"""Analysis-run records + an on-disk run store (plan 24).

Pure module: importing it pulls in only the stdlib, so `import smile_msi.runs`
stays cheap and Qt-free. pandas/numpy are imported LAZILY inside the few methods
that touch them, because a run's *metadata* (the dataclass, its JSON round-trip,
the digest) never needs them — only saving/loading a heavyweight result payload
does.

An ``AnalysisRun`` is the durable receipt for one analysis: what step ran, on
which dataset, with which inputs/params, and where its result payload landed.
``RunStore`` lays these out one-directory-per-run under a root so a session can
embed a lightweight index (see ``RunStore.index``) and rehydrate results on
demand.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

VERSION = 1
STATUSES = ("running", "done", "error", "cancelled")

# sentinel a NaN collapses to in the canonical digest form — a plain float('nan')
# is not JSON-representable AND compares unequal to itself, so it could never give
# a stable hash. Coercing it to a fixed string makes the digest deterministic.
_NAN_SENTINEL = "__NaN__"


def _utcnow_iso() -> str:
    # ISO-8601, UTC, second precision (no microseconds) — matches the human-facing
    # 'created' stamps used elsewhere and keeps newest-first sorts lexicographic.
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_default(o):
    """json ``default=`` hook: coerce the awkward payloads a digest may meet.

    Numpy scalars/arrays, datetimes, sets and NaN all reach here (or are handled
    by ``_canonical`` before) so ``json.dumps`` never raises on a run's inputs.
    """
    # numpy scalar (has .item()) or ndarray (has .tolist()) — checked without
    # importing numpy at module load.
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        try:
            return _canonical(o.item())
        except Exception:
            pass
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, (set, frozenset)):
        return sorted((_canonical(x) for x in o), key=_sort_key)
    if isinstance(o, bytes):
        return o.decode("utf-8", "replace")
    return str(o)


def _sort_key(x):
    """A total order over canonicalised set members. A heterogeneous set has no natural
    ordering — bare ``sorted({1, "a"})`` raises — so order by type name, then repr."""
    return (type(x).__name__, repr(x))


def _canonical(o):
    """Recursively rewrite ``o`` into a JSON-native form whose encoding is stable.

    Total by construction: every branch returns JSON-native output, so
    ``json.dumps(_canonical(x), allow_nan=False)`` cannot raise. The ``default=`` hook is
    belt-and-braces, never load-bearing — which matters because that hook cannot fix a NaN
    it hands back inside a list.

    - dicts: values canonicalised (json ``sort_keys`` handles key order).
    - sets/frozensets: sorted lists under ``_sort_key`` → a set and its sorted list hash alike.
    - lists/tuples: order preserved (order IS meaningful for a sequence).
    - NaN: the sentinel string — ``float('nan') != itself``, so it can never hash stably.
    - numpy scalars *and arrays of any size*: unwrapped via ``.tolist()`` then re-canonicalised,
      so a NaN nested inside an array normalises too.
    """
    if isinstance(o, float):                      # np.float64 subclasses float; catches its NaN
        return _NAN_SENTINEL if math.isnan(o) else o
    if o is None or isinstance(o, (str, bool, int)):
        return o
    if isinstance(o, dict):
        return {k: _canonical(v) for k, v in o.items()}
    if isinstance(o, (set, frozenset)):
        return sorted((_canonical(x) for x in o), key=_sort_key)
    if isinstance(o, (list, tuple)):
        return [_canonical(x) for x in o]
    if isinstance(o, bytes):
        return o.decode("utf-8", "replace")
    if isinstance(o, datetime):
        return o.isoformat()
    if hasattr(o, "tolist"):                      # numpy scalar or ndarray of any size
        return _canonical(o.tolist())
    if hasattr(o, "item"):
        try:
            return _canonical(o.item())
        except Exception:
            pass
    return str(o)


def json_safe(obj, _depth=0) -> bool:
    """True if ``obj`` survives :meth:`RunStore.save_result`'s ``json.dump`` without loss.

    :func:`_json_default` is a *total* hook — its last line is ``str(o)`` — so an unsupported
    object never raises, it silently persists as its own repr. A pandas DataFrame is exactly
    that case: neither JSON-native nor caught by the ``.tolist()`` / ``.item()`` probes, so a
    dict carrying one reopens with a meaningless string where the table should be. Steps whose
    result dict holds frames (``filter_auc``, ``marker_panel``, ``region_membership``) call
    this and persist their flat table instead.

    So the predicate is precisely "would ``_json_default`` have to stringify this?". Numpy
    scalars and arrays pass (the hook unwraps them); containers recurse, bounded because a
    lossy value nested deeper than any payload we ship is not worth the traversal.
    """
    if obj is None or isinstance(obj, (str, bytes, bool, int, float, datetime)):
        return True
    if isinstance(obj, dict):
        return _depth >= 4 or all(json_safe(v, _depth + 1) for v in obj.values())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return _depth >= 4 or all(json_safe(v, _depth + 1) for v in obj)
    return hasattr(obj, "tolist") or hasattr(obj, "item")     # numpy scalar / ndarray


@dataclass
class AnalysisRun:
    """One logged analysis. All fields are JSON-native (or coerced on write)."""

    run_id: str
    step_id: str
    title: str
    created: str
    status: str = "running"
    target: str = "slide"
    dataset: str = ""
    dataset_fingerprint: str = ""
    inputs: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    summary: str = ""
    error: str = ""
    result_ref: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AnalysisRun":
        # tolerant of missing keys (old/partial records): only known fields are
        # read, each falling back to its dataclass default.
        d = d or {}
        return cls(
            run_id=d.get("run_id", ""),
            step_id=d.get("step_id", ""),
            title=d.get("title", ""),
            created=d.get("created", ""),
            status=d.get("status", "running"),
            target=d.get("target", "slide"),
            dataset=d.get("dataset", ""),
            dataset_fingerprint=d.get("dataset_fingerprint", ""),
            inputs=dict(d.get("inputs") or {}),
            params=dict(d.get("params") or {}),
            summary=d.get("summary", ""),
            error=d.get("error", ""),
            result_ref=d.get("result_ref"),
        )

    def inputs_digest(self) -> str:
        """Stable sha1 over the run's inputs.

        Invariant to dict key insertion order (json ``sort_keys``) and to
        set-vs-sorted-list (sets canonicalise to sorted lists). Never raises on
        numpy scalars/arrays, NaN, or datetimes — the ``_canonical`` pass plus the
        ``default=`` hook coerce them all.
        """
        payload = json.dumps(
            _canonical(self.inputs),
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
            allow_nan=False,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def is_stale(self, dataset_fingerprint, inputs) -> bool:
        """True if the dataset fingerprint OR the inputs digest has drifted."""
        if dataset_fingerprint != self.dataset_fingerprint:
            return True
        other = AnalysisRun(
            run_id="", step_id="", title="", created="", inputs=dict(inputs or {})
        )
        return other.inputs_digest() != self.inputs_digest()


def new_run(step_id, *, title, target="slide", dataset="", dataset_fingerprint="",
            inputs=None, params=None, now=None) -> AnalysisRun:
    """Mint a fresh 'running' AnalysisRun. ``now`` is injectable for tests."""
    return AnalysisRun(
        run_id=_new_run_id(now),
        step_id=step_id,
        title=title,
        created=(now or _utcnow_iso()),
        status="running",
        target=target,
        dataset=dataset,
        dataset_fingerprint=dataset_fingerprint,
        inputs=dict(inputs or {}),
        params=dict(params or {}),
    )


def _new_run_id(now=None) -> str:
    # timestamp prefix keeps ids sortable/legible; a short random suffix avoids
    # collisions when two runs start in the same second.
    import uuid

    stamp = (now or _utcnow_iso())
    safe = "".join(c if c.isalnum() else "-" for c in stamp)
    return f"{safe}-{uuid.uuid4().hex[:8]}"


class RunStore:
    """One-directory-per-run store rooted at ``root``.

    Layout::

        root/<run_id>/meta.json      # the AnalysisRun record
        root/<run_id>/result.{csv,npz,json}
        root/<run_id>/thumb.png

    Meta writes are atomic (temp + fsync + os.replace), mirroring
    ``session.save_session`` so a crash mid-write can't corrupt an existing run.
    """

    META = "meta.json"
    THUMB = "thumb.png"

    def __init__(self, root):
        self.root = str(root)

    def run_dir(self, run_id) -> str:
        # a run_id is a single path segment; reject anything that could escape the
        # root (separators or parent refs). No sanitising/escaping — refuse it.
        rid = str(run_id)
        if (not rid or rid in (".", "..") or os.sep in rid
                or "/" in rid or "\\" in rid or ".." in rid):
            raise ValueError(f"unsafe run_id {run_id!r}: must be a single path segment")
        return os.path.join(self.root, rid)

    def save(self, run) -> str:
        d = self.run_dir(run.run_id)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, self.META)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            # default= coerces the odd numpy scalar / set that can slip into
            # inputs/params, so a heavyweight caller can't crash the meta write.
            json.dump(run.to_dict(), f, indent=2, default=_json_default)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return path

    def load(self, run_id) -> AnalysisRun:
        path = os.path.join(self.run_dir(run_id), self.META)
        with open(path, encoding="utf-8") as f:
            return AnalysisRun.from_dict(json.load(f))

    def save_result(self, run, obj):
        """Persist a result payload beside the run, set + return ``run.result_ref``.

        DataFrame → result.csv (index dropped); ndarray → result.npz (key 'a');
        dict|list → result.json; None → no-op (returns None). Anything else is a
        TypeError. pyarrow is NOT installed — DataFrames go to CSV, never parquet.
        """
        if obj is None:
            return None
        d = self.run_dir(run.run_id)
        os.makedirs(d, exist_ok=True)

        # lazy: only touch pandas/numpy when a payload actually needs them.
        try:
            import pandas as pd
        except Exception:
            pd = None
        try:
            import numpy as np
        except Exception:
            np = None

        if pd is not None and isinstance(obj, pd.DataFrame):
            name = "result.csv"
            obj.to_csv(os.path.join(d, name), index=False)
        elif np is not None and isinstance(obj, np.ndarray):
            name = "result.npz"
            np.savez(os.path.join(d, name), a=obj)
        elif isinstance(obj, (dict, list)):
            name = "result.json"
            with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                json.dump(obj, f, default=_json_default)
        else:
            raise TypeError(
                f"save_result cannot persist {type(obj).__name__}; "
                "supported: pandas.DataFrame, numpy.ndarray, dict, list, or None"
            )

        run.result_ref = name
        self.save(run)
        return name

    def load_result(self, run):
        """Rehydrate the saved payload, or None if none/ the file is gone."""
        ref = getattr(run, "result_ref", None)
        if not ref:
            return None
        path = os.path.join(self.run_dir(run.run_id), ref)
        if not os.path.exists(path):
            return None
        if ref.endswith(".csv"):
            import pandas as pd

            return pd.read_csv(path)
        if ref.endswith(".npz"):
            import numpy as np

            with np.load(path) as z:
                return z["a"]
        if ref.endswith(".json"):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        # unknown ref extension — read raw bytes rather than guess.
        with open(path, "rb") as f:
            return f.read()

    def save_thumb(self, run, png_bytes) -> str:
        d = self.run_dir(run.run_id)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, self.THUMB)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(png_bytes)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return path

    def thumb_path(self, run):
        p = os.path.join(self.run_dir(run.run_id), self.THUMB)
        return p if os.path.exists(p) else None

    def list_runs(self):
        """All runs, newest-first by 'created'. Corrupt/unreadable meta skipped."""
        runs = []
        try:
            entries = os.listdir(self.root)
        except FileNotFoundError:
            return []
        for name in entries:
            meta = os.path.join(self.root, name, self.META)
            if not os.path.isfile(meta):
                continue
            try:
                with open(meta, encoding="utf-8") as f:
                    runs.append(AnalysisRun.from_dict(json.load(f)))
            except Exception:
                # unreadable / malformed JSON — skip, don't abort the listing.
                continue
        runs.sort(key=lambda r: r.created, reverse=True)
        return runs

    def delete(self, run_id) -> bool:
        """Remove a run dir. False if absent; tolerates a half-written dir."""
        d = self.run_dir(run_id)
        if not os.path.isdir(d):
            return False
        import shutil

        shutil.rmtree(d, ignore_errors=True)
        return True

    def index(self):
        """Newest-first list of to_dict()s — the lightweight session embed."""
        return [r.to_dict() for r in self.list_runs()]
