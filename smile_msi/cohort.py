"""Cohorts — working with many samples at once.

A *cohort* is the level **above** a sample's regions. Where a region groups pixels
*within* one slide, a cohort groups whole slides *across* files, each tagged with a
**group** label (e.g. ``"control"`` vs ``"synkinetic"``), so the same kind of
A-vs-B / multi-group comparison the app already does between regions can run
sample-to-sample. It is a lightweight roster, not a second data store:

    Cohort                      ← this module (a named list of samples + group labels)
      └─ Sample                 ← a pointer to a managed session: a *whole slide* or,
           │                       when SampleRef.region is set, one region within it
           └─ Region            ← pixels within that slide
                └─ Pixel

A cohort holds **no spectra**. A :class:`SampleRef` is just a pointer to a sample's
managed session JSON (see :mod:`smile_msi.session`). Cross-sample tables are built by
reading each sample's *session* — the m/z and intensities its peaks were saved with (a
whole slide reads the session's top-level peaks; a **region sample** reads that region's
saved feature scope) — so a batch overview costs **zero dataset loads**: the heavy cubes
stay on disk. Callers
that need exact per-region extraction can supply a loader; this module deliberately
stays GUI- and IO-light so it is unit-testable without Qt or a real imzML.

The functions are pure (dict/array in, dict/DataFrame/file out), mirroring
:mod:`smile_msi.session` and :mod:`smile_msi.spatial`.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import numpy as np

from . import library, session
from .constants import CONSENSUS_TOL_PPM, DEFAULT_TOL_PPM

VERSION = 3   # v2 adds SampleRef.batch (batch axis); v3 adds SampleRef.modality (co-mapping);
              # older rosters forward-fill both. from_dict ignores the version field.


# --------------------------------------------------------------------------- #
# locations
# --------------------------------------------------------------------------- #
def cohorts_dir() -> str:
    """Directory holding saved cohorts; created on demand (mirrors
    :func:`smile_msi.session.sessions_dir`)."""
    d = os.path.join(library.home_dir(), "cohorts")
    os.makedirs(d, exist_ok=True)
    return d


def cohort_path(name: str) -> str:
    """Path of the named cohort's JSON file under :func:`cohorts_dir`."""
    return os.path.join(cohorts_dir(), f"{session._sanitize(name)}.json")


def list_cohorts() -> list[dict]:
    """``{name, path, n_samples, mtime}`` for every saved cohort, newest first.
    Corrupt/partial files are skipped (mirrors :func:`session.list_managed`)."""
    return session._scan_json_dir(cohorts_dir(), lambda fn, data: {
        "name": data.get("name", os.path.splitext(fn)[0]),
        "n_samples": len(data.get("samples", []) or []),
    })


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #
def _session_mtime(s: "SampleRef") -> float:
    """Modification time of a sample's managed session file (``-1`` if missing) — used to
    keep the most recently worked-on copy when collapsing duplicates."""
    try:
        return os.stat(s.session_path).st_mtime
    except OSError:
        return -1.0


@dataclass
class SampleRef:
    """A pointer to one sample in a cohort — its managed session plus the group it
    belongs to. Carries enough identity (fingerprint, n_pixels) to detect when a
    session was captured on a different slide than its recorded source now holds.

    A sample is normally a *whole slide*. When ``region`` is set, the sample is instead
    one named region **within** that slide: its intensities are read from the session's
    saved per-region feature scope of the same name (see :func:`_session_peaks`), so an
    area of a slide can act as an independent replicate alongside whole slides."""
    name: str                       # display name (session basename / source basename)
    session_path: str               # managed session JSON for this sample
    source: str = ""                # raw data path the session reloads from (may be moved)
    group: str = ""                 # cohort group label (free text; "" = ungrouped)
    batch: str = ""                 # acquisition batch label (technical; "" = unassigned)
    fingerprint: str = ""           # dataset_fingerprint at save time (identity check)
    n_pixels: int | None = None     # slide pixel count (whole-slide identity check)
    region: str = ""                # region / feature-scope name; "" = the whole slide
    region_n_pixels: int | None = None   # pixel count of the region (display only)
    meta: dict = field(default_factory=dict)   # free-form donor metadata (age, sex, bmi, …)
    modality: dict | None = None   # second-modality pointer (path + kind + transform) for co-mapping

    def to_dict(self) -> dict:
        return {"name": self.name, "session_path": self.session_path, "source": self.source,
                "group": self.group, "batch": self.batch,
                "fingerprint": self.fingerprint, "n_pixels": self.n_pixels,
                "region": self.region, "region_n_pixels": self.region_n_pixels,
                "meta": dict(self.meta),
                "modality": (dict(self.modality) if self.modality else None)}

    @classmethod
    def from_dict(cls, d: dict) -> "SampleRef":
        return cls(name=str(d.get("name", "")), session_path=str(d.get("session_path", "")),
                   source=str(d.get("source", "")), group=str(d.get("group", "")),
                   batch=str(d.get("batch", "")),
                   fingerprint=str(d.get("fingerprint", "")), n_pixels=d.get("n_pixels"),
                   region=str(d.get("region", "")), region_n_pixels=d.get("region_n_pixels"),
                   meta=dict(d.get("meta") or {}), modality=d.get("modality"))

    def label_with_meta(self) -> str:
        """Display name with any age/sex/bmi appended — the bubble-plot donor row label."""
        bits = [str(self.meta[k]) for k in ("age", "sex", "bmi") if self.meta.get(k) not in (None, "")]
        return f"{self.name} ({', '.join(bits)})" if bits else self.name

    def key(self) -> str:
        """Stable identity for dedup. Keys on the raw data **source** path — the one thing
        that survives the managed session being re-keyed. (A drifting dataset fingerprint
        re-keys the session file, which used to mint a fresh duplicate of the same slide on
        every load; keying on the source makes dedup immune to that.) Falls back to the
        session path, then the display name. A region sample appends the region name so
        several regions of one slide coexist.

        Identity is normalized via :func:`_canonical_source` (separator- and case-insensitive)
        so the same slide doesn't mint duplicate roster rows when referenced with Windows
        casing, ``\\`` vs ``/``, or after a cross-OS move."""
        if self.source:
            base = _canonical_source(self.source)
        elif self.session_path:
            base = _canonical_source(self.session_path)
        else:
            base = (self.name or "").lower()
        return f"{base}::{self.region}" if self.region else base

    def scope(self) -> str | None:
        """The session feature-scope to read intensities from — the region name for a
        region sample, ``None`` for a whole slide (which reads the top-level peaks)."""
        return self.region or None

    def folder(self) -> str:
        """The sample's containing folder name — the natural 'group by folder' bucket
        (mirrors how FlowJo auto-groups by the folder a file came from). Empty when the
        sample has no on-disk source. Separator-agnostic so a Windows path groups correctly
        on a POSIX host (and vice-versa)."""
        base = self.source or self.session_path
        return _path_dirname(base) if base else ""

    def match_ids(self) -> list[str]:
        """Identifiers a metadata row can be keyed on, lower-cased: the display name, the
        source/session basenames (with and without extension), and the full source path.
        Lets a metadata CSV match a sample by whichever column the user has."""
        out: list[str] = []
        if self.name:
            out.append(self.name)
        for p in (self.source, self.session_path):
            if not p:
                continue
            b = _path_basename(p)                    # separator-agnostic (Windows path on POSIX)
            out += [b, os.path.splitext(b)[0], p]    # no os.path.abspath — it cwd-anchors (#5)
        # de-dupe, lower-case, drop blanks
        return list({s.strip().lower() for s in out if s and s.strip()})


@dataclass
class Cohort:
    """A named roster of :class:`SampleRef`. Add/remove samples, label their groups,
    save/load as JSON. Holds no spectra — see the module docstring."""
    name: str = "Cohort"
    samples: list[SampleRef] = field(default_factory=list)

    # ---- membership ----------------------------------------------------- #
    def add(self, ref: SampleRef) -> SampleRef:
        """Add a sample, or return the existing one if its session is already present
        (so re-adding the active sample is idempotent)."""
        for s in self.samples:
            if s.key() == ref.key():
                return s
        self.samples.append(ref)
        return ref

    def remove(self, key: str) -> bool:
        n = len(self.samples)
        self.samples = [s for s in self.samples if s.key() != key]
        return len(self.samples) != n

    def dedupe(self) -> int:
        """Collapse samples that share an identity :meth:`SampleRef.key` (same source +
        region) into one, keeping the ref whose managed session file is newest and carrying
        over any non-empty group label. Returns the number of samples removed. Repairs
        rosters minted before the key was based on the stable source path — a drifting
        fingerprint used to add a fresh duplicate of the same slide on every load."""
        best: dict[str, SampleRef] = {}
        order: list[str] = []
        removed = 0
        for s in self.samples:
            k = s.key()
            cur = best.get(k)
            if cur is None:
                best[k] = s
                order.append(k)
                continue
            removed += 1
            keep, drop = (s, cur) if _session_mtime(s) > _session_mtime(cur) else (cur, s)
            # Carry over EVERY label the kept (newer-mtime) ref lacks — not just group. The
            # survivor is chosen by mtime, an axis independent of which duplicate the user
            # labelled, so keeping only `group` silently dropped batch/meta/modality and broke
            # later ComBat / group-by-metadata on the next launch (dedupe runs + persists then).
            for attr in ("group", "batch", "fingerprint", "modality"):
                if not getattr(keep, attr) and getattr(drop, attr):
                    setattr(keep, attr, getattr(drop, attr))
            if keep.n_pixels is None and drop.n_pixels is not None:
                keep.n_pixels = drop.n_pixels
            if keep.region_n_pixels is None and drop.region_n_pixels is not None:
                keep.region_n_pixels = drop.region_n_pixels
            if drop.meta:                            # kept values win; dropped fills the gaps
                keep.meta = {**drop.meta, **(keep.meta or {})}
            best[k] = keep
        if removed:
            self.samples = [best[k] for k in order]
        return removed

    def find(self, key: str) -> SampleRef | None:
        for s in self.samples:
            if s.key() == key:
                return s
        return None

    def set_group(self, key: str, group: str) -> bool:
        s = self.find(key)
        if s is None:
            return False
        s.group = str(group or "")
        return True

    def groups(self) -> list[str]:
        """Distinct non-empty group labels, in first-seen order."""
        seen, out = set(), []
        for s in self.samples:
            g = s.group or ""
            if g and g not in seen:
                seen.add(g)
                out.append(g)
        return out

    def by_group(self) -> dict[str, list[SampleRef]]:
        """Samples bucketed by group label; ungrouped samples land under ``""``."""
        out: dict[str, list[SampleRef]] = {}
        for s in self.samples:
            out.setdefault(s.group or "", []).append(s)
        return out

    # ---- bulk / criteria-based grouping --------------------------------- #
    # Assigning a group to 30 samples one click at a time is the cohort-scale pain;
    # these derive the group from something the samples already carry (their folder,
    # a piece of their name, or a metadata field) in one pass — the FlowJo "group by
    # keyword / folder" move. Each returns the number of samples whose group changed and
    # skips samples that yield no label (so a partial match never blanks the rest).
    def _assign_derived_to(self, attr: str, label_of) -> int:
        """Set ``attr`` (``'group'`` or ``'batch'``) on each sample from ``label_of(s)``;
        blank labels are skipped so a partial match never blanks the rest. Returns the
        number of samples changed."""
        changed = 0
        for s in self.samples:
            v = (label_of(s) or "").strip()
            if v and v != getattr(s, attr, ""):
                setattr(s, attr, v)
                changed += 1
        return changed

    def _assign_derived(self, label_of) -> int:
        return self._assign_derived_to("group", label_of)

    def group_by_folder(self) -> int:
        """Group each sample by its containing folder name."""
        return self._assign_derived(lambda s: s.folder())

    def group_by_pattern(self, pattern: str) -> int:
        """Group by a regex matched against the sample name. The first capture group (or
        the whole match, if the pattern has none) becomes the group label. Samples that
        don't match keep their current group. Raises ``re.error`` on a bad pattern."""
        rx = re.compile(pattern)

        def label(s: SampleRef) -> str:
            m = rx.search(s.name or "")
            if not m:
                return ""
            return m.group(1) if m.groups() else m.group(0)

        return self._assign_derived(label)

    def group_by_meta(self, key: str) -> int:
        """Group by the value of metadata ``key`` (e.g. ``"genotype"``)."""
        return self._assign_derived(
            lambda s: "" if s.meta.get(key) in (None, "") else str(s.meta[key]))

    # ---- acquisition batch (a technical axis, independent of biological group) --
    # Batch and group are deliberately separate: batch correction must remove the
    # technical (batch) shift while *protecting* the biological (group) contrast.
    def batches(self) -> list[str]:
        """Distinct non-empty batch labels, in first-seen order."""
        seen, out = set(), []
        for s in self.samples:
            b = getattr(s, "batch", "") or ""
            if b and b not in seen:
                seen.add(b)
                out.append(b)
        return out

    def by_batch(self) -> dict[str, list[SampleRef]]:
        """Samples bucketed by batch label; unassigned land under ``""``."""
        out: dict[str, list[SampleRef]] = {}
        for s in self.samples:
            out.setdefault(getattr(s, "batch", "") or "", []).append(s)
        return out

    def set_batch(self, name: str, batch: str) -> bool:
        """Set the batch label of the sample whose name matches; returns True if found."""
        for s in self.samples:
            if s.name == name:
                s.batch = str(batch).strip()
                return True
        return False

    def batch_by_folder(self) -> int:
        """Batch each sample by its containing folder name (e.g. one folder per run)."""
        return self._assign_derived_to("batch", lambda s: s.folder())

    def batch_by_pattern(self, pattern: str) -> int:
        """Batch by a regex captured from the sample name (cf. :meth:`group_by_pattern`)."""
        rx = re.compile(pattern)

        def label(s: SampleRef) -> str:
            m = rx.search(s.name or "")
            if not m:
                return ""
            return m.group(1) if m.groups() else m.group(0)

        return self._assign_derived_to("batch", label)

    def batch_by_meta(self, key: str) -> int:
        """Batch by the value of metadata ``key`` (e.g. ``"acq_date"`` / ``"matrix_lot"``)."""
        return self._assign_derived_to(
            "batch", lambda s: "" if s.meta.get(key) in (None, "") else str(s.meta[key]))

    def meta_keys(self) -> list[str]:
        """Distinct metadata keys present across the roster, in first-seen order — the
        columns the roster can show and 'group by metadata' can offer."""
        seen, out = set(), []
        for s in self.samples:
            for k in s.meta:
                if k not in seen:
                    seen.add(k)
                    out.append(k)
        return out

    def apply_metadata(self, rows, *, key_col=None, group_col=None) -> int:
        """Attach metadata from CSV-like ``rows`` (a list of ``{column: value}`` dicts).

        Each row is matched to a sample by ``key_col`` (defaults to the first column)
        against the sample's name / source basename / path (see
        :meth:`SampleRef.match_ids`). Non-key columns fill the sample's ``meta`` dict;
        ``group_col``, if given, also sets the group. Returns the number of samples that
        matched a row."""
        rows = list(rows or [])
        if not rows:
            return 0
        if key_col is None:
            key_col = next(iter(rows[0]))
        # index samples by every identifier they answer to, so any key column lines up
        by_id: dict[str, SampleRef] = {}
        for s in self.samples:
            for mid in s.match_ids():
                by_id.setdefault(mid, s)
        matched: set[int] = set()
        for row in rows:
            kv = str(row.get(key_col, "")).strip().lower()
            s = by_id.get(kv)
            if s is None:
                continue
            for col, val in row.items():
                if col == key_col or val in (None, ""):
                    continue
                if group_col is not None and col == group_col:
                    s.group = str(val).strip()
                else:
                    s.meta[col] = val
            matched.add(id(s))
        return len(matched)

    # ---- persistence ---------------------------------------------------- #
    def to_dict(self) -> dict:
        return {"version": VERSION, "name": self.name,
                "samples": [s.to_dict() for s in self.samples]}

    @classmethod
    def from_dict(cls, d: dict) -> "Cohort":
        return cls(name=str(d.get("name", "Cohort")),
                   samples=[SampleRef.from_dict(s) for s in (d.get("samples") or [])])

    def save(self, path: str | None = None) -> str:
        """Atomic write (temp + fsync + replace), like :func:`session.save_session`."""
        path = path or cohort_path(self.name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: str) -> "Cohort":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def ref_from_session(session_path: str, session_dict: dict | None = None,
                     group: str = "") -> SampleRef:
    """Build a :class:`SampleRef` from a managed session file (loading it if the dict
    isn't supplied) — the bridge from the per-sample store to a cohort."""
    data = session_dict if session_dict is not None else session.load_session(session_path)
    src = data.get("source", "") or ""
    return SampleRef(name=session._display_name(src), session_path=session_path, source=src,
                     group=group, fingerprint=data.get("dataset_fingerprint") or "",
                     n_pixels=data.get("n_pixels"))


def discover_samples() -> list[SampleRef]:
    """Every auto-saved managed session as an (ungrouped) :class:`SampleRef`, newest
    first — so a fresh cohort can be seeded from the samples you've already worked on."""
    return [SampleRef(name=m["name"], session_path=m["path"], source=m.get("source", "") or "",
                      n_pixels=m.get("n_pixels")) for m in session.list_managed()]


# --------------------------------------------------------------------------- #
# relocation  (heal stale absolute paths after a cohort is moved / copied)
# --------------------------------------------------------------------------- #
def _path_basename(p: str) -> str:
    """Basename of a *stored* path that may use either separator, regardless of the host OS
    running us. ``os.path.basename`` only splits on the local separator, so a Windows-made
    cohort (``\\`` paths) opened on macOS/Linux — or vice-versa — would not split correctly."""
    return re.split(r"[\\/]", p)[-1] if p else p


def _path_dirname(p: str) -> str:
    """Parent-folder *name* of a stored path, separator-agnostic (the 'group by folder'
    bucket). ``os.path.dirname`` only splits on the local separator, so a Windows path on a
    POSIX host would yield ``''`` for every sample (#folder-bug)."""
    parts = re.split(r"[\\/]", p) if p else []
    return parts[-2] if len(parts) >= 2 else ""


def _canonical_source(p: str) -> str:
    """Separator- and case-normalized identity for a stored source/session path. Separators are
    unified (``\\`` → ``/``) unconditionally so a cohort compares equal across a cross-OS move;
    case is folded **only on Windows** (``os.name == 'nt'``) — mirroring :func:`os.path.normcase`
    — because Windows paths are case-insensitive (the same slide opened with different casing
    must map to one identity) while POSIX filesystems are case-sensitive (folding there could
    wrongly merge two distinct files). Avoids ``os.path.abspath``, which *cwd-anchors* a
    foreign-OS absolute path (e.g. ``C:\\x`` → ``/cwd/C:\\x``), making the key cwd-dependent."""
    if not p:
        return p
    q = re.sub(r"[\\/]+", "/", p).rstrip("/")
    return q.lower() if os.name == "nt" else q


def source_exists(ref) -> bool:
    """True when ``ref``'s raw ``source`` is an imzML present on *this* machine — exactly the
    condition :class:`SectionLoader` needs to load the sample. Synthetic / CSV / empty sources
    are not loadable and return False (they aren't 'missing', just not imzML-backed)."""
    src = (getattr(ref, "source", "") or "")
    return bool(src) and src.lower().endswith(".imzml") and os.path.exists(src)


def missing_source_samples(samples) -> list:
    """Samples whose imzML ``source`` looks like a real path (ends ``.imzml``) but isn't here —
    the 'cohort moved / new machine' case a relocation can repair. Excludes synthetic samples
    (empty / non-imzML source), which are legitimately unloadable and can't be relocated."""
    out = []
    for s in samples:
        src = (getattr(s, "source", "") or "")
        if src.lower().endswith(".imzml") and not os.path.exists(src):
            out.append(s)
    return out


def sample_load_problem(ref) -> str | None:
    """``None`` if ``ref`` can load for a pooled embedding, else a short reason — mirrors
    :meth:`SectionLoader.load`'s checks so a caller can pre-flight a run and say *which*
    samples won't load and *why*, instead of grinding to a generic "None could be loaded".

    A whole slide needs an imzML ``source`` present here. A **region** sample additionally
    needs a readable ``session_path`` whose saved region resolves to non-empty pixels — a
    constraint :func:`source_exists` alone misses, so a region sample with a stale session
    silently fails even when its imzML is present (the moved-cohort gap)."""
    src = (getattr(ref, "source", "") or "")
    if not src.lower().endswith(".imzml"):
        return "source is not an imzML"
    if not os.path.exists(src):
        return "source file not found on this machine"
    region = (getattr(ref, "region", "") or "")
    if region:
        sp = (getattr(ref, "session_path", "") or "")
        if not sp or not os.path.exists(sp):
            return "region's session file not found on this machine"
        try:
            data = session.load_session(sp)
        except (OSError, ValueError):
            return "region's session file is unreadable"
        if region_pixels(data, region).size == 0:
            return f"region '{region}' has no saved pixels in its session"
    return None


def unloadable_samples(samples) -> list:
    """``[(ref, reason)]`` for every sample that can't load for a pooled embedding — the
    session-aware superset of :func:`missing_source_samples` (it also catches region samples
    whose imzML is present but whose session/region pixels are not)."""
    out = []
    for s in samples:
        reason = sample_load_problem(s)
        if reason is not None:
            out.append((s, reason))
    return out


@dataclass
class RelocationReport:
    """Outcome of :func:`relocate_samples`. ``fixed`` is ``(name, old_source, new_source)``;
    ``ambiguous`` is ``(name, basename, n_candidates)`` for names that matched >1 file (left
    untouched — the caller can't safely guess); ``unresolved`` is ``(name, source)`` not found
    under any root."""
    fixed: list = field(default_factory=list)
    repointed_sessions: int = 0
    ambiguous: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)


def _index_imzml_by_basename(roots) -> dict:
    """``basename(lower) -> [full paths]`` for every ``.imzml`` under any of ``roots``
    (searched recursively). Roots that don't exist are skipped."""
    index: dict[str, list[str]] = {}
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if f.lower().endswith(".imzml"):
                    index.setdefault(f.lower(), []).append(os.path.join(dirpath, f))
    return index


def _rewrite_session_source(session_path: str, new_source: str) -> bool:
    """Point a managed session JSON's own ``source`` at ``new_source`` so reopening that
    sample on its own reloads from the right place too. Best-effort; never raises."""
    if not session_path or not os.path.isfile(session_path):
        return False
    try:
        data = session.load_session(session_path)
        data["source"] = new_source
        session.save_session(session_path, data)
    except (OSError, ValueError):
        return False
    return True


def relocate_samples(samples, data_roots, *, fix_sessions: bool = False) -> RelocationReport:
    """Repair stale absolute paths in ``samples`` **in place** after a cohort was moved to
    another folder or machine.

    For every sample whose imzML ``source`` is gone, search ``data_roots`` recursively for a
    file with the same basename and, on a *unique* match, rewrite ``source`` to it (a name that
    matches several files is reported ``ambiguous`` and left alone — guessing could embed the
    wrong slide). A missing ``session_path`` is repointed at this machine's managed-sessions
    store by basename. ``fix_sessions`` also rewrites the ``source`` recorded *inside* each
    relocated sample's session JSON.

    Mutates the :class:`SampleRef` objects; the caller persists the roster (:meth:`Cohort.save`)
    and gets back a :class:`RelocationReport`."""
    index = _index_imzml_by_basename(list(data_roots))
    report = RelocationReport()
    sess_dir = session.sessions_dir()
    # How many *distinct roster samples* are missing each basename. When two different slides
    # were exported under the same generic basename (e.g. runA/Analysis.imzML, runB/Analysis.
    # imzML), basename alone can't say which file is which — so if >1 sample needs a basename we
    # must NOT relocate any of them onto a single found file (that collapses two distinct slides
    # onto one source → identical key() → dedup deletes one / the other becomes unreachable).
    needed: dict[str, int] = {}
    for s in samples:
        src = (getattr(s, "source", "") or "")
        if src.lower().endswith(".imzml") and not os.path.exists(src):
            bn = _path_basename(src).lower()
            needed[bn] = needed.get(bn, 0) + 1
    for s in samples:
        # Repoint session_path to this machine's store FIRST, so a `fix_sessions` rewrite below
        # targets the local (existing) session JSON rather than the stale, absent one.
        sp = (getattr(s, "session_path", "") or "")
        if sp and not os.path.exists(sp):
            cand = os.path.join(sess_dir, _path_basename(sp))
            if os.path.exists(cand):
                s.session_path = cand
                report.repointed_sessions += 1
        src = (getattr(s, "source", "") or "")
        if src.lower().endswith(".imzml") and not os.path.exists(src):
            bn = _path_basename(src).lower()
            hits = index.get(bn, [])
            if needed.get(bn, 0) > 1:
                # several distinct roster samples share this basename — refuse to guess which
                # found file belongs to which slide; surface for manual resolution.
                report.ambiguous.append((getattr(s, "name", ""), _path_basename(src),
                                         max(len(hits), needed[bn])))
            elif len(hits) == 1:
                report.fixed.append((getattr(s, "name", ""), src, hits[0]))
                s.source = hits[0]
                if fix_sessions:
                    _rewrite_session_source(getattr(s, "session_path", "") or "", hits[0])
            elif len(hits) > 1:
                report.ambiguous.append((getattr(s, "name", ""), _path_basename(src), len(hits)))
            else:
                report.unresolved.append((getattr(s, "name", ""), src))
    return report


# --------------------------------------------------------------------------- #
# cross-sample feature tables  (from session metadata — no dataset loads)
# --------------------------------------------------------------------------- #
def _session_peaks(data: dict, value: str = "rel_intensity", scope: str | None = None):
    """``(mz_sorted, value_sorted)`` from a session's saved peaks. ``value`` is the
    per-peak field to read — ``rel_intensity`` (0..1, most comparable across slides) or
    ``intensity`` (raw mean). ``scope`` selects which peak list to read: ``None`` (the
    default) reads the slide's top-level ``peaks``; a name reads that region's saved
    feature scope (``feature_scopes[scope]``), so a region can act as its own sample.
    Returns empties when the scope is absent (a region whose features were never built or
    has since been renamed). Peaks are returned sorted by m/z for fast matching."""
    if scope is None:
        peaks = data.get("peaks") or []
    else:
        peaks = (data.get("feature_scopes") or {}).get(scope) or []
    if not peaks:
        return np.empty(0), np.empty(0)
    mz = np.array([float(p.get("mz", 0.0)) for p in peaks], dtype=float)
    val = np.array([float(p.get(value, p.get("rel_intensity", 0.0))) for p in peaks], dtype=float)
    order = np.argsort(mz)
    return mz[order], val[order]


def region_pixels(data: dict, name: str) -> np.ndarray:
    """Pixel indices of named region ``name`` in a loaded session ``data`` — its saved
    ``mask`` or, for a cluster-backed region (empty mask, the app's primary region type),
    reconstructed from its ``segments`` over the session's ``segmentation.labels``. Empty
    array when the region is absent / its mask is gone. Shared by :class:`SectionLoader` and
    the region-mean embedding so both resolve a region's pixels identically."""
    match = next((r for r in (data.get("named_regions") or []) if r.get("name") == name), None)
    if match is None:
        return np.empty(0, dtype=int)
    pix = np.asarray(match.get("mask") or [], dtype=int)
    if pix.size == 0:
        segs = match.get("segments") or []
        labels = np.asarray((data.get("segmentation") or {}).get("labels") or [], dtype=int)
        if segs and labels.size:
            pix = np.nonzero(np.isin(labels, np.asarray(segs, dtype=int)))[0]
    return pix


def _match(mz_sorted: np.ndarray, target: float, tol_ppm: float) -> int:
    """Index of the m/z in ``mz_sorted`` nearest ``target`` within ``tol_ppm``, else -1."""
    if mz_sorted.size == 0 or target <= 0:
        return -1
    i = int(np.searchsorted(mz_sorted, target))
    best, best_d = -1, np.inf
    for j in (i - 1, i):
        if 0 <= j < mz_sorted.size:
            d = abs(mz_sorted[j] - target) / target * 1e6
            if d <= tol_ppm and d < best_d:
                best, best_d = j, d
    return best


def _match_values(mz_sorted: np.ndarray, val_sorted: np.ndarray, targets: np.ndarray,
                  tol_ppm: float) -> np.ndarray:
    """Vectorized nearest-within-tol match: for every target, the ``val`` of the m/z in
    ``mz_sorted`` nearest it within ``tol_ppm`` (NaN when none). Bit-for-bit equivalent to
    calling :func:`_match` per target — including its prefer-the-lower-index-on-tie rule —
    but O((S+T)·logS) instead of a Python loop over S×T (≈58× faster at 100 samples ×
    10k targets; matching ran on the GUI thread). ``mz_sorted`` must be ascending."""
    targets = np.asarray(targets, dtype=float)
    out = np.full(targets.shape, np.nan, dtype=float)
    if mz_sorted.size == 0:
        return out
    pos = np.searchsorted(mz_sorted, targets)
    lo = np.clip(pos - 1, 0, mz_sorted.size - 1)
    hi = np.clip(pos, 0, mz_sorted.size - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        # boundaries: j=pos-1 invalid when pos==0; j=pos invalid when pos==size → distance ∞
        dlo = np.where(pos - 1 >= 0, np.abs(mz_sorted[lo] - targets) / targets * 1e6, np.inf)
        dhi = np.where(pos < mz_sorted.size, np.abs(mz_sorted[hi] - targets) / targets * 1e6, np.inf)
    pick_lo = dlo <= dhi                                  # _match checks j=i-1 first (ties → lower)
    best_idx = np.where(pick_lo, lo, hi)
    best_d = np.where(pick_lo, dlo, dhi)
    ok = (best_d <= tol_ppm) & (targets > 0)
    out[ok] = val_sorted[best_idx[ok]]
    return out


def _cluster_center(mzs: np.ndarray, ws: np.ndarray, center: str = "apex") -> float:
    """Representative m/z for one consensus cluster.

    ``'apex'`` (default) returns the m/z of the most-intense member — a **real observed
    peak from a real slide**, so the emitted target always sits on a measured apex. This is
    the SCiLS-style "collapse to local max", computed once and shared across the cohort.
    ``'centroid'`` returns the legacy intensity-weighted mean, which can fall *between* two
    slides' apexes (the off-apex "blank ion image" artifact) and is biased toward the
    highest-signal samples — kept only for bit-for-bit reproducibility of pre-2026-07 runs."""
    if center == "centroid":
        return float(np.average(mzs, weights=ws) if ws.sum() > 0 else mzs.mean())
    return float(mzs[int(np.argmax(ws))] if ws.sum() > 0 else np.median(mzs))


def consensus_targets(refs, tol_ppm: float = CONSENSUS_TOL_PPM, min_prevalence: float = 0.0,
                      value: str = "rel_intensity", sessions: dict | None = None, *,
                      center: str = "apex", deisotope: bool = True) -> list[float]:
    """A shared feature axis built from the union of every sample's peaks.

    Peaks across all samples are sorted and greedily clustered (a new cluster starts
    whenever the gap to the previous m/z exceeds ``tol_ppm`` *or* the cluster would span
    more than ``tol_ppm`` end-to-end — the width cap stops single-linkage chaining from
    fusing distinct ions in a dense spectrum); each cluster yields one representative m/z.

    ``center`` picks that representative: ``'apex'`` (default) uses the most-intense pooled
    peak's m/z — a real measured apex, so extracting the target on any slide lands on a real
    peak rather than the valley between two — while ``'centroid'`` reproduces the legacy
    intensity-weighted mean (off-apex, high-signal-biased; reproducibility only). ``deisotope``
    (default on) drops ¹³C M+1/M+2 satellites via :func:`isotopes.deisotope`, so an isotope
    like the PI 38:4 M+1 near 886.55 never becomes a standalone monoisotopic target.

    ``min_prevalence`` (0..1) drops clusters seen in fewer than that fraction of samples — so
    you can keep only ions common to, say, half the cohort. Use this when there's no predefined
    target list to compare on. Pass a shared ``sessions`` cache (path → session dict, mutated in
    place) to avoid re-parsing the same JSONs that :func:`batch_feature_table` reads moments
    later."""
    pts = []   # (mz, value, sample_index)
    n = 0
    cache: dict[str, dict | None] = sessions if sessions is not None else {}
    for si, r in enumerate(refs):
        if r.session_path not in cache:
            try:
                cache[r.session_path] = session.load_session(r.session_path)
            except (OSError, json.JSONDecodeError, ValueError):
                cache[r.session_path] = None
        data = cache[r.session_path]
        if data is None:
            continue
        mz, val = _session_peaks(data, value, scope=r.scope())
        if mz.size:
            n += 1
            for m, v in zip(mz, val):
                pts.append((float(m), float(v), si))
    if not pts or n == 0:
        return []
    pts.sort(key=lambda t: t[0])
    clusters: list[list[tuple]] = [[pts[0]]]
    for p in pts[1:]:
        cl = clusters[-1]
        gap_ok = (p[0] - cl[-1][0]) / p[0] * 1e6 <= tol_ppm          # close to the previous peak
        width_ok = (p[0] - cl[0][0]) / p[0] * 1e6 <= tol_ppm         # cluster stays ≤ tol wide
        if gap_ok and width_ok:
            cl.append(p)
        else:
            clusters.append([p])
    out = []
    for cl in clusters:
        prevalence = len({p[2] for p in cl}) / n
        if prevalence + 1e-9 < min_prevalence:
            continue
        mzs = np.array([p[0] for p in cl]); ws = np.array([p[1] for p in cl])
        out.append({"mz": round(_cluster_center(mzs, ws, center), 4),
                    "intensity": float(ws.sum())})    # cluster signal, for deisotoping
    if deisotope and len(out) > 1:
        from . import isotopes                          # lazy: keeps masses/scipy off cohort import
        out, _ = isotopes.deisotope(out, charge=1, tol_ppm=tol_ppm)
    return [d["mz"] for d in out]


def features_present(refs, targets, tol_ppm: float = 20.0, *, value: str = "rel_intensity",
                     sessions: dict | None = None) -> tuple[int, int]:
    """Cheap pre-flight for a pooled embedding / comparison: do any of ``targets`` actually
    occur in the cohort's peaks? Reads only the saved **session peak lists** (no cube load),
    so it can run before a long pooling job. Returns ``(n_with_peaks, n_overlapping)`` — how
    many of ``refs`` exposed any peaks at all, and how many of those carried ≥1 target within
    ``tol_ppm``. A feature set foreign to these slides → ``n_overlapping == 0`` while
    ``n_with_peaks > 0``, the classic "list from another dataset" mistake. Stops early once one
    overlap is found. ``sessions`` is an optional path→session cache (mutated in place) so it
    can reuse :func:`consensus_targets`' reads.

    This is a *proxy*: pixel extraction reads the raw cube, which can carry signal at an m/z
    that was never picked as a peak — so a zero here means "likely all-zero extraction", not a
    guarantee. Callers should warn, not hard-block."""
    tgt = np.asarray([float(t) for t in targets], dtype=float)
    if tgt.size == 0:
        return 0, 0
    cache: dict[str, dict | None] = sessions if sessions is not None else {}
    n_with = 0
    for r in refs:
        if r.session_path not in cache:
            try:
                cache[r.session_path] = session.load_session(r.session_path)
            except (OSError, json.JSONDecodeError, ValueError):
                cache[r.session_path] = None
        data = cache[r.session_path]
        if data is None:
            continue
        mz, val = _session_peaks(data, value, scope=r.scope())
        if mz.size == 0:
            continue
        n_with += 1
        if np.isfinite(_match_values(mz, val, tgt, tol_ppm)).any():
            return n_with, 1                       # ≥1 overlap → caller won't warn; stop early
    return n_with, 0


def _normalize_samples(values: np.ndarray, method: str) -> np.ndarray:
    """Cross-slide (per-sample / per-row) normalization of a sample×feature matrix, so a
    group comparison isn't biased by slide-to-slide total-signal differences. ``method``:

    * ``none``   — leave values as-is.
    * ``median`` — divide each sample by its own median (of positive features), rescaled to
      the cohort-mean median → samples share a common central intensity (robust default).
    * ``sum`` / ``tic`` — divide each sample by its feature sum (total-ion equivalent),
      rescaled to the cohort-mean sum.

    NaNs (a feature absent in a sample) are ignored when computing the factor and stay NaN.
    """
    method = (method or "none").lower()
    if method == "none":
        return values
    X = np.array(values, dtype=float)
    with np.errstate(invalid="ignore"):
        if method == "median":
            pos = np.where(X > 0, X, np.nan)
            factor = np.nanmedian(pos, axis=1)
        elif method in ("sum", "tic"):
            factor = np.nansum(X, axis=1)
        else:
            return values
    factor = np.where(np.isfinite(factor) & (factor > 0), factor, np.nan)
    scale = np.nanmean(factor)
    if not np.isfinite(scale) or scale <= 0:
        return values
    return X / factor[:, None] * scale


def batch_feature_table(refs, targets, tol_ppm: float = 20.0, value: str = "rel_intensity",
                        normalize: str = "none", missing: str = "nan",
                        sessions: dict | None = None):
    """A sample x feature table built from session metadata — **no dataset loads**.

    ``refs`` are :class:`SampleRef`; ``targets`` an iterable of m/z. Each cell is the
    sample's saved ``value`` for the peak nearest that target within ``tol_ppm``, or — when
    the sample has no peak there — NaN (``missing='nan'``, the default) or 0.0
    (``missing='zero'``). For MSI 'no saved peak near m/z X' usually means *below detection*
    (≈0), not missing-at-random, so ``'zero'`` removes the presence bias that NaN-dropping
    introduces into group means/fold-change; ``'nan'`` is kept as the default for backward
    compatibility (pair it with :func:`group_comparison`'s ≥2-per-group detection gate).
    Zero-fill is applied *before* ``normalize`` so the median/sum factors aren't shifted.

    ``normalize`` applies cross-slide per-sample normalization (``none`` / ``median`` /
    ``sum``; see :func:`_normalize_samples`). Returns a pandas DataFrame indexed by sample
    name with a ``group`` column followed by one column per target (named ``mz_xxx.xxxx``).
    Duplicate sample names get a ``#2`` suffix so rows never collide. Samples that couldn't
    be read are recorded in ``df.attrs['skipped']`` (``missing/corrupt session`` vs ``no
    saved peaks``) and all-absent rows in ``df.attrs['no_signal']`` — so the caller can show
    'k of N skipped' instead of silently comparing a shrunken cohort. Pass a shared
    ``sessions`` cache to avoid re-parsing JSONs already read by :func:`consensus_targets`.
    """
    import pandas as pd

    targets = [float(t) for t in targets]
    targets_arr = np.asarray(targets, dtype=float)
    cols = feature_columns(targets)
    rows, index, used, included = [], [], {}, []
    skipped: list[dict] = []                 # samples dropped before the table (reason recorded)
    cache: dict[str, dict | None] = sessions if sessions is not None else {}
    for r in refs:
        if r.session_path not in cache:
            try:
                cache[r.session_path] = session.load_session(r.session_path)
            except (OSError, json.JSONDecodeError, ValueError):
                cache[r.session_path] = None
        data = cache[r.session_path]
        nm = r.name or session._display_name(r.source)
        if data is None:
            skipped.append({"name": nm, "reason": "missing/corrupt session"})
            continue
        mz, val = _session_peaks(data, value, scope=r.scope())
        if mz.size == 0:
            skipped.append({"name": nm, "reason": "no saved peaks"})
            continue
        vals = _match_values(mz, val, targets_arr, tol_ppm)
        row = {"group": r.group or ""}
        row.update(dict(zip(cols, (float(v) for v in vals))))
        used[nm] = used.get(nm, 0) + 1
        if used[nm] > 1:
            nm = f"{nm} #{used[nm]}"
        rows.append(row); index.append(nm); included.append(r)
    df = pd.DataFrame(rows, index=index, columns=["group", *cols])
    if cols:
        M = df[cols].to_numpy(dtype=float)
        no_signal = [index[i] for i in range(len(index)) if np.isnan(M[i]).all()]
        if missing == "zero":
            M = np.where(np.isnan(M), 0.0, M)            # absent ion → below-detection ≈ 0
        if normalize and normalize != "none":
            M = _normalize_samples(M, normalize)
        df[cols] = M
    else:
        no_signal = []
    df.attrs["targets"] = targets
    df.attrs["value"] = value
    df.attrs["tol_ppm"] = float(tol_ppm)
    df.attrs["normalize"] = normalize
    df.attrs["missing"] = missing
    df.attrs["skipped"] = skipped
    df.attrs["no_signal"] = no_signal
    df.attrs["refs"] = included     # per-row SampleRef, for batch_design row alignment
    return df


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def batch_design(table, *, protect=("group",), numeric_meta=()):
    """Build ComBat inputs aligned **row-for-row** to a :func:`batch_feature_table`.

    Reads the per-row :class:`SampleRef` list the table records in ``df.attrs['refs']``.
    Returns ``(batch_labels, covariate_design, biology_labels)``:

    * ``batch_labels`` — each sample's ``SampleRef.batch`` (``""`` when unassigned).
    * ``covariate_design`` — a float ``(n_rows, k)`` design of the *protected* factors: a
      drop-first one-hot of the biological ``group`` (when ``'group'`` is in ``protect``)
      plus any ``numeric_meta`` keys coerced to float, or ``None`` when there is nothing
      to protect. Pass it as ``covariates=`` to :func:`smile_msi.batchfx.combat` so the
      biological contrast is preserved via ComBat's covariate model matrix (not
      reComBat's regularized regression).
    * ``biology_labels`` — each sample's ``group`` (for silhouette-by-biology QC).
    """
    refs = list(table.attrs.get("refs") or [])
    batch = np.array([(getattr(r, "batch", "") or "") for r in refs], dtype=object)
    biology = np.array([(r.group or "") for r in refs], dtype=object)
    cols = []
    if "group" in protect:
        levels = sorted({str(g) for g in biology})
        for g in levels[1:]:                       # drop-first to avoid collinearity with batch
            cols.append((biology == g).astype(float))
    for key in numeric_meta:
        cols.append(np.array([_to_float(r.meta.get(key)) for r in refs], dtype=float))
    cov = np.column_stack(cols) if cols else None
    return batch, cov, biology


def correct_batches(refs, targets, *, protect=("group",), numeric_meta=(), mean_only=False,
                    tol_ppm=20.0, normalize="none", missing="zero", random_state=0,
                    sessions=None):
    """Region/sample-summarized ComBat batch correction over a cohort.

    Builds the sample x feature table (no cube loads), removes cross-batch technical shifts
    with :func:`smile_msi.batchfx.combat` while protecting the biological ``group`` contrast,
    and returns ``(corrected_table, CorrectionReport, MixingReport)``. The corrected table
    mirrors :func:`batch_feature_table` (a ``group`` column + one ``mz_xxxx`` column per
    target) with the feature block replaced by corrected values.

    ``missing='zero'`` (the default here) treats an absent ion as below-detection ≈ 0 so the
    ComBat input carries no NaNs. With fewer than two distinct batches this is a no-op
    (``combat`` returns the input unchanged) and the QC reports reflect that.
    """
    from . import batchfx

    table = batch_feature_table(refs, targets, tol_ppm=tol_ppm, value="rel_intensity",
                                normalize=normalize, missing=missing, sessions=sessions)
    feat_cols = [c for c in table.columns if c != "group"]
    X = table[feat_cols].to_numpy(dtype=float)
    batch, cov, biology = batch_design(table, protect=protect, numeric_meta=numeric_meta)
    Xc = batchfx.combat(X, batch, covariates=cov, mean_only=mean_only)
    out = table.copy()
    out[feat_cols] = Xc
    out.attrs.update(table.attrs)
    summary = batchfx.correction_summary(X, Xc, batch)
    mixing = batchfx.compare_mixing(X, Xc, batch, biology=biology, random_state=random_state)
    out.attrs["batch_correction"] = {
        "n_batches": int(len({b for b in batch.tolist() if b})),
        "protected": list(protect) + list(numeric_meta),
        "mean_only": bool(mean_only),
        "var_reduction": summary.var_reduction,
        "rejection_before": mixing.rejection_before,
        "rejection_after": mixing.rejection_after,
    }
    return out, summary, mixing


def feature_columns(targets) -> list:
    """One **unique** column name per target, in order.

    ``f"mz_{t:.4f}"`` collides whenever two targets agree to four decimal places — which a
    consensus axis can emit for a pair of tightly clustered peaks. A duplicate column name is
    not a cosmetic problem: ``table[feat_cols]`` then expands each duplicated name into every
    matching column, so the feature/column counts silently disagree. Downstream that either
    raises ("All arrays must be of the same length") or, worse, doubles that feature's
    per-group ``n`` in :func:`group_comparison` — pseudoreplication introduced by a format
    string. Collisions get a ``#2`` suffix; :func:`targets_from_columns` reverses it.
    """
    names: list[str] = []
    seen: dict[str, int] = {}
    for t in targets:
        base = f"mz_{float(t):.4f}"
        seen[base] = seen.get(base, 0) + 1
        names.append(base if seen[base] == 1 else f"{base}#{seen[base]}")
    return names


def targets_from_columns(feat_cols) -> list:
    """The m/z encoded in :func:`feature_columns` names, ``#n`` disambiguators stripped. Only a
    fallback — a table built by this module carries the exact targets in ``attrs['targets']``,
    which are not rounded to the column name's four decimal places."""
    return [float(c.split("_", 1)[1].split("#", 1)[0]) for c in feat_cols]


def _check_feature_columns(feat_cols, targets) -> None:
    """Fail loud when a table's feature columns can't be trusted.

    Duplicate names make ``table[cols]`` return more columns than it was asked for, so a
    per-feature loop reads a ``(n × k)`` block as one feature: ``group_comparison`` then counted
    ``k × n`` replicates instead of ``n``, and the nested path died inside pandas with "All
    arrays must be of the same length". Both are silent-corruption shapes, so refuse the table
    rather than let either happen."""
    if len(set(feat_cols)) != len(feat_cols):
        dupes = sorted({c for c in feat_cols if list(feat_cols).count(c) > 1})
        raise ValueError(f"feature columns are not unique ({', '.join(dupes[:3])}) — a "
                         f"duplicated column silently multiplies that feature's replicate "
                         f"count; rebuild the table with batch_feature_table/pseudobulk_table")
    if len(targets) != len(feat_cols):
        raise ValueError(f"table has {len(feat_cols)} feature columns but {len(targets)} "
                         f"targets — rebuild it with batch_feature_table/pseudobulk_table")


def _pair_keys(table, pair_by: str):
    """Per-row pairing key (lower-cased ``SampleRef.meta[pair_by]``), as a ``{row_label: key}``
    map aligned to ``table.index`` via ``table.attrs['refs']``. Rows with a blank/absent key
    map to ``""`` (unpairable). Raises when the refs aren't row-aligned (the table was
    rebuilt/re-sorted without them) so a paired comparison never silently mispairs samples."""
    refs = list(table.attrs.get("refs") or [])
    if len(refs) != len(table):
        raise ValueError("paired comparison needs table.attrs['refs'] aligned to rows — "
                         "rebuild the table with batch_feature_table and don't re-sort it")
    keys = [str((getattr(r, "meta", {}) or {}).get(pair_by, "")).strip().lower() for r in refs]
    return dict(zip(list(table.index), keys))


def group_comparison(table, group_a: str, group_b: str, method: str = "mwu",
                     min_per_group: int = 2, paired: bool = False,
                     pair_by: str | None = None):
    """Per-feature comparison of two cohort groups, sample-as-replicate.

    ``table`` is a :func:`batch_feature_table`. For each feature column it compares the
    per-sample values of ``group_a`` vs ``group_b`` and reports ``mz, n_A, n_B, mean_A,
    mean_B, log2_fc, p_value, q_value, n_detected, prevalence``. ``prevalence`` is the
    fraction of the compared samples in which the ion is actually detected (value > 0) — a
    cross-sample **reproducibility** axis orthogonal to significance and to the mass-based ID
    confidence, so a biomarker can be ranked by *how many samples it shows up in*, not only by
    its p-value. A feature is **tested only** when it has
    ``≥ min_per_group`` detections in *each* group and the test returns a finite p — a
    detection gate that doubles as the small-n guard. Untested features (too sparse, or a
    degenerate/zero-variance column) get ``p_value = q_value = NaN`` rather than a
    misleading 1.0, and — crucially — are **excluded from the BH-FDR denominator** so that
    adding more sparse features to a bigger cohort can't paradoxically inflate q and hide a
    real biomarker. ``df.attrs`` carries ``n_testable`` / ``n_features`` and, when nothing
    is testable, a ``warning``. Sorted by ascending p (untested rows last).

    ``method``: ``'mwu'`` (default, Mann-Whitney U), ``'welch'`` (unequal-variance t) or
    ``'student'`` (pooled-variance t). ``min_per_group`` floors at 2 (a 1-vs-N rank test is
    vacuous — two-sided MWU can't reach p≤0.05 below 4-vs-4).

    **Paired designs** (``paired=True``) — for repeated-measures data where each subject
    contributes to *both* groups (e.g. left/right nerves, treated/control of one donor,
    a compartment pair within a slide). Samples are matched across the two groups by
    ``pair_by`` — a :class:`SampleRef.meta` key (e.g. ``"subject"``/``"donor"``); only
    subjects present in **both** groups with a finite value form a pair. The test becomes
    the **Wilcoxon signed-rank** (``method`` in mwu/rank) or the **paired t-test**
    (``method`` in welch/student/t) on the within-pair values, and ``log2_fc`` is the mean
    within-pair log2 ratio. ``n_A``/``n_B`` report the number of *pairs*; ``df.attrs['paired']``
    and ``['pair_by']`` record the design. A paired test on matched subjects removes
    between-subject variance, so it is both more correct and more powerful than the unpaired
    default for these designs. Requires ``pair_by`` and a row-aligned ``table.attrs['refs']``.

    ``log2_fc`` is pseudocount-regularized with a data-scaled τ (5th percentile of the two
    groups' nonzero intensities, :func:`smile_msi.spatial._fold_tau`) so two near-noise means
    read as ``log2 FC ≈ 0`` instead of a spurious ratio (KNOWN_ISSUES artifact #2), matching
    the per-slide fold metrics rather than a fixed tiny epsilon."""
    import pandas as pd
    from scipy.stats import mannwhitneyu, ttest_ind, ttest_rel, wilcoxon

    from .spatial import _bh_fdr, _fold_tau

    m = (method or "mwu").lower()
    if paired:
        if m in ("mwu", "mannwhitney", "mann-whitney", "rank", "wilcoxon", "signed-rank"):
            def test(a, b):
                return wilcoxon(a, b, zero_method="wilcox")[1]
            test_name = "Wilcoxon signed-rank (paired)"
        elif m in ("welch", "student", "pooled", "t", "ttest", "t-test", "paired-t",
                   "ttest_rel", "rel"):
            def test(a, b):
                return ttest_rel(a, b).pvalue
            test_name = "Paired t-test"
        else:
            raise ValueError(f"unknown paired method {method!r}; use 'wilcoxon' or 'paired-t'")
        if not pair_by:
            raise ValueError("paired=True requires pair_by=<metadata key> to match samples "
                             "across the two groups")
    elif m in ("mwu", "mannwhitney", "mann-whitney", "rank"):
        def test(a, b):
            # When every observation across both groups is tied (a constant, detected
            # feature with no between-group difference), the U-statistic's variance is 0.
            # scipy's normal approximation returns p=1.0 on some versions but NaN on newer
            # ones (≥1.16) — and a NaN here flips the feature to "untestable" (see the
            # np.isfinite gate below), silently changing n_testable on a dependency bump.
            # There is no evidence of a location shift, so report 1.0 deterministically.
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            if a.size and b.size and np.ptp(np.concatenate([a, b])) == 0.0:
                return 1.0
            return mannwhitneyu(a, b, alternative="two-sided")[1]
        test_name = "Mann-Whitney U"
    elif m in ("welch", "t", "ttest", "t-test"):
        test, test_name = lambda a, b: ttest_ind(a, b, equal_var=False).pvalue, "Welch's t-test"
    elif m in ("student", "pooled", "ttest_equal"):
        test, test_name = lambda a, b: ttest_ind(a, b, equal_var=True).pvalue, "Student's t-test"
    else:
        raise ValueError(f"unknown method {method!r}; use 'mwu', 'welch', or 'student'")
    min_n = max(2, int(min_per_group))

    feat_cols = [c for c in table.columns if c != "group"]
    targets = table.attrs.get("targets") or targets_from_columns(feat_cols)
    _check_feature_columns(feat_cols, targets)
    A = table[table["group"] == group_a]
    B = table[table["group"] == group_b]
    # Data-scaled fold pseudocount over the two groups' feature block (one source shared with
    # the per-slide fold metrics), NaN/zero-safe. Empty block → 1.0.
    tau = _fold_tau(pd.concat([A, B])[feat_cols].to_numpy(dtype=float)) if feat_cols else 1.0

    # For a paired design, precompute each group's {pair_key: row_label}. First occurrence of a
    # key in a group wins; a key repeated within one group is ambiguous (counted for a warning).
    a_map = b_map = None
    ambiguous_pairs = 0
    if paired:
        key_by_label = _pair_keys(table, pair_by)
        a_map, b_map = {}, {}
        for dst, sub in ((a_map, A), (b_map, B)):
            for lbl in sub.index:
                k = key_by_label.get(lbl, "")
                if not k:
                    continue
                if k in dst:
                    ambiguous_pairs += 1
                    continue
                dst[k] = lbl
        shared_keys = [k for k in a_map if k in b_map]

    n_compared = int(len(A) + len(B))                    # cohort samples in this A-vs-B contrast
    mz_l, nA_l, nB_l, mA_l, mB_l, fc_l, p_l, tested_l, prev_l, det_l = (
        [], [], [], [], [], [], [], [], [], [])
    for t, c in zip(targets, feat_cols):
        fc, p, is_tested = np.nan, np.nan, False
        # Cross-sample reproducibility: in how many of the compared samples is this ion actually
        # detected (value > 0)? A NaN/zero cell is "below detection". Counted over the raw group
        # columns so it's independent of paired/unpaired and of the missing='nan'/'zero' fill —
        # an orthogonal trust axis: a biomarker seen in 10/11 samples is far more credible than a
        # significant hit riding on 2 detections.
        col_a = A[c].to_numpy(dtype=float); col_b = B[c].to_numpy(dtype=float)
        det = int((col_a > 0).sum() + (col_b > 0).sum())
        prev = (det / n_compared) if n_compared else np.nan
        if paired:
            av, bv = [], []
            for k in shared_keys:
                x = float(A.at[a_map[k], c]); y = float(B.at[b_map[k], c])
                if np.isnan(x) or np.isnan(y):
                    continue
                av.append(x); bv.append(y)
            a = np.asarray(av, dtype=float); b = np.asarray(bv, dtype=float)
            mA = float(a.mean()) if a.size else np.nan
            mB = float(b.mean()) if b.size else np.nan
            if a.size >= min_n:
                fc = float(np.mean(np.log2((b + tau) / (a + tau))))
                # A flat feature has zero within-pair difference everywhere. scipy's Wilcoxon
                # returns p=1.0 (not a raise) for that degenerate case; running it would count
                # the feature as "tested" and pad the BH denominator, so gate on real signal.
                if np.any(a != b):
                    try:
                        p = float(test(a, b))
                    except ValueError:               # all retained diffs zero → undefined
                        p = np.nan
                    if not np.isfinite(p):
                        p = np.nan
                    is_tested = bool(np.isfinite(p))
        else:
            a = A[c].to_numpy(dtype=float); a = a[~np.isnan(a)]
            b = B[c].to_numpy(dtype=float); b = b[~np.isnan(b)]
            mA = float(a.mean()) if a.size else np.nan
            mB = float(b.mean()) if b.size else np.nan
            if a.size >= min_n and b.size >= min_n:
                fc = float(np.log2((mB + tau) / (mA + tau)))
                try:
                    p = float(test(a, b))
                except ValueError:                   # all-equal / degenerate column
                    p = np.nan
                if not np.isfinite(p):               # zero-variance / undefined → untestable
                    p = np.nan
                is_tested = bool(np.isfinite(p))
        mz_l.append(float(t)); nA_l.append(int(a.size)); nB_l.append(int(b.size))
        mA_l.append(mA); mB_l.append(mB); fc_l.append(fc); p_l.append(p); tested_l.append(is_tested)
        prev_l.append(prev); det_l.append(det)
    p_arr = np.array(p_l, dtype=float)
    tested = np.array(tested_l, dtype=bool)
    q_arr = np.full(p_arr.shape, np.nan)
    if tested.any():
        q_arr[tested] = _bh_fdr(p_arr[tested])           # BH over tested features ONLY
    df = pd.DataFrame({"mz": mz_l, "n_A": nA_l, "n_B": nB_l, "mean_A": mA_l, "mean_B": mB_l,
                       "log2_fc": fc_l, "p_value": p_arr, "q_value": q_arr,
                       "n_detected": det_l, "prevalence": prev_l})
    df.attrs["a_label"], df.attrs["b_label"] = group_a, group_b
    df.attrs["test"] = test_name
    df.attrs["n_features"] = int(len(p_arr))
    df.attrs["n_testable"] = int(tested.sum())
    df.attrs["n_compared"] = n_compared
    df.attrs["n_a_total"], df.attrs["n_b_total"] = int(len(A)), int(len(B))
    df.attrs["paired"] = bool(paired)
    if paired:
        df.attrs["pair_by"] = pair_by
        df.attrs["n_pairs"] = int(len(shared_keys))
        if ambiguous_pairs:
            df.attrs["pair_warning"] = (f"{ambiguous_pairs} sample(s) shared a '{pair_by}' value "
                                        "within one group; only the first of each was paired.")
    if not tested.any():
        if paired:
            df.attrs["warning"] = (
                f"No feature had ≥{min_n} complete pairs matched on '{pair_by}' between "
                f"{group_a} (n={len(A)}) and {group_b} (n={len(B)}) — no valid paired test. "
                f"{len(shared_keys)} subject(s) present in both groups.")
        else:
            df.attrs["warning"] = (f"No feature had ≥{min_n} detections in both groups "
                                   f"({group_a} n={len(A)} vs {group_b} n={len(B)}) — no valid test.")
    return df.sort_values("p_value", kind="stable", na_position="last", ignore_index=True)


# --------------------------------------------------------------------------- #
# Nested designs — subject × compartment pseudobulk, then a hierarchical test
# --------------------------------------------------------------------------- #
#: ``subject_by`` sentinel: each source file is one subject. Right when the design is one
#: nerve/donor per slide, and it needs no metadata import.
SUBJECT_BY_SLIDE = "__slide__"

#: ``subject_by`` prefix: take the subject from a token of the region name — ``'region:first'``
#: turns ``"S01 endo"`` into ``s01``. Right when several subjects were sectioned onto ONE slide,
#: where the slide identifies nothing.
SUBJECT_BY_REGION = "region:"


def _name_tokens(name: str) -> list:
    """Whitespace-, ``-`` and ``_``-delimited tokens of a label, empties dropped."""
    return [p for p in re.split(r"[\s_\-]+", (name or "").strip()) if p]


def subject_id(ref, subject_by: str) -> str:
    """The subject a region-sample belongs to. Lower-cased; ``''`` when unresolvable, which
    callers must treat as an error rather than as a subject named ``''``.

    ``subject_by``:

    * :data:`SUBJECT_BY_SLIDE` — the source file. One nerve per imzML.
    * ``'region:first'`` / ``'region:last'`` — a token of the region name, for when several
      subjects share one slide and the ROI label is the only thing that names them
      (``"S01 endo"`` → ``s01``).
    * anything else — ``meta[subject_by]``, the same key :func:`group_comparison`'s paired mode
      uses for ``pair_by``, and what the GUI's compartment mapping writes.

    Getting this wrong is the one error the whole nested module exists to prevent: if two nerves
    resolve to one subject their compartments are pooled, and if one nerve resolves to three
    subjects its compartments become independent replicates.
    """
    if subject_by == SUBJECT_BY_SLIDE:
        src = (getattr(ref, "source", "") or "")
        if src:
            return session._display_name(src).strip().lower()
        return (getattr(ref, "session_path", "") or "").strip().lower()
    if subject_by.startswith(SUBJECT_BY_REGION):
        mode = subject_by[len(SUBJECT_BY_REGION):]
        parts = _name_tokens(getattr(ref, "region", "") or "")
        if not parts:
            return ""
        if mode == "first":
            return parts[0].lower()
        if mode == "last":
            return parts[-1].lower()
        raise ValueError(f"unknown subject_by {subject_by!r}; use 'region:first' or 'region:last'")
    return str((getattr(ref, "meta", {}) or {}).get(subject_by, "")).strip().lower()


#: ``compartment_from`` prefix: read the compartment from ``SampleRef.meta[key]`` — what the
#: GUI's "Define compartments…" mapping writes, and the only source that survives region names
#: no rule can parse.
COMPARTMENT_BY_META = "meta:"


def compartment_id(ref, compartment_from: str = "exact") -> str:
    """The compartment a region-sample represents.

    A region is named *inside one slide*, so nothing stops the same anatomical compartment
    from being called ``"s01 endo"`` in one nerve and ``"s02 endo"`` in the next. A nested
    model needs ``endo`` to denote the same compartment in every subject, or each compartment
    appears in exactly one subject and the design is unestimable.

    ``compartment_from``:

    * ``'meta:<key>'`` — read ``meta[key]``. An explicit, per-sample mapping; the only source
      that copes with region names no rule can parse. Returns ``''`` when unset, which callers
      must treat as "this sample has no compartment", never as a compartment named ``''``.
    * ``'exact'`` — the region name verbatim. Right when regions were named identically
      across slides.
    * ``'last'`` / ``'first'`` — the last / first whitespace-, ``-`` or ``_``-delimited token,
      which collapses ``"s01 endo"`` and ``"s02 endo"`` onto ``endo``.

    The token rules are offered, not inferred: a silent guess at which token is the subject id
    would be a quiet way to merge two different compartments.
    """
    if compartment_from.startswith(COMPARTMENT_BY_META):
        key = compartment_from[len(COMPARTMENT_BY_META):]
        return str((getattr(ref, "meta", {}) or {}).get(key, "")).strip()
    name = (getattr(ref, "region", "") or "").strip()
    if not name or compartment_from == "exact":
        return name
    parts = _name_tokens(name)
    if not parts:
        return name
    if compartment_from == "last":
        return parts[-1].lower()
    if compartment_from == "first":
        return parts[0].lower()
    raise ValueError(f"unknown compartment_from {compartment_from!r}; use 'exact', 'last', "
                     f"'first' or 'meta:<key>'")


def _scope_rows_by_source(refs, sessions: dict | None = None):
    """``{source: pixel rows}`` — the union of each slide's region pixels across ``refs``.

    This is the on-tissue population a cross-sample normalization should take its mean scale
    over (see :meth:`smile_msi.msi.MSIDataset.norm_factors`). Resolved from the session JSONs
    alone, so it costs no dataset loads and stays independent of the cube loader's eviction
    order. A source contributing a whole-slide sample maps to ``None`` (no restriction).
    """
    cache = sessions if sessions is not None else {}
    acc: dict[str, list | None] = {}
    for r in refs:
        src = getattr(r, "source", "") or ""
        region = getattr(r, "region", "") or ""
        if not region:
            acc[src] = None                       # a whole-slide sample: the scope IS the slide
            continue
        if src in acc and acc[src] is None:
            continue
        sp = getattr(r, "session_path", "") or ""
        data = cache.get(sp)
        if data is None:
            try:
                data = session.load_session(sp)
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            cache[sp] = data
        pix = region_pixels(data, region)
        if pix is None or pix.size == 0:
            continue
        acc.setdefault(src, []).append(pix)
    return {s: (np.unique(np.concatenate(v)) if isinstance(v, list) and v else None)
            for s, v in acc.items()}


def pseudobulk_table(refs, targets, tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic",
                     summary: str = "median", reduce: str = "sum", loader=None,
                     norm_scope: str = "tissue", sessions: dict | None = None,
                     progress=None):
    """A sample × feature table whose cells are re-extracted from **pixels**, not read from
    the session's saved peaks — one row per :class:`SampleRef`, one summary per feature.

    :func:`batch_feature_table` is free (zero dataset loads) but can only report the value a
    region's peaks were *saved* with, which is a mean. When the analysis is a nested one —
    several compartments per subject, tested with :mod:`smile_msi.hierstats` — the summary
    statistic matters: a compartment with a few matrix hot-pixels or an ablation artifact
    drags its mean far more than its median. ``summary='median'`` (the default here, unlike
    the saved-peak path) is the robust choice; ``'mean'`` reproduces Cardinal's ``meansTest``
    convention exactly.

    Each ref is loaded once per source via :class:`SectionLoader` (lazy, so only the region's
    pixels stream). Refs whose imzML is missing/moved, or whose region no longer resolves to
    pixels, land in ``df.attrs['skipped']`` rather than failing the run.

    **Normalization is deliberately not delegated to** ``features_for_rows(norm=...)``. On a
    lazy dataset that method computes its per-pixel scale over *only the rows it was asked
    for* (``msi.py``'s streamed branch), so each compartment would be divided by its own mean
    TIC — self-referential, and it would quietly flatten exactly the between-compartment
    differences this table exists to model (TRUST_AUDIT_2026-06-25.md, Confounder #1). We
    extract raw (``norm='none'``) and divide by factors computed over a *slide-wide* scope.

    ``norm_scope`` picks that scope. ``'tissue'`` (the default) takes the mean scale over the
    union of the slide's region pixels; ``'slide'`` takes it over every pixel, background
    included. The scale enters each normalized intensity as one multiplicative constant per
    slide — identical across that slide's compartments, so within-subject contrasts never
    care, but it differs between slides according to how much empty matrix each section sits
    in. A mixed model's ``(1 | subject)`` term absorbs that on the log2 scale; a stratified
    test does not, and if one group's sections carry systematically more background than the
    other's it becomes a confound rather than noise. ``'tissue'`` removes the dependence at
    the source.

    Returns a DataFrame shaped like :func:`batch_feature_table` — indexed by sample name, a
    ``group`` column then one column per target, with ``attrs['targets']`` / ``attrs['refs']``
    row-aligned — so it drops straight into :func:`nested_comparison`.
    ``attrs['summary']`` / ``attrs['norm']`` / ``attrs['norm_scope']`` record how the cells
    were made.
    """
    import pandas as pd

    if summary not in ("median", "mean"):
        raise ValueError(f"unknown summary {summary!r}; use 'median' or 'mean'")
    if norm_scope not in ("tissue", "slide"):
        raise ValueError(f"unknown norm_scope {norm_scope!r}; use 'tissue' or 'slide'")
    targets = [float(t) for t in targets]
    targets_arr = np.asarray(targets, dtype=float)
    cols = feature_columns(targets)
    agg = np.nanmedian if summary == "median" else np.nanmean

    refs = list(refs)
    own_loader = loader is None
    loader = loader or SectionLoader(dense=False)   # lazy: stream the region's pixels only
    if own_loader:
        loader.prime(refs)                          # evict each cube as its last region is done
    # Resolved from session JSONs, before any cube is touched — so the scope of a slide's
    # normalization never depends on the order the loader happens to visit its regions in.
    scope_rows = (_scope_rows_by_source(refs, sessions) if norm_scope == "tissue"
                  and norm and norm != "none" else {})
    rows, index, used, included = [], [], {}, []
    skipped: list[dict] = []
    n_pixels: list[int] = []
    try:
        for i, r in enumerate(refs):
            nm = r.name or session._display_name(r.source)
            if progress is not None:
                progress(i, len(refs), nm)
            got = loader.load(r)
            if got is None:
                skipped.append({"name": nm, "reason": "source missing / region not resolvable"})
                continue
            ds, pix = got
            if pix is None:
                pix = np.arange(int(ds.n_pixels), dtype=int)
            if pix.size == 0:
                skipped.append({"name": nm, "reason": "region has no pixels"})
                continue
            X = ds.features_for_rows(targets_arr, pix, tol_ppm=tol_ppm, reduce=reduce,
                                     norm="none")
            if norm and norm != "none":
                scope = scope_rows.get(getattr(r, "source", "") or "")
                facs = np.asarray(ds.norm_factors(norm, scope=scope), dtype=float)[pix]
                X = X / facs[:, None]
            vals = agg(np.asarray(X, dtype=float), axis=0)
            row = {"group": r.group or ""}
            row.update(dict(zip(cols, (float(v) for v in vals))))
            used[nm] = used.get(nm, 0) + 1
            if used[nm] > 1:
                nm = f"{nm} #{used[nm]}"
            rows.append(row); index.append(nm); included.append(r); n_pixels.append(int(pix.size))
    finally:
        if own_loader:
            loader.close()

    df = pd.DataFrame(rows, index=index, columns=["group", *cols])
    df.attrs["targets"] = targets
    df.attrs["tol_ppm"] = float(tol_ppm)
    df.attrs["norm"] = norm
    df.attrs["norm_scope"] = norm_scope
    df.attrs["summary"] = summary
    df.attrs["value"] = f"{summary} of per-pixel {norm}-normalized intensity"
    df.attrs["skipped"] = skipped
    df.attrs["refs"] = included
    df.attrs["n_pixels"] = n_pixels      # per row: the pixels the summary was taken over
    return df


def nested_design(table, subject_by: str, compartment_from: str = "exact"):
    """``(subject, compartment)`` label arrays aligned to ``table``'s rows.

    The subject is :func:`subject_id` — the slide, or a metadata key (the same key
    :func:`group_comparison`'s paired mode uses for ``pair_by``). The compartment is
    :func:`compartment_id` — the region name, optionally reduced to one token so that a
    per-nerve region called ``"s01 endo"`` joins ``"s02 endo"`` under ``endo``.

    Raises when a row has no subject, because a nested model that silently dropped the nerve
    a compartment came from would be the very pseudoreplication it exists to prevent.
    """
    refs = list(table.attrs.get("refs") or [])
    if len(refs) != len(table):
        raise ValueError("a nested comparison needs table.attrs['refs'] aligned to rows — "
                         "rebuild the table and don't re-sort it")
    subject = np.array([subject_id(r, subject_by) for r in refs], dtype=object)
    compartment = np.array([compartment_id(r, compartment_from) for r in refs], dtype=object)
    missing = [str(lbl) for lbl, s in zip(table.index, subject) if not s]
    if missing:
        where = ("the slide each sample came from" if subject_by == SUBJECT_BY_SLIDE
                 else f"the '{subject_by}' metadata key")
        raise ValueError(
            f"{len(missing)} sample(s) could not be traced to a subject via {where} "
            f"(e.g. {', '.join(missing[:3])}) — every compartment must name the subject it "
            f"came from, or its pixels would be counted as an independent replicate.")
    return subject, compartment


def pseudobulk_tidy(table, subject_by: str, compartment_from: str = "exact"):
    """Melt a :func:`pseudobulk_table` (or :func:`batch_feature_table`) into one row per
    subject × compartment × feature — the shape a between-nerve export needs, as opposed
    to ``table`` itself, which is one row per sample with one column per feature.

    Adds ``subject``/``compartment`` via :func:`nested_design`, so every row can be traced
    back to the nerve and region it came from, alongside the sample's ``group`` and (when
    ``table`` carries it, i.e. it came from :func:`pseudobulk_table`) the pixel count the
    mean/median was taken over.
    """
    import pandas as pd

    subject, compartment = nested_design(table, subject_by, compartment_from)
    feat_cols = [c for c in table.columns if c != "group"]
    targets = table.attrs.get("targets") or targets_from_columns(feat_cols)
    n_pixels = table.attrs.get("n_pixels")

    long = table[feat_cols].copy()
    long.insert(0, "sample", table.index)
    long.insert(1, "subject", subject)
    long.insert(2, "compartment", compartment)
    long.insert(3, "group", table["group"].to_numpy())
    if n_pixels is not None and len(n_pixels) == len(table):
        long.insert(4, "n_pixels", n_pixels)

    id_vars = [c for c in long.columns if c not in feat_cols]
    out = long.melt(id_vars=id_vars, value_vars=feat_cols, var_name="_col",
                    value_name="intensity")
    out["mz"] = out["_col"].map(dict(zip(feat_cols, targets)))
    out = out.drop(columns="_col").reset_index(drop=True)
    out.attrs["value"] = table.attrs.get("value")
    out.attrs["norm"] = table.attrs.get("norm")
    out.attrs["norm_scope"] = table.attrs.get("norm_scope")
    out.attrs["summary"] = table.attrs.get("summary")
    out.attrs["subject_by"] = subject_by
    out.attrs["compartment_from"] = compartment_from
    return out


def nested_comparison(table, group_a: str, group_b: str, subject_by: str, model: str = "lmm",
                      compartments=None, method: str = "modt", transform: str = "log2",
                      min_subjects_per_group: int = 2, df_method: str = "bw",
                      compartment_from: str = "exact"):
    """Compare two groups across compartments with the **subject** as the unit of replication.

    ``table`` is a :func:`pseudobulk_table` (or a :func:`batch_feature_table`) whose rows are
    one subject × one compartment. ``subject_by`` names the :attr:`SampleRef.meta` key holding
    the subject id, so the three compartments of one nerve are recognised as one nerve.

    ``model``:

    * ``'lmm'`` (default) — one :func:`smile_msi.hierstats.nested_mixed_model` per feature,
      ``intensity ~ group * compartment + (1 | subject)``. Reports the group × compartment
      interaction, the group effect averaged over compartments, and the group's simple effect
      within each compartment. This is the model that answers "does the difference differ by
      compartment".
    * ``'stratified'`` — :func:`smile_msi.hierstats.stratified_comparison`: an independent
      two-group test inside each compartment (``method='modt'`` moderated-t by default;
      ``'welch'``/``'student'``/``'mwu'`` also available). Simpler to explain; the three
      contrasts are not independent of one another because they share subjects.

    ``transform='log2'`` (default) models ``log2(intensity + τ)`` with the same data-scaled τ
    every other fold metric in the app uses (:func:`smile_msi.spatial._fold_tau`). Both tests
    assume roughly normal residuals, and raw MSI intensities are strongly right-skewed, so the
    transform is on by default; the reported ``log2_fc__*`` columns are always computed from
    the **raw** means so they stay comparable with :func:`group_comparison`'s ``log2_fc``.

    ``df.attrs`` carries ``subject_by``, ``transform``, ``tau``, ``n_subjects``,
    ``subjects_a``/``subjects_b``, ``compartments``, ``skipped`` (from the table) and a
    ``feasibility`` verdict from :func:`smile_msi.hierstats.bh_feasibility` for the actual
    subject counts — the small-n guard that says, before you read any q-value, whether a rank
    test on this design could ever have rejected anything.
    """
    from . import hierstats
    from .spatial import _fold_tau

    if group_a == group_b:
        raise ValueError("group_a and group_b must differ")
    subject, compartment = nested_design(table, subject_by, compartment_from)
    feat_cols = [c for c in table.columns if c != "group"]
    targets = table.attrs.get("targets") or targets_from_columns(feat_cols)
    _check_feature_columns(feat_cols, targets)
    group = table["group"].to_numpy(dtype=object)

    keep = np.isin(group, [group_a, group_b])
    if not keep.any():
        raise ValueError(f"no samples labelled {group_a!r} or {group_b!r}")
    raw = table[feat_cols].to_numpy(dtype=float)[keep]
    group, subject, compartment = group[keep], subject[keep], compartment[keep]

    if compartments is None:
        compartments = sorted({str(c) for c in compartment})
    compartments = [str(c) for c in compartments]

    tau = _fold_tau(raw) if raw.size else 1.0
    if transform == "log2":
        X = np.log2(np.clip(raw, 0.0, None) + tau)
    elif transform in ("none", "", None):
        X = raw
    else:
        raise ValueError(f"unknown transform {transform!r}; use 'log2' or 'none'")

    if model == "lmm":
        res = hierstats.nested_mixed_model(
            X, group, compartment, subject, group_a, group_b, ids=targets,
            compartments=compartments, min_subjects_per_group=min_subjects_per_group,
            df_method=df_method, raw=raw)
    elif model == "stratified":
        res = hierstats.stratified_comparison(
            X, group, compartment, subject, group_a, group_b, ids=targets,
            compartments=compartments, method=method,
            min_n=min_subjects_per_group, raw=raw)
    else:
        raise ValueError(f"unknown model {model!r}; use 'lmm' or 'stratified'")

    subs_a = sorted({str(s) for s, g in zip(subject, group) if g == group_a})
    subs_b = sorted({str(s) for s, g in zip(subject, group) if g == group_b})
    res.attrs["subject_by"] = subject_by
    res.attrs["compartment_from"] = compartment_from
    res.attrs["norm_scope"] = table.attrs.get("norm_scope")
    res.attrs["transform"] = transform
    res.attrs["tau"] = float(tau)
    res.attrs["model"] = model
    res.attrs["subjects_a"], res.attrs["subjects_b"] = subs_a, subs_b
    res.attrs["n_subjects_a"], res.attrs["n_subjects_b"] = len(subs_a), len(subs_b)
    res.attrs["n_observations"] = int(keep.sum())
    res.attrs["skipped"] = list(table.attrs.get("skipped") or [])
    res.attrs["summary"] = table.attrs.get("summary")
    res.attrs["feasibility"] = hierstats.bh_feasibility(len(subs_a), len(subs_b), len(targets))
    return res


def nested_class_comparison(table, res, contrast: str, class_col: str = "best_class",
                            min_ions: int = 3, min_confidence: float = 0.0,
                            use_ranks: bool = True, inter_ion_cor=None, min_vif: float = 1.0,
                            center_observations: bool = True):
    """Roll a :func:`nested_comparison` result up to **lipid classes**, competitively.

    Per-feature FDR is hopeless at the replicate counts a nested MSI study actually has, but a
    class is not one test — it is a coherent set of ions whose members should move together if
    the biology is real. Asking "is this class shifted relative to the rest of the lipidome"
    aggregates that coherence into one testable statement, and being *competitive* makes it
    immune to the per-slide scale offsets that a self-contained class test would report as
    findings.

    The catch, and the reason this is not a one-liner over ``res``: a lipid class is a
    chain-length series. Its ions rise and fall together, so they carry far less information
    than their count suggests. :func:`smile_msi.hierstats.camera` measures that coupling
    directly — ``r̄``, the mean pairwise residual correlation — and inflates the test's variance
    by ``1 + (m-1)·r̄``. Ignoring it turns 19 sulfatides into 19 independent votes when they are
    worth about four, and reports ``p = 0.03`` where the honest answer is ``p = 0.30``.

    ``table`` is the :func:`pseudobulk_table` the model ran on and ``res`` its
    :func:`nested_comparison` output (already annotated, so it carries ``class_col``); the two
    must be the same run, because the residuals are recomputed from ``table`` against the design
    ``res`` records. ``contrast`` is a compartment name (the group's simple effect there) or
    ``'group'`` (averaged over compartments) — the interaction has no signed per-feature
    statistic, so it cannot be rolled up this way.

    ``min_confidence`` drops ions whose MS1 ``id_confidence`` is below it before forming the
    classes. A third of accurate-mass IDs at 5 ppm can be chance matches, and a class assembled
    from them is a class of noise; filtering costs ions and buys interpretability.

    ``center_observations`` (default on) subtracts each profile's mean across features before
    estimating ``r̄``. Every ion of one section shares that section's overall scale — the
    residual normalization offset — so without centering a *completely uncorrelated* class
    reports ``r̄ ≈ 0.6`` and every class is deflated by the same nuisance. That component shifts
    the class and the background equally, so a competitive test is already immune to it, and
    letting it into ``r̄`` only buys conservatism that isn't evidence of anything. What centering
    leaves behind is the correlation that actually manufactures false positives: a nerve rich in
    myelin is rich in every sulfatide *relative to its own lipidome*, and if that nerve happens
    to be a control the whole class looks depleted in the cases. The statistic itself is never
    centered — only the residuals used to measure coupling.

    Returns :func:`~smile_msi.hierstats.camera`'s table with ``class`` in place of ``set`` and a
    ``median_log2_fc`` column, plus ``df.attrs`` recording the contrast, the statistic, the
    filters and how many ions each stage dropped.
    """
    import pandas as pd

    from . import hierstats
    from .hierstats import _nested_design

    if contrast == "group":
        eff_col, se_col = "effect_group", "se_group"
    else:
        eff_col, se_col = f"effect__{contrast}", f"se__{contrast}"
    if eff_col not in res.columns:
        raise ValueError(f"no contrast {contrast!r} in this result; expected 'group' or one of "
                         f"{res.attrs.get('compartments')}")
    if se_col not in res.columns:
        # The stratified path reports effects and p-values but no standard errors, and the
        # competitive statistic must be the standardized effect — an unstandardized log2 FC
        # would rank a noisy low-abundance ion alongside a well-measured one.
        raise ValueError(
            f"the class test needs per-contrast standard errors ({se_col}), which only the "
            f"mixed model reports — re-run with model='lmm'")
    if class_col not in res.columns:
        raise ValueError(f"{class_col!r} is missing — annotate the result first "
                         f"(pipeline.annotate_df)")

    subject_by = res.attrs["subject_by"]
    comp_from = res.attrs.get("compartment_from", "exact")
    compartments = list(res.attrs["compartments"])
    group_a, group_b = res.attrs["a_label"], res.attrs["b_label"]
    tau = float(res.attrs.get("tau", 1.0))
    transform = res.attrs.get("transform", "log2")

    subject, compartment = nested_design(table, subject_by, comp_from)
    feat_cols = [c for c in table.columns if c != "group"]
    targets = table.attrs.get("targets") or targets_from_columns(feat_cols)
    _check_feature_columns(feat_cols, targets)
    if len(res) != len(targets) or not np.allclose(res["mz"].to_numpy(dtype=float),
                                                   np.asarray(targets, dtype=float), atol=1e-6):
        raise ValueError("res is not row-aligned with table's targets — pass the result of "
                         "nested_comparison(table, ...) for this same table, unsorted")

    group = table["group"].to_numpy(dtype=object)
    keep = np.isin(group, [group_a, group_b]) & np.isin(
        np.array([str(c) for c in compartment], dtype=object), compartments)
    raw = table[feat_cols].to_numpy(dtype=float)[keep]
    Y = np.log2(np.clip(raw, 0.0, None) + tau) if transform == "log2" else raw
    g_bin = (group[keep] == group_b).astype(float)
    comp_pos = {c: k for k, c in enumerate(compartments)}
    c_idx = np.array([comp_pos[str(c)] for c in compartment[keep]], dtype=int)
    D = _nested_design(g_bin, c_idx, len(compartments))

    # Features the model couldn't fit, or with a missing observation, have no residual to
    # correlate — they can't inform r̄ and shouldn't dilute the background either.
    fitted = (res["converged"].to_numpy(dtype=bool) if "converged" in res.columns
              else np.ones(len(res), dtype=bool))
    complete = np.isfinite(Y).all(axis=0) & fitted
    Yr = Y - np.nanmean(Y, axis=1, keepdims=True) if center_observations else Y
    resid, usable = hierstats.standardized_residuals(Yr, D)
    stat = (res[eff_col].to_numpy(dtype=float) / res[se_col].to_numpy(dtype=float))
    stat = np.where(complete & usable, stat, np.nan)

    keep_ion = complete & usable & np.isfinite(stat)
    if min_confidence > 0 and "id_confidence" in res.columns:
        conf = res["id_confidence"].to_numpy(dtype=float)
        keep_ion &= ~(np.isfinite(conf) & (conf < min_confidence))
    cls = res[class_col].astype(object).to_numpy()
    sets: dict = {}
    for i in np.flatnonzero(keep_ion):
        c = cls[i]
        if isinstance(c, str) and c:
            sets.setdefault(c, []).append(i)

    out = hierstats.camera(np.where(keep_ion, stat, np.nan), sets, resid_unit=resid,
                           inter_feature_cor=inter_ion_cor, use_ranks=use_ranks,
                           min_vif=min_vif, df_residual=float(D.shape[0] - D.shape[1]),
                           min_size=min_ions)
    out = out.rename(columns={"set": "class", "n_features": "n_ions"})
    fc_col = f"log2_fc__{contrast}" if contrast != "group" else None
    if fc_col and fc_col in res.columns:
        med = {name: float(np.nanmedian(res[fc_col].to_numpy(dtype=float)[idx]))
               for name, idx in sets.items()}
        out.insert(2, "median_log2_fc", out["class"].map(med))
    out.attrs["contrast"] = contrast
    out.attrs["statistic"] = f"{eff_col} / {se_col}"
    out.attrs["center_observations"] = bool(center_observations)
    out.attrs["a_label"], out.attrs["b_label"] = group_a, group_b
    out.attrs["min_confidence"] = float(min_confidence)
    out.attrs["min_ions"] = int(min_ions)
    out.attrs["n_ions_annotated"] = int(sum(len(v) for v in sets.values()))
    out.attrs["n_ions_background"] = int(keep_ion.sum())
    out.attrs["n_ions_dropped_unfitted"] = int((~(complete & usable)).sum())
    out.attrs["test"] = out.attrs.get("test", "")
    return out


# --------------------------------------------------------------------------- #
# SectionLoader — load one cube per source, reused across region-samples
# --------------------------------------------------------------------------- #
class SectionLoader:
    """Loads an imzML-backed sample's dataset **once per source path**, so several
    region-samples of one slide (or several Studio sections) share a single ``to_ram`` load,
    and the cube can be evicted the moment a source's last section is done — bounding peak RAM
    to one cube even across a large cohort.

    This is the loader the cohort *pooled-embedding* and the *Export Studio* both use, lifted
    out of the GUI so there is one cube-loading path (and one place to add the cache-sidecar
    fast-path). ``load(ref)`` accepts anything with ``.source`` and (optionally) ``.region`` /
    ``.session_path`` — a :class:`SampleRef` or a :class:`smile_msi.studio.Section`.
    """

    def __init__(self, prepare=None, dense: bool = True):
        self._cache: dict[str, object] = {}      # source path → loaded MSIDataset
        self._prepare = prepare                  # optional prepare(ds, ref) hook (e.g. sidecar)
        self._dense = dense                      # False → skip to_ram/prime (whole-slide passes):
                                                 # the region/sample-mean path streams only the
                                                 # region pixels, so loading the whole cube is waste
        self._remaining: dict[str, int] | None = None   # source → not-yet-consumed loads (primed)
        self._pending_evict: str | None = None   # a finished source, held until we move off it

    def prime(self, refs) -> None:
        """Declare the full set of ``refs`` this loader will be asked to ``load`` (the cohort
        roster), so each source's cube can be **evicted the moment its last ref is consumed** —
        bounding resident RAM to ~one cube even across a large cohort. ``load`` calls must then
        be one-per-ref (as :func:`~smile_msi.multivariate.pooled_embedding` does). Without
        priming the loader keeps every cube until :meth:`close` (the legacy behaviour)."""
        counts: dict[str, int] = {}
        for r in refs:
            src = (getattr(r, "source", "") or "")
            if src:
                counts[src] = counts.get(src, 0) + 1
        self._remaining = counts
        self._pending_evict = None

    def load(self, ref):
        """Return ``(ds, pix)`` for ``ref`` — ``pix`` is the region's pixel-index array, or
        ``None`` for a whole slide — or ``None`` to skip (non-imzML / moved / lost region)."""
        src = (getattr(ref, "source", "") or "")
        # Primed eviction: once we move off a source whose last ref was already consumed, free
        # its cube before loading the next — so at most the in-flight cube is resident.
        if self._pending_evict is not None and self._pending_evict != src:
            self.release(self._pending_evict)
            self._pending_evict = None
        if not src.lower().endswith(".imzml") or not os.path.exists(src):
            self._consume(src)                   # still count it so refcounts stay aligned
            return None                          # synthetic / CSV / moved sources can't be loaded
        self._consume(src)                       # one consume per load() call (all paths below)
        ds = self._cache.get(src)
        if ds is None:
            from .msi import MSIDataset           # lazy: keep heavy imports off module import
            ds = MSIDataset.from_imzml(src, lazy=True)
            self._restore_sidecar(ds, ref)           # reuse on-disk cube + prime stats if present
            if self._dense:                          # dense path (pooled pixels): whole cube in RAM
                ds.to_ram()
                ds.prime()                           # no-op when the sidecar carried prime stats
            # else: stay lazy — features_for_rows streams only the region pixels we ask for
            if self._prepare is not None:
                try:
                    self._prepare(ds, ref)
                except Exception:  # noqa: BLE001 — a failed prepare must not block the load
                    pass
            self._cache[src] = ds
        # Identity guard: when the roster records the slide's pixel count, a loaded file with a
        # different count is a DIFFERENT (or resized) slide — e.g. relocation matched the right
        # basename but the wrong acquisition, or a hand-edited roster. Don't silently embed the
        # wrong slide under this sample's name (mirrors the interactive reopen's mismatch guard).
        n_ref = getattr(ref, "n_pixels", None)
        n_got = getattr(ds, "n_pixels", None)
        if n_ref is not None and n_got is not None and int(n_got) != int(n_ref):
            import logging
            logging.getLogger(__name__).warning(
                "cohort sample %r: %s has %d pixels but the roster expected %d — wrong/resized "
                "slide; skipping.", getattr(ref, "name", "?"), src, int(n_got), int(n_ref))
            return None
        pix = None
        region = (getattr(ref, "region", "") or "")
        if region:
            sp = (getattr(ref, "session_path", "") or "")
            try:
                data = session.load_session(sp)
            except (OSError, ValueError):
                return None
            # Resolve the region's pixels (saved mask, or reconstructed from cluster `segments`
            # over the session labels — the app's primary region type saves an empty mask). So
            # these regions aren't silently dropped from the embedding while the group comparison
            # (which reads feature_scopes) keeps them — both analyses agree on cohort membership.
            pix = region_pixels(data, region)
            if pix is None or pix.size == 0:
                return None                      # region truly gone → skip cleanly
        return ds, pix

    def _restore_sidecar(self, ds, ref) -> bool:
        """Reuse the fingerprint-guarded ``.cache.npz`` sidecar (prime stats + cube) written
        beside a sample's managed session, so a cohort member **skips the prime pass** — and a
        lazy ion-image render (Export Studio) gets a ready cube — instead of recomputing it on
        every cohort run. The same fast-path the interactive reopen uses. A cache miss/mismatch
        is normal and silent; only a genuine read error is logged (never swallowed blindly).

        Skipped for processed/ragged data (no shared m/z axis): there the sidecar key
        (:func:`library.dataset_fingerprint`) would itself cost a full streaming pass, defeating
        the point."""
        sp = (getattr(ref, "session_path", "") or "")
        store = getattr(ds, "store", None)
        if not sp or store is None:
            return False
        try:
            if store.shared_axis() is None:
                return False
            got = session.load_cube(sp, library.dataset_fingerprint(ds), ds.n_pixels)
        except Exception:  # noqa: BLE001 — a broken sidecar must never block a cohort load
            import logging
            logging.getLogger(__name__).warning(
                "cube sidecar reuse failed for %s", sp, exc_info=True)
            return False
        if not got:
            return False
        ds._cube = got["cube"]
        if got.get("mean") is not None and got.get("pix") is not None:
            ds._mean, ds._pix = got["mean"], got["pix"]
        return True

    def _consume(self, src: str) -> None:
        """Account one consumed load of ``src`` (primed mode only) and, once its last ref is
        done, mark its cube for eviction the moment the next *different* source is loaded."""
        if self._remaining is None or not src:
            return
        left = self._remaining.get(src)
        if left is None:
            return
        left -= 1
        self._remaining[src] = left
        if left <= 0:
            self._pending_evict = src

    def release(self, source: str) -> None:
        """Evict a source's cube (called once its last section is extracted)."""
        ds = self._cache.pop(source, None)
        if ds is not None and hasattr(ds, "release"):
            try:
                ds.release()
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        self._pending_evict = None
        for src in list(self._cache):
            self.release(src)
