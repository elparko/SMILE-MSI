"""Open a slide — and the analysis state saved on it — without the desktop app.

The app auto-saves each sample's whole analysis to a *managed session* JSON under
``~/.smile-msi/sessions`` (:mod:`smile_msi.session`): the source path, the extraction
settings, the picked features, every named region and its group tag, the segmentation
labels, the saved feature lists. Everything a script needs to pick up where the user
left off is therefore already on disk — but the only code that turned it back into live
state lived in the GUI (``MainWindow._prepare_imzml`` / ``ScriptConsole._build_api``).

This module is that reconstruction, without Qt:

* :func:`managed_sessions` / :func:`read_session` / :func:`session_summary` — inspect
  what has been worked on **without opening a dataset at all** (instant: it is a JSON
  read, not a disk pass over the spectra).
* :func:`open_dataset` — an imzML opened the way the app opens it: preprocessing
  re-applied, fast cube restored from the session sidecar when one matches, so a reopen
  does not re-stream the ``.ibd``.
* :func:`open_slide` — the whole thing: dataset + region masks + group masks + working
  feature set, handed back as a :class:`ScriptAPI` identical to the one the in-app
  Script Console builds. Analyses run through it are the *same* registry steps the GUI
  runs, so results cannot drift from what the app would show.

Pure (no Qt), and the heavy engine imports stay inside the functions that need them, so
importing this module is cheap.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

from . import library, session

# Per-pixel normalizations / window reducers a session may carry; anything else in a
# hand-edited session file falls back to the engine defaults rather than raising.
_SETTING_DEFAULTS = {"ppm": 10.0, "norm": "tic", "reduce": "sum", "id_ppm": 5.0,
                     "mode": "negative"}


# --------------------------------------------------------------------------- #
# managed sessions — metadata only (no dataset opened)
# --------------------------------------------------------------------------- #
def managed_sessions() -> list[dict]:
    """Every auto-saved session on this machine, newest first, each with the row
    :func:`smile_msi.session.list_managed` builds plus ``source_exists`` — the flag that
    separates "you can reopen this" from a slide whose drive is unplugged or whose folder
    moved (the failure mode :mod:`scripts.rebase_cohort` heals for cohorts)."""
    out = []
    for row in session.list_managed():
        row = dict(row)
        src = str(row.get("source") or "")
        row["source_exists"] = bool(src) and os.path.exists(src)
        out.append(row)
    return out


def resolve_session_ref(ref: str) -> str:
    """The session-JSON path a loose reference names: an explicit path, a managed session's
    file stem, its display name (``slide.imzML``), or the source dataset path it was saved
    on. Raises ``FileNotFoundError`` naming the available sessions when nothing matches."""
    ref = str(ref)
    if os.path.isfile(ref):
        return ref
    if os.path.isfile(ref + ".json"):
        return ref + ".json"
    rows = managed_sessions()
    for row in rows:
        path = str(row["path"])
        stem = os.path.basename(path).removesuffix(".json")
        if ref in (path, stem, str(row.get("name") or ""), str(row.get("source") or "")):
            return path if path.endswith(".json") else path + ".json"
    names = ", ".join(sorted({str(r.get("name") or "") for r in rows})) or "none"
    raise FileNotFoundError(f"no session matches {ref!r}. Available: {names}")


def read_session(ref: str) -> dict:
    """Load the session dict for a reference :func:`resolve_session_ref` understands."""
    return session.load_session(resolve_session_ref(ref))


def session_summary(data: dict, *, path: str = "") -> dict:
    """A small JSON-safe digest of a session: what slide it is, how it is being extracted,
    and what state has accumulated on it (regions and their group tags, saved feature
    lists, the segmentation, the logged analysis runs).

    Deliberately cheap and shallow — no pixel masks, no spectra, no peak table — so it can
    describe a 200k-pixel slide in milliseconds, without its dataset present at all."""
    regions = data.get("named_regions") or []
    seg = data.get("segmentation") or None
    settings = dict(data.get("settings") or {})
    source = str(data.get("source") or "")
    return {
        "path": path,
        "source": source,
        "source_exists": bool(source) and os.path.exists(source),
        "n_pixels": data.get("n_pixels"),
        "settings": {k: settings.get(k, v) for k, v in _SETTING_DEFAULTS.items()},
        "preprocessing": data.get("preprocessing") or None,
        "calibration": data.get("calibration") or None,
        "n_features": len(data.get("peaks") or []),
        "active_mz": data.get("active_mz"),
        "regions": [{"name": r.get("name"), "group": r.get("group") or "",
                     "parent": r.get("parent"), "sample": r.get("sample") or "",
                     "n_segments": len(r.get("segments") or []),
                     "drawn": r.get("mask") is not None,
                     "n_pixels": (len(r["mask"]) if r.get("mask") is not None else None)}
                    for r in regions],
        "groups": sorted({(r.get("group") or "").strip() for r in regions} - {""}),
        "segmentation": ({"n_clusters": seg.get("n_clusters")} if seg else None),
        "feature_lists": {name: len(feats)
                          for name, feats in (data.get("feature_lists") or {}).items()},
        "feature_scopes": sorted(data.get("feature_scopes") or {}),
        # the run index only — never a run's ``inputs`` (it carries the whole m/z list) or
        # its result payload; :func:`run_result` fetches those one run at a time.
        "analysis_runs": [{"run_id": r.get("run_id"), "step_id": r.get("step_id"),
                           "title": r.get("title"), "status": r.get("status"),
                           "created": r.get("created"), "summary": r.get("summary"),
                           "params": r.get("params") or {}}
                          for r in (data.get("analysis_runs") or [])],
    }


def run_store(ref: str):
    """The :class:`smile_msi.runs.RunStore` holding a session's analysis runs — one
    directory per run beside the session file (``<session>.runs/<run_id>/``)."""
    from .runs import RunStore

    return RunStore(session.runs_dir_path(resolve_session_ref(ref)))


def run_result(ref: str, run_id: str):
    """The saved payload of one logged analysis run: a ``DataFrame`` for a stats table, an
    array for an image, a dict for everything else — or ``None`` when the run kept none.

    This is how a question like *what did that comparison find?* gets answered without
    re-running anything: the app already wrote the result to disk when it ran."""
    store = run_store(ref)
    return store.load_result(store.load(run_id))


# --------------------------------------------------------------------------- #
# regions → masks  (the GUI's ScriptConsole._build_api, from a session dict)
# --------------------------------------------------------------------------- #
def region_masks(data: dict, n_pixels: int, source: str = "") -> tuple[dict, dict]:
    """``({region name: bool[n_pixels]}, {group name: bool[n_pixels]})`` from a session.

    Mirrors ``SegmentTabMixin._region_pixel_mask`` (smile_msi/gui/segment.py:1940) exactly:
    a drawn ROI's saved pixel indices win; else the union of the segmentation clusters the
    region names; else — for an *aggregate parent* such as a compartment grouping several
    fascicle ROIs — the union of its children. Regions tagged to a different slide are
    skipped when ``source`` is given (the region↔slide tie), so a leftover from another
    sample can never be compared against this one's pixels.

    Groups are the union of every region carrying that group tag, which is how the app
    turns "tag these ROIs Group A" into the two masks a comparison runs on."""
    regions = [r for r in (data.get("named_regions") or [])
               if not source or not (r.get("sample") or "") or r.get("sample") == source]
    labels = (data.get("segmentation") or {}).get("labels")
    labels = np.asarray(labels, dtype=int) if labels else None

    def resolve(rg, seen):
        name = rg.get("name")
        if name in seen:                                   # malformed parent cycle
            return None
        seen.add(name)
        if rg.get("mask") is not None:
            return session.mask_from_indices(rg["mask"], n_pixels)
        segs = rg.get("segments") or []
        if segs and labels is not None and labels.size == n_pixels:
            return np.isin(labels, [int(s) for s in segs])
        union = None
        for child in regions:
            if child is rg or child.get("parent") != name:
                continue
            mask = resolve(child, seen)
            if mask is None:
                continue
            union = mask.copy() if union is None else (union | mask)
        return union

    masks, groups = {}, {}
    for rg in regions:
        mask = resolve(rg, set())
        if mask is None or not mask.any():
            continue
        masks[str(rg.get("name"))] = mask
        tag = (rg.get("group") or "").strip()
        if tag:
            groups[tag] = (groups[tag] | mask) if tag in groups else mask.copy()
    return masks, groups


def unresolved_recal_regions(cfg: dict | None, masks: dict) -> list:
    """Per-region lock-mass blocks in ``cfg`` naming a region that ``masks`` does not have.
    Region names are user-editable, so a rename, a delete, or a slide moved to a path that no
    longer matches the regions' ``sample`` tag leaves those pixels uncorrected. Callers must
    surface this rather than keep reporting the slide as calibrated."""
    blocks = ((cfg or {}).get("recalibrate") or {}).get("regions") or []
    return [b.get("name") for b in blocks if b.get("name") not in masks]


def apply_preprocessing(ds, data: dict) -> dict | None:
    """Re-apply the preprocessing a session was saved with, resolving any per-region
    lock-mass block against that session's own regions. Every reopen path must go through
    this rather than calling ``build_pipeline`` bare, which would resolve no regions at all
    and silently degrade a per-region recalibration to the slide-wide offset — the one number
    it exists to avoid.

    The config is stored exactly as the session recorded it, so a caller's
    "already applied?" check still matches; use :func:`unresolved_recal_regions` to find
    blocks that did not resolve. Returns the config applied, or ``None``."""
    from . import preprocess          # engine imports stay inside the functions (see module doc)
    cfg = (data or {}).get("preprocessing") or None
    if not cfg:
        return None
    masks = {}
    if (cfg.get("recalibrate") or {}).get("regions"):
        masks, _ = region_masks(data, ds.n_pixels, getattr(ds, "source", ""))
    ds.set_preprocessing(preprocess.build_pipeline(cfg, masks=masks), config=cfg)
    return cfg


# --------------------------------------------------------------------------- #
# opening a dataset the way the app opens one
# --------------------------------------------------------------------------- #
def cube_fingerprint(ds) -> str:
    """The key a fast-cube sidecar is stored under: the slide fingerprint plus a hash of
    the preprocessing it was built with, so a cube built on raw spectra is never served to
    a recalibrated session. Mirrors ``MainWindow._cube_fingerprint`` (gui/main.py:747)."""
    fp = library.dataset_fingerprint(ds)
    cfg = getattr(ds, "preprocessing", None)
    if cfg:
        fp += "|pp:" + hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    return fp


def open_dataset(source: str, *, session_path: str | None = None, stride: int = 1,
                 dense: bool = False, progress=None):
    """Open an imzML the way the app does: re-apply the session's preprocessing pipeline
    **before** any pass, then restore the fast m/z cube (and prime statistics) from the
    sidecar saved beside that session when it matches this slide.

    That ordering is the whole point. Preprocessing changes the spectra, so a cube built
    on raw data must not be served to a recalibrated session — and priming a large slide
    from disk costs a full pass over the ``.ibd`` (minutes on an external drive), which the
    sidecar is there to avoid. ``dense=True`` additionally pulls the spectra into RAM
    (``to_ram``), which is only worth it for repeated whole-slide passes.

    ``stride`` is the quick-load every-Nth-pixel preview; leave it 1 for real analysis
    (a strided dataset has different pixel indices, so saved region masks will not line up).
    """
    from . import preprocess
    from .msi import MSIDataset

    source = str(source)
    ibd = os.path.splitext(source)[0] + ".ibd"
    if not os.path.exists(ibd):
        raise FileNotFoundError(
            f"missing the companion data file {os.path.basename(ibd)} — an imzML stores only "
            "metadata; the .ibd holds the spectra and must sit in the same folder.")
    ds = MSIDataset.from_imzml(source, lazy=True, stride=int(stride))

    if session_path is None and int(stride) == 1:
        try:
            found = session.resolve_session_path(source, library.dataset_fingerprint(ds),
                                                 ds.n_pixels)
            session_path = found if found and os.path.exists(found) else None
        except Exception:  # noqa: BLE001 — the lookup must never block a load
            session_path = None

    if session_path:
        apply_preprocessing(ds, read_session(session_path))
        try:
            got = session.load_cube(session_path, cube_fingerprint(ds), ds.n_pixels)
        except Exception:  # noqa: BLE001 — a stale sidecar must never block a load
            got = None
        if got:
            ds._cube = got["cube"]
            if got.get("mean") is not None and got.get("pix") is not None:
                ds._mean, ds._pix = got["mean"], got["pix"]
            ds.prime(progress=progress)          # no-op when the sidecar carried prime stats
            return ds

    if dense:
        ds.to_ram(progress=progress)
    ds.prime(progress=progress)
    return ds


@dataclass
class Slide:
    """One opened slide plus the analysis state saved on it.

    ``api`` is a :class:`smile_msi.scripting.ScriptAPI` built exactly as the in-app Script
    Console builds it (gui/scriptconsole.py:280), so ``slide.api.run_analysis(...)`` runs
    the same registry step, with the same inputs, that the app would run."""
    ds: object
    api: object
    data: dict = field(default_factory=dict)
    session_path: str = ""

    @property
    def source(self) -> str:
        return str(getattr(self.ds, "source", "") or "")

    def summary(self) -> dict:
        """The session digest plus what actually resolved once the dataset was opened —
        region masks can resolve to no pixels if a session was saved against a different
        slide, and that is worth seeing before an analysis silently runs on nothing."""
        out = session_summary(self.data, path=self.session_path)
        out["source"] = self.source                  # the open slide wins over the saved path
        out["source_exists"] = os.path.exists(self.source)
        out["n_pixels"] = int(self.ds.n_pixels)
        out["resolved_regions"] = {name: int(mask.sum())
                                   for name, mask in self.api.masks.items()}
        out["resolved_groups"] = {name: int(mask.sum())
                                  for name, mask in self.api.groups.items()}
        out["working_features"] = len(self.api.get_features())
        return out


def working_features(data: dict) -> list:
    """The peak set the app would land on when it reopens this session: the active
    per-sample *scope*, falling back to the slide's own peaks (``gui/main.py:1050``).

    Taking ``peaks`` alone would silently analyse the wrong feature set on any sample whose
    working set had been switched to a scope — which is the normal state on a multi-sample
    slide."""
    scopes = data.get("feature_scopes") or {}
    active = data.get("active_feature_scope")
    return scopes.get(active) or data.get("peaks") or []


def bind(ds, data: dict | None = None, *, session_path: str = ""):
    """Wrap an already-open dataset and (optionally) a session dict into a :class:`Slide`."""
    from . import scripting

    data = dict(data or {})
    settings = dict(data.get("settings") or {})
    # The app applies the saved rotation before any view renders (gui/main.py:1021). Without
    # it every image and mask here comes out at a different orientation from the one on screen.
    with contextlib.suppress(Exception):
        ds.set_orientation(int(settings.get("orientation", 0) or 0))
    masks, groups = region_masks(data, int(ds.n_pixels), source=str(getattr(ds, "source", "")))
    api = scripting.ScriptAPI(
        ds,
        ppm=float(settings.get("ppm", _SETTING_DEFAULTS["ppm"])),
        norm=str(settings.get("norm", _SETTING_DEFAULTS["norm"])),
        reduce=str(settings.get("reduce", _SETTING_DEFAULTS["reduce"])),
        masks=masks, groups=groups,
        features=[p.get("mz") for p in working_features(data)],
        active_mz=data.get("active_mz"))
    return Slide(ds=ds, api=api, data=data, session_path=session_path)


def open_slide(ref: str, *, stride: int = 1, dense: bool = False, progress=None) -> Slide:
    """Open a saved session as a live :class:`Slide` — dataset, regions, groups, features.

    ``ref`` is anything :func:`resolve_session_ref` understands (a session path or stem, a
    managed sample name, or the source ``.imzML`` path). A dataset path with no saved
    session is also accepted: it opens the slide with engine defaults and no regions."""
    src = str(ref)
    if src.lower().endswith(".imzml") and os.path.isfile(src):
        try:
            path = resolve_session_ref(src)
        except FileNotFoundError:
            ds = open_dataset(src, stride=stride, dense=dense, progress=progress)
            return bind(ds, {"source": src})
    else:
        path = resolve_session_ref(ref)
    data = session.load_session(path)
    source = str(data.get("source") or "")
    if not source or not os.path.exists(source):
        raise FileNotFoundError(
            f"the slide this session was saved on is not where it was left: {source!r}. "
            "Reconnect the drive, or re-point the session at the moved file.")
    ds = open_dataset(source, session_path=path, stride=stride, dense=dense, progress=progress)
    return bind(ds, data, session_path=path)


def open_demo(**kwargs) -> Slide:
    """A synthetic slide with two tagged groups — the zero-data way to exercise every
    analysis (the same generator the app's *Load demo dataset* button uses)."""
    from . import demo

    ds = demo.make_synthetic(**kwargs)
    ds.prime()
    _, cols = ds._pixel_rows_cols()
    left = cols < ds.width / 2
    named = [{"name": "left", "color": "#e6194b", "group": "Group A",
              "mask": np.flatnonzero(left).tolist()},
             {"name": "right", "color": "#3cb44b", "group": "Group B",
              "mask": np.flatnonzero(~left).tolist()}]
    return bind(ds, {"source": ds.source, "named_regions": named,
                     "settings": {"ppm": 20.0, "norm": "tic", "reduce": "sum"}})
