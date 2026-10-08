"""Session / project files — a sample's whole analysis, saved and resumed.

A session is a JSON file capturing *what you did*, not the raw spectra: the dataset's
source path, the global settings, the picked peaks, the **named feature lists** and
**per-sample working scopes**, the segmentation labels, the named regions, and the
active m/z. Re-opening it reloads the dataset from its source and restores that state,
so a review is reproducible and shareable.

Sessions are the single per-sample store: as you work, the app auto-saves to a
*managed* file under ``~/.smile-msi/sessions/`` (see :func:`managed_path` /
:func:`list_managed`); ``Save/Open session…`` export/import a standalone copy.

The functions here are pure (dict in, dict out / file) so they are testable without
a GUI; the desktop app gathers and applies the state.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

import numpy as np

VERSION = 5   # v5 adds analysis_runs (the re-openable analysis-run index, plan 24); older sessions load fine
              # v4 added acquisition_meta / calibration_models / msms_lib


def _feature_record(f) -> dict:
    """Normalize one saved-feature-list entry to ``{mz, lipid, note}``."""
    return {"mz": float(f["mz"]), "lipid": str(f.get("lipid", "")),
            "note": str(f.get("note", ""))}


def _peak_record(p) -> dict:
    """Serialize one working peak, keeping the optional scope/window/visibility keys
    the GUI stores on it so a restored scope is identical to the saved one."""
    rec = {"mz": float(p["mz"]), "intensity": float(p.get("intensity", 0.0)),
           "snr": float(p.get("snr", 0.0)),
           "rel_intensity": float(p.get("rel_intensity", 0.0))}
    if p.get("region") is not None:
        rec["region"] = p["region"]
    for k in ("lo", "hi"):
        if p.get(k) is not None:
            rec[k] = float(p[k])
    if p.get("hidden"):
        rec["hidden"] = True
    if p.get("label_override"):                    # a custom/renamed legend label
        rec["label_override"] = str(p["label_override"])
    return rec


def build_session(*, source, settings, peaks, active_mz=None, labels=None,
                  n_clusters=None, seg_colors=None, region_a=None, region_b=None,
                  named_regions=None, n_pixels=None, dataset_fingerprint=None,
                  feature_lists=None, feature_scopes=None,
                  active_feature_scope=None, flist_name=None, optical=None,
                  cube_fingerprint=None, crop_preset=None, report_items=None,
                  flow=None,   # retired (plan 24): the Flow designer is gone; kept as a no-op
                  lipid_lists=None, provenance=None, studio_plan=None,
                  acquisition_meta=None, calibration_models=None, msms_lib=None,
                  analysis_runs=None, preprocessing=None, calibration=None) -> dict:
    """Assemble a JSON-serializable session dict from analysis state.

    ``n_pixels`` / ``dataset_fingerprint`` (from
    :func:`smile_msi.library.dataset_fingerprint`) record the identity of the slide
    this state was captured on, so re-opening onto a different (even same-sized)
    dataset can warn that the restored pixel indices may not line up.

    ``feature_lists`` (``{name: [{mz,lipid,note}]}``) are the named saved lists and
    ``feature_scopes`` (``{scope: [peak]}``) the per-sample working sets; together with
    ``active_feature_scope`` / ``flist_name`` they let a reopened session land on the
    same active feature set it was left on (mirrors the undo snapshot in the GUI)."""
    def idx_list(mask):
        if mask is None:
            return None
        # .tolist() is C-speed and yields plain Python ints (JSON-identical to a loop) —
        # this runs on the GUI thread on every debounced auto-save, so keep it cheap.
        return np.flatnonzero(np.asarray(mask, dtype=bool)).tolist()

    seg = None
    if labels is not None:
        seg = {"labels": np.asarray(labels).astype(int).tolist(), "n_clusters": int(n_clusters)}
        if seg_colors:                       # tree-aware segment colours (treecolors)
            seg["colors"] = [str(c) for c in seg_colors]
    return {
        "version": VERSION,
        "source": source,
        # the per-spectrum preprocessing pipeline config the analysis was done with
        # (preprocess.build_pipeline dict) — re-applied before any pass on reopen — and the
        # lock-mass calibration state the Calibration check dialog reports
        "preprocessing": (dict(preprocessing) if preprocessing else None),
        "calibration": (dict(calibration) if calibration else None),
        "n_pixels": (int(n_pixels) if n_pixels is not None else None),
        "dataset_fingerprint": (str(dataset_fingerprint) if dataset_fingerprint else None),
        "settings": dict(settings),
        "peaks": [_peak_record(p) for p in peaks],
        "active_mz": (float(active_mz) if active_mz is not None else None),
        "segmentation": seg,
        "regions": {"A": idx_list(region_a), "B": idx_list(region_b)},
        # named regions: cluster groups and/or drawn ROI 'sample' regions (+ sub-regions)
        "named_regions": [{"name": str(r["name"]), "color": str(r["color"]),
                           "segments": sorted(int(s) for s in r.get("segments", [])),
                           "mask": idx_list(r.get("mask")),
                           "parent": r.get("parent"),
                           "visible": bool(r.get("visible", True)),
                           # cohort group tag (Group A/B/…) the Analysis Flow assigns; "" = none
                           "group": str(r.get("group", "") or ""),
                           # the slide (dataset source) this region was drawn on, so it only
                           # renders on its own tissue; "" for legacy regions → treated as current
                           "sample": str(r.get("sample", "") or ""),
                           # manual export crop box (display-grid bounds + the orientation it
                           # was drawn at, so a later Rotate can fall back to the bbox)
                           "crop": ([int(v) for v in r["crop"]] if r.get("crop") else None),
                           "crop_orient": r.get("crop_orient")}
                          for r in (named_regions or [])],
        # per-sample analysis store (formerly the global library)
        "feature_lists": {str(name): [_feature_record(f) for f in feats]
                          for name, feats in (feature_lists or {}).items()},
        "feature_scopes": {str(name): [_peak_record(p) for p in plist]
                           for name, plist in (feature_scopes or {}).items()},
        # lipid lists: named collections of lipid classes (each a colour + its member m/z) —
        # the class-level twin of feature_lists, rendered as a per-class tissue overlay
        "lipid_lists": {str(name): [{"class": str(e.get("class", "")),
                                     "color": str(e.get("color", "")),
                                     "mzs": [float(m) for m in (e.get("mzs") or [])]}
                                    for e in entries]
                        for name, entries in (lipid_lists or {}).items()},
        "active_feature_scope": active_feature_scope,
        "flist_name": flist_name,
        # optical/histology backdrop: path + alignment + display (pixels not embedded)
        "optical": (dict(optical) if optical else None),
        # fingerprint of the slide the fast m/z cube sidecar (<stem>.cache.npz) was built
        # on; on reopen it's loaded instead of rebuilt when this still matches (see
        # save_cube / load_cube). None when no cube was cached.
        "cube_fingerprint": (str(cube_fingerprint) if cube_fingerprint else None),
        # project-wide default close-up framing (Crop Studio): {aspect, pad_frac, size_um}
        "crop_preset": (dict(crop_preset) if crop_preset else None),
        # curated PDF report contents (Report tab): ordered list of plain-dict analysis items
        "report_items": [dict(it) for it in (report_items or [])],
        # saved Export Studio plan (sections × feature-lists/ions × outputs): a
        # StudioPlan.to_dict() or plain dict; persisted so the assembled export survives
        # reopen on the same sample (mirrors report_items).
        "studio_plan": (studio_plan.to_dict() if hasattr(studio_plan, "to_dict")
                        else (dict(studio_plan) if studio_plan else None)),
        # provenance / audit trail: how every result on this sample was generated — input
        # checksums, dataset, each analysis step + its full settings + the ROIs that fed it,
        # software versions. A Provenance obj (.to_dict()) or a plain dict; persisted so the
        # methods record survives close/reopen instead of being rebuilt empty.
        "provenance": (provenance.to_dict() if hasattr(provenance, "to_dict")
                       else (dict(provenance) if provenance else None)),
        # acquisition / reporting metadata for standards-compliant imzML export + METASPACE
        # (plan 02): an AcquisitionMeta.to_dict() or plain dict; {} when unset.
        "acquisition_meta": (acquisition_meta.to_dict() if hasattr(acquisition_meta, "to_dict")
                             else (dict(acquisition_meta) if acquisition_meta else {})),
        # absolute-quantification calibration models per analyte (plan 03): a list of
        # CalibrationModel.to_dict() dicts; [] when none fitted.
        "calibration_models": [(m.to_dict() if hasattr(m, "to_dict") else dict(m))
                               for m in (calibration_models or [])],
        # MS/MS spectral-library pointer + per-ion library-match detail (plan 04); None when unused.
        "msms_lib": (dict(msms_lib) if msms_lib else None),
        # logged analysis runs (plan 24): ordered list of plain-dict run records
        # (kind + params + result summary) forming the re-openable analysis-run index;
        # [] when none. JSON-native values only — mirrors report_items' plain-dict copy.
        "analysis_runs": [dict(r) for r in (analysis_runs or [])],
    }


def save_session(path: str, session: dict) -> str:
    # atomic write: temp + fsync + replace, so a crash mid-write can't corrupt an
    # existing session file (same safeguard as library.save_library).
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def load_session(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # forward/backward-compatible: default any keys a v1 (or future) file may lack so
    # callers can read them unconditionally.
    data.setdefault("settings", {})
    data.setdefault("feature_lists", {})
    data.setdefault("feature_scopes", {})
    data.setdefault("lipid_lists", {})
    data.setdefault("active_feature_scope", None)
    data.setdefault("flist_name", None)
    data.setdefault("optical", None)
    data.setdefault("cube_fingerprint", None)
    data.setdefault("report_items", [])
    data.setdefault("studio_plan", None)
    data.setdefault("flow", None)
    data.setdefault("provenance", None)
    data.setdefault("acquisition_meta", {})       # v4: standards/reporting metadata (plan 02)
    data.setdefault("calibration_models", [])     # v4: absolute-quant calibrations (plan 03)
    data.setdefault("msms_lib", None)             # v4: MS/MS spectral-library match (plan 04)
    data.setdefault("single_cell", None)          # v4: single-cell segmentation result (plan 07)
    data.setdefault("analysis_runs", [])          # v5: re-openable analysis-run index (plan 24)
    # data-value back-compat shim (plan 18, Issue A; shipped v4-era): rewrite any legacy
    # report-item fold_change → signed log2_fc. NOT a version-gated schema migration —
    # runs unconditionally, load never reads data["version"].
    _migrate_fold_change_to_log2(data)
    return data


def _migrate_fold_change_to_log2(data: dict) -> None:
    """Back-compat shim for sessions saved before the per-slide ``fold_change`` column
    became signed ``log2_fc`` (audit plan 18, Issue A). Logged analysis report items
    persist their result table as JSON ``records``; rewrite any stale ``fold_change``
    key to ``log2_fc = log2(fold_change)`` so old report books still render. Directional
    sources (discriminating / ROI localization) keep their sign; the old magnitude form
    (ROI / class comparison, always ≥ 1) maps to a non-negative log2 — its direction was
    not recoverable from the stored magnitude, which is the documented limitation."""
    import math

    def _fix(rec):
        if isinstance(rec, dict) and "fold_change" in rec and "log2_fc" not in rec:
            try:
                fc = float(rec.pop("fold_change"))
                rec["log2_fc"] = math.log2(fc) if fc > 0 else 0.0
            except (TypeError, ValueError):
                rec.pop("fold_change", None)

    for item in data.get("report_items", []):
        if isinstance(item, dict):
            for rec in item.get("records", []) or []:
                _fix(rec)


# --------------------------------------------------------------------------- #
# fast-cube cache — the expensive m/z cube + prime stats are persisted beside the
# session so reopening skips rebuilding them. Current format: a chunked, lazily-read
# Zarr store (<stem>.cube.zarr, see cubestore.py). The legacy <stem>.cache.npz is no
# longer written; old ones are still read and migrated to Zarr on first load.
# --------------------------------------------------------------------------- #
def cube_sidecar_path(session_path: str) -> str:
    """Path of the legacy (no longer written) npz cube sidecar beside a session file —
    still read by :func:`load_cube` so pre-Zarr caches migrate transparently."""
    return os.path.splitext(str(session_path))[0] + ".cache.npz"


def cube_zarr_path(session_path: str) -> str:
    """Path of the chunked, lazily-read Zarr cube store beside a session file — the cube
    cache (:class:`smile_msi.cubestore.CubeStore`). Reopening reads only the chunks an ion
    window touches instead of materialising the whole CSC, so warm-reopen RSS no longer
    scales with the whole cube."""
    return os.path.splitext(str(session_path))[0] + ".cube.zarr"


def runs_dir_path(session_path: str) -> str:
    """Directory holding the analysis-run payload store beside a session file —
    ``<stem>.runs/`` (plan 24). Mirrors the :func:`cube_zarr_path` sidecar idiom: the
    heavyweight run payloads (result tables, thumbnails) live on disk next to the session,
    while only the lightweight ``analysis_runs`` index is embedded in the session JSON."""
    return os.path.splitext(str(session_path))[0] + ".runs"


def save_cube(session_path: str, cube, *, mean=None, pix=None,
              fingerprint: str = "") -> str | None:
    """Persist the fast m/z cube (and, when given, the :meth:`MSIDataset.prime` results —
    mean spectrum + per-pixel TIC/RMS/median) to the chunked, lazily-read **Zarr** store
    ``<stem>.cube.zarr`` beside the session, so reopening reads it on demand instead of
    re-streaming the whole dataset to rebuild.

    ``cube`` is the ``(axis, scipy-sparse pixels×bins, edges)`` tuple from
    :meth:`MSIDataset.build_mz_cube`. The slide ``fingerprint`` is embedded and checked on
    load (:func:`load_cube`) so a stale or mismatched cache is ignored, never misapplied.
    The write is atomic (temp + replace). Returns the store path, or ``None`` if there's no
    cube (or zarr is unavailable / the write failed — caching is best-effort, never fatal).

    The legacy uncompressed ``.cache.npz`` sidecar is no longer written; old npz caches are
    still read and transparently migrated to a Zarr store on the next :func:`load_cube`."""
    if cube is None:
        return None

    from .cubestore import CubeStore

    # A reopened cube is already an on-disk CubeStore (e.g. after a streaming build) — there's
    # nothing to (re)persist, and it can't be unpacked as (axis, csc, edges).
    if isinstance(cube, CubeStore):
        zp = cube_zarr_path(session_path)
        return zp if os.path.exists(zp) else None

    axis, mat, edges = cube
    mat = mat.tocsc()
    try:
        store = CubeStore.create(cube_zarr_path(session_path), axis, mat, edges,
                                 fingerprint=str(fingerprint), n_pixels=int(mat.shape[0]),
                                 mean=mean, pix=pix)
        store.close()
    except Exception:  # noqa: BLE001 — the cube cache is a best-effort optimization
        return None
    return cube_zarr_path(session_path)


def _load_npz_cube(path: str, fingerprint: str, n_pixels: int) -> dict | None:
    """Read a legacy ``.cache.npz`` cube sidecar, or ``None`` if it's missing, corrupt, or
    doesn't match this slide (same ``fingerprint`` + ``n_pixels`` guard as the Zarr path).
    Returns ``{"cube": (axis, csc, edges)}`` (+ ``mean``/``pix`` when present). Kept only so
    pre-Zarr caches still load and can be migrated; nothing writes this format any more."""
    if not os.path.exists(path):
        return None
    try:
        import numpy as np
        from scipy import sparse

        with np.load(path, allow_pickle=False) as z:
            if str(z["fingerprint"]) != str(fingerprint):
                return None
            shape = tuple(int(x) for x in z["shape"])
            if shape[0] != int(n_pixels):
                return None
            axis = z["axis"]
            cube = sparse.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=shape)
            out = {"cube": (axis, cube, z["edges"])}
            if "mean" in z.files:
                out["mean"] = (axis, z["mean"])
            if "tic" in z.files:
                out["pix"] = {"tic": z["tic"], "rms": z["rms"], "median": z["median"]}
        return out
    except Exception:  # noqa: BLE001 — a corrupt/partial cache must never be fatal
        return None


def load_cube(session_path: str, fingerprint: str, n_pixels: int) -> dict | None:
    """Load the cube cache for a session, or ``None`` if it's missing, unreadable, or doesn't
    match this slide (``fingerprint`` + pixel-count guard, so a cache from a different/resized
    acquisition is never misapplied).

    Returns ``{"cube": <CubeStore | (axis, csc, edges)>}`` plus, when they were saved,
    ``"mean"`` (axis, mean) and ``"pix"`` ({tic, rms, median}) — ready to assign onto a
    dataset to skip both the cube build and the prime pass.

    Prefers the chunked, lazily-read **Zarr** store (returns a :class:`CubeStore`, so
    reopening doesn't materialise the whole CSC). If only a legacy **npz** cache is present it
    is read *and transparently migrated* to a Zarr store (then removed), so the next reopen is
    lazy — a one-time, best-effort upgrade that never fails a load."""
    zpath = cube_zarr_path(session_path)
    try:
        from .cubestore import CubeStore
        store = CubeStore.open(zpath, fingerprint=str(fingerprint), n_pixels=int(n_pixels))
        if store is not None:
            out = {"cube": store}
            mean, pix = store.stored_extras()
            if mean is not None:
                out["mean"] = mean
            if pix is not None:
                out["pix"] = pix
            return out
    except Exception:  # noqa: BLE001 — zarr unavailable/corrupt → try the legacy npz below
        pass

    got = _load_npz_cube(cube_sidecar_path(session_path), fingerprint, n_pixels)
    if got is None:
        return None
    # One-time migration: rewrite the npz cube as a lazily-read Zarr store and hand back THAT,
    # so this (and every later) reopen gets the low-RAM path. Best-effort — if the rewrite
    # fails (read-only share, no zarr) we just return the in-RAM npz cube as before.
    try:
        from .cubestore import CubeStore
        axis, csc, edges = got["cube"]
        store = CubeStore.create(zpath, axis, csc, edges, fingerprint=str(fingerprint),
                                 n_pixels=int(n_pixels), mean=got.get("mean"),
                                 pix=got.get("pix"))
        if store is not None:
            try:
                os.remove(cube_sidecar_path(session_path))   # superseded by the Zarr store
            except OSError:
                pass
            out = {"cube": store}
            if got.get("mean") is not None:
                out["mean"] = got["mean"]
            if got.get("pix") is not None:
                out["pix"] = got["pix"]
            return out
    except Exception:  # noqa: BLE001 — migration is a bonus; the npz cube still works
        pass
    return got


def parse_cache_path(imzml_path: str) -> str:
    """Parse-sidecar path for an imzML — co-located as ``<stem>.parse.npz`` (like the .ibd),
    so it moves with the data and a tmp/throwaway file's cache cleans up with it. Writing is
    best-effort: a read-only data share just means no cache (and a fresh parse next time)."""
    return os.path.splitext(str(imzml_path))[0] + ".parse.npz"


def _parse_stat_key(imzml_path: str, ibd_path: str):
    """A staleness key = (imzML mtime_ns, imzML size, ibd mtime_ns, ibd size). The parse
    sidecar stores ABSOLUTE byte offsets into the .ibd, so it's only valid while both files
    are byte-for-byte the ones that were parsed — a re-export at the same path must force a
    re-parse. ``None`` if either file is missing."""
    try:
        si, sb = os.stat(imzml_path), os.stat(ibd_path)
    except OSError:
        return None
    return [int(si.st_mtime_ns), int(si.st_size), int(sb.st_mtime_ns), int(sb.st_size)]


def save_parse_cache(imzml_path: str, *, mz_offsets, mz_lengths, int_offsets, int_lengths,
                     mz_precision: str, int_precision: str, coordinates,
                     shared=None, polarity: str = "", spec_mode: str = "",
                     pixel_size_um=None, pixel_size_source: str = "") -> str | None:
    """Persist the pyimzml XML-parse result (per-spectrum byte offsets/lengths, precisions,
    coordinates, shared m/z axis + a few header fields) so reopening the file rebuilds the
    store WITHOUT re-running the single-threaded XML parse. Best-effort; returns the path
    or ``None``. Staleness-guarded by :func:`_parse_stat_key` on load."""
    try:
        ibd = os.path.splitext(str(imzml_path))[0] + ".ibd"
        key = _parse_stat_key(imzml_path, ibd)
        if key is None:
            return None
        arrs = {
            "stat_key": np.asarray(key, dtype=np.int64),
            "mz_offsets": np.asarray(mz_offsets, dtype=np.int64),
            "mz_lengths": np.asarray(mz_lengths, dtype=np.int64),
            "int_offsets": np.asarray(int_offsets, dtype=np.int64),
            "int_lengths": np.asarray(int_lengths, dtype=np.int64),
            "coordinates": np.asarray(coordinates, dtype=np.int64),
            "precisions": np.asarray([str(mz_precision), str(int_precision)]),
            "text": np.asarray([polarity or "", spec_mode or "", pixel_size_source or ""]),
            "pixel": np.asarray([np.nan if pixel_size_um is None else float(pixel_size_um)]),
        }
        if shared is not None:
            arrs["shared"] = np.asarray(shared, dtype=np.float64)
        path = parse_cache_path(imzml_path)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez(f, **arrs)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return path
    except Exception:  # noqa: BLE001 — caching is best-effort, never fatal
        return None


def load_parse_cache(imzml_path: str) -> dict | None:
    """Load a parse sidecar written by :func:`save_parse_cache`, or ``None`` if it's missing,
    unreadable, or stale (the imzML/.ibd mtime+size changed since it was written)."""
    path = parse_cache_path(imzml_path)
    if not os.path.exists(path):
        return None
    ibd = os.path.splitext(str(imzml_path))[0] + ".ibd"
    key = _parse_stat_key(imzml_path, ibd)
    if key is None:
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if not np.array_equal(z["stat_key"], np.asarray(key, dtype=np.int64)):
                return None                       # re-exported / touched → offsets may be invalid
            pix = float(z["pixel"][0])
            txt = z["text"]                       # [polarity, spec_mode, pixel_size_source]
            src = str(txt[2]) if len(txt) > 2 else ""   # absent in pre-source sidecars
            return {
                "mz_offsets": z["mz_offsets"], "mz_lengths": z["mz_lengths"],
                "int_offsets": z["int_offsets"], "int_lengths": z["int_lengths"],
                "coordinates": z["coordinates"],
                "mz_precision": str(z["precisions"][0]),
                "int_precision": str(z["precisions"][1]),
                "polarity": str(txt[0]), "spec_mode": str(txt[1]),
                "pixel_size_um": (None if np.isnan(pix) else pix),
                "pixel_size_source": (src or None),
                "shared": (z["shared"] if "shared" in z.files else None),
                "mz_bounds": (z["mz_bounds"] if "mz_bounds" in z.files else None),
            }
    except Exception:  # noqa: BLE001 — a corrupt/partial cache must never be fatal
        return None


def amend_parse_cache(imzml_path: str, **extra) -> str | None:
    """Add or replace arrays in an existing, still-valid parse sidecar (e.g. the m/z bounds
    once a first pass has measured them) without re-parsing. No-op when the sidecar is
    missing or stale. Atomic like :func:`save_parse_cache`; best-effort."""
    path = parse_cache_path(imzml_path)
    if not os.path.exists(path):
        return None
    ibd = os.path.splitext(str(imzml_path))[0] + ".ibd"
    key = _parse_stat_key(imzml_path, ibd)
    if key is None:
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if not np.array_equal(z["stat_key"], np.asarray(key, dtype=np.int64)):
                return None
            arrs = {k: z[k] for k in z.files}
        arrs.update({k: np.asarray(v) for k, v in extra.items()})
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez(f, **arrs)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return path
    except Exception:  # noqa: BLE001 — caching is best-effort, never fatal
        return None


def mask_from_indices(indices, n_pixels) -> np.ndarray | None:
    """Rebuild a boolean pixel mask from a saved index list."""
    if indices is None:
        return None
    mask = np.zeros(int(n_pixels), dtype=bool)
    idx = np.asarray(indices, dtype=int)
    idx = idx[(idx >= 0) & (idx < n_pixels)]
    mask[idx] = True
    return mask


# --------------------------------------------------------------------------- #
# managed per-sample store  (~/.smile-msi/sessions/) — the app auto-saves here
# --------------------------------------------------------------------------- #
def sessions_dir() -> str:
    """Directory holding the auto-saved per-sample sessions; created on demand.
    (``save_session`` does not makedirs, so this must run before the first write.)"""
    from . import library
    d = os.path.join(library.home_dir(), "sessions")
    os.makedirs(d, exist_ok=True)
    return d


def _sanitize(name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")
    return base or "session"


def _display_name(source) -> str:
    if not source:
        return "(unknown)"
    if source == "synthetic":
        return "Demo dataset"
    return os.path.basename(str(source))


def stable_fingerprint(fingerprint) -> str:
    """Reduce a full dataset fingerprint to only the parts that pin a slide's pixel space
    and never drift between loads: the pixel count and the coordinate hash.

    ``library.dataset_fingerprint`` has the shape ``n{px}:w{w}:h{h}:mz{lo}-{hi}:{coordhash}``.
    The grid width/height and the m/z range can legitimately shift between loads of the SAME
    slide — an orientation change transposes w/h, and a single misread spectrum flips the m/z
    bounds to ``NA`` or absurd values — even though the pixels (and therefore the region /
    segmentation indices) are identical. Keying the managed-session filename on the whole
    fingerprint (see :func:`managed_path`) therefore re-keyed the file on every such shift,
    stranding the sample's ROIs under the old name. The pixel count + coordinate hash alone
    are the right identity for the on-disk key and for :func:`fingerprint_mismatch`.

    Keeps the first (``n{px}``) and last (coordhash) colon-separated fields. Returns the
    input unchanged when it doesn't parse (older/other formats), so callers stay
    backward-compatible — an arbitrary/legacy fingerprint string keys exactly as before."""
    s = str(fingerprint or "")
    parts = s.split(":")
    if len(parts) >= 3 and parts[0].startswith("n"):
        return f"{parts[0]}:{parts[-1]}"
    return s


def managed_path(source, fingerprint) -> str:
    """Stable managed-file path for a sample: ``<basename>__<fp8>.json``. The 8-char
    suffix is a hash of the slide's *stable* identity (:func:`stable_fingerprint`, falling
    back to the source), so two different slides sharing a file name — or the same slide
    loaded at a different subsample stride (different pixel set) — get distinct files, while
    the SAME slide always maps to the SAME file across reloads even if its m/z range or grid
    orientation reads differently (which used to orphan its ROIs)."""
    from . import library
    key = library.dataset_key(source)
    stem = _sanitize(os.path.splitext(os.path.basename(str(key)))[0])
    seed = stable_fingerprint(fingerprint) or str(source) or stem
    fp8 = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]
    return os.path.join(sessions_dir(), f"{stem}__{fp8}.json")


def _fingerprint_coordhash(fingerprint) -> str:
    """The trailing coordinate-hash field of a full fingerprint (the last ``:``-token), or
    ``""`` when it doesn't parse. Two managed sessions with the same coordhash + pixel count
    describe the same pixel space, so their region masks are mergeable/comparable."""
    parts = str(fingerprint or "").split(":")
    return parts[-1] if len(parts) >= 3 and parts[0].startswith("n") else ""


def _same_slide_session(data: dict, source, fingerprint, n_pixels) -> bool:
    """Whether a loaded managed-session ``data`` belongs to the slide identified by
    ``(source, fingerprint, n_pixels)``.

    Identity is source basename AND pixel geometry — BOTH must match:
      * basename (``library.dataset_key``) — because two slides can share a coordinate
        raster (same instrument grid → identical coordinate hash), so geometry alone would
        cross-adopt one slide's ROIs onto another;
      * geometry — the stable fingerprint (pixel count + coordinate hash) when both sides
        carry a parseable fingerprint, else the pixel count — because two *different* slides
        can share a basename (same file name in different folders), which basename alone
        can't tell apart.
    This is exactly the pair the on-disk key (``<basename>__<hash(stable_fp)>``) encodes."""
    from . import library
    if library.dataset_key(data.get("source", "")) != library.dataset_key(source):
        return False
    saved_fp = data.get("dataset_fingerprint")
    if saved_fp and fingerprint and _fingerprint_coordhash(saved_fp) and _fingerprint_coordhash(fingerprint):
        return stable_fingerprint(saved_fp) == stable_fingerprint(fingerprint)
    if n_pixels is not None and data.get("n_pixels") is not None:
        return int(data.get("n_pixels")) == int(n_pixels)
    return True   # same basename, no geometry to compare on either side → treat as same slide


def best_existing_session(source, fingerprint, n_pixels=None) -> str | None:
    """Path of the richest existing managed session for this slide, or ``None``.

    "Richest" = the most named regions, tie-broken by newest mtime. Used to find a sample's
    ROIs even when a past fingerprint drift auto-saved them under a differently-hashed
    filename (the orphaning bug). Only files that pass :func:`_same_slide_session` are
    considered, so it never returns another slide's session."""
    d = sessions_dir()
    best, best_key = None, None
    for fn in os.listdir(d):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(d, fn)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            mtime = os.stat(path).st_mtime
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if not _same_slide_session(data, source, fingerprint, n_pixels):
            continue
        key = (len(data.get("named_regions") or []), mtime)
        if best_key is None or key > best_key:
            best, best_key = path, key
    return best


def resolve_session_path(source, fingerprint, n_pixels=None) -> str:
    """The managed-session path to read/write for this slide, self-healing the historical
    fingerprint-drift orphaning: the canonical stable-key path when it exists; else the
    richest existing sibling for the *same* slide (so its ROIs are picked up instead of
    lost); else the canonical path for a genuinely new sample."""
    canonical = managed_path(source, fingerprint)
    if os.path.exists(canonical):
        return canonical
    return best_existing_session(source, fingerprint, n_pixels) or canonical


def existing_named_regions(path) -> list:
    """The ``named_regions`` stored in an existing managed session, or ``[]`` when the file
    is missing/unreadable. Lets the auto-save refuse to blank a populated ROI set on disk
    with an empty in-memory one (a restore that never ran / failed / mis-scoped overlay)."""
    try:
        if not path or not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("named_regions") or []
    except (OSError, json.JSONDecodeError, ValueError):
        return []


def _scan_json_dir(directory: str, summarize) -> list[dict]:
    """Load every ``*.json`` in ``directory`` (corrupt/partial files skipped) and map each
    parsed ``(filename, data)`` through ``summarize(fn, data) -> dict``, stamping ``path``
    and ``mtime`` onto every record. Returns the summaries newest-first. Shared by
    :func:`list_managed` and :func:`smile_msi.cohort.list_cohorts`."""
    out = []
    for fn in sorted(os.listdir(directory)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(directory, fn)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            mtime = os.stat(path).st_mtime
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        rec = summarize(fn, data)
        rec["path"] = path
        rec["mtime"] = mtime
        out.append(rec)
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


def list_managed() -> list[dict]:
    """Summaries of every managed session, newest first, for the startup launcher.
    Corrupt/partial files are skipped (mirrors library's defensive load)."""
    return _scan_json_dir(sessions_dir(), lambda fn, data: {
        "source": data.get("source", ""),
        "name": _display_name(data.get("source", "")),
        "n_pixels": data.get("n_pixels"),
        "n_features": len(data.get("peaks", []) or []),
        "active_mz": data.get("active_mz"),
    })


# --------------------------------------------------------------------------- #
# cache inventory — what the managed store holds and what can go
# --------------------------------------------------------------------------- #
def _path_size(path: str) -> int:
    if os.path.isdir(path):
        total = 0
        for dirpath, _dirs, files in os.walk(path):
            for fn in files:
                try:
                    total += os.stat(os.path.join(dirpath, fn)).st_size
                except OSError:
                    pass
        return total
    try:
        return os.stat(path).st_size
    except OSError:
        return 0


_COMPANION_SUFFIXES = (".cube.zarr", ".cache.npz", ".runs", "__thumbs")


def cache_inventory(directory: str | None = None) -> list[dict]:
    """Every file and folder in the managed store with what it is and whether it can go.

    Rows are ``{path, name, kind, status, size, removable, stem}``. Kinds: ``session``
    (the JSON), ``cube`` (``.cube.zarr``), ``legacy cube`` (``.cache.npz``), ``runs``,
    ``thumbnails``, ``temporary`` (``.tmp`` / ``.cube_spill_*``) and ``other``. A companion
    whose session JSON is gone is *orphaned*; a session that is an older-fingerprint copy of
    a slide that also has a richer/newer session is a *duplicate* (its companions with it);
    a legacy npz that already has a Zarr cube beside it is *superseded*; leftovers of an
    interrupted build are *temporary*. Only those are marked removable — a session whose
    imzML is missing is reported (``source missing``) but kept, since it holds the ROIs."""
    from . import library
    d = directory or sessions_dir()
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return []
    sessions: dict[str, dict] = {}          # stem -> loaded JSON (+ path/mtime)
    rows: list[dict] = []
    for fn in names:
        path = os.path.join(d, fn)
        if fn.endswith(".json"):
            stem = fn[:-5]
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                sessions[stem] = {"data": data, "path": path, "mtime": os.stat(path).st_mtime}
            except (OSError, json.JSONDecodeError, ValueError):
                rows.append({"path": path, "name": fn, "kind": "session", "stem": stem,
                             "status": "unreadable", "size": _path_size(path),
                             "removable": True})
    # duplicates: same slide (basename + stable geometry) saved under several fingerprints
    groups: dict[tuple, list[str]] = {}
    for stem, rec in sessions.items():
        data = rec["data"]
        src = data.get("source", "")
        fp = data.get("dataset_fingerprint") or ""
        geo = stable_fingerprint(fp) if (fp and _fingerprint_coordhash(fp)) else f"n={data.get('n_pixels')}"
        groups.setdefault((library.dataset_key(src), geo), []).append(stem)
    dup: set[str] = set()
    for stems in groups.values():
        if len(stems) < 2:
            continue
        keep = max(stems, key=lambda st: (len(sessions[st]["data"].get("named_regions") or []),
                                          sessions[st]["mtime"]))
        dup.update(st for st in stems if st != keep)
    for stem, rec in sessions.items():
        src = rec["data"].get("source", "")
        if stem in dup:
            status, removable = "duplicate (older fingerprint of the same slide)", True
        elif src and src != "synthetic" and not os.path.exists(src):
            status, removable = "source missing (ROIs kept)", False
        else:
            status, removable = "in use", False
        rows.append({"path": rec["path"], "name": stem + ".json", "kind": "session",
                     "stem": stem, "status": status, "size": _path_size(rec["path"]),
                     "removable": removable})
    for fn in names:
        path = os.path.join(d, fn)
        if fn.endswith(".json"):
            continue
        if fn.startswith(".cube_spill_") or fn.endswith(".tmp"):
            rows.append({"path": path, "name": fn, "kind": "temporary", "stem": "",
                         "status": "leftover from an interrupted build",
                         "size": _path_size(path), "removable": True})
            continue
        kind, stem = "other", ""
        for suf in _COMPANION_SUFFIXES:
            if fn.endswith(suf):
                stem = fn[: -len(suf)]
                kind = {".cube.zarr": "cube", ".cache.npz": "legacy cube",
                        ".runs": "runs", "__thumbs": "thumbnails"}[suf]
                break
        if kind == "other":
            rows.append({"path": path, "name": fn, "kind": kind, "stem": "",
                         "status": "not managed here", "size": _path_size(path),
                         "removable": False})
            continue
        if stem not in sessions:
            status, removable = "orphaned (no session)", True
        elif stem in dup:
            status, removable = "belongs to a duplicate session", True
        elif kind == "legacy cube" and (stem + ".cube.zarr") in names:
            status, removable = "superseded by the Zarr cube", True
        elif kind == "legacy cube":
            status, removable = "legacy (migrates on next open)", False
        else:
            status, removable = "in use", False
        rows.append({"path": path, "name": fn, "kind": kind, "stem": stem,
                     "status": status, "size": _path_size(path), "removable": removable})
    rows.sort(key=lambda r: (r["stem"] or r["name"], r["kind"]))
    return rows


def delete_cache_paths(paths) -> tuple[list[str], list[str]]:
    """Remove the given store entries (files or folders). Returns ``(deleted, failed)``."""
    import shutil
    deleted, failed = [], []
    for p in paths:
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
            deleted.append(p)
        except OSError:
            failed.append(p)
    return deleted, failed


def unique_feature_list_name(existing, base: str) -> str:
    """A feature-list name based on ``base`` that doesn't collide with ``existing``
    (an iterable of names) — appends ``(2)``, ``(3)``… as needed. Used when a list is
    auto-named on generation (co-localized panels, Venn compartments)."""
    base = (base or "").strip() or "Feature list"
    names = set(existing)
    if base not in names:
        return base
    i = 2
    while f"{base} ({i})" in names:
        i += 1
    return f"{base} ({i})"


def fingerprint_mismatch(saved_fp, cur_fp, saved_n=None, cur_n=None) -> bool:
    """True when a restored session was captured on a *different* slide than the one
    now loaded: a fingerprint mismatch (preferred) or, for legacy sessions without a
    fingerprint, a pixel-count mismatch. Unknown/missing on either side → False.

    Compares on the *stable* identity (pixel count + coordinate hash), not the raw string,
    so the same slide reloaded with a differently-read m/z range or transposed grid does not
    read as a mismatch (that drift is expected and must not warn)."""
    if saved_fp and cur_fp:
        return stable_fingerprint(saved_fp) != stable_fingerprint(cur_fp)
    if not saved_fp and saved_n is not None and cur_n is not None:
        return int(saved_n) != int(cur_n)
    return False
