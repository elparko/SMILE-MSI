"""Analysis Profiles — named, versioned, shareable bundles of the processing
settings that should stay **constant across samples and even across slides**.

Where :mod:`smile_msi.prefs` is a flat key→value store for app-wide UI state and
:mod:`smile_msi.session` captures one *dataset's* analysis, a Profile is the third
leg: the *method*. It answers "what settings does this study use?" with a single,
citable object — peak-picking thresholds, mass tolerances, normalization, the
pre-processing chain, segmentation defaults, the spatial-finder gates, and the
random seed — stamped with the app + engine versions that produced it.

Design (matches the answers the feature was scoped against):

* **Global defaults, soft.** The *active* profile only ever *pre-fills* defaults; it
  never blocks a user from changing a value for one sample. Deviations are recorded
  (:func:`diff`), not prevented.
* **Named / versioned / shareable.** Each profile is a JSON file in
  ``<home>/profiles/<slug>.json``; the active one is remembered in ``prefs.json``
  (so it persists between sessions). "Save as new version" bumps ``version`` and the
  file can be exported/imported to share with the lab.
* **A read-only baseline.** :data:`BUILTIN_NAME` ("SMILE default") is synthesised from
  the schema defaults — the field-consensus values — so there is *always* an active
  profile and a validated starting point.

The :data:`SCHEMA` is the single source of truth: it drives the editor UI, the
canonical defaults, forward-compatible filling of older files, and the human-readable
methods record. Add a knob once, here, and every consumer picks it up.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from . import library, prefs, provenance

BUILTIN_NAME = "SMILE default"
_ACTIVE_KEY = "active_profile"          # prefs key holding the active profile's name

# prefs keys the active profile mirrors so subsystems that read prefs directly
# (segmentation, the stochastic engine) reflect the active profile without a handle
# to it. Kept in sync by :func:`apply_to_prefs` (called on set_active / apply).
SEED_PREF = "random_seed"
SEG_METHOD_PREF = "seg_method_default"
SEG_METRIC_PREF = "seg_metric_default"
# segment count / detail already have keys owned by gui.segment — reuse them verbatim
SEG_COUNT_PREF = "segment_default_count"
SEG_DETAIL_PREF = "segment_detail_max"


@dataclass(frozen=True)
class Param:
    """One profile knob. ``kind`` drives the editor widget and value coercion."""
    key: str
    label: str
    group: str
    kind: str                       # 'choice' | 'float' | 'int' | 'text'
    default: object
    choices: tuple = ()
    lo: float | None = None
    hi: float | None = None
    decimals: int = 3
    help: str = ""


# --------------------------------------------------------------------------- #
# The canonical schema — defaults here ARE the built-in "SMILE default" profile.
# Mirror the live-widget defaults in gui/main.py and gui/segment.py.
# --------------------------------------------------------------------------- #
SCHEMA: tuple[Param, ...] = (
    # ---- Acquisition & matching -----------------------------------------
    Param("mode", "Polarity", "Acquisition", "choice", "negative",
          choices=("negative", "positive"),
          help="Ionization mode; sets the adduct set used for lipid identification."),
    Param("ppm", "Extraction tolerance (ppm)", "Acquisition", "float", 10.0,
          lo=1.0, hi=200.0, decimals=1,
          help="m/z integration window for ion images, the feature matrix, and spatial stats."),
    Param("id_ppm", "Identification tolerance (ppm)", "Acquisition", "float", 5.0,
          lo=1.0, hi=30.0, decimals=1,
          help="Mass-accuracy window for in-silico lipid identification (tighter than extraction)."),
    # ---- Ion image ------------------------------------------------------
    Param("norm", "Normalization", "Ion image", "choice", "tic",
          choices=("tic", "none", "rms"),
          help="Per-pixel intensity normalization for ion images and analyses."),
    Param("reduce", "Reduce (m/z window)", "Ion image", "choice", "sum",
          choices=("sum", "max", "mean"),
          help="How intensities inside the extraction window collapse to one value."),
    Param("tic_max_amp", "TIC amplification cap (×)", "Ion image", "float", 3.0,
          lo=1.0, hi=50.0, decimals=1,
          help="Caps per-pixel TIC amplification so faint/low-TIC pixels can't explode "
               "into fake hotspots."),
    # ---- Peak picking ---------------------------------------------------
    Param("snr", "Min S/N", "Peak picking", "float", 3.0, lo=1.0, hi=50.0, decimals=1,
          help="Signal-to-noise floor a peak must clear."),
    Param("prominence", "Min prominence (S/N)", "Peak picking", "float", 1.0,
          lo=0.0, hi=20.0, decimals=1,
          help="How far a peak must rise above its neighbouring saddle; 0 keeps every local max."),
    Param("minrel", "Min rel. intensity", "Peak picking", "float", 0.002,
          lo=0.0, hi=1.0, decimals=4,
          help="Fraction of the base peak below which candidates are dropped."),
    Param("projection", "Spectrum", "Peak picking", "choice", "mean",
          choices=("mean", "skyline (max)"),
          help="Pick on the mean spectrum or the per-m/z maximum (skyline)."),
    Param("max_peaks", "Max peaks (0 = no limit)", "Peak picking", "int", 0,
          lo=0, hi=200000,
          help="Keep at most this many peaks, most intense first. 0 keeps every peak that "
               "clears the gates; a cap makes two regions with different numbers of "
               "detectable ions report the same count."),
    # ---- Pre-processing chain (preprocess.py) ---------------------------
    Param("baseline_method", "Baseline", "Pre-processing", "choice", "none",
          choices=("none", "SNIP", "local minimum", "convex hull", "median"),
          help="Per-spectrum baseline reduction applied before peak picking."),
    Param("baseline_param", "Baseline strength", "Pre-processing", "int", 40,
          lo=3, hi=400, help="SNIP iterations / window size for the baseline method."),
    Param("smooth_method", "Smoothing", "Pre-processing", "choice", "none",
          choices=("none", "Savitzky-Golay", "Gaussian"),
          help="Per-spectrum smoothing."),
    Param("smooth_param", "Smoothing strength", "Pre-processing", "float", 9.0,
          lo=0.5, hi=51.0, decimals=1, help="Window (Savitzky-Golay) or sigma (Gaussian)."),
    Param("recal_refs", "Lock-mass refs (m/z)", "Pre-processing", "text", "",
          help="Comma/space-separated reference m/z for lock-mass recalibration; blank = off."),
    Param("recal_tol", "Lock-mass tol. (ppm)", "Pre-processing", "float", 200.0,
          lo=1.0, hi=1000.0, decimals=0, help="Search window around each lock mass."),
    Param("pp_norm_method", "Per-spectrum norm.", "Pre-processing", "choice", "none",
          choices=("none", "vector (L2)", "reference m/z"),
          help="Axis-preserving per-spectrum normalization during pre-processing."),
    Param("pp_norm_ref", "Per-spectrum norm. ref m/z", "Pre-processing", "text", "",
          help="Reference m/z when per-spectrum normalization is 'reference m/z'."),
    # ---- Segmentation ---------------------------------------------------
    Param("seg_method", "Algorithm", "Segmentation", "choice", "bisecting",
          choices=("bisecting", "ward"),
          help="Clustering algorithm for spatial segmentation."),
    Param("seg_metric", "Distance metric", "Segmentation", "choice", "correlation",
          choices=("correlation", "euclidean"),
          help="Distance metric for segmentation."),
    Param("segment_default_count", "Default segment count", "Segmentation", "int", 8,
          lo=2, hi=64, help="Starting number of segments when a slide is first segmented."),
    Param("segment_detail_max", "Detail slider max", "Segmentation", "int", 24,
          lo=2, hi=200, help="Upper bound of the segmentation Detail slider."),
    # ---- Spatial feature finder ----------------------------------------
    Param("spatial_projection", "Candidates", "Spatial finder", "choice", "mean",
          choices=("mean", "skyline (max)", "both"),
          help="Candidate spectrum for the spatially-aware feature finder."),
    Param("spatial_max_candidates", "Max candidates (0 = no limit)", "Spatial finder",
          "int", 2000, lo=0, hi=200000,
          help="How many detected peaks enter the spatial gates, most intense first. Each "
               "extra candidate costs one Moran's I evaluation."),
    Param("spatial_min_freq", "Min frequency (%)", "Spatial finder", "float", 1.0,
          lo=0.0, hi=100.0, decimals=1,
          help="Minimum fraction of pixels a feature must occupy to survive."),
    Param("spatial_min_morans", "Min Moran's I", "Spatial finder", "float", 0.05,
          lo=-1.0, hi=1.0, decimals=2,
          help="Spatial-autocorrelation floor; strips spatially-random noise."),
    Param("spatial_fdr", "FDR threshold", "Spatial finder", "choice", "off",
          choices=("off", "≤ 5%", "≤ 10%", "≤ 20%"),
          help="Annotation false-discovery-rate gate for the spatial finder."),
    # ---- Batch correction ----------------------------------------------
    Param("batch_method", "Method", "Batch correction", "choice", "combat",
          choices=("combat",),
          help="Cross-batch harmonization method (empirical-Bayes ComBat)."),
    Param("batch_level", "Correction level", "Batch correction", "choice", "region means",
          choices=("region means", "sample means", "pixels"),
          help="What ComBat operates on. Region/sample means are recommended — pixel-level "
               "correction can erase genuine within-sample spatial biology."),
    Param("batch_mean_only", "Location-only (skip scale)", "Batch correction", "choice", "no",
          choices=("no", "yes"),
          help="Correct only the per-batch location shift, leaving the scale untouched."),
    # ---- Reporting (standards-compliant imzML export / METASPACE) -------
    Param("report_instrument", "Instrument", "Reporting", "text", "",
          help="Instrument model recorded in exported imzML metadata (minimum-reporting)."),
    Param("report_mass_analyzer", "Mass analyzer", "Reporting", "text", "",
          help="e.g. FT-ICR, Orbitrap, TOF — recorded in imzML reporting metadata."),
    Param("report_matrix", "MALDI matrix", "Reporting", "text", "",
          help="Matrix + application (e.g. DHB, sublimation) for the reporting checklist."),
    Param("report_ionization", "Ionization", "Reporting", "choice", "MALDI",
          choices=("MALDI", "DESI", "SIMS", "other"),
          help="Ionization source recorded in imzML reporting metadata."),
    # ---- Quantification (on-tissue calibration) ------------------------
    Param("quant_tol_ppm", "Extraction tol (ppm)", "Quantification", "float", 10.0,
          lo=1.0, hi=100.0, decimals=1,
          help="m/z window for reading the calibration analyte / internal standard."),
    Param("quant_weighting", "Curve weighting", "Quantification", "choice", "none",
          choices=("none", "1/x", "1/x2"),
          help="Least-squares weighting for the calibration fit (down-weights high levels)."),
    Param("quant_through_origin", "Force through origin", "Quantification", "choice", "no",
          choices=("no", "yes"),
          help="Constrain the calibration line to pass through (0, 0)."),
    Param("quant_units", "Concentration units", "Quantification", "text", "a.u.",
          help="Units reported on the back-calculated concentration maps."),
    # ---- MS/MS spectral-library matching -------------------------------
    Param("ms2_metric", "Similarity metric", "MS/MS", "choice", "entropy",
          choices=("entropy", "cosine", "modified_cosine"),
          help="Spectral-similarity metric for library MS/MS confirmation (Li et al. 2021)."),
    Param("ms2_tol_da", "Fragment tol (Da)", "MS/MS", "float", 0.02,
          lo=0.001, hi=1.0, decimals=3,
          help="Mass tolerance for matching fragment peaks against the library."),
    # ---- Registration (MSI <-> histology) ------------------------------
    Param("registration_kind", "Transform", "Registration", "choice", "affine",
          choices=("similarity", "rigid", "affine", "piecewise-affine"),
          help="Landmark transform model aligning the optical/histology image to the MSI grid."),
    Param("registration_deformable", "Deformable refine", "Registration", "choice", "no",
          choices=("no", "yes"),
          help="Optional elastic B-spline refinement after the affine (needs the register extra)."),
    # ---- Single-cell (subcellular spatial metabolomics) ----------------
    Param("sc_backend", "Cell segmentation", "Single-cell", "choice", "watershed",
          choices=("watershed", "cellpose", "stardist"),
          help="Cell-segmentation backend on the co-registered microscopy image (watershed needs "
               "no extra; cellpose/stardist are optional)."),
    Param("sc_min_diameter_um", "Min cell diameter (µm)", "Single-cell", "float", 6.0,
          lo=1.0, hi=100.0, decimals=1, help="Smallest expected cell diameter for segmentation."),
    Param("sc_min_support", "Min pixel support", "Single-cell", "float", 0.25,
          lo=0.0, hi=10.0, decimals=2,
          help="Minimum MSI-pixel overlap area for a cell to be kept in the per-cell matrix."),
    # ---- 3D reconstruction (serial sections) ---------------------------
    Param("stack_reg_mode", "Section registration", "3D reconstruction", "choice", "rigid",
          choices=("rigid", "affine", "deformable"),
          help="How serial sections are aligned before stacking (rigid = FFT phase correlation)."),
    Param("stack_spacing_um", "Section spacing (µm)", "3D reconstruction", "float", 10.0,
          lo=0.1, hi=1000.0, decimals=1, help="Physical z spacing between serial sections."),
    # ---- Multi-omics co-mapping ----------------------------------------
    Param("comap_corr_method", "Correlation", "Multi-omics", "choice", "spearman",
          choices=("spearman", "pearson", "cosine"),
          help="Metabolite↔gene spatial-correlation metric for the co-mapped modality."),
    Param("comap_min_pixels", "Min overlapping pixels", "Multi-omics", "int", 20,
          lo=1, hi=100000, help="Minimum co-registered pixels required to report a correlation."),
    # ---- Learned embedding (self-supervised ion-image representation) ---
    Param("embed_dim", "Embedding dim", "Learned embedding", "int", 32, lo=4, hi=512,
          help="Dimensionality of the learned per-ion embedding (needs the ion-embed extra)."),
    Param("embed_epochs", "Training epochs", "Learned embedding", "int", 30, lo=1, hi=500,
          help="Contrastive (NT-Xent) training epochs for the ion-image encoder."),
    Param("embed_p_missing", "Pixel dropout (aug)", "Learned embedding", "float", 0.2,
          lo=0.0, hi=0.9, decimals=2,
          help="Fraction of on-tissue pixels randomly zeroed per augmented view."),
    # ---- Reproducibility ------------------------------------------------
    Param("random_seed", "Random seed", "Reproducibility", "int", 0,
          lo=0, hi=2_147_483_647,
          help="Seed for every stochastic step (clustering, embeddings) so a re-run on the "
               "same machine reproduces identical results."),
)

_BY_KEY = {p.key: p for p in SCHEMA}
GROUPS = tuple(dict.fromkeys(p.group for p in SCHEMA))   # ordered, unique


def _coerce(p: Param, v):
    """Coerce a stored/edited value to the param's type, clamping numerics to range."""
    try:
        if p.kind == "float":
            v = float(v)
        elif p.kind == "int":
            v = int(round(float(v)))
        elif p.kind == "choice":
            v = str(v)
            return v if v in p.choices else p.default
        else:                                   # text
            return str(v)
    except (TypeError, ValueError):
        return p.default
    if p.lo is not None:
        v = max(p.lo if p.kind == "float" else int(p.lo), v)
    if p.hi is not None:
        v = min(p.hi if p.kind == "float" else int(p.hi), v)
    return v


def default_params() -> dict:
    """The canonical defaults — the parameter set of the built-in profile."""
    return {p.key: p.default for p in SCHEMA}


def merge_defaults(params: dict | None) -> dict:
    """Return ``params`` with any missing/unknown keys reconciled against the schema —
    so an older saved profile (or a newer app) always yields a complete, valid dict."""
    src = params or {}
    return {p.key: _coerce(p, src.get(p.key, p.default)) for p in SCHEMA}


# --------------------------------------------------------------------------- #
# Profile objects (plain dicts: name, version, created, params, app/engine ver)
# --------------------------------------------------------------------------- #
def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def content_hash(profile: dict) -> str:
    """Short, stable hash of a profile's *parameters* (ignores name/version/timestamp) —
    so two profiles with identical settings collide, and any settings change is visible."""
    blob = json.dumps(merge_defaults(profile.get("params")), sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]


def make_profile(name: str, params: dict | None = None, *, version: int = 1) -> dict:
    """Assemble a profile dict, stamping the app + engine versions that defined it."""
    from . import __version__
    return {
        "name": str(name).strip() or "Untitled",
        "version": int(version),
        "created": _now(),
        "params": merge_defaults(params),
        "app_version": __version__,
        "engine_versions": provenance._versions(),
    }


def builtin() -> dict:
    """The read-only baseline profile, synthesised from the schema defaults."""
    p = make_profile(BUILTIN_NAME, default_params(), version=1)
    p["builtin"] = True
    return p


def bump_version(profile: dict, *, name: str | None = None) -> dict:
    """A new profile snapshot with ``version`` incremented (and a fresh timestamp/versions)."""
    return make_profile(name or profile.get("name", "Untitled"),
                        profile.get("params"), version=int(profile.get("version", 1)) + 1)


def label(profile: dict) -> str:
    """Display label, e.g. ``Nerve-Lipid v2``."""
    return f"{profile.get('name', '?')} v{int(profile.get('version', 1))}"


def diff(current_params: dict, profile: dict) -> dict:
    """Keys where ``current_params`` deviates from ``profile``'s params.

    Returns ``{key: [profile_value, current_value]}`` — the record written into a
    session/provenance so any departure from the active profile is auditable, never silent.
    """
    base = merge_defaults(profile.get("params"))
    out = {}
    for k, cur in (current_params or {}).items():
        if k not in _BY_KEY:
            continue
        cv = _coerce(_BY_KEY[k], cur)
        if cv != base.get(k):
            out[k] = [base.get(k), cv]
    return out


# --------------------------------------------------------------------------- #
# File store  (<home>/profiles/<slug>-v<version>.json)
#
# Every (name, version) is its own immutable file, so "save as new version" keeps the
# old one on disk — a result that recorded "Nerve-Lipid v2" can still be reloaded and
# reproduced after the lab has moved on to v3. The active pointer references a *name*;
# the active/latest version is the highest on disk for that name.
# --------------------------------------------------------------------------- #
def _dir() -> str:
    d = os.path.join(library.home_dir(), "profiles")
    os.makedirs(d, exist_ok=True)
    return d


def _slug(name: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in str(name)]
    s = "".join(keep).strip("-")
    while "--" in s:
        s = s.replace("--", "-")
    return s or "profile"


def _path(name: str, version: int) -> str:
    return os.path.join(_dir(), f"{_slug(name)}-v{int(version)}.json")


def _read_all() -> list[dict]:
    """Every saved (non-builtin) profile file, params reconciled against the schema."""
    out = []
    try:
        files = sorted(f for f in os.listdir(_dir()) if f.endswith(".json"))
    except OSError:
        files = []
    for f in files:
        try:
            with open(os.path.join(_dir(), f), encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("name") and data.get("name") != BUILTIN_NAME:
            data["params"] = merge_defaults(data.get("params"))
            out.append(data)
    return out


def list_profiles() -> list[dict]:
    """The built-in baseline first, then the *latest* version of each saved profile."""
    latest: dict[str, dict] = {}
    for d in _read_all():
        n = d["name"]
        if n not in latest or int(d.get("version", 1)) > int(latest[n].get("version", 1)):
            latest[n] = d
    return [builtin()] + [latest[n] for n in sorted(latest)]


def versions_of(name: str) -> list[int]:
    """Sorted version numbers present on disk for ``name`` (empty for the built-in)."""
    return sorted({int(d.get("version", 1)) for d in _read_all() if d.get("name") == name})


def load(name: str | None) -> dict:
    """The *latest* version of ``name`` (the built-in, or any missing name, is the baseline)."""
    if not name or name == BUILTIN_NAME:
        return builtin()
    cands = [d for d in _read_all() if d.get("name") == name]
    return max(cands, key=lambda d: int(d.get("version", 1))) if cands else builtin()


def load_exact(name: str | None, version: int) -> dict:
    """A specific (name, version) — used to reproduce a result from its recorded profile.
    Falls back to the latest version, then the built-in, if that exact file is gone."""
    if not name or name == BUILTIN_NAME:
        return builtin()
    try:
        with open(_path(name, version), encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            data["params"] = merge_defaults(data.get("params"))
            return data
    except (OSError, ValueError):
        pass
    return load(name)


def save(profile: dict) -> str:
    """Persist a user profile as its own (name, version) file. Refuses the built-in name."""
    name = str(profile.get("name", "")).strip()
    if not name or name == BUILTIN_NAME:
        raise ValueError(f"'{BUILTIN_NAME}' is read-only — save under a different name.")
    profile = dict(profile)
    profile.pop("builtin", None)
    profile["params"] = merge_defaults(profile.get("params"))
    path = _path(name, int(profile.get("version", 1)))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def delete(name: str, version: int | None = None) -> None:
    """Remove one version (``version`` given) or every version of ``name``.
    No-op for the built-in. Resets the active pointer if its target is gone."""
    if not name or name == BUILTIN_NAME:
        return
    targets = [version] if version is not None else versions_of(name)
    for v in targets:
        try:
            os.remove(_path(name, v))
        except OSError:
            pass
    if prefs.get(_ACTIVE_KEY) == name and not versions_of(name):
        set_active(BUILTIN_NAME)              # don't leave a dangling active pointer


def export_to(profile: dict, path: str) -> str:
    """Write a profile to an arbitrary path for sharing."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2, ensure_ascii=False)
    return path


def import_from(path: str) -> dict:
    """Read a shared profile file, validating + reconciling it against the schema."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or "params" not in data:
        raise ValueError("Not a profile file (missing 'params').")
    data["params"] = merge_defaults(data.get("params"))
    data.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    data.setdefault("version", 1)
    data.pop("builtin", None)
    return data


# --------------------------------------------------------------------------- #
# Active profile
# --------------------------------------------------------------------------- #
def active() -> dict:
    """The currently active profile (built-in if none is set)."""
    return load(prefs.get(_ACTIVE_KEY))


def active_params() -> dict:
    return merge_defaults(active().get("params"))


def active_seed() -> int:
    """Resolved random seed: the prefs mirror first (so an applied profile wins), else
    the active profile's value. Pass this as ``random_state`` at GUI call sites."""
    return int(prefs.get(SEED_PREF, active_params()["random_seed"]))


def set_active(name: str) -> None:
    """Make ``name`` the active profile and mirror its prefs-backed knobs."""
    prefs.set(_ACTIVE_KEY, name or BUILTIN_NAME)
    apply_to_prefs(active())


def mirror_prefs(params: dict) -> None:
    """Mirror the prefs-backed subset of ``params`` into ``prefs.json`` so the subsystems
    that read prefs directly (segmentation defaults, the stochastic seed) reflect it."""
    p = merge_defaults(params)
    prefs.set(SEED_PREF, int(p["random_seed"]))
    prefs.set(SEG_METHOD_PREF, p["seg_method"])
    prefs.set(SEG_METRIC_PREF, p["seg_metric"])
    prefs.set(SEG_COUNT_PREF, int(p["segment_default_count"]))
    prefs.set(SEG_DETAIL_PREF, int(p["segment_detail_max"]))


def apply_to_prefs(profile: dict) -> None:
    """Mirror a whole profile's prefs-backed knobs (see :func:`mirror_prefs`)."""
    mirror_prefs(profile.get("params"))
