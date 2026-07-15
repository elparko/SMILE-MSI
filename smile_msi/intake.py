"""Adaptive intake inspection — detect what kind of MSI data was just loaded and
suggest the matching processing settings, instead of making the user pick
normalization and tolerance blind.

Two properties drive every downstream decision and are routinely *mislabeled* in
the file, so we detect them from the data itself rather than trusting metadata:

1. **Representation** — *centroid* vs *profile*. imzML carries an MS:1000127 /
   MS:1000128 flag (pyimzml surfaces it as ``ds.spec_mode``), but SCiLS and other
   exporters sometimes set it wrong — and a "processed" storage layout does **not**
   imply centroided data — so we verify the flag against the spectra. Profile peaks
   span several adjacent samples (a hump); centroid peaks are isolated spikes. For
   continuous (shared-axis) data we measure the mean run-length of contiguous
   non-zero samples; for processed (per-spectrum m/z) data we measure the spacing of
   recorded m/z points in ppm.

2. **Normalization state** — *raw* vs already *TIC / RMS / median-normalized*. There
   is no imzML flag for this, so it is purely statistical: if every pixel's total ion
   current is (nearly) identical — coefficient of variation ≈ 0 — the data was
   already TIC-normalized, and re-normalizing it is a redundant (and slightly harmful)
   no-op. Raw biological MSI shows CV ≈ 0.3–1+ because TIC tracks tissue morphology.
   (SCiLS ComparisonExport files measure CV ≈ 0.)

The recommendations are deliberately conservative: we only ever suggest ``"tic"`` or
``"none"`` normalization — never the median / exclusion-list recipes that are the
subject of a pending normalization-patent review (see the withheld ``"median"`` mode
in the Display panel).

Pure NumPy, GUI-free, so it runs headless (tests, CLI, and the future
centroided fast-path that skips profile peak-picking).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import DEFAULT_TOL_PPM

# A per-pixel summary statistic whose coefficient of variation falls below this is
# treated as "held constant on purpose" — i.e. the data was already normalized to it.
NORMALIZED_CV = 0.05
# Profile peaks occupy a run of at least this many adjacent samples on a shared axis;
# centroid peaks are isolated spikes (run length 1).
PROFILE_MIN_RUN = 2.0
# On per-spectrum (processed) storage, profile sampling steps are this tight or
# tighter (instrument grid); centroid peak-to-peak gaps are far wider and irregular.
PROFILE_MAX_GAP_PPM = 30.0
# How many spectra to sample for the representation heuristic (evenly across pixels).
REP_SAMPLE = 48


@dataclass
class IntakeReport:
    """What the loaded dataset is, and the settings that match it."""

    # ---- detected facts ----
    representation: str            # "centroid" | "profile" | "unknown"
    representation_source: str     # how it was determined (flag / heuristic / disputed)
    storage: str                   # "continuous" | "processed"
    normalization: str             # "raw" | "tic" | "rms" | "median" | "unknown"
    polarity: str
    mz_lo: float
    mz_hi: float
    n_pixels: int
    # ---- measured signals (for transparency / tests) ----
    cv: dict = field(default_factory=dict)        # per-pixel CV of tic / rms / median
    rep_metrics: dict = field(default_factory=dict)
    # ---- suggestions (never forced) ----
    suggested_norm: str = "tic"    # "tic" | "none" only
    suggested_tol_ppm: float = 25.0
    suggested_pick: str = "mean-spectrum"  # "mean-spectrum" | "centroids"
    notes: list = field(default_factory=list)      # human-readable recommendations
    warnings: list = field(default_factory=list)   # e.g. flag-vs-data mismatch

    @property
    def already_normalized(self) -> bool:
        return self.normalization in ("tic", "rms", "median")


# --------------------------------------------------------------------------- #
# Representation: centroid vs profile
# --------------------------------------------------------------------------- #
def _run_lengths(mask: np.ndarray) -> np.ndarray:
    """Lengths of each contiguous run of True in a boolean mask."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.array([], dtype=int)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return edges[1::2] - edges[0::2]


def _spectrum_vote(mz: np.ndarray, inten: np.ndarray):
    """Classify one raw spectrum as 'profile' / 'centroid' / 'unknown' with metrics."""
    inten = np.asarray(inten, dtype=np.float64)
    mz = np.asarray(mz, dtype=np.float64)
    if inten.size < 8:
        return "unknown", {}
    nz = inten > 0
    n_nz = int(nz.sum())
    if n_nz < 4:
        return "unknown", {"n_nonzero": n_nz}
    zero_frac = 1.0 - n_nz / inten.size
    if zero_frac >= 0.2:
        # The array stores zeros between peaks (continuous grid / zero-padded): a
        # profile peak spans a run of adjacent samples, a centroid peak is one spike.
        runs = _run_lengths(nz)
        mean_run = float(runs.mean()) if runs.size else 0.0
        label = "profile" if mean_run >= PROFILE_MIN_RUN else "centroid"
        return label, {"zero_frac": zero_frac, "mean_run": mean_run}
    # Sparse storage with no interleaved zeros (processed/centroid-style): fall back
    # to the spacing of the recorded m/z points — tight & regular ⇒ profile grid.
    p = mz[nz]
    p = np.sort(p[np.isfinite(p) & (p > 0)])
    if p.size < 8:
        return "unknown", {"n_points": int(p.size)}
    gaps_ppm = np.diff(p) / p[:-1] * 1e6
    med_gap = float(np.median(gaps_ppm))
    label = "profile" if med_gap <= PROFILE_MAX_GAP_PPM else "centroid"
    return label, {"median_gap_ppm": med_gap, "zero_frac": zero_frac}


def detect_representation(ds, sample: int = REP_SAMPLE):
    """Return ``(label, source, metrics)`` for centroid-vs-profile, reconciling the
    file's spec_mode flag with a data heuristic (the data wins on disagreement —
    SCiLS sometimes mislabels)."""
    store = ds.store
    n = len(store)
    idx = _sample_indices(n, sample)
    votes, runs, gaps = [], [], []
    for i in idx:
        try:
            mz, inten = store.get(int(i))
        except Exception:  # noqa: BLE001 - one bad read shouldn't sink detection
            continue
        v, m = _spectrum_vote(mz, inten)
        if v == "unknown":
            continue
        votes.append(v)
        if "mean_run" in m:
            runs.append(m["mean_run"])
        if "median_gap_ppm" in m:
            gaps.append(m["median_gap_ppm"])
    metrics = {}
    if runs:
        metrics["mean_run"] = float(np.median(runs))
    if gaps:
        metrics["median_gap_ppm"] = float(np.median(gaps))

    if votes:
        n_prof = votes.count("profile")
        data_label = "profile" if n_prof * 2 >= len(votes) else "centroid"
        metrics["profile_fraction"] = n_prof / len(votes)
    else:
        data_label = "unknown"

    flag = (getattr(ds, "spec_mode", "") or "").strip().lower()
    flag = flag if flag in ("centroid", "profile") else ""

    if flag and data_label != "unknown":
        if flag == data_label:
            return data_label, "file flag (confirmed by data)", metrics
        return data_label, f"data heuristic (file flag said '{flag}')", metrics
    if flag:
        return flag, "file flag", metrics
    if data_label != "unknown":
        return data_label, "data heuristic", metrics
    return "unknown", "undetermined", metrics


# --------------------------------------------------------------------------- #
# Normalization state
# --------------------------------------------------------------------------- #
def detect_normalization(ds):
    """Return ``(label, cvs)`` where label is raw / tic / rms / median: a per-pixel
    statistic that is essentially constant across pixels means the data was already
    normalized to it. TIC is checked first (the dominant convention)."""
    ds.prime()
    pix = getattr(ds, "_pix", None) or {}

    def cv(key):
        a = np.asarray(pix.get(key, []), dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size < 2:
            return float("nan")
        m = a.mean()
        return float(a.std() / m) if m > 0 else float("nan")

    cvs = {k: cv(k) for k in ("tic", "rms", "median")}
    label = "raw"
    # NB: the per-pixel *median* is deliberately NOT in the auto-detected set. On
    # genuinely raw data a near-constant noise floor makes median(positive intensities)
    # an extremely stable statistic (CV ≈ 0) even while the TIC swings widely — which
    # would falsely flag raw data as "already median-normalized" and suppress the TIC
    # suggestion (the harmful false-positive direction). Median-based normalization is
    # withheld pending patent review anyway (METHODS.md §3); TIC is the motivated default.
    for key in ("tic", "rms"):
        c = cvs[key]
        if c == c and c <= NORMALIZED_CV:   # c == c rejects NaN
            label = key
            break
    return label, cvs


def _sample_indices(n: int, k: int):
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    return [int(i) for i in np.unique(np.linspace(0, n - 1, k).astype(int))]


# --------------------------------------------------------------------------- #
# Mass-drift QC (does this data even need alignment?)
# --------------------------------------------------------------------------- #
def mass_drift(ds, ref_mz=None, tol_ppm: float = DEFAULT_TOL_PPM, sample: int = 2000) -> dict:
    """Per-pixel mass-drift QC: the ppm deviation of each pixel's observed apex from a
    reference m/z. This is the **measure-before-you-align** check — if the spread is far
    below the extraction tolerance, peak alignment/recalibration is unnecessary (the
    common case for well-calibrated high-res data); if it approaches the tolerance, apply
    :func:`preprocess.auto_recalibrate`. Continuous (shared-axis) data reports ~0 by
    construction (one axis for all pixels); the QC is meaningful for processed-mode data.

    Auto-picks the strongest mean-spectrum peak as the reference when ``ref_mz`` is None.
    Returns ``{ref_mz, n, median_ppm, iqr_ppm, max_abs_ppm, rms_ppm}`` (ppm)."""
    empty = {"ref_mz": ref_mz, "n": 0, "median_ppm": float("nan"),
             "iqr_ppm": float("nan"), "max_abs_ppm": float("nan"), "rms_ppm": float("nan")}
    ds.prime()
    if ref_mz is None:
        pk = ds.pick_peaks(max_peaks=1)
        if not pk:
            return empty
        ref_mz = float(pk[0]["mz"])
    win = ref_mz * tol_ppm / 1e6
    lo, hi = ref_mz - win, ref_mz + win
    n = len(ds.store)
    idx = _sample_indices(n, sample) if (sample and sample < n) else range(n)
    devs = []
    for i in idx:
        try:
            mz, inten = ds.store.get(int(i))
        except Exception:  # noqa: BLE001
            continue
        mz = np.asarray(mz, float)
        inten = np.asarray(inten, float)
        m = (mz >= lo) & (mz <= hi) & (inten > 0)
        if not m.any():
            continue
        sel = np.flatnonzero(m)
        obs = float(mz[sel[int(np.argmax(inten[sel]))]])
        devs.append((obs - ref_mz) / ref_mz * 1e6)
    devs = np.asarray(devs, float)
    if devs.size == 0:
        return {**empty, "ref_mz": ref_mz}
    q1, q3 = np.percentile(devs, [25, 75])
    return {"ref_mz": float(ref_mz), "n": int(devs.size),
            "median_ppm": float(np.median(devs)), "iqr_ppm": float(q3 - q1),
            "max_abs_ppm": float(np.max(np.abs(devs))),
            "rms_ppm": float(np.sqrt(np.mean(devs ** 2)))}


# --------------------------------------------------------------------------- #
# Absolute calibration offset (measure, then lock-mass correct)
# --------------------------------------------------------------------------- #
# Curated, abundant, unambiguous anchor ions — chosen so the nearest strong mean-spectrum
# apex is unmistakably the anchor (no close isobar). Given as (species name in the in-silico
# DB, adduct); the true m/z is computed from the engine (masses.ion_mz over the DB neutral
# mass), so no lock mass is ever hardcoded and the anchors stay consistent with the matcher.
_NEG_CALIB_ANCHORS = [
    ("FA 18:1", "[M-H]-"),            # oleic acid (~281.2486)
    ("FA 22:6", "[M-H]-"),            # DHA
    ("FA 24:1", "[M-H]-"),            # nervonic acid (myelin)
    ("PE 38:4", "[M-H]-"),
    ("PS 40:6", "[M-H]-"),            # brain/nerve marker
    ("PI 38:4", "[M-H]-"),
    ("Sulfatide 42:1;O2", "[M-H]-"),  # myelin sulfatides — the mislabelled ions
    ("Sulfatide 42:2;O2", "[M-H]-"),
]
_POS_CALIB_ANCHORS = [
    ("PC 32:0", "[M+H]+"),
    ("PC 34:1", "[M+H]+"),
    ("SM 34:1", "[M+H]+"),
    ("PE 38:4", "[M+H]+"),
]


def default_calibration_anchors(mode: str = "negative", db=None) -> list:
    """The curated anchor set for ``mode`` resolved to ``[(label, true_mz), ...]`` using the
    in-silico DB neutral masses (so no lock mass is hardcoded). Anchors whose species is not
    in the (possibly filtered/external) database are skipped."""
    from .lipiddb import build_database
    from .masses import ion_mz
    by = {lip.name: lip for lip in (db if db is not None else build_database())}
    anchors = _NEG_CALIB_ANCHORS if mode == "negative" else _POS_CALIB_ANCHORS
    out = []
    for name, adduct in anchors:
        lip = by.get(name)
        if lip is not None:
            out.append((f"{name} {adduct}", ion_mz(lip.neutral_mass, adduct)))
    return out


def _apex_centroid(axis, spec, k, half: int = 2) -> float:
    """Intensity-weighted m/z of the peak at bin ``k`` (± ``half`` bins) — a sub-bin apex
    refine so the observed m/z isn't quantised to the axis grid. Falls back to the grid
    position when the local window has no signal."""
    lo = max(0, k - half)
    hi = min(len(axis), k + half + 1)
    w = spec[lo:hi]
    tot = float(w.sum())
    return float(np.average(axis[lo:hi], weights=w)) if tot > 0 else float(axis[k])


def measure_calibration_offset(ds, references=None, *, mode: str = "negative",
                               tol_ppm: float = 30.0) -> dict:
    """Measure the dataset's **absolute** m/z calibration offset against known reference ions.

    Unlike :func:`mass_drift` (per-pixel *spread* vs. an auto-picked peak) and
    :func:`preprocess.auto_recalibrate` (self-referential spread reduction), this compares
    the **mean spectrum**'s apexes to *known* anchor ion m/z, so it recovers the systematic
    offset (e.g. "−5.6 ppm low") that drives confident lipid mislabels — exactly what a
    one-point lock-mass recalibration should remove.

    ``references`` is ``[(label, true_mz), ...]`` (or bare m/z floats); ``None`` uses the
    curated engine-computed anchor set for ``mode`` (:func:`default_calibration_anchors`).
    For each anchor the nearest mean-spectrum apex within ``tol_ppm`` gives the observed m/z
    and a signed ppm error (sub-bin refined). ``tol_ppm`` must exceed the expected offset
    (default 30 ppm — wide enough to catch a ~−12 ppm drift without grabbing a neighbour).

    Returns ``{anchors: [{label, ref_mz, obs_mz, intensity, ppm}], n, median_ppm, iqr_ppm,
    max_abs_ppm, slope_ppm_per_da, factor}`` where ``factor = 1/(1 + median_ppm·1e-6)`` is
    the multiplicative m/z correction to apply, and ``slope_ppm_per_da`` flags a
    mass-dependent stretch (≈0 → a pure constant offset a single-point lock mass fixes)."""
    empty = {"anchors": [], "n": 0, "median_ppm": float("nan"), "iqr_ppm": float("nan"),
             "max_abs_ppm": float("nan"), "slope_ppm_per_da": float("nan"), "factor": 1.0}
    if references is None:
        references = default_calibration_anchors(mode)
    norm_refs = []
    for x in references:
        if isinstance(x, (tuple, list)):
            norm_refs.append((str(x[0]), float(x[1])))
        else:
            norm_refs.append((f"{float(x):.4f}", float(x)))
    if not norm_refs:
        return empty

    axis, spec = ds.mean_spectrum()
    axis = np.asarray(axis, float)
    spec = np.asarray(spec, float)
    if axis.size == 0:
        return empty

    anchors = []
    for lbl, ref_mz in norm_refs:
        win = ref_mz * tol_ppm / 1e6
        lo = int(np.searchsorted(axis, ref_mz - win, "left"))
        hi = int(np.searchsorted(axis, ref_mz + win, "right"))
        if hi <= lo:
            continue
        seg = spec[lo:hi]
        if not np.any(seg > 0):
            continue
        k = lo + int(np.argmax(seg))
        obs = _apex_centroid(axis, spec, k)
        anchors.append({"label": lbl, "ref_mz": float(ref_mz), "obs_mz": obs,
                        "intensity": float(spec[k]),
                        "ppm": (obs - ref_mz) / ref_mz * 1e6})
    if not anchors:
        return empty

    ppms = np.array([a["ppm"] for a in anchors], float)
    refm = np.array([a["ref_mz"] for a in anchors], float)
    q1, q3 = (np.percentile(ppms, [25, 75]) if ppms.size > 1 else (ppms[0], ppms[0]))
    slope = float(np.polyfit(refm, ppms, 1)[0]) if ppms.size >= 2 else 0.0
    median = float(np.median(ppms))
    return {"anchors": anchors, "n": int(ppms.size), "median_ppm": median,
            "iqr_ppm": float(q3 - q1), "max_abs_ppm": float(np.max(np.abs(ppms))),
            "slope_ppm_per_da": slope, "factor": 1.0 / (1.0 + median * 1e-6)}


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #
def inspect_dataset(ds, sample: int = REP_SAMPLE) -> IntakeReport:
    """Inspect a primed (or primeable) dataset and return an :class:`IntakeReport`
    with detected facts and conservative, overridable setting suggestions."""
    rep, rep_src, rep_metrics = detect_representation(ds, sample)
    norm, cvs = detect_normalization(ds)
    storage = "continuous" if ds.store.shared_axis() is not None else "processed"
    lo, hi = ds.mz_range

    already = norm in ("tic", "rms", "median")
    suggested_norm = "none" if already else "tic"
    suggested_tol = 10.0 if rep == "centroid" else 25.0
    suggested_pick = "centroids" if rep == "centroid" else "mean-spectrum"

    notes, warnings = [], []
    if already:
        c = cvs.get(norm, float("nan"))
        notes.append(
            f"Already {norm.upper()}-normalized (per-pixel {norm} is constant, "
            f"CV≈{c:.3f}). Recommend normalization OFF — re-normalizing is redundant.")
    else:
        c = cvs.get("tic", float("nan"))
        notes.append(
            f"Raw intensities (per-pixel TIC varies, CV≈{c:.2f}). "
            f"Recommend TIC normalization.")
    if rep == "centroid":
        notes.append(
            "Centroided data — skip profile peak-picking; bin/align the existing "
            "centroids and use a tight (~10 ppm) extraction tolerance.")
    elif rep == "profile":
        notes.append(
            "Profile data — pick peaks on the mean spectrum; a wider (~25 ppm) "
            "extraction tolerance suits multi-sample peaks.")

    if "file flag said" in rep_src:
        warnings.append(
            "The file's centroid/profile flag disagrees with the data "
            f"({rep_src}); trusting the data.")

    return IntakeReport(
        representation=rep,
        representation_source=rep_src,
        storage=storage,
        normalization=norm,
        polarity=getattr(ds, "polarity", "") or "",
        mz_lo=float(lo),
        mz_hi=float(hi),
        n_pixels=int(ds.n_pixels),
        cv=cvs,
        rep_metrics=rep_metrics,
        suggested_norm=suggested_norm,
        suggested_tol_ppm=suggested_tol,
        suggested_pick=suggested_pick,
        notes=notes,
        warnings=warnings,
    )
