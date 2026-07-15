"""Tests for the adaptive intake inspector (smile_msi.intake): centroid-vs-profile
representation, raw-vs-normalized detection, and the resulting suggestions."""
import numpy as np
import pytest

from smile_msi import intake, spatial
from smile_msi.msi import MSIDataset

AXIS = np.linspace(150.0, 900.0, 1500)
CENTERS = np.linspace(220.0, 850.0, 20)
COORDS = [(x + 1, y + 1) for y in range(5) for x in range(8)]   # 40 pixels


def _profile_pixel(rng):
    """A dense profile spectrum — each peak is a multi-sample Gaussian hump."""
    inten = np.zeros_like(AXIS)
    for c in CENTERS:
        j = int(np.argmin(np.abs(AXIS - c)))
        lo, hi = max(0, j - 8), min(len(AXIS), j + 9)
        x = np.arange(lo, hi)
        inten[lo:hi] += 100.0 * np.exp(-0.5 * ((x - j) / 3.0) ** 2)
    return inten * rng.uniform(0.3, 3.0)                         # raw: TIC varies per pixel


def _centroid_pixel(rng):
    """A centroided spectrum — each peak is a single isolated spike on a zero grid."""
    inten = np.zeros_like(AXIS)
    for c in CENTERS:
        j = int(np.argmin(np.abs(AXIS - c)))
        inten[j] = 100.0
    return inten * rng.uniform(0.3, 3.0)


def _make(build_pixel, normalize=False, target=1000.0, spec_mode="", seed=0):
    rng = np.random.default_rng(seed)
    mzs, ints = [], []
    for _ in COORDS:
        inten = build_pixel(rng)
        if normalize:
            t = inten.sum()
            if t > 0:
                inten = inten / t * target                      # force identical per-pixel TIC
        mzs.append(AXIS)
        ints.append(inten)
    return MSIDataset.from_arrays(COORDS, mzs, ints, spec_mode=spec_mode)


def test_profile_raw():
    rep = intake.inspect_dataset(_make(_profile_pixel))
    assert rep.representation == "profile"
    assert rep.storage == "continuous"
    assert rep.normalization == "raw"
    assert rep.cv["tic"] > intake.NORMALIZED_CV
    assert rep.suggested_norm == "tic"
    assert rep.suggested_pick == "mean-spectrum"
    assert not rep.already_normalized


def test_centroid_raw():
    rep = intake.inspect_dataset(_make(_centroid_pixel))
    assert rep.representation == "centroid"
    assert rep.normalization == "raw"
    assert rep.suggested_norm == "tic"
    assert rep.suggested_tol_ppm == pytest.approx(10.0)
    assert rep.suggested_pick == "centroids"


def test_already_tic_normalized_suggests_off():
    rep = intake.inspect_dataset(_make(_centroid_pixel, normalize=True))
    assert rep.normalization == "tic"
    assert rep.already_normalized
    assert rep.cv["tic"] < intake.NORMALIZED_CV
    assert rep.suggested_norm == "none"          # don't double-normalize
    assert any("OFF" in n or "redundant" in n for n in rep.notes)


def test_flag_disputed_by_data_warns():
    # Profile data mislabeled as centroid in the file — the data must win, with a warning.
    rep = intake.inspect_dataset(_make(_profile_pixel, spec_mode="centroid"))
    assert rep.representation == "profile"
    assert "file flag said" in rep.representation_source
    assert rep.warnings


def test_is_centroided_detection():
    assert _make(_centroid_pixel).is_centroided() is True
    assert _make(_profile_pixel).is_centroided() is False


def test_pick_peaks_centroid_uses_exact_recorded_mz():
    # Centroided data → the centroid branch keeps the recorded m/z exactly (no profile
    # weighted-centroid re-estimate) and treats every recorded peak as detected.
    ds = _make(_centroid_pixel)
    peaks = ds.pick_peaks(snr=1.0, min_rel_intensity=0.0)
    got = sorted(p["mz"] for p in peaks)
    expected = sorted(float(AXIS[int(np.argmin(np.abs(AXIS - c)))]) for c in CENTERS)
    assert len(peaks) == len(CENTERS)
    assert got == pytest.approx(expected, abs=1e-9)


def test_processed_centroid_via_gap_heuristic():
    # Per-spectrum m/z (no shared axis, no stored zeros): sparse peaks with wide,
    # irregular gaps → centroid through the m/z-spacing path.
    rng = np.random.default_rng(3)
    mzs, ints = [], []
    for k in range(len(COORDS)):
        npk = 25 + (k % 7)                       # vary length so the axis isn't shared
        mz = np.sort(rng.uniform(200.0, 800.0, npk))
        mzs.append(mz)
        ints.append(rng.uniform(10.0, 100.0, npk))
    ds = MSIDataset.from_arrays(COORDS, mzs, ints)
    rep = intake.inspect_dataset(ds)
    assert rep.storage == "processed"
    assert rep.representation == "centroid"


def test_find_spatial_features_isotope_collapse():
    # Four spatially-structured features; 801.003 is the M+1 satellite of 800.0 (less
    # intense). collapse_isotopes folds it into the monoisotopic peak → 4 features become 3.
    axis = np.linspace(700.0, 900.0, 40001)             # fine grid (~0.005 Da) so the
    i = {m: int(np.argmin(np.abs(axis - m)))            # 13C spacing survives quantization
         for m in (720.0, 800.0, 801.00336, 850.0)}
    W, H = 12, 10
    coords = [(x + 1, y + 1) for y in range(H) for x in range(W)]
    mzs, ints = [], []
    for (x, y) in coords:
        v = np.zeros_like(axis)
        if x < W // 2:                                  # left block = coherent "tissue"
            v[i[720.0]] = 80.0
            v[i[800.0]] = 100.0                         # monoisotopic
            v[i[801.00336]] = 30.0                      # M+1 satellite (less intense)
            v[i[850.0]] = 60.0
        mzs.append(axis); ints.append(v)
    ds = MSIDataset.from_arrays(coords, mzs, ints)
    kw = dict(snr=1.0, min_frequency=0.0, min_morans=-1.0, tol_ppm=20.0, norm="none")

    base = spatial.find_spatial_features(ds, **kw)
    coll = spatial.find_spatial_features(ds, collapse_isotopes=True, **kw)

    assert len(base.peaks) == 4
    assert base.n_after_collapse == -1                  # collapse off → sentinel
    assert len(coll.peaks) == 3                         # M+1 folded into monoisotopic
    assert coll.n_after_collapse == 3
    assert coll.n_after_morans == 4                     # pre-collapse survivor count preserved
    coll_mz = sorted(round(p["mz"], 2) for p in coll.peaks)
    assert 800.0 in coll_mz and 801.0 not in coll_mz    # mono kept, satellite removed


def test_mass_drift_qc_measures_spread():
    # Processed-mode data with a known per-pixel ppm drift injected on the 800 peak.
    rng = np.random.default_rng(5)
    coords = [(x + 1, y + 1) for y in range(6) for x in range(8)]   # 48 px
    offs = rng.uniform(-12e-6, 12e-6, len(coords))                  # ±12 ppm drift
    mzs, ints = [], []
    for k in range(len(coords)):
        mzs.append(np.array([400.0, 600.0, 800.0 * (1 + offs[k])]))  # per-pixel m/z → processed
        ints.append(np.array([50.0, 70.0, 100.0]))
    ds = MSIDataset.from_arrays(coords, mzs, ints)
    qc = intake.mass_drift(ds, ref_mz=800.0, tol_ppm=50.0)
    assert qc["n"] == len(coords)
    assert abs(qc["median_ppm"]) < 4.0                 # centered near zero
    assert 8.0 < qc["max_abs_ppm"] < 16.0              # ≈ the ±12 ppm injected spread


def test_auto_recalibrate_reduces_drift():
    from smile_msi import preprocess
    rng = np.random.default_rng(7)
    coords = [(x + 1, y + 1) for y in range(6) for x in range(8)]
    offs = rng.uniform(-12e-6, 12e-6, len(coords))
    mzs, ints = [], []
    for k in range(len(coords)):
        mzs.append(np.array([800.0 * (1 + offs[k]), 900.0 * (1 + offs[k])]))
        ints.append(np.array([100.0, 80.0]))
    ds = MSIDataset.from_arrays(coords, mzs, ints)
    before = intake.mass_drift(ds, ref_mz=800.0, tol_ppm=80.0)["max_abs_ppm"]
    ds.set_preprocessing([preprocess.auto_recalibrate(ds, n_refs=2, min_rel_intensity=0.0)])
    # after recalibration the per-spectrum shift (median of the two ref offsets) is removed
    after = intake.mass_drift(ds, ref_mz=800.0, tol_ppm=80.0)["max_abs_ppm"]
    assert after <= before                              # alignment doesn't worsen spread
