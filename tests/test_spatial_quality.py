"""Ion-image quality metrics: hotspot fraction + measure of spatial chaos.

Both run end-to-end through the real feature-matrix extraction on a synthetic slide whose
channels have known spatial character, so the metrics are exercised exactly as annotation
would call them:

* ``uniform`` — constant everywhere (organized, but no autocorrelation gradient)
* ``struct``  — a solid left-half blob (spatially structured, real-ion-like)
* ``speckle`` — dense continuous random noise (realistic salt-and-pepper)
* ``checker`` — a checkerboard (every pixel isolated → *maximal* spatial chaos)
* ``hot``     — all signal in a single pixel (crystallization/delocalization artifact)

The point of having several metrics is that they catch different failure modes: Moran's I and
:func:`spatial_chaos` agree that speckle/checker are bad, but only ``spatial_chaos`` calls
``uniform`` organized, and only :func:`hotspot_fraction` flags the single-pixel ``hot`` artifact.
"""
import numpy as np

from smile_msi.msi import MSIDataset
from smile_msi.spatial import (
    hotspot_fraction, spatial_chaos, spatial_autocorrelation, CHAOS_MIN_PIXELS,
)

MZ = {"checker": 500.5, "uniform": 600.5, "struct": 700.5,
      "speckle": 800.5, "hot": 900.5, "absent": 1000.5}
_ORDER = ["checker", "uniform", "struct", "speckle", "hot"]
AXIS = np.array([MZ[k] for k in _ORDER])
PEAKS = [MZ[k] for k in _ORDER]
TOL = 50.0   # channels are ~100 Da apart → no cross-talk; wide window just guarantees the hit


def _quality_slide(W=10, H=10, seed=0):
    """A W×H slide carrying the named channels with known spatial character."""
    rng = np.random.default_rng(seed)
    hot_xy = (W // 2, H // 2)
    coords, spectra = [], []
    for y in range(H):
        for x in range(W):
            coords.append([x, y, 0])
            vals = {
                "checker": float((x + y) % 2),               # isolated pixels → max chaos
                "uniform": 1.0,                               # organized, flat
                "struct": 1.0 if x < W // 2 else 0.0,        # solid left half → one blob
                "speckle": float(rng.random()),              # dense random noise
                "hot": 100.0 if (x, y) == hot_xy else 0.0,   # all signal in one pixel
            }
            spectra.append(np.array([vals[k] for k in _ORDER]))
    ds = MSIDataset.from_arrays(np.array(coords), [AXIS.copy() for _ in coords], spectra)
    ds.prime()
    return ds


def _by_mz(peaks, values):
    return {round(float(m), 4): float(v) for m, v in zip(peaks, values)}


def test_hotspot_fraction_flags_single_pixel_artifact():
    """A single-pixel hotspot concentrates ~all signal in the top pixel (→1); spread ions
    keep their signal diffuse (→ small)."""
    ds = _quality_slide()
    h = _by_mz(PEAKS, hotspot_fraction(ds, PEAKS, tol_ppm=TOL))    # top_frac default 1%

    assert h[MZ["hot"]] > 0.9                                       # essentially all in one pixel
    for name in ("uniform", "struct", "speckle", "checker"):
        assert h[MZ[name]] < 0.1, name                             # diffuse ions barely concentrate
        assert h[MZ["hot"]] > 5 * h[MZ[name]], name                # the hotspot dwarfs them


def test_hotspot_fraction_bounds_and_absent_ion():
    """Fractions stay in [0, 1]; an ion with no signal reads exactly 0, not nan."""
    ds = _quality_slide()
    h = hotspot_fraction(ds, PEAKS + [MZ["absent"]], tol_ppm=TOL)
    assert np.all((h >= 0.0) & (h <= 1.0))
    assert h[-1] == 0.0                                             # absent ion → 0


def test_spatial_chaos_separates_structure_from_speckle():
    """Structured and uniform images score high (organized); a checkerboard scores ~0 (maximal
    chaos); dense random speckle sits below the structured ceiling. norm='none' isolates each
    channel's morphology from the others."""
    ds = _quality_slide()
    c = _by_mz(PEAKS, spatial_chaos(ds, PEAKS, tol_ppm=TOL, norm="none"))

    assert c[MZ["struct"]] > 0.8                                    # one contiguous blob
    assert c[MZ["uniform"]] > 0.8                                   # organized, not chaotic
    assert c[MZ["checker"]] < 0.1                                   # every pixel isolated → ρ≈0
    # realistic speckle is clearly worse than structure, and clearly better than a checkerboard
    assert c[MZ["struct"]] > c[MZ["speckle"]] > c[MZ["checker"]]
    assert c[MZ["uniform"]] > c[MZ["speckle"]]


def test_spatial_chaos_undefined_for_too_sparse():
    """The single-pixel hotspot (< CHAOS_MIN_PIXELS non-zero pixels) has no morphology → nan;
    an absent ion is likewise nan."""
    assert CHAOS_MIN_PIXELS >= 2
    ds = _quality_slide()
    peaks = PEAKS + [MZ["absent"]]
    c = _by_mz(peaks, spatial_chaos(ds, peaks, tol_ppm=TOL, norm="none"))
    assert np.isnan(c[MZ["hot"]])                                   # 1 non-zero pixel
    assert np.isnan(c[MZ["absent"]])                               # 0 non-zero pixels


def test_chaos_and_morans_are_complementary_axes():
    """The reason to keep both: a *uniform* ion is organized (high chaos ρ) yet has no
    autocorrelation gradient (Moran's ≈ 0), while dense speckle is bad on both."""
    ds = _quality_slide()
    c = _by_mz(PEAKS, spatial_chaos(ds, PEAKS, tol_ppm=TOL, norm="none"))
    sa = spatial_autocorrelation(ds, PEAKS, tol_ppm=TOL, norm="none")
    m = _by_mz(sa["mz"].tolist(), sa["morans_i"].tolist())

    # structured left-half blob: high on both
    assert m[MZ["struct"]] > 0.4 and c[MZ["struct"]] > 0.8
    # uniform ion: Moran's collapses to ~0 (flat → no gradient) but chaos ρ stays high
    assert abs(m[MZ["uniform"]]) < 0.2 and c[MZ["uniform"]] > 0.8
    # dense speckle: weak on both
    assert m[MZ["speckle"]] < 0.2 and c[MZ["speckle"]] < c[MZ["struct"]]
