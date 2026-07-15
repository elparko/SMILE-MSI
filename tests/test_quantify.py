"""Unit tests for the absolute-quantification calibration engine (smile_msi.quantify).

Headless, synthetic data only. A small in-RAM MSIDataset is hand-built so that each
calibration "region" (a block of pixels) carries a known analyte intensity proportional
to a known concentration, plus an optional internal-standard peak. With ``norm="none"``
the extracted feature value equals the integrated intensity we placed, so region-mean
responses are exactly predictable and the fit must recover the planted slope/intercept."""

import numpy as np
import pytest

from smile_msi.msi import MSIDataset
from smile_msi import quantify
from smile_msi.quantify import CalLevel, CalibrationModel


# --------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------- #
# Shared m/z axis with two well-separated peaks: analyte and internal standard.
ANALYTE_MZ = 500.0
IS_MZ = 400.0
_AXIS = np.array([300.0, 400.0, 500.0, 600.0], dtype=float)
_I_ANALYTE = 2          # axis index of the analyte bin
_I_IS = 1               # axis index of the IS bin


def _build_dataset(blocks):
    """Build a MSIDataset from a list of pixel blocks.

    Each block is a dict {analyte: float, is_: float, n: int}. Every pixel in the block
    gets the same spectrum (analyte amplitude at ANALYTE_MZ, IS amplitude at IS_MZ).
    Returns (ds, masks) where masks[k] is the bool[n_pixels] mask of block k."""
    coords = []
    ints = []
    mzs = []
    masks_idx = []
    p = 0
    x = 1
    for blk in blocks:
        idx = []
        for _ in range(blk["n"]):
            coords.append((x, 1))
            spec = np.zeros_like(_AXIS)
            spec[_I_ANALYTE] = blk["analyte"]
            spec[_I_IS] = blk.get("is_", 0.0)
            ints.append(spec.astype(np.float32))
            mzs.append(_AXIS.copy())
            idx.append(p)
            p += 1
            x += 1
        masks_idx.append(idx)
    ds = MSIDataset.from_arrays(coords, mzs, ints, polarity="negative",
                               spec_mode="profile", source="synthetic")
    n = ds.n_pixels
    masks = []
    for idx in masks_idx:
        m = np.zeros(n, dtype=bool)
        m[idx] = True
        masks.append(m)
    return ds, masks


def _levels(concs, slope, intercept, masks, *, is_amp=None):
    """Build CalLevels where each region's analyte intensity = slope*conc + intercept.
    If ``is_amp`` is given every region shares that IS amplitude."""
    levels = []
    blocks_meta = []
    for c in concs:
        blocks_meta.append({"analyte": slope * c + intercept,
                            "is_": (is_amp or 0.0), "n": 4})
    return blocks_meta


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_recovers_slope_and_intercept():
    concs = [0.0, 1.0, 2.0, 4.0, 8.0]
    slope, intercept = 3.0, 5.0
    blocks = [{"analyte": slope * c + intercept, "is_": 0.0, "n": 4} for c in concs]
    ds, masks = _build_dataset(blocks)
    levels = [CalLevel(f"std{i}", c, masks[i]) for i, c in enumerate(concs)]

    model = quantify.fit_calibration(ds, levels, ANALYTE_MZ, norm="none")
    assert model.slope == pytest.approx(slope, rel=1e-6)
    assert model.intercept == pytest.approx(intercept, rel=1e-6)
    assert model.r2 == pytest.approx(1.0, abs=1e-9)
    assert model.n_levels == 5
    assert model.conc_min == 0.0 and model.conc_max == 8.0
    assert model.response == "intensity"


def test_through_origin():
    concs = [1.0, 2.0, 3.0, 5.0]
    slope = 4.0
    blocks = [{"analyte": slope * c, "is_": 0.0, "n": 3} for c in concs]
    ds, masks = _build_dataset(blocks)
    levels = [CalLevel(f"std{i}", c, masks[i]) for i, c in enumerate(concs)]

    model = quantify.fit_calibration(ds, levels, ANALYTE_MZ, norm="none",
                                     through_origin=True)
    assert model.intercept == 0.0
    assert model.through_origin is True
    assert model.slope == pytest.approx(slope, rel=1e-6)
    assert model.r2 == pytest.approx(1.0, abs=1e-9)


def test_is_ratio_matrix_cancellation():
    """Doubling analyte and IS together leaves the analyte/IS ratio (and thus the
    back-calculated concentration) unchanged — the matrix-effect cancellation."""
    concs = [1.0, 2.0, 3.0, 4.0]
    slope = 2.0    # ratio = slope * conc
    is_amp = 10.0
    blocks = [{"analyte": slope * c * is_amp, "is_": is_amp, "n": 4} for c in concs]
    ds, masks = _build_dataset(blocks)
    levels = [CalLevel(f"std{i}", c, masks[i]) for i, c in enumerate(concs)]
    model = quantify.fit_calibration(ds, levels, ANALYTE_MZ, is_mz=IS_MZ, norm="none",
                                     through_origin=True)
    assert model.response == "is_ratio"
    assert model.slope == pytest.approx(slope, rel=1e-6)

    # Now scale BOTH analyte and IS by 2x in every region: ratio is identical, so the
    # fitted slope (and any back-calculated concentration) must be unchanged.
    blocks2 = [{"analyte": slope * c * is_amp * 2.0, "is_": is_amp * 2.0, "n": 4}
               for c in concs]
    ds2, masks2 = _build_dataset(blocks2)
    levels2 = [CalLevel(f"std{i}", c, masks2[i]) for i, c in enumerate(concs)]
    model2 = quantify.fit_calibration(ds2, levels2, ANALYTE_MZ, is_mz=IS_MZ, norm="none",
                                      through_origin=True)
    assert model2.slope == pytest.approx(model.slope, rel=1e-9)


def test_apply_calibration_roundtrip_and_out_of_range():
    concs = [0.0, 1.0, 2.0, 4.0]
    slope, intercept = 5.0, 2.0
    blocks = [{"analyte": slope * c + intercept, "is_": 0.0, "n": 4} for c in concs]
    # An extra block ABOVE the calibrated range (conc 10 -> intensity 52).
    over_conc = 10.0
    blocks.append({"analyte": slope * over_conc + intercept, "is_": 0.0, "n": 4})
    ds, masks = _build_dataset(blocks)
    levels = [CalLevel(f"std{i}", c, masks[i]) for i, c in enumerate(concs)]

    model = quantify.fit_calibration(ds, levels, ANALYTE_MZ, norm="none")
    conc, in_range = quantify.apply_calibration(ds, model, norm="none")

    assert conc.shape == (ds.n_pixels,)
    # Standard-region pixels back-calculate to their nominal concentration.
    for i, c in enumerate(concs):
        vals = conc[masks[i]]
        assert np.allclose(vals, c, atol=1e-6)
        assert in_range[masks[i]].all()
    # The over-range block: value ~10 but flagged out of range (NOT clamped to conc_max).
    over_mask = masks[len(concs)]
    assert np.allclose(conc[over_mask], over_conc, atol=1e-6)
    assert not in_range[over_mask].any()
    # Raw value preserved, not clamped to conc_max.
    assert conc[over_mask].max() > model.conc_max


def test_lod_loq_ordering_and_ratio():
    # Build a series with a tiny, controlled residual so sigma > 0 and lod/loq exist.
    concs = [1.0, 2.0, 3.0, 4.0, 5.0]
    slope, intercept = 4.0, 1.0
    # Add small alternating residuals to the response.
    resid = [0.1, -0.1, 0.1, -0.1, 0.1]
    blocks = [{"analyte": slope * c + intercept + r, "is_": 0.0, "n": 4}
              for c, r in zip(concs, resid)]
    ds, masks = _build_dataset(blocks)
    levels = [CalLevel(f"std{i}", c, masks[i]) for i, c in enumerate(concs)]

    model = quantify.fit_calibration(ds, levels, ANALYTE_MZ, norm="none")
    assert model.residual_se > 0
    assert model.lod is not None and model.loq is not None
    assert 0 < model.lod < model.loq
    # IUPAC/ICH ratio LOQ/LOD = 10/3.3.
    assert model.loq / model.lod == pytest.approx(10.0 / 3.3, rel=1e-9)


def test_weighting_favors_low_concentration():
    """On a heteroscedastic series (noise grows with concentration) the 1/x weighting
    should pull the fit toward the low-concentration points relative to OLS."""
    concs = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
    true_slope, true_intercept = 2.0, 0.5
    # Distort only the high-concentration points (proportional error) so OLS, which
    # treats absolute residuals equally, is pulled off the low end.
    resp = true_slope * concs + true_intercept
    resp[-1] += 8.0       # big absolute error at the highest level
    resp[-2] += 3.0
    blocks = [{"analyte": float(r), "is_": 0.0, "n": 4} for r in resp]
    ds, masks = _build_dataset(blocks)
    levels = [CalLevel(f"std{i}", float(c), masks[i]) for i, c in enumerate(concs)]

    m_ols = quantify.fit_calibration(ds, levels, ANALYTE_MZ, norm="none", weighting="none")
    m_w = quantify.fit_calibration(ds, levels, ANALYTE_MZ, norm="none", weighting="1/x")

    # The weighted intercept should be closer to the true low-end intercept than OLS,
    # because 1/x weighting trusts the low-concentration points more.
    assert abs(m_w.intercept - true_intercept) < abs(m_ols.intercept - true_intercept)
    assert m_w.weighting == "1/x"


def test_errors_on_too_few_or_degenerate_levels():
    blocks = [{"analyte": 10.0, "is_": 0.0, "n": 4},
              {"analyte": 20.0, "is_": 0.0, "n": 4}]
    ds, masks = _build_dataset(blocks)

    # Fewer than 2 levels.
    with pytest.raises(ValueError):
        quantify.fit_calibration(ds, [CalLevel("a", 1.0, masks[0])], ANALYTE_MZ,
                                 norm="none")

    # Two levels but identical concentration => zero-variance design.
    deg = [CalLevel("a", 3.0, masks[0]), CalLevel("b", 3.0, masks[1])]
    with pytest.raises(ValueError):
        quantify.fit_calibration(ds, deg, ANALYTE_MZ, norm="none")


def test_calibration_report_keys_and_roundtrip():
    concs = [0.0, 1.0, 2.0, 3.0]
    slope, intercept = 2.0, 1.0
    blocks = [{"analyte": slope * c + intercept, "is_": 0.0, "n": 4} for c in concs]
    ds, masks = _build_dataset(blocks)
    levels = [CalLevel(f"std{i}", c, masks[i]) for i, c in enumerate(concs)]
    model = quantify.fit_calibration(ds, levels, ANALYTE_MZ, norm="none", units="pmol/mm2")

    rep = quantify.calibration_report(model)
    expected = {
        "analyte_mz", "is_mz", "slope", "intercept", "r2", "residual_se",
        "lod", "loq", "units", "n_levels", "conc_min", "conc_max",
        "weighting", "through_origin", "response",
    }
    assert set(rep.keys()) == expected
    assert rep["units"] == "pmol/mm2"
    # All values are plain JSON-friendly scalars (no numpy types).
    for v in rep.values():
        assert v is None or isinstance(v, (int, float, str, bool))

    # Model to_dict / from_dict round-trip.
    d = model.to_dict()
    m2 = CalibrationModel.from_dict(d)
    assert m2.slope == pytest.approx(model.slope)
    assert m2.intercept == pytest.approx(model.intercept)
    assert m2.units == "pmol/mm2"
    assert m2.is_mz is None
    assert m2.response == "intensity"
    # to_dict must be JSON-serializable.
    import json
    json.loads(json.dumps(d))


def test_region_response_is_ratio_undefined_raises():
    blocks = [{"analyte": 10.0, "is_": 0.0, "n": 4}]
    ds, masks = _build_dataset(blocks)
    with pytest.raises(ValueError):
        quantify.region_response(ds, masks[0], ANALYTE_MZ, is_mz=IS_MZ, norm="none")
