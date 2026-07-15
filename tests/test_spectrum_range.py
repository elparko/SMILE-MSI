"""Tests for smile_msi.spectrum_range.signal_mz_range — the pure-numpy helper that
crops a spectrum's m/z axis to the window actually holding signal.

Pure/headless: numpy only, small in-memory fixtures, no GUI/network/files.

Behaviour verified against the source:
  * empty axis              -> None
  * no spec matching axis    -> None  (mag stays None)
  * all-zero / no peak       -> None  (peak not > 0)
  * all-NaN spectrum         -> None  (nanmax is NaN, ``not NaN > 0`` is True)
  * single qualifying bin    -> None  (lo == hi, ``not hi > lo``)
  * multi-peak               -> (lo, hi) padded by pad_frac of the span
  * signed / difference       -> magnitudes are absolute (|spec|), so it spans
                                where *either* side has signal
"""
import warnings

import numpy as np
import pytest

from smile_msi.spectrum_range import signal_mz_range


def _peaky_axis():
    """A 4000-bin axis with three Gaussian peaks confined to ~382..1040 and a long
    empty acquired tail out to 2300."""
    axis = np.linspace(200.0, 2300.0, 4000)
    spec = np.zeros_like(axis)
    for mz, h in [(382.0, 5e4), (560.0, 8e3), (1040.0, 6e3)]:
        spec += h * np.exp(-0.5 * ((axis - mz) / 0.4) ** 2)
    return axis, spec


# --- degenerate / None-returning inputs -----------------------------------

def test_empty_axis_returns_none():
    assert signal_mz_range(np.array([]), np.array([])) is None
    # an empty axis short-circuits before the specs are even inspected
    assert signal_mz_range(np.array([]), np.array([1.0, 2.0, 3.0])) is None


def test_no_spec_given_returns_none():
    # no intensity arrays at all -> mag stays None
    axis = np.linspace(100.0, 200.0, 50)
    assert signal_mz_range(axis) is None


def test_spec_length_mismatch_is_skipped():
    # a spec whose length != axis.size is skipped; with no other spec -> None
    axis = np.linspace(100.0, 200.0, 50)
    assert signal_mz_range(axis, np.ones(10)) is None
    # a matching spec alongside a mismatched one still works (mismatch ignored)
    good = np.zeros_like(axis)
    good[10] = 5.0
    good[40] = 5.0
    assert signal_mz_range(axis, np.ones(10), good) is not None


def test_all_zero_spectrum_returns_none():
    axis = np.linspace(100.0, 200.0, 50)
    assert signal_mz_range(axis, np.zeros_like(axis)) is None


def test_all_nan_spectrum_returns_none():
    # nanmax over all-NaN is NaN; ``not (NaN > 0)`` is True -> None.
    # nanmax warns on all-NaN input; that's expected, not a failure.
    axis = np.linspace(100.0, 200.0, 50)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert signal_mz_range(axis, np.full_like(axis, np.nan)) is None


def test_nan_bins_are_ignored_alongside_real_signal():
    # NaNs scattered in an otherwise real spectrum don't poison the result:
    # nanmax skips them and the qualifying bins still define the window.
    axis = np.linspace(100.0, 200.0, 11)   # spacing of 10
    spec = np.zeros_like(axis)
    spec[3] = 10.0
    spec[7] = 8.0
    spec[0] = np.nan
    spec[10] = np.nan
    rng = signal_mz_range(axis, spec)
    assert rng is not None
    lo, hi = rng
    # spans bins 3..7 (axis[3]=130, axis[7]=170), padded outward
    assert lo < axis[3]
    assert hi > axis[7]


def test_single_qualifying_bin_returns_none():
    # exactly one bin clears the threshold -> lo == hi -> ``not hi > lo`` -> None.
    axis = np.linspace(100.0, 200.0, 11)
    spec = np.zeros_like(axis)
    spec[5] = 1.0
    assert signal_mz_range(axis, spec) is None


# --- the main multi-peak crop ---------------------------------------------

def test_multi_peak_crops_to_signal_with_padding():
    axis, spec = _peaky_axis()
    rng = signal_mz_range(axis, spec)
    assert rng is not None
    lo, hi = rng
    # opens just left of the first peak, crops the empty tail past the last peak
    assert 330.0 < lo < 382.0
    assert 1040.0 < hi < 1200.0
    # the returned window must sit inside the padded acquired span
    assert lo > axis[0] - 1.0
    assert hi < axis[-1]


def test_padding_matches_pad_frac_of_span():
    # Build a clean two-bin window so the un-padded span is exact, then check the
    # padding is pad_frac * span on each side.
    axis = np.linspace(0.0, 100.0, 101)   # integer m/z, spacing 1
    spec = np.zeros_like(axis)
    spec[20] = 1.0
    spec[60] = 1.0
    pad_frac = 0.1
    lo, hi = signal_mz_range(axis, spec, frac=0.5, pad_frac=pad_frac)
    raw_lo, raw_hi = 20.0, 60.0
    span = raw_hi - raw_lo            # 40
    pad = span * pad_frac            # 4
    assert lo == pytest.approx(raw_lo - pad)
    assert hi == pytest.approx(raw_hi + pad)


def test_pad_frac_zero_gives_unpadded_edges():
    axis = np.linspace(0.0, 100.0, 101)
    spec = np.zeros_like(axis)
    spec[20] = 1.0
    spec[60] = 1.0
    lo, hi = signal_mz_range(axis, spec, frac=0.5, pad_frac=0.0)
    assert lo == pytest.approx(20.0)
    assert hi == pytest.approx(60.0)


def test_larger_pad_frac_widens_window():
    axis, spec = _peaky_axis()
    lo_s, hi_s = signal_mz_range(axis, spec, pad_frac=0.0)
    lo_w, hi_w = signal_mz_range(axis, spec, pad_frac=0.1)
    # more padding pushes lo down and hi up symmetrically around the same raw span
    assert lo_w < lo_s
    assert hi_w > hi_s
    # the raw (un-padded) midpoint is unchanged by padding
    assert (lo_s + hi_s) / 2 == pytest.approx((lo_w + hi_w) / 2, rel=1e-9)


# --- frac (fraction-of-max threshold) sensitivity --------------------------

def test_larger_frac_narrows_window():
    # A tall central peak plus tiny shoulder peaks. A higher frac threshold drops
    # the shoulders, narrowing the crop.
    axis = np.linspace(0.0, 100.0, 1001)
    spec = np.zeros_like(axis)
    spec += 1000.0 * np.exp(-0.5 * ((axis - 50.0) / 0.2) ** 2)   # tall centre
    spec += 1.0 * np.exp(-0.5 * ((axis - 10.0) / 0.2) ** 2)      # faint left shoulder
    spec += 1.0 * np.exp(-0.5 * ((axis - 90.0) / 0.2) ** 2)      # faint right shoulder
    lo_lo, hi_lo = signal_mz_range(axis, spec, frac=1e-4, pad_frac=0.0)  # keeps shoulders
    lo_hi, hi_hi = signal_mz_range(axis, spec, frac=0.5, pad_frac=0.0)   # only the centre
    assert lo_lo < lo_hi
    assert hi_lo > hi_hi
    # the strict threshold collapses onto the central peak
    assert 45.0 < lo_hi < 50.0
    assert 50.0 < hi_hi < 55.0


def test_frac_threshold_is_inclusive():
    # the comparison is ``mag >= peak * frac`` — a bin exactly at the threshold is kept.
    axis = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    spec = np.array([0.0, 5.0, 0.0, 10.0, 0.0])   # peak 10; bin 1 is 5 == 10*0.5
    lo, hi = signal_mz_range(axis, spec, frac=0.5, pad_frac=0.0)
    assert lo == pytest.approx(1.0)   # the 5.0 bin (== threshold) is included
    assert hi == pytest.approx(3.0)


# --- signed / difference spectra ------------------------------------------

def test_negative_only_spectrum_uses_absolute_magnitude():
    # a purely negative spectrum still crops to its (absolute) signal.
    axis = np.linspace(0.0, 100.0, 101)
    spec = np.zeros_like(axis)
    spec[20] = -8.0
    spec[60] = -4.0
    rng = signal_mz_range(axis, spec, pad_frac=0.0)
    assert rng is not None
    lo, hi = rng
    assert lo == pytest.approx(20.0)
    assert hi == pytest.approx(60.0)


def test_difference_spectrum_spans_either_side():
    # signed difference / multi-spectrum: window spans where *either* side has signal.
    axis, spec = _peaky_axis()
    lo, hi = signal_mz_range(axis, spec)
    lo2, hi2 = signal_mz_range(axis, spec, -np.flip(spec))
    # the flipped/negated copy adds signal on the mirror side, widening the window
    assert lo2 < lo
    assert hi2 > hi


def test_multiple_specs_take_elementwise_max_magnitude():
    # two non-overlapping one-sided specs combine to span both of their peaks.
    axis = np.linspace(0.0, 100.0, 101)
    a = np.zeros_like(axis)
    a[10] = 5.0
    b = np.zeros_like(axis)
    b[80] = -5.0
    lo, hi = signal_mz_range(axis, a, b, frac=0.5, pad_frac=0.0)
    assert lo == pytest.approx(10.0)
    assert hi == pytest.approx(80.0)


# --- type / shape robustness ----------------------------------------------

def test_accepts_python_lists():
    # axis and specs are coerced via np.asarray(dtype=float); lists are fine.
    axis = list(range(11))
    spec = [0.0] * 11
    spec[3] = 1.0
    spec[7] = 1.0
    lo, hi = signal_mz_range(axis, spec, frac=0.5, pad_frac=0.0)
    assert lo == pytest.approx(3.0)
    assert hi == pytest.approx(7.0)
