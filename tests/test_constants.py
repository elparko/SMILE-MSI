"""Guards for the named extraction tolerances (audit plan 18, Issue B).

Two windows are deliberately different: the broad :data:`DEFAULT_TOL_PPM` for
ion-image / feature-matrix extraction, and the tighter :data:`QUANT_TOL_PPM`
for absolute quantification (minimizes co-integration of interferents). These
tests pin the defaults to the constants so the intentional gap can't be silently
equalized, and document the behavioural consequence of the two windows.
"""
import inspect

import numpy as np

from smile_msi import quantify, spatial
from smile_msi.constants import DEFAULT_TOL_PPM, QUANT_TOL_PPM
from smile_msi.msi import MSIDataset


def _default(fn, name="tol_ppm"):
    return inspect.signature(fn).parameters[name].default


def test_quant_window_is_tighter_than_extraction():
    """The whole point of two constants: quant integrates a narrower window."""
    assert QUANT_TOL_PPM < DEFAULT_TOL_PPM


def test_quantify_entrypoints_default_to_quant_window():
    for fn in (quantify.region_response, quantify.fit_calibration,
               quantify.apply_calibration):
        assert _default(fn) == QUANT_TOL_PPM, fn.__name__


def test_extraction_entrypoints_default_to_default_window():
    for fn in (spatial.feature_matrix, spatial.roi_comparison,
               spatial.discriminating_features, MSIDataset.ion_image,
               MSIDataset.ensure_features):
        assert _default(fn) == DEFAULT_TOL_PPM, fn.__name__


def _two_peak_dataset():
    """A single pixel with two peaks ~30 ppm apart around m/z 500.

    30 ppm at 500 Da ≈ 0.015 Da, so a 10 ppm window (0.005 Da) catches only the
    analyte while a 50 ppm window (0.025 Da) sums both peaks into one feature.
    """
    analyte = 500.000
    neighbour = 500.000 * (1 + 30e-6)   # +30 ppm
    axis = np.array([analyte, neighbour], dtype=float)
    coords = np.array([[1, 1]], dtype=int)
    # MemoryStore takes per-pixel (mz, intensity) arrays: one pixel here.
    ds = MSIDataset.from_arrays(coords, [axis], [np.array([10.0, 7.0])])
    return ds, analyte


def test_window_width_changes_integration():
    """Documents the intended difference: 10 ppm = analyte only, 50 ppm = both."""
    ds, analyte = _two_peak_dataset()
    narrow = ds.ion_vector(analyte, tol_ppm=QUANT_TOL_PPM, norm="none")[0]
    broad = ds.ion_vector(analyte, tol_ppm=DEFAULT_TOL_PPM, norm="none")[0]
    assert narrow == 10.0          # only the analyte bin
    assert broad == 17.0           # analyte + neighbour summed
