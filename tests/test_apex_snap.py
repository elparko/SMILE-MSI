"""Ion-image extraction must centre on the true spectral apex, not the catalogued
feature m/z.

A picked-peak / consensus feature m/z is a centroid (``msi._centroid``) or a pooled
cluster centre that can sit several ppm off the real apex — and cross-slide calibration
adds more. With the GUI's tight extraction window (10 ppm) that offset slides the boxcar
off the peak, so a correctly *labelled* feature renders a *blank* ion image (the reported
"PC labelled, no ions"). ``_apex_bin`` / ``_apex_mz`` snap the extraction centre back to
the apex while the catalogued m/z stays the identity. See gui/ion.py:_apex_mz.
"""
import numpy as np
import pytest

from smile_msi.msi import MSIDataset, _centroid

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from smile_msi.gui.ion import _apex_bin  # noqa: E402


def _peak_slide(apex_mz=850.5729, npix=6):
    """A small profile slide with one sharp, mildly asymmetric peak at ``apex_mz``."""
    axis = np.round(np.arange(apex_mz - 0.17, apex_mz + 0.23, 0.0010), 4)
    apex_i = int(np.argmin(np.abs(axis - apex_mz)))
    prof = np.zeros_like(axis)
    for off, val in [(-2, 30), (-1, 70), (0, 100), (1, 55), (2, 18)]:
        prof[apex_i + off] = val
    coords = np.array([[x, 0, 0] for x in range(npix)])
    ds = MSIDataset.from_arrays(coords, [axis] * npix, [prof.copy() for _ in range(npix)])
    ds.prime()
    return ds, axis, apex_i, prof


def test_apex_bin_recovers_apex_from_offapex_center():
    """A centre a few ppm off the apex snaps back to the apex bin (the halo + the ion
    image share this)."""
    _, axis, apex_i, prof = _peak_slide()
    apex_mz = float(axis[apex_i])
    for off_ppm in (5.0, 10.0, 20.0, -8.0):          # centroid / consensus / calibration span
        c = apex_mz * (1.0 - off_ppm / 1e6)
        assert _apex_bin(axis, prof, c) == apex_i, off_ppm


def test_offapex_center_blanks_image_and_apex_snap_fixes_it():
    """The reported failure and its fix, end-to-end through the real extraction:
    extracting at an off-apex centre blanks the image; extracting at the snapped apex
    recovers full signal — while the catalogued m/z (identity) is never mutated."""
    ds, axis, apex_i, prof = _peak_slide()
    apex_mz = float(axis[apex_i])

    catalogued = apex_mz * (1.0 - 15.0 / 1e6)         # 15 ppm below apex (>1 half-window)
    blank = ds.ion_image(catalogued, tol_ppm=10.0)
    assert np.nansum(blank) == 0.0                    # boxcar fell off the peak → dark image

    snapped = float(axis[_apex_bin(axis, prof, catalogued)])
    assert snapped == apex_mz                         # extraction centre != catalogued m/z
    lit = ds.ion_image(snapped, tol_ppm=10.0)
    assert np.nansum(lit) > 0.0 and np.nanmax(lit) > 0.0

    # identity is untouched: we only moved the *extraction* centre, not the stored m/z
    assert catalogued != snapped


def test_apex_bin_degenerate_inputs():
    """No spectrum → None (caller falls back to the m/z unchanged); a real apex maps to
    itself (snapping is idempotent, so already-on-apex selections don't move)."""
    _, axis, apex_i, prof = _peak_slide()
    assert _apex_bin(None, None, 850.5) is None
    assert _apex_bin(axis, prof, None) is None
    assert _apex_bin(np.array([]), np.array([]), 850.5) is None
    assert _apex_bin(axis, prof, float(axis[apex_i])) == apex_i


def test_profile_centroid_is_the_off_apex_source():
    """Documents the mechanism: the stored feature m/z is the 3-point weighted centroid,
    which on an asymmetric peak sits off the apex — the same value the ion image would
    otherwise extract at."""
    _, axis, apex_i, prof = _peak_slide()
    centroid = _centroid(axis, prof, apex_i)
    assert centroid != float(axis[apex_i])            # centroid biased off the apex bin
    assert _apex_bin(axis, prof, centroid) == apex_i  # snap undoes the bias
