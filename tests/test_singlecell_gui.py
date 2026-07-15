"""GUI smoke test for the Single-cell profiling dialog (plan 07, headless).

Quick sanity only: the dialog builds, gates Run on an optical + dataset, runs the
engine pipeline end-to-end on a synthetic optical image + tiny MSIDataset (inline,
no host ``_run``), and rolls cell clusters up into regions via ``_new_region``.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402

from smile_msi.msi import MSIDataset  # noqa: E402
from smile_msi.gui import singlecell as scd  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _tiny_ds():
    """A 6x6 flat-spectrum MSIDataset with two built features (mirrors the engine test)."""
    axis = np.array([100.0, 150.0, 200.0, 250.0], dtype=float)
    coords, ints = [], []
    rng = np.random.default_rng(0)
    for r in range(6):
        for c in range(6):
            coords.append((c + 1, r + 1))
            ints.append((1.0 + 0.1 * rng.standard_normal(axis.size)).astype(np.float32))
    ds = MSIDataset.from_arrays(coords, [axis] * len(coords), ints,
                                polarity="negative", spec_mode="profile",
                                source="synthetic")
    ds.build_features([150.0, 200.0], tol_ppm=400.0)
    return ds


def _optical_blobs():
    """A small RGB image with a few bright square blobs on a dark background, sized so
    the default watershed (min 6 px diameter → ~28 px disc area) keeps them."""
    img = np.zeros((36, 36, 3), dtype=np.uint8)
    for (r, c) in [(6, 6), (6, 24), (24, 6), (24, 24)]:
        img[r:r + 7, c:c + 7, :] = 255
    return img


class _FakeMain(QtWidgets.QMainWindow):
    def __init__(self, *, optical=None, ds=None):
        super().__init__()
        self.ds = ds
        self._optical = optical
        self._optical_align = {"tx": 0.0, "ty": 0.0, "scale": 1.0, "angle": 0.0,
                               "flipx": False, "flipy": False}
        self.regions = []
        self.recorded = []

    # region creation (minimal _new_region)
    def _new_region(self, name=None, color=None, mask=None, **kw):
        self.regions.append({"name": name, "mask": mask})

    def record_step(self, kind, *, label="", params=None, regions=None):
        self.recorded.append((kind, label, dict(params or {})))
        return {}


def test_dialog_builds_and_gates_run(app):
    main = _FakeMain()                              # no ds, no optical
    dlg = scd.SingleCellDialog(main)
    dlg.load_from_main()
    assert not dlg.b_run.isEnabled()               # nothing loaded → disabled
    assert "dataset" in dlg.report.toPlainText().lower()

    main.ds = _tiny_ds()
    main._optical = _optical_blobs()
    dlg.load_from_main()
    assert dlg.b_run.isEnabled()                    # both present → enabled


def test_align_matrix_is_3x3_invertible():
    M = scd._align_matrix({"tx": 0.0, "ty": 0.0, "scale": 1.0, "angle": 0.0,
                           "flipx": False, "flipy": False},
                          hp=24, wp=24, h0=6, w0=6)
    assert M.shape == (3, 3)
    np.linalg.inv(M)                                # must be invertible


def test_run_pipeline_and_make_regions(app):
    main = _FakeMain(optical=_optical_blobs(), ds=_tiny_ds())
    dlg = scd.SingleCellDialog(main)
    dlg.load_from_main()
    assert dlg.b_run.isEnabled()
    dlg.min_diam.setValue(4.0)                      # small blobs survive the min-area filter

    dlg._run()                                      # inline (no host _run)
    assert dlg._cells is not None
    assert dlg._cellmat is not None
    assert dlg._coords is not None and dlg._coords.shape[1] == 2
    assert dlg._labels is not None
    txt = dlg.report.toPlainText().lower()
    assert "cell" in txt
    assert main.recorded and main.recorded[0][0] == "single_cell"

    # roll clusters up into regions
    if dlg.b_regions.isEnabled():
        dlg._make_regions()
        assert len(main.regions) >= 1
        for r in main.regions:
            assert r["mask"].dtype == bool
            assert r["mask"].shape[0] == main.ds.n_pixels
