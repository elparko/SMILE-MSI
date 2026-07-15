"""GUI smoke test for the spatial multi-omics co-mapping dialog (plan 10, headless).

Builds a tiny real MSIDataset + an in-frame ModalityGrid (same convention as
tests/test_comap.py), instantiates the eagerly-built dialog against a _FakeMain host,
and drives the import → run → top-pairs path. No Qt event loop, no real ST file.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402

from smile_msi import comap  # noqa: E402
from smile_msi.msi import MSIDataset  # noqa: E402
from smile_msi.gui import comapdialog as cd  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _ds():
    W = H = 6
    coords = np.array([[x, y] for y in range(H) for x in range(W)], dtype=int)
    mzs = [np.array([100.0, 200.0]) for _ in range(coords.shape[0])]
    inten = [np.array([1.0 + x, 1.0 + y], dtype=float) for x, y in coords]
    return MSIDataset.from_arrays(coords, mzs, inten)


def _grid_for(ds):
    from smile_msi import spatial
    X = spatial.feature_matrix(ds, [100.0, 200.0], norm="none")
    coords = np.asarray(ds.coordinates, dtype=float)
    vals = np.column_stack([3.0 * X[:, 0] + 7.0, -2.0 * X[:, 1] + 5.0])
    return comap.ModalityGrid(kind="generic", feature_names=["g0", "g1"],
                              coords_um=coords.copy(), values=vals)


class _FakeMain(QtWidgets.QMainWindow):
    def __init__(self, ds=None):
        super().__init__()
        self.ds = ds
        self.peaks = [{"mz": 100.0}, {"mz": 200.0}]
        self.regions = [{"name": "roi-A", "mask": np.array([True] * 18 + [False] * 18)}]
        self.recorded = []

    def _region_pixel_mask(self, rg, _seen=None):
        return rg.get("mask")

    def record_step(self, kind, *, label="", params=None, regions=None, now=None):
        self.recorded.append((kind, label, dict(params or {})))
        return {}


def test_dialog_builds_and_populates(app):
    main = _FakeMain()
    dlg = cd.CoMapDialog(main)
    dlg.load_from_main()
    assert dlg.table.columnCount() == 3
    # region combo: foreground + the one named region
    assert dlg.region.count() == 2
    assert dlg.method_label.text()  # method from profile is shown


def test_run_without_dataset_is_graceful(app):
    main = _FakeMain(ds=None)
    dlg = cd.CoMapDialog(main)
    dlg.load_from_main()
    dlg._run()  # ds is None → status message, no crash, no record
    assert main.recorded == []


def test_import_and_run_populates_top_pairs(app):
    ds = _ds()
    main = _FakeMain(ds=ds)
    dlg = cd.CoMapDialog(main)
    dlg.load_from_main()
    dlg.grid = _grid_for(ds)
    dlg._populate_features(dlg.grid)
    assert dlg.feat_list.count() == 2

    dlg.topk.setValue(4)
    dlg._run()
    assert dlg._result is not None
    assert dlg.table.rowCount() > 0
    # strongest pair should be a strong correlation (TIC-normalized, profile metric)
    top = abs(float(dlg.table.item(0, 2).text()))
    assert top > 0.5
    # a provenance step was recorded with the right kind
    assert main.recorded and main.recorded[0][0] == "cross_modality_correlation"
    assert main.recorded[0][2]["method"] in ("pearson", "spearman")


def test_h5ad_import_missing_backend_is_clear(app, tmp_path, monkeypatch):
    main = _FakeMain(ds=_ds())
    dlg = cd.CoMapDialog(main)
    dlg.load_from_main()
    # simulate a missing anndata backend by pointing at a .h5ad path that load_modality
    # will try to import anndata for; if anndata IS installed, the file is malformed →
    # still a graceful error message (no crash).
    fake = tmp_path / "spatial.h5ad"
    fake.write_bytes(b"not really an h5ad")
    dlg._import(str(fake))
    assert dlg.grid is None
    assert dlg.grid_status.text()  # some error text shown, not a crash
