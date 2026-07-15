"""GUI tests for the UMAP Studio editor + its Cohort-UMAP wiring (headless, offscreen Qt).

Skipped when the desktop deps aren't installed, like the rest of the GUI suite."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtWidgets  # noqa: E402

from smile_msi import umapstudio as us  # noqa: E402
from smile_msi.gui import umapstudiodialog as usd  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def _fake_prefs(monkeypatch):
    """Back the prefs store with a per-test dict so the editor's save/restore doesn't touch
    the real prefs.json."""
    store = {}
    monkeypatch.setattr(usd.prefs, "get", lambda k, d=None: store.get(k, d))
    monkeypatch.setattr(usd.prefs, "set", lambda k, v: store.__setitem__(k, v))
    return store


@pytest.fixture
def data():
    rng = np.random.default_rng(7)
    n = 600
    coords = np.vstack([rng.normal([0, 0], 0.5, (n, 2)), rng.normal([3, 2], 0.5, (n, 2))])
    return us.EmbeddingData(
        coords=coords, method="UMAP",
        categorical={"Sample": np.array(["s1"] * n + ["s2"] * n, dtype=object),
                     "Group": np.array(["A"] * n + ["B"] * n, dtype=object)},
        features=np.column_stack([np.r_[rng.normal(5, 1, n), rng.normal(9, 1, n)],
                                  rng.normal(3, 1, 2 * n)]),
        feature_mz=np.array([744.55, 810.52]))


def test_dialog_builds_and_renders(app, data, _fake_prefs):
    win = QtWidgets.QMainWindow()
    dlg = usd.UMAPStudioDialog(win, data)
    assert dlg.fig is not None
    assert len(dlg.fig._umap_panel_axes) == 1                 # single-panel default
    assert "Sample" in dlg._color_keys


def test_layout_switch_grows_panels(app, data, _fake_prefs):
    dlg = usd.UMAPStudioDialog(QtWidgets.QMainWindow(), data)
    dlg._set_combo(dlg.layout_combo, "four")
    dlg._on_layout_changed()
    dlg._render()                                             # flush the debounced redraw
    assert len(dlg.fig._umap_panel_axes) == 4
    # the seeded multi-panel defaults vary donor vs group (not four identical panels)
    assert {p.color_by for p in dlg.spec.panels} >= {"Sample", "Group"}


def test_continuous_channel_renders_colorbar(app, data, _fake_prefs):
    dlg = usd.UMAPStudioDialog(QtWidgets.QMainWindow(), data)
    name = data.set_feature_channel(0)                        # 744.55 m/z
    dlg._panels[0].color_by = name
    dlg._refresh_panel_combos()
    dlg._render()
    # a continuous panel adds a colourbar axis beyond the single panel axis
    assert len(dlg.fig.axes) > len(dlg.fig._umap_panel_axes)


def test_add_annotation_then_export(app, data, _fake_prefs, monkeypatch, tmp_path):
    dlg = usd.UMAPStudioDialog(QtWidgets.QMainWindow(), data)
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("myelin loss", True)))
    dlg._add_label()
    assert dlg.spec.annotations and dlg.spec.annotations[0].text == "myelin loss"
    assert len(dlg._ann_artists) == 1

    out = tmp_path / "umap.png"
    monkeypatch.setattr(usd.filedialogs, "get_save_file_name",
                        staticmethod(lambda *a, **k: (str(out), "")))
    dlg.fmt_combo.setCurrentText("PNG")
    dlg.dpi.setValue(96)
    dlg._export()
    assert out.exists() and out.stat().st_size > 0
    assert _fake_prefs.get(usd.PREFS_KEY)                     # the look was remembered


def test_new_label_uses_text_size(app, data, _fake_prefs, monkeypatch):
    dlg = usd.UMAPStudioDialog(QtWidgets.QMainWindow(), data)
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("PT", True)))
    dlg.fontsize_spin.setValue(20)
    dlg._add_label()
    assert dlg.spec.annotations[0].fontsize == 20.0          # new label adopts the spinbox size
    assert dlg._selected == ("annotation", 0)                # and is selected so the spinbox
    dlg._on_fontsize_changed(30)                             # can resize it right away
    assert dlg.spec.annotations[0].fontsize == 30.0


def test_text_size_resizes_selected_label(app, data, _fake_prefs, monkeypatch):
    dlg = usd.UMAPStudioDialog(QtWidgets.QMainWindow(), data)
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("PT", True)))
    dlg._add_label()
    dlg._selected = ("annotation", 0)                        # simulate clicking the label
    dlg._on_fontsize_changed(28)
    assert dlg.spec.annotations[0].fontsize == 28.0          # spec persists for export
    assert dlg._ann_artists[0][0].get_fontsize() == 28.0     # on-canvas artist resized live
    # selecting reflects the size back into the spinbox without re-triggering a resize
    dlg.spec.annotations[0].fontsize = 9.0
    dlg._sync_fontsize_spin(9.0)
    assert dlg.fontsize_spin.value() == 9.0


def test_open_with_no_data_guides_user(app, monkeypatch):
    from smile_msi.gui.umapstudiodialog import UMAPStudioMixin

    class Win(UMAPStudioMixin):
        def __init__(self):
            self.msg = None
            self.viewed = None

        def statusBar(self):
            return type("S", (), {"showMessage": lambda _s, m: setattr(self, "msg", m)})()

        def reveal_view(self, label):
            self.viewed = label

    w = Win()
    w.open_umap_studio(None)                                  # no embedding yet
    assert "Cohort UMAP" in (w.msg or "")
    assert w.viewed == "Cohort UMAP"


# --------------------------------------------------------------------------- #
# Cohort UMAP tab wiring on the real MainWindow
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def win(app):
    from smile_msi.gui import main as M
    return M.MainWindow()


def test_cohort_tab_has_studio_controls(win):
    assert hasattr(win, "b_cembed_studio")
    assert hasattr(win, "cembed_metric") and hasattr(win, "cembed_nn")
    assert not win.b_cembed_studio.isEnabled()                # disabled until an embedding runs


def test_atlas_knobs_are_the_defaults(win):
    # the former 'Atlas preset' button was removed; its kidney-atlas values are now the
    # out-of-the-box defaults (cosine · n_neighbors 250 · min_dist 0 · align on)
    assert win.cembed_metric.currentText() == "cosine"
    assert win.cembed_nn.value() == 250
    assert win.cembed_mindist.value() == 0.0
    assert win.cembed_norm_chk.isChecked()
    assert not hasattr(win, "_cembed_atlas_preset")


def test_open_cembed_studio_from_pooled_embedding(win, monkeypatch):
    from smile_msi.multivariate import PooledEmbedding

    rng = np.random.default_rng(11)
    n = 400
    coords = rng.normal(size=(2 * n, 2))
    emb = PooledEmbedding(
        method="UMAP", coords=coords,
        sample_id=np.array([0] * n + [1] * n),
        group=np.array(["A"] * n + ["B"] * n, dtype=object),
        sample_names=["donorA", "donorB"], targets=np.array([744.5, 810.5]),
        counts={"donorA": n, "donorB": n},
        features=rng.normal(size=(2 * n, 2)))
    win._cembed_emb = emb
    captured = {}
    # don't actually open a modal dialog under test — just confirm the data routing
    monkeypatch.setattr(usd.UMAPStudioDialog, "exec", lambda self: captured.update(
        {"channels": list(self.data.categorical), "method": self.data.method}) or 0)
    win._open_cembed_studio()
    assert captured["method"] == "UMAP"
    assert "Sample" in captured["channels"] and "Group" in captured["channels"]
    assert win._umap_studio_data.features is not None         # features retained for colour-by
