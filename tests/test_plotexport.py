"""Tests for the publication volcano render (export.render_volcano_figure) and its export
dialog (gui.plotexport.VolcanoExportDialog)."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from smile_msi import export


def _demo_arrays(n=120, seed=4):
    rng = np.random.default_rng(seed)
    fc = rng.normal(0, 1.1, n)
    p = 10 ** (-np.abs(rng.normal(0, 1.0, n)))
    q = np.clip(p * 3, 0, 1)
    strong = rng.choice(n, 8, replace=False)
    fc[strong] += np.sign(fc[strong]) * 2.2
    p[strong] = 10 ** (-rng.uniform(4, 8, len(strong)))
    q[strong] = p[strong] * 1.5
    mz = rng.uniform(700, 900, n)
    labels = [f"PC {30 + i % 12}:1" for i in range(n)]
    return fc, p, q, mz, labels


def _title_size(fig):
    for t in fig.texts:
        if t.get_text():
            return t.get_fontsize()
    return None


# --------------------------------------------------------------------------- #
# Pure render
# --------------------------------------------------------------------------- #
def test_volcano_renders_and_has_axes():
    fc, p, q, mz, labels = _demo_arrays()
    fig = export.render_volcano_figure(fc, p, q, mz=mz, labels=labels,
                                       a_label="A", b_label="B", dpi=100)
    assert fig.axes, "no axes drawn"
    ax = fig.axes[0]
    assert "fold-change" in ax.get_xlabel()
    # points were scattered
    assert any(len(c.get_offsets()) for c in ax.collections)


def test_volcano_handles_all_nan_and_empty():
    # empty
    fig = export.render_volcano_figure([], [], [], dpi=80)
    assert fig.axes
    # all-nan (nothing tested) must not raise
    nan = np.full(10, np.nan)
    fig2 = export.render_volcano_figure(nan, nan, nan, dpi=80)
    assert fig2.axes


def test_volcano_nothing_significant_still_draws_points():
    # q all > 0.05 → no significant, but points still drawn (not an all-grey broken plot)
    fc = np.linspace(-2, 2, 30)
    p = np.full(30, 0.2)
    q = np.full(30, 0.9)
    fig = export.render_volcano_figure(fc, p, q, dpi=80)
    ax = fig.axes[0]
    pts = sum(len(c.get_offsets()) for c in ax.collections)
    assert pts == 30


def test_volcano_annotate_by_label_and_mz():
    fc, p, q, mz, labels = _demo_arrays()
    fig_mz = export.render_volcano_figure(fc, p, q, mz=mz, annotate_top=6, annotate_by="mz", dpi=80)
    fig_lab = export.render_volcano_figure(fc, p, q, mz=mz, labels=labels, annotate_top=6,
                                           annotate_by="label", dpi=80)
    # both produced some annotation texts on the axes
    assert any(t.get_text() for t in fig_mz.axes[0].texts)
    lab_texts = [t.get_text() for t in fig_lab.axes[0].texts]
    assert any(t.startswith("PC ") for t in lab_texts)


def test_volcano_honours_style_preset():
    fc, p, q, mz, _ = _demo_arrays()
    base = export.render_volcano_figure(fc, p, q, mz=mz, title="V", dpi=80)
    with export.active_style("Poster (large type)"):
        big = export.render_volcano_figure(fc, p, q, mz=mz, title="V", dpi=80)
    assert _title_size(big) > _title_size(base) * 1.2
    assert export.current_style_spec() is None       # restored


def test_volcano_length_mismatch_is_truncated():
    # shorter arrays must not raise (min length used)
    fig = export.render_volcano_figure([0.1, 0.2, 0.3], [0.5, 0.6], [0.9], dpi=80)
    assert fig.axes


# --------------------------------------------------------------------------- #
# Export dialog (offscreen Qt)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def app():
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _demo_res():
    fc, p, q, mz, labels = _demo_arrays()
    res = pd.DataFrame({"mz": mz, "log2_fc": fc, "p_value": p, "q_value": q,
                        "best_lipid": [lab if i % 2 else "" for i, lab in enumerate(labels)]})
    res.attrs["a_label"] = "Normal"
    res.attrs["b_label"] = "Sciatic"
    return res


def test_dialog_builds_and_previews(app):
    from PySide6 import QtWidgets
    from smile_msi.gui.plotexport import VolcanoExportDialog
    win = QtWidgets.QMainWindow()
    dlg = VolcanoExportDialog(win, _demo_res(), a_label="Normal", b_label="Sciatic")
    assert dlg._pixmap is not None                    # preview rendered
    assert dlg.preset_combo.count() >= 6              # style presets present
    assert dlg.a_edit.text() == "Normal"


def test_dialog_options_change_and_export(app, tmp_path):
    from PySide6 import QtWidgets
    from smile_msi.gui.plotexport import VolcanoExportDialog
    win = QtWidgets.QMainWindow()
    dlg = VolcanoExportDialog(win, _demo_res(), a_label="Normal", b_label="Sciatic")
    dlg.theme_combo.setCurrentText("Dark")
    dlg.label_combo.setCurrentIndex(4)                # Top 20
    dlg._render()
    assert dlg._pixmap is not None
    # export path (drive the real save, bypassing the native file dialog)
    out = str(tmp_path / "v.pdf")
    fig = dlg._make_figure(dpi=150)
    export.save_figure(fig, out, dpi=150, fmt="pdf")
    assert os.path.getsize(out) > 0
    assert export.current_style_spec() is None         # dialog render left no active style


def test_dialog_applies_preset_to_preview(app):
    from PySide6 import QtWidgets
    from smile_msi.gui.plotexport import VolcanoExportDialog
    win = QtWidgets.QMainWindow()
    dlg = VolcanoExportDialog(win, _demo_res())
    i = dlg.preset_combo.findText("Poster (large type)")
    dlg.preset_combo.setCurrentIndex(i)
    dlg._render()
    fig = dlg._make_figure(dpi=90)
    base = export.render_volcano_figure(dlg._fc, dlg._p, dlg._q, mz=dlg._mz, **dlg._opts(), dpi=90)
    assert _title_size(fig) > _title_size(base)        # preset enlarged the title


# --------------------------------------------------------------------------- #
# Heatmap render + generic FigureExportDialog
# --------------------------------------------------------------------------- #
def test_heatmap_renders_with_groups_and_scaling():
    rng = np.random.default_rng(5)
    M = rng.gamma(2.0, 1.0, (12, 40))
    M[rng.random(M.shape) < 0.1] = np.nan                # some absent → floor, no crash
    rows = [f"S{i}" for i in range(12)]
    cols = [f"{m:.1f}" for m in np.linspace(700, 900, 40)]
    groups = ["A"] * 6 + ["B"] * 6
    for scale in ("col", "row", "none"):
        fig = export.render_heatmap_figure(M, row_labels=rows, col_labels=cols, scale=scale,
                                           group_labels=groups, dpi=90)
        assert fig.axes
    # honours a preset
    base = export.render_heatmap_figure(M, row_labels=rows, title="H", dpi=80)
    with export.active_style("Poster (large type)"):
        big = export.render_heatmap_figure(M, row_labels=rows, title="H", dpi=80)
    assert _title_size(big) > _title_size(base) * 1.2


def test_scale_matrix_normalises_and_fills_nan():
    from smile_msi.export import _scale_matrix
    M = np.array([[1.0, np.nan], [3.0, 10.0]])
    out = _scale_matrix(M, "col")
    assert np.all((out >= 0) & (out <= 1))
    assert not np.isnan(out).any()                       # NaN → 0 (floor)


def test_segmentation_grid_and_montage_render():
    from smile_msi import palettes
    rng = np.random.default_rng(7)
    # segmentation grid: 3 slides, shared palette, NaN off-tissue
    panels = []
    for i in range(3):
        lab = rng.integers(0, 4, (30, 40)).astype(float)
        lab[rng.random(lab.shape) < 0.3] = np.nan
        panels.append((f"Slide {i + 1}", lab))
    colors = {cl: palettes.SEGMENT[cl] for cl in range(4)}
    fig = export.render_segmentation_grid(panels, colors, dpi=80)
    assert len(fig.axes) >= 3
    # montage: 5 ion images
    imgs = [rng.gamma(2, 1, (25, 30)) for _ in range(5)]
    labs = [f"m/z {700 + 40 * i:.2f}" for i in range(5)]
    fig2 = export.render_montage_figure(imgs, labs, cmap="inferno", dpi=80)
    assert len(fig2.axes) >= 5
    # both honour a preset (title grows)
    base = export.render_montage_figure(imgs, labs, title="M", dpi=70)
    with export.active_style("Poster (large type)"):
        big = export.render_montage_figure(imgs, labs, title="M", dpi=70)
    bt = next((t.get_fontsize() for t in base.texts if t.get_text()), None)
    gt = next((t.get_fontsize() for t in big.texts if t.get_text()), None)
    assert bt and gt and gt > bt * 1.2


def test_embedding_export_dialog(app, tmp_path):
    from PySide6 import QtWidgets
    from smile_msi import umapstudio as us
    from smile_msi.gui.plotexport import EmbeddingExportDialog
    rng = np.random.default_rng(8)
    coords = np.vstack([rng.normal([-2, 0], 0.6, (150, 2)), rng.normal([2, 1], 0.6, (150, 2))])
    labels = np.array(["A"] * 150 + ["B"] * 150, dtype=object)
    data = us.EmbeddingData(coords=coords, method="Component", categorical={"Class": labels})
    win = QtWidgets.QMainWindow()
    dlg = EmbeddingExportDialog(win, data, default_color_by="Class", title="Export scores")
    assert dlg._pixmap is not None
    assert dlg.preset_combo.count() >= 6
    dlg._controls["mode"][0].setCurrentText("Dots (scatter)")
    dlg._render()
    out = str(tmp_path / "e.png")
    export.save_figure(dlg._figure(dpi=120), out, dpi=120)
    assert os.path.getsize(out) > 0
    assert export.current_style_spec() is None


def test_generic_dialog_builds_options_and_exports(app, tmp_path):
    from PySide6 import QtWidgets
    from smile_msi.gui.plotexport import FigureExportDialog
    rng = np.random.default_rng(6)
    M = rng.random((10, 30))
    calls = {}

    def render(opts):
        calls.update(opts)
        return export.render_heatmap_figure(M, cmap=opts["cmap"], theme=opts["theme"],
                                            title=opts["title"], dpi=opts["dpi"])
    options = [
        {"key": "title", "label": "Title", "kind": "text", "default": "H"},
        {"key": "theme", "label": "Theme", "kind": "combo",
         "choices": [("Light", "light"), ("Dark", "dark")], "default": "light"},
        {"key": "cmap", "label": "Colormap", "kind": "combo",
         "choices": [("viridis", "viridis"), ("magma", "magma")], "default": "viridis"},
    ]
    win = QtWidgets.QMainWindow()
    dlg = FigureExportDialog(win, render=render, options=options, default_name="h.png")
    assert dlg._pixmap is not None
    # options collected + passed through (plus injected dpi)
    assert calls["title"] == "H" and calls["cmap"] == "viridis" and "dpi" in calls
    dlg._controls["cmap"][0].setCurrentText("magma")
    dlg._render()
    assert calls["cmap"] == "magma"
    out = str(tmp_path / "h.png")
    export.save_figure(dlg._figure(dpi=120), out, dpi=120)
    assert os.path.getsize(out) > 0
    assert export.current_style_spec() is None
