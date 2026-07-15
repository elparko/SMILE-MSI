"""Interactive result renderers for the generic analysis dialog (plan 24, Phase 2a).

Every registry analysis produces one raw result; :func:`render_result` turns it into a
QWidget keyed by the step's ``result_kind`` (features|segmentation|stats|table|components|
classifier|embedding). Two concrete views cover the field:

* :class:`TableResultView` — a sortable table over a DataFrame with CSV export; clicking a
  row that carries an m/z sets the window's active ion (so a stats/marker table drives the
  ion image, exactly as the dedicated tabs do).
* :class:`IonGridView` — a small grid of ion images built from ``sd.rep_ions(result, n, ds)``
  (the representative ions a step nominates), for the image-first kinds (features/components).

Dispatch is table-first and total: an unknown or image-less result falls back to the step's
own ``to_table`` DataFrame, and a step with no table at all yields a plain note. Nothing here
raises — a renderer that cannot build (no dataset, empty result) returns ``None`` so the
dispatcher moves on to the fallback.
"""
from __future__ import annotations

import traceback

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from . import common
from .. import imaging


# how many representative ions an IonGridView shows by default (mirrors Flow's rep_top_n feel)
_REP_N = 12


def _note_widget(text):
    w, v = common.tab_page()
    v.addWidget(common.note(text))
    v.addStretch(1)
    return w


def _mz_column(headers):
    """Index of the m/z column in ``headers`` (an ``mz`` / ``m/z`` cell), else None. Used to
    make a result table's rows click-to-select the active ion."""
    low = [str(h).strip().lower() for h in headers]
    for exact in ("mz", "m/z"):
        if exact in low:
            return low.index(exact)
    for i, h in enumerate(low):
        if h.startswith("mz") or "m/z" in h:
            return i
    return None


class VolcanoView(QtWidgets.QWidget):
    """A volcano scatter over an A-vs-B stats table: signed log2 fold-change (x) vs
    −log10 q (or p) (y), one point per ion, click-to-select the active ion. Significant
    points (|log2FC| ≥ 1 and q ≤ 0.05) are highlighted. Builds nothing when the table lacks
    the fold-change / significance columns, so the dispatcher falls back to the plain table."""

    def __init__(self, win, df, *, parent=None):
        super().__init__(parent)
        self.win = win
        self._built = False
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        cols = {str(c).lower(): c for c in df.columns}
        fc_col = cols.get("log2_fc") or cols.get("log2fc")
        sig_col = cols.get("q_value") or cols.get("q") or cols.get("p_value") or cols.get("p")
        mz_col = cols.get("mz") or cols.get("m/z")
        if fc_col is None or sig_col is None or mz_col is None:
            outer.addWidget(common.note("No volcano columns (need log2_fc + q/p + mz)."))
            return
        fc = np.asarray(df[fc_col], dtype=float)
        q = np.asarray(df[sig_col], dtype=float)
        self._mz = np.asarray(df[mz_col], dtype=float)
        with np.errstate(divide="ignore"):
            y = -np.log10(np.clip(q, 1e-300, 1.0))
        sig = (np.abs(fc) >= 1.0) & (q <= 0.05)

        gw = pg.GraphicsLayoutWidget()
        common.dark_image_view(gw)
        self._plot = gw.addPlot()
        self._plot.setLabel("bottom", "log2 fold-change (B / A)")
        self._plot.setLabel("left", "-log10 q")
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        brushes = [pg.mkBrush(220, 70, 70, 220) if s else pg.mkBrush(150, 150, 150, 120)
                   for s in sig]
        self._sc = pg.ScatterPlotItem(x=fc, y=y, brush=brushes, pen=None, size=7, data=self._mz)
        self._sc.sigClicked.connect(self._on_click)
        self._plot.addItem(self._sc)
        outer.addWidget(gw, 1)
        outer.addWidget(common.plot_caption(
            f"{int(sig.sum())} of {len(fc)} ions significant (|log2FC|≥1, q≤0.05). "
            "Click a point to show its ion image."))
        self._built = True

    def _on_click(self, _plot, points):
        if not points or not hasattr(self.win, "set_active_mz"):
            return
        try:
            self.win.set_active_mz(float(points[0].data()))
        except (TypeError, ValueError):
            pass

    def has_content(self) -> bool:
        return self._built


class SpectraView(QtWidgets.QWidget):
    """Mean-spectrum overlay (or A−B difference) of two regions — the Region-comparison view.

    Reads a dict ``{axis, spec_a, spec_b, a_label, b_label}``. A toggle switches between the
    overlaid spectra and their difference. Builds nothing without the spectra arrays."""

    def __init__(self, win, result, *, parent=None):
        super().__init__(parent)
        self._built = False
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        if not isinstance(result, dict) or "axis" not in result:
            outer.addWidget(common.note("No spectra to show."))
            return
        self._axis = np.asarray(result["axis"], dtype=float)
        self._a = np.asarray(result["spec_a"], dtype=float)
        self._b = np.asarray(result["spec_b"], dtype=float)
        self._la = str(result.get("a_label", "A"))
        self._lb = str(result.get("b_label", "B"))

        bar = common.ControlBar()
        self._mode = common.NoScrollComboBox()
        self._mode.addItems(["Overlay", "Difference (A − B)"])
        self._mode.currentIndexChanged.connect(self._draw)
        bar.add(QtWidgets.QLabel("View"), self._mode)
        outer.addWidget(bar)

        gw = pg.GraphicsLayoutWidget()
        common.dark_image_view(gw)
        self._plot = gw.addPlot()
        self._plot.setLabel("bottom", "m/z")
        self._plot.addLegend()
        outer.addWidget(gw, 1)
        self._draw()
        self._built = True

    def _draw(self, *_):
        self._plot.clear()
        if self._mode.currentIndex() == 1:
            self._plot.plot(self._axis, self._a - self._b, pen=pg.mkPen("#e0662d", width=1),
                            name=f"{self._la} − {self._lb}")
        else:
            self._plot.plot(self._axis, self._a, pen=pg.mkPen("#2d7dd2", width=1), name=self._la)
            self._plot.plot(self._axis, self._b, pen=pg.mkPen("#d24d4d", width=1), name=self._lb)

    def has_content(self) -> bool:
        return self._built


class TableResultView(QtWidgets.QWidget):
    """A sortable table over a result DataFrame, with a ``⤓ CSV`` export.

    If the table carries an m/z column, clicking a row sets the owning window's active ion
    (``win.set_active_mz``) so the result immediately drives the ion image — the same link
    the Region-comparison / marker tables give."""

    def __init__(self, win, df, *, parent=None):
        super().__init__(parent)
        self.win = win
        self._df = df
        headers = [str(c) for c in df.columns]
        self._mz_col = _mz_column(headers)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        bar = common.ControlBar()
        bar.add(common.button("⤓ CSV", self._export_csv,
                              tooltip="Save this table as a CSV file"))
        if self._mz_col is not None:
            bar.set_status("Click a row to show its ion image.")
        v.addWidget(bar)

        self.table = QtWidgets.QTableWidget()
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QtWidgets.QTableWidget.SelectRows)
        self.table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        rows = [[("" if val is None else val) for val in row] for row in df.itertuples(index=False)]
        common.fill_table(self.table, headers, rows)
        if self._mz_col is not None:
            self.table.cellClicked.connect(self._on_cell)
        v.addWidget(self.table, 1)

    def _on_cell(self, row, _col):
        if self._mz_col is None or not hasattr(self.win, "set_active_mz"):
            return
        item = self.table.item(row, self._mz_col)
        if item is None:
            return
        try:
            self.win.set_active_mz(float(item.text()))
        except (TypeError, ValueError):
            pass

    def _export_csv(self):
        common.export_table(self, self.table, stem="analysis_result",
                            title="Export result table")


class IonGridView(QtWidgets.QWidget):
    """A grid of ion images for a step's representative ions (``sd.rep_ions``).

    Each cell is an independent tissue canvas (pinned to the fixed dark ion background)
    showing ``ds.ion_image(mz)`` under the window's current tolerance/reduce/normalization.
    A cell click sets the active ion. Builds nothing (and reports ``has_content() == False``)
    when there is no dataset or the step nominates no ions — the dispatcher then falls back to
    a table."""

    def __init__(self, win, sd, result, *, n=_REP_N, parent=None):
        super().__init__(parent)
        self.win = win
        ds = getattr(win, "ds", None)
        try:
            reps = list(sd.rep_ions(result, int(n), ds) or [])
        except Exception:                                   # noqa: BLE001 — never break rendering
            traceback.print_exc()
            reps = []
        self._built = 0

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        if ds is None or not reps:
            outer.addWidget(common.note("No ion images to show."))
            return

        cmap_name = "viridis"
        combo = getattr(win, "cmap_combo", None)
        if combo is not None:
            try:
                cmap_name = combo.currentText()
            except Exception:                               # noqa: BLE001
                cmap_name = "viridis"
        lut = common.colormap(cmap_name).getLookupTable(0.0, 1.0, 256)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        inner = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(inner)
        grid.setContentsMargins(2, 2, 2, 2)
        grid.setSpacing(8)

        ppm = float(getattr(win, "ppm", 10.0))
        reduce = getattr(win, "reduce", "sum")
        norm = getattr(win, "norm", "none")
        cols = 3
        for k, rep in enumerate(reps):
            mz = float(rep[0])
            label = str(rep[1]) if len(rep) > 1 and rep[1] else ""
            try:
                img = ds.ion_image(mz, tol_ppm=ppm, reduce=reduce, norm=norm)
            except Exception:                               # noqa: BLE001 — skip an ion that won't image
                traceback.print_exc()
                continue
            cell = self._ion_cell(mz, label, img, lut)
            grid.addWidget(cell, k // cols, k % cols)
            self._built += 1

        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)
        if not self._built:
            outer.addWidget(common.note("No ion images to show."))

    def _ion_cell(self, mz, label, img, lut):
        box = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(box)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(2)

        gw = pg.GraphicsLayoutWidget()
        gw.setFixedHeight(160)
        common.dark_image_view(gw)
        vb = gw.addViewBox()
        vb.setAspectLocked(True)
        vb.invertY(True)
        vb.setMouseEnabled(False, False)
        item = pg.ImageItem()
        clipped = imaging.quantile_clip(np.asarray(img, dtype=float), high=99.0)
        item.setImage(clipped, autoLevels=True)
        item.setLookupTable(lut)
        vb.addItem(item)
        # a cell click selects the ion (the tissue view then follows the active feature)
        item.mouseClickEvent = lambda ev, m=mz: self._pick(ev, m)

        cap = common.plot_caption(f"m/z {mz:.4f}" + (f"  {label}" if label else ""))
        cap.setWordWrap(True)
        bl.addWidget(gw)
        bl.addWidget(cap)
        return box

    def _pick(self, ev, mz):
        try:
            ev.accept()
        except Exception:                                   # noqa: BLE001
            pass
        if hasattr(self.win, "set_active_mz"):
            self.win.set_active_mz(float(mz))

    def has_content(self) -> bool:
        return self._built > 0


class ImageResultView(QtWidgets.QWidget):
    """A single labelled/scalar image on the fixed dark ion canvas.

    Built from a result payload carrying a 2-D ``image`` array (e.g. the per-ion DGMM
    segmentation into intensity zones), colour-mapped by the window's current colormap, with a
    caption of the component means. Static — the view is read-only. Builds nothing (and reports
    ``has_content() == False``) when the payload has no image, so the dispatcher falls back to a
    table."""

    def __init__(self, win, result, *, parent=None):
        super().__init__(parent)
        self.win = win
        self._built = False
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        img = None
        if isinstance(result, dict):
            img = result.get("image")
        if img is None:
            outer.addWidget(common.note("No image to show."))
            return

        cmap_name = "viridis"
        combo = getattr(win, "cmap_combo", None)
        if combo is not None:
            try:
                cmap_name = combo.currentText()
            except Exception:                               # noqa: BLE001
                cmap_name = "viridis"

        gw = pg.GraphicsLayoutWidget()
        common.dark_image_view(gw)
        vb = gw.addViewBox()
        vb.setAspectLocked(True)
        vb.invertY(True)
        vb.setMouseEnabled(False, False)
        item = pg.ImageItem()
        item.setImage(np.asarray(img, dtype=float), autoLevels=True)
        try:
            item.setColorMap(common.colormap(cmap_name))
        except Exception:                                   # noqa: BLE001 — colormap is cosmetic
            pass
        vb.addItem(item)
        outer.addWidget(gw, 1)

        means = result.get("means") or []
        mz = float(result.get("mz", 0.0))
        if means:
            zones = ", ".join(f"L{i}={m:.3g}" for i, m in enumerate(means))
            cap = common.plot_caption(
                f"m/z {mz:.4f} — {len(means)} intensity zones ({zones}). Brightest = highest label.")
            cap.setWordWrap(True)
            outer.addWidget(cap)
        self._built = True

    def has_content(self) -> bool:
        return self._built


class EmbeddingView(QtWidgets.QWidget):
    """A 2-D pixel embedding (UMAP / t-SNE) scatter coloured by each pixel's molecular
    fingerprint, with freehand **lasso → region**: drag a loop to select a cloud of pixels
    and save it as a named region (the same carve-a-region-in-molecular-space affordance the
    old Feature-space tab gave).

    Reads ``.coords`` (N×2), ``.rgba`` (N×4) and ``.index`` (dataset pixel ids per point) off
    the :class:`~smile_msi.multivariate.Embedding` result. Builds nothing when there are no
    coordinates, so the dispatcher falls back to a table."""

    def __init__(self, win, result, *, parent=None):
        super().__init__(parent)
        self.win = win
        self._built = False
        self._sel = None
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        coords = getattr(result, "coords", None)
        if coords is None:
            outer.addWidget(common.note("No embedding coordinates to show."))
            return
        self._coords = np.asarray(coords, dtype=float)
        if self._coords.ndim != 2 or self._coords.shape[0] == 0:
            outer.addWidget(common.note("No embedding coordinates to show."))
            return
        self._index = getattr(result, "index", None)
        rgb = None
        try:
            rgb = result.point_rgb()                        # (N,3) float 0..1, matches the tissue map
        except Exception:                                   # noqa: BLE001
            rgb = None

        from .scatter import _LassoViewBox, points_in_polygon
        self._pip = points_in_polygon

        bar = common.ControlBar()
        self._sel_info = QtWidgets.QLabel("Drag to lasso-select pixels.")
        bar.add(self._sel_info)
        self._to_region = common.button("Selection → region", self._save_region,
                                        tooltip="Save the lassoed pixels as a named region")
        self._to_region.setEnabled(False)
        bar.add(self._to_region)
        outer.addWidget(bar)

        gw = pg.GraphicsLayoutWidget()
        common.dark_image_view(gw)
        self._vb = _LassoViewBox()
        self._vb.setAspectLocked(True)
        gw.addItem(self._vb)
        self._scatter = pg.ScatterPlotItem(pen=None, size=4)
        self._vb.addItem(self._scatter)
        xy = self._coords
        if rgb is not None and np.asarray(rgb).ndim == 2:
            cols = (np.clip(np.asarray(rgb, dtype=float), 0, 1) * 255).astype(int)
            brushes = [pg.mkBrush(int(c[0]), int(c[1]), int(c[2]), 255) for c in cols]
            self._scatter.setData(x=xy[:, 0], y=xy[:, 1], brush=brushes, pen=None)
        else:
            self._scatter.setData(x=xy[:, 0], y=xy[:, 1])
        self._vb.lassoFinished.connect(self._on_lasso)
        self._vb.setLasso(True)
        outer.addWidget(gw, 1)
        outer.addWidget(common.plot_caption(
            "Each point is a pixel, coloured by its molecular fingerprint. Drag a loop to "
            "select a cloud → Selection → region."))
        self._built = True

    def _on_lasso(self, poly):
        try:
            sub = self._pip(self._coords, np.asarray(poly, dtype=float))
        except Exception:                                   # noqa: BLE001
            traceback.print_exc()
            return
        self._sel = np.asarray(sub, dtype=bool)
        n = int(self._sel.sum())
        self._sel_info.setText(f"{n:,} pixels selected")
        self._to_region.setEnabled(n > 0)

    def _full_mask(self):
        """Lift the scatter-space selection to a full per-dataset-pixel mask (identity for a
        whole-slide embedding; scattered back into place for a region-scoped one)."""
        sub = self._sel
        ds = getattr(self.win, "ds", None)
        if sub is None or ds is None:
            return None
        idx = self._index
        if idx is None or len(idx) == ds.n_pixels:
            return sub
        full = np.zeros(ds.n_pixels, dtype=bool)
        full[np.asarray(idx)[sub]] = True
        return full

    def _save_region(self):
        mask = self._full_mask()
        if mask is None or not mask.any():
            return
        if hasattr(self.win, "_new_region"):
            self.win._new_region(name="feature-space selection", mask=mask, select=True)
            self.win.statusBar().showMessage(f"Saved region ({int(mask.sum()):,} pixels).")

    def has_content(self) -> bool:
        return self._built


class HeatmapView(QtWidgets.QWidget):
    """A labelled similarity matrix (region×region or module correlation) on a diverging
    colour scale, with per-cell value text for small matrices.

    Reads a ``.matrix`` (2-D array) and optional ``.names`` off the result object. Builds
    nothing (``has_content() == False``) when there is no square matrix, so the dispatcher
    falls back to the step's table."""

    def __init__(self, win, result, *, parent=None):
        super().__init__(parent)
        self._built = False
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        mat = getattr(result, "matrix", None)
        if mat is None and isinstance(result, dict):
            mat = result.get("matrix")
        try:
            mat = np.asarray(mat, dtype=float)
        except Exception:                                   # noqa: BLE001
            mat = None
        if mat is None or mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] == 0:
            outer.addWidget(common.note("No matrix to show."))
            return
        names = list(getattr(result, "names", None) or getattr(result, "peaks", None) or
                     range(mat.shape[0]))
        names = [str(n) for n in names][:mat.shape[0]]

        gw = pg.GraphicsLayoutWidget()
        common.dark_image_view(gw)
        plot = gw.addPlot()
        plot.setAspectLocked(True)
        plot.invertY(True)
        item = pg.ImageItem()
        item.setImage(mat, autoLevels=False)
        # symmetric diverging scale centred at 0 (correlations run -1..1)
        hi = float(np.nanmax(np.abs(mat))) or 1.0
        item.setLevels((-hi, hi))
        try:
            item.setColorMap(common.colormap("magma"))
        except Exception:                                   # noqa: BLE001
            pass
        plot.addItem(item)
        # tick labels from the names (only when few enough to read)
        if len(names) <= 30:
            ticks = list(enumerate(names))
            for ax in ("left", "bottom"):
                a = plot.getAxis(ax)
                a.setTicks([[(i + 0.5, n) for i, n in ticks]])
            if len(names) <= 12:                            # value text for small matrices
                for i in range(mat.shape[0]):
                    for j in range(mat.shape[1]):
                        t = pg.TextItem(f"{mat[i, j]:.2f}", color="w", anchor=(0.5, 0.5))
                        t.setPos(j + 0.5, i + 0.5)
                        plot.addItem(t)
        plot.setMouseEnabled(False, False)
        outer.addWidget(gw, 1)
        self._built = True

    def has_content(self) -> bool:
        return self._built


# --------------------------------------------------------------------------- #
# result_kind -> factory(win, sd, result) -> QWidget | None
# --------------------------------------------------------------------------- #
def _image_factory(win, sd, result):
    try:
        view = ImageResultView(win, result)
    except Exception:                                       # noqa: BLE001 — never break rendering
        traceback.print_exc()
        return None
    return view if view.has_content() else None


def _table_factory(win, sd, result):
    try:
        df = sd.to_table(result)
    except Exception:                                       # noqa: BLE001
        traceback.print_exc()
        return None
    if df is None or not len(df):
        return None
    return TableResultView(win, df)


def _ion_factory(win, sd, result):
    view = IonGridView(win, sd, result)
    return view if view.has_content() else None


def _embedding_factory(win, sd, result):
    try:
        view = EmbeddingView(win, result)
    except Exception:                                       # noqa: BLE001 — never break rendering
        traceback.print_exc()
        return None
    return view if view.has_content() else None


def _stats_factory(win, sd, result):
    """An A-vs-B comparison: the volcano over its stats table (both work off the DataFrame, so
    a rehydrated CSV renders identically to a live run)."""
    try:
        import pandas as pd
        df = result if isinstance(result, pd.DataFrame) else sd.to_table(result)
    except Exception:                                       # noqa: BLE001
        traceback.print_exc()
        df = None
    if df is None or not len(df):
        return None
    table = TableResultView(win, df)
    try:
        volcano = VolcanoView(win, df)
    except Exception:                                       # noqa: BLE001
        traceback.print_exc()
        volcano = None
    if volcano is None or not volcano.has_content():
        return table
    split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
    split.addWidget(volcano)
    split.addWidget(table)
    split.setSizes([380, 260])
    return split


def _spectra_factory(win, sd, result):
    try:
        view = SpectraView(win, result)
    except Exception:                                       # noqa: BLE001
        traceback.print_exc()
        return None
    return view if view.has_content() else None


def _matrix_factory(win, sd, result):
    """A similarity matrix over a summary table: the heatmap on top, the step's ``to_table``
    (e.g. each region's closest match) below. Falls back to the table alone if the heatmap
    can't build."""
    try:
        heat = HeatmapView(win, result)
    except Exception:                                       # noqa: BLE001
        traceback.print_exc()
        heat = None
    table = _table_factory(win, sd, result)
    if heat is None or not heat.has_content():
        return table
    if table is None:
        return heat
    split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
    split.addWidget(heat)
    split.addWidget(table)
    split.setSizes([420, 240])
    return split


RENDERERS = {
    "features": _ion_factory,
    "components": _ion_factory,
    "ion_segmentation": _image_factory,     # a single labelled ion image (per-ion DGMM)
    "matrix": _matrix_factory,              # region×region / module similarity heatmap + summary
    "segmentation": _table_factory,
    "stats": _stats_factory,                # A-vs-B volcano over the stats table
    "spectra": _spectra_factory,            # two regions' mean-spectrum overlay / difference
    "table": _table_factory,
    "classifier": _table_factory,
    "embedding": _embedding_factory,
}


def render_result(win, sd, result):
    """Dispatch ``result`` to the renderer for ``sd.result_kind``, falling back to a table
    over ``sd.to_table`` and finally to a plain note. Never raises — a renderer that cannot
    build returns ``None`` and dispatch moves on."""
    kind = getattr(sd, "result_kind", "table")
    # A result that is ALREADY a DataFrame (a step that returns a table raw, or a run rehydrated
    # from the store where the persisted payload is the table) renders directly — sd.to_table
    # expects the RAW engine result, so it must not be re-applied to an already-tabular payload.
    # EXCEPT the "stats" kind, whose renderer (volcano + table) itself works off the DataFrame,
    # so both a live run and a rehydrated CSV go through it rather than short-circuiting to a
    # bare table.
    try:
        import pandas as pd
        if isinstance(result, pd.DataFrame) and kind != "stats":
            return TableResultView(win, result) if len(result) else \
                _note_widget("The result table is empty.")
    except Exception:                                       # noqa: BLE001
        traceback.print_exc()
    factory = RENDERERS.get(kind)
    if factory is not None:
        try:
            w = factory(win, sd, result)
            if w is not None:
                return w
        except Exception:                                   # noqa: BLE001
            traceback.print_exc()
    # universal fallback: the step's own table, else a note.
    fallback = _table_factory(win, sd, result)
    if fallback is not None:
        return fallback
    return _note_widget("No tabular result to display.")
