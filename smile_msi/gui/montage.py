"""MontageTabMixin — the ion-montage grid, demoted off the tab strip to a modeless dialog
opened from File ▸ Ion montage grid… (plan 24 Phase 6). It was never a registry view."""
from __future__ import annotations


import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets

from .. import (spatial, imaging)
from .common import (colormap, ControlBar, dark_image_view, icon, tab_page, tool_button,
                     NoScrollComboBox)
from .scope import ScopeBar


class MontageTabMixin:
    def _open_montage_dialog(self):
        """Ion montage grid — demoted off the tab strip to File ▸ Ion montage grid… (plan 24
        Phase 6). Builds the montage UI once into a modeless dialog; later opens just raise it.
        It was never a registry view, so nothing reveal_view()s it."""
        win = getattr(self, "_montage_window", None)
        if win is None:
            win = QtWidgets.QDialog(self)
            win.setWindowTitle("Ion montage grid")
            win.setModal(False)
            lay = QtWidgets.QVBoxLayout(win)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(self._build_montage_body())
            try:
                geo = win.screen().availableGeometry()
                win.resize(int(geo.width() * 0.6), int(geo.height() * 0.72))
            except Exception:  # noqa: BLE001
                win.resize(900, 720)
            self._montage_window = win
        win.show()
        win.raise_()
        win.activateWindow()

    def _build_montage_body(self):
        w, v = tab_page()
        # "Data in this analysis": the montage ranks/draws ion images from exactly
        # these features — pick a subset to montage just those.
        self.montage_scope = ScopeBar(self, feature=True)
        v.addWidget(self.montage_scope)
        bar = ControlBar()
        self.montage_n = NoScrollComboBox()
        self.montage_n.addItems(["4", "6", "9", "12", "16", "20"])
        self.montage_n.setCurrentText("9")
        self.montage_src = NoScrollComboBox()
        self.montage_src.addItems(["top intensity", "most spatial (Moran's I)"])
        b = QtWidgets.QPushButton("Render")
        b.setObjectName("primaryAction")                  # the screen's single run action
        b.setIcon(icon("run"))
        b.clicked.connect(self.do_montage)
        b_reset = tool_button(name="refresh", tooltip="Reset view",
                              slot=self._reset_montage_view)
        b_save = tool_button(name="export", tooltip="Export the montage as a publication figure "
                                                    "(live preview + colormap / contrast / columns "
                                                    "+ Style preset).",
                             slot=self._export_montage_dialog)
        bar.add_group("# ions", self.montage_n, "rank by", self.montage_src)
        bar.add(b, b_reset, b_save)
        v.addWidget(bar)
        self.montage_view = pg.GraphicsLayoutWidget()
        dark_image_view(self.montage_view)   # ion-image montage reads against black in both themes
        v.addWidget(self.montage_view)
        return w

    def do_montage(self):
        if not self.peaks:
            self.statusBar().showMessage("Find peaks first.")
            return
        n = int(self.montage_n.currentText())
        mzs = self._scope_mzs(self.montage_scope)              # bar's subset, else visible features
        if self.montage_src.currentText().startswith("most spatial"):
            sa = spatial.spatial_autocorrelation(self.ds, mzs, tol_ppm=self.ppm, norm=self.norm)
            ordered = sa["mz"].tolist()[:n]
        else:
            # rank the scoped features by intensity (self.peaks is sorted by m/z, not intensity)
            keyset = {round(float(m), 4) for m in mzs}
            pool = [p for p in self.peaks if round(float(p["mz"]), 4) in keyset]
            vis = sorted(pool, key=lambda p: p.get("intensity", 0.0), reverse=True)
            ordered = [p["mz"] for p in vis[:n]]
        self._montage_mzs = [float(m) for m in ordered]        # remembered for figure export
        self.montage_view.clear()
        # Re-rendering rebuilds the panel list; drop any links to discarded viewboxes.
        self._montage_vbs = []
        ncol = int(np.ceil(np.sqrt(n)))            # square-ish grid (4→2×2, 9→3×3, 16→4×4)
        lut = colormap(self.cmap_combo.currentText()).getLookupTable()
        high = float(self.contrast_spin.value())
        first_vb = None
        for k, mz in enumerate(ordered):
            r, c = divmod(k, ncol)
            img = imaging.quantile_clip(
                self.ds.ion_image(mz, tol_ppm=self.ppm, reduce=self.reduce, norm=self.norm), high=high)
            lab = self.annotate(mz)
            short = (lab.split(" (")[0] if lab else "")
            self.montage_view.addLabel(f"m/z {mz:.3f}  {short}", row=2 * r, col=c, size="8pt")
            vb = self.montage_view.addViewBox(row=2 * r + 1, col=c)
            vb.invertY(True)
            # Square pixels, but keep interactive pan/zoom (aspect lock does not block it).
            vb.setAspectLocked(True)
            vb.setMouseEnabled(x=True, y=True)
            it = pg.ImageItem(img)
            it.setLookupTable(lut)
            vb.addItem(it)
            # Link every subsequent panel's X/Y range to the first so pan/zoom moves all
            # panels together. All ion images share the same pixel grid, so linking keeps
            # the field of view aligned across panels.
            if first_vb is None:
                first_vb = vb
            else:
                vb.setXLink(first_vb)
                vb.setYLink(first_vb)
            self._montage_vbs.append(vb)
        # autoRange after links are wired: ranging the first propagates to the linked rest.
        if first_vb is not None:
            first_vb.autoRange()
        self.statusBar().showMessage(f"Montage of {len(ordered)} ions.")

    def _export_montage_dialog(self):
        """Publication export of the ion montage (live preview + colormap / contrast / columns +
        Style preset), replacing the raw pyqtgraph screenshot."""
        mzs = getattr(self, "_montage_mzs", None)
        if not mzs or self.ds is None:
            self.statusBar().showMessage("Render a montage first.")
            return
        from .. import export, palettes
        from .plotexport import FigureExportDialog
        images = [self.ds.ion_image(mz, tol_ppm=self.ppm, reduce=self.reduce, norm=self.norm)
                  for mz in mzs]
        labels = []
        for mz in mzs:
            lab = self.annotate(mz)
            short = (lab.split(" (")[0] if lab else "")
            labels.append(f"m/z {mz:.3f}  {short}".strip())
        cmap0 = self.cmap_combo.currentText() if getattr(self, "cmap_combo", None) is not None else "viridis"
        high0 = float(self.contrast_spin.value()) if getattr(self, "contrast_spin", None) is not None else 99.0

        def render(opts):
            return export.render_montage_figure(
                images, labels, cmap=opts["cmap"], high=float(opts["high"]),
                title=opts["title"], theme=opts["theme"], ncols=(int(opts["ncols"]) or None),
                dpi=opts["dpi"], pixel_size_um=getattr(self.ds, "pixel_size_um", None))

        options = [
            {"key": "title", "label": "Title", "kind": "text", "default": "Ion montage"},
            {"key": "theme", "label": "Theme", "kind": "combo",
             "choices": [("Dark", "dark"), ("Light", "light")], "default": "dark"},
            {"key": "cmap", "label": "Colormap", "kind": "combo",
             "choices": [(c, c) for c in palettes.SEQUENTIAL_CMAPS],
             "default": cmap0 if cmap0 in palettes.SEQUENTIAL_CMAPS else "viridis"},
            {"key": "high", "label": "Contrast", "kind": "float", "min": 80.0, "max": 100.0,
             "step": 0.5, "default": high0, "suffix": " %",
             "tooltip": "Per-panel hotspot-clip percentile (matches the on-screen montage)."},
            {"key": "ncols", "label": "Columns (0 = auto)", "kind": "int", "min": 0, "max": 10,
             "default": 0},
        ]
        FigureExportDialog(self, render=render, options=options, default_name="montage.png",
                           title="Export montage").exec()

    def _reset_montage_view(self):
        """Reset the linked montage panels so each image fills its panel again."""
        vbs = getattr(self, "_montage_vbs", [])
        if not vbs:
            return
        # autoRange on the link master propagates to all linked panels.
        vbs[0].autoRange()
        self.statusBar().showMessage("Montage view reset.")

