"""AnalysisTabsMixin — extracted from the monolithic MainWindow (no behavior change).

The Co-localization tab was retired in plan 24 (its ranking/module engine now runs through
the registry into :mod:`smile_msi.spatial`). All that survives here is the focused
co-localization *overlay*, reached from the Features panel's "Co-localized…" button — a
lightweight alternative to the old tab (composite overlay + save-as-feature-list).
"""
from __future__ import annotations


import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .. import (spatial, imaging)
from .common import (MUTED_QSS, CheckList, colormap, dark_image_view, icon,
                     NoScrollComboBox, NoScrollSpinBox)


class AnalysisTabsMixin:
    # ----- co-localization overlay (from the Features panel's "Co-localized…") -- #
    def open_coloc_overlay(self, target_mz):
        """Find the ions co-localized with ``target_mz``, render them as a single
        composite overlay image, and offer to turn the selection into a feature list.
        A focused alternative to the full Co-localization tab (matrix + modules)."""
        if self.ds is None or not self.peaks:
            self.statusBar().showMessage("Find peaks first.")
            return
        target_mz = float(target_mz)
        old = getattr(self, "_coloc_dlg", None)
        if old is not None:
            old.close()

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Co-localized with m/z {target_mz:.4f}")
        dlg.resize(860, 580)
        self._coloc_dlg = dlg
        lay = QtWidgets.QVBoxLayout(dlg)

        ctl = QtWidgets.QHBoxLayout()
        lab = self.annotate(target_mz) or "(unannotated)"
        ctl.addWidget(QtWidgets.QLabel(f"<b>m/z {target_mz:.4f}</b> · {lab}"))
        ctl.addStretch(1)
        ctl.addWidget(QtWidgets.QLabel("Show top"))
        topn = NoScrollSpinBox()
        topn.setRange(1, max(1, len(self.peaks)))
        topn.setValue(min(12, len(self.peaks)))
        ctl.addWidget(topn)
        ctl.addWidget(QtWidgets.QLabel("Similarity"))
        method = NoScrollComboBox()
        method.addItems(["pearson", "cosine", "moc", "m1", "m2", "dice"])
        # the Co-localization tab was retired (plan 24); default the overlay's measure to
        # pearson when its combo is gone rather than depend on the tab widget.
        _cm = getattr(self, "coloc_method", None)
        method.setCurrentText(_cm.currentText() if _cm is not None else "pearson")
        ctl.addWidget(method)
        lay.addLayout(ctl)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        view = pg.GraphicsLayoutWidget()
        dark_image_view(view)            # coloc detail ion images read against black in both themes
        vb = view.addViewBox()
        vb.setAspectLocked(True)
        vb.invertY(True)
        img_item = pg.ImageItem()
        vb.addItem(img_item)
        split.addWidget(view)
        lst = CheckList(noun="ion")
        lst.setToolTip("Tick the ions to include in the overlay and the feature list")
        split.addWidget(lst)
        split.setSizes([520, 340])
        lay.addWidget(split, 1)

        btns = QtWidgets.QHBoxLayout()
        self._coloc_count = QtWidgets.QLabel()
        self._coloc_count.setStyleSheet(MUTED_QSS)
        btns.addWidget(self._coloc_count)
        btns.addStretch(1)
        b_make = QtWidgets.QPushButton("Save as feature list")
        b_make.setIcon(icon("save"))
        b_make.setToolTip("Save the ticked co-localized ions as a feature list — you'll be asked "
                          "to name it (pre-filled) — and load it as the working set")
        b_close = QtWidgets.QPushButton("Close")
        b_close.setIcon(icon("close"))
        btns.addWidget(b_make)
        btns.addWidget(b_close)
        lay.addLayout(btns)

        state = {"res": []}

        def ticked():
            return lst.checked_in_order()                  # score order → overlay/list order

        def render_overlay():
            mzs = ticked()
            self._coloc_count.setText(f"{len(mzs)} ion(s) in overlay")
            if not mzs:
                img_item.clear()
                return
            img = imaging.quantile_clip(
                self.ds.composite_image(mzs, tol_ppm=self.ppm, reduce=self.reduce, norm=self.norm,
                                        weight=self.composite_weight),
                high=float(self.contrast_spin.value()))
            img_item.setImage(img, autoLevels=True)
            img_item.setLookupTable(colormap(self.cmap_combo.currentText()).getLookupTable())
            # keep the user's pan/zoom as ions are ticked; only refit when the grid changes
            if getattr(vb, "_iv_img_shape", None) != img.shape:
                vb._iv_img_shape = img.shape
                vb.autoRange()

        def populate(res):
            state["res"] = res
            entries = []
            for r in res[:int(topn.value())]:
                mz, score = float(r["mz"]), float(r["score"])
                name = self.annotate(mz) or "—"
                tag = "  ← target" if abs(mz - target_mz) < 1e-3 else ""
                entries.append((mz, f"m/z {mz:.4f}   {name}   (r={score:+.2f}){tag}"))
            lst.set_entries(entries)
            lst.set_checked([k for k, _ in entries])       # a fresh result starts fully ticked
            render_overlay()

        def recompute():
            mzs_all = self._visible_mzs()
            m = method.currentText()

            def job():
                return spatial.colocalize(self.ds, target_mz, mzs_all,
                                          tol_ppm=self.ppm, norm=self.norm, method=m)
            self._run(job, on_done=populate, busy="Finding co-localized ions…")

        def make_list():
            mzs = ticked()
            if not mzs:
                self.statusBar().showMessage("Tick at least one ion first.")
                return
            feats = [{"mz": float(m), "lipid": self.annotate(m) or "",
                      "note": f"co-localized with m/z {target_mz:.4f}"} for m in mzs]
            name = self._save_feature_list_to_library(f"{target_mz:.3f} co-localized peaks",
                                                      feats, activate=True, prompt=True)
            if not name:                                  # user cancelled the name prompt
                return
            self.record_undo("co-localized list")
            ctx = self._mean_spec_ctx()                   # shared mean spectrum for all ions
            self._set_peaks([self._peak_from_mz(m, ctx) for m in mzs],
                            msg=(f"Saved feature list '{name}' ({len(mzs)} ions co-localized "
                                 f"with m/z {target_mz:.4f})."))
            dlg.accept()

        topn.valueChanged.connect(lambda *_: populate(state["res"]))
        method.currentTextChanged.connect(lambda *_: recompute())
        lst.changed.connect(render_overlay)
        b_make.clicked.connect(make_list)
        b_close.clicked.connect(dlg.reject)

        recompute()
        dlg.show()
