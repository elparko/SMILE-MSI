"""ClassCompareMixin — compare lipid *classes* (not single ions) between two regions.

A 'Lipid classes…' button on the Stats tab opens a self-contained dialog that rolls the
per-ion identifications up to whole lipid classes (PE, PC, SM, Sulfatide, …) and answers two
questions about the same A/B selection at once:

* **Abundance (A vs B)** — which classes shift between the two regions (the exact rank-based
  AUC + signed log2 fold-change + test from :func:`smile_msi.spatial.class_comparison`), drawn as a
  diverging bar chart and a ranked table.
* **Composition** — how each region's annotated-lipid signal is split across classes
  (:func:`smile_msi.spatial.class_composition`), drawn as side-by-side stacked bars that sum
  to 100% and a per-class percentage table.

Results render inside the dialog (the m/z-keyed shared volcano/ROC widgets don't apply to
class rows), so the dialog stays open after Run.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .. import spatial, imaging
from .common import (REGION_A_COLOR, REGION_B_COLOR, colormap, fill_table,
                     set_header_tooltips, install_table_export, table_placeholder, note,
                     plot_caption, RegionMultiSelect, icon, GUIDE_LINE, NoScrollComboBox)


def _plot_pane(plot, caption_text):
    """Wrap a plot + a thin caption beneath it into one widget (so the caption rides under
    the plot in a splitter pane instead of grabbing its own resizable strip)."""
    w = QtWidgets.QWidget()
    v = QtWidgets.QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(2)
    v.addWidget(plot, 1)
    v.addWidget(plot_caption(caption_text))
    return w


def _class_color(i, total):
    """A distinct, stable colour for class ``i`` of ``total`` — evenly spaced hues so even a
    dozen-plus classes stay separable in the composition legend."""
    return pg.intColor(i, hues=max(9, int(total)), maxValue=210)


class ClassCompareMixin:
    # ----- build (eager + hidden, like the other stats dialogs) ------------- #
    def _make_cc_table(self, placeholder, stem, title):
        """A class-comparison result table: row-select, sortable, double-click a row to image
        that class, with the shared placeholder + export wiring."""
        t = QtWidgets.QTableWidget()
        t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        t.setSortingEnabled(True)
        t.setToolTip("Double-click a class to view its composite ion image "
                     "(all of the class's ions summed) in the Ion image tab.")
        t.cellDoubleClicked.connect(lambda r, _c: self._cc_class_image_for_row(t, r))
        table_placeholder(t, placeholder)
        install_table_export(t, self, stem=stem, title=title)
        return t

    def _build_class_compare_dialog(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Lipid class comparison — A vs B")
        dlg.resize(980, 640)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(note(
            "Roll the per-ion identifications up to whole lipid classes and compare two "
            "regions: which classes shift in abundance (A vs B), and how each region's signal "
            "is split across classes (composition). Tick one or more ROIs per side, pick the "
            "feature set, then Run. Double-click any class to see its composite ion image."))
        self.cc_region_a = RegionMultiSelect()
        self.cc_region_a.setMinimumWidth(220)
        self.cc_region_a.setToolTip("Region(s) for group A — their union is compared / profiled as A.")
        self.cc_region_b = RegionMultiSelect()
        self.cc_region_b.setMinimumWidth(220)
        self.cc_region_b.setToolTip("Region(s) for group B — their union is compared / profiled as B.")
        self.cc_region_a.currentIndexChanged.connect(lambda *_: self._update_class_run_enabled())
        self.cc_region_b.currentIndexChanged.connect(lambda *_: self._update_class_run_enabled())
        self.cc_feat = self._make_feature_combo()
        self.cc_method = NoScrollComboBox()
        self.cc_method.addItems(["mwu", "welch", "student"])
        self.cc_method.setToolTip("Per-class significance test (AUC is always the rank-based "
                                  "effect size): Mann-Whitney U (default), Welch's t, Student's t.")
        # Replication unit — the same pseudoreplication guard the Region-comparison tab uses.
        # Pixels within one ROI are spatially correlated, NOT independent replicates, so per-pixel
        # class p/q are overconfident. "Auto" summarizes each ticked ROI to one value and tests
        # across ROIs when both sides have ≥2 regions, else falls back to a descriptive pixel read.
        self.cc_unit_combo = NoScrollComboBox()
        self.cc_unit_combo.addItem("Auto (ROI replicates when available)", "auto")
        self.cc_unit_combo.addItem("ROI (each region = replicate)", "roi")
        self.cc_unit_combo.addItem("Pixel (descriptive only)", "pixel")
        self.cc_unit_combo.setToolTip(
            "How the class p/q are computed:\n"
            "• Auto: test across ROIs (each ticked region = one replicate) when both sides have "
            "≥2 regions, else a clearly-labelled descriptive per-pixel read.\n"
            "• ROI: always summarize each region to one replicate → tick ≥2 per side.\n"
            "• Pixel: per-pixel (descriptive only — pixels are pseudoreplicated, not a population "
            "inference).")
        form = QtWidgets.QFormLayout()
        form.addRow("Region A:", self.cc_region_a)
        form.addRow("vs Region B:", self.cc_region_b)
        form.addRow("Features:", self.cc_feat)
        form.addRow("Statistical test:", self.cc_method)
        form.addRow("Replicate unit:", self.cc_unit_combo)
        lay.addLayout(form)
        self.b_class_run = QtWidgets.QPushButton("Run class comparison")
        self.b_class_run.setObjectName("primaryAction")   # the dialog's single run action
        self.b_class_run.setIcon(icon("run"))
        self.b_class_run.clicked.connect(self._run_class_compare)
        self.b_class_save_list = QtWidgets.QPushButton("Save as lipid list…")
        self.b_class_save_list.setIcon(icon("save"))
        self.b_class_save_list.setToolTip("Save the compared classes as a ◆ lipid list — each "
                                          "class auto-coloured. Pick it in the Feature set "
                                          "selector to paint the classes on the tissue.")
        self.b_class_save_list.setEnabled(False)                # enabled once a comparison has run
        self.b_class_save_list.clicked.connect(self._save_cc_as_lipid_list)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.b_class_save_list)
        row.addStretch(1)
        row.addWidget(self.b_class_run)
        lay.addLayout(row)

        tabs = QtWidgets.QTabWidget()
        # -- Abundance (A vs B) -------------------------------------------------
        ab = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.cc_abund_table = self._make_cc_table(
            "Run a class comparison — each class's A-vs-B effect appears here.",
            "lipid_class_abundance", "Export class abundance")
        ab.addWidget(self.cc_abund_table)
        self.cc_abund_plot = pg.PlotWidget()
        self.cc_abund_plot.setLabel("bottom", "effect size  (2·(AUC−0.5))")
        self.cc_abund_plot.setMenuEnabled(False)
        ab.addWidget(_plot_pane(self.cc_abund_plot,
                                "Bars right = higher in B · left = higher in A · "
                                "dim grey = not significant (q > 0.05)."))
        ab.setSizes([480, 480])
        tabs.addTab(ab, "Abundance (A vs B)")
        # -- Composition --------------------------------------------------------
        co = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.cc_comp_table = self._make_cc_table(
            "Run a class comparison — each region's class composition (%) appears here.",
            "lipid_class_composition", "Export class composition")
        co.addWidget(self.cc_comp_table)
        self.cc_comp_plot = pg.PlotWidget()
        self.cc_comp_plot.setLabel("left", "% of annotated-lipid signal")
        self.cc_comp_plot.setMenuEnabled(False)
        co.addWidget(_plot_pane(self.cc_comp_plot,
                                "Each stacked bar is one region; a segment's height is that "
                                "class's share of the region's annotated-lipid signal."))
        co.setSizes([480, 480])
        tabs.addTab(co, "Composition")
        lay.addWidget(tabs, 1)
        self._class_tabs = tabs
        dlg.hide()
        self._class_dlg = dlg
        self._update_class_run_enabled()

    # ----- open / enable ---------------------------------------------------- #
    def _open_class_compare(self):
        self._refresh_feature_combo(self.cc_feat)          # pick up newly-saved ★ lists / scopes
        self._update_class_run_enabled()
        self._class_dlg.show()
        self._class_dlg.raise_()
        self._class_dlg.activateWindow()

    def _update_class_run_enabled(self):
        b = getattr(self, "b_class_run", None)
        if b is None:
            return
        ok = (getattr(self, "ds", None) is not None and bool(getattr(self, "peaks", None))
              and bool(self.cc_region_a.selected_names())
              and bool(self.cc_region_b.selected_names()))
        b.setEnabled(ok)

    def _run_class_compare(self):
        self.do_class_compare()                            # keep the dialog open — results land in it

    # ----- run -------------------------------------------------------------- #
    def do_class_compare(self):
        if self.ds is None or not self.peaks:
            self.statusBar().showMessage("Load data and find peaks first.")
            return
        mask_a, mask_b = self._resolve_regions(self.cc_region_a, self.cc_region_b)
        if mask_a is None:
            self.statusBar().showMessage(mask_b)           # mask_b carries the failure message
            return
        la, lb = self._region_labels
        mzs = self._combo_feature_mzs(getattr(self, "cc_feat", None))
        ann = getattr(self, "ann", None)
        classes = ann.classes_for(mzs) if ann is not None else [""] * len(mzs)
        if not any(classes):
            self.statusBar().showMessage("No lipids identified in this feature set — annotate "
                                         "peaks (or widen the identification tolerance) first.")
            return
        # remember which ions feed each class (from THIS feature set) so a double-clicked class
        # renders the same ions the comparison rolled up
        self._cc_class_mzs = {}
        for mz, c in zip(mzs, classes):
            if c:
                self._cc_class_mzs.setdefault(c, []).append(float(mz))
        method = self.cc_method.currentText()
        ppm, norm = self.ppm, self.norm
        # Resolve the replication unit so the class p/q aren't pseudoreplicated per-pixel —
        # mirrors the Region-comparison tab (each ticked ROI = one replicate).
        choice = self.cc_unit_combo.currentData() if hasattr(self, "cc_unit_combo") else "auto"
        rep_samples, n_a_reg, n_b_reg = self._resolve_region_replicates(
            self.cc_region_a, self.cc_region_b)
        if choice == "pixel":
            use_samples = None
        elif choice == "roi":
            use_samples = rep_samples
        else:                                             # auto
            use_samples = (rep_samples if (rep_samples is not None
                                           and n_a_reg >= 2 and n_b_reg >= 2) else None)

        def compute():
            cmp = spatial.class_comparison(self.ds, mask_a, mask_b, mzs, classes, tol_ppm=ppm,
                                           norm=norm, a_label=la, b_label=lb, method=method,
                                           samples=use_samples)
            comp = spatial.class_composition(self.ds, [mask_a, mask_b], mzs, classes,
                                             names=[la, lb], tol_ppm=ppm, norm=norm)
            return cmp, comp

        self._run(compute, on_done=self._on_class_compare, busy="Comparing lipid classes…")

    def _on_class_compare(self, payload):
        cmp_df, comp_df = payload
        self._last_class_cmp = cmp_df
        self._last_class_comp = comp_df
        self._fill_class_abundance(cmp_df)
        self._fill_class_composition(comp_df)
        self.b_class_save_list.setEnabled(bool(getattr(self, "_cc_class_mzs", None)))
        unit = cmp_df.attrs.get("unit", "pixel")
        if unit == "sample":
            unit_desc = f"ROI-replicate test (n={cmp_df.attrs.get('n_a', '?')} vs {cmp_df.attrs.get('n_b', '?')})"
        else:
            unit_desc = "per-pixel (descriptive; pixels pseudoreplicated, not inferential)"
        if self.prov is not None:
            self.prov.step("class_comparison",
                           test="lipid-class roll-up · rank AUC + MWU + BH-FDR",
                           unit=unit_desc, norm=self.norm, tol_ppm=self.ppm)
        la = cmp_df.attrs.get("a_label", "A")
        lb = cmp_df.attrs.get("b_label", "B")
        n_sig = int((cmp_df["q_value"].astype(float) <= 0.05).sum())
        warn = cmp_df.attrs.get("warning")
        msg = (f"Lipid classes: {len(cmp_df)} classes · {n_sig} differ (q ≤ 0.05) "
               f"between {la} and {lb} · {unit_desc}.")
        self.statusBar().showMessage(("⚠ " + warn + "  " + msg) if warn else msg)

    # ----- abundance render ------------------------------------------------- #
    def _fill_class_abundance(self, df):
        d = df.copy()
        d["effect"] = (d["AUC"].astype(float) - 0.5) * 2.0
        ranked = d.reindex(d["effect"].abs().sort_values(ascending=False).index)
        rows = []
        for _, x in ranked.iterrows():
            rows.append((str(x["class"]), str(int(x["n_ions"])), f"{x['AUC']:.3f}",
                         f"{x['effect']:+.2f}", f"{x['p_value']:.2e}", f"{x['q_value']:.2e}",
                         f"{x['log2_fc']:+.2f}", f"{x['mean_A']:.0f}", f"{x['mean_B']:.0f}"))
        self.cc_abund_table.setSortingEnabled(False)
        fill_table(self.cc_abund_table,
                   ["class", "ions", "AUC", "effect", "p", "q (FDR)", "log2 FC", "mean A", "mean B"],
                   rows)
        set_header_tooltips(self.cc_abund_table, {
            "ions": "How many identified ions were summed into this class.",
            "AUC": "Rank-based AUC of the class's summed signal (0.5 = no separation, "
                   ">0.5 = higher in B, <0.5 = higher in A).",
            "effect": "AUC on a signed −1…+1 axis: 2·(AUC−0.5).",
            "q (FDR)": "Benjamini-Hochberg FDR-corrected p across the classes; q ≤ 0.05 is the usual cutoff.",
            "log2 FC": "Signed log2 fold-change (B/A) of the two regions' mean class "
                       "intensities: + = higher in B, − = higher in A, 0 = no change.",
        })
        self.cc_abund_table.setSortingEnabled(True)
        self._render_class_abund_plot(d)

    def _render_class_abund_plot(self, d):
        plot = self.cc_abund_plot
        plot.clear()
        dd = d.sort_values("effect")                       # ascending → 'higher in A' at the bottom
        names = [str(c) for c in dd["class"]]
        effect = dd["effect"].to_numpy(dtype=float)
        sig = dd["q_value"].to_numpy(dtype=float) <= 0.05
        ys = np.arange(len(names))
        brushes = [pg.mkBrush(REGION_B_COLOR if e > 0 else REGION_A_COLOR) if s
                   else pg.mkBrush(180, 180, 180, 120) for e, s in zip(effect, sig)]
        x0 = np.minimum(0.0, effect)                       # draw each bar on its correct side of 0
        width = np.abs(effect)
        bar = pg.BarGraphItem(x0=x0, width=width, y=ys, height=0.62, brushes=brushes,
                              pen=pg.mkPen("#333", width=0.4))
        plot.addItem(bar)
        plot.addItem(pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(GUIDE_LINE)))
        plot.getAxis("left").setTicks([[(i, n) for i, n in enumerate(names)]])
        plot.setYRange(-0.6, len(names) - 0.4)
        span = float(np.max(width)) if width.size else 1.0
        span = max(span, 0.05)
        plot.setXRange(-1.08 * span, 1.08 * span)

    # ----- composition render ----------------------------------------------- #
    def _fill_class_composition(self, comp):
        cols = list(comp.columns)
        a_col, b_col = cols[0], cols[1]
        inten = comp.attrs.get("intensity")
        order = list(comp.sum(axis=1).sort_values(ascending=False).index)   # biggest share first
        rows = []
        for c in order:
            a = float(comp.loc[c, a_col])
            b = float(comp.loc[c, b_col])
            ia = float(inten.loc[c, a_col]) if inten is not None else 0.0
            ib = float(inten.loc[c, b_col]) if inten is not None else 0.0
            rows.append((str(c), f"{a:.1f}", f"{b:.1f}", f"{b - a:+.1f}", f"{ia:.0f}", f"{ib:.0f}"))
        self.cc_comp_table.setSortingEnabled(False)
        fill_table(self.cc_comp_table,
                   ["class", f"% {a_col}", f"% {b_col}", "Δ pp (B−A)",
                    f"mean {a_col}", f"mean {b_col}"], rows)
        set_header_tooltips(self.cc_comp_table, {
            "Δ pp (B−A)": "Change in the class's percentage share from A to B, in percentage points.",
        })
        self.cc_comp_table.setSortingEnabled(True)
        self._render_class_comp_plot(comp, order, a_col, b_col)

    def _render_class_comp_plot(self, comp, order, a_col, b_col):
        plot = self.cc_comp_plot
        self._clear_plot_legend(plot)
        plot.clear()
        legend = plot.addLegend(offset=(-8, 8))
        off_a = off_b = 0.0
        total = len(order)
        for i, c in enumerate(order):
            color = _class_color(i, total)
            a = float(comp.loc[c, a_col])
            b = float(comp.loc[c, b_col])
            bar = pg.BarGraphItem(x=[0, 1], width=0.6, height=[a, b], y0=[off_a, off_b],
                                  brush=pg.mkBrush(color), pen=pg.mkPen("#222", width=0.4))
            plot.addItem(bar)
            # a square scatter proxy gives the legend a clean colour swatch (BarGraphItem has
            # none). It's referenced only by the legend's sample — never added to the plot — so
            # it stays out of the bars; keep it 'visible' or pyqtgraph slashes the swatch as if
            # the series were hidden.
            proxy = pg.ScatterPlotItem([0], [0], symbol="s", size=11,
                                       brush=pg.mkBrush(color), pen=pg.mkPen("#222", width=0.4))
            legend.addItem(proxy, str(c))
            off_a += a
            off_b += b
        plot.getAxis("bottom").setTicks([[(0, str(a_col)), (1, str(b_col))]])
        plot.setXRange(-0.6, 1.6)
        plot.setYRange(0, 100)

    # ----- save the compared classes as a ◆ lipid list ---------------------- #
    def _save_cc_as_lipid_list(self):
        mzs_by_class = getattr(self, "_cc_class_mzs", None)
        if not mzs_by_class:
            self.statusBar().showMessage("Run a class comparison first.")
            return
        la = self._last_class_cmp.attrs.get("a_label", "") if getattr(self, "_last_class_cmp", None) is not None else ""
        lb = self._last_class_cmp.attrs.get("b_label", "") if getattr(self, "_last_class_cmp", None) is not None else ""
        suggested = f"{la} vs {lb} classes" if la and lb else None
        return self.save_lipid_list(mzs_by_class, suggested=suggested)

    # ----- "show this class as an ion image" -------------------------------- #
    def _cc_class_image_for_row(self, table, row):
        """Double-clicking a class row paints that class's composite ion image — every ion in
        the class summed — in the main Ion image view, exactly like the Features-tab 'Σ class'
        composite but driven straight from the comparison."""
        if row is None or row < 0:
            return
        it = table.item(row, 0)
        if it is not None:
            self._show_class_ion_image(it.text())

    def _show_class_ion_image(self, cls):
        mzs = (getattr(self, "_cc_class_mzs", {}) or {}).get(cls)
        if self.ds is None or not mzs:
            return
        img = imaging.quantile_clip(
            self.ds.composite_image(mzs, tol_ppm=self.ppm, reduce=self.reduce, norm=self.norm,
                                    weight=self.composite_weight),
            high=float(self.contrast_spin.value()))
        self.iv.setImage(img, autoLevels=True)
        self.iv.setColorMap(colormap(self.cmap_combo.currentText()))
        self._set_ion_tab_caption(f"total {cls} ({len(mzs)} ions)")
        self.reveal_view("Ion image")
        self.statusBar().showMessage(
            f"Composite image: total {cls} ({len(mzs)} ions) — the class painted like one ion.")

    def _clear_plot_legend(self, plot):
        """Drop a plot's legend before a re-render (PlotItem.clear() leaves it behind)."""
        pi = plot.getPlotItem()
        leg = getattr(pi, "legend", None)
        if leg is not None:
            try:
                leg.scene().removeItem(leg)
            except Exception:  # noqa: BLE001
                pass
            pi.legend = None
