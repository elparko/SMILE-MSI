"""Report tab — curate the analyses that go into the PDF data book.

The data book used to be assembled blindly from whatever was live on screen at export time
(active ion + the first 8 visible features, the one current segmentation, the single most
recent comparison). This tab gives the report a **curated, ordered, editable list** instead:
you collect analyses from the live state ("Add current…"), give each a title + caption,
reorder / delete them, and the PDF builds straight from the list.

Each report item is a plain JSON-serializable dict so it persists in the per-sample session
(``session.build_session(report_items=…)``) and feeds the exporter unchanged. Figure items
store *parameters* and are re-rendered crisply at export (mirroring ``_panel_kwargs``);
ephemeral results (segmentation / statistics) are *snapshotted* so they survive later edits.

``report_document_items(opts)`` turns the stored items into the ordered ``items`` list that
:func:`smile_msi.export.build_book` consumes (rebuilding the heavy arrays — ion images, the
overlay composite, spectra — from the stored params).
"""
from __future__ import annotations

import json
import os

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .. import export
from .common import (ControlBar, confirm, fig_to_pixmap, glossary_button, icon, note,
                     tab_page, button, MUTED_QSS, MUTED_FG, NoScrollComboBox)
from . import filedialogs

_BADGE = {"ion": "🖼 Ion", "overlay": "🎨 Overlay", "spectrum": "📈 Spectrum",
          "segmentation": "🗺 Segmentation", "stats": "📊 Stats", "features": "📋 Features",
          "analysis": "📊 Analysis", "note": "📝 Note"}

# Where each item type lands when "Export all → files" writes loose files — one folder per
# kind of analysis (analysis-result tables get a folder named after the analysis itself).
_EXPORT_FOLDERS = {"ion": "ion_images", "overlay": "overlays", "spectrum": "spectra",
                   "segmentation": "segmentation", "features": "feature_tables",
                   "stats": "statistics", "note": "notes"}


class ReportTabMixin:
    """Lives on ``MainWindow``. Owns ``self.report_items`` (a list of plain dicts) and the
    Report tab UI; the export hub pulls the curated list via :meth:`report_document_items`."""

    # ----- tab construction ------------------------------------------------ #
    def _tab_report(self):
        w, v = tab_page()
        bar = ControlBar()
        self._report_bar = bar
        add_btn = QtWidgets.QToolButton()
        add_btn.setText("Add current")                    # ▾ drawn by QSS (menuButton chevron)
        add_btn.setIcon(icon("add"))
        add_btn.setObjectName("menuButton")               # shared dropdown pill (single arrow)
        add_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        add_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        add_btn.setToolTip("Snapshot an analysis from the current view into the report.")
        self._report_add_menu = QtWidgets.QMenu(add_btn)
        self._report_add_menu.aboutToShow.connect(self._populate_report_add_menu)
        add_btn.setMenu(self._report_add_menu)
        note_btn = button("Add note / heading…", self._report_add_note, icon="add")
        export_btn = button("Export report (PDF)…", lambda: self.open_export_hub("book"),
                            icon="export",
                            tooltip="Open the export hub on the data book, built from this list.")
        self._report_export_btn = export_btn
        files_btn = button("Export all → files…", self._report_export_all_files, icon="export",
                           tooltip="Render every item in this list to an individual image / CSV "
                                   "in a folder you choose — the same artifacts the PDF book "
                                   "bundles, as loose files.")
        self._report_files_btn = files_btn
        studio_btn = button("Export Studio…", self.open_export_studio, icon="export",
                            tooltip="Batch-export ion images across several feature lists and "
                                    "sections at once, plus their feature-list CSVs and analyses.")
        self._report_studio_btn = studio_btn
        clear_btn = button("Clear report", self._report_clear, icon="delete", danger=True,
                           tooltip="Empty the report list (the logged images / tables / notes). "
                                   "Leaves your regions, segmentation, statistics and features "
                                   "as they are.")
        self._report_clear_btn = clear_btn
        wipe_btn = button("Wipe && restart analysis", self.wipe_analysis, icon="delete",
                          danger=True,
                          tooltip="Reset the whole analysis — report, regions, segmentation, "
                                  "statistics and picked features — back to the freshly-loaded "
                                  "dataset. The dataset, optical overlay and saved ★ lists are "
                                  "kept.")
        name_chk = QtWidgets.QCheckBox("Ask to name on create")
        name_chk.setToolTip("Prompt for a name each time you capture an analysis into the report.")
        from .. import prefs as _prefs
        name_chk.setChecked(bool(_prefs.get("report_ask_name_on_create", True)))
        name_chk.toggled.connect(self._set_report_ask_name)
        bar.add(add_btn, note_btn, export_btn, files_btn, studio_btn, clear_btn, wipe_btn, name_chk, glossary_button([
            ["Data book", "The multi-page PDF this list builds: cover, dataset summary, methods, ion images, spectra, segmentation, statistics."],
            ["· auto", "An item logged automatically from an export (⌘E) or CSV analysis, vs one you added by hand with 'Add current…'."],
            ["Provenance", "The recorded context behind an item — dataset, regions/pixels, feature list, normalization and extraction tolerance."],
            ["Data bundle", "A folder of CSV/Markdown files written beside the PDF holding the report's underlying numbers."],
            ["Normalization", "The per-pixel scaling (TIC / RMS / median / none) the logged item was computed under."],
        ], parent=self))
        v.addWidget(bar)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        # left — the ordered item list + row actions
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        # browse controls: search across titles/types/provenance + group the list
        filt = QtWidgets.QHBoxLayout()
        self.report_search = QtWidgets.QLineEdit()
        self.report_search.setPlaceholderText("Search analyses…")
        self.report_search.setClearButtonEnabled(True)
        self.report_search.textChanged.connect(lambda *_: self._refresh_report_list())
        self.report_groupby = NoScrollComboBox()
        self.report_groupby.addItems(["No grouping", "Type", "Analysis kind", "Dataset",
                                      "Date logged"])
        from .. import prefs as _gp
        _gi = self.report_groupby.findText(_gp.get("report_group_by", "No grouping"))
        if _gi >= 0:
            self.report_groupby.setCurrentIndex(_gi)
        self.report_groupby.currentIndexChanged.connect(self._on_report_groupby)
        filt.addWidget(self.report_search, 1)
        filt.addWidget(QtWidgets.QLabel("Group:"))
        filt.addWidget(self.report_groupby)
        lv.addLayout(filt)
        self.report_list = QtWidgets.QListWidget()
        self.report_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.report_list.setUniformItemSizes(True)
        self.report_list.currentRowChanged.connect(self._report_preview_current)
        self.report_list.itemDoubleClicked.connect(lambda *_: self._report_edit())
        self.report_list.model().rowsMoved.connect(self._report_sync_order)
        lv.addWidget(self.report_list, 1)
        rb = QtWidgets.QHBoxLayout()
        for txt, slot, tip, ic in [("↑", lambda: self._report_move(-1), "Move up", "up"),
                                   ("↓", lambda: self._report_move(1), "Move down", "down"),
                                   ("Edit…", self._report_edit, "Edit title / caption", "settings"),
                                   ("Delete", self._report_delete, "Remove from report", "delete")]:
            b = QtWidgets.QPushButton(txt)
            b.setIcon(icon(ic))
            if txt in ("↑", "↓"):                      # symbol-only → icon-only button
                b.setText("")
                b.setAccessibleName(tip)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            rb.addWidget(b)
        rb.addStretch(1)
        lv.addLayout(rb)
        split.addWidget(left)

        # right — preview / detail of the selected item, with a provenance block beneath
        # ("where did this come from": the regions + feature list + dataset that fed it).
        pv_inner = QtWidgets.QWidget()
        pvl = QtWidgets.QVBoxLayout(pv_inner)
        pvl.setContentsMargins(6, 6, 6, 6)
        self.report_preview = QtWidgets.QLabel("Add analyses on the left, then preview them here.")
        self.report_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.report_preview.setWordWrap(True)
        self.report_preview.setMinimumWidth(360)
        self.report_preview.setStyleSheet(MUTED_QSS)
        pvl.addWidget(self.report_preview)
        # table items (stats / features / analysis) preview their actual data here — the
        # first rows of the persisted records, so you read the numbers without re-running.
        self.report_table_preview = QtWidgets.QTableWidget()
        self.report_table_preview.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.report_table_preview.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.report_table_preview.verticalHeader().setVisible(False)
        self.report_table_preview.horizontalHeader().setStretchLastSection(True)
        self.report_table_preview.setMaximumHeight(260)
        self.report_table_preview.hide()
        pvl.addWidget(self.report_table_preview)
        self.report_source = QtWidgets.QLabel("")
        self.report_source.setWordWrap(True)
        self.report_source.setTextFormat(QtCore.Qt.RichText)
        self.report_source.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.report_source.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.report_source.setStyleSheet(
            f"{MUTED_QSS} border-top:1px solid rgba(128,128,128,0.25); padding-top:6px;")
        pvl.addWidget(self.report_source)
        # reveal where the source dataset was loaded from (full path lives in item.source)
        self.report_open_folder_btn = QtWidgets.QPushButton("Open data folder")
        self.report_open_folder_btn.setToolTip(
            "Reveal the folder the source dataset for this item was loaded from.")
        self.report_open_folder_btn.clicked.connect(self._report_open_data_folder)
        self.report_open_folder_btn.setEnabled(False)
        pvl.addWidget(self.report_open_folder_btn, 0, QtCore.Qt.AlignLeft)
        pvl.addStretch(1)
        pv_scroll = QtWidgets.QScrollArea()
        pv_scroll.setWidgetResizable(True)
        pv_scroll.setWidget(pv_inner)
        split.addWidget(pv_scroll)
        split.setSizes([430, 540])
        v.addWidget(split, 1)
        v.addWidget(note("Every analysis you run and everything you export (⌘E) is logged here "
                         "automatically (marked “· auto”); items you name yourself are marked “✎”. "
                         "Search and group the list to find an analysis, then select it to preview "
                         "its data and see the method, settings, regions and data path that fed it "
                         "(“Open data folder” reveals where it came from). In “No grouping” the order "
                         "is the export order — drag to reorder, edit titles, or delete. “Export "
                         "report (PDF)” bundles them into one book; “Export all → files” writes each "
                         "into a per-type folder with a report_log.csv / README.md."))
        self._report_tab_page = w
        self.tabs.addTab(w, "Report book")           # a sub-tab of the Analyses group (plan 24)
        self._refresh_report_list()

    def _show_report_tab(self):
        """Bring the Report tab to the front (File → Report builder)."""
        page = getattr(self, "_report_tab_page", None)
        if page is not None:
            self.tabs.setCurrentWidget(page)

    # ----- the "Add current…" menu ----------------------------------------- #
    def _populate_report_add_menu(self):
        m = self._report_add_menu
        m.clear()
        has_ds = self.ds is not None
        a = m.addAction("Ion image (active m/z)", self._report_add_ion)
        a.setEnabled(has_ds and getattr(self, "active_mz", None) is not None)
        a = m.addAction("Colour overlay (visible features)", self._report_add_overlay)
        a.setEnabled(has_ds and bool(self._overlay_peaks()))
        a = m.addAction("Mean spectrum", self._report_add_spectrum_mean)
        a.setEnabled(has_ds)
        regions = [r for r in (getattr(self, "regions", None) or []) if r.get("visible", True)]
        if has_ds and regions:
            sub = m.addMenu("Region spectrum")
            for rg in regions:
                sub.addAction(rg["name"],
                              lambda _=False, n=rg["name"]: self._report_add_spectrum_region(n))
        a = m.addAction("Segmentation map", self._report_add_segmentation)
        a.setEnabled(getattr(self, "seg", None) is not None)
        a = m.addAction("Statistics comparison (last run)", self._report_add_stats)
        a.setEnabled(getattr(self, "last_stats", None) is not None)
        a = m.addAction("Feature table", self._report_add_features)
        a.setEnabled(has_ds and (getattr(self, "feat_df", None) is not None or bool(self.peaks)))

    # ----- provenance / "where did this come from" ------------------------- #
    # Every captured item carries a ``source`` dict recording the live analysis context
    # (dataset, normalization, extraction tolerance, the active feature list, and a
    # timestamp). The *regions* and *features* that actually fed an item are derived from
    # the item's own fields at display/export time (see :meth:`_item_regions_text` /
    # :meth:`_item_features_text`) so a crop or label set after capture stays in sync.
    def _active_feature_list_name(self):
        """Human label for the feature list currently driving the working set."""
        name = getattr(self, "_active_feature_scope", None) or getattr(self, "_flist_name", None)
        return name or "working set"

    def _report_context(self):
        """The immutable when/where context stamped on every captured report item."""
        from datetime import datetime
        ds = getattr(self, "ds", None)
        full = (getattr(ds, "source", "") or "") if ds is not None else ""
        src = os.path.basename(full)
        return {"dataset": src or "—",
                "dataset_path": full,            # full source path (the basename above loses where)
                "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "norm": getattr(self, "norm", None) or "none",
                "ppm": float(getattr(self, "ppm", 0) or 0),
                "feature_list": self._active_feature_list_name()}

    def _item_regions_text(self, it):
        """Plain-language description of which pixels/regions fed an item."""
        t = (it.get("type") or "").lower()
        if t in ("ion", "overlay"):
            cr = it.get("crop_region")
            base = f"Cropped to “{cr}”" if cr else "Full image — every pixel (no ROI)"
            if t == "overlay" and it.get("outline"):
                base += " · region outlines drawn"
            return base
        if t == "spectrum":
            return it.get("region") or "Whole slide — mean spectrum (every pixel)"
        if t == "stats":
            return f"{it.get('a_label', 'Region A')}  vs  {it.get('b_label', 'Region B')}"
        return (it.get("source") or {}).get("regions") or ("Whole slide" if t == "segmentation" else "")

    def _item_features_text(self, it):
        """Plain-language description of the feature list / ions that fed an item."""
        t = (it.get("type") or "").lower()
        src = it.get("source") or {}
        if t == "ion":
            mz = it.get("mz")
            if mz is None:
                return ""
            lab = it.get("label")
            return f"m/z {float(mz):.4f}" + (f" · {lab}" if lab else "")
        if t == "overlay":
            chs = it.get("channels") or []
            shown = ", ".join(f"{float(c.get('mz', 0)):.4f}" for c in chs[:6])
            return f"{len(chs)} ions ({shown}{'…' if len(chs) > 6 else ''})" if chs else ""
        if t in ("stats", "features", "analysis"):
            n = len(it.get("records") or [])
            fl = src.get("feature_list")
            return f"{n} features" + (f" · {fl}" if fl else "")
        if t == "segmentation":
            return src.get("features") or "all detected features"
        if t == "spectrum":
            return "full spectrum (all m/z)"
        return ""

    def _report_source_pairs(self, it):
        """Ordered (label, value) provenance rows for an item — drives the preview block,
        the row tooltip, and the export log. When an analysis recorded an audit block
        (:meth:`AuditMixin.record_step`), its method label, full settings string, and the
        ROIs-with-pixel-counts are surfaced here as a "Methods & data sources" block."""
        src = it.get("source") or {}
        pairs = []
        if src.get("method"):
            pairs.append(("Method", src["method"]))
        # the audit ROI string (names + pixel counts + how-made) is richer than the
        # item-derived text, so prefer it when an analysis recorded one
        reg = src.get("regions") or self._item_regions_text(it)
        if reg:
            pairs.append(("Regions / pixels", reg))
        feat = self._item_features_text(it)
        if feat:
            pairs.append(("Feature list", feat))
        if src.get("params"):
            pairs.append(("Settings", src["params"]))
        if src.get("dataset"):
            pairs.append(("Dataset", src["dataset"]))
        if src.get("dataset_path") and src["dataset_path"] not in ("", src.get("dataset")):
            pairs.append(("Data path", src["dataset_path"]))
        if src.get("norm"):
            pairs.append(("Normalization", str(src["norm"])))
        if src.get("ppm"):
            pairs.append(("Extraction tol.", f"±{src['ppm']:g} ppm"))
        if src.get("created"):
            pairs.append(("Logged", src["created"]))
        return pairs

    @staticmethod
    def _esc(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _report_source_html(self, it):
        pairs = self._report_source_pairs(it)
        if not pairs:
            return ""
        rows = "".join(
            f"<tr><td style='color:{MUTED_FG};padding-right:12px;vertical-align:top;"
            f"white-space:nowrap;'>{self._esc(k)}</td><td>{self._esc(v)}</td></tr>"
            for k, v in pairs)
        return ("<div style='font-size:11px;'><b>Source / provenance</b>"
                f"<table style='margin-top:4px;'>{rows}</table></div>")

    def _report_source_tooltip(self, it):
        pairs = self._report_source_pairs(it)
        head = it.get("title") or it.get("type") or ""
        body = "\n".join(f"{k}: {v}" for k, v in pairs)
        return f"{head}\n{body}".strip() if pairs else head

    # ----- item builders (snapshot from live state) ------------------------ #
    # Each ``_capture_*`` returns a plain item dict (or None when there's nothing to
    # capture) *without* touching the UI, so both the manual "Add current…" actions and
    # the automatic export log (:meth:`_log_export_to_report`) can reuse them. Every item
    # carries a ``source`` block (:meth:`_report_context`) so the report records exactly
    # what fed it.
    def _capture_ion_item(self, p=None):
        """Snapshot an ion-image item for peak ``p`` (defaults to the active m/z)."""
        if p is None:
            if getattr(self, "active_mz", None) is None:
                return None
            p = self._peak_for_mz(self.active_mz) or {"mz": self.active_mz}
        mz = float(p["mz"])
        if "lo" in p or "hi" in p:
            lo, hi = self._feature_window(p)
        else:
            lo, hi = 0.0, float(self.contrast_spin.value())
        label = self._clean_label(mz)
        cmap = self.cmap_combo.currentText() if getattr(self, "cmap_combo", None) else "viridis"
        title = f"m/z {mz:.4f}" + (f" · {label}" if label else "")
        return {"type": "ion", "title": title, "caption": "", "mz": mz, "ppm": float(self.ppm),
                "window": [float(lo), float(hi)], "cmap": cmap, "label": label,
                "crop_region": None, "source": self._report_context()}

    def _capture_overlay_item(self):
        peaks = self._overlay_peaks() or self._visible_peaks()
        if not peaks:
            return None
        channels = []
        for p, hexc in self._overlay_channels(peaks):
            lo, hi = self._feature_window(p)
            channels.append({"mz": float(p["mz"]), "color": hexc, "ppm": float(self.ppm),
                             "lo": float(lo), "hi": float(hi), "label": self._clean_label(p["mz"])})
        return {"type": "overlay", "title": "Colour overlay", "caption": "",
                "channels": channels, "crop_region": None, "outline": False,
                "source": self._report_context()}

    def _capture_spectrum_item(self, name=None):
        if self.ds is None:
            return None
        title = f"{name} spectrum" if name else "Mean spectrum"
        return {"type": "spectrum", "title": title, "caption": "", "region": name,
                "source": self._report_context()}

    def _capture_segmentation_item(self):
        seg = getattr(self, "seg", None)
        if seg is None:
            return None
        colors, legend = self._seg_colors_legend()
        src = self._report_context()
        src["regions"] = "Whole slide"
        src["features"] = f"all detected features → {int(seg.n_clusters)} clusters"
        return {"type": "segmentation", "title": "Segmentation", "caption": "",
                "labels": [int(x) for x in np.asarray(seg.labels).tolist()],
                "n_clusters": int(seg.n_clusters),
                "colors": {str(int(k)): v for k, v in colors.items()},
                "legend": [[c, str(n)] for c, n in legend], "source": src}

    def _capture_stats_item(self):
        if getattr(self, "last_stats", None) is None:
            return None
        la, lb = getattr(self, "_region_labels", ("Group A", "Group B"))
        records = json.loads(self.last_stats.to_json(orient="records"))
        src = self._report_context()
        src["regions"] = f"{la} vs {lb}"
        return {"type": "stats", "title": f"{la} vs {lb}", "caption": "",
                "records": records, "a_label": la, "b_label": lb, "source": src}

    def _capture_features_item(self):
        df = getattr(self, "feat_df", None)
        if df is None:
            df = self._fallback_feature_df()
        if df is None or len(df) == 0:
            return None
        return {"type": "features", "title": "Feature table", "caption": "",
                "records": json.loads(df.to_json(orient="records")),
                "source": self._report_context()}

    def _report_add_ion(self):
        it = self._capture_ion_item()
        if it is None:
            self.statusBar().showMessage("Select a feature (active m/z) first.")
            return
        self._report_append(it)

    def _report_add_overlay(self):
        it = self._capture_overlay_item()
        if it is None:
            self.statusBar().showMessage("No visible features to overlay.")
            return
        self._report_append(it)

    def _report_add_spectrum_mean(self):
        it = self._capture_spectrum_item()
        if it is not None:
            self._report_append(it)

    def _report_add_spectrum_region(self, name):
        it = self._capture_spectrum_item(name)
        if it is not None:
            self._report_append(it)

    def _report_add_segmentation(self):
        it = self._capture_segmentation_item()
        if it is None:
            self.statusBar().showMessage("Run segmentation first.")
            return
        self._report_append(it)

    def _report_add_stats(self):
        it = self._capture_stats_item()
        if it is None:
            self.statusBar().showMessage("Run a region comparison first (Statistics tab).")
            return
        self._report_append(it)

    def _report_add_features(self):
        it = self._capture_features_item()
        if it is None:
            self.statusBar().showMessage("No features to add — find peaks first.")
            return
        self._report_append(it)

    # ----- automatic export log -------------------------------------------- #
    # Every successful export (⌘E) calls :meth:`_log_export_to_report`, which mirrors the
    # exported artifact into this list so the Report tab accumulates a running record of
    # everything produced. Items are de-duplicated by :meth:`_report_signature` (re-exporting
    # the same view doesn't pile up copies) and tagged ``auto`` so the row shows "· auto".
    def _report_signature(self, it):
        """A stable identity for an item so equivalent captures collapse to one row."""
        t = (it.get("type") or "").lower()
        if t == "ion":
            w = tuple(round(float(x), 3) for x in (it.get("window") or []))
            return ("ion", round(float(it.get("mz", 0)), 4), w, it.get("cmap"),
                    it.get("crop_region"))
        if t == "overlay":
            chs = tuple(sorted(round(float(c.get("mz", 0)), 4) for c in (it.get("channels") or [])))
            return ("overlay", chs, it.get("crop_region"))
        if t == "spectrum":
            return ("spectrum", it.get("region"))
        if t == "segmentation":
            return ("segmentation", it.get("n_clusters"))
        if t == "stats":
            return ("stats", it.get("a_label"), it.get("b_label"), len(it.get("records") or []))
        if t == "features":
            # key on the list's name + its m/z set so distinct named lists stay separate
            # (re-saving the same list still collapses to one row).
            mzs = tuple(sorted(round(float(r["mz"]), 4) for r in (it.get("records") or [])
                               if isinstance(r, dict) and r.get("mz") is not None))
            return ("features", it.get("title") or it.get("list_name") or "", mzs)
        if t == "analysis":
            return ("analysis", it.get("analysis_kind"), it.get("source", {}).get("regions"),
                    len(it.get("records") or []))
        return (t, it.get("title"))

    def _log_export_to_report(self, scope, opts):
        """Append a Report-tab item mirroring the export just performed for ``scope``.

        Best-effort and quiet: it never changes the current selection or overrides the
        export's own status message, and any failure is swallowed (logging must never break
        an export). Returns the number of items added."""
        if getattr(self, "report_list", None) is None:    # Report tab not built yet
            return 0
        crop = opts.get("crop_region")
        single_crop = crop if (crop and crop != "__each__") else None
        items = []
        try:
            if scope == "ion":
                it = self._capture_ion_item()
                if it:
                    it["crop_region"] = single_crop
                    items.append(it)
            elif scope == "overlay":
                it = self._capture_overlay_item()
                if it:
                    it["crop_region"] = single_crop
                    it["outline"] = bool(opts.get("roi_outline"))
                    items.append(it)
            elif scope == "gallery":
                for p in (self._visible_peaks() or []):
                    it = self._capture_ion_item(p)
                    if it:
                        items.append(it)
            elif scope in ("specfig", "specdata"):
                it = self._capture_spectrum_item()
                if it:
                    items.append(it)
            elif scope == "features":
                it = self._capture_features_item()
                if it:
                    items.append(it)
            elif scope == "stats":
                it = self._capture_stats_item()
                if it:
                    items.append(it)
            elif scope == "seg":
                it = self._capture_segmentation_item()
                if it:
                    items.append(it)
        except Exception:  # noqa: BLE001 — logging is best-effort, never fatal to an export
            import traceback
            traceback.print_exc()
            return 0
        return self._append_logged_items(items)

    def _append_logged_items(self, items):
        """Append ``items`` (each tagged ``auto``), skipping any whose signature already
        appears in the list. Refreshes the list quietly (no selection/status change)."""
        existing = {self._report_signature(it) for it in self.report_items}
        added = 0
        for it in items:
            if not it:
                continue
            it = {**it, "auto": True}
            sig = self._report_signature(it)
            if sig in existing:
                continue
            existing.add(sig)
            self.report_items.append(it)
            self._report_maybe_write_thumb(it)
            added += 1
        if added:
            self._refresh_report_list()
            self._mark_dirty()
            self.statusBar().showMessage(f"Logged to Report ({added} item{'s' if added != 1 else ''}) — see the Report tab.", 4000)
        return added

    # ----- generic analysis + named feature-list logging ------------------- #
    # Two thin wrappers over :meth:`_append_logged_items` so *every* analysis the user runs,
    # and every named feature list they save out of one, lands in the Report tab — auto-tagged
    # ("· auto") and de-duplicated exactly like the ⌘E export log. Both are best-effort: they
    # return 0 (and never raise) when the Report tab isn't built yet or there's nothing to log,
    # so a logging slip can never break the analysis it mirrors.
    def _log_analysis_to_report(self, kind, df, *, regions=None, title=None, caption=None,
                                source_extra=None):
        """Append a CSV-analysis result as a generic ``analysis`` table item. ``df`` is a
        pandas DataFrame (serialized to JSON records) or an already-built list of row dicts.
        ``kind`` names the analysis (drives de-dup + the per-type export folder)."""
        if getattr(self, "report_list", None) is None or df is None:
            return 0
        try:
            if hasattr(df, "to_json"):
                if not len(df):
                    return 0
                records = json.loads(df.to_json(orient="records"))
            else:
                records = [dict(r) for r in (df or [])]
            if not records:
                return 0
        except Exception:  # noqa: BLE001 — logging never breaks the analysis it mirrors
            import traceback
            traceback.print_exc()
            return 0
        src = self._report_context()
        if regions:
            src["regions"] = regions
        if source_extra:
            src.update(source_extra)
        item = {"type": "analysis", "analysis_kind": kind, "title": title or kind,
                "caption": caption or "", "records": records, "source": src}
        return self._append_logged_items([item])

    def _log_feature_list_to_report(self, name, features, *, caption="", source_extra=None):
        """Append a NAMED feature list (saved from an analysis or by hand) as a ``features``
        table item titled with the list's name — so "export a feature list from an analysis"
        leaves a named, ordered table in the report. De-duplicated by name + m/z set."""
        if getattr(self, "report_list", None) is None or not name:
            return 0
        try:
            feats = (self._norm_feature_list(features)
                     if hasattr(self, "_norm_feature_list") else list(features or []))
        except Exception:  # noqa: BLE001
            feats = list(features or [])
        records = []
        for f in feats:
            if isinstance(f, dict):
                records.append({"mz": f.get("mz"), "lipid": f.get("lipid", ""),
                                "note": f.get("note", "")})
            else:
                try:
                    records.append({"mz": float(f), "lipid": "", "note": ""})
                except (TypeError, ValueError):
                    continue
        if not records:
            return 0
        src = self._report_context()
        src["feature_list"] = name
        if source_extra:
            src.update(source_extra)
        item = {"type": "features", "title": name, "caption": caption,
                "records": records, "list_name": name, "source": src}
        return self._append_logged_items([item])

    def _report_add_note(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Add note / heading")
        form = QtWidgets.QFormLayout(dlg)
        heading = QtWidgets.QLineEdit()
        heading.setPlaceholderText("Section heading")
        body = QtWidgets.QPlainTextEdit()
        body.setPlaceholderText("Optional body text…")
        body.setFixedHeight(120)
        form.addRow("Heading", heading)
        form.addRow("Body", body)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)
        if dlg.exec() == QtWidgets.QDialog.Accepted and heading.text().strip():
            self._report_append({"type": "note", "title": heading.text().strip(),
                                 "heading": heading.text().strip(),
                                 "text": body.toPlainText().strip(), "caption": ""})

    # ----- list mutation + display ----------------------------------------- #
    def _report_append(self, item):
        if not item:
            return
        # name-at-creation: notes already carry a user-typed heading; everything else is
        # prompted (prefilled with the smart auto-title) so each analysis gets a meaningful name.
        if item.get("type") == "note":
            item.setdefault("name_origin", "user")
        else:
            name, origin = self._prompt_item_name(item.get("title") or item.get("type"),
                                                  item.get("type"))
            if name is None:                         # user cancelled → abort the capture
                return
            item["title"], item["name_origin"] = name, origin
        self.report_items.append(item)
        self._report_maybe_write_thumb(item)
        self._refresh_report_list()
        self._report_select_index(len(self.report_items) - 1)   # group/filter-safe selection
        self._mark_dirty()
        self.statusBar().showMessage(f"Added to report: {item.get('title') or item.get('type')}")

    def _set_report_ask_name(self, on):
        from .. import prefs
        prefs.set("report_ask_name_on_create", bool(on))

    def _prompt_item_name(self, default, kind):
        """Prompt to name a newly-captured item, prefilled with the smart default. Returns
        ``(name, origin)``; ``name`` is None when cancelled. Skips the dialog (returns the
        default, tagged ``auto``) when naming is turned off in prefs or in a headless/test
        context, so it never blocks an automated run."""
        from .. import prefs
        default = str(default or kind or "Item")
        app = QtWidgets.QApplication.instance()
        headless = bool(os.environ.get("PYTEST_CURRENT_TEST")) or (
            app is not None and app.platformName() == "offscreen")
        if headless or not prefs.get("report_ask_name_on_create", True):
            return default, "auto"
        text, ok = QtWidgets.QInputDialog.getText(
            self, f"Name this {kind or 'item'}", "Name:",
            QtWidgets.QLineEdit.Normal, default)
        if not ok:
            return None, "auto"
        text = text.strip()
        return (text, "user") if text else (default, "auto")

    def _on_report_groupby(self, *_):
        from .. import prefs
        prefs.set("report_group_by", self.report_groupby.currentText())
        self._refresh_report_list()

    def _report_item_matches(self, it, query):
        """True when the search ``query`` (lower-case) appears in an item's title, type, or
        provenance block — the haystack the browser filters on."""
        hay = " ".join(str(x) for x in (
            it.get("title") or "", it.get("type") or "", it.get("analysis_kind") or "",
            _BADGE.get(it.get("type"), ""), self._report_source_tooltip(it))).lower()
        return query in hay

    def _report_group_key(self, it, mode):
        src = it.get("source") or {}
        if mode == "Type":
            return _BADGE.get(it.get("type"), it.get("type") or "Other")
        if mode == "Analysis kind":
            return it.get("analysis_kind") or _BADGE.get(it.get("type"), "Other")
        if mode == "Dataset":
            return src.get("dataset") or "—"
        if mode == "Date logged":
            return (src.get("created") or "")[:10] or "—"
        return "Other"

    def _report_group_rows(self, shown, mode):
        """Order the filtered ``shown`` ``(idx, it)`` pairs into display rows
        ``(header|None, idx, it)``: flat when not grouping, else group-header rows followed
        by their members (preserving report-items order within each group)."""
        if mode == "No grouping" or not shown:
            return [(None, i, it) for i, it in shown]
        groups = {}
        for i, it in shown:
            groups.setdefault(self._report_group_key(it, mode), []).append((i, it))
        rows = []
        for g, members in groups.items():        # dict keeps first-seen (report-items) order
            rows.append((f"{g}  ({len(members)})", -1, None))
            rows += [(None, i, it) for i, it in members]
        return rows

    def _refresh_report_list(self):
        lst = getattr(self, "report_list", None)
        if lst is None:
            return
        sel_idx = self._report_sel_index()       # remember selection by report-items index
        query = (self.report_search.text().strip().lower()
                 if getattr(self, "report_search", None) else "")
        mode = (self.report_groupby.currentText()
                if getattr(self, "report_groupby", None) else "No grouping")
        shown = [(i, it) for i, it in enumerate(self.report_items)
                 if not query or self._report_item_matches(it, query)]
        rows = self._report_group_rows(shown, mode)
        flat = mode == "No grouping" and not query   # drag-reorder only on a flat, full list
        lst.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove if flat
                            else QtWidgets.QAbstractItemView.NoDragDrop)
        lst.blockSignals(True)
        lst.clear()
        sel_row = -1
        for header, idx, it in rows:
            if header is not None:               # non-selectable group header
                hi = QtWidgets.QListWidgetItem(header)
                hi.setData(QtCore.Qt.UserRole, None)
                hi.setFlags(QtCore.Qt.ItemIsEnabled)
                f = hi.font(); f.setBold(True); hi.setFont(f)
                hi.setForeground(QtGui.QBrush(QtGui.QColor(MUTED_FG)))
                lst.addItem(hi)
                continue
            suffix = "   · auto" if it.get("auto") else ""
            named = "  ✎" if it.get("name_origin") == "user" else ""
            wi = QtWidgets.QListWidgetItem(f"{_BADGE.get(it.get('type'), '•')}    "
                                           f"{it.get('title') or it.get('type')}{named}{suffix}")
            if it.get("auto"):                       # auto-logged exports read muted vs curated adds
                wi.setForeground(QtGui.QBrush(QtGui.QColor(MUTED_FG)))
            wi.setData(QtCore.Qt.UserRole, idx)
            wi.setToolTip(self._report_source_tooltip(it))
            lst.addItem(wi)
            if idx == sel_idx:
                sel_row = lst.count() - 1
        lst.blockSignals(False)
        if getattr(self, "_report_bar", None) is not None:
            n = len(self.report_items)
            if not n:
                status = "Report is empty — add analyses to build the PDF."
            else:
                status = f"{n} item{'s' if n != 1 else ''} in report"
                if query and len(shown) != n:
                    status += f" · {len(shown)} shown"
            self._report_bar.set_status(status)
        if sel_row >= 0:
            lst.setCurrentRow(sel_row)
        has_items = bool(self.report_items)
        for b in (getattr(self, "_report_export_btn", None), getattr(self, "_report_files_btn", None), getattr(self, "_report_clear_btn", None)):
            if b is not None:
                b.setEnabled(has_items)

    def _report_sel_index(self):
        """The ``report_items`` index of the selected row (its ``UserRole``), or -1 when
        nothing — or a group header — is selected. Stable across filter/group changes."""
        lst = getattr(self, "report_list", None)
        cur = lst.currentItem() if lst is not None else None
        idx = cur.data(QtCore.Qt.UserRole) if cur is not None else None
        return idx if isinstance(idx, int) and 0 <= idx < len(self.report_items) else -1

    def _report_select_index(self, idx):
        """Select the visible row whose ``UserRole`` == report-items index ``idx`` (if any)."""
        lst = getattr(self, "report_list", None)
        if lst is None:
            return
        for r in range(lst.count()):
            if lst.item(r).data(QtCore.Qt.UserRole) == idx:
                lst.setCurrentRow(r)
                return

    def _report_sync_order(self, *_):
        """Rebuild ``report_items`` to match the list after a drag-reorder, reading each
        row's original index from its ``UserRole`` payload. Only fires in flat mode (drag is
        disabled while grouped/filtered), so every row is a leaf with a valid index."""
        lst = self.report_list
        order = []
        for r in range(lst.count()):
            idx = lst.item(r).data(QtCore.Qt.UserRole)
            if idx is None:                       # a stray header — bail rather than drop items
                return
            if not isinstance(idx, int) or not (0 <= idx < len(self.report_items)):
                return
            order.append(self.report_items[idx])
        if len(order) == len(self.report_items):
            self.report_items = order
            self._refresh_report_list()       # re-stamp UserRole indices to the new order
            self._mark_dirty()

    def _report_move(self, delta):
        i = self._report_sel_index()
        j = i + delta
        if i < 0 or not (0 <= j < len(self.report_items)):
            return
        self.report_items[i], self.report_items[j] = self.report_items[j], self.report_items[i]
        self._refresh_report_list()
        self._report_select_index(j)
        self._mark_dirty()

    def _report_delete(self):
        i = self._report_sel_index()
        if i < 0:
            return
        it = self.report_items.pop(i)
        self._report_prune_thumbs()
        self._refresh_report_list()
        if self.report_items:
            self._report_select_index(min(i, len(self.report_items) - 1))
        else:
            self._report_preview_current(-1)
        self._mark_dirty()
        self.statusBar().showMessage(f"Removed from report: {it.get('title') or it.get('type')}")

    def _report_edit(self):
        r = self._report_sel_index()
        if r < 0:
            return
        it = self.report_items[r]
        is_note = it.get("type") == "note"
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Edit report item")
        form = QtWidgets.QFormLayout(dlg)
        title_edit = QtWidgets.QLineEdit(it.get("title", ""))
        body_edit = QtWidgets.QPlainTextEdit(it.get("text", "") if is_note else it.get("caption", ""))
        body_edit.setFixedHeight(120)
        form.addRow("Heading" if is_note else "Title", title_edit)
        form.addRow("Body" if is_note else "Caption", body_edit)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        it["title"] = title_edit.text().strip()
        it["name_origin"] = "user"                   # an explicit edit is a user-chosen name
        body = body_edit.toPlainText().strip()
        if is_note:
            it["text"] = body
            it["heading"] = it["title"]
        else:
            it["caption"] = body
        self._refresh_report_list()
        self._report_select_index(r)
        self._mark_dirty()

    # ----- preview --------------------------------------------------------- #
    def _report_preview_current(self, row):
        """Slot for the list's currentRowChanged: ``row`` is the *visible* row, which maps
        to a report-items index via the row's ``UserRole`` (group headers / out-of-range →
        clear)."""
        lst = getattr(self, "report_list", None)
        it = None
        if lst is not None and 0 <= row < lst.count():
            idx = lst.item(row).data(QtCore.Qt.UserRole)
            if isinstance(idx, int) and 0 <= idx < len(self.report_items):
                it = self.report_items[idx]
        self._report_show_preview(it)

    def _report_show_preview(self, it):
        """Render the detail pane for one item (or clear when ``it`` is None): a data-table
        preview for table items, a figure thumbnail / live render for figure items, plus the
        provenance block and the 'Open data folder' button state."""
        prev = getattr(self, "report_preview", None)
        if prev is None:
            return
        tbl = getattr(self, "report_table_preview", None)
        srclbl = getattr(self, "report_source", None)
        folder_btn = getattr(self, "report_open_folder_btn", None)
        if it is None:
            prev.setPixmap(QtGui.QPixmap())
            prev.setText("Add analyses on the left, then preview them here.")
            prev.show()
            if tbl is not None:
                tbl.hide()
            if srclbl is not None:
                srclbl.setText("")
            if folder_btn is not None:
                folder_btn.setEnabled(False)
            return
        t = (it.get("type") or "").lower()
        if t in ("stats", "features", "analysis") and tbl is not None:
            prev.hide()
            self._fill_table_preview(it.get("records") or [])
            tbl.show()
        else:
            if tbl is not None:
                tbl.hide()
            prev.show()
            pix = self._report_item_pixmap(it)
            if pix is not None:
                prev.setText("")
                prev.setPixmap(pix)
            else:
                prev.setPixmap(QtGui.QPixmap())
                prev.setText(self._report_detail_text(it))
        if srclbl is not None:
            srclbl.setText(self._report_source_html(it))
        if folder_btn is not None:
            self._report_folder_path = self._report_item_data_dir(it)
            folder_btn.setEnabled(bool(self._report_folder_path))

    # ----- data previews + thumbnails -------------------------------------- #
    def _report_item_pixmap(self, it):
        """Best preview pixmap for a figure item: a crisp live render when the dataset is
        loaded, else the cached thumbnail (instant, and the only thing available after a
        reopen before the m/z cube reloads). None → caller shows text."""
        live = None
        if getattr(self, "ds", None) is not None:
            try:
                live = self._report_render_pixmap(it)
            except Exception:  # noqa: BLE001 — preview is best-effort, never fatal
                import traceback
                traceback.print_exc()
        if live is not None:
            return live
        from . import reportthumbs
        return reportthumbs.load_pixmap(self._report_thumb_path(it))

    @staticmethod
    def _fmt_preview_cell(v):
        if isinstance(v, float):
            return f"{v:.4g}"
        return "" if v is None else str(v)

    def _fill_table_preview(self, records, max_rows=12, max_cols=8):
        """Populate the mini data-table from an item's persisted ``records`` (first rows ×
        first columns) — instant, needs no dataset. Width-capped to avoid pane overflow."""
        tbl = self.report_table_preview
        records = [r for r in (records or []) if isinstance(r, dict)]
        if not records:
            tbl.clear(); tbl.setRowCount(0); tbl.setColumnCount(0)
            return
        rows = records[:max_rows]
        cols = list(dict.fromkeys(k for r in rows for k in r))[:max_cols]
        tbl.clear()
        tbl.setColumnCount(len(cols))
        tbl.setRowCount(len(rows))
        tbl.setHorizontalHeaderLabels([str(c) for c in cols])
        for ri, rec in enumerate(rows):
            for ci, c in enumerate(cols):
                tbl.setItem(ri, ci, QtWidgets.QTableWidgetItem(self._fmt_preview_cell(rec.get(c))))
        tbl.resizeColumnsToContents()

    def _report_item_id(self, it):
        """Stable per-item id (the thumbnail filename stem), minted lazily for legacy items."""
        rid = it.get("id")
        if not rid:
            import uuid
            rid = uuid.uuid4().hex[:12]
            it["id"] = rid
        return rid

    def _report_thumb_dir(self):
        from . import reportthumbs
        return reportthumbs.thumb_dir_for(getattr(self, "_session_path", None))

    def _report_thumb_path(self, it):
        from . import reportthumbs
        rid = it.get("id")
        return reportthumbs.thumb_path(self._report_thumb_dir(), rid) if rid else None

    def _report_maybe_write_thumb(self, it):
        """Best-effort cache of a small PNG thumbnail for a figure item, so the browser can
        preview it instantly and after a reopen. No-op for table/note items, a synthetic
        dataset, or before the first auto-save (no session path yet)."""
        if (it.get("type") or "").lower() not in ("ion", "overlay", "spectrum", "segmentation"):
            return
        if getattr(self, "ds", None) is None:
            return
        from . import reportthumbs
        path = reportthumbs.thumb_path(self._report_thumb_dir(), self._report_item_id(it))
        if not path:
            return
        try:
            pix = self._report_render_pixmap(it)
        except Exception:  # noqa: BLE001 — a thumbnail is a cache, never break a capture
            return
        if pix is not None and reportthumbs.write_pixmap(path, pix):
            it["thumb"] = os.path.basename(path)

    def _report_item_data_dir(self, it):
        """The folder the item's source dataset was loaded from, when it still exists."""
        path = (it.get("source") or {}).get("dataset_path") or ""
        if path and os.path.exists(path):
            return os.path.dirname(path) or path
        d = os.path.dirname(path)
        return d if d and os.path.isdir(d) else ""

    def _report_open_data_folder(self):
        path = getattr(self, "_report_folder_path", "")
        if path and os.path.isdir(path):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))

    def _report_prune_thumbs(self):
        """Drop cached thumbnail PNGs for items no longer in the report (best-effort)."""
        from . import reportthumbs
        reportthumbs.prune(self._report_thumb_dir(),
                           [it.get("id") for it in self.report_items if it.get("id")])

    def _report_detail_text(self, it):
        t = it.get("type")
        lines = [it.get("title") or t, ""]
        if t == "stats":
            lines.append(f"{len(it.get('records') or [])} features · "
                         f"{it.get('a_label', 'A')} vs {it.get('b_label', 'B')}")
        elif t == "features":
            lines.append(f"{len(it.get('records') or [])} features")
        elif t == "analysis":
            lines.append(f"{it.get('analysis_kind') or 'Analysis'} · "
                         f"{len(it.get('records') or [])} features")
        elif t == "note":
            lines.append(it.get("text", "") or "(no body)")
        if it.get("caption"):
            lines += ["", it["caption"]]
        return "\n".join(lines)

    def _report_item_figure(self, it, opts):
        """Render a figure-bearing item to a matplotlib ``Figure`` (None for non-figure
        items). Shared by the live preview and "Export all → files" so both stay identical."""
        t = (it.get("type") or "").lower()
        if t not in ("ion", "overlay", "spectrum", "segmentation"):
            return None
        built = self._report_build_export_item(it, opts)
        if not built:
            return None
        if t == "ion":
            return export.render_ion_panel(**built["panel"])
        if t == "overlay":
            p = built["panel"]
            return export.render_overlay_panel(p["rgb"], p.get("channels"), crop=p.get("crop"),
                                               outlines=p.get("outlines"),
                                               show_legend=p.get("show_legend", True),
                                               **(p.get("design") or {}))
        if t == "spectrum":
            return export.render_spectrum_figure(built["spectra"], theme=opts["theme"],
                                                 width_in=opts.get("width", 7.0), dpi=opts["dpi"])
        s = built["segmentation"]                          # segmentation
        # honour the report's scale-bar toggle; the renderer auto-sizes the bar to the map
        px = getattr(self.ds, "pixel_size_um", None) if opts.get("scalebar_on") else None
        return export.render_segmentation_figure(s["image"], s["colors"], legend=s.get("legend"),
                                                 title=s.get("title", "Segmentation"),
                                                 theme=opts["theme"], pixel_size_um=px,
                                                 width_in=opts.get("width", 6.0), dpi=opts["dpi"])

    def _report_render_pixmap(self, it):
        """A small live render of a figure-bearing item (None → caller shows text)."""
        fig = self._report_item_figure(it, self._report_preview_opts())
        return self._fig_to_pixmap(fig) if fig is not None else None

    @staticmethod
    def _fig_to_pixmap(fig):
        pix = fig_to_pixmap(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)            # this preview render owns the throwaway figure
        return pix

    def _report_preview_opts(self):
        cmap = self.cmap_combo.currentText() if getattr(self, "cmap_combo", None) else "viridis"
        return {"theme": "light", "cmap": cmap, "corner": "lower left", "card": True,
                "spectrum": True, "colorbar": True, "scalebar_on": False, "scalebar_um": 0.0,
                "title": True, "width": 6.0, "dpi": 96}

    def _report_export_opts(self):
        """Full-quality render settings for writing report items as loose files (the preview
        settings, bumped to print DPI with the scale bar on)."""
        o = self._report_preview_opts()
        # size the bar to this slide (~20% of its width), not a fixed 500 µm that reads as a
        # sliver on a 2 cm slide; falls back to no bar when the pixel size is unknown.
        sb = self.ds.auto_scale_bar_um() if getattr(self, "ds", None) is not None else None
        o.update(dpi=300, width=7.0, scalebar_on=bool(sb), scalebar_um=(sb or 0.0))
        return o

    # ----- wipe / restart -------------------------------------------------- #
    def _report_clear(self):
        """Empty the report list only — the logged images / tables / notes. The rest of the
        analysis (regions, segmentation, statistics, features) is left untouched."""
        n = len(self.report_items)
        if not n:
            self.statusBar().showMessage("Report is already empty.")
            return
        if not confirm(self, "Clear report", f"Remove all {n} item{'s' if n != 1 else ''} from the report?\n\nThis clears the report list only — your regions, segmentation, statistics and features stay as they are.", ok_text="Clear report"):
            return
        self.report_items = []
        self._report_prune_thumbs()
        self._refresh_report_list()
        self._report_preview_current(-1)
        self._mark_dirty()
        self.statusBar().showMessage("Report cleared.")

    # ----- "Export all → files" (one loose image / CSV per item) ----------- #
    def _report_export_folder(self, it):
        """Sub-folder name for an item — one folder per kind of analysis. Logged analysis
        results land in a folder named after the analysis itself (e.g. ``discriminating_…``)."""
        t = (it.get("type") or "").lower()
        if t == "analysis":
            return self._safe_name(it.get("analysis_kind") or "analysis")
        return _EXPORT_FOLDERS.get(t, "other")

    def _report_manifest_row(self, idx, it, rel_path):
        """One log row recording where an exported item came from."""
        src = it.get("source") or {}
        t = (it.get("type") or "").lower()
        return {"#": idx + 1, "file": rel_path or "",
                "type": (it.get("analysis_kind") or "analysis") if t == "analysis" else t,
                "title": it.get("title") or "",
                "method": src.get("method", ""),
                "regions": src.get("regions") or self._item_regions_text(it),
                "feature_list": self._item_features_text(it),
                "settings": src.get("params", ""),
                "dataset": src.get("dataset", ""),
                "dataset_path": src.get("dataset_path", ""),
                "normalization": src.get("norm", ""),
                "ppm": src.get("ppm", ""),
                "logged": src.get("created", ""),
                "caption": it.get("caption") or ""}

    def _write_report_logs(self, folder, manifest):
        """Write the contents/provenance log beside the foldered files: a machine-readable
        ``report_log.csv`` and a human-readable ``README.md`` — both record the regions and
        feature list that fed every item."""
        if not manifest:
            return
        import pandas as pd
        cols = ["#", "file", "type", "title", "method", "regions", "feature_list",
                "settings", "dataset", "dataset_path", "normalization", "ppm", "logged",
                "caption"]
        export.write_table(pd.DataFrame(manifest, columns=cols),
                           os.path.join(folder, "report_log.csv"), fmt="csv")
        lines = ["# Report export — contents & provenance log", "",
                 "Each item is written into a folder named for its analysis type. The entries "
                 "below record the **regions** and **feature list** that fed every item.", ""]
        for row in manifest:
            lines += [f"## {row['#']:02d}. {row['title'] or row['type']}", "",
                      f"- **File:** `{row['file']}`",
                      f"- **Type:** {row['type']}"]
            if row.get("method"):
                lines.append(f"- **Method:** {row['method']}")
            lines += [f"- **Regions / pixels:** {row['regions'] or '—'}",
                      f"- **Feature list:** {row['feature_list'] or '—'}"]
            if row.get("settings"):
                lines.append(f"- **Settings:** {row['settings']}")
            if row["dataset"]:
                lines.append(f"- **Dataset:** {row['dataset']}")
            if row.get("dataset_path"):
                lines.append(f"- **Data path:** `{row['dataset_path']}`")
            if row["normalization"]:
                lines.append(f"- **Normalization:** {row['normalization']}  ·  "
                             f"**Extraction tol.:** ±{row['ppm']} ppm")
            if row["logged"]:
                lines.append(f"- **Logged:** {row['logged']}")
            if row["caption"]:
                lines.append(f"- **Caption:** {row['caption']}")
            lines.append("")
        with open(os.path.join(folder, "README.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _report_export_all_files(self):
        """Render every item in the report list to an individual file, sorted into one
        folder per analysis type (figures → images, tables → CSV, notes → text), and write
        a ``report_log.csv`` + ``README.md`` logging the regions / feature list behind each."""
        if not self.report_items:
            self.statusBar().showMessage("Report is empty — add or export something first.")
            return
        d = filedialogs.get_existing_directory(self, "Export all report items → choose a folder")
        if not d:
            return
        import pandas as pd
        opts = self._report_export_opts()
        total = len(self.report_items)
        manifest = []
        n = 0
        for idx, it in enumerate(self.report_items):
            t = (it.get("type") or "").lower()
            sub_name = self._report_export_folder(it)
            base = f"{idx + 1:02d}_{self._safe_name(it.get('title') or t)}"
            rel = None
            try:
                if t in ("ion", "overlay", "spectrum", "segmentation"):
                    fig = self._report_item_figure(it, opts)
                    if fig is None:
                        continue
                    rel = os.path.join(sub_name, base + ".png")
                    os.makedirs(os.path.join(d, sub_name), exist_ok=True)
                    export.save_figure(fig, os.path.join(d, rel), dpi=opts["dpi"])
                elif t in ("stats", "features", "analysis"):
                    df = pd.DataFrame(it.get("records") or [])
                    if not len(df):
                        continue
                    rel = os.path.join(sub_name, base + ".csv")
                    os.makedirs(os.path.join(d, sub_name), exist_ok=True)
                    export.write_table(df, os.path.join(d, rel), fmt="csv")
                elif t == "note":
                    rel = os.path.join(sub_name, base + ".txt")
                    os.makedirs(os.path.join(d, sub_name), exist_ok=True)
                    body = (it.get("heading") or it.get("title") or "") + "\n\n" + (it.get("text") or "")
                    with open(os.path.join(d, rel), "w", encoding="utf-8") as f:
                        f.write(body.strip() + "\n")
                else:
                    continue
                n += 1
                manifest.append(self._report_manifest_row(idx, it, rel))
                self.statusBar().showMessage(f"Exporting report files… {n}/{total}")
                QtWidgets.QApplication.processEvents()
            except Exception:  # noqa: BLE001 — one bad item shouldn't sink the batch
                import traceback
                traceback.print_exc()
        try:
            self._write_report_logs(d, manifest)
        except Exception:  # noqa: BLE001 — the log is best-effort, never fatal to the export
            import traceback
            traceback.print_exc()
        if getattr(self, "prov", None) is not None:
            try:
                self.prov.output(d, description="report — all items as files (foldered + logged)")
            except Exception:  # noqa: BLE001
                pass
        self.statusBar().showMessage(
            f"Wrote {n} file{'s' if n != 1 else ''} into per-type folders in {d} "
            "(see report_log.csv / README.md)")

    # ----- export bridge --------------------------------------------------- #
    def report_document_items(self, opts):
        """The ordered ``items`` list for :func:`smile_msi.export.build_book`, rebuilt from
        the stored params (heavy arrays reconstructed here, off the persisted state)."""
        out = []
        for it in self.report_items:
            try:
                built = self._report_build_export_item(it, opts)
            except Exception:  # noqa: BLE001 — one bad item shouldn't sink the whole book
                import traceback
                traceback.print_exc()
                built = None
            if built:
                out.append(built)
        return out

    def _report_build_export_item(self, it, opts):
        fn = {"ion": self._report_ion_panel, "overlay": self._report_overlay_panel,
              "spectrum": self._report_spectrum_item, "segmentation": self._report_segmentation_item,
              "stats": self._report_stats_item, "features": self._report_features_item,
              "analysis": self._report_analysis_item,
              "note": self._report_note_item}.get((it.get("type") or "").lower())
        return fn(it, opts) if fn else None

    def _region_by_name(self, name):
        if not name:
            return None
        for r in (getattr(self, "regions", None) or []):
            if r.get("name") == name:
                return r
        return None

    def _report_ion_panel(self, it, opts):
        mz = float(it["mz"])
        ppm = float(it.get("ppm") or self.ppm)
        # extract on the apex (display centre); ``mz`` below stays the catalogued label
        img = self.ds.ion_image(self._apex_mz(mz), tol_ppm=ppm, reduce=self.reduce, norm=self.norm)
        win = it.get("window") or [0.0, float(self.contrast_spin.value())]
        lo, hi = float(win[0]), float(win[1])
        kw = dict(image=img, mz=mz, label=it.get("label", "") or "", low=lo, high=hi,
                  window=(lo, hi))
        dk = self._design_kwargs(opts)
        if it.get("cmap"):
            dk["cmap"] = it["cmap"]
        kw.update(dk)
        kw["mean_spectrum"] = self._mean_spec_xy()
        rg = self._region_by_name(it.get("crop_region"))
        if rg is not None:
            kw.update(self._crop_kwargs(rg))
        return {"type": "ion", "title": it.get("title"), "caption": it.get("caption"), "panel": kw}

    def _report_overlay_rgb(self, channels):
        acc = np.zeros((self.ds.height, self.ds.width, 3), float)
        for ch in channels:
            img = self.ds.ion_image(self._apex_mz(ch["mz"]), tol_ppm=float(ch.get("ppm") or self.ppm),
                                    reduce=self.reduce, norm=self.norm)
            norm = self._windowed(img, float(ch.get("lo", 0.0)), float(ch.get("hi", 100.0)))
            col = QtGui.QColor(ch.get("color", "#ffffff"))
            acc[..., 0] += norm * col.redF()
            acc[..., 1] += norm * col.greenF()
            acc[..., 2] += norm * col.blueF()
        return (np.clip(acc, 0, 1) * 255).astype(np.uint8)

    def _report_overlay_panel(self, it, opts):
        channels = it.get("channels") or []
        if not channels:
            return None
        rgb = self._report_overlay_rgb(channels)
        design = self._design_kwargs(opts)
        outlines = self._overlay_outlines() if it.get("outline") else None
        crop = None
        rg = self._region_by_name(it.get("crop_region"))
        if rg is not None:
            crop = self._region_bbox(rg)
            outs = list(outlines or [])
            om = self._region_mask_2d(rg)
            if om is not None:
                outs.append((om, rg.get("color", "#ffffff")))
            outlines = outs or None
        panel = {"kind": "overlay", "rgb": rgb, "channels": channels, "crop": crop,
                 "outlines": outlines, "show_legend": True, "design": design}
        return {"type": "overlay", "title": it.get("title"), "caption": it.get("caption"),
                "panel": panel}

    def _report_spectrum_item(self, it, opts):
        spectra = [("mean spectrum", *self._mean_spec_xy())]
        name = it.get("region")
        rg = self._region_by_name(name)
        if rg is not None:
            m = self._region_pixel_mask(rg)
            if m is not None:
                ax, y = self.ds.mean_spectrum(mask=np.asarray(m, bool))
                spectra.append((f"{name} spectrum", np.asarray(ax), np.asarray(y),
                                rg.get("color", "#ffcc00")))
        return {"type": "spectrum", "title": it.get("title"), "caption": it.get("caption"),
                "spectra": spectra}

    def _report_segmentation_item(self, it, opts):
        labels = np.asarray(it.get("labels") or [], int)
        image = self.ds.to_image(labels) if labels.size == self.ds.n_pixels else None
        if image is None:
            return None
        colors = {int(k): v for k, v in (it.get("colors") or {}).items()}
        seg = {"image": image, "colors": colors, "legend": it.get("legend"),
               "title": it.get("title") or "Segmentation"}
        return {"type": "segmentation", "title": it.get("title"), "caption": it.get("caption"),
                "segmentation": seg}

    def _report_stats_item(self, it, opts):
        import pandas as pd
        df = pd.DataFrame(it.get("records") or [])
        st = {"df": df, "a_label": it.get("a_label", "Group A"),
              "b_label": it.get("b_label", "Group B"),
              "title": it.get("title") or "Discriminating features"}
        return {"type": "stats", "title": it.get("title"), "caption": it.get("caption"),
                "stats": st}

    def _report_features_item(self, it, opts):
        return {"type": "features", "title": it.get("title") or "Feature table",
                "caption": it.get("caption"), "records": list(it.get("records") or [])}

    def _report_analysis_item(self, it, opts):
        """A logged CSV-analysis result (discriminating / multi-group …) — rendered as a
        generic table page in the curated PDF book."""
        title = it.get("title") or it.get("analysis_kind") or "Analysis"
        return {"type": "features", "title": title, "caption": it.get("caption"),
                "records": list(it.get("records") or [])}

    def _report_note_item(self, it, opts):
        head = it.get("title") or it.get("heading") or "Note"
        return {"type": "note", "title": head, "heading": head,
                "caption": it.get("caption"), "text": it.get("text", "")}
