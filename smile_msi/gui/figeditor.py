"""FigureEditorDialog — a small WYSIWYG editor for an exportable figure.

One generic dialog drives both figure shapes (the SHAP bubble plot and an ion-image
legend): a **zoomable** preview of the actual export on the right, and on the left one
reorderable, checkable list per axis (rows = ions, columns = regions/donors), a title
field, and any figure-specific style choices (e.g. the bubble colormap). The caller
supplies a ``render(spec)`` callback that turns the current :class:`~smile_msi.figedit.
FigEditSpec` into a matplotlib ``Figure`` using the very same ``export`` render function
it will save with — so the preview *is* the export.

What you can do: drag to **reorder**, untick to **drop**, double-click to **rename** (a
label override), set a title, pick style options, **zoom** the preview (Fit / %), then
**Export…**. The edited spec is kept on the dialog (``.spec``) for the caller to remember.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from .. import export, figedit as fe, stylelib
from . import common, filedialogs

_OVERRIDE_ROLE = QtCore.Qt.UserRole + 1          # per-item label override text
_ORIG_ROLE = QtCore.Qt.UserRole                  # per-item original axis index

_ZOOM_LEVELS = [("Fit", None), ("50%", 0.5), ("75%", 0.75), ("100%", 1.0),
                ("150%", 1.5), ("200%", 2.0)]


class FigureEditorDialog(QtWidgets.QDialog):
    def __init__(self, parent, *, title, render, row_items=None, col_items=None, spec=None,
                 row_axis_name="Ions", col_axis_name="Columns", default_name="figure",
                 style_options=None, row_sorts=None, regions=None, topn=None,
                 row_provider=None, extra_actions=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(1040, 680)
        self._render_cb = render
        self._default_name = default_name
        self._style_options = list(style_options or [])
        self._extra_actions = list(extra_actions or [])
        self._pixmap = None

        # Optional region-aware mode: a "Region" combo + "Top ions" spinbox whose values feed
        # ``row_provider(region, n) -> [labels]`` so the row axis is rebuilt (and re-ranked)
        # whenever either changes — this is how the per-region importance histogram shows each
        # region's OWN top-N ions instead of one shared, globally-ranked set.
        self._regions = [str(r) for r in regions] if regions else None
        self._topn_cfg = tuple(topn) if topn else None      # (min, max, default)
        self._row_provider = row_provider
        opts0 = dict(spec.options) if spec else {}
        init_region = opts0.get("region") or (self._regions[0] if self._regions else None)
        init_topn = opts0.get("topn") or (self._topn_cfg[2] if self._topn_cfg else None)
        if row_provider is not None:
            row_items = row_provider(init_region, init_topn)

        self._row_originals = [str(x) for x in (row_items or [])]
        self._col_originals = None if col_items is None else [str(x) for x in col_items]
        n_cols = None if self._col_originals is None else len(self._col_originals)
        self.spec = spec or fe.FigEditSpec.identity(len(self._row_originals), n_cols)
        if row_provider is not None:                     # rows are owned by the provider
            self.spec.rows = fe.AxisEdit.identity(len(self._row_originals))
        if init_region is not None:
            self.spec.options["region"] = init_region
        if init_topn is not None:
            self.spec.options["topn"] = int(init_topn)
        for opt in self._style_options:                  # seed unset options with defaults
            self.spec.options.setdefault(opt["key"], opt.get("default"))

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)

        # ---- left: the edit controls ----------------------------------------- #
        controls = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(controls)
        cv.setContentsMargins(10, 10, 10, 10)
        cv.setSpacing(6)
        cv.addWidget(common.section_title("Title"))
        self.title_edit = QtWidgets.QLineEdit(self.spec.title or "")
        self.title_edit.setPlaceholderText("(default title)")
        self.title_edit.textEdited.connect(self._schedule_render)
        cv.addWidget(self.title_edit)

        self._opt_combos = {}
        if self._style_options:
            cv.addWidget(common.section_title("Style"))
            for opt in self._style_options:
                cv.addWidget(QtWidgets.QLabel(opt["label"]))
                combo = common.NoScrollComboBox()
                for clabel, cval in opt["choices"]:
                    combo.addItem(clabel, cval)
                i = combo.findData(self.spec.options.get(opt["key"]))
                combo.setCurrentIndex(max(0, i))
                combo.activated.connect(self._schedule_render)
                self._opt_combos[opt["key"]] = combo
                cv.addWidget(combo)

        # Style preset — a reusable design-system look (fonts, type sizes, line weights,
        # category colours) applied LIVE to the preview and the export, on top of the per-figure
        # style options above. Because the preview renders through the very same export function,
        # picking a look here shows exactly what you'll save.
        self._style_preset = None
        cv.addWidget(common.section_title("Style preset"))
        self.preset_combo = common.NoScrollComboBox()
        self.preset_combo.setToolTip(
            "A reusable look shared with every figure the app exports. Copy its recipe (JSON) to "
            "tweak elsewhere, then Import the result — the preview updates instantly.")
        self._refresh_presets()
        self.preset_combo.activated.connect(self._on_preset_change)
        cv.addWidget(self.preset_combo)
        prow = QtWidgets.QHBoxLayout()
        prow.setSpacing(4)
        prow.addWidget(common.button("Copy recipe", self._copy_recipe,
                                     tooltip="Copy this look's recipe (JSON) to the clipboard."))
        prow.addWidget(common.button("Import…", self._import_recipe,
                                     tooltip="Paste a recipe (JSON) to add it as a saved look."))
        cv.addLayout(prow)

        # region selector + top-N (drives row_provider; rows re-rank per region on change)
        self.region_combo = None
        self.topn_spin = None
        if self._regions:
            cv.addWidget(common.section_title("Region"))
            self.region_combo = common.NoScrollComboBox()
            self.region_combo.addItems(self._regions)
            self.region_combo.setToolTip("Which region's biomarker histogram to show — each "
                                         "region ranks its OWN top ions by importance.")
            i = self.region_combo.findText(str(self.spec.options.get("region")))
            self.region_combo.setCurrentIndex(max(0, i))
            self.region_combo.activated.connect(self._on_data_change)
            cv.addWidget(self.region_combo)
        if self._topn_cfg:
            cv.addWidget(QtWidgets.QLabel("Top ions"))
            self.topn_spin = common.NoScrollSpinBox()
            self.topn_spin.setRange(int(self._topn_cfg[0]), int(self._topn_cfg[1]))
            self.topn_spin.setValue(int(self.spec.options.get("topn") or self._topn_cfg[2]))
            self.topn_spin.setToolTip("How many of this region's strongest ions to show.")
            self.topn_spin.valueChanged.connect(self._on_data_change)
            cv.addWidget(self.topn_spin)

        cv.addWidget(common.section_title(row_axis_name))
        cv.addWidget(common.note("Drag or ↑ ↓ to reorder · untick to drop · double-click to "
                                 "rename · shift-click a run, then Space to drop it"))
        self._row_sorts = list(row_sorts or [])
        self.sort_combo = None
        if self._row_sorts:
            srow = QtWidgets.QHBoxLayout()
            srow.addWidget(QtWidgets.QLabel("Sort by"))
            self.sort_combo = common.NoScrollComboBox()
            self.sort_combo.addItem("(choose…)", None)
            for si, s in enumerate(self._row_sorts):
                self.sort_combo.addItem(s["label"], si)
            self.sort_combo.setToolTip("One-click reorder of the ions by m/z, importance, or "
                                       "direction within a column. Drag still works after.")
            self.sort_combo.activated.connect(self._apply_sort)
            srow.addWidget(self.sort_combo, 1)
            cv.addLayout(srow)
        self.row_list = self._make_list()
        cv.addWidget(self._add_bar(self.row_list, row_axis_name))
        cv.addWidget(self.row_list, 1)

        self.col_list = None
        if self._col_originals is not None:
            cv.addWidget(common.section_title(col_axis_name))
            self.col_list = self._make_list()
            cv.addWidget(self._add_bar(self.col_list, col_axis_name))
            cv.addWidget(self.col_list, 1)

        cv.addWidget(common.button("Reset", self._reset, tooltip="Restore original order, "
                                   "labels, included items, and style"))
        split.addWidget(controls)

        # ---- right: zoomable preview of the actual export -------------------- #
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        zbar = QtWidgets.QHBoxLayout()
        zbar.addWidget(QtWidgets.QLabel("Zoom"))
        self.zoom_combo = common.NoScrollComboBox()
        for zlabel, zval in _ZOOM_LEVELS:
            self.zoom_combo.addItem(zlabel, zval)
        self.zoom_combo.setToolTip("Zoom the preview. 'Fit' shows the whole figure so you "
                                   "can judge the real layout; pick a % to inspect detail.")
        self.zoom_combo.currentIndexChanged.connect(self._apply_zoom)
        zbar.addWidget(self.zoom_combo)
        zbar.addStretch(1)
        rv.addLayout(zbar)
        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(QtCore.Qt.AlignCenter)
        self._scroll.setBackgroundRole(QtGui.QPalette.Base)
        self._img_label = QtWidgets.QLabel()
        self._img_label.setAlignment(QtCore.Qt.AlignCenter)
        self._scroll.setWidget(self._img_label)
        rv.addWidget(self._scroll, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([330, 710])

        outer = QtWidgets.QVBoxLayout(self)
        outer.addWidget(split, 1)
        foot = QtWidgets.QHBoxLayout()
        foot.addStretch(1)
        foot.addWidget(common.button("Close", self.accept))
        for label, cb in self._extra_actions:            # e.g. "Export every region…"
            foot.addWidget(common.button(label, cb))
        foot.addWidget(common.primary_button("Export…", self._export, action="export"))
        outer.addLayout(foot)

        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)                 # debounce rapid edits
        self._timer.timeout.connect(self._render)

        self._populate(self.row_list, self._row_originals, self.spec.rows)
        if self.col_list is not None:
            self._populate(self.col_list, self._col_originals, self.spec.cols)
        self._render()

    # ----- list plumbing --------------------------------------------------- #
    def _make_list(self):
        lw = QtWidgets.QListWidget()
        lw.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        # Extended, not single: a 30-ion bubble plot should let you shift-click a run and drop
        # it with one Space, and drag the whole run at once.
        lw.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        lw.setMaximumHeight(240)
        lw.itemChanged.connect(self._schedule_render)          # checkbox toggled
        lw.itemDoubleClicked.connect(self._rename_item)
        lw.model().rowsMoved.connect(self._schedule_render)    # drag reorder
        common.install_space_toggle(lw)
        return lw

    def _populate(self, lw, originals, axis: fe.AxisEdit):
        lw.blockSignals(True)
        lw.clear()
        for oi in axis.order:
            ov = axis.labels.get(oi)
            text = str(ov) if ov else originals[oi]
            it = QtWidgets.QListWidgetItem(text)
            it.setData(_ORIG_ROLE, int(oi))
            it.setData(_OVERRIDE_ROLE, ov)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Unchecked if oi in axis.dropped else QtCore.Qt.Checked)
            if ov:
                it.setToolTip(f"renamed (was {originals[oi]})")
            lw.addItem(it)
        lw.blockSignals(False)
        bar = getattr(lw, "_bulk_bar", None)      # the bar was built over an empty list
        if bar is not None:
            bar.refresh_count()

    def _add_bar(self, lw, axis_name):
        """All/None/Invert + ↑ ↓ over one axis. Dropping half a 30-ion figure was 15 clicks."""
        noun = axis_name.rstrip("s").lower() or "item"      # "Ions" -> "ion", "Columns" -> "column"
        bar = common.check_list_bar(lw, noun, on_change=self._schedule_render, reorder=True)
        lw._bulk_bar = bar                                  # _populate refreshes the count
        return bar

    def _rename_item(self, item):
        oi = int(item.data(_ORIG_ROLE))
        originals = self._row_originals if item.listWidget() is self.row_list \
            else self._col_originals
        cur = item.data(_OVERRIDE_ROLE) or originals[oi]
        text, ok = QtWidgets.QInputDialog.getText(self, "Rename label",
                                                  f"Label for {originals[oi]}:", text=str(cur))
        if not ok:
            return
        text = text.strip()
        item.setData(_OVERRIDE_ROLE, text or None)
        item.setText(text or originals[oi])
        item.setToolTip(f"renamed (was {originals[oi]})" if text else "")
        self._schedule_render()

    def _axis_from_list(self, lw) -> fe.AxisEdit:
        order, labels, dropped = [], {}, set()
        for row in range(lw.count()):
            it = lw.item(row)
            oi = int(it.data(_ORIG_ROLE))
            order.append(oi)
            if it.checkState() == QtCore.Qt.Unchecked:
                dropped.add(oi)
            ov = it.data(_OVERRIDE_ROLE)
            if ov:
                labels[oi] = str(ov)
        return fe.AxisEdit(order=order, labels=labels, dropped=dropped)

    def _on_data_change(self, *_):
        """Region or top-N changed → ask the provider for that region's ranked ions, reset the
        row axis to the fresh ranking (dropping stale per-row edits), and re-render."""
        if self._row_provider is None:
            return
        region = self.region_combo.currentText() if self.region_combo else None
        n = self.topn_spin.value() if self.topn_spin else None
        self._row_originals = [str(x) for x in self._row_provider(region, n)]
        self.spec.rows = fe.AxisEdit.identity(len(self._row_originals))
        self.spec.options["region"] = region
        if n is not None:
            self.spec.options["topn"] = int(n)
        self._populate(self.row_list, self._row_originals, self.spec.rows)
        self._render()

    def _rebuild_spec(self):
        self.spec.rows = self._axis_from_list(self.row_list)
        if self.col_list is not None:
            self.spec.cols = self._axis_from_list(self.col_list)
        self.spec.title = self.title_edit.text().strip() or None
        for key, combo in self._opt_combos.items():
            self.spec.options[key] = combo.currentData()
        if self.region_combo is not None:
            self.spec.options["region"] = self.region_combo.currentText()
        if self.topn_spin is not None:
            self.spec.options["topn"] = int(self.topn_spin.value())

    def _apply_sort(self, *_):
        """One-shot reorder of the ion rows by a chosen key, keeping current drops/renames."""
        si = self.sort_combo.currentData()
        if si is None:
            return
        s = self._row_sorts[si]
        self._rebuild_spec()                         # capture current drops/renames first
        self.spec.rows.sort_by(s["values"], s.get("descending", False))
        self._populate(self.row_list, self._row_originals, self.spec.rows)
        self.sort_combo.setCurrentIndex(0)           # the sort is a one-shot action
        self._render()

    # ----- render / zoom / actions ----------------------------------------- #
    def _schedule_render(self, *_):
        self._timer.start()

    # ----- style presets --------------------------------------------------- #
    def _refresh_presets(self, select=None):
        want = select or (self.preset_combo.currentText() if self.preset_combo.count() else "App Default")
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        for nm in stylelib.names():
            self.preset_combo.addItem(nm)
        i = self.preset_combo.findText(want)
        self.preset_combo.setCurrentIndex(i if i >= 0 else 0)
        self.preset_combo.blockSignals(False)

    def _on_preset_change(self, *_):
        self._style_preset = stylelib.get(self.preset_combo.currentText())
        self._render()

    def _copy_recipe(self):
        spec = stylelib.get(self.preset_combo.currentText())
        if spec is not None:
            QtWidgets.QApplication.clipboard().setText(spec.to_json())

    def _import_recipe(self):
        clip = QtWidgets.QApplication.clipboard().text()
        text, ok = QtWidgets.QInputDialog.getMultiLineText(
            self, "Import style recipe",
            "Paste a style recipe (JSON). It will be saved as a look you can reuse:", clip)
        if not ok or not text.strip():
            return
        try:
            spec = stylelib.import_json(text)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Import failed", f"Not a valid recipe:\n{e}")
            return
        self._refresh_presets(select=spec.name)
        self._style_preset = spec
        self._render()

    def _render(self):
        self._rebuild_spec()
        try:
            with export.active_style(self._style_preset):
                fig = self._render_cb(self.spec)
            self._pixmap = common.fig_to_pixmap(fig)
        except Exception as e:  # noqa: BLE001 — a bad arrangement must never crash the editor
            self._img_label.setText(f"Could not render: {e}")
            self._pixmap = None
            return
        self._apply_zoom()

    def _apply_zoom(self, *_):
        if self._pixmap is None:
            return
        z = self.zoom_combo.currentData()
        pm = self._pixmap
        if z is None:                                # Fit the whole figure in the viewport
            vp = self._scroll.viewport().size()
            target = pm.size().scaled(vp, QtCore.Qt.KeepAspectRatio)
        else:
            target = QtCore.QSize(int(pm.width() * z), int(pm.height() * z))
        if target.width() < 1 or target.height() < 1:
            target = pm.size()
        scaled = pm.scaled(target, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self._img_label.setPixmap(scaled)
        self._img_label.resize(scaled.size())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.zoom_combo.currentData() is None:    # re-fit on dialog resize
            self._apply_zoom()

    def _reset(self):
        n_cols = None if self._col_originals is None else len(self._col_originals)
        self.spec = fe.FigEditSpec.identity(len(self._row_originals), n_cols)
        for opt in self._style_options:
            self.spec.options[opt["key"]] = opt.get("default")
            combo = self._opt_combos.get(opt["key"])
            if combo is not None:
                combo.setCurrentIndex(max(0, combo.findData(opt.get("default"))))
        self.title_edit.blockSignals(True)
        self.title_edit.setText("")
        self.title_edit.blockSignals(False)
        self._populate(self.row_list, self._row_originals, self.spec.rows)
        if self.col_list is not None:
            self._populate(self.col_list, self._col_originals, self.spec.cols)
        self._render()

    def _export(self):
        self._rebuild_spec()
        path, _ = filedialogs.get_save_file_name(
            self, "Export figure", f"{self._default_name}.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;TIFF (*.tif)")
        if not path:
            return
        try:
            with export.active_style(self._style_preset):
                fig = self._render_cb(self.spec)
            export.save_figure(fig, path, dpi=300)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Export failed", str(e))
            return
        if self.parent() is not None and hasattr(self.parent(), "statusBar"):
            self.parent().statusBar().showMessage(f"Exported figure: {path}")
