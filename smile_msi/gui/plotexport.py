"""Publication export dialogs for the analytics plots (volcano, …) that were previously
exported as raw pyqtgraph **screenshots** with no options.

Each dialog pairs a small options panel with a **live matplotlib preview** rendered through
the very same ``export.render_*`` function it saves with — so the preview *is* the export —
and every option is honoured, including the shared **Style preset** (fonts / weights / theme /
palette) so an analytics figure matches the rest of a paper's figures. This is the "export
screen" the volcano/heatmap buttons open instead of jumping straight to a file dialog.
"""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets

from .. import export, stylelib
from . import common, filedialogs

_FMTS = [("PNG (raster)", "png"), ("TIFF (raster, lossless)", "tiff"),
         ("JPEG (raster)", "jpg"), ("PDF (vector)", "pdf"), ("SVG (vector)", "svg")]


def _col(res, name):
    """Float column ``name`` from a result DataFrame, or an empty array if absent."""
    try:
        return res[name].to_numpy(dtype=float)
    except Exception:  # noqa: BLE001
        return np.array([], dtype=float)


class FigureExportDialog(QtWidgets.QDialog):
    """Generic publication-figure **export screen**: a Style preset + declared options on the
    left, a **live matplotlib preview** on the right, and format/DPI output — the reusable
    replacement for the old "grab the pyqtgraph widget and save a screenshot" path.

    ``render(opts)`` is a caller callback that returns a :class:`matplotlib.figure.Figure`;
    ``opts`` is ``{option_key: value}`` for every declared option plus ``"dpi"``. The dialog
    wraps the call in :func:`export.active_style` for the chosen preset, so the shared look is
    applied without the caller doing anything. ``options`` is a list of control specs::

        {"key","label","kind": "combo"|"text"|"int"|"float"|"check",
         "default", ...kind-specific: choices=[(label,value)], min/max/step/suffix/special}

    Keeping the widget-building generic means each screenshot site becomes a few lines: declare
    its options, pass a render lambda that closes over its data. See the callers in this module.
    """

    def __init__(self, parent, *, render, options=None, default_name="figure.png",
                 title="Export figure", preview_dpi=110):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(1000, 660)
        self._render_cb = render
        self._default_name = default_name
        self._preview_dpi = int(preview_dpi)
        self._pixmap = None
        self._specs = list(options or [])
        self._controls = {}

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)

        controls = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(controls)
        cv.setContentsMargins(10, 10, 10, 10)
        cv.setSpacing(6)

        cv.addWidget(common.section_title("Style preset"))
        self.preset_combo = common.NoScrollComboBox()
        self.preset_combo.setToolTip(
            "A reusable look (fonts / type sizes / weights / theme) shared with every figure the "
            "app exports. Copy the recipe (JSON) to tweak elsewhere, then Import it back.")
        self._refresh_presets()
        self.preset_combo.activated.connect(self._schedule_render)
        cv.addWidget(self.preset_combo)
        prow = QtWidgets.QHBoxLayout()
        prow.setSpacing(4)
        prow.addWidget(common.button("Copy recipe", self._copy_recipe,
                                     tooltip="Copy this look's recipe (JSON) to the clipboard."))
        prow.addWidget(common.button("Import…", self._import_recipe,
                                     tooltip="Paste a recipe (JSON) to add it as a saved look."))
        cv.addLayout(prow)

        if self._specs:
            cv.addWidget(common.section_title("Figure"))
            form = QtWidgets.QFormLayout()
            form.setLabelAlignment(QtCore.Qt.AlignRight)
            for spec in self._specs:
                w = self._build_control(spec)
                form.addRow(spec.get("label", spec["key"]), w)
            cv.addLayout(form)

        cv.addWidget(common.section_title("Output"))
        oform = QtWidgets.QFormLayout()
        oform.setLabelAlignment(QtCore.Qt.AlignRight)
        self.fmt_combo = common.NoScrollComboBox()
        for label, ext in _FMTS:
            self.fmt_combo.addItem(label, ext)
        oform.addRow("Format", self.fmt_combo)
        self.dpi_spin = common.NoScrollSpinBox()
        self.dpi_spin.setRange(72, 1200)
        self.dpi_spin.setValue(300)
        self.dpi_spin.setSuffix(" dpi")
        oform.addRow("Resolution", self.dpi_spin)
        cv.addLayout(oform)
        cv.addStretch(1)
        split.addWidget(controls)

        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(common.note("Live preview — exactly what will be saved."))
        self._img = QtWidgets.QLabel()
        self._img.setAlignment(QtCore.Qt.AlignCenter)
        self._img.setMinimumWidth(420)
        rv.addWidget(self._img, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([320, 680])

        outer = QtWidgets.QVBoxLayout(self)
        outer.addWidget(split, 1)
        foot = QtWidgets.QHBoxLayout()
        foot.addStretch(1)
        foot.addWidget(common.button("Close", self.reject))
        foot.addWidget(common.primary_button("Export…", self._export, action="export"))
        outer.addLayout(foot)

        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._render)
        self._render()

    # ----- option controls ------------------------------------------------- #
    def _build_control(self, spec):
        kind = spec.get("kind", "text")
        default = spec.get("default")
        if kind == "combo":
            w = common.NoScrollComboBox()
            for clabel, cval in spec["choices"]:
                w.addItem(clabel, cval)
            i = w.findData(default)
            w.setCurrentIndex(max(0, i))
            w.activated.connect(self._schedule_render)
        elif kind == "check":
            w = QtWidgets.QCheckBox()
            w.setChecked(bool(default))
            w.toggled.connect(self._schedule_render)
        elif kind in ("int", "float"):
            w = common.NoScrollSpinBox() if kind == "int" else common.NoScrollDoubleSpinBox()
            w.setRange(spec.get("min", 0), spec.get("max", 100))
            if "step" in spec:
                w.setSingleStep(spec["step"])
            if kind == "float":
                w.setDecimals(spec.get("decimals", 1))
            if spec.get("suffix"):
                w.setSuffix(spec["suffix"])
            if spec.get("special"):
                w.setSpecialValueText(spec["special"])
            w.setValue(default if default is not None else 0)
            w.valueChanged.connect(self._schedule_render)
        else:                                       # text
            w = QtWidgets.QLineEdit(str(default) if default is not None else "")
            w.textEdited.connect(self._schedule_render)
        if spec.get("tooltip"):
            w.setToolTip(spec["tooltip"])
        if spec.get("enabled") is False:
            w.setEnabled(False)
        self._controls[spec["key"]] = (w, kind)
        return w

    def _value(self, key):
        w, kind = self._controls[key]
        if kind == "combo":
            return w.currentData()
        if kind == "check":
            return w.isChecked()
        if kind in ("int", "float"):
            return w.value()
        return w.text().strip()

    def opts(self):
        return {key: self._value(key) for key in self._controls}

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
        self._render()

    # ----- render / export ------------------------------------------------- #
    def _schedule_render(self, *_):
        self._timer.start()

    def _figure(self, dpi):
        spec = stylelib.get(self.preset_combo.currentText())
        opts = self.opts()
        opts["dpi"] = int(dpi)
        with export.active_style(spec):
            return self._render_cb(opts)

    def _render(self):
        try:
            self._pixmap = common.fig_to_pixmap(self._figure(self._preview_dpi))
        except Exception as e:  # noqa: BLE001 — a bad option combo must never crash the dialog
            self._img.setText(f"Could not render: {e}")
            self._pixmap = None
            return
        self._fit()

    def _fit(self):
        if self._pixmap is not None:
            self._img.setPixmap(self._pixmap.scaled(self._img.size(), QtCore.Qt.KeepAspectRatio,
                                                    QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def _export(self):
        ext = self.fmt_combo.currentData()
        base = self._default_name.rsplit(".", 1)[0]
        path, _ = filedialogs.get_save_file_name(
            self, self.windowTitle(), f"{base}.{ext}", f"{ext.upper()} (*.{ext})")
        if not path:
            return
        if not path.lower().endswith("." + ext):
            path += "." + ext
        try:
            fig = self._figure(int(self.dpi_spin.value()))
            export.save_figure(fig, path, dpi=int(self.dpi_spin.value()), fmt=ext)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Export failed", str(e))
            return
        par = self.parent()
        if par is not None and hasattr(par, "statusBar"):
            par.statusBar().showMessage(f"Wrote {path}")
        self.accept()


class EmbeddingExportDialog(FigureExportDialog):
    """Export screen for a 2-D embedding / scores scatter, rendered via
    :func:`smile_msi.umapstudio.render_figure` (density atlas or scatter, coloured by a chosen
    categorical channel) — the publication-quality replacement for the raw pyqtgraph screenshot
    of a cohort-UMAP / component / discriminant-scores cloud. ``data`` is a
    :class:`smile_msi.umapstudio.EmbeddingData`."""

    def __init__(self, parent, data, *, default_color_by=None, default_name="embedding.png",
                 title="Export embedding", default_theme="light"):
        from .. import umapstudio as us
        channels = data.channels() or []
        color_by0 = default_color_by if default_color_by in channels else (channels[0] if channels else "")

        def render(opts):
            spec = us.UMAPStudioSpec(
                panels=[us.PanelSpec(color_by=opts.get("color_by", ""))],
                layout="single",
                mode=(us.MODE_SCATTER if opts["mode"] == "scatter" else us.MODE_DENSITY),
                background=("black" if opts["theme"] == "dark" else "white"),
                axis_style=opts["axis"], legend=True, dpi=int(opts["dpi"]))
            return us.render_figure(data, spec)

        options = []
        if channels:
            options.append({"key": "color_by", "label": "Colour by", "kind": "combo",
                            "choices": [(c, c) for c in channels], "default": color_by0})
        options += [
            {"key": "mode", "label": "Style", "kind": "combo",
             "choices": [("Density (atlas)", "density"), ("Dots (scatter)", "scatter")],
             "default": "density"},
            {"key": "axis", "label": "Axes", "kind": "combo",
             "choices": [("Corner labels", us.AXIS_CORNER), ("Full labels", us.AXIS_LABELED),
                         ("None", us.AXIS_OFF)], "default": us.AXIS_CORNER},
            {"key": "theme", "label": "Background", "kind": "combo",
             "choices": [("Light", "light"), ("Dark", "dark")], "default": default_theme},
        ]
        super().__init__(parent, render=render, options=options, default_name=default_name,
                         title=title, preview_dpi=96)


class VolcanoExportDialog(QtWidgets.QDialog):
    """Options + live preview for exporting a **volcano plot** as a publication figure.

    ``res`` is the differential-comparison DataFrame (``log2_fc`` / ``p_value`` / ``q_value`` /
    ``mz`` / ``best_lipid``) the Cohort and Region-comparison tabs already build.
    """

    def __init__(self, parent, res, *, a_label="A", b_label="B", default_name="volcano.png"):
        super().__init__(parent)
        self.setWindowTitle("Export volcano plot")
        self.setModal(True)
        self.resize(1000, 660)
        self._res = res
        self._default_name = default_name
        self._pixmap = None

        # pull the arrays once (the render is otherwise pure)
        self._fc = _col(res, "log2_fc")
        self._p = _col(res, "p_value")
        self._q = _col(res, "q_value")
        self._mz = _col(res, "mz")
        try:
            self._labels = [str(x) if x else "" for x in res.get("best_lipid", [])]
        except Exception:  # noqa: BLE001
            self._labels = []

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)

        # ---- left: options ---------------------------------------------------- #
        controls = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(controls)
        cv.setContentsMargins(10, 10, 10, 10)
        cv.setSpacing(6)

        cv.addWidget(common.section_title("Style preset"))
        self.preset_combo = common.NoScrollComboBox()
        self.preset_combo.setToolTip(
            "A reusable look (fonts / type sizes / weights / theme) shared with every figure the "
            "app exports. Copy the recipe (JSON) to tweak elsewhere, then Import it back.")
        self._refresh_presets()
        self.preset_combo.activated.connect(self._schedule_render)
        cv.addWidget(self.preset_combo)
        prow = QtWidgets.QHBoxLayout()
        prow.setSpacing(4)
        prow.addWidget(common.button("Copy recipe", self._copy_recipe,
                                     tooltip="Copy this look's recipe (JSON) to the clipboard."))
        prow.addWidget(common.button("Import…", self._import_recipe,
                                     tooltip="Paste a recipe (JSON) to add it as a saved look."))
        cv.addLayout(prow)

        cv.addWidget(common.section_title("Figure"))
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignRight)

        self.theme_combo = common.NoScrollComboBox()
        self.theme_combo.addItems(["Light", "Dark"])
        self.theme_combo.activated.connect(self._schedule_render)
        form.addRow("Theme", self.theme_combo)

        self.title_edit = QtWidgets.QLineEdit("Differential abundance")
        self.title_edit.textEdited.connect(self._schedule_render)
        form.addRow("Title", self.title_edit)

        self.a_edit = QtWidgets.QLineEdit(str(a_label or "A"))
        self.a_edit.textEdited.connect(self._schedule_render)
        form.addRow("Group A", self.a_edit)
        self.b_edit = QtWidgets.QLineEdit(str(b_label or "B"))
        self.b_edit.textEdited.connect(self._schedule_render)
        form.addRow("Group B", self.b_edit)

        self.label_combo = common.NoScrollComboBox()
        for txt, n in [("No labels", 0), ("Top 5 hits", 5), ("Top 8 hits", 8),
                       ("Top 12 hits", 12), ("Top 20 hits", 20)]:
            self.label_combo.addItem(txt, n)
        self.label_combo.setCurrentIndex(2)      # Top 8
        self.label_combo.activated.connect(self._schedule_render)
        form.addRow("Label hits", self.label_combo)

        self.labelby_combo = common.NoScrollComboBox()
        self.labelby_combo.addItem("m/z", "mz")
        self.labelby_combo.addItem("Lipid name", "label")
        self.labelby_combo.setEnabled(bool(any(self._labels)))
        self.labelby_combo.activated.connect(self._schedule_render)
        form.addRow("Label by", self.labelby_combo)

        self.fc_spin = common.NoScrollDoubleSpinBox()
        self.fc_spin.setRange(0.0, 10.0)
        self.fc_spin.setSingleStep(0.5)
        self.fc_spin.setDecimals(1)
        self.fc_spin.setSpecialValueText("off")
        self.fc_spin.setToolTip("Draw fold-change guide lines at ±this log₂ FC (0 = off).")
        self.fc_spin.valueChanged.connect(self._schedule_render)
        form.addRow("Fold-change guide", self.fc_spin)

        cv.addLayout(form)

        cv.addWidget(common.section_title("Output"))
        oform = QtWidgets.QFormLayout()
        oform.setLabelAlignment(QtCore.Qt.AlignRight)
        self.fmt_combo = common.NoScrollComboBox()
        for label, ext in _FMTS:
            self.fmt_combo.addItem(label, ext)
        oform.addRow("Format", self.fmt_combo)
        self.dpi_spin = common.NoScrollSpinBox()
        self.dpi_spin.setRange(72, 1200)
        self.dpi_spin.setValue(300)
        self.dpi_spin.setSuffix(" dpi")
        oform.addRow("Resolution", self.dpi_spin)
        cv.addLayout(oform)
        cv.addStretch(1)
        split.addWidget(controls)

        # ---- right: live preview --------------------------------------------- #
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(common.note("Live preview — exactly what will be saved."))
        self._img = QtWidgets.QLabel()
        self._img.setAlignment(QtCore.Qt.AlignCenter)
        self._img.setMinimumWidth(420)
        rv.addWidget(self._img, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([320, 680])

        outer = QtWidgets.QVBoxLayout(self)
        outer.addWidget(split, 1)
        foot = QtWidgets.QHBoxLayout()
        foot.addStretch(1)
        foot.addWidget(common.button("Close", self.reject))
        foot.addWidget(common.primary_button("Export…", self._export, action="export"))
        outer.addLayout(foot)

        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._render)
        self._render()

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
        self._render()

    # ----- render ---------------------------------------------------------- #
    def _schedule_render(self, *_):
        self._timer.start()

    def _opts(self):
        return dict(
            a_label=self.a_edit.text().strip() or "A",
            b_label=self.b_edit.text().strip() or "B",
            title=self.title_edit.text().strip(),
            annotate_top=int(self.label_combo.currentData() or 0),
            annotate_by=self.labelby_combo.currentData(),
            fc_line=(float(self.fc_spin.value()) or None),
            theme=("dark" if self.theme_combo.currentText() == "Dark" else "light"),
        )

    def _make_figure(self, dpi):
        spec = stylelib.get(self.preset_combo.currentText())
        with export.active_style(spec):
            return export.render_volcano_figure(
                self._fc, self._p, self._q, mz=self._mz, labels=self._labels,
                dpi=dpi, **self._opts())

    def _render(self):
        try:
            fig = self._make_figure(dpi=110)          # preview dpi
            self._pixmap = common.fig_to_pixmap(fig)
        except Exception as e:  # noqa: BLE001 — a bad option combo must never crash the dialog
            self._img.setText(f"Could not render: {e}")
            self._pixmap = None
            return
        self._fit()

    def _fit(self):
        if self._pixmap is None:
            return
        vp = self._img.size()
        self._img.setPixmap(self._pixmap.scaled(vp, QtCore.Qt.KeepAspectRatio,
                                                QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def _export(self):
        ext = self.fmt_combo.currentData()
        base = self._default_name.rsplit(".", 1)[0]
        path, _ = filedialogs.get_save_file_name(
            self, "Export volcano plot", f"{base}.{ext}",
            f"{ext.upper()} (*.{ext})")
        if not path:
            return
        if not path.lower().endswith("." + ext):
            path += "." + ext
        try:
            fig = self._make_figure(dpi=int(self.dpi_spin.value()))
            export.save_figure(fig, path, dpi=int(self.dpi_spin.value()), fmt=ext)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Export failed", str(e))
            return
        par = self.parent()
        if par is not None and hasattr(par, "statusBar"):
            par.statusBar().showMessage(f"Wrote {path}")
        self.accept()
