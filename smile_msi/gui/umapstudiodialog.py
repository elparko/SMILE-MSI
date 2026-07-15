"""UMAP Studio — an interactive editor for publication-quality embedding figures.

This is the on-screen twin of :mod:`smile_msi.umapstudio`: the dialog embeds a Matplotlib
canvas that previews the *exact* :class:`~matplotlib.figure.Figure` it will export (WYSIWYG,
the same contract the ion-image overlay keeps via :mod:`smile_msi.annotations`). You pick how
each panel is coloured (donor / group / tissue-type label / a lipid-feature intensity), tune
the density compositing (palette, alpha mapping, spread, background), choose a 1-/2-/4-panel
layout, drag on free-text cluster labels (double-click to rename), and export to PDF / PNG /
SVG / TIFF at publication DPI.

The heavy lifting — turning millions of points into a density raster and assembling the figure
— lives in the pure engine; this module is just Qt glue + the live interactions. It is opened
from the Cohort UMAP tab ("Edit & annotate…") on a freshly-pooled embedding, and from
**File ▸ UMAP Studio…** (⇧⌘U) on the last embedding computed this session.
"""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets

from .. import umapstudio as us
from .. import export, prefs
from .common import (note, section_title, primary_button, icon_button, ControlBar,
                     NoScrollComboBox, NoScrollDoubleSpinBox, NoScrollSpinBox)
from . import filedialogs

PREFS_KEY = "umap_studio_spec"

# A short menu of good sequential colormaps for continuous (feature-intensity) panels.
CONT_CMAPS = ["viridis", "magma", "inferno", "plasma", "cividis", "turbo"]
LAYOUT_CHOICES = [("Single panel", "single"), ("Two panels (A, B)", "two"),
                  ("Four panels (A–D)", "four")]
MODE_CHOICES = [("Scatter (crisp dots)", us.MODE_SCATTER), ("Density (atlas cloud)", us.MODE_DENSITY)]
HOW_CHOICES = [("Equalized (eq_hist)", us.HOW_EQ_HIST), ("Linear", us.HOW_LINEAR),
               ("Log", us.HOW_LOG)]
AXIS_CHOICES = [("Corner labels", us.AXIS_CORNER), ("Full axes", us.AXIS_LABELED),
                ("None", us.AXIS_OFF)]
PALETTE_CHOICES = [("Auto", "auto"), ("Glasbey (many)", "glasbey"), ("tab10", "tab10"),
                   ("Okabe–Ito (colourblind-safe)", "okabe_ito"), ("Set2", "set2")]
PICK_FEATURE = "__pick_feature__"               # sentinel item in a colour-by combo


def _leading_mz(name: str):
    """The m/z that a saved feature/lipid channel name starts with (``"744.5500 m/z · …"`` →
    744.55), or ``None`` if the name isn't a feature channel."""
    try:
        return float(str(name).split()[0])
    except (ValueError, IndexError):
        return None


class UMAPStudioDialog(QtWidgets.QDialog):
    """Live editor over an :class:`smile_msi.umapstudio.EmbeddingData`."""

    def __init__(self, win, data: us.EmbeddingData, spec: us.UMAPStudioSpec | None = None):
        super().__init__(win)
        self.win = win
        self.data = data
        self.spec = spec or self._restore_spec()
        # internal pool of 4 panel specs (the spec exposes only the first n for the layout).
        # Seed the extra slots with the *second* categorical channel (e.g. Group) when there
        # is one, so switching to a multi-panel layout reads like the atlas (donor vs group)
        # out of the box instead of four identical panels.
        self._panels = list(self.spec.panels)[:4]
        cats = list(data.categorical)
        alt = cats[1] if len(cats) > 1 else self._default_channel()
        while len(self._panels) < 4:
            self._panels.append(us.PanelSpec(color_by=(alt if len(self._panels) % 2 else
                                                       self._default_channel())))
        self._selected = None
        self._color_keys: dict = {}              # stable category→colour across tweaks
        self._drag = None                        # (artist, annotation) while dragging
        self._feature_labels = (
            [data.feature_label(j) for j in range(len(data.feature_mz))]
            if data.features is not None and data.feature_mz is not None else [])

        self.setWindowTitle("UMAP Studio")
        self.setMinimumSize(1040, 680)
        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(note(
            "Tune the look, colour each panel, and drag on labels (double-click to rename, "
            "Delete to remove) — then export. The preview is exactly what you export."))

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self._build_canvas_pane())
        ctrl = self._build_controls_pane()
        split.addWidget(ctrl)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([720, 320])
        root.addWidget(split, 1)
        root.addWidget(self._build_footer())

        self._sync_controls_from_spec()
        self._render()

    # ------------------------------------------------------------------ panes
    def _build_canvas_pane(self):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        box = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(6, 5))
        self.canvas = FigureCanvasQTAgg(self.fig)       # reused across renders
        self.canvas.setMinimumSize(480, 420)
        self.toolbar = NavigationToolbar2QT(self.canvas, box)
        v.addWidget(self.toolbar)
        v.addWidget(self.canvas, 1)
        # interactions: drag + rename annotations
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        # the widget owns the figure size (render uses resize=False); re-lay-out on resize so the
        # legend/colorbar margin stays a true ~2 in of the new width, not a stale fraction
        self.canvas.mpl_connect("resize_event", self._schedule_render)
        return box

    def _build_controls_pane(self):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        inner = QtWidgets.QWidget()
        scroll.setWidget(inner)
        v = QtWidgets.QVBoxLayout(inner)
        v.setSpacing(10)

        # --- layout + per-panel colour-by ---------------------------------- #
        v.addWidget(section_title("Layout & panels"))
        self.layout_combo = self._combo(LAYOUT_CHOICES, self._on_layout_changed)
        v.addLayout(self._labeled("Layout", self.layout_combo))
        self._panel_rows = []
        for i in range(4):
            row = self._build_panel_row(i)
            v.addWidget(row)
            self._panel_rows.append(row)

        # --- appearance ---------------------------------------------------- #
        v.addWidget(section_title("Appearance"))
        self.mode_combo = self._combo(MODE_CHOICES, self._on_mode_changed)
        self.mode_combo.setToolTip(
            "Scatter draws crisp coloured dots + legend (like the live plot — best for region/"
            "sample means and modest pixel counts). Density paints the atlas cloud (best for "
            "millions of pooled pixels, where dots overplot into a blob).")
        v.addLayout(self._labeled("Style", self.mode_combo))
        self.palette_combo = self._combo(PALETTE_CHOICES, self._on_palette_changed)
        v.addLayout(self._labeled("Palette", self.palette_combo))
        # --- scatter-only ---
        self.psize = NoScrollDoubleSpinBox(); self.psize.setRange(1.0, 200.0)
        self.psize.setSingleStep(2.0)
        self.psize.setToolTip("Marker size for the scatter dots.")
        self.psize.valueChanged.connect(self._schedule_render)
        self._row_psize = self._labeled_row("Point size", self.psize)
        v.addWidget(self._row_psize)
        self.palpha = NoScrollDoubleSpinBox(); self.palpha.setRange(0.05, 1.0)
        self.palpha.setSingleStep(0.05)
        self.palpha.setToolTip("Dot opacity — lower it when many dots overlap.")
        self.palpha.valueChanged.connect(self._schedule_render)
        self._row_palpha = self._labeled_row("Point alpha", self.palpha)
        v.addWidget(self._row_palpha)
        # --- density-only ---
        self.how_combo = self._combo(HOW_CHOICES, self._schedule_render)
        self._row_how = self._labeled_row("Density", self.how_combo)
        v.addWidget(self._row_how)
        self.minalpha = NoScrollDoubleSpinBox()
        self.minalpha.setRange(0.0, 1.0); self.minalpha.setSingleStep(0.02)
        self.minalpha.setToolTip("Floor opacity for the faintest occupied pixel — raise it so "
                                 "lone outliers stay visible.")
        self.minalpha.valueChanged.connect(self._schedule_render)
        self._row_minalpha = self._labeled_row("Min alpha", self.minalpha)
        v.addWidget(self._row_minalpha)
        self.spread = NoScrollSpinBox(); self.spread.setRange(0, 5)
        self.spread.setSuffix(" px")
        self.spread.setToolTip("Fatten lone points so they survive print downscaling "
                               "(datashader's dynspread idea). 0 = off, for dense atlases.")
        self.spread.valueChanged.connect(self._schedule_render)
        self._row_spread = self._labeled_row("Spread", self.spread)
        v.addWidget(self._row_spread)
        # --- both modes ---
        self.bg_combo = self._combo([("White (print)", "white"), ("Black (screen)", "black"),
                                     ("Transparent", "transparent")], self._schedule_render)
        v.addLayout(self._labeled("Background", self.bg_combo))
        self.res = NoScrollSpinBox(); self.res.setRange(200, 3000); self.res.setSingleStep(100)
        self.res.setToolTip("Density-raster resolution (output pixels per side). Higher = "
                            "crisper, slower.")
        self.res.valueChanged.connect(self._schedule_render)
        self._row_res = self._labeled_row("Resolution", self.res)
        v.addWidget(self._row_res)
        self.axis_combo = self._combo(AXIS_CHOICES, self._schedule_render)
        v.addLayout(self._labeled("Axes", self.axis_combo))
        self.legend_chk = QtWidgets.QCheckBox("Show legend / colourbar")
        self.legend_chk.toggled.connect(self._schedule_render)
        v.addWidget(self.legend_chk)

        # --- per-category colours ------------------------------------------ #
        # The palette sets every category's colour at once; this lets you override an
        # individual one (e.g. paint "Endoneurium" your lab's signature blue). Overrides
        # are remembered in the spec and win over the palette in every panel + the legend.
        self._color_section = QtWidgets.QWidget()
        cs = QtWidgets.QVBoxLayout(self._color_section)
        cs.setContentsMargins(0, 0, 0, 0)
        cs.setSpacing(6)
        cs.addWidget(section_title("Category colours"))
        self.color_channel_combo = NoScrollComboBox()
        self.color_channel_combo.setToolTip("Which categorical channel to recolour "
                                            "(region, sample, group, cluster…).")
        self.color_channel_combo.activated.connect(lambda *_: self._refresh_color_swatches())
        cs.addLayout(self._labeled("Channel", self.color_channel_combo))
        self._color_swatch_box = QtWidgets.QVBoxLayout()
        self._color_swatch_box.setSpacing(3)
        cs.addLayout(self._color_swatch_box)
        b_reset_col = QtWidgets.QPushButton("Reset to palette")
        b_reset_col.setToolTip("Drop every custom colour on this channel and fall back to "
                               "the palette.")
        b_reset_col.clicked.connect(self._reset_channel_colours)
        cs.addWidget(b_reset_col)
        v.addWidget(self._color_section)

        # --- annotate ------------------------------------------------------ #
        v.addWidget(section_title("Annotate"))
        b_add = icon_button("add", "Add text label", slot=self._add_label,
                            tooltip="Drop a label on the figure — drag to place it, "
                                    "double-click to rename, Delete to remove.")
        v.addWidget(b_add)
        # Text size sets the size of the selected label live, and the starting size for the
        # next label added (the property-panel idiom — reflects the selection, edits it).
        self.fontsize_spin = NoScrollDoubleSpinBox()
        self.fontsize_spin.setRange(4.0, 72.0); self.fontsize_spin.setSingleStep(1.0)
        self.fontsize_spin.setDecimals(0); self.fontsize_spin.setSuffix(" pt")
        self.fontsize_spin.setValue(us.Annotation.fontsize)
        self.fontsize_spin.setToolTip("Text size for labels. Click a label to select it, then "
                                      "change this to resize it; new labels start at this size.")
        self.fontsize_spin.valueChanged.connect(self._on_fontsize_changed)
        v.addLayout(self._labeled("Text size", self.fontsize_spin))
        v.addWidget(note("Drag a label to move it · double-click to rename · click to select "
                         "then set Text size · select + Delete to remove. Panel letters (A–D) "
                         "are added automatically in multi-panel layouts."))
        v.addStretch(1)
        return scroll

    def _build_panel_row(self, i):
        box = QtWidgets.QGroupBox()
        g = QtWidgets.QVBoxLayout(box)
        g.setContentsMargins(8, 6, 8, 6)
        g.setSpacing(4)
        g.addWidget(section_title(f"Panel {'ABCD'[i]}"))
        combo = NoScrollComboBox()
        combo.setToolTip("What this panel colours pixels by.")
        combo.activated.connect(lambda _=0, idx=i: self._on_color_by(idx))
        cmap = NoScrollComboBox()
        cmap.addItems(CONT_CMAPS)
        cmap.setToolTip("Colormap for a continuous (feature-intensity) channel.")
        cmap.activated.connect(lambda _=0, idx=i: self._on_cmap(idx))
        title = QtWidgets.QLineEdit()
        title.setPlaceholderText("Panel title (optional)")
        title.editingFinished.connect(lambda idx=i: self._on_title(idx))
        g.addLayout(self._labeled("Colour by", combo))
        g.addLayout(self._labeled("Colormap", cmap))
        g.addWidget(title)
        box._combo, box._cmap, box._title = combo, cmap, title
        return box

    def _build_footer(self):
        bar = ControlBar()
        self.fmt_combo = NoScrollComboBox()
        self.fmt_combo.addItems(["PDF", "PNG", "SVG", "TIFF"])
        self.fmt_combo.setToolTip("Vector PDF/SVG keep text crisp; the dense point layer is "
                                  "embedded as a high-DPI raster either way.")
        self.dpi = NoScrollSpinBox(); self.dpi.setRange(72, 1200); self.dpi.setValue(600)
        self.dpi.setSuffix(" dpi")
        b_export = primary_button("Export figure…", self._export, action="export",
                                  tooltip="Render the figure to disk at the chosen format/DPI.")
        b_close = QtWidgets.QPushButton("Close"); b_close.clicked.connect(self.accept)
        bar.add_group("Format", self.fmt_combo)
        bar.add_group("DPI", self.dpi)
        bar.add(b_export, b_close)
        self.status = bar.set_status("")
        return bar

    # ------------------------------------------------------------- small helpers
    @staticmethod
    def _labeled(label, widget):
        row = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(label)
        lbl.setMinimumWidth(86)
        row.addWidget(lbl)
        row.addWidget(widget, 1)
        return row

    def _labeled_row(self, label, widget):
        """A labelled row wrapped in a QWidget so the whole row can be shown/hidden (used for
        the mode-specific controls that only apply to scatter or to density)."""
        box = QtWidgets.QWidget()
        box.setLayout(self._labeled(label, widget))
        box.layout().setContentsMargins(0, 0, 0, 0)
        return box

    def _combo(self, choices, slot):
        c = NoScrollComboBox()
        for label, key in choices:
            c.addItem(label, key)
        c.activated.connect(lambda *_: slot())
        return c

    def _default_channel(self):
        cats = list(self.data.categorical)
        return cats[0] if cats else (list(self.data.continuous)[:1] or [""])[0]

    def _n_panels(self):
        return us.LAYOUTS.get(self.layout_combo.currentData(), us.LAYOUTS["single"])[1]

    # --------------------------------------------------------- spec <-> controls
    def _restore_spec(self) -> us.UMAPStudioSpec:
        saved = prefs.get(PREFS_KEY)
        if saved:
            try:
                spec = us.UMAPStudioSpec.from_dict(saved)
                # a remembered channel may not exist on this embedding — re-materialise a saved
                # lipid/feature channel by m/z, else fall back cleanly (never a blank panel).
                for p in spec.panels:
                    p.color_by = self._resolve_saved_channel(p.color_by)
                spec.annotations = []            # don't resurrect labels onto a new embedding
                spec.legend_xy = {}              # …nor a dragged legend spot from another figure
                return spec
            except Exception:  # noqa: BLE001 — corrupt/old prefs → defaults
                pass
        first = self._default_channel()
        return us.UMAPStudioSpec(panels=[us.PanelSpec(color_by=first)])

    def _resolve_saved_channel(self, name: str) -> str:
        """Map a remembered ``color_by`` onto THIS embedding: keep it if the channel exists,
        re-materialise a saved lipid/feature channel by matching its m/z, else fall back to a
        real channel — so a stale remembered channel can never blank the figure."""
        if not name:
            return self._default_channel()
        if self.data.is_categorical(name) or name in self.data.continuous:
            return name
        mz = _leading_mz(name)
        if mz is not None and self.data.features is not None \
                and self.data.feature_mz is not None and len(self.data.feature_mz):
            fmz = np.asarray(self.data.feature_mz, dtype=float)
            j = int(np.argmin(np.abs(fmz - mz)))
            if abs(float(fmz[j]) - mz) <= 0.01:        # same feature within 0.01 Da
                return self.data.set_feature_channel(j)
        return self._default_channel()

    def _sync_controls_from_spec(self):
        self._block(True)
        self._set_combo(self.layout_combo, self.spec.layout)
        self._set_combo(self.mode_combo, self.spec.mode)
        self._set_combo(self.palette_combo, self.spec.palette)
        self._set_combo(self.how_combo, self.spec.how)
        self._set_combo(self.bg_combo, self.spec.background)
        self._set_combo(self.axis_combo, self.spec.axis_style)
        self.minalpha.setValue(self.spec.min_alpha)
        self.spread.setValue(int(self.spec.spread_px))
        self.psize.setValue(float(self.spec.point_size))
        self.palpha.setValue(float(self.spec.point_alpha))
        self.res.setValue(int(self.spec.resolution))
        self.legend_chk.setChecked(bool(self.spec.legend))
        self.dpi.setValue(int(self.spec.dpi))
        self._refresh_panel_combos()
        self._refresh_color_channels()
        self._update_panel_visibility()
        self._update_mode_visibility()
        self._block(False)

    def _block(self, on):
        for w in (getattr(self, n, None) for n in (
                "layout_combo", "mode_combo", "palette_combo", "how_combo", "bg_combo",
                "axis_combo", "minalpha", "spread", "psize", "palpha", "res", "legend_chk",
                "dpi")):
            if w is not None:
                w.blockSignals(on)

    @staticmethod
    def _set_combo(combo, key):
        i = combo.findData(key)
        if i >= 0:
            combo.setCurrentIndex(i)

    def _refresh_panel_combos(self):
        cats = list(self.data.categorical)
        conts = list(self.data.continuous)
        for i, row in enumerate(self._panel_rows):
            combo = row._combo
            combo.blockSignals(True)
            combo.clear()
            for name in cats:
                combo.addItem(f"{name}  (categories)", name)
            for name in conts:
                combo.addItem(f"{name}  (intensity)", name)
            if self._feature_labels:
                combo.addItem("＋ Colour by feature / lipid…", PICK_FEATURE)
            self._set_combo(combo, self._panels[i].color_by)
            row._cmap.blockSignals(True)
            ci = row._cmap.findText(self._panels[i].cmap)
            row._cmap.setCurrentIndex(ci if ci >= 0 else 0)
            row._cmap.blockSignals(False)
            row._title.setText(self._panels[i].title)
            combo.blockSignals(False)

    def _update_panel_visibility(self):
        n = self._n_panels()
        for i, row in enumerate(self._panel_rows):
            row.setVisible(i < n)
            row._cmap.setVisible(self.data.is_categorical(self._panels[i].color_by) is False)

    def _build_spec(self) -> us.UMAPStudioSpec:
        n = self._n_panels()
        self.spec.panels = [self._panels[i] for i in range(n)]
        for i in range(n):                       # auto panel letters for multi-panel
            self._panels[i].panel_letter = "ABCD"[i] if n > 1 else ""
        self.spec.layout = self.layout_combo.currentData()
        self.spec.mode = self.mode_combo.currentData()
        self.spec.palette = self.palette_combo.currentData()
        self.spec.how = self.how_combo.currentData()
        self.spec.background = self.bg_combo.currentData()
        self.spec.axis_style = self.axis_combo.currentData()
        self.spec.min_alpha = float(self.minalpha.value())
        self.spec.spread_px = int(self.spread.value())
        self.spec.point_size = float(self.psize.value())
        self.spec.point_alpha = float(self.palpha.value())
        self.spec.resolution = int(self.res.value())
        self.spec.legend = self.legend_chk.isChecked()
        self.spec.dpi = int(self.dpi.value())
        self.spec.method = self.data.method
        return self.spec

    # --------------------------------------------------------------- rendering
    def _schedule_render(self, *_):
        # debounce: coalesce rapid control changes into one redraw
        if not hasattr(self, "_render_timer"):
            self._render_timer = QtCore.QTimer(self)
            self._render_timer.setSingleShot(True)
            self._render_timer.timeout.connect(self._render)
        self._render_timer.start(140)

    def _render(self):
        spec = self._build_spec()
        try:
            # resize=False: this is the live embedded canvas, sized by its Qt widget. Forcing an
            # absolute figsize would leave the Agg buffer smaller than the widget and paint sheared
            # garbage in the uncovered strip (see render_figure's docstring).
            us.render_figure(self.data, spec, color_keys=self._color_keys, fig=self.fig,
                             resize=False)
            # keep the keys the engine resolved so colours stay stable across tweaks
            self._color_keys = dict(getattr(self.fig, "_umap_color_keys", {}) or {})
            self._ann_artists = list(getattr(self.fig, "_umap_annotation_artists", []))
            # every draggable element (text labels, panel titles / A–D letters, legends)
            self._draggables = list(getattr(self.fig, "_umap_draggables", []))
            self.canvas.draw_idle()
            self.status.setText(self._status_text())
        except Exception as e:  # noqa: BLE001 — surface render errors in-dialog, never crash
            self.status.setText(f"Render error: {e}")

    def _status_text(self):
        n = int(np.asarray(self.data.coords).shape[0])
        emb = self.data.method
        return (f"{n:,} points · {emb} · {self._n_panels()} panel(s). "
                "Drag labels to place · double-click to rename.")

    # --------------------------------------------------------- control handlers
    def _on_layout_changed(self):
        self._update_panel_visibility()
        self._schedule_render()

    def _on_mode_changed(self):
        self._update_mode_visibility()
        self._schedule_render()

    def _update_mode_visibility(self):
        scatter = (self.mode_combo.currentData() == us.MODE_SCATTER)
        for w in (self._row_psize, self._row_palpha):
            w.setVisible(scatter)
        for w in (self._row_how, self._row_minalpha, self._row_spread, self._row_res):
            w.setVisible(not scatter)

    def _on_palette_changed(self):
        self._color_keys = {}                    # palette changed → rebuild colour keys
        self._refresh_color_swatches()
        self._schedule_render()

    # ----------------------------------------------------- per-category colours
    def _effective_color_key(self, ch):
        """The channel's category→hex map as it will render: the palette assignment with
        any user overrides laid on top. Independent of render timing so the swatches are
        right the moment the dialog opens."""
        key = us.build_color_key(self.data.categorical[ch], self.spec.palette)
        key.update({c: h for c, h in (self.spec.color_overrides.get(ch) or {}).items()
                    if c in key and h})
        return key

    def _refresh_color_channels(self):
        """Populate the channel chooser with every categorical channel; hide the whole
        section when the embedding has none (a pure feature-intensity figure)."""
        cats = [c for c in self.data.categorical]
        self._color_section.setVisible(bool(cats))
        combo = self.color_channel_combo
        prev = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(cats)
        i = combo.findText(prev)
        combo.setCurrentIndex(i if i >= 0 else 0)
        combo.blockSignals(False)
        self._refresh_color_swatches()

    def _refresh_color_swatches(self):
        """Rebuild the swatch rows for the chosen channel — one clickable colour chip per
        category, showing the colour it will render with."""
        box = self._color_swatch_box
        while box.count():
            it = box.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        ch = self.color_channel_combo.currentText()
        if not ch or ch not in self.data.categorical:
            return
        key = self._effective_color_key(ch)
        for cat, hexcol in key.items():
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            swatch = QtWidgets.QPushButton()
            swatch.setFixedSize(22, 18)
            swatch.setStyleSheet(
                f"background:{hexcol}; border:1px solid palette(mid); border-radius:3px;")
            swatch.setToolTip("Click to choose a custom colour for this category.")
            swatch.clicked.connect(lambda _=0, c=cat, ce=ch: self._pick_category_colour(ce, c))
            lbl = QtWidgets.QLabel(str(cat))
            lbl.setToolTip(str(cat))
            row.addWidget(swatch)
            row.addWidget(lbl, 1)
            holder = QtWidgets.QWidget()
            holder.setLayout(row)
            box.addWidget(holder)

    def _pick_category_colour(self, ch, cat):
        from . import colorpicker
        cur = self._effective_color_key(ch).get(cat, "#888888")
        hexc = colorpicker.pick_color(self, initial=cur, title=f"Colour for “{cat}”")
        if not hexc:
            return
        self.spec.color_overrides.setdefault(ch, {})[cat] = hexc
        self._color_keys = {}                    # force the engine to re-resolve with the override
        self._refresh_color_swatches()
        self._schedule_render()

    def _reset_channel_colours(self):
        ch = self.color_channel_combo.currentText()
        if self.spec.color_overrides.pop(ch, None) is not None:
            self._color_keys = {}
            self._refresh_color_swatches()
            self._schedule_render()

    def _on_color_by(self, i):
        combo = self._panel_rows[i]._combo
        key = combo.currentData()
        if key == PICK_FEATURE:
            self._pick_feature(i)                # async-safe: re-points the combo itself
            return
        self._panels[i].color_by = key
        self._update_panel_visibility()
        self._schedule_render()

    def _on_cmap(self, i):
        self._panels[i].cmap = self._panel_rows[i]._cmap.currentText()
        self._schedule_render()

    def _on_title(self, i):
        self._panels[i].title = self._panel_rows[i]._title.text().strip()
        self._schedule_render()

    def _pick_feature(self, i):
        """Searchable lipid/feature picker → add a continuous channel for that column and
        point panel ``i`` at it. Deferred so we never rebuild a combo inside its own
        ``activated`` handler (a known pyqtgraph/Qt foot-gun)."""
        def go():
            label, ok = QtWidgets.QInputDialog.getItem(
                self, "Colour by feature", "Lipid / feature (type to search):",
                self._feature_labels, 0, True)
            if ok and label:
                j = self._feature_labels.index(label)
                name = self.data.set_feature_channel(j)
                self._panels[i].color_by = name
            self._refresh_panel_combos()          # restore selection (or revert on cancel)
            self._update_panel_visibility()
            self._schedule_render()
        QtCore.QTimer.singleShot(0, go)

    # ---------------------------------------------------------- annotation edit
    def _add_label(self):
        text, ok = QtWidgets.QInputDialog.getText(self, "Add label", "Label text:")
        if not ok or not text.strip():
            return
        (x0, x1), (y0, y1) = getattr(self.fig, "_umap_ranges", ((0, 1), (0, 1)))
        ann = us.Annotation(text=text.strip(), x=(x0 + x1) / 2.0, y=(y0 + y1) / 2.0,
                            panel=0, bold=True, fontsize=float(self.fontsize_spin.value()),
                            color=("#f0f0f0" if self.bg_combo.currentData() == "black" else "#111111"))
        self.spec.annotations.append(ann)
        self._selected = ("annotation", len(self.spec.annotations) - 1)  # spinbox now targets it
        self._sync_fontsize_spin(ann.fontsize)
        self._render()

    def _sync_fontsize_spin(self, size):
        """Show ``size`` in the Text-size spinbox without re-triggering a resize."""
        self.fontsize_spin.blockSignals(True)
        self.fontsize_spin.setValue(float(size))
        self.fontsize_spin.blockSignals(False)

    def _on_fontsize_changed(self, val):
        """Resize the selected label live; with nothing selected the value is just the
        starting size for the next label added."""
        sel = getattr(self, "_selected", None)
        if not (isinstance(sel, tuple) and sel[0] == "annotation"):
            return
        key = sel[1]
        if not (0 <= key < len(self.spec.annotations)):
            return
        self.spec.annotations[key].fontsize = float(val)
        # resize the on-canvas artist directly — far cheaper than re-rastering the cloud; the
        # spec already carries the new size so a later full render reproduces it.
        artists = getattr(self, "_ann_artists", [])
        if 0 <= key < len(artists):
            artists[key][0].set_fontsize(float(val))
            self.canvas.draw_idle()

    def _on_press(self, event):
        # hit-test every draggable (text labels, panel titles / A–D letters, legends) topmost
        # first. Labels/titles/letters live in axes; legends in the figure margin — so we can't
        # gate on event.inaxes here.
        for artist, kind, key in reversed(getattr(self, "_draggables", [])):
            try:
                hit, _ = artist.contains(event)
            except Exception:  # noqa: BLE001 — a stale artist between renders
                hit = False
            if hit:
                if kind == "annotation" and event.dblclick:
                    self._rename_idx(key, artist)
                else:
                    self._drag = (artist, kind, key)
                    self._drag_xy = None
                    self._selected = ("annotation", key) if kind == "annotation" else None
                    if kind == "annotation":         # reflect its size in the Text-size spinbox
                        self._sync_fontsize_spin(self.spec.annotations[key].fontsize)
                return

    def _event_xy(self, kind, artist, event):
        """Mouse position in the artist's own coordinate space: data coords for annotations,
        axes fraction for panel titles/letters, figure fraction for legends. Returns ``None``
        for an annotation dragged outside the axes (no data coords there)."""
        if kind == "annotation":
            if event.xdata is None:
                return None
            return (float(event.xdata), float(event.ydata))
        if event.x is None or event.y is None:
            return None
        inv = (self.fig.transFigure.inverted() if kind == "legend"
               else artist.axes.transAxes.inverted())
        x, y = inv.transform((event.x, event.y))
        return (float(x), float(y))

    def _on_motion(self, event):
        if self._drag is None:
            return
        artist, kind, key = self._drag
        xy = self._event_xy(kind, artist, event)
        if xy is None:
            return
        if kind == "legend":
            artist.set_bbox_to_anchor(xy, transform=self.fig.transFigure)
        else:
            artist.set_position(xy)
        self._drag_xy = xy
        self.canvas.draw_idle()

    def _on_release(self, _event):
        if self._drag is None:
            return
        artist, kind, key = self._drag
        xy = getattr(self, "_drag_xy", None)
        if xy is not None:                       # persist into the spec so export reproduces it
            if kind == "annotation":
                ann = self.spec.annotations[key]
                ann.x, ann.y = float(xy[0]), float(xy[1])
            elif kind == "letter":
                self._panels[key].letter_xy = (float(xy[0]), float(xy[1]))
            elif kind == "title":
                self._panels[key].title_xy = (float(xy[0]), float(xy[1]))
            elif kind == "legend":
                self.spec.legend_xy[key] = (float(xy[0]), float(xy[1]))
        self._drag = None
        self._drag_xy = None

    def _rename_idx(self, key, artist):
        if not (0 <= key < len(self.spec.annotations)):
            return
        ann = self.spec.annotations[key]
        text, ok = QtWidgets.QInputDialog.getText(self, "Rename label", "Label text:",
                                                  text=ann.text)
        if ok:
            ann.text = text.strip()
            if not ann.text:
                self.spec.annotations.pop(key)
                self._render()
            else:
                artist.set_text(ann.text)
                self.canvas.draw_idle()

    def keyPressEvent(self, ev):
        if ev.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
            sel = getattr(self, "_selected", None)
            if isinstance(sel, tuple) and sel[0] == "annotation" \
                    and 0 <= sel[1] < len(self.spec.annotations):
                self.spec.annotations.pop(sel[1])
                self._selected = None
                self._render()
                return
        super().keyPressEvent(ev)

    # --------------------------------------------------------------- exporting
    def _export(self):
        fmt = self.fmt_combo.currentText().lower()
        ext = {"tiff": "tif"}.get(fmt, fmt)
        path, _ = filedialogs.get_save_file_name(
            self, "Export UMAP figure", f"umap_figure.{ext}", f"{fmt.upper()} (*.{ext})")
        if not path:
            return
        spec = self._build_spec()
        spec.dpi = int(self.dpi.value())
        try:
            import matplotlib
            matplotlib.rcParams["pdf.fonttype"] = 42      # editable text in vector PDFs
            matplotlib.rcParams["svg.fonttype"] = "none"
            fig = us.render_figure(self.data, spec, color_keys=self._color_keys)  # fresh figure
            out = export.save_figure(fig, path, dpi=spec.dpi, fmt=ext)
            prefs.set(PREFS_KEY, spec.to_dict())          # remember the look for next time
            self.status.setText(f"Saved {out}")
            self.win.statusBar().showMessage(f"UMAP figure → {out}")
        except Exception as e:  # noqa: BLE001
            self.status.setText(f"Export failed: {e}")


class UMAPStudioMixin:
    """Adds the UMAP Studio entry points to ``MainWindow``."""

    def open_umap_studio(self, data: us.EmbeddingData | None = None):
        """Open the editor. ``data`` comes from the Cohort UMAP tab; without it we reuse the
        last embedding pooled this session (``self._umap_studio_data``)."""
        if data is None:
            data = getattr(self, "_umap_studio_data", None)
        if data is None:
            self.statusBar().showMessage(
                "Run a Cohort UMAP first, then open UMAP Studio to make it publication-ready.")
            self.reveal_view("Cohort UMAP")
            return
        self._umap_studio_data = data
        UMAPStudioDialog(self, data).exec()
