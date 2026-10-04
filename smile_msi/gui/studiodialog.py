"""Export Studio — the flexible, multi-list / multi-section export surface.

One place to choose **which feature lists and ions**, across **which sections**, and **which
outputs** (per-ion photos, a combined matrix figure, the feature-list CSVs and the analyses
bundle), then render them all in one action — instead of switching lists + sections and
re-exporting per group.

* :class:`StudioDialog` is the three-pane UI (Sections · Feature-lists→ions · Outputs+design).
  It only *collects a selection* — it reads the saved-list registries directly and never
  mutates the live working set ``self.peaks``.
* :class:`StudioMixin` lives on ``MainWindow`` and turns a compiled :class:`smile_msi.studio`
  plan into rendered files, supplying the loader / extract / render callbacks that close over
  the live slide (P2) and, later, each cohort section's freshly-loaded cube (P3).
"""
from __future__ import annotations

import os

import numpy as np
from PySide6 import QtCore, QtWidgets

from .. import export, studio, imaging, palettes, stylelib
from .. import annotations as annot
from .common import (note, section_title, MUTED_QSS, RangeSliderField, icon, button,
                     check_tree_bar, coalesce, NoScrollComboBox, NoScrollDoubleSpinBox,
                     NoScrollSpinBox)
from . import filedialogs

# the same label-content choices the Export hub offers (shared, single source)
LABEL_CHOICES = annot.LABEL_CHOICES

OUTPUT_ROWS = [
    (studio.OutputKind.ION_IMAGES, "Per-ion photos (folder tree)",
     "One styled ion image per section × ion, organised into folders."),
    (studio.OutputKind.MATRIX, "Combined matrix figure",
     "A single ion-row × section-column grid figure (publication contact sheet)."),
    (studio.OutputKind.OVERLAY, "Colour overlay per section",
     "An additive multi-ion composite of all picked ions, one per section."),
    (studio.OutputKind.LIST_CSV, "Feature-list CSVs",
     "One CSV per picked feature list (m/z, lipid, source list)."),
    (studio.OutputKind.ANALYSES, "Analyses bundle",
     "Region intensities, discriminating statistics and methods (live slide)."),
]


class StudioDialog(QtWidgets.QDialog):
    """Collects the sections, the ions (across lists) and the outputs, compiles a
    :class:`smile_msi.studio.StudioPlan` and hands it to ``win.run_studio_plan``."""

    PREFS_KEY = "studio_options"

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        # Ticking a feature list cascades itemChanged over every ion in it; recount once at
        # the end of the cascade, not once per ion (O(N) instead of O(N²)).
        self._recount = coalesce(self, self._update_count)
        self.setWindowTitle("Export Studio")
        self.setMinimumSize(1000, 520)
        root = QtWidgets.QVBoxLayout(self)

        root.addWidget(note("Pick feature lists + ions, the sections to render them across, and "
                            "the outputs — then export everything in one pass."))

        # --- three panes side by side ------------------------------------- #
        panes = QtWidgets.QHBoxLayout()
        panes.addWidget(self._build_sections_pane(), 4)
        panes.addWidget(self._build_lists_pane(), 5)
        panes.addWidget(self._build_outputs_pane(), 6)
        root.addLayout(panes, 1)

        # --- footer: live count + buttons --------------------------------- #
        self.count_lbl = QtWidgets.QLabel("")
        self.count_lbl.setStyleSheet(MUTED_QSS)
        root.addWidget(self.count_lbl)

        btns = QtWidgets.QDialogButtonBox()
        self.b_export = btns.addButton("Export…", QtWidgets.QDialogButtonBox.AcceptRole)
        self.b_export.setObjectName("primaryAction")      # the dialog's primary run action
        btns.addButton(QtWidgets.QDialogButtonBox.Close)
        self.b_export.clicked.connect(self._do_export)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        self._populate_sections()
        self._populate_lists()
        self._restore_prefs()
        self._restore_plan()                 # re-tick a previously-saved plan for this sample
        self._update_count()

        # Never open taller than the screen (the design pane scrolls) so the footer Export
        # button is always reachable and the title bar stays on-screen to move/resize.
        scr = QtWidgets.QApplication.primaryScreen()
        if scr is not None:
            avail = scr.availableGeometry()
            self.setMaximumHeight(avail.height())
            self.resize(min(1180, max(1000, avail.width() - 80)),
                        min(760, avail.height() - 60))

    # ------------------------------------------------------------------ panes
    def _build_sections_pane(self):
        box = QtWidgets.QGroupBox("Sections")
        v = QtWidgets.QVBoxLayout(box)
        v.addWidget(section_title("Render across"))
        # a tree (not a flat list) so each region section can carry its own crop control.
        # Column 0 (the name) takes the slack; column 1 sizes to its crop button so the button is
        # always fully visible even in a narrow pane (the old fixed 200px column pushed it off-screen).
        self.sec_tree = QtWidgets.QTreeWidget()
        self.sec_tree.setHeaderLabels(["Section", "Close-up crop"])
        self.sec_tree.setRootIsDecorated(False)
        hdr = self.sec_tree.header()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.sec_tree.setMinimumWidth(230)   # keep room for the name column beside the crop button
        self.sec_tree.itemChanged.connect(self._recount)
        self._sec_bar = check_tree_bar(self.sec_tree, "section", on_change=self._update_count)
        v.addWidget(self._sec_bar)
        v.addWidget(self.sec_tree, 1)
        v.addWidget(note("The live slide and the regions you've drawn on it are always here; "
                         "cohort samples appear once registered in the Samples panel. A region "
                         "renders as a close-up — set its crop with “Edit crop…”."))
        return box

    def _build_lists_pane(self):
        box = QtWidgets.QGroupBox("Feature lists → ions")
        v = QtWidgets.QVBoxLayout(box)
        v.addWidget(section_title("Pick ions across any lists"))
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Feature / ion", "◆ class mode"])
        # responsive columns (same fix as the sections tree): the ion column takes the slack and
        # the class-mode combo column self-sizes, so it can't be clipped off in a narrow pane.
        lhdr = self.tree.header()
        lhdr.setStretchLastSection(False)
        lhdr.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        lhdr.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.tree.setMinimumWidth(300)       # room for the indented ion labels + class-mode combo
        self.tree.itemChanged.connect(self._tree_item_changed)
        # All/None/Invert over the *ion* leaves — the pane had a "Check none" and no way back.
        # The parents are ItemIsAutoTristate, so ticking the leaves settles them for us.
        self._list_bar = check_tree_bar(self.tree, "ion", on_change=self._update_count)
        lay = self._list_bar.layout()
        for i, (label, fn) in enumerate([("Expand all", self.tree.expandAll),
                                         ("Collapse all", self.tree.collapseAll)]):
            lay.insertWidget(3 + i, button(label, fn))     # after All/None/Invert, before stretch
        v.addWidget(self._list_bar)
        v.addWidget(self.tree, 1)
        return box

    def _build_outputs_pane(self):
        box = QtWidgets.QGroupBox("Outputs and design")
        # The design form is the tall column that drives the dialog's height. Put it in a
        # scroll area so the whole dialog can shrink below the form's natural height (and never
        # open taller than the screen) while every control stays reachable by scrolling here.
        box_v = QtWidgets.QVBoxLayout(box)
        box_v.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        box_v.addWidget(scroll)
        inner = QtWidgets.QWidget()
        scroll.setWidget(inner)
        v = QtWidgets.QVBoxLayout(inner)
        v.addWidget(section_title("Generate"))
        self.out_chks = {}
        for kind, label, tip in OUTPUT_ROWS:
            c = QtWidgets.QCheckBox(label); c.setToolTip(tip)
            c.toggled.connect(lambda *_: self._update_count())
            self.out_chks[kind] = c
            v.addWidget(c)
        self.out_chks[studio.OutputKind.ION_IMAGES].setChecked(True)
        self.out_chks[studio.OutputKind.LIST_CSV].setChecked(True)

        v.addSpacing(6)
        v.addWidget(section_title("Design"))
        form = QtWidgets.QFormLayout()
        self.cmap_combo = NoScrollComboBox()
        items = [self.win.cmap_combo.itemText(i) for i in range(self.win.cmap_combo.count())] \
            if getattr(self.win, "cmap_combo", None) is not None else []
        self.cmap_combo.addItems(items or ["viridis", "inferno", "magma", "cividis", "turbo", "hot"])
        if getattr(self.win, "cmap_combo", None) is not None:
            self.cmap_combo.setCurrentText(self.win.cmap_combo.currentText())
        form.addRow("Colormap", self.cmap_combo)

        self.theme_combo = NoScrollComboBox()
        self.theme_combo.addItems(["Light", "Dark glass"])
        form.addRow("Card theme", self.theme_combo)

        # Style preset — one reusable look (fonts / type sizes / line weights / category
        # colours) applied to EVERY panel in the batch, so a whole cohort renders consistently.
        self.preset_combo = NoScrollComboBox()
        self.preset_combo.setToolTip(
            "A reusable look applied to every ion panel, overlay and the matrix figure this "
            "batch renders — so the whole set matches. Copy the recipe (JSON) to tweak "
            "elsewhere, then Import the result.")
        self.preset_combo.setMinimumWidth(120)   # don't let the Copy/Import buttons squeeze the name
        self._refresh_preset_combo()
        self.preset_combo.activated.connect(self._apply_preset)
        copy_btn = QtWidgets.QPushButton("Copy")
        copy_btn.setToolTip("Copy this look's recipe (JSON) to the clipboard.")
        copy_btn.clicked.connect(self._copy_recipe)
        imp_btn = QtWidgets.QPushButton("Import…")
        imp_btn.setToolTip("Paste a recipe (JSON) to add it as a saved look.")
        imp_btn.clicked.connect(self._import_recipe)
        prow = QtWidgets.QHBoxLayout()
        prow.setContentsMargins(0, 0, 0, 0)
        prow.setSpacing(4)
        prow.addWidget(self.preset_combo, 1)
        prow.addWidget(copy_btn)
        prow.addWidget(imp_btn)
        pwrap = QtWidgets.QWidget()
        pwrap.setLayout(prow)
        form.addRow("Style preset", pwrap)

        self.label_combo = NoScrollComboBox()
        for text, _m in LABEL_CHOICES:
            self.label_combo.addItem(text)
        form.addRow("Label", self.label_combo)

        self.contrast_mode = NoScrollComboBox()
        self.contrast_mode.addItem("Shared scale per ion (comparable)", studio.ContrastMode.SHARED.value)
        self.contrast_mode.addItem("Per-section auto (each looks best)", studio.ContrastMode.PER_SECTION.value)
        self.contrast_mode.setToolTip("Shared: one intensity window per ion across all sections, so "
                                      "brightness is honestly comparable. Per-section: each section "
                                      "auto-contrasts to look its best (not comparable).")
        form.addRow("Contrast mode", self.contrast_mode)

        self.layout_combo = NoScrollComboBox()
        self.layout_combo.addItem("List ▸ section ▸ ion", "list_outer")
        self.layout_combo.addItem("Section ▸ list ▸ ion", "section_outer")
        form.addRow("Folder layout", self.layout_combo)

        self.fmt_combo = NoScrollComboBox()
        for label, ext in [("PNG", "png"), ("TIFF", "tiff"), ("JPEG", "jpg"),
                           ("PDF", "pdf"), ("SVG", "svg")]:
            self.fmt_combo.addItem(label, ext)
        form.addRow("Image format", self.fmt_combo)

        self.dpi_spin = NoScrollSpinBox(); self.dpi_spin.setRange(72, 1200); self.dpi_spin.setValue(300)
        self.dpi_spin.setSuffix(" dpi")
        form.addRow("Resolution", self.dpi_spin)

        # Automatic length (a round 1-2-5 length ≈20% of each section, from its pixel size);
        # the checkbox only toggles it — no hand-typed size that could be wrong.
        self.chk_scale = QtWidgets.QCheckBox("Auto (sized from the pixel size)")
        self.chk_scale.setChecked(True)
        self.chk_scale.setToolTip("Draw an automatic scale bar on each section. Needs a known "
                                  "pixel size (set it in the main Export dialog if the file lacks one).")
        form.addRow("Scale bar", self.chk_scale)

        # Intensity scaling for the image outputs — type the contrast clip percentile (the
        # value mapped to 100% brightness). Lower = brighter/more saturated. Owned by the
        # Studio (typed here), defaulting to the main view's current contrast so existing
        # exports are unchanged until you touch it.
        self.clip_spin = NoScrollDoubleSpinBox()
        self.clip_spin.setRange(50.0, 100.0)
        self.clip_spin.setDecimals(1)
        self.clip_spin.setSingleStep(0.5)
        self.clip_spin.setSuffix(" %")
        self.clip_spin.setKeyboardTracking(False)        # commit on Enter/focus-out, not each keystroke
        cs = getattr(self.win, "contrast_spin", None)
        self.clip_spin.setValue(float(cs.value()) if cs is not None else 99.0)
        self.clip_spin.setToolTip("Intensity scaling for exported images: the percentile mapped to "
                                  "100% brightness (the hotspot clip). Lower = brighter. Type a value.")
        form.addRow("Contrast clip", self.clip_spin)

        # Standardise the intensity window across the whole batch. Off (default): every cell
        # fills 0–100% of its own clip/anchor, so each ion×section looks its best but the
        # brightness of a fixed % differs cell to cell. On: pin one low/high window — dragged or
        # typed — so every exported panel (and the matrix) shares an identical contrast, the way
        # the Export hub's "Override intensity window" does. The window is relative intensity:
        # % of each cell's 100% reference (the clip percentile, or the shared per-ion anchor in
        # Shared mode), so it stays honestly comparable in both contrast modes.
        self.window_override_chk = QtWidgets.QCheckBox("Override intensity window")
        self.window_override_chk.setToolTip(
            "Off: each cell fills 0–100% of its clip/anchor (each looks best). On: every exported "
            "image and the matrix use the one low/high window below — a single standardised "
            "contrast across the whole batch.")
        lo0, hi0 = self._initial_window()
        self.window_slider = RangeSliderField(lo0, hi0)
        self.window_slider.setToolTip(
            "Standardised contrast window for the batch — % of each cell's 100% reference (the "
            "clip percentile, or the shared per-ion anchor in Shared mode). Drag the handles or "
            "type exact values.")
        self.window_slider.setEnabled(False)
        self.window_override_chk.toggled.connect(self.window_slider.setEnabled)
        form.addRow(self.window_override_chk)
        form.addRow("Intensity window", self.window_slider)

        v.addLayout(form)
        v.addStretch(1)
        return box

    # --------------------------------------------------------------- populate
    def _populate_sections(self):
        self.sec_tree.blockSignals(True)
        self.sec_tree.clear()
        for sec in self.win._studio_sections():
            it = QtWidgets.QTreeWidgetItem(self.sec_tree)
            tag = " — current" if (sec.is_live and not sec.region) else ""
            it.setText(0, sec.name + tag)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            # the whole live slide starts checked; regions / cohort sections start unchecked
            it.setCheckState(0, QtCore.Qt.Checked if (sec.is_live and not sec.region)
                             else QtCore.Qt.Unchecked)
            it.setData(0, QtCore.Qt.UserRole, sec.to_dict())
            if sec.region and sec.is_live:        # a drawn region of the current slide → editable crop
                self._add_region_crop_widget(it, sec)
        self.sec_tree.blockSignals(False)
        self._sec_bar.refresh_count()             # the bar was built over an empty tree

    def _crop_btn_text(self, region_name):
        """Terse crop-state label for the row button, derived from the single source of truth
        (``_studio_region_crop_state`` → '✎ custom crop' | 'auto (region bbox)') so it can't drift."""
        return "custom crop" if self.win._studio_region_crop_state(region_name).startswith("✎") \
            else "auto crop"

    def _add_region_crop_widget(self, item, sec):
        """A single compact crop button on a live region row: a gear icon + terse state, clicking
        opens the Crop Studio. One button (no separate stretched label) so it always fits the
        narrow Sections column instead of being pushed off-screen behind a scrollbar."""
        btn = QtWidgets.QPushButton("  " + self._crop_btn_text(sec.region))
        btn.setIcon(icon("settings"))
        btn.setToolTip("Draw or adjust this region's close-up crop (non-destructive; saved on "
                       "the region). Default is the region's bounding box + margin.")
        btn.clicked.connect(lambda _=False, sid=sec.sid, rn=sec.region, b=btn:
                            self._edit_region_crop(sid, rn, b))
        self.sec_tree.setItemWidget(item, 1, btn)

    def _edit_region_crop(self, sid, region_name, button):
        if self.win._studio_edit_region_crop(region_name):
            button.setText("  " + self._crop_btn_text(region_name))

    def _populate_lists(self):
        """Build the two-level tree from the three list registries — without ever touching
        ``self.peaks`` (we read ``_feature_scopes`` / ``_feature_lists`` / ``_lipid_lists``)."""
        self.tree.blockSignals(True)
        self.tree.clear()
        for group in self.win._studio_list_groups():
            parent = QtWidgets.QTreeWidgetItem(self.tree)
            parent.setText(0, f"{group['badge']} {group['name']}  ({len(group['items'])})")
            parent.setFlags(parent.flags() | QtCore.Qt.ItemIsUserCheckable
                            | QtCore.Qt.ItemIsAutoTristate)
            parent.setCheckState(0, QtCore.Qt.Unchecked)
            parent.setData(0, QtCore.Qt.UserRole, {"role": "group"})
            for entry in group["items"]:
                child = QtWidgets.QTreeWidgetItem(parent)
                child.setText(0, entry["text"])
                child.setFlags(child.flags() | QtCore.Qt.ItemIsUserCheckable)
                child.setCheckState(0, QtCore.Qt.Unchecked)
                child.setData(0, QtCore.Qt.UserRole, {"role": "ion", "rec": entry["rec"],
                                                      "is_class": entry.get("is_class", False)})
                if entry.get("is_class"):
                    combo = NoScrollComboBox()
                    combo.addItem("Composite", "composite")
                    combo.addItem("Expand ions", "expand")
                    combo.currentIndexChanged.connect(lambda *_: self._update_count())
                    self.tree.setItemWidget(child, 1, combo)
        self.tree.expandToDepth(0)
        self.tree.blockSignals(False)
        self._list_bar.refresh_count()            # the bar was built over an empty tree

    # ------------------------------------------------------------- selections
    def _iter_children(self):
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            for j in range(top.childCount()):
                yield top.child(j)

    def selections(self):
        """Resolved :class:`SelectionRecord` list from every ticked ion / class."""
        out = []
        for child in self._iter_children():
            if child.checkState(0) != QtCore.Qt.Checked:
                continue
            d = child.data(0, QtCore.Qt.UserRole) or {}
            rec = dict(d.get("rec") or {})
            if d.get("is_class"):
                combo = self.tree.itemWidget(child, 1)
                mode = combo.currentData() if combo is not None else "composite"
                members = tuple(float(m) for m in (rec.get("member_mzs") or []))
                if mode == "expand":
                    for m in members:
                        out.append(studio.SelectionRecord(
                            mz=m, label=f"{rec.get('lipid_class', '')} {m:.4f}".strip(),
                            list_name=rec.get("list_name", ""), kind="ion",
                            source_kind=rec.get("source_kind", "lipid_class"),
                            lipid_class=rec.get("lipid_class", ""), color=rec.get("color", "")))
                else:
                    out.append(studio.SelectionRecord(
                        mz=(members[0] if members else None),
                        label=rec.get("label", rec.get("lipid_class", "")),
                        list_name=rec.get("list_name", ""), kind="composite",
                        source_kind=rec.get("source_kind", "lipid_class"),
                        lipid_class=rec.get("lipid_class", ""), member_mzs=members,
                        color=rec.get("color", "")))
            else:
                out.append(studio.SelectionRecord(
                    mz=(float(rec["mz"]) if rec.get("mz") is not None else None),
                    label=rec.get("label", ""), list_name=rec.get("list_name", ""),
                    kind="ion", source_kind=rec.get("source_kind", "saved"),
                    color=rec.get("color", "")))
        return out

    def sections(self):
        out = []
        for i in range(self.sec_tree.topLevelItemCount()):
            it = self.sec_tree.topLevelItem(i)
            if it.checkState(0) == QtCore.Qt.Checked:
                out.append(studio.Section.from_dict(it.data(0, QtCore.Qt.UserRole)))
        return out

    def outputs(self):
        return [k for k, c in self.out_chks.items() if c.isChecked()]

    def _initial_window(self):
        """Seed the standardised intensity window from the active feature's dock window, so the
        override starts where the live view is; fall back to the full 0–100 range when there's no
        active feature yet (mirrors the Export hub's ``_initial_export_window``)."""
        try:
            p = self.win._peak_for_mz(self.win.active_mz)
            lo, hi = self.win._feature_window(p)
            return float(lo), float(hi)
        except Exception:  # noqa: BLE001 — no active feature / not picked yet
            return 0.0, 100.0

    def design(self):
        lo, hi = self.window_slider.values()
        return {
            "cmap": self.cmap_combo.currentText(),
            "theme": ("light" if self.theme_combo.currentText().startswith("Light") else "dark"),
            "label_mode": LABEL_CHOICES[self.label_combo.currentIndex()][1],
            "dpi": int(self.dpi_spin.value()),
            "scalebar_on": bool(self.chk_scale.isChecked()),
            "clip": float(self.clip_spin.value()),
            "window_override": bool(self.window_override_chk.isChecked()),
            "window": [float(lo), float(hi)],
            "style_spec": self._style_spec_for_render(),
        }

    # ----- style-preset plumbing (mirrors the Export hub) ------------------ #
    def _refresh_preset_combo(self, select=None):
        want = select or (self.preset_combo.currentText() if self.preset_combo.count() else "App Default")
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        for nm in stylelib.names():
            self.preset_combo.addItem(nm)
        i = self.preset_combo.findText(want)
        self.preset_combo.setCurrentIndex(i if i >= 0 else 0)
        self.preset_combo.blockSignals(False)

    def _current_style_spec(self):
        return stylelib.get(self.preset_combo.currentText())

    def _style_spec_for_render(self):
        """The preset recipe (dict) for :func:`export.active_style`, minus the theme + image
        colormap the dialog owns with its own pickers. ``None`` for the no-op 'App Default'."""
        spec = self._current_style_spec()
        if spec is None or spec.is_default():
            return None
        d = spec.to_dict()
        d.pop("theme", None)
        d.pop("image_cmap", None)
        return d if len(d) > 1 else None

    def _apply_preset(self, *_):
        spec = self._current_style_spec()
        if spec is None:
            return
        if spec.theme:
            self._set_text(self.theme_combo,
                           "Light" if spec.theme in ("light", "print", "paper") else "Dark glass")
        if spec.image_cmap:
            i = self.cmap_combo.findText(spec.image_cmap)
            if i < 0:
                self.cmap_combo.addItem(spec.image_cmap)
                i = self.cmap_combo.findText(spec.image_cmap)
            if i >= 0:
                self.cmap_combo.setCurrentIndex(i)

    def _copy_recipe(self):
        spec = self._current_style_spec()
        if spec is not None:
            QtWidgets.QApplication.clipboard().setText(spec.to_json())
            self.win.statusBar().showMessage(f"Copied recipe for “{spec.name}”.")

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
        self._refresh_preset_combo(select=spec.name)
        self._apply_preset()
        self.win.statusBar().showMessage(f"Imported look “{spec.name}”.")

    def _compile(self):
        return studio.compile_plan(
            self.sections(), self.selections(), self.outputs(),
            design=self.design(), layout=self.layout_combo.currentData(),
            contrast=self.contrast_mode.currentData(), ext=self.fmt_combo.currentData())

    # --------------------------------------------------------------- handlers
    def _tree_item_changed(self, *_):
        self._recount()

    def _update_count(self):
        if not hasattr(self, "count_lbl"):
            return                          # a default-check fired before the footer was built
        n_sec = len(self.sections())
        n_ion = len(self.selections())
        n_out = len(self.outputs())
        imgs = n_sec * n_ion if studio.OutputKind.ION_IMAGES in self.outputs() else 0
        self.count_lbl.setText(
            f"{n_sec} section(s) × {n_ion} ion(s) · {n_out} output type(s)"
            + (f"  →  {imgs} ion image(s)" if imgs else ""))
        self.b_export.setEnabled(n_sec >= 1 and n_ion >= 1 and n_out >= 1)

    def _do_export(self):
        plan = self._compile()
        if plan.job_count() == 0 and not (plan.wants(studio.OutputKind.LIST_CSV)
                                          or plan.wants(studio.OutputKind.ANALYSES)
                                          or plan.wants(studio.OutputKind.OVERLAY)
                                          or plan.wants(studio.OutputKind.MATRIX)):
            self.win.statusBar().showMessage("Pick at least one ion and one output.")
            return
        self._save_prefs()
        self.win._studio_plan = plan          # remember on the sample (persisted in the session)
        self.win._mark_dirty()
        if self.win.run_studio_plan(plan):
            self.accept()

    # ----------------------------------------------------------------- prefs
    def _save_prefs(self):
        from .. import prefs
        prefs.set(self.PREFS_KEY, {
            "outputs": [studio.OutputKind(o).value for o in self.outputs()],
            "cmap": self.cmap_combo.currentText(), "theme": self.theme_combo.currentText(),
            "label": self.label_combo.currentIndex(), "contrast": self.contrast_mode.currentData(),
            "layout": self.layout_combo.currentData(), "fmt": self.fmt_combo.currentData(),
            "dpi": int(self.dpi_spin.value()), "scalebar_on": bool(self.chk_scale.isChecked()),
            "clip": float(self.clip_spin.value()),
            "window_override": bool(self.window_override_chk.isChecked()),
            "window": [float(v) for v in self.window_slider.values()],
            "style_preset": self.preset_combo.currentText(),
        })

    def _restore_prefs(self):
        from .. import prefs
        d = prefs.get(self.PREFS_KEY) or {}
        if not isinstance(d, dict) or not d:
            return
        try:
            want = {studio.OutputKind(o) for o in (d.get("outputs") or [])}
            if want:
                for k, c in self.out_chks.items():
                    c.setChecked(k in want)
            self._set_text(self.cmap_combo, d.get("cmap"))
            self._set_text(self.theme_combo, d.get("theme"))
            if isinstance(d.get("label"), int) and 0 <= d["label"] < self.label_combo.count():
                self.label_combo.setCurrentIndex(d["label"])
            self._set_data(self.contrast_mode, d.get("contrast"))
            self._set_data(self.layout_combo, d.get("layout"))
            self._set_data(self.fmt_combo, d.get("fmt"))
            if "dpi" in d:
                self.dpi_spin.setValue(int(d["dpi"]))
            if "scalebar_on" in d:
                self.chk_scale.setChecked(bool(d["scalebar_on"]))
            elif "scalebar_um" in d:                       # migrate an old length-based plan
                self.chk_scale.setChecked(float(d["scalebar_um"]) > 0)
            if "clip" in d:
                self.clip_spin.setValue(float(d["clip"]))
            w = d.get("window")
            if isinstance(w, (list, tuple)) and len(w) == 2:
                self.window_slider.setValues(float(w[0]), float(w[1]))
            # set the checkbox last: its toggled signal enables/disables the slider
            self.window_override_chk.setChecked(bool(d.get("window_override", False)))
            if d.get("style_preset"):
                self._refresh_preset_combo(select=d["style_preset"])
                self._apply_preset()
        except Exception:  # noqa: BLE001 — restoring prefs must never break the dialog
            import traceback
            traceback.print_exc()

    def _restore_plan(self):
        """Re-tick the sections/ions/outputs of the plan last saved on this sample, so the
        Studio reopens where the user left it (mirrors the Report tab's persistence)."""
        plan = getattr(self.win, "_studio_plan", None)
        if plan is None:
            return
        try:
            want_secs = {s.sid for s in plan.sections}
            for i in range(self.sec_tree.topLevelItemCount()):
                it = self.sec_tree.topLevelItem(i)
                sid = (it.data(0, QtCore.Qt.UserRole) or {}).get("sid")
                if sid in want_secs:
                    it.setCheckState(0, QtCore.Qt.Checked)
            want_uids = {s.uid() for s in plan.selections}
            self.tree.blockSignals(True)
            for child in self._iter_children():
                d = child.data(0, QtCore.Qt.UserRole) or {}
                rec = d.get("rec") or {}
                sel = studio.SelectionRecord.from_dict(rec) if rec else None
                if sel is not None and sel.uid() in want_uids:
                    child.setCheckState(0, QtCore.Qt.Checked)
            self.tree.blockSignals(False)
        except Exception:  # noqa: BLE001
            self.tree.blockSignals(False)

    @staticmethod
    def _set_text(combo, text):
        if text:
            i = combo.findText(text)
            if i >= 0:
                combo.setCurrentIndex(i)

    @staticmethod
    def _set_data(combo, data):
        if data is not None:
            i = combo.findData(data)
            if i >= 0:
                combo.setCurrentIndex(i)


# --------------------------------------------------------------------------- #
# Mixin — lives on MainWindow
# --------------------------------------------------------------------------- #
class StudioMixin:
    # ----- entry point ----------------------------------------------------- #
    def open_export_studio(self):
        if self.ds is None:
            self.statusBar().showMessage("Load a dataset first.")
            return
        dlg = StudioDialog(self)
        dlg.exec()

    # ----- section + list enumeration (read-only) -------------------------- #
    def _studio_sections(self):
        """The renderable sections: the live whole slide, then every region drawn on it (each a
        close-up rendered from the same in-memory cube — no reload), then every cohort sample.

        The live slide + its regions fully represent the current slide, so any cohort sample
        whose source IS the live slide is skipped (no duplicates). Cohort sections of *other*
        slides are loaded one cube at a time by the shared :class:`cohort.SectionLoader`."""
        out = []
        live_src = getattr(self.ds, "source", "") or ""
        slide = os.path.splitext(os.path.basename(live_src))[0] or "Current slide"
        out.append(studio.Section(sid="<live>", name=slide, source=live_src,
                                  session_path=(self._active_session_path() or ""), is_live=True))
        # regions drawn on the current slide → close-up sections (same cube, no reload)
        for rg in (getattr(self, "regions", None) or []):
            nm = rg.get("name")
            if not nm or not rg.get("visible", True):
                continue
            if self._region_pixel_mask(rg) is None:        # unresolved (no mask/segments) → skip
                continue
            out.append(studio.Section(sid=f"<live>::{nm}", name=f"{slide} - {nm}",
                                      source=live_src, region=nm,
                                      session_path=(self._active_session_path() or ""), is_live=True))
        cohort = getattr(self, "cohort", None)
        live_abs = os.path.abspath(live_src) if live_src else ""
        for ref in (getattr(cohort, "samples", None) or []):
            src = ref.source or ""
            if not src.lower().endswith(".imzml"):
                continue
            if live_abs and os.path.abspath(src) == live_abs:   # the live slide + its regions cover this
                continue
            out.append(studio.Section(sid=ref.key(), name=ref.name, source=src,
                                      session_path=ref.session_path, region=ref.region,
                                      is_live=False))
        return out

    def _studio_find_region(self, region_name):
        return next((r for r in (getattr(self, "regions", None) or [])
                     if r.get("name") == region_name), None)

    def _studio_region_crop_state(self, region_name):
        """Short label of a region's close-up framing: a manually-drawn Crop Studio box, or the
        automatic padded bounding box."""
        rg = self._studio_find_region(region_name)
        if rg is None:
            return ""
        has_crop = bool(rg.get("crop")) and \
            rg.get("crop_orient", getattr(self.ds, "orientation", 0)) == getattr(self.ds, "orientation", 0)
        return "✎ custom crop" if has_crop else "auto (region bbox)"

    def _studio_edit_region_crop(self, region_name):
        """Open the Crop Studio for a region so its close-up crop can be set/adjusted. Returns
        True if a crop was saved."""
        rg = self._studio_find_region(region_name)
        if rg is None:
            return False
        return self._open_crop_studio(rg)

    def _studio_list_groups(self):
        """Structured, read-only enumeration of the feature-list registries for the picker.

        Returns ``[{badge, name, kind, items:[{text, rec, is_class}]}]`` over the working
        scopes, the saved ★ lists and the ◆ lipid-class lists — built straight from
        ``_feature_scopes`` / ``_feature_lists`` / ``_lipid_lists`` so ``self.peaks`` is never
        touched."""
        groups = []
        # working scopes (peaks) — "All slide" + per-region picked lists
        for name in self._display_scope_names():
            peaks = (getattr(self, "_feature_scopes", {}) or {}).get(name) or []
            items = []
            for p in peaks:
                mz = p.get("mz")
                if mz is None:
                    continue
                lab = p.get("label_override") or self._clean_label(mz)
                items.append({"text": self._ion_text(mz, lab),
                              "rec": {"mz": float(mz), "label": lab, "list_name": name,
                                      "source_kind": "scope", "color": p.get("color", "")},
                              "is_class": False})
            if items:
                groups.append({"badge": "▣", "name": name, "kind": "scope", "items": items})
        # saved ★ feature lists
        for name in sorted(getattr(self, "_feature_lists", {}) or {}):
            items = []
            for f in self._feature_lists[name]:
                mz = f.get("mz")
                if mz is None:
                    continue
                lab = f.get("lipid", "") or self._clean_label(mz)
                items.append({"text": self._ion_text(mz, lab),
                              "rec": {"mz": float(mz), "label": lab, "list_name": name,
                                      "source_kind": "saved"},
                              "is_class": False})
            if items:
                groups.append({"badge": "★", "name": name, "kind": "saved", "items": items})
        # ◆ lipid-class lists — each class is a composite (member m/z carried for expand)
        for name in sorted(getattr(self, "_lipid_lists", {}) or {}):
            items = []
            for e in self._lipid_lists[name]:
                cls = e.get("class", "")
                mzs = [float(m) for m in (e.get("mzs") or [])]
                if not cls or not mzs:
                    continue
                items.append({"text": f"{cls}  ·  {len(mzs)} ion(s)",
                              "rec": {"mz": None, "label": cls, "list_name": name,
                                      "kind": "composite", "source_kind": "lipid_class",
                                      "lipid_class": cls, "member_mzs": mzs,
                                      "color": e.get("color", "")},
                              "is_class": True})
            if items:
                groups.append({"badge": "◆", "name": name, "kind": "lipidlist", "items": items})
        return groups

    @staticmethod
    def _ion_text(mz, label):
        return f"m/z {float(mz):.4f}" + (f"  ·  {label}" if label else "")

    # ----- run a compiled plan -------------------------------------------- #
    def run_studio_plan(self, plan):
        """Render a compiled :class:`StudioPlan` into a chosen folder, on the background
        worker. Returns True once the job is launched (so the dialog can close)."""
        d = filedialogs.get_existing_directory(self, "Export Studio — choose an output folder")
        if not d:
            return False
        out_dir = os.path.join(d, "export_studio")
        # freeze the live display settings so a later UI change can't alter a render mid-batch
        settings = {"ppm": float(self.ppm), "reduce": self.reduce, "norm": self.norm,
                    "weight": self.composite_weight,
                    "pixel_size_um": getattr(self.ds, "pixel_size_um", None)}
        # Shared contrast: anchor 100% to the pooled clip-percentile across sections (not each
        # section's own max), and record what that reference means so a figure/caption is honest.
        _clip = float(plan.design.get("clip", 99.0))
        _shared = studio.ContrastMode(plan.contrast) == studio.ContrastMode.SHARED
        _norm_label = {"rms": "RMS-normalized", "tic": "TIC-normalized",
                       "none": "raw (un-normalized)"}.get(self.norm, str(self.norm))
        contrast_pct = _clip if _shared else None
        contrast_note = (f"100% = pooled {_clip:g}th percentile across sections · {_norm_label}"
                         if _shared else None)
        self._studio_contrast_note_text = contrast_note   # frozen; the matrix figure captions it

        from .. import cohort as cohort_engine
        section_loader = cohort_engine.SectionLoader()
        self._studio_section_crops = {}          # section.sid → {bbox, mask2d, color} (region close-ups)

        loader = self._studio_make_loader(section_loader)
        extract = self._studio_make_extract(settings)
        render_image, build_spec = self._studio_make_render(plan, settings)
        measure = self._studio_measure
        release = self._studio_make_release(section_loader)
        write_outputs = self._studio_make_write_outputs(plan, settings)
        # Fan the (CPU-bound, independent) ion-panel renders across processes — the batch's
        # bottleneck. Capped so a big machine doesn't spawn dozens of matplotlib workers for a
        # moderate batch; run_plan renders serially below its own small-batch threshold.
        workers = max(1, min((os.cpu_count() or 2) - 1, 8))

        def job(progress=None):
            try:
                return studio.run_plan(
                    plan, loader=loader, extract=extract, render_image=render_image,
                    measure=measure, contrast_percentile=contrast_pct, contrast_note=contrast_note,
                    write_outputs=write_outputs, release=release,
                    out_dir=out_dir, progress=(lambda i, n, m: progress(i, n) if progress else None),
                    render_cell=export.render_cell, build_spec=build_spec, max_workers=workers)
            finally:
                section_loader.close()           # free every cohort cube the run loaded

        def done(result):
            if result is None:
                return
            self._export_pending = getattr(self, "_export_pending", [])
            self._export_pending.append(out_dir)
            if getattr(self, "prov", None) is not None:
                for p in result.written:
                    self.prov.output(p, description="export studio")
            n_done = sum(1 for j in result.jobs if j.status == "done")
            extra = ", ".join(os.path.basename(str(v)) for k, v in result.extra.items()
                              if not k.startswith("_"))
            msg = f"Export Studio → {out_dir}  ·  {n_done} image(s)"
            if result.skipped:
                msg += f", {result.skipped} skipped"
            if extra:
                msg += f"  ·  {extra}"
            if result.cancelled:
                msg = "Export Studio cancelled (partial output written)  ·  " + msg
            self.statusBar().showMessage(msg)
            self._log_studio_to_report(plan, result)

        self._run(job, want_progress=True, busy="Export Studio — rendering…", on_done=done)
        return True

    # ----- injected callbacks --------------------------------------------- #
    def _studio_make_loader(self, section_loader):
        """Return ``loader(section) -> ds`` that resolves the live slide in-memory and loads
        each cohort section's cube once via the shared :class:`cohort.SectionLoader`. For a
        region section it also reconstructs the region's 2-D mask + bounding box (while the cube
        is live) and stashes it for the render close-up — the cube can then be evicted."""
        def loader(section):
            if section.is_live:
                if section.region:               # a region of the current slide → close-up crop
                    rg = self._studio_find_region(section.region)
                    if rg is not None:
                        self._studio_section_crops[section.sid] = {
                            "bbox": self._region_bbox(rg), "mask2d": self._region_mask_2d(rg),
                            "color": rg.get("color", "#ffffff")}
                return self.ds
            got = section_loader.load(section)   # Section duck-types as a SampleRef (.source/.region)
            if got is None:
                return None
            ds, pix = got
            if section.region and pix is not None and pix.size:
                self._studio_section_crops[section.sid] = self._studio_region_crop(ds, section, pix)
            return ds
        return loader

    def _studio_make_release(self, section_loader):
        def release(load_key):
            if load_key != "<live>":             # the live slide stays put; cohort cubes evict
                section_loader.release(load_key)
        return release

    def _studio_region_crop(self, ds, section, pix):
        """A region section's close-up spec: its 2-D footprint mask, bbox and outline colour,
        from the region's pixel indices into ``ds``. Computed while the cube is live so the cube
        can be evicted before rendering."""
        try:
            vec = np.zeros(ds.n_pixels, dtype=float)
            pix = pix[(pix >= 0) & (pix < ds.n_pixels)]
            vec[pix] = 1.0
            mask2d = np.asarray(ds.to_image(vec, fill=0.0)) > 0.5
            rows = np.any(mask2d, axis=1); cols = np.any(mask2d, axis=0)
            bbox = None
            if rows.any() and cols.any():
                r0, r1 = np.where(rows)[0][[0, -1]]
                c0, c1 = np.where(cols)[0][[0, -1]]
                pad_r = max(2, int(0.06 * (r1 - r0 + 1)))
                pad_c = max(2, int(0.06 * (c1 - c0 + 1)))
                bbox = (max(0, r0 - pad_r), min(mask2d.shape[0], r1 + 1 + pad_r),
                        max(0, c0 - pad_c), min(mask2d.shape[1], c1 + 1 + pad_c))
            return {"bbox": bbox, "mask2d": mask2d, "color": "#ffffff"}
        except Exception:  # noqa: BLE001 — a bad region just renders the full section
            return {"bbox": None, "mask2d": None, "color": "#ffffff"}

    def _studio_make_extract(self, settings):
        ppm = settings["ppm"]; reduce = settings["reduce"]; norm = settings["norm"]
        weight = settings.get("weight", "raw")

        def extract(ds, section, sel):
            mzr = getattr(ds, "mz_range", None)
            if sel.kind == "composite":
                mzs = [m for m in sel.member_mzs if mzr is None or mzr[0] <= m <= mzr[1]]
                if not mzs:
                    return None
                img = ds.composite_image(mzs, tol_ppm=ppm, reduce=reduce, norm=norm, weight=weight)
            else:
                if sel.mz is None or (mzr is not None and not (mzr[0] <= sel.mz <= mzr[1])):
                    return None              # out of this section's acquired range → skip
                img = ds.ion_image(float(sel.mz), tol_ppm=ppm, reduce=reduce, norm=norm)
            # for a region section, restrict to the region so the shared anchor + the rendered
            # close-up reflect the region's signal, not the whole slide around it
            crop = self._studio_section_crops.get(section.sid)
            if crop is not None and crop.get("mask2d") is not None:
                m = np.asarray(crop["mask2d"], bool)
                img = np.asarray(img, float)
                if m.shape == img.shape:
                    img = np.where(m, img, np.nan)
            return img
        return extract

    @staticmethod
    def _studio_measure(arr):
        """Per-section anchor for shared contrast: the max tissue intensity (so the shared
        window spans the brightest section's signal)."""
        a = np.asarray(arr, float)
        finite = a[np.isfinite(a)]
        return float(finite.max()) if finite.size else 0.0

    def _studio_make_render(self, plan, settings):
        """Return ``(render_image, build_spec)``. ``build_spec`` produces the picklable per-cell
        render kwargs (no Qt, no live state — the styling is frozen in ``design``) so the render
        phase can fan out to worker processes; ``render_image`` is the serial fallback, expressed
        in terms of the same spec so the two paths render byte-identically."""
        clip = float(plan.design.get("clip", 99.0))
        lo, hi = self._studio_window(plan.design)
        design = self._studio_design_kwargs(plan.design)
        dpi = int(plan.design.get("dpi", 300))
        ext = plan.ext
        style_spec = plan.design.get("style_spec")   # applied per-worker in export.render_cell

        def build_spec(arr, selection, section, anchor):
            kw = dict(mz=selection.mz, label=selection.label,
                      low=lo, high=hi, window=(lo, hi), clip=clip, ppm=settings["ppm"],
                      subject_mask=np.isfinite(np.asarray(arr, float)),
                      title=f"{section.name} · {selection.label or ''}".strip(" ·"),
                      anchor=anchor)        # shared per-ion absolute window when contrast=SHARED
            kw.update(design)
            crops = getattr(self, "_studio_section_crops", None) or {}
            crop = crops.get(section.sid)
            if crop is not None:            # a cohort region section → close-up + outline
                kw["crop"] = crop.get("bbox")
                kw["outline_mask"] = crop.get("mask2d")
                kw["outline_color"] = crop.get("color", "#ffffff")
            return {"kw": kw, "dpi": dpi, "fmt": ext, "style": style_spec}

        def render_image(arr, *, selection, section, anchor, path):
            export.render_cell(arr, build_spec(arr, selection, section, anchor), path)

        return render_image, build_spec

    @staticmethod
    def _studio_window(design):
        """The ``(lo, hi)`` relative-intensity window for the batch: the user's standardised
        override when 'Override intensity window' is on, else the full ``(0, 100)`` range (each
        cell auto-scales to its own clip/anchor). ``lo``/``hi`` are percentages of each cell's
        100% reference, so one window is comparable across sections in both contrast modes."""
        if design.get("window_override"):
            w = design.get("window")
            if isinstance(w, (list, tuple)) and len(w) == 2:   # tolerate a malformed saved plan
                return float(w[0]), float(w[1])
        return 0.0, 100.0

    def _studio_scale_bar_um(self, design):
        """Automatic scale-bar length for the batch (a round 1-2-5 length ≈20% of the active
        slide, refit to any crop), or None when the bar is toggled off or the pixel size is
        unknown. Back-compat: an old saved plan stored a length in ``scalebar_um`` (>0 == on)."""
        on = design.get("scalebar_on", design.get("scalebar_um", 0) > 0)
        return self.ds.auto_scale_bar_um() if (on and getattr(self, "ds", None) is not None) else None

    def _studio_design_kwargs(self, design):
        """Styling kwargs for :func:`export.render_ion_panel` from the Studio's frozen design
        dict (independent of the live spinboxes, so a batch renders consistently)."""
        px = getattr(self.ds, "pixel_size_um", None)
        sb = self._studio_scale_bar_um(design)
        return dict(cmap=design.get("cmap", "viridis"), theme=design.get("theme", "light"),
                    width_in=7.0, dpi=int(design.get("dpi", 300)), pixel_size_um=px,
                    scale_bar_um=sb, show_scalebar=bool(sb), show_spectrum=False,
                    label_mode=design.get("label_mode", annot.LABEL_MZ_PPM))

    def _studio_make_write_outputs(self, plan, settings):
        """Build the non-per-cell outputs from the already-extracted cells: the combined matrix
        figure, one colour overlay per section, the feature-list CSVs and the analyses bundle."""
        def write_outputs(_plan, cells, anchors, out_dir):
            extra = {}
            if plan.wants(studio.OutputKind.MATRIX):
                p = self._studio_write_matrix(plan, cells, anchors, out_dir)
                if p:
                    extra["matrix"] = p
            if plan.wants(studio.OutputKind.OVERLAY):
                p = self._studio_write_overlays(plan, cells, anchors, out_dir, settings)
                if p:
                    extra["overlays"] = p
            if plan.wants(studio.OutputKind.LIST_CSV):
                p = self._studio_write_list_csvs(plan, out_dir)
                if p:
                    extra["feature lists"] = p
            if plan.wants(studio.OutputKind.ANALYSES):
                p = self._studio_write_analyses(out_dir)
                if p:
                    extra["analyses"] = p
            return extra
        return write_outputs

    def _studio_write_matrix(self, plan, cells, anchors, out_dir):
        """The combined ion-row × section-column contact sheet from the already-extracted
        cells, honouring the shared per-ion anchor when contrast is shared."""
        rows = plan.selections
        cols = plan.sections
        if not rows or not cols:
            return None

        def cell_for(sec, sel):
            arr = cells.get((sec.sid, sel.uid()))
            if arr is None:
                return None
            crop = self._studio_section_crops.get(sec.sid) if hasattr(self, "_studio_section_crops") else None
            bbox = crop.get("bbox") if crop else None
            if bbox is not None:                 # frame a region section's cell to its close-up
                r0, r1, c0, c1 = bbox
                arr = np.asarray(arr)[r0:r1, c0:c1]
            return arr

        grid = [[cell_for(sec, sel) for sec in cols] for sel in rows]
        if all(c is None for row in grid for c in row):
            return None                          # nothing extracted → no matrix
        shared = studio.ContrastMode(plan.contrast) == studio.ContrastMode.SHARED
        row_anchors = [anchors.get(sel.uid()) if shared else None for sel in rows]
        row_labels = [self._studio_row_label(s) for s in rows]
        col_labels = [s.name for s in cols]
        design = plan.design
        lo, hi = self._studio_window(design)
        px = getattr(self.ds, "pixel_size_um", None)
        sb = self._studio_scale_bar_um(design)
        with export.active_style(design.get("style_spec")):   # match the batch's panel look
            fig = export.render_matrix_figure(
                grid, row_labels=row_labels, col_labels=col_labels,
                cmap=design.get("cmap", "viridis"), low=lo, high=hi,
                clip=float(design.get("clip", 99.0)), row_anchors=row_anchors,
                theme=design.get("theme", "light"), dpi=int(design.get("dpi", 300)),
                title="Export Studio — ions × sections", pixel_size_um=px, scale_bar_um=sb,
                subtitle=getattr(self, "_studio_contrast_note_text", None))
        # a matrix is a single sheet; PDF/SVG/PNG all fine, but force a sheet-friendly ext
        ext = plan.ext if plan.ext in ("png", "tiff", "pdf", "svg", "jpg") else "png"
        path = os.path.join(out_dir, f"matrix.{ext}")
        export.save_figure(fig, path, dpi=int(design.get("dpi", 300)), fmt=ext)
        return path

    def _studio_write_overlays(self, plan, cells, anchors, out_dir, settings):
        """One additive multi-ion colour composite per section: every picked ion stretched by the
        batch's intensity window, tinted by its feature colour (or a CVD-safe fallback hue), and
        summed so co-located ions mix — the same composite the Export hub and Report tab build,
        rendered full-frame with a channel legend. Region sections zoom to their close-up crop.
        Returns the overlays folder, or ``None`` when nothing composited."""
        from matplotlib.colors import to_rgb

        sels = plan.selections
        if not sels or not plan.sections:
            return None
        design = plan.design
        lo, hi = self._studio_window(design)
        clip = float(design.get("clip", 99.0))
        shared = studio.ContrastMode(plan.contrast) == studio.ContrastMode.SHARED
        px = getattr(self.ds, "pixel_size_um", None)
        sb = self._studio_scale_bar_um(design)
        theme = design.get("theme", "light")
        dpi = int(design.get("dpi", 300))
        label_mode = design.get("label_mode", annot.LABEL_MZ_PPM)
        ppm = settings.get("ppm")
        ext = plan.ext if plan.ext in ("png", "tiff", "jpg", "pdf", "svg") else "png"

        # a stable channel colour per selection: its own colour when set, else the next CVD-safe
        # overlay hue — so the same ion keeps one colour across every section's composite.
        fallback = palettes.overlay_colors(n=len(sels))
        colors, auto = {}, 0
        for sel in sels:
            c = (getattr(sel, "color", "") or "").strip()
            if not c:
                c = fallback[auto % len(fallback)]
                auto += 1
            colors[sel.uid()] = c

        crops = getattr(self, "_studio_section_crops", None) or {}
        folder = os.path.join(out_dir, "overlays")
        written = None
        for sec in plan.sections:
            acc, chans = None, []
            for sel in sels:
                arr = cells.get((sec.sid, sel.uid()))
                if arr is None:
                    continue
                a = np.asarray(arr, float)
                if acc is None:
                    acc = np.zeros((a.shape[0], a.shape[1], 3), float)
                anchor = anchors.get(sel.uid()) if shared else None
                norm = np.nan_to_num(imaging.relative_window(a, lo, hi, clip, anchor), nan=0.0)
                r, g, b = to_rgb(colors[sel.uid()])
                acc[..., 0] += norm * r
                acc[..., 1] += norm * g
                acc[..., 2] += norm * b
                chans.append(dict(mz=sel.mz, color=colors[sel.uid()], ppm=ppm, lo=lo, hi=hi,
                                  peak=imaging.relative_max(a, clip), label=sel.label))
            if acc is None:
                continue                                 # nothing extracted for this section
            rgb = (np.clip(acc, 0.0, 1.0) * 255).astype(np.uint8)
            crop = crops.get(sec.sid)
            bbox = crop.get("bbox") if crop else None
            outlines = None
            if crop and crop.get("mask2d") is not None:
                outlines = [(crop.get("mask2d"), crop.get("color", "#ffffff"))]
            with export.active_style(plan.design.get("style_spec")):   # match the batch look
                fig = export.render_overlay_panel(
                    rgb, chans, theme=theme, pixel_size_um=px, scale_bar_um=sb,
                    show_scalebar=bool(sb), show_legend=True, dpi=dpi, crop=bbox,
                    outlines=outlines, label_mode=label_mode)
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"overlay_{studio.safe_name(sec.name)}.{ext}")
            export.save_figure(fig, path, dpi=dpi, fmt=ext)
            written = folder
        return written

    @staticmethod
    def _studio_row_label(sel):
        if sel.kind == "composite":
            return sel.lipid_class or sel.label or "class"
        base = f"m/z {sel.mz:.4f}" if sel.mz is not None else "feature"
        return f"{base}\n{sel.label}" if sel.label else base

    def _studio_write_list_csvs(self, plan, out_dir):
        """One CSV per picked feature list (ions + a ``list`` traceability column)."""
        import pandas as pd
        folder = os.path.join(out_dir, "feature_lists")
        by_list: dict[str, list] = {}
        for sel in plan.selections:
            by_list.setdefault(sel.list_name or "features", []).append(sel)
        if not by_list:
            return None
        os.makedirs(folder, exist_ok=True)
        for name, sels in by_list.items():
            rows = []
            for s in sels:
                rows.append({"list": name, "kind": s.kind,
                             "mz": (round(s.mz, 4) if s.mz is not None else ""),
                             "lipid": s.label, "lipid_class": s.lipid_class,
                             "member_mzs": " ".join(f"{m:.4f}" for m in s.member_mzs)})
            export.write_table(pd.DataFrame(rows),
                               os.path.join(folder, studio.safe_name(name) + ".csv"), fmt="csv")
        return folder

    def _studio_write_analyses(self, out_dir):
        """The live slide's analyses bundle (feature table, region intensities, statistics,
        methods) — reuses the Export hub's data-bundle builder."""
        folder = os.path.join(out_dir, "analyses")
        doc = {}
        if self.peaks:
            fdf = getattr(self, "feat_df", None)
            doc["feature_table"] = fdf if fdf is not None else self._fallback_feature_df()
            rstats = self._region_intensity_table()
            if rstats is not None:
                doc["region_stats"] = rstats
        if getattr(self, "last_stats", None) is not None:
            la, lb = getattr(self, "_region_labels", ("Group A", "Group B"))
            doc["stats"] = {"df": self.last_stats, "a_label": la, "b_label": lb}
        doc["provenance"] = getattr(self, "prov", None)
        if "feature_table" not in doc and "stats" not in doc:
            return None                          # nothing to bundle (no peaks, no stats)
        export.build_data_bundle(doc, folder, options={"features": True, "region_stats": True,
                                                       "statistics": True, "spectra": False,
                                                       "methods": True})
        return folder

    # ----- report log ----------------------------------------------------- #
    def _log_studio_to_report(self, plan, result):
        """Best-effort summary note into the Report tab log (never blocks the export)."""
        try:
            if not hasattr(self, "_report_append"):
                return
            n_done = sum(1 for j in result.jobs if j.status == "done")
            text = (f"{len(plan.sections)} section(s) × {len(plan.selections)} ion(s) → "
                    f"{n_done} image(s)"
                    + (f", {result.skipped} skipped" if result.skipped else "")
                    + f". Outputs: {', '.join(studio.OutputKind(o).value for o in plan.outputs)} "
                    f"→ {os.path.basename(result.out_dir)}/")
            self._report_append({"type": "note", "title": "Export Studio",
                                 "heading": "Export Studio", "text": text, "caption": "",
                                 "name_origin": "auto"})
        except Exception:  # noqa: BLE001
            pass
