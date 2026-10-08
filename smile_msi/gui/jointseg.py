"""CohortSegMixin — joint (consensus) segmentation across several slides at once.

The single-slide Segmentation / Feature-space tabs cluster one ``self.ds`` and so can
only ever show one segmentation. This tab builds **one** granularity tree over the pixels
of *several* cohort samples on a shared feature axis (see
:func:`smile_msi.spatial.joint_hierarchy`), so a single Detail cut assigns the **same
cluster ids to every slide** — "cluster 3" means the same molecular profile in each
sample. That's the cross-slide comparison independent per-sample runs can't give (their
labels are unrelated), and the natural sibling of the Cohort UMAP tab.

Like the Cohort tabs it reads the cohort roster and loads each slide's cube in a worker
(this one *does* need the pixels). Per-sample z-scoring before pooling ("Align samples")
keeps a slide's overall intensity offset from masquerading as biology. Each sample's
segmentation renders on its own grid in a shared palette; a cluster can be promoted to a
region on the active sample.
"""
from __future__ import annotations

import os

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .. import spatial, session, profiles
from .common import (PALETTE, MUTED_QSS, CheckList, ControlBar, add_copy_actions,
                     dark_image_view, fill_table,
                     icon, tab_page, tool_button, NoScrollComboBox, NoScrollDoubleSpinBox,
                     NoScrollSlider)


class CohortSegMixin:
    # ===================================================================== #
    # Tab construction
    # ===================================================================== #
    def _tab_cohort_segment(self):
        w, v = tab_page()
        bar = ControlBar()
        self.jseg_feat_combo = self._make_consensus_feature_combo(
            "Feature axis for the joint clustering:\n"
            "Consensus — a shared m/z axis clustered from every sample's saved peaks "
            "(recommended, so every slide spans the same molecular features).\n"
            "Or any named feature set — the working set, a region scope, or a saved ★ list, "
            "extracted from each slide.")
        self.jseg_method = NoScrollComboBox()
        self.jseg_method.addItem("Ward (agglomerative)", "ward")      # default
        self.jseg_method.addItem("Bisecting k-means", "bisecting")
        self.jseg_method.setToolTip("Clustering algorithm for the joint tree (as in the "
                                    "single-slide Segmentation tab).")
        self.jseg_metric = NoScrollComboBox()
        self.jseg_metric.addItem("Correlation", "correlation")
        self.jseg_metric.addItem("Euclidean", "euclidean")
        self.jseg_metric.setToolTip("Distance metric: correlation groups pixels by spectral "
                                    "shape, Euclidean by overall profile.")
        self.jseg_align_chk = QtWidgets.QCheckBox("Align samples")
        self.jseg_align_chk.setChecked(True)
        self.jseg_align_chk.setToolTip(
            "Per-sample z-score each feature before pooling, so a slide's overall intensity "
            "offset/scale can't dominate the joint clusters (a light batch alignment). "
            "Strongly recommended across independent acquisitions.")
        self.jseg_tol_spin = NoScrollDoubleSpinBox()
        self.jseg_tol_spin.setRange(1.0, 200.0)
        self.jseg_tol_spin.setValue(20.0)
        self.jseg_tol_spin.setSuffix(" ppm")
        self.jseg_tol_spin.setToolTip("Consensus only: m/z tolerance for merging peaks across "
                                      "samples into shared features.")
        self.jseg_prev_spin = NoScrollDoubleSpinBox()
        self.jseg_prev_spin.setRange(0.0, 1.0)
        self.jseg_prev_spin.setSingleStep(0.1)
        self.jseg_prev_spin.setValue(0.5)
        self.jseg_prev_spin.setToolTip("Consensus only: keep ions present in at least this "
                                       "fraction of samples.")
        b_samples = QtWidgets.QPushButton("Samples…")
        b_samples.setIcon(icon("open"))
        b_samples.setToolTip("Open the Samples panel to add/label the slides this segments")
        b_samples.clicked.connect(self._reveal_samples_panel)
        b_run = QtWidgets.QPushButton("Run joint segmentation")
        b_run.setObjectName("primaryAction")              # the tab's single run action
        b_run.setIcon(icon("run"))
        b_run.clicked.connect(self._jseg_run)
        for lab, wdg in (("Features", self.jseg_feat_combo), ("Algorithm", self.jseg_method),
                         ("Distance", self.jseg_metric), ("Match", self.jseg_tol_spin),
                         ("Min prevalence", self.jseg_prev_spin)):
            bar.add_group(lab, wdg)
        bar.add(self.jseg_align_chk, b_samples, b_run)
        # --- Detail slider: cut the prebuilt joint tree live, all slides together ---
        self.jseg_detail = NoScrollSlider(QtCore.Qt.Horizontal)
        self.jseg_detail.setRange(2, 24)
        self.jseg_detail.setValue(6)
        self.jseg_detail.setEnabled(False)
        self.jseg_detail.setMaximumWidth(200)
        self.jseg_detail.setToolTip("Drag to re-segment every slide live: left = a few big "
                                    "regions, right = many fine ones. One shared model.")
        self.jseg_detail.valueChanged.connect(self._jseg_detail_changed)
        self.jseg_detail.sliderPressed.connect(lambda: setattr(self, "_jseg_dragging", True))
        self.jseg_detail.sliderReleased.connect(self._jseg_detail_released)
        self._jseg_detail_timer = QtCore.QTimer(self)   # debounce live re-segmentation
        self._jseg_detail_timer.setSingleShot(True)
        self._jseg_detail_timer.timeout.connect(self._jseg_commit_live)
        self.jseg_detail_lbl = QtWidgets.QLabel("")
        bar.add_group("Detail", QtWidgets.QLabel("coarse"), self.jseg_detail,
                      QtWidgets.QLabel("fine"), self.jseg_detail_lbl)
        self.b_jseg_png = tool_button(
            name="export", tooltip="Export the segmentation panels as a publication figure "
                                    "(live preview + theme / columns + Style preset).",
            slot=self._export_jseg_dialog)
        self.b_jseg_png.setEnabled(False)
        bar.add(self.b_jseg_png)
        self.jseg_info = bar.set_status(
            "Add ≥2 samples to the cohort (Samples panel), tick the ones to segment on the "
            "left, then Run: one tree is built over every slide so the same cluster id means "
            "the same tissue in each.")
        v.addWidget(bar)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        # --- left: which cohort samples to segment ---------------------------
        leftw = QtWidgets.QWidget()
        leftv = QtWidgets.QVBoxLayout(leftw)
        leftv.setContentsMargins(0, 0, 0, 0)
        leftv.setSpacing(2)
        leftv.addWidget(QtWidgets.QLabel("Samples to segment"))
        self.jseg_sample_list = CheckList(noun="sample")
        self.jseg_sample_list.setToolTip("Tick the cohort slides to segment jointly "
                                         "(imzML-backed samples only).")
        leftv.addWidget(self.jseg_sample_list, 1)
        note = QtWidgets.QLabel("Only imzML-backed samples can be segmented (the pixels are "
                                "loaded). Set groups in the Samples panel.")
        note.setWordWrap(True)
        note.setStyleSheet(MUTED_QSS)
        leftv.addWidget(note)
        leftw.setMaximumWidth(240)
        split.addWidget(leftw)

        # --- centre: one segmentation panel per slide (shared palette) -------
        midw = QtWidgets.QWidget()
        midv = QtWidgets.QVBoxLayout(midw)
        midv.setContentsMargins(0, 0, 0, 0)
        midv.setSpacing(2)
        self.jseg_view = pg.GraphicsLayoutWidget()
        dark_image_view(self.jseg_view)   # joint-seg map montage reads against black in both themes
        midv.addWidget(self.jseg_view, 1)
        self.jseg_legend = QtWidgets.QLabel()
        self.jseg_legend.setWordWrap(True)
        self.jseg_legend.setTextFormat(QtCore.Qt.RichText)
        self.jseg_legend.setStyleSheet(MUTED_QSS)
        midv.addWidget(self.jseg_legend)
        split.addWidget(midw)

        # --- right: shared cluster signatures --------------------------------
        rightw = QtWidgets.QWidget()
        rightv = QtWidgets.QVBoxLayout(rightw)
        rightv.setContentsMargins(0, 0, 0, 0)
        rightv.setSpacing(2)
        rightv.addWidget(QtWidgets.QLabel("Shared clusters — right-click to make a region"))
        self.jseg_table = QtWidgets.QTableWidget()
        self.jseg_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.jseg_table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.jseg_table.customContextMenuRequested.connect(self._jseg_table_menu)
        rightv.addWidget(self.jseg_table, 1)
        rightw.setMaximumWidth(360)
        split.addWidget(rightw)

        split.setChildrenCollapsible(False)
        split.setSizes([220, 760, 320])
        v.addWidget(split, 1)
        self._jseg_items, self._jseg_vbs = [], []
        self._jseg_dragging = False
        self._cohort_tabs.addTab(w, "Cohort segmentation")
        self.tabs.currentChanged.connect(self._jseg_on_tab_changed)

    # ===================================================================== #
    # Sample picker
    # ===================================================================== #
    def _jseg_on_tab_changed(self, *_):
        # nesting-aware: Cohort segmentation is a sub-tab of the "Cohort" container now.
        if self._view_visible("Cohort segmentation"):
            self._jseg_refresh_sample_list()

    def _jseg_imzml_refs(self):
        """Cohort samples that can be segmented: imzML-backed and the file still on disk."""
        out = []
        for r in getattr(getattr(self, "cohort", None), "samples", []) or []:
            src = getattr(r, "source", "") or ""
            if src.lower().endswith(".imzml") and os.path.exists(src):
                out.append(r)
        return out

    def _jseg_refresh_sample_list(self):
        """Rebuild the checkable sample list from the cohort, preserving ticks by key.
        Fresh rosters default to every imzML sample ticked (so Run works out of the box)."""
        lw = getattr(self, "jseg_sample_list", None)
        if lw is None:
            return
        entries = [(r.key(), r.name, getattr(r, "group", "") or "")
                   for r in self._jseg_imzml_refs()]
        lw.set_entries(entries, check_new=True)           # a sample joins the run by default

    def _jseg_checked_refs(self):
        lw = getattr(self, "jseg_sample_list", None)
        if lw is None:
            return []
        keys = lw.checked_keys()
        return [r for r in self._jseg_imzml_refs() if r.key() in keys]

    # ===================================================================== #
    # Run
    # ===================================================================== #
    def _jseg_run(self):
        self._jseg_refresh_sample_list()
        refs = self._jseg_checked_refs()
        if len(refs) < 2:
            self.statusBar().showMessage(
                "Tick at least two imzML-backed cohort samples to segment jointly "
                "— add them in the Samples panel.")
            self._reveal_samples_panel()
            return
        tol = float(self.jseg_tol_spin.value())
        targets = self._cohort_feature_targets(
            self.jseg_feat_combo, refs, tol_ppm=tol,
            min_prevalence=float(self.jseg_prev_spin.value()), value="rel_intensity")
        if len(targets) < 2:
            self.statusBar().showMessage("Need ≥2 shared features — lower min-prevalence, widen "
                                         "Match, or find peaks for the active feature set.")
            return
        method = self.jseg_method.currentData() or "ward"
        metric = self.jseg_metric.currentData() or "correlation"
        align = self.jseg_align_chk.isChecked()
        ext_ppm, norm = self.ppm, self.norm
        active_src = os.path.abspath(self.ds.source) if (self.ds is not None and self.ds.source) else ""

        def work(progress=None):
            from ..msi import MSIDataset           # lazy: keep heavy imports off GUI startup
            cache: dict[str, object] = {}           # one slide load shared by its region samples
            loaded = []                             # (name, ds, mask, is_active)
            total = len(refs) + 1
            for si, r in enumerate(refs):
                if progress is not None:
                    progress(si, total)
                src = getattr(r, "source", "") or ""
                if not src.lower().endswith(".imzml") or not os.path.exists(src):
                    continue
                if src not in cache:
                    ds = MSIDataset.from_imzml(src, lazy=True)
                    ds.to_ram()
                    ds.prime()
                    cache[src] = ds
                ds = cache[src]
                mask = None
                if getattr(r, "region", ""):
                    try:
                        data = session.load_session(r.session_path)
                    except (OSError, ValueError):
                        continue
                    idx = None
                    for rg in (data.get("named_regions") or []):
                        if rg.get("name") == r.region:
                            idx = np.asarray(rg.get("mask") or [], dtype=int)
                            break
                    if idx is None or idx.size == 0:
                        continue
                    mask = np.zeros(ds.n_pixels, dtype=bool)
                    idx = idx[(idx >= 0) & (idx < ds.n_pixels)]
                    mask[idx] = True
                is_active = (bool(active_src) and os.path.abspath(src) == active_src
                             and mask is None)
                loaded.append((r.name, ds, mask, is_active))
            if len(loaded) < 2:
                return None
            datasets = [t[1] for t in loaded]
            masks = [t[2] for t in loaded]
            names = [t[0] for t in loaded]
            jh = spatial.joint_hierarchy(datasets, targets, masks=masks, names=names,
                                         tol_ppm=ext_ppm, norm=norm, method=method, metric=metric,
                                         standardize_per_sample=align,
                                         random_state=profiles.active_seed(),
                                         release_cubes=True)   # hold ≤1 cube — geometry survives for label maps
            if progress is not None:
                progress(total, total)
            actives = [i for i, t in enumerate(loaded) if t[3]]
            return {"datasets": datasets, "names": names, "jh": jh,
                    "active_idx": actives[0] if actives else None, "targets": targets}

        self._run(work, want_progress=True,
                  busy=f"Joint segmentation over {len(refs)} slides…",
                  on_done=self._jseg_on_built)

    def _jseg_on_built(self, result):
        if not result:
            self.statusBar().showMessage("Joint segmentation needs ≥2 loadable imzML slides "
                                         "with signal on the shared feature axis.")
            return
        self._jseg_datasets = result["datasets"]
        self._jseg_names = result["names"]
        self._jseg_jh = result["jh"]
        self._jseg_active_idx = result["active_idx"]
        self._jseg_targets = result["targets"]
        jh = self._jseg_jh
        hi = int(min(24, max(2, jh.max_clusters)))
        self.jseg_detail.blockSignals(True)
        self.jseg_detail.setRange(2, hi)
        k0 = int(min(max(2, self.jseg_detail.value()), hi))
        self.jseg_detail.setValue(k0)
        self.jseg_detail.setEnabled(True)
        self.jseg_detail.blockSignals(False)
        self._jseg_dragging = False
        self.b_jseg_png.setEnabled(True)
        self._jseg_commit(k0, with_table=True)
        n = len(self._jseg_datasets)
        ev = jh.hier.explained_variance
        ev_txt = f" · PCA EV {ev:.0%}" if np.isfinite(ev) else ""
        self.statusBar().showMessage(
            f"Joint tree built over {n} slides ({len(self._jseg_targets)} shared features"
            f"{ev_txt}). Drag Detail to re-segment every slide together.")

    # ===================================================================== #
    # Detail slider → cut + render
    # ===================================================================== #
    def _jseg_detail_changed(self, value):
        if getattr(self, "_jseg_jh", None) is None:
            return
        self.jseg_detail_lbl.setText(f"{value} segments")
        if getattr(self, "_jseg_dragging", False):
            # Mid-drag: debounce — coalesce rapid ticks into one re-cut+repaint ~150 ms after
            # movement settles, instead of re-segmenting every slide on every intermediate value.
            self._jseg_detail_timer.start(150)
        else:
            # A single keyboard/click/programmatic change commits immediately (with table).
            self._jseg_commit(value, with_table=True)

    def _jseg_commit_live(self):
        """Debounced Detail-slider commit: maps only while dragging, full commit (with table)
        once the drag settles or on a keyboard/click change."""
        if getattr(self, "_jseg_jh", None) is None:
            return
        self._jseg_commit(self.jseg_detail.value(),
                          with_table=not getattr(self, "_jseg_dragging", False))

    def _jseg_detail_released(self):
        self._jseg_dragging = False
        self._jseg_detail_timer.stop()
        self._jseg_commit(self.jseg_detail.value(), with_table=True)

    def _jseg_commit(self, k, with_table=False):
        jh = getattr(self, "_jseg_jh", None)
        if jh is None:
            return
        segs = spatial.joint_segmentation_at(self._jseg_datasets, jh, int(k),
                                             with_silhouette=False)
        self._jseg_segs = segs
        self._jseg_k = segs[0].n_clusters if segs else 0
        self._jseg_render(segs)
        if with_table:
            self._jseg_update_table(segs)

    def _jseg_render(self, segs):
        """(Re)build the per-slide panel grid and paint each segmentation in the shared
        palette. Panels are rebuilt when the sample set changes, else just re-imaged."""
        names = self._jseg_names
        need_rebuild = len(self._jseg_items) != len(segs)
        if need_rebuild:
            self.jseg_view.clear()
            self._jseg_items, self._jseg_vbs = [], []
            cols = max(1, min(3, len(segs)))
            for i, name in enumerate(names):
                r, c = (i // cols) * 2, i % cols
                self.jseg_view.addLabel(name, row=r, col=c)
                vb = self.jseg_view.addViewBox(row=r + 1, col=c)
                vb.invertY(True)
                vb.setAspectLocked(True)
                img = pg.ImageItem()
                vb.addItem(img)
                self._jseg_vbs.append(vb)
                self._jseg_items.append(img)
        for img, vb, seg in zip(self._jseg_items, self._jseg_vbs, segs):
            img.setImage(self._jseg_rgba(seg))
            if need_rebuild:
                vb.autoRange()
        self._jseg_update_legend(self._jseg_k)

    def _export_jseg_dialog(self):
        """Publication export of the per-slide segmentation montage (live preview + theme /
        columns + Style preset), replacing the raw pyqtgraph screenshot."""
        segs = getattr(self, "_jseg_segs", None)
        if not segs:
            self.statusBar().showMessage("Run joint segmentation first.")
            return
        from .. import export
        from .plotexport import FigureExportDialog
        names = list(self._jseg_names)
        panels = [(names[i] if i < len(names) else f"Slide {i + 1}", seg.label_image)
                  for i, seg in enumerate(segs)]
        colors = {cl: self._jseg_color(cl) for cl in range(self._jseg_k)}

        def render(opts):
            # one scale bar per slide panel (slides can differ in pixel size)
            pixel_sizes = [getattr(ds, "pixel_size_um", None) for ds in self._jseg_datasets]
            return export.render_segmentation_grid(
                panels, colors, title=opts["title"], theme=opts["theme"],
                ncols=(int(opts["ncols"]) or None), dpi=opts["dpi"], pixel_sizes=pixel_sizes)

        options = [
            {"key": "title", "label": "Title", "kind": "text",
             "default": "Joint segmentation (shared clusters)"},
            {"key": "theme", "label": "Theme", "kind": "combo",
             "choices": [("Dark", "dark"), ("Light", "light")], "default": "dark"},
            {"key": "ncols", "label": "Columns (0 = auto)", "kind": "int",
             "min": 0, "max": 8, "default": 0},
        ]
        FigureExportDialog(self, render=render, options=options,
                           default_name="joint_segmentation.png",
                           title="Export segmentation").exec()

    def _jseg_color(self, cl):
        """Shared colour of joint cluster ``cl`` — tree-aware (siblings share a hue family)
        from the first slide's cut, else the cycling palette."""
        segs = getattr(self, "_jseg_segs", None) or []
        cols = getattr(segs[0], "colors", None) if segs else None
        return cols[cl] if cols and cl < len(cols) else PALETTE[cl % len(PALETTE)]

    @staticmethod
    def _jseg_rgba(seg):
        """Paint a segmentation onto an (h,w,4) image: each cluster its (tree-aware)
        colour, off-tissue (label_image NaN) transparent."""
        lab = seg.label_image
        h, wd = lab.shape
        rgba = np.zeros((h, wd, 4), dtype=np.ubyte)
        for cl in range(seg.n_clusters):
            mask = lab == cl
            if not mask.any():
                continue
            cols = getattr(seg, "colors", None)
            col = QtGui.QColor(cols[cl] if cols and cl < len(cols) else PALETTE[cl % len(PALETTE)])
            rgba[mask] = [col.red(), col.green(), col.blue(), 255]
        return rgba

    def _jseg_update_legend(self, k):
        chips = []
        for cl in range(min(k, 16)):
            col = self._jseg_color(cl)
            chips.append(f'<span style="background:{col}">&nbsp;&nbsp;&nbsp;</span>&nbsp;{cl}')
        note = f"  (+{k - 16} more)" if k > 16 else ""
        self.jseg_legend.setText(" &nbsp; ".join(chips)
                                 + f" &nbsp; <i>shared cluster colours{note}</i>")

    def _jseg_update_table(self, segs):
        """Per-cluster pixel counts (total + per sample) and pooled top-3 enriched lipids,
        on the shared feature axis — so a cluster's molecular identity reads at a glance."""
        k = self._jseg_k
        jh = self._jseg_jh
        targets = jh.targets
        names = self._jseg_names
        sums = np.zeros((k, len(targets)))
        counts = np.zeros(k)
        per_sample = np.zeros((len(segs), k), dtype=int)
        # Per-sample cluster pixel counts (from the labels; cheap, cube-free).
        per_lab = []
        for si, seg in enumerate(segs):
            mask = jh.masks[si] if si < len(jh.masks) else None
            lab_i = seg.labels if mask is None else seg.labels[mask]   # pooled-order chunk
            per_lab.append(lab_i)
            for cl in range(k):
                per_sample[si, cl] = int((lab_i == cl).sum())
        # Cluster mean signatures: prefer the pooled feature matrix retained on the hierarchy
        # (so released cubes don't matter); fall back to re-extracting from the cubes.
        feats = getattr(jh, "features", None)
        pooled_lab = np.concatenate(per_lab) if per_lab else np.zeros(0, dtype=int)
        if feats is not None and feats.shape[0] == pooled_lab.shape[0]:
            for cl in range(k):
                m = pooled_lab == cl
                if m.any():
                    sums[cl] = feats[m].sum(0)
                    counts[cl] = int(m.sum())
        else:                                            # legacy path (cubes still resident)
            for si, (ds, seg) in enumerate(zip(self._jseg_datasets, segs)):
                X = ds.feature_matrix(self.norm)
                lab = seg.labels
                for cl in range(k):
                    m = lab == cl
                    if m.any():
                        sums[cl] += X[m].sum(0)
                        counts[cl] += int(m.sum())
        rows = []
        for cl in range(k):
            if counts[cl] > 0:
                means = sums[cl] / counts[cl]
                order = np.argsort(-means)[:3]
                top = ", ".join(self.annotate(float(targets[j])) or f"m/z {targets[j]:.3f}"
                                for j in order)
            else:
                top = ""
            per = " / ".join(f"{int(per_sample[si, cl]):,}" for si in range(len(segs)))
            rows.append((str(cl), f"{int(counts[cl]):,}", per, top))
        heads = ["cluster", "total px", "px per sample (" + " / ".join(names) + ")", "top lipids"]
        fill_table(self.jseg_table, heads, rows)

    # ===================================================================== #
    # Cluster → region on the active sample
    # ===================================================================== #
    def _jseg_selected_cluster(self):
        rows = self.jseg_table.selectionModel().selectedRows()
        if not rows:
            return None
        it = self.jseg_table.item(rows[0].row(), 0)
        try:
            return int(it.text()) if it else None
        except ValueError:
            return None

    def _jseg_table_menu(self, pos):
        if getattr(self, "_jseg_segs", None) is None:
            return
        cl = self._jseg_selected_cluster()
        if cl is None:
            return
        menu = QtWidgets.QMenu(self.jseg_table)
        ai = getattr(self, "_jseg_active_idx", None)
        can = ai is not None and self._jseg_active_alignable()
        act = menu.addAction(f"New region from cluster {cl} (active sample)")
        act.setIcon(icon("add"))
        act.setEnabled(can)
        if not can:
            act.setToolTip("The active sample must be one of the jointly-segmented whole "
                           "slides with a matching pixel grid.")
        act.triggered.connect(lambda: self._jseg_cluster_to_region(cl))
        menu.addSeparator()
        add_copy_actions(menu, self.jseg_table)
        menu.exec(self.jseg_table.viewport().mapToGlobal(pos))

    def _jseg_active_alignable(self):
        """True when cluster labels for the active sample can be mapped back onto
        ``self.ds`` — i.e. it was one of the segmented slides and the freshly-loaded copy
        has the same pixel count (not a stride/preview load)."""
        ai = getattr(self, "_jseg_active_idx", None)
        if ai is None or self.ds is None:
            return False
        return self._jseg_datasets[ai].n_pixels == self.ds.n_pixels

    def _jseg_cluster_to_region(self, cl):
        if not self._jseg_active_alignable():
            self.statusBar().showMessage("Can't map this cluster back to the active sample.")
            return
        seg = self._jseg_segs[self._jseg_active_idx]
        mask = seg.labels == cl
        if not mask.any():
            self.statusBar().showMessage(f"Cluster {cl} has no pixels on the active sample.")
            return
        self.record_undo("joint cluster → region", domains=("regions",))
        i = self._new_region(f"Joint cluster {cl}", mask=mask.copy())
        if hasattr(self, "reveal_view"):
            self.reveal_view("Ion image")      # grouped-tab safe (raw index is brittle post-IA)
        self.statusBar().showMessage(
            f"Region '{self.regions[i]['name']}' created on the active sample from cluster "
            f"{cl} — {int(mask.sum()):,} pixels. Find it in the Regions panel.")
