"""SMILE MSI desktop main window (PySide6 + pyqtgraph).

The MainWindow is assembled from per-tab mixins (see ion.py, features.py,
segment.py, montage.py, stats.py, analysis.py, librarytab.py); shared helpers
live in common.py. Every method still runs on one combined instance, so
cross-tab ``self.*`` calls resolve exactly as before.
"""
from __future__ import annotations

import copy
import os
from contextlib import contextmanager

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .. import (spatial, imaging, preprocess, provenance, session, library,
                prefs, profiles, intake)
from .. import APP_NAME
from ..match import Annotator
from ..msi import MSIDataset
import sys
from . import filedialogs
from . import common  # common.icon(...) — bare `icon` is shadowed locally in this file
from .workers import Worker
from .common import (apply_theme, set_theme, retheme_open_plots, THEME_MODES, DEFAULT_THEME,
                     colormap, confirm, load_demo, load_imzml, pick_and_build, run_segment,
                     run_components, section_title, note, tab_page, NoScrollComboBox,
                     NoScrollDoubleSpinBox, NoScrollSpinBox)
from .ion import IonTabMixin
from .features import FeaturesTabMixin
from .segment import SegmentTabMixin
from .montage import MontageTabMixin
from .stats import StatsTabMixin
from .classcompare import ClassCompareMixin
from .lipidlists import LipidListMixin
from .lipidtree import LipidTreeMixin
from .analysis import AnalysisTabsMixin
from .librarytab import LibraryTabMixin
from .featurepanel import RightPanelMixin
from .exportdialog import ExportMixin
from .studiodialog import StudioMixin
from .umapstudiodialog import UMAPStudioMixin
from .reporttab import ReportTabMixin
from .analyses import HistoryMixin
# The Flow designer (gui/flowdialog.py) was retired in plan 24; the analysis registry lives on
# in smile_msi.registry and the batch/replay surface is the scripting API. Saved flows migrate
# to scripting presets via scripts/migrate_flows_to_presets.py.
from .cohortview import CohortMixin
from .jointseg import CohortSegMixin
from .scope import ScopeMixin
from .optical import OpticalMixin
from .gallery import GalleryMixin
from .scriptconsole import ScriptConsoleMixin
from .audit import AuditMixin


HELP_HTML = """
<h2>SMILE MSI — Quick start</h2>
<ol>
<li><b>Load data</b> — <i>File → Open imzML…</i>, <i>Open table (CSV/TSV)…</i>, or <i>Load demo dataset</i>.</li>
<li>Set <b>Mode</b> / tolerance under <i>Data → Acquisition &amp; matching…</i> (set once).</li>
<li><b>Data → Find peaks…</b> (or press <b>P</b>) to build the feature set — it fills the
    <i>Features</i> panel on the right.</li>
<li>Tune <b>Colormap / Hotspot clip / Normalization</b> in the right-hand <i>Display</i> panel —
    every change re-renders the ion image live.</li>
<li><b>Segmentation</b> tab — tick <i>Auto</i> + <i>Spatially-aware</i>, Run; group clusters into
    named <i>Regions</i> (right panel).</li>
<li>Build named <i>Regions</i> from drawn ROIs (Regions panel → "Add ROI") or segmentation
    clusters, pick two as A/B, then <b>ROI statistics</b> / <b>Region comparison</b>.</li>
<li><b>Identify lipids</b> (Features panel) to annotate every peak; export the <b>Excel report</b>.</li>
<li><b>Analyze</b> gallery — pick any analysis (markers, per-ion segmentation, comparisons,
    co-localization…) from one menu; each opens a Configure &amp; Run popup and records a
    re-openable run in <b>Analyses → History</b>. For batch/replay, <i>Data → Analysis script…</i>
    drives the same engine from Python and saves reusable workflow presets.</li>
<li><b>Export</b> (<i>File → Export…</i>, or <b>⌘E</b>) — ion images with the spectrum overlay
    composited into a corner, colour overlays, spectra, tables, or a full multi-page
    <b>PDF data book</b> (cover · methods + provenance · ion gallery · segmentation · statistics)
    in your choice of PNG / TIFF / JPEG / PDF / SVG / CSV / XLSX.</li>
</ol>
<p><b>Tips:</b> single-click a feature row to view its ion image; with the ROI tool off,
click any pixel to overlay its spectrum; "Save image…" opens the Export hub with design
options (theme, colormap, corner, scale bar, resolution).
The right-hand panel is the persistent home for features, regions, and display settings.</p>
<p><b>Overlays &amp; extras</b> live next to the views they affect: an optical/histology
backdrop via <i>File → Image setup…</i> (toggle with <i>View → Show optical image</i>);
total-class &amp; ratio composites via the <i>Composite…</i> button on the ion-image
toolbar; and the show/hide list of overlaid spectra via the <i>Traces…</i> button on the
spectrum toolbar.</p>
<p>Full manual: <b>Help → Open User Guide</b>.</p>
"""


# --------------------------------------------------------------------------- #
# Compute functions (pure-ish; run inside workers, no Qt) — kept module-level so
# they can be exercised headlessly in tests.
# --------------------------------------------------------------------------- #


class _DockTabWidget(QtWidgets.QTabWidget):
    """A QTabWidget whose tab strip is mirrored onto an external ``QTabBar`` — so the
    strip can live in a full-width toolbar *above the right-hand dock* instead of being
    penned into the narrower central column (where it overflowed into scroll arrows).

    Every existing ``addTab`` / ``setTabText`` / ``setCurrentIndex`` call keeps working
    unchanged: tab inserts/removes/renames and selection changes mirror across both ways,
    so the rest of the app never has to know the visible strip is a separate widget.
    """
    def attach_strip(self, bar: QtWidgets.QTabBar):
        self._strip = bar
        bar.blockSignals(True)
        while bar.count():
            bar.removeTab(0)
        for i in range(self.count()):
            bar.addTab(self.tabText(i))
        bar.setCurrentIndex(self.currentIndex())
        bar.blockSignals(False)
        bar.currentChanged.connect(self.setCurrentIndex)     # click strip → switch page
        self.currentChanged.connect(self._mirror_current)    # switch page → light strip
        self.tabBar().hide()                                 # the strip is the tab UI now

    def _mirror_current(self, i):
        bar = getattr(self, "_strip", None)
        if bar is not None and bar.currentIndex() != i:
            bar.blockSignals(True)
            bar.setCurrentIndex(i)
            bar.blockSignals(False)

    def tabInserted(self, index):
        super().tabInserted(index)
        bar = getattr(self, "_strip", None)                  # None while tabs are first built
        if bar is not None:
            bar.blockSignals(True)
            bar.insertTab(index, self.tabText(index))
            bar.blockSignals(False)

    def tabRemoved(self, index):
        super().tabRemoved(index)
        bar = getattr(self, "_strip", None)
        if bar is not None and index < bar.count():
            bar.blockSignals(True)
            bar.removeTab(index)
            bar.blockSignals(False)

    def setTabText(self, index, text):
        super().setTabText(index, text)
        bar = getattr(self, "_strip", None)
        if bar is not None and index < bar.count():
            bar.setTabText(index, text)


class _NoWheelScroll(QtCore.QObject):
    """Eats wheel events on a widget so it can't scroll. Used on the tab-strip viewport
    to keep the strip anchored at the left edge — the leftmost tab ('Ion image') is always
    visible and tabs only ever fall off the *right*, reached via the » overflow menu."""
    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Type.Wheel:
            return True                                   # swallow → no scrolling
        return super().eventFilter(obj, ev)


class MainWindow(AuditMixin, ScopeMixin, IonTabMixin, FeaturesTabMixin, SegmentTabMixin, MontageTabMixin, StatsTabMixin, ClassCompareMixin, LipidListMixin, LipidTreeMixin, AnalysisTabsMixin, LibraryTabMixin, RightPanelMixin, OpticalMixin, ExportMixin, StudioMixin, UMAPStudioMixin, ReportTabMixin, HistoryMixin, CohortMixin, CohortSegMixin, GalleryMixin, ScriptConsoleMixin, QtWidgets.QMainWindow):
    # Signal bus (plan 24): long-lived modeless dialogs observe state through these instead
    # of being hand-synced by _sync_region_combos / _refresh_action_states. Each fires once at
    # the very end of its funnel; nothing connects yet, so behaviour is unchanged.
    datasetChanged = QtCore.Signal()   # a new MSIDataset was loaded / cleared
    peaksChanged   = QtCore.Signal()   # the active feature list changed
    regionsChanged = QtCore.Signal()   # a region was added / removed / renamed / recoloured
    segChanged     = QtCore.Signal()   # a segmentation was applied / reset

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SMILE MSI — Spatial Mass Imaging of Lipid Environments")
        self._size_to_screen()
        self.pool = QtCore.QThreadPool.globalInstance()

        self.ds: MSIDataset | None = None
        self.peaks: list = []
        self.seg: spatial.Segmentation | None = None
        self.active_mz: float | None = None
        self._disp_axis = None           # m/z axis of the currently displayed base spectrum
        self._disp_y = None              # intensities of the currently displayed base spectrum
        self.regions: list = []          # named regions: {name, color, segments:set[int]}
        self._region_labels = ("Region A", "Region B")
        self._crop_preset = None          # project default close-up framing (aspect / size / padding)
        self.last_stats = None            # ROI two-region comparison (rich Excel report + ⌘E hub)
        self.report_items: list = []      # curated, ordered PDF-book contents (Report tab)
        self._studio_plan = None          # saved Export Studio plan (studio.StudioPlan, per-sample)
        self._script_console = None       # lazily-built Analysis script console
        self._stats_export = None         # table currently shown on the stats tab, for its Export buttons
        self._pix_lookup = None
        self._undo_stack = []            # (label, restore-callable) for ⌘Z
        self._undo_busy = False          # re-entrancy guard: one undo step per action
        self.prov = None
        self._acquisition_meta = {}            # reporting metadata for standards-compliant imzML export
        self._calibration_models = []          # absolute-quant calibration models (plan 03)
        self._cancel = False
        self._init_optical_state()       # global optical/histology backdrop layer
        self._class_map = {}
        self._active_workers = set()     # strong refs so workers die on the main thread
        from .jobqueue import JobQueue
        self.jobs = JobQueue(self)       # serialises heavy analyses (one at a time)
        self._running_job = None         # the Job backing the currently-running worker, if queued
        self._lipid_db = None            # external DB (LIPID MAPS/SwissLipids); None = built-in
        self.ann = Annotator(mode="negative", ppm_tol=5.0)
        self._flist_name = "(picked peaks)"   # which feature list drives the pipelines
        self._feature_consumer_labels = []    # pipeline labels kept in sync with it
        self._scope_bars = []                 # per-analysis "data in" bars (scope.py)
        self._feature_scopes = {}             # scope name -> peaks ("All slide" + per-sample)
        self._active_feature_scope = None     # which sample's features are active (None = all)
        self._feature_lists = {}              # name -> [{mz,lipid,note}] saved lists (per session)
        self._lipid_lists = {}                # name -> [{class,color,mzs}] lipid-class lists
        self._active_lipid_list = None        # name of the lipid list driving the class overlay
        # auto-save: this sample's whole analysis persists to a managed session file
        self._session_path = None             # managed file for the loaded sample (lazy)
        self._dirty = False                   # unsaved analysis changes pending a flush
        self._restoring = False               # True while applying a session (suppresses autosave)

        self._build_setup_widgets()      # the "set-once" controls (now homed in menus/dialogs)
        self._build_tabs()
        # The Analyze gallery is the signal bus's first consumer (plan 24): every funnel that
        # changes readiness re-evaluates the cards. _refresh_gallery no-ops until its widgets
        # exist, so connecting here (post-_build_tabs) is safe regardless of build order.
        for sig in (self.datasetChanged, self.peaksChanged, self.regionsChanged, self.segChanged):
            sig.connect(self._refresh_gallery)
        self._build_right_panel()        # persistent panel: Display · Regions · Features
                                         # (also assembles the central splitter around the tabs)
        self._build_samples_panel()      # cohort roster: collapsible section in the right column
        self._build_image_setup_dialog() # optical backdrop setup — File → Image setup… (built
                                         # eagerly + hidden so its controls exist for restore)
        self._build_menu()
        self._install_status_progress()  # progress + cancel live in the status bar now
        self._setup_shortcuts()
        self.statusBar().showMessage("Open an imzML file (File menu), or Load demo, to begin.  "
                                     "(Help → Quick start)")
        # After the window is up: re-apply a remembered lipid DB and, on the very first launch,
        # open the setup wizard. Non-blocking (singleShot) and never during tests.
        QtCore.QTimer.singleShot(0, self._run_startup_setup)

    def _size_to_screen(self):
        """Open at a sensible fraction of the available screen (never larger than it),
        centered — so the window fits laptops and external monitors alike."""
        screen = QtWidgets.QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else None
        if avail and avail.width() > 0:
            w = min(1360, int(avail.width() * 0.92))
            h = min(860, int(avail.height() * 0.92))
            self.resize(w, h)
            self.move(avail.x() + (avail.width() - w) // 2,
                      avail.y() + (avail.height() - h) // 2)
            self.setMinimumSize(min(820, w), min(560, h))
        else:                                     # headless / unknown screen
            self.resize(1100, 760)
            self.setMinimumSize(820, 560)

    def _setup_shortcuts(self):
        def sc(keys, fn):
            QtGui.QShortcut(QtGui.QKeySequence(keys), self, activated=fn)
        sc("P", self.do_pick_peaks)                                   # pick peaks
        sc("Shift+P", self.do_find_spatial_features)                  # spatial feature finder
        sc("Ctrl+D", self._add_active_to_list)                        # ⌘D: add selected peak to the live list
        sc("R", lambda: self._reset_spectrum_view())                  # reset spectrum view
        sc("+", lambda: self.spectrum.getViewBox().scaleBy(x=0.7))    # zoom in
        sc("=", lambda: self.spectrum.getViewBox().scaleBy(x=0.7))
        sc("-", lambda: self.spectrum.getViewBox().scaleBy(x=1.4))    # zoom out
        sc(QtCore.Qt.Key_Left, lambda: self._step_peak(-1))           # previous peak
        sc(QtCore.Qt.Key_Right, lambda: self._step_peak(+1))          # next peak
        sc(QtCore.Qt.Key_Escape, self._on_escape)                     # cancel ROI / deselect
        sc("Ctrl+Z", self._undo)                                      # ⌘Z: undo last reversible action
        sc("[", lambda: self._nudge_brush(-1))                        # shrink freehand brush
        sc("]", lambda: self._nudge_brush(+1))                        # grow freehand brush

    def _nudge_brush(self, step):
        """Grow/shrink the freehand brush radius (the [ / ] shortcuts)."""
        if getattr(self, "brush_size", None) is not None:
            self.brush_size.setValue(self.brush_size.value() + step)

    # ----- undo: snapshot the touched state before each destructive action - #
    #
    # ``record_undo(label, domains)`` deep-copies the named state domains and pushes
    # a restore onto the stack; ⌘Z pops the last one and re-renders. A re-entrancy
    # guard collapses a whole user action into ONE undo step even when its handler
    # fans out through several shared helpers (e.g. ROI→region builds a region AND a
    # feature list): only the first record_undo in a synchronous event survives, so
    # its ``domains`` must cover everything that action transitively touches. The
    # guard clears on the next event-loop turn (or after app.processEvents() in tests).
    _UNDO_LIMIT = 40

    def push_undo(self, label, restore):
        """Low-level: register a reversible action with an explicit restore callable."""
        stack = getattr(self, "_undo_stack", None)
        if stack is None:
            stack = self._undo_stack = []
        stack.append((label, restore))
        del stack[:-self._UNDO_LIMIT]                 # cap memory: keep the most recent N

    def record_undo(self, label, domains=("features", "regions")):
        """Snapshot ``domains`` (any of features/regions/seg/roi) so ⌘Z can roll the
        action back. Call it right before the mutation, after any early-return guards
        and modal dialogs, so a cancelled/no-op action leaves no undo entry."""
        # every reversible action changes the saved analysis → schedule an auto-save.
        # Done before the re-entrancy guard so a multi-step action still marks dirty once.
        self._mark_dirty()
        if getattr(self, "_undo_busy", False):        # nested call within one action
            return
        snap = self._capture_state(domains)
        self.push_undo(label, lambda: self._restore_state(snap))
        self._undo_busy = True
        QtCore.QTimer.singleShot(0, self._end_undo_action)

    def _end_undo_action(self):
        self._undo_busy = False

    def _undo(self):
        stack = getattr(self, "_undo_stack", None)
        if not stack:
            self.statusBar().showMessage("Nothing to undo.")
            return
        label, restore = stack.pop()
        try:
            restore()
        except Exception as e:                         # a stale snapshot shouldn't wedge the app
            self.statusBar().showMessage(f"Couldn't undo {label}: {e}")
            return
        self.statusBar().showMessage(f"Undid {label}.")

    # --- snapshot / restore helpers --- #
    def _copy_regions(self, regions):
        """Deep-copy the region list (each dict owns a mutable segments set + mask)."""
        out = []
        for rg in regions:
            c = dict(rg)
            if c.get("segments") is not None:
                c["segments"] = set(c["segments"])
            m = c.get("mask")
            if m is not None:
                c["mask"] = np.asarray(m).copy()
            out.append(c)
        return out

    def _capture_state(self, domains):
        snap = {"domains": tuple(domains)}
        if "features" in domains:
            snap["peaks"] = [dict(p) for p in self.peaks]
            snap["scopes"] = {k: [dict(p) for p in v]
                              for k, v in (getattr(self, "_feature_scopes", {}) or {}).items()}
            snap["scope_active"] = getattr(self, "_active_feature_scope", None)
            snap["flist"] = getattr(self, "_flist_name", None)
            snap["lipid_active"] = getattr(self, "_active_lipid_list", None)
            snap["active_mz"] = self.active_mz
        if "regions" in domains:
            snap["regions"] = self._copy_regions(self.regions)
        if "seg" in domains:
            snap["seg"] = copy.deepcopy(getattr(self, "seg", None))
            snap["seg_lineage"] = copy.deepcopy(getattr(self, "_seg_lineage", None))
            snap["seg_hidden"] = set(getattr(self, "_seg_hidden", set()) or set())
            snap["seg_original"] = copy.deepcopy(getattr(self, "_seg_original", None))
        if "roi" in domains:
            bm = getattr(self, "_brush_mask", None)
            snap["brush"] = None if bm is None else bm.copy()
        if "optical" in domains:
            opt = getattr(self, "_optical", None)
            snap["optical_img"] = None if opt is None else np.asarray(opt).copy()
            snap["optical_path"] = getattr(self, "_optical_path", None)
            snap["optical_on"] = bool(getattr(self, "_optical_on", False))
            snap["optical_align"] = dict(getattr(self, "_optical_align", {}) or {})
            snap["optical_alpha"] = float(getattr(self, "_optical_spectra_alpha", 0.5))
        return snap

    def _restore_state(self, snap):
        domains = snap["domains"]
        if "features" in domains:
            self._feature_scopes = {k: [dict(p) for p in v] for k, v in snap["scopes"].items()}
            self._active_feature_scope = snap["scope_active"]
            self._flist_name = snap["flist"]
            self._active_lipid_list = snap.get("lipid_active")
            self.active_mz = snap["active_mz"]
            self.peaks = [dict(p) for p in snap["peaks"]]
        if "regions" in domains:
            self.regions = self._copy_regions(snap["regions"])
        if "seg" in domains:
            self.seg = snap["seg"]
            self._seg_lineage = snap["seg_lineage"]
            self._seg_hidden = snap["seg_hidden"]
            self._seg_original = snap["seg_original"]
        if "roi" in domains:
            self._brush_mask = snap["brush"]
        if "optical" in domains:
            self._optical_align = dict(snap["optical_align"]) or self._default_optical_align()
            self._optical_spectra_alpha = snap["optical_alpha"]
            self._optical_on = snap["optical_on"]
            # re-push the snapshotted pixels straight into the views (no disk reload needed)
            self._set_optical_image(snap["optical_img"], snap["optical_path"])
        self._rerender_undo(domains)

    def _rerender_undo(self, domains):
        """Re-render whatever the restored domains affect. Every call is guarded so a
        view that isn't built yet (or a None dataset) can't wedge the undo."""
        safe = common.guarded            # best-effort refresh; a dead view must not wedge undo
        if "features" in domains:
            safe(self._set_peaks, self.peaks)          # rebuilds peak table/combos/marks + IDs
            safe(self._refresh_feature_set_combo)
            if self.active_mz is not None:
                safe(self.set_active_mz, self.active_mz)
        if "seg" in domains or "regions" in domains:
            if getattr(self, "seg", None) is not None:
                safe(self._render_seg_base)
                safe(self._rebuild_seg_table)
            safe(self._refresh_region_list)
            safe(self._sync_region_combos)
            safe(self._render_region_map)
            safe(self._refresh_seg_assignments)
        if "roi" in domains:
            safe(self._render_brush_overlay)
            safe(self._roi_spectrum)
        if "optical" in domains:
            safe(self._sync_optical_controls)
            safe(self._refresh_optical)
        safe(self.refresh_ion_image)
        self._refresh_action_states()

    def _refresh_action_states(self):
        """Re-evaluate every tab's Run/Compare/Render button enabled-state after the
        dataset, peaks, regions or segmentation change (A5: never click into an error)."""
        for name in ("_update_cohort_run_enabled",
                     "_refresh_seg_run_state",
                     "_update_build_region_enabled", "_refresh_gallery"):
            fn = getattr(self, name, None)
            if fn is not None:
                common.guarded(fn)       # a tab not built yet must not block the others' sweep

    def _build_menu(self):
        filem = self.menuBar().addMenu("&File")
        a_setup = filem.addAction("Setup wizard…", self._open_setup_wizard)
        a_setup.setIcon(common.icon("settings"))
        a_setup.setToolTip("Guided setup: ionization mode, mass tolerance, and the lipid "
                           "database (built-in or LIPID MAPS). Re-runnable any time.")
        filem.addSeparator()
        a_open = filem.addAction("Open imzML…", self.open_imzml)
        a_open.setIcon(common.icon("open"))
        a_open.setShortcut(QtGui.QKeySequence.Open)          # ⌘O / Ctrl+O
        a_table = filem.addAction("Open table (CSV/TSV)…", self.open_table)
        a_table.setIcon(common.icon("open"))
        a_table.setToolTip("Imaging table: long (x,y,mz,intensity) or wide (x,y then m/z columns)")
        a_demo = filem.addAction("Load demo dataset", self.do_load_demo)
        a_demo.setIcon(common.icon("open"))
        # subsample stride is a load-time control → tuck it into the File menu
        sub = QtWidgets.QWidget()
        sh = QtWidgets.QHBoxLayout(sub)
        sh.setContentsMargins(22, 2, 8, 2)
        sh.addWidget(QtWidgets.QLabel("Quick-load every Nth px"))
        sh.addWidget(self.subsample_spin)
        wa_sub = QtWidgets.QWidgetAction(filem)
        wa_sub.setDefaultWidget(sub)
        filem.addAction(wa_sub)
        filem.addSeparator()
        a_imgset = filem.addAction("Image setup…", self._open_image_setup_dialog)
        a_imgset.setIcon(common.icon("settings"))
        a_imgset.setToolTip("Load and align an optical / histology photo of the slide as a "
                            "backdrop behind the ion image, colour overlay, and segmentation")
        filem.addSeparator()
        a_exp = filem.addAction("Export…", lambda: self.open_export_hub())
        a_exp.setIcon(common.icon("export"))
        a_exp.setShortcut(QtGui.QKeySequence("Ctrl+E"))      # ⌘E / Ctrl+E
        a_exp.setToolTip("Export images, spectra, tables, or a full PDF data book — "
                         "multiple formats with design options")
        a_studio = filem.addAction("Export Studio…", lambda: self.open_export_studio())
        a_studio.setIcon(common.icon("export"))
        a_studio.setShortcut(QtGui.QKeySequence("Ctrl+Shift+E"))   # ⇧⌘E / Ctrl+Shift+E
        a_studio.setToolTip("Batch-export ion images across multiple feature lists and "
                            "sections at once — plus their feature-list CSVs and analyses")
        a_umap = filem.addAction("UMAP Studio…", lambda: self.open_umap_studio())
        a_umap.setIcon(common.icon("analysis"))
        a_umap.setShortcut(QtGui.QKeySequence("Ctrl+Shift+U"))     # ⇧⌘U / Ctrl+Shift+U
        a_umap.setToolTip("Turn a Cohort UMAP into a publication-quality, density-shaded, "
                          "annotated figure (multi-panel, colour by donor / group / region / lipid)")
        a_book = filem.addAction("Export data book (PDF)…", lambda: self.open_export_hub("book"))
        a_book.setIcon(common.icon("export"))
        a_book.setToolTip("One multi-page PDF: cover, methods + provenance, ion images, "
                          "spectra, segmentation, and statistics")
        a_rep = filem.addAction("Analyses…", self._show_report_tab)
        a_rep.setToolTip("Curate the analyses (ion images, overlays, spectra, comparisons…) "
                         "that go into the data book")
        a_methods = filem.addAction("Export methods report…", self.export_methods)
        a_methods.setIcon(common.icon("export"))
        a_montage = filem.addAction("Ion montage grid…", self._open_montage_dialog)
        a_montage.setIcon(common.icon("analysis"))
        a_montage.setToolTip("A grid of ion images ranked by intensity or Moran's I — render "
                             "on screen, then export as a publication figure")
        filem.addSeparator()
        a_export = filem.addAction("Export to imzML…", self.export_imzml)
        filem.addAction("Acquisition metadata && reporting…", self._open_reporting_dialog)
        a_export.setIcon(common.icon("export"))
        a_export.setToolTip("Convert the loaded dataset to the open imzML standard")
        filem.addSeparator()
        a_save = filem.addAction("Export analysis to share…", self.save_session_dialog)
        a_save.setIcon(common.icon("export"))
        a_save.setToolTip("Save your regions + analyses to one file a colleague can open on "
                          "top of their own copy of this dataset")
        a_opensess = filem.addAction("Open analysis…", self.open_session_dialog)
        a_opensess.setIcon(common.icon("open"))
        a_opensess.setToolTip("Open a saved or shared analysis; you'll be asked to locate the "
                              "dataset if it isn't on this machine")

        # everything that's set once at startup (or is a one-shot workflow) lives here
        datam = self.menuBar().addMenu("&Data")
        a_pick = datam.addAction("Find peaks…", self._open_pick_dialog)
        a_pick.setIcon(common.icon("find"))
        a_pick.setToolTip("Detect peaks → build the working feature set (P)")
        a_spatial = datam.addAction("Find spatial features…",
                                    lambda: self._open_pick_dialog(focus_spatial=True))
        a_spatial.setIcon(common.icon("find"))
        a_spatial.setToolTip("Spatial feature finder: mean candidates → reproducibility "
                             "frequency → Moran's I denoise → auto-width (⇧P)")
        datam.addSeparator()
        a_script = datam.addAction("Analysis script…", self._open_script_console)
        a_script.setShortcut(QtGui.QKeySequence("Ctrl+Shift+J"))
        a_script.setToolTip("Drive the analysis engine with Python (find_peaks, segment, "
                            "compare…), run it on this slide, and save it as a workflow. "
                            "Includes a copy-paste AI scripting guide. (⇧⌘J)")
        datam.addSeparator()
        datam.addAction("Acquisition && matching…", self._open_acquisition_dialog).setIcon(common.icon("settings"))
        datam.addAction("Calibration check…", self._open_calibration_dialog).setIcon(common.icon("settings"))
        datam.addAction("Preprocessing…", self._open_preprocess_dialog).setIcon(common.icon("settings"))
        a_prof = datam.addAction("Analysis profiles…", self._open_profiles_dialog)
        datam.addAction("Quantification (calibration)…", self._open_quantify_dialog)
        datam.addAction("Single-cell profiling…", self._open_single_cell_dialog)
        datam.addAction("3D reconstruction…", self._open_stack3d_dialog)
        datam.addAction("Spatial multi-omics…", self._open_comap_dialog)
        a_prof.setIcon(common.icon("settings"))
        a_prof.setToolTip("Named, versioned, shareable defaults for processing settings that "
                          "should stay constant across samples and slides — the active profile "
                          "pre-fills new samples; deviations are recorded.")
        datam.addAction("Import lipid database…", self._import_lipid_db).setIcon(common.icon("import"))
        a_cache = datam.addAction("Build fast ion cache", self.build_ion_cache)
        a_cache.setToolTip("One streaming pass → instant ion images for any m/z (best for large files)")

        viewm = self.menuBar().addMenu("&View")
        a_panel = viewm.addAction("Show/hide right panel")
        a_panel.setShortcut("Ctrl+\\")
        a_panel.setToolTip("Collapse the Display · Regions · Features panel to the edge, or "
                           "bring it back (you can also drag the divider)")
        a_panel.triggered.connect(self._toggle_right_panel)
        a_reset = viewm.addAction("Reset panel layout")
        a_reset.setIcon(common.icon("refresh"))
        a_reset.setToolTip("Re-open the right-hand panel at its default width if it was dragged shut")
        a_reset.triggered.connect(self._reset_panel_layout)
        viewm.addSeparator()
        self._build_appearance_menu(viewm)
        viewm.addSeparator()
        # Quick show/hide for the optical backdrop — the frequent toggle, separate from the
        # once-per-sample load/align setup (File → Image setup…). Kept in step with the
        # dialog's own checkbox via _sync_optical_show_controls.
        self.optical_show_action = viewm.addAction("Show optical image")
        self.optical_show_action.setCheckable(True)
        self.optical_show_action.setEnabled(False)
        self.optical_show_action.setToolTip("Toggle the optical backdrop in every spatial view "
                                            "(load and align it via File → Image setup…)")
        self.optical_show_action.toggled.connect(self._set_optical_on)

        helpm = self.menuBar().addMenu("&Help")
        qs = helpm.addAction("Quick start")
        qs.setIcon(common.icon("help"))
        qs.triggered.connect(self._show_quickstart)
        ug = helpm.addAction("Open User Guide")
        ug.setIcon(common.icon("help"))
        ug.triggered.connect(self._open_user_guide)
        tl = helpm.addAction("Show timeline…")
        tl.setToolTip("See where startup and each load / preprocessing pass spent time "
                      "(the same events written to smile_msi_perf.log)")
        tl.triggered.connect(self._show_timeline)
        helpm.addSeparator()
        upd = helpm.addAction("Check for updates…")
        upd.setIcon(common.icon("refresh"))
        upd.setToolTip("See if newer code is available on GitHub — every push for a source "
                       "install, or the latest release for a frozen bundle")
        upd.triggered.connect(lambda: self._check_for_updates())
        from . import updater
        if updater.is_git_checkout():
            # Source install: offer a real in-place update (git pull + reinstall + restart).
            gp = helpm.addAction("Update from GitHub (git pull)…")
            gp.setIcon(common.icon("refresh"))
            gp.setToolTip("Pull the latest code into this checkout, reinstall if "
                          "dependencies changed, and restart")
            gp.triggered.connect(self._update_from_git)
        tok = helpm.addAction("GitHub token…")
        tok.setIcon(common.icon("settings"))
        tok.setToolTip("Save a GitHub token — used both to file feature requests and to "
                       "check for updates on the private repo")
        tok.triggered.connect(self._set_github_token)
        helpm.addSeparator()
        fb = helpm.addAction("Send feedback / Feature request…")
        fb.setToolTip("File a feature request or bug report as a GitHub issue")
        fb.triggered.connect(self._open_feedback)
        mr = helpm.addAction("My requests…")
        mr.setToolTip("See the status of feature requests / bugs you've submitted")
        mr.triggered.connect(self._open_my_requests)
        about = helpm.addAction("About")
        about.setIcon(common.icon("help"))
        about.triggered.connect(lambda: QtWidgets.QMessageBox.about(
            self, "About SMILE MSI",
            "SMILE MSI — Spatial Mass Imaging of Lipid Environments.\n"
            "Open-source MALDI-MSI analysis.\n\n"
            "Ion imaging, segmentation, co-localization, exact ROI statistics, and\n"
            "in-silico lipid identification. Runs locally; Apache-2.0 licensed."))

    # ----- session save / load -------------------------------------------- #
    def _current_settings(self):
        s = {"mode": self.mode_combo.currentText(), "ppm": self.ppm, "id_ppm": self.id_ppm,
             "norm": self.norm, "reduce": self.reduce, "composite_weight": self.composite_weight,
             "cmap": self.cmap_combo.currentText(),
             "contrast": float(self.contrast_spin.value()), "snr": float(self.snr_spin.value()),
             "minrel": float(self.minrel_spin.value()),
             "prominence": float(self.prominence_spin.value()),
             "orientation": int(getattr(self.ds, "orientation", 0)) if self.ds is not None else 0}
        # stamp the active profile (name+version+hash) and where this sample deviates from
        # it, so the data trail records the method and any per-sample overrides (soft lock).
        try:
            act = profiles.active()
            s["profile"] = {"name": act.get("name"), "version": int(act.get("version", 1)),
                            "hash": profiles.content_hash(act)}
            s["profile_deviations"] = profiles.diff(self._profile_widget_params(), act)
        except Exception:  # noqa: BLE001 — provenance stamping must never break a save
            pass
        return s

    def _session_state(self) -> dict:
        """The current sample's whole analysis as a session dict — the single builder
        used by both auto-save and the explicit Save session… export."""
        return session.build_session(
            source=self.ds.source, settings=self._current_settings(), peaks=self.peaks,
            active_mz=self.active_mz, labels=(self.seg.labels if self.seg else None),
            n_clusters=(self.seg.n_clusters if self.seg else None),
            named_regions=self.regions, n_pixels=self.ds.n_pixels,
            dataset_fingerprint=library.dataset_fingerprint(self.ds),
            feature_lists=self._feature_lists, feature_scopes=self._feature_scopes,
            lipid_lists=self._lipid_lists,
            active_feature_scope=self._active_feature_scope,
            flist_name=getattr(self, "_flist_name", None),
            optical=self._optical_session_dict(),
            cube_fingerprint=(library.dataset_fingerprint(self.ds)
                              if getattr(self.ds, "_cube", None) is not None else None),
            crop_preset=getattr(self, "_crop_preset", None),
            report_items=getattr(self, "report_items", None),
            studio_plan=getattr(self, "_studio_plan", None),
            provenance=getattr(self, "prov", None),
            acquisition_meta=getattr(self, "_acquisition_meta", None),
            calibration_models=getattr(self, "_calibration_models", None),
            analysis_runs=self._analysis_runs_index())

    def _analysis_runs_index(self):
        """The lightweight ``analysis_runs`` index for the session embed — the run *metadata*
        (never the payloads, which live under ``<session>.runs/``). Best-effort: a store read
        must never break a save."""
        store = self.run_store
        if store is None:
            return None
        try:
            return store.index()
        except Exception:  # noqa: BLE001 — index read is best-effort
            return None

    @staticmethod
    def _restore_studio_plan(stored):
        """Decode a persisted Export Studio plan (``studio.StudioPlan.to_dict()``) back into a
        plan; best-effort so a stale/partial record from an older build never blocks restore."""
        if not stored:
            return None
        try:
            from .. import studio
            return studio.StudioPlan.from_dict(dict(stored))
        except Exception:  # noqa: BLE001 — a malformed plan must never block session restore
            return None

    # ----- auto-save: the per-sample session is the source of truth -------- #
    def _mark_dirty(self):
        """Flag that the loaded sample's analysis changed and schedule a debounced
        auto-save. No-ops with no dataset, on the throwaway demo, or while restoring a
        session (so we never rewrite a file we just read). GUI-thread only — never call
        from a worker ``fn`` (only from on_done callbacks / Qt slots / record_undo)."""
        if self.ds is None or self.ds.source == "synthetic" or getattr(self, "_restoring", False):
            return
        self._dirty = True
        QtCore.QTimer.singleShot(1000, self._flush_autosave)

    def _flush_autosave(self):
        """Write the managed session file if anything is pending. Best-effort: an
        auto-save must never crash the app or interrupt the user."""
        if not self._dirty or self.ds is None or self.ds.source == "synthetic":
            return
        try:
            if self._session_path is None:
                # resolve_session_path adopts an existing (possibly differently-keyed) session
                # for this slide instead of minting a fresh blank one, so a first edit can't
                # start a new empty file alongside the one that holds the ROIs.
                self._session_path = session.resolve_session_path(
                    self.ds.source, library.dataset_fingerprint(self.ds), self.ds.n_pixels)
            state = self._session_state()
            # Never let an empty in-memory region set silently overwrite a populated ROI set on
            # disk. That only happens when this sample's session was NOT restored this load (a
            # fresh open that skipped restore, a restore that raised, or an overlay onto the
            # wrong slide) — in which case self.regions is untrustworthy, so preserve what's on
            # disk. A restored sample with no regions is a real user clear and saves normally.
            if not state.get("named_regions") and not getattr(self, "_session_restored", False):
                on_disk = session.existing_named_regions(self._session_path)
                if on_disk:
                    state["named_regions"] = on_disk
            session.save_session(self._session_path, state)
            self._dirty = False
            self._save_cube_sidecar()            # cache the fast cube next to the session
            self._register_active_in_cohort()    # a sample you work on joins the roster
        except Exception:  # noqa: BLE001 — autosave is best-effort
            import traceback
            traceback.print_exc()

    def _save_cube_sidecar(self):
        """Persist the fast m/z cube (+ prime stats) beside the managed session once per
        dataset, so reopening this sample loads them instead of re-streaming the whole
        acquisition to rebuild. The cube is derived from the raw data + fixed bin params,
        so it never needs rewriting — gated on a per-dataset ``_cube_saved`` flag."""
        ds = self.ds
        if (ds is None or self._session_path is None
                or getattr(ds, "_cube", None) is None
                or getattr(self, "_cube_saved", False)):
            return
        self._cube_saved = True              # claim now so back-to-back autosaves don't double-write
        # snapshot immutable references and write off the GUI thread — the npz can be tens
        # of MB and the cube never changes once built, so a background write can't race.
        path, cube, mean, pix = self._session_path, ds._cube, ds._mean, ds._pix
        fp = library.dataset_fingerprint(ds)

        def write():
            try:
                session.save_cube(path, cube, mean=mean, pix=pix, fingerprint=fp)
            except Exception:  # noqa: BLE001 — caching is best-effort, never fatal
                import traceback
                traceback.print_exc()

        import threading
        threading.Thread(target=write, daemon=True, name="cube-sidecar").start()

    def _restore_cache_from_sidecar(self, ds, session_path) -> bool:
        """Load the fast cube (+ prime stats) saved beside a session, skipping the rebuild.
        Returns True only when a sidecar exists and matches this slide (fingerprint +
        pixel count); the cube/mean/stats are then assigned onto ``ds`` directly so the
        reopen touches the .ibd only for the shared m/z axis, not a full re-stream."""
        try:
            got = session.load_cube(session_path, library.dataset_fingerprint(ds),
                                    ds.n_pixels)
        except Exception:  # noqa: BLE001
            got = None
        if not got:
            return False
        ds._cube = got["cube"]
        if got.get("mean") is not None and got.get("pix") is not None:
            ds._mean, ds._pix = got["mean"], got["pix"]
        self._cube_saved = True              # came from disk — don't immediately rewrite it
        return True

    def _prepare_imzml(self, src, progress=None, stride=1, session_path=None, stage=None):
        """Load an imzML dataset ready to use, off the GUI thread under the load bar:
        restore the fast cube from a session sidecar when one matches (no disk re-stream),
        else read into RAM (budget-guarded :meth:`MSIDataset.to_ram`), prime, and build the
        cube — so prime/feature passes are vectorized on a large-RAM machine."""
        import os
        ibd = os.path.splitext(src)[0] + ".ibd"
        if not os.path.exists(ibd):
            raise ValueError(
                "This imzML is missing its companion .ibd data file. "
                f"'{os.path.basename(ibd)}' must sit in the same folder as the .imzML "
                "(imzML stores only the metadata; the .ibd holds the spectra).")
        ds = MSIDataset.from_imzml(src, lazy=True, stride=stride)
        if session_path and self._restore_cache_from_sidecar(ds, session_path):
            if stage:
                stage("Restoring fast cache")
            ds.prime(progress=progress)            # no-op when the sidecar carried prime stats
            return ds
        if stage:
            stage("Reading spectra into memory")
        ds.to_ram(progress=progress)               # one disk pass into RAM if it fits the budget
        if stage:
            stage("Priming spectra")
        ds.prime(progress=progress)                # vectorized when in RAM
        # Do NOT build the m/z cube on the modal load bar — it's the slowest pass (and the
        # biggest memory spike). In-RAM stores serve arbitrary-m/z ion images straight from
        # the dense matrix (ion_vector → _dense_ion), so they need no cube at all; a
        # disk-backed (over-budget) store gets its cube warmed in the BACKGROUND by
        # _autobuild_cache once _on_dataset has painted the TIC. First paint then waits only
        # for parse + to_ram + prime, not the cube.
        return ds

    def closeEvent(self, event):
        """Flush a pending auto-save synchronously on exit — the event loop stops after
        this, so the debounce timer would otherwise never fire."""
        try:
            self._flush_autosave()
        except Exception:  # noqa: BLE001
            pass
        super().closeEvent(event)

    def save_session_dialog(self):
        """Explicit export of this sample's analysis (regions, segmentation, peaks, feature
        lists, settings, provenance) to a chosen file — for sharing outside the managed
        ~/.smile-msi/sessions/ store, which auto-saves continuously. A colleague opens it via
        File ▸ Open analysis… on top of their own copy of the same dataset."""
        if self.ds is None:
            self.statusBar().showMessage("Load a dataset first.")
            return
        import os
        stem = (os.path.splitext(os.path.basename(self.ds.source))[0]
                if self.ds.source and self.ds.source != "synthetic" else "analysis")
        path, _ = filedialogs.get_save_file_name(self, "Export analysis",
                                                 f"{stem}-analysis.json", "Analysis (*.json)")
        if not path:
            return
        session.save_session(path, self._session_state())
        self.statusBar().showMessage(f"Exported analysis: {path}")

    def open_session_dialog(self):
        path, _ = filedialogs.get_open_file_name(self, "Open session", "", "Session (*.json)")
        if path:
            self._open_session_path(path)

    # File-name → this-machine path for datasets whose recorded source path is foreign
    # (e.g. an analysis shared from another computer); persisted across runs in prefs.
    _RELOC_PREF_KEY = "dataset_relocations"

    def _relocated_source(self, src: str) -> str:
        """A remembered local path for a session's imzML ``src`` whose recorded path is
        absent here, matched by file name; '' if none is known or it no longer exists."""
        import os
        from .. import prefs
        base = os.path.basename(str(src))
        mapping = prefs.get(self._RELOC_PREF_KEY, {}) or {}
        cand = mapping.get(base, "")
        return cand if cand and os.path.exists(cand) else ""

    def _prompt_locate_dataset(self, src: str) -> str:
        """Ask the user to point at this machine's copy of a shared analysis's dataset when
        the recorded path isn't here, and remember it by file name so future opens resolve
        automatically. Returns the chosen path, or '' if cancelled."""
        import os
        base = os.path.basename(str(src)) or "dataset.imzML"
        QtWidgets.QMessageBox.information(
            self, "Locate dataset",
            f"This analysis was saved against:\n\n{src}\n\n"
            f"That file isn't on this computer. Choose your copy of “{base}” to "
            "open the analysis on top of it.")
        path, _ = filedialogs.get_open_file_name(
            self, f"Locate {base}", base, "imzML (*.imzML *.imzml)")
        if not path:
            return ""
        from .. import prefs
        mapping = dict(prefs.get(self._RELOC_PREF_KEY, {}) or {})
        mapping[base] = path
        prefs.set(self._RELOC_PREF_KEY, mapping)
        return path

    def _open_session_path(self, path):
        """Load a session/analysis file (managed or shared) and apply it onto its dataset —
        reloading from the recorded source when present, overlaying onto an already-loaded
        slide, or (for an analysis shared from another machine) asking the user to locate
        their own copy of the dataset and remembering it for next time."""
        import os
        try:
            self._pending_session = session.load_session(path)
        except Exception as e:
            self._on_error(
                "Couldn't open this analysis file — it may be corrupt or not a SMILE MSI "
                f"session:\n{os.path.basename(path)}", details=str(e))
            return
        src = self._pending_session.get("source", "")
        if src == "synthetic":
            self._run(load_demo, on_done=self._apply_session, want_progress=True,
                      modal=True, busy="Loading demo + analysis…")
            return
        is_imzml = src.lower().endswith(".imzml")
        if is_imzml and os.path.exists(src):
            resolved = src                        # same machine: the recorded path is here
        elif self.ds is not None and not session.fingerprint_mismatch(
                self._pending_session.get("dataset_fingerprint"),
                library.dataset_fingerprint(self.ds),
                self._pending_session.get("n_pixels"), self.ds.n_pixels):
            self._apply_session(self.ds)          # the SAME slide is already loaded → overlay
            return
        elif is_imzml:
            # Shared from another machine — or a DIFFERENT slide is currently loaded: the
            # recorded path is absent here. Reuse a remembered relocation, else ask the user to
            # point at their copy. Never silently overlay this analysis onto a different loaded
            # slide — that would write one slide's ROIs into another slide's session file.
            resolved = self._relocated_source(src) or self._prompt_locate_dataset(src)
        else:
            resolved = ""
        if not resolved:
            self.statusBar().showMessage(
                "Dataset not found — load it first, then open the analysis.")
            return

        def load(progress=None, stage=None):
            # reuse the cached cube next to the session when it matches (no re-stream);
            # otherwise load into RAM, prime, and build the cube up front.
            return self._prepare_imzml(resolved, progress=progress, session_path=path, stage=stage)
        self._run(load, on_done=self._apply_session, want_progress=True, modal=True,
                  want_stage=True,
                  stages=["Reading spectra into memory", "Priming spectra"],
                  busy=f"Loading {os.path.basename(resolved)} + analysis + fast cache…")

    def _apply_session(self, ds):
        # _on_dataset resets feature lists/scopes/regions/peaks → must run BEFORE we
        # restore them. Everything after is guarded by _restoring so the restore itself
        # doesn't trip the auto-save (we'd be rewriting the file we just read).
        self._on_dataset(ds)
        data = self._pending_session
        self._restoring = True
        try:
            # Warn if this slide isn't the one the session was captured on: a fingerprint
            # mismatch (preferred) or, for legacy sessions, a pixel-count mismatch. The
            # restored peaks still apply; segmentation/region indices may not line up.
            if session.fingerprint_mismatch(data.get("dataset_fingerprint"),
                                            library.dataset_fingerprint(ds),
                                            data.get("n_pixels"), ds.n_pixels):
                self.statusBar().showMessage(
                    "⚠ Session opened on a different slide than it was saved on — "
                    "segmentation/regions may not align.")
            s = data.get("settings", {})
            self.mode_combo.setCurrentText(s.get("mode", "negative"))
            self.ppm_spin.setValue(float(s.get("ppm", 10.0)))
            self.id_ppm_spin.setValue(float(s.get("id_ppm", 5.0)))
            self.norm_combo.setCurrentText(s.get("norm", "tic"))
            self.reduce_combo.setCurrentText(s.get("reduce", "sum"))
            if hasattr(self, "composite_combo"):
                self.composite_combo.setCurrentText(
                    "balanced" if s.get("composite_weight") == "balanced" else "raw sum")
            self.cmap_combo.setCurrentText(s.get("cmap", "viridis"))
            self.contrast_spin.setValue(float(s.get("contrast", 99.0)))
            self.snr_spin.setValue(float(s.get("snr", 3.0)))
            self.minrel_spin.setValue(float(s.get("minrel", 0.002)))
            self.prominence_spin.setValue(float(s.get("prominence", 1.0)))
            if self.ds is not None:                       # restore saved rotation before views render
                self.ds.set_orientation(int(s.get("orientation", 0)))

            # saved feature lists (★) — restore into this sample's in-memory store
            self._feature_lists = {str(n): self._norm_feature_list(feats)
                                   for n, feats in (data.get("feature_lists") or {}).items()}
            # lipid lists (◆) — class collections with their colours
            self._lipid_lists = self._norm_lipid_lists(data.get("lipid_lists"))
            self._active_lipid_list = None

            # working scopes + active peaks. Set the scope dict directly rather than via
            # _on_peaks, which would collapse every scope into a single "All slide" entry.
            scopes = {str(n): [dict(p) for p in plist]
                      for n, plist in (data.get("feature_scopes") or {}).items()}
            peaks = data.get("peaks", [])
            if peaks:
                ds.ensure_features([p["mz"] for p in peaks], tol_ppm=self.ppm, reduce=self.reduce)
            if scopes:
                self._feature_scopes = scopes
                self._active_feature_scope = data.get("active_feature_scope")
                # share the scope's list object (not a copy) so in-place edits persist
                # into the scope; fall back to a fresh list only when no scope is active
                self.peaks = self._feature_scopes.get(self._active_feature_scope)
                if self.peaks is None:
                    self.peaks = list(peaks)
                self._flist_name = data.get("flist_name") or "All slide"
                # refresh the peak-derived views without re-picking (mirrors _on_peaks)
                self._populate_peak_table()
                self._populate_peak_combos()
                self._build_class_map()
                self._mark_peaks()
                self._refresh_feature_set_combo()
            elif peaks:                                     # legacy session without scopes
                self._on_peaks(peaks)

            seg = data.get("segmentation")
            if seg and peaks:
                labels = np.asarray(seg["labels"], dtype=int)
                if labels.size != ds.n_pixels:             # saved on a different-sized dataset
                    self.statusBar().showMessage(
                        "Session segmentation skipped — it was saved on a dataset with a different "
                        f"pixel count ({labels.size} vs {ds.n_pixels}).")
                else:
                    self.seg = spatial.Segmentation(
                        labels=labels, label_image=ds.to_image(labels),
                        peaks=[p["mz"] for p in peaks], n_clusters=int(seg["n_clusters"]),
                        explained_variance=float("nan"), silhouette=float("nan"))
                    self._on_seg(self.seg)                  # note: this resets self.regions
            named = data.get("named_regions") or []
            # restore after _on_seg cleared regions. Regions in a slide's session belong to
            # that slide, so an old session with no 'sample' defaults to this slide's source.
            self.regions = [{"name": r["name"], "color": r["color"],
                             "segments": set(int(s) for s in r.get("segments", [])),
                             "mask": session.mask_from_indices(r.get("mask"), ds.n_pixels),
                             "parent": r.get("parent"), "visible": r.get("visible", True),
                             "group": r.get("group", "") or "",
                             "sample": (r.get("sample") or ds.source or ""),
                             "crop": (tuple(r["crop"]) if r.get("crop") else None),
                             "crop_orient": r.get("crop_orient")}
                            for r in named]
            # this slide's saved regions are now in memory (even if the saved list was empty),
            # so a subsequent empty self.regions is a genuine user clear, not an un-restored
            # blank — release the auto-save's blank-ROI guard (see _flush_autosave).
            self._session_restored = True
            # refresh unconditionally — switching to a slide whose session has NO regions must
            # still rebuild the (now empty) list/combos/map, else the previous slide's regions
            # keep showing on this tissue (the cross-sample region leak).
            self._refresh_region_list()
            self._sync_region_combos()
            self._render_region_map()
            self._crop_preset = data.get("crop_preset")     # project close-up framing standard
            # restore the audit trail / methods record (_on_dataset made a fresh one with just
            # the dataset + inputs; the saved record carries every analysis step too)
            if data.get("provenance"):
                try:
                    self.prov = provenance.Provenance.from_dict(data["provenance"])
                except Exception:  # noqa: BLE001 — a malformed record must never block restore
                    pass
            self._acquisition_meta = dict(data.get("acquisition_meta") or {})
            self._calibration_models = list(data.get("calibration_models") or [])
            self.report_items = [dict(it) for it in (data.get("report_items") or [])]
            self._refresh_report_list()
            self._studio_plan = self._restore_studio_plan(data.get("studio_plan"))
            # regions are now authoritative: drop feature scopes whose region is gone
            # (stale 'ROI 1/2/3' relics; the active scope + 'All slide' are always kept)
            # and rebuild the selector so it reflects only live scopes.
            self._reconcile_feature_scopes()
            self._refresh_feature_set_combo()
            # switching to a cohort 'region sample' asked us to land on a specific region's
            # scope once its session finished applying (set in cohortview._switch_to_sample)
            pending = getattr(self, "_pending_region_sample", None)
            self._pending_region_sample = None
            if pending:
                self._activate_region_sample(pending)
            self._apply_optical_session(data)        # optical backdrop (path + alignment)
            if data.get("active_mz"):
                self.set_active_mz(data["active_mz"])
            self.statusBar().showMessage("Session restored.")
            self._refresh_action_states()
        finally:
            self._restoring = False
            self._register_active_in_cohort()    # reflect the now-active sample in the roster

    def _show_quickstart(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("SMILE MSI — Quick start")
        dlg.resize(560, 480)
        lay = QtWidgets.QVBoxLayout(dlg)
        view = QtWidgets.QTextBrowser()
        view.setHtml(HELP_HTML)
        view.setOpenExternalLinks(True)
        lay.addWidget(view)
        dlg.show()

    def _open_user_guide(self):
        import os
        guide = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "USER_GUIDE.md")
        if os.path.exists(guide):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(guide))
        else:
            self._show_quickstart()

    def _build_appearance_menu(self, viewm):
        """View → Appearance: Light / Dark, persisted in prefs and applied live."""
        from .. import prefs
        current = prefs.get("theme", DEFAULT_THEME)
        if current not in THEME_MODES:           # migrates the retired "system" pref
            current = DEFAULT_THEME
        sub = viewm.addMenu("Appearance")
        sub.menuAction().setIcon(common.icon("settings"))
        sub.setToolTip("Light or dark theme")
        group = QtGui.QActionGroup(self)
        group.setExclusive(True)
        for mode in THEME_MODES:                 # ("light", "dark")
            act = sub.addAction(mode.capitalize())
            act.setCheckable(True)
            act.setChecked(mode == current)
            act.triggered.connect(lambda _checked, m=mode: self._set_appearance(m))
            group.addAction(act)

    def _set_appearance(self, mode):
        """Switch the live theme and remember it for next launch."""
        from .. import prefs
        app = QtWidgets.QApplication.instance()
        set_theme(app, mode)
        retheme_open_plots(app)                  # repaint already-built pyqtgraph views
        prefs.set("theme", mode)

    def _check_for_updates(self, *, silent=False):
        """Check for updates. A **source checkout** compares against the remote
        commit-by-commit (every push counts — no GitHub release needed) and offers an
        in-place git pull; a **frozen bundle** falls back to the GitHub Releases check."""
        import threading
        from . import updater
        root = updater.repo_root()
        if root is None:                        # frozen bundle → versioned-release check
            from .. import __version__
            from . import update_check
            update_check.check_for_updates(self, __version__, silent=silent)
            return
        prog = None
        if not silent:
            prog = QtWidgets.QProgressDialog("Checking GitHub for updates…", "", 0, 0, self)
            prog.setWindowTitle("Check for updates")
            prog.setWindowModality(QtCore.Qt.WindowModal)
            prog.setMinimumDuration(0)
            prog.setCancelButton(None)
            prog.show()
        worker = updater.CheckWorker()
        self._update_check_worker = worker      # keep a ref so it isn't GC'd before it fires

        def _on_done(res, error):
            if prog is not None:
                prog.close()
            worker.deleteLater()
            self._update_check_worker = None
            if error or not res.get("ok"):
                if not silent:
                    QtWidgets.QMessageBox.information(
                        self, "Check for updates",
                        f"Couldn't check GitHub for updates.\n\n{error or res.get('message', '')}")
                return
            behind = int(res.get("behind", 0))
            if behind <= 0:
                if not silent:
                    QtWidgets.QMessageBox.information(
                        self, "Check for updates", "You're up to date — no new commits on GitHub.")
                return
            log = res.get("log") or []
            preview = "\n".join(f"  • {ln}" for ln in log[:8]) + ("\n  …" if behind > len(log) else "")
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("Update available")
            box.setText(f"{behind} new update{'s' if behind != 1 else ''} available on GitHub.")
            box.setInformativeText(f"{preview}\n\nUpdate now (git pull + restart)?")
            box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            box.setDefaultButton(QtWidgets.QMessageBox.Yes)
            if box.exec() == QtWidgets.QMessageBox.Yes:
                self._update_from_git(confirm=False)        # already confirmed here

        worker.done.connect(_on_done)
        threading.Thread(target=worker.run, args=(root,), daemon=True).start()

    def _update_from_git(self, confirm=True):
        """In-place update for a source checkout: git pull → reinstall if deps changed →
        offer to restart. The slow work runs on a daemon thread (see updater.Worker).
        ``confirm=False`` skips the pull prompt (used when the caller already confirmed)."""
        import threading
        from . import updater
        root = updater.repo_root()
        if root is None:                        # not a checkout (frozen bundle) — shouldn't reach here
            self._check_for_updates()
            return
        if confirm and QtWidgets.QMessageBox.question(
                self, "Update from GitHub",
                f"Pull the latest code into:\n{root}\n\nand restart if anything changed?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes) != QtWidgets.QMessageBox.Yes:
            return

        prog = QtWidgets.QProgressDialog("Contacting GitHub…", "", 0, 0, self)
        prog.setWindowTitle("Update from GitHub")
        prog.setWindowModality(QtCore.Qt.WindowModal)
        prog.setMinimumDuration(0)
        prog.setCancelButton(None)              # the pull/reinstall isn't safely interruptible

        worker = updater.Worker()
        self._git_update_worker = worker        # keep a ref so it isn't GC'd before it fires

        def _on_done(res, error):
            prog.close()
            worker.deleteLater()
            self._git_update_worker = None
            if error:
                QtWidgets.QMessageBox.warning(self, "Update failed", error)
                return
            if not res.get("changed"):
                QtWidgets.QMessageBox.information(
                    self, "Update", "Already up to date — nothing to pull.")
                return
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("Update installed")
            box.setText(f"Updated {res['before']} → {res['after']}"
                        + (" (dependencies reinstalled)." if res.get("deps_changed") else "."))
            box.setInformativeText("Restart now to run the new version?")
            box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            box.setDefaultButton(QtWidgets.QMessageBox.Yes)
            if box.exec() == QtWidgets.QMessageBox.Yes:
                updater.restart(root)
                QtWidgets.QApplication.quit()

        worker.progress.connect(prog.setLabelText)
        worker.done.connect(_on_done)
        prog.show()
        threading.Thread(target=worker.run, args=(root,), daemon=True).start()

    def _set_github_token(self):
        """Set/replace the saved GitHub token via the shared feedback token dialog.

        One token (stored in the OS keyring, or a 0600 file fallback) serves both the
        feature-request / agent-fix flow and *Check for updates*. Give it *Issues: Read and
        write* and *Contents: Read* on the SMILE-MSI repo, with a long expiry."""
        from . import feedback
        if feedback._prompt_for_token(self) is not None:
            QtWidgets.QMessageBox.information(self, "GitHub token", "Token saved.")

    def _open_feedback(self):
        from . import feedback
        feedback.open_feedback_dialog(self)

    def _open_my_requests(self):
        from . import feedback
        feedback.open_my_requests_dialog(self)

    # ----- annotation ------------------------------------------------------ #
    def annotate(self, mz) -> str:
        cs = self.ann.annotate_mz(float(mz))
        if not cs:
            return ""
        b = cs[0]
        return f"{b.lipid.name} {b.adduct} ({b.ppm:+.1f} ppm)"

    @property
    def norm(self):
        return self.norm_combo.currentText()

    @property
    def ppm(self):
        return float(self.ppm_spin.value())

    @property
    def id_ppm(self):
        """Identification (lipid-matching) tolerance in ppm — distinct from the
        extraction window ``ppm``."""
        return float(self.id_ppm_spin.value())

    @property
    def reduce(self):
        return self.reduce_combo.currentText()

    @property
    def composite_weight(self):
        """How lipid-class composites combine member ions: ``"raw"`` (total-abundance
        sum, default) or ``"balanced"`` (each ion scaled to its own 99th-percentile
        before summing, so faint members stay visible). See ``MSIDataset.composite_vector``."""
        combo = getattr(self, "composite_combo", None)
        return "balanced" if combo is not None and combo.currentText() == "balanced" else "raw"

    # ----- set-once controls (no left dock; homed in menus / dialogs) ----- #
    def _build_setup_widgets(self):
        """Create the controls that used to crowd the left column. They're plain
        widget attributes (so every ``self.mode_combo`` / ``self.ppm`` call site is
        unchanged); the actual *placement* is now a menu or an on-demand dialog. The
        display knobs that re-render the viewer (colormap/clip/normalization) are built
        with the right-hand Display panel instead — see ``_populate_display_section``."""
        # dataset summary banner — persistent reference data, shown atop the right dock
        self.info = QtWidgets.QLabel("No dataset loaded.")
        self.info.setWordWrap(True)

        # load-time: subsample stride (lives in the File menu)
        self.subsample_spin = NoScrollSpinBox()
        self.subsample_spin.setRange(1, 50)
        self.subsample_spin.setValue(1)
        self.subsample_spin.setToolTip("Subsample large files for a fast preview (1 = full resolution)")

        # acquisition & matching (Data → Acquisition & matching…)
        self.mode_combo = NoScrollComboBox()
        self.mode_combo.addItems(["negative", "positive"])
        self.mode_combo.currentTextChanged.connect(self._mode_changed)
        self.ppm_spin = NoScrollDoubleSpinBox()
        self.ppm_spin.setRange(1.0, 200.0)
        self.ppm_spin.setValue(10.0)
        self.ppm_spin.setSingleStep(1.0)
        self.ppm_spin.setToolTip(
            "m/z integration window used to extract ion images, the feature matrix, and "
            "all spatial statistics. This is NOT the lipid-identification tolerance — "
            "see 'Identification tolerance' below.")
        self.ppm_spin.valueChanged.connect(self._invalidate_features)
        # identification tolerance — the mass-accuracy window for in-silico lipid ID,
        # kept separate from (and tighter than) the extraction window above.
        self.id_ppm_spin = NoScrollDoubleSpinBox()
        self.id_ppm_spin.setRange(1.0, 30.0)
        self.id_ppm_spin.setValue(5.0)
        self.id_ppm_spin.setSingleStep(1.0)
        self.id_ppm_spin.setToolTip(
            "Mass-accuracy window for in-silico lipid identification — the feature list, "
            "peak-table lipid labels, annotation FDR, and report IDs. Tight (~5–10 ppm) "
            "for confident assignments; independent of the extraction tolerance above.")
        self.id_ppm_spin.valueChanged.connect(self._id_ppm_changed)
        self.reduce_combo = NoScrollComboBox()
        self.reduce_combo.addItems(["sum", "max", "mean"])
        self.reduce_combo.currentTextChanged.connect(self._invalidate_features)

        # peak picking (Data → Find peaks…)
        self.snr_spin = NoScrollDoubleSpinBox()
        self.snr_spin.setRange(1.0, 50.0)
        self.snr_spin.setValue(3.0)
        # Min prominence (in S/N units) for both finders: a peak must rise this far above
        # its neighbouring saddle. High values merge shoulders / closely-spaced peaks
        # (under-counts on dense qTOF spectra); 0 keeps every local maximum.
        self.prominence_spin = NoScrollDoubleSpinBox()
        self.prominence_spin.setRange(0.0, 20.0)
        self.prominence_spin.setSingleStep(0.5)
        self.prominence_spin.setValue(1.0)
        self.prominence_spin.setToolTip(
            "Min prominence (S/N units): how far a peak must rise above the neighbouring "
            "saddle to count. Higher merges shoulders / closely-spaced peaks; set 0 to "
            "keep every local maximum — surfaces more features on dense qTOF spectra. "
            "Applies to both the spectrum picker and the spatial finder.")
        self.minrel_spin = NoScrollDoubleSpinBox()
        self.minrel_spin.setRange(0.0, 1.0)
        self.minrel_spin.setDecimals(4)
        self.minrel_spin.setSingleStep(0.001)
        # 0.002 (0.2% of base peak) on the mean spectrum: the floor where the
        # cross-nerve-reproducible feature set stabilizes at ~270 features at
        # ~94% coverage. The old 0.02 (2%) discarded ~2/3 of real features. Raise toward
        # 0.005 for a tighter list, drop to 0.001 for maximum sensitivity.
        self.minrel_spin.setValue(0.002)
        self.proj_combo = NoScrollComboBox()
        self.proj_combo.addItems(["mean", "skyline (max)"])
        self.proj_combo.setToolTip("Pick on the mean spectrum, or the skyline (per-m/z maximum) "
                                   "which surfaces ions confined to a few pixels")
        self.proj_combo.currentTextChanged.connect(lambda *_: self._plot_mean_spectrum())
        # scope Find peaks to region(s): a checkable multi-select so you can pick over the
        # whole slide (nothing ticked), one region, or several taken *together* (the union
        # of their pixels → one feature list). Populated by _sync_pick_region_list.
        self.pick_region_list = common.CheckList(noun="region")
        self.pick_region_list.setMaximumHeight(180)
        self.pick_region_list.setToolTip(
            "Detect peaks over the whole slide (leave all unticked), restrict to one "
            "region, or tick several to pick over them together — one feature list scoped "
            "to the combined pixels.")
        self.pick_save_list_chk = QtWidgets.QCheckBox("Also save as a feature list (★)")
        self.pick_save_list_chk.setToolTip("After picking, snapshot the resulting features into a "
                                           "named ★ list in the library")
        # spatially-aware feature finder (Data → Find spatial features…)
        self.spatial_proj_combo = NoScrollComboBox()
        self.spatial_proj_combo.addItems(["mean", "skyline (max)", "both"])
        self.spatial_proj_combo.setToolTip(
            "Candidate source. 'mean' (recommended) starts from the high-coverage, "
            "low-noise mean-spectrum feature set; 'skyline (max)' surfaces ions bright "
            "in only a few pixels but is noisier; 'both' unions the two and lets the "
            "denoise gate strip the extra skyline noise.")
        self.spatial_freq_spin = NoScrollDoubleSpinBox()
        self.spatial_freq_spin.setRange(0.0, 100.0)
        self.spatial_freq_spin.setSingleStep(0.5)
        self.spatial_freq_spin.setValue(1.0)
        self.spatial_freq_spin.setSuffix(" %")
        self.spatial_freq_spin.setToolTip("Reproducibility gate: keep ions detected (above their "
                                          "own noise floor) in at least this % of pixels — drops "
                                          "single-pixel spikes")
        self.spatial_morans_spin = NoScrollDoubleSpinBox()
        self.spatial_morans_spin.setRange(-1.0, 1.0)
        self.spatial_morans_spin.setSingleStep(0.05)
        self.spatial_morans_spin.setValue(0.05)
        self.spatial_morans_spin.setToolTip(
            "Spatial-denoise gate: keep ions whose image is spatially structured "
            "(Moran's I ≥ this) and drop spatially-random chemical/electronic noise. "
            "Real focal or ubiquitous ions score far above the noise floor, so a low "
            "value (≈0.05) strips noise without culling signal; raise for a stricter cut.")
        self.spatial_deiso_chk = QtWidgets.QCheckBox("Collapse isotopes (M+1/M+2)")
        self.spatial_deiso_chk.setChecked(False)
        self.spatial_deiso_chk.setToolTip(
            "Fold M+1/M+2 isotope satellites into their monoisotopic peak so one compound "
            "is one feature — the 'region-complete' reduction (one compound → one feature). On nerve "
            "data ~⅓–⅖ of picked peaks are isotopologues; turn this on to bring the feature "
            "count down. Off = the raw feature set (keeps satellites).")
        self.spatial_fdr_combo = NoScrollComboBox()
        self.spatial_fdr_combo.addItems(["off", "≤ 5%", "≤ 10%", "≤ 20%"])
        self.spatial_fdr_combo.setToolTip(
            "Threshold the feature list by target–decoy FDR (METASPACE-style q-value) "
            "instead of an intensity cutoff — the publication-grade way to size a feature "
            "list (a controlled error rate, not a slider position). Unannotated 'unknown' "
            "ions are kept regardless, so a novel biomarker is never dropped by the lipid DB.")

        # preprocessing (Data → Preprocessing…)
        self.pp_baseline_combo = NoScrollComboBox()
        self.pp_baseline_combo.addItems(["none", "SNIP", "local minimum", "convex hull", "median"])
        self.pp_baseline_combo.setToolTip("Baseline subtraction method.")
        self.pp_baseline_param = NoScrollSpinBox()
        self.pp_baseline_param.setRange(3, 400)
        self.pp_baseline_param.setValue(40)
        self.pp_baseline_param.setToolTip("SNIP: iterations. local-min/median: window (points).")
        self.pp_smooth_combo = NoScrollComboBox()
        self.pp_smooth_combo.addItems(["none", "Savitzky-Golay", "Gaussian"])
        self.pp_smooth_param = NoScrollDoubleSpinBox()
        self.pp_smooth_param.setRange(0.5, 51.0)
        self.pp_smooth_param.setValue(9.0)
        self.pp_smooth_param.setToolTip("Savitzky-Golay: window (points). Gaussian: sigma.")
        self.pp_recal_edit = QtWidgets.QLineEdit()
        self.pp_recal_edit.setPlaceholderText("lock-mass m/z, comma-separated (e.g. 255.2330, 885.5499)")
        self.pp_recal_tol = NoScrollDoubleSpinBox()
        self.pp_recal_tol.setRange(1.0, 1000.0)
        self.pp_recal_tol.setValue(200.0)
        self.pp_recal_tol.setSuffix(" ppm")
        self.pp_norm_combo = NoScrollComboBox()
        self.pp_norm_combo.addItems(["none", "vector (L2)", "reference m/z"])
        self.pp_norm_combo.setToolTip("Per-spectrum normalization (separate from the feature-matrix "
                                      "TIC/RMS/median used for imaging/stats).")
        self.pp_norm_ref = QtWidgets.QLineEdit()
        self.pp_norm_ref.setPlaceholderText("internal-standard m/z")

    def _install_status_progress(self):
        """Progress + cancel are persistent app state → pin them to the status bar
        (bottom-right) so they're visible from every tab without a left column."""
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setMaximumWidth(170)
        self.progress.setTextVisible(False)
        self.b_cancel = QtWidgets.QPushButton("Cancel")
        self.b_cancel.setEnabled(False)
        self.b_cancel.setMaximumWidth(80)
        self.b_cancel.clicked.connect(self._cancel_work)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.b_cancel)

    # ----- on-demand setup dialogs (reused; built once, then re-shown) ----- #
    @staticmethod
    def _show_dialog(dlg):
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _build_image_setup_dialog(self):
        """The optical / histology backdrop is a once-per-sample setup (load · align · show ·
        opacity), so it lives in this small non-modal dialog (File → Image setup…) instead of
        a right-panel section. Non-modal so 'Drag to move' alignment works against the live ion
        image. Built eagerly + hidden so the optical controls exist for session restore and the
        View → Show optical image quick toggle even before the dialog is first opened."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Image setup — optical / histology backdrop")
        lay = QtWidgets.QVBoxLayout(dlg)
        self._populate_optical_section(lay)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btns.rejected.connect(dlg.hide)
        lay.addWidget(btns)
        self._image_setup_dialog = dlg

    def _open_image_setup_dialog(self):
        self._sync_optical_controls()
        self._show_dialog(self._image_setup_dialog)

    # ----- analysis profiles (Data → Analysis profiles…) ----------------- #
    # profile-key → (live-widget attribute, kind). The prefs-backed knobs
    # (segmentation defaults, random seed) are mirrored via profiles.mirror_prefs
    # rather than a live widget, so they're handled separately below.
    _PROFILE_WIDGETS = {
        "mode": ("mode_combo", "combo"),
        "ppm": ("ppm_spin", "dspin"),
        "id_ppm": ("id_ppm_spin", "dspin"),
        "norm": ("norm_combo", "combo"),
        "reduce": ("reduce_combo", "combo"),
        "snr": ("snr_spin", "dspin"),
        "prominence": ("prominence_spin", "dspin"),
        "minrel": ("minrel_spin", "dspin"),
        "projection": ("proj_combo", "combo"),
        "baseline_method": ("pp_baseline_combo", "combo"),
        "baseline_param": ("pp_baseline_param", "ispin"),
        "smooth_method": ("pp_smooth_combo", "combo"),
        "smooth_param": ("pp_smooth_param", "dspin"),
        "recal_refs": ("pp_recal_edit", "text"),
        "recal_tol": ("pp_recal_tol", "dspin"),
        "pp_norm_method": ("pp_norm_combo", "combo"),
        "pp_norm_ref": ("pp_norm_ref", "text"),
        "spatial_projection": ("spatial_proj_combo", "combo"),
        "spatial_min_freq": ("spatial_freq_spin", "dspin"),
        "spatial_min_morans": ("spatial_morans_spin", "dspin"),
        "spatial_fdr": ("spatial_fdr_combo", "combo"),
    }
    _PROFILE_PREFS = {                              # profile-key → prefs key (no live widget)
        "seg_method": profiles.SEG_METHOD_PREF,
        "seg_metric": profiles.SEG_METRIC_PREF,
        "segment_default_count": profiles.SEG_COUNT_PREF,
        "segment_detail_max": profiles.SEG_DETAIL_PREF,
        "random_seed": profiles.SEED_PREF,
    }

    def _open_profiles_dialog(self):
        dlg = getattr(self, "_profiles_dialog", None)
        if dlg is None:
            from .profiles import ProfilesDialog
            dlg = ProfilesDialog(self)
            self._profiles_dialog = dlg
        else:
            dlg.refresh()
        self._show_dialog(dlg)

    def _apply_profile_params(self, params: dict) -> None:
        """Push a profile's parameters into the live controls (soft: only sets defaults,
        the user can still change anything afterward) + the dataset + the prefs-backed
        knobs. Called on a fresh dataset open and from the profile dialog's Apply button."""
        params = profiles.merge_defaults(params)
        for key, (attr, kind) in self._PROFILE_WIDGETS.items():
            w = getattr(self, attr, None)
            if w is None:
                continue
            v = params[key]
            if kind == "combo":
                w.setCurrentText(str(v))
            elif kind == "dspin":
                w.setValue(float(v))
            elif kind == "ispin":
                w.setValue(int(v))
            else:
                w.setText(str(v))
        if self.ds is not None:
            self.ds.tic_max_amp = float(params["tic_max_amp"])
        profiles.mirror_prefs(params)              # segmentation defaults + random seed

    def _profile_widget_params(self) -> dict:
        """Read the current live settings into a profile-shaped dict — used to detect (and
        record) where the open sample has deviated from the active profile."""
        out = {}
        for key, (attr, kind) in self._PROFILE_WIDGETS.items():
            w = getattr(self, attr, None)
            if w is None:
                continue
            if kind == "combo":
                out[key] = w.currentText()
            elif kind in ("dspin", "ispin"):
                out[key] = w.value()
            else:
                out[key] = w.text()
        ap = profiles.active_params()
        out["tic_max_amp"] = (float(getattr(self.ds, "tic_max_amp", ap["tic_max_amp"]))
                              if self.ds is not None else ap["tic_max_amp"])
        for key, pref in self._PROFILE_PREFS.items():
            out[key] = prefs.get(pref, ap[key])
        return profiles.merge_defaults(out)

    def _prefill_from_active_profile(self):
        """Apply the active profile's defaults to a freshly-loaded sample. Runs inside
        _on_dataset *before* any saved-session settings, so a restored session still wins
        (per-sample, soft) while a brand-new sample inherits the lab's standard defaults."""
        try:
            self._apply_profile_params(profiles.active_params())
        except Exception:  # noqa: BLE001 — a profile hiccup must never block a data load
            pass

    def _open_acquisition_dialog(self):
        dlg = getattr(self, "_acq_dialog", None)
        if dlg is None:
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("Acquisition & matching")
            form = QtWidgets.QFormLayout(dlg)
            form.addRow("Mode (polarity)", self.mode_combo)
            form.addRow("Extraction tolerance (ppm)", self.ppm_spin)
            form.addRow("Identification tolerance (ppm)", self.id_ppm_spin)
            form.addRow("Reduce (m/z window)", self.reduce_combo)
            btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
            btns.rejected.connect(dlg.hide)
            form.addRow(btns)
            self._acq_dialog = dlg
        self._show_dialog(dlg)

    def _ask_db_import_mode(self):
        """Ask whether to merge an external lipid DB onto the built-in one or replace it.
        Returns ``'merge'`` / ``'replace'`` / ``None`` (cancel). Split out so tests can drive
        the import headless by calling ``_import_lipid_db(mode=...)`` directly."""
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Import lipid database")
        box.setText("Add these lipids to the built-in database, or replace it?")
        box.setInformativeText(
            "Merge keeps the built-in nerve/myelin tissue priors and the curated metabolites "
            "(NAA, taurine, …) and adds the imported DB's extra species — recommended for a "
            "lipids-only export like LIPID MAPS.\nReplace annotates against only the imported DB.")
        merge_btn = box.addButton("Merge (recommended)", QtWidgets.QMessageBox.AcceptRole)
        replace_btn = box.addButton("Replace", QtWidgets.QMessageBox.DestructiveRole)
        box.addButton(QtWidgets.QMessageBox.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is merge_btn:
            return "merge"
        if clicked is replace_btn:
            return "replace"
        return None

    def _open_setup_wizard(self, *_a):
        """Open the guided Setup wizard (File ▸ Setup wizard…). Re-runnable any time."""
        from .setupwizard import SetupWizard
        SetupWizard(self, self).exec()

    def _run_startup_setup(self):
        """Startup: re-apply the remembered lipid DB, then open the setup wizard on first launch
        only. Best-effort — a failure here must never block startup. Skipped under pytest so the
        modal wizard can't hang the headless GUI suite (tests drive the wizard explicitly)."""
        import sys
        try:
            self._apply_remembered_lipid_db()
        except Exception:  # noqa: BLE001
            pass
        if "pytest" not in sys.modules and not prefs.get("setup_done"):
            try:
                self._open_setup_wizard()
            except Exception:  # noqa: BLE001
                pass

    _LIPID_DB_PREF = "lipid_db"      # remembered external DB so setup persists across launches

    def _load_lipid_db_file(self, path, *, mode="merge", categories=None,
                            min_mass=None, max_mass=None):
        """Load an external lipid DB and return ``(db, note)``. A ``.sdf`` (LIPID MAPS LMSD)
        goes through :func:`lipiddb.load_sdf` with the optional category/mass filters; anything
        else is a CSV/TSV via :func:`lipiddb.load_external_db`. ``mode='merge'`` keeps the
        built-in tissue priors + curated metabolites; ``mode='replace'`` uses only the file.
        Raises on read/parse error. The one loader shared by the importer, the setup wizard,
        and the startup auto-load."""
        from .. import lipiddb
        is_sdf = str(path).lower().endswith(".sdf")
        if is_sdf:
            ext = lipiddb.load_sdf(path, categories=categories, min_mass=min_mass, max_mass=max_mass)
        if mode == "merge":
            if is_sdf:
                db, n_ext, n_added = lipiddb.merge_lipids(lipiddb.build_database(), ext)
            else:
                db, n_ext, n_added = lipiddb.merge_external_db(path)
            note = f"Merged {n_added} new lipids (of {n_ext} usable) onto the built-in DB"
        else:
            db = ext if is_sdf else lipiddb.load_external_db(path)
            note = f"Loaded {len(db)} lipids (built-in replaced)"
        return db, note

    def _set_lipid_db(self, db):
        """Make ``db`` (or ``None`` → built-in) the active annotation database and drop cached
        feature lists so the next build annotates against it."""
        self._lipid_db = db
        self._invalidate_feature_list_cache()
        self.ann = Annotator(mode=self.mode_combo.currentText(), ppm_tol=self.id_ppm, db=db)

    def _apply_remembered_lipid_db(self):
        """On startup, re-load the external lipid DB the user set up in a previous session so the
        choice sticks. A missing/moved file is reported, not fatal — the app falls back to the
        built-in DB and points the user at the wizard.

        Runs as a background job (:meth:`_run`), not inline: a large remembered LMSD ``.sdf``
        (~48k records) parsed synchronously here would freeze the window before it's even
        shown. The load error path stays a quiet status-bar message rather than the standard
        error dialog — this must never interrupt startup with a popup."""
        import os
        cfg = prefs.get(self._LIPID_DB_PREF) or None
        if not isinstance(cfg, dict) or not cfg.get("path"):
            return
        if not os.path.exists(cfg["path"]):
            self.statusBar().showMessage(
                f"Set-up lipid database not found ({os.path.basename(cfg['path'])}) — re-run "
                "File ▸ Setup wizard to point at it.")
            return
        path, mode = cfg["path"], cfg.get("mode", "merge")
        categories, min_mass, max_mass = cfg.get("categories"), cfg.get("min_mass"), cfg.get("max_mass")

        def load():
            try:
                return self._load_lipid_db_file(path, mode=mode, categories=categories,
                                                 min_mass=min_mass, max_mass=max_mass)
            except Exception as e:  # noqa: BLE001 — a bad file must not block startup
                return None, f"Could not load set-up lipid database: {e}"

        def done(result):
            db, note = result
            if db:
                self._set_lipid_db(db)
                note = f"{note} (from your saved setup)."
            self.statusBar().showMessage(note)

        self._run(load, on_done=done, busy="Loading your saved lipid database…",
                  title="Loading lipid database…")

    def _import_lipid_db(self, *_a, mode=None):
        """Load an external lipid/metabolite database (LIPID MAPS ``.sdf`` / SwissLipids / custom
        CSV) for annotation. ``mode='merge'`` adds it onto the built-in in-silico DB (keeping the
        tissue priors + curated metabolites); ``mode='replace'`` uses only the imported DB;
        ``None`` prompts. Re-pick or rebuild the feature list to apply. Import again to revert."""
        if getattr(self, "_lipid_db", None):           # already using an imported DB → toggle back
            self._set_lipid_db(None)
            prefs.set(self._LIPID_DB_PREF, None)       # and stop restoring it next launch
            self.statusBar().showMessage("Reverted to the built-in in-silico database.")
            return
        path, _ = filedialogs.get_open_file_name(
            self, "Import lipid database (LMSD .sdf / CSV / TSV)", "",
            "Lipid databases (*.sdf *.csv *.tsv *.txt);;All (*)")
        if not path:
            return
        if mode is None:
            mode = self._ask_db_import_mode()
        if mode not in ("merge", "replace"):
            return

        # The full LIPID MAPS LMSD .sdf is ~48k records with a molblock per entry — parsing
        # it synchronously here used to freeze the whole window (reads as "crashed") for as
        # long as the parse took. Route it through the same background-worker path every
        # other heavy analysis uses; a load error surfaces via the standard error dialog
        # (Worker catches it and routes to _on_error) instead of a manual try/except here.
        def load():
            return self._load_lipid_db_file(path, mode=mode)

        def done(result):
            db, note = result
            if not db:
                QtWidgets.QMessageBox.warning(self, "Empty database",
                                              "No usable rows (need name + formula columns).")
                return
            self._set_lipid_db(db)
            # Remember it, exactly as the setup wizard does — otherwise a DB imported from this
            # menu is silently forgotten at the next launch and _apply_remembered_lipid_db has
            # nothing to restore. Same pref key, same shape, so either entry point round-trips.
            prefs.set(self._LIPID_DB_PREF, {"path": path, "mode": mode, "categories": None,
                                            "min_mass": None, "max_mass": None})
            self.statusBar().showMessage(f"{note} — rebuild the feature list to annotate "
                                         "against them. It will load again next launch. "
                                         "(Import again to revert.)")

        self._run(load, on_done=done, busy="Loading lipid database…", modal=True,
                  title="Importing lipid database…")

    def _open_preprocess_dialog(self):
        dlg = getattr(self, "_pp_dialog", None)
        if dlg is None:
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("Preprocessing")
            v = QtWidgets.QVBoxLayout(dlg)
            v.addWidget(QtWidgets.QLabel("Re-primes the dataset; find peaks again afterward."))
            form = QtWidgets.QFormLayout()
            bl_row = QtWidgets.QHBoxLayout()
            bl_row.addWidget(self.pp_baseline_combo); bl_row.addWidget(self.pp_baseline_param)
            form.addRow("Baseline", bl_row)
            sm_row = QtWidgets.QHBoxLayout()
            sm_row.addWidget(self.pp_smooth_combo); sm_row.addWidget(self.pp_smooth_param)
            form.addRow("Smoothing", sm_row)
            rc_row = QtWidgets.QHBoxLayout()
            rc_row.addWidget(self.pp_recal_edit); rc_row.addWidget(self.pp_recal_tol)
            form.addRow("Lock-mass recal.", rc_row)
            nm_row = QtWidgets.QHBoxLayout()
            nm_row.addWidget(self.pp_norm_combo); nm_row.addWidget(self.pp_norm_ref)
            form.addRow("Per-spectrum norm.", nm_row)
            v.addLayout(form)
            b_apply = QtWidgets.QPushButton("Apply preprocessing")
            b_apply.setObjectName("primaryAction")        # the dialog's single commit action
            b_apply.clicked.connect(self.apply_preprocessing)
            v.addWidget(b_apply)
            btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
            btns.rejected.connect(dlg.hide)
            v.addWidget(btns)
            self._pp_dialog = dlg
        self._show_dialog(dlg)

    # ----- absolute calibration check (measure → one-click lock-mass recal) ------ #
    def _open_calibration_dialog(self):
        """Measure the dataset's absolute m/z offset against known reference ions and, if it's
        a constant offset, apply a one-click lock-mass recalibration. Closes the gap between
        the QC (:func:`intake.mass_drift`, spread only) and the fix (Preprocessing recal)."""
        dlg = getattr(self, "_calib_dialog", None)
        if dlg is None:
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("Calibration check")
            v = QtWidgets.QVBoxLayout(dlg)
            v.addWidget(QtWidgets.QLabel(
                "Compares the mean spectrum to known reference lipid ions to find the "
                "systematic m/z offset that causes mislabels.\nApply lock-mass recalibration "
                "if the offset is a constant ppm shift (flat slope), then re-pick peaks."))
            self.calib_table = QtWidgets.QTableWidget(0, 4)
            self.calib_table.setHorizontalHeaderLabels(["Anchor", "ref m/z", "obs m/z", "ppm"])
            self.calib_table.horizontalHeader().setStretchLastSection(True)
            self.calib_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            v.addWidget(self.calib_table)
            self.calib_summary = QtWidgets.QLabel("Load a dataset, then Measure.")
            self.calib_summary.setWordWrap(True)
            v.addWidget(self.calib_summary)
            row = QtWidgets.QHBoxLayout()
            b_measure = QtWidgets.QPushButton("Measure")
            b_measure.setToolTip("Measure the absolute m/z offset against the reference ions")
            b_measure.clicked.connect(lambda: self._run_calibration_check())
            self.calib_apply_btn = QtWidgets.QPushButton("Apply lock-mass recalibration")
            self.calib_apply_btn.setObjectName("primaryAction")
            self.calib_apply_btn.setEnabled(False)
            self.calib_apply_btn.setToolTip(
                "Fill the Preprocessing lock-mass field with the matched anchors, apply, "
                "and re-verify. Re-pick peaks afterward.")
            self.calib_apply_btn.clicked.connect(self._apply_calibration_recal)
            row.addWidget(b_measure); row.addWidget(self.calib_apply_btn)
            v.addLayout(row)
            btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
            btns.rejected.connect(dlg.hide)
            v.addWidget(btns)
            self._calib_dialog = dlg
            self._calib_result = None
        self._show_dialog(dlg)

    def _run_calibration_check(self, *, verify=False):
        if self.ds is None:
            self.statusBar().showMessage("Load a dataset first.")
            return
        mode = self.mode_combo.currentText()
        self._run(intake.measure_calibration_offset, self.ds, busy="Measuring calibration…",
                  on_done=lambda r: self._on_calibration_result(r, verify=verify),
                  label="Calibration check", mode=mode)

    def _on_calibration_result(self, res, *, verify=False):
        self._calib_result = res
        anchors = res.get("anchors", []) if res else []
        self.calib_table.setRowCount(len(anchors))
        for i, a in enumerate(anchors):
            vals = [a["label"], f"{a['ref_mz']:.4f}", f"{a['obs_mz']:.4f}", f"{a['ppm']:+.2f}"]
            for j, txt in enumerate(vals):
                self.calib_table.setItem(i, j, QtWidgets.QTableWidgetItem(txt))
        if not anchors:
            self.calib_summary.setText("No reference ions matched — check polarity/mode, or "
                                       "that the dataset covers the anchor m/z range.")
            self.calib_apply_btn.setEnabled(False)
            return
        med = res["median_ppm"]
        slope = res["slope_ppm_per_da"]
        mass_dependent = abs(slope) > 0.5
        if abs(med) < 2.0:
            msg = f"Well calibrated: median {med:+.2f} ppm (n={res['n']}). No recalibration needed."
            self.calib_apply_btn.setEnabled(False)
        elif mass_dependent:
            msg = (f"Median {med:+.2f} ppm but mass-dependent (slope {slope:+.3f} ppm/Da) — a "
                   f"single-point lock mass won't fully fix this; use multi-point recalibration.")
            self.calib_apply_btn.setEnabled(True)
        else:
            msg = (f"Constant offset: median {med:+.2f} ppm (n={res['n']}, slope {slope:+.3f} "
                   f"ppm/Da). Apply lock-mass recalibration (×{res['factor']:.7f}).")
            self.calib_apply_btn.setEnabled(True)
        if verify:
            msg = ("After recalibration — " + msg + ("  ✓ target reached (<2 ppm)."
                   if abs(med) < 2.0 else "  Re-pick peaks; residual still present."))
        self.calib_summary.setText(msg)

    def _apply_calibration_recal(self):
        res = getattr(self, "_calib_result", None)
        if self.ds is None or not res or not res.get("anchors"):
            return
        refs = " ".join(f"{a['ref_mz']:.4f}" for a in res["anchors"])
        self.pp_recal_edit.setText(refs)
        # window must exceed the measured offset so recalibrate locks onto the right apex
        self.pp_recal_tol.setValue(max(float(self.pp_recal_tol.value()),
                                       abs(res["max_abs_ppm"]) * 3.0 + 10.0))
        self.apply_preprocessing()             # queued: re-primes on the recalibrated axis
        if self.prov is not None:
            self.prov.step("calibration_recalibrate", n_anchors=len(res["anchors"]),
                           median_ppm=round(res["median_ppm"], 3))
        # queued after the reprime job → measures on the corrected mean spectrum
        self._run_calibration_check(verify=True)

    def _open_pick_dialog(self, focus_spatial=False):
        """Combined feature-detection dialog. Two finders share one set of controls
        (region / S/N / ★-save) so no widget is parented into two dialogs: the plain
        spectrum picker on top, and the spatially-aware finder below.
        ``focus_spatial`` puts the cursor on the spatial gates (Data → Find spatial
        features…)."""
        dlg = getattr(self, "_pick_dialog", None)
        if dlg is None:
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("Find features")
            v = QtWidgets.QVBoxLayout(dlg)
            shared = QtWidgets.QFormLayout()
            shared.addRow("Region(s)", self.pick_region_list)
            shared.addRow("Min S/N", self.snr_spin)
            shared.addRow("Min prominence (S/N)", self.prominence_spin)
            shared.addRow("", self.pick_save_list_chk)
            v.addLayout(shared)
            # The two finders share the form above; split their own controls into
            # one-click-deeper inner tabs so only the shared knobs sit on the surface.
            tabs = QtWidgets.QTabWidget()
            self.pick_tabs = tabs
            # Tab 1 — plain spectrum picker
            pp_page = QtWidgets.QWidget()
            pp_v = QtWidgets.QVBoxLayout(pp_page)
            pf = QtWidgets.QFormLayout()
            pf.addRow("Spectrum", self.proj_combo)
            pf.addRow("Min rel. intensity", self.minrel_spin)
            pp_v.addLayout(pf)
            b_pick = QtWidgets.QPushButton("Find peaks")
            b_pick.setObjectName("primaryAction")         # the tab's single run action
            b_pick.setIcon(common.icon("find"))
            b_pick.setToolTip("Detect peaks with these settings and build the working feature set")
            b_pick.clicked.connect(self.do_pick_peaks)
            pp_v.addWidget(b_pick)
            pp_v.addStretch()
            tabs.addTab(pp_page, "Peak picking")
            # Tab 2 — spatially-aware finder
            sp_page = QtWidgets.QWidget()
            sp_v = QtWidgets.QVBoxLayout(sp_page)
            sp_v.addWidget(note("Mean candidates → frequency gate → spatial-denoise gate → "
                                "per-feature auto-width. Starts from the high-coverage mean "
                                "feature set, then strips spatially-random noise without "
                                "culling real diffuse ions."))
            sf = QtWidgets.QFormLayout()
            sf.addRow("Candidates", self.spatial_proj_combo)
            sf.addRow("Min frequency", self.spatial_freq_spin)
            sf.addRow("Min Moran's I", self.spatial_morans_spin)
            sf.addRow("FDR threshold", self.spatial_fdr_combo)
            sp_v.addLayout(sf)
            sp_v.addWidget(self.spatial_deiso_chk)
            b_find = QtWidgets.QPushButton("Find spatial features")
            b_find.setObjectName("primaryAction")         # the tab's single run action
            b_find.setIcon(common.icon("find"))
            b_find.setToolTip("Run the spatial pipeline and build the working feature set")
            b_find.clicked.connect(self.do_find_spatial_features)
            sp_v.addWidget(b_find)
            sp_v.addStretch()
            tabs.addTab(sp_page, "Spatial finder")
            v.addWidget(tabs)
            btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
            btns.rejected.connect(dlg.hide)
            v.addWidget(btns)
            self._pick_dialog = dlg
        self._sync_pick_region_list()         # reflect any regions added/removed since last open
        self._show_dialog(dlg)
        if focus_spatial:
            self.pick_tabs.setCurrentIndex(1)
            self.spatial_morans_spin.setFocus()

    # ----- central tabs ---------------------------------------------------- #
    def _build_tabs(self):
        self.tabs = _DockTabWidget()
        # NB: the central widget is set later by _build_right_panel → _assemble_center_split,
        # which wraps these tabs and the right panel in a collapsible splitter.
        self._tab_ion()
        self._tab_segment()
        self._tab_gallery()          # the Analyze launcher — a card per registry step (plan 24)
        # Feature space + Components (PCA/NMF) retired as tabs (plan 24 Phase 4) — pca / nmf /
        # embedding now open in the AnalysisDialog (embedding keeps its lasso→region carve).
        # Discriminating / Multi-group / Venn / ROI-localization / Region-comparison (per-ion
        # stats + spectra overlay) likewise open in the AnalysisDialog (plan 24 Phase 4.5/4.6).
        # Co-localization retired as a tab (plan 24 Phase 4) — colocalize / modules / region
        # correlation now open in the AnalysisDialog from the Analyze gallery.
        # Montage (ion-image grid) demoted off the strip (plan 24 Phase 6) — File ▸ Ion montage
        # grid… opens do_montage in a modeless dialog; it was never a registry view.
        # Cohort screens live in an on-demand window, not the strip (plan 24 Phase 5). The four
        # _tab_cohort* builders add into this container instead of self.tabs; the Analyze
        # gallery's Cohort-scope cards reveal them (reveal_view → _show_cohort_screen).
        self._cohort_tabs = QtWidgets.QTabWidget()
        self._cohort_tabs.setDocumentMode(True)
        self._cohort_window = None   # lazily wrapped around _cohort_tabs on first reveal
        self._tab_cohort()           # cross-sample / between-group comparison
        self._tab_cohort_nested()    # subject × compartment nested model (mixed / moderated-t)
        self._tab_cohort_embed()     # pooled UMAP/t-SNE over every sample's pixels
        self._tab_cohort_segment()   # one segmentation tree shared across every sample
        # Feature lists demoted off the strip (plan 24 Phase 6) — the Features ⋯ menu's
        # "Manage feature lists…" (and reveal_view("Feature lists")) open it as a modeless
        # dialog; it only ever mirrored the dock's feature-set selector. Built (hidden) now so
        # the mirror table stays in sync from startup, exactly as the old strip tab did.
        self._build_feature_lists_dialog()
        # Classify (plsda/classify_cv/classify_map) and SHAP biomarkers retired as tabs
        # (plan 24 Phase 4) — both now open in the AnalysisDialog from the Analyze gallery.
        # Markers (SSC) + Per-ion (DGMM) are no longer tabs — they run in the generic
        # AnalysisDialog launched from the Analyze gallery (plan 24, Phase 2b).
        self._tab_history()          # Analyses ▸ History — the re-openable run log (plan 24)
        self._tab_report()           # Analyses ▸ Report book — curate the PDF data book
        self._regroup_tabs()         # collapse the flat strip into ~9 grouped top-level tabs
        self._build_tab_strip()

    def _build_tab_strip(self):
        """Lift the tab strip out of the central column into a full-width toolbar that
        spans above the central area *and* the right-hand dock — so all the view tabs read
        as one clean strip instead of overflowing into scroll arrows."""
        self.tab_strip = QtWidgets.QTabBar()
        self.tab_strip.setDocumentMode(True)          # flatter, modern tab look
        self.tab_strip.setExpanding(False)            # tabs take natural width, left-aligned
        self.tab_strip.setUsesScrollButtons(False)    # scroll the whole strip, no arrow paging
        self.tab_strip.setElideMode(QtCore.Qt.ElideNone)
        # Wrap the strip in a scroll area: when the window is too narrow to show every tab,
        # the strip scrolls sideways (two-finger swipe) and the » overflow button jumps to
        # any off-screen tab. usesScrollButtons(False) keeps the bar's minimum width = all
        # tabs, so the scroll range grows exactly when tabs don't fit.
        #
        # Both scrollbars are forced OFF: a *visible* horizontal scrollbar here is a macOS
        # overlay that paints ON TOP of the tab strip (which is pinned to a fixed height
        # with no room below it) — that read as ghosted/clipped tabs with a stray grey bar
        # across them. The » button + swipe cover scrolling without it; _sync_tab_overflow
        # still keys off the scrollbar's range, which updates even while the bar is hidden.
        self.tab_scroll = QtWidgets.QScrollArea()
        self.tab_scroll.setWidget(self.tab_strip)
        self.tab_scroll.setWidgetResizable(True)
        self.tab_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.tab_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.tab_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        tb = QtWidgets.QToolBar("View")
        tb.setObjectName("view_tab_strip")
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setStyleSheet("QToolBar { border: 0px; padding: 0px; spacing: 0px; }")
        tb.addWidget(self.tab_scroll)
        # Overflow menu: a » button at the right edge, shown only when tabs don't all fit.
        # It drops down the views that are currently off-screen — one click jumps to any of
        # them (and scrolls it into view), so a small screen never hides a tab behind a swipe.
        self.tab_overflow = QtWidgets.QToolButton()
        self.tab_overflow.setIcon(common.icon("overflow"))
        self.tab_overflow.setText("")
        self.tab_overflow.setAccessibleName("More views (tabs that don't fit)")
        self.tab_overflow.setAutoRaise(True)
        self.tab_overflow.setToolTip("More views (tabs that don't fit)")
        self.tab_overflow.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        of_menu = QtWidgets.QMenu(self.tab_overflow)
        of_menu.aboutToShow.connect(self._populate_tab_overflow)
        self.tab_overflow.setMenu(of_menu)
        # addWidget wraps the button in a QWidgetAction; toolbar visibility must be driven
        # through THAT action — button.setVisible(False) alone won't hide it (it'd sit there
        # as dead clutter), which is the bug that left » showing when every tab already fit.
        self._tab_overflow_action = tb.addWidget(self.tab_overflow)
        self._tab_overflow_action.setVisible(False)
        # Queue button, pinned to the right end of the strip: a spacer eats the slack so it
        # sits beside the view tabs with a live "(N queued)" badge.
        self._build_queue_button(tb)
        self.addToolBar(QtCore.Qt.TopToolBarArea, tb)
        self.tabs.attach_strip(self.tab_strip)
        # Opening the Region-comparison tab after changing the active ion elsewhere should
        # show that ion, not whatever was last picked inside the tab (_sync_cmp_active no-ops
        # unless a comparison has been run and the tab is now on-screen).
        self.tabs.currentChanged.connect(
            lambda _i: self._sync_cmp_active(getattr(self, "active_mz", None)))
        # likewise the ROI-stats ROC curve follows the active ion when that tab opens.
        self.tabs.currentChanged.connect(
            lambda _i: self._sync_stats_active(getattr(self, "active_mz", None)))
        # pin the strip's height now that it holds tabs (an empty bar reports too short).
        # No scrollbar is shown anymore, so the viewport gets the full height — +2 is just
        # breathing room so descenders in the tab text aren't clipped.
        self.tab_scroll.setFixedHeight(self.tab_strip.sizeHint().height() + 2)
        # Anchor the strip at the left so 'Ion image' (tab 0) is ALWAYS visible: eat wheel
        # events (no swipe-scroll) and snap any stray scroll back to 0. Overflow therefore
        # only ever happens on the right, where the » menu reaches the off-screen tabs.
        self._tab_nowheel = _NoWheelScroll(self)
        self.tab_scroll.viewport().installEventFilter(self._tab_nowheel)
        hbar = self.tab_scroll.horizontalScrollBar()
        hbar.valueChanged.connect(lambda v: v and hbar.setValue(0))
        # show/hide the » button as overflow appears/disappears (range changes on resize)
        self.tab_scroll.horizontalScrollBar().rangeChanged.connect(self._sync_tab_overflow)
        # rangeChanged drives every later resize, but the scroll range isn't known until the
        # window has laid out — so the build-time call below sees max=0 and wrongly hides ».
        # Defer one more sync to the next event-loop turn (post-show) to fix the initial state.
        self._sync_tab_overflow()
        QtCore.QTimer.singleShot(0, self._sync_tab_overflow)

    def _build_queue_button(self, tb):
        """Add the 'Queue (N)' button to the view-tab toolbar and its dockable job panel."""
        from .jobqueue import JobQueuePanel
        # No separate spacer here: tab_scroll (a QScrollArea) is already horizontally
        # Expanding by default. A second Expanding widget in the same toolbar splits the
        # slack between the two, so the tab strip only ever claimed ~half the bar and
        # tripped the » overflow well before it needed to. tab_scroll alone now grows to
        # fill everything up to this button.
        self.b_queue = QtWidgets.QToolButton()
        self.b_queue.setText("Queue")
        self.b_queue.setToolTip("Show queued and running analyses")
        self.b_queue.setAutoRaise(True)
        self.b_queue.clicked.connect(self._toggle_queue_panel)
        tb.addWidget(self.b_queue)

        self._queue_panel = JobQueuePanel(self.jobs, self._cancel_job)
        self._queue_dock = QtWidgets.QDockWidget("Analysis queue", self)
        self._queue_dock.setObjectName("analysis_queue_dock")
        self._queue_dock.setWidget(self._queue_panel)
        self._queue_dock.setAllowedAreas(QtCore.Qt.RightDockWidgetArea | QtCore.Qt.BottomDockWidgetArea)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._queue_dock)
        self._queue_dock.hide()

        self.jobs.changed.connect(self._update_queue_badge)
        self._update_queue_badge()

    def _toggle_queue_panel(self):
        self._queue_dock.setVisible(not self._queue_dock.isVisible())
        if self._queue_dock.isVisible():
            self._queue_dock.raise_()

    def _update_queue_badge(self):
        n = self.jobs.pending_count()
        running = self.jobs.running is not None
        self.b_queue.setText(f"Queue ({n})" if n else "Queue")
        bits = []
        if running:
            bits.append("1 running")
        if n:
            bits.append(f"{n} queued")
        self.b_queue.setToolTip(", ".join(bits) if bits else "No analyses queued")

    def _sync_tab_overflow(self, *_):
        """Show the » overflow button only when the tab strip is wider than its viewport
        (some tabs scrolled off the right). Measured from the live widths rather than the
        scrollbar's range, which can lag a widen-resize and leave » stuck on when every
        tab already fits."""
        act = getattr(self, "_tab_overflow_action", None)
        strip = getattr(self, "tab_strip", None)
        scroll = getattr(self, "tab_scroll", None)
        if act is None or strip is None or scroll is None:
            return
        act.setVisible(strip.sizeHint().width() > scroll.viewport().width() + 1)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        # the tab strip's viewport tracks the window width — re-evaluate overflow after the
        # layout settles (deferred so viewport().width() reflects the new size).
        if hasattr(self, "tab_overflow"):
            QtCore.QTimer.singleShot(0, self._sync_tab_overflow)

    def _populate_tab_overflow(self):
        """Fill the » menu on-demand with the tabs that aren't fully visible right now —
        computed from the live scroll position, so there's no resize-time fit math to drift."""
        menu = self.tab_overflow.menu()
        menu.clear()
        bar = self.tab_strip
        hbar = self.tab_scroll.horizontalScrollBar()
        x0 = hbar.value()
        x1 = x0 + self.tab_scroll.viewport().width()
        cur = bar.currentIndex()
        for i in range(bar.count()):
            r = bar.tabRect(i)
            if r.left() >= x0 and r.right() <= x1:
                continue                                  # already on screen
            act = menu.addAction(bar.tabText(i))          # same &-mnemonic rules as the tab
            act.setCheckable(True)
            act.setChecked(i == cur)
            act.triggered.connect(lambda _=False, idx=i: self._goto_tab(idx))
        if menu.isEmpty():
            menu.addAction("(all views visible)").setEnabled(False)

    def _goto_tab(self, idx):
        """Switch to a tab chosen from the » menu. The strip stays anchored left (so
        'Ion image' is never pushed off), so we only switch the page — the » menu's
        checkmark marks the active view when it's one of the off-screen (right) tabs."""
        self.tabs.setCurrentIndex(idx)

    # ------------------------------------------------------------------ #
    # Information architecture — collapse the flat 17-tab strip into ~9
    # grouped top-level tabs. The everyday workflow stays top-level; the
    # near-duplicate / niche analyses nest under one container each:
    #   Compare regions ← Discriminating&ROI stats + Region comparison
    #   Cohort          ← Cohort + Cohort UMAP + Cohort segmentation
    #   Explore         ← Feature space + Components
    #   Advanced        ← Classify + Markers (SSC) + Per-ion (DGMM)
    #   More            ← Montage + Feature lists
    # NOTHING is removed — every original view is still present, just one
    # click deeper for the nested ones. Runs BEFORE attach_strip, so the
    # inserts/removes here never have to sync the external strip (it mirrors
    # the final layout). reveal_view()/_set_view_title() then address a view
    # by its ORIGINAL label, so callers don't track tab indices this
    # regrouping invalidates.
    # ------------------------------------------------------------------ #
    def _regroup_tabs(self):
        def has(*subs):
            return lambda L: any(s in L for s in subs)
        # Top-level strip order reads left→right as the workflow: view the slide → segment
        # it → explore its structure → compare regions → relate ions (co-loc) → step up to
        # the cohort (cross-sample) → niche/advanced → utilities → the report. Cohort sits
        # after the single-slide analyses (it's the multi-sample step); Explore sits next to
        # Segmentation (structure-finding). Matching is by label, so this order is the single
        # place the strip sequence is defined.
        SLOTS = [
            ("Ion image",       lambda L: L == "Ion image"),
            ("Segmentation",    lambda L: L == "Segmentation"),
            ("Analyze",         lambda L: L == "Analyze"),        # exact match; cannot collide
            # Montage + Feature lists left the strip (plan 24 Phase 6), emptying the old "More"
            # bucket — that slot is gone. "Analyses" still nests History + Report book here.
            ("Analyses",        has("History", "Report book", "Analyses", "Report")),
        ]
        # snapshot current (label, widget) in build order, route each into the first slot it
        # matches; any view that matches nothing keeps its own top-level slot (never lost).
        pages = [(self.tabs.tabText(i), self.tabs.widget(i)) for i in range(self.tabs.count())]
        buckets = {name: [] for name, _ in SLOTS}
        order = [name for name, _ in SLOTS]
        for label, w in pages:
            for name, match in SLOTS:
                if match(label):
                    buckets[name].append((label, w))
                    break
            else:
                buckets.setdefault(label, []).append((label, w))
                order.append(label)
        while self.tabs.count():                                     # detach (does not delete)
            self.tabs.removeTab(0)
        self._view_loc = {}                                          # original label → (outer, inner|None, inner_idx)
        self._inner_tabwidgets = []
        for name in order:
            members = buckets.get(name) or []
            if not members:
                continue
            if len(members) == 1:                                    # don't nest a lone view
                label, w = members[0]
                outer = self.tabs.addTab(w, label)
                self._view_loc[label] = (outer, None, 0)
                self._view_loc.setdefault(name, (outer, None, 0))
                continue
            inner = QtWidgets.QTabWidget()
            inner.setDocumentMode(True)
            for label, w in members:
                inner.addTab(w, label)
            outer = self.tabs.addTab(inner, name)
            self._inner_tabwidgets.append(inner)
            for inner_idx, (label, _w) in enumerate(members):
                self._view_loc[label] = (outer, inner, inner_idx)
            self._view_loc.setdefault(name, (outer, inner, 0))
            # Every on-show refresher (active-ion syncs, Feature-space rebuild, cohort-seg
            # redraw, …) is connected to the OUTER currentChanged. Switching an INNER sub-tab
            # must re-run them too, so re-emit the outer signal — each handler is page/label
            # gated, so only the now-visible view actually refreshes.
            inner.currentChanged.connect(
                lambda *_: self.tabs.currentChanged.emit(self.tabs.currentIndex()))

    _COHORT_SCREENS = ("Cohort", "Cohort nested stats", "Cohort UMAP", "Cohort segmentation")

    def _show_cohort_screen(self, label=None):
        """Open the on-demand Cohort window (the four cohort screens live here, not the strip —
        plan 24 Phase 5) and select the requested screen. Lazily wraps ``_cohort_tabs`` in a
        modeless dialog the first time; subsequent reveals just raise + reselect it."""
        tabs = getattr(self, "_cohort_tabs", None)
        if tabs is None:
            return
        win = getattr(self, "_cohort_window", None)
        if win is None:
            win = QtWidgets.QDialog(self)
            win.setWindowTitle("Cohort")
            win.setModal(False)
            lay = QtWidgets.QVBoxLayout(win)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(tabs)
            try:
                geo = QtGui.QGuiApplication.primaryScreen().availableGeometry()
                win.resize(int(geo.width() * 0.82), int(geo.height() * 0.85))
            except Exception:  # noqa: BLE001
                win.resize(1200, 800)
            self._cohort_window = win
        if label:
            for i in range(tabs.count()):
                if tabs.tabText(i) == label:
                    tabs.setCurrentIndex(i)
                    break
        win.show()
        win.raise_()
        win.activateWindow()

    def open_analysis(self, dlg):
        """Dock an :class:`~smile_msi.gui.analysisdialog.AnalysisDialog` as a **closable tab**
        in the strip and switch to it (plan 24). Several analyses can sit open side by side;
        closing a tab keeps the run in Analyses ▸ History. Returns the dialog."""
        title = dlg.sd.name if getattr(dlg, "sd", None) is not None else "Analysis"
        idx = self.tabs.addTab(dlg, title)
        btn = QtWidgets.QToolButton()
        btn.setText("✕")
        btn.setAutoRaise(True)
        btn.setToolTip("Close this analysis (it stays in Analyses ▸ History)")
        btn.clicked.connect(lambda _=False, d=dlg: self._close_analysis(d))
        try:
            self.tab_strip.setTabButton(idx, QtWidgets.QTabBar.RightSide, btn)
        except Exception:  # noqa: BLE001 — a strip without per-tab buttons still works
            pass
        self.tabs.setCurrentIndex(idx)
        return dlg

    def _close_analysis(self, dlg):
        """Close an analysis tab: detach it, tear down its signal-bus subscriptions, and land
        back on the Analyze gallery. The run itself persists in History."""
        idx = self.tabs.indexOf(dlg)
        if idx >= 0:
            self.tabs.removeTab(idx)
        try:
            dlg._teardown()
        except Exception:  # noqa: BLE001
            pass
        dlg.deleteLater()
        self.reveal_view("Analyze")

    def reveal_view(self, label):
        """Switch to a view by its ORIGINAL tab label, selecting the container tab *and* its
        inner sub-tab when _regroup_tabs nested it. Falls back to a top-level name search if
        the registry isn't built. Use this instead of a stored tab index."""
        if label in self._COHORT_SCREENS:                   # cohort screens live in their own window
            self._show_cohort_screen(label)
            return
        if label == "Feature lists":                        # demoted off the strip (Phase 6)
            self._open_feature_lists_manager()
            return
        loc = getattr(self, "_view_loc", {}).get(label)
        if loc is None:
            idx = self._tab_index(label) if hasattr(self, "_tab_index") else -1
            if idx >= 0:
                self.tabs.setCurrentIndex(idx)
            return
        outer, inner, inner_idx = loc
        self.tabs.setCurrentIndex(outer)
        if inner is not None:
            inner.setCurrentIndex(inner_idx)

    def _page_on_screen(self, page):
        """True if ``page`` (an original tab's content widget) is the one currently visible —
        directly as a top-level tab, or as the selected sub-tab of the container QTabWidget
        that _regroup_tabs nested it into. Lets visibility-gated syncs (e.g. the comparison /
        ROC active-ion followers) keep working after the strip was regrouped."""
        if page is None:
            return False
        # a cohort screen is "on screen" when its window is visible and it's the current
        # cohort tab (the four cohort views moved off the strip into a window, plan 24 Phase 5).
        ctabs = getattr(self, "_cohort_tabs", None)
        cwin = getattr(self, "_cohort_window", None)
        if ctabs is not None and cwin is not None and cwin.isVisible() and ctabs.currentWidget() is page:
            return True
        cur = self.tabs.currentWidget()
        if cur is page:
            return True
        return isinstance(cur, QtWidgets.QTabWidget) and cur.currentWidget() is page

    def _view_visible(self, label):
        """True if the view with this ORIGINAL label is the one currently on-screen — its
        container tab is selected and (when nested) its inner sub-tab is selected. The
        label-based twin of :meth:`_page_on_screen`, for on-show refreshers that key off a
        view name (Feature space, Cohort segmentation, …)."""
        loc = getattr(self, "_view_loc", {}).get(label)
        if loc is None:
            idx = self._tab_index(label) if hasattr(self, "_tab_index") else -1
            return idx >= 0 and self.tabs.currentIndex() == idx
        outer, inner, inner_idx = loc
        if self.tabs.currentIndex() != outer:
            return False
        return inner is None or inner.currentIndex() == inner_idx

    def _set_view_title(self, label, text):
        """Set a view's tab title by its ORIGINAL label — on the inner QTabWidget when the
        view was nested, else on the outer strip."""
        loc = getattr(self, "_view_loc", {}).get(label)
        if loc is None:
            idx = self._tab_index(label) if hasattr(self, "_tab_index") else -1
            if idx >= 0:
                self.tabs.setTabText(idx, text)
            return
        outer, inner, inner_idx = loc
        if inner is not None:
            inner.setTabText(inner_idx, text)
        else:
            self.tabs.setTabText(outer, text)

    # ----- persistent library tab ----------------------------------------- #
    def _show_timeline(self):
        """Help ▸ Show timeline… — surface, in-app, what startup and each subsequent
        load / preprocessing pass spent time on (the events _perf writes to
        smile_msi_perf.log), so a slow run is diagnosable without opening a log file."""
        from .timeline import TimelineDialog, read_perf_log
        clock = getattr(self, "_startup_clock", None)
        rows = clock.rows() if clock is not None else []
        total = clock.total_ms() if clock is not None else 0.0
        dlg = TimelineDialog(self, startup_rows=rows, startup_total_ms=total,
                             operations=read_perf_log())
        dlg.exec()

    def _perf(self, msg):
        """Append a timestamped perf line to stderr + ~/smile_msi_perf.log (overridable via
        $SMILE_MSI_HOME). Best-effort timing so slow operations can be diagnosed from a real
        run — open the log after reproducing to see exactly what took how long."""
        try:
            import os, sys, time
            line = f"[{time.strftime('%H:%M:%S')}] {msg}"
            print(line, file=sys.stderr, flush=True)
            home = os.environ.get("SMILE_MSI_HOME") or os.path.expanduser("~")
            with open(os.path.join(home, "smile_msi_perf.log"), "a") as fh:
                fh.write(line + "\n")
        except Exception:  # noqa: BLE001 — diagnostics must never break the app
            pass

    def _run(self, fn, *args, on_done=None, want_progress=False, busy="Working…",
             modal=False, title="Loading sample…", queue=True, label=None,
             run_id=None, on_cancelled=None, on_error=None,
             want_stage=False, stages=None, **kwargs):
        """Dispatch a background analysis.

        By default the job is funnelled through :attr:`jobs` so heavy analyses run **one at
        a time** instead of piling onto the thread pool and starving each other. Modal work
        (dataset loads — they block the UI anyway and are prerequisites for everything else)
        and explicit ``queue=False`` callers run immediately, preserving the old behaviour.

        ``run_id`` ties the queue row to a plan-24 :class:`~smile_msi.runs.AnalysisRun` history
        record. ``on_cancelled`` / ``on_error`` are optional per-call terminal hooks (in
        addition to the global cancel/error handling) so a caller — the AnalysisDialog — can
        finalise its own run record on *every* terminal path, not only success.

        v1 limitation: a modal load started while a queued analysis is still running will run
        concurrently with it. In practice the UI is blocked during a modal load, so the user
        can't trigger one mid-analysis through the normal flow."""
        if modal or not queue:
            return self._start_run(fn, *args, on_done=on_done, want_progress=want_progress,
                                   busy=busy, modal=modal, title=title,
                                   on_cancelled=on_cancelled, on_error=on_error,
                                   want_stage=want_stage, stages=stages, **kwargs)
        from .jobqueue import Job
        job = Job(label or busy, cancellable=want_progress, run_id=run_id)
        job.starter = lambda j: self._start_run(
            fn, *args, on_done=on_done, want_progress=want_progress, busy=busy,
            modal=False, title=title, _job=j,
            on_cancelled=on_cancelled, on_error=on_error,
            want_stage=want_stage, stages=stages, **kwargs)
        self.jobs.submit(job)

    def _start_run(self, fn, *args, on_done=None, want_progress=False, busy="Working…",
                   modal=False, title="Loading sample…", _job=None,
                   on_cancelled=None, on_error=None, want_stage=False, stages=None, **kwargs):
        import time
        self.statusBar().showMessage(busy)
        self.progress.setValue(0)
        self._cancel = False
        self.b_cancel.setEnabled(want_progress)
        # modal load bar: a blocking window over the workspace while the sample
        # streams in + the fast cache builds (status-bar progress alone left the
        # half-loaded window clickable). Closed by every terminal handler below.
        self._close_load_dialog()                     # never stack two
        if modal:
            from .loading import LoadingWindow
            dlg = LoadingWindow(self, message=busy, cancelable=want_progress, title=title,
                                stages=stages)
            if want_progress:
                dlg.on_cancel(self._cancel_work)
            self._load_dialog = dlg
            dlg.show()
            dlg.raise_()
            # Paint it now: the worker starts on the next line, and without an
            # explicit flush the frameless card can appear blank or arrive a beat
            # late (it only repaints once the event loop next idles).
            QtWidgets.QApplication.processEvents()
        t0 = time.perf_counter()
        self._perf(f"START  {busy}")
        wk = Worker(fn, *args, want_progress=want_progress, want_stage=want_stage,
                    cancel_check=(lambda: self._cancel) if (want_progress or want_stage) else None,
                    **kwargs)
        self._active_workers.add(wk)                  # keep alive until a terminal signal
        if want_progress:
            wk.signals.progress.connect(lambda d, t: self.progress.setValue(int(100 * d / max(t, 1))))
            if modal:
                wk.signals.progress.connect(lambda d, t, _d=dlg: _d.set_progress(d, t))
        if want_stage:
            # Narrate the current sub-step: the modal card's stage list / detail line, or the
            # status bar when non-modal; and record it on the session timeline either way.
            if modal:
                wk.signals.stage.connect(lambda name, _d=dlg: _d.set_stage(name))
            else:
                wk.signals.stage.connect(lambda name: self.statusBar().showMessage(name))
            wk.signals.stage.connect(lambda name, _b=busy: self._perf(f"STAGE  {_b} :: {name}"))
        wk.signals.error.connect(self._on_error)
        wk.signals.cancelled.connect(self._on_cancelled)
        # per-call terminal hooks (plan 24: let a caller finalise its AnalysisRun record on the
        # cancelled / error paths too). Guarded so a buggy hook can't crash the event loop.
        if on_cancelled is not None:
            wk.signals.cancelled.connect(lambda *_: self._safe_hook(on_cancelled))
        if on_error is not None:
            wk.signals.error.connect(lambda msg=" ", _h=on_error: self._safe_hook(_h, str(msg)))
        wk.signals.finished.connect(lambda *_: self._perf(f"DONE   {busy}  ({time.perf_counter()-t0:.2f}s)"))
        wk.signals.error.connect(lambda *_: self._perf(f"ERROR  {busy}  ({time.perf_counter()-t0:.2f}s)"))

        def done(result):
            self.progress.setValue(100)
            self.b_cancel.setEnabled(False)
            self._close_load_dialog()
            if on_done:
                # on_done runs on the GUI thread; a bug in it would otherwise be an
                # unhandled event-loop exception (crash). Surface it like a worker error.
                try:
                    on_done(result)
                except Exception as e:
                    import traceback
                    self._on_error(f"{e}", details=traceback.format_exc())
        wk.signals.finished.connect(done)
        # drop the strong ref on the main thread (any terminal outcome)
        for sig in (wk.signals.finished, wk.signals.error, wk.signals.cancelled):
            sig.connect(lambda *_: self._active_workers.discard(wk))
        # Report back to the queue so the panel updates and the next job auto-starts.
        if _job is not None:
            from .jobqueue import JobStatus
            self._running_job = _job
            if want_progress:
                wk.signals.progress.connect(
                    lambda d, t, _j=_job: self.jobs.on_progress(_j, d, t))
            wk.signals.finished.connect(
                lambda *_, _j=_job: self.jobs.on_terminal(_j, JobStatus.DONE))
            wk.signals.cancelled.connect(
                lambda *_, _j=_job: self.jobs.on_terminal(_j, JobStatus.CANCELLED))
            wk.signals.error.connect(
                lambda msg, _j=_job: self.jobs.on_terminal(_j, JobStatus.FAILED, str(msg).split("\n", 1)[0]))
        self.pool.start(wk)

    def _close_load_dialog(self):
        """Tear down the modal load window (if any). Called from every terminal
        outcome so the workspace is never left blocked behind a stale loader."""
        dlg = getattr(self, "_load_dialog", None)
        if dlg is not None:
            self._load_dialog = None
            dlg.close()
            dlg.deleteLater()

    @staticmethod
    def _safe_hook(fn, *a):
        """Run a caller-supplied terminal hook on the GUI thread, swallowing (but logging) any
        error so a buggy hook can never turn into an unhandled event-loop exception."""
        try:
            fn(*a)
        except Exception:  # noqa: BLE001 — a terminal hook must never crash the app
            import traceback
            traceback.print_exc()

    # ----- analysis-run history store (plan 24) ---------------------------- #
    def _ensure_session_path(self):
        """The managed session path for the loaded sample, resolving it lazily the way
        :meth:`_flush_autosave` does. Returns ``None`` for the throwaway demo or when no
        dataset is loaded — callers then simply skip persistence."""
        if self.ds is None or getattr(self.ds, "source", "") == "synthetic":
            return None
        if self._session_path is None:
            try:
                self._session_path = session.resolve_session_path(
                    self.ds.source, library.dataset_fingerprint(self.ds), self.ds.n_pixels)
            except Exception:  # noqa: BLE001 — no managed path ⇒ no persistence, never fatal
                return None
        return self._session_path

    @property
    def run_store(self):
        """The analysis-run payload store for the loaded sample, rooted at ``<session>.runs/``
        beside the managed session file (the :func:`session.cube_zarr_path` sidecar idiom).

        ``None`` when there is no persistable session yet (the demo, or no dataset) — the
        AnalysisDialog then runs without recording, exactly as before plan 24."""
        path = self._ensure_session_path()
        if path is None:
            return None
        from .. import runs as runs_mod
        root = session.runs_dir_path(path)
        st = getattr(self, "_run_store", None)
        if st is None or st.root != root:
            self._run_store = runs_mod.RunStore(root)
        return self._run_store

    @property
    def cohort_run_store(self):
        """The analysis-run payload store for the loaded **cohort**, rooted at
        ``<cohorts_dir>/<cohort name>.runs/`` — a sibling of the cohort's own JSON.

        A cohort analysis (nested stats, group comparison, …) spans every sample in the
        cohort, not the one slide that happens to be open, so it cannot live in
        :attr:`run_store` — that store is keyed to the loaded sample's session file and
        would reassign or vanish across a slide switch. This store instead follows the
        cohort by name, independent of which sample is currently loaded."""
        from .. import cohort as cohort_mod
        from .. import runs as runs_mod
        name = getattr(getattr(self, "cohort", None), "name", "") or "Workspace"
        root = os.path.join(cohort_mod.cohorts_dir(), f"{session._sanitize(name)}.runs")
        st = getattr(self, "_cohort_run_store", None)
        if st is None or st.root != root:
            self._cohort_run_store = runs_mod.RunStore(root)
        return self._cohort_run_store

    @contextmanager
    def _busy_popup(self, message, title="Loading…"):
        """Application-modal loading popup over the workspace for a **synchronous** heavy
        step (no worker thread), so stray clicks can't interleave with it. Reuses the
        dataset-load :class:`LoadingWindow`; painted once before the blocking work and torn
        down on exit. Don't wrap work that dispatches its own ``_run(modal=True)`` — that
        opens its own loader and the exit here would close it early; pass ``modal=True`` to
        that call instead."""
        from .loading import LoadingWindow
        self._close_load_dialog()                     # never stack two
        dlg = LoadingWindow(self, message=message, cancelable=False, title=title)
        self._load_dialog = dlg
        dlg.show()
        dlg.raise_()
        QtWidgets.QApplication.processEvents()        # paint the popup before we block the thread
        try:
            yield
        finally:
            self._close_load_dialog()

    def _cancel_work(self):
        self._cancel = True
        self.statusBar().showMessage("Cancelling…")

    def _cancel_job(self, job):
        """Cancel one job from the queue panel: drop it if still queued, else (if it is the
        running, cancellable job) trigger the cooperative cancel the status-bar button uses."""
        if self.jobs.cancel_pending(job):
            return
        if job is self.jobs.running and job.cancellable:
            self._cancel_work()

    def _on_cancelled(self):
        self._close_load_dialog()
        self.b_cancel.setEnabled(False)
        self.progress.setValue(0)
        self.statusBar().showMessage("Cancelled.")

    def _on_error(self, msg, details=None):
        self._close_load_dialog()
        self.statusBar().showMessage("Error.")
        self.b_cancel.setEnabled(False)
        # Workers emit "<message>\n\n<traceback>"; show the sentence and tuck the stack
        # behind "Show details" so a non-developer (a PI clicking around) sees a readable
        # message, not a wall of Python traceback.
        headline, _, rest = str(msg).partition("\n\n")
        if details is None and rest:
            details = rest
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Critical)
        box.setWindowTitle("Error")
        box.setText(headline.strip() or "Something went wrong.")
        if details and str(details).strip():
            box.setDetailedText(str(details).strip())
        box.exec()

    def _rebuild_annotator(self):
        """Ensure the on-the-fly annotator matches the current mode + lipid DB — it drives the
        peak-table lipid labels and class map. The annotator's index is a pure function of
        (mode, DB); the identification tolerance is applied per-lookup, not baked into the index.
        So when only the tolerance changed we retune the existing annotator in place (drops its
        m/z memo, O(1)) instead of re-enumerating the whole DB. This is what makes the annotator
        "fire once" — it is reconstructed only on a real mode/DB switch, not on every ID-tolerance
        nudge, feature-list interaction, or session-restore step (a full rebuild is ~40-120 ms on
        a large merged LIPID MAPS DB and was previously running on the UI thread per spinbox tick)."""
        mode = self.mode_combo.currentText()
        db = getattr(self, "_lipid_db", None)
        ann = getattr(self, "ann", None)
        if ann is not None and ann.matches(mode, db):
            ann.set_ppm_tol(self.id_ppm)
            return
        self.ann = Annotator(mode=mode, ppm_tol=self.id_ppm, db=db)

    def _mode_changed(self, mode):
        self._rebuild_annotator()
        if self.peaks:
            self._populate_peak_table()
        self._mark_dirty()                   # polarity is part of saved settings

    def _id_ppm_changed(self, *_):
        self._rebuild_annotator()
        if self.peaks:
            self._populate_peak_table()      # refresh lipid labels at the new tolerance
        self._mark_dirty()

    def _invalidate_features(self, *_):
        if self.ds is not None:
            self.ds._feat = None
        self._mark_dirty()                   # extraction tolerance / reduce changed

    # ----- dataset load ---------------------------------------------------- #
    def open_imzml(self):
        path, _ = filedialogs.get_open_file_name(self, "Open imzML", "", "imzML (*.imzML)")
        if path:
            self._load_path(path)

    @staticmethod
    def _build_cache_inplace(ds, progress=None):
        """Build the fast m/z cube on ``ds`` *up front* (called inside the load worker, so
        it runs under the load progress bar, off the GUI thread). Disk-backed datasets get
        the cube here so every in-app interaction afterward — ROI / region spectra,
        arbitrary-m/z ion images — is served from RAM and is instant. In-RAM stores (demo /
        imaging-table CSV) are already instant, so they're skipped."""
        if ds is None or getattr(ds, "_cube", None) is not None:
            return
        if getattr(getattr(ds, "store", None), "in_memory", False):
            return
        _, spec = ds.mean_spectrum()
        ds.build_mz_cube(min_intensity=float(spec.max()) * 0.001, progress=progress)

    def _load_path(self, path):
        stride = int(self.subsample_spin.value())

        def load(progress=None, stage=None):
            # read into RAM (if it fits), prime, and build the fast cube once, during load
            return self._prepare_imzml(path, progress=progress, stride=stride, stage=stage)
        self._run(load, on_done=self._on_dataset_opened, want_progress=True,
                  want_stage=True, stages=["Reading spectra into memory", "Priming spectra"],
                  modal=True, busy=f"Loading {path} + building fast cache…")

    def _load_table_path(self, path):
        def load(progress=None):
            from .. import ingest
            ds = ingest.read_imaging_csv(path)
            ds.prime(progress=progress)
            return ds
        self._run(load, on_done=self._on_dataset_opened, want_progress=True, modal=True,
                  busy=f"Loading {path}…")

    def _on_dataset_opened(self, ds):
        """Completion handler for a *fresh* file open (File ▸ Open, the startup launcher,
        or the cohort 'Open files…' button). Initialize the dataset, then add it to the
        cohort roster right away so it shows in the Samples panel immediately — before any
        analysis. Registration used to wait for the first auto-save (which only fires once
        you Find peaks / edit), so a just-opened slide stayed invisible in the roster.

        Plain :meth:`_on_dataset` stays the shared reset for the demo (synthetic — skipped
        by the roster), session restore (which registers after the regions are applied) and
        wipe-and-restart (already on the roster), none of which should join here."""
        # If this slide already has an auto-saved session (ROIs, feature lists, segmentation…),
        # restore it instead of starting blank — otherwise the first edit auto-saves an empty
        # analysis over the real one. A roster switch already restores via _open_session_path;
        # this covers File ▸ Open / the cohort 'Open files…' button, which used to skip it.
        if self._restore_managed_session(ds):
            return
        self._on_dataset(ds)
        self._register_active_in_cohort()

    def _restore_managed_session(self, ds) -> bool:
        """Load and apply this slide's existing managed session onto a freshly-opened ``ds``
        (restoring its saved ROIs/analysis), returning True when one was found and applied.
        No-ops (returns False) for the demo, a source-less dataset, or when no managed session
        exists for the slide — the caller then falls back to a blank :meth:`_on_dataset`."""
        if ds is None or not getattr(ds, "source", "") or ds.source == "synthetic":
            return False
        try:
            sp = session.resolve_session_path(
                ds.source, library.dataset_fingerprint(ds), ds.n_pixels)
            if not sp or not os.path.exists(sp):
                return False
            self._pending_session = session.load_session(sp)
        except Exception:  # noqa: BLE001 — a bad/locked file must never block opening the slide
            import traceback
            traceback.print_exc()
            return False
        self._pending_region_sample = None
        self._apply_session(ds)          # _on_dataset reset + restore, guarded by _restoring
        self._session_path = sp          # keep writing back to the file we restored from
        return True

    def _load_dataset_path(self, path):
        """Programmatic dataset open (used by the startup launcher): dispatch imzML vs
        imaging-table by extension."""
        if str(path).lower().endswith(".imzml"):
            self._load_path(path)
        else:
            self._load_table_path(path)

    def open_table(self):
        path, _ = filedialogs.get_open_file_name(
            self, "Open imaging table", "", "Tables (*.csv *.tsv *.txt)")
        if path:
            self._load_table_path(path)

    def export_imzml(self):
        if self.ds is None:
            self.statusBar().showMessage("Load a dataset first.")
            return
        path, _ = filedialogs.get_save_file_name(self, "Export to imzML", "dataset.imzML",
                                                        "imzML (*.imzML)")
        if not path:
            return
        meta = getattr(self, "_acquisition_meta", None) or None
        if meta:
            from .. import standards
            am = standards.AcquisitionMeta.from_dict(meta)
            self._run(lambda: standards.write_imzml_with_metadata(self.ds, path, am,
                                                                  getattr(self, "prov", None)),
                      on_done=lambda _: self.statusBar().showMessage(
                          f"Wrote {path} (+ .ibd) with reporting metadata"),
                      busy="Converting to imzML…")
        else:
            from .. import ingest
            self._run(lambda: ingest.to_imzml(self.ds, path), on_done=lambda _: self.statusBar()
                      .showMessage(f"Wrote {path} (+ .ibd)"), busy="Converting to imzML…")

    def _open_reporting_dialog(self):
        """File ▸ Acquisition metadata & reporting… — edit/validate the standards metadata."""
        from .reportingdialog import ReportingDialog
        dlg = getattr(self, "_reporting_dialog", None)
        if dlg is None:
            dlg = self._reporting_dialog = ReportingDialog(self)
        dlg.load_from_main()
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _open_quantify_dialog(self):
        """Data ▸ Quantification (calibration)… — on-tissue absolute quantification."""
        from .quantify import QuantifyDialog
        dlg = getattr(self, "_quantify_dialog", None)
        if dlg is None:
            dlg = self._quantify_dialog = QuantifyDialog(self)
        dlg.load_from_main()
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _open_single_cell_dialog(self):
        """Data ▸ Single-cell profiling… — per-cell metabolite profiles (plan 07)."""
        from .singlecell import SingleCellDialog
        dlg = getattr(self, "_single_cell_dialog", None)
        if dlg is None:
            dlg = self._single_cell_dialog = SingleCellDialog(self)
        dlg.load_from_main()
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _open_stack3d_dialog(self):
        """Data ▸ 3D reconstruction… — serial-section molecular volume (plan 09)."""
        from .stack3d import Stack3DDialog
        dlg = getattr(self, "_stack3d_dialog", None)
        if dlg is None:
            dlg = self._stack3d_dialog = Stack3DDialog(self)
        dlg.load_from_main()
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _open_comap_dialog(self):
        """Data ▸ Spatial multi-omics… — co-map a second modality (plan 10)."""
        from .comapdialog import CoMapDialog
        dlg = getattr(self, "_comap_dialog", None)
        if dlg is None:
            dlg = self._comap_dialog = CoMapDialog(self)
        dlg.load_from_main()
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def do_load_demo(self):
        def loaded(ds):
            self._on_dataset(ds)
            # The demo is meaningless as a blank TIC — auto-find peaks the moment it loads so
            # the feature list, ion images, segmentation and stats are all immediately live.
            # A first-time viewer (or a PI being shown the app) shouldn't have to know to click
            # 'Find peaks' before anything interesting appears. Demo path only: session restore
            # calls _on_dataset directly and must keep its saved peaks.
            self.do_pick_peaks()
        self._run(load_demo, on_done=loaded, want_progress=True,
                  modal=True, busy="Synthesizing demo dataset…")

    def _on_dataset(self, ds):
        self.ds = ds
        self.peaks = []
        self.seg = None
        self.regions = []
        # a fresh/other slide starts with no regions on screen — clear the list, combos and
        # on-image overlay so a previous sample's regions can't linger. _apply_session refreshes
        # again after restoring; the plain-open and wipe paths never reach it, so it must run
        # here too. Best-effort: the region views may not be built yet on the very first load.
        for _fn in (self._refresh_region_list, self._sync_region_combos, self._hide_region_overlay):
            common.guarded(_fn)
        self.report_items = []           # report contents are per-sample (session re-applies)
        self._studio_plan = None         # Export Studio plan is per-sample too
        self.active_mz = None
        self._displayed_mz = None         # drop the backdrop m/z so a new slide starts blank
        self._iv_img_shape = None         # new grid → let the first ion image fit the view
        self._exp_mean_spec = None        # invalidate the export mean-spectrum cache
        self._feature_scopes = {}
        self._active_feature_scope = None
        self._feature_lists = {}         # saved lists are per-sample now → start empty
        self._lipid_lists = {}           # lipid lists are per-sample too
        self._active_lipid_list = None
        self._session_path = None        # managed file is (re)assigned lazily for this sample
        self._session_restored = False   # True once _apply_session restores this slide's saved
                                         # analysis; gates the auto-save's blank-ROI guard so an
                                         # un-restored empty region set can't overwrite the file
        self._cube_saved = False         # cube sidecar is written once per dataset
        self._dirty = False
        self._undo_stack = []            # drop snapshots that point at the old dataset
        self._clear_optical()            # the optical backdrop is per-sample (session re-applies it)
        self._build_pixel_lookup()
        # soft pre-fill: a fresh sample inherits the active profile's standardized defaults
        # (a restored session overwrites these afterwards in _apply_session).
        self._prefill_from_active_profile()
        self.prov = provenance.Provenance(title=f"SMILE MSI — {os.path.basename(ds.source)}")
        self.prov.set_dataset(ds, self.mode_combo.currentText())
        try:
            if ds.source and ds.source != "synthetic" and os.path.exists(ds.source):
                self.prov.set_input(ds.source)
        except Exception:  # noqa: BLE001
            pass
        # stamp the active analysis profile + resolved random seed into the audit trail, so
        # the persisted provenance records the standardized method and the seed that make
        # results reproducible (ties the profiles feature into the data trail).
        try:
            act = profiles.active()
            self.prov.step("analysis_profile", profile=act.get("name"),
                           version=int(act.get("version", 1)), profile_hash=profiles.content_hash(act),
                           random_seed=profiles.active_seed())
        except Exception:  # noqa: BLE001 — provenance stamping is best-effort
            pass
        # inspect the just-loaded data (centroid/profile + already-normalized?) and surface
        # matching, overridable suggestions — never block a load on a detection hiccup.
        report = None
        try:
            from .. import intake
            report = intake.inspect_dataset(ds)
        except Exception:  # noqa: BLE001
            report = None
        self.info.setText(self._dataset_banner_html(ds, report))
        self._show_intake_suggestion(report)
        if ds.polarity in ("positive", "negative"):
            self.mode_combo.setCurrentText(ds.polarity)
        self._plot_mean_spectrum()
        # show the TIC image right away so the tissue is visible before peaks are picked
        self.iv.setImage(imaging.quantile_clip(ds.tic_image(), high=99), autoLevels=True)
        self.iv.setColorMap(colormap(self.cmap_combo.currentText()))
        self._set_ion_tab_caption("TIC (total ion current)")
        self._refresh_library_tab()              # reset the (now per-sample) feature-list tab
        self._refresh_report_list()              # clear the report list for the new sample
        self.statusBar().showMessage("Loaded — showing TIC. Click 'Find peaks', then click a peak "
                                     "(table or spectrum) to view its ion image.")
        self._autobuild_cache()                  # warm the fast cube so ROI/region views never stream
        self._refresh_action_states()
        self.datasetChanged.emit()

    def wipe_analysis(self):
        """Reset the analysis to a freshly-loaded state — clears the report, all regions, the
        segmentation, every statistics / comparison result and the picked features (and the
        methods record), back to a just-loaded dataset. The dataset stays loaded, and the
        optical overlay + saved ★ feature lists are preserved. Invoked from the Report tab's
        'Wipe & restart analysis'."""
        if self.ds is None:
            self.statusBar().showMessage("Load a dataset first.")
            return
        if not confirm(self, "Wipe & restart analysis", "Start the analysis over?\n\nThis clears the report, all regions, the segmentation, every statistics / comparison result and the picked features, back to a freshly-loaded dataset. The dataset stays loaded, and your optical overlay and saved ★ feature lists are kept.\n\n(This can't be undone with ⌘Z.)", ok_text="Wipe & restart"):
            return
        # preserve the cross-cutting context that isn't part of "the analysis": this sample's
        # session file (so the restart overwrites it, not a new one), the optical overlay and
        # the saved ★ feature lists.
        session_path = self._session_path
        optical = self._optical_session_dict()
        saved_lists = dict(getattr(self, "_feature_lists", {}) or {})
        self._on_dataset(self.ds)                 # known-good "freshly loaded" reset (cube is
                                                  # already built, so _autobuild_cache no-ops)
        self._reset_stats_views()                 # _on_dataset doesn't touch the stats caches
        self._session_path = session_path
        self._feature_lists = saved_lists
        if optical:
            self._apply_optical_session({"optical": optical})
        self._refresh_library_tab()               # re-show the restored saved ★ lists
        self._refresh_scope_bars()
        self._mark_dirty()                         # the restart differs from the saved session
        self.statusBar().showMessage("Analysis wiped — back to the freshly-loaded dataset.")

    def _preprocess_config(self) -> dict:
        """Build a preprocess.build_pipeline config from the Preprocessing dialog widgets."""
        import re as _re
        cfg = {}
        bl = self.pp_baseline_combo.currentText()
        param = int(self.pp_baseline_param.value())
        if bl == "SNIP":
            cfg["baseline"] = param                                    # int → SNIP (iterations)
        elif bl == "local minimum":
            cfg["baseline"] = {"method": "locmin", "window": param}
        elif bl == "convex hull":
            cfg["baseline"] = {"method": "hull"}
        elif bl == "median":
            cfg["baseline"] = {"method": "median", "window": param}
        sm = self.pp_smooth_combo.currentText()
        if sm == "Savitzky-Golay":
            cfg["smooth"] = {"savgol": int(self.pp_smooth_param.value())}
        elif sm == "Gaussian":
            cfg["smooth"] = {"gaussian": float(self.pp_smooth_param.value())}
        refs = [float(x) for x in _re.split(r"[,\s]+", self.pp_recal_edit.text().strip()) if x]
        if refs:
            cfg["recalibrate"] = {"refs": refs, "tol_ppm": float(self.pp_recal_tol.value())}
        nm = self.pp_norm_combo.currentText()
        if nm.startswith("vector"):
            cfg["normalize"] = "vector"
        elif nm.startswith("reference"):
            try:
                cfg["normalize"] = {"reference": float(self.pp_norm_ref.text())}
            except ValueError:
                pass
        return cfg

    def apply_preprocessing(self):
        if self.ds is None:
            return
        cfg = self._preprocess_config()
        self.ds.set_preprocessing(preprocess.build_pipeline(cfg))
        if self.prov is not None and cfg:                              # record for the methods report
            self.prov.step("preprocess", **{k: (v if not isinstance(v, dict) else
                                                 {kk: vv for kk, vv in v.items()})
                                            for k, v in cfg.items() if k != "normalize"})
            if "normalize" in cfg:
                self.prov.step("normalize", method=str(cfg["normalize"]))
        self.peaks = []
        self.seg = None

        def reprime(progress=None, stage=None):
            # set_preprocessing() cleared the prime cache and, with transforms now active,
            # _dense() returns None — so this prime takes the slow per-pixel streaming path.
            # It's the most expensive pass, so it MUST report a moving bar + stage, not hang
            # silently. prime() emits progress(done, n) every 256 pixels on that path.
            if stage:
                stage("Re-priming spectra")
            self.ds.prime(progress=progress)
            return True
        self._run(reprime, on_done=lambda _: self._after_preprocess(),
                  want_progress=True, want_stage=True, modal=True,
                  stages=["Re-priming spectra"],
                  busy="Applying preprocessing & re-priming…")

    def _after_preprocess(self):
        self._exp_mean_spec = None        # mean spectrum changed after re-priming
        self._plot_mean_spectrum()
        self.statusBar().showMessage("Preprocessing applied. Find peaks again to refresh analyses.")

    # A cube this big (in-RAM CSC bytes) is built out-of-core straight to its on-disk sidecar
    # (bounded RAM) instead of materialising the whole CSC; smaller cubes keep the fast in-RAM
    # build, whose resident matrix serves ion images a touch faster than the lazy store.
    _STREAM_CUBE_BYTES = 1_500_000_000

    def _cube_stream_target(self, ds, min_intensity):
        """``(out_path, fingerprint)`` to build a large slide's cube out-of-core straight to
        its session sidecar (the Phase-2 bounded-memory path), or ``(None, None)`` to use the
        in-RAM build. Only disk-backed real slides with a big estimated cube qualify; any
        failure falls back to the in-RAM build."""
        try:
            src = getattr(ds, "source", None)
            if not src or src == "synthetic":
                return None, None
            if ds.estimate_cube_bytes(min_intensity=min_intensity) < self._STREAM_CUBE_BYTES:
                return None, None
            fp = library.dataset_fingerprint(ds)
            if self._session_path is None:
                self._session_path = session.resolve_session_path(src, fp, ds.n_pixels)
            return session.cube_zarr_path(self._session_path), fp
        except Exception:  # noqa: BLE001 — never let the estimate block a build
            return None, None

    def _build_cube(self, ds, spec, progress=None):
        """Build the fast cube, choosing the bounded-memory streaming path for large slides.
        Returns True when the cube was streamed straight to its sidecar (already persisted)."""
        min_intensity = float(spec.max()) * 0.001
        out_path, fp = self._cube_stream_target(ds, min_intensity)
        if out_path:
            ds.build_mz_cube(min_intensity=min_intensity, progress=progress,
                             out_path=out_path, fingerprint=fp)
            return True                       # on disk at its sidecar → no separate save needed
        ds.build_mz_cube(min_intensity=min_intensity, progress=progress)
        return False

    def build_ion_cache(self):
        if self.ds is None:
            self.statusBar().showMessage("Load a dataset first.")
            return

        def build(progress=None):
            _, spec = self.ds.mean_spectrum()
            if self._build_cube(self.ds, spec, progress=progress):
                self._cube_saved = True       # streamed cube is already at its sidecar
            return True
        self._run(build, want_progress=True,
                  on_done=lambda _: self.statusBar().showMessage(
                      "Fast ion cache built — arbitrary-m/z ion images are now instant."),
                  busy="Building fast ion-image cache…")

    def _autobuild_cache(self):
        """After a disk-backed dataset loads, build the fast m/z cube in the background so
        ROI/region spectra and arbitrary-m/z ion images are served from RAM instead of
        streaming pixels off disk on the GUI thread (the main cause of the "not responding"
        freezes). In-RAM stores (demo, imaging-table CSV) are already instant, so skip them;
        the parallel streaming pass keeps the build quick and fully off the GUI thread."""
        ds = self.ds
        if ds is None or getattr(ds, "_cube", None) is not None:
            return
        if getattr(getattr(ds, "store", None), "in_memory", False):
            return        # in-RAM (demo / imaging-table CSV): masked means already instant
        if getattr(self, "_cube_building", False):
            return        # a build is already in flight — don't start a second
        # NOTE: build for every disk-backed store, even if parallel reads are unavailable —
        # a serial build still yields the cube, and without it every ROI/region select would
        # stream from disk. (Parallelism only affects how fast this background build is.)
        self._cube_building = True

        def build(progress=None):
            try:
                _, spec = ds.mean_spectrum()
                if self._build_cube(ds, spec, progress=progress):
                    self._cube_saved = True       # streamed cube is already at its sidecar
                return True
            finally:
                self._cube_building = False       # clear on success OR error (so retry can run)

        self._run(build, want_progress=True,
                  on_done=lambda _: self.statusBar().showMessage(
                      "Fast cache ready — ROI, region & ion-image views are now instant."),
                  busy="Building fast cache (ROI / ion images)…")

    # ----- image export (photo tools) ------------------------------------- #
    def _export_view(self, widget, default):
        """Export a pyqtgraph composite view (montage / components) in a choice of
        formats. Raster output is rendered at 2× the on-screen size for a crisp result;
        SVG is written as true vector."""
        path, _ = filedialogs.get_save_file_name(
            self, "Export view", default,
            "PNG (*.png);;TIFF (*.tif);;JPEG (*.jpg);;SVG (*.svg)")
        if not path:
            return
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        try:
            if ext == "svg":
                from pyqtgraph.exporters import SVGExporter
                SVGExporter(widget.scene()).export(path)
            else:
                from pyqtgraph.exporters import ImageExporter
                exp = ImageExporter(widget.scene())
                try:                                   # 2× upscale for a publication-crisp raster
                    params = exp.parameters()
                    w0, h0 = int(params["width"]), int(params["height"])
                    params["width"] = w0 * 2
                    params["height"] = h0 * 2
                except Exception:  # noqa: BLE001
                    pass
                exp.export(path)
            self.statusBar().showMessage(f"Wrote {path}")
        except Exception as e:  # noqa: BLE001
            self._on_error(f"Export failed: {e}")

    # ----- segmentation ---------------------------------------------------- #
    def export_methods(self):
        if self.prov is None:
            self.statusBar().showMessage("Load a dataset first.")
            return
        path, _ = filedialogs.get_save_file_name(self, "Save methods report",
                                                        "methods.md", "Markdown (*.md)")
        if path:
            self.prov.to_markdown(path)
            json_path = path.rsplit(".", 1)[0] + ".provenance.json"
            self.prov.to_json(json_path)
            self.statusBar().showMessage(f"Wrote {path} + {json_path}")

    # ----- component analysis ---------------------------------------------- #


# These two run *before* the QApplication/heavy-import stage, so they live in the
# pyqtgraph-free ``_appid`` module that the startup splash (``splash.boot``) imports.
# Re-exported here so ``main._name_macos_menubar`` / ``main._set_windows_app_id`` (and the
# frozen-bundle selftest in scripts/launch.py) keep resolving.
from ._appid import _name_macos_menubar, _set_windows_app_id  # noqa: E402


def app_icon():
    """The app/window icon, from an asset bundled inside the package so it resolves
    regardless of the working directory (frozen bundle, ``python -m``, or a .bat launcher).

    Prefers the *square* multi-resolution ``smile_msi.ico`` (the proper taskbar/Dock icon —
    the wide ``smile_msi_logo.png`` wordmark would be squashed in a square slot); falls back
    to the PNG, then a null QIcon (callers no-op on null)."""
    here = os.path.join(os.path.dirname(__file__), "assets")
    for name in ("smile_msi.ico", "smile_msi_logo.png"):
        path = os.path.join(here, name)
        if os.path.exists(path):
            icon = QtGui.QIcon(path)
            if not icon.isNull():
                return icon
    return QtGui.QIcon()


def apply_app_identity(app):
    """Give Qt/the OS the app's real name everywhere it would otherwise show "Python":
    window titles, the About box, Dock, ⌘-Tab, and ~/Library paths. Safe to call twice."""
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    app.setOrganizationDomain("smilemsi.org")
    app.setDesktopFileName(APP_NAME)
    icon = app_icon()                # title bar + taskbar/Dock (top-level windows inherit it)
    if not icon.isNull():
        app.setWindowIcon(icon)


def run(argv=None, open_path=None, *, app=None, splash=None, clock=None):
    """Build the main window and start the event loop.

    Normally entered via :func:`smile_msi.gui.splash.boot`, which creates the
    ``QApplication`` and shows the startup ``splash`` *before* this module is imported,
    then passes both in along with the shared ``clock``. Called with none of those
    (``run()``), it recreates the old standalone behaviour — no splash — so importing and
    calling ``run`` directly (and the headless tests) keep working unchanged.
    """
    from .splash import StageClock
    if clock is None:
        clock = StageClock()
    if app is None:                  # standalone entry (no splash): preserve old ordering
        _name_macos_menubar()        # must precede QApplication (Cocoa) creation
        _set_windows_app_id()        # taskbar icon/grouping — before any window shows
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv or sys.argv)

    def _stage(name):
        clock.start(name)

    _stage("Loading appearance")
    apply_app_identity(app)          # menu bar / Dock / window titles say "SMILE MSI"
    from .. import prefs             # saved Appearance: Light / Dark (defaults to Dark; legacy "system" → Dark)
    set_theme(app, prefs.get("theme", DEFAULT_THEME))   # before any widget — pyqtgraph reads bg/fg at build

    _stage("Building workspace")     # the dominant phase — pyqtgraph tabs (~800 ms)
    win = MainWindow()
    win._startup_clock = clock       # retained for Help ▸ Show timeline…

    def _dismiss_splash():
        """Close the startup splash once a real window is up (idempotent)."""
        clock.finish_all()
        if splash is not None:
            splash.close()
            splash.deleteLater()

    if open_path:                    # explicit CLI intent (smile_msi data.imzML) → skip the launcher
        clock.finish_all()
        win.show()
        _dismiss_splash()
        win._load_dataset_path(open_path)
        return app.exec()
    # startup launcher: resume a saved sample, open a dataset/session, or start fresh.
    _stage("Scanning saved sessions")
    from .launcher import StartupDialog
    dlg = StartupDialog(win)
    clock.finish_all()
    _dismiss_splash()                # the launcher is itself interactive — don't leave a
                                     # stay-on-top splash trapping input over its modal exec()
    dlg.exec()                       # modal; closing it ⇒ ("fresh", None) ⇒ empty workspace
    win.show()
    kind, payload = dlg.result_action
    if kind == "dataset":
        win._load_dataset_path(payload)
    elif kind == "session":
        win._open_session_path(payload)
    elif kind == "demo":
        win.do_load_demo()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run())
