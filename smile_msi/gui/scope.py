"""Per-analysis "what data is going in" selectors.

Every analysis runs on some set of features and, for the per-region tests, some set
of regions/groups. Historically each test silently grabbed whatever happened to be
selected in the side panel — the active feature list (``_visible_mzs``) and the A/B
region combos — so it was easy to run a test without realising exactly what fed it.
This module adds a compact inline :class:`ScopeBar` that each analysis embeds at the
top of its controls. It states the inputs plainly ("▶ 23 features · All slide") and
lets the user override them right there:

* **Features** — a dropdown of *named feature sets*: the working set, every region
  that has its own picked feature list, and every saved ★ list. Picking one runs this
  one analysis on that list's ions (the engines extract any m/z on demand), without
  disturbing the window-wide active set. The choice is by named list/region — not a
  hand-ticked pile of raw m/z.
* **Regions / groups** — A vs B combos for the two-region tests, and a *Choose
  groups…* picker for the per-region tests, with a readout of the grouping used.

The bar always *defaults* to today's behaviour (the visible working set, the default
grouping), so leaving it untouched changes nothing. :class:`ScopeMixin` provides the
resolver helpers the ``do_*`` methods call instead of reaching for ``_visible_mzs``.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from .common import (MUTED_QSS, ElidedLabel, icon, NoScrollComboBox, filter_edit, _natural_key,
                     check_tree_bar)


# A thin single-line strip, not a tall boxed panel: a hairline top/bottom rule instead
# of a full border + filled background, so the "what data feeds this analysis" header
# costs ~one row of height on every tab instead of two-plus.
_SCOPE_QSS = """
QFrame#scopeBar {
    border: none;
    border-top: 1px solid rgba(128,128,128,0.22);
    border-bottom: 1px solid rgba(128,128,128,0.22);
}
/* scopeTitle colour + scopeReadout weight are set DIRECTLY on the labels (in __init__): a
   QLabel color/font rule written in this container stylesheet does NOT reach the child label
   — Qt draws QLabel text from the palette and drops the property inherited from an ancestor
   widget's stylesheet, so the title rendered at the default colour and the readout wasn't bold. */
"""


def _key(mz):
    """4-dp rounding key — matches the feature-table m/z precision used elsewhere."""
    return round(float(mz), 4)


class _GroupPickDialog(QtWidgets.QDialog):
    """Pick which named regions feed an analysis, grouped under their owning sample.

    Two uses, chosen by ``note``/``title``: pick the *groups* for a per-region test
    (discriminating / multi-group — needs ≥2), or pick the *ROIs* an analysis runs on
    (PCA on a couple of ROIs — needs ≥1). Regions render in a tree (sample → its regions)
    keyed by ``rg['sample']``, with a filter box matching region OR sample name. The min
    count is only advisory text here; the caller (`group_regions` / `region_mask`) enforces
    it. ``chosen()`` still returns the checked region names, so the selection API to callers
    is unchanged."""

    def __init__(self, win, preselected, parent=None,
                 title="Choose groups for this analysis", note=None):
        super().__init__(parent)
        self.win = win
        self.setWindowTitle(title)
        self.resize(440, 500)
        v = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(note or (
            "Tick the regions to compare as groups. Leave fewer than two ticked to use "
            "the default grouping (the segmentation if present, otherwise every region)."))
        note.setWordWrap(True)
        v.addWidget(note)

        self.filter = filter_edit("Filter regions or samples…")   # shared search-box helper
        self.filter.textChanged.connect(self._apply_filter)
        v.addWidget(self.filter)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        # All/None/Invert over the *shown* regions — filter to one slide, hit All, done.
        self._bulk = check_tree_bar(self.tree, "region")
        v.addWidget(self._bulk)
        v.addWidget(self.tree, 1)

        from ..session import _display_name
        pre = set(preselected or [])
        active_src = win.ds.source if getattr(win, "ds", None) is not None else ""

        groups = {}                                    # sample-key -> [rg]; first-seen order kept
        for rg in win.regions:
            groups.setdefault(rg.get("sample") or active_src, []).append(rg)

        def _order(key):                               # active slide first, then natural by name
            return (key != active_src, _natural_key(_display_name(key)))

        for key in sorted(groups, key=_order):
            label = _display_name(key) + ("  (this slide)" if key == active_src else "")
            head = QtWidgets.QTreeWidgetItem([label])
            head.setFlags(QtCore.Qt.ItemIsEnabled)     # a header: not checkable, not selectable
            hf = head.font(0); hf.setBold(True); head.setFont(0, hf)
            self.tree.addTopLevelItem(head)
            for rg in groups[key]:
                mask = win._region_pixel_mask(rg)
                npx = int(mask.sum()) if mask is not None else 0
                it = QtWidgets.QTreeWidgetItem([f"{rg['name']}   ·   {npx:,} px"])
                it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
                it.setCheckState(0, QtCore.Qt.Checked if rg["name"] in pre
                                 else QtCore.Qt.Unchecked)
                it.setData(0, QtCore.Qt.UserRole, rg["name"])
                try:
                    it.setIcon(0, win._color_icon(rg["color"]))
                except Exception:  # noqa: BLE001 — icon is cosmetic
                    pass
                head.addChild(it)
            head.setExpanded(True)

        if not win.regions:
            ph = QtWidgets.QTreeWidgetItem(
                ["(no regions yet — draw an ROI or group clusters)"])
            ph.setFlags(QtCore.Qt.NoItemFlags)
            self.tree.addTopLevelItem(ph)
        self._bulk.refresh_count()                     # the bar was built over an empty tree

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        v.addWidget(btns)

    def _apply_filter(self, *_):
        """Hide leaves whose region name doesn't contain the needle; a matching *sample*
        header reveals all its regions. Hide a header whose leaves are all filtered out.
        Hidden-not-removed, so check states survive the filter (matches attach_list_filter)."""
        needle = self.filter.text().strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            head = self.tree.topLevelItem(i)
            if head.childCount() == 0:                  # the empty-state placeholder
                continue
            sample_match = bool(needle) and needle in head.text(0).lower()
            shown = 0
            for j in range(head.childCount()):
                it = head.child(j)
                hide = bool(needle) and needle not in it.text(0).lower() and not sample_match
                it.setHidden(hide)
                shown += (not hide)
            head.setHidden(bool(needle) and shown == 0)
            if needle:
                head.setExpanded(True)

    def chosen(self):
        out = []
        for i in range(self.tree.topLevelItemCount()):
            head = self.tree.topLevelItem(i)
            for j in range(head.childCount()):
                it = head.child(j)
                if bool(it.flags() & QtCore.Qt.ItemIsUserCheckable) \
                        and it.checkState(0) == QtCore.Qt.Checked:
                    out.append(str(it.data(0, QtCore.Qt.UserRole)))
        return out


class ScopeBar(QtWidgets.QFrame):
    """A compact "Data in this analysis" panel embedded at the top of an analysis view.

    Reads everything live off the owning window, so it always reflects the current
    feature lists / regions. The feature choice is a named set (a region's own list, a
    saved ★ list, or the working set); a fresh bar with nothing chosen behaves exactly
    like the old ``_visible_mzs`` / default-grouping path.
    """
    changed = QtCore.Signal()

    def __init__(self, win, *, feature=True, ab=False, groups=False, regions=False,
                 title="Data in this analysis", parent=None):
        super().__init__(parent)
        self.win = win
        self._feature_choice = None          # None (auto) | ("scope"|"list", name) explicit pick
        self._feature_touched = False        # True once the user picks a set in THIS bar — then
                                             # we stop auto-adopting the app-wide default here
        self._group_override = None          # list[str] region names | None
        self._region_override = None         # list[str] ROI names to run on | None (all pixels)
        self.feat_readout = None
        self.feat_combo = None
        self.group_readout = None
        self.region_readout = None
        self.combo_a = self.combo_b = None
        self.setObjectName("scopeBar")
        self.setStyleSheet(_SCOPE_QSS)
        # A thin strip: never grow taller than its content. Without this, a tab whose main
        # content (a table/splitter) has no vertical stretch shares the slack equally with
        # this frame — inflating the one-row header into a tall grey band.
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 2, 8, 2)
        outer.setSpacing(2)
        title_lbl = QtWidgets.QLabel(title)
        title_lbl.setObjectName("scopeTitle")
        title_lbl.setStyleSheet(MUTED_QSS)             # muted header — set on the label (see _SCOPE_QSS)

        if feature:
            # One compact line: muted title · bold readout · [Use features ▾]. The title
            # no longer takes its own row (the readout already says what's loaded), so the
            # scope header is a single strip instead of a stacked box.
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(8)
            row.addWidget(title_lbl)
            self.feat_readout = ElidedLabel("")
            self.feat_readout.setObjectName("scopeReadout")
            self.feat_readout.setStyleSheet("font-weight: 600;")   # bold readout — set on the label
            self.feat_readout.setToolTip("The features this analysis will run on")
            row.addWidget(self.feat_readout, 1)
            row.addWidget(self._muted("Use features:"))
            self.feat_combo = NoScrollComboBox()           # hover-scroll must not re-pick the set
            self.feat_combo.setMinimumWidth(190)
            self.feat_combo.setToolTip(
                "Run this analysis on a named feature set — the working set, a region's "
                "own feature list, or a saved ★ list — without changing the active set.")
            self.feat_combo.activated.connect(self._on_feat_combo)
            row.addWidget(self.feat_combo)
            outer.addLayout(row)
        else:
            outer.addWidget(title_lbl)

        if ab:
            row = QtWidgets.QHBoxLayout()
            row.addWidget(self._muted("Compare"))
            row.addWidget(QtWidgets.QLabel("A:"))
            self.combo_a = NoScrollComboBox()
            self.combo_a.setMinimumWidth(150)
            self.combo_a.setToolTip("Region A — group segmentation clusters, or draw an ROI")
            row.addWidget(self.combo_a)
            row.addWidget(QtWidgets.QLabel("vs B:"))
            self.combo_b = NoScrollComboBox()
            self.combo_b.setMinimumWidth(150)
            self.combo_b.setToolTip("Region B")
            row.addWidget(self.combo_b)
            row.addStretch(1)
            outer.addLayout(row)

        if regions:
            # "Run on ROIs" — restrict the analysis to the pixels in one or a few regions
            # (their union). Empty = the whole slide (today's behaviour).
            row = QtWidgets.QHBoxLayout()
            self.region_readout = ElidedLabel("")
            self.region_readout.setToolTip("The pixels this analysis runs on — the whole "
                                           "slide, or the union of the chosen ROIs")
            row.addWidget(self.region_readout, 1)
            self.region_btn = QtWidgets.QPushButton("Run on ROIs…")
            self.region_btn.setIcon(icon("pick"))
            self.region_btn.setToolTip("Restrict this analysis to the pixels inside one or a "
                                       "few regions (e.g. run PCA on just 2 ROIs)")
            self.region_btn.clicked.connect(self._pick_regions)
            row.addWidget(self.region_btn)
            outer.addLayout(row)

        if groups:
            row = QtWidgets.QHBoxLayout()
            self.group_readout = ElidedLabel("")
            self.group_readout.setToolTip("The groups the per-region tests "
                                          "(discriminating / multi-group) will compare")
            row.addWidget(self.group_readout, 1)
            self.group_btn = QtWidgets.QPushButton("Choose groups…")
            self.group_btn.setIcon(icon("pick"))
            self.group_btn.setToolTip("Pick which regions act as the groups for the "
                                      "per-region tests")
            self.group_btn.clicked.connect(self._pick_groups)
            row.addWidget(self.group_btn)
            outer.addLayout(row)

        bars = getattr(win, "_scope_bars", None)
        if bars is None:
            bars = win._scope_bars = []
        bars.append(self)
        self.refresh()

    @staticmethod
    def _muted(text):
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet(MUTED_QSS)
        return lbl

    # ----- features -------------------------------------------------------- #
    def feature_mzs(self):
        """The m/z this analysis should run on: the chosen named set's ions — an explicit pick,
        or the app-wide default for an untouched bar — else every visible feature. A set that
        has vanished falls back to the working set."""
        kind, name = self._effective_choice()
        if kind == "default":
            return list(self.win._visible_mzs())
        mzs = self._set_mzs(kind, name)
        if not mzs:                            # the chosen/default list was deleted → fall back
            if self._feature_choice is not None:
                self._feature_choice = None    # clear only an explicit pick; an untouched bar
            return list(self.win._visible_mzs())   # stays auto so it can re-adopt a valid default
        return mzs

    def _set_mzs(self, kind, name):
        """The m/z of a named feature set — a working scope (peaks) or a saved ★ list."""
        win = self.win
        if kind == "scope":
            src = (getattr(win, "_feature_scopes", {}) or {}).get(name) or []
        else:
            src = (getattr(win, "_feature_lists", {}) or {}).get(name) or []
        return [float(r["mz"]) for r in src]

    def _app_default(self):
        """The app-wide default feature set as (kind, name), or None when there's no default
        (or the host window predates the stats mixin). Lets every ScopeBar honour the list the
        user pinned / last selected — the same source the analysis-popup combos use."""
        fn = getattr(self.win, "_default_feature_set_data", None)
        if not callable(fn):
            return None
        try:
            return fn()
        except Exception:  # noqa: BLE001 — a missing default must never break a readout
            return None

    def _effective_choice(self):
        """The feature set this bar resolves to, as (kind, name): the user's explicit pick if
        any; else the app-wide default for an untouched bar; else the working set
        ('default', None). One source so the readout, the selection and feature_mzs() agree."""
        if self._feature_choice is not None:
            return self._feature_choice
        if not self._feature_touched:
            d = self._app_default()
            if d is not None:
                return d
        return ("default", None)

    def _available_sets(self):
        """(label, (kind, name)) for every selectable named set: the working set first,
        then region scopes, then saved ★ lists. Mirrors the right-dock Feature-set combo."""
        win = self.win
        out = [("Working set (visible)", ("default", None))]
        for name in (win._display_scope_names() if hasattr(win, "_display_scope_names") else []):
            label = name if name == "All slide" else f"{name}  (region)"
            out.append((label, ("scope", name)))
        for name in sorted(getattr(win, "_feature_lists", {}) or {}):
            out.append((f"★ {name}", ("list", name)))
        return out

    def _rebuild_feat_combo(self):
        """Repopulate the named-set dropdown from the live feature sets, preserving the
        current choice. Done with signals blocked so it never re-fires _on_feat_combo —
        and only ever from refresh() (an external sync), never the combo's own handler."""
        combo = self.feat_combo
        if combo is None:
            return
        combo.blockSignals(True)
        combo.clear()
        prev_kind = None
        for label, data in self._available_sets():
            kind = data[0]
            if kind == "list" and prev_kind in ("default", "scope"):
                combo.insertSeparator(combo.count())     # ★ saved lists sit below a rule
            combo.addItem(label, data)
            prev_kind = kind
        combo.blockSignals(False)
        self._select_current_choice()

    def _select_current_choice(self):
        combo = self.feat_combo
        want = self._effective_choice()
        idx = next((i for i in range(combo.count()) if combo.itemData(i) == want), -1)
        if idx < 0:                            # chosen/default set not present → working set
            if self._feature_choice is not None:
                self._feature_choice = None    # an explicit pick vanished; an untouched bar stays
            idx = 0                            #   auto so a later refresh can re-adopt a default
        if idx != combo.currentIndex():
            combo.blockSignals(True)
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def _on_feat_combo(self, idx):
        data = self.feat_combo.itemData(idx)
        if data is None:                       # a separator
            self._select_current_choice()
            return
        kind, name = data
        self._feature_choice = None if kind == "default" else (kind, name)
        self._feature_touched = True           # an explicit pick (incl. 'Working set') — from now
                                               # on keep it, don't auto-adopt the app-wide default
        self._update_feature_readout()         # NB: don't rebuild the combo from its own handler
        self.changed.emit()

    # ----- groups ---------------------------------------------------------- #
    def group_regions(self):
        """Chosen group region-names (≥2, pruned to those still alive), or ``None`` to
        use the window's default grouping."""
        if not self._group_override:
            return None
        live = {rg["name"] for rg in self.win.regions}
        keep = [n for n in self._group_override if n in live]
        if len(keep) < 2:
            self._group_override = None
            return None
        return keep

    def _pick_groups(self):
        dlg = _GroupPickDialog(self.win, self._group_override or [], self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        chosen = dlg.chosen()
        self._group_override = chosen if len(chosen) >= 2 else None
        self.refresh()
        self.changed.emit()

    # ----- ROI scope (run on a subset of pixels) --------------------------- #
    def region_names(self):
        """Chosen ROI names (pruned to those still alive), or ``None`` for the whole slide."""
        if not self._region_override:
            return None
        live = {rg["name"] for rg in self.win.regions}
        keep = [n for n in self._region_override if n in live]
        if not keep:
            self._region_override = None
            return None
        return keep

    def region_mask(self):
        """Union pixel mask (bool[n_pix]) of the chosen ROIs, or ``None`` to run on every
        pixel. A chosen ROI with no pixels is skipped; if none resolve, returns ``None``."""
        names = self.region_names()
        if not names:
            return None
        import numpy as np
        regions = [rg for rg in self.win.regions if rg["name"] in names]
        mask = None
        for rg in regions:
            m = self.win._region_pixel_mask(rg)
            if m is None:
                continue
            m = np.asarray(m, bool)
            mask = m if mask is None else (mask | m)
        return mask if (mask is not None and mask.any()) else None

    def _pick_regions(self):
        dlg = _GroupPickDialog(
            self.win, self._region_override or [], self,
            title="Run this analysis on which ROIs?",
            note=("Tick one or a few regions — the analysis runs on the union of their "
                  "pixels (e.g. PCA on just 2 ROIs). Leave none ticked to run on the whole "
                  "slide."))
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        chosen = dlg.chosen()
        self._region_override = chosen or None
        self.refresh()
        self.changed.emit()

    def _region_text(self):
        names = self.region_names()
        if not names:
            return "On: whole slide"
        m = self.region_mask()
        npx = int(m.sum()) if m is not None else 0
        return f"On: {', '.join(names)}   ({npx:,} px)"

    # ----- readouts -------------------------------------------------------- #
    def refresh(self):
        """External sync (feature lists / regions / segmentation changed): rebuild the
        dropdown and both readouts. Never call from the combo's own handler."""
        if self.feat_combo is not None:
            self._rebuild_feat_combo()
            self._update_feature_readout()
        if self.region_readout is not None:
            self.region_readout.setText(self._region_text())
        if self.group_readout is not None:
            self.group_readout.setText(self._group_text())

    def _update_feature_readout(self):
        if self.feat_readout is not None:
            self.feat_readout.setText("▶ " + self._feature_text())

    def _feature_text(self):
        win = self.win
        kind, name = self._effective_choice()
        if kind == "default":
            if not win.peaks:
                return "no features yet — find peaks first"
            n = len(win._visible_mzs())
            nm = getattr(win, "_flist_name", None) or "working set"
            return f"{n} features · {nm}"
        n = len(self.feature_mzs())
        label = f"★ {name}" if kind == "list" else name
        return f"{n} features · {label}"

    def _group_text(self):
        win = self.win
        chosen = self.group_regions()
        if chosen:
            return "Across groups: " + ", ".join(chosen) + f"   ({len(chosen)})"
        if getattr(win, "seg", None) is not None:
            return f"Across groups: segmentation ({win.seg.n_clusters} clusters) · default"
        n = sum(1 for rg in win.regions if win._region_pixel_mask(rg) is not None)
        if n >= 2:
            return f"Across groups: all {n} regions · default"
        return "Across groups: none yet — run segmentation or add ≥2 regions"


class ScopeMixin:
    """Resolver helpers the ``do_*`` analysis methods call so a test's inputs come
    from its :class:`ScopeBar` (with the historical defaults as the fall-back)."""

    def _scope_mzs(self, bar):
        """Features feeding an analysis — the bar's chosen set, or every visible m/z."""
        if bar is not None:
            return bar.feature_mzs()
        return self._visible_mzs()

    def _scope_grouping(self, bar):
        """``(labels, names)`` for the per-region tests. Uses the bar's chosen groups
        when set, else the window's default grouping. Returns ``(None, message)`` on
        failure, matching :meth:`_region_grouping`."""
        chosen = bar.group_regions() if bar is not None else None
        if chosen:
            regions = [rg for rg in self.regions if rg["name"] in chosen]
            labels, names = self._labels_from_regions(regions)
            if labels is None:
                return None, ("The chosen groups need pixels — assign clusters or draw "
                              "ROIs, or pick different groups.")
            return labels, names
        return self._region_grouping()

    def _refresh_scope_bars(self):
        """Keep every embedded scope bar's readout/dropdown current (feature lists /
        regions / segmentation changed). Best-effort — a destroyed bar is skipped."""
        for bar in getattr(self, "_scope_bars", []):
            try:
                bar.refresh()
            except RuntimeError:              # underlying C++ widget gone
                pass
        # the classifier's 'Predict in' region picker isn't a scope bar but tracks the same regions
        if hasattr(self, "_refresh_cls_predict_regions"):
            try:
                self._refresh_cls_predict_regions()
            except RuntimeError:
                pass
