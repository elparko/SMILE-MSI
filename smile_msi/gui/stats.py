"""StatsTabMixin — extracted from the monolithic MainWindow (no behavior change)."""
from __future__ import annotations


import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .. import (spatial, imaging, studio)
from .common import (PALETTE, REGION_A_COLOR, REGION_B_COLOR, AB_COLORS, hex_to_rgba,
                     colormap, copy_table, export_table, fill_table, install_table_export,
                     ControlBar, section_title, tab_page, ACCENT, GUIDE_LINE, signal_mz_range,
                     note, glossary_button, set_header_tooltips, plot_caption,
                     table_placeholder, RegionMultiSelect, icon, dark_image_view, MUTED_FG,
                     MUTED_QSS, NoScrollComboBox, ElidedLabel, CheckList, check_table_bar,
                     button, NoScrollDoubleSpinBox, NoScrollSpinBox)
from .scope import ScopeBar
from . import filedialogs


class StatsTabMixin:

    def _cmp_compare_set(self):
        """Regions whose ion image appears in the preview strip: A, B, then up to two
        extra ticked regions — capped at 4 panels. A/B keep the orange/blue scheme; the
        extras use their own region color."""
        ab = self._cmp_ab
        out = [{"name": ab["na"], "color": REGION_A_COLOR, "mask": ab["ma"]},
               {"name": ab["nb"], "color": REGION_B_COLOR, "mask": ab["mb"]}]
        for name in self._cmp_extra:
            if len(out) >= 4:
                break
            if name in (ab["na"], ab["nb"]):
                continue
            rg = next((r for r in self.regions if r["name"] == name), None)
            if rg is None:
                continue
            m = self._region_pixel_mask(rg)
            if m is not None and m.any():
                out.append({"name": name, "color": rg.get("color", "#888"), "mask": m})
        return out

    def _render_cmp_previews(self, mz):
        """Light up the selected ion across the compared regions: one small ion image per
        region, each spotlighting that region's pixels on a SHARED color scale so the
        intensities are directly comparable."""
        panels = self._cmp_panels
        lay = getattr(self, "_cmp_prev_layout", None)

        def show_panel(i, visible):
            panels[i]["cell"].setVisible(visible)
            if lay is not None:                        # only visible panels share the row width
                lay.setStretch(i, 1 if visible else 0)

        if mz is None or getattr(self, "_cmp_payload", None) is None or self.ds is None:
            for i in range(len(panels)):
                show_panel(i, False)
            self._cmp_n_previews = 0
            return
        img = self.ds.ion_image(float(mz), tol_ppm=self.ppm, reduce=self.reduce, norm=self.norm)
        high = float(self.contrast_spin.value())
        clipped = imaging.quantile_clip(img, high=high)
        finite = clipped[np.isfinite(clipped)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        if hi <= lo:
            hi = lo + 1.0
        lut = colormap(self.cmap_combo.currentText()).getLookupTable()
        regions = self._cmp_compare_set()
        for i, pn in enumerate(panels):
            if i >= len(regions):
                show_panel(i, False)
                continue
            rg = regions[i]
            region2d = self.ds.to_image(rg["mask"].astype(float), fill=0.0) > 0.5
            pim = clipped.copy()
            pim[~region2d] = lo                        # spotlight: dim everything outside the region
            pn["item"].setImage(pim, levels=(lo, hi))
            pn["item"].setLookupTable(lut)
            meanv = float(img[region2d].mean()) if region2d.any() else 0.0
            pn["title"].setText(f"{rg['name']} · mean {meanv:.0f} · {int(region2d.sum())} px")
            pn["title"].setStyleSheet(f"color:{rg['color']}; font-weight:bold")
            was_hidden = not pn["cell"].isVisible()
            show_panel(i, True)
            # only refit the view when the panel first appears or its grid changes —
            # keep the user's pan/zoom across peak changes
            fit_key = (rg["name"], pim.shape)
            if was_hidden or pn.get("fit_key") != fit_key:
                pn["vb"].autoRange()
                pn["fit_key"] = fit_key
        self._cmp_n_previews = len(regions)

    def _select_cmp_ion(self, mz, sync=True):
        """Select a differential ion: refresh the previews, highlight it in the tables +
        difference spectrum, and (on a user click) mirror it to the window-wide active
        feature so the Ion image + feature list follow."""
        if mz is None:
            return
        self._cmp_selected_mz = float(mz)
        self._render_cmp_previews(self._cmp_selected_mz)
        self._highlight_cmp_selection(self._cmp_selected_mz)
        if sync and hasattr(self, "set_active_mz"):
            self.set_active_mz(self._cmp_selected_mz)

    def _sync_cmp_active(self, mz):
        """Make the Region-comparison view follow the window-wide active ion. When the user
        scrolls peaks (arrow keys), clicks the feature list, or drags the m/z cursor, the
        comparison previews + difference-spectrum cursor + per-region tables track it too —
        not just clicks made inside the comparison tab. Only acts once a comparison has been
        run and its tab is on-screen, and always sync=False so we don't loop back into
        set_active_mz."""
        if mz is None or getattr(self, "_cmp_payload", None) is None:
            return
        if not self._page_on_screen(getattr(self, "_cmp_tab", None)):
            return                                # not looking at the comparison — skip the redraw
        if self._cmp_selected_mz is not None and float(mz) == self._cmp_selected_mz:
            return                                # already showing this ion — nothing to redraw
        self._select_cmp_ion(float(mz), sync=False)

    def _highlight_cmp_selection(self, mz):
        self.cmp_cursor.setValue(float(mz))
        self.cmp_cursor.show()
        key = f"{float(mz):.4f}"
        for side in ("A", "B"):
            tbl = getattr(self, f"cmp_table_{side}")
            tbl.blockSignals(True)
            tbl.clearSelection()
            for r in range(tbl.rowCount()):
                it = tbl.item(r, 0)
                if it is not None and it.text() == key:
                    tbl.selectRow(r)
                    tbl.scrollToItem(it)
                    break
            tbl.blockSignals(False)

    # ---- per-popup feature-list selector (working set / region scope / ★ list) ---- #
    def _feature_set_items(self):
        """(label, (kind, name)) for each selectable named feature set — the working set,
        each region's own picked list, every saved ★ list, and every ◆ lipid list. Mirrors
        the dock's feature-set dropdown so each analysis popup offers the same choice."""
        out = [("Working set (visible)", ("default", None))]
        names = self._display_scope_names() if hasattr(self, "_display_scope_names") else []
        for n in names:
            out.append((n if n == "All slide" else f"{n}  (region)", ("scope", n)))
        for n in sorted(getattr(self, "_feature_lists", {}) or {}):
            out.append((f"★ {n}", ("list", n)))
        for n in sorted(getattr(self, "_lipid_lists", {}) or {}):
            out.append((f"◆ {n}", ("lipidlist", n)))
        return out

    def _refresh_feature_combo(self, combo):
        """Repopulate a feature combo from the live named sets. An untouched combo (the user has
        not picked in it yet) re-points at the app-wide default feature set (the last one
        selected, or a pinned default) on every refresh — so analyses default to the list you
        work off even when the combo was built, hidden, before that list loaded or before you
        pinned a default. Once the user picks in the combo, their choice is preserved instead."""
        if combo.property("user_touched"):
            cur = combo.currentData()
        else:
            cur = self._default_feature_set_data()
        combo.blockSignals(True)
        combo.clear()
        for label, data in self._feature_set_items():
            combo.addItem(label, data)
        idx = next((i for i in range(combo.count()) if combo.itemData(i) == cur), 0)
        combo.setCurrentIndex(max(0, idx))
        combo.blockSignals(False)

    def _default_feature_set_data(self):
        """The default feature set for new analysis popups: the set the user last actively
        selected, else a list pinned via 'Set current list as default' (persisted in prefs)
        when it exists in this dataset, else None (→ the working set)."""
        d = getattr(self, "_default_feature_set", None)
        if d is not None:
            return d
        try:
            from .. import prefs
            p = prefs.get("default_feature_list", None)
        except Exception:  # noqa: BLE001
            p = None
        if p and p.get("name"):
            want = (p.get("kind", "list"), p["name"])
            if want in {it[1] for it in self._feature_set_items()}:
                self._default_feature_set = want
                return want
        return None

    def _sync_default_feature_selectors(self):
        """Push a just-changed app-wide default into the already-built analysis selectors so it
        takes effect without reopening anything: every ScopeBar (untouched ones adopt it) and the
        eager, hidden System-A combos (roi_feat / cc_feat) — which aren't ScopeBars, so
        _refresh_scope_bars doesn't reach them. User-overridden selectors keep their pick.
        Best-effort: a destroyed widget is skipped."""
        if hasattr(self, "_refresh_scope_bars"):
            self._refresh_scope_bars()
        for attr in ("roi_feat", "cc_feat", "shap_feat"):
            combo = getattr(self, attr, None)
            if combo is None:
                continue
            try:
                self._refresh_feature_combo(combo)
            except RuntimeError:        # underlying C++ widget gone
                pass
        # cohort combos keep their Consensus default (untouched) but must still pick up a
        # newly saved ★ list / pinned default in their dropdown
        if hasattr(self, "_refresh_cohort_feature_combos"):
            self._refresh_cohort_feature_combos()

    def _combo_feature_mzs(self, combo):
        """The m/z this popup's feature combo selects — its named set, else the visible set."""
        data = combo.currentData() if combo is not None else None
        if not data or data[0] in ("default", "consensus"):
            return list(self._visible_mzs())
        kind, name = data
        if kind == "lipidlist":                       # ◆ list: flatten every class's ions
            entries = (getattr(self, "_lipid_lists", {}) or {}).get(name) or []
            mzs = [float(m) for e in entries for m in (e.get("mzs") or [])]
            return mzs or list(self._visible_mzs())
        src = (getattr(self, "_feature_scopes", {}) if kind == "scope"
               else getattr(self, "_feature_lists", {})) or {}
        mzs = [float(r["mz"]) for r in (src.get(name) or [])]
        return mzs or list(self._visible_mzs())

    # ---- cohort analyses: a feature combo whose first item is the cross-sample Consensus,
    #      then every named feature set — so a cohort comparison/embedding can run on a saved
    #      ★ list or region scope, not just Consensus / the active set (app-wide parity). ---- #
    def _make_consensus_feature_combo(self, tooltip=""):
        combo = NoScrollComboBox()
        combo.setMinimumWidth(200)
        if tooltip:
            combo.setToolTip(tooltip)
        combo.setProperty("user_touched", False)        # until picked, stays on Consensus
        combo.activated.connect(lambda _i, c=combo: c.setProperty("user_touched", True))
        bag = getattr(self, "_cohort_feature_combos", None)
        if bag is None:
            bag = self._cohort_feature_combos = []
        bag.append(combo)
        self._refresh_consensus_feature_combo(combo)
        return combo

    def _refresh_consensus_feature_combo(self, combo):
        """Repopulate one cohort feature combo: Consensus first, then the live named sets.
        An untouched combo stays on Consensus (the safe cross-slide default); once the user
        picks, their choice is preserved while newly saved ★ lists still appear."""
        cur = combo.currentData() if combo.property("user_touched") else ("consensus", None)
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Consensus (shared peaks)", ("consensus", None))
        for label, data in self._feature_set_items():
            combo.addItem(label, data)
        idx = next((i for i in range(combo.count()) if combo.itemData(i) == cur), 0)
        combo.setCurrentIndex(max(0, idx))
        combo.blockSignals(False)

    def _refresh_cohort_feature_combos(self):
        """Refresh every cohort feature combo (Cohort / Cohort UMAP / Cohort segmentation) so a
        list saved or a default pinned after startup shows up without reopening. Best-effort."""
        for combo in (getattr(self, "_cohort_feature_combos", None) or []):
            try:
                self._refresh_consensus_feature_combo(combo)
            except RuntimeError:                        # underlying C++ widget gone
                pass

    def _cohort_combo_is_consensus(self, combo):
        data = combo.currentData() if combo is not None else None
        return (not data) or data[0] == "consensus"

    def _cohort_feature_targets(self, combo, refs, *, tol_ppm, min_prevalence, value,
                                sessions=None):
        """The shared m/z axis a cohort analysis runs on, resolved from its feature combo:
        Consensus clustering across samples, or the m/z of the chosen named set (snapshotted
        at call time so a later slide-switch can't silently change the comparison axis)."""
        from .. import cohort as cohort_engine        # lazy: keep cohort off GUI startup
        if self._cohort_combo_is_consensus(combo):
            return cohort_engine.consensus_targets(
                refs, tol_ppm=tol_ppm, min_prevalence=min_prevalence, value=value,
                sessions=sessions)
        return sorted({round(float(m), 4) for m in self._combo_feature_mzs(combo)})

    def _drop_nested_children(self, regions):
        """Of ``regions``, drop any whose parent (or a deeper ancestor) is also in the set.

        A **compartment** parent region (created by grouping sub-regions) owns the *union*
        of its children's pixels, so in a partition the parent and its own children fight
        over the same pixels — and because nesting orders the parent first, first-wins lets
        the parent swallow every child, starving them out of the comparison entirely (a
        compartment built from segmented sub-regions then reads as 'nothing found'). Keeping
        only the top-level region of each family makes the compartment itself the group, the
        natural unit to compare. Children that are *not* under any region in the set (e.g. the
        user explicitly picked the sub-regions, not the parent) are left untouched."""
        names = {r["name"] for r in regions}
        by_name = {r["name"]: r for r in (getattr(self, "regions", None) or [])}

        def ancestor_in_set(rg, seen):
            p = rg.get("parent")
            if not p or p == rg["name"] or p in seen:
                return False
            if p in names:
                return True
            seen.add(p)
            par = by_name.get(p)
            return ancestor_in_set(par, seen) if par is not None else False

        return [r for r in regions if not ancestor_in_set(r, set())]

    def _labels_from_regions(self, regions=None):
        """Build a per-pixel label array from named regions (ROI- or cluster-backed),
        so the per-region analyses work even without a segmentation. Unassigned pixels
        get ``-1`` (the engine drops them); overlapping pixels go to the first region.
        A compartment parent and its own sub-regions never both become groups — the
        parent wins (see :meth:`_drop_nested_children`). Returns ``(labels, names)`` —
        ``names[k]`` is the region for label ``k`` — or ``(None, [])`` if fewer than two
        regions resolve to pixels."""
        regions = self.regions if regions is None else regions
        regions = self._drop_nested_children(regions)
        labels = np.full(self.ds.n_pixels, -1, dtype=int)
        names = []
        for rg in regions:
            m = self._region_pixel_mask(rg)
            if m is None or not m.any():
                continue
            take = m & (labels < 0)                  # first region wins on any overlap
            if not take.any():
                continue
            labels[take] = len(names)
            names.append(rg["name"])
        return (labels, names) if len(names) >= 2 else (None, [])

    def _region_grouping(self):
        """Resolve the labelling the group-wise tests (Discriminating + Multi-group) run
        on. The 'Group by' selector chooses between the data-driven segmentation clusters
        and the user's named regions; whichever is picked, we fall back to the other when
        the chosen source isn't available. Returns ``(labels, names)`` where ``names`` is
        ``None`` for a segmentation (label = cluster id), or ``(None, msg)`` on failure."""
        gb = getattr(self, "group_by_combo", None)
        prefer_regions = gb is not None and gb.currentText() == "Named regions"
        have_seg = self.seg is not None
        if prefer_regions:
            labels, names = self._labels_from_regions()
            if labels is not None:
                return labels, names
            if have_seg:                                  # asked for regions, none defined
                return self.seg.labels, None
            return None, ("Group by 'Named regions' needs 2+ regions — draw ROIs → 'Add ROI' "
                          "in the Regions panel, or run segmentation.")
        if have_seg:
            return self.seg.labels, None
        labels, names = self._labels_from_regions()       # no segmentation → fall back
        if labels is None:
            return None, ("Run segmentation, or define at least two regions (draw ROIs → "
                          "'Add ROI' in the Regions panel), first.")
        return labels, names

    # ----- per-ion ROC curve (changes with the selected ion) --------------- #
    def _render_roc(self, mz):
        """Draw the ROC curve for one ion between Region A and Region B. The chance
        diagonal runs up the middle; the curve bows above it (blue) when the ion is
        enriched in B and below (orange) when enriched in A, and the shaded area under
        the curve is the AUC the table reports. Redrawn on every ion selection."""
        plot = getattr(self, "roc_plot", None)
        if plot is None:
            return
        plot.clear()
        roc = getattr(self, "_roc", None)
        diag = pg.mkPen(GUIDE_LINE, style=QtCore.Qt.DashLine)
        if roc is None or mz is None:
            plot.plot([0, 1], [0, 1], pen=diag)
            plot.setTitle("ROC — run ROI statistics, then click an ion")
            return
        mzs = roc["mzs"]
        j = int(np.argmin(np.abs(mzs - float(mz)))) if len(mzs) else -1
        if j < 0 or abs(float(mzs[j]) - float(mz)) > 1e-3:
            plot.plot([0, 1], [0, 1], pen=diag)
            plot.setTitle("ROC — that ion isn't in this result")
            return
        fpr, tpr, auc = spatial.roc_curve(roc["XA"][:, j], roc["XB"][:, j])
        higher_b = auc >= 0.5
        color = REGION_B_COLOR if higher_b else REGION_A_COLOR
        enriched = roc["lb"] if higher_b else roc["la"]
        plot.plot(fpr, tpr, pen=pg.mkPen(color, width=2.4),    # area under = the AUC
                  fillLevel=0, brush=hex_to_rgba(color, 60))
        plot.plot([0, 1], [0, 1], pen=diag)                   # chance line, on top
        plot.setXRange(0, 1, padding=0.02)
        plot.setYRange(0, 1, padding=0.02)
        lab = (self.annotate(float(mz)) or "").split(" (")[0] or f"m/z {float(mz):.4f}"
        q = self._roc_stat.get(round(float(mz), 4), (auc, float("nan"), float("nan")))[1]
        qtxt = f" · q {q:.1e}" if np.isfinite(q) else ""
        plot.setTitle(f"{lab} — AUC {auc:.3f}  (↑ {enriched}){qtxt}", color=color)

    def _select_stats_ion(self, mz, sync=True):
        """Select an ion on the stats tab: redraw its ROC, highlight its table row and
        volcano point, and (on a user action) mirror it to the window-wide active ion so
        the ion image and other views follow. ``sync=False`` keeps the selection local."""
        if mz is None:
            return
        self._stats_selected_mz = float(mz)
        self._render_roc(self._stats_selected_mz)
        self._highlight_stats_row(self._stats_selected_mz)
        self._highlight_volcano(self._stats_selected_mz)
        if sync and hasattr(self, "set_active_mz"):
            self.set_active_mz(self._stats_selected_mz)

    def _highlight_stats_row(self, mz):
        tbl = self.stats_table
        key = f"{float(mz):.4f}"
        tbl.blockSignals(True)
        for r in range(tbl.rowCount()):
            it = tbl.item(r, 0)
            if it is not None and it.text() == key:
                tbl.selectRow(r)
                tbl.scrollToItem(it)
                break
        tbl.blockSignals(False)

    def _highlight_volcano(self, mz):
        """Ring the selected ion's point on the volcano so it's easy to connect the
        overview dot to the ROC curve below."""
        pts = getattr(self, "_volcano_pts", None)
        m = getattr(self, "_volcano_marker", None)
        if m is not None:
            try:
                self.volcano.removeItem(m)
            except Exception:  # noqa: BLE001
                pass
            self._volcano_marker = None
        xy = pts.get(round(float(mz), 4)) if pts else None
        if xy is None:
            return
        m = pg.ScatterPlotItem(x=[xy[0]], y=[xy[1]], size=16, symbol="o",
                               brush=pg.mkBrush(0, 0, 0, 0), pen=pg.mkPen(ACCENT, width=2.5))
        self.volcano.addItem(m)
        self._volcano_marker = m

    def _sync_stats_active(self, mz):
        """Follow the window-wide active ion on the stats tab: when it changes elsewhere
        (feature list, ion image, arrow keys) and this tab is on-screen, redraw the ROC
        for that ion. No-ops until an ROI-stats result exists; always sync=False so we
        don't loop back into set_active_mz."""
        if mz is None or getattr(self, "_roc", None) is None:
            return
        if not self._page_on_screen(getattr(self, "_stats_tab", None)):
            return
        if getattr(self, "_stats_selected_mz", None) is not None and float(mz) == self._stats_selected_mz:
            return
        self._select_stats_ion(float(mz), sync=False)

    def _clear_volcano_legend(self):
        """Drop the per-region legend the discriminating plot adds, so it doesn't
        linger over a subsequent ROI volcano (PlotItem.clear() leaves it behind)."""
        pi = self.volcano.getPlotItem()
        leg = getattr(pi, "legend", None)
        if leg is not None:
            try:
                leg.scene().removeItem(leg)
            except Exception:  # noqa: BLE001 — legend already detached; clearing the ref is enough
                pass
            pi.legend = None

    def _reset_stats_views(self):
        """Clear every statistics / region-comparison result and its widgets, so a wipe &
        restart leaves no stale table, volcano, ROC curve or comparison panel behind.
        Best-effort per widget (some may not exist yet)."""
        self.last_stats = None
        self._stats_export = None
        self._disc_groups = None
        self._roc = None
        self._roc_stat = {}
        self._stats_selected_mz = None
        self._cmp_payload = None
        self._cmp_ab = {}
        self._cmp_stat = {}
        self._cmp_selected_mz = None
        tbl = getattr(self, "stats_table", None)
        if tbl is not None:
            tbl.clearContents()
            tbl.setRowCount(0)
        if getattr(self, "volcano", None) is not None:
            try:
                self._clear_volcano_legend()
            except Exception:  # noqa: BLE001
                pass
            self.volcano.clear()
        if getattr(self, "roc_plot", None) is not None:
            self._render_roc(None)
        for side in ("A", "B"):
            t = getattr(self, f"cmp_table_{side}", None)
            if t is not None:
                t.clearContents()
                t.setRowCount(0)
        if getattr(self, "cmp_diff", None) is not None:
            self.cmp_diff.clear()
        try:
            self._render_cmp_previews(None)
        except Exception:  # noqa: BLE001
            pass
        for b in (getattr(self, "_cmp_export_btns", ()) or ()):
            try:
                b.setEnabled(False)
            except Exception:  # noqa: BLE001
                pass
        for name in ("b_csv", "b_xlsx", "b_roc_report", "b_save_lists"):
            b = getattr(self, name, None)
            if b is not None:
                b.setEnabled(False)
