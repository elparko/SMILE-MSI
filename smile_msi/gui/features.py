"""FeaturesTabMixin — extracted from the monolithic MainWindow (no behavior change)."""
from __future__ import annotations

import hashlib
import os
import traceback
from collections import OrderedDict

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .. import (spatial, imaging, isotopes, session, msms, library, studio)
from ..msi import _mad
from .common import (add_copy_actions, colormap, confirm, fill_table, set_header_tooltips)
from . import filedialogs


class FeaturesTabMixin:
    def _feat_table_menu(self, pos):
        menu = QtWidgets.QMenu(self.feat_table)
        row = self.feat_table.rowAt(pos.y())
        mz = self._feat_mz_at(row) if row is not None and row >= 0 else None
        if mz is not None:
            menu.addAction("Rename label…", lambda: self._rename_feature_label(mz))
            menu.addSeparator()
        n_sel = len(self._selected_feature_mzs())
        act_sub = menu.addAction(f"New feature list from {n_sel} selected…",
                                 self._save_selected_as_feature_list)
        act_sub.setEnabled(n_sel > 0)
        n_vis = sum(1 for p in self.peaks if not p.get("hidden"))
        act_vis = menu.addAction(f"New feature list from {n_vis} visible (eye-ticked)…",
                                 self._save_visible_as_feature_list)
        act_vis.setEnabled(0 < n_vis < len(self.peaks))
        menu.addSeparator()
        menu.addAction("Add feature…", self._add_feature)
        menu.addAction("Remove selected", self._remove_selected_features)
        menu.addAction("Clear all features", self._clear_features)
        menu.addAction("Import targets…", self._import_targets)
        menu.addSeparator()
        add_copy_actions(menu, self.feat_table)
        menu.addAction("Export feature list (CSV, with analyses)…", self.export_features)
        menu.addAction("Export…  (images · spectra · book)", lambda: self.open_export_hub("features"))
        menu.exec(self.feat_table.viewport().mapToGlobal(pos))

    # ----- worker plumbing ------------------------------------------------- #
    def _build_class_map(self):
        self._class_map = {}
        for p in self.peaks:
            cs = self.ann.annotate_mz(p["mz"])
            if cs:
                self._class_map.setdefault(cs[0].lipid.lipid_class, []).append(p["mz"])
        self.class_combo.clear()
        self.class_combo.addItems(sorted(self._class_map, key=lambda c: -len(self._class_map[c])))

    def do_class_image(self):
        if self.ds is None or not self._class_map:
            self.statusBar().showMessage("Find peaks first.")
            return
        cls = self.class_combo.currentText()
        mzs = self._class_map.get(cls, [])
        if not mzs:
            return
        img = imaging.quantile_clip(
            self.ds.composite_image(mzs, tol_ppm=self.ppm, reduce=self.reduce, norm=self.norm,
                                    weight=self.composite_weight),
            high=float(self.contrast_spin.value()))
        self.iv.setImage(img, autoLevels=True)
        self.iv.setColorMap(colormap(self.cmap_combo.currentText()))
        self.tabs.setCurrentIndex(0)
        self._set_ion_tab_caption(f"total {cls} ({len(mzs)} ions)")
        dlg = getattr(self, "_composite_dialog", None)       # step aside to reveal the result
        if dlg is not None:
            dlg.hide()
        self.statusBar().showMessage(f"Composite image: total {cls} ({len(mzs)} ions).")

    def _populate_peak_combos(self):
        labels = [f"{p['mz']:.4f}  {self.annotate(p['mz']) or ''}".strip() for p in self.peaks]
        for combo in (self.ratio_a, self.ratio_b):
            combo.clear()
            combo.addItems(labels)
        if len(labels) > 1:
            self.ratio_b.setCurrentIndex(1)

    def do_ratio_image(self):
        if not self.peaks:
            self.statusBar().showMessage("Find peaks first.")
            return
        ia, ib = self.ratio_a.currentIndex(), self.ratio_b.currentIndex()
        if ia < 0 or ib < 0:
            return
        mz_a, mz_b = self.peaks[ia]["mz"], self.peaks[ib]["mz"]
        img = imaging.quantile_clip(
            self.ds.ratio_image(mz_a, mz_b, tol_ppm=self.ppm, norm=self.norm),
            high=float(self.contrast_spin.value()))
        self.iv.setImage(img, autoLevels=True)
        self.iv.setColorMap(colormap(self.cmap_combo.currentText()))
        self.tabs.setCurrentIndex(0)
        self._set_ion_tab_caption(f"ratio {mz_a:.3f}/{mz_b:.3f}")
        dlg = getattr(self, "_composite_dialog", None)       # step aside to reveal the result
        if dlg is not None:
            dlg.hide()
        self.statusBar().showMessage(f"Ratio image: {mz_a:.4f} / {mz_b:.4f}.")

    def _active_roi_mask(self):
        """Pixel mask of the region backing the active feature scope, or None when
        the working set isn't scoped to a region (e.g. 'All slide' or a saved list).
        Drives the per-feature 'how confined to this region' score in the table."""
        scope = getattr(self, "_active_feature_scope", None)
        if not scope or scope == "All slide" or self.ds is None:
            return None
        rg = next((r for r in getattr(self, "regions", []) if r.get("name") == scope), None)
        if rg is None:
            return None
        mask = self._region_pixel_mask(rg)
        return mask if mask is not None and mask.any() else None

    # How many recently-built feature-list bundles to keep in the RAM cache. Re-selecting
    # or switching back to a list the user already loaded is the common case and the slow
    # one (the full build_feature_list + target/decoy estimate_fdr storm runs otherwise),
    # so an exact memoization on the total input signature turns it into an instant hit.
    _FL_CACHE_MAX = 24

    def _feature_list_cache_key(self, ds, peaks, ppm, norm, mode, id_ppm, roi_mask, lipid_db):
        """A *total* key for the do_feature_list bundle — every input that can change the
        annotated table or its FDR. Returns None when it can't be keyed safely (no ds /
        no peaks), which disables caching for that call rather than risk a stale hit."""
        if ds is None or not peaks:
            return None
        token = getattr(ds, "cache_token", None)
        if token is None:
            return None                                  # unknown ds identity → don't cache
        peak_sig = tuple((round(float(p["mz"]), 5), p.get("window")) for p in peaks)
        if roi_mask is None:
            roi_sig = None
        else:
            m = np.ascontiguousarray(np.asarray(roi_mask, dtype=bool))
            roi_sig = (m.shape, hashlib.sha1(m.tobytes()).hexdigest()[:16])
        db_sig = None if lipid_db is None else id(lipid_db)
        return (token(), peak_sig, round(float(ppm), 6), str(norm), str(mode),
                round(float(id_ppm), 6), str(getattr(self, "reduce", "")), roi_sig, db_sig)

    def _fl_cache_get(self, key):
        cache = getattr(self, "_fl_cache", None)
        if not cache or key not in cache:
            return None
        cache.move_to_end(key)
        df, fdr, n_mono = cache[key]
        return (df.copy(), dict(fdr), n_mono)            # serve copies — downstream mutates feat_df

    def _fl_cache_put(self, key, result):
        cache = getattr(self, "_fl_cache", None)
        if cache is None:
            cache = self._fl_cache = OrderedDict()
        df, fdr, n_mono = result
        cache[key] = (df.copy(), dict(fdr), n_mono)      # store copies — caller keeps mutating the live df
        cache.move_to_end(key)
        while len(cache) > self._FL_CACHE_MAX:
            cache.popitem(last=False)

    def _invalidate_feature_list_cache(self):
        """Drop every cached do_feature_list bundle. Called when something outside the cache
        key changes the annotation — currently the lipid DB (keyed by id(), which a
        re-imported DB could otherwise collide with on a freed address)."""
        cache = getattr(self, "_fl_cache", None)
        if cache:
            cache.clear()

    def do_feature_list(self, modal=False):
        # ``modal`` blocks the workspace behind a loader while the lipid ID runs — used when
        # the user switches feature sets, so they can't poke a half-rebuilt view mid-load.
        if not self.peaks:
            self.statusBar().showMessage("Find peaks first.")
            return
        mode = self.mode_combo.currentText()
        peaks, ds, ppm, norm = list(self.peaks), self.ds, self.ppm, self.norm
        id_ppm = self.id_ppm                                 # lipid-ID tolerance (≠ extraction)
        self._last_id_ppm = id_ppm
        roi_mask = self._active_roi_mask()       # region-scoped → score confinement vs. rest of tissue

        lipid_db = getattr(self, "_lipid_db", None)      # external DB if imported, else built-in

        # Switching back to a feature list already built this session is a hit → skip the
        # whole worker (no build, no FDR decoy storm, no modal loader): show it instantly.
        cache_key = self._feature_list_cache_key(ds, peaks, ppm, norm, mode, id_ppm,
                                                  roi_mask, lipid_db)
        if cache_key is not None:
            hit = self._fl_cache_get(cache_key)
            if hit is not None:
                self._on_feature_list(hit)
                return

        def compute():
            from .. import annotate  # lazy: keeps pandas out of GUI startup
            # One per-build isotope-score cache shared by build_feature_list and the
            # following estimate_fdr so the monoisotopic peaks' M+1/M+2 images are
            # pulled and scored once, not twice (audit plan 23, item B). Fresh per
            # compute() — never a module global (ds is mutable across builds).
            iso_cache = {}
            df = annotate.build_feature_list(ds, peaks, mode=mode, match_ppm=id_ppm,
                                             image_ppm=ppm, norm=norm, db=lipid_db,
                                             iso_cache=iso_cache)
            if roi_mask is not None:                         # "is this ion actually confined here?"
                loc = spatial.roi_localization(ds, roi_mask, peaks, tol_ppm=ppm, norm=norm)
                lut = {round(float(m), 4): (a, f) for m, a, f
                       in zip(loc["mz"], loc["roi_auc"], loc["roi_log2_fc"])}
                df["roi_auc"] = [lut.get(round(float(m), 4), (np.nan, np.nan))[0] for m in df["mz"]]
                df["roi_log2_fc"] = [lut.get(round(float(m), 4), (np.nan, np.nan))[1] for m in df["mz"]]
            mono, _ = isotopes.deisotope(peaks)              # FDR on monoisotopic peaks only
            mono_mzs = [p["mz"] for p in mono]
            # image-based target–decoy FDR (MSM scoring); ds enables the spatial/
            # spectral evidence that gives a real per-ID q-value, n_decoy capped for speed.
            # Match at the SAME recalibrated m/z the feature list used, so q-values line up.
            offset_ppm = df.attrs.get("mass_offset_ppm", 0.0)
            fdr = annotate.estimate_fdr(mono_mzs, mode=mode, ppm=id_ppm, ds=ds,
                                        norm=norm, image_ppm=ppm, n_decoy=10, db=lipid_db,
                                        iso_cache=iso_cache, offset_ppm=offset_ppm)
            qmap = {round(float(m), 4): q for m, q in zip(mono_mzs, fdr["q_values"])}
            df["fdr_q"] = [qmap.get(round(float(m), 4), np.nan) for m in df["mz"]]
            return df, fdr, len(mono)

        def finish(result, _key=cache_key):
            if _key is not None:
                self._fl_cache_put(_key, result)         # remember so the next switch-back is instant
            self._on_feature_list(result)
        self._run(compute, on_done=finish, busy="Identifying lipids…",
                  modal=bool(modal), title="Loading features…")

    def _on_feature_list(self, result):
        df, fdr, n_mono = result
        # identifying lipids is a per-ion annotation step — reveal the flat annotated
        # table even if a ◆ lipid list's class tree was showing.
        self._show_lipid_tree(False)
        self.feat_df = df
        if self.prov is not None:
            # record both tolerances honestly: the identification tolerance actually
            # used for matching, and self.ppm (the extraction window).
            self.prov.step("annotation", mode=self.mode_combo.currentText(),
                           match_ppm=getattr(self, "_last_id_ppm", self.id_ppm),
                           extraction_ppm=self.ppm,
                           recalibration_ppm=round(float(df.attrs.get("mass_offset_ppm", 0.0) or 0.0), 2),
                           n_identified=int((df["lipid"] != "").sum()),
                           annotation_fdr=round(float(fdr["fdr"]), 3))
        n_id = int((df["lipid"] != "").sum())
        n_iso = int(df["isotopologue"].sum())
        # short peak lists make the decoy/target ratio noisy — flag it rather than imply precision
        caveat = "" if fdr.get("reliable", True) else " — few matches, treat as rough"
        lv = fdr.get("levels", {})
        # how many IDs survive at each image-scored FDR tier
        tiers = " · ".join(f"{int(lv[k])}@{int(k*100)}%" for k in (0.05, 0.10, 0.20)
                           if k in lv) if lv else ""
        # surface the gated self-calibration when it fired, so the shift is visible/auditable
        offset = float(df.attrs.get("mass_offset_ppm", 0.0) or 0.0)
        recal = (f" · auto-recalibrated {offset:+.1f} ppm ({int(df.attrs.get('n_calibration_matches', 0))} matches)"
                 if offset else "")
        self.feat_info.setText(
            f"{len(df)} features ({n_mono} monoisotopic, {n_iso} isotopologues) · {n_id} identified · "
            f"annotation FDR ≈ {fdr['fdr']:.1%} on monoisotopic "        # 1 dp so a real 0.4% isn't shown as 0%
            + (f"· IDs passing: {tiers} " if tiers else "")
            + f"(target {fdr['target_rate']:.0%} vs decoy {fdr['decoy_rate']:.0%}){caveat}{recal}")
        self._render_feature_table()
        self.statusBar().showMessage("Feature list built. Double-click a row to open PubMed; "
                                     "single-click to view its ion image.")

    def _render_feature_table(self):
        df = self.feat_df
        # the confidence column shows the 0–100 score as a sortable "%"; the folded
        # label + the reasons ride along as a tooltip + cell colour (_decorate_confidence)
        # S/N sits next to m/z: it is a property of the detected peak, not of the annotation,
        # so the lipid/class/adduct/ppm/confidence block stays contiguous.
        cols = ["mz", "snr", "lipid", "class", "adduct", "ppm", "confidence_score", "msi_level",
                "isotopologue", "spatial_morans_i", "isotope_ok", "isotope_spectral",
                "isotope_spatial", "n_adducts", "alternatives"]
        heads = ["m/z", "S/N", "lipid", "class", "adduct", "ppm", "confidence", "MSI",
                 "isotopologue?", "spatial (Moran I)", "isotope?", "iso fit", "iso co-loc",
                 "#adducts", "alternatives"]
        if "snr" not in df.columns:                       # lists saved before S/N was shown
            heads.pop(cols.index("snr")); cols.remove("snr")
        # only show the new isotope-evidence columns when present (older lists may lack them)
        if "isotope_spectral" not in df.columns:
            for c in ("isotope_spectral", "isotope_spatial"):
                heads.pop(cols.index(c)); cols.remove(c)
        if "msi_level" not in df.columns:                 # older lists predate the MSI level
            heads.pop(cols.index("msi_level")); cols.remove("msi_level")
        if "mz_calibrated" in df.columns:                 # self-cal correction, next to raw mz
            at = cols.index("mz") + 1
            cols.insert(at, "mz_calibrated")
            heads.insert(at, "m/z (cal.)")
        if "fdr_q" in df.columns:                         # per-ID target–decoy q-value
            at = cols.index("confidence_score") + 1
            cols.insert(at, "fdr_q")
            heads.insert(at, "FDR q")
        # MS/MS is identification evidence, so it sits just before the confidence it feeds.
        # Anchored by name: a fixed index moved whenever an optional column above it appeared.
        if "msms" in df.columns:
            at = cols.index("confidence_score")
            cols.insert(at, "msms")
            heads.insert(at, "MS/MS")
        if "msms_lib" in df.columns:                      # spectral-library match (plan 04)
            at = (cols.index("msms") + 1) if "msms" in cols else cols.index("confidence_score")
            cols.insert(at, "msms_lib")
            heads.insert(at, "MS/MS (library)")
        # region-scoped lists carry a confinement score: how much higher each ion is
        # inside the ROI than in the rest of the tissue (signed log2 fold-change + ROC AUC,
        # where 0.5 = no preference, →1 = confined here). Surface it right after ppm.
        roi_cols = []
        if "roi_auc" in df.columns:
            at = cols.index("ppm") + 1
            cols[at:at] = ["roi_log2_fc", "roi_auc"]
            heads[at:at] = ["ROI log2 FC", "ROI AUC"]
            roi_cols = [cols.index("roi_log2_fc"), cols.index("roi_auc")]
        conf_col = cols.index("confidence_score")
        q_col = cols.index("fdr_q") if "fdr_q" in cols else None
        msi_col = cols.index("msi_level") if "msi_level" in cols else None
        rows = [[r[c] for c in cols] for _, r in df.iterrows()]
        for row in rows:                                  # render the score as "82%"
            v = row[conf_col]
            row[conf_col] = f"{int(v)}%" if isinstance(v, (int, float)) and v == v else ""
            if msi_col is not None:                       # MSI identification level as "L2".."L5"
                mv = row[msi_col]
                row[msi_col] = f"L{int(mv)}" if isinstance(mv, (int, float)) and mv == mv else ""
            if q_col is not None:                          # q-value as a percent, blank if unscored
                qv = row[q_col]
                row[q_col] = f"{float(qv):.0%}" if isinstance(qv, (int, float)) and qv == qv else ""
            for rc in roi_cols:                           # numeric, blank when NaN (unscored)
                fv = row[rc]
                row[rc] = f"{float(fv):.2f}" if isinstance(fv, (int, float)) and fv == fv else ""
        self._feat_repopulating = True
        try:
            fill_table(self.feat_table, heads, rows)
            if msi_col is not None:
                set_header_tooltips(self.feat_table, {"MSI": (
                    "Metabolomics-Standards-Initiative identification level (Schymanski 2014):\n"
                    "L1 = confirmed (MS/MS or standard) · L2 = probable (clean isotope + spatial "
                    "match) · L3 = tentative / isobaric tie · L4 = formula only · L5 = unknown "
                    "(no database match).")})
            self._decorate_feature_swatches()
            self._decorate_confidence(conf_col)
        finally:
            self._feat_repopulating = False
        self._apply_feature_filter()
        self._sync_feature_consumers()
        self._update_selected_feature_ui()

    # colour + tooltip for the confidence column: green/amber/grey by label, with the
    # mass/isotope/adduct/runner-up breakdown spelled out on hover.
    _CONF_COLORS = {"High": "#2e8b57", "Medium": "#c08a00", "Low": "#9a9a9a"}

    def _decorate_confidence(self, conf_col):
        t = self.feat_table
        if self.feat_df is None or conf_col is None:
            return
        info = {round(float(r["mz"]), 4): (str(r.get("confidence", "")),
                                           str(r.get("confidence_why", "")))
                for _, r in self.feat_df.iterrows()}
        t.blockSignals(True)
        for row in range(t.rowCount()):
            cell = t.item(row, conf_col)
            if cell is None:
                continue
            mz = self._feat_mz_at(row)
            label, why = info.get(round(mz, 4), ("", "")) if mz is not None else ("", "")
            if why:
                cell.setToolTip(f"{label} confidence — {why}"
                                if label and label != "unidentified" else why)
            # don't override the grey that _decorate_feature_swatches gives hidden rows
            m0 = t.item(row, 0)
            hidden = m0 is not None and m0.checkState() == QtCore.Qt.Unchecked
            if not hidden and label in self._CONF_COLORS:
                cell.setForeground(QtGui.QBrush(QtGui.QColor(self._CONF_COLORS[label])))
        t.blockSignals(False)

    # ----- unified feature-set selector (working scopes + saved lists) ----- #
    def _refresh_feature_set_combo(self):
        """Rebuild the one selector that drives feature-set choice: the working scopes
        ('All slide' + every region with its own picked list) first, then the saved
        library lists (prefixed ★) below a separator. Each item carries
        ('scope'|'list', name) so a pick is dispatched without parsing the label. The
        active set shows as the current item; disabled only when there's nothing in it.

        This is the single chokepoint every feature-list mutation funnels through (save /
        rename / delete / load, flow auto-saves, find-peaks), so it also refills the
        'Feature lists' tab table — otherwise a list saved from anywhere but that tab left
        it stale until the manual Refresh button."""
        if hasattr(self, "_fill_library_table"):
            self._fill_library_table()                    # keep the Feature lists tab in sync
        combo = getattr(self, "feat_set_combo", None)
        if combo is None:
            return                                        # right dock not built yet
        scopes = self._display_scope_names()
        saved = sorted(self._feature_lists)
        combo.blockSignals(True)
        combo.clear()
        for name in scopes:
            combo.addItem(name, ("scope", name))
        if scopes and saved:
            combo.insertSeparator(combo.count())
        for name in saved:
            combo.addItem(f"★ {name}", ("list", name))
        # lipid lists (◆): the class-level twin of ★ lists, in the same dropdown but a
        # distinct kind so picking one paints classes on the tissue instead of loading ions.
        lipid = sorted(getattr(self, "_lipid_lists", {}) or {})
        if lipid and (scopes or saved):
            combo.insertSeparator(combo.count())
        for name in lipid:
            combo.addItem(f"◆ {name}", ("lipidlist", name))
        combo.blockSignals(False)
        combo.setEnabled(combo.count() > 0)
        self._set_feature_set_current()               # point at the active set
        if hasattr(self, "_refresh_cohort_feature_combos"):
            self._refresh_cohort_feature_combos()     # cohort combos pick up the saved/renamed list
        # SHAP's Features combo lives in an always-visible tab, so — unlike the ROI / class-
        # compare dialogs, which refresh their combo on open — nothing else refreshes it.
        # Do it here so a list saved/loaded anywhere shows up in SHAP without a reopen.
        shap_combo = getattr(self, "shap_feat", None)
        if shap_combo is not None and hasattr(self, "_refresh_feature_combo"):
            try:
                self._refresh_feature_combo(shap_combo)
            except RuntimeError:                      # underlying C++ widget gone
                pass

    def _display_scope_names(self):
        """Scope names to show in the selector, in a stable order: 'All slide' first, then
        one entry per *existing* region (in region order) that has a picked feature list,
        then the active scope if it somehow isn't already covered.

        A scope is keyed by the region name it was picked under (see ``ion._on_peaks``).
        Scopes whose region no longer exists — e.g. an auto-named 'ROI 1/2/3' left behind
        after the region was renamed or deleted, or carried over from an older session —
        are *hidden* here so the selector never shows stale names. ``_reconcile_feature_scopes``
        prunes them from storage once the regions are known to be authoritative."""
        have = getattr(self, "_feature_scopes", {}) or {}
        names = []
        if "All slide" in have:
            names.append("All slide")
        for rg in getattr(self, "regions", []) or []:
            nm = rg.get("name")
            if nm in have and nm not in names:
                names.append(nm)
        active = getattr(self, "_active_feature_scope", None)
        if active in have and active not in names:    # keep the active scope visible even if orphaned
            names.append(active)
        return names

    def _reconcile_feature_scopes(self):
        """Drop per-region feature scopes whose region no longer exists, so the persisted
        session stops carrying stale 'ROI 1/2/3' relics. 'All slide' and the active scope
        are always kept. Call this only once regions are authoritative (e.g. at the end of
        a session restore) — never mid-load, when ``self.regions`` may not be populated yet.
        Returns the number of orphaned scopes removed."""
        scopes = getattr(self, "_feature_scopes", None)
        if not scopes:
            return 0
        keep = {"All slide", getattr(self, "_active_feature_scope", None)}
        keep |= {rg.get("name") for rg in (getattr(self, "regions", []) or [])}
        orphans = [k for k in list(scopes) if k not in keep]
        for k in orphans:
            scopes.pop(k, None)
        return len(orphans)

    def _set_feature_set_current(self):
        """Point the selector at the active set WITHOUT rebuilding its items. Safe to
        call from inside the combo's own ``activated`` handler — a clear()/re-add there
        resets the index to 0 ('All slide') and fights Qt's in-flight popup selection,
        which is what made the box snap back to the first item. Items don't change when
        you merely *switch* sets, so just move the current index."""
        combo = getattr(self, "feat_set_combo", None)
        if combo is None:
            return
        # An active scope is the source of truth (so e.g. saving the working set as a
        # named list doesn't yank the selector onto the ★ list while you're still
        # working in the scope). Only show a ★ list as current when one is genuinely
        # loaded — i.e. no scope is active.
        flist = getattr(self, "_flist_name", None)
        lipid = getattr(self, "_active_lipid_list", None)
        if self._active_feature_scope is not None:
            want = ("scope", self._active_feature_scope)
        elif lipid in (getattr(self, "_lipid_lists", {}) or {}):
            want = ("lipidlist", lipid)
        elif flist in self._feature_lists:
            want = ("list", flist)
        else:
            return
        # NB: scan items by hand rather than QComboBox.findData — findData compares the
        # tuple payloads unreliably in this PySide6 build (returns -1 even on an exact
        # match), which left the box stuck on the first item ('All slide').
        idx = next((i for i in range(combo.count()) if combo.itemData(i) == want), -1)
        if idx >= 0 and idx != combo.currentIndex():
            combo.blockSignals(True)
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def _on_feature_set_activated(self, idx):
        """User picked an item → switch to that working scope or load that saved list.
        Separators (no item data) are ignored."""
        data = self.feat_set_combo.itemData(idx)
        if not data:
            return
        kind, name = data
        if kind in ("scope", "list"):
            self._default_feature_set = data              # latest selection → analysis-popup
                                                          # default; set BEFORE the load so any
                                                          # sync it triggers adopts the new default
        if kind == "scope":
            self._switch_feature_scope(name)
        elif kind == "lipidlist":
            self._load_lipid_list(name)
        else:
            self._load_feature_list(name)
        if kind in ("scope", "list") and hasattr(self, "_sync_default_feature_selectors"):
            self._sync_default_feature_selectors()        # untouched analysis selectors follow
        # With no region selected, the chosen set should stand alone in the spectrum:
        # drop any lingering region / ROI / per-pixel overlays so only the base
        # (mean/skyline) trace and this set's peak markers remain at the bottom. When
        # regions ARE selected their traces are intentional, so leave them.
        if not self._selected_region_indices():
            self._clear_region_spectra()                  # region mean-spectrum traces
            self._remove_overlays("ROI", prefix=True)     # drawn-ROI traces
            self._remove_overlays("pixel", prefix=True)   # per-pixel probe traces

    def _set_default_feature_set(self, data=None):
        """Pin a feature set as the app-wide default for every analysis popup, and
        remember it across sessions — so 'working off one list' needs no re-picking. (The
        default also tracks your latest selection automatically; this makes it stick.)

        ``data`` is a ``(kind, name)`` tuple when a caller (e.g. the Feature lists tab's
        context menu) targets a specific row; the ⋯-menu slot passes a bool 'checked', so
        anything that isn't a real tuple falls back to the dock selector's current item."""
        if not isinstance(data, (tuple, list)):           # QAction slot passes a bool
            combo = getattr(self, "feat_set_combo", None)
            data = combo.currentData() if combo is not None else None
        if not data or data[0] not in ("scope", "list"):
            self.statusBar().showMessage(
                "Pick a working scope or a ★ saved list in the selector first, then set it as default.")
            return
        self._default_feature_set = data
        kind, name = data
        try:
            from .. import prefs
            prefs.set("default_feature_list", {"kind": kind, "name": name})
        except Exception:  # noqa: BLE001
            pass
        if hasattr(self, "_sync_default_feature_selectors"):
            self._sync_default_feature_selectors()        # apply across tabs now, no reopen needed
        self.statusBar().showMessage(
            f"Default feature set → “{name}”. New analyses default to it (remembered across sessions).")

    def _selected_saved_list(self):
        """Name of the saved list currently chosen in the selector, or None when the
        current item is a working scope — rename/delete/load apply only to saved lists."""
        combo = getattr(self, "feat_set_combo", None)
        data = combo.currentData() if combo is not None else None
        return data[1] if data and data[0] == "list" else None

    def _switch_feature_scope(self, name):
        """Make ``name`` the active working scope — every pipeline reads ``self.peaks``,
        so swapping it is all it takes to 'work with this sample's features only'."""
        if not name or name not in self._feature_scopes or name == self._active_feature_scope:
            return
        self._active_feature_scope = name
        self._active_lipid_list = None                # a working scope, not a lipid list
        self._flist_name = "All slide" if name == "All slide" else f"{name} (sample)"
        # Hand the scope's own list to _set_peaks (it re-sorts into a fresh list and
        # writes it straight back to this scope) so the working set and the scope stay
        # the same object — edits made from here on persist without a copy step.
        peaks = self._feature_scopes[name]
        # The rebuild (cube access + table/marker refresh) is synchronous, so show a brief
        # modal loader over it — the user can't interrupt it, and it gives clear feedback.
        with self._busy_popup(f"Switching to {name} ({len(peaks)} features)…",
                              title="Loading features…"):
            # extract=False: don't re-stream the whole slide to rebuild the feature matrix on a
            # cosmetic switch — nothing here reads ds._feat; ion images / analyses re-extract
            # lazily (and single-ion display is cube-backed). Mirrors the lipid-list load path.
            self._set_peaks(peaks, reannotate=False, extract=False,
                            msg=f"Working with {name} features ({len(peaks)}).")
        self._set_feature_set_current()               # selection only — items unchanged

    # ----- feature-list management (named lists, add/remove, import, filter) -- #
    def _current_features(self):
        """Working features as ``[{mz, lipid, note}]`` from the picked peaks."""
        return [{"mz": float(p["mz"]), "lipid": self.annotate(p["mz"]) or "", "note": ""}
                for p in self.peaks]

    def _mean_spec_ctx(self):
        """The loop-invariant inputs to :meth:`_peak_from_mz` — ``(axis, spec, base, noise)``
        for the cached mean spectrum. Loading a list / importing targets snaps many m/z against
        the *same* mean spectrum, so ``spec.max()`` and the robust ``_mad`` noise are computed
        once here and threaded into each ``_peak_from_mz`` call rather than recomputed over the
        whole axis per feature (that per-feature recompute was the dominant cost of a big load)."""
        axis, spec = self.ds.mean_spectrum()
        base = float(spec.max()) or 1.0
        noise = _mad(spec)            # same robust MAD noise as pick_peaks, so SNR is comparable
        return axis, spec, base, noise

    def _peak_from_mz(self, mz, ctx=None):
        """Build a peak record for a manually-added m/z, reading height/SNR off the mean
        spectrum. Pass ``ctx`` (from :meth:`_mean_spec_ctx`) when building many peaks at once so
        the spectrum-wide reductions are shared instead of repeated per feature."""
        axis, spec, base, noise = ctx if ctx is not None else self._mean_spec_ctx()
        # nearest bin via binary search on the ascending m/z axis (was a full-axis argmin)
        i = int(np.searchsorted(axis, mz))
        if i <= 0:
            i = 0
        elif i >= len(axis):
            i = len(axis) - 1
        elif (mz - axis[i - 1]) <= (axis[i] - mz):    # equidistant → lower index, matching argmin
            i -= 1
        inten = float(spec[i])
        return {"mz": float(mz), "intensity": inten, "rel_intensity": inten / base,
                "snr": (inten / noise if noise > 0 else 0.0)}

    def _set_peaks(self, peaks, msg=None, reannotate=True, modal=False, extract=True):
        """Replace the working peak set, refresh every peak-derived view, and
        (optionally) rebuild the annotated feature list. ``modal`` blocks the workspace
        behind a loader while that rebuild runs (set when switching feature sets).
        ``extract=False`` skips the up-front feature-matrix build (each ion image is then
        pulled lazily on first display / when an analysis needs it) — used for a big
        loaded lipid list so the load is instant instead of streaming a column per ion."""
        # Any fresh working set defaults to the flat per-ion table; _load_lipid_list
        # flips back to the class tree right after it calls us (see LipidTreeMixin).
        self._show_lipid_tree(False)
        peaks = sorted(peaks, key=lambda p: p["mz"])
        self.peaks = peaks
        # Mirror the working set back into its scope so switching feature sets and back
        # doesn't silently drop edits (add/remove/import/colour/hide). self.peaks IS the
        # scope's list afterwards, so later in-place edits (colour/hide/window) land in
        # the scope too. A loaded saved list has no active scope, so nothing is written.
        scope = getattr(self, "_active_feature_scope", None)
        if scope is not None and scope in getattr(self, "_feature_scopes", {}):
            self._feature_scopes[scope] = self.peaks
        if extract and self.ds is not None and peaks:
            self.ds.ensure_features([p["mz"] for p in peaks], tol_ppm=self.ppm, reduce=self.reduce)
        self._populate_peak_table()
        self._populate_peak_combos()
        self._build_class_map()
        self._mark_peaks()
        if msg:
            self.statusBar().showMessage(msg)
        if reannotate and self.ds is not None and peaks:
            self.do_feature_list(modal=modal)
        elif not peaks:
            self.feat_df = None
            fill_table(self.feat_table, ["m/z"], [])
        else:
            # Deferred ID (scope switch / saved-list load / add / import / remove pass
            # reannotate=False): the previous set's annotated table + FDR no longer describe
            # these peaks, so drop it. The quick _populate_peak_table view above is now the live
            # one; the annotated columns + FDR fill in when 'Identify lipids' runs. Centralised
            # here so every deferred path stops leaking a stale feat_df into export / reports /
            # MS-MS (the load path used to clear it by hand; the switch path never did).
            self.feat_df = None
            # Skip the flat-table caption for a ◆ lipid list — it shows a class tree, sets its
            # own status, and 'Identify lipids' doesn't apply to class composites.
            if getattr(self, "feat_info", None) is not None and not getattr(
                    self, "_active_lipid_list", None):
                self.feat_info.setText(
                    f"{len(peaks)} features — click ‘Identify lipids’ to annotate (with FDR).")
        if hasattr(self, "_refresh_action_states"):
            self._refresh_action_states()
        self.peaksChanged.emit()

    def _add_feature(self):
        if self.ds is None:
            self.statusBar().showMessage("Load a dataset first.")
            return
        default = float(self.active_mz) if self.active_mz is not None else 0.0
        mz, ok = QtWidgets.QInputDialog.getDouble(
            self, "Add feature", "m/z:", default, 0.0, 1e6, 4)
        if not ok:
            return
        if mz <= 0:
            self.statusBar().showMessage("Enter a positive m/z to add.")
            return
        lo, hi = self.ds.mz_range
        if not (lo <= mz <= hi):
            self.statusBar().showMessage(
                f"m/z {mz:.4f} is outside the acquired range {lo:.2f}–{hi:.2f}.")
            return
        if any(abs(p["mz"] - mz) < 1e-4 for p in self.peaks):
            self.statusBar().showMessage(f"m/z {mz:.4f} is already in the list.")
            return
        self.record_undo("add feature")
        # reannotate=False: don't re-run the lipid-ID + FDR storm on every add (it's an explicit
        # 'Identify lipids' step now, matching load). extract=False: the new column is pulled
        # lazily on display (cube-backed) instead of re-streaming the whole slide here.
        self._set_peaks(self.peaks + [self._peak_from_mz(mz)], msg=f"Added feature m/z {mz:.4f}.",
                        reannotate=False, extract=False)
        self.set_active_mz(mz)

    def _selected_feature_mzs(self):
        """m/z of the selected feature rows, robust to full-row vs single-cell vs
        current-row selection (the table is sortable, so m/z is read off the cell).
        Empty when nothing is selected."""
        rows = {ix.row() for ix in self.feat_table.selectionModel().selectedRows()}
        if not rows:
            rows = {it.row() for it in self.feat_table.selectedItems()}
        if not rows and self.feat_table.currentRow() >= 0:
            rows = {self.feat_table.currentRow()}
        return [m for m in (self._feat_mz_at(r) for r in sorted(rows)) if m is not None]

    def _save_selected_as_feature_list(self):
        """Save just the highlighted feature rows as a new named ★ list, leaving the
        working set untouched (it's not a scope edit — nothing is removed). Mirrors the
        Stats tab's 'Build feature list from N selected'; pick the list from the
        feature-set selector to work in it."""
        mzs = self._selected_feature_mzs()
        if not mzs:
            self.statusBar().showMessage("Select feature row(s) to save as a list.")
            return
        feats = [{"mz": float(m), "lipid": self.annotate(m) or "", "note": ""} for m in mzs]
        saved = self._save_feature_list_to_library("Selected features", feats, prompt=True)
        if not saved:                                     # user cancelled the name prompt
            return
        self.statusBar().showMessage(
            f"Saved feature list '{saved}' from {len(feats)} selected feature(s) — "
            "pick it from the feature-set selector to work in it.")

    def _save_visible_as_feature_list(self):
        """Save just the eye-ticked (visible) features as a new named ★ list — the easy
        way to whittle a list down: hide the ions you don't want with the eye toggle, then
        keep the rest. Strictly the visible set (no fall-back to all), and the working set
        is left untouched."""
        vis = [p for p in self.peaks if not p.get("hidden")]
        if not vis:
            self.statusBar().showMessage(
                "No visible (eye-ticked) features — tick the ions to keep first.")
            return
        feats = [{"mz": float(p["mz"]), "lipid": self.annotate(p["mz"]) or "", "note": ""}
                 for p in vis]
        saved = self._save_feature_list_to_library("Visible features", feats, prompt=True)
        if not saved:                                     # user cancelled the name prompt
            return
        self.statusBar().showMessage(
            f"Saved feature list '{saved}' from {len(feats)} visible feature(s) — "
            "pick it from the feature-set selector to work in it.")

    def _remove_selected_features(self):
        mzs = self._selected_feature_mzs()
        if not mzs:
            self.statusBar().showMessage("Select feature row(s) to remove.")
            return
        # for each selected m/z, drop the single nearest peak within 0.01 Da
        drop = set()
        for m in mzs:
            best, best_d = None, 0.01
            for i, p in enumerate(self.peaks):
                if i in drop:
                    continue
                d = abs(p["mz"] - m)
                if d <= best_d:
                    best, best_d = i, d
            if best is not None:
                drop.add(best)
        if not drop:
            self.statusBar().showMessage("No matching feature(s) to remove.")
            return
        keep = [p for i, p in enumerate(self.peaks) if i not in drop]
        self.record_undo("remove features")
        # reannotate=False: editing the list defers ID to the explicit 'Identify lipids' step
        # (same contract as add/import), so a removal doesn't trigger the FDR storm. Extraction
        # stays on — a removal is a subset of the cached matrix, so it's a cheap column slice.
        self._set_peaks(keep, msg=f"Removed {len(drop)} feature(s).", reannotate=False)

    def _clear_features(self):
        """Clear the feature list *from view* without destroying it. The working table,
        its annotations, and the active selection are emptied, but the underlying list —
        the active scope's peaks or the loaded saved list — is left intact and stays in
        the feature-set selector above, so it can be reopened anytime (or restored with
        ⌘Z). To actually remove a saved list, use the ⋯ menu's 'Delete list'."""
        if not self.peaks:
            self.statusBar().showMessage("No features in view to clear.")
            return
        n = len(self.peaks)
        self.record_undo("clear features from view")
        # Detach from the active scope / loaded list FIRST so emptying the working set
        # below can't write an empty list back over the stored one (see _set_peaks): the
        # scope keeps its peaks and the saved list keeps its entry — both stay pickable.
        kept = self._active_feature_scope or getattr(self, "_flist_name", None)
        self._active_feature_scope = None
        self._set_flist_name(None)
        self._deselect_feature()                      # stop pulsing, clear halo + active m/z
        self._set_peaks([], msg=(
            f"Cleared {n} features from view — reopen '{kept}' from the feature-set "
            "selector, or find peaks to build a new list." if kept else
            f"Cleared {n} features from view — find peaks to build a new list."))
        self.feat_info.setText("Find peaks, then identify lipids.")
        self._refresh_feature_set_combo()
        # leave nothing current in the set picker so re-choosing the same list re-fires
        combo = getattr(self, "feat_set_combo", None)
        if combo is not None:
            combo.blockSignals(True)
            combo.setCurrentIndex(-1)
            combo.blockSignals(False)
        if getattr(self, "color_overlay_chk", None) is not None and self.color_overlay_chk.isChecked():
            self.refresh_ion_image()                  # drop the now-empty overlay

    def _import_targets(self):
        if self.ds is None:
            self.statusBar().showMessage("Load a dataset first.")
            return
        path, _ = filedialogs.get_open_file_name(
            self, "Import target m/z list", "", "Target list (*.csv *.txt *.tsv);;All files (*)")
        if not path:
            return
        with open(path, encoding="utf-8-sig") as f:
            targets = library.parse_target_mzs(f.read())
        if not targets:
            self.statusBar().showMessage("No m/z values found in that file.")
            return
        # drop targets the instrument never acquired (otherwise _peak_from_mz snaps
        # them to an edge bin and fabricates a feature with bogus intensity)
        lo, hi = self.ds.mz_range
        in_range = [m for m in targets if lo <= m <= hi]
        n_oob = len(targets) - len(in_range)
        existing = [p["mz"] for p in self.peaks]
        ctx = self._mean_spec_ctx()                       # snap all targets against one mean spectrum
        added = [self._peak_from_mz(m, ctx) for m in in_range
                 if not any(abs(m - e) < 1e-4 for e in existing)]
        oob_note = f" ({n_oob} outside {lo:.1f}–{hi:.1f} skipped)" if n_oob else ""
        if not added:
            self.statusBar().showMessage(
                f"No new in-range targets to add{oob_note}." if n_oob
                else "All target m/z were already in the list.")
            return
        self.record_undo("import targets")
        # reannotate=False / extract=False: same deferral as add — show the new rows instantly,
        # ID + matrix materialise on demand instead of an FDR storm + whole-slide re-stream here.
        self._set_peaks(self.peaks + added, reannotate=False, extract=False,
                        msg=f"Imported {len(added)} target(s) from {os.path.basename(path)}{oob_note}.")

    @staticmethod
    def _norm_feature_list(features):
        """Normalize an arbitrary feature iterable to ``[{mz, lipid, note}]`` sorted by
        m/z, dropping non-numeric / duplicate m/z. Accepts dicts or bare m/z floats."""
        out, seen = [], set()
        for f in features or []:
            try:
                mz = round(float(f["mz"] if isinstance(f, dict) else f), 6)
            except (TypeError, ValueError, KeyError):
                continue
            if mz <= 0 or mz in seen:
                continue
            seen.add(mz)
            lipid = str(f.get("lipid", "")) if isinstance(f, dict) else ""
            note = str(f.get("note", "")) if isinstance(f, dict) else ""
            out.append({"mz": mz, "lipid": lipid, "note": note})
        out.sort(key=lambda d: d["mz"])
        return out

    def _save_feature_list_to_library(self, base_name, features, activate=False,
                                      prompt=False):
        """Save ``features`` ([{mz, lipid, note}] or m/z floats) into this sample's
        session under a unique name derived from ``base_name`` and refresh the saved-list
        selector. Returns the final name. Shared by the auto-named generators
        (co-localization, distinct/shared Venn) so they register like a manual Save as….

        ``prompt`` asks the user to name the list (pre-filled with the auto-derived name,
        so Enter accepts it) instead of saving silently — for explicit 'make a feature
        list' actions. Returns ``None`` if the user cancels the prompt."""
        suggested = session.unique_feature_list_name(self._feature_lists, base_name)
        if prompt:
            name, ok = QtWidgets.QInputDialog.getText(
                self, "New feature list", "Name:", text=suggested)
            if not (ok and name.strip()):
                return None
            name = session.unique_feature_list_name(self._feature_lists, name.strip())
        else:
            name = suggested
        self._feature_lists[name] = self._norm_feature_list(features)
        self._mark_dirty()
        # every named ★ list (selected rows, AUC cutoff, Venn compartment, co-loc module,
        # classifier markers, combine/merge…) funnels through here — mirror it into the
        # Report tab as a named feature-table item so the report captures lists made from
        # an analysis. Best-effort + de-duplicated; never blocks the save.
        self._log_feature_list_to_report(name, self._feature_lists[name])
        if activate:
            # treat it as a loaded saved list, not a scope edit: drop the active scope
            # so the working set the caller is about to install (via _set_peaks) can't
            # write back over the scope it was generated from.
            self._active_feature_scope = None
            self._set_flist_name(name)
        self._refresh_feature_set_combo()         # registers the list (current if activated)
        return name

    def _save_feature_list_as(self):
        if not self.peaks:
            self.statusBar().showMessage("Nothing to save — pick or add features first.")
            return
        # always a fresh suggested name (don't reuse the combo's current text — that made
        # it easy to accidentally re-save a stray name like "40")
        default = f"Feature list {len(self._feature_lists) + 1}"
        name, ok = QtWidgets.QInputDialog.getText(self, "Save feature list", "Name:", text=default)
        if not (ok and name.strip()):
            return
        nm = name.strip()
        self._feature_lists[nm] = self._norm_feature_list(self._current_features())
        self._mark_dirty()
        self._log_feature_list_to_report(nm, self._feature_lists[nm])   # mirror into the Report tab
        # Saving while working in a scope is just a snapshot — stay in the scope so the
        # selector keeps pointing at the live set (pick the ★ list later to work in it).
        # Working from a loaded list, the new name becomes the active list.
        if self._active_feature_scope is None:
            self._set_flist_name(nm)
        self._refresh_feature_set_combo()
        self.statusBar().showMessage(f"Saved feature list '{nm}' ({len(self.peaks)} features).")

    def _load_feature_list(self, name=None):
        if self.ds is None:
            self.statusBar().showMessage("Load a dataset first.")
            return
        if not isinstance(name, str):                 # button/menu slots pass a bool 'checked'
            name = self._selected_saved_list()
        if not name:
            self.statusBar().showMessage("Select a saved list (★) to load.")
            return
        feats = self._feature_lists.get(name, [])
        if not feats:
            self.statusBar().showMessage(f"Feature list '{name}' is empty.")
            return
        # Saved (★) lists are auto-generated from many sources — cohort/multi-group
        # comparisons, co-localization, classifier markers — so they can carry m/z this
        # particular sample never acquired. Those snap to an edge bin in _peak_from_mz and
        # render as blank ion images ("0 ions shown"). Keep only the m/z inside this
        # dataset's acquired range (same guard as Import targets) and report what dropped.
        lo, hi = self.ds.mz_range
        in_range = [f for f in feats if lo <= float(f["mz"]) <= hi]
        n_oob = len(feats) - len(in_range)
        if not in_range:
            self.statusBar().showMessage(
                f"Feature list '{name}': none of its {len(feats)} m/z fall in this sample's "
                f"acquired range {lo:.1f}–{hi:.1f}, so there's nothing to show here.")
            return
        self.record_undo("load feature list")
        self._active_feature_scope = None             # a saved list isn't a working scope
        self._active_lipid_list = None                # …nor a lipid list
        self.feat_df = None                           # the previous list's annotated table no longer
                                                      # applies; loading no longer auto-IDs, so drop it
                                                      # (export/reports fall back to raw centroid markers
                                                      # until 'Identify lipids' runs — see export_features)
        ctx = self._mean_spec_ctx()                       # one mean spectrum for the whole list
        peaks = [self._peak_from_mz(float(f["mz"]), ctx) for f in in_range]
        oob_note = f" · {n_oob} outside {lo:.1f}–{hi:.1f} skipped" if n_oob else ""
        # reannotate=False: loading a list no longer auto-runs the lipid-ID worker — it's an
        # explicit step now (the 'Identify lipids' button), matching scope / lipid-list loads.
        # The quick m/z table shows at once; FDR/confidence scoring waits until it's asked for.
        # extract=False: pull ion columns lazily on display rather than streaming a column per ion.
        self._set_peaks(peaks, msg=f"Loaded feature list '{name}' ({len(peaks)} features){oob_note}. "
                                   "Segmentation and every pipeline now use this list.",
                        reannotate=False, extract=False)
        self._set_flist_name(name)
        self._set_feature_set_current()               # selection only — items unchanged
        self.feat_info.setText(
            f"Loaded ‘{name}’ ({len(peaks)} features) — click ‘Identify lipids’ to annotate (with FDR).")
        if peaks:
            self.set_active_mz(peaks[0]["mz"])

    def _resolve_saved_list_for_action(self, verb):
        """The saved ★ list a ⋯-menu action (rename / delete) should target. Prefer the
        list currently shown in the feature-set selector — but that selector usually points
        at a *working scope* ('All slide' / a region), in which case nothing renameable is
        'selected', so fall back to the saved lists themselves: use the only one if there's
        just one, else pop a small chooser. Returns None (with a hint) when no saved ★ list
        exists yet. This is the fix for Rename/Delete silently doing nothing while you were
        working in a scope (the common state right after Find peaks)."""
        name = self._selected_saved_list()
        if name:
            return name
        saved = sorted(self._feature_lists)
        if not saved:
            self.statusBar().showMessage(
                f"No saved ★ lists to {verb} yet — use 'Save as…' to make one first.")
            return None
        if len(saved) == 1:
            return saved[0]
        choice, ok = QtWidgets.QInputDialog.getItem(
            self, f"{verb.capitalize()} feature list",
            f"Saved ★ list to {verb}:", saved, 0, False)
        return choice if (ok and choice) else None

    def _rename_saved_list(self, old):
        """Rename saved ★ list ``old`` (prompting for the new name); returns the new name,
        or None if cancelled / unchanged / clashing. Shared by the dock ⋯ menu and the
        Feature lists tab context menu — both refresh through ``_refresh_feature_set_combo``,
        which also refills the tab table, so the two surfaces stay in sync."""
        if old not in self._feature_lists:
            return None
        new, ok = QtWidgets.QInputDialog.getText(
            self, "Rename feature list", "Name:", text=old)
        new = new.strip() if ok else ""
        if not new or new == old:
            return None
        if new in self._feature_lists:
            self.statusBar().showMessage(f"A feature list named '{new}' already exists.")
            return None
        self._feature_lists[new] = self._feature_lists.pop(old)
        self._mark_dirty()
        if getattr(self, "_flist_name", None) == old:
            self._set_flist_name(new)
        self._refresh_feature_set_combo()
        self.statusBar().showMessage(f"Renamed feature list '{old}' → '{new}'.")
        return new

    def _rename_feature_list(self):
        old = self._resolve_saved_list_for_action("rename")
        if old:
            self._rename_saved_list(old)

    def _delete_feature_list(self):
        name = self._resolve_saved_list_for_action("delete")
        if not name:
            return
        if not confirm(self, "Delete feature list",
                       f"Delete saved feature list '{name}'?"):
            return
        self._feature_lists.pop(name, None)
        self._mark_dirty()
        self._refresh_feature_set_combo()
        self.statusBar().showMessage(f"Deleted feature list '{name}'.")

    def _combine_feature_lists(self, names=None):
        """Union two or more saved ★ lists into a new one, collapsing duplicate m/z.
        The sources are left untouched; the merged result is saved as a fresh named
        list and registered in the selector (``_norm_feature_list`` on save already
        drops repeated m/z and sorts). Handy for OR-keeping per-region marker lists
        into the single shared working set every downstream step reads.

        ``names`` lets callers/tests pass the source lists directly; the menu slot
        passes a bool ('checked'), so anything but a real sequence pops the picker."""
        if not isinstance(names, (list, tuple)):          # menu/button slots pass a bool
            names = self._pick_feature_lists_to_combine()
        names = [n for n in (names or []) if n in self._feature_lists]
        if len(names) < 2:
            self.statusBar().showMessage("Pick at least two saved lists (★) to combine.")
            return
        merged = [f for nm in names for f in self._feature_lists.get(nm, [])]
        n_raw = len(merged)
        saved = self._save_feature_list_to_library(" + ".join(names), merged, prompt=True)
        if not saved:                                     # user cancelled the name prompt
            return
        dup = n_raw - len(self._feature_lists[saved])
        self.statusBar().showMessage(
            f"Combined {len(names)} lists into '{saved}' — "
            f"{len(self._feature_lists[saved])} unique feature(s)"
            + (f", {dup} duplicate m/z merged." if dup else "."))

    def _pick_feature_lists_to_combine(self):
        """Modal multi-select picker → the saved-list names to merge (or ``None`` if
        cancelled / fewer than two exist)."""
        names = sorted(self._feature_lists)
        if len(names) < 2:
            self.statusBar().showMessage("Need at least two saved lists (★) to combine.")
            return None
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Combine feature lists")
        v = QtWidgets.QVBoxLayout(dlg)
        v.addWidget(QtWidgets.QLabel(
            "Pick two or more lists to merge — shared m/z are kept once:"))
        lw = QtWidgets.QListWidget()
        lw.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        for nm in names:
            QtWidgets.QListWidgetItem(f"{nm}   ({len(self._feature_lists[nm])} features)", lw)
        v.addWidget(lw)
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return None
        return [names[i.row()] for i in lw.selectedIndexes()]

    def _apply_feature_filter(self):
        t = self.feat_table
        text = self.feat_filter.text().strip().lower()
        idonly = self.feat_idonly.isChecked()
        for r in range(t.rowCount()):
            show = True
            if idonly:
                lip = t.item(r, 1)
                if lip is None or not lip.text().strip() or lip.text().strip() == "—":
                    show = False
            if show and text:
                show = any(text in (t.item(r, c).text().lower())
                           for c in range(t.columnCount()) if t.item(r, c) is not None)
            t.setRowHidden(r, not show)

    def do_msms_confirm(self):
        if self.feat_df is None:
            self.statusBar().showMessage("Build the feature list first.")
            return
        path, _ = filedialogs.get_open_file_name(self, "Open MGF (fragment spectra)", "",
                                                        "MGF (*.mgf)")
        if not path:
            return
        mode = self.mode_combo.currentText()
        spectra = msms.parse_mgf(path)
        tol = 0.02
        verdicts = []
        for _, r in self.feat_df.iterrows():
            mz, cls = float(r["mz"]), r["class"]
            match = min(spectra, key=lambda s: abs(s.precursor - mz), default=None) if spectra else None
            if match is None or abs(match.precursor - mz) > tol:
                verdicts.append("")
            elif not cls:                                  # unidentified: suggest a class
                ranked = msms.classify(match, mode)
                verdicts.append(f"→ {ranked[0][0]}?" if ranked else "")
            else:
                verdict = msms.confirm_class(cls, match, mode)
                if mode == "negative":
                    chains = msms.acyl_chains(match, max_chains=3)
                    if chains:
                        verdict += " (" + ", ".join(chains) + ")"
                verdicts.append(verdict)
        self.feat_df["msms"] = verdicts
        self._render_feature_table()
        n_conf = sum(1 for x in verdicts if x == "confirmed")
        self.statusBar().showMessage(f"MS/MS: {n_conf} confirmed, "
                                     f"{sum(1 for x in verdicts if x == 'unsupported')} unsupported "
                                     f"(from {len(spectra)} spectra).")

    def do_msms_library(self):
        """Score each feature's acquired MS2 against a reference spectral library
        (cosine / modified-cosine / spectral-entropy, per the active profile; plan 04).
        Loads the acquired spectra and the library as .msp/.mgf and writes the top hit
        to an ``MS/MS (library)`` column."""
        if self.feat_df is None:
            self.statusBar().showMessage("Build the feature list first.")
            return
        from .. import profiles, specmatch
        acq_path, _ = filedialogs.get_open_file_name(
            self, "Open acquired MS/MS spectra (MSP/MGF)", "", "MS/MS (*.msp *.mgf)")
        if not acq_path:
            return
        lib_path, _ = filedialogs.get_open_file_name(
            self, "Open reference library (MSP/MGF)", "", "Library (*.msp *.mgf)")
        if not lib_path:
            return
        try:
            acquired = specmatch.load_library(acq_path)
            library = specmatch.load_library(lib_path)
        except Exception as e:  # noqa: BLE001 — a parse failure must not crash the tab
            self.statusBar().showMessage(f"Could not load spectra: {e}")
            return
        prof = profiles.active()
        metric = prof.get("ms2_metric", "entropy")
        tol = float(prof.get("ms2_tol_da", 0.02))
        labels, n_hit = [], 0
        for _, r in self.feat_df.iterrows():
            mz = float(r["mz"])
            query = min(acquired, key=lambda s: abs(s.precursor - mz), default=None)
            if query is None or abs(query.precursor - mz) > tol:
                labels.append("")
                continue
            hits = specmatch.match_spectrum_to_library(query, library, metric=metric,
                                                       tol_da=tol, top_n=1)
            if hits:
                labels.append(f"{hits[0].ref.name} ({hits[0].score:.2f})")
                n_hit += 1
            else:
                labels.append("")
        self.feat_df["msms_lib"] = labels
        self._render_feature_table()
        self.statusBar().showMessage(
            f"MS/MS library: {n_hit} of {len(self.feat_df)} features matched "
            f"(metric={metric}, {len(library)} refs).")

    def _feat_mz_at(self, row):
        # the table may be sorted, so read m/z from the cell (col 0), not the df index
        item = self.feat_table.item(row, 0)
        try:
            return float(item.text()) if item else None
        except ValueError:
            return None

    def _feature_selected(self):
        # Selecting a feature drives the spectrum cursor + ion image even before lipids
        # are identified — the m/z comes from the table cell, not the annotation df.
        rows = self.feat_table.selectionModel().selectedRows()
        if rows:
            mz = self._feat_mz_at(rows[0].row())
            if mz is not None:
                self.set_active_mz(mz)
        elif self.active_mz is not None:
            self._deselect_feature()                  # clicked empty space → stop pulsing

    def _feat_hover(self, row, _col):
        """Hovering a feature row points the Co-localization starting ion at it (no
        click needed). Scoped to that tab inside ``_coloc_hover_feature`` so it never
        disturbs the picker — or the window-wide active ion — from anywhere else."""
        fn = getattr(self, "_coloc_hover_feature", None)
        if fn is not None:
            fn(self._feat_mz_at(row))

    def _feature_double_clicked(self, row, col):
        if self.feat_df is None:
            return
        mz = self._feat_mz_at(row)
        match = self.feat_df[np.isclose(self.feat_df["mz"], mz)] if mz is not None else None
        if match is not None and len(match):
            url = match.iloc[0].get("pubmed_url", "")
            if url:
                QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))

    def _current_feature_set_name(self):
        """Filename-safe name of the active feature set, for export defaults. An active
        working scope ('All slide' or a region) is the source of truth (mirrors
        ``_set_feature_set_current``); otherwise a loaded ★ list; else a plain fallback.
        Run through the shared filesystem-safe sanitiser so it drops straight into a save
        dialog on any OS."""
        name = (getattr(self, "_active_feature_scope", None)
                or getattr(self, "_flist_name", None) or "feature_list")
        # Shared ``studio.safe_name`` so a label like 'PG 46:2' can't leave a Windows-forbidden
        # ':' in the default save name (the old inline squash only handled spaces and '/').
        return studio.safe_name(name)

    def export_features(self):
        """Export the working feature list to CSV, carrying every analysis that scored it.

        The list is the spine — the annotated table when lipids have been identified, else the
        raw centroid markers (m/z, label, intensity, S/N…) so a freshly-picked peak list is
        still exportable before ID. Every completed analysis whose result is keyed by m/z then
        joins on as a namespaced block of columns, so one CSV carries the annotation *and* the
        AUCs, q-values, loadings and co-localization scores. With no analyses recorded there is
        nothing to choose, so it saves straight away.
        """
        df = self.feat_df if self.feat_df is not None else self._centroid_marker_df()
        if df is None or not len(df):
            self.statusBar().showMessage("No features to export — find peaks first.")
            return
        attachments = self._analysis_attachments()
        if attachments:
            from .featureexport import FeatureExportDialog
            dlg = FeatureExportDialog(self, df, attachments,
                                      default_name=self._current_feature_set_name())
            if dlg.exec() and dlg.path:
                self.statusBar().showMessage(
                    f"Wrote {dlg.path} ({dlg.n_features} features × {dlg.n_columns} columns)")
            return
        default = f"{self._current_feature_set_name()}.csv"
        path, _ = filedialogs.get_save_file_name(self, "Save feature list",
                                                        default, "CSV (*.csv)")
        if path:
            df.to_csv(path, index=False, encoding="utf-8-sig")
            self.statusBar().showMessage(f"Wrote {path} ({len(df)} features)")

    def _analysis_attachments(self):
        """Every completed analysis in this sample's run store, reduced to joinable columns.

        Unusable ones are kept (greyed, with a reason) rather than filtered out — an analysis
        missing from the picker looks like it was never run. Returns ``[]`` when there is no
        run store at all (the demo, or an unsaved dataset), which is what sends
        :meth:`export_features` down its plain single-table path."""
        store = getattr(self, "run_store", None)
        if store is None:
            return []
        try:
            runs_list = store.list_runs()
        except Exception:                     # noqa: BLE001 — a bad store never blocks the export
            traceback.print_exc()
            return []
        from .. import featuretable, registry

        def name_of(run):
            sd = registry.REGISTRY.get(run.step_id)
            return sd.name if sd is not None else run.step_id

        return featuretable.attachments_from_runs(runs_list, store.load_result, name_of=name_of)

    def _centroid_marker_df(self):
        """The picked-peak centroid markers as a DataFrame — what we have before lipid ID.
        Mirrors the spectrum's centroid markers: m/z, our best label, and whatever per-peak
        metrics the picker stored (intensity, relative intensity, S/N), plus the user's
        custom label, colour, hidden flag, and scoping region."""
        import pandas as pd  # lazy: keeps pandas out of GUI startup
        if not self.peaks:
            return None
        rows = []
        for p in self.peaks:
            mz = float(p["mz"])
            rows.append({
                "mz": round(mz, 4),
                "label": p.get("label_override") or (self.annotate(mz) or ""),
                "intensity": p.get("intensity"),
                "rel_intensity": p.get("rel_intensity"),
                "snr": p.get("snr"),
                "region": p.get("region"),
                "color": p.get("color"),
                "hidden": bool(p.get("hidden")),
            })
        return pd.DataFrame(rows)

