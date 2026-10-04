"""The one dialog every registry analysis opens in (plan 24, Phase 2a).

:class:`AnalysisDialog` is the generic, modeless *configure → Run → explore result* surface
for any :class:`registry.StepDef`. It composes the pieces the dedicated tabs already use — a
:class:`~smile_msi.gui.scope.ScopeBar` for data selection, a :class:`common.ParamForm` for the
step's parameters, the shared ``win._run`` worker path, and :mod:`resultviews` for interactive
output — so a single class replaces N bespoke tab bodies without changing any analysis.

Readiness (the Run button) is driven by the same pure gate the Analyze gallery uses:
``registry.unmet_needs(win._gallery_state(), sd)``. Inputs are assembled by the pure
``registry.resolve_inputs`` from that same Qt-free state, so the dialog never re-implements the
runner's token wiring.

Phase 2a builds the base and result views only. It does NOT yet mint / persist
:class:`~smile_msi.runs.AnalysisRun` records (Phase 3) — but ``__init__`` already takes
``run`` / ``store`` so that is a fill-in, and :meth:`from_run` hydrates a saved run's params +
result from a :class:`~smile_msi.runs.RunStore` WITHOUT recomputing, flagging a stale receipt.
"""
from __future__ import annotations

import traceback

from PySide6 import QtCore, QtGui, QtWidgets

from . import common, jobqueue, resultviews
from .scope import ScopeBar
from .. import registry


def _safe_call(fn, *a, default=None):
    try:
        return fn(*a)
    except Exception:                                       # noqa: BLE001 — wiring callables are best-effort
        traceback.print_exc()
        return default


def _is_dataframe(result):
    try:
        import pandas as pd
        return isinstance(result, pd.DataFrame)
    except Exception:                                       # noqa: BLE001
        return False


def _result_table(sd, result):
    """The result's table DataFrame (``None`` if there is none).

    A result that is ALREADY a DataFrame — a step returning a table raw, or a run rehydrated
    from the store whose persisted payload is the table — is returned as-is; ``sd.to_table``
    expects the RAW engine result, so applying it to an already-tabular payload would raise."""
    if _is_dataframe(result):
        return result
    return _safe_call(sd.to_table, result, default=None)


class AnalysisDialog(QtWidgets.QWidget):
    """One self-contained view for every registry analysis: params form → Run → interactive
    results, with its own export/report actions.

    A plain widget (not a window) so it docks as a closable tab in the main strip — you can
    switch between several open analyses and come back to any of them; closing a tab keeps the
    run in Analyses ▸ History. Non-singleton — one instance per open run. Held alive by a list
    on the owning window (``win._analysis_dialogs``) until closed."""

    def __init__(self, win, sd, *, run=None, store=None, parent=None):
        super().__init__(parent)
        self.win = win
        self.sd = sd
        self.run = run
        self.store = store
        self.result = None
        self._result_widget = None
        self._sigs = []

        self.setWindowTitle(sd.name if sd is not None else "Analysis")

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(8)

        if sd is None:                                      # unknown step id — degrade, never crash
            v.addWidget(common.note("This analysis is no longer available."))
            self._keep_alive()
            return

        # header ----------------------------------------------------------- #
        title = common.section_title(sd.name)
        title.setWordWrap(True)
        v.addWidget(title)
        if sd.description:
            v.addWidget(common.note(sd.description))

        # stale banner (hidden until from_run detects drift) --------------- #
        self._stale = common.note("")
        self._stale.setObjectName("staleBanner")
        self._stale.setStyleSheet(f"color:{common.DANGER}; font-weight:600;")
        self._stale.setVisible(False)
        v.addWidget(self._stale)

        # data scope + params --------------------------------------------- #
        needs = sd.needs or set()
        self.scope = ScopeBar(
            win,
            feature=("feature_set" in needs),
            ab=("ab" in needs),
            groups=("groups" in needs or "region" in needs or sd.id == "region_membership"),
        )
        v.addWidget(self.scope)

        seed = dict(run.params) if run is not None else registry.default_params(sd.id)
        self.form = common.ParamForm(sd.params, seed,
                                     active_mz=lambda: getattr(win, "active_mz", None))
        v.addWidget(self.form)

        # run row ---------------------------------------------------------- #
        run_row = QtWidgets.QHBoxLayout()
        self._summary = common.note("")
        run_row.addWidget(self._summary, 1)
        self.b_cancel = common.button("Cancel", self._cancel, danger=True,
                                      tooltip="Stop this analysis")
        self.b_cancel.setVisible(False)
        run_row.addWidget(self.b_cancel)
        self.b_run = common.primary_button("Re-run" if run is not None else "Run", self._run)
        run_row.addWidget(self.b_run)
        v.addLayout(run_row)

        # This analysis's own progress. The status bar and the queue panel are shared, so with
        # several analyses open at once they can't say which one is working — the bar belongs
        # next to the result it is producing. Driven off jobs.changed via this dialog's run_id.
        self._active_run_id = None
        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        self.progress.setVisible(False)
        v.addWidget(self.progress)
        jobs = getattr(win, "jobs", None)
        if jobs is not None:
            jobs.changed.connect(self._sync_job_progress)
            self._sigs.append((jobs.changed, self._sync_job_progress))

        # result area + apply/report actions (populated on a result) ------- #
        self._holder = QtWidgets.QWidget()
        QtWidgets.QVBoxLayout(self._holder).setContentsMargins(0, 0, 0, 0)
        v.addWidget(self._holder, 1)

        self._actions = QtWidgets.QWidget()
        arow = QtWidgets.QHBoxLayout(self._actions)
        arow.setContentsMargins(0, 0, 0, 0)
        arow.addStretch(1)
        self._apply_holder = QtWidgets.QWidget()          # swapped for a fresh menu per result
        QtWidgets.QHBoxLayout(self._apply_holder).setContentsMargins(0, 0, 0, 0)
        arow.addWidget(self._apply_holder)
        # export / display actions available on every result kind ---------- #
        self.b_copy = common.menu_button("Copy", [
            ("Copy figure", self._copy_figure),
            ("Copy table", self._copy_table),
        ], tooltip="Copy the figure image, or the table, to the clipboard")
        arow.addWidget(self.b_copy)
        self.b_image = common.button("⤓ Image", self._export_image,
                                     tooltip="Save the result figure as a PNG / SVG image")
        arow.addWidget(self.b_image)
        self.b_report = common.button("Add to report", self._add_to_report,
                                      tooltip="Append this result to the Report book")
        arow.addWidget(self.b_report)
        self.b_hub = common.button("Export hub…", self._open_in_export_hub,
                                   tooltip="Send this result to the report and open the Export "
                                           "hub (PDF data book + batch export)")
        arow.addWidget(self.b_hub)
        self._actions.setVisible(False)
        v.addWidget(self._actions)

        # re-evaluate readiness on every state funnel + on a scope edit ----- #
        for name in ("datasetChanged", "peaksChanged", "regionsChanged", "segChanged"):
            sig = getattr(win, name, None)
            if sig is not None:
                sig.connect(self._refresh_run_enabled)
                self._sigs.append((sig, self._refresh_run_enabled))
        self.scope.changed.connect(self._refresh_run_enabled)
        self._refresh_run_enabled()

        self._keep_alive()

    # ------------------------------------------------------------------ #
    def _keep_alive(self):
        dialogs = getattr(self.win, "_analysis_dialogs", None)
        if dialogs is None:
            dialogs = self.win._analysis_dialogs = []
        dialogs.append(self)

    def _teardown(self):
        """Disconnect the signal-bus subscriptions and drop the keep-alive ref. Called when
        the analysis tab is closed (or the widget is closed as a window). Idempotent."""
        # (signal, slot) pairs — not a bare signal list: this widget subscribes with two
        # different slots, and disconnecting the wrong one would leave a live connection into
        # a deleted C++ object (PySide6 raises "Internal C++ object already deleted" later).
        for sig, slot in self._sigs:
            try:
                sig.disconnect(slot)
            except Exception:                               # noqa: BLE001
                pass
        self._sigs = []
        try:
            self.win._analysis_dialogs.remove(self)
        except (AttributeError, ValueError):
            pass

    def closeEvent(self, ev):                               # noqa: N802 (Qt API)
        self._teardown()
        super().closeEvent(ev)

    def _set_summary(self, text):
        self._summary.setText(text or "")

    # ------------------------------------------------------------------ #
    # readiness
    # ------------------------------------------------------------------ #
    def _state(self):
        """The window's gallery state with THIS dialog's scope-bar picks applied: the chosen
        feature set, the chosen groups (over the Setup tags and the segmentation fallback)
        and the A/B pair. The run, the readiness check and the staleness probe all read
        this, so what the bar says is what the analysis gets."""
        state = self.win._gallery_state()
        bar = self.scope
        if bar.feat_readout is not None:                       # bar has a feature picker
            state["mzs"] = [float(m) for m in bar.feature_mzs()]
        chosen = bar.group_regions()
        if chosen:
            regions = state.get("regions") or {}
            groups = {}
            for name in chosen:
                mk = regions.get(name)
                if mk is not None and mk.any():
                    groups[name] = [mk]
            if len(groups) >= 2:
                state["groups"] = groups
                state["seg_labels"] = None                     # chosen groups, never the fallback
                state["seg_names"] = None
        a, b = bar.ab_regions()
        if a and b:
            state["scope_a"], state["scope_b"] = [a], [b]
        return state

    def _refresh_run_enabled(self, *_):
        if self.sd is None:
            return
        try:
            reasons = registry.unmet_needs(self._state(), self.sd)
        except Exception:                                   # noqa: BLE001 — never let a bad state disable forever
            traceback.print_exc()
            reasons = []
        self.b_run.setEnabled(not reasons)
        self.b_run.setToolTip("\n".join(reasons) if reasons else "Run this analysis")

    # ------------------------------------------------------------------ #
    # run
    # ------------------------------------------------------------------ #
    def _run(self):
        win, sd = self.win, self.sd
        state = self._state()
        try:
            inputs = registry.resolve_inputs(state, sd)
        except ValueError as exc:
            self._set_summary(f"Cannot run: {exc}")
            return
        # backfill registry defaults under the form values — an older/partial preset that omits
        # a hard-indexed param won't KeyError at run (mirrors flowdialog._run_step).
        params = {**registry.default_params(sd.id), **self.form.values()}

        # Mint + persist a 'running' receipt BEFORE the work starts, so a hard crash mid-run
        # still leaves a trace in the history (plan 24 Phase 3). None when there's no store.
        self.run = self._begin_record(state, params, inputs)
        rid = self.run.run_id if self.run is not None else None
        self._active_run_id = rid

        def work(progress=None):
            # mirror flowdialog._run_step's contract: catch everything, print, return a tagged
            # payload so the worker never raises off-thread and the dialog can show the error.
            # `progress` rides in on the inputs dict: StepDef.run's (ds, inp, params) signature
            # has no channel for it, and a run callable that ignores the key is unaffected.
            try:
                return ("ok", sd.run(win.ds, {**inputs, "progress": progress}, params))
            except Exception as exc:                        # noqa: BLE001
                traceback.print_exc()
                return ("err", f"{type(exc).__name__}: {exc}")

        def done(payload):
            kind, val = payload
            if kind == "err":
                self._set_summary(val)
                self._finish_record("error", error=val)
            else:
                self.run_params = params
                self._on_done(val)
                self._finish_record("done", result=val)
                self._record_provenance(state, params)      # session audit trail + methods

        # on_cancelled / on_error close out the record on the non-success terminal paths too.
        win._run(work, on_done=done, want_progress=True, queue=True, label=sd.name,
                 run_id=rid,
                 on_cancelled=lambda: self._finish_record("cancelled"),
                 on_error=lambda msg: self._finish_record("error", error=str(msg)))
        self._sync_job_progress()   # show 'Queued…' now; jobs.changed drives it from here

    # ------------------------------------------------------------------ #
    # this analysis's own progress
    # ------------------------------------------------------------------ #
    def _sync_job_progress(self):
        """Mirror *this* dialog's job into its inline bar. Keyed on ``run_id``, so N open
        analyses each track their own work rather than the shared status bar's."""
        jobs = getattr(self.win, "jobs", None)
        job = jobs.job_for(self._active_run_id) if jobs is not None else None
        if job is None or job.is_terminal:
            self.progress.setVisible(False)
            self.b_cancel.setVisible(False)
            self._refresh_run_enabled()               # prerequisites decide, not the job
            return

        self.progress.setVisible(True)
        self.b_run.setEnabled(False)
        # A queued job can always be dropped; a running one only if its worker reports
        # progress (JobQueue.cancellable) and can therefore stop cooperatively.
        self.b_cancel.setVisible(job.status == jobqueue.JobStatus.QUEUED or job.cancellable)

        if job.status == jobqueue.JobStatus.QUEUED:
            self.progress.setRange(0, 0)              # indeterminate
            self._set_summary("Queued…")
        elif job.total:
            self.progress.setRange(0, job.total)
            self.progress.setValue(job.done)
            self._set_summary(f"Running… {round(100 * job.done / job.total)}%")
        else:
            self.progress.setRange(0, 0)
            self._set_summary("Running…")

    def _cancel(self):
        jobs = getattr(self.win, "jobs", None)
        job = jobs.job_for(self._active_run_id) if jobs is not None else None
        if job is not None:
            self.win._cancel_job(job)

    def _on_done(self, result, *, hydrated=False):
        """``hydrated`` marks a result read back from the run store rather than returned by
        ``sd.run``. Provenance cannot be inferred from the result's type: several steps
        (``roi_comparison``, ``multigroup_features``, …) return a DataFrame *raw*, so a type
        probe would rob a live run of its summary and its apply-to-session wiring."""
        self.result = result
        # A store round-trip is lossy (CSV), so the raw payload sd.summary expects is gone —
        # a rehydrated run carries its saved one-liner instead.
        if hydrated:
            summ = (self.run.summary if self.run is not None else "") or ""
        else:
            summ = _safe_call(self.sd.summary, result, default="") or ""
        self._set_summary(summ)
        self._render(result)
        self._build_actions(result, hydrated=hydrated)

    def _render(self, result):
        lay = self._holder.layout()
        if self._result_widget is not None:
            self._result_widget.setParent(None)
            self._result_widget.deleteLater()
        self._result_widget = resultviews.render_result(self.win, self.sd, result)
        lay.addWidget(self._result_widget)

    # ------------------------------------------------------------------ #
    # run-history records (plan 24 Phase 3)
    # ------------------------------------------------------------------ #
    def _scope_descriptor(self, state):
        """A compact, JSON-safe identity of the inputs this run consumed — region/group names,
        the feature m/z set, the target ion — filtered by the step's ``needs``.

        Deliberately name/id based (not raw pixel masks): keeps the embedded session index
        small and matches the plan's "inputs = region ids, groups, feature list, target m/z".
        Used both as the stored ``AnalysisRun.inputs`` and as the current-state probe in
        :meth:`_check_stale`, so the two always compare like-for-like."""
        sd = self.sd
        needs = sd.needs or set()
        d = {}
        if "feature_set" in needs:
            d["mzs"] = sorted(round(float(m), 4) for m in (state.get("mzs") or []))
        if (needs & {"groups", "region", "ab"}) or sd.id == "region_membership":
            d["groups"] = sorted((state.get("groups") or {}).keys())
            d["regions"] = sorted((state.get("regions") or {}).keys())
            if state.get("scope_a") or state.get("scope_b"):    # the chosen A/B pair
                d["ab"] = [list(state.get("scope_a") or []), list(state.get("scope_b") or [])]
        if "target_mz" in needs:
            t = state.get("target_mz")
            d["target_mz"] = (round(float(t), 4) if t is not None else None)
        if "stats" in needs:
            sdf = state.get("stats_df")
            d["stats_rows"] = (int(len(sdf)) if sdf is not None else 0)
        return d

    def _dataset_id(self):
        import os
        ds = getattr(self.win, "ds", None)
        if ds is None:
            return "", ""
        name = os.path.basename(str(getattr(ds, "source", ""))) or "dataset"
        try:
            from .. import library
            fp = library.dataset_fingerprint(ds)
        except Exception:                                   # noqa: BLE001
            fp = ""
        return name, fp

    def _run_title(self, inputs):
        """The analysis name, plus what makes *this* run of it different.

        Two 'Region comparison (A vs B)' runs are otherwise indistinguishable in History and in
        a feature-list export, because ``_scope_descriptor`` records every group on the slide
        rather than the chosen pair. The resolved inputs know the pair, so name the run after
        it. Title only — the ``inputs`` digest and staleness comparison are untouched."""
        name = self.sd.name
        needs = self.sd.needs or set()
        if not inputs:
            return name
        if "ab" in needs:
            a, b = inputs.get("a_label"), inputs.get("b_label")
            if a and b:
                return f"{name} — {a} vs {b}"
        if "target_mz" in needs and inputs.get("target_mz") is not None:
            return f"{name} — m/z {float(inputs['target_mz']):.4f}"
        return name

    def _begin_record(self, state, params, inputs=None):
        """Persist a fresh ``status='running'`` :class:`~smile_msi.runs.AnalysisRun` to the
        window's run store before the work starts. Returns ``None`` (run simply not recorded)
        when there is no persistable store — the demo, or a dataset with no managed session."""
        store = getattr(self.win, "run_store", None)
        if store is None:
            return None
        from .. import runs
        name, fp = self._dataset_id()
        targets = getattr(self.sd, "targets", {"slide"}) or {"slide"}
        scope = getattr(self.win, "_gallery_scope", "slide")
        target = scope if scope in targets else "slide"
        run = runs.new_run(self.sd.id, title=self._run_title(inputs), target=target,
                           dataset=name, dataset_fingerprint=fp,
                           inputs=self._scope_descriptor(state), params=params)
        try:
            store.save(run)
        except Exception:                                   # noqa: BLE001 — recording is best-effort
            traceback.print_exc()
        return run

    def _finish_record(self, status, *, result=None, error=""):
        """Close out the run record on a terminal path (done / error / cancelled), persisting
        the result payload + summary + thumbnail on success. Best-effort throughout."""
        run = getattr(self, "run", None)
        store = getattr(self.win, "run_store", None)
        if run is None or store is None:
            return
        run.status = status
        run.error = error or ""
        if status == "done":
            run.summary = _safe_call(self.sd.summary, result, default="") or ""
            payload = self._persist_payload(result)
            try:
                if payload is not None:
                    store.save_result(run, payload)         # persists the payload AND saves the run
                else:
                    _safe_call(store.save, run)             # no re-openable payload → still mark done
            except Exception:                               # noqa: BLE001 — unpersistable payload
                traceback.print_exc()
                _safe_call(store.save, run)                 # still record the 'done' metadata
            self._save_thumb(run, store)
        else:
            _safe_call(store.save, run)
        self._notify_runs_changed()

    #: Registry id → provenance step name (drives the auto-methods paragraph + citations).
    #: Unmapped ids record under their own id — still traceable in the methods report, just
    #: without bespoke prose. Ids that already match a recognized name are omitted.
    _PROV_KIND = {
        "colocalize": "colocalization", "coloc_modules": "colocalization",
        "region_correlation": "region_match",
        "plsda": "classification", "classify_cv": "classification",
        "classify_map": "classification", "shrunken_centroids": "classification",
        "auto_segment": "segmentation", "dgmm": "segmentation",
        "find_peaks": "peak_picking", "find_spatial_features": "spatial_feature_finding",
        "multigroup_features": "multigroup",
        "discriminating_features": "statistics", "roi_localization": "statistics",
    }

    def _record_provenance(self, state, params):
        """Log this run into the session audit trail (``self.prov`` via ``win.record_step``) so
        it feeds the methods paragraph / traceability report — the provenance the dedicated
        tabs recorded before plan 24. Best-effort; never blocks the analysis."""
        rec = getattr(self.win, "record_step", None)
        if not callable(rec):
            return
        kind = self._PROV_KIND.get(self.sd.id, self.sd.id)
        try:
            regions = sorted((state.get("regions") or {}).keys()) or None
        except Exception:                                   # noqa: BLE001
            regions = None
        try:
            rec(kind, label=self.sd.name, params=dict(params or {}), regions=regions)
        except Exception:                                   # noqa: BLE001 — audit never breaks a run
            traceback.print_exc()

    def _persist_payload(self, result):
        """The re-openable payload for this result: the structured dict for an ion-segmentation
        image (JSON round-trips and re-renders as the image), else the step's flat table (CSV
        round-trips and re-renders as a :class:`~resultviews.TableResultView`)."""
        sd = self.sd
        # a structured dict result (ion-segmentation image, spectra overlay, classifier map)
        # round-trips as JSON and re-renders from its own kind renderer. A dict carrying a
        # DataFrame does NOT — json stringifies the frame, and the run reopens holding the
        # table's repr — so those persist their flat table instead (filter_auc, marker_panel).
        from .. import runs
        if isinstance(result, dict) and runs.json_safe(result):
            return result
        return _result_table(sd, result)

    def _save_thumb(self, run, store):
        w = self._result_widget
        if w is None:
            return
        try:
            pix = w.grab()
            if pix.isNull():
                return
            ba = QtCore.QByteArray()
            buf = QtCore.QBuffer(ba)
            buf.open(QtCore.QIODevice.WriteOnly)
            pix.scaled(240, 180, QtCore.Qt.KeepAspectRatio,
                       QtCore.Qt.SmoothTransformation).save(buf, "PNG")
            store.save_thumb(run, bytes(ba.data()))
        except Exception:                                   # noqa: BLE001 — thumbnails are cosmetic
            traceback.print_exc()

    def _notify_runs_changed(self):
        """Tell the window a run record changed so the History tab can refresh and the session
        index (embedded ``analysis_runs``) is re-saved. Best-effort; both hooks are optional."""
        win = self.win
        fn = getattr(win, "_refresh_analysis_history", None)
        if callable(fn):
            win._safe_hook(fn) if hasattr(win, "_safe_hook") else _safe_call(fn)
        mark = getattr(win, "_mark_dirty", None)
        if callable(mark):
            _safe_call(mark)

    # ------------------------------------------------------------------ #
    # apply-to-session / report
    # ------------------------------------------------------------------ #
    def _build_actions(self, result, hydrated=False):
        """Show the action row, rebuilding the 'Apply to session ▾' menu for this result.

        The menu appears ONLY when the step has non-trivial ``peaks`` / ``produce`` / ``lists``
        wiring — detected behaviourally: the trivial defaults ``StepDef.__post_init__`` installs
        return ``None`` / ``{}``, so a truthy return means the step really produces something to
        apply."""
        win, sd = self.win, self.sd
        # Only a store round-trip loses the raw payload that peaks/produce/lists expect. A live
        # result is raw whatever its type, so probe it — gating on `isinstance(result, DataFrame)`
        # would silently strip apply-to-session from every step that returns a table raw.
        pk = None if hydrated else _safe_call(sd.peaks, result, default=None)
        produced = {} if hydrated else (_safe_call(sd.produce, result, default={}) or {})
        listed = {} if hydrated else (_safe_call(sd.lists, result, default={}) or {})
        regions = {} if hydrated else (_safe_call(sd.regions_out, result, self.win.ds, default={}) or {})

        items = []
        if pk:
            items.append(("Set as working peaks", lambda: self._apply_peaks(pk)))
        if sd.result_kind == "segmentation" or "segmentation" in produced:
            items.append(("Apply segmentation", lambda: self._apply_seg(result)))
        if listed:
            items.append((f"Save {len(listed)} feature list(s)",
                          lambda l=listed: self._apply_lists(l)))
        if regions:
            items.append((f"Create {len(regions)} region(s)",
                          lambda r=regions: self._apply_new_regions(r)))

        holder_lay = self._apply_holder.layout()
        while holder_lay.count():
            old = holder_lay.takeAt(0).widget()
            if old is not None:
                old.deleteLater()
        if items:
            holder_lay.addWidget(common.menu_button("Apply to session", items,
                                                    tooltip="Commit this result into the session"))

        # 'Add to report' / 'Export hub' need a table; copy-figure / export-image work on any
        # rendered result (the copy-table sub-action guards on its own).
        has_table = _result_table(sd, result) is not None
        self.b_report.setEnabled(has_table)
        self.b_hub.setEnabled(has_table)
        self.b_copy.setEnabled(self._result_widget is not None)
        self.b_image.setEnabled(self._result_widget is not None)
        self._actions.setVisible(True)

    def _apply_peaks(self, pk):
        # mirror flowdialog._route_result's peak-producer branch.
        self.win._pending_pick_region = None
        self.win._pending_pick_save = False
        self.win._on_peaks(pk)

    def _apply_seg(self, result):
        self.win._on_seg(result)

    def _apply_new_regions(self, regions):
        """Create a named region per (name → mask) entry the step produced — the
        prediction-map classes and the matched-region merges. Mirrors the tabs' record_undo
        + _new_region write-back."""
        made = 0
        if hasattr(self.win, "record_undo"):
            _safe_call(self.win.record_undo, f"Create {len(regions)} region(s)")
        for name, mask in regions.items():
            try:
                if mask is None or not getattr(mask, "any", lambda: False)():
                    continue
                self.win._new_region(name=name, mask=mask, select=False)
                made += 1
            except Exception:                               # noqa: BLE001
                traceback.print_exc()
        self.win.statusBar().showMessage(f"Created {made} region(s).")

    def _apply_lists(self, listed):
        for name, feats in listed.items():
            if not feats:
                continue
            self.win._feature_lists.pop(name, None)         # deterministic name → replace on re-run
            self.win._save_feature_list_to_library(name, feats)

    def _add_to_report(self):
        """Append the result to the Report tab. Mirrors flowdialog._log_result's table path —
        the stats/segmentation live-capture helpers there need the result routed into their
        views first, which the dialog doesn't do, so a step's own ``to_table`` is the safe,
        universal source here."""
        df = _result_table(self.sd, self.result)
        if df is None or not len(df):
            self.win.statusBar().showMessage("Nothing to add to the report.")
            return
        title = self.run.title if (self.run is not None and self.run.title) else self.sd.name
        # backlink the report item to its run record (plan 24), so the two stores can converge.
        extra = ({"run_id": self.run.run_id}
                 if (self.run is not None and getattr(self.run, "run_id", None)) else None)
        self.win._log_analysis_to_report(self.sd.name, df, title=title, source_extra=extra)
        self.win.statusBar().showMessage(f"Added “{title}” to the report.")

    # ------------------------------------------------------------------ #
    # export / display actions (available on every result kind)
    # ------------------------------------------------------------------ #
    def _copy_figure(self):
        w = self._result_widget
        if w is None:
            self.win.statusBar().showMessage("Nothing to copy.")
            return
        pix = w.grab()
        if pix.isNull():
            return
        QtWidgets.QApplication.clipboard().setPixmap(pix)
        self.win.statusBar().showMessage("Figure copied to clipboard.")

    def _copy_table(self):
        df = _result_table(self.sd, self.result)
        if df is None or not len(df):
            self.win.statusBar().showMessage("This result has no table to copy.")
            return
        QtWidgets.QApplication.clipboard().setText(df.to_csv(sep="\t", index=False))
        self.win.statusBar().showMessage(f"Copied {len(df)} rows to the clipboard.")

    def _export_image(self):
        w = self._result_widget
        if w is None:
            self.win.statusBar().showMessage("Nothing to export.")
            return
        path, _flt = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export figure", f"{self.sd.id}_result.png",
            "PNG image (*.png);;SVG image (*.svg)")
        if not path:
            return
        if path.lower().endswith(".svg") and not self._export_svg(w, path):
            path = path[:-4] + ".png"                       # no vector source → fall back to PNG
        if not path.lower().endswith(".svg"):
            w.grab().save(path)
        self.win.statusBar().showMessage(f"Saved {path}")

    @staticmethod
    def _export_svg(widget, path) -> bool:
        """Try a true vector SVG export of a contained pyqtgraph plot; False if there's no
        vector source (the caller then falls back to a PNG raster)."""
        try:
            import pyqtgraph as pg
            from pyqtgraph.exporters import SVGExporter
            plot = widget.findChild(pg.PlotItem) or widget.findChild(pg.GraphicsLayoutWidget)
            target = plot.getPlotItem() if isinstance(plot, pg.GraphicsLayoutWidget) else plot
            if target is None:
                return False
            SVGExporter(target).export(path)
            return True
        except Exception:                                   # noqa: BLE001
            traceback.print_exc()
            return False

    def _open_in_export_hub(self):
        self._add_to_report()                               # give the hub something to show
        hub = getattr(self.win, "open_export_hub", None)
        if not callable(hub):
            self.win.statusBar().showMessage("The Export hub isn't available.")
            return
        try:
            hub("book")
        except TypeError:
            hub()
        except Exception:                                   # noqa: BLE001
            traceback.print_exc()

    # ------------------------------------------------------------------ #
    # hydrate a stored run (Phase 3 persists them; here we only READ)
    # ------------------------------------------------------------------ #
    @classmethod
    def from_run(cls, win, run, store):
        """Open a dialog over a saved :class:`~smile_msi.runs.AnalysisRun` — params and result
        rehydrated from ``store``, NEVER recomputed. The Run button reads 'Re-run'. If the run
        is stale against the current dataset/inputs, a prominent banner says so; the (stale)
        result still displays — it is never passed off as fresh."""
        sd = registry.REGISTRY.get(getattr(run, "step_id", None))
        dlg = cls(win, sd, run=run, store=store)
        if sd is None:
            return dlg
        try:
            dlg.form.set_values(run.params or {})
        except Exception:                                   # noqa: BLE001
            traceback.print_exc()
        result = None
        if store is not None:
            try:
                result = store.load_result(run)
            except Exception:                               # noqa: BLE001
                traceback.print_exc()
        if result is not None:
            dlg._on_done(result, hydrated=True)             # render WITHOUT running sd.run
        dlg._check_stale(run)
        return dlg

    def _check_stale(self, run):
        fp = ""
        try:
            from .. import library
            if getattr(self.win, "ds", None) is not None:
                fp = library.dataset_fingerprint(self.win.ds)
        except Exception:                                   # noqa: BLE001
            fp = ""
        try:
            cur_inputs = self._scope_descriptor(self._state())
        except Exception:                                   # noqa: BLE001 — unmet inputs still count as drift
            cur_inputs = {}
        if run.is_stale(fp, cur_inputs):
            self._stale.setText(
                f"Computed {run.created} against state that has since changed — "
                "Re-run to refresh.")
            self._stale.setVisible(True)
