"""The Analyses ▸ History sub-tab (plan 24, Phase 3).

A live table of every :class:`~smile_msi.runs.AnalysisRun` in the loaded sample's run store
(``MainWindow.run_store``), merged with every run in the loaded **cohort**'s run store
(``MainWindow.cohort_run_store`` — nested stats, group comparison, … span every sample in
the cohort, not the one slide that happens to be open, so they're recorded separately and
folded in here rather than living only in the Report book), with per-row actions:

* **Open** — re-open the stored result in an :class:`~smile_msi.gui.analysisdialog.AnalysisDialog`
  via ``from_run`` (never recomputed; a stale banner flags drift).
* **Re-run** — open it and run again against the *current* data (mints a fresh record).
* **Duplicate…** — a fresh dialog pre-filled with the run's parameters, ready to edit + run.
* **Add to report** — append the stored table to the Report book, stamping ``source['run_id']``.
* **Delete** — remove the run and its on-disk payload (after a confirm).

The Report book sub-tab is today's :class:`~smile_msi.gui.reporttab.ReportTabMixin` tab, moved
under the same ``Analyses`` group untouched — both are nested by ``_regroup_tabs``.
"""
from __future__ import annotations

import traceback

from PySide6 import QtWidgets

from .common import ControlBar, button, confirm, fill_table, note, tab_page
from .. import registry

_COLS = ["When", "Analysis", "Dataset", "Status", "Summary"]


class HistoryMixin:
    """Lives on ``MainWindow``. Builds the History sub-tab and keeps it synced to the store."""

    def _tab_history(self):
        w, v = tab_page()
        bar = ControlBar()
        bar.add(
            button("Open", self._history_open,
                   tooltip="Re-open the selected run's stored result (no recompute)"),
            button("Re-run", self._history_rerun,
                   tooltip="Re-open and run again against the current data"),
            button("Duplicate…", self._history_duplicate,
                   tooltip="Start a fresh analysis pre-filled with these parameters"),
            button("Add to report", self._history_add_report,
                   tooltip="Append the selected run's result table to the Report book"),
            button("Delete", self._history_delete, danger=True,
                   tooltip="Delete the selected run and its stored payload"),
        )
        v.addWidget(bar)

        self.history_table = QtWidgets.QTableWidget()
        self.history_table.setSelectionBehavior(QtWidgets.QTableWidget.SelectRows)
        self.history_table.setSelectionMode(QtWidgets.QTableWidget.SingleSelection)
        self.history_table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        # newest-first row order maps 1:1 to the run list, so leave sorting off (row == index).
        self.history_table.doubleClicked.connect(lambda *_: self._history_open())
        v.addWidget(self.history_table, 1)
        v.addWidget(note(
            "Every analysis you run is recorded here. Open re-shows a past result exactly as it "
            "was computed; Re-run recomputes it against the current data; Duplicate starts a fresh "
            "run from the same settings."))

        self._history_runs = []                       # row index → AnalysisRun (newest first)
        self._history_stores = []                      # row index → the RunStore it came from
        sig = getattr(self, "datasetChanged", None)
        if sig is not None:
            sig.connect(lambda *_: self._refresh_analysis_history())
        self.tabs.addTab(w, "History")
        self._refresh_analysis_history()

    # ------------------------------------------------------------------ #
    def _refresh_analysis_history(self):
        """Rebuild the table from the current sample's run store AND the current cohort's run
        store, merged newest-first. Called on a dataset switch and by the dialog on every run
        transition (``AnalysisDialog._notify_runs_changed``)."""
        table = getattr(self, "history_table", None)
        if table is None:
            return
        pairs = []                                     # (run, store), the store it came from
        for store in (self.run_store, getattr(self, "cohort_run_store", None)):
            if store is None:
                continue
            try:
                pairs.extend((r, store) for r in store.list_runs())
            except Exception:                          # noqa: BLE001 — a bad store never breaks the view
                traceback.print_exc()
        pairs.sort(key=lambda p: p[0].created, reverse=True)
        self._history_runs = [r for r, _s in pairs]
        self._history_stores = [s for _r, s in pairs]
        rows = []
        for r in self._history_runs:
            sd = registry.REGISTRY.get(r.step_id)
            name = sd.name if sd is not None else r.step_id
            rows.append([r.created, name, r.dataset, r.status, (r.summary or r.error or "")])
        fill_table(table, _COLS, rows)

    def _selected_run(self):
        run, _store = self._selected_run_and_store()
        return run

    def _selected_run_and_store(self):
        table = getattr(self, "history_table", None)
        if table is None:
            return None, None
        row = table.currentRow()
        if 0 <= row < len(self._history_runs):
            return self._history_runs[row], self._history_stores[row]
        self.statusBar().showMessage("Select a run first.")
        return None, None

    # ------------------------------------------------------------------ #
    def _history_open(self):
        run, store = self._selected_run_and_store()
        if run is None:
            return
        from .analysisdialog import AnalysisDialog
        self.open_analysis(AnalysisDialog.from_run(self, run, store))

    def _history_rerun(self):
        run, store = self._selected_run_and_store()
        if run is None:
            return
        from .analysisdialog import AnalysisDialog
        dlg = self.open_analysis(AnalysisDialog.from_run(self, run, store))
        dlg._run()

    def _history_duplicate(self):
        run = self._selected_run()
        if run is None:
            return
        sd = registry.REGISTRY.get(run.step_id)
        if sd is None:
            self.statusBar().showMessage("This analysis is no longer available.")
            return
        from .analysisdialog import AnalysisDialog
        dlg = AnalysisDialog(self, sd)                 # a fresh 'Run' analysis, not a Re-run
        try:
            dlg.form.set_values(run.params or {})
        except Exception:                              # noqa: BLE001
            traceback.print_exc()
        self.open_analysis(dlg)

    def _history_add_report(self):
        run, store = self._selected_run_and_store()
        if run is None:
            return
        payload = store.load_result(run) if store is not None else None
        try:
            import pandas as pd
            df = payload if isinstance(payload, pd.DataFrame) else None
        except Exception:                              # noqa: BLE001
            df = None
        if df is None or not len(df):
            self.statusBar().showMessage("This run has no table to add to the report.")
            return
        sd = registry.REGISTRY.get(run.step_id)
        kind = sd.name if sd is not None else run.step_id
        self._log_analysis_to_report(kind, df, title=(run.title or kind),
                                     source_extra={"run_id": run.run_id})
        self.statusBar().showMessage(f"Added “{run.title or kind}” to the report.")

    def _history_delete(self):
        run, store = self._selected_run_and_store()
        if run is None:
            return
        sd = registry.REGISTRY.get(run.step_id)
        name = run.title or (sd.name if sd is not None else run.step_id)
        if not confirm(self, "Delete run",
                       f"Delete the recorded run “{name}”? Its stored result is removed too.",
                       ok_text="Delete"):
            return
        if store is not None:
            try:
                store.delete(run.run_id)
            except Exception:                          # noqa: BLE001
                traceback.print_exc()
        self._refresh_analysis_history()
        if hasattr(self, "_mark_dirty"):
            self._mark_dirty()
