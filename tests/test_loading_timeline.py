"""Load-card stage narration, re-priming progress, and the perf-timeline view.

Covers the three UX-plumbing guarantees added for the "what's going on?" pass:

* ``LoadingWindow`` accepts a ``stages=`` checklist and ``set_stage()`` advances it
  (the modal load card narrates *which* pass runs), with a caption fallback for an
  unlisted stage and full back-compat for callers that pass no stages;
* a re-prime with preprocessing active takes the slow streaming path and still reports
  a moving, completing bar plus a stage through the ``Worker`` — no silent hang; and
* the perf-log parser / summarizer and the ``TimelineDialog`` that surface it in-app.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402

from smile_msi.demo import make_synthetic  # noqa: E402
from smile_msi.gui.loading import LoadingWindow  # noqa: E402
from smile_msi.gui.workers import Worker  # noqa: E402
from smile_msi.gui.timeline import (  # noqa: E402
    parse_perf_log, summarize_operations, read_perf_log, TimelineDialog)


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# --- Workstream C: LoadingWindow stage checklist ------------------------------ #

def test_loading_window_accepts_stages_and_advances(app):
    """The exact construction ``_start_run`` makes for a modal staged job — previously a
    TypeError because LoadingWindow supported neither ``stages=`` nor ``set_stage``."""
    dlg = LoadingWindow(None, message="Loading sample…", cancelable=True,
                        stages=["Reading spectra into memory", "Priming spectra"])
    first_icon = dlg._stage_rows["Reading spectra into memory"][0]
    cur_icon = dlg._stage_rows["Priming spectra"][0]
    assert first_icon.text() == "○" and cur_icon.text() == "○"   # all pending initially

    dlg.set_stage("Reading spectra into memory")
    assert first_icon.text() == "▸"                              # active
    dlg.set_stage("Priming spectra")
    assert first_icon.text() == "✓"                              # earlier row now done
    assert cur_icon.text() == "▸"                                # current row active


def test_loading_window_unlisted_stage_falls_back(app):
    """A stage name not in the list routes to the detail caption instead of vanishing."""
    dlg = LoadingWindow(None, stages=["Priming spectra"])
    dlg.set_stage("Something ad-hoc")            # must not raise
    assert dlg._detail.text()                    # caption populated


def test_loading_window_without_stages_back_compat(app):
    """Callers that pass no ``stages=`` keep the old behaviour and set_stage is a no-op-ish
    fallback (writes the caption), never a crash."""
    dlg = LoadingWindow(None, message="x", cancelable=False)
    assert dlg._stage_rows == {}
    dlg.set_stage("anything")                    # no rows → caption fallback, no error
    dlg.set_progress(1, 2)


# --- Workstream D: re-prime reports progress + stage on the slow path --------- #

def test_reprime_with_preprocessing_reports_progress_and_stage(app):
    """set_preprocessing clears the prime cache and, with transforms active, ``_dense()``
    returns None, forcing the slow per-pixel streaming prime. That path must still emit a
    moving, completing bar and a stage — the fix for the silent preprocessing hang."""
    ds = make_synthetic(width=20, height=16)     # 320 px > the 256-pixel progress stride
    ds.to_ram(); ds.prime()
    ds.set_preprocessing([lambda mz, inten: (mz, inten)])
    assert ds._dense() is None                   # transforms force the streaming path

    def reprime(progress=None, stage=None):      # mirrors MainWindow.apply_preprocessing
        if stage:
            stage("Re-priming spectra")
        ds.prime(progress=progress)
        return True

    prog, stages, done = [], [], []
    wk = Worker(reprime, want_progress=True, want_stage=True)
    wk.signals.progress.connect(lambda d, t: prog.append((d, t)))
    wk.signals.stage.connect(stages.append)
    wk.signals.finished.connect(done.append)
    wk.run()

    assert stages == ["Re-priming spectra"]
    assert prog and prog[-1][0] == prog[-1][1]   # bar reaches 100%
    assert done == [True]


# --- Workstream E: perf-log parsing + timeline dialog ------------------------- #

_SAMPLE_LOG = """\
[12:18:40] START  Loading a.imzML + building fast cache…
[12:18:40] STAGE  Loading a.imzML + building fast cache… :: Reading spectra into memory
[12:18:41] STAGE  Loading a.imzML + building fast cache… :: Priming spectra
[12:18:43] DONE   Loading a.imzML + building fast cache…  (3.20s)
[12:18:50] START  Applying preprocessing & re-priming…
[12:18:50] STAGE  Applying preprocessing & re-priming… :: Re-priming spectra
[12:18:52] DONE   Applying preprocessing & re-priming…  (2.10s)
a stray line that is not a perf record
[12:19:00] START  An operation that never finished…
"""


def test_parse_perf_log_ignores_junk_and_extracts_fields():
    events = parse_perf_log(_SAMPLE_LOG)
    kinds = [e["kind"] for e in events]
    assert kinds.count("START") == 3 and kinds.count("DONE") == 2
    stage_ev = next(e for e in events if e["kind"] == "STAGE")
    assert stage_ev["stage"] == "Reading spectra into memory"
    done_ev = next(e for e in events if e["kind"] == "DONE")
    assert done_ev["seconds"] == 3.20


def test_summarize_pairs_stages_and_flags_running():
    ops = summarize_operations(parse_perf_log(_SAMPLE_LOG))
    assert len(ops) == 3
    load = ops[0]
    assert load["status"] == "done" and load["seconds"] == 3.20
    assert load["stages"] == ["Reading spectra into memory", "Priming spectra"]
    assert ops[2]["status"] == "running"     # the unfinished op is kept, marked running


def test_read_perf_log_missing_file_is_empty(tmp_path):
    assert read_perf_log(str(tmp_path / "nope.log")) == []


def test_read_perf_log_reads_real_file(tmp_path):
    p = tmp_path / "smile_msi_perf.log"
    p.write_text(_SAMPLE_LOG, encoding="utf-8")
    ops = read_perf_log(str(p))
    assert [o["label"] for o in ops][:2] == [
        "Loading a.imzML + building fast cache…",
        "Applying preprocessing & re-priming…",
    ]


def test_timeline_dialog_renders_startup_and_operations(app):
    ops = summarize_operations(parse_perf_log(_SAMPLE_LOG))
    dlg = TimelineDialog(None, startup_rows=[("Imports", 250.0, False),
                                             ("MainWindow()", 793.0, False)],
                         startup_total_ms=1370.0, operations=ops)
    tables = dlg.findChildren(QtWidgets.QTableWidget)
    assert tables[0].rowCount() == 2         # startup breakdown
    assert tables[1].rowCount() == 3         # operations


def test_timeline_dialog_empty_state_is_safe(app):
    TimelineDialog(None)                      # no clock, no ops → must not raise
