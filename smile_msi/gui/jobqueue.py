"""A serial job queue for background analyses.

Why this exists: every heavy analysis used to be started straight onto the global
``QThreadPool`` the instant it was triggered, so two analyses kicked off close together
ran *concurrently* — fighting over CPU/RAM and (worse) over the single shared cancel
flag, progress bar and Cancel button. The symptom users hit was an analysis getting
"eaten" when another one started on top of it.

This module serialises that work: heavy jobs are submitted to a :class:`JobQueue`, which
runs **one at a time** and starts the next only when the current one reaches a terminal
state. A :class:`JobQueuePanel` shows what is running / queued / finished and lets the
user cancel an individual job (a queued job is simply dropped; the running job is
cancelled cooperatively, when it supports it).
"""
from __future__ import annotations

import itertools

from PySide6 import QtCore, QtWidgets


class JobStatus:
    QUEUED = "Queued"
    RUNNING = "Running"
    DONE = "Done"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


_TERMINAL = {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED}


class Job:
    """One unit of queued background work.

    ``starter`` is a callable ``starter(job)`` that actually launches the worker (set by
    the window once it has built the closure); the queue calls it when the job reaches the
    front. ``cancellable`` is True only when the underlying worker reports progress and so
    can be stopped cooperatively mid-run — a job without progress can still be cancelled
    while it is *queued*, just not once it is running."""

    _ids = itertools.count(1)

    def __init__(self, label: str, *, cancellable: bool = False, run_id: str | None = None):
        self.id = next(Job._ids)
        self.label = label or "Working…"
        self.run_id = run_id        # ties this queue row to its AnalysisRun history record (plan 24)
        self.cancellable = bool(cancellable)
        self.status = JobStatus.QUEUED
        self.done = 0
        self.total = 0
        self.error = ""
        self.starter = None        # set by the owner before submit()

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL


class JobQueue(QtCore.QObject):
    """Runs submitted :class:`Job` objects one at a time.

    The queue never starts a worker itself — it calls ``job.starter(job)``, and the owner
    is responsible for reporting back via :meth:`on_progress` and :meth:`on_terminal` (wired
    to the worker's Qt signals). It emits :attr:`changed` on every state transition so the
    panel and the toolbar badge can refresh."""

    changed = QtCore.Signal()
    _MAX_FINISHED = 50            # cap the finished history so a long session can't grow it unbounded

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pending: list[Job] = []
        self._running: Job | None = None
        self._finished: list[Job] = []       # most-recent-last

    # ---- lookup ------------------------------------------------------------------
    def job_for(self, run_id: str):
        """The job carrying ``run_id`` — running, queued, or most-recently finished.

        Lets a plan-24 AnalysisDialog show *its own* progress inline instead of only in the
        shared status bar, where several concurrent analyses are indistinguishable. Newest
        finished job wins, so a re-run supersedes the record of its predecessor."""
        if not run_id:
            return None
        if self._running is not None and self._running.run_id == run_id:
            return self._running
        for job in self._pending:
            if job.run_id == run_id:
                return job
        for job in reversed(self._finished):
            if job.run_id == run_id:
                return job
        return None

    # ---- submission / scheduling -------------------------------------------------
    def submit(self, job: Job) -> None:
        self._pending.append(job)
        self.changed.emit()
        self._maybe_start()

    def _maybe_start(self) -> None:
        if self._running is not None or not self._pending:
            return
        job = self._pending.pop(0)
        self._running = job
        job.status = JobStatus.RUNNING
        self.changed.emit()
        # starter launches the worker; terminal/progress callbacks below drive the rest.
        job.starter(job)

    # ---- callbacks from the running worker (GUI thread) --------------------------
    def on_progress(self, job: Job, done: int, total: int) -> None:
        job.done, job.total = done, total
        self.changed.emit()

    def on_terminal(self, job: Job, status: str, error: str = "") -> None:
        # Ignore a late terminal for a job already retired (e.g. cancelled while queued
        # then its worker — which never started — somehow signals).
        if job.is_terminal:
            return
        job.status = status
        job.error = error
        if self._running is job:
            self._running = None
        self._retire(job)
        self.changed.emit()
        self._maybe_start()

    def _retire(self, job: Job) -> None:
        self._finished.append(job)
        if len(self._finished) > self._MAX_FINISHED:
            del self._finished[: len(self._finished) - self._MAX_FINISHED]

    # ---- cancellation ------------------------------------------------------------
    def cancel_pending(self, job: Job) -> bool:
        """Drop a job that hasn't started yet. Returns True if it was queued."""
        if job in self._pending:
            self._pending.remove(job)
            job.status = JobStatus.CANCELLED
            self._retire(job)
            self.changed.emit()
            return True
        return False

    def cancel_all_pending(self) -> None:
        if not self._pending:
            return
        for job in self._pending:
            job.status = JobStatus.CANCELLED
            self._retire(job)
        self._pending.clear()
        self.changed.emit()

    def clear_finished(self) -> None:
        if self._finished:
            self._finished.clear()
            self.changed.emit()

    # ---- views -------------------------------------------------------------------
    @property
    def running(self) -> Job | None:
        return self._running

    def pending_count(self) -> int:
        return len(self._pending)

    def rows(self) -> list[Job]:
        """Display order: running first, then queued, then finished (newest first)."""
        out: list[Job] = []
        if self._running is not None:
            out.append(self._running)
        out.extend(self._pending)
        out.extend(reversed(self._finished))
        return out


_STATUS_COLOR = {
    JobStatus.RUNNING: "#2d7dd2",
    JobStatus.QUEUED: "#888888",
    JobStatus.DONE: "#3a9a52",
    JobStatus.FAILED: "#c0392b",
    JobStatus.CANCELLED: "#8a8a8a",
}


class JobQueuePanel(QtWidgets.QWidget):
    """Live list of jobs with per-job cancel. ``cancel_cb(job)`` is invoked when the user
    clicks a job's ✕ — the owner decides whether that means "drop from queue" or "cancel
    the running worker"."""

    def __init__(self, queue: JobQueue, cancel_cb, parent=None):
        super().__init__(parent)
        self._queue = queue
        self._cancel_cb = cancel_cb

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        bar = QtWidgets.QHBoxLayout()
        self._summary = QtWidgets.QLabel("No jobs.")
        bar.addWidget(self._summary, 1)
        self._b_cancel_all = QtWidgets.QPushButton("Cancel queued")
        self._b_cancel_all.clicked.connect(self._queue.cancel_all_pending)
        self._b_clear = QtWidgets.QPushButton("Clear finished")
        self._b_clear.clicked.connect(self._queue.clear_finished)
        bar.addWidget(self._b_cancel_all)
        bar.addWidget(self._b_clear)
        root.addLayout(bar)

        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._list = QtWidgets.QWidget()
        self._list_layout = QtWidgets.QVBoxLayout(self._list)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(4)
        self._list_layout.addStretch(1)
        self._scroll.setWidget(self._list)
        root.addWidget(self._scroll, 1)

        self._queue.changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        # Rebuild the rows. The list is short and only changes on real transitions
        # (submit / start / finish) plus the running job's ~per-256-pixel progress tick,
        # so a full rebuild is cheap and avoids stale-widget bookkeeping.
        while self._list_layout.count() > 1:        # keep the trailing stretch
            item = self._list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        rows = self._queue.rows()
        for job in rows:
            self._list_layout.insertWidget(self._list_layout.count() - 1, self._make_row(job))

        running = self._queue.running
        n_pending = self._queue.pending_count()
        parts = []
        if running is not None:
            parts.append("1 running")
        if n_pending:
            parts.append(f"{n_pending} queued")
        self._summary.setText(", ".join(parts) if parts else "No jobs.")
        self._b_cancel_all.setEnabled(n_pending > 0)
        self._b_clear.setEnabled(any(j.is_terminal for j in rows))

    def _make_row(self, job: Job) -> QtWidgets.QWidget:
        row = QtWidgets.QFrame()
        row.setFrameShape(QtWidgets.QFrame.StyledPanel)
        lay = QtWidgets.QHBoxLayout(row)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(8)

        dot = QtWidgets.QLabel("●")
        dot.setStyleSheet(f"color: {_STATUS_COLOR.get(job.status, '#888')};")
        lay.addWidget(dot)

        text = QtWidgets.QVBoxLayout()
        text.setSpacing(1)
        name = QtWidgets.QLabel(job.label)
        name.setToolTip(job.error or job.label)
        text.addWidget(name)
        sub = QtWidgets.QLabel(job.status if not job.error
                               else f"{job.status} — {job.error.splitlines()[0]}")
        sub.setStyleSheet("color: #888; font-size: 11px;")
        text.addWidget(sub)
        lay.addLayout(text, 1)

        if job.status == JobStatus.RUNNING and job.cancellable:
            pb = QtWidgets.QProgressBar()
            pb.setMaximumWidth(120)
            pb.setTextVisible(False)
            if job.total > 0:
                pb.setRange(0, 100)
                pb.setValue(int(100 * job.done / max(job.total, 1)))
            else:
                pb.setRange(0, 0)        # indeterminate
            lay.addWidget(pb)

        # ✕ to cancel: available while queued (drop it) or while running *if* cancellable.
        can_cancel = (job.status == JobStatus.QUEUED) or \
                     (job.status == JobStatus.RUNNING and job.cancellable)
        if can_cancel:
            x = QtWidgets.QToolButton()
            x.setText("✕")
            x.setAutoRaise(True)
            x.setToolTip("Cancel this job")
            x.clicked.connect(lambda _=False, j=job: self._cancel_cb(j))
            lay.addWidget(x)

        return row
