"""Background workers — run engine calls off the GUI thread so the window stays
responsive while large files load and analyses run."""
from __future__ import annotations

import math
import traceback

from PySide6 import QtCore


def _safe_int(x) -> int:
    """Coerce a progress count to a finite int. A progress tick is purely cosmetic, so a
    non-finite / NaN / non-numeric value (e.g. a degenerate ``total`` from an upstream
    estimate) must NEVER abort the running analysis with
    ``OverflowError: cannot convert float infinity to integer`` — clamp it to 0 (which the
    loading bar reads as indeterminate) instead."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0
    return int(x) if math.isfinite(x) else 0


class CancelledError(Exception):
    """Raised inside a worker's progress callback when the user cancels."""


class WorkerSignals(QtCore.QObject):
    finished = QtCore.Signal(object)      # result of fn
    error = QtCore.Signal(str)
    cancelled = QtCore.Signal()
    progress = QtCore.Signal(int, int)    # done, total
    stage = QtCore.Signal(str)            # human-readable name of the current sub-step


class Worker(QtCore.QRunnable):
    """Run ``fn(*args, **kwargs)`` in a thread-pool thread.

    If ``fn`` accepts a ``progress`` keyword it receives a callback that emits
    :attr:`WorkerSignals.progress` (safe to connect to a GUI progress bar — Qt
    queues the cross-thread signal) and raises :class:`CancelledError` when
    ``cancel_check()`` returns True, so long streaming passes stop cooperatively.

    ``want_stage`` injects a second optional callback — ``stage(name)`` — that emits
    :attr:`WorkerSignals.stage`, so a job can narrate *what* it's doing (\"Reading spectra
    into memory\" → \"Priming\") separately from the numeric bar. Both are opt-in per call:
    a ``fn`` only receives ``progress`` / ``stage`` if the caller asks for it *and* ``fn``
    declares the matching keyword, so every existing call site is unaffected."""

    def __init__(self, fn, *args, want_progress: bool = False, want_stage: bool = False,
                 cancel_check=None, **kwargs):
        super().__init__()
        # Do NOT let Qt auto-delete this QRunnable on the worker thread — that would
        # tear down the WorkerSignals QObject off the main thread and segfault shiboken.
        # The owning window keeps a reference and drops it on a terminal signal (main thread).
        self.setAutoDelete(False)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self._cancel_check = cancel_check
        if want_progress:
            self.kwargs["progress"] = self._progress
        if want_stage:
            self.kwargs["stage"] = self._stage

    def _progress(self, d, t):
        if self._cancel_check and self._cancel_check():
            raise CancelledError()
        self.signals.progress.emit(_safe_int(d), _safe_int(t))

    def _stage(self, name):
        # A stage change is also a cooperative cancellation point — cheap, and it lets a
        # job that reports stages but not fine-grained progress still stop on Cancel.
        if self._cancel_check and self._cancel_check():
            raise CancelledError()
        self.signals.stage.emit(str(name))

    @QtCore.Slot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except CancelledError:
            self.signals.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            self.signals.error.emit(f"{exc}\n\n{traceback.format_exc()}")
        else:
            self.signals.finished.emit(result)
