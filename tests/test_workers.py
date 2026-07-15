"""Background-worker progress sink — a cosmetic progress tick must never crash an
analysis, even when an upstream pass emits a non-finite count."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402

from smile_msi.gui.workers import Worker, _safe_int  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_safe_int_handles_non_finite_and_junk():
    assert _safe_int(float("inf")) == 0
    assert _safe_int(float("-inf")) == 0
    assert _safe_int(float("nan")) == 0
    assert _safe_int(5.9) == 5
    assert _safe_int(0) == 0
    assert _safe_int("nope") == 0
    assert _safe_int(None) == 0


def test_progress_tick_with_inf_does_not_raise(app):
    """Regression: a non-finite progress count used to crash with
    'cannot convert float infinity to integer' (e.g. on Find spatial features)."""
    w = Worker(lambda progress=None: None, want_progress=True)
    w._progress(0, float("inf"))      # previously: OverflowError
    w._progress(float("nan"), 100)    # previously: ValueError
    w._progress(3, 10)                # normal tick still works
