"""Shared pytest fixtures + automatic test segmentation.

Every test is auto-tagged so dev can run a fast subset (see ``[tool.pytest.ini_options]`` in
pyproject): a test whose module pulls in Qt / ``smile_msi.gui`` is marked ``gui`` (builds a
MainWindow, slower), and heavy-compute tests (UMAP / SHAP / cohort / 3D / registration) are
marked ``slow``. No test needs a hand-written marker.
"""
import pathlib
import sys

import pytest

# substrings in a nodeid that flag a heavy-compute test (skippable with -m "not slow")
_SLOW_HINTS = ("umap", "embedding", "shap", "cohort", "nested", "studio",
               "volume3d", "stack3d", "register", "singlecell", "ionembed")
_GUI_FILE_CACHE: dict = {}


def _is_gui_file(path) -> bool:
    """A GUI test file imports Qt (PySide6) / ``smile_msi.gui`` / pyqtgraph — those build a
    MainWindow and are the slow, offscreen-Qt tests we segment out of the fast dev loop."""
    p = str(path)
    if p not in _GUI_FILE_CACHE:
        try:
            txt = pathlib.Path(p).read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            txt = ""
        _GUI_FILE_CACHE[p] = ("PySide6" in txt or "smile_msi.gui" in txt
                              or "from .gui" in txt or "pyqtgraph" in txt)
    return _GUI_FILE_CACHE[p]


def pytest_collection_modifyitems(config, items):
    for it in items:
        if _is_gui_file(it.fspath):
            it.add_marker(pytest.mark.gui)
        low = it.nodeid.lower()
        if any(h in low for h in _SLOW_HINTS):
            it.add_marker(pytest.mark.slow)


_EXIT_STATUS: dict = {}


def pytest_sessionfinish(session, exitstatus):
    """Record the real status; the hard-exit itself happens in :func:`pytest_unconfigure`."""
    _EXIT_STATUS["code"] = int(exitstatus)


def pytest_unconfigure(config):
    """Offscreen Qt + pyqtgraph can segfault during interpreter teardown (a C++-side cleanup
    race), turning a green run into a spurious exit 139 that reads as a failing suite. Close any
    lingering windows and hard-exit with the REAL status so the shutdown crash can't mask a
    passing run. Skipped when coverage is active (it needs a normal exit to write its data).

    This *must not* live in ``pytest_sessionfinish``. TerminalReporter's own sessionfinish is a
    hookwrapper: it yields, every plain hook runs, and only then does it print the summary. An
    ``os._exit`` from a plain hook lands in that gap and kills the process before the ``FAILED``
    list is ever written — so a failing suite reported which tests failed nowhere at all.
    ``pytest_unconfigure`` runs after the summary."""
    if "PySide6.QtWidgets" not in sys.modules or "coverage" in sys.modules:
        return
    try:
        from PySide6 import QtWidgets
        appx = QtWidgets.QApplication.instance()
    except Exception:  # noqa: BLE001
        return
    if appx is None:                      # no Qt app was ever created → a pure-engine run, leave it
        return
    import os
    try:
        for w in list(appx.topLevelWidgets()):
            try:
                w.close()
            except Exception:  # noqa: BLE001
                pass
        appx.processEvents()
    except Exception:  # noqa: BLE001
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_EXIT_STATUS.get("code", 0))


# Phase-2 of the UX pass routed destructive actions (delete region / clear report /
# remove-from-cohort / reset segmentation / clear painted ROI / wipe-and-restart) through
# the standardized ``common.confirm()`` dialog. In headless tests there is no user to click
# it and ``QMessageBox.exec()`` would block forever, so auto-accept every ``confirm()`` —
# returning True reproduces the pre-gate behaviour (the action simply proceeds), which is
# exactly what the existing destructive-path tests assert. A test that wants the *cancel*
# path can re-patch the specific module's ``confirm``. Only gui modules already imported are
# patched, so pure-engine tests (masses/msi/pipeline) never pull in PySide6.
_CONFIRM_MODULES = ("cohortview", "ion", "reporttab", "main", "segment")


@pytest.fixture(autouse=True)
def _auto_accept_confirm(monkeypatch):
    for mod in _CONFIRM_MODULES:
        module = sys.modules.get(f"smile_msi.gui.{mod}")
        if module is not None and hasattr(module, "confirm"):
            monkeypatch.setattr(module, "confirm", lambda *a, **k: True)
