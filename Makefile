# SMILE MSI — segmented test targets so day-to-day dev never waits on the slow Qt suite.
# The `gui` / `slow` markers are applied automatically (tests/conftest.py); see pyproject
# [tool.pytest.ini_options]. Use PY=.venv/bin/python (or your interpreter).
PY ?= .venv/bin/python
PYTEST = QT_QPA_PLATFORM=offscreen $(PY) -m pytest

.PHONY: test test-fast test-engine test-gui test-analysis test-all test-collect

## test        — the quick dev loop: fast pure-engine tests only (no Qt, no heavy compute)
test test-fast:
	$(PYTEST) -m "not gui and not slow"

## test-engine — every non-Qt test (includes heavy engine compute: cohort, register, …)
test-engine:
	$(PYTEST) -m "not gui"

## test-gui     — the Qt/MainWindow tests (slower; a hung test fails on the 180s timeout)
test-gui:
	$(PYTEST) -m gui

## test-analysis — just the Analyze surface (dialog, gallery, provenance) — fast, one window
test-analysis:
	$(PYTEST) tests/test_analysis_dialog.py tests/test_analysis_provenance.py

## test-all      — everything (run per-file in CI; the whole Qt suite is memory-heavy in one process)
test-all:
	$(PYTEST) tests/

## test-collect  — sanity: collect the whole suite with zero import errors
test-collect:
	$(PYTEST) tests/ --collect-only -q
