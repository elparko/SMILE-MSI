"""Headless tests for the source-checkout updater's commit-based 'check for updates'."""
from pathlib import Path
import types

import pytest

pytest.importorskip("PySide6")
from smile_msi.gui import updater  # noqa: E402


def _r(code=0, out="", err=""):
    return types.SimpleNamespace(returncode=code, stdout=out, stderr=err)


def test_remote_status_counts_behind_commits(monkeypatch):
    canned = {
        ("rev-parse", "HEAD"): _r(0, "abc123\n"),
        ("fetch", "--quiet"): _r(0, ""),
        ("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"): _r(0, "origin/main\n"),
        ("rev-list", "--count", "HEAD..origin/main"): _r(0, "3\n"),
        ("log", "--oneline", "--no-decorate", "--no-color", "HEAD..origin/main"):
            _r(0, "a feat one\nb fix two\nc feat three\n"),
    }
    monkeypatch.setattr(updater, "_git", lambda args, cwd: canned.get(tuple(args), _r(0, "")))
    res = updater.remote_status(Path("/x"))
    assert res["ok"] and res["behind"] == 3
    assert res["log"] == ["a feat one", "b fix two", "c feat three"]
    assert res["upstream"] == "origin/main"


def test_remote_status_up_to_date(monkeypatch):
    canned = {
        ("rev-parse", "HEAD"): _r(0, "abc\n"),
        ("fetch", "--quiet"): _r(0, ""),
        ("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"): _r(0, "origin/main\n"),
        ("rev-list", "--count", "HEAD..origin/main"): _r(0, "0\n"),
    }
    monkeypatch.setattr(updater, "_git", lambda args, cwd: canned.get(tuple(args), _r(0, "")))
    res = updater.remote_status(Path("/x"))
    assert res["ok"] and res["behind"] == 0 and res["log"] == []


def test_remote_status_fetch_failure(monkeypatch):
    canned = {("rev-parse", "HEAD"): _r(0, "abc\n"),
              ("fetch", "--quiet"): _r(1, "", "could not resolve host")}
    monkeypatch.setattr(updater, "_git", lambda args, cwd: canned.get(tuple(args), _r(0, "")))
    res = updater.remote_status(Path("/x"))
    assert not res["ok"] and "could not resolve host" in res["message"]
