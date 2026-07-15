"""Tests for the persistent app-preferences store (smile_msi.prefs) and the last-folder
seeding logic in the GUI file-dialog wrappers (smile_msi.gui.filedialogs).

The prefs store is pure (no Qt); the file-dialog seeding helpers are pure functions over
the store, so they're tested without spinning up a QApplication.
"""
import os

import pytest

from smile_msi import prefs


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point the prefs store at an isolated JSON file with a clean cache."""
    monkeypatch.setattr(prefs, "_path", lambda: str(tmp_path / "prefs.json"))
    monkeypatch.setattr(prefs, "_CACHE", None)
    return tmp_path


def test_defaults_then_set_get(store):
    assert prefs.get("last_dir", "") == ""           # default when unset
    assert prefs.get("missing") is None
    prefs.set("last_dir", "/tmp/x")
    prefs.set("export_options", {"scope": "overlay", "dpi": 600})
    assert prefs.get("last_dir") == "/tmp/x"
    assert prefs.get("export_options")["dpi"] == 600


def test_persists_to_disk_and_survives_reload(store, monkeypatch):
    prefs.set("last_dir", "/tmp/y")
    assert os.path.exists(prefs._path())             # written eagerly
    monkeypatch.setattr(prefs, "_CACHE", None)        # simulate a fresh app run
    assert prefs.get("last_dir") == "/tmp/y"


def test_corrupt_file_degrades_to_defaults(store, monkeypatch):
    with open(prefs._path(), "w", encoding="utf-8") as fh:
        fh.write("{ not json")
    monkeypatch.setattr(prefs, "_CACHE", None)
    assert prefs.get("last_dir", "fallback") == "fallback"   # no crash, just defaults


def test_filedialog_seed_and_last_dir(store, tmp_path, monkeypatch):
    fd = pytest.importorskip("smile_msi.gui.filedialogs")
    # an existing dir is remembered and seeds bare filenames
    prefs.set("last_dir", str(tmp_path))
    assert fd._last_dir() == str(tmp_path)
    assert fd._seed("report.pdf") == os.path.join(str(tmp_path), "report.pdf")
    # an explicit directory in the default is respected verbatim
    assert fd._seed("/explicit/dir/x.csv") == "/explicit/dir/x.csv"
    # a stale (missing) remembered dir is ignored, so we fall back to the bare default
    prefs.set("last_dir", str(tmp_path / "deleted"))
    assert fd._last_dir() == ""
    assert fd._seed("x.csv") == "x.csv"


def test_filedialog_remember_uses_parent_of_a_file(store, tmp_path):
    fd = pytest.importorskip("smile_msi.gui.filedialogs")
    f = tmp_path / "out.csv"
    f.write_text("x")
    fd._remember(str(f))
    assert prefs.get("last_dir") == str(tmp_path)    # stored the folder, not the file
