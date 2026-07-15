"""Tests for the named figure-style preset library (built-ins + user-saved)."""
from __future__ import annotations

import pytest

from smile_msi import prefs, stylelib
from smile_msi.stylespec import StyleSpec


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Point the prefs store at a throwaway home so user presets don't touch real prefs."""
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    monkeypatch.setattr(prefs, "_CACHE", None)     # drop any cached prefs from earlier tests
    yield tmp_path
    monkeypatch.setattr(prefs, "_CACHE", None)


def test_builtins_always_present(isolated_home):
    names = stylelib.names()
    assert "App Default" in names
    assert "Poster (large type)" in names
    assert stylelib.is_builtin("App Default")
    assert stylelib.get("Poster (large type)").type_gain == 1.6


def test_save_and_load_user_preset(isolated_home):
    spec = StyleSpec(name="My Look", type_gain=1.3, theme="light")
    stylelib.save(spec)
    assert "My Look" in stylelib.names()
    loaded = stylelib.get("My Look")
    assert loaded.type_gain == 1.3
    assert loaded.theme == "light"
    assert not stylelib.is_builtin("My Look")


def test_user_preset_persists_across_cache_reset(isolated_home, monkeypatch):
    stylelib.save(StyleSpec(name="Persisted", spine_w=2.2))
    monkeypatch.setattr(prefs, "_CACHE", None)     # simulate an app restart (reload from disk)
    assert stylelib.get("Persisted").spine_w == 2.2


def test_delete_user_preset_only(isolated_home):
    stylelib.save(StyleSpec(name="Temp", type_gain=1.1))
    assert stylelib.delete("Temp") is True
    assert "Temp" not in stylelib.names()
    # built-ins can't be deleted
    assert stylelib.delete("App Default") is False
    assert "App Default" in stylelib.names()


def test_user_preset_shadows_builtin_then_reveals(isolated_home):
    stylelib.save(StyleSpec(name="Poster (large type)", type_gain=2.5))
    assert stylelib.get("Poster (large type)").type_gain == 2.5   # shadowed
    assert not stylelib.is_builtin("Poster (large type)")
    stylelib.delete("Poster (large type)")
    assert stylelib.get("Poster (large type)").type_gain == 1.6   # built-in revealed
    assert stylelib.is_builtin("Poster (large type)")


def test_empty_name_rejected(isolated_home):
    with pytest.raises(ValueError):
        stylelib.save(StyleSpec(name="   "))


def test_json_round_trip_import_export(isolated_home):
    stylelib.save(StyleSpec(name="Exported", type_gain=1.4, category_palette="Tableau 10"))
    text = stylelib.export_json("Exported")
    imported = stylelib.import_json(text, name="Reimported")
    assert imported.name == "Reimported"
    assert imported.type_gain == 1.4
    assert stylelib.get("Reimported").category_palette == "Tableau 10"


def test_import_unnamed_recipe_gets_default_name(isolated_home):
    spec = stylelib.import_json('{"type_gain": 1.2}', save_it=False)
    assert spec.name == "Imported"
    assert spec.type_gain == 1.2


def test_corrupt_user_entry_is_skipped(isolated_home, monkeypatch):
    # a malformed stored entry must not break the whole library
    prefs.set(stylelib.STYLE_PREFS_KEY, {"Good": {"name": "Good", "type_gain": 1.2},
                                         "Bad": "not-a-dict"})
    up = stylelib.user_presets()
    assert "Good" in up
    assert up["Good"].type_gain == 1.2
    assert "Bad" not in up          # tolerated, skipped
