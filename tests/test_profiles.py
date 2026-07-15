"""Tests for Analysis Profiles (smile_msi.profiles) — the named, versioned, shareable
bundles of standardized processing settings.

Pure engine/store logic (no Qt): the file store, versioning, the active pointer, the
prefs mirror, diff/merge, and export/import. The GUI dialog is exercised elsewhere.
"""
import json

import pytest

from smile_msi import prefs, profiles


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolate the app home (so profiles/ and prefs.json live under tmp) with a clean
    prefs cache — both the profile store and the active pointer resolve through here."""
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    monkeypatch.setattr(prefs, "_CACHE", None)
    return tmp_path


def test_builtin_is_complete_and_readonly(home):
    b = profiles.builtin()
    assert b["name"] == profiles.BUILTIN_NAME
    assert b.get("builtin") is True
    # every schema key has a value, with the schema's default
    assert set(b["params"]) == {p.key for p in profiles.SCHEMA}
    assert b["params"]["ppm"] == 10.0 and b["params"]["random_seed"] == 0
    with pytest.raises(ValueError):                       # the baseline can't be overwritten
        profiles.save(profiles.make_profile(profiles.BUILTIN_NAME))


def test_save_load_roundtrip_and_hash(home):
    prof = profiles.make_profile("Nerve-Lipid", {**profiles.default_params(), "ppm": 5.0})
    h0 = profiles.content_hash(prof)
    profiles.save(prof)
    again = profiles.load("Nerve-Lipid")
    assert again["params"]["ppm"] == 5.0
    assert profiles.content_hash(again) == h0            # hash is over params, stable
    # hash ignores name/version/timestamp but tracks param changes
    assert profiles.content_hash(profiles.make_profile("Other", again["params"])) == h0
    changed = profiles.make_profile("Nerve-Lipid", {**again["params"], "ppm": 6.0})
    assert profiles.content_hash(changed) != h0


def test_versions_are_retained(home):
    profiles.save(profiles.make_profile("P", {**profiles.default_params(), "snr": 4.0}))
    profiles.set_active("P")
    profiles.save(profiles.bump_version(profiles.active()))   # -> v2, keeps v1 on disk
    assert profiles.versions_of("P") == [1, 2]
    assert profiles.active()["version"] == 2                  # active resolves to latest
    assert profiles.load("P")["version"] == 2
    assert profiles.load_exact("P", 1)["version"] == 1        # old version still reproducible
    # list shows only the latest per name, plus the builtin
    labels = [profiles.label(x) for x in profiles.list_profiles()]
    assert labels == [f"{profiles.BUILTIN_NAME} v1", "P v2"]


def test_diff_detects_deviation(home):
    act = profiles.builtin()
    assert profiles.diff(act["params"], act) == {}           # identical → no deviation
    cur = {**act["params"], "ppm": 8.0, "snr": 5.0}
    d = profiles.diff(cur, act)
    assert d == {"ppm": [10.0, 8.0], "snr": [3.0, 5.0]}      # [profile, current]


def test_merge_defaults_fills_and_coerces(home):
    merged = profiles.merge_defaults({"ppm": "7", "snr": 999, "bogus": 1})
    assert merged["ppm"] == 7.0                              # string coerced to float
    assert merged["snr"] == 50.0                             # clamped to schema max
    assert "bogus" not in merged                             # unknown key dropped
    assert merged["mode"] == "negative"                     # missing key gets default
    assert set(merged) == {p.key for p in profiles.SCHEMA}


def test_active_pointer_persists_and_mirrors_prefs(home, monkeypatch):
    profiles.save(profiles.make_profile(
        "Lab", {**profiles.default_params(), "random_seed": 42, "seg_method": "ward",
                "segment_default_count": 12}))
    profiles.set_active("Lab")
    # active pointer survives a "fresh app run" (cleared cache, reread from disk)
    monkeypatch.setattr(prefs, "_CACHE", None)
    assert profiles.active()["name"] == "Lab"
    # set_active mirrored the prefs-backed knobs so subsystems that read prefs see them
    assert profiles.active_seed() == 42
    assert prefs.get(profiles.SEG_METHOD_PREF) == "ward"
    assert prefs.get(profiles.SEG_COUNT_PREF) == 12


def test_delete_resets_dangling_active(home):
    profiles.save(profiles.make_profile("Temp"))
    profiles.set_active("Temp")
    assert profiles.active()["name"] == "Temp"
    profiles.delete("Temp")
    assert profiles.versions_of("Temp") == []
    assert profiles.active()["name"] == profiles.BUILTIN_NAME   # pointer reset, no dangle


def test_export_import_roundtrip(home, tmp_path):
    prof = profiles.make_profile("Shared", {**profiles.default_params(), "id_ppm": 5.0})
    path = str(tmp_path / "shared.json")
    profiles.export_to(prof, path)
    imported = profiles.import_from(path)
    assert imported["params"]["id_ppm"] == 5.0
    # a non-profile file is rejected
    bad = str(tmp_path / "bad.json")
    with open(bad, "w", encoding="utf-8") as fh:
        json.dump({"not": "a profile"}, fh)
    with pytest.raises(ValueError):
        profiles.import_from(bad)
