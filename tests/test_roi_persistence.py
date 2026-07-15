"""Regression tests for the ROI-loss-on-add/remove bug.

A sample's ROIs live in its managed session file, named after a hash of the slide's content
fingerprint. That fingerprint used to include the m/z range and grid width/height, both of
which drift between loads of the same slide — so the slide re-keyed to a new (empty) file and
its ROIs were stranded. These tests lock in the stable-key + adoption + blank-guard fixes.
"""
import json
import os

from smile_msi import session


# --- drift variants of ONE slide's fingerprint (seen on real data) -------------------- #
FP_CLEAN = "n361271:w4952:h771:mz200.00-2000.00:c8506c52c8b5"
FP_NA = "n361271:w4952:h771:mzNA:c8506c52c8b5"                 # misread m/z bounds
FP_TRANSPOSED = "n361271:w771:h4952:mz200.00-2000.00:c8506c52c8b5"  # orientation flip
FP_OTHER_SLIDE = "n94713:w5254:h1021:mz200.00-2000.00:b333f221693b"


def test_stable_fingerprint_strips_volatile_parts():
    assert session.stable_fingerprint(FP_CLEAN) == "n361271:c8506c52c8b5"
    assert session.stable_fingerprint(FP_NA) == "n361271:c8506c52c8b5"
    assert session.stable_fingerprint(FP_TRANSPOSED) == "n361271:c8506c52c8b5"
    # a non-standard / legacy fingerprint passes through unchanged (back-compat)
    assert session.stable_fingerprint("WRONGFP") == "WRONGFP"
    assert session.stable_fingerprint("") == ""


def test_managed_path_stable_across_mz_and_orientation_drift(tmp_path, monkeypatch):
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    src = "/data/negrun2.imzML"
    p_clean = session.managed_path(src, FP_CLEAN)
    assert session.managed_path(src, FP_NA) == p_clean          # m/z NA drift → same file
    assert session.managed_path(src, FP_TRANSPOSED) == p_clean  # w/h transpose → same file
    assert session.managed_path(src, FP_OTHER_SLIDE) != p_clean  # different slide → different file


def test_fingerprint_mismatch_tolerates_same_slide_drift():
    assert not session.fingerprint_mismatch(FP_NA, FP_CLEAN)         # same slide, drifted → no warn
    assert not session.fingerprint_mismatch(FP_TRANSPOSED, FP_CLEAN)
    assert session.fingerprint_mismatch(FP_CLEAN, FP_OTHER_SLIDE)    # genuinely different slide


def _write(tmp_path, monkeypatch, fname, *, source, fingerprint, regions, n_pixels=361271):
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    sdir = session.sessions_dir()
    path = os.path.join(sdir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"source": source, "dataset_fingerprint": fingerprint, "n_pixels": n_pixels,
                   "named_regions": [{"name": n, "color": "#f00"} for n in regions]}, f)
    return path


def test_resolve_session_path_adopts_richest_sibling(tmp_path, monkeypatch):
    src = "/data/negrun2.imzML"
    # a past load stranded 3 ROIs under the NA-keyed filename…
    orphan = _write(tmp_path, monkeypatch, "negrun2__aaaa.json",
                    source=src, fingerprint=FP_NA, regions=["A", "B", "C"])
    # …a poorer file also exists
    _write(tmp_path, monkeypatch, "negrun2__bbbb.json",
           source=src, fingerprint=FP_CLEAN, regions=["A"])
    # resolving the slide (canonical file doesn't exist yet) finds the richest existing one
    resolved = session.resolve_session_path(src, FP_CLEAN, 361271)
    assert resolved == orphan
    assert len(session.existing_named_regions(resolved)) == 3


def test_best_existing_session_never_crosses_slides(tmp_path, monkeypatch):
    """A different slide that happens to share a coordinate raster (same coordhash) must NOT
    be adopted — identity is basename AND geometry."""
    mine = _write(tmp_path, monkeypatch, "negrun2__aaaa.json",
                  source="/data/negrun2.imzML", fingerprint=FP_CLEAN, regions=["A", "B"])
    # same coordhash, DIFFERENT basename → a genuinely different slide
    _write(tmp_path, monkeypatch, "otherslide__cccc.json",
           source="/data/otherslide.imzML", fingerprint=FP_CLEAN, regions=["X", "Y", "Z"])
    resolved = session.resolve_session_path("/data/negrun2.imzML", FP_NA, 361271)
    assert resolved == mine                       # not the richer other-slide file
    assert sorted(r["name"] for r in session.existing_named_regions(resolved)) == ["A", "B"]


def test_resolve_prefers_canonical_when_present(tmp_path, monkeypatch):
    src = "/data/negrun2.imzML"
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    canonical = session.managed_path(src, FP_CLEAN)
    with open(canonical, "w", encoding="utf-8") as f:
        json.dump({"source": src, "dataset_fingerprint": FP_CLEAN, "n_pixels": 361271,
                   "named_regions": []}, f)
    # even though a sibling has more regions, an existing canonical file wins (it's the live one)
    _write(tmp_path, monkeypatch, "negrun2__zzzz.json",
           source=src, fingerprint=FP_NA, regions=["A", "B"])
    assert session.resolve_session_path(src, FP_CLEAN, 361271) == canonical


def test_existing_named_regions_handles_missing_and_bad(tmp_path):
    assert session.existing_named_regions(None) == []
    assert session.existing_named_regions(str(tmp_path / "nope.json")) == []
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert session.existing_named_regions(str(bad)) == []
