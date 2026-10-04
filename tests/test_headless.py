"""Headless session rehydration — the GUI's live state, rebuilt from a session file.

Covers the three things that have to be exactly right for a script (or the MCP server) to
see what the app sees: which regions resolve to which pixels, how group tags become the two
masks a comparison runs on, and that the rebuilt ``ScriptAPI`` really drives the registry.
No Qt, no real imzML — the synthetic slide plus a hand-built session file.
"""
import json
import os

import numpy as np
import pytest

from smile_msi import demo, headless, library, session


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture(scope="module")
def ds():
    d = demo.make_synthetic(width=24, height=20, seed=1)
    d.prime()
    return d


@pytest.fixture
def saved(home, ds):
    """A session saved the way the app saves one: two drawn ROIs tagged into groups, a
    cluster-backed region, an aggregate parent over the two ROIs, and a region belonging
    to a different slide."""
    _, cols = ds._pixel_rows_cols()
    left = cols < ds.width / 2
    labels = np.where(left, 0, 1)
    data = session.build_session(
        source=ds.source,
        settings={"ppm": 20.0, "norm": "rms", "reduce": "max", "id_ppm": 3.0},
        peaks=[{"mz": 700.5, "rel_intensity": 1.0}, {"mz": 800.5, "rel_intensity": 0.5}],
        active_mz=700.5, labels=labels, n_clusters=2,
        named_regions=[
            {"name": "L", "color": "#e6194b", "mask": left, "group": "Group A"},
            {"name": "R", "color": "#3cb44b", "mask": ~left, "group": "Group B"},
            {"name": "cluster 1", "color": "#4363d8", "segments": [1]},
            {"name": "both", "color": "#f58231"},
            {"name": "elsewhere", "color": "#911eb4", "mask": left,
             "sample": "/some/other/slide.imzML"},
        ],
        n_pixels=ds.n_pixels, dataset_fingerprint=library.dataset_fingerprint(ds))
    for rg in data["named_regions"]:                 # parent tie the builder doesn't set
        if rg["name"] in ("L", "R"):
            rg["parent"] = "both"
    path = os.path.join(session.sessions_dir(), "demo__test.json")
    session.save_session(path, data)
    return path


# --------------------------------------------------------------------------- #
# discovery + summary (no dataset opened)
# --------------------------------------------------------------------------- #
def test_managed_sessions_flags_a_missing_source(home, saved):
    rows = headless.managed_sessions()
    assert [r["name"] for r in rows] == ["Demo dataset"]
    assert rows[0]["source_exists"] is False        # the synthetic slide has no file on disk


def test_resolve_session_ref_accepts_path_stem_and_name(home, saved):
    stem = saved.removesuffix(".json")
    assert headless.resolve_session_ref(saved) == saved
    assert headless.resolve_session_ref(stem) == saved
    assert headless.resolve_session_ref("Demo dataset") == saved


def test_resolve_session_ref_lists_what_it_knows(home, saved):
    with pytest.raises(FileNotFoundError, match="Demo dataset"):
        headless.resolve_session_ref("no-such-slide")


def test_session_summary_is_shallow_and_json_safe(home, saved):
    s = headless.session_summary(headless.read_session(saved), path=saved)
    json.dumps(s)                                    # must survive a protocol round-trip
    assert s["n_features"] == 2 and s["settings"]["norm"] == "rms"
    assert s["groups"] == ["Group A", "Group B"]
    assert s["segmentation"] == {"n_clusters": 2}
    assert {r["name"] for r in s["regions"]} == {"L", "R", "cluster 1", "both", "elsewhere"}


# --------------------------------------------------------------------------- #
# regions → masks
# --------------------------------------------------------------------------- #
def test_region_masks_resolve_drawn_clusters_parents_and_slide_tie(home, saved, ds):
    data = headless.read_session(saved)
    masks, groups = headless.region_masks(data, ds.n_pixels, source=ds.source)

    _, cols = ds._pixel_rows_cols()
    left = cols < ds.width / 2
    assert np.array_equal(masks["L"], left)                     # drawn ROI
    assert np.array_equal(masks["cluster 1"], ~left)            # cluster-backed
    assert np.array_equal(masks["both"], left | ~left)          # aggregate parent
    assert "elsewhere" not in masks                             # tagged to another slide
    assert np.array_equal(groups["Group A"], left)
    assert np.array_equal(groups["Group B"], ~left)


def test_region_masks_keep_other_slides_when_no_source_given(home, saved, ds):
    masks, _ = headless.region_masks(headless.read_session(saved), ds.n_pixels)
    assert "elsewhere" in masks


def test_region_masks_ignore_a_segmentation_of_the_wrong_length(home, saved, ds):
    data = headless.read_session(saved)
    data["segmentation"]["labels"] = [0, 1, 0]                  # a different slide's labels
    masks, _ = headless.region_masks(data, ds.n_pixels, source=ds.source)
    assert "cluster 1" not in masks                             # no mask beats a wrong mask


def test_region_masks_survive_a_parent_cycle(home, saved, ds):
    data = headless.read_session(saved)
    for rg in data["named_regions"]:                            # both → L → both
        if rg["name"] == "both":
            rg["parent"] = "L"
        if rg["name"] == "L":
            rg["parent"] = "both"
    masks, _ = headless.region_masks(data, ds.n_pixels, source=ds.source)
    assert masks["L"].any()


# --------------------------------------------------------------------------- #
# binding + running
# --------------------------------------------------------------------------- #
def test_bind_matches_the_script_consoles_api(home, saved, ds):
    slide = headless.bind(ds, headless.read_session(saved), session_path=saved)
    assert (slide.api.ppm, slide.api.norm, slide.api.reduce) == (20.0, "rms", "max")
    assert slide.api.active_mz == 700.5
    assert slide.api.get_features() == [700.5, 800.5]
    assert slide.api.group_names() == ["Group A", "Group B"]


def test_the_active_feature_scope_beats_the_slides_own_peaks(home, saved, ds):
    """A sample whose working set was switched to a scope must not analyse the slide's
    top-level peaks instead (gui/main.py:1050)."""
    data = headless.read_session(saved)
    data["feature_scopes"] = {"sample 2": [{"mz": 900.1}, {"mz": 901.2}, {"mz": 902.3}]}
    data["active_feature_scope"] = "sample 2"
    assert headless.bind(ds, data).api.get_features() == [900.1, 901.2, 902.3]


def test_an_inactive_scope_falls_back_to_the_saved_peaks(home, saved, ds):
    data = headless.read_session(saved)
    data["feature_scopes"] = {"sample 2": [{"mz": 900.1}]}
    data["active_feature_scope"] = None
    assert headless.bind(ds, data).api.get_features() == [700.5, 800.5]


def test_the_saved_rotation_is_restored(home, saved, ds):
    """Images and masks must come out at the orientation the user sees (gui/main.py:1021)."""
    data = headless.read_session(saved)
    data["settings"]["orientation"] = 1
    try:
        headless.bind(ds, data)
        assert int(getattr(ds, "orientation", 0)) == 1
    finally:
        ds.set_orientation(0)


def test_summary_reports_what_actually_resolved(home, saved, ds):
    slide = headless.bind(ds, headless.read_session(saved), session_path=saved)
    s = slide.summary()
    assert s["resolved_groups"]["Group A"] == int((ds._pixel_rows_cols()[1] < ds.width / 2).sum())
    assert s["working_features"] == 2


def test_open_demo_runs_a_registry_analysis_end_to_end():
    slide = headless.open_demo(width=24, height=20, seed=1)
    slide.api.find_peaks(snr=3, max_peaks=20)
    df = slide.api.run_analysis("roi_comparison", method="mwu")
    assert len(df) == len(slide.api.get_features())
    assert {"mz", "AUC", "q_value"} <= set(df.columns)


def test_open_slide_refuses_a_slide_that_moved(home, saved):
    with pytest.raises(FileNotFoundError, match="not where it was left"):
        headless.open_slide(saved)                   # synthetic source has no file on disk


def test_open_dataset_names_the_missing_ibd(tmp_path):
    imzml = tmp_path / "slide.imzML"
    imzml.write_text("<mzML/>")
    with pytest.raises(FileNotFoundError, match="slide.ibd"):
        headless.open_dataset(str(imzml))


# --------------------------------------------------------------------------- #
# logged analysis runs
# --------------------------------------------------------------------------- #
def test_run_result_reads_back_a_stored_table(home, saved):
    import pandas as pd

    from smile_msi import runs

    store = headless.run_store(saved)
    run = runs.new_run("roi_comparison", title="Region comparison")
    store.save(run)
    store.save_result(run, pd.DataFrame({"mz": [700.5], "AUC": [0.9]}))

    got = headless.run_result(saved, run.run_id)
    assert list(got.columns) == ["mz", "AUC"] and float(got["AUC"][0]) == 0.9
