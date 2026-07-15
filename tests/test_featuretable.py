"""Folding every analysis into one feature-list CSV (``smile_msi.featuretable``).

Pure module, no Qt: the tables here are the real shapes the registry's ``to_table`` callables
emit — wide per-ion stats, long per-(region, m/z) markers, PCA loadings, a per-ion
segmentation, a class-keyed comparison — so a step changing shape breaks a test here rather
than silently dropping its columns out of the export.
"""
import numpy as np
import pandas as pd
import pytest

from smile_msi import featuretable as ft
from smile_msi import runs


# --------------------------------------------------------------------------- #
# the table shapes the registry actually produces
# --------------------------------------------------------------------------- #
def _roi_comparison():                       # wide, one row per m/z
    return pd.DataFrame({"mz": [700.5, 750.25], "AUC": [0.9, 0.3], "q_value": [0.01, 0.4]})


def _discriminating():                       # long, one row per (region, m/z)
    return pd.DataFrame({"region": ["endo", "endo", "peri", "peri"],
                         "mz": [700.5, 800.1, 700.5, 800.1],
                         "AUC": [0.8, 0.2, 0.4, 0.95],
                         "q_value": [0.01, 0.5, 0.3, 0.001]})


def _pca():                                  # long, one row per (component, m/z)
    return pd.DataFrame({"component": ["PC1", "PC2", "PC1", "PC2"],
                         "mz": [700.5, 700.5, 800.1, 800.1],
                         "loading": [0.4, -0.1, 0.2, 0.7]})


def _base():
    return pd.DataFrame({"mz": [700.5, 750.25, 800.1], "lipid": ["PE 34:1", "", "PC 36:2"]})


# --------------------------------------------------------------------------- #
# attach: which analyses can contribute columns at all
# --------------------------------------------------------------------------- #
def test_wide_table_attaches_its_columns_namespaced():
    a = ft.attach(_roi_comparison(), prefix="roi_comparison")
    assert a.usable
    assert a.columns == ["roi_comparison.AUC", "roi_comparison.q_value"]
    assert a.n_features == 2


def test_long_table_pivots_to_one_column_per_key():
    a = ft.attach(_discriminating(), prefix="disc")
    assert a.columns == ["disc.endo.AUC", "disc.endo.q_value",
                         "disc.peri.AUC", "disc.peri.q_value"]
    assert "pivoted" in a.reason
    # a feature stays exactly one row — the whole point of the export
    assert a.n_features == 2


def test_pivot_columns_group_by_key_and_sort_naturally():
    df = _pca()
    df.loc[len(df)] = ["PC10", 700.5, 0.05]
    df.loc[len(df)] = ["PC10", 800.1, 0.06]
    a = ft.attach(df, prefix="pca")
    # PC2 before PC10 (natural, not lexical), and each component's columns adjacent
    assert a.columns == ["pca.pc1.loading", "pca.pc2.loading", "pca.pc10.loading"]


def test_a_numeric_key_value_borrows_its_column_name():
    # PCA's registry table numbers its components 0…4 — 'pca.0.loading' names nothing
    df = pd.DataFrame({"component": [0, 1, 0, 1], "mz": [700.5, 700.5, 800.1, 800.1],
                       "loading": [0.4, -0.1, 0.2, 0.7]})
    assert ft.attach(df, prefix="pca").columns == ["pca.component0.loading",
                                                   "pca.component1.loading"]


@pytest.mark.parametrize("table, expect", [
    (pd.DataFrame({"class": ["PE", "PC"], "AUC": [0.7, 0.3]}), "no m/z column"),
    (pd.DataFrame({"level": [0, 1, 2], "mz": [700.5] * 3, "mean_intensity": [1.0, 5.0, 9.0]}),
     "single ion"),
    (pd.DataFrame({"mz": [700.5, 800.1]}), "no columns besides m/z"),
    (pd.DataFrame({"mz": [], "AUC": []}), "empty"),
    (None, "no result table"),
    # an image / spectra result persists as JSON, so load_result hands back a dict
    ({"image": [[1, 2]], "mz": 700.5}, "not a table"),
])
def test_non_feature_results_are_refused_with_a_reason(table, expect):
    a = ft.attach(table, prefix="x")
    assert not a.usable
    assert expect in a.reason


def test_single_feature_list_is_not_mistaken_for_a_single_ion_result():
    # one row, one m/z: a legitimate one-feature comparison, not a per-ion segmentation
    a = ft.attach(pd.DataFrame({"mz": [700.5], "AUC": [0.9]}), prefix="roi")
    assert a.usable and a.columns == ["roi.AUC"]


def test_unkeyed_duplicate_rows_keep_the_first_and_say_so():
    df = pd.DataFrame({"mz": [700.5, 700.5, 800.1], "score": [0.9, 0.1, 0.5]})
    a = ft.attach(df, prefix="x")
    assert a.usable and a.n_features == 2
    assert a.table["x.score"].iloc[0] == 0.9
    assert "kept the first" in a.reason


def test_a_float_statistic_is_never_chosen_as_the_pivot_key():
    # 'score' is unique per row and would make (mz, score) unique — but it is the data,
    # not a key. With no categorical key the table falls back to first-row-per-m/z.
    df = pd.DataFrame({"mz": [700.5, 700.5, 800.1], "score": [0.9, 0.1, 0.5],
                       "extra": [1.0, 2.0, 3.0]})
    a = ft.attach(df, prefix="x")
    assert "kept the first" in a.reason


def test_high_cardinality_key_is_rejected_as_a_pivot():
    n = ft.MAX_PIVOT_VALUES + 1
    df = pd.DataFrame({"tag": [f"t{i}" for i in range(n)] * 2,
                       "mz": [700.5] * n + [800.1] * n,
                       "v": list(range(2 * n))})
    a = ft.attach(df, prefix="x")
    assert "kept the first" in a.reason          # would have exploded into n columns


def test_pivot_key_values_that_slug_alike_stay_distinct_columns():
    df = pd.DataFrame({"region": ["endo", "endo ", "endo", "endo "],
                       "mz": [700.5, 700.5, 800.1, 800.1], "AUC": [0.1, 0.2, 0.3, 0.4]})
    a = ft.attach(df, prefix="x")
    assert len(set(a.columns)) == 2             # no duplicate header


def test_unnamed_index_column_from_a_csv_round_trip_is_dropped():
    df = pd.DataFrame({"Unnamed: 0": [0, 1], "mz": [700.5, 750.25], "AUC": [0.9, 0.3]})
    assert ft.attach(df, prefix="x").columns == ["x.AUC"]


# --------------------------------------------------------------------------- #
# the m/z join
# --------------------------------------------------------------------------- #
def test_nearest_within_respects_the_ppm_window():
    ref = np.array([700.5, 800.1])
    # +2 ppm on 700.5 is ~0.0014 Da: inside a 5 ppm window, outside a 1 ppm one
    q = np.array([700.5 * (1 + 2e-6)])
    assert ft.nearest_within(q, ref, 5.0)[0] == 0
    assert ft.nearest_within(q, ref, 1.0)[0] == -1


def test_nearest_within_handles_empty_and_nan():
    assert ft.nearest_within([700.0], np.array([]), 5.0).tolist() == [-1]
    assert ft.nearest_within([], np.array([700.0]), 5.0).tolist() == []
    assert ft.nearest_within([np.nan], np.array([700.0]), 5.0).tolist() == [-1]


def test_nearest_within_picks_the_closer_of_two_candidates():
    ref = np.array([700.5, 700.5001])
    assert ft.nearest_within([700.50008], ref, 500.0)[0] == 1


def test_join_keeps_every_feature_and_nans_the_unscored_ones():
    out, notes = ft.join_by_mz(_base(), [ft.attach(_roi_comparison(), prefix="roi", title="ROI")])
    assert len(out) == 3                            # 800.1 was never scored — still exported
    assert out["roi.AUC"].tolist()[:2] == [0.9, 0.3]
    assert np.isnan(out["roi.AUC"].iloc[2])
    assert "2 of 3 features matched" in notes[0]


def test_join_survives_a_recalibrated_feature_list():
    base = _base()
    base["mz"] = base["mz"] * (1 + 3e-6)            # +3 ppm since the analysis ran
    out, _ = ft.join_by_mz(base, [ft.attach(_roi_comparison(), prefix="roi")], tol_ppm=5.0)
    assert out["roi.AUC"].iloc[0] == 0.9            # an equality join would have dropped it
    out, _ = ft.join_by_mz(base, [ft.attach(_roi_comparison(), prefix="roi")], tol_ppm=1.0)
    assert np.isnan(out["roi.AUC"].iloc[0])


def test_join_stacks_several_analyses_side_by_side():
    atts = [ft.attach(_roi_comparison(), prefix="roi"),
            ft.attach(_discriminating(), prefix="disc"),
            ft.attach(_pca(), prefix="pca"),
            ft.attach(pd.DataFrame({"class": ["PE"], "AUC": [0.5]}), prefix="cls")]
    out, notes = ft.join_by_mz(_base(), atts)
    assert list(out.columns) == [
        "mz", "lipid", "roi.AUC", "roi.q_value",
        "disc.endo.AUC", "disc.endo.q_value", "disc.peri.AUC", "disc.peri.q_value",
        "pca.pc1.loading", "pca.pc2.loading"]      # 'cls' is not per-feature → contributes nothing
    assert len(notes) == 3
    assert out["disc.peri.AUC"].iloc[2] == 0.95


def test_join_preserves_bool_and_string_columns():
    t = pd.DataFrame({"mz": [700.5, 750.25], "kept": [True, False], "note": ["a", "b"]})
    out, _ = ft.join_by_mz(_base(), [ft.attach(t, prefix="f")])
    assert out["f.kept"].tolist()[:2] == [True, False]
    assert out["f.note"].tolist()[:2] == ["a", "b"]
    assert pd.isna(out["f.note"].iloc[2])           # unmatched feature, not the string 'nan'


def test_join_needs_an_mz_column_on_the_base():
    with pytest.raises(ValueError, match="mz"):
        ft.join_by_mz(pd.DataFrame({"lipid": ["PE"]}), [])


def test_join_with_no_analyses_returns_the_list_unchanged():
    out, notes = ft.join_by_mz(_base(), [])
    assert notes == [] and list(out.columns) == ["mz", "lipid"]


# --------------------------------------------------------------------------- #
# runs → attachments
# --------------------------------------------------------------------------- #
def _run(step_id, created, run_id, status="done", title=None):
    r = runs.new_run(step_id, title=(title or step_id), now=created)
    r.run_id, r.status = run_id, status
    return r


def test_prefixes_are_the_registry_id_and_only_ordinal_when_repeated():
    rs = [_run("roi_comparison", "2026-07-01", "b"), _run("pca", "2026-07-02", "c"),
          _run("roi_comparison", "2026-07-03", "d")]
    p = ft.run_prefixes(rs)
    assert p == {"b": "roi_comparison_1", "d": "roi_comparison_2", "c": "pca"}


def test_repeated_runs_are_named_by_what_made_them_different():
    # AnalysisDialog._run_title stamps the contrast after an em-dash
    rs = [_run("roi_comparison", "2026-07-01", "b", title="Region comparison — endo vs peri"),
          _run("roi_comparison", "2026-07-03", "d", title="Region comparison — epi vs peri")]
    assert ft.run_prefixes(rs) == {"b": "roi_comparison_endo_vs_peri",
                                   "d": "roi_comparison_epi_vs_peri"}


@pytest.mark.parametrize("titles", [
    ("Region comparison", "Region comparison — endo vs peri"),      # one says nothing
    ("Region comparison — endo vs peri", "Region comparison — endo vs peri"),   # both the same
])
def test_ordinals_return_when_the_titles_cannot_tell_the_runs_apart(titles):
    rs = [_run("roi_comparison", "2026-07-01", "b", title=titles[0]),
          _run("roi_comparison", "2026-07-03", "d", title=titles[1])]
    assert ft.run_prefixes(rs) == {"b": "roi_comparison_1", "d": "roi_comparison_2"}


def test_prefixes_are_stable_whatever_order_the_runs_arrive_in():
    rs = [_run("roi_comparison", "2026-07-03", "d"), _run("roi_comparison", "2026-07-01", "b")]
    assert ft.run_prefixes(rs) == ft.run_prefixes(list(reversed(rs)))


def test_attachments_skip_unfinished_runs_and_grey_out_unreadable_ones():
    ok, failed, running = (_run("roi_comparison", "2026-07-01", "a"),
                           _run("pca", "2026-07-02", "b"),
                           _run("nmf", "2026-07-03", "c", status="running"))

    def load(run):
        if run.run_id == "b":
            raise OSError("payload gone")
        return _roi_comparison()

    atts = ft.attachments_from_runs([ok, failed, running], load, name_of=lambda r: r.step_id.upper())
    assert [a.key for a in atts] == ["a", "b"]      # the running one never reaches the picker
    assert atts[0].usable and atts[0].title == "ROI_COMPARISON"
    assert not atts[1].usable and "could not be read" in atts[1].reason


def test_attachments_carry_the_run_summary_as_a_subtitle():
    r = _run("roi_comparison", "2026-07-01", "a")
    r.summary = "12 ions q<0.05 of 300"
    a = ft.attachments_from_runs([r], lambda _r: _roi_comparison())[0]
    assert "12 ions q<0.05" in a.subtitle and "2026-07-01" in a.subtitle


def test_the_picker_title_keeps_a_runs_contrast_but_prefers_the_live_analysis_name():
    plain = _run("roi_comparison", "2026-07-01", "a", title="An old name")
    keyed = _run("roi_comparison", "2026-07-02", "b", title="Old name — endo vs peri")
    atts = ft.attachments_from_runs([plain, keyed], lambda _r: _roi_comparison(),
                                    name_of=lambda _r: "Region comparison (A vs B)")
    assert atts[0].title == "Region comparison (A vs B)"        # renamed step, nothing to add
    assert atts[1].title == "Old name — endo vs peri"           # the contrast is worth keeping


# --------------------------------------------------------------------------- #
# the payload that used to be lost (runs.json_safe)
# --------------------------------------------------------------------------- #
def test_json_safe_rejects_a_dict_carrying_a_dataframe():
    # what _run_filter_auc returns — json.dump would persist str(df) and the run would
    # reopen as a meaningless string, so it must persist its flat table instead
    assert not runs.json_safe({"full": pd.DataFrame({"mz": [1.0]}), "cut": 0.7})
    # what _run_marker_panel returns: the frames sit inside a list of dicts
    assert not runs.json_safe({"pairs": [{"full": pd.DataFrame({"mz": [1.0]})}]})


def test_json_safe_accepts_the_dicts_that_really_do_round_trip():
    assert runs.json_safe({"image": np.zeros((3, 3)), "means": [1.0, 2.0], "mz": 700.5,
                           "names": ("a", "b"), "ok": True, "none": None})


def test_json_safe_rejects_an_object_the_hook_would_stringify():
    class Opaque:
        pass

    assert not runs.json_safe({"x": Opaque()})


def test_a_dataframe_payload_round_trips_through_the_run_store(tmp_path):
    store = runs.RunStore(tmp_path)
    run = runs.new_run("filter_auc", title="Keep features by AUC")
    store.save_result(run, _roi_comparison())
    assert run.result_ref == "result.csv"
    back = store.load_result(run)
    # and it is immediately attachable — the point of persisting the table, not the dict
    assert ft.attach(back, prefix="filter_auc").columns == ["filter_auc.AUC",
                                                            "filter_auc.q_value"]
