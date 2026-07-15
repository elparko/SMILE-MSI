"""Pure tests for the analysis-run store + its session integration (plan 24).

Qt-free: exercises ``smile_msi.runs`` (records, digest, on-disk store) and the
``session`` sidecar path + ``analysis_runs`` index round-trip without a GUI.
"""
import numpy as np
import pandas as pd

from smile_msi import runs, session


def test_run_store_roundtrip(tmp_path):
    store = runs.RunStore(tmp_path / "s.runs")
    r = runs.new_run("dgmm", title="Per-ion", dataset="a.imzML",
                     dataset_fingerprint="fp1", inputs={"target_mz": 700.1}, params={"k": 3})
    assert r.status == "running"
    store.save(r)
    back = store.load(r.run_id)
    assert back.step_id == "dgmm" and back.params == {"k": 3}
    assert [x.run_id for x in store.list_runs()] == [r.run_id]
    assert store.delete(r.run_id) and store.list_runs() == []


def test_run_store_result_payloads(tmp_path):
    store = runs.RunStore(tmp_path / "s.runs")

    # DataFrame → CSV
    rt = runs.new_run("roi_comparison", title="t", inputs={}, params={})
    store.save_result(rt, pd.DataFrame({"mz": [1.0, 2.0], "AUC": [0.9, 0.1]}))
    assert rt.result_ref == "result.csv"
    assert list(store.load_result(rt)["mz"]) == [1.0, 2.0]

    # dict (with an ndarray) → JSON, ndarray coerced to a list
    rj = runs.new_run("dgmm", title="t", inputs={}, params={})
    store.save_result(rj, {"image": np.zeros((2, 2)), "means": [0.0, 1.0], "mz": 700.0})
    assert rj.result_ref == "result.json"
    loaded = store.load_result(rj)
    assert loaded["means"] == [0.0, 1.0] and np.asarray(loaded["image"]).shape == (2, 2)

    # None → no payload
    rn = runs.new_run("x", title="t", inputs={}, params={})
    assert store.save_result(rn, None) is None and rn.result_ref is None


def test_run_is_stale_on_fingerprint_or_inputs_drift():
    r = runs.new_run("dgmm", title="t", dataset_fingerprint="fp1",
                     inputs={"target_mz": 700.0}, params={})
    assert not r.is_stale("fp1", {"target_mz": 700.0})
    assert r.is_stale("fp2", {"target_mz": 700.0})           # dataset changed
    assert r.is_stale("fp1", {"target_mz": 701.0})           # inputs changed


def test_inputs_digest_is_order_and_type_stable():
    a = runs.new_run("x", title="t", inputs={"g": ["B", "A"], "mz": 1.0}, params={})
    b = runs.new_run("x", title="t", inputs={"mz": 1.0, "g": ["B", "A"]}, params={})
    assert a.inputs_digest() == b.inputs_digest()            # key order irrelevant
    # a NaN-bearing ndarray must not raise (regression guarded in runs._canonical)
    c = runs.new_run("x", title="t", inputs={"arr": np.array([1.0, np.nan])}, params={})
    assert isinstance(c.inputs_digest(), str)


def test_runs_dir_path_is_a_session_sidecar():
    assert session.runs_dir_path("/data/sample_x.smile.json") == "/data/sample_x.smile.runs"


def test_session_roundtrips_analysis_runs_index():
    idx = [runs.new_run("dgmm", title="t", dataset="a", inputs={"target_mz": 1.0},
                        params={"k": 3}).to_dict()]
    state = session.build_session(source="a.imzML", settings={}, peaks=[], active_mz=None,
                                  labels=None, n_clusters=None, named_regions=[], n_pixels=10,
                                  analysis_runs=idx)
    assert state["version"] == 5
    assert state["analysis_runs"] and state["analysis_runs"][0]["step_id"] == "dgmm"


def test_v4_session_loads_with_empty_analysis_runs(tmp_path):
    # a v4 document (no analysis_runs key) must load with an empty index, not KeyError.
    p = tmp_path / "old.smile.json"
    import json
    p.write_text(json.dumps({"version": 4, "source": "a.imzML", "peaks": []}))
    data = session.load_session(str(p))
    assert data.get("analysis_runs") == []
