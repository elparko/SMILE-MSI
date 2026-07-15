"""SHAP biomarker workflow (smile_msi.explain). Skips cleanly if shap isn't installed."""
import numpy as np
import pytest

shap = pytest.importorskip("shap")          # optional dependency

from smile_msi import demo, spatial, explain


@pytest.fixture(scope="module")
def ds():
    d = demo.make_synthetic(width=26, height=20, seed=7)
    d.prime()
    return d


@pytest.fixture(scope="module")
def peaks(ds):
    return [p["mz"] for p in ds.pick_peaks(snr=3, min_rel_intensity=0.01)]


@pytest.fixture(scope="module")
def planted(ds, peaks):
    """Define two classes by a single high-variance 'biomarker' ion: pixels above its
    median are class 1. SHAP should then surface that exact ion as the top driver with a
    positive direction for the high class."""
    X = spatial.feature_matrix(ds, peaks)
    biomarker = int(np.argmax(X.var(axis=0)))
    labels = (X[:, biomarker] > np.median(X[:, biomarker])).astype(int)
    return biomarker, labels


def test_shap_importance_shapes_and_ranges(ds, peaks, planted):
    biomarker, labels = planted
    res = explain.shap_importance(ds, peaks, labels, n_estimators=120, random_state=0)

    assert res.importance.shape == (2, len(peaks))
    assert res.direction.shape == (2, len(peaks))
    assert (res.importance >= 0).all()                       # mean(|SHAP|) is non-negative
    assert np.all(res.direction >= -1) and np.all(res.direction <= 1)
    assert res.accuracy > 0.9                                # separable by construction
    assert res.n_pixels == len(labels)


def test_shap_recovers_planted_biomarker(ds, peaks, planted):
    biomarker, labels = planted
    res = explain.shap_importance(ds, peaks, labels, n_estimators=120, random_state=0)

    # class "1" = high-intensity-of-biomarker pixels; its top ion should BE the biomarker
    top_ions = [int(np.argmax(res.importance[c])) for c in range(2)]
    assert biomarker in top_ions
    # and high intensity must push *toward* the high class (positive Spearman direction)
    high_class = res.classes.index("1")
    assert res.direction[high_class, biomarker] > 0.3


def test_shap_needs_two_classes(ds, peaks):
    labels = np.zeros(ds.n_pixels, dtype=int)                # one class only
    with pytest.raises(ValueError):
        explain.shap_importance(ds, peaks, labels)


def test_shap_names_and_long_df(ds, peaks, planted):
    _, labels = planted
    res = explain.shap_importance(ds, peaks, labels, names={0: "cortex", 1: "medulla"},
                                  n_estimators=80, random_state=0)
    assert set(res.classes) == {"cortex", "medulla"}
    df = res.to_long_df()
    assert len(df) == 2 * len(peaks)
    assert set(df.columns) >= {"ftu", "mz", "importance", "direction"}


def test_shap_flow_step_registered_and_runs(ds, peaks, planted):
    from smile_msi import registry
    _, labels = planted
    sd = registry.REGISTRY["shap_biomarkers"]
    assert sd.needs == {"groups", "feature_set"}
    inp = {"mzs": peaks, "labels": labels, "names": {0: "a", 1: "b"}}
    res = sd.run(ds, inp, {"n_estimators": 80, "max_pixels": 20000, "tol_ppm": 50.0, "norm": "tic"})
    df = sd.to_table(res)
    assert list(df.columns) == ["donor", "ftu", "mz", "importance", "direction"]
    assert len(sd.rep_ions(res, 3)) == 3
    assert "regions" in sd.summary(res)


def test_concat_shap_preserves_list_and_array_forms():
    # list-of-per-class form (legacy shap return): concat each class along the sample axis
    a = [np.arange(6).reshape(3, 2), np.ones((3, 2))]
    b = [np.arange(6, 12).reshape(3, 2), np.zeros((3, 2))]
    out = explain._concat_shap([a, b])
    assert isinstance(out, list) and len(out) == 2
    assert out[0].shape == (6, 2)
    assert np.array_equal(out[0], np.vstack([a[0], b[0]]))
    # 3-D (samples, features, classes) form (modern shap return)
    rng = np.random.default_rng(0)
    a3, b3 = rng.normal(size=(3, 2, 4)), rng.normal(size=(5, 2, 4))
    out3 = explain._concat_shap([a3, b3])
    assert out3.shape == (8, 2, 4)
    assert np.array_equal(out3[:3], a3) and np.array_equal(out3[3:], b3)


def test_shap_parallel_matches_serial(ds, peaks, planted, monkeypatch):
    """Row-chunk parallelism must be numerically identical to the single-thread path —
    TreeSHAP is independent per pixel, so splitting and concatenating changes nothing."""
    _, labels = planted
    serial = explain.shap_importance(ds, peaks, labels, n_estimators=120,
                                     random_state=0, n_jobs=1)
    # force the parallel path even on the small demo dataset
    monkeypatch.setattr(explain, "_PARALLEL_MIN_ROWS", 50)
    monkeypatch.setattr(explain, "_MIN_ROWS_PER_CHUNK", 100)
    par = explain.shap_importance(ds, peaks, labels, n_estimators=120,
                                  random_state=0, n_jobs=2)
    assert par.classes == serial.classes
    assert np.allclose(par.importance, serial.importance)
    assert np.allclose(par.direction, serial.direction)


def test_normalize_importance_modes():
    v = np.array([2.0, 1.0, 1.0])
    assert np.allclose(explain.normalize_importance(v, "l1"), [0.5, 0.25, 0.25])   # sums to 1
    assert np.allclose(explain.normalize_importance(v, "max"), [1.0, 0.5, 0.5])    # top ion = 1
    assert np.allclose(explain.normalize_importance(v, "none"), v)                 # unchanged
    # 2-D: each row (donor) is normalized independently along the ion axis
    M = np.array([[2.0, 2.0], [1.0, 3.0]])
    assert np.allclose(explain.normalize_importance(M, "l1"), [[0.5, 0.5], [0.25, 0.75]])
    # zero / empty vectors are safe (no divide-by-zero, no NaN)
    assert np.allclose(explain.normalize_importance(np.zeros(3), "l1"), 0.0)
    assert explain.normalize_importance(np.array([]), "l1").size == 0
    with pytest.raises(ValueError):
        explain.normalize_importance(v, "bogus")


def test_importance_bars_normalizes_across_donors():
    """Two donors with the SAME relative biomarker profile but different model scale (donor B's
    RandomForest separates 10× more confidently): raw pooling lets B dominate and inflates the
    'SD across donors'; L1 normalization makes them comparable → the spread collapses."""
    peaks = np.array([700.0, 800.0, 900.0])
    prof = np.array([[0.6, 0.3, 0.1]])                         # (1 class) × 3 ions
    rA = explain.ShapResult(classes=["ftu"], peaks=peaks, importance=prof.copy(),
                            direction=np.zeros((1, 3)), donor="A")
    rB = explain.ShapResult(classes=["ftu"], peaks=peaks, importance=prof * 10.0,
                            direction=np.zeros((1, 3)), donor="B")
    raw = explain.importance_bars([rA, rB], "ftu", peaks, norm="none")
    assert raw["std"].max() > 0.5                             # spread is pure model scale
    l1 = explain.importance_bars([rA, rB], "ftu", peaks, norm="l1")
    assert np.allclose(l1["std"], 0.0, atol=1e-9)            # biology agrees → no spread
    assert l1["mz"][0] == 700.0 and np.isclose(l1["mean"][0], 0.6)   # strongest first, fraction


def test_shap_panel_bubble_frame_normalized():
    """Panel aggregates default to L1; bubble_frame importance sums to 1 per donor, and an
    explicit norm override is honoured."""
    peaks = np.array([700.0, 800.0, 900.0])
    r = explain.ShapResult(classes=["ftu"], peaks=peaks,
                           importance=np.array([[4.0, 4.0, 2.0]]),
                           direction=np.zeros((1, 3)), donor="A")
    panel = explain.ShapPanel(results=[r], peaks=peaks, classes=["ftu"])
    assert panel.donor_norm == "l1"
    assert np.isclose(panel.bubble_frame("ftu")["importance"].sum(), 1.0)
    assert np.isclose(panel.bubble_frame("ftu", norm="none")["importance"].sum(), 10.0)


def test_shap_panel_across_donors(ds, peaks, planted):
    _, labels = planted

    def loader():
        for i, age in enumerate((54, 61)):
            yield (f"donor{i}", ds, labels, {0: "cortex", 1: "medulla"},
                   {"age": age, "sex": "F", "bmi": 27.0 + i})

    panel = explain.shap_panel(loader(), peaks, n_estimators=80, random_state=0)
    assert len(panel.results) == 2
    assert panel.donors() == ["donor0", "donor1"]
    frame = panel.bubble_frame("cortex")
    assert len(frame) == 2 * len(peaks)                      # 2 donors × ions
    assert set(frame["age"]) == {54, 61}

    # per-category importance histogram aggregation (Farrow Fig. S151 data)
    bars = panel.histogram_bars("cortex", n=3)
    assert bars is not None
    assert len(bars["mz"]) == len(bars["mean"]) == len(bars["std"]) == 3
    assert bars["n_donors"] == 2
    assert np.all(np.diff(bars["mean"]) <= 1e-12)            # strongest first
    assert np.all(bars["std"] >= 0)
    assert np.all((bars["direction"] >= -1) & (bars["direction"] <= 1))
    # a lone result → zero-width error bars (SD over one donor is 0)
    lone = explain.importance_bars([panel.results[0]], "cortex", peaks)
    assert np.allclose(lone["std"], 0.0)
    assert explain.importance_bars(panel.results, "no-such-region", peaks) is None
    df = panel.histogram_frame("cortex", n=3)
    assert list(df.columns) == ["mz", "importance_mean", "importance_std", "direction_mean"]
    assert len(df) == 3
