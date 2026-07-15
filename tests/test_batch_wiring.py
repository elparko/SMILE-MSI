"""Plan-01 wiring tests: SampleRef.batch, Cohort batch helpers, cohort.batch_design /
correct_batches over real sessions, the profile knobs, and the pooled_embedding seam."""
import numpy as np
import pytest

from smile_msi import cohort, multivariate, profiles, session


def _peak(mz, rel):
    return {"mz": float(mz), "intensity": float(rel * 100), "snr": 10.0,
            "rel_intensity": float(rel)}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------- data model
def test_sampleref_batch_roundtrip():
    s = cohort.SampleRef(name="x", session_path="/p.json", group="G0", batch="B1")
    d = s.to_dict()
    assert d["batch"] == "B1"
    assert cohort.SampleRef.from_dict(d).batch == "B1"
    # a legacy roster (v1, no batch key) loads with batch=""
    legacy = {k: v for k, v in d.items() if k != "batch"}
    assert cohort.SampleRef.from_dict(legacy).batch == ""


def test_cohort_batch_helpers():
    c = cohort.Cohort(name="c")
    c.add(cohort.SampleRef(name="a", session_path="/a.json", group="G0", meta={"acq": "day1"}))
    c.add(cohort.SampleRef(name="b", session_path="/b.json", group="G1", meta={"acq": "day2"}))
    c.add(cohort.SampleRef(name="d", session_path="/d.json", group="G0", meta={"acq": "day1"}))
    assert c.batches() == []                          # none assigned yet
    assert c.batch_by_meta("acq") == 3
    assert set(c.batches()) == {"day1", "day2"}
    assert {k: [s.name for s in v] for k, v in c.by_batch().items()} == {
        "day1": ["a", "d"], "day2": ["b"]}
    assert c.set_batch("a", "dayX")
    assert c.by_batch()["dayX"][0].name == "a"
    assert c.groups() == ["G0", "G1"]               # group axis untouched by batch assignment


def test_batch_design_alignment():
    import pandas as pd
    refs = [cohort.SampleRef(name="a", session_path="", group="G0", batch="B0", meta={"age": 5}),
            cohort.SampleRef(name="b", session_path="", group="G1", batch="B0", meta={"age": 7}),
            cohort.SampleRef(name="d", session_path="", group="G1", batch="B1", meta={"age": 9})]
    df = pd.DataFrame({"group": ["G0", "G1", "G1"], "mz_100.0000": [1.0, 2.0, 3.0]},
                      index=["a", "b", "d"])
    df.attrs["refs"] = refs
    batch, cov, biology = cohort.batch_design(df, protect=("group",), numeric_meta=("age",))
    assert list(batch) == ["B0", "B0", "B1"]
    assert list(biology) == ["G0", "G1", "G1"]
    assert cov.shape == (3, 2)                        # drop-first group one-hot (1) + age (1)
    assert cov[:, 0].tolist() == [0.0, 1.0, 1.0]      # G1 indicator
    assert cov[:, 1].tolist() == [5.0, 7.0, 9.0]      # age


# --------------------------------------------------- end-to-end over real sessions
def _mk_sample(idx, batch_shift, group_shift, targets, fp):
    rng = np.random.default_rng(idx + 100)
    peaks = [_peak(t, 10.0 + batch_shift + (group_shift if j == 0 else 0.0) + rng.normal(0, 0.2))
             for j, t in enumerate(targets)]
    src = f"/d/s{idx}.imzML"
    sess = session.build_session(source=src, settings={"mode": "negative"}, peaks=peaks,
                                 n_pixels=100, dataset_fingerprint=fp)
    return session.save_session(session.managed_path(src, fp), sess)


def test_correct_batches_removes_batch_keeps_biology(home):
    targets = [700.0, 720.0, 740.0, 760.0, 780.0, 800.0]
    refs, idx = [], 0
    # 2 batches x 2 groups x 2 reps; batch adds +6 to every feature, group adds +4 to feature[0]
    for b, bshift in enumerate([0.0, 6.0]):
        for g, gshift in enumerate([0.0, 4.0]):
            for _ in range(2):
                p = _mk_sample(idx, bshift, gshift, targets, fp=f"f{idx}")
                r = cohort.ref_from_session(p, group=f"G{g}")
                r.batch = f"B{b}"
                refs.append(r)
                idx += 1

    out, summary, mixing = cohort.correct_batches(refs, targets, protect=("group",),
                                                  random_state=0)
    feat_cols = [c for c in out.columns if c != "group"]
    assert len(feat_cols) == len(targets)
    assert summary.var_reduction > 0.3               # batch variance dropped
    g0 = out[out["group"] == "G0"][feat_cols[0]].mean()
    g1 = out[out["group"] == "G1"][feat_cols[0]].mean()
    assert (g1 - g0) > 2.0                           # group contrast preserved (planted ≈4)
    assert out.attrs["batch_correction"]["n_batches"] == 2


def test_correct_batches_single_batch_is_noop(home):
    targets = [700.0, 720.0, 740.0]
    refs = [cohort.ref_from_session(_mk_sample(i, 0.0, 0.0, targets, fp=f"n{i}"), group="G0")
            for i in range(4)]                       # no batch assigned → one batch
    out, _, _ = cohort.correct_batches(refs, targets)
    base = cohort.batch_feature_table(refs, targets, value="rel_intensity", missing="zero")
    fc = [c for c in out.columns if c != "group"]
    assert np.allclose(out[fc].to_numpy(float), base[fc].to_numpy(float))


# ------------------------------------------------------------------- profiles
def test_profile_batch_params_present():
    params = {p.key: p for p in profiles.SCHEMA}
    assert {"batch_method", "batch_level", "batch_mean_only"} <= set(params)
    assert params["batch_level"].default == "region means"
    assert "Batch correction" in profiles.GROUPS


# ----------------------------------------------------- pooled_embedding seam
class _TinyDS:
    def __init__(self, counts):
        self._counts = np.asarray(counts, dtype=float)
        self.n_pixels = int(self._counts.shape[0])

    def prime(self):
        pass

    def features_for_rows(self, peaks, rows, tol_ppm=20.0, reduce="sum", norm="none"):
        return self._counts[np.asarray(rows, dtype=int)].astype(np.float32)


def _samp(name, group, batch):
    return type("S", (), {"name": name, "group": group, "batch": batch})()


def test_pooled_embedding_batch_correct_runs_and_differs():
    rng = np.random.default_rng(0)
    targets = np.array([100.0, 200.0, 300.0, 400.0])
    a = rng.normal(5, 1, size=(8, 4))
    b = rng.normal(5, 1, size=(8, 4)) + 4.0           # batch B1 shifted up
    dss = {"A": _TinyDS(a), "B": _TinyDS(b)}
    samples = [_samp("A", "G0", "B0"), _samp("B", "G0", "B1")]
    loader = lambda s: (dss[s.name], None)

    base = multivariate.pooled_embedding(samples, targets, loader=loader, method="tsne",
                                         per_sample_cap=None, standardize_per_sample=False)
    corr = multivariate.pooled_embedding(samples, targets, loader=loader, method="tsne",
                                         per_sample_cap=None, standardize_per_sample=False,
                                         batch_correct=True)
    assert base.coords.shape == corr.coords.shape == (16, 2)
    assert np.isfinite(corr.coords).all()
    assert not np.allclose(corr.coords, base.coords)  # correction changed the input → embedding
