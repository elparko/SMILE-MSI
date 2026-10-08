"""Analysis registry tests — pure, headless (no Qt, no real imzML).

Exercises the step-type registry's shape/completeness, its default-params helper, and the
startup-perf invariant that importing it stays light. (The Flow *recipe* model + its on-disk
preset store were retired in plan 24; the registry itself lives on as smile_msi.registry.)
"""
import subprocess
import sys

from smile_msi import registry


# --------------------------------------------------------------------------- #
# registry shape + completeness
# --------------------------------------------------------------------------- #
def test_registry_is_well_formed():
    assert registry.REGISTRY, "registry must not be empty"
    for type_id, sd in registry.REGISTRY.items():
        assert sd.id == type_id
        assert sd.name and isinstance(sd.name, str)
        assert sd.category in registry.CATEGORIES, f"{type_id} has unknown category {sd.category}"
        assert sd.view and isinstance(sd.view, str)
        assert sd.needs <= registry.NEEDS, f"{type_id} declares unknown needs {sd.needs - registry.NEEDS}"
        assert sd.targets <= {"slide", "cohort"}
        # callables defaulted by __post_init__
        for fn in (sd.run, sd.peaks, sd.produce, sd.to_table, sd.rep_ions, sd.summary):
            assert callable(fn)
        # param names unique; choices present for choice kind; mz/numeric sane
        names = [p.name for p in sd.params]
        assert len(names) == len(set(names)), f"{type_id} has duplicate param names"
        for p in sd.params:
            assert p.kind in {"float", "int", "choice", "bool", "mz"}
            if p.kind == "choice":
                assert p.choices, f"{type_id}.{p.name} choice without choices"


def test_every_category_has_steps_and_targets_cover_both_run_modes():
    by_cat = registry.registry_by_category()
    # menu groups appear in declared order and are non-empty
    assert list(by_cat) == [c for c in registry.CATEGORIES if c in by_cat]
    for cat, defs in by_cat.items():
        assert defs, f"empty category {cat}"
    # at least one cohort-capable step exists (cross-slide group comparison)
    assert any("cohort" in sd.targets for sd in registry.REGISTRY.values())
    # the core group-comparison + a peak feeder are present
    for need in ("find_spatial_features", "roi_comparison", "discriminating_features", "auto_segment"):
        assert need in registry.REGISTRY


def test_default_params_match_registry_defaults():
    p = registry.default_params("find_spatial_features")
    assert p["snr"] == 3.0 and p["tol_ppm"] == 10.0 and p["norm"] == "tic"
    # the reworked spatial finder also carries footprint-frequency / Moran's-denoise floors
    assert p["min_frequency"] == 0.01 and p["min_morans"] == 0.05
    assert registry.default_params("nonexistent") == {}


# --------------------------------------------------------------------------- #
# startup-perf invariant: importing the registry must NOT pull heavy engine modules
# --------------------------------------------------------------------------- #
def test_import_does_not_load_heavy_engine_modules():
    code = ("import sys, smile_msi.registry as f; "
            "assert f.REGISTRY; "
            "heavy = [m for m in ('smile_msi.spatial', 'smile_msi.multivariate', "
            "'smile_msi.annotate', 'scipy', 'sklearn') if m in sys.modules]; "
            "assert not heavy, heavy")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# --------------------------------------------------------------------------- #
# discriminating features: filter, then cap — and say what the cap cut
# --------------------------------------------------------------------------- #
def test_top_n_cut_keeps_every_ion_tied_at_the_cut():
    import pandas as pd

    d = pd.DataFrame({"mz": [1.0, 2, 3, 4, 5], "AUC": [0.9, 0.795, 0.795, 0.795, 0.6],
                      "present": [True] * 5})
    assert list(registry._top_n_with_ties(d, 2)["mz"]) == [1, 2, 3, 4]
    assert list(registry._top_n_with_ties(d, 5)["mz"]) == [1, 2, 3, 4, 5]
    assert list(registry._top_n_with_ties(d, 1)["mz"]) == [1]


def test_discriminating_filters_before_capping_and_reports_the_cap():
    import numpy as np

    from smile_msi import demo

    ds = demo.make_synthetic(width=20, height=16, seed=2)
    mzs = [p["mz"] for p in ds.pick_peaks(snr=3, max_peaks=40)]
    rows, cols = ds._pixel_rows_cols()
    core = np.hypot((rows - ds.height / 2) / ds.height, (cols - ds.width / 2) / ds.width) < 0.2
    labels = np.where(core, 0, 1)            # 'rest': its strongest ions are core markers (depleted)
    sd = registry.REGISTRY["discriminating_features"]
    params = registry.default_params("discriminating_features")
    full = sd.run(ds, {"labels": labels, "mzs": mzs}, {**params, "top_n": 500})
    capped = sd.run(ds, {"labels": labels, "mzs": mzs}, {**params, "top_n": 3})
    assert len(full[1]) > 3                  # 'rest' has more than 3 enriched markers …
    for cl in full:
        n = len(capped[cl])
        assert n >= min(3, len(full[cl]))    # … so the cap is filled with them, not lost to the
        assert list(capped[cl]["mz"]) == list(full[cl]["mz"][:n])   # depleted ions ranked first
        assert capped.n_capped[cl] == len(full[cl]) - n
    assert "cap" in sd.summary(capped) and "cap" not in sd.summary(full)
    assert "pseudoreplication" in capped.warning


# --------------------------------------------------------------------------- #
# stochastic steps take the session seed (profiles.active_seed), as the GUI's calls do
# --------------------------------------------------------------------------- #
def test_stochastic_steps_run_with_the_session_seed(monkeypatch):
    import numpy as np
    import pytest

    from smile_msi import demo, explain, multivariate, profiles, spatial

    class Stop(Exception):
        pass

    seen = {}

    def spy(name):
        def fn(*args, **kwargs):
            seen[name] = kwargs.get("random_state")
            raise Stop
        return fn

    for mod, name in ((multivariate, "spatial_segment"), (spatial, "auto_segment"),
                      (spatial, "segment"), (multivariate, "pca_images"),
                      (multivariate, "nmf_images"), (multivariate, "embedding"),
                      (multivariate, "spatial_dgmm"), (multivariate, "cross_validate"),
                      (explain, "shap_importance")):
        monkeypatch.setattr(mod, name, spy(name))
    monkeypatch.setattr(profiles, "active_seed", lambda: 1234)
    ds = demo.make_synthetic(width=12, height=10, seed=2)
    inp = {"mzs": [885.5499, 888.6236], "labels": np.arange(ds.n_pixels) % 2,
           "target_mz": 885.5499}
    for step_id, extra in (("auto_segment", {"spatial": True}),
                           ("auto_segment", {"spatial": False, "n_clusters": 0}),
                           ("auto_segment", {"spatial": False, "n_clusters": 3}),
                           ("pca", {}), ("nmf", {}), ("embedding", {}), ("dgmm", {}),
                           ("classify_cv", {}), ("shap_biomarkers", {})):
        params = {**registry.default_params(step_id), **extra}
        with pytest.raises(Stop):
            registry.REGISTRY[step_id].run(ds, dict(inp), params)
    assert len(seen) == 9 and set(seen.values()) == {1234}
    with pytest.raises(Stop):                                  # an explicit seed still wins
        registry.REGISTRY["pca"].run(ds, dict(inp), {**registry.default_params("pca"),
                                                     "random_state": 7})
    assert seen["pca_images"] == 7
