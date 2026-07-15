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
