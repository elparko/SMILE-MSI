"""Programmatic scripting engine tests — pure, headless (no Qt).

Exercises the :class:`smile_msi.scripting.ScriptAPI` facade against a synthetic slide, the
``run_script`` sandbox (output helpers, stdout capture, error reporting), the namespace /
guide self-consistency, and the on-disk workflow preset store under a throwaway
``SMILE_MSI_HOME``.
"""
import numpy as np
import pytest

from smile_msi import demo, registry, scripting


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
def api(ds):
    rows, cols = ds._pixel_rows_cols()
    left = cols < ds.width / 2
    right = ~left
    return scripting.ScriptAPI(ds, ppm=20, masks={"L": left, "R": right},
                               groups={"Group A": left, "Group B": right})


# --------------------------------------------------------------------------- #
# API facade
# --------------------------------------------------------------------------- #
def test_find_peaks_sets_working_features(api):
    peaks = api.find_peaks(snr=3, max_peaks=40)
    assert peaks and isinstance(peaks[0], dict) and "mz" in peaks[0]
    assert api.features == [float(p["mz"]) for p in peaks]
    assert api.last_peaks == peaks


def test_features_required_error_is_friendly(api):
    with pytest.raises(scripting.ScriptError, match="no features"):
        api.segment()


def test_segment_and_compare(api):
    api.find_peaks(snr=3, max_peaks=40)
    seg = api.segment(n_clusters=3)
    assert seg.n_clusters == 3 and api.segmentation is seg
    df = api.compare("Group A", "Group B")
    assert len(df) == len(api.features)
    for col in ("mz", "AUC", "q_value"):
        assert col in df.columns


def test_compare_accepts_masks_and_names(api):
    api.find_peaks(snr=3, max_peaks=30)
    df = api.compare(api.region("L"), "Group B")
    assert len(df) == len(api.features)


def test_unknown_region_message_lists_available(api):
    with pytest.raises(scripting.ScriptError, match="unknown region/group 'Nope'"):
        api.region("Nope")


def test_groups_fall_back_to_segmentation(ds):
    """With <2 tagged groups, group steps reuse the last segmentation's labels."""
    a = scripting.ScriptAPI(ds, ppm=20)
    a.find_peaks(snr=3, max_peaks=30)
    a.segment(n_clusters=3)
    labels, names = a._labels(None)
    assert names is None and set(np.unique(labels)) <= {0, 1, 2}


def test_colocalize_and_run_generic(api):
    api.find_peaks(snr=3, max_peaks=30)
    target = api.features[0]
    ranked = api.colocalize(target_mz=target)
    assert ranked and "score" in ranked[0]
    # the generic escape hatch reaches the same step set
    peaks = api.run("find_peaks", snr=4, max_peaks=20)
    assert peaks and len(peaks) <= 20


def test_annotate_returns_dataframe(api):
    api.find_peaks(snr=3, max_peaks=30)
    df = api.annotate(mode="negative")
    assert len(df) == len(api.features) and "mz" in df.columns


# --------------------------------------------------------------------------- #
# run_script sandbox
# --------------------------------------------------------------------------- #
def test_run_script_surfaces_outputs(api):
    code = (
        "peaks = find_peaks(snr=3, max_peaks=25)\n"
        "log('have', len(peaks))\n"
        "print('hello stdout')\n"
        "res = compare('Group A', 'Group B')\n"
        "table(res.head(5), 'A vs B')\n"
        "image(float(peaks[0]['mz']), 'first ion')\n"
        "record('n', len(peaks))\n"
    )
    r = scripting.run_script(code, api)
    assert r.ok and r.error is None
    assert any("have 25" in m for m in r.logs)
    assert "hello stdout" in r.stdout
    assert r.tables and r.tables[0][0] == "A vs B"
    assert r.images and r.images[0][1].ndim == 2
    assert r.values == {"n": 25}
    assert r.peaks and r.features
    assert "table(s)" in r.summary()


def test_run_script_reports_user_error_with_lineno(api):
    r = scripting.run_script("x = 1\nundefined_name + 2\n", api)
    assert not r.ok
    assert "NameError" in r.error
    assert "<workflow>" in r.error and "line 2" in r.error


def test_run_script_scripterror_is_message_only(api):
    r = scripting.run_script("segment()", api)            # no features yet
    assert not r.ok
    assert r.error.startswith("no features")
    assert "Traceback" not in r.error


def test_run_script_syntax_error(api):
    r = scripting.run_script("this is not python !!", api)
    assert not r.ok and "SyntaxError" in r.error


def test_run_script_never_raises(api):
    # a hard error inside user code must still return a result, not propagate
    r = scripting.run_script("raise RuntimeError('boom')", api)
    assert not r.ok and "boom" in r.error


@pytest.mark.parametrize("name", list(scripting.EXAMPLES))
def test_shipped_examples_run(api, name):
    """Every starter script the console offers must run clean on the demo slide."""
    api.active_mz = 888.62                                # the coloc example may use it
    r = scripting.run_script(scripting.EXAMPLES[name], api)
    assert r.ok, f"{name}: {r.error}"


# --------------------------------------------------------------------------- #
# namespace / guide self-consistency
# --------------------------------------------------------------------------- #
def test_namespace_covers_declared_funcs(api):
    r = scripting.ScriptResult()
    ns = scripting.build_namespace(api, r)
    for name in scripting.ScriptAPI.NAMESPACE_FUNCS:
        assert callable(ns[name]), name
        assert hasattr(scripting.ScriptAPI, name), name
    for helper in ("log", "table", "image", "record", "ds", "np"):
        assert helper in ns


def test_capabilities_doc_is_complete(api):
    doc = scripting.capabilities_doc(api)
    assert "Analysis Workflow Scripting Guide" in doc
    # every bare-name function is documented
    for name in scripting.ScriptAPI.NAMESPACE_FUNCS:
        assert f"`{name}(" in doc, name
    # every registry step id appears in the run() table
    for step_id in registry.REGISTRY:
        assert f"`{step_id}`" in doc, step_id
    # runtime context reflects the bound slide
    assert "runtime context" in doc and "Groups tagged" in doc


def test_capabilities_doc_static_without_api():
    doc = scripting.capabilities_doc()
    assert "Best practices" in doc and "runtime context" not in doc


# --------------------------------------------------------------------------- #
# workflow preset store
# --------------------------------------------------------------------------- #
def test_workflow_roundtrip():
    wf = scripting.Workflow(name="My WF", code="find_peaks()\n", description="demo")
    again = scripting.Workflow.from_dict(wf.to_dict())
    assert again.name == "My WF" and again.code == "find_peaks()\n" and again.description == "demo"


def test_workflow_store_save_list_load(home):
    p = scripting.Workflow(name="Nerve markers", code="segment()", description="d").save()
    assert p.startswith(str(home))
    listed = scripting.list_workflows()
    assert [w["name"] for w in listed] == ["Nerve markers"]
    assert scripting.Workflow.load(listed[0]["path"]).code == "segment()"


def test_workflow_store_skips_corrupt(home):
    scripting.Workflow(name="good", code="x").save()
    with open(scripting.workflow_path("broken"), "w", encoding="utf-8") as f:
        f.write("{ not json")
    names = [w["name"] for w in scripting.list_workflows()]
    assert "good" in names and "broken" not in names


# --------------------------------------------------------------------------- #
# grouped-ROI roi_comparison stays across-replicate (Plan 24 blocker regression)
# --------------------------------------------------------------------------- #
def _blobs_dataset():
    """A slide with four spatially-separate tissue blobs (2 → A, 2 → B) so tissue
    detection recovers four replicate samples. Off-tissue pixels are exact zeros so
    ``detect_samples`` thresholds cleanly."""
    from smile_msi.msi import MSIDataset
    W, H = 48, 32
    centers = np.array([600.0, 700.0, 810.0, 900.0])
    axis = np.arange(590.0, 910.0, 0.05)
    sig = centers / 20000.0 / 2.355
    rng = np.random.default_rng(0)
    blobs = {("A", 0): (10, 8), ("A", 1): (10, 24), ("B", 0): (38, 8), ("B", 1): (38, 24)}
    coords, mzs, ints, tag = [], [], [], []
    for r in range(H):
        for c in range(W):
            coords.append((c + 1, r + 1)); mzs.append(axis)
            key = next((k for k, (bx, by) in blobs.items()
                        if (c - bx) ** 2 + (r - by) ** 2 <= 9), None)
            tag.append(key[0] if key else "")
            if key is None:
                ints.append(np.zeros(len(axis), dtype=np.float32)); continue
            amp = 1.0 + 0.1 * rng.standard_normal(len(centers))
            if key[0] == "B":
                amp[1] *= 2.5
            else:
                amp[0] *= 2.5
            spec = np.zeros_like(axis)
            for k in range(len(centers)):
                spec += max(amp[k], 0.0) * np.exp(-0.5 * ((axis - centers[k]) / sig[k]) ** 2)
            ints.append(spec.astype(np.float32))
    ds = MSIDataset.from_arrays(coords, mzs, ints, polarity="negative", spec_mode="continuous")
    ds.prime()
    tag = np.array(tag)
    return ds, (tag == "A"), (tag == "B"), [float(m) for m in centers]


def test_run_analysis_grouped_roi_comparison_is_across_replicate():
    """A grouped-ROI (Flow ``@grouped``) roi_comparison run through ``run_analysis`` must
    summarise across replicates (``unit=='sample'``), not pseudoreplicate over pixels — the
    regression the flow→preset migration would otherwise reintroduce."""
    ds, mask_a, mask_b, feats = _blobs_dataset()
    api = scripting.ScriptAPI(ds, ppm=10, norm="tic",
                              masks={"A_roi": mask_a, "B_roi": mask_b},
                              groups={"Group A": mask_a, "Group B": mask_b})
    api.set_features(feats)
    df = api.run_analysis("roi_comparison", method="mwu", tol_ppm=10.0, norm="tic",
                          mask=(mask_a | mask_b))
    assert df.attrs["unit"] == "sample"
    assert df.attrs["n_a"] == 2 and df.attrs["n_b"] == 2


def test_run_analysis_explicit_ab_stays_per_pixel():
    """An explicit ``a=/b=`` comparison is left on the per-pixel path — the auto replicate
    derivation is scoped to the grouped-ROI default, and never silently rewrites a call the
    caller spelled out."""
    ds, mask_a, mask_b, feats = _blobs_dataset()
    api = scripting.ScriptAPI(ds, ppm=10, norm="tic",
                              masks={"A_roi": mask_a, "B_roi": mask_b},
                              groups={"Group A": mask_a, "Group B": mask_b})
    api.set_features(feats)
    df = api.run_analysis("roi_comparison", a="A_roi", b="B_roi",
                          method="mwu", tol_ppm=10.0, norm="tic")
    assert df.attrs["unit"] == "pixel"


def test_run_analysis_single_piece_falls_back_to_pixel():
    """When the slide has no resolvable replicate structure (one tissue piece), the grouped
    path degrades gracefully to per-pixel instead of crashing on an empty per-sample summary."""
    ds, mask_a, mask_b, feats = _blobs_dataset()
    rows, cols = ds._pixel_rows_cols()
    one = (cols >= 8) & (cols <= 12) & (rows >= 6) & (rows <= 10)     # a single blob
    left = cols < 10
    api = scripting.ScriptAPI(ds, ppm=10, norm="tic",
                              groups={"Group A": one & left, "Group B": one & ~left})
    api.set_features(feats)
    df = api.run_analysis("roi_comparison", method="mwu", tol_ppm=10.0, norm="tic", mask=one)
    assert df.attrs["unit"] == "pixel"


# --------------------------------------------------------------------------- #
# regions by construction (plan 25)
# --------------------------------------------------------------------------- #
def test_composite_is_the_sum_of_ion_vectors(api):
    mzs = [888.6236, 885.5499]
    expect = api.ion_vector(mzs[0]) + api.ion_vector(mzs[1])
    assert np.allclose(api.composite(mzs), expect)
    with pytest.raises(scripting.ScriptError):
        api.composite([])


def test_threshold_ring_invert_partition_the_slide(api):
    ds = api.ds
    endo = api.threshold_mask(api.composite([888.6236]), 60)
    assert endo.dtype == bool and endo.shape == (ds.n_pixels,) and endo.any()
    peri = api.ring(endo, width_px=2, mode="outer")
    assert peri.any() and not (peri & endo).any()
    epi = api.invert(endo | peri)
    assert int(endo.sum() + peri.sum() + epi.sum()) == ds.n_pixels
    assert np.array_equal(api.threshold_mask(888.6236, 60), endo)      # an m/z is accepted too
    with pytest.raises(scripting.ScriptError):
        api.ring(endo)                                                  # no width at all
    with pytest.raises(scripting.ScriptError):
        api.ring(endo, width_um=30)                                     # synthetic: no pixel size
    ds.pixel_size_um = 5.0
    try:
        assert np.array_equal(api.ring(endo, width_um=10), api.ring(endo, width_px=2))
    finally:
        ds.pixel_size_um = None


def test_add_region_stages_and_is_usable_by_name(api):
    api.find_peaks(snr=5)
    endo = api.threshold_mask(888.6236, 60)
    peri = api.ring(endo, width_px=2, mode="outer")
    epi = api.invert(endo | peri)
    api.add_region("endo", endo)
    api.add_region("peri", peri, color="#ff8800")
    api.add_region("epi", epi)
    assert [r["name"] for r in api.new_regions] == ["endo", "peri", "epi"]
    assert api.new_regions[1]["color"] == "#ff8800"
    assert "endo" in api.region_names()
    df = api.multigroup(groups=["endo", "peri", "epi"])                 # by name
    assert len(df) == len(api.features)
    api.add_region("endo", peri)                                        # re-staging replaces
    assert [r["name"] for r in api.new_regions] == ["peri", "epi", "endo"]
    with pytest.raises(scripting.ScriptError):
        api.add_region("empty", np.zeros(api.ds.n_pixels, bool))


def test_run_script_returns_staged_regions(api):
    r = scripting.run_script(
        "m = threshold_mask(888.6236, 60)\nadd_region('core', m)\n", api)
    assert r.ok, r.error
    assert [rg["name"] for rg in r.regions] == ["core"]
    assert "1 region(s)" in r.summary()
    assert any("add_region" in line for line in r.logs)


# --------------------------------------------------------------------------- #
# findings from an assistant-driven run of the demo slide
# --------------------------------------------------------------------------- #
def test_annotate_carries_the_picked_peaks_intensity_and_snr(api):
    api.find_peaks(snr=5, max_peaks=10)
    df = api.annotate()
    assert (df["intensity"] != "").all() and (df["snr"] != "").all()
    one = api.annotate(features=[round(api.features[0], 4)])   # an m/z read back off a table
    assert one.iloc[0]["intensity"] != ""


def test_threshold_mask_warns_when_filling_closes_a_ring(api):
    rows, cols = api.ds._pixel_rows_cols()
    r = np.hypot(rows - rows.mean(), cols - cols.mean())
    ring = np.where((r > 3) & (r < 6), 10.0, 0.0)                # a bright annulus
    out = scripting.run_script(
        "record('n', (int(threshold_mask(v, 50).sum()),\n"
        "             int(threshold_mask(v, 50, fill_holes=False).sum())))", api,
        extra_globals={"v": ring})
    assert out.ok, out.error
    n_filled, n_open = out.values["n"]
    assert n_filled > n_open                                     # the default fills the core
    assert sum("fill_holes=False" in line for line in out.logs) == 1   # … and says so, once


def test_segment_takes_a_mask_and_flags_background_without_one(api):
    api.find_peaks(snr=5, max_peaks=20)
    left = api.region("L")
    seg = api.segment(n_clusters=2, mask="L")
    assert (seg.labels[~left] == -1).all() and (seg.labels[left] >= 0).all()
    out = scripting.run_script("segment(n_clusters=2)", api)
    assert out.ok, out.error
    assert any("off-tissue background" in line for line in out.logs)


def test_run_uses_the_session_tolerance_like_the_named_wrappers(api):
    api.find_peaks(snr=5, max_peaks=20)
    assert registry.default_params("auto_segment")["tol_ppm"] != api.ppm
    via_run = api.run("auto_segment", n_clusters=2)
    via_wrapper = api.segment(n_clusters=2)
    assert via_run.silhouette == via_wrapper.silhouette
    assert np.array_equal(via_run.labels, via_wrapper.labels)


def test_guide_documents_stdout_ratio_image_and_hole_filling():
    doc = scripting.capabilities_doc()
    assert "captured into the run log" not in doc and "`stdout`" in doc
    assert "ds.ratio_image(" in doc
    assert "fill_holes=False" in doc and "percentile of every pixel with signal" in doc
