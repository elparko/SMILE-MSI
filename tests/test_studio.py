"""Tests for the Export Studio engine (smile_msi.studio) — the pure, Qt-free batch
compiler + runner. No display, no MSIDataset, no matplotlib: every heavyweight step is a
fake callback, so these prove the orchestration (cross-product, source-outer load/evict,
shared contrast, manifest) in isolation, mirroring tests/test_export.py's headless style.
"""
import csv
import os

import numpy as np

from smile_msi import studio
from smile_msi.studio import (ContrastMode, OutputKind, Section, SelectionRecord,
                               compile_plan, run_plan)


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def _sections(n=2):
    return [Section(sid=f"s{i}", name=f"section_{i:02d}", source=f"/data/s{i}.imzML",
                    session_path=f"/data/s{i}.json") for i in range(n)]


def _selections():
    return [
        SelectionRecord(mz=744.5548, label="PC 34:1", list_name="PC_list", source_kind="saved"),
        SelectionRecord(mz=810.5291, label="PE 40:6", list_name="PC_list", source_kind="saved"),
        SelectionRecord(mz=None, label="Total ST", list_name="classes", kind="composite",
                        source_kind="lipid_class", lipid_class="ST", member_mzs=(888.6, 890.6)),
    ]


# --------------------------------------------------------------------------- #
# safe_name + selection identity
# --------------------------------------------------------------------------- #
def test_safe_name_is_windows_safe():
    assert studio.safe_name("PG 46:2") == "PG_46-2"
    assert studio.safe_name('a/b\\c:d*e?f"g<h>i|j') == "a-b-c-d-e-f-g-h-i-j"
    assert studio.safe_name("   ...  ") == "item"           # never empty


def test_selection_uid_and_stem():
    s = SelectionRecord(mz=744.5548, label="PC 34:1", list_name="PC list")
    assert s.uid() == "PC_list::ion::744.5548"
    assert s.file_stem() == "ion_744.5548_PC_34-1"
    c = SelectionRecord(label="Total ST", list_name="classes", kind="composite", lipid_class="ST")
    assert c.file_stem() == "class_ST"
    assert "composite" in c.uid()


# --------------------------------------------------------------------------- #
# compile_plan
# --------------------------------------------------------------------------- #
def test_compile_plan_cross_product_cardinality():
    secs, sels = _sections(2), _selections()
    plan = compile_plan(secs, sels, [OutputKind.ION_IMAGES])
    assert plan.job_count() == len(secs) * len(sels)        # 2 × 3 = 6


def test_compile_plan_is_section_outer_in_order():
    secs, sels = _sections(2), _selections()
    plan = compile_plan(secs, sels, [OutputKind.ION_IMAGES])
    # all of section 0's jobs come before any of section 1's (render order = source-outer)
    sids = [j.section.sid for j in plan.jobs]
    assert sids == ["s0"] * len(sels) + ["s1"] * len(sels)


def test_compile_plan_dedupes_same_cell():
    secs = _sections(1)
    dup = SelectionRecord(mz=744.5548, label="PC 34:1", list_name="L")
    plan = compile_plan(secs, [dup, dup], [OutputKind.ION_IMAGES])
    assert plan.job_count() == 1                            # same (section, selection) → one job


def test_compile_plan_layout_paths():
    secs, sels = _sections(1), _selections()[:1]
    lo = compile_plan(secs, sels, [OutputKind.ION_IMAGES], layout="list_outer", ext="png")
    assert lo.jobs[0].path == os.path.join("PC_list", "section_00", "ion_744.5548_PC_34-1.png")
    so = compile_plan(secs, sels, [OutputKind.ION_IMAGES], layout="section_outer", ext="tiff")
    assert so.jobs[0].path == os.path.join("section_00", "PC_list", "ion_744.5548_PC_34-1.tiff")


def test_compile_plan_no_ion_jobs_when_not_requested():
    plan = compile_plan(_sections(2), _selections(), [OutputKind.LIST_CSV])
    assert plan.job_count() == 0                            # CSV-only run renders no images
    assert plan.wants(OutputKind.LIST_CSV) and not plan.wants(OutputKind.ION_IMAGES)


# --------------------------------------------------------------------------- #
# run_plan — load-once / evict / render / skip / manifest
# --------------------------------------------------------------------------- #
def _fake_run(tmp_path, secs, sels, *, contrast=ContrastMode.PER_SECTION,
              absent=None, fail_source=None):
    """Drive run_plan with fakes; return (result, telemetry dict)."""
    absent = absent or set()                                # (sid, sel_uid) cells that are 'absent'
    tel = {"loads": [], "releases": [], "live_at_load": [], "rendered": []}
    live = {}                                               # load_key → currently-loaded sentinel

    def loader(section):
        if fail_source and section.source == fail_source:
            raise RuntimeError("boom")
        key = section.load_key()
        if key not in live:                                 # one load per source
            live[key] = object()
            tel["loads"].append(key)
        tel["live_at_load"].append(len(live))               # how many cubes are live right now
        return ("ds", section.sid)

    def extract(ds, section, sel):
        if (section.sid, sel.uid()) in absent:
            return None
        # a deterministic little image whose brightness varies by section, for shared-contrast
        base = 1.0 + 0.5 * int(section.sid[1:])
        return np.full((4, 4), base, float)

    def measure(arr):
        return float(np.nanmax(arr))

    def release(load_key):
        live.pop(load_key, None)
        tel["releases"].append(load_key)

    def render_image(arr, *, selection, section, anchor, path):
        tel["rendered"].append((section.sid, selection.uid(), anchor))
        with open(path, "w") as f:                          # write a real (tiny) file
            f.write("x")

    plan = compile_plan(secs, sels, [OutputKind.ION_IMAGES], contrast=contrast)
    result = run_plan(plan, loader=loader, extract=extract, render_image=render_image,
                      measure=measure, release=release, out_dir=str(tmp_path))
    return result, tel


# --------------------------------------------------------------------------- #
# run_plan — parallel render phase
# --------------------------------------------------------------------------- #
class _SyncExecutor:
    """Runs each submitted task immediately in-process — stands in for the spawn process pool
    so the parallel orchestration is tested deterministically (no real subprocess to drive)."""
    def __init__(self, max_workers=None):
        pass

    def submit(self, fn, *args, **kw):
        from concurrent.futures import Future
        fut = Future()
        try:
            fut.set_result(fn(*args, **kw))
        except BaseException as e:  # noqa: BLE001
            fut.set_exception(e)
        return fut

    def shutdown(self, wait=True, cancel_futures=False):
        pass


def _fake_build_spec(arr, selection, section, anchor):
    return {"sid": section.sid, "uid": selection.uid(), "anchor": anchor}


def _fake_render_cell(image, spec, path):
    with open(path, "w") as f:                              # write a real (tiny) file
        f.write("x")
    return path


def test_run_plan_parallel_renders_every_cell(tmp_path, monkeypatch):
    monkeypatch.setattr(studio, "_make_render_executor", lambda mw: _SyncExecutor())
    secs, sels = _sections(4), _selections()                # 12 cells ≥ _PARALLEL_MIN_JOBS

    def loader(section):
        return ("ds", section.sid)

    def extract(ds, section, sel):
        return np.full((4, 4), 1.0, float)

    def boom_render_image(*a, **k):                          # serial path must NOT be used
        raise AssertionError("serial render_image used; expected the parallel path")

    plan = compile_plan(secs, sels, [OutputKind.ION_IMAGES])
    seen = []
    result = run_plan(plan, loader=loader, extract=extract, render_image=boom_render_image,
                      out_dir=str(tmp_path), on_path=seen.append,
                      render_cell=_fake_render_cell, build_spec=_fake_build_spec, max_workers=4)
    n = len(secs) * len(sels)
    assert not result.cancelled
    assert sum(1 for j in result.jobs if j.status == "done") == n
    assert len(seen) == n                                    # on_path fired once per rendered cell
    assert all(os.path.exists(os.path.join(str(tmp_path), j.path)) for j in result.jobs)
    assert os.path.exists(os.path.join(str(tmp_path), "manifest.csv"))


def test_run_plan_small_batch_stays_serial(tmp_path, monkeypatch):
    # Below the threshold the process pool's spawn cost isn't worth it → render serially; the
    # executor factory must never be called.
    monkeypatch.setattr(studio, "_make_render_executor",
                        lambda mw: (_ for _ in ()).throw(AssertionError("pool used for small batch")))
    secs, sels = _sections(1), _selections()                # 3 cells < _PARALLEL_MIN_JOBS
    used = []

    def loader(section):
        return ("ds", section.sid)

    def extract(ds, section, sel):
        return np.full((4, 4), 1.0, float)

    def render_image(arr, *, selection, section, anchor, path):
        used.append(path)
        with open(path, "w") as f:
            f.write("x")

    plan = compile_plan(secs, sels, [OutputKind.ION_IMAGES])
    result = run_plan(plan, loader=loader, extract=extract, render_image=render_image,
                      out_dir=str(tmp_path),
                      render_cell=_fake_render_cell, build_spec=_fake_build_spec, max_workers=4)
    assert len(used) == len(sels)                            # serial render_image did the work
    assert all(j.status == "done" for j in result.jobs)


def test_run_plan_writes_tree_and_manifest(tmp_path):
    secs, sels = _sections(2), _selections()
    result, tel = _fake_run(tmp_path, secs, sels)
    # every cell rendered to a real file under the chosen layout
    assert len(tel["rendered"]) == len(secs) * len(sels)
    for job in result.jobs:
        assert job.status == "done"
        assert os.path.exists(os.path.join(str(tmp_path), job.path))
    # manifest.csv + README.md exist and the manifest has a row per cell
    man = os.path.join(str(tmp_path), "manifest.csv")
    assert os.path.exists(man) and os.path.exists(os.path.join(str(tmp_path), "README.md"))
    with open(man, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(secs) * len(sels)
    assert all(r["status"] == "done" for r in rows)


def test_run_plan_loads_each_source_once_and_evicts(tmp_path):
    # two region-samples of ONE slide + a second slide → the shared slide loads once
    shared = "/data/slideA.imzML"
    secs = [
        Section(sid="r1", name="A_fascicle", source=shared, region="fascicle"),
        Section(sid="r2", name="A_epineurium", source=shared, region="epineurium"),
        Section(sid="b", name="slideB", source="/data/slideB.imzML"),
    ]
    result, tel = _fake_run(tmp_path, secs, _selections()[:1])
    # slideA loaded once (shared by its two regions), slideB once → 2 loads, not 3
    assert tel["loads"].count(shared) == 1
    assert len(tel["loads"]) == 2
    # peak live-cube count never exceeded 1 (each source evicted before the next loads)
    assert max(tel["live_at_load"]) == 1
    # every source released exactly once
    assert sorted(tel["releases"]) == sorted(set(s.load_key() for s in secs))


def test_run_plan_skips_absent_ion_with_manifest_note(tmp_path):
    secs, sels = _sections(2), _selections()
    absent = {("s1", sels[0].uid())}                        # PC 34:1 missing on section 1 only
    result, tel = _fake_run(tmp_path, secs, sels, absent=absent)
    assert result.skipped == 1
    skipped = [j for j in result.jobs if j.status == "skipped"]
    assert len(skipped) == 1 and skipped[0].section.sid == "s1"
    assert skipped[0].note                                  # carries a reason
    # the rendered cells are all the others
    assert len(tel["rendered"]) == len(secs) * len(sels) - 1


def test_run_plan_failed_source_skips_its_cells_not_the_batch(tmp_path):
    secs, sels = _sections(2), _selections()[:1]
    result, tel = _fake_run(tmp_path, secs, sels, fail_source="/data/s0.imzML")
    # s0 failed to load → its cell skipped; s1 still rendered
    statuses = {j.section.sid: j.status for j in result.jobs}
    assert statuses["s0"] == "skipped"
    assert statuses["s1"] == "done"


def test_shared_contrast_uses_one_anchor_per_ion_across_sections(tmp_path):
    secs, sels = _sections(2), _selections()[:1]
    result, tel = _fake_run(tmp_path, secs, sels, contrast=ContrastMode.SHARED)
    anchors = {r[2] for r in tel["rendered"]}               # the anchor each render received
    # both sections of the same ion get the SAME anchor = max over sections
    # (s0 brightness 1.0, s1 brightness 1.5 → shared anchor 1.5)
    assert anchors == {1.5}


def test_per_section_contrast_passes_no_anchor(tmp_path):
    secs, sels = _sections(2), _selections()[:1]
    result, tel = _fake_run(tmp_path, secs, sels, contrast=ContrastMode.PER_SECTION)
    assert all(r[2] is None for r in tel["rendered"])       # auto-contrast per section


def test_run_plan_cancel_leaves_valid_partial_tree(tmp_path):
    secs, sels = _sections(3), _selections()
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 3                               # bail after a few extraction ticks

    def loader(section):
        return ("ds", section.sid)

    def extract(ds, section, sel):
        return np.ones((3, 3))

    def render_image(arr, *, selection, section, anchor, path):
        with open(path, "w") as f:
            f.write("x")

    plan = compile_plan(secs, sels, [OutputKind.ION_IMAGES])
    result = run_plan(plan, loader=loader, extract=extract, render_image=render_image,
                      out_dir=str(tmp_path), cancel=cancel)
    assert result.cancelled
    assert os.path.exists(os.path.join(str(tmp_path), "manifest.csv"))   # manifest still written


# --------------------------------------------------------------------------- #
# composite (◆ class) flatten + write_outputs hook
# --------------------------------------------------------------------------- #
def test_composite_selection_renders_one_cell_with_members(tmp_path):
    secs = _sections(1)
    comp = SelectionRecord(label="Total ST", list_name="classes", kind="composite",
                           lipid_class="ST", member_mzs=(888.6, 890.6))
    seen = {}

    def loader(s):
        return "ds"

    def extract(ds, section, sel):
        seen["members"] = sel.member_mzs
        seen["kind"] = sel.kind
        return np.ones((2, 2))

    def render_image(arr, *, selection, section, anchor, path):
        with open(path, "w") as f:
            f.write("x")

    plan = compile_plan(secs, [comp], [OutputKind.ION_IMAGES])
    result = run_plan(plan, loader=loader, extract=extract, render_image=render_image,
                      out_dir=str(tmp_path))
    assert plan.job_count() == 1
    assert seen["kind"] == "composite" and seen["members"] == (888.6, 890.6)
    assert result.jobs[0].path.endswith(os.path.join("classes", "section_00", "class_ST.png"))


def test_write_outputs_hook_receives_cells_and_records_paths(tmp_path):
    secs, sels = _sections(2), _selections()[:2]
    captured = {}

    def loader(s):
        return "ds"

    def extract(ds, section, sel):
        return np.ones((2, 2))

    def render_image(arr, *, selection, section, anchor, path):
        with open(path, "w") as f:
            f.write("x")

    def write_outputs(plan, cells, anchors, out_dir):
        captured["n_cells"] = len(cells)
        p = os.path.join(out_dir, "matrix.png")
        with open(p, "w") as f:
            f.write("matrix")
        return {"matrix": p}

    plan = compile_plan(secs, sels, [OutputKind.ION_IMAGES, OutputKind.MATRIX])
    result = run_plan(plan, loader=loader, extract=extract, render_image=render_image,
                      write_outputs=write_outputs, out_dir=str(tmp_path))
    assert captured["n_cells"] == len(secs) * len(sels)
    assert result.extra.get("matrix", "").endswith("matrix.png")
    assert os.path.exists(result.extra["matrix"])


# --------------------------------------------------------------------------- #
# plan persistence round-trip (session.studio_plan)
# --------------------------------------------------------------------------- #
def test_plan_roundtrips_through_dict():
    secs, sels = _sections(2), _selections()
    plan = compile_plan(secs, sels, [OutputKind.ION_IMAGES, OutputKind.LIST_CSV],
                        design={"cmap": "inferno", "dpi": 300}, layout="section_outer",
                        contrast=ContrastMode.SHARED, ext="tiff")
    back = studio.StudioPlan.from_dict(plan.to_dict())
    assert len(back.sections) == 2 and len(back.selections) == 3
    assert back.layout == "section_outer" and back.ext == "tiff"
    assert back.contrast == ContrastMode.SHARED.value
    assert back.design["cmap"] == "inferno"
    assert back.wants(OutputKind.LIST_CSV)
    # the composite selection survives with its members
    comp = [s for s in back.selections if s.kind == "composite"][0]
    assert comp.lipid_class == "ST" and comp.member_mzs == (888.6, 890.6)
    # recompiled jobs match the original cardinality
    assert back.job_count() == plan.job_count()


# --------------------------------------------------------------------------- #
# cohort.SectionLoader — load-once-per-source / region mask / evict
# --------------------------------------------------------------------------- #
def test_section_loader_caches_per_source_and_reconstructs_region(tmp_path, monkeypatch):
    import json
    from smile_msi import cohort

    loads = {"n": 0}

    class FakeDS:
        def __init__(self, src):
            self.source = src; self.n_pixels = 9
        def to_ram(self): pass
        def prime(self): pass

    class FakeMSIDataset:
        @classmethod
        def from_imzml(cls, src, lazy=True):
            loads["n"] += 1
            return FakeDS(src)

    # SectionLoader does `from .msi import MSIDataset` lazily inside .load()
    import smile_msi.msi as msi
    monkeypatch.setattr(msi, "MSIDataset", FakeMSIDataset, raising=False)

    # a real imzML path must exist on disk (the loader guards on os.path.exists + .imzml)
    src = tmp_path / "slideA.imzML"
    src.write_text("x")
    # a session carrying a named region's pixel mask
    sess = tmp_path / "slideA.json"
    sess.write_text(json.dumps({"named_regions": [{"name": "fascicle", "mask": [0, 1, 2]}]}))

    class Ref:
        def __init__(self, region=""):
            self.source = str(src); self.session_path = str(sess); self.region = region

    sl = cohort.SectionLoader()
    a = sl.load(Ref())                       # whole slide
    b = sl.load(Ref(region="fascicle"))      # region of the SAME slide → shares the load
    assert loads["n"] == 1                   # one cube for both
    assert a[1] is None                      # whole slide → no pixel mask
    assert list(b[1]) == [0, 1, 2]           # region mask reconstructed from the session
    # non-imzml / missing source → skipped cleanly
    class Bad:
        source = "/nope/synthetic.csv"; session_path = ""; region = ""
    assert sl.load(Bad()) is None
    # evict frees the cache; a re-load reloads
    sl.release(str(src))
    sl.load(Ref())
    assert loads["n"] == 2


def test_section_loader_missing_region_mask_returns_none(tmp_path, monkeypatch):
    import json
    from smile_msi import cohort

    class FakeDS:
        def __init__(self, src): self.source = src; self.n_pixels = 4
        def to_ram(self): pass
        def prime(self): pass

    class FakeMSIDataset:
        @classmethod
        def from_imzml(cls, src, lazy=True): return FakeDS(src)

    import smile_msi.msi as msi
    monkeypatch.setattr(msi, "MSIDataset", FakeMSIDataset, raising=False)
    src = tmp_path / "s.imzML"; src.write_text("x")
    sess = tmp_path / "s.json"; sess.write_text(json.dumps({"named_regions": []}))

    class Ref:
        source = str(src); session_path = str(sess); region = "ghost"
    assert cohort.SectionLoader().load(Ref()) is None   # region whose mask is gone → skip
