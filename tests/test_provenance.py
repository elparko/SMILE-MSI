"""Snapshot/format tests for the provenance & traceability record.

Pure formatting only — no GUI, no network, no real files beyond tmp_path. Every
``Provenance`` here is built with an explicit ``started``/``now`` so the output is
deterministic. We assert the module's *current* behaviour:

* :meth:`Provenance.to_json` emits the documented key set and a version field.
* :meth:`Provenance.to_markdown` renders all recorded sections, non-empty.
* :meth:`Provenance.methods_paragraph` always auto-drafts a non-empty paragraph.
* :meth:`Provenance.references_used` cites exactly the methods that ran (plus the
  always-on software stack), in report order, with the correlation/PCA split.
"""
import json

import pytest

from smile_msi import provenance


# --------------------------------------------------------------------------- #
# small in-memory fixtures
# --------------------------------------------------------------------------- #
class FakeDataset:
    """Minimal stand-in for what Provenance.set_dataset() reads off a dataset."""
    def __init__(self):
        self.n_pixels = 600
        self.width = 30
        self.height = 20
        self.mz_range = (400.123456, 900.987654)
        self.polarity = "negative"
        self.spec_mode = "centroid"
        self.source = "synthetic.imzML"


def _basic_prov():
    """A Provenance with a dataset + a couple of steps, frozen in time."""
    p = provenance.Provenance(title="Snapshot", started="2026-06-14T00:00:00+00:00")
    p.set_dataset(FakeDataset(), "negative")
    p.step("peak_picking", snr=3.0, norm="tic", now="2026-06-14T00:00:01+00:00")
    p.step("segmentation", spatial=True, auto=True, n_clusters=4,
           now="2026-06-14T00:00:02+00:00")
    return p


_TOP_KEYS = {"title", "started", "inputs", "dataset", "steps", "outputs", "environment"}


# --------------------------------------------------------------------------- #
# to_dict / to_json
# --------------------------------------------------------------------------- #
def test_to_dict_has_exact_top_level_keys():
    d = _basic_prov().to_dict()
    assert set(d) == _TOP_KEYS
    assert d["title"] == "Snapshot"
    assert d["started"] == "2026-06-14T00:00:00+00:00"


def test_to_dict_environment_carries_version_field():
    env = _basic_prov().to_dict()["environment"]
    # _versions() always seeds python, platform and a smile_msi version (falls back
    # to a literal if the package metadata is unavailable).
    assert "python" in env and "platform" in env
    assert "smile_msi" in env and env["smile_msi"]


def test_to_json_writes_readable_file_with_keys(tmp_path):
    p = _basic_prov()
    out = tmp_path / "prov.json"
    ret = p.to_json(str(out))
    assert ret == str(out)
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    # round-trips to exactly what to_dict produced
    assert set(loaded) == _TOP_KEYS
    assert loaded == p.to_dict()
    assert "smile_msi" in loaded["environment"]


def test_to_json_records_steps_in_order(tmp_path):
    p = _basic_prov()
    out = tmp_path / "prov.json"
    p.to_json(str(out))
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert [s["step"] for s in loaded["steps"]] == ["peak_picking", "segmentation"]
    assert loaded["dataset"]["pixels"] == 600
    assert loaded["dataset"]["mz_min"] == 400.1235  # rounded to 4 dp


# --------------------------------------------------------------------------- #
# to_markdown
# --------------------------------------------------------------------------- #
def test_to_markdown_renders_nonempty_with_all_sections():
    md = _basic_prov().to_markdown()
    assert md.strip()
    assert md.startswith("# Snapshot — Provenance & methods")
    for section in ("## Dataset", "## Processing steps", "## Software",
                    "## Methods (draft)", "## References"):
        assert section in md


def test_to_markdown_writes_file(tmp_path):
    out = tmp_path / "methods.md"
    p = _basic_prov()
    text = p.to_markdown(str(out))
    assert out.exists()
    assert out.read_text(encoding="utf-8") == text


def test_to_markdown_lists_numbered_steps():
    md = _basic_prov().to_markdown()
    # steps are 1-indexed bold entries with their params inline
    assert "1. **peak_picking**" in md
    assert "2. **segmentation**" in md
    assert "snr=3.0" in md


def test_to_markdown_inputs_table_when_fingerprinted(tmp_path):
    f = tmp_path / "data.imzML"
    f.write_bytes(b"x" * 64)
    p = provenance.Provenance(title="In", started="2026-06-14T00:00:00+00:00")
    p.set_input(str(f))
    md = p.to_markdown()
    assert "## Input files" in md
    assert "data.imzML" in md


def test_to_markdown_minimal_still_renders():
    """Even with no dataset/steps, the Software + Methods + References sections
    always render (the software stack is always cited)."""
    p = provenance.Provenance(title="Bare", started="2026-06-14T00:00:00+00:00")
    md = p.to_markdown()
    assert md.strip()
    assert "## Software" in md
    assert "## Methods (draft)" in md
    assert "## References" in md          # software refs are always emitted
    assert "## Dataset" not in md         # no dataset recorded
    assert "## Processing steps" not in md


# --------------------------------------------------------------------------- #
# methods_paragraph
# --------------------------------------------------------------------------- #
def test_methods_paragraph_nonempty_with_steps():
    para = _basic_prov().methods_paragraph()
    assert para.strip()
    assert "Peaks were detected" in para
    assert "segmented" in para
    assert "SMILE MSI" in para           # always closes with the software sentence


def test_methods_paragraph_nonempty_with_no_steps():
    """The auto-draft is never empty: the closing software sentence is unconditional."""
    p = provenance.Provenance(title="Empty", started="2026-06-14T00:00:00+00:00")
    para = p.methods_paragraph()
    assert para.strip()
    assert "Analyses were performed with SMILE MSI" in para


def test_methods_paragraph_covers_recorded_methods():
    """A wider pipeline drafts the matching prose for each recorded step."""
    p = provenance.Provenance(title="Wide", started="2026-06-15T00:00:00+00:00")
    p.step("preprocess", baseline={"method": "snip"}, smooth=True,
           now="2026-06-15T00:00:01+00:00")
    p.step("multigroup", test="Kruskal-Wallis", now="2026-06-15T00:00:02+00:00")
    p.step("region_membership", n_compartments=4, now="2026-06-15T00:00:03+00:00")
    p.step("classification", method="PLS-DA", now="2026-06-15T00:00:04+00:00")
    para = p.methods_paragraph()
    assert "baseline-corrected" in para
    assert "Savitzky" in para
    assert "Kruskal" in para
    assert "Venn" in para
    assert "PLS-DA" in para


def _nested_prov(model="lmm", test="linear mixed model", summary="median"):
    p = provenance.Provenance(title="Nested", started="2026-07-09T00:00:00+00:00")
    p.step("cohort_nested_comparison", model=model, test=test, subject_by="nerve",
           summary=summary, compartments=["endoneurium", "perineurium", "epineurium"],
           group_a="normal", group_b="treated", df_method="bw",
           now="2026-07-09T00:00:01+00:00")
    return p


def test_methods_paragraph_describes_the_nested_mixed_model():
    """The nested design's prose must state the replication unit, the model and the df rule —
    a reader can't otherwise tell a subject-level fit from a pixel-level one."""
    para = _nested_prov().methods_paragraph()
    assert "unit of replication" in para
    assert "one profile per nerve per compartment" in para
    assert "the median of the normalized per-pixel intensities" in para
    assert "intensity ~ group * compartment + (1 | nerve)" in para
    assert "between-within denominator degrees of freedom" in para
    assert "endoneurium, perineurium, epineurium" in para
    assert "Benjamini" in para


def test_methods_paragraph_describes_the_stratified_moderated_t():
    para = _nested_prov(model="stratified",
                        test="moderated t (empirical-Bayes, limma/Smyth 2004)",
                        summary="mean").methods_paragraph()
    assert "Each compartment was then tested separately" in para
    assert "normal-versus-treated" in para
    assert "the mean of the normalized per-pixel intensities" in para
    assert "empirical Bayes (Smyth 2004)" in para
    assert "(1 | nerve)" not in para              # no mixed model ran


def test_references_used_cites_the_nested_model_it_actually_ran():
    refs = _nested_prov().references_used()
    joined = " ".join(refs)
    assert "Laird" in joined and "Benjamini" in joined and "Bemis" in joined
    assert "Smyth" not in joined                 # the LMM path is not a moderated t
    strat = _nested_prov(model="stratified",
                         test="moderated t (empirical-Bayes, limma/Smyth 2004)").references_used()
    joined = " ".join(strat)
    assert "Smyth" in joined
    assert "Laird" not in joined


# --------------------------------------------------------------------------- #
# references_used / filtering
# --------------------------------------------------------------------------- #
def test_references_used_always_includes_software_stack():
    """Even with no analytical steps, the four always-on software citations appear."""
    p = provenance.Provenance(title="Bare", started="2026-06-14T00:00:00+00:00")
    refs = p.references_used()
    assert refs                                  # never empty
    assert any("NumPy" in r for r in refs)
    assert any("SciPy 1.0" in r for r in refs)
    assert any("Scikit-learn" in r for r in refs)
    assert any("Matplotlib" in r for r in refs)


def test_references_used_filters_to_methods_run():
    """Peak picking pulls in find_peaks (SciPy) + MAD; segmentation (k-means) pulls in
    PCA + Lloyd + silhouette + spatially-aware seg. Methods NOT run are absent."""
    p = _basic_prov()
    refs = p.references_used()
    assert any("absolute deviation" in r for r in refs)         # mad
    assert any("Least squares quantization" in r for r in refs)  # kmeans (Lloyd)
    assert any("Silhouettes" in r for r in refs)                 # silhouette
    assert any("spatially aware clustering" in r for r in refs)  # spatial_seg
    # nothing classification/annotation-specific was recorded
    assert not any("PLS-regression" in r for r in refs)
    assert not any("target-decoy" in r.lower() for r in refs)


def test_references_used_dedupes_and_orders_software_last():
    """references_used is deduped (a software tag pulled in early by a method step is
    not repeated by the always-on stack) and the software tags trail the method refs."""
    p = _basic_prov()                            # peak_picking already pulls in 'scipy'
    tags = p._used_ref_tags()
    assert len(tags) == len(set(tags))           # no duplicate tags (deduped)
    # every always-on software tag is present exactly once...
    for soft in ("numpy", "scipy", "sklearn", "matplotlib"):
        assert tags.count(soft) == 1
    # ...and 'scipy', already used by peak_picking, is NOT re-appended at the tail;
    # the surviving tail is the rest of the stack in order.
    assert tags[-3:] == ["numpy", "sklearn", "matplotlib"]
    refs = p.references_used()
    assert len(refs) == len(tags)


def test_references_used_software_tail_intact_without_method_overlap():
    """With no step that pre-uses a software tag, the full always-on stack trails
    the method refs in order."""
    p = provenance.Provenance(title="m", started="2026-06-15T00:00:00+00:00")
    p.step("multigroup", test="Kruskal-Wallis", now="2026-06-15T00:00:01+00:00")
    tags = p._used_ref_tags()
    assert tags[-4:] == ["numpy", "scipy", "sklearn", "matplotlib"]


def test_references_used_correlation_segmentation_skips_pca():
    """The correlation-distance segmentation path clusters spectral shape directly and
    must NOT cite PCA, while the default (PCA) path does."""
    pca = provenance.Provenance(title="pca", started="2026-06-15T00:00:00+00:00")
    pca.step("segmentation", method="ward", metric="euclidean", n_clusters=5,
             now="2026-06-15T00:00:01+00:00")
    corr = provenance.Provenance(title="corr", started="2026-06-15T00:00:00+00:00")
    corr.step("segmentation", method="ward", metric="correlation", n_clusters=5,
              now="2026-06-15T00:00:01+00:00")
    pca_refs = pca.references_used()
    corr_refs = corr.references_used()
    assert any("closest fit" in r for r in pca_refs)        # Pearson PCA cited
    assert not any("closest fit" in r for r in corr_refs)   # correlation path omits PCA
    # both still cite the agglomerative/Ward stack
    assert any("Hierarchical grouping" in r for r in corr_refs)


def test_references_used_matches_module_catalog():
    """Every returned citation string is a verbatim entry from REFERENCES."""
    refs = _basic_prov().references_used()
    catalog = set(provenance.REFERENCES.values())
    assert all(r in catalog for r in refs)


def test_markdown_references_match_references_used():
    """The Markdown References section is numbered straight from references_used()."""
    p = _basic_prov()
    md = p.to_markdown()
    refs = p.references_used()
    body = md.split("## References", 1)[1]
    for i, r in enumerate(refs, 1):
        assert f"{i}. {r}" in body


def test_miamsie_checklist_autofills_and_flags_blanks():
    from smile_msi import demo, provenance
    ds = demo.make_synthetic(width=12, height=10, seed=3)
    prov = provenance.Provenance(title="t")
    prov.set_dataset(ds, "negative")
    prov.step("find spatial features", snr=3.0, min_rel_intensity=0.005)
    chk = prov.miamsie_checklist()
    R = provenance.Provenance.REQUIRED
    # auto-filled from the dataset
    assert chk["Instrument & acquisition"]["Polarity"] == "negative"
    assert "Data & preprocessing" in chk and chk["Data & preprocessing"]["Spectrum type"]
    assert any("find spatial features" in s
               for s in chk["Data & preprocessing"]["Processing steps & parameters"])
    # acquisition/prep the app can't observe are flagged for the user
    assert chk["Sample & preparation"]["Matrix & application method"] == R
    assert chk["Instrument & acquisition"]["Instrument / mass analyzer"] == R
    # markdown render marks the blanks
    md = prov.miamsie_markdown()
    assert "MIAMSIE" in md and "⚠️" in md


# --------------------------------------------------------------------------- #
# audit-trail additions: ROI descriptors, from_dict restore, CSV header
# --------------------------------------------------------------------------- #
def test_region_descriptor_and_fingerprint_are_stable():
    import numpy as np
    m = np.zeros(50, bool)
    m[5:20] = True
    d = provenance.region_descriptor("nerve", created_via="drawn/derived ROI", mask=m,
                                     role="A", bbox=[1, 2, 8, 9])
    assert d["name"] == "nerve" and d["role"] == "A" and d["via"] == "drawn/derived ROI"
    assert d["n_pixels"] == 15 and d["bbox"] == [1, 2, 8, 9]
    # the mask fingerprint is content-derived and stable, and changes when the mask changes
    assert d["mask_sha1"] == provenance.mask_fingerprint(m)
    m2 = m.copy()
    m2[0] = True
    assert provenance.mask_fingerprint(m2) != d["mask_sha1"]


def test_region_descriptor_falls_back_to_n_pixels_without_mask():
    d = provenance.region_descriptor("cluster group", n_pixels=320, role="B")
    assert d["n_pixels"] == 320 and "mask_sha1" not in d and d["role"] == "B"


def test_format_settings_and_regions_render_compact():
    s = provenance.format_settings({"norm": "tic", "tol_ppm": 10.0, "max_q": 0.05,
                                    "regions": "ignored", "blank": None})
    assert s == "norm=tic tol_ppm=10 max_q=0.05"          # regions + blank dropped, float trimmed
    line = provenance.format_regions([
        provenance.region_descriptor("nerve", n_pixels=3204, role="A", created_via="drawn ROI"),
        provenance.region_descriptor("bg", n_pixels=5991, role="B", created_via="clusters 2+5")])
    assert "A=nerve(3,204px, drawn ROI)" in line and "B=bg(5,991px, clusters 2+5)" in line


def test_from_dict_round_trips_full_record():
    p = provenance.Provenance(title="Round", started="2026-06-19T00:00:00+00:00")
    p.step("roi_comparison", test="mwu", norm="tic", tol_ppm=10,
           now="2026-06-19T00:00:01+00:00")
    p.output("out.csv", "stats")
    back = provenance.Provenance.from_dict(p.to_dict())
    assert back.to_dict() == p.to_dict()                  # exact round-trip
    assert [s["step"] for s in back.steps] == ["roi_comparison"]


def test_from_dict_tolerates_missing_keys():
    back = provenance.Provenance.from_dict({"title": "Bare"})
    assert back.title == "Bare" and back.steps == [] and back.environment  # env reseeded


def test_csv_header_lines_carry_settings_regions_software():
    import numpy as np
    p = provenance.Provenance(title="H", started="2026-06-19T00:00:00+00:00")
    p.set_dataset(_FakeDS(), "negative")
    descs = [provenance.region_descriptor("nerve", mask=np.ones(10, bool), role="A")]
    lines = p.csv_header_lines(analysis="ROI comparison", settings={"norm": "tic", "tol_ppm": 10},
                               regions=descs, now="2026-06-19T00:00:00+00:00")
    assert all(ln.startswith("# ") for ln in lines)       # every line is a CSV comment
    text = "\n".join(lines)
    assert "analysis: ROI comparison" in text
    assert "settings: norm=tic tol_ppm=10" in text
    assert "ROIs: A=nerve" in text
    assert "software:" in text and "smile_msi" in text
    assert "negative" in text                              # dataset polarity rendered


class _FakeDS:
    n_pixels = 600
    width = 30
    height = 20
    mz_range = (400.1, 900.9)
    polarity = "negative"
    spec_mode = "centroid"
    source = "synthetic.imzML"


def test_session_persists_and_restores_provenance(tmp_path):
    """The audit trail survives a session save/load round-trip, so reopening a
    sample keeps its methods record instead of rebuilding it empty."""
    from smile_msi import session
    p = provenance.Provenance(title="S", started="2026-06-19T00:00:00+00:00")
    p.step("spatial_feature_finding", snr=3.0, min_morans=0.05,
           now="2026-06-19T00:00:01+00:00")
    sess = session.build_session(source="x.imzML", settings={}, peaks=[], provenance=p)
    assert sess["version"] == session.VERSION
    assert sess["provenance"]["steps"][0]["step"] == "spatial_feature_finding"
    path = tmp_path / "s.json"
    session.save_session(str(path), sess)
    loaded = session.load_session(str(path))
    restored = provenance.Provenance.from_dict(loaded["provenance"])
    assert [s["step"] for s in restored.steps] == ["spatial_feature_finding"]


def test_load_session_defaults_provenance_for_old_files(tmp_path):
    """A pre-v3 session with no provenance key loads with provenance=None (not a KeyError)."""
    from smile_msi import session
    path = tmp_path / "old.json"
    path.write_text('{"version": 2, "source": "x", "peaks": []}', encoding="utf-8")
    loaded = session.load_session(str(path))
    assert loaded["provenance"] is None


# --------------------------------------------------------------------------- #
# _provenance_sections — shared extraction for the bundle methods file + PDF book
# (audit plan 23, item D: dedup of the duplicated provenance walk)
# --------------------------------------------------------------------------- #
def _prov_with_input(tmp_path):
    p = provenance.Provenance(title="t", started="2026-06-19T00:00:00+00:00")
    f = tmp_path / "demo.imzML"
    f.write_bytes(b"hello world")
    p.set_input(str(f))
    p.step("annotation", match_ppm=10, now="2026-06-19T00:00:01+00:00")
    return p


def test_provenance_sections_order_and_content(tmp_path):
    from smile_msi.export import _provenance_sections
    p = _prov_with_input(tmp_path)
    sects = dict(_provenance_sections(p))
    heads = [h for h, _ in _provenance_sections(p)]
    # Input files and Processing steps always present here; Software/References
    # only if populated — assert relative order of those that appear.
    assert heads == [h for h in ("Input files", "Processing steps", "Software",
                                 "References") if h in heads]
    assert "Input files" in sects and "Processing steps" in sects
    assert any("annotation" in ln and "match_ppm=10" in ln
               for ln in sects["Processing steps"])


def test_provenance_sections_include_method_toggle(tmp_path):
    from smile_msi.export import _provenance_sections
    p = _prov_with_input(tmp_path)
    method = p.inputs[0].get("method", "")
    without = dict(_provenance_sections(p, include_method=False))["Input files"][0]
    withm = dict(_provenance_sections(p, include_method=True))["Input files"][0]
    # Both carry the explicit "sha256" checksum label; only the book variant
    # appends the fingerprint method before the closing paren.
    assert without.endswith("…)")
    assert withm != without
    if method:
        assert withm.endswith(f"{method})")


# --------------------------------------------------------------------------- #
# Session load-shim: legacy fold_change → signed log2_fc (audit plan 18, Issue A)
# --------------------------------------------------------------------------- #
def test_load_session_migrates_fold_change_to_log2(tmp_path):
    from smile_msi import session
    legacy = {
        "version": 4, "source": "x.imzML", "peaks": [],
        "report_items": [
            {"type": "analysis", "analysis_kind": "roi_comparison",
             "records": [{"mz": 700.1, "fold_change": 2.0},     # → log2 = 1.0
                         {"mz": 800.2, "fold_change": 0.5},     # → log2 = -1.0
                         {"mz": 650.0, "fold_change": 0.0}]},   # → 0.0 (guarded)
            {"type": "note", "text": "no records here"},
        ],
    }
    path = tmp_path / "legacy.json"
    path.write_text(__import__("json").dumps(legacy), encoding="utf-8")
    loaded = session.load_session(str(path))
    recs = loaded["report_items"][0]["records"]
    assert all("fold_change" not in r for r in recs)
    assert recs[0]["log2_fc"] == 1.0
    assert recs[1]["log2_fc"] == -1.0
    assert recs[2]["log2_fc"] == 0.0
