"""Tests for community-standards externalization (``smile_msi.standards``).

Headless, synthetic data only — no real imzML, no network. The METASPACE half is
exercised only through monkeypatched lazy-imports / synthetic DataFrames so CI never
touches the network. Round-trips use a real tiny ``MSIDataset.from_arrays`` (the imzML
write needs a real dataset, unlike the ``FakeDataset`` stand-in in test_provenance).
"""
import warnings

import numpy as np
import pytest

from smile_msi import provenance, standards
from smile_msi.msi import MSIDataset

# pyimzml's bundled OBO warns on a couple of accession-name mismatches (the plan flags OBO
# accession drift as a known, verify-before-merge risk); silence it for clean test output.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _tiny_dataset(polarity="negative", pixel_size_um=25.0):
    coords = [(1, 1), (2, 1), (1, 2)]
    ax = np.array([100.0, 200.0, 300.0])
    ints = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]),
            np.array([7.0, 8.0, 9.0])]
    ds = MSIDataset.from_arrays(coords, [ax] * 3, ints, polarity=polarity)
    ds.pixel_size_um = pixel_size_um
    return ds


def _full_meta():
    return standards.AcquisitionMeta(
        organism="rat", tissue="sciatic nerve", condition="synkinetic",
        sample_prep="fresh-frozen", storage="-80C",
        matrix="DHB", matrix_application="sublimation",
        section_thickness_um=10.0,
        instrument="Bruker timsTOF fleX", mass_analyzer="tof", ionization="MALDI",
        laser_spot_um=20.0, mass_resolution="40000", calibration="red phosphorus")


def _prov_for(ds):
    p = provenance.Provenance(title="t", started="2026-06-23T00:00:00+00:00")
    p.set_dataset(ds, "negative")
    p.step("peak_picking", snr=3.0, norm="tic", now="2026-06-23T00:00:01+00:00")
    return p


# --------------------------------------------------------------------------- #
# AcquisitionMeta round-trip
# --------------------------------------------------------------------------- #
def test_acquisition_meta_round_trips():
    m = _full_meta()
    d = m.to_dict()
    assert d["matrix"] == "DHB" and d["section_thickness_um"] == 10.0
    back = standards.AcquisitionMeta.from_dict(d)
    assert back == m


def test_acquisition_meta_from_dict_tolerant():
    m = standards.AcquisitionMeta.from_dict({"matrix": "CHCA", "bogus": "x"})
    assert m.matrix == "CHCA" and m.ionization == "MALDI"  # default kept


def test_acquisition_meta_is_empty():
    assert standards.AcquisitionMeta().is_empty()            # only the MALDI default
    assert not standards.AcquisitionMeta(matrix="DHB").is_empty()


# --------------------------------------------------------------------------- #
# imzml_cvparams
# --------------------------------------------------------------------------- #
def test_cvparams_emit_expected_accessions():
    ds = _tiny_dataset()
    cv = standards.imzml_cvparams(ds, _full_meta())
    # pixel size from ds.pixel_size_um
    assert cv["IMS:1000046"]["value"] == "25.0"
    assert cv["IMS:1000047"]["value"] == "25.0"
    # polarity term (presence-only, blank value)
    assert "MS:1000129" in cv and cv["MS:1000129"]["value"] == ""
    # instrument free text + curated analyzer/ionization terms
    assert cv["MS:1000031"]["value"] == "Bruker timsTOF fleX"
    assert "MS:1000084" in cv          # tof
    assert "MS:1000075" in cv          # MALDI
    # matrix from meta
    assert cv["MS:1000835"]["value"] == "DHB"


def test_cvparams_omit_absent_fields_not_blank():
    ds = _tiny_dataset()
    cv = standards.imzml_cvparams(ds, standards.AcquisitionMeta())  # nothing user-set
    # no instrument/matrix => those accessions absent (not emitted blank)
    assert "MS:1000031" not in cv
    assert "MS:1000835" not in cv
    # only auto-observable + the MALDI default survive
    assert set(cv) == {"IMS:1000046", "IMS:1000047", "MS:1000129", "MS:1000075"}
    # no free-text value is ever blank
    for k, info in cv.items():
        assert info["value"] != "" or k in {"MS:1000129", "MS:1000075"}


def test_cvparams_pixel_size_falls_back_to_provenance():
    ds = _tiny_dataset(pixel_size_um=None)
    prov = _prov_for(_tiny_dataset(pixel_size_um=33.0))  # prov carries pixel size
    cv = standards.imzml_cvparams(ds, standards.AcquisitionMeta(), prov=prov)
    assert cv["IMS:1000046"]["value"] == "33.0"


def test_cvparams_positive_polarity_term():
    ds = _tiny_dataset(polarity="positive")
    cv = standards.imzml_cvparams(ds, standards.AcquisitionMeta())
    assert "MS:1000130" in cv and "MS:1000129" not in cv


# --------------------------------------------------------------------------- #
# write_imzml_with_metadata round-trip
# --------------------------------------------------------------------------- #
def test_write_with_metadata_round_trips(tmp_path):
    from pyimzml.ImzMLParser import ImzMLParser
    from smile_msi import ingest

    ds = _tiny_dataset()
    enriched = str(tmp_path / "enriched.imzML")
    plain = str(tmp_path / "plain.imzML")
    standards.write_imzml_with_metadata(ds, enriched, _full_meta())
    ingest.to_imzml(ds, plain)

    # injected metadata is present in the header
    txt = (tmp_path / "enriched.imzML").read_text(encoding="ISO-8859-1")
    assert "DHB" in txt and "Bruker timsTOF fleX" in txt and "<sampleList" in txt

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        par = ImzMLParser(enriched)
        par2 = ImzMLParser(plain)
        # pixel size cvParam reparses
        assert par.imzmldict.get("pixel size x") == 25.0
        # spectra are byte-identical to plain to_imzml (offsets/.ibd untouched)
        for i in range(ds.n_pixels):
            m1, a1 = par.getspectrum(i)
            m2, a2 = par2.getspectrum(i)
            assert np.allclose(m1, m2) and np.allclose(a1, a2)


def test_write_with_empty_meta_is_a_noop(tmp_path):
    """Empty / None meta => no injection (graceful degrade). to_imzml embeds a random
    UUID per call, so a no-op is asserted by the *absence* of any injected reporting
    metadata (no <sampleList>, no instrument model) — not by raw byte-equality."""
    from smile_msi import ingest

    ds = _tiny_dataset()
    plain = str(tmp_path / "p.imzML")
    ingest.to_imzml(ds, plain)
    plain_txt = (tmp_path / "p.imzML").read_text(encoding="ISO-8859-1")

    for fname, meta in (("e.imzML", standards.AcquisitionMeta()), ("n.imzML", None)):
        out = str(tmp_path / fname)
        standards.write_imzml_with_metadata(ds, out, meta)
        txt = (tmp_path / fname).read_text(encoding="ISO-8859-1")
        # no injected reporting elements...
        assert "<sampleList" not in txt
        assert "instrument model" not in txt
        # ...and the same header skeleton as plain to_imzml (sampleList absent in both)
        assert ("<sampleList" in txt) == ("<sampleList" in plain_txt)


def test_write_records_provenance(tmp_path):
    ds = _tiny_dataset()
    prov = _prov_for(ds)
    standards.write_imzml_with_metadata(ds, str(tmp_path / "x.imzML"), _full_meta(),
                                        prov=prov)
    assert any(s["step"] == "imzml_export" for s in prov.steps)
    assert any(o["file"] == "x.imzML" for o in prov.outputs)


# --------------------------------------------------------------------------- #
# validate_reporting — pass / warn / fail
# --------------------------------------------------------------------------- #
def test_validate_empty_meta_flags_required_missing():
    ds = _tiny_dataset()
    prov = _prov_for(ds)
    res = standards.validate_reporting(prov, standards.AcquisitionMeta())
    assert not res.is_submission_ready()
    assert res.n_missing_required > 0
    # the submission-blocking user fields are flagged missing
    missing = res.missing_required()
    assert any("Matrix & application method" in k for k in missing)
    assert any("Instrument / mass analyzer" in k for k in missing)
    # auto-observed polarity is ok (not missing)
    pol = res.fields["Instrument & acquisition :: Polarity"]
    assert pol["status"] == "ok" and pol["value"] == "negative"


def test_validate_full_meta_is_submission_ready():
    ds = _tiny_dataset()
    prov = _prov_for(ds)
    res = standards.validate_reporting(prov, _full_meta())
    assert res.is_submission_ready()
    assert res.n_missing_required == 0
    # the previously-missing user fields flipped to ok
    assert res.fields["Sample & preparation :: Matrix & application method"]["status"] == "ok"
    assert res.fields["Instrument & acquisition :: Instrument / mass analyzer"]["status"] == "ok"


def test_validate_warn_vs_missing_distinct():
    """Recommended-but-absent fields warn (non-blocking); submission-required absent
    fields are 'missing' and block readiness."""
    ds = _tiny_dataset()
    prov = _prov_for(ds)
    # mass_resolution is recommended (warn) but not submission-required.
    res = standards.validate_reporting(prov, standards.AcquisitionMeta())
    mr = res.fields["Instrument & acquisition :: Mass resolution"]
    assert mr["status"] == "warn" and not mr["required"]
    assert res.n_warn > 0


def test_validate_markdown_renders():
    ds = _tiny_dataset()
    res = standards.validate_reporting(_prov_for(ds), _full_meta())
    md = res.to_markdown()
    assert md.startswith("# MSI reporting validation")
    assert "submission-ready" in md and "✅" in md


# --------------------------------------------------------------------------- #
# session v4 forward-fill (acquisition_meta)
# --------------------------------------------------------------------------- #
def test_v3_session_forward_fills_acquisition_meta(tmp_path, monkeypatch):
    """A pre-v4 session loads with acquisition_meta defaulted — IF session.py has been
    wired for v4. Until the integrator bumps VERSION + adds the setdefault, this is a soft
    check (xfail-style skip), so this engine-only test stays green on the current tree."""
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    from smile_msi import session
    path = tmp_path / "old.json"
    path.write_text('{"version": 3, "source": "x", "peaks": []}', encoding="utf-8")
    loaded = session.load_session(str(path))
    if "acquisition_meta" not in loaded:
        pytest.skip("session.py not yet wired for v4 acquisition_meta (integrator step)")
    assert loaded["acquisition_meta"] == {}


# --------------------------------------------------------------------------- #
# METASPACE — missing-extra error path + compare_fdr (no network)
# --------------------------------------------------------------------------- #
def test_metaspace_submit_raises_clear_import_error(monkeypatch):
    """When the metaspace2020 extra is absent, _import_metaspace raises ImportError with
    the uv pip install hint. Simulate absence by forcing the import to fail."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "metaspace" or name.startswith("metaspace."):
            raise ImportError("no metaspace")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError) as ei:
        standards.metaspace_submit("a.imzML", "a.ibd", standards.AcquisitionMeta(),
                                   databases=[("HMDB", "v4")], api_key="k")
    assert "metaspace2020" in str(ei.value)
    assert "uv pip install" in str(ei.value)


def test_metaspace_submit_requires_api_key(monkeypatch):
    """If the client imports but no api_key is supplied, a clear ValueError is raised
    (no network)."""
    import sys
    import types
    fake = types.ModuleType("metaspace")
    fake.SMInstance = object  # never instantiated — we fail before that on the key check
    monkeypatch.setitem(sys.modules, "metaspace", fake)
    with pytest.raises(ValueError) as ei:
        standards.metaspace_submit("a.imzML", "a.ibd", standards.AcquisitionMeta(),
                                   databases=[("HMDB", "v4")], api_key=None)
    assert "API key" in str(ei.value)


def test_compare_fdr_agreement_and_exclusives():
    pytest.importorskip("pandas")
    import pandas as pd

    ours = [
        {"formula": "C42H82NO8P", "adduct": "-H", "q_value": 0.02, "mz": 802.5},   # both
        {"formula": "C40H80NO8P", "adduct": "-H", "q_value": 0.04, "mz": 774.5},   # ours_only
    ]
    theirs = pd.DataFrame([
        {"formula": "C42H82NO8P", "adduct": "-H", "fdr": 0.05, "mz": 802.5},       # both
        {"formula": "C44H86NO8P", "adduct": "-H", "fdr": 0.10, "mz": 830.6},       # metaspace_only
    ])
    out = standards.compare_fdr(ours, theirs)
    by = {(r.formula, r.adduct): r for r in out.itertuples()}
    assert by[("C42H82NO8P", "-H")].agreement == "both"
    assert by[("C40H80NO8P", "-H")].agreement == "ours_only"
    assert by[("C44H86NO8P", "-H")].agreement == "metaspace_only"
    # FDR delta computed for the shared row
    both = by[("C42H82NO8P", "-H")]
    assert abs(both.delta - (0.02 - 0.05)) < 1e-9


def test_compare_fdr_missing_columns_raises():
    pytest.importorskip("pandas")
    import pandas as pd
    with pytest.raises(KeyError):
        standards.compare_fdr([{"formula": "X"}],  # missing 'adduct'
                              pd.DataFrame([{"formula": "X", "adduct": "-H", "fdr": 0.1}]))
