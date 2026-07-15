"""Tests for the spectral-library MS/MS matching engine (smile_msi.specmatch):
.msp/.mgf parsing, the cosine / modified-cosine / spectral-entropy metrics, the
greedy peak aligner, the top-level scorer, and the rule-tier reconciler.

All fixtures are synthetic: in-memory MS2Spectrum objects and small .msp/.mgf
strings written to tmp_path. No GUI, network, or external files.
"""
import numpy as np
import pytest

from smile_msi import specmatch
from smile_msi.msms import MS2Spectrum


def _spec(precursor, mz, inten):
    return MS2Spectrum(precursor, np.asarray(mz, float), np.asarray(inten, float))


# --------------------------------------------------------------------------- #
# Metric identities
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("metric", [
    "cosine", "spectral_entropy_similarity", "modified_cosine",
])
def test_identical_spectrum_scores_one(metric):
    """Every similarity metric returns 1.0 (and aligns all 3 peaks) for a spectrum vs itself."""
    s = _spec(700.5, [100.0, 200.0, 300.0], [10.0, 5.0, 1.0])
    score, n = getattr(specmatch, metric)(s, s)
    assert score == pytest.approx(1.0, abs=1e-9)
    assert n == 3


def test_orthogonal_spectra_score_zero():
    a = _spec(500.0, [100.0, 110.0], [1.0, 1.0])
    b = _spec(500.0, [800.0, 810.0], [1.0, 1.0])
    assert specmatch.cosine(a, b)[0] == 0.0
    assert specmatch.spectral_entropy_similarity(a, b)[0] == pytest.approx(0.0, abs=1e-9)
    assert specmatch.modified_cosine(a, b)[0] == 0.0


def test_entropy_similarity_bounds_zero_one():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n_a, n_b = rng.integers(1, 8), rng.integers(1, 8)
        a = _spec(600.0, rng.uniform(100, 900, n_a), rng.uniform(0.1, 10, n_a))
        b = _spec(600.0, rng.uniform(100, 900, n_b), rng.uniform(0.1, 10, n_b))
        score, _ = specmatch.spectral_entropy_similarity(a, b)
        assert 0.0 <= score <= 1.0


# --------------------------------------------------------------------------- #
# Modified-cosine shift
# --------------------------------------------------------------------------- #
def test_modified_cosine_recovers_shifted_peak():
    # b is a's fragments shifted by +50 Da; precursors differ by the same 50 Da.
    a = _spec(700.0, [100.0, 200.0, 300.0], [5.0, 3.0, 1.0])
    b = _spec(750.0, [150.0, 250.0, 350.0], [5.0, 3.0, 1.0])
    plain, _ = specmatch.cosine(a, b)
    mod, n = specmatch.modified_cosine(a, b)
    assert plain == 0.0            # no direct peak overlap
    assert mod == pytest.approx(1.0, abs=1e-9)
    assert n == 3


def test_modified_cosine_mixes_direct_and_shifted():
    # one shared peak at 100 (direct), two more recovered only via the shift.
    a = _spec(700.0, [100.0, 200.0, 300.0], [4.0, 2.0, 1.0])
    b = _spec(740.0, [100.0, 240.0, 340.0], [4.0, 2.0, 1.0])
    mod, n = specmatch.modified_cosine(a, b)
    assert mod == pytest.approx(1.0, abs=1e-9)
    assert n == 3


# --------------------------------------------------------------------------- #
# _align correctness
# --------------------------------------------------------------------------- #
def test_align_is_one_to_one_within_tol():
    mz_a = np.array([100.0, 100.01, 200.0])
    mz_b = np.array([100.005, 200.0])
    pairs = specmatch._align(mz_a, mz_b, tol_da=0.02)
    # no b index used twice, no a index used twice
    a_idx = [i for i, _ in pairs]
    b_idx = [j for _, j in pairs]
    assert len(set(a_idx)) == len(a_idx)
    assert len(set(b_idx)) == len(b_idx)
    # 200.0 must align; the 100.* cluster gives exactly one pair to b[0]
    assert (2, 1) in pairs
    assert sum(1 for _, j in pairs if j == 0) == 1


def test_align_respects_tolerance():
    mz_a = np.array([100.0])
    mz_b = np.array([100.5])
    assert specmatch._align(mz_a, mz_b, tol_da=0.02) == []
    assert specmatch._align(mz_a, mz_b, tol_da=1.0) == [(0, 0)]


# --------------------------------------------------------------------------- #
# Entropy discrimination (Li et al. property on synthetic data)
# --------------------------------------------------------------------------- #
def test_entropy_ranks_true_ref_above_noisy_decoy():
    # Reference: a few dominant peaks. True query = ref + faint noise peaks.
    ref = _spec(600.0, [150.0, 250.0, 350.0], [10.0, 8.0, 6.0])
    query = _spec(600.0,
                  [150.0, 250.0, 350.0, 175.0, 225.0, 275.0],
                  [10.0, 8.0, 6.0, 0.3, 0.3, 0.3])
    # Decoy shares the same dominant peaks but is dominated by spread-out noise,
    # which entropy weighting penalizes.
    decoy = _spec(600.0,
                  [150.0, 250.0, 350.0, 400.0, 450.0, 500.0, 550.0],
                  [3.0, 3.0, 3.0, 9.0, 9.0, 9.0, 9.0])
    ent_true, _ = specmatch.spectral_entropy_similarity(query, ref)
    ent_decoy, _ = specmatch.spectral_entropy_similarity(query, decoy)
    assert ent_true > ent_decoy


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #
def test_parse_msp_roundtrip(tmp_path):
    msp = (
        "Name: PC 34:1 [M+H]+\n"
        "PrecursorMZ: 760.585\n"
        "Precursor_type: [M+H]+\n"
        "SMILES: CCCCCC\n"
        "Num Peaks: 3\n"
        "184.0733 999\n"
        "86.0964 120\n"
        "garbage line that is not a peak\n"
        "478.3293 55\n"
        "\n"
        "Name: PE 36:2 [M-H]-\n"
        "PrecursorMZ: 742.539\n"
        "Num Peaks: 1\n"
        "196.0380 500\n"
    )
    p = tmp_path / "lib.msp"
    p.write_text(msp)
    refs = specmatch.parse_msp(str(p))
    assert len(refs) == 2
    r0 = refs[0]
    assert r0.name == "PC 34:1 [M+H]+"
    assert r0.precursor == pytest.approx(760.585)
    assert r0.lipid_class == "PC"
    assert r0.adduct == "[M+H]+"
    assert r0.smiles == "CCCCCC"
    # malformed line skipped -> 3 valid peaks
    np.testing.assert_allclose(r0.mz, [184.0733, 86.0964, 478.3293])
    np.testing.assert_allclose(r0.intensity, [999, 120, 55])
    assert refs[1].lipid_class == "PE"


def test_parse_mgf_library_roundtrip(tmp_path):
    mgf = (
        "BEGIN IONS\n"
        "TITLE=Sulfatide d18:1/24:1\n"
        "PEPMASS=888.6234\n"
        "SMILES=OS(=O)(=O)O\n"
        "CHARGE=1-\n"
        "96.9601 1000\n"
        "not-a-peak\n"
        "241.0119 200\n"
        "END IONS\n"
    )
    p = tmp_path / "lib.mgf"
    p.write_text(mgf)
    refs = specmatch.parse_mgf_library(str(p))
    assert len(refs) == 1
    r = refs[0]
    assert r.name == "Sulfatide d18:1/24:1"
    assert r.precursor == pytest.approx(888.6234)
    assert r.smiles == "OS(=O)(=O)O"
    np.testing.assert_allclose(r.mz, [96.9601, 241.0119])
    np.testing.assert_allclose(r.intensity, [1000, 200])


def test_load_library_dispatch(tmp_path):
    msp = tmp_path / "a.msp"
    msp.write_text("Name: X\nPrecursorMZ: 100\nNum Peaks: 1\n100.0 1\n")
    mgf = tmp_path / "b.mgf"
    mgf.write_text("BEGIN IONS\nPEPMASS=100\n100.0 1\nEND IONS\n")
    assert len(specmatch.load_library(str(msp))) == 1
    assert len(specmatch.load_library(str(mgf))) == 1
    with pytest.raises(ValueError):
        specmatch.load_library(str(tmp_path / "c.txt"))


# --------------------------------------------------------------------------- #
# match_spectrum_to_library
# --------------------------------------------------------------------------- #
def _ref(name, pre, mz, inten, cls=""):
    return specmatch.RefSpectrum(
        name=name, precursor=pre, mz=np.asarray(mz, float),
        intensity=np.asarray(inten, float), lipid_class=cls,
    )


def test_match_ranks_true_reference_first():
    query = _spec(700.0, [100.0, 200.0, 300.0], [10.0, 6.0, 2.0])
    true = _ref("TRUE", 700.0, [100.0, 200.0, 300.0], [10.0, 6.0, 2.0], "PC")
    near = _ref("NEAR", 700.0, [100.0, 200.0, 999.0], [10.0, 6.0, 9.0], "PE")
    off = _ref("OFF", 720.0, [100.0, 200.0, 300.0], [10.0, 6.0, 2.0], "PS")
    lib = [near, off, true]
    hits = specmatch.match_spectrum_to_library(lib and query, lib, metric="entropy")
    assert hits[0].ref.name == "TRUE"
    # off-precursor ref filtered by require_precursor=True
    assert all(h.ref.name != "OFF" for h in hits)
    assert hits[0].n_matched == 3
    # all three metrics populated
    assert 0.0 <= hits[0].cosine <= 1.0
    assert 0.0 <= hits[0].modified_cosine <= 1.0
    assert 0.0 <= hits[0].entropy <= 1.0


def test_match_require_precursor_false_includes_off_precursor():
    query = _spec(700.0, [100.0, 200.0], [5.0, 5.0])
    off = _ref("OFF", 900.0, [100.0, 200.0], [5.0, 5.0], "PC")
    hits = specmatch.match_spectrum_to_library(
        query, [off], require_precursor=False, metric="cosine")
    assert len(hits) == 1 and hits[0].ref.name == "OFF"


def test_match_min_score_and_top_n():
    query = _spec(700.0, [100.0, 200.0, 300.0], [10.0, 6.0, 2.0])
    good = _ref("GOOD", 700.0, [100.0, 200.0, 300.0], [10.0, 6.0, 2.0])
    poor = _ref("POOR", 700.0, [100.0, 999.0], [1.0, 9.0])
    hits = specmatch.match_spectrum_to_library(
        query, [good, poor], metric="entropy", min_score=0.5, top_n=1)
    assert len(hits) == 1 and hits[0].ref.name == "GOOD"


def test_match_unknown_metric_raises():
    with pytest.raises(ValueError):
        specmatch.match_spectrum_to_library(_spec(1, [1], [1]), [], metric="nope")


# --------------------------------------------------------------------------- #
# confirm_with_library
# --------------------------------------------------------------------------- #
def test_confirm_with_library_class_agreeing_strong():
    query = _spec(700.0, [100.0, 200.0, 300.0], [10.0, 6.0, 2.0])
    ref = _ref("PC 34:1", 700.0, [100.0, 200.0, 300.0], [10.0, 6.0, 2.0], "PC")
    verdict, best = specmatch.confirm_with_library("PC", query, [ref], strong_thr=0.7)
    assert verdict == "confirmed (library)"
    assert best is not None and best.ref.name == "PC 34:1"


def test_confirm_with_library_class_disagreeing_suggests():
    query = _spec(700.0, [100.0, 200.0, 300.0], [10.0, 6.0, 2.0])
    ref = _ref("PE 36:2", 700.0, [100.0, 200.0, 300.0], [10.0, 6.0, 2.0], "PE")
    verdict, best = specmatch.confirm_with_library("PC", query, [ref], strong_thr=0.7)
    assert verdict.startswith("library: PE 36:2")
    assert best is not None


def test_confirm_with_library_weak_falls_through_to_rule():
    # weak library match -> falls through to msms.confirm_class. With no rule
    # diagnostics present, the rule verdict is "inconclusive".
    query = _spec(700.0, [123.456, 234.567], [1.0, 1.0])
    ref = _ref("PC 34:1", 700.0, [999.0], [1.0], "PC")
    verdict, _ = specmatch.confirm_with_library(
        "PC", query, [ref], strong_thr=0.7, weak_thr=0.5, mode="negative")
    assert verdict == "inconclusive"


# --------------------------------------------------------------------------- #
# Plan 23 item A — shared-alignment fast path must equal the standalone metrics
# --------------------------------------------------------------------------- #
def test_match_scores_equal_standalone_metrics():
    """The pre-aligned fast path in match_spectrum_to_library reproduces the public
    cosine / modified_cosine / entropy scores exactly (the safety net for the dedup)."""
    query = _spec(700.0, [100.0, 200.0, 300.0, 450.0], [10.0, 6.0, 2.0, 4.0])
    lib = [
        _ref("A", 700.0, [100.0, 200.0, 300.0], [10.0, 6.0, 2.0], "PC"),
        _ref("B", 700.02, [100.0, 250.0, 300.0, 450.0], [3.0, 7.0, 2.0, 5.0], "PE"),
        _ref("C", 740.0, [150.0, 240.0, 350.0], [5.0, 3.0, 1.0], "PS"),
        _ref("D", 700.0, [999.0], [1.0], "SM"),
    ]
    for metric in ("cosine", "modified_cosine", "entropy"):
        hits = specmatch.match_spectrum_to_library(
            query, lib, metric=metric, require_precursor=False,
            min_score=0.0, top_n=len(lib))
        by_name = {h.ref.name: h for h in hits}
        for ref in lib:
            h = by_name[ref.name]
            cs, _ = specmatch.cosine(query, ref)
            ms, _ = specmatch.modified_cosine(query, ref)
            es, _ = specmatch.spectral_entropy_similarity(query, ref)
            assert h.cosine == pytest.approx(cs, abs=1e-12)
            assert h.modified_cosine == pytest.approx(ms, abs=1e-12)
            assert h.entropy == pytest.approx(es, abs=1e-12)


def test_match_normalizes_query_once(monkeypatch):
    """Scoring N references must normalize the *query* once, not 3N times."""
    calls = {"n": 0}
    orig = specmatch._clean_normalize
    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    monkeypatch.setattr(specmatch, "_clean_normalize", counting)
    query = _spec(700.0, [100.0, 200.0, 300.0], [10.0, 6.0, 2.0])
    lib = [_ref(f"R{i}", 700.0, [100.0, 200.0, 300.0], [1.0, 2.0, 3.0]) for i in range(5)]
    specmatch.match_spectrum_to_library(query, lib, require_precursor=False)
    # 1 query normalize + 1 per reference = 6 (was 3 + 3*5 = 18 before the dedup).
    assert calls["n"] == 1 + len(lib)
