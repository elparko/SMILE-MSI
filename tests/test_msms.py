"""Tests for the MS/MS confirmation engine: MGF parsing, diagnostic-fragment
matching, and the confirmed/unsupported/inconclusive verdict logic.

All fixtures are synthetic in-memory MGF strings written to tmp_path, plus a few
hand-built ``MS2Spectrum`` objects. No GUI, network, or external files.
"""
import numpy as np
import pytest

from smile_msi import msms
from smile_msi.masses import formula_mass, ELECTRON, ELEMENTS


def _carboxylate(c, d):
    """[FA-H]- carboxylate m/z for a C:D fatty acyl — mirrors msms.acyl_chains."""
    fa = formula_mass({"C": c, "H": 2 * c - 2 * d, "O": 2})
    return fa - ELEMENTS["H"] + ELECTRON


# --------------------------------------------------------------------------- #
# parse_mgf
# --------------------------------------------------------------------------- #
def test_parse_single_block(tmp_path):
    p = tmp_path / "one.mgf"
    p.write_text("BEGIN IONS\nPEPMASS=888.62\n96.9601 9000\n255.2330 2000\nEND IONS\n")
    specs = msms.parse_mgf(str(p))
    assert len(specs) == 1
    sp = specs[0]
    assert isinstance(sp, msms.MS2Spectrum)
    assert sp.precursor == 888.62
    assert isinstance(sp.mz, np.ndarray) and isinstance(sp.intensity, np.ndarray)
    np.testing.assert_allclose(sp.mz, [96.9601, 255.2330])
    np.testing.assert_allclose(sp.intensity, [9000.0, 2000.0])


def test_parse_multiple_blocks(tmp_path):
    p = tmp_path / "multi.mgf"
    p.write_text(
        "BEGIN IONS\nPEPMASS=100.0\n50.0 1\nEND IONS\n"
        "BEGIN IONS\nPEPMASS=200.0\n60.0 2\n70.0 3\nEND IONS\n"
    )
    specs = msms.parse_mgf(str(p))
    assert len(specs) == 2
    assert specs[0].precursor == 100.0 and specs[0].mz.size == 1
    assert specs[1].precursor == 200.0 and specs[1].mz.size == 2
    np.testing.assert_allclose(specs[1].mz, [60.0, 70.0])


def test_parse_pepmass_with_intensity_field(tmp_path):
    """PEPMASS may carry a precursor-intensity second token; only the m/z is used."""
    p = tmp_path / "pi.mgf"
    p.write_text("BEGIN IONS\nPEPMASS=703.5746 12345.6\n184.0733 9000\nEND IONS\n")
    sp = msms.parse_mgf(str(p))[0]
    assert sp.precursor == 703.5746


def test_parse_missing_pepmass_defaults_zero(tmp_path):
    """A block with no PEPMASS gets precursor 0.0 (the ``pre or 0.0`` fallback)."""
    p = tmp_path / "nopm.mgf"
    p.write_text("BEGIN IONS\n184.0733 9000\n264.2686 3000\nEND IONS\n")
    sp = msms.parse_mgf(str(p))[0]
    assert sp.precursor == 0.0
    assert sp.mz.size == 2


def test_parse_peak_without_intensity_defaults_one(tmp_path):
    """A peak line with only an m/z gets intensity 1.0."""
    p = tmp_path / "noint.mgf"
    p.write_text("BEGIN IONS\nPEPMASS=500.0\n241.0119\nEND IONS\n")
    sp = msms.parse_mgf(str(p))[0]
    np.testing.assert_allclose(sp.mz, [241.0119])
    np.testing.assert_allclose(sp.intensity, [1.0])


def test_parse_tolerates_malformed_lines(tmp_path):
    """Non-numeric junk, header metadata, and blank lines are skipped; the real
    peaks survive. This is the malformed-line tolerance case."""
    p = tmp_path / "messy.mgf"
    p.write_text(
        "BEGIN IONS\n"
        "TITLE=scan=42 some description\n"   # key=value metadata, not PEPMASS
        "CHARGE=1-\n"                          # another metadata key
        "PEPMASS=888.62\n"
        "\n"                                   # blank line
        "96.9601 9000\n"                       # valid peak
        "garbage line that is not a peak\n"    # non-numeric -> ValueError -> skip
        "   \n"                                # whitespace-only
        "255.2330 2000\n"                      # valid peak
        "NaNlike notanumber here\n"            # leading token non-numeric -> skip
        "END IONS\n"
    )
    specs = msms.parse_mgf(str(p))
    assert len(specs) == 1
    sp = specs[0]
    assert sp.precursor == 888.62
    # only the two genuine peaks remain
    np.testing.assert_allclose(sp.mz, [96.9601, 255.2330])
    np.testing.assert_allclose(sp.intensity, [9000.0, 2000.0])


def test_parse_empty_block(tmp_path):
    """A BEGIN/END pair with no peaks yields an empty-array spectrum, precursor 0."""
    p = tmp_path / "empty.mgf"
    p.write_text("BEGIN IONS\nEND IONS\n")
    sp = msms.parse_mgf(str(p))[0]
    assert sp.precursor == 0.0
    assert sp.mz.size == 0 and sp.intensity.size == 0


# --------------------------------------------------------------------------- #
# matched_classes
# --------------------------------------------------------------------------- #
def test_matched_classes_sulfatide_fragment():
    sp = msms.MS2Spectrum(888.62, np.array([96.9601, 281.2486]), np.array([9e3, 4e3]))
    found = msms.matched_classes(sp, "negative")
    assert "Sulfatide" in found


def test_matched_classes_ganglioside_neu5ac():
    """The shared 290.0881 sialic-acid B-ion matches every ganglioside class."""
    sp = msms.MS2Spectrum(772.0, np.array([290.0881, 264.2686]), np.array([9e3, 3e3]))
    found = msms.matched_classes(sp, "negative")
    assert {"GM3", "GM2", "GM1", "GD3", "GD1", "GT1"} <= found


def test_matched_classes_neutral_loss_ps():
    """PS is matched via a serine neutral loss (precursor - 87.0320), not a fragment."""
    pre = 788.5447
    sp = msms.MS2Spectrum(pre, np.array([pre - 87.0320]), np.array([5e3]))
    found = msms.matched_classes(sp, "negative")
    assert "PS" in found


def test_matched_classes_respects_mode():
    """A positive-mode fragment (184.0733 phosphocholine) only matches in positive
    mode; the same spectrum yields nothing diagnostic in negative mode."""
    sp = msms.MS2Spectrum(703.57, np.array([184.0733, 264.2686]), np.array([9e3, 3e3]))
    assert "PC" in msms.matched_classes(sp, "positive")
    assert "SM" in msms.matched_classes(sp, "positive")
    assert msms.matched_classes(sp, "negative") == set()


def test_matched_classes_unknown_mode_is_empty():
    sp = msms.MS2Spectrum(500.0, np.array([184.0733, 96.9601]), np.array([1.0, 1.0]))
    assert msms.matched_classes(sp, "sideways") == set()


def test_matched_classes_tolerance_window():
    """An ion just outside tol_da is not matched; widening tol_da matches it."""
    sp = msms.MS2Spectrum(800.0, np.array([96.9601 + 0.02]), np.array([1.0]))
    assert "Sulfatide" not in msms.matched_classes(sp, "negative", tol_da=0.01)
    assert "Sulfatide" in msms.matched_classes(sp, "negative", tol_da=0.05)


def test_matched_classes_empty_spectrum():
    sp = msms.MS2Spectrum(700.0, np.array([]), np.array([]))
    assert msms.matched_classes(sp, "negative") == set()


# --------------------------------------------------------------------------- #
# confirm_class verdict logic
# --------------------------------------------------------------------------- #
def test_confirm_class_confirmed():
    sp = msms.MS2Spectrum(888.62, np.array([96.9601]), np.array([9e3]))
    assert msms.confirm_class("Sulfatide", sp, "negative") == "confirmed"


def test_confirm_class_unsupported():
    """A spectrum whose only diagnostic is PI's 241.0119 should mark a competing
    rule-bearing class (PE) as 'unsupported' — other-class evidence is present but
    this class's diagnostic is absent."""
    sp = msms.MS2Spectrum(885.55, np.array([241.0119]), np.array([5e3]))
    found = msms.matched_classes(sp, "negative")
    assert "PI" in found and "PE" not in found
    assert msms.confirm_class("PE", sp, "negative") == "unsupported"
    # and the matched class itself is confirmed
    assert msms.confirm_class("PI", sp, "negative") == "confirmed"


def test_confirm_class_inconclusive_no_diagnostics():
    """No diagnostic ion at all -> inconclusive regardless of the queried class."""
    sp = msms.MS2Spectrum(700.0, np.array([100.0, 200.0]), np.array([1.0, 1.0]))
    assert msms.confirm_class("PI", sp, "negative") == "inconclusive"


def test_confirm_class_inconclusive_class_without_rule():
    """Other-class diagnostics are present but the queried class has no rule in this
    mode -> inconclusive (not 'unsupported', because has_rule is False)."""
    sp = msms.MS2Spectrum(888.62, np.array([96.9601]), np.array([9e3]))  # Sulfatide hit
    found = msms.matched_classes(sp, "negative")
    assert found  # there IS other-class evidence
    # 'PC' is a positive-mode-only class -> no negative-mode rule
    assert "PC" not in msms.DIAGNOSTIC_FRAGMENTS["negative"]
    assert "PC" not in msms.DIAGNOSTIC_LOSSES["negative"]
    assert msms.confirm_class("PC", sp, "negative") == "inconclusive"


def test_confirm_class_full_workflow_from_mgf(tmp_path):
    """End-to-end: parse a synthetic sulfatide MGF then confirm the class."""
    p = tmp_path / "sulf.mgf"
    p.write_text(
        "BEGIN IONS\nPEPMASS=888.62\n96.9601 9000\n281.2486 4000\n255.2330 2000\nEND IONS\n"
    )
    sp = msms.parse_mgf(str(p))[0]
    assert sp.precursor == 888.62 and sp.mz.size == 3
    assert msms.confirm_class("Sulfatide", sp, "negative") == "confirmed"
    # PI has a negative-mode rule but no PI diagnostic here -> unsupported
    assert msms.confirm_class("PI", sp, "negative") == "unsupported"


# --------------------------------------------------------------------------- #
# classify (ranking of an unknown spectrum)
# --------------------------------------------------------------------------- #
def test_classify_ranks_pi_top():
    """Three PI diagnostic ions outscore the single shared 152.9958 of PG/PA."""
    sp = msms.MS2Spectrum(885.55, np.array([241.0119, 223.0013, 152.9958]),
                          np.array([5e3, 3e3, 2e3]))
    ranked = msms.classify(sp, "negative")
    assert ranked and ranked[0][0] == "PI"
    score = dict(ranked)
    assert score["PI"] == 3
    # PG and PA share only the 152.9958 ion -> score 1 each, ranked below PI
    assert score.get("PG") == 1 and score.get("PA") == 1
    assert score["PI"] > score["PG"]


def test_classify_empty_when_nothing_diagnostic():
    sp = msms.MS2Spectrum(700.0, np.array([111.0, 222.0]), np.array([1.0, 1.0]))
    assert msms.classify(sp, "negative") == []


def test_classify_counts_neutral_loss():
    """A neutral loss contributes to the score (HexCer hexose loss in negative)."""
    pre = 888.62
    sp = msms.MS2Spectrum(pre, np.array([pre - 162.0528]), np.array([4e3]))
    ranked = dict(msms.classify(sp, "negative"))
    assert ranked.get("HexCer") == 1


# --------------------------------------------------------------------------- #
# acyl_chains
# --------------------------------------------------------------------------- #
def test_carboxylate_helper_matches_literature():
    """Ground the ``_carboxylate`` helper (and, transitively, the production formula it
    mirrors) against LIPID MAPS [FA-H]- masses, so the detection tests below are anchored
    to an independent ground truth rather than re-deriving production's own arithmetic."""
    assert _carboxylate(16, 0) == pytest.approx(255.2330, abs=1e-3)   # palmitate
    assert _carboxylate(18, 1) == pytest.approx(281.2486, abs=1e-3)   # oleate


def test_acyl_chains_reads_carboxylates():
    """16:0 (255.2330) and 18:1 (281.2486) carboxylates are detected. Fed as literal
    literature m/z (not ``_carboxylate(...)``) so a bug shared between the test helper and
    ``acyl_chains`` cannot hide -- the production matcher must hit the real masses."""
    sp = msms.MS2Spectrum(
        888.62,
        np.array([96.9601, 281.2486, 255.2330]),
        np.array([9e3, 4e3, 2e3]),
    )
    chains = msms.acyl_chains(sp)
    assert set(chains) >= {"16:0", "18:1"}


def test_acyl_chains_ranked_by_intensity():
    """Most intense carboxylate first: 18:1 (4000) before 16:0 (2000)."""
    sp = msms.MS2Spectrum(
        888.62,
        np.array([_carboxylate(18, 1), _carboxylate(16, 0)]),
        np.array([4000.0, 2000.0]),
    )
    assert msms.acyl_chains(sp)[0] == "18:1"
    assert msms.acyl_chains(sp, max_chains=1) == ["18:1"]


def test_acyl_chains_none_present():
    sp = msms.MS2Spectrum(700.0, np.array([96.9601, 100.0]), np.array([1.0, 1.0]))
    assert msms.acyl_chains(sp) == []
