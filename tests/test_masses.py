"""Verify class templates against literature neutral monoisotopic masses,
and adduct ion m/z against known values. Tolerance 1 mDa."""
import math

import pytest

from smile_msi.masses import (formula_mass, ion_mz, ppm_error, ADDUCTS, adduct_charge,
                              ELEMENTS, ELECTRON)
from smile_msi.lipiddb import build_database

TOL = 1e-3

# (name, reference neutral monoisotopic mass) from LIPID MAPS
REF_NEUTRAL = {
    "PA 34:1": 674.4887,
    "PC 34:1": 759.5778,
    "PE 34:1": 717.5309,
    "PS 36:1": 789.5519,
    "PG 34:1": 748.5254,
    "PI 34:1": 836.5415,
    "SM 34:1": 702.5676,    # d18:1/16:0
    "Cer 34:1;O2": 537.5121,  # d18:1/16:0
    "HexCer 34:1;O2": 699.5650,
    "Sulfatide 34:1;O2": 779.5217,  # HexCer 34:1;O2 + SO3 (+79.9568)
    "FA 18:1": 282.2559,
}


def _by_name():
    return {lip.name: lip for lip in build_database()}


def test_class_neutral_masses():
    db = _by_name()
    for name, ref in REF_NEUTRAL.items():
        assert name in db, f"{name} not generated"
        got = db[name].neutral_mass
        assert math.isclose(got, ref, abs_tol=TOL), f"{name}: {got:.4f} != {ref:.4f}"


def test_negative_adducts():
    # PE 34:1 [M-H]- and PC 34:1 [M-CH3]- and [M+HCOO]-
    pe = formula_mass({"C": 39, "H": 76, "N": 1, "O": 8, "P": 1})
    assert math.isclose(ion_mz(pe, "[M-H]-"), 716.5236, abs_tol=TOL)
    pc = 759.5778
    assert math.isclose(ion_mz(pc, "[M-CH3]-"), 744.5549, abs_tol=TOL)
    assert math.isclose(ion_mz(pc, "[M+HCOO]-"), 804.5760, abs_tol=TOL)


def test_chloride_and_formate_deltas():
    assert math.isclose(ADDUCTS["[M+Cl]-"], 34.9694, abs_tol=TOL)
    assert math.isclose(ADDUCTS["[M+HCOO]-"], 44.9982, abs_tol=TOL)


def test_multiply_charged_ion_mz():
    """m/z of [M-nH]^n- = (M - n*proton)/n; singly-charged is unchanged."""
    proton = ELEMENTS["H"] - ELECTRON
    M = 1000.0
    assert adduct_charge("[M-2H]2-") == 2 and adduct_charge("[M-3H]3-") == 3
    assert math.isclose(ion_mz(M, "[M-2H]2-"), (M - 2 * proton) / 2, abs_tol=TOL)
    assert math.isclose(ion_mz(M, "[M-3H]3-"), (M - 3 * proton) / 3, abs_tol=TOL)
    assert math.isclose(ion_mz(M, "[M+2H]2+"), (M + 2 * proton) / 2, abs_tol=TOL)
    # regression: singly-charged m/z must not change
    assert math.isclose(ion_mz(M, "[M-H]-"), M - proton, abs_tol=TOL)


def test_ammonium_adduct():
    """[M+NH4]+ delta + a known neutral-lipid m/z (TG 52:2 → 876.80)."""
    assert math.isclose(ADDUCTS["[M+NH4]+"], 18.0338, abs_tol=TOL)
    assert math.isclose(ion_mz(858.7677, "[M+NH4]+"), 876.8015, abs_tol=2e-3)


def test_neutral_glycerolipids_present():
    db = _by_name()
    for name, ref in [("MG 18:1", 356.2927), ("DG 36:2", 620.5380), ("TG 52:2", 858.7677)]:
        assert name in db, f"{name} not generated"
        assert abs(db[name].neutral_mass - ref) / ref * 1e6 < 5


def test_oxfa_is_its_own_class():
    classes = {l.lipid_class for l in build_database()}
    assert "OxFA" in classes and "FA" in classes        # oxylipins split out from plain FAs


def test_even_chain_and_db_cap_filter():
    """Default DB drops odd total-carbon and >6-total-double-bond glycerolipids / free FAs
    (rare mass-coincidence hits that a small calibration offset lets win), keeps common nerve
    species, and does NOT filter sphingolipids (myelin has genuine odd-chain / 2-OH species)."""
    names = {l.name for l in build_database()}
    assert "PC 35:2" not in names                        # odd total carbons: gone
    assert "FA 17:0" not in names
    def total_db(n):
        return int(n.split()[1].split(":")[1].split(";")[0])
    assert not any(n.startswith(("PC ", "PE ", "PS ", "PI ")) and total_db(n) > 6
                   for n in names)                        # >6 total double bonds: gone
    for n in ("PE 40:6", "PC 34:1", "FA 24:1", "PS 40:6"):
        assert n in names, f"{n} should be kept"
    # sphingolipids/gangliosides exempt — odd-chain and 2-OH species retained
    assert any(n.startswith("Cer 33:") for n in names)    # odd-chain ceramide
    assert any(n.startswith("Sulfatide 4") and ";O3" in n for n in names)   # 2-OH sulfatide


def test_even_chain_filter_can_be_disabled():
    """even_chain_only=False restores odd-carbon ester/acyl species for tissues where they
    matter (the filter is a default, not a hard-coded restriction)."""
    names = {l.name for l in build_database(even_chain_only=False)}
    assert "PC 35:2" in names


def test_neutral_lipids_positive_only_and_priors():
    """TG/DG/MG are enumerated in positive mode but never in negative (they don't
    deprotonate); the positive adduct priors favour their usual adducts."""
    from smile_msi.match import Annotator
    db = build_database()
    neg = {lip.lipid_class for _, _, lip in Annotator(mode="negative", db=db)._index}
    pos = {lip.lipid_class for _, _, lip in Annotator(mode="positive", db=db)._index}
    assert {"TG", "DG", "MG"}.isdisjoint(neg)
    assert {"TG", "DG", "MG"} <= pos

    ann = Annotator(mode="positive", db=db)
    assert ann._adduct_prior("TG", "[M+NH4]+") > 0      # neutral lipid → ammonium
    assert ann._adduct_prior("PC", "[M+H]+") > 0        # choline headgroup → protonated
    assert ann._adduct_prior("TG", "[M-H]-") == 0       # no positive prior for a neg adduct

    # the TG ammonium ion is an actual candidate at its m/z
    by = {l.name: l for l in db}
    mz = ion_mz(by["TG 52:2"].neutral_mass, "[M+NH4]+")
    assert any(c.lipid.name == "TG 52:2" and c.adduct == "[M+NH4]+" for c in ann.annotate_mz(mz))


def test_gangliosides_matchable_multicharge():
    """Gangliosides are reachable at their multiply-charged m/z; plain lipids
    don't spawn chemically implausible multiply-charged phantoms."""
    from smile_msi.match import Annotator
    db = build_database()
    by = {lip.name: lip for lip in db}
    ann = Annotator(mode="negative", ppm_tol=10.0, db=db)

    # a di-sialo ganglioside should match at its [M-3H]3- m/z (out of z=1 range)
    gd1 = next(n for n in by if n.startswith("GD1 "))
    mz3 = ion_mz(by[gd1].neutral_mass, "[M-3H]3-")
    assert any(c.lipid.name == gd1 and c.adduct == "[M-3H]3-" for c in ann.annotate_mz(mz3))

    # mono-sialo GM3 (one carboxyl) is reachable as 2- but NOT as 3-
    gm3 = next(n for n in by if n.startswith("GM3 "))
    mz2 = ion_mz(by[gm3].neutral_mass, "[M-2H]2-")
    assert any(c.lipid.name == gm3 and c.adduct == "[M-2H]2-" for c in ann.annotate_mz(mz2))
    mz3_gm3 = ion_mz(by[gm3].neutral_mass, "[M-3H]3-")
    assert not any(c.adduct == "[M-3H]3-" for c in ann.annotate_mz(mz3_gm3))

    # a plain fatty acid must not produce a multiply-charged candidate
    fa = next(lip for lip in db if lip.lipid_class == "FA")
    cands = ann.annotate_mz(ion_mz(fa.neutral_mass, "[M-2H]2-"))
    assert all(adduct_charge(c.adduct) == 1 for c in cands)


# ----- isotope patterns ---------------------------------------------------- #
def test_isotope_distribution_carbon_rule():
    """M+1/M0 tracks ~1.08% per carbon; cholesterol C27 -> ~0.30."""
    from smile_msi.masses import isotope_distribution
    dist = isotope_distribution({"C": 27, "H": 46, "O": 1}, max_offset=2)
    m0, m1 = dist[0][1], dist[1][1]
    assert abs(m1 / m0 - 0.30) < 0.04
    # abundances sum to ~1 across the kept offsets
    assert abs(sum(p for _, p in dist) - 1.0) < 0.01


def test_ion_isotope_pattern_spacing_and_anchor():
    """Monoisotopic m/z equals ion_mz; isotopologue spacing ~1.00336/charge."""
    from smile_msi.masses import ion_isotope_pattern
    f = {"C": 42, "H": 82, "N": 1, "O": 8, "P": 1}
    p1 = ion_isotope_pattern(f, "[M-H]-", 3)
    assert abs(p1[0][0] - ion_mz(formula_mass(f), "[M-H]-")) < 1e-6
    assert abs((p1[1][0] - p1[0][0]) - 1.00336) < 2e-3
    p2 = ion_isotope_pattern(f, "[M-2H]2-", 3)
    assert abs((p2[1][0] - p2[0][0]) - 0.50168) < 2e-3


def test_chloride_adduct_boosts_m2():
    """[M+Cl]- gains a big M+2 from 37Cl (~24%); [M-H]- does not."""
    from smile_msi.masses import ion_isotope_pattern
    f = {"C": 40, "H": 80, "N": 1, "O": 8, "P": 1}
    cl = ion_isotope_pattern(f, "[M+Cl]-", 3)
    h = ion_isotope_pattern(f, "[M-H]-", 3)
    assert cl[2][1] > 0.30 and cl[2][1] > 2 * h[2][1]


# ----- external database import -------------------------------------------- #
def test_load_external_db(tmp_path):
    """LIPID MAPS / SwissLipids-style CSV import: parse formulas, skip unsupported
    elements, and feed the in-silico Annotator."""
    import pandas as pd
    from smile_msi.lipiddb import load_external_db, parse_formula
    from smile_msi.match import Annotator

    assert parse_formula("C42H82NO8P") == {"C": 42, "H": 82, "N": 1, "O": 8, "P": 1}
    rows = [{"ABBREVIATION": "PC 34:1", "FORMULA": "C42H82NO8P", "CATEGORY": "GP"},
            {"ABBREVIATION": "PE 36:2", "FORMULA": "C41H76NO8P", "CATEGORY": "GP"},
            {"ABBREVIATION": "FluoroLipid", "FORMULA": "C20H30F2O2", "CATEGORY": "X"}]
    p = tmp_path / "lm.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    db = load_external_db(str(p))
    names = [l.name for l in db]
    assert "PC 34:1" in names and "FluoroLipid" not in names    # F unsupported -> skipped
    assert all(l.formula for l in db)                            # formulas carried for isotopes
    ann = Annotator("negative", 10.0, db=db)
    hit = ann.annotate_mz(ion_mz(db[0].neutral_mass, "[M-H]-"))
    assert hit and hit[0].lipid.name == "PC 34:1"


def test_merge_external_db_adds_new_formulas_keeps_builtin(tmp_path):
    """Merge adds only the external species whose composition isn't already in the built-in
    DB (keeping the curated metabolites + tissue priors), deduping by formula."""
    import pandas as pd
    from smile_msi.lipiddb import build_database, merge_external_db

    base = build_database()
    base_formulas = {tuple(sorted(l.formula.items())) for l in base}
    # one lipid whose formula the built-in already covers (PE 34:1) + one genuinely new class
    # (cardiolipin, not enumerated) + one unsupported-element row that must be dropped.
    cl_formula = "C81H156O17P2"                          # a cardiolipin — absent from the in-silico DB
    rows = [{"ABBREVIATION": "PE 34:1", "FORMULA": "C39H76NO8P", "CATEGORY": "GP"},
            {"ABBREVIATION": "CL 72:8", "FORMULA": cl_formula, "CATEGORY": "GP"},
            {"ABBREVIATION": "FluoroLipid", "FORMULA": "C20H30F2O2", "CATEGORY": "X"}]
    p = tmp_path / "ext.csv"
    pd.DataFrame(rows).to_csv(p, index=False)

    merged, n_ext, n_added = merge_external_db(p, base=base)
    assert n_ext == 2                                    # fluoro row dropped by load_external_db
    assert n_added == 1                                  # only the cardiolipin is a new formula
    assert len(merged) == len(base) + 1
    names = {l.name for l in merged}
    assert "CL 72:8" in names                            # new class added
    assert "N-acetylaspartate (NAA)" in names           # built-in curated metabolite kept
    # the built-in PE 34:1 (with its tissue prior) is kept, not a duplicated external copy
    pe = [l for l in merged if l.lipid_class == "PE" and l.name == "PE 34:1"]
    assert len(pe) == 1 and pe[0].tissue_prior > 0.5     # built-in prior, not the flat 0.5 external default


_MINI_SDF = """
  Marvin

M  END
> <ABBREVIATION>
PC 34:1

> <FORMULA>
C42H82NO8P

> <CATEGORY>
Glycerophospholipids

$$$$

  Marvin

M  END
> <ABBREVIATION>
CL 72:8

> <FORMULA>
C81H156O17P2

> <CATEGORY>
Glycerophospholipids

$$$$

  Marvin

M  END
> <ABBREVIATION>
FluoroStd

> <FORMULA>
C20H30F2O2

> <CATEGORY>
Fatty Acyls

$$$$
"""


def test_load_sdf_direct_and_filters(tmp_path):
    """LMSD .sdf → Lipid list directly (the wizard's one-step path): drop unsupported-element
    rows, honour category + mass filters, and merge onto the built-in by new formula."""
    from smile_msi.lipiddb import build_database, load_sdf, merge_lipids

    sdf = tmp_path / "mini.sdf"
    sdf.write_text(_MINI_SDF, encoding="utf-8")

    lipids = load_sdf(str(sdf))
    names = {l.name for l in lipids}
    assert {"PC 34:1", "CL 72:8"} <= names and "FluoroStd" not in names   # F dropped
    # mass filter: PC 34:1 ~759 Da, CL 72:8 ~1466 Da
    assert {l.name for l in load_sdf(str(sdf), min_mass=1000.0)} == {"CL 72:8"}
    # category filter (case-insensitive substring)
    assert {l.name for l in load_sdf(str(sdf), categories=["glycerophospho"])} == {"PC 34:1", "CL 72:8"}

    merged, n_ext, n_added = merge_lipids(build_database(), lipids)
    assert n_ext == 2                                     # PC + CL usable (fluoro dropped)
    assert n_added == 1                                   # PC 34:1 formula already built-in; CL is new
    assert any(l.name == "CL 72:8" for l in merged)


def test_ppm_error_sign_and_zero_guard():
    """Signed ppm and a degenerate-input guard (theo == 0 -> nan, not ZeroDivisionError)."""
    # observed heavier than theoretical -> positive ppm; 1 ppm at m/z 1000 = 1 mDa
    assert ppm_error(1000.001, 1000.0) == pytest.approx(1.0, abs=1e-6)
    assert ppm_error(999.999, 1000.0) == pytest.approx(-1.0, abs=1e-6)
    # a non-positive theoretical (never a real ion m/z) returns nan instead of raising
    assert math.isnan(ppm_error(100.0, 0.0))


def test_formula_mass_unsupported_element_raises_clear_error():
    """An element outside the supported table raises a clear ValueError naming it,
    not a bare KeyError (callers like load_external_db pre-filter such rows)."""
    with pytest.raises(ValueError, match="unsupported element"):
        formula_mass({"C": 20, "H": 30, "F": 2, "O": 2})   # fluorine unsupported
