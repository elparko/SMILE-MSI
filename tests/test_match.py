"""Unit tests for :mod:`smile_msi.match` — the ppm-window Annotator that ranks
in-silico lipid candidates for an observed m/z.

Ranking score (higher = better):  ``-|ppm| + tissue_prior + adduct_prior``
with the priors deliberately small so they only break near-ppm ties.

All fixtures are tiny in-memory ``Lipid`` lists built so each candidate's adduct
ion m/z lands on a hand-chosen target; expected ranking orders are hand-computed
from the scoring formula. No GUI / network / files.
"""
import math

import pytest

from smile_msi.lipiddb import Lipid, build_database
from smile_msi.masses import (
    ADDUCTS,
    ADDUCT_CHARGE,
    NEG_ADDUCTS,
    POS_ADDUCTS,
    ion_mz,
)
from smile_msi.match import (
    _MULTICHARGE_ADDUCT_CLASSES,
    _POSITIVE_ONLY_CLASSES,
    PRIOR_WINDOW_FRAC,
    Annotator,
    Candidate,
)

TOL = 1e-9


def _lipid_for_ion(name, cls, ion_target, adduct, prior):
    """Build a Lipid whose `adduct` ion m/z equals `ion_target` exactly.

    ion_mz(M, add) = (M + ADDUCTS[add]) / |z|  =>  M = ion_target*|z| - ADDUCTS[add].
    """
    z = ADDUCT_CHARGE[adduct]
    neutral = ion_target * z - ADDUCTS[adduct]
    lip = Lipid(name, cls, neutral, prior, formula={})
    # sanity: the ion really lands where we asked
    assert math.isclose(ion_mz(neutral, adduct), ion_target, abs_tol=1e-6)
    return lip


# --------------------------------------------------------------------------- #
# init / index construction
# --------------------------------------------------------------------------- #
def test_init_defaults_and_adduct_set_by_mode():
    db = [_lipid_for_ion("PE 34:1", "PE", 700.0, "[M-H]-", 0.9)]
    neg = Annotator(db=db)
    assert neg.mode == "negative"
    assert neg.ppm_tol == 5.0
    assert neg.adducts == NEG_ADDUCTS
    assert neg.lipids is db

    pos = Annotator(mode="positive", ppm_tol=10.0, db=db)
    assert pos.adducts == POS_ADDUCTS
    assert pos.ppm_tol == 10.0


def test_index_is_sorted_by_mz():
    db = [
        _lipid_for_ion("FA a", "FA", 800.0, "[M-H]-", 0.5),
        _lipid_for_ion("FA b", "FA", 700.0, "[M-H]-", 0.5),
        _lipid_for_ion("FA c", "FA", 900.0, "[M-H]-", 0.5),
    ]
    ann = Annotator(mode="negative", db=db)
    assert ann._mzs == sorted(ann._mzs)
    assert ann._mzs == [t[0] for t in ann._index]


# --------------------------------------------------------------------------- #
# ranking: mass accuracy dominates, priors break near-ppm ties
# --------------------------------------------------------------------------- #
def test_annotate_ranking_mass_accuracy_then_priors():
    """Observed 700.0 at ppm_tol=5 -> prior weight scale = 5*0.1 = 0.5:
      PE 34:1 [M-H]-  exact (ppm 0), scale*(0.9+0.15) -> 0.525  (rank 1)
      PG      [M-H]-  exact (ppm 0), scale*(0.4+0.15) -> 0.275  (rank 2)
      FA      [M-H]-  +2 ppm off,    -2 + scale*(0.5+0.15)      (rank 3)
    """
    obs = 700.0
    tol = 5.0
    scale = tol * PRIOR_WINDOW_FRAC
    db = [
        _lipid_for_ion("PE 34:1", "PE", obs, "[M-H]-", 0.9),
        _lipid_for_ion("FA test", "FA", obs + obs * 2e-6, "[M-H]-", 0.5),
        _lipid_for_ion("PG test", "PG", obs, "[M-H]-", 0.4),
    ]
    ann = Annotator(mode="negative", ppm_tol=tol, db=db)
    res = ann.annotate_mz(obs)
    assert isinstance(res[0], Candidate)
    assert [c.lipid.name for c in res] == ["PE 34:1", "PG test", "FA test"]
    # exact scores (priors are tolerance-scaled)
    assert math.isclose(res[0].score, scale * (0.9 + 0.15), abs_tol=TOL)        # PE, ppm 0
    assert math.isclose(res[1].score, scale * (0.4 + 0.15), abs_tol=TOL)        # PG, ppm 0
    # FA observed is 2 ppm *below* its theoretical ion -> signed ppm = -2
    assert math.isclose(res[2].ppm, -2.0, abs_tol=1e-3)
    assert math.isclose(res[2].score, -2.0 + scale * (0.5 + 0.15), abs_tol=1e-3)
    # every Candidate echoes the queried mz
    assert all(c.mz == obs for c in res)


def test_mass_accuracy_overrides_a_big_prior():
    """A closer mass with no priors must beat a far mass with a big prior:
    Sulfatide (prior 1.0 + adduct 0.15) at +4 ppm = -2.85; FA exact (0.5+0.15) = 0.65.
    """
    obs = 750.0
    db = [
        _lipid_for_ion("Sulfatide 34:1;O2", "Sulfatide", obs + obs * 4e-6, "[M-H]-", 1.0),
        _lipid_for_ion("FA exact", "FA", obs, "[M-H]-", 0.5),
    ]
    ann = Annotator(mode="negative", ppm_tol=10.0, db=db)
    res = ann.annotate_mz(obs)
    assert res[0].lipid.name == "FA exact"
    assert res[1].lipid.name == "Sulfatide 34:1;O2"


def test_negative_adduct_prior_tiebreak():
    """Two candidates, identical neutral mass / tissue prior / adduct [M-H]-: the one whose
    class is in the [M-H]- adduct-prior table (PE) beats the one that isn't (PC) by exactly
    the (tolerance-scaled) adduct-prior bonus. [M-H]- is enumerated for every class, so the
    bonus — not gating — decides the order. (Cl/formate/acetate are now class-gated so their
    enumerated set equals their prior set — the exotic-adduct gate is covered by
    :func:`test_exotic_adducts_gated_to_chemical_classes`.)"""
    obs = 720.0
    tol = 5.0
    bonus = tol * PRIOR_WINDOW_FRAC * 0.15
    db = [
        _lipid_for_ion("favored", "PE", obs, "[M-H]-", 0.6),
        _lipid_for_ion("plain", "PC", obs, "[M-H]-", 0.6),
    ]
    ann = Annotator(mode="negative", ppm_tol=tol, db=db)
    res = ann.annotate_mz(obs)
    assert res[0].lipid.name == "favored"
    assert math.isclose(res[0].score - res[1].score, bonus, abs_tol=TOL)


def test_positive_adduct_prior_tiebreak():
    """Positive mode: PC is in the [M+Na]+ prior set; LPC is not -> PC wins by 0.15."""
    obs = 800.0
    db = [
        _lipid_for_ion("PC 36:1", "PC", obs, "[M+Na]+", 0.6),
        _lipid_for_ion("LPC 18:1", "LPC", obs, "[M+Na]+", 0.6),
    ]
    tol = 5.0
    ann = Annotator(mode="positive", ppm_tol=tol, db=db)
    res = ann.annotate_mz(obs)
    assert res[0].lipid.name == "PC 36:1"
    assert math.isclose(res[0].score - res[1].score,
                        tol * PRIOR_WINDOW_FRAC * 0.15, abs_tol=TOL)


# --------------------------------------------------------------------------- #
# ppm window
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ppm_tol", [1.0, 5.0, 20.0])
def test_ppm_window_inclusion(ppm_tol):
    """A candidate 3 ppm away is returned iff the tolerance is >= 3 ppm."""
    obs = 600.0
    off = _lipid_for_ion("off3ppm", "FA", obs + obs * 3e-6, "[M-H]-", 0.5)
    ann = Annotator(mode="negative", ppm_tol=ppm_tol, db=[off])
    res = ann.annotate_mz(obs)
    if ppm_tol >= 3.0:
        assert [c.lipid.name for c in res] == ["off3ppm"]
        assert abs(res[0].ppm) <= ppm_tol
    else:
        assert res == []


# --------------------------------------------------------------------------- #
# multicharge adduct filtering
# --------------------------------------------------------------------------- #
def _adducts_for(ann, name_prefix):
    return sorted({a for (_m, a, lip) in ann._index if lip.name.startswith(name_prefix)})


def test_multicharge_only_for_ganglioside_classes():
    """[M-2H]2- enumerated for GM1 (mono-sialo ganglioside) but never for a PE.
    [M-3H]3- only for di-/tri-sialo (GD/GT), so GM1 has 2- but NOT 3-, while GD1 has both."""
    gm1 = Lipid("GM1 36:1", "GM1", 1500.0, 0.95, formula={})
    gd1 = Lipid("GD1 36:1", "GD1", 1800.0, 0.95, formula={})
    pe = Lipid("PE 40:1", "PE", 900.0, 0.9, formula={})
    ann = Annotator(mode="negative", ppm_tol=5.0, db=[gm1, gd1, pe])

    gm1_adducts = _adducts_for(ann, "GM1")
    assert "[M-2H]2-" in gm1_adducts
    assert "[M-3H]3-" not in gm1_adducts          # mono-sialo: only reaches 2-

    gd1_adducts = _adducts_for(ann, "GD1")
    assert "[M-2H]2-" in gd1_adducts and "[M-3H]3-" in gd1_adducts

    pe_adducts = _adducts_for(ann, "PE")
    assert "[M-2H]2-" not in pe_adducts and "[M-3H]3-" not in pe_adducts
    # PE still gets all singly-charged negative adducts
    assert "[M-H]-" in pe_adducts


def test_multicharge_adduct_prior_applies():
    """A GM1 [M-2H]2- ion carries the +0.15 adduct prior (class is in the 2- prior set)."""
    # neutral mass chosen so its [M-2H]2- ion lands at 700.0
    gm1 = _lipid_for_ion("GM1 36:1", "GM1", 700.0, "[M-2H]2-", 0.95)
    tol = 5.0
    ann = Annotator(mode="negative", ppm_tol=tol, db=[gm1])
    res = ann.annotate_mz(700.0)
    assert len(res) == 1 and res[0].adduct == "[M-2H]2-"
    assert math.isclose(res[0].score, tol * PRIOR_WINDOW_FRAC * (0.95 + 0.15), abs_tol=TOL)   # ppm 0


def test_no_positive_doubly_charged_species():
    """[M+2H]2+ maps to an empty class set, so it is never enumerated in positive mode."""
    db = [
        Lipid("PC 34:1", "PC", 759.5, 0.8, formula={}),
        Lipid("PE 34:1", "PE", 717.5, 0.9, formula={}),
    ]
    ann = Annotator(mode="positive", ppm_tol=5.0, db=db)
    assert not any(a == "[M+2H]2+" for (_m, a, _l) in ann._index)
    assert _MULTICHARGE_ADDUCT_CLASSES["[M+2H]2+"] == set()


# --------------------------------------------------------------------------- #
# positive-only class masking
# --------------------------------------------------------------------------- #
def test_neutral_glycerolipids_masked_in_negative_mode():
    """TG/DG/MG (neutral glycerolipids) are skipped entirely in negative mode but
    present in positive mode."""
    assert _POSITIVE_ONLY_CLASSES == {"TG", "DG", "MG"}
    db = [
        Lipid("TG 52:2", "TG", 858.0, 0.3, formula={}),
        Lipid("DG 36:2", "DG", 620.5, 0.25, formula={}),
        Lipid("MG 18:1", "MG", 356.3, 0.2, formula={}),
        Lipid("PE 34:1", "PE", 717.5, 0.9, formula={}),
    ]
    neg = Annotator(mode="negative", ppm_tol=5.0, db=db)
    assert {lip.lipid_class for (_m, _a, lip) in neg._index} == {"PE"}

    pos = Annotator(mode="positive", ppm_tol=5.0, db=db)
    assert {"TG", "DG", "MG", "PE"} <= {lip.lipid_class for (_m, _a, lip) in pos._index}


# --------------------------------------------------------------------------- #
# edge cases: no database, no match, isobaric tie
# --------------------------------------------------------------------------- #
def test_build_database_returns_fresh_mutable_list():
    """build_database hands back a fresh list each call (copied from the shared immutable
    cache), so sorting/appending one caller's copy can't corrupt another's. (Regression:
    the lru_cache previously returned the same mutable list to every caller.)"""
    a = build_database()
    b = build_database()
    assert a is not b                       # distinct list objects
    assert len(a) == len(b) and a == b      # but identical contents
    n = len(a)
    a.append("sentinel")                    # mutate one copy
    assert "sentinel" not in build_database()    # others are unaffected
    assert len(build_database()) == n


def test_empty_database():
    ann = Annotator(mode="negative", ppm_tol=5.0, db=[])
    assert ann._index == [] and ann._mzs == []
    assert ann.annotate_mz(700.0) == []


def test_no_match_returns_empty():
    db = [_lipid_for_ion("PE 34:1", "PE", 700.0, "[M-H]-", 0.9)]
    ann = Annotator(mode="negative", ppm_tol=5.0, db=db)
    assert ann.annotate_mz(900.0) == []           # far from the only ion


@pytest.mark.parametrize("bad", [float("nan"), 0.0, -700.0, float("inf")])
def test_nonfinite_or_nonpositive_mz_returns_empty(bad):
    """A NaN/inf/zero/negative observed m/z must return no candidates — not the whole DB.
    (Regression: NaN made the bisect window span the entire index, leaking an arbitrary
    class label into class_of and polluting class-level rollups.)"""
    db = [
        _lipid_for_ion("PE 34:1", "PE", 700.0, "[M-H]-", 0.9),
        _lipid_for_ion("FA 18:1", "FA", 281.0, "[M-H]-", 0.5),
    ]
    ann = Annotator(mode="negative", ppm_tol=5.0, db=db)
    assert ann.annotate_mz(bad) == []
    assert ann.class_of(bad) == ""                 # and no spurious class label


def test_cross_class_tie_break_is_deterministic():
    """An exact isobaric tie across two classes resolves the same way regardless of DB
    insertion order (deterministic secondary sort), so class_of is reproducible."""
    obs = 600.0
    pe = _lipid_for_ion("PE x", "PE", obs, "[M-H]-", 0.6)
    ps = _lipid_for_ion("PS y", "PS", obs, "[M-H]-", 0.6)
    a = Annotator(mode="negative", ppm_tol=5.0, db=[pe, ps])
    b = Annotator(mode="negative", ppm_tol=5.0, db=[ps, pe])   # reversed insertion order
    assert a.class_of(obs) == b.class_of(obs)
    assert [c.lipid.name for c in a.annotate_mz(obs)] == \
           [c.lipid.name for c in b.annotate_mz(obs)]


def test_demethylation_adduct_only_enumerated_for_choline_classes():
    """[M-CH3]- (headgroup demethylation) is enumerated for choline lipids (PC) but NOT
    for PE/FA etc. — the ion is chemically impossible there and must not be generated."""
    pc = _lipid_for_ion("PC 34:1", "PC", 780.0, "[M-CH3]-", 0.8)
    pe = _lipid_for_ion("PE 34:1", "PE", 700.0, "[M-CH3]-", 0.9)
    ann = Annotator(mode="negative", ppm_tol=5.0, db=[pc, pe])
    adducts_by_class = {}
    for _m, add, lip in ann._index:
        adducts_by_class.setdefault(lip.lipid_class, set()).add(add)
    assert "[M-CH3]-" in adducts_by_class["PC"]
    assert "[M-CH3]-" not in adducts_by_class.get("PE", set())


def test_exotic_adducts_gated_to_chemical_classes():
    """Cl / formate / acetate adducts are enumerated only for choline (PC/SM/PC-O/LPC) and
    neutral ceramide (HexCer/Cer) classes — never for acidic-headgroup lipids (PE/PS/FA/
    Sulfatide), which ionize as [M-H]-. This kills the exotic-adduct false hits (e.g. an
    ether-PE formate winning over the true sulfatide)."""
    db = [
        _lipid_for_ion("PC 34:1", "PC", 780.0, "[M+Cl]-", 0.8),
        _lipid_for_ion("PE 34:1", "PE", 760.0, "[M-H]-", 0.9),
        _lipid_for_ion("PS 40:6", "PS", 800.0, "[M-H]-", 0.85),
        _lipid_for_ion("Sulfatide 42:2;O2", "Sulfatide", 888.0, "[M-H]-", 1.0),
        _lipid_for_ion("HexCer 34:1;O2", "HexCer", 700.0, "[M+HCOO]-", 0.95),
    ]
    ann = Annotator(mode="negative", ppm_tol=5.0, db=db)
    by_class = {}
    for _m, add, lip in ann._index:
        by_class.setdefault(lip.lipid_class, set()).add(add)
    exotic = {"[M+Cl]-", "[M+HCOO]-", "[M+CH3COO]-"}
    assert exotic & by_class.get("PC", set())            # choline: kept
    assert exotic & by_class.get("HexCer", set())        # neutral ceramide: kept
    assert not (exotic & by_class.get("PE", set()))      # acidic headgroup: gated out
    assert not (exotic & by_class.get("PS", set()))
    assert not (exotic & by_class.get("Sulfatide", set()))


def test_base_ion_preferred_on_exact_isobar_tie():
    """On an *exact* score tie between a base [M-H]- ion and an adduct form of another lipid,
    the [M-H]- sorts first (prefer the simpler assignment); the scores stay equal. Without
    the base-ion tie-break the old class/name order would put PC (an adduct) ahead of PE."""
    obs = 800.0
    tol = 5.0
    # PE [M-H]- (base) and PC [M+Cl]- (adduct), same tissue prior 0.6 and same +0.15 adduct
    # prior (both classes are in their adduct's prior set) → identical score.
    base = _lipid_for_ion("PE 38:4", "PE", obs, "[M-H]-", 0.6)
    add = _lipid_for_ion("PC 36:1", "PC", obs, "[M+Cl]-", 0.6)
    ann = Annotator(mode="negative", ppm_tol=tol, db=[add, base])   # adduct listed first
    res = ann.annotate_mz(obs)
    assert len(res) == 2
    assert math.isclose(res[0].score, res[1].score, abs_tol=TOL)    # genuine tie
    assert res[0].adduct == "[M-H]-" and res[0].lipid.name == "PE 38:4"


def test_isobaric_tie_keeps_both_with_equal_score():
    """Two distinct lipids with identical neutral mass, adduct, ppm and priors get
    identical scores; both are returned (the tie is not collapsed)."""
    obs = 600.0
    a = _lipid_for_ion("A iso", "FA", obs, "[M-H]-", 0.5)
    b = _lipid_for_ion("B iso", "FA", obs, "[M-H]-", 0.5)
    tol = 5.0
    ann = Annotator(mode="negative", ppm_tol=tol, db=[a, b])
    res = ann.annotate_mz(obs)
    assert len(res) == 2
    assert math.isclose(res[0].score, res[1].score, abs_tol=TOL)
    assert math.isclose(res[0].score, tol * PRIOR_WINDOW_FRAC * (0.5 + 0.15), abs_tol=TOL)
    assert {c.lipid.name for c in res} == {"A iso", "B iso"}


# --------------------------------------------------------------------------- #
# annotate_many / top_n
# --------------------------------------------------------------------------- #
def test_annotate_many_top_n_and_keys():
    obs = 700.0
    # All exact (ppm 0), same [M-H]- adduct prior (+0.15); rank is then by tissue prior:
    # PE 0.9 > FA 0.5 > PG 0.4.
    db = [
        _lipid_for_ion("PE 34:1", "PE", obs, "[M-H]-", 0.9),
        _lipid_for_ion("PG test", "PG", obs, "[M-H]-", 0.4),
        _lipid_for_ion("FA test", "FA", obs, "[M-H]-", 0.5),
    ]
    ann = Annotator(mode="negative", ppm_tol=5.0, db=db)

    full = ann.annotate_mz(obs)
    assert len(full) == 3
    assert [c.lipid.name for c in full] == ["PE 34:1", "FA test", "PG test"]

    many = ann.annotate_many([obs, 9999.0], top_n=2)
    assert set(many) == {obs, 9999.0}
    assert [c.lipid.name for c in many[obs]] == ["PE 34:1", "FA test"]   # top 2 by score
    assert many[9999.0] == []                                           # no match
