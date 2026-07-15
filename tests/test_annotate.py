"""Tests for the feature-list annotator (smile_msi.annotate) — the literature-link
builder, the rich feature-list DataFrame (columns incl. isotope-consistency +
confidence), and the confidence-scoring edge cases (no match / single candidate /
multiple candidates).

Pure/headless: a tiny hand-built lipid ``db`` (custom neutral masses so the matched
m/z is exact) is passed straight to :func:`annotate.build_feature_list`, and ``ds`` is
``None`` so the isotope/spatial image checks degrade to their no-dataset fallbacks.
No GUI, no network (``research_urls`` builds URLs offline), no external files.
"""
import numbers
import urllib.parse

import pandas as pd
import pytest

from smile_msi import annotate
from smile_msi.isotopes import DELTA_C13
from smile_msi.lipiddb import Lipid
from smile_msi.masses import ion_mz


# --------------------------------------------------------------------------- #
# fixtures: hand-built lipids whose [M-H]- ion lands on a chosen observed m/z
# --------------------------------------------------------------------------- #
def _neutral_for(adduct: str, mz: float) -> float:
    """Neutral mass whose ``adduct`` ion sits at ``mz`` (ion_mz is monotonic in the
    neutral mass for these adducts, so a plain bisection nails it)."""
    lo, hi = mz - 5.0, mz + 50.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if ion_mz(mid, adduct) < mz:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# observed m/z used throughout (a plausible PE [M-H]-)
MZ = 742.5392


@pytest.fixture
def pe_lipid():
    """A single PE whose [M-H]- ion is exactly at MZ (0 ppm)."""
    return Lipid("PE 36:2", "PE", _neutral_for("[M-H]-", MZ), 0.8,
                 formula={"C": 41, "H": 78, "N": 1, "O": 8, "P": 1})


@pytest.fixture
def ps_runner_up():
    """A PS ~4 ppm above MZ with a *lower* tissue prior — it loses to the PE but is a
    real second candidate (drives the multiple-candidate / runner-up-gap path)."""
    return Lipid("PS 34:1", "PS", _neutral_for("[M-H]-", MZ + 0.0030), 0.3,
                 formula={"C": 40, "H": 76, "N": 1, "O": 10, "P": 1})


FEATURE_COLS = [
    "mz", "mz_calibrated", "intensity", "snr", "lipid", "class", "adduct", "ppm", "n_candidates",
    "alternatives", "isotope_m1", "isotope_spectral", "isotope_spatial", "isotope_ok",
    "adducts_seen", "n_adducts", "isotopologue", "spatial_morans_i", "confidence",
    "confidence_score", "confidence_why", "msi_level", "pubmed_url", "europepmc_url",
    "scholar_url",
]


# --------------------------------------------------------------------------- #
# research_urls
# --------------------------------------------------------------------------- #
def test_research_urls_keys_and_hosts():
    urls = annotate.research_urls("PE 36:2", "PE")
    assert set(urls) == {"pubmed", "europepmc", "scholar"}
    assert urls["pubmed"].startswith("https://pubmed.ncbi.nlm.nih.gov/?term=")
    assert urls["europepmc"].startswith("https://europepmc.org/search?query=")
    assert urls["scholar"].startswith("https://scholar.google.com/scholar?q=")


def test_research_urls_percent_encode_lipid_name():
    """The lipid name is URL-quoted into each engine's query parameter; the colon and
    space in a shorthand name become %3A / %20 (not left raw)."""
    name = "PE 36:2"
    q = urllib.parse.quote(name)
    assert q == "PE%2036%3A2"                       # space -> %20, ':' -> %3A
    urls = annotate.research_urls(name)
    assert urls["pubmed"].endswith("term=" + q)
    assert urls["europepmc"].endswith("query=" + q)
    assert urls["scholar"].endswith("q=" + q)
    # round-trips back to the original name
    parsed = urllib.parse.urlparse(urls["pubmed"])
    assert urllib.parse.parse_qs(parsed.query)["term"] == [name]


def test_research_urls_ignores_class_argument():
    """``lipid_class`` is part of the signature but does not change the (name-only)
    queries — same URLs whether or not a class is supplied."""
    assert annotate.research_urls("Cer d18:1/24:1", "Cer") == \
        annotate.research_urls("Cer d18:1/24:1")


# --------------------------------------------------------------------------- #
# build_feature_list: DataFrame shape / columns
# --------------------------------------------------------------------------- #
def test_feature_list_columns_present(pe_lipid):
    df = annotate.build_feature_list(None, [{"mz": MZ, "intensity": 1000.0, "snr": 25.0}],
                                     mode="negative", match_ppm=10.0, db=[pe_lipid])
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == FEATURE_COLS
    assert len(df) == 1
    # isotope-consistency columns exist and (no dataset) read as the blank/False fallback
    assert df.loc[0, "isotope_ok"] == False          # noqa: E712  (numpy bool)
    assert df.loc[0, "isotope_m1"] == ""             # ds=None -> NaN -> blank
    assert df.loc[0, "isotope_spectral"] == ""
    assert df.loc[0, "isotope_spatial"] == ""
    assert df.loc[0, "spatial_morans_i"] == 0.0       # no dataset -> 0.0 default


def test_feature_list_empty_peaks_is_empty_frame(pe_lipid):
    """An empty peak list yields an empty frame (no rows) without raising."""
    df = annotate.build_feature_list(None, [], db=[pe_lipid])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_feature_list_bare_float_peaks_blank_intensity(pe_lipid):
    """Bare-float peaks (no intensity/snr) still annotate; the numeric columns that
    can't be computed render as empty strings, not NaN."""
    df = annotate.build_feature_list(None, [MZ], db=[pe_lipid])
    assert df.loc[0, "intensity"] == ""
    assert df.loc[0, "snr"] == ""
    assert df.loc[0, "lipid"] == "PE 36:2"           # still matched on m/z alone


# --------------------------------------------------------------------------- #
# confidence-scoring edge case: NO MATCH
# --------------------------------------------------------------------------- #
def test_no_match_row_is_unidentified(pe_lipid):
    """A peak with nothing within tolerance gets the empty/unidentified row: no lipid,
    n_candidates 0, blank confidence_score, the 'no match' reason, and no URLs."""
    df = annotate.build_feature_list(None, [{"mz": 123.456, "intensity": 5.0, "snr": 2.0}],
                                     db=[pe_lipid])
    r = df.iloc[0]
    assert r["lipid"] == "" and r["class"] == "" and r["adduct"] == ""
    assert r["n_candidates"] == 0
    assert r["confidence"] == "unidentified"
    assert r["confidence_score"] == ""               # no numeric score for a non-ID
    assert r["confidence_why"] == "no match within tolerance"
    assert r["pubmed_url"] == "" and r["europepmc_url"] == "" and r["scholar_url"] == ""
    assert r["adducts_seen"] == "" and r["n_adducts"] == 0
    assert r["msi_level"] == 5                        # no DB match → genuine unknown


# --------------------------------------------------------------------------- #
# confidence-scoring edge case: SINGLE CANDIDATE
# --------------------------------------------------------------------------- #
def test_single_candidate_only_candidate_reason(pe_lipid):
    """One candidate -> score_gap is None -> the uniqueness term reads 'only candidate'
    (maximally unique), n_candidates == 1, and there are no alternatives listed."""
    df = annotate.build_feature_list(None, [{"mz": MZ, "intensity": 1000.0, "snr": 25.0}],
                                     match_ppm=10.0, db=[pe_lipid])
    r = df.iloc[0]
    assert r["lipid"] == "PE 36:2" and r["class"] == "PE" and r["adduct"] == "[M-H]-"
    assert r["n_candidates"] == 1
    assert r["alternatives"] == ""
    assert "only candidate" in r["confidence_why"]
    assert r["ppm"] == 0.0
    assert r["msi_level"] == 3                        # annotated, but MS1-mass-only (no image) → tentative
    # a real numeric 0-100 confidence with a folded label drawn from it
    # (pandas stores it as a numpy integer in the mixed-dtype column)
    assert isinstance(r["confidence_score"], numbers.Integral)
    assert 0 <= int(r["confidence_score"]) <= 100
    assert r["confidence"] in ("High", "Medium", "Low")
    # the matched row carries research links built from the lipid name
    assert r["pubmed_url"] == annotate.research_urls("PE 36:2")["pubmed"]


def test_msi_level_mapping_and_spatial_ok():
    """The MSI level maps evidence → 1..5 (Schymanski), and _spatial_ok gates level 2 on
    real spatial support (strong isotopologue co-localization or a structured image)."""
    m = annotate.msi_confidence_level
    assert m(annotated=False) == 5                          # no DB match → unknown
    assert m(True, has_msms=True) == 1                      # MS/MS confirmation wins
    assert m(True, isotope_ok=True, spatial_ok=True) == 2   # clean isotope + spatial
    assert m(True, isotope_ok=True, spatial_ok=True, ambiguous=True) == 3  # isobaric tie caps at 3
    assert m(True, isotope_ok=True, spatial_ok=False) == 3  # isotope only → tentative
    assert m(True, formula_only=True) == 4

    ok = annotate._spatial_ok
    assert ok({"spatial": 0.8}) is True                     # strong isotopologue co-loc
    assert ok({"spatial": 0.2}, morans_i=0.5) is True       # weak co-loc but structured image
    assert ok({"spatial": float("nan")}) is False           # no dataset / no evidence
    assert ok({"spatial": 0.2}, morans_i=0.1) is False


def _shifted_peaks(off_ppm, n=40):
    """n well-separated real [M-H]- ions, each shifted by ``off_ppm`` to simulate a
    miscalibrated dataset. Returns (observed_mzs, true_mzs)."""
    from smile_msi.lipiddb import build_database
    from smile_msi.masses import ion_mz
    db = build_database()
    trues = sorted({round(ion_mz(l.neutral_mass, "[M-H]-"), 5) for l in db
                    if l.lipid_class in ("PE", "PC", "PS", "PI", "FA", "Sulfatide")})[::7][:n]
    obs = [t * (1 + off_ppm * 1e-6) for t in trues]
    return obs, trues


def test_estimate_mass_offset_recovers_injected_shift():
    obs, _ = _shifted_peaks(4.0)
    off, n = annotate.estimate_mass_offset(obs, mode="negative")
    assert n >= 30 and abs(off - 4.0) < 0.5                  # recovers the +4 ppm miscalibration


def test_recalibration_gate_skips_when_unsupported_or_small():
    # too few matches → gated off (offset 0), never perturbs a short/odd list
    off, n = annotate.estimate_mass_offset([700.5, 800.6], mode="negative")
    assert off == 0.0 and n < annotate.CAL_MIN_MATCHES
    # a materially-zero offset (already calibrated) → gated off even with many matches
    obs, _ = _shifted_peaks(0.0)
    off0, n0 = annotate.estimate_mass_offset(obs, mode="negative")
    assert n0 >= 30 and off0 == 0.0


def test_build_feature_list_recalibrates_shifted_data():
    obs, _ = _shifted_peaks(4.0)
    peaks = [{"mz": m, "intensity": 100.0, "snr": 10.0} for m in obs]
    off_df = annotate.build_feature_list(None, peaks, match_ppm=2.0, recalibrate=False)
    on_df = annotate.build_feature_list(None, peaks, match_ppm=2.0, recalibrate=True)
    n_off = int((off_df["lipid"] != "").sum())
    n_on = int((on_df["lipid"] != "").sum())
    assert n_on > n_off + 20                                 # +4 ppm ions recovered at a 2 ppm tol
    assert abs(on_df.attrs["mass_offset_ppm"] - 4.0) < 0.5
    assert off_df.attrs["mass_offset_ppm"] == 0.0            # disabled → no correction recorded
    # the reported m/z stays the OBSERVED value (ion images unaffected), not the corrected one
    assert list(on_df["mz"]) == [round(m, 4) for m in obs]
    # ppm is the RESIDUAL after correction (small), not the ~4 ppm raw error
    resid = on_df[on_df["lipid"] != ""]["ppm"].abs()
    assert resid.max() < 2.0
    # the correction the "mz" column drops is still recoverable via "mz_calibrated"
    factor = 1.0 / (1.0 + on_df.attrs["mass_offset_ppm"] * 1e-6)
    assert list(on_df["mz_calibrated"]) == [round(m * factor, 4) for m in obs]
    # gate disabled (or gated off) → mz_calibrated == mz, no phantom correction
    assert list(off_df["mz_calibrated"]) == list(off_df["mz"])


def test_estimate_fdr_offset_matches_shifted_targets():
    obs, _ = _shifted_peaks(4.0)
    r_off = annotate.estimate_fdr(obs, mode="negative", ppm=2.0, offset_ppm=0.0)
    r_on = annotate.estimate_fdr(obs, mode="negative", ppm=2.0, offset_ppm=4.0)
    assert r_on["n_target"] > r_off["n_target"] + 20         # FDR matches the recalibrated masses


def test_attach_fdr_recalibrates_shifted_data():
    """attach_fdr must score the FDR on the SAME gated self-calibration build_feature_list
    applies — otherwise a mis-calibrated dataset has its targets scored off-centre and the
    gate turns spuriously strict. Recalibration recovers the +4 ppm targets a raw (0-offset)
    scoring drops at a 2 ppm tolerance, and records the applied offset in the summary."""
    obs, _ = _shifted_peaks(4.0)
    peaks = [{"mz": m, "intensity": 100.0, "snr": 10.0} for m in obs]
    off, s_off = annotate.attach_fdr(None, peaks, ppm=2.0, recalibrate=False)
    on, s_on = annotate.attach_fdr(None, peaks, ppm=2.0, recalibrate=True)
    assert s_on["n_target"] > s_off["n_target"] + 20         # +4 ppm ions recovered at 2 ppm tol
    assert abs(s_on["mass_offset_ppm"] - 4.0) < 0.5          # the applied offset is auditable
    assert s_off["mass_offset_ppm"] == 0.0                   # disabled → no correction recorded
    assert s_on["n_calibration_matches"] >= annotate.CAL_MIN_MATCHES


def test_single_candidate_score_drops_as_ppm_worsens(pe_lipid):
    """Mass accuracy is the dominant confidence term: the same lipid matched off-centre
    (worse ppm) scores strictly lower than an on-centre (0 ppm) match."""
    on = annotate.build_feature_list(None, [MZ], match_ppm=10.0, db=[pe_lipid]).iloc[0]
    # +0.005 m/z ~ 6.7 ppm, still inside the 10 ppm window
    off = annotate.build_feature_list(None, [MZ + 0.005], match_ppm=10.0, db=[pe_lipid]).iloc[0]
    assert off["lipid"] == "PE 36:2"                 # still the same ID
    assert abs(off["ppm"]) > abs(on["ppm"])
    assert off["confidence_score"] < on["confidence_score"]


# --------------------------------------------------------------------------- #
# confidence-scoring edge case: MULTIPLE CANDIDATES
# --------------------------------------------------------------------------- #
def test_multiple_candidates_winner_and_alternatives(pe_lipid, ps_runner_up):
    """Two candidates within tolerance: the higher-tissue-prior PE wins on score, the PS
    runner-up is listed in ``alternatives`` with a signed ppm, n_candidates == 2, and the
    decisive score gap reads 'clear best match'."""
    df = annotate.build_feature_list(None, [{"mz": MZ, "intensity": 1000.0, "snr": 25.0}],
                                     match_ppm=10.0, db=[pe_lipid, ps_runner_up])
    r = df.iloc[0]
    assert r["n_candidates"] == 2
    assert r["lipid"] == "PE 36:2"                   # PE prior (0.8) beats PS prior (0.3)
    assert "PS 34:1" in r["alternatives"]
    assert "[M-H]-" in r["alternatives"]
    # the alternatives string embeds the runner-up's signed ppm e.g. "(-4.0)"
    assert "(" in r["alternatives"] and ")" in r["alternatives"]
    assert "clear best match" in r["confidence_why"]


def test_multiple_candidates_top_alts_truncates(pe_lipid):
    """``top_alts`` caps how many runner-ups appear in ``alternatives``. Three near-
    isobaric extra lipids + the winner = 4 candidates; with top_alts=1 only one
    alternative is rendered."""
    extras = [
        Lipid(f"PS {i}", "PS", _neutral_for("[M-H]-", MZ + 0.001 * (i + 1)), 0.30 - 0.01 * i,
              formula={"C": 40, "H": 76, "N": 1, "O": 10, "P": 1})
        for i in range(3)
    ]
    df = annotate.build_feature_list(None, [MZ], match_ppm=10.0, top_alts=1,
                                     db=[pe_lipid, *extras])
    r = df.iloc[0]
    assert r["n_candidates"] == 4
    # alternatives are " | "-joined; top_alts=1 keeps exactly one
    assert r["alternatives"].count(" | ") == 0
    assert r["alternatives"] != ""                   # but there IS one alternative


def test_winner_score_at_least_runner_up(pe_lipid, ps_runner_up):
    """Sanity on the ranking the gap is computed from: the reported confidence for the
    decisive PE win is a high-confidence-eligible number (and never below the no-ID 0)."""
    df = annotate.build_feature_list(None, [MZ], match_ppm=10.0, db=[pe_lipid, ps_runner_up])
    r = df.iloc[0]
    assert isinstance(r["confidence_score"], numbers.Integral) and int(r["confidence_score"]) > 0


# --------------------------------------------------------------------------- #
# isotopologue flagging (deisotope mask feeds the 'isotopologue' column)
# --------------------------------------------------------------------------- #
def test_isotopologue_satellite_flagged(pe_lipid):
    """An M+1 satellite (one 13C-step above a *more intense* mono peak) is flagged as an
    isotopologue; the mono peak is not."""
    mono = {"mz": MZ, "intensity": 1000.0}
    sat = {"mz": MZ + DELTA_C13, "intensity": 200.0}
    df = annotate.build_feature_list(None, [mono, sat], db=[pe_lipid])
    assert list(df["isotopologue"]) == [False, True]


def test_isotopologue_not_flagged_when_no_intensity(pe_lipid):
    """Bare-float peaks carry no intensity, so the deisotope step (which needs a strictly
    more intense neighbour below) flags nothing — even on a perfect 13C spacing."""
    df = annotate.build_feature_list(None, [MZ, MZ + DELTA_C13], db=[pe_lipid])
    assert list(df["isotopologue"]) == [False, False]


# --------------------------------------------------------------------------- #
# multi-row integration: order preserved, mixed matched / unmatched
# --------------------------------------------------------------------------- #
def test_mixed_matched_and_unmatched_rows_keep_order(pe_lipid):
    peaks = [
        {"mz": 100.0, "intensity": 3.0, "snr": 1.5},     # no match
        {"mz": MZ, "intensity": 900.0, "snr": 20.0},     # PE match
    ]
    df = annotate.build_feature_list(None, peaks, db=[pe_lipid])
    assert len(df) == 2
    assert list(df["mz"]) == [100.0, round(MZ, 4)]
    assert df.iloc[0]["confidence"] == "unidentified"
    assert df.iloc[1]["lipid"] == "PE 36:2"


def test_attach_fdr_gates_but_keeps_unknowns(monkeypatch):
    """attach_fdr thresholds by q-value but never drops unannotated (unknown) ions."""
    peaks = [{"mz": 700.1}, {"mz": 800.2}, {"mz": 900.3}, {"mz": 1000.4}]
    # q for 700/800/1000; 900 has NO database match (NaN) -> a genuine "unknown"
    fake = {"q_values": [0.02, 0.15, float("nan"), 0.50],
            "levels": {0.05: 1, 0.10: 1, 0.20: 2, 0.50: 3},
            "fdr": 0.3, "reliable": True, "n_target": 3, "n_peaks": 4,
            "image_based": True}
    monkeypatch.setattr(annotate, "estimate_fdr", lambda *a, **k: fake)

    # no threshold: every feature kept, q_value attached; annotated → fdr_pass
    allp, summ = annotate.attach_fdr(None, peaks, q_max=None)
    assert len(allp) == 4
    assert allp[0]["q_value"] == 0.02 and allp[0]["fdr_pass"] is True
    assert allp[1]["fdr_pass"] is True                      # annotated, no threshold applied
    assert allp[2]["q_value"] != allp[2]["q_value"]         # NaN preserved for the unknown
    assert allp[2]["fdr_pass"] is False                     # unannotated never "passes"

    # q<=0.10: keep the passing ID (0.02) AND the unknown (NaN), drop 0.15 and 0.50
    kept, summ = annotate.attach_fdr(None, peaks, q_max=0.10, keep_unannotated=True)
    mzs = sorted(p["mz"] for p in kept)
    assert mzs == [700.1, 900.3]                            # 0.02-passer + the unknown
    assert summ["n_unannotated_kept"] == 1
    assert summ["levels"][0.10] == 1

    # keep_unannotated=False: only the q<=0.10 passer survives
    strict, _ = annotate.attach_fdr(None, peaks, q_max=0.10, keep_unannotated=False)
    assert [p["mz"] for p in strict] == [700.1]


def test_estimate_fdr_qvalues_are_monotonic_not_inverted():
    """Regression for the inverted q-value direction (q built best-first with a running
    MAX pinned every ID to the inflated top-of-ranking FDR; the fix walks worst-first
    with a running MIN). With distinct target scores and no decoy hits, FDR(t) =
    1/(n_target+1) at the accept-all threshold, so every monotone q must equal that —
    NOT the 0.5 the inverted code produced. Distinct scores are essential: all-tied
    scores accidentally mask the bug.
    """
    from smile_msi.lipiddb import Lipid as _Lipid
    from smile_msi.masses import ADDUCTS, ADDUCT_CHARGE

    def lipid_for_ion(name, ion_target, adduct="[M-H]-"):
        z = ADDUCT_CHARGE[adduct]
        neutral = ion_target * z - ADDUCTS[adduct]
        return _Lipid(name, "FA", neutral, 0.9, formula={})

    n = 25
    # irregular (non-grid) spacing so a decoy ion (neutral + implausible-element mass)
    # can't systematically alias onto another peak
    ions = [410.4 + 37.13 * i for i in range(n)]
    db = [lipid_for_ion(f"L{i}", m) for i, m in enumerate(ions)]
    # place each peak a *distinct* sub-tolerance ppm off its ion so mass-only scores differ
    peaks = [m * (1.0 + (i * 0.3) * 1e-6) for i, m in enumerate(ions)]

    res = annotate.estimate_fdr(peaks, mode="negative", ppm=10.0, db=db, ds=None)

    assert res["n_target"] == n                            # every peak matched its lipid
    finite_q = [q for q in res["q_values"] if q == q]
    assert len(finite_q) == n
    assert res["fdr"] < 0.5 - 1e-9                          # accept-all FDR well below the bug's 0.5
    # the core invariant of a correct target–decoy q-value: q(score) ≤ accept-all FDR for
    # every ID (the accept-all threshold is in every "min over t ≤ s" set). The inverted
    # code violated this — best IDs got q = 0.5 ≫ fdr.
    assert max(finite_q) <= res["fdr"] + 1e-9
    # with no decoy hits FDR = 1/(n+1) ≈ 0.038, so every ID passes at 5%; the inverted
    # code (q ≡ 0.5) reported 0 passing.
    assert res["levels"][0.05] == n


def test_rerank_by_msm_promotes_better_image_fit():
    """With a dataset, candidates are re-ranked by MSM (mass x isotope x spatial): an
    isobaric runner-up the mass+prior ranking placed second is promoted when its predicted
    isotope pattern clearly fits the image better. Two candidates share the observed peak
    (same neutral_mass); A has the higher tissue prior (so it wins on mass+prior) but B's
    formula predicts the M+1/M0 ratio the data actually shows."""
    import numpy as np
    from smile_msi import isotopes
    from smile_msi.lipiddb import Lipid
    from smile_msi.masses import formula_mass, ion_mz
    from smile_msi.match import Annotator

    fA = {"C": 40, "H": 70, "O": 4}        # predicted M+1 ~0.44 (40 carbons)
    fB = {"C": 20, "H": 70, "O": 19}       # predicted M+1 ~0.22 (20 carbons)
    N = formula_mass(fA)
    A = Lipid("A highprior", "PC", N, 0.9, formula=fA)
    B = Lipid("B lowprior", "PE", N, 0.3, formula=fB)
    obs = ion_mz(N, "[M-H]-")

    patA = isotopes.theoretical_isotope_pattern(fA, "[M-H]-", 3)
    patB = isotopes.theoretical_isotope_pattern(fB, "[M-H]-", 3)
    m0s = [patA[0][0], patB[0][0]]
    m1s = [patA[1][0], patB[1][0]]
    m2s = [patA[2][0], patB[2][0]]
    base = np.linspace(1.0, 5.0, 60)       # structured image so correlations are defined
    ratios = [1.0, 0.22, 0.03]             # measured envelope: matches B, not A

    class FakeDS:
        def ion_vector(self, mz, tol_ppm=50.0, norm="tic"):
            for grp, r in zip((m0s, m1s, m2s), ratios):
                if any(abs(mz - g) < 0.3 for g in grp):
                    return base * r
            return np.zeros_like(base)

    ds = FakeDS()
    ann = Annotator(mode="negative", ppm_tol=10.0, db=[A, B])
    # mass + prior alone: A wins (same 0 ppm, higher tissue prior)
    assert [c.lipid.name for c in ann.annotate_mz(obs)] == ["A highprior", "B lowprior"]

    cands = ann.annotate_mz(obs)
    reordered, best_iso = annotate._rerank_by_msm(
        ds, cands, match_ppm=10.0, image_ppm=50.0, norm="tic", iso_cache=None)
    assert reordered[0].lipid.name == "B lowprior"     # promoted on image evidence
    assert reordered[0].lipid.lipid_class == "PE"
    assert best_iso["spectral"] >= 0.85                # the new best fits the isotope pattern


def test_estimate_fdr_median_is_robust_to_one_overaliased_decoy_element():
    """Regression for the pooled-decoy skew (#3): one decoy element that happens to alias
    many peaks must not inflate the FDR, because the estimator takes the MEDIAN over decoy
    samples — the outlier element is a single vote, not summed into the pool. Adding noise
    peaks that all alias the SAME decoy element leaves the clean IDs' levels/FDR unchanged.
    (Pooling every element-decoy into one ranking — the old code — would inflate both.)"""
    from smile_msi.lipiddb import Lipid as _Lipid
    from smile_msi.masses import ADDUCTS, ADDUCT_CHARGE, DECOY_ELEMENT_MASSES

    z = ADDUCT_CHARGE["[M-H]-"]

    def lip(name, ion):
        return _Lipid(name, "FA", ion * z - ADDUCTS["[M-H]-"], 0.9, formula={})

    ions = [410.4 + 37.13 * i for i in range(22)]          # irregular spacing -> no clean aliasing
    db = [lip(f"L{i}", m) for i, m in enumerate(ions)]
    peaks = [m * (1.0 + (i * 0.3) * 1e-6) for i, m in enumerate(ions)]
    base = annotate.estimate_fdr(peaks, mode="negative", ppm=10.0, db=db, ds=None)

    # 8 noise peaks far from the targets, each aliasing the SAME decoy element (F) via a
    # bait lipid whose neutral mass + F lands exactly on the peak.
    F = DECOY_ELEMENT_MASSES["F"]
    noise = [1500.0 + 211.3 * i for i in range(8)]
    baits = [_Lipid(f"bait{i}", "FA", P - F, 0.9, formula={}) for i, P in enumerate(noise)]
    res = annotate.estimate_fdr(peaks + noise, mode="negative", ppm=10.0,
                                db=db + baits, ds=None)

    assert res["n_target"] == base["n_target"] == 22       # noise peaks match nothing real
    assert res["levels"] == base["levels"]                 # clean IDs unaffected by the outlier
    assert res["fdr"] == pytest.approx(base["fdr"])


def test_estimate_fdr_decoy_sample_is_reproducible():
    """The decoy-element sample is seeded, so estimate_fdr is deterministic run-to-run
    (no FDR jitter from a random decoy draw) and reports the sample size."""
    from smile_msi.lipiddb import Lipid as _Lipid
    from smile_msi.masses import ADDUCTS, ADDUCT_CHARGE, DECOY_ELEMENT_MASSES

    z = ADDUCT_CHARGE["[M-H]-"]
    ions = [410.4 + 37.13 * i for i in range(12)]
    db = [_Lipid(f"L{i}", "FA", m * z - ADDUCTS["[M-H]-"], 0.9, formula={})
          for i, m in enumerate(ions)]
    peaks = [m * (1.0 + (i * 0.3) * 1e-6) for i, m in enumerate(ions)]
    a = annotate.estimate_fdr(peaks, mode="negative", ppm=10.0, db=db, ds=None, n_decoy=20)
    b = annotate.estimate_fdr(peaks, mode="negative", ppm=10.0, db=db, ds=None, n_decoy=20)
    assert a["q_values"] == pytest.approx(b["q_values"], nan_ok=True)
    assert a["fdr"] == b["fdr"]
    assert a["n_decoy"] == min(20, len(DECOY_ELEMENT_MASSES))


def test_msi_confidence_levels():
    L = annotate.msi_confidence_level
    assert L(annotated=False) == 5                                  # no DB match → unknown
    assert L(annotated=True, has_msms=True) == 1                    # MS/MS confirmed
    assert L(annotated=True, isotope_ok=True, spatial_ok=True) == 2  # isotope+spatial
    assert L(annotated=True, ambiguous=True) == 3                   # isobaric tie
    assert L(annotated=True, formula_only=True) == 4
    assert L(annotated=True) == 3                                   # bare match, weak evidence
    assert set(annotate.MSI_LEVELS) == {1, 2, 3, 4, 5}


# --------------------------------------------------------------------------- #
# Plan 23 item B — optional isotope-score cache (build_feature_list ↔ estimate_fdr)
# --------------------------------------------------------------------------- #
def test_iso_key_collides_for_same_inputs_and_differs_on_tol():
    """The key build_feature_list and estimate_fdr each derive must collide for the
    same (formula, adduct, mz, ppm, norm) and differ when any component differs."""
    from smile_msi.isotopes import _iso_key
    f = {"C": 40, "H": 80, "O": 8, "P": 1}
    k1 = _iso_key(f, "[M-H]-", 700.4912, 50.0, "tic")
    k2 = _iso_key(dict(f), "[M-H]-", 700.49123, 50.0, "tic")   # same to 4 dp
    assert k1 == k2
    assert k1 != _iso_key(f, "[M-H]-", 700.4912, 30.0, "tic")  # tol differs
    assert k1 != _iso_key(f, "[M+Cl]-", 700.4912, 50.0, "tic")  # adduct differs
    assert k1 != _iso_key(f, "[M-H]-", 700.4912, 50.0, "rms")   # norm differs


def test_isotope_scores_cache_skips_image_pull():
    """A cache hit returns the stored dict without re-pulling the isotopologue images
    (the expensive ds.ion_vector calls)."""
    from smile_msi import demo, isotopes
    ds = demo.make_synthetic(width=16, height=12, seed=11)
    ds.prime()
    pat = isotopes.theoretical_isotope_pattern({"C": 30, "H": 58, "O": 2}, "[M-H]-", n_peaks=3)
    cache, key = {}, ("k",)

    calls = {"n": 0}
    orig = ds.ion_vector
    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    ds.ion_vector = spy

    first = isotopes.isotope_scores(ds, pat, cache=cache, key=key)
    after_first = calls["n"]
    assert after_first > 0 and key in cache
    second = isotopes.isotope_scores(ds, pat, cache=cache, key=key)
    assert calls["n"] == after_first          # no further image pulls
    assert second is first                     # same cached dict


def test_isotope_scores_uncached_recomputes_each_call():
    """cache=None (default) reproduces the old behaviour: every call pulls images."""
    from smile_msi import demo, isotopes
    ds = demo.make_synthetic(width=16, height=12, seed=7)
    ds.prime()
    pat = isotopes.theoretical_isotope_pattern({"C": 30, "H": 58, "O": 2}, "[M-H]-", n_peaks=3)
    calls = {"n": 0}
    orig = ds.ion_vector
    ds.ion_vector = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), orig(*a, **k))[1]
    isotopes.isotope_scores(ds, pat)
    n1 = calls["n"]
    isotopes.isotope_scores(ds, pat)
    assert calls["n"] == 2 * n1                # recomputed, not memoized


def test_estimate_fdr_iso_cache_is_noop_when_absent():
    """Passing iso_cache must not change estimate_fdr output (the cache is a pure memo)."""
    from smile_msi.lipiddb import Lipid as _Lipid
    from smile_msi.masses import ADDUCTS, ADDUCT_CHARGE
    z = ADDUCT_CHARGE["[M-H]-"]
    ions = [410.4 + 37.13 * i for i in range(12)]
    db = [_Lipid(f"L{i}", "FA", m * z - ADDUCTS["[M-H]-"], 0.9, formula={})
          for i, m in enumerate(ions)]
    peaks = [m * (1.0 + (i * 0.3) * 1e-6) for i, m in enumerate(ions)]
    a = annotate.estimate_fdr(peaks, mode="negative", ppm=10.0, db=db, ds=None)
    b = annotate.estimate_fdr(peaks, mode="negative", ppm=10.0, db=db, ds=None, iso_cache={})
    assert a["q_values"] == pytest.approx(b["q_values"], nan_ok=True)
    assert a["levels"] == b["levels"]
