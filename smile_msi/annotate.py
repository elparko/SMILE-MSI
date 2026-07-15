"""Feature-list annotation — turn a picked-peak list into an information-rich
table of lipid identifications, the way you'd want it "on hand" while exploring.

For each feature this combines:
  * the in-silico match (best lipid + alternative candidates, adduct, ppm)
  * **isotope consistency** (M+1 pattern) from the data
  * **adduct corroboration** (sibling adduct ions present in the list)
  * a folded **confidence** label *and* a 0–100 ``confidence_score`` with a
    ``confidence_why`` breakdown (mass accuracy, isotope, adducts, runner-up gap)
  * **research links** (PubMed / Europe PMC / Google Scholar) per lipid

and adds a dataset-level **annotation FDR** estimate via a target-decoy search, so
you know how trustworthy the whole feature list is.
"""
from __future__ import annotations

import bisect
import random
import urllib.parse

import numpy as np
import pandas as pd

from . import isotopes
from .match import Annotator


def research_urls(lipid_name: str, lipid_class: str = "") -> dict:
    """Offline-built literature-search URLs for a lipid (no network call here)."""
    q = urllib.parse.quote(lipid_name)
    return {
        "pubmed": f"https://pubmed.ncbi.nlm.nih.gov/?term={q}",
        "europepmc": f"https://europepmc.org/search?query={q}",
        "scholar": f"https://scholar.google.com/scholar?q={q}",
    }


# How many top mass+prior candidates to re-score by image evidence (MSM) per peak when
# a dataset is available. Bounds the extra isotope-image pulls; the true ion is virtually
# always within the few closest-mass candidates.
RERANK_TOPK = 6

# MSI identification level 2 ("probable") requires the ion to show real spatial structure, not
# just a mass + isotope match. Treat a strong isotopologue spatial co-localization (≥ this) OR a
# clearly structured single-ion image (Moran's I ≥ this) as the spatial support the level needs.
MSI_SPATIAL_MIN = 0.5
MSI_MORANS_MIN = 0.3
# A runner-up within this score gap (≈ ppm units) of the winner is an unresolved isobaric tie —
# the ID can't be pinned to one structure by MS1 alone, so it caps at MSI level 3.
MSI_AMBIGUOUS_GAP = 1.0


def _spatial_ok(iso, morans_i=None) -> bool:
    """Whether an annotation carries enough spatial evidence for MSI level 2 — a strong
    isotopologue co-localization score or a clearly structured image (Moran's I). NaN/absent
    evidence (e.g. no dataset) reads as *not* spatially supported."""
    spat = iso.get("spatial")
    if spat is not None and spat == spat and spat >= MSI_SPATIAL_MIN:
        return True
    return morans_i is not None and morans_i == morans_i and morans_i >= MSI_MORANS_MIN


def _candidate_iso(ds, cand, image_ppm, norm, iso_cache):
    """Image isotope scores (:func:`isotopes.isotope_scores`) for one candidate, keyed so
    the result is shared with :func:`estimate_fdr` via ``iso_cache``. Empty-formula
    candidates can't be scored — return the neutral/blank dict."""
    formula = cand.lipid.formula
    if not formula:
        return {"m1_ratio": 0.0, "m1_expected": 0.0, "spectral": float("nan"),
                "spatial": float("nan"), "consistent": False, "n_iso": 0}
    pat = isotopes.theoretical_isotope_pattern(formula, cand.adduct, n_peaks=3)
    return isotopes.isotope_scores(
        ds, pat, tol_ppm=image_ppm, norm=norm, cache=iso_cache,
        key=isotopes._iso_key(formula, cand.adduct, cand.mz, image_ppm, norm))


def _rerank_by_msm(ds, cands, match_ppm, image_ppm, norm, iso_cache, k=RERANK_TOPK):
    """Re-rank the top-``k`` mass+prior candidates by their **MSM** score
    (mass x spectral-isotope x spatial-isotope) so a runner-up the priors ranked below the
    winner can be promoted when the image evidence decisively supports it (METASPACE's
    ranking criterion; Palmer et al. 2017). Candidates past ``k`` keep their order after.

    Returns ``(reordered_candidates, iso_dict_of_the_new_best)`` — the best candidate's
    isotope scores are reused for the feature row so it isn't re-pulled."""
    head = cands[:k]
    scored = []
    for c in head:
        iso = _candidate_iso(ds, c, image_ppm, norm, iso_cache)
        msm = _msm(_mass_score(c.ppm, match_ppm), iso.get("spectral", float("nan")),
                   iso.get("spatial", float("nan")))
        scored.append((msm, c, iso))
    # MSM desc; exact ties keep the original mass+prior order (stable via the index key).
    order = sorted(range(len(scored)), key=lambda i: (-scored[i][0], i))
    reordered = [scored[i][1] for i in order] + list(cands[k:])
    return reordered, scored[order[0]][2]


# ----- per-dataset self-calibration ---------------------------------------- #
# A systematic m/z offset is applied to the DB match only when it is well-supported (>= this
# many confident base-ion matches) AND materially non-zero (median |ppm| above this) — so an
# already-calibrated list is left untouched (measured: correcting a +1.2 ppm list lost IDs). A
# wide window is used only to *measure* the offset; the match itself stays at the tight tol.
CAL_WIDE_PPM = 15.0
CAL_MIN_MATCHES = 30
CAL_MIN_ABS_PPM = 2.0


def estimate_mass_offset(peak_mzs, mode: str = "negative", db=None, *,
                         wide_ppm: float = CAL_WIDE_PPM, min_matches: int = CAL_MIN_MATCHES,
                         min_abs_ppm: float = CAL_MIN_ABS_PPM) -> tuple:
    """Median signed **ppm offset** of a picked-peak list against the database — a per-dataset
    self-calibration measured from the peaks' *own* confident matches (no lock-mass anchors
    needed, unlike :func:`smile_msi.intake.measure_calibration_offset`, which reads the mean
    spectrum vs known reference ions).

    Each peak is matched at a **wide** window; for peaks that match, the signed ppm of the
    closest **base-ion** ([M-H]-/[M+H]+) candidate is collected (adduct assignments are
    ambiguous and would bias the fit). Returns ``(offset_ppm, n_used)``. The offset is **0.0
    unless** it is both well-supported (``>= min_matches`` base-ion matches) and materially
    non-zero (``|median| > min_abs_ppm``) — so a well-calibrated dataset is not perturbed. A
    *constant* offset is used (a mass-dependent linear fit measured worse on real MSI lists).

    Apply it as ``mz_true = mz / (1 + offset_ppm * 1e-6)`` before matching — recovering the
    ~-3 ppm systematic error that otherwise leaves real ions just outside a tight tolerance.
    On real nerve data this lifted the 5 ppm annotation rate from 124/500 to ~150/500 while
    *lowering* the target-decoy FDR (the error distribution re-centres on zero)."""
    ann = Annotator(mode=mode, ppm_tol=wide_ppm, db=db)
    base = ann._base_adduct
    errs = []
    for mz in peak_mzs:
        cs = ann.annotate_mz(float(mz))
        if not cs:
            continue
        pool = [c for c in cs if c.adduct == base] or cs
        errs.append(min(pool, key=lambda c: abs(c.ppm)).ppm)
    n = len(errs)
    if n < min_matches:
        return 0.0, n
    off = float(np.median(errs))
    return (off if abs(off) > min_abs_ppm else 0.0), n


def build_feature_list(ds, peaks, mode: str = "negative", match_ppm: float = 5.0,
                       image_ppm: float = 50.0, norm: str = "tic", top_alts: int = 3,
                       db=None, iso_cache=None, rerank: bool = True,
                       recalibrate: bool = True) -> pd.DataFrame:
    """Annotate a list of peaks (dicts with ``mz``/``intensity``/``snr``, or bare
    m/z floats) into a feature-list DataFrame. ``ds`` enables the isotope check
    (pass ``None`` to skip it).

    When ``ds`` is provided and ``rerank`` is true, each peak's candidates are re-ranked
    by image evidence (MSM = mass x spectral x spatial; :func:`_rerank_by_msm`) rather than
    by mass + class/adduct priors alone — so an isobaric runner-up that the data clearly
    supports becomes the reported ID. With ``ds=None`` the mass+prior ranking is used as-is.

    ``recalibrate`` (default on) measures the list's systematic m/z offset
    (:func:`estimate_mass_offset`) and matches each peak against the DB at the **corrected**
    m/z — so a dataset that sits a few ppm off the database still annotates at a tight
    tolerance instead of falling just outside it. It is **gated**: a well-calibrated list
    (small offset / too few matches) is matched unchanged. The ``mz`` column stays the raw
    measured value — everything keyed off it (ion image extraction, the ``roi_auc``/``fdr_q``
    joins in the GUI, peak lookups elsewhere) extracts from the cube at that observed m/z, so
    changing it would desync those joins. The applied correction is instead exposed as its own
    ``mz_calibrated`` column (``mz`` unchanged when the gate didn't fire) so a user diffing
    against theoretical masses isn't silently shown only the uncorrected value; ``ppm`` is the
    residual after correction. The offset is also recorded on ``df.attrs['mass_offset_ppm']``
    (with ``['n_calibration_matches']``) so it is auditable.

    ``iso_cache`` is an optional caller-owned dict memoizing per-``(formula, adduct,
    mz, ppm, norm)`` isotope scores; pass the *same* dict to a following
    :func:`estimate_fdr` so the shared monoisotopic peaks aren't scored twice
    (audit plan 23, item B)."""
    ann = Annotator(mode=mode, ppm_tol=match_ppm, db=db)
    peak_mzs = [float(p["mz"]) if isinstance(p, dict) else float(p) for p in peaks]
    _, is_iso = isotopes.deisotope(peaks)            # mark M+1/M+2 satellites
    iso_flag = {peak_mzs[i]: bool(is_iso[i]) for i in range(len(peak_mzs))}
    # Per-dataset self-calibration: correct the m/z used for DB matching (not the reported
    # "mz" column, nor cube extraction) by the measured systematic offset — the correction is
    # separately exposed as "mz_calibrated" below so it isn't lost to the export. Gated → 0
    # for a good list.
    offset_ppm, n_cal = (estimate_mass_offset(peak_mzs, mode=mode, db=db)
                         if (recalibrate and peak_mzs) else (0.0, 0))
    cal_factor = 1.0 / (1.0 + offset_ppm * 1e-6) if offset_ppm else 1.0
    peak_mzs_q = [m * cal_factor for m in peak_mzs] if offset_ppm else peak_mzs
    # spatial structure per peak (Moran's I) when a dataset is provided — on the OBSERVED m/z
    morans = {}
    if ds is not None and peak_mzs:
        from . import spatial as _spatial
        sa = _spatial.spatial_autocorrelation(ds, peak_mzs, tol_ppm=image_ppm, norm=norm)
        morans = dict(zip(sa["mz"], sa["morans_i"]))
    rows = []
    for p in peaks:
        mz = float(p["mz"]) if isinstance(p, dict) else float(p)
        intensity = p.get("intensity", np.nan) if isinstance(p, dict) else np.nan
        snr = p.get("snr", np.nan) if isinstance(p, dict) else np.nan
        cands = ann.annotate_mz(mz * cal_factor)     # match at the recalibrated m/z
        # With a dataset, re-rank candidates by image evidence (MSM) so a runner-up the
        # priors mis-ranked can win; reuse the new best's isotope scores for the row.
        best_iso = None
        if ds is not None and rerank and cands:
            cands, best_iso = _rerank_by_msm(ds, cands, match_ppm, image_ppm, norm, iso_cache)
        best = cands[0] if cands else None
        # Isotope check using the best candidate's **real** formula + adduct so the M+1/M+2
        # expectation is the true envelope (METASPACE spectral + spatial scores), not a
        # carbon proxy. Reuse the score from re-ranking; otherwise compute it now (no
        # candidate -> averagine proxy fallback).
        if ds is None:
            iso = {"m1_ratio": np.nan, "consistent": False,
                   "spectral": np.nan, "spatial": np.nan}
        elif best_iso is not None:
            iso = best_iso
        elif best is not None:
            iso = _candidate_iso(ds, best, image_ppm, norm, iso_cache)
        else:
            iso = isotopes.isotope_consistency(ds, mz, charge=1, tol_ppm=image_ppm,
                                               norm=norm, cache=iso_cache)
        if best:
            add = isotopes.corroborating_adducts(best.lipid.neutral_mass, mode, peak_mzs_q)
            # how decisively the top candidate beat the runner-up (score units ≈ ppm);
            # None when it's the only candidate, which reads as maximally unique
            gap = (best.score - cands[1].score) if len(cands) > 1 else None
            cd = isotopes.confidence_detail(best.ppm, iso["consistent"], len(add),
                                            ppm_tol=match_ppm, score_gap=gap,
                                            spectral=iso.get("spectral"),
                                            spatial=iso.get("spatial"),
                                            morans=morans.get(mz))
            # Metabolomics-Standards-Initiative identification level (Schymanski 2014), MS1-only:
            # an isobaric tie caps at 3; a clean isotope + spatial match earns 2; otherwise 3.
            # (MS/MS confirmation, when available, promotes a row to level 1 downstream.)
            ambiguous_id = bool(len(cands) > 1 and gap is not None and gap < MSI_AMBIGUOUS_GAP)
            msi_lvl = msi_confidence_level(True, isotope_ok=iso["consistent"],
                                           spatial_ok=_spatial_ok(iso, morans.get(mz)),
                                           ambiguous=ambiguous_id)
            urls = research_urls(best.lipid.name, best.lipid.lipid_class)
            alts = " | ".join(f"{c.lipid.name} {c.adduct} ({c.ppm:+.1f})"
                              for c in cands[1:1 + top_alts])
            rows.append({
                "mz": round(mz, 4), "mz_calibrated": round(mz * cal_factor, 4),
                "intensity": round(float(intensity), 1) if intensity == intensity else "",
                "snr": round(float(snr), 1) if snr == snr else "",
                "lipid": best.lipid.name, "class": best.lipid.lipid_class, "adduct": best.adduct,
                "ppm": round(best.ppm, 2), "n_candidates": len(cands), "alternatives": alts,
                "isotope_m1": round(iso["m1_ratio"], 3) if iso["m1_ratio"] == iso["m1_ratio"] else "",
                "isotope_spectral": round(float(iso.get("spectral")), 3)
                    if iso.get("spectral") == iso.get("spectral") else "",
                "isotope_spatial": round(float(iso.get("spatial")), 3)
                    if iso.get("spatial") == iso.get("spatial") else "",
                "isotope_ok": iso["consistent"], "adducts_seen": ",".join(add), "n_adducts": len(add),
                "isotopologue": iso_flag.get(mz, False),
                "spatial_morans_i": round(float(morans.get(mz, 0.0)), 3),
                "confidence": cd["label"], "confidence_score": cd["score"],
                "confidence_why": " · ".join(cd["reasons"]), "msi_level": msi_lvl,
                "pubmed_url": urls["pubmed"], "europepmc_url": urls["europepmc"], "scholar_url": urls["scholar"],
            })
        else:
            rows.append({
                "mz": round(mz, 4), "mz_calibrated": round(mz * cal_factor, 4),
                "intensity": round(float(intensity), 1) if intensity == intensity else "",
                "snr": round(float(snr), 1) if snr == snr else "",
                "lipid": "", "class": "", "adduct": "", "ppm": "", "n_candidates": 0, "alternatives": "",
                "isotope_m1": round(iso["m1_ratio"], 3) if iso["m1_ratio"] == iso["m1_ratio"] else "",
                "isotope_spectral": round(float(iso.get("spectral")), 3)
                    if iso.get("spectral") == iso.get("spectral") else "",
                "isotope_spatial": round(float(iso.get("spatial")), 3)
                    if iso.get("spatial") == iso.get("spatial") else "",
                "isotope_ok": iso["consistent"], "adducts_seen": "", "n_adducts": 0,
                "isotopologue": iso_flag.get(mz, False),
                "spatial_morans_i": round(float(morans.get(mz, 0.0)), 3), "confidence": "unidentified",
                "confidence_score": "", "confidence_why": "no match within tolerance", "msi_level": 5,
                "pubmed_url": "", "europepmc_url": "", "scholar_url": "",
            })
    df = pd.DataFrame(rows)
    df.attrs["mass_offset_ppm"] = float(offset_ppm)       # 0.0 when the gate left the list as-is
    df.attrs["n_calibration_matches"] = int(n_cal)
    return df


FDR_LEVELS = (0.05, 0.10, 0.20, 0.50)   # METASPACE-style reporting tiers

# Seed for the (reproducible) random sample of decoy elements in estimate_fdr.
_DECOY_SEED = 0


def _mass_score(ppm: float, tol: float) -> float:
    """Mass-accuracy component in [0,1]: 1 at 0 ppm, 0 at the tolerance edge."""
    return max(0.0, 1.0 - abs(ppm) / tol) if tol > 0 else 0.0


def _msm(mass: float, spectral: float, spatial: float) -> float:
    """MSM-style annotation score: product of the [0,1] evidence measures
    (METASPACE multiplies mass × spectral isotope × spatial isotope). Missing
    image evidence (NaN, e.g. no dataset) is treated as neutral so the score
    degrades gracefully to mass accuracy alone."""
    s = mass
    if spectral == spectral:                       # not NaN
        s *= spectral
    if spatial == spatial:
        s *= spatial
    return s


def _decoy_pattern(formula: dict, base_mz: float, n_peaks: int = 3):
    """Isotope pattern a decoy *claims*: the real formula's neutral abundances at
    ¹³C spacing above the matched peak. Its M+1/M+2 fall on whatever happens to be
    there, so the spatial/spectral scores collapse for coincidental matches — which
    is exactly what should drive the false annotations' scores down."""
    from .masses import isotope_distribution
    dist = isotope_distribution(formula, max_offset=n_peaks - 1)
    pmax = max((p for _, p in dist if p > 0), default=1.0) or 1.0
    return [(base_mz + k * isotopes.DELTA_C13, p / pmax)
            for k, (_m, p) in enumerate(dist) if p > 0]


def estimate_fdr(peak_mzs, mode: str = "negative", ppm: float = 5.0, db=None,
                 ds=None, norm: str = "tic", image_ppm: float = 50.0,
                 n_decoy: int = 20, iso_cache=None, offset_ppm: float = 0.0) -> dict:
    """Per-annotation target–decoy FDR for a peak list (METASPACE-style).

    ``offset_ppm`` applies the same per-dataset mass recalibration as
    :func:`build_feature_list` (match at ``mz / (1 + offset_ppm·1e-6)``) so the FDR is computed
    on the *same* corrected masses the feature list annotated — pass ``df.attrs['mass_offset_ppm']``.
    The default 0.0 leaves matching unchanged.

    **Targets** = each peak's best real database ion within ``ppm``, scored by the best
    **MSM** over its in-tolerance candidates (so the FDR target is the strongest-evidence
    annotation, not merely the closest-mass one). **Decoys** = the same formulas paired
    with chemically *implausible element adducts* (He, Be, Sc, …;
    :data:`masses.DECOY_ELEMENT_MASSES`), a **random sample** of ``n_decoy`` of them
    (seeded, so the result is reproducible) — a null whose matches are pure mass
    coincidence. Every annotation gets an **MSM score**; without a dataset that is mass
    accuracy alone, and with ``ds`` it is ``mass × spectral-isotope × spatial-isotope``
    (the image evidence that separates real ions from coincidences).

    Each sampled decoy element yields **one** best decoy match per peak — an independent
    null the same size as the target set. The FDR at a score threshold ``t`` is the
    **median over the decoy samples** of

        FDR_e(t) = ( #decoys_e≥t + 1 ) / ( #targets≥t + 1 )

    (the ``+1`` is the rule of succession, avoiding a spurious 0%; the median over
    samples is METASPACE's estimator and is far more stable than pooling every
    element-decoy into one ranking, which over-counts a single peak that coincides with
    many decoy elements), made monotonic in score. Each peak's **q-value** is the FDR at
    its target score; ``levels`` reports how many IDs pass at 5 / 10 / 20 / 50 % FDR.

    Returns ``{n_peaks, n_target, target_rate, decoy_rate, fdr, reliable, q_values,
    levels, image_based, n_decoy}``. ``q_values`` is aligned to ``peak_mzs`` (NaN where
    a peak has no target match). ``fdr`` is the global FDR (accept-all threshold).

    References: target–decoy FDR — Elias & Gygi (2007), Nat. Methods 4(3):207–214,
    doi:10.1038/nmeth1019; MSM scoring & implausible-adduct decoys — Palmer et al.
    (2017), Nat. Methods 14(1):57–60, doi:10.1038/nmeth.4072.
    """
    from .lipiddb import build_database
    from .masses import DECOY_ELEMENT_MASSES

    peak_mzs = np.asarray(peak_mzs, float)
    n_peaks = int(len(peak_mzs))
    empty = {"n_peaks": n_peaks, "n_target": 0, "target_rate": 0.0, "decoy_rate": 0.0,
             "fdr": 1.0, "reliable": False, "q_values": [float("nan")] * n_peaks,
             "levels": {lv: 0 for lv in FDR_LEVELS}, "image_based": ds is not None,
             "n_decoy": int(n_decoy)}
    if n_peaks == 0:
        return empty
    base = db if db is not None else build_database()
    ann = Annotator(mode=mode, ppm_tol=ppm, db=base)
    cal_factor = 1.0 / (1.0 + offset_ppm * 1e-6) if offset_ppm else 1.0   # recalibrated match m/z

    def score_pattern(formula, adduct, peak_mz, decoy_base=None):
        if ds is None or not formula:
            return float("nan"), float("nan")
        if decoy_base is not None:
            # Decoys use a fabricated pattern at shifted positions — must NOT share
            # the target cache key, so leave them uncached.
            pat = _decoy_pattern(formula, decoy_base)
            sc = isotopes.isotope_scores(ds, pat, tol_ppm=image_ppm, norm=norm)
        else:
            pat = isotopes.theoretical_isotope_pattern(formula, adduct, n_peaks=3)
            sc = isotopes.isotope_scores(
                ds, pat, tol_ppm=image_ppm, norm=norm, cache=iso_cache,
                key=isotopes._iso_key(formula, adduct, peak_mz, image_ppm, norm))
        return sc["spectral"], sc["spatial"]

    # ---- targets: best real candidate per peak, scored by the best MSM over its
    # in-tolerance candidates (not merely the closest-mass one) ----
    target_scores, q_index = [], {}
    for j, mz in enumerate(peak_mzs):
        mzc = float(mz) * cal_factor
        cands = ann.annotate_mz(mzc)
        if not cands:
            continue
        s = None
        for c in cands[:RERANK_TOPK]:
            spec, spat = score_pattern(c.lipid.formula, c.adduct, mzc)
            msm = _msm(_mass_score(c.ppm, ppm), spec, spat)
            if s is None or msm > s:
                s = msm
        target_scores.append(s)
        q_index[j] = s
    n_target = len(target_scores)

    # ---- decoys: real formulas + implausible element adducts, a RANDOM (seeded) sample.
    # Each element is its own independent null: one best decoy match per peak, the same
    # size as the target set. (Pooling all elements into one ranking over-counts a peak
    # that happens to coincide with many decoy elements — audit finding #3.)
    syms = sorted(DECOY_ELEMENT_MASSES)
    rng = random.Random(_DECOY_SEED)
    chosen = rng.sample(syms, min(max(1, n_decoy), len(syms)))
    elements = [(s, DECOY_ELEMENT_MASSES[s]) for s in sorted(chosen)]
    n_decoy_used = len(elements)
    # Sort the base lipids by neutral mass ONCE. Each decoy element adds a constant
    # offset `em` to every neutral mass, which preserves the ascending order, so the
    # per-element decoy m/z list is just `base_mz + em` — no re-sort per element.
    base_sorted = sorted(((l.neutral_mass, l.formula) for l in base), key=lambda t: t[0])
    base_mz = np.array([t[0] for t in base_sorted], dtype=float)
    base_formula = [t[1] for t in base_sorted]
    decoy_per_element = []        # one ascending-sorted score array per sampled element
    decoy_hits = 0
    for _sym, em in elements:
        dmz = (base_mz + em).tolist()
        scores_e = []
        for mz in peak_mzs:
            mz = float(mz) * cal_factor              # recalibrated match m/z (same as targets)
            win = mz * ppm / 1e6
            lo = bisect.bisect_left(dmz, mz - win)
            hi = bisect.bisect_right(dmz, mz + win)
            if hi <= lo:
                continue
            # best (closest) decoy match for this peak under this element
            bestk = min(range(lo, hi), key=lambda k: abs(dmz[k] - mz))
            spec, spat = score_pattern(base_formula[bestk], None, mz, decoy_base=dmz[bestk])
            scores_e.append(_msm(_mass_score((mz - dmz[bestk]) / dmz[bestk] * 1e6, ppm),
                                 spec, spat))
        decoy_hits += len(scores_e)
        decoy_per_element.append(np.sort(np.asarray(scores_e, float)))

    target_rate = n_target / n_peaks
    decoy_rate = decoy_hits / (n_decoy_used * n_peaks)

    # ---- per-sample rule-of-succession FDR, then the MEDIAN over decoy samples ----
    tgt_asc = np.sort(np.asarray(target_scores, float))     # ascending

    def fdr_at(t):
        n_t = n_target - int(np.searchsorted(tgt_asc, t, side="left"))   # #targets >= t
        per = [(len(de) - int(np.searchsorted(de, t, side="left")) + 1.0) / (n_t + 1.0)
               for de in decoy_per_element]                              # FDR_e(t)
        return min(1.0, float(np.median(per)))

    # raw FDR at each distinct target score, then make monotonic (lower score -> higher FDR)
    q_values = [float("nan")] * n_peaks
    levels = {lv: 0 for lv in FDR_LEVELS}
    if n_target:
        # q-value is non-increasing in score: q(s) = min over all thresholds t <= s of FDR(t).
        # Walk worst-first (ascending score) taking a running minimum so the best-scoring IDs
        # inherit the smallest achievable FDR. (Walking best-first with a running *max* — the
        # previous code — pinned every ID to the inflated top-of-ranking FDR; see Storey 2003,
        # Kall et al. 2008.)
        order = sorted(q_index.items(), key=lambda kv: kv[1])  # worst (lowest score) first
        running = 1.0
        qmap = {}
        for j, s in order:
            running = min(running, fdr_at(s))   # monotone non-increasing in score
            qmap[j] = running
        for j, q in qmap.items():
            q_values[j] = q
        for lv in FDR_LEVELS:
            levels[lv] = int(sum(1 for q in qmap.values() if q <= lv))
    fdr = float(min(1.0, np.median([(len(de) + 1.0) / (n_target + 1.0)
                                    for de in decoy_per_element]))) if n_target else 1.0
    reliable = bool(n_peaks >= 10 and n_target >= 5)
    return {"n_peaks": n_peaks, "n_target": n_target, "target_rate": target_rate,
            "decoy_rate": decoy_rate, "fdr": fdr, "reliable": reliable,
            "q_values": q_values, "levels": levels, "image_based": ds is not None,
            "n_decoy": n_decoy_used}


MSI_LEVELS = {
    1: "confirmed (MS/MS match or authentic standard)",
    2: "probable (isotope + spatial co-localization match)",
    3: "tentative candidate / compound class",
    4: "molecular formula only",
    5: "exact mass of interest (unknown)",
}


def msi_confidence_level(annotated: bool, has_msms: bool = False,
                         isotope_ok: bool = False, spatial_ok: bool = False,
                         ambiguous: bool = False, formula_only: bool = False) -> int:
    """Metabolomics-Standards-Initiative identification confidence level (1 = best …
    5 = unknown), per Schymanski et al. (2014), from the evidence the pipeline has.

    MS/MS or an authentic standard → **1**; a clean isotope-pattern + spatial-coherence
    match → **2**; an isobaric/ambiguous or class-only match → **3**; a bare formula with
    no structure → **4**; no database match at all (a genuine unknown ion) → **5**. Map
    every reported annotation to a level so a reviewer sees the ID confidence explicitly
    (the standard expectation for MSI annotation; SCiLS does not natively report it).

    Reference: Schymanski, E.L. et al. (2014), Environ. Sci. Technol. 48(4):2097–2098,
    doi:10.1021/es5002105."""
    if not annotated:
        return 5
    if has_msms:
        return 1
    if formula_only:
        return 4
    if ambiguous:
        return 3
    if isotope_ok and spatial_ok:
        return 2
    return 3


def attach_fdr(ds, peaks, mode: str = "negative", ppm: float = 5.0,
               image_ppm: float = 50.0, norm: str = "tic", n_decoy: int = 20,
               q_max: float | None = None, keep_unannotated: bool = True, db=None,
               recalibrate: bool = True):
    """Attach a target–decoy q-value to each feature so a feature list can be reported
    and thresholded by **FDR** instead of by an intensity cutoff — the publication-grade
    way to size a feature list (METASPACE-style; :func:`estimate_fdr`).

    Writes ``q_value`` (float, NaN where the peak has no database match) and ``fdr_pass``
    (bool) onto each peak dict. With ``q_max`` set, returns only features that pass —
    *but* unannotated features (no lipid-DB match → NaN q) are kept when
    ``keep_unannotated`` so a genuine **unknown** ion (e.g. a novel biomarker that isn't
    in the lipid database) is never silently dropped by an annotation filter. Returns
    ``(peaks_out, summary)`` where ``summary`` carries the 5/10/20/50 % FDR ID counts,
    the global FDR, the reliability flag, and the applied ``mass_offset_ppm`` /
    ``n_calibration_matches``.

    ``recalibrate`` (default on) measures the list's systematic m/z offset
    (:func:`estimate_mass_offset`, the *same gated* self-calibration
    :func:`build_feature_list` applies to the identities) and scores the FDR at the
    corrected m/z — so on a dataset that sits a few ppm off the database the q-values are
    computed against the same recentred masses the feature list annotated. Without it, a
    real offset leaves every target off-centre, which erodes its mass-accuracy score,
    collapses the target/decoy separation, and makes the gate spuriously strict (dropping
    real features at a given ``q_max``). Gated identically → 0.0 for a well-calibrated
    list, so a good list is matched unchanged.

    This makes "the feature list" a statistically defined object (q-thresholded) rather
    than a slider position. Refs as in :func:`estimate_fdr` (Elias & Gygi 2007; Palmer
    et al. 2017)."""
    peaks = [dict(p) for p in peaks]
    peak_mzs = [float(p["mz"]) for p in peaks]
    offset_ppm, n_cal = (estimate_mass_offset(peak_mzs, mode=mode, db=db)
                         if (recalibrate and peak_mzs) else (0.0, 0))
    res = estimate_fdr(peak_mzs, mode=mode, ppm=ppm, ds=ds, norm=norm,
                       image_ppm=image_ppm, n_decoy=n_decoy, db=db, offset_ppm=offset_ppm)
    out = []
    for p, q in zip(peaks, res["q_values"]):
        annotated = (q == q)                      # q == q is False for NaN
        p["q_value"] = float(q) if annotated else float("nan")
        p["fdr_pass"] = bool(annotated and (q_max is None or q <= q_max))
        if q_max is None or p["fdr_pass"] or (not annotated and keep_unannotated):
            out.append(p)
    summary = {"levels": res["levels"], "fdr": res["fdr"], "reliable": res["reliable"],
               "n_target": res["n_target"], "n_peaks": res["n_peaks"],
               "image_based": res["image_based"], "q_max": q_max, "n_kept": len(out),
               "n_unannotated_kept": sum(1 for p in out if p["q_value"] != p["q_value"]),
               "mass_offset_ppm": float(offset_ppm), "n_calibration_matches": int(n_cal)}
    return out, summary
