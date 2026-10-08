"""Isotope-pattern and adduct corroboration — annotation-confidence checks used
by deisotoping/annotation tools (METASPACE-style).

An m/z match on mass alone is weak. Two cheap, orthogonal signals raise (or lower)
confidence in a candidate:

* **isotope consistency** — a real ion shows an M+1 (and often M+2) peak; its
  M+1/M0 ratio should be ~1.1% per carbon. :func:`isotope_consistency` checks the
  data for that pattern.
* **adduct corroboration** — the same neutral lipid often appears under more than
  one adduct (e.g. [M-H]- and [M+Cl]-). :func:`corroborating_adducts` looks for the
  sibling ions in the picked peak list.

:func:`confidence` folds these together with the ppm error into a label.
"""
from __future__ import annotations

import numpy as np

from .constants import DEFAULT_TOL_PPM

from .masses import ion_mz, ion_isotope_pattern, NEG_ADDUCTS, POS_ADDUCTS

DELTA_C13 = 1.0033548   # 13C - 12C mass difference


def theoretical_isotope_pattern(formula: dict, adduct: str, n_peaks: int = 3):
    """Theoretical ion isotope pattern ``[(m/z, rel_intensity), ...]`` (max=1.0).

    Thin wrapper over :func:`masses.ion_isotope_pattern` — the real,
    composition-derived envelope used by every isotope score below."""
    return ion_isotope_pattern(formula, adduct, n_peaks=n_peaks)


def _spectral_isotope_score(theo, meas) -> float:
    """METASPACE spectral isotope score: ``1 - mean|L2(theory) - L2(measured)|``."""
    theo = np.asarray(theo, float)
    meas = np.asarray(meas, float)
    if len(theo) < 2 or meas[0] <= 0:
        return float("nan")
    tn = np.linalg.norm(theo)
    mn = np.linalg.norm(meas)
    if tn <= 0 or mn <= 0:
        return 0.0
    return float(max(0.0, 1.0 - np.mean(np.abs(theo / tn - meas / mn))))


def _spatial_isotope_score(images, theo) -> float:
    """METASPACE spatial isotope score: theory-weighted Pearson of each M+k image
    against the monoisotopic image.

    The per-image correlations are kept **signed** and only the final theory-weighted
    average is clipped to [0, 1] (Palmer et al. 2017 / pyImagingMSpec). Clipping each
    correlation to 0 first would treat an anti-correlated satellite (r ≈ −1, strong
    evidence *against* the candidate) the same as an uninformative one (r ≈ 0), biasing
    poor candidates upward."""
    if len(images) < 2:
        return float("nan")
    base = np.asarray(images[0], float)
    if base.std() == 0:
        return 0.0
    cors, weights = [], []
    for k in range(1, len(images)):
        img = np.asarray(images[k], float)
        if img.std() == 0:
            c = 0.0
        else:
            c = np.corrcoef(base, img)[0, 1]
            c = 0.0 if np.isnan(c) else float(c)
        cors.append(c)
        weights.append(max(float(theo[k]), 0.0))
    w = np.asarray(weights, float)
    if w.sum() <= 0:
        return 0.0
    return float(np.clip(np.average(cors, weights=w), 0.0, 1.0))


def _iso_key(formula, adduct, mz, tol_ppm, norm):
    """Hashable cache key for an :func:`isotope_scores` result. The theoretical
    pattern is a pure function of ``(formula, adduct)``, and the pulled images of
    ``(mz, tol_ppm, norm)`` on a fixed ``ds`` — so this tuple keys the score safely.
    Built by callers that score the *same* monoisotopic peak twice per feature-list
    build (audit plan 23, item B)."""
    fk = tuple(sorted(formula.items())) if formula else None
    return (fk, adduct, round(float(mz), 4), round(float(tol_ppm), 3), norm)


def isotope_scores(ds, pattern, tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic",
                   *, cache=None, key=None) -> dict:
    """METASPACE-style isotope evidence for a theoretical ``pattern`` = ``[(mz, rel), ...]``.

    Pulls the M, M+1, M+2 ion images from the data and returns
    ``{spectral, spatial, m1_ratio, m1_expected, n_iso, consistent}``:

    * **spectral** — ``1 - mean|L2(theory) - L2(measured)|`` over the isotopologues
      (does the *abundance* pattern match).
    * **spatial** — theory-weighted Pearson of each M+k image vs the M0 image
      (do the isotopologues *co-localize*, as a real ion's must).
    * **consistent** — spectral ≥ 0.85 and (spatial ≥ 0.3 or only one peak).

    ``cache``/``key`` are an optional, opt-in memo (a plain dict + an :func:`_iso_key`
    tuple): when both are given and the key is present, the stored result dict is
    returned without re-pulling the isotopologue ion images. ``cache=None`` (default)
    reproduces the uncached behaviour exactly, so every standalone caller is untouched.
    """
    if cache is not None and key is not None and key in cache:
        return cache[key]
    empty = {"spectral": float("nan"), "spatial": float("nan"), "m1_ratio": 0.0,
             "m1_expected": 0.0, "n_iso": 0, "consistent": False}
    if not pattern:
        return empty
    theo = [p[1] for p in pattern]
    images = [np.asarray(ds.ion_vector(mz, tol_ppm=tol_ppm, norm=norm), float)
              for mz, _ in pattern]
    meas = np.array([img.sum() for img in images])
    spec = _spectral_isotope_score(theo, meas)
    spat = _spatial_isotope_score(images, theo)
    m1_ratio = float(meas[1] / meas[0]) if len(meas) > 1 and meas[0] > 0 else 0.0
    m1_expected = float(theo[1] / theo[0]) if len(theo) > 1 and theo[0] > 0 else 0.0
    consistent = bool((spec == spec and spec >= 0.85)
                      and (len(pattern) < 2 or (spat == spat and spat >= 0.3)))
    result = {"spectral": spec, "spatial": spat, "m1_ratio": m1_ratio,
              "m1_expected": m1_expected, "n_iso": len(pattern), "consistent": consistent}
    if cache is not None and key is not None:
        cache[key] = result
    return result


def isotope_consistency(ds, mz: float, charge: int = 1, tol_ppm: float = 100.0,
                        norm: str = "tic", formula: dict | None = None,
                        adduct: str | None = None, *, cache=None) -> dict:
    """Check the M+1 / M+2 isotope pattern of ``mz`` in the dataset.

    With ``formula`` + ``adduct`` the expectation comes from the **real** isotope
    pattern (:func:`isotope_scores` — spectral + spatial); without them it falls
    back to the averagine **carbon proxy** (~1.1% M+1 per carbon, carbon ≈ m/z·z/14).
    Returns ``{m1_ratio, m2_ratio, n_carbon_est, expected_m1, consistent, spectral, spatial}``.

    Reference: averagine isotope model — Senko, Beu & McLafferty (1995), J. Am. Soc.
    Mass Spectrom. 6(4):229–233, doi:10.1016/1044-0305(95)00017-8.
    """
    if formula and adduct:
        pat = theoretical_isotope_pattern(formula, adduct, n_peaks=3)
        sc = isotope_scores(ds, pat, tol_ppm=tol_ppm, norm=norm, cache=cache,
                            key=_iso_key(formula, adduct, mz, tol_ppm, norm))
        return {"m1_ratio": sc["m1_ratio"], "m2_ratio": float("nan"),
                "n_carbon_est": int(formula.get("C", 0)), "expected_m1": sc["m1_expected"],
                "consistent": sc["consistent"], "spectral": sc["spectral"],
                "spatial": sc["spatial"]}
    # --- fallback: averagine carbon proxy (no candidate formula available) ---
    m0 = float(np.mean(ds.ion_vector(mz, tol_ppm=tol_ppm, norm=norm)))
    if m0 <= 0:
        return {"m1_ratio": 0.0, "m2_ratio": 0.0, "n_carbon_est": 0, "expected_m1": 0.0,
                "consistent": False, "spectral": float("nan"), "spatial": float("nan")}
    step = DELTA_C13 / charge
    m1 = float(np.mean(ds.ion_vector(mz + step, tol_ppm=tol_ppm, norm=norm)))
    m2 = float(np.mean(ds.ion_vector(mz + 2 * step, tol_ppm=tol_ppm, norm=norm)))
    m1_ratio, m2_ratio = m1 / m0, m2 / m0
    # carbon count tracks the *neutral* mass (~= m/z * charge), ~CH2 per 14 Da
    n_c = max(1, int(round(mz * charge / 14.0)))
    expected = 0.011 * n_c
    consistent = bool(m1_ratio > 0 and 0.5 * expected <= m1_ratio <= 2.5 * expected)
    return {"m1_ratio": m1_ratio, "m2_ratio": m2_ratio, "n_carbon_est": n_c,
            "expected_m1": expected, "consistent": consistent,
            "spectral": float("nan"), "spatial": float("nan")}


def corroborating_adducts(neutral_mass: float, mode: str, peak_mzs, tol_ppm: float = 20.0):
    """Which adduct ions of a neutral mass are present among ``peak_mzs``.

    Returns the list of adduct names found (e.g. ``['[M-H]-', '[M+Cl]-']``); two or
    more is strong corroboration that the assignment is real."""
    peak_mzs = np.asarray(peak_mzs, float)
    adducts = NEG_ADDUCTS if mode == "negative" else POS_ADDUCTS
    found = []
    for add in adducts:
        target = ion_mz(neutral_mass, add)
        win = target * tol_ppm / 1e6
        if np.any(np.abs(peak_mzs - target) <= win):
            found.append(add)
    return found


def deisotope(peaks, charge: int = 1, tol_ppm: float = 20.0, max_iso: int = 2):
    """Flag isotopologue (M+1/M+2) satellites in a peak list.

    A peak is a satellite if a more-intense peak sits ``n * 1.00336/charge`` Da below
    it (within ``tol_ppm``) for some n in 1..``max_iso`` — i.e. it is the heavier
    isotope of a monoisotopic peak. Collapsing these to the monoisotopic peak gives
    a cleaner feature list and a more honest annotation FDR.

    Flagging relies on intensity: a peak is a satellite only when a **strictly
    more intense** peak sits the 13C-step below it. Dicts (``mz``/``intensity``)
    work as intended; bare m/z floats are treated as equal intensity, so nothing is
    flagged (a spacing-only mode would flag on m/z spacing alone). Returns
    ``(monoisotopic_peaks, is_isotopologue_mask)``.
    """
    items = [(float(p["mz"]) if isinstance(p, dict) else float(p),
              float(p.get("intensity", 1.0)) if isinstance(p, dict) else 1.0) for p in peaks]
    mzs = np.array([m for m, _ in items])
    ints = np.array([v for _, v in items])
    step = DELTA_C13 / charge
    is_iso = np.zeros(len(items), dtype=bool)
    for i, (m, v) in enumerate(items):
        win = m * tol_ppm / 1e6
        for n in range(1, max_iso + 1):
            cand = np.where(np.abs(mzs - (m - n * step)) <= win)[0]
            if len(cand) and ints[cand].max() > v:
                is_iso[i] = True
                break
    mono = [peaks[i] for i in range(len(peaks)) if not is_iso[i]]
    return mono, is_iso


def confidence_detail(ppm: float, isotope_ok: bool, n_adducts: int,
                      ppm_tol: float = 5.0, score_gap: float | None = None,
                      spectral: float | None = None, spatial: float | None = None,
                      morans: float | None = None) -> dict:
    """Per-candidate confidence as a 0–100 score, a folded label, and the *why*.

    A transparent weighted blend of the orthogonal evidence the annotation carries —
    not a calibrated posterior, but an explainable triage number. Two modes:

    **Mass-only (default, ``spectral is None``)** — the original four-term blend:
    mass accuracy (50%), binary isotope ✓ (22%), adduct corroboration (16%),
    uniqueness vs runner-up (12%).

    **MSM-style (``spectral`` given)** — the METASPACE-shaped blend that folds in the
    *continuous* image evidence: mass accuracy (40%), spectral isotope score (22%),
    spatial evidence (16% = isotopologue co-localization + Moran's I structure),
    adduct corroboration (12%), uniqueness (10%).

    ``spectral``/``spatial`` are the [0,1] scores from :func:`isotope_scores`;
    ``morans`` is the feature's Moran's I (negatives clipped). Returns
    ``{"score": int 0-100, "label": "High"|"Medium"|"Low", "reasons": [...]}``. "High"
    needs ``isotope_ok``: a score that would reach it without one is labelled "Medium".

    This is the **multi-signal** Features-tab confidence — a different metric from the
    Excel report's "Mass-match confidence" (``pipeline.build_report``), which uses mass
    accuracy + adduct plausibility only, so the two can legitimately differ for one ion.
    """
    reasons = []
    tol = ppm_tol if ppm_tol and ppm_tol > 0 else 10.0
    mass = max(0.0, 1.0 - abs(ppm) / tol)
    reasons.append(f"{abs(ppm):.1f} ppm" + (" (tight)" if abs(ppm) <= 2
                   else " (near tol)" if abs(ppm) >= 0.7 * tol else ""))

    n_sib = max(0, n_adducts - 1)                         # n_adducts counts the ion itself
    add = min(1.0, n_sib / 2.0)
    add_reason = (f"{n_sib} sibling adduct{'s' if n_sib != 1 else ''}"
                  if n_sib else "no sibling adduct")

    if score_gap is None:
        uniq = 1.0
        uniq_reason = "only candidate"
    else:
        uniq = max(0.0, min(1.0, score_gap / 2.0))        # ~2 score units (≈2 ppm) = decisive
        uniq_reason = "clear best match" if score_gap >= 1.0 else "isobaric tie"

    if spectral is None:
        # --- mass-only blend (unchanged; keeps the bare-call contract stable) ---
        iso = 1.0 if isotope_ok else 0.0
        reasons.append("M+1 isotope ✓" if isotope_ok else "no M+1 support")
        reasons += [add_reason, uniq_reason]
        score01 = 0.50 * mass + 0.22 * iso + 0.16 * add + 0.12 * uniq
    else:
        # --- MSM-style blend with continuous image evidence ---
        iso = spectral if spectral == spectral else (1.0 if isotope_ok else 0.0)
        reasons.append(f"isotope fit {iso:.2f}" if spectral == spectral
                       else ("M+1 isotope ✓" if isotope_ok else "no M+1 support"))
        coloc = spatial if (spatial is not None and spatial == spatial) else 0.0
        struct = max(0.0, morans) if (morans is not None and morans == morans) else 0.0
        spat_term = 0.6 * coloc + 0.4 * struct
        if spatial is not None and spatial == spatial:
            reasons.append(f"isotope co-loc {coloc:.2f}")
        if morans is not None and morans == morans:
            reasons.append(f"Moran's I {morans:+.2f}")
        reasons += [add_reason, uniq_reason]
        score01 = 0.40 * mass + 0.22 * iso + 0.16 * spat_term + 0.12 * add + 0.10 * uniq

    pct = int(round(100 * score01))
    label = "High" if pct >= 70 else ("Medium" if pct >= 45 else "Low")
    if label == "High" and not isotope_ok:
        # mass accuracy and spatial structure alone can outscore a missing isotope pattern;
        # "High" claims the envelope was seen
        label = "Medium"
        reasons.append("capped at Medium: isotope check failed")
    return {"score": pct, "label": label, "reasons": reasons}


def confidence(ppm: float, isotope_ok: bool, n_adducts: int,
               ppm_tol: float = 5.0, score_gap: float | None = None) -> str:
    """Folded confidence label — see :func:`confidence_detail` for the full breakdown."""
    return confidence_detail(ppm, isotope_ok, n_adducts, ppm_tol, score_gap)["label"]
