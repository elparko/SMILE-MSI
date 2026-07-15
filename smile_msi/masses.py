"""Monoisotopic element masses, formula math, and MS adduct ion m/z.

Bare-bones reimplementation: everything needed to go from an elemental
formula (as a dict) to a neutral monoisotopic mass, and from a neutral mass
to an observed ion m/z for a given adduct.
"""
from __future__ import annotations

# Monoisotopic atomic masses (u). Refs: AME2020 — Wang et al. (2021), Chin. Phys. C
# 45(3):030003, doi:10.1088/1674-1137/abddaf; IUPAC atomic weights — Meija et al. (2016),
# Pure Appl. Chem. 88(3):265–291, doi:10.1515/pac-2015-0305.
ELEMENTS = {
    "H": 1.0078250319,
    "C": 12.0,
    "N": 14.0030740052,
    "O": 15.9949146221,
    "P": 30.97376151,
    "S": 31.97207069,
    "Na": 22.98976928,
    "K": 38.96370649,
    "Cl": 34.96885268,
}
ELECTRON = 0.00054857990907


def formula_mass(formula: dict[str, int]) -> float:
    """Neutral monoisotopic mass of an elemental composition like {'C':42,'H':82,...}.

    Only the elements in :data:`ELEMENTS` are supported; an unsupported symbol raises a
    clear ``ValueError`` naming it (callers that ingest external databases — e.g.
    ``lipiddb.load_external_db`` — pre-filter such rows, so this only fires on a genuinely
    malformed formula rather than the bare ``KeyError`` it used to surface).
    """
    unknown = [el for el in formula if el not in ELEMENTS]
    if unknown:
        raise ValueError(
            f"formula_mass: unsupported element(s) {unknown}; "
            f"supported = {sorted(ELEMENTS)}"
        )
    return sum(ELEMENTS[el] * n for el, n in formula.items())


# Adduct mass shift = total mass added to the neutral (already electron-corrected:
# an anion gains an electron mass per charge, a cation loses one).  For a multiply-
# charged ion the observed m/z is (neutral + shift) / |charge| (see ``ion_mz``); for
# the singly-charged ions the shift is exactly the historical additive delta, so a
# direct ``ADDUCTS[name]`` read is still the m/z offset for those.
ADDUCTS = {
    # ---- negative mode ----
    "[M-H]-":      -ELEMENTS["H"] + ELECTRON,                       # -1.0072765
    "[M-2H]2-":    2 * (-ELEMENTS["H"] + ELECTRON),                 # multiply-charged, z=2
    "[M-3H]3-":    3 * (-ELEMENTS["H"] + ELECTRON),                 # multiply-charged, z=3
    "[M-CH3]-":    -(ELEMENTS["C"] + 3 * ELEMENTS["H"]) + ELECTRON, # PC/SM demethylation, -15.0229
    "[M+Cl]-":      ELEMENTS["Cl"] + ELECTRON,                      # +34.9694
    "[M+HCOO]-":    ELEMENTS["C"] + ELEMENTS["H"] + 2 * ELEMENTS["O"] + ELECTRON,        # formate +44.9982
    "[M+CH3COO]-":  2 * ELEMENTS["C"] + 3 * ELEMENTS["H"] + 2 * ELEMENTS["O"] + ELECTRON, # acetate +59.0139
    # ---- positive mode ----
    "[M+H]+":       ELEMENTS["H"] - ELECTRON,
    "[M+2H]2+":    2 * (ELEMENTS["H"] - ELECTRON),                  # multiply-charged, z=2
    "[M+NH4]+":     ELEMENTS["N"] + 4 * ELEMENTS["H"] - ELECTRON,   # ammonium +18.0338 (neutral lipids)
    "[M+Na]+":      ELEMENTS["Na"] - ELECTRON,
    "[M+K]+":       ELEMENTS["K"] - ELECTRON,
}

# Absolute charge |z| of each adduct (used to divide the mass shift into m/z).
ADDUCT_CHARGE = {
    "[M-H]-": 1, "[M-2H]2-": 2, "[M-3H]3-": 3, "[M-CH3]-": 1,
    "[M+Cl]-": 1, "[M+HCOO]-": 1, "[M+CH3COO]-": 1,
    "[M+H]+": 1, "[M+2H]2+": 2, "[M+NH4]+": 1, "[M+Na]+": 1, "[M+K]+": 1,
}

NEG_ADDUCTS = ["[M-H]-", "[M-2H]2-", "[M-3H]3-", "[M-CH3]-",
               "[M+Cl]-", "[M+HCOO]-", "[M+CH3COO]-"]
POS_ADDUCTS = ["[M+H]+", "[M+2H]2+", "[M+NH4]+", "[M+Na]+", "[M+K]+"]


def adduct_charge(adduct: str) -> int:
    """Absolute charge |z| of an adduct (defaults to 1 for unknown names)."""
    return ADDUCT_CHARGE.get(adduct, 1)


def ion_mz(neutral_mass: float, adduct: str) -> float:
    """Observed m/z of a neutral mass under the given adduct (charge-aware)."""
    return (neutral_mass + ADDUCTS[adduct]) / ADDUCT_CHARGE[adduct]


def ppm_error(observed: float, theoretical: float) -> float:
    """Signed mass error in ppm: (obs - theo)/theo * 1e6.

    A non-positive ``theoretical`` (degenerate input — a real ion m/z is always > 0)
    returns ``nan`` rather than raising ``ZeroDivisionError`` or producing a spurious
    finite value, so a bad row propagates as missing instead of crashing the caller.
    """
    if theoretical == 0:
        return float("nan")
    return (observed - theoretical) / theoretical * 1e6


# ---------------------------------------------------------------------------
# Isotope patterns
# ---------------------------------------------------------------------------
# Natural terrestrial isotope abundances and exact masses. Source: CIAAW /
# IUPAC — Meija et al. (2016), Pure Appl. Chem. 88(3):265–291. Each element maps
# to [(exact_mass, abundance), ...] with the monoisotopic (most abundant, lightest)
# peak first. This replaces the old "carbon ≈ m/z/14" proxy with a real
# composition-derived isotope envelope (the input every isotope score needs).
ISOTOPES = {
    "H":  [(1.0078250319, 0.999885), (2.0141017779, 0.000115)],
    "C":  [(12.0, 0.9893), (13.0033548378, 0.0107)],
    "N":  [(14.0030740052, 0.996205), (15.0001088989, 0.003795)],
    "O":  [(15.9949146221, 0.99757), (16.99913170, 0.00038), (17.99915961, 0.00205)],
    "P":  [(30.97376151, 1.0)],
    "S":  [(31.97207069, 0.9499), (32.97145850, 0.0075),
           (33.96786683, 0.0425), (35.96708088, 0.0001)],
    "Na": [(22.98976928, 1.0)],
    "K":  [(38.96370649, 0.932581), (39.96399817, 0.000117), (40.96182526, 0.067302)],
    "Cl": [(34.96885268, 0.7576), (36.96590258, 0.2424)],
}

# Atom-count deltas each adduct applies to the neutral formula, so the ion's
# isotope envelope reflects the *ion* composition — this matters for [M+Cl]-
# (³⁷Cl is 24%) and [M+K]+ (⁴¹K is 6.7%), which visibly shift the M+2/M+1 ratio.
ADDUCT_FORMULA_DELTA = {
    "[M-H]-":      {"H": -1},
    "[M-2H]2-":    {"H": -2},
    "[M-3H]3-":    {"H": -3},
    "[M-CH3]-":    {"C": -1, "H": -3},
    "[M+Cl]-":     {"Cl": 1},
    "[M+HCOO]-":   {"C": 1, "H": 1, "O": 2},
    "[M+CH3COO]-": {"C": 2, "H": 3, "O": 2},
    "[M+H]+":      {"H": 1},
    "[M+2H]2+":    {"H": 2},
    "[M+NH4]+":    {"N": 1, "H": 4},
    "[M+Na]+":     {"Na": 1},
    "[M+K]+":      {"K": 1},
}


# Chemically implausible "element adducts" for target–decoy FDR. METASPACE forms
# decoys by pairing each real formula with an element that would never realistically
# adduct (He, Be, transition metals, …), giving a null whose mass density matches the
# real search space but whose matches are pure coincidence. Monoisotopic masses (u).
DECOY_ELEMENT_MASSES = {
    "He": 4.0026032, "Li": 7.0160034, "Be": 9.0121822, "B": 11.0093054,
    "F": 18.9984032, "Ne": 19.9924401, "Al": 26.9815385, "Si": 27.9769265,
    "Ar": 39.9623831, "Ca": 39.9625909, "Sc": 44.9559082, "Ti": 47.9479463,
    "V": 50.9439569, "Cr": 51.9405062, "Mn": 54.9380439, "Fe": 55.9349363,
    "Co": 58.9331943, "Ni": 57.9353424, "Cu": 62.9295977, "Zn": 63.9291422,
    "Ga": 68.9255735, "Ge": 73.9211778, "As": 74.9215946, "Se": 79.9165218,
    "Br": 78.9183376, "Rb": 84.9117897, "Sr": 87.9056125, "Y": 88.9058403,
    "Zr": 89.9046977, "Mo": 97.9054069, "Ag": 106.9050916, "Cd": 113.9033585,
    "Sn": 119.9021947, "Sb": 120.9038157, "I": 126.9044719, "Cs": 132.9054520,
    "Ba": 137.9052472, "Au": 196.9665688,
}


def _combine(a: dict, b: dict, max_offset: int) -> dict:
    """Convolve two nucleon-offset distributions, each ``offset -> [prob, mean_mass]``.

    Masses add (it is a molecule); the combined mean mass at an offset is the
    probability-weighted average over every (oa, ob) pair that reaches it. The
    result is renormalized to mean masses so it can feed straight into the next
    convolution (this normalization is what makes the spacing correct)."""
    out: dict[int, list] = {}
    for oa, (pa, ma) in a.items():
        for ob, (pb, mb) in b.items():
            o = oa + ob
            if o > max_offset:
                continue
            p = pa * pb
            if p <= 0:
                continue
            e = out.setdefault(o, [0.0, 0.0])
            e[0] += p
            e[1] += p * (ma + mb)
    return {o: [p, mw / p] for o, (p, mw) in out.items() if p > 0}


def _single_atom(el: str, max_offset: int) -> dict:
    iso = ISOTOPES[el]
    mono = iso[0][0]
    d: dict[int, list] = {}
    for mass, ab in iso:
        o = int(round(mass - mono))
        if o > max_offset:
            continue
        e = d.setdefault(o, [0.0, 0.0])
        e[0] += ab
        e[1] += ab * mass
    return {o: [p, mw / p] for o, (p, mw) in d.items() if p > 0}


def _atom_power(el: str, n: int, max_offset: int) -> dict:
    """Distribution for ``n`` atoms of one element, via exponentiation by squaring."""
    result = {0: [1.0, 0.0]}
    base = _single_atom(el, max_offset)
    while n > 0:
        if n & 1:
            result = _combine(result, base, max_offset)
        n >>= 1
        if n:
            base = _combine(base, base, max_offset)
    return result


def isotope_distribution(formula: dict[str, int], max_offset: int = 4) -> list[tuple[float, float]]:
    """Isotopologue envelope of a neutral formula, grouped by nucleon offset.

    Returns ``[(mean_mass, abundance), ...]`` for offsets 0..``max_offset`` (M, M+1,
    M+2, …), abundances summing to ~1. Computed by convolving each element's
    isotope distribution (no averagine approximation) — exact for the supported
    elements (H, C, N, O, P, S, Na, K, Cl).
    """
    dist = {0: [1.0, 0.0]}
    for el, n in formula.items():
        if n <= 0 or el not in ISOTOPES:
            continue
        dist = _combine(dist, _atom_power(el, n, max_offset), max_offset)
    out = []
    for o in range(max_offset + 1):
        if o in dist and dist[o][0] > 0:
            p, mean_mass = dist[o]
            out.append((mean_mass, p))
        else:
            out.append((float("nan"), 0.0))
    return out


def _apply_adduct(formula: dict[str, int], adduct: str) -> dict[str, int]:
    f = dict(formula)
    for el, dn in ADDUCT_FORMULA_DELTA.get(adduct, {}).items():
        f[el] = f.get(el, 0) + dn
    return {el: n for el, n in f.items() if n > 0}


def ion_isotope_pattern(formula: dict[str, int], adduct: str,
                        n_peaks: int = 3) -> list[tuple[float, float]]:
    """Theoretical isotope pattern of an ion: ``[(m/z, rel_intensity), ...]``.

    The monoisotopic m/z is anchored to :func:`ion_mz` (electron-corrected and
    test-validated); higher isotopologue m/z come from the *ion* formula's
    envelope spacing divided by charge, so ³⁷Cl / ⁴¹K satellites are correct.
    Intensities are normalized to the base peak (max = 1.0). ``n_peaks`` counts
    M, M+1, M+2, …
    """
    if not formula:
        return [(ion_mz(0.0, adduct), 1.0)]
    z = ADDUCT_CHARGE.get(adduct, 1)
    ion_formula = _apply_adduct(formula, adduct)
    dist = isotope_distribution(ion_formula, max_offset=n_peaks - 1)
    base_mass = dist[0][0]
    mz0 = ion_mz(formula_mass(formula), adduct)
    pmax = max((p for _, p in dist if p > 0), default=1.0) or 1.0
    peaks = []
    for k, (m, p) in enumerate(dist):
        if p <= 0 or m != m:                      # skip empty / NaN offsets
            continue
        peaks.append((mz0 + (m - base_mass) / z, p / pmax))
    return peaks
