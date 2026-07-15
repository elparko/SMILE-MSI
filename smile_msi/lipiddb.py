"""In-silico lipid database: generate neutral monoisotopic masses for the major
lipid classes by enumerating sum compositions (total acyl carbons : double bonds).

No external database file needed. Class formulas were derived from glycerol-3-
phosphate / sphingoid backbones and verified against reference species in
tests/test_masses.py.  Each chain adds CH2 (+1 C, +2 H) per carbon and removes
2 H per double bond, so a class is fully described by a base formula at C=D=0.

`tissue_prior` encodes how expected a class is in nerve / myelin (sulfatides,
galactosylceramides, plasmalogens, PE, PS, PC, SM dominate myelin).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from .masses import formula_mass, ELEMENTS


@dataclass(frozen=True)
class Lipid:
    name: str          # e.g. "PE 36:1"
    lipid_class: str   # e.g. "PE"
    neutral_mass: float
    tissue_prior: float  # 0..1, higher = more expected in nerve/myelin
    # Elemental composition {'C':.., 'H':.., ...}. Excluded from eq/hash so Lipid
    # stays hashable; carried so isotope patterns use the real formula, not a proxy.
    formula: dict = field(default_factory=dict, compare=False, hash=False)


# ----- glycerophospholipid class templates -----------------------------------
# base = formula at total carbons C and double bonds D (diacyl).
# H is written as (a + 2*C - 2*D); we store (a, fixed C-offset, other atoms).
# Each entry: class -> (C_offset, H_const, extra_atoms dict, n_chains)
#   total_C = C_offset + C ; total_H = H_const + 2*C - 2*D
GPL_TEMPLATES = {
    # diacyl (2 chains)
    "PA": dict(c=3, h=5,  atoms={"O": 8,  "P": 1}, chains=2, prior=0.4),
    "PC": dict(c=8, h=16, atoms={"N": 1, "O": 8,  "P": 1}, chains=2, prior=0.8),
    "PE": dict(c=5, h=10, atoms={"N": 1, "O": 8,  "P": 1}, chains=2, prior=0.9),
    "PS": dict(c=6, h=10, atoms={"N": 1, "O": 10, "P": 1}, chains=2, prior=0.85),
    "PG": dict(c=6, h=11, atoms={"O": 10, "P": 1},         chains=2, prior=0.4),
    "PI": dict(c=9, h=15, atoms={"O": 13, "P": 1},         chains=2, prior=0.6),
    # ether / plasmalogen variants: one ester -> ether (-O, +2H).
    "PE-O": dict(c=5, h=12, atoms={"N": 1, "O": 7, "P": 1}, chains=2, prior=0.9),
    "PC-O": dict(c=8, h=18, atoms={"N": 1, "O": 7, "P": 1}, chains=2, prior=0.7),
    # lyso (1 chain): one acyl removed, +H2O back (one fewer ester).
    "LPA": dict(c=3, h=7,  atoms={"O": 7, "P": 1},          chains=1, prior=0.2),
    "LPC": dict(c=8, h=18, atoms={"N": 1, "O": 7, "P": 1},  chains=1, prior=0.3),
    "LPE": dict(c=5, h=12, atoms={"N": 1, "O": 7, "P": 1},  chains=1, prior=0.35),
    "LPI": dict(c=9, h=17, atoms={"O": 12, "P": 1},         chains=1, prior=0.25),
    "LPS": dict(c=6, h=12, atoms={"N": 1, "O": 9, "P": 1},  chains=1, prior=0.3),
    "LPG": dict(c=6, h=13, atoms={"O": 9, "P": 1},          chains=1, prior=0.2),
}

# ----- neutral glycerolipids (no headgroup) -----------------------------------
# glycerol (C3H8O3) esterified with 1/2/3 acyl chains (each -H2O). Detected in
# positive mode as ammonium/sodium adducts, not negative. Same C=D=0 base scheme
# as the GPLs. Verified MG 18:1=356.2927, DG 36:2=620.5380, TG 52:2=858.7677.
GLYCEROLIPID_TEMPLATES = {
    "MG": dict(c=3, h=6, atoms={"O": 4}, chains=1, prior=0.2),
    "DG": dict(c=3, h=4, atoms={"O": 5}, chains=2, prior=0.25),
    "TG": dict(c=3, h=2, atoms={"O": 6}, chains=3, prior=0.3),
}

# ----- sphingolipid templates (sum comp C = sphingoid+acyl carbons) -----------
# dihydroxy ("d", ;O2) base; extra_O adds 2-hydroxy / trihydroxy variants.
SPH_TEMPLATES = {
    "Cer":      dict(c=0, h=1,  atoms={"N": 1, "O": 3},                prior=0.6),
    "SM":       dict(c=5, h=13, atoms={"N": 2, "O": 6, "P": 1},        prior=0.85),
    "HexCer":   dict(c=6, h=11, atoms={"N": 1, "O": 8},                prior=0.95),  # GalCer/GlcCer
    "Sulfatide":dict(c=6, h=11, atoms={"N": 1, "O": 11, "S": 1},       prior=1.0),   # SHexCer / ST
}


# ----- ganglioside templates (ceramide + sialylated glycan headgroup) ---------
# Built as HexCer + extra glycan residues (each monosaccharide minus a water):
#   Hex C6H10O5, HexNAc C8H13NO5, Neu5Ac C11H17NO8. C/h are the C=D=0 base, the
# sphingoid+acyl carbons are added by _sph. Verified vs GM3 d36:1=1180.744,
# GM1 d36:1=1545.876, GD1 d36:1=1836.972.
GANGLIO_TEMPLATES = {
    "GM3": dict(c=23, h=38, atoms={"N": 2, "O": 21}, prior=0.9),
    "GM2": dict(c=31, h=51, atoms={"N": 3, "O": 26}, prior=0.85),
    "GM1": dict(c=37, h=61, atoms={"N": 3, "O": 31}, prior=0.95),
    "GD3": dict(c=34, h=55, atoms={"N": 3, "O": 29}, prior=0.85),
    "GD1": dict(c=48, h=78, atoms={"N": 4, "O": 39}, prior=0.95),
    "GT1": dict(c=59, h=95, atoms={"N": 5, "O": 47}, prior=0.9),
}

# ----- sterols (fixed compounds) and cholesteryl esters -----------------------
STEROLS = [
    ("Cholesterol", {"C": 27, "H": 46, "O": 1}),
    ("Cholesterol sulfate", {"C": 27, "H": 46, "O": 4, "S": 1}),
    ("Desmosterol", {"C": 27, "H": 44, "O": 1}),
]


def _ce(C, D):
    # cholesteryl ester = cholesterol + acyl(C:D) - H2O. Verified CE 18:1=650.601.
    f = {"C": 27 + C, "H": 44 + 2 * C - 2 * D, "O": 2}
    return Lipid(f"CE {C}:{D}", "CE", formula_mass(f), 0.3, formula=f)


def _car(C, D):
    # acylcarnitine = carnitine(C7H15NO3) + acyl(C:D) - H2O. Verified CAR 16:0=399.335.
    f = {"C": 7 + C, "H": 13 + 2 * C - 2 * D, "N": 1, "O": 4}
    return Lipid(f"CAR {C}:{D}", "CAR", formula_mass(f), 0.4, formula=f)


def _oxfa(C, D, n_o):
    # oxidized fatty acid / oxylipin = FA(C:D) + n_o extra O (hydroxy/oxo). Its own
    # "OxFA" class so class-level images/markers don't lump oxylipins with plain FAs.
    f = {"C": C, "H": 2 * C - 2 * D, "O": 2 + n_o}
    return Lipid(f"FA {C}:{D};O{n_o}", "OxFA", formula_mass(f), 0.4, formula=f)


# Curated low-mass metabolites prominent in nervous tissue (neutral formulas).
METABOLITES = [
    ("N-acetylaspartate (NAA)", {"C": 6, "H": 9, "N": 1, "O": 5}),
    ("N-acetylaspartylglutamate (NAAG)", {"C": 11, "H": 16, "N": 2, "O": 8}),
    ("Glutamate", {"C": 5, "H": 9, "N": 1, "O": 4}),
    ("Glutamine", {"C": 5, "H": 10, "N": 2, "O": 3}),
    ("Aspartate", {"C": 4, "H": 7, "N": 1, "O": 4}),
    ("GABA", {"C": 4, "H": 9, "N": 1, "O": 2}),
    ("Taurine", {"C": 2, "H": 7, "N": 1, "O": 3, "S": 1}),
    ("Creatine", {"C": 4, "H": 9, "N": 3, "O": 2}),
    ("Phosphocreatine", {"C": 4, "H": 10, "N": 3, "O": 5, "P": 1}),
    ("Citrate", {"C": 6, "H": 8, "O": 7}),
    ("Succinate", {"C": 4, "H": 6, "O": 4}),
    ("Malate", {"C": 4, "H": 6, "O": 5}),
    ("Lactate", {"C": 3, "H": 6, "O": 3}),
    ("Ascorbate", {"C": 6, "H": 8, "O": 6}),
    ("myo-Inositol", {"C": 6, "H": 12, "O": 6}),
    ("Glutathione (GSH)", {"C": 10, "H": 17, "N": 3, "O": 6, "S": 1}),
    ("AMP", {"C": 10, "H": 14, "N": 5, "O": 7, "P": 1}),
]


def _gpl(name_class, tmpl, C, D):
    f = {"C": tmpl["c"] + C, "H": tmpl["h"] + 2 * C - 2 * D}
    for el, n in tmpl["atoms"].items():
        f[el] = f.get(el, 0) + n
    return Lipid(f"{name_class} {C}:{D}", name_class, formula_mass(f), tmpl["prior"], formula=f)


_OH_VARIANT_CLASSES = {"Cer", "HexCer", "Sulfatide"}


def _sph(name_class, tmpl, C, D, extra_o):
    atoms = dict(tmpl["atoms"])
    atoms["O"] = atoms["O"] + extra_o
    f = {"C": tmpl["c"] + C, "H": tmpl["h"] + 2 * C - 2 * D}
    for el, n in atoms.items():
        f[el] = f.get(el, 0) + n
    # Show ;O2 / ;O3 for classes where the hydroxylation state matters (myelin
    # 2-hydroxy sulfatides/HexCer); SM stays unsuffixed.
    suffix = f";O{2 + extra_o}" if name_class in _OH_VARIANT_CLASSES else ""
    return Lipid(f"{name_class} {C}:{D}{suffix}", name_class, formula_mass(f), tmpl["prior"],
                 formula=f)


def _fa(C, D):
    f = {"C": C, "H": 2 * C - 2 * D, "O": 2}
    return Lipid(f"FA {C}:{D}", "FA", formula_mass(f), 0.5, formula=f)


# Biological-plausibility bounds. Mass-only matching otherwise invents species
# like "PI 26:4" (two 13-carbon chains) that win on mass but don't exist.
PER_CHAIN_C = (14, 24)   # acyl carbons per chain for glycerophospholipids
PER_CHAIN_DB = 6         # max double bonds per chain (DHA 22:6 is the practical ceiling)
MAX_TOTAL_DB = 6         # cap on total double bonds for a whole glycerolipid (see below)

# Odd total-carbon glycerolipids/free-fatty-acids (an odd chain somewhere) and very highly
# unsaturated species are rare mass-coincidence hits that a small calibration offset lets win
# over the true ion. `even_chain_only` and `MAX_TOTAL_DB` drop them by default for the
# ester/acyl classes. NOTE: sphingolipids & gangliosides are deliberately NOT even-chain
# filtered — myelin genuinely contains odd-chain and 2-hydroxy sulfatides/HexCer — they use
# `_plausible_sph`, which already bounds C/D tightly.


def _plausible_gpl(chains, C, D, even_chain_only=True, max_total_db=MAX_TOTAL_DB):
    if not (PER_CHAIN_C[0] * chains <= C <= PER_CHAIN_C[1] * chains):
        return False
    if even_chain_only and C % 2:
        return False                                  # odd total carbons: an odd chain, rare
    return 0 <= D <= min(PER_CHAIN_DB * chains, max_total_db)


def _plausible_sph(C, D):
    # sphingoid base (~18C) + N-acyl (14-26C); 1 base double bond + a few in the acyl.
    # Intentionally not even-chain filtered (odd-chain 2-OH sulfatides are real in myelin).
    return 32 <= C <= 44 and 0 <= D <= 4


def _plausible_fa(C, D, even_chain_only=True):
    if even_chain_only and C % 2:
        return False
    return 12 <= C <= 26 and 0 <= D <= 6


# ----- external-DB plausibility (same idea as _plausible_gpl/_sph/_fa, generalized to an
# arbitrary imported (class, name) pair instead of a generation loop) -----------------------
_CHAIN_COMP = re.compile(r"(\d+):(\d+)")

# Chain count per class family, so an external import gets the same per-chain carbon/
# double-bond envelope as the in-silico builder without re-deriving it per class.
_EXT_CHAINS_BY_CLASS = {cls: t["chains"] for cls, t in GPL_TEMPLATES.items()}
_EXT_CHAINS_BY_CLASS.update({cls: t["chains"] for cls, t in GLYCEROLIPID_TEMPLATES.items()})
_EXT_CHAINS_BY_CLASS.update({"FA": 1, "CE": 1, "CAR": 1, "OxFA": 1})
_EXT_SPH_CLASSES = set(SPH_TEMPLATES) | set(GANGLIO_TEMPLATES)


def _parse_chain_comp(name: str):
    """Best-effort total ``(C, D)`` sum composition from a lipid name, by **summing every
    'C:D' token found** — handles both the sum-composition form used throughout this module
    (``'PC 34:1'`` -> one token, sums to itself) and sn-position/multi-chain notation
    (``'PC(16:0/18:1)'``, ``'Cer d18:1/16:0'`` -> two tokens, (16,0)+(18,1) = (34,1), the same
    total either naming convention represents). ``None`` when the name carries no ``C:D``
    token at all (a metabolite name, a bare LM_ID, ...), which the caller treats as "can't
    judge, don't filter"."""
    matches = _CHAIN_COMP.findall(str(name))
    if not matches:
        return None
    return sum(int(c) for c, _ in matches), sum(int(d) for _, d in matches)


def _plausible_external(name: str) -> bool:
    """Generic sum-composition sanity check for an externally-imported lipid (LMSD/CSV/SDF):
    same per-chain carbon/double-bond envelope as the in-silico enumeration
    (:func:`_plausible_gpl` / :func:`_plausible_sph`), so mass-coincidence species a small
    calibration offset would otherwise let win (contrived short/hyper-unsaturated chains)
    are dropped on import too, not just from the built-in generator.

    The class family is read from the name's own shorthand token (:func:`_class_from_name`,
    e.g. ``'PC'`` out of ``'PC 34:1'``), not the record's CATEGORY/MAIN_CLASS field — LMSD's
    own category fields are free text ("Glycerophospholipids") that won't match a chain-count
    model keyed by abbreviation, while the name token reliably does.

    Deliberately **not** even-chain filtered here (unlike the in-silico default): a curated
    external DB is exactly where genuine odd-chain species earn their keep, so only the
    carbon/double-bond range is enforced. Names that don't parse as ``C:D`` (metabolites,
    unusual naming) or classes this module has no chain-count model for pass through
    unfiltered — the goal is dropping obvious mass-coincidence junk, not narrowing coverage
    for a naming convention/class family we don't recognize."""
    comp = _parse_chain_comp(name)
    if comp is None:
        return True
    C, D = comp
    tok = _class_from_name(name)
    if tok in _EXT_SPH_CLASSES:
        return _plausible_sph(C, D)
    chains = _EXT_CHAINS_BY_CLASS.get(tok)
    if chains is None:
        return True
    return _plausible_gpl(chains, C, D, even_chain_only=False)


@lru_cache(maxsize=8)
def _build_database_cached(
    gpl_c=(28, 48), gpl_d=(0, 12),
    sph_c=(32, 44), sph_d=(0, 4),
    fa_c=(12, 26), fa_d=(0, 6),
    plausible_only=True, even_chain_only=True, max_total_db=MAX_TOTAL_DB,
) -> tuple:
    """Memoized enumeration core — returns an immutable ``tuple`` so the single shared
    cached object can't be mutated by a caller. The public :func:`build_database`
    wraps this in a fresh list per call."""
    lipids: list[Lipid] = []

    for name_class, tmpl in GPL_TEMPLATES.items():
        chains = tmpl["chains"]
        crange = fa_c if chains == 1 else gpl_c   # lyso = single chain
        for C in range(crange[0], crange[1] + 1):
            for D in range(gpl_d[0], gpl_d[1] + 1):
                if plausible_only and not _plausible_gpl(chains, C, D, even_chain_only,
                                                         max_total_db):
                    continue
                lipids.append(_gpl(name_class, tmpl, C, D))

    # neutral glycerolipids (MG/DG/TG): chain count drives the carbon envelope, which
    # is the per-chain plausibility range × chains — so bounds hold even when
    # plausible_only is off (these classes are never enumerated outside chemical reason).
    for name_class, tmpl in GLYCEROLIPID_TEMPLATES.items():
        chains = tmpl["chains"]
        for C in range(PER_CHAIN_C[0] * chains, PER_CHAIN_C[1] * chains + 1):
            for D in range(0, PER_CHAIN_DB * chains + 1):
                if plausible_only and not _plausible_gpl(chains, C, D, even_chain_only,
                                                         max_total_db):
                    continue
                lipids.append(_gpl(name_class, tmpl, C, D))

    for name_class, tmpl in SPH_TEMPLATES.items():
        for C in range(sph_c[0], sph_c[1] + 1):
            for D in range(sph_d[0], sph_d[1] + 1):
                if plausible_only and not _plausible_sph(C, D):   # not even-chain filtered
                    continue
                # ;O2 (normal d-base) and ;O3 (2-hydroxy, common in myelin sulfatides/HexCer)
                extras = (0, 1) if name_class in ("HexCer", "Sulfatide", "Cer") else (0,)
                for extra_o in extras:
                    lipids.append(_sph(name_class, tmpl, C, D, extra_o))

    for C in range(fa_c[0], fa_c[1] + 1):
        for D in range(fa_d[0], fa_d[1] + 1):
            if plausible_only and not _plausible_fa(C, D, even_chain_only):
                continue
            lipids.append(_fa(C, D))

    # gangliosides: ceramide (sphingoid+acyl) carbons enumerated like sphingolipids
    for name_class, tmpl in GANGLIO_TEMPLATES.items():
        for C in range(sph_c[0], sph_c[1] + 1):
            for D in range(sph_d[0], sph_d[1] + 1):
                if plausible_only and not _plausible_sph(C, D):   # not even-chain filtered
                    continue
                lipids.append(_sph(name_class, tmpl, C, D, 0))

    # cholesteryl esters (acyl chain enumerated like fatty acids)
    for C in range(fa_c[0], fa_c[1] + 1):
        for D in range(fa_d[0], fa_d[1] + 1):
            if plausible_only and not _plausible_fa(C, D, even_chain_only):
                continue
            lipids.append(_ce(C, D))

    # fixed sterols
    for name, f in STEROLS:
        lipids.append(Lipid(name, "ST", formula_mass(f), 0.5, formula=f))

    # acylcarnitines
    for C in range(fa_c[0], fa_c[1] + 1):
        for D in range(fa_d[0], fa_d[1] + 1):
            if plausible_only and not _plausible_fa(C, D, even_chain_only):
                continue
            lipids.append(_car(C, D))

    # oxidized fatty acids / oxylipins (mono/di-hydroxy or oxo)
    for C in range(fa_c[0], fa_c[1] + 1):
        for D in range(fa_d[0], fa_d[1] + 1):
            if plausible_only and not _plausible_fa(C, D, even_chain_only):
                continue
            for n_o in (1, 2):
                lipids.append(_oxfa(C, D, n_o))

    # curated low-mass metabolites
    for name, f in METABOLITES:
        lipids.append(Lipid(name, "Metab", formula_mass(f), 0.4, formula=f))

    return tuple(lipids)


def build_database(
    gpl_c=(28, 48), gpl_d=(0, 12),
    sph_c=(32, 44), sph_d=(0, 4),
    fa_c=(12, 26), fa_d=(0, 6),
    plausible_only=True, even_chain_only=True, max_total_db=MAX_TOTAL_DB,
) -> list[Lipid]:
    """Enumerate the in-silico lipid space. Ranges are (min, max) inclusive.

    ``plausible_only`` applies per-chain carbon/unsaturation bounds so mass-coincidence
    species (short, hyper-unsaturated formulas) are not generated. On top of that,
    ``even_chain_only`` drops odd total-carbon ester/acyl species (an odd chain somewhere —
    rare) and ``max_total_db`` caps a whole glycerolipid's double bonds; both remove the
    contrived hits (odd-chain PC, 8-double-bond ether lipids) that a small calibration
    offset would otherwise let win over the true ion. Set ``even_chain_only=False`` /
    raise ``max_total_db`` to widen the search for a tissue where odd/poly-unsaturated
    species matter. Sphingolipids/gangliosides are never even-chain filtered (myelin has
    genuine odd-chain / 2-OH sulfatides).

    Deterministic in its (hashable) args, so the heavy enumeration is memoized
    (:func:`_build_database_cached`) — repeated ``Annotator``/``estimate_fdr``
    construction reuses one build. Each call returns a **fresh list** copied from the
    shared immutable cache, so callers may sort/filter the result in place without
    corrupting other callers' view of the database.
    """
    return list(_build_database_cached(
        gpl_c, gpl_d, sph_c, sph_d, fa_c, fa_d, plausible_only,
        even_chain_only, max_total_db))


# --------------------------------------------------------------------------- #
# External databases (LIPID MAPS / SwissLipids / custom) — standardize against
# the community references instead of (or alongside) the in-silico enumeration.
# --------------------------------------------------------------------------- #
_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(s: str) -> dict:
    """Parse a Hill-notation formula string (e.g. ``'C42H82NO8P'``) into a dict.
    Unparseable input returns ``{}``."""
    out: dict[str, int] = {}
    for el, n in _FORMULA_TOKEN.findall(str(s).strip()):
        if el:
            out[el] = out.get(el, 0) + (int(n) if n else 1)
    return out


def _pick_col(columns, candidates):
    low = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


def _class_from_name(name: str) -> str:
    """Best-effort lipid class from a name like ``'PC 34:1'`` -> ``'PC'``."""
    tok = str(name).strip().split()
    return tok[0] if tok else ""


def load_external_db(path, name_col=None, formula_col=None, class_col=None,
                     tissue_prior: float = 0.5, sep=None, plausible_only: bool = True) -> list:
    """Load a lipid/metabolite database from a CSV/TSV (LIPID MAPS export, SwissLipids,
    or any custom table) into a list of :class:`Lipid`, ready to pass as ``db=`` to
    :class:`~smile_msi.match.Annotator` or :func:`~smile_msi.annotate.estimate_fdr`.

    Columns are auto-detected (name / formula / class) but can be named explicitly.
    Rows whose formula contains an element this engine can't mass (outside
    H, C, N, O, P, S, Na, K, Cl) or that duplicate a (name, formula) already seen are
    skipped, so ``len`` of the result reflects the usable coverage. ``plausible_only``
    (default on) additionally drops rows whose parsed ``C:D`` sum composition falls outside
    the same per-chain carbon/double-bond envelope :func:`build_database` enforces on its
    own enumeration (:func:`_plausible_external`) — an unfiltered lipid export otherwise
    carries the odd contrived species (a 3-carbon acyl chain, 10 double bonds on a diacyl
    glycerophospholipid, ...) that only win a match because they happen to coincide in mass.
    """
    import pandas as pd

    df = pd.read_csv(path, sep=sep, engine="python")
    name_col = name_col or _pick_col(df.columns, ["abbreviation", "name", "common_name",
                                                  "lipid", "systematic_name"])
    formula_col = formula_col or _pick_col(df.columns, ["formula", "molecular_formula",
                                                        "mol_formula", "chemical_formula"])
    class_col = class_col or _pick_col(df.columns, ["class", "category", "main_class",
                                                    "lipid_class", "sub_class"])
    if name_col is None or formula_col is None:
        raise ValueError("could not find name/formula columns; pass name_col / formula_col")

    lipids, seen = [], set()
    for _, row in df.iterrows():
        f = parse_formula(row[formula_col])
        if not f or any(el not in ELEMENTS for el in f):
            continue
        name = str(row[name_col]).strip()
        if not name or name.lower() == "nan":
            continue
        cls = str(row[class_col]).strip() if class_col and pd.notna(row.get(class_col)) \
            else _class_from_name(name)
        if plausible_only and not _plausible_external(name):
            continue
        key = (name, tuple(sorted(f.items())))
        if key in seen:
            continue
        seen.add(key)
        lipids.append(Lipid(name, cls or "Lipid", formula_mass(f), float(tissue_prior), formula=f))
    return lipids


def merge_lipids(base, ext):
    """Merge ``ext`` :class:`Lipid` list onto ``base``, adding only species whose exact
    elemental **composition** isn't already present (same composition = same monoisotopic mass
    = redundant for annotation *coverage*; the ``base`` entry is preferred, so its tissue prior
    is kept). Returns ``(merged, n_external, n_added)``. The single dedup core shared by the
    CSV (:func:`merge_external_db`) and SDF (wizard) import paths."""
    base = list(base)
    seen = {tuple(sorted(l.formula.items())) for l in base}
    added = [l for l in ext if tuple(sorted(l.formula.items())) not in seen]
    return base + added, len(ext), len(added)


def merge_external_db(path, base=None, *, tissue_prior: float = 0.5,
                      name_col=None, formula_col=None, class_col=None, sep=None,
                      plausible_only: bool = True):
    """Load an external database (:func:`load_external_db`) and **merge** it onto the in-silico
    enumeration (or a provided ``base``) instead of replacing it — so you gain the external DB's
    extra classes/species while keeping the built-in **tissue priors** and the **curated
    metabolites** (NAA, taurine, …) that a lipids-only export like LIPID MAPS doesn't carry.
    Dedup is by exact elemental composition (:func:`merge_lipids`). ``plausible_only`` (default
    on) is forwarded to :func:`load_external_db`.

    Returns ``(merged_db, n_external, n_added)`` — the combined list, how many usable lipids the
    file yielded, and how many were new (not already covered by ``base``)."""
    base = base if base is not None else build_database()
    ext = load_external_db(path, name_col=name_col, formula_col=formula_col,
                           class_col=class_col, tissue_prior=tissue_prior, sep=sep,
                           plausible_only=plausible_only)
    return merge_lipids(base, ext)


# --------------------------------------------------------------------------- #
# LIPID MAPS LMSD .sdf import — parse the structure-data file directly into Lipids, so the
# setup wizard can import a downloaded LMSD.sdf in one step (no separate CSV conversion). The
# same field extraction the scripts/lmsd_to_csv.py CLI uses.
# --------------------------------------------------------------------------- #
_SDF_FIELD = re.compile(r"^>\s+<([^>]+)>")
_SDF_NAME_FIELDS = ["ABBREVIATION", "COMMON_NAME", "NAME", "SYSTEMATIC_NAME", "LM_ID"]
_SDF_FORMULA_FIELDS = ["FORMULA", "MOLECULAR_FORMULA"]
_SDF_CLASS_FIELDS = ["MAIN_CLASS", "CATEGORY", "SUB_CLASS"]


def parse_sdf(text: str) -> list:
    """Parse the *data fields* of every record in an SDF into a list of ``{FIELD: value}``
    dicts. The molblock (connection table) is ignored — only ``> <FIELD>``/value pairs are
    read; records are delimited by ``$$$$``. Pure text, no chemistry toolkit."""
    records, fields, key = [], {}, None
    for line in text.splitlines():
        if line.strip() == "$$$$":
            if fields:
                records.append(fields)
            fields, key = {}, None
            continue
        m = _SDF_FIELD.match(line)
        if m:
            key = m.group(1).strip()
            fields.setdefault(key, "")
            continue
        if key is not None:
            if line.strip() == "":
                key = None
            else:
                fields[key] = (fields[key] + " " + line.strip()).strip() if fields[key] else line.strip()
    if fields:
        records.append(fields)
    return records


def _sdf_first(rec: dict, keys) -> str:
    for k in keys:
        v = (rec.get(k) or "").strip()
        if v and v.lower() not in ("n/a", "na", "none", "-"):
            return v
    return ""


def load_sdf(path, *, categories=None, min_mass=None, max_mass=None,
             tissue_prior: float = 0.5, plausible_only: bool = True) -> list:
    """Load a LIPID MAPS LMSD ``.sdf`` directly into a list of :class:`Lipid` — the wizard's
    one-step import. Skips rows whose formula uses an element this engine can't mass (outside
    H,C,N,O,P,S,Na,K,Cl) or that duplicate a (name, formula) already seen; optional
    ``categories`` (CATEGORY/MAIN_CLASS substrings) and ``min_mass``/``max_mass`` filters trim
    the ~48k full LMSD to a tissue-relevant, less-isobaric slice. ``plausible_only`` (default
    on) additionally drops the same contrived-chain-composition rows :func:`load_external_db`
    does (:func:`_plausible_external`). Same output contract as :func:`load_external_db`."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        records = parse_sdf(fh.read())
    cats = [c.strip().lower() for c in categories] if categories else None
    lipids, seen = [], set()
    for rec in records:
        name = _sdf_first(rec, _SDF_NAME_FIELDS)
        formula = _sdf_first(rec, _SDF_FORMULA_FIELDS)
        if not name or not formula:
            continue
        f = parse_formula(formula)
        if not f or any(el not in ELEMENTS for el in f):
            continue
        if cats is not None:
            hay = f"{rec.get('CATEGORY', '')} {rec.get('MAIN_CLASS', '')} {rec.get('SUB_CLASS', '')}".lower()
            if not any(c in hay for c in cats):
                continue
        mass = formula_mass(f)
        if (min_mass is not None and mass < min_mass) or (max_mass is not None and mass > max_mass):
            continue
        cls = _sdf_first(rec, _SDF_CLASS_FIELDS) or _class_from_name(name)
        if plausible_only and not _plausible_external(name):
            continue
        key = (name, tuple(sorted(f.items())))
        if key in seen:
            continue
        seen.add(key)
        lipids.append(Lipid(name, cls or "Lipid", mass, float(tissue_prior), formula=f))
    return lipids
