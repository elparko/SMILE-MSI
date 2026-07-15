"""Match observed m/z peaks against the in-silico lipid database within a ppm
tolerance, across a set of adducts, and rank the candidates.

Ranking score (higher = better):
    -|ppm error|                       mass accuracy (dominant term)
    + scale * tissue_prior             nerve/myelin class expectation
    + scale * adduct_prior             how usual that adduct is for the class in this mode

with ``scale = ppm_tol * PRIOR_WINDOW_FRAC`` (0.1). The priors are deliberately small
(tissue 0.2-1.0, adduct <=0.15) and scaled to the *tolerance window* so their combined
swing is a fixed ~11% of the window — they only break ties between candidates within a
fraction of ``ppm_tol`` of each other; beyond that the closest mass always wins. (At the
default 10 ppm tolerance ``scale`` is 1.0, the historical value.)

This ranks on mass + class/adduct priors alone — it has no image evidence. When a
dataset is available, :func:`smile_msi.annotate.build_feature_list` re-ranks these
candidates by their MSM score (mass x spectral-isotope x spatial-isotope) so strong
image evidence can promote an isobaric runner-up the priors alone would miss.
"""
from __future__ import annotations

import bisect
import math
import re
from dataclasses import dataclass

from .lipiddb import Lipid, build_database
from .masses import NEG_ADDUCTS, POS_ADDUCTS, ion_mz, adduct_charge, ppm_error

# Sum-composition token in a lipid name, e.g. "PE 34:1", "Sulfatide 42:2;O3" -> (34, 1).
_CD_RE = re.compile(r"(\d+):(\d+)")


def _carbons_dbs(name: str) -> tuple[int | None, int | None]:
    """(total acyl carbons, total double bonds) parsed from a lipid name, or (None, None)
    for fixed compounds without a ``C:D`` token (sterols, curated metabolites)."""
    m = _CD_RE.search(name or "")
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)

# Multiply-charged adducts are only enumerated for classes that actually form them
# (multiple acidic groups) — otherwise every lipid would spawn a spurious half-/
# third-mass ion cluttering the low-mass region with chemically implausible
# candidates. Mono-sialo gangliosides (one carboxyl) reach 2-; di-/tri-sialo reach 3-.
_MULTICHARGE_ADDUCT_CLASSES = {
    "[M-2H]2-": {"GM3", "GM2", "GM1", "GD3", "GD1", "GT1"},
    "[M-3H]3-": {"GD3", "GD1", "GT1"},
    "[M+2H]2+": set(),   # no multiply-charged positive species in this nerve DB
}

# Neutral glycerolipids don't deprotonate, so they're never seen in negative mode —
# skip enumerating them under negative adducts (otherwise every neg peak gains a
# spurious low-prior TG/DG/MG candidate that can only mislead).
_POSITIVE_ONLY_CLASSES = {"TG", "DG", "MG"}

# Some adducts are chemically possible only for specific classes, so we must not even
# *enumerate* them elsewhere (a +0 prior still lets the ion win a tie or pad
# n_candidates / FDR-decoy noise — audit 2026-06-24, finding #6). Two cases:
#   * Headgroup demethylation [M-CH3]- requires a quaternary-ammonium (choline) headgroup;
#     generating it for PE/FA/sulfatide/etc. is chemically wrong.
#   * Chloride / formate / acetate adducts form for species that do NOT readily deprotonate
#     — choline lipids (PC/SM/PC-O/LPC, quaternary ammonium) and neutral ceramides
#     (HexCer/Cer). Acidic-headgroup lipids (PE/PS/PI/PG/PA, FA, sulfatides, gangliosides)
#     ionize as plain [M-H]-, so enumerating their Cl/HCOO/CH3COO adducts only manufactures
#     false hits that a small mass offset lets win over the true [M-H]- (the sulfatide
#     mislabel this fixes). Classes here mirror the [M+Cl]-/[M+HCOO]-/[M+CH3COO]- entries
#     in _NEG_ADDUCT_PRIOR. An adduct absent from this map is enumerated for every class
#     (subject to the filters above).
_ADDUCT_CLASS_WHITELIST = {
    "[M-CH3]-":     {"PC", "SM", "PC-O", "LPC"},
    "[M+Cl]-":      {"PC", "SM", "PC-O", "LPC", "HexCer", "Cer"},
    "[M+HCOO]-":    {"PC", "SM", "PC-O", "LPC", "HexCer", "Cer"},
    "[M+CH3COO]-":  {"PC", "SM", "PC-O", "LPC", "HexCer", "Cer"},
}

# Priors are tie-breakers, not mass evidence: their combined swing is capped at this
# fraction of the ppm window (PRIOR_WINDOW_FRAC * ppm_tol score units). At the default
# 10 ppm tolerance the scale is 1.0, so legacy behaviour is preserved; at a tight
# tolerance the priors can no longer override a clearly-closer mass (which was the old
# behaviour — a ~0.95 ppm absolute swing, i.e. ~95% of a 1 ppm window). Audit finding #5.
PRIOR_WINDOW_FRAC = 0.1

# Which adducts are chemically usual for each class in negative mode.
_NEG_ADDUCT_PRIOR = {
    "[M-H]-": {"PE", "PS", "PI", "PG", "PA", "Sulfatide", "FA", "OxFA",
               "LPE", "LPS", "LPI", "LPG", "LPA", "PE-O",
               "GM3", "GM2", "GM1", "GD3", "GD1", "GT1", "ST", "Metab"},
    # gangliosides are typically seen multiply-charged in a <=1000 m/z window
    "[M-2H]2-": {"GM3", "GM2", "GM1", "GD3", "GD1", "GT1"},
    "[M-3H]3-": {"GD3", "GD1", "GT1"},   # di-/tri-sialo carry the extra charge
    "[M-CH3]-": {"PC", "SM", "PC-O", "LPC"},
    "[M+Cl]-": {"PC", "SM", "PC-O", "LPC", "HexCer", "Cer"},
    "[M+HCOO]-": {"PC", "SM", "PC-O", "LPC", "HexCer", "Cer"},
    "[M+CH3COO]-": {"PC", "SM", "PC-O", "LPC", "HexCer", "Cer"},
}

# Which adducts are chemically usual for each class in positive mode. Protonatable
# headgroups (choline/amine) favour [M+H]+; neutral lipids (TG/DG/MG/CE) ionize as
# ammonium/sodium; alkali adducts are ubiquitous in tissue. Like the negative table
# these are small tie-breakers (≤0.15), not hard filters.
_POS_ADDUCT_PRIOR = {
    "[M+H]+":  {"PC", "PC-O", "SM", "LPC", "PE", "PE-O", "LPE", "PS", "LPS",
                "Cer", "HexCer", "CAR", "Metab", "ST"},
    "[M+NH4]+": {"TG", "DG", "MG", "CE", "HexCer", "Cer"},
    "[M+Na]+": {"PC", "PC-O", "SM", "PE", "PG", "PS", "PI", "Cer", "HexCer",
                "TG", "DG", "MG", "CE", "ST"},
    "[M+K]+":  {"PC", "SM", "PE", "Cer", "HexCer", "TG", "DG"},
}


@dataclass
class Candidate:
    mz: float
    adduct: str
    lipid: Lipid
    ppm: float
    score: float


class Annotator:
    def __init__(self, mode="negative", ppm_tol=5.0, db=None):
        self.mode = mode
        self.ppm_tol = ppm_tol
        self.adducts = NEG_ADDUCTS if mode == "negative" else POS_ADDUCTS
        # The base deprotonated/protonated ion — preferred over adduct forms on an
        # exact-score tie (see the sort in annotate_mz).
        self._base_adduct = "[M-H]-" if mode == "negative" else "[M+H]+"
        self._db_arg = db            # the raw DB argument (None = built-in) — see matches()
        self.lipids = db if db is not None else build_database()
        # Precompute a sorted (ion_mz, adduct, lipid) index for fast lookup.
        # Multiply-charged adducts are only enumerated for classes that actually
        # form them (see _MULTICHARGE_CLASSES).
        neg = self.mode == "negative"
        self._index = []
        for lip in self.lipids:
            if neg and lip.lipid_class in _POSITIVE_ONLY_CLASSES:
                continue                                  # neutral lipids: positive mode only
            for add in self.adducts:
                if adduct_charge(add) > 1 and lip.lipid_class not in \
                        _MULTICHARGE_ADDUCT_CLASSES.get(add, ()):
                    continue
                allowed = _ADDUCT_CLASS_WHITELIST.get(add)
                if allowed is not None and lip.lipid_class not in allowed:
                    continue                                  # adduct chemically impossible for this class
                self._index.append((ion_mz(lip.neutral_mass, add), add, lip))
        self._index.sort(key=lambda t: t[0])
        self._mzs = [t[0] for t in self._index]
        # Per-instance memo for annotate_mz. The index and tolerance are fixed at construction
        # (any mode/tolerance/DB change builds a *fresh* Annotator), so the candidates for a
        # given m/z never change for this instance. Switching or rendering a feature list calls
        # annotate_mz ~3x per peak (peak table + ratio combos + class map), so caching collapses
        # that to one real lookup per distinct m/z. Keyed on the exact float → zero behaviour
        # change; callers treat the result read-only (slice/index), never mutate it in place.
        self._mz_cache: dict[float, list] = {}

    def set_ppm_tol(self, ppm_tol: float) -> None:
        """Retune the match tolerance in place, WITHOUT rebuilding the index. `_index`/`_mzs`
        are enumerated purely from (mode, db) and never reference `ppm_tol` — it's applied only
        in annotate_mz (the bisect window `win` and the tie-break `prior_scale`). So a tolerance
        change is a field update, not a full `lipids x adducts` re-enumeration + sort. The
        per-instance memo IS tolerance-dependent (both the candidate window and each score's
        prior scale with ppm_tol), so it must be dropped. On a large merged (LIPID MAPS) DB this
        replaces the ~40-120 ms reconstruction the GUI was doing on every ID-tolerance spinbox
        tick with ~0 — behaviour is identical to a freshly built Annotator at the new tolerance."""
        if ppm_tol == self.ppm_tol:
            return
        self.ppm_tol = ppm_tol
        self._mz_cache.clear()

    def matches(self, mode, db) -> bool:
        """True when this annotator's index is already current for (mode, db). The index is
        enumerated purely from the mode and the lipid DB, so a caller holding an annotator that
        `matches` can retune the tolerance via set_ppm_tol instead of rebuilding it. `db` is
        compared by identity (None = the built-in DB), matching how __init__ consumed it."""
        return self.mode == mode and self._db_arg is db

    def _adduct_prior(self, lipid_class, adduct):
        table = _NEG_ADDUCT_PRIOR if self.mode == "negative" else _POS_ADDUCT_PRIOR
        return 0.15 if lipid_class in table.get(adduct, ()) else 0.0

    def annotate_mz(self, mz: float) -> list[Candidate]:
        """All candidates for one observed m/z, best first."""
        # Guard non-finite / non-positive m/z: NaN makes the bisect window span the whole
        # index (returning the entire DB as NaN-scored candidates, which then leaks an
        # arbitrary class label into class_of); zero/negative can't be a real ion. (#1)
        if not math.isfinite(mz) or mz <= 0:
            return []
        cached = self._mz_cache.get(mz)
        if cached is not None:
            return cached
        win = mz * self.ppm_tol / 1e6
        lo = bisect.bisect_left(self._mzs, mz - win)
        hi = bisect.bisect_right(self._mzs, mz + win)
        prior_scale = self.ppm_tol * PRIOR_WINDOW_FRAC      # tolerance-relative tie-break weight
        out = []
        for theo, add, lip in self._index[lo:hi]:
            ppm = ppm_error(mz, theo)
            prior = lip.tissue_prior + self._adduct_prior(lip.lipid_class, add)
            score = -abs(ppm) + prior_scale * prior
            out.append(Candidate(mz, add, lip, ppm, score))
        # Sort by score (desc). On an *exact* score tie (isobars within a fraction of the
        # window), prefer the simpler/more-likely assignment — base ion ([M-H]-/[M+H]+) over
        # an adduct, then even-chain over odd, then fewer double bonds — before the fully
        # deterministic class/name/adduct fallback (so class_of never depends on DB insertion
        # order, #7). This only reorders equal-score candidates; the closest mass still wins
        # beyond the tie, so it's a robustness tie-break, not a mass-evidence change.
        def _tiebreak(c):
            tc, ndb = _carbons_dbs(c.lipid.name)
            base_rank = 0 if c.adduct == self._base_adduct else 1
            odd = 1 if (tc is not None and tc % 2 == 1) else 0
            return (-c.score, base_rank, odd, ndb if ndb is not None else 0,
                    c.lipid.lipid_class, c.lipid.name, c.adduct)
        out.sort(key=_tiebreak)
        if len(self._mz_cache) >= 65536:        # bound a pathological caller (e.g. decoy storms)
            self._mz_cache.clear()
        self._mz_cache[mz] = out
        return out

    def annotate_many(self, mzs, top_n=3):
        """{mz: [top_n candidates]} for a list of observed m/z."""
        return {mz: self.annotate_mz(mz)[:top_n] for mz in mzs}

    def class_of(self, mz: float) -> str:
        """Top-ranked lipid class for one observed m/z, or ``''`` if nothing matches."""
        cs = self.annotate_mz(float(mz))
        return cs[0].lipid.lipid_class if cs else ""

    def classes_for(self, mzs) -> list[str]:
        """Top lipid class per m/z, parallel to ``mzs`` (``''`` = unannotated). The single
        source of the per-peak class labels the class-level statistics roll up by."""
        return [self.class_of(m) for m in mzs]
