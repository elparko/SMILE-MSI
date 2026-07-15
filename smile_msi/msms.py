"""MS/MS confirmation — use fragment spectra to confirm or refute a lipid class.

An MS1 m/z match is ambiguous (isobars tie on mass). A fragmentation spectrum
resolves it: each lipid class produces **diagnostic** product ions or neutral
losses (sulfate ``HSO4-`` 96.96 → sulfatide; phosphocholine 184.07 → PC/SM;
serine neutral loss 87.03 → PS; inositol phosphate 241.01 → PI; sialic-acid
``[Neu5Ac-H]-`` 290.09 → ganglioside). This module
reads an MGF (exportable from Bruker DataAnalysis / SCiLS), matches those rules,
and returns a verdict that raises or lowers confidence in an MS1 annotation.

Rules are conservative — only well-established diagnostics — so a *confirmed*
verdict is meaningful. ``parse_mgf`` and the rule matching are pure and testable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Diagnostic product ions (m/z of the fragment anion/cation), by class and mode.
DIAGNOSTIC_FRAGMENTS = {
    "negative": {
        "Sulfatide": [96.9601],            # HSO4-
        "PI": [241.0119, 223.0013, 152.9958],
        "PG": [152.9958, 171.0064],
        "PA": [152.9958],
        "PE": [196.0380, 140.0118],        # glycerophosphoethanolamine / phosphoethanolamine
        # lyso species share their diacyl headgroup fragments
        "LPI": [241.0119, 223.0013, 152.9958],
        "LPG": [152.9958, 171.0064],
        "LPA": [152.9958],
        "LPS": [152.9958],
        "LPE": [196.0380, 140.0118],
        # gangliosides: the sialic-acid (Neu5Ac) B-ion [Neu5Ac-H]- is the hallmark
        # diagnostic — singly charged, so it appears regardless of the (often 2-/3-)
        # precursor charge, unlike a charge-dependent neutral loss.
        "GM3": [290.0881], "GM2": [290.0881], "GM1": [290.0881],
        "GD3": [290.0881], "GD1": [290.0881], "GT1": [290.0881],
    },
    "positive": {
        "PC": [184.0733],                  # phosphocholine
        "SM": [184.0733, 264.2686],        # phosphocholine + sphingosine (d18:1) base
        "Cer": [264.2686],
        "HexCer": [264.2686],
    },
}
# Diagnostic neutral losses from the precursor, by class and mode.
DIAGNOSTIC_LOSSES = {
    "negative": {
        "PS": [87.0320],                   # serine
        "LPS": [87.0320],
        "HexCer": [162.0528],              # hexose (Gal/Glc) loss
    },
    "positive": {
        "PE": [141.0191],                  # phosphoethanolamine
        "PS": [185.0089],                  # phosphoserine
        "HexCer": [162.0528],              # hexose loss
    },
}


@dataclass
class MS2Spectrum:
    precursor: float
    mz: np.ndarray
    intensity: np.ndarray


def parse_mgf(path: str):
    """Parse an MGF file into a list of :class:`MS2Spectrum`."""
    spectra = []
    pre, mzs, ints = None, [], []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line == "BEGIN IONS":
                pre, mzs, ints = None, [], []
            elif line == "END IONS":
                spectra.append(MS2Spectrum(pre or 0.0, np.asarray(mzs), np.asarray(ints)))
            elif "=" in line:
                key, _, val = line.partition("=")
                if key.upper() == "PEPMASS":
                    pre = float(val.split()[0])
            else:
                parts = line.split()
                try:                                  # tolerate stray/malformed lines
                    mz = float(parts[0])
                    inten = float(parts[1]) if len(parts) > 1 else 1.0
                except (ValueError, IndexError):
                    continue                          # not a numeric peak line — skip it
                mzs.append(mz)
                ints.append(inten)
    return spectra


def matched_classes(spectrum: MS2Spectrum, mode: str = "negative", tol_da: float = 0.01):
    """Which lipid classes have a diagnostic fragment or neutral loss present."""
    found = set()
    frags = np.asarray(spectrum.mz)
    for cls, targets in DIAGNOSTIC_FRAGMENTS.get(mode, {}).items():
        if any(np.any(np.abs(frags - t) <= tol_da) for t in targets):
            found.add(cls)
    pre = spectrum.precursor
    for cls, losses in DIAGNOSTIC_LOSSES.get(mode, {}).items():
        if any(np.any(np.abs(frags - (pre - dl)) <= tol_da) for dl in losses):
            found.add(cls)
    return found


def classify(spectrum: MS2Spectrum, mode: str = "negative", tol_da: float = 0.01):
    """Rank candidate lipid classes for an *unknown* MS2 spectrum by how many of each
    class's diagnostic ions / neutral losses are present. Returns ``[(class, n_hits)]``
    sorted by descending evidence (empty if nothing diagnostic matched)."""
    frags = np.asarray(spectrum.mz)
    pre = spectrum.precursor
    scores: dict = {}
    for cls, targets in DIAGNOSTIC_FRAGMENTS.get(mode, {}).items():
        n = sum(1 for t in targets if np.any(np.abs(frags - t) <= tol_da))
        if n:
            scores[cls] = scores.get(cls, 0) + n
    for cls, losses in DIAGNOSTIC_LOSSES.get(mode, {}).items():
        n = sum(1 for dl in losses if np.any(np.abs(frags - (pre - dl)) <= tol_da))
        if n:
            scores[cls] = scores.get(cls, 0) + n
    return sorted(scores.items(), key=lambda kv: -kv[1])


def confirm_class(lipid_class: str, spectrum: MS2Spectrum, mode: str = "negative",
                  tol_da: float = 0.01) -> str:
    """Verdict for an MS1 class assignment given an MS2 spectrum:

    * ``confirmed`` — a diagnostic for this class is present;
    * ``unsupported`` — diagnostics for *other* classes are present but not this one;
    * ``inconclusive`` — no diagnostic rule matched (this class has no rule, or the
      spectrum lacks diagnostic ions).
    """
    found = matched_classes(spectrum, mode=mode, tol_da=tol_da)
    has_rule = (lipid_class in DIAGNOSTIC_FRAGMENTS.get(mode, {})
                or lipid_class in DIAGNOSTIC_LOSSES.get(mode, {}))
    if lipid_class in found:
        return "confirmed"
    if found and has_rule:
        return "unsupported"
    return "inconclusive"


def acyl_chains(spectrum: MS2Spectrum, fa_range=(12, 24), db_max=6, tol_da: float = 0.01,
                max_chains: int | None = None):
    """Fatty-acyl chains read from [FA-H]- carboxylate ions (negative mode), ranked
    by the intensity of the matched carboxylate (most intense first). Returns a list
    of ``"C:D"`` strings; pass ``max_chains`` to keep only the top few."""
    from .masses import formula_mass, ELECTRON, ELEMENTS

    frags = np.asarray(spectrum.mz)
    inten = np.asarray(spectrum.intensity)
    have_int = inten.shape == frags.shape and inten.size > 0
    hits = []
    for c in range(fa_range[0], fa_range[1] + 1):
        for d in range(0, db_max + 1):
            fa = formula_mass({"C": c, "H": 2 * c - 2 * d, "O": 2})
            carboxylate = fa - ELEMENTS["H"] + ELECTRON          # [FA-H]-
            idx = np.flatnonzero(np.abs(frags - carboxylate) <= tol_da)
            if idx.size:
                strength = float(inten[idx].max()) if have_int else 1.0
                hits.append((strength, f"{c}:{d}"))
    hits.sort(key=lambda h: h[0], reverse=True)
    chains = [name for _, name in hits]
    return chains[:max_chains] if max_chains else chains
