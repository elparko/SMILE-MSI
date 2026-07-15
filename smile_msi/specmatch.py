"""Spectral-library MS/MS matching — cosine, modified-cosine and spectral-entropy
similarity between an acquired MS2 spectrum and reference library entries.

This is a *structure-level* confidence tier that sits **alongside** the conservative
rule-based confirmation in :mod:`smile_msi.msms`. Where the rule path asks only
"is diagnostic fragment X present?", this module scores a full acquired MS2
against reference spectra (parsed from ``.msp`` / ``.mgf`` libraries — e.g.
LIPID MAPS or in-house) and ranks candidate identifications.

Three similarity metrics are provided, each reimplemented clean-room from the
equations in its source publication (``matchms`` (MIT) and GNPS source were *not*
copied):

* ``cosine`` — the classical normalized dot product with √-intensity weighting.
  Stein, S.E. & Scott, D.R. (1994), *Optimization and testing of mass spectral
  library search algorithms for compound identification.* J. Am. Soc. Mass
  Spectrom. 5, 859–866. doi:10.1016/1044-0305(94)87009-8
* ``modified_cosine`` — greedy peak alignment that additionally allows a
  precursor-mass-difference shift, so analog/related spectra still score. Wang,
  M. et al. (2016), *Sharing and community curation of mass spectrometry data
  with GNPS.* Nat. Biotechnol. 34, 828–837. doi:10.1038/nbt.3597
* ``spectral_entropy_similarity`` — entropy-weighted spectral similarity (the
  default; more discriminating and noise-robust than the dot product). Li, Y.
  et al. (2021), *Spectral entropy outperforms MS/MS dot product similarity for
  small-molecule compound identification.* Nat. Methods 18, 1524–1531.
  doi:10.1038/s41592-021-01331-z

All matching is **deterministic** (greedy alignment, no sampling — no random
seed). Pure NumPy; no new hard dependency.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .msms import MS2Spectrum  # reuse the dataclass (precursor, mz, intensity)


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class RefSpectrum:
    """A reference library spectrum with parsed metadata."""

    name: str                       # e.g. "PC 34:1 [M+H]+"
    precursor: float
    mz: np.ndarray
    intensity: np.ndarray
    lipid_class: str = ""           # parsed from name where possible
    adduct: str = ""
    smiles: str = ""
    meta: dict | None = None        # raw .msp/.mgf header fields


@dataclass
class LibraryHit:
    """A ranked library match for a query spectrum."""

    ref: RefSpectrum
    score: float                    # the chosen primary metric (entropy by default)
    cosine: float
    modified_cosine: float
    entropy: float
    n_matched: int                  # aligned peaks (provenance / explainability)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _as_arrays(spec) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (mz, intensity, precursor) float arrays for an MS2Spectrum/RefSpectrum."""
    mz = np.asarray(spec.mz, dtype=float).ravel()
    inten = np.asarray(spec.intensity, dtype=float).ravel()
    if inten.shape != mz.shape:
        inten = np.ones_like(mz)
    pre = float(getattr(spec, "precursor", 0.0) or 0.0)
    return mz, inten, pre


def _lipid_class_from_name(name: str) -> str:
    """Best-effort lipid class token from a reference name, e.g. "PC 34:1 [M+H]+" → "PC".

    Falls back to the empty string when the leading token does not look like a
    lipid abbreviation (purely heuristic; the parser stores the full name too).
    """
    if not name:
        return ""
    head = name.strip().split()[0] if name.strip().split() else ""
    # strip a trailing punctuation, keep alphanumerics only for the class token
    head = head.strip("[](),;:")
    if head and head[0].isalpha():
        return head
    return ""


# --------------------------------------------------------------------------- #
# Import — .msp / .mgf libraries
# --------------------------------------------------------------------------- #
_MSP_PRECURSOR_KEYS = {
    "precursormz", "precursor_mz", "precursor m/z", "precursor", "exactmass",
    "exact_mass", "pepmass",
}
_MSP_NAME_KEYS = {"name", "title", "compound_name"}
_MSP_SMILES_KEYS = {"smiles"}
_MSP_ADDUCT_KEYS = {"precursor_type", "precursortype", "adduct", "ion_mode_adduct"}


def _float_or_none(val: str):
    try:
        return float(str(val).split()[0])
    except (ValueError, IndexError):
        return None


def parse_msp(path: str) -> list[RefSpectrum]:
    """Parse a NIST/MoNA/LIPID MAPS ``.msp`` library into :class:`RefSpectrum` records.

    Tolerant of dialect differences: ``Name``/``PrecursorMZ``/``Precursor_type``/
    ``SMILES``/``Num Peaks`` headers (case-insensitive) followed by ``mz intensity``
    peak lines. Malformed peak lines are skipped (mirrors :func:`msms.parse_mgf`).
    A blank line terminates a record.
    """
    refs: list[RefSpectrum] = []
    meta: dict = {}
    mzs: list[float] = []
    ints: list[float] = []

    def _flush():
        if not meta and not mzs:
            return
        name = ""
        for k in _MSP_NAME_KEYS:
            if k in meta:
                name = meta[k]
                break
        pre = None
        for k in _MSP_PRECURSOR_KEYS:
            if k in meta:
                pre = _float_or_none(meta[k])
                if pre is not None:
                    break
        smiles = next((meta[k] for k in _MSP_SMILES_KEYS if k in meta), "")
        adduct = next((meta[k] for k in _MSP_ADDUCT_KEYS if k in meta), "")
        refs.append(RefSpectrum(
            name=name,
            precursor=pre if pre is not None else 0.0,
            mz=np.asarray(mzs, dtype=float),
            intensity=np.asarray(ints, dtype=float),
            lipid_class=_lipid_class_from_name(name),
            adduct=adduct,
            smiles=smiles,
            meta=dict(meta),
        ))

    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                # blank line ends the current record
                if meta or mzs:
                    _flush()
                    meta, mzs, ints = {}, [], []
                continue
            if ":" in line and not _looks_like_peak(line):
                key, _, val = line.partition(":")
                key = key.strip().lower()
                val = val.strip()
                # "Num Peaks" just signals peak block start; no need to store
                meta[key] = val
                continue
            # otherwise treat as a peak line ("mz intensity", possibly ; separated)
            _parse_peak_line(line, mzs, ints)
    if meta or mzs:
        _flush()
    return refs


def _looks_like_peak(line: str) -> bool:
    """A peak line starts with a number; a header line ("Name: ...") does not."""
    first = line.split(":", 1)[0].strip().split()
    if not first:
        return False
    try:
        float(first[0])
        return True
    except ValueError:
        return False


def _parse_peak_line(line: str, mzs: list, ints: list) -> None:
    """Append numeric mz/intensity pairs from a peak line; skip malformed tokens.

    Handles both "mz intensity" per line and several pairs on one line (e.g.
    "100.0 5; 200.0 9")."""
    cleaned = line.replace(";", " ").replace(",", " ")
    parts = cleaned.split()
    if not parts:
        return
    # The common case is exactly one pair per line.
    try:
        mz = float(parts[0])
    except ValueError:
        return
    inten = 1.0
    if len(parts) > 1:
        try:
            inten = float(parts[1])
        except ValueError:
            inten = 1.0
    mzs.append(mz)
    ints.append(inten)


def parse_mgf_library(path: str) -> list[RefSpectrum]:
    """Parse an ``.mgf`` library into :class:`RefSpectrum` records.

    Richer than :func:`msms.parse_mgf`: captures ``TITLE``/``NAME``/``SMILES``/
    ``CHARGE``/``PEPMASS`` as metadata so library entries keep their identity.
    Malformed peak lines are skipped.
    """
    refs: list[RefSpectrum] = []
    meta: dict = {}
    mzs: list[float] = []
    ints: list[float] = []
    in_ions = False

    def _flush():
        name = meta.get("name") or meta.get("title") or ""
        pre = _float_or_none(meta["pepmass"]) if "pepmass" in meta else None
        refs.append(RefSpectrum(
            name=name,
            precursor=pre if pre is not None else 0.0,
            mz=np.asarray(mzs, dtype=float),
            intensity=np.asarray(ints, dtype=float),
            lipid_class=_lipid_class_from_name(name),
            adduct=meta.get("precursor_type", "") or meta.get("adduct", ""),
            smiles=meta.get("smiles", ""),
            meta=dict(meta),
        ))

    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line == "BEGIN IONS":
                in_ions, meta, mzs, ints = True, {}, [], []
            elif line == "END IONS":
                _flush()
                in_ions = False
            elif "=" in line and not _looks_like_peak(line):
                key, _, val = line.partition("=")
                meta[key.strip().lower()] = val.strip()
            elif in_ions:
                _parse_peak_line(line, mzs, ints)
    return refs


def load_library(path: str) -> list[RefSpectrum]:
    """Load a spectral library, dispatching on file extension (``.msp`` / ``.mgf``)."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".msp":
        return parse_msp(path)
    if ext == ".mgf":
        return parse_mgf_library(path)
    raise ValueError(
        f"Unsupported spectral-library extension {ext!r}: expected .msp or .mgf."
    )


# --------------------------------------------------------------------------- #
# Normalization & alignment
# --------------------------------------------------------------------------- #
def _clean_normalize(mz, inten, *, min_rel: float = 0.0, max_peaks=None
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Drop peaks below ``min_rel`` of the base peak, optionally keep top-N, sort by m/z.

    Returns (mz, intensity) with intensities left on their original scale (metric
    code does its own L2 / probability normalization)."""
    mz = np.asarray(mz, dtype=float).ravel()
    inten = np.asarray(inten, dtype=float).ravel()
    if inten.shape != mz.shape:
        inten = np.ones_like(mz)
    if mz.size == 0:
        return mz, inten
    inten = np.clip(inten, 0.0, None)
    base = inten.max() if inten.size else 0.0
    if base > 0 and min_rel > 0:
        keep = inten >= min_rel * base
        mz, inten = mz[keep], inten[keep]
    if max_peaks is not None and mz.size > max_peaks:
        idx = np.argsort(inten)[::-1][:max_peaks]
        mz, inten = mz[idx], inten[idx]
    order = np.argsort(mz)
    return mz[order], inten[order]


def _align(mz_a, mz_b, *, tol_da: float = 0.02, shift: float = 0.0):
    """Greedy 1:1 nearest-neighbour peak alignment within ``tol_da``.

    Matches each peak of ``a`` to at most one peak of ``b`` (and vice versa),
    in ascending order of mass difference, so no peak is double-assigned. When
    ``shift`` is non-zero, ``b`` peaks are compared at ``mz_b + shift`` (used by
    :func:`modified_cosine` to align a precursor-mass-difference shifted copy).

    Returns a list of ``(i, j)`` index pairs into ``a`` and ``b``.
    """
    mz_a = np.asarray(mz_a, dtype=float).ravel()
    mz_b = np.asarray(mz_b, dtype=float).ravel()
    if mz_a.size == 0 or mz_b.size == 0:
        return []
    b_shift = mz_b + shift
    # candidate (diff, i, j) within tolerance
    cands = []
    for i, a in enumerate(mz_a):
        diffs = np.abs(b_shift - a)
        for j in np.flatnonzero(diffs <= tol_da):
            cands.append((float(diffs[j]), i, int(j)))
    cands.sort(key=lambda t: t[0])
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, i, j in cands:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j))
    return pairs


# --------------------------------------------------------------------------- #
# Pre-aligned scoring cores — operate on already-normalized arrays + a precomputed
# direct alignment, so match_spectrum_to_library can normalize each spectrum once
# and align once per reference instead of 3× (audit plan 23, item A). The public
# metric functions below are thin wrappers over these, so their standalone contract
# (and the existing direct-call tests) is byte-for-byte unchanged.
# --------------------------------------------------------------------------- #
def _cosine_from(q_mz, q_in, r_mz, r_in, pairs, *, sqrt_weight: bool = True
                 ) -> tuple[float, int]:
    """:func:`cosine` body given normalized arrays + a direct alignment ``pairs``."""
    if q_mz.size == 0 or r_mz.size == 0 or not pairs:
        return 0.0, 0
    wa = np.sqrt(q_in) if sqrt_weight else q_in.astype(float)
    wb = np.sqrt(r_in) if sqrt_weight else r_in.astype(float)
    norm_a = np.linalg.norm(wa)
    norm_b = np.linalg.norm(wb)
    if norm_a == 0 or norm_b == 0:
        return 0.0, 0
    dot = sum(wa[i] * wb[j] for i, j in pairs)
    score = float(dot / (norm_a * norm_b))
    return max(0.0, min(1.0, score)), len(pairs)


def _entropy_from(q_mz, q_in, r_mz, r_in, pairs) -> tuple[float, int]:
    """:func:`spectral_entropy_similarity` body given normalized arrays + direct ``pairs``."""
    if q_mz.size == 0 or r_mz.size == 0:
        return 0.0, 0
    sum_a = q_in.sum()
    sum_b = r_in.sum()
    if sum_a <= 0 or sum_b <= 0:
        return 0.0, 0
    pa = q_in / sum_a
    pb = r_in / sum_b

    matched_a = {i for i, _ in pairs}
    matched_b = {j for _, j in pairs}

    # Build the merged probability spectrum over the union of peaks.
    merged: list[float] = []
    pair_b_for_a = {i: j for i, j in pairs}
    for i in range(q_mz.size):
        if i in matched_a:
            merged.append(pa[i] + pb[pair_b_for_a[i]])
        else:
            merged.append(pa[i])
    for j in range(r_mz.size):
        if j not in matched_b:
            merged.append(pb[j])
    merged_arr = np.asarray(merged, dtype=float)
    total = merged_arr.sum()
    if total <= 0:
        return 0.0, len(pairs)
    p_ab = merged_arr / total  # sums to 1 (each input summed to 1, total = 2)

    s_a = _entropy(pa)
    s_b = _entropy(pb)
    s_ab = _entropy(p_ab)
    # 2*S_AB - S_A - S_B ranges in [0, ln 4]; similarity = 1 - that / ln4.
    sim = 1.0 - (2.0 * s_ab - s_a - s_b) / np.log(4.0)
    return float(max(0.0, min(1.0, sim))), len(pairs)


# --------------------------------------------------------------------------- #
# Similarity metrics
# --------------------------------------------------------------------------- #
def cosine(a, b, *, tol_da: float = 0.02, sqrt_weight: bool = True
           ) -> tuple[float, int]:
    """√-intensity-weighted normalized dot product (Stein & Scott 1994).

    Greedy peak alignment within ``tol_da``, then the cosine of the aligned
    intensity vectors. ``sqrt_weight`` applies the classical √-intensity weighting
    that down-weights large peaks. Returns ``(score in [0, 1], n_matched)``.

    Stein, S.E. & Scott, D.R. (1994). doi:10.1016/1044-0305(94)87009-8
    """
    mz_a, in_a = _clean_normalize(*_as_arrays(a)[:2])
    mz_b, in_b = _clean_normalize(*_as_arrays(b)[:2])
    pairs = _align(mz_a, mz_b, tol_da=tol_da) if mz_a.size and mz_b.size else []
    return _cosine_from(mz_a, in_a, mz_b, in_b, pairs, sqrt_weight=sqrt_weight)


def modified_cosine(a, b, *, tol_da: float = 0.02, sqrt_weight: bool = True,
                    pre=None) -> tuple[float, int]:
    """Modified cosine allowing a precursor-mass-difference shift (Wang et al. 2016 / GNPS).

    Aligns peaks directly *and* with ``b`` shifted by ``Δprecursor = a.precursor −
    b.precursor``, then keeps, per peak, the better of the two assignments without
    double-counting. Recovers the score for analog spectra that differ by a
    constant mass offset. Returns ``(score in [0, 1], n_matched)``.

    ``pre`` is an internal fast-path: a tuple ``(mz_a, in_a, mz_b, in_b,
    direct_pairs, pre_a, pre_b)`` of already-normalized arrays and the precomputed
    *direct* alignment, supplied by :func:`match_spectrum_to_library` so the shared
    normalize/align work is not repeated. The shifted leg is still computed here.

    Wang, M. et al. (2016). Nat. Biotechnol. 34, 828–837. doi:10.1038/nbt.3597
    """
    if pre is not None:
        mz_a, in_a, mz_b, in_b, direct_pairs, pre_a, pre_b = pre
    else:
        mz_a, in_a, pre_a = _as_arrays(a)
        mz_b, in_b, pre_b = _as_arrays(b)
        mz_a, in_a = _clean_normalize(mz_a, in_a)
        mz_b, in_b = _clean_normalize(mz_b, in_b)
        direct_pairs = None
    if mz_a.size == 0 or mz_b.size == 0:
        return 0.0, 0
    wa = np.sqrt(in_a) if sqrt_weight else in_a.astype(float)
    wb = np.sqrt(in_b) if sqrt_weight else in_b.astype(float)
    norm_a = np.linalg.norm(wa)
    norm_b = np.linalg.norm(wb)
    if norm_a == 0 or norm_b == 0:
        return 0.0, 0
    shift = pre_a - pre_b
    # gather direct + shifted candidate pairs, then greedily pick best 1:1
    cands: list[tuple[float, int, int]] = []
    for which_shift in (0.0, shift):
        if which_shift != 0.0 and abs(which_shift) <= tol_da:
            continue  # shifted alignment would duplicate the direct one
        if which_shift == 0.0 and direct_pairs is not None:
            aligned = direct_pairs           # reuse the shared direct alignment
        else:
            aligned = _align(mz_a, mz_b, tol_da=tol_da, shift=which_shift)
        for i, j in aligned:
            cands.append((wa[i] * wb[j], i, j))
    cands.sort(key=lambda t: -t[0])
    used_a: set[int] = set()
    used_b: set[int] = set()
    dot = 0.0
    n = 0
    for prod, i, j in cands:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        dot += prod
        n += 1
    score = float(dot / (norm_a * norm_b))
    return max(0.0, min(1.0, score)), n


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy of a probability vector (natural log; 0·log0 := 0)."""
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def spectral_entropy_similarity(a, b, *, tol_da: float = 0.02) -> tuple[float, int]:
    """Entropy-weighted spectral similarity (Li et al. 2021).

    Each spectrum's intensities are normalized to a probability vector. The merged
    (aligned) spectrum's entropy is compared against the per-spectrum entropies:

        similarity = 1 − (2·S_AB − S_A − S_B) / ln(4)

    where S_A, S_B are the Shannon entropies of the two probability spectra and
    S_AB is the entropy of their peak-wise sum (re-normalized). The result lies in
    [0, 1]; identical spectra give 1, fully disjoint spectra give 0. Returns
    ``(score, n_matched)``.

    Li, Y. et al. (2021). Nat. Methods 18, 1524–1531.
    doi:10.1038/s41592-021-01331-z
    """
    mz_a, in_a = _clean_normalize(*_as_arrays(a)[:2])
    mz_b, in_b = _clean_normalize(*_as_arrays(b)[:2])
    pairs = _align(mz_a, mz_b, tol_da=tol_da) if mz_a.size and mz_b.size else []
    return _entropy_from(mz_a, in_a, mz_b, in_b, pairs)


# --------------------------------------------------------------------------- #
# Top-level scorer
# --------------------------------------------------------------------------- #
_METRIC_FUNCS = {
    "cosine": cosine,
    "modified_cosine": modified_cosine,
    "entropy": spectral_entropy_similarity,
}


def match_spectrum_to_library(
    query, library: list[RefSpectrum], *,
    tol_da: float = 0.02, precursor_tol_da: float = 0.02,
    metric: str = "entropy",
    min_score: float = 0.0, top_n: int = 5,
    require_precursor: bool = True,
) -> list[LibraryHit]:
    """Rank library reference spectra against an acquired query MS2 spectrum.

    Computes all three similarity metrics per reference; the primary ranking score
    is ``metric`` (``'entropy'`` | ``'cosine'`` | ``'modified_cosine'``). When
    ``require_precursor`` is True, only references within ``precursor_tol_da`` of
    the query precursor are scored; set it False for an analog search (best paired
    with ``metric='modified_cosine'``). Returns the top ``top_n`` hits with score
    ≥ ``min_score``, sorted by descending primary score (ties broken by descending
    ``n_matched``).
    """
    if metric not in _METRIC_FUNCS:
        raise ValueError(
            f"Unknown metric {metric!r}: expected one of {sorted(_METRIC_FUNCS)}."
        )
    q_pre = float(getattr(query, "precursor", 0.0) or 0.0)
    # Normalize the query once for the whole loop; per reference, normalize the ref
    # once and build the direct alignment once — shared by cosine + entropy + the
    # direct leg of modified_cosine (audit plan 23, item A).
    q_mz, q_in = _clean_normalize(*_as_arrays(query)[:2])
    hits: list[LibraryHit] = []
    for ref in library:
        if require_precursor:
            if abs(ref.precursor - q_pre) > precursor_tol_da:
                continue
        r_mz, r_in = _clean_normalize(*_as_arrays(ref)[:2])
        pairs = _align(q_mz, r_mz, tol_da=tol_da) if q_mz.size and r_mz.size else []
        r_pre = float(getattr(ref, "precursor", 0.0) or 0.0)
        cos_s, cos_n = _cosine_from(q_mz, q_in, r_mz, r_in, pairs)
        mod_s, mod_n = modified_cosine(
            query, ref, tol_da=tol_da,
            pre=(q_mz, q_in, r_mz, r_in, pairs, q_pre, r_pre))
        ent_s, ent_n = _entropy_from(q_mz, q_in, r_mz, r_in, pairs)
        primary = {"cosine": cos_s, "modified_cosine": mod_s, "entropy": ent_s}[metric]
        n_matched = {"cosine": cos_n, "modified_cosine": mod_n, "entropy": ent_n}[metric]
        if primary < min_score:
            continue
        hits.append(LibraryHit(
            ref=ref, score=primary, cosine=cos_s, modified_cosine=mod_s,
            entropy=ent_s, n_matched=n_matched,
        ))
    hits.sort(key=lambda h: (h.score, h.n_matched), reverse=True)
    return hits[:top_n] if top_n else hits


def confirm_with_library(
    lipid_class: str, query, library: list[RefSpectrum], *,
    strong_thr: float = 0.7, weak_thr: float = 0.5, mode: str = "negative",
    tol_da: float = 0.02, **kw,
) -> tuple[str, LibraryHit | None]:
    """Reconcile a library search with the rule-based class verdict.

    Returns ``(verdict, best_hit)``:

    * top hit's ``lipid_class`` equals ``lipid_class`` and ``score ≥ strong_thr``
      → ``"confirmed (library)"``;
    * ``score ≥ weak_thr`` but the class differs → ``"library: <name>?"`` (an
      alternate suggestion);
    * otherwise → fall through to :func:`msms.confirm_class` (the conservative
      rule tier), returned as that string verdict with the best hit (if any).
    """
    from . import msms

    hits = match_spectrum_to_library(
        query, library, tol_da=tol_da, top_n=1, min_score=0.0, **kw
    )
    best = hits[0] if hits else None
    if best is not None and best.score >= strong_thr and best.ref.lipid_class == lipid_class:
        return "confirmed (library)", best
    if best is not None and best.score >= weak_thr and best.ref.lipid_class != lipid_class:
        return f"library: {best.ref.name}?", best
    return msms.confirm_class(lipid_class, query, mode=mode, tol_da=tol_da), best
