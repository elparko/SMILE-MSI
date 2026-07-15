"""Synthetic MALDI-MSI dataset generator.

Builds a small, fully in-memory :class:`~smile_msi.msi.MSIDataset` shaped like a
coronal nerve/brain section: an elliptical tissue mask split into a myelin-rich
"white matter" rim, a "gray matter" core, and a focal "lesion" spot. Each region
is enriched in a different set of lipids whose m/z come straight from the
in-silico database, so peak picking, segmentation, co-localization, ROI
statistics, *and* lipid annotation all produce meaningful results with no real
data and no network.

Used by the Streamlit app's "Load demo" button and by the test suite.
"""
from __future__ import annotations

import numpy as np

from .lipiddb import build_database
from .masses import ion_mz
from .msi import MSIDataset


def _demo_peaks():
    """Return [(mz, region_weights)] using real in-silico [M-H]-/[M-CH3]- masses.

    region_weights = (white_matter, gray_matter, lesion) relative abundances.
    """
    db = build_database()
    spec = [
        # species,            adduct,        (WM,  GM,  lesion)
        ("Sulfatide 42:2;O3", "[M-H]-",      (1.0, 0.15, 0.2)),   # myelin marker -> white matter
        ("Sulfatide 42:1;O2", "[M-H]-",      (0.9, 0.1,  0.1)),
        ("HexCer 42:2;O3",    "[M-H]-",      (0.8, 0.2,  0.2)),   # galactosylceramide -> white matter
        ("PE 38:4",           "[M-H]-",      (0.4, 1.0,  0.5)),   # gray-matter enriched
        ("PS 40:6",           "[M-H]-",      (0.2, 1.0,  0.3)),   # gray-matter enriched
        ("PI 38:4",           "[M-H]-",      (0.3, 0.9,  0.4)),
        ("PC 34:1",           "[M-CH3]-",    (0.6, 0.6,  0.6)),   # ubiquitous membrane lipid
        ("FA 18:1",           "[M-H]-",      (0.5, 0.5,  1.0)),   # free fatty acid -> lesion
        ("FA 22:6",           "[M-H]-",      (0.3, 0.6,  1.0)),   # DHA -> lesion / injury
        ("LPA 18:1",          "[M-H]-",      (0.1, 0.2,  1.0)),   # lyso species -> lesion
    ]
    db_index = {lip.name: lip for lip in db}
    out = []
    for name, adduct, weights in spec:
        lip = db_index[name]
        out.append((ion_mz(lip.neutral_mass, adduct), np.array(weights), name))
    return out


def make_synthetic(width: int = 44, height: int = 36, resolution: float = 20000.0,
                   mz_step: float = 0.01, noise: float = 0.02, seed: int = 0) -> MSIDataset:
    """Generate a synthetic negative-mode section as a continuous-mode MSIDataset.

    Profile spectra share one m/z axis (the imzML "continuous" case). Each peak is
    a Gaussian at instrument resolution ``resolution``; per-pixel intensity follows
    the region map plus multiplicative noise. Off-tissue pixels are still emitted
    (near-zero signal) so normalization and tissue masking behave realistically.

    The 44×36 grid (~1.6k pixels) keeps the three-region tissue (white-matter rim,
    gray-matter core, focal lesion) clearly resolved while holding the in-RAM cube to
    ~half of the old 60×48 — the per-pixel profile axis is ~63k bins, so every extra
    pixel costs ~250 KB and the GUI test suite builds the demo in dozens of windows.
    The mz_step stays at 0.01: the lowest-mass peak's FWHM is ~0.014 Da, so a coarser
    grid would under-sample it. Engine tests that need a specific size pass it explicitly.
    """
    rng = np.random.default_rng(seed)
    peaks = _demo_peaks()
    centers = np.array([p[0] for p in peaks])

    # Shared profile m/z axis spanning the peaks with margin.
    lo, hi = centers.min() - 3.0, centers.max() + 3.0
    axis = np.arange(lo, hi, mz_step)

    # ----- region map ------------------------------------------------------ #
    yy, xx = np.mgrid[0:height, 0:width]
    cy, cx = height / 2, width / 2
    rx, ry = width * 0.45, height * 0.45
    ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    tissue = ellipse <= 1.0
    inner = ellipse <= 0.45                       # gray-matter core
    white = tissue & ~inner                       # myelin-rich rim
    # focal lesion: a small disc offset into the gray matter
    lx, ly = cx + width * 0.12, cy - height * 0.10
    lesion = ((xx - lx) ** 2 + (yy - ly) ** 2) <= (min(width, height) * 0.10) ** 2
    lesion &= tissue

    # per-pixel (WM, GM, lesion) membership, soft-ish
    w_wm = white.astype(float)
    w_gm = inner.astype(float)
    w_les = lesion.astype(float)
    w_gm = np.where(lesion, 0.3, w_gm)            # lesion sits within gray matter

    coords, mzs, ints = [], [], []
    sigma = np.array([c / resolution / 2.355 for c in centers])   # FWHM = mz/R
    for r in range(height):
        for c in range(width):
            coords.append((c + 1, r + 1))         # imzML is 1-indexed
            if not tissue[r, c]:
                ints.append((rng.random(len(axis)) * noise * 0.05).astype(np.float32))
                mzs.append(axis)
                continue
            wts = np.array([w_wm[r, c], w_gm[r, c], w_les[r, c]])
            spec = np.zeros_like(axis)
            for k, (_mz, region_w, _name) in enumerate(peaks):
                amp = float(region_w @ wts) * (1.0 + noise * rng.standard_normal())
                amp = max(amp, 0.0)
                n_c = max(1, int(round(centers[k] / 14.0)))        # ~CH2 per 14 Da
                m1 = 0.011 * n_c                                   # M+1 / M0
                for iso, rel in ((0, 1.0), (1, m1), (2, 0.5 * m1 * m1)):
                    mu = centers[k] + iso * 1.0033548
                    half = 6.0 * sigma[k]                          # ±6σ window; tail beyond is ~0
                    a = int(np.searchsorted(axis, mu - half, "left"))
                    b = int(np.searchsorted(axis, mu + half, "right"))
                    if b > a:                                      # add the Gaussian only where it matters
                        spec[a:b] += amp * rel * np.exp(-0.5 * ((axis[a:b] - mu) / sigma[k]) ** 2)
            spec += rng.random(len(axis)) * noise * 0.1            # baseline noise
            ints.append(spec.astype(np.float32))
            mzs.append(axis)

    ds = MSIDataset.from_arrays(coords, mzs, ints, polarity="negative",
                                spec_mode="profile", source="synthetic")
    return ds


def write_synthetic_imzml(path: str, **kwargs) -> str:
    """Write a synthetic dataset to ``path`` (.imzML + .ibd) for round-trip testing."""
    from .ingest import to_imzml
    return to_imzml(make_synthetic(**kwargs), path)
