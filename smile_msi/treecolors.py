"""Tree-aware segment colours — related segments get related hues.

A segmentation cut from a :class:`~smile_msi.spatial.Hierarchy` used to be painted from a
fixed, cycling categorical palette (``PALETTE[cl % 12]``): two sibling segments that are
chemically near-identical could land on unrelated colours, and past 12 segments the colours
repeated. This module colours the **tree** instead, so colour carries the hierarchy:

* every tree node owns a slice of the hue circle; its children split that slice between
  them, so siblings sit on neighbouring hues and a whole branch reads as one colour family;
* a segment is painted with the colour of the node it *is* (the subtree root the cut
  produced), so dragging the Detail slider finer splits a segment into shades *of its own
  hue* rather than reshuffling every colour on the map.

This is the "Tree Colors" scheme of Tennekes & de Jonge (2014, *IEEE TVCG* 20:2072,
doi:10.1109/TVCG.2014.2346277) — recursive hue-range partitioning with a gap between
siblings and depth-dependent lightness/chroma — computed in **OKLCh** (Ottosson 2020), a
perceptually uniform polar space, then gamut-clipped to sRGB by reducing chroma. It is the
colouring idea behind hierarchical lipid-territory atlases such as the "lipizones" of
Fusar Bassini et al. (bioRxiv 2025, doi:10.1101/2025.10.13.682018).

Two refinements for MSI granularity trees, which are often unbalanced (Ward peels off small
outlier groups early):

* **Hue is allocated by how many segments a branch holds at a reference cut** (``ref_k``),
  not split 50/50 — otherwise a tiny outlier branch would take half the wheel and the
  segments users actually look at (around the default Detail) would crowd into a sliver.
* **The two children of a split alternate lightness** (as the EUCLID atlas code alternates
  saturation), so deep siblings whose hues have converged stay distinguishable.

Pure NumPy, no Qt, deterministic (no randomness at all).
"""
from __future__ import annotations

import numpy as np

#: Fraction of a parent's hue slice its children use; the rest is a gap between siblings.
FRACTION = 0.8
#: Hue (degrees) where the root's slice starts.
START_HUE = 20.0
#: Reference cut the hue allocation is tuned for (clamped to the tree size).
REF_K = 24


# --------------------------------------------------------------------------- #
# OKLCh -> sRGB
# --------------------------------------------------------------------------- #
def _oklab_to_linear_srgb(L, a, b):
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    return np.array([
        +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    ])


def _in_gamut(rgb, eps=1e-6):
    return bool(np.all(rgb >= -eps) and np.all(rgb <= 1 + eps))


def oklch_to_hex(L: float, C: float, h: float) -> str:
    """OKLCh (``L`` 0..1, chroma ``C``, hue ``h`` degrees) → ``#rrggbb``. Out-of-gamut
    colours keep their lightness and hue and lose chroma until they fit sRGB."""
    rad = np.deg2rad(h)
    ca, sa = np.cos(rad), np.sin(rad)
    rgb = _oklab_to_linear_srgb(L, C * ca, C * sa)
    if not _in_gamut(rgb):
        lo, hi = 0.0, float(C)
        for _ in range(24):                          # bisect the largest in-gamut chroma
            mid = 0.5 * (lo + hi)
            if _in_gamut(_oklab_to_linear_srgb(L, mid * ca, mid * sa)):
                lo = mid
            else:
                hi = mid
        rgb = _oklab_to_linear_srgb(L, lo * ca, lo * sa)
    rgb = np.clip(rgb, 0.0, 1.0)
    srgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(rgb, 1 / 2.4) - 0.055)
    r, g, b = (int(round(float(v) * 255)) for v in np.clip(srgb, 0.0, 1.0))
    return f"#{r:02x}{g:02x}{b:02x}"


# --------------------------------------------------------------------------- #
# tree colouring
# --------------------------------------------------------------------------- #
def _ref_weights(linkage: np.ndarray, n: int, ref_k: int) -> np.ndarray:
    """Per node (``2n-1``, scipy ids): how many segments of the cut at ``ref_k`` lie in
    its subtree, at least 1. Linkage rows are in merge order, so the cut at ``k`` undoes
    the last ``k-1`` rows; nodes created by those rows are *split*, every other node is
    (inside) one segment."""
    n_nodes = 2 * n - 1
    k = int(max(1, min(ref_k, n)))
    split = np.zeros(n_nodes, dtype=bool)
    if k > 1:
        split[n + (n - 1) - (k - 1):] = True              # nodes of the last k-1 rows
    w = np.ones(n_nodes, dtype=float)
    for r in range(n - 1):                                 # children before parents
        node = n + r
        if split[node]:
            a, b = int(linkage[r, 0]), int(linkage[r, 1])
            w[node] = (w[a] if split[a] else 1.0) + (w[b] if split[b] else 1.0)
    return w


def node_colors(linkage, *, ref_k: int = REF_K, start_hue: float = START_HUE,
                fraction: float = FRACTION) -> list[str]:
    """A hex colour for every node of a scipy-format ``linkage`` (``n-1`` rows over ``n``
    leaves) → list of ``2n-1`` colours indexed by scipy node id (leaves ``0..n-1``,
    the node created by row ``r`` is ``n + r``).

    A node's hue is the centre of its hue slice; lightness falls and chroma rises gently
    with depth (coarse branches pale, fine ones richer), and the second child of every
    split is set a step darker than the first so close siblings still separate."""
    link = np.asarray(linkage, dtype=float)
    n = int(link.shape[0]) + 1
    if link.shape[0] == 0:
        return [oklch_to_hex(0.74, 0.12, start_hue + 180.0)]
    n_nodes = 2 * n - 1
    w = _ref_weights(link, n, ref_k)
    lo = np.zeros(n_nodes)
    hi = np.zeros(n_nodes)
    depth = np.zeros(n_nodes, dtype=int)
    second = np.zeros(n_nodes, dtype=bool)                 # the 2nd child of its split
    root = n_nodes - 1
    lo[root], hi[root] = start_hue, start_hue + 360.0
    for r in range(n - 2, -1, -1):                         # parents before children
        node = n + r
        a, b = int(link[r, 0]), int(link[r, 1])
        span = hi[node] - lo[node]
        cut = lo[node] + span * w[a] / (w[a] + w[b])
        for child, c_lo, c_hi in ((a, lo[node], cut), (b, cut, hi[node])):
            mid, half = 0.5 * (c_lo + c_hi), 0.5 * (c_hi - c_lo) * fraction
            lo[child], hi[child] = mid - half, mid + half
            depth[child] = depth[node] + 1
        second[b] = True
    d = np.minimum(depth, 6)
    L = 0.80 - 0.035 * d - np.where(second, 0.06, 0.0)
    C = 0.10 + 0.012 * d
    hue = (0.5 * (lo + hi)) % 360.0
    return [oklch_to_hex(float(L[i]), float(C[i]), float(hue[i])) for i in range(n_nodes)]


def cluster_colors(linkage, micro_to_macro, node_hex=None, **kw) -> list[str]:
    """Colour each cluster of a cut: ``micro_to_macro`` maps every leaf (micro-cluster) of
    ``linkage`` to its cluster id ``0..k-1``; cluster ``c`` gets the colour of the subtree
    root whose leaves are exactly its members. ``node_hex`` reuses a precomputed
    :func:`node_colors` result (the per-tick path of a live Detail slider)."""
    link = np.asarray(linkage, dtype=float)
    m2m = np.asarray(micro_to_macro, dtype=int)
    n = int(link.shape[0]) + 1
    if node_hex is None:
        node_hex = node_colors(link, **kw)
    k = int(m2m.max()) + 1 if m2m.size else 0
    if n <= 1 or k == 0:
        return list(node_hex[:1]) * k
    parent = np.full(2 * n - 1, -1, dtype=int)
    count = np.ones(2 * n - 1, dtype=int)
    for r in range(n - 1):
        a, b = int(link[r, 0]), int(link[r, 1])
        parent[a] = parent[b] = n + r
        count[n + r] = count[a] + count[b]
    size = np.bincount(m2m, minlength=k)
    first_leaf = np.full(k, -1, dtype=int)
    for leaf in range(len(m2m) - 1, -1, -1):
        first_leaf[m2m[leaf]] = leaf
    out = []
    for c in range(k):
        node = int(first_leaf[c])
        if node < 0:                                       # empty id (shouldn't happen)
            out.append(node_hex[-1])
            continue
        # climb while the ancestor still lies wholly inside cluster c (cut clusters are
        # whole subtrees, so leaf count alone decides it)
        while parent[node] >= 0 and count[parent[node]] <= size[c]:
            node = int(parent[node])
        out.append(node_hex[node])
    return out


def hex_to_oklch(hexc: str) -> tuple[float, float, float]:
    """``#rrggbb`` → OKLCh ``(L, C, h°)`` — the inverse of :func:`oklch_to_hex`."""
    h = hexc.lstrip("#")
    srgb = np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    r, g, b = lin
    l_ = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m_ = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s_ = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    bb = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return float(L), float(np.hypot(a, bb)), float(np.degrees(np.arctan2(bb, a)) % 360.0)


def shades(hexc: str, n: int) -> list[str]:
    """``n`` variants of ``hexc`` for the children of a manual split: same hue family,
    stepping lighter / darker in turn (and nudging hue a few degrees) so the sub-segments
    read as belonging to their parent yet stay distinguishable."""
    L0, C0, h0 = hex_to_oklch(hexc)
    out = []
    for j in range(1, int(n) + 1):
        sign = 1.0 if j % 2 else -1.0
        step = (j + 1) // 2
        L = float(np.clip(L0 + sign * 0.08 * step, 0.38, 0.92))
        out.append(oklch_to_hex(L, max(C0, 0.06), h0 + sign * 7.0 * step))
    return out
