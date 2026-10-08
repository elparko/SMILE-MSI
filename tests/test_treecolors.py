"""Tree-aware segment colours (:mod:`smile_msi.treecolors`) — siblings share a hue family,
an unsplit segment keeps its colour across cuts, and the colours ride a Segmentation into
the session file."""
import re

import numpy as np
import pytest
from scipy.cluster.hierarchy import linkage as scipy_linkage

from smile_msi import session, spatial, treecolors

_HEX = re.compile(r"^#[0-9a-f]{6}$")


def _hue_gap(h1, h2):
    d = abs(h1 - h2) % 360.0
    return min(d, 360.0 - d)


def _hue(hexc):
    return treecolors.hex_to_oklch(hexc)[2]


def _toy_hier(method="ward"):
    """Micro-cluster centroids in two well-separated super-groups (A, B), each made of two
    tighter sub-groups — so the cut at 2 is {A, B} and the cut at 4 is {A1, A2, B1, B2}.
    A has more pixels, so the size-ordered labels put it first."""
    rng = np.random.default_rng(0)
    centres = {"A1": (0, 0), "A2": (0, 6), "B1": (40, 0), "B2": (40, 6)}
    cent, sizes = [], []
    for name, (x, y) in centres.items():
        for _ in range(5):
            cent.append((x + rng.normal(0, 0.3), y + rng.normal(0, 0.3)))
            sizes.append(30 if name.startswith("A") else 10)
    cent = np.asarray(cent)
    micro = np.repeat(np.arange(len(cent)), sizes)
    if method == "bisecting":
        link = spatial._bisecting_linkage(cent, np.asarray(sizes, float), 0)
    else:
        link = scipy_linkage(cent, method="ward")
    return spatial.Hierarchy(scores=cent[micro], micro_labels=micro, linkage=link, peaks=[1.0],
                             explained_variance=1.0, n_micro=len(cent), method=method)


def test_node_colors_are_valid_and_deterministic():
    h = _toy_hier()
    cols = treecolors.node_colors(h.linkage)
    assert len(cols) == 2 * h.n_micro - 1
    assert all(_HEX.match(c) for c in cols)
    assert cols == treecolors.node_colors(h.linkage)


def test_single_leaf_tree():
    assert len(treecolors.node_colors(np.empty((0, 4)))) == 1


@pytest.mark.parametrize("method", ["ward", "bisecting"])
def test_siblings_share_a_hue_family(method):
    h = _toy_hier(method)
    labels4 = spatial.cut(h, 4)
    cols4 = spatial.segment_colors(h, 4)
    assert len(cols4) == 4 and len(set(cols4)) == 4
    # which k=4 segments sit inside which k=2 segment
    labels2 = spatial.cut(h, 2)
    parent = {int(c): int(labels2[labels4 == c][0]) for c in range(4)}
    hue = [_hue(c) for c in cols4]
    for a in range(4):
        for b in range(4):
            for c in range(4):
                if parent[a] == parent[b] != parent[c] and a != b:
                    assert _hue_gap(hue[a], hue[b]) < _hue_gap(hue[a], hue[c])


def test_unsplit_segment_keeps_its_colour_across_cuts():
    h = _toy_hier()
    labels2, labels3 = spatial.cut(h, 2), spatial.cut(h, 3)
    cols2, cols3 = spatial.segment_colors(h, 2), spatial.segment_colors(h, 3)
    for c3 in range(3):
        members = labels3 == c3
        (c2,) = np.unique(labels2[members])
        if np.array_equal(labels2 == c2, members):        # the segment that did not split
            assert cols3[c3] == cols2[c2]
            break
    else:
        pytest.fail("expected one segment to survive the 2 → 3 cut unchanged")


def test_cluster_colors_are_the_subtree_roots():
    h = _toy_hier()
    node_hex = treecolors.node_colors(h.linkage)
    n = h.n_micro
    # the cut at 1 is the root itself
    assert treecolors.cluster_colors(h.linkage, np.zeros(n, int), node_hex=node_hex) == \
        [node_hex[-1]]
    # the cut at n is every leaf
    assert treecolors.cluster_colors(h.linkage, np.arange(n), node_hex=node_hex) == \
        node_hex[:n]


def test_oklch_round_trip_and_gamut_clip():
    for hexc in ("#4c72b0", "#dd8452", "#55a868", "#ffffff", "#000000"):
        L, C, hh = treecolors.hex_to_oklch(hexc)
        assert treecolors.oklch_to_hex(L, C, hh) == hexc
    # an impossible chroma is clipped into sRGB rather than wrapping
    assert _HEX.match(treecolors.oklch_to_hex(0.7, 0.9, 140.0))


def test_shades_stay_in_the_parent_family():
    base = "#4c72b0"
    out = treecolors.shades(base, 4)
    assert len(set(out)) == 4 and all(_HEX.match(c) for c in out)
    assert all(_hue_gap(_hue(c), _hue(base)) < 30 for c in out)


class _DS:
    def __init__(self, n):
        self.n_pixels = n

    def to_image(self, v):
        return np.asarray(v, float).reshape(1, -1)


def test_segmentation_carries_tree_colours_into_the_session():
    h = _toy_hier()
    seg = spatial.segmentation_at(_DS(len(h.micro_labels)), h, 4)
    assert seg.colors == spatial.segment_colors(h, 4)
    s = session.build_session(source="x", settings={}, peaks=[], labels=seg.labels,
                              n_clusters=seg.n_clusters, seg_colors=seg.colors)
    assert s["segmentation"]["colors"] == seg.colors
    # k-means style segmentations carry no colours → the session stays as before
    s = session.build_session(source="x", settings={}, peaks=[], labels=seg.labels,
                              n_clusters=seg.n_clusters)
    assert "colors" not in s["segmentation"]
