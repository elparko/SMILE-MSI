"""Region-construction helpers in :mod:`smile_msi.spatial` — threshold → fill → ring →
invert on hand-built grids, so every pixel count is checkable by eye."""
import numpy as np
import pytest

from smile_msi import spatial


class _Grid:
    """A fully acquired H×W slide: pixel i sits at (i // W, i % W)."""

    def __init__(self, h, w, pixel_size_um=None):
        self.height, self.width = h, w
        self.n_pixels = h * w
        self.pixel_size_um = pixel_size_um
        rr, cc = np.mgrid[0:h, 0:w]
        self._rc = (rr.ravel(), cc.ravel())

    def _pixel_rows_cols(self):
        return self._rc

    def grid(self, mask):
        return np.asarray(mask, bool).reshape(self.height, self.width)


@pytest.fixture
def ds():
    return _Grid(7, 7)


def _square(ds, r0, r1, c0, c1):
    g = np.zeros((ds.height, ds.width), bool)
    g[r0:r1, c0:c1] = True
    return g.ravel()


def test_fill_holes_closes_an_enclosed_core(ds):
    ring = _square(ds, 1, 6, 1, 6) & ~_square(ds, 2, 5, 2, 5)      # 5×5 square, 3×3 hole
    assert int(ring.sum()) == 16
    filled = spatial.fill_holes(ds, ring)
    assert int(filled.sum()) == 25
    assert filled[ring].all()


def test_fill_holes_leaves_open_shapes_alone(ds):
    cup = _square(ds, 1, 6, 1, 6) & ~_square(ds, 2, 6, 2, 5)        # open at the bottom
    assert np.array_equal(spatial.fill_holes(ds, cup), cup)


def test_drop_small_removes_specks_keeps_bodies(ds):
    m = _square(ds, 0, 3, 0, 3)                                     # 9-px body
    m[ds.n_pixels - 1] = True                                       # a lone corner pixel
    m[3 * ds.width + 5] = True                                      # another lone pixel
    out = spatial.drop_small(ds, m, 4)
    assert int(out.sum()) == 9
    assert np.array_equal(spatial.drop_small(ds, m, 1), m)          # ≤1 is a no-op


def test_invert_mask_is_the_complement(ds):
    m = _square(ds, 0, 2, 0, 7)
    inv = spatial.invert_mask(ds, m)
    assert int(inv.sum()) == ds.n_pixels - 14
    assert not (inv & m).any()
    with pytest.raises(ValueError):
        spatial.invert_mask(ds, np.zeros(3, bool))


def test_threshold_percentile_ignores_zero_pixels(ds):
    vals = np.zeros(ds.n_pixels)
    core = _square(ds, 2, 5, 2, 5)                                  # 9 signal pixels
    vals[core] = np.arange(1, 10)                                   # 1..9
    m = spatial.threshold_mask(ds, vals, 50, fill_holes=False)      # 50th pct of 1..9 = 5
    assert int(m.sum()) == 5                                        # 5, 6, 7, 8, 9
    assert not m[~core].any()
    m0 = spatial.threshold_mask(ds, vals, 0, fill_holes=False)      # ≥ min → every signal px
    assert int(m0.sum()) == 9


def test_threshold_absolute_and_nonfinite(ds):
    vals = np.full(ds.n_pixels, 2.0)
    vals[:5] = np.nan
    vals[5:10] = 10.0
    m = spatial.threshold_mask(ds, vals, 5.0, percentile=False, fill_holes=False)
    assert int(m.sum()) == 5 and m[5:10].all()
    assert not spatial.threshold_mask(ds, np.zeros(ds.n_pixels), 60).any()   # no signal → empty


def test_threshold_fills_holes_and_drops_islands(ds):
    vals = np.zeros(ds.n_pixels)
    ring = _square(ds, 1, 6, 1, 6) & ~_square(ds, 2, 5, 2, 5)
    vals[ring] = 10.0
    vals[0] = 10.0                                                  # a lone bright pixel
    m = spatial.threshold_mask(ds, vals, 0, fill_holes=True, min_pixels=2)
    assert int(m.sum()) == 25                                       # solid square, speck gone
    raw = spatial.threshold_mask(ds, vals, 0, fill_holes=False)
    assert int(raw.sum()) == 17


def test_ring_mask_widths_on_a_filled_square(ds):
    sq = _square(ds, 1, 6, 1, 6)                                    # 5×5 = 25 px
    inner = spatial.ring_mask(ds, sq, 1, mode="inner")
    assert int(inner.sum()) == 16                                   # 25 − 3×3 core
    outer = spatial.ring_mask(ds, sq, 1, mode="outer")
    # a 1-px euclidean collar: the 4-neighbour frame (20 px) without the diagonal corners
    assert int(outer.sum()) == 20
    assert not (outer & sq).any()
    band = spatial.ring_mask(ds, sq, 1, mode="band")
    assert int(band.sum()) == 36
    assert np.array_equal(spatial.ring_mask(ds, sq, 0, mode="outer"), sq)   # width 0 = as is


def test_compartments_by_construction(ds):
    """endo = threshold → peri = outer collar → epi = everything else: disjoint, complete."""
    vals = np.zeros(ds.n_pixels)
    vals[_square(ds, 2, 5, 2, 5)] = 5.0
    endo = spatial.threshold_mask(ds, vals, 0)
    peri = spatial.ring_mask(ds, endo, 1, mode="outer")
    epi = spatial.invert_mask(ds, endo | peri)
    assert int(endo.sum()) == 9 and int(peri.sum()) == 12
    assert not (endo & peri).any() and not (epi & (endo | peri)).any()
    assert int(endo.sum() + peri.sum() + epi.sum()) == ds.n_pixels
