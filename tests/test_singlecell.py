"""Engine tests for smile_msi.singlecell (single-cell spatial metabolomics).

Headless, synthetic data only — no real microscopy / imzML. We test the load-bearing
engine logic: watershed splitting of touching blobs, area-weighted pixel->cell
apportioning on a synthetic mask + grid, per-cell matrix shape/values, orientation
invariance of the mapping, graceful behaviour when pixel sizes are None, and the
clear-error path for an absent optional backend.
"""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from smile_msi import singlecell as sc
from smile_msi.msi import MSIDataset


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _disc(img, cy, cx, r, val):
    yy, xx = np.ogrid[: img.shape[0], : img.shape[1]]
    img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = val


def _make_ds(coords, per_pixel_amp, axis=None):
    """A tiny MSIDataset: each pixel is a flat spectrum scaled by per_pixel_amp."""
    if axis is None:
        axis = np.linspace(100.0, 200.0, 64)
    ints = [np.full(axis.size, float(a), dtype=np.float32) for a in per_pixel_amp]
    ds = MSIDataset.from_arrays(coords, [axis] * len(coords), ints,
                                spec_mode="profile")
    return ds, axis


def _grid_transform(foot_opt):
    """optical-px -> MSI-base-px affine for a grid where MSI pixel (x,y) tiles the
    optical block [x*foot, (x+1)*foot). base = optical/foot - 0.5+0.5/foot ... we
    derive from: optical_center(base=k) = k*foot + (foot-1)/2."""
    s = 1.0 / float(foot_opt)
    t = -((foot_opt - 1.0) / 2.0) / float(foot_opt)
    return np.array([[s, 0.0, t], [0.0, s, t], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------- #
# 1. Segmentation
# --------------------------------------------------------------------------- #
def test_watershed_three_separated_blobs():
    img = np.zeros((60, 90), dtype=np.uint8)
    _disc(img, 30, 15, 8, 255)
    _disc(img, 30, 45, 8, 255)
    _disc(img, 30, 75, 8, 255)
    cs = sc.segment_cells(img, backend="watershed", min_diameter_um=4.0)
    assert cs.n_cells == 3
    assert cs.centroids_opt.shape == (3, 2)
    assert (cs.areas_opt > 0).all()


def test_watershed_splits_touching_blobs():
    """Two blobs whose discs touch must be split into 2 cells by the distance-
    transform markers (a plain connected-components labeller would merge them)."""
    img = np.zeros((60, 100), dtype=np.uint8)
    _disc(img, 30, 38, 12, 255)
    _disc(img, 30, 62, 12, 255)          # rims touch around x=50
    # confirm they are a single connected component (the hard case)
    from scipy import ndimage as ndi
    _, ncc = ndi.label(img > 0)
    assert ncc == 1
    cs = sc.segment_cells(img, backend="watershed", min_diameter_um=6.0)
    assert cs.n_cells == 2


def test_watershed_drops_tiny_specks():
    img = np.zeros((60, 80), dtype=np.uint8)
    _disc(img, 30, 20, 9, 255)           # real cell
    _disc(img, 10, 70, 1, 255)           # a 1-2 px speck below min area
    cs = sc.segment_cells(img, backend="watershed", min_diameter_um=8.0)
    assert cs.n_cells == 1


def test_segment_unknown_backend_raises():
    img = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        sc.segment_cells(img, backend="nope")


def test_segment_cellpose_missing_raises(monkeypatch):
    """The optional cellpose backend, when absent, raises a clear RuntimeError naming
    the install command — never a silent fallback to watershed."""
    img = np.zeros((20, 20), dtype=np.uint8)
    _disc(img, 10, 10, 5, 255)
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "cellpose" or name.startswith("cellpose."):
            raise ImportError("no cellpose")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError) as ei:
        sc.segment_cells(img, backend="cellpose")
    assert "cellpose" in str(ei.value)


# --------------------------------------------------------------------------- #
# 2. Pixel -> cell area weighting
# --------------------------------------------------------------------------- #
def _one_cell_over_grid(foot_opt=10, gx=2, gy=2):
    """A single cell covering the whole optical grid that backs a gx*gy MSI grid."""
    H = gy * foot_opt
    W = gx * foot_opt
    lab = np.ones((H, W), dtype=np.int32)
    cells = sc.CellSet(
        label_image=lab, n_cells=1,
        centroids_opt=np.array([[(H - 1) / 2.0, (W - 1) / 2.0]]),
        areas_opt=np.array([float(H * W)]), labels=np.array([1]),
        backend="watershed", params={})
    coords = [(x + 1, y + 1) for y in range(gy) for x in range(gx)]
    return cells, coords


def test_area_weighting_cell_over_four_pixels_equal_weights():
    cells, coords = _one_cell_over_grid(foot_opt=10, gx=2, gy=2)
    ds, _ = _make_ds(coords, [1.0, 2.0, 3.0, 4.0])
    ds.pixel_size_um = 10.0
    M = _grid_transform(10)
    m = sc.map_pixels_to_cells(ds, cells, M, optical_pixel_size_um=1.0,
                               weighting="area")
    assert m.n_cells == 1
    w = m.cell_weights[0]
    assert np.allclose(w.sum(), 1.0)
    assert np.allclose(w, 0.25, atol=1e-9)        # 4 equal pixels
    assert np.allclose(m.cell_support[0], 4.0, atol=1e-6)
    assert (m.cells_per_pixel == 1).all()


def test_area_weighting_straddle_two_pixels_proportional():
    """A cell occupying 75% of pixel 0's footprint and 25% of pixel 1's gets weights
    in a 3:1 ratio (area-proportional)."""
    foot = 20
    H, W = foot, 2 * foot
    lab = np.zeros((H, W), dtype=np.int32)
    # cell spans columns [foot-15, foot+5): 15 cols in pixel0 footprint, 5 in pixel1
    lab[:, foot - 15: foot + 5] = 1
    cells = sc.CellSet(
        label_image=lab, n_cells=1, centroids_opt=np.array([[(H - 1) / 2.0, foot - 5.0]]),
        areas_opt=np.array([float(lab.sum())]), labels=np.array([1]),
        backend="watershed", params={})
    coords = [(1, 1), (2, 1)]
    ds, _ = _make_ds(coords, [5.0, 9.0])
    ds.pixel_size_um = float(foot)
    M = _grid_transform(foot)
    m = sc.map_pixels_to_cells(ds, cells, M, optical_pixel_size_um=1.0,
                               weighting="area")
    w = m.cell_weights[0]
    pix = m.cell_pixels[0]
    assert set(pix.tolist()) == {0, 1}
    # order by pixel index
    order = np.argsort(pix)
    w0, w1 = w[order]
    assert np.isclose(w0 / (w0 + w1), 0.75, atol=0.03)
    assert np.isclose(w1 / (w0 + w1), 0.25, atol=0.03)


def test_cells_per_pixel_counts_two_cells_in_one_pixel():
    """One MSI pixel whose footprint is split between two cells -> cells_per_pixel==2."""
    foot = 20
    lab = np.zeros((foot, foot), dtype=np.int32)
    lab[:, : foot // 2] = 1               # left half = cell 1
    lab[:, foot // 2:] = 2                # right half = cell 2
    uniq, cen, ar = sc._label_props(lab)
    cells = sc.CellSet(label_image=lab, n_cells=2, centroids_opt=cen, areas_opt=ar,
                       labels=uniq, backend="watershed", params={})
    ds, _ = _make_ds([(1, 1)], [7.0])
    ds.pixel_size_um = float(foot)
    M = _grid_transform(foot)
    m = sc.map_pixels_to_cells(ds, cells, M, optical_pixel_size_um=1.0,
                               weighting="area")
    assert m.n_cells == 2
    assert m.cells_per_pixel[0] == 2
    # each cell gets the single pixel, weight 1 (only one pixel touches it)
    for ci in range(2):
        assert m.cell_pixels[ci].tolist() == [0]
        assert np.allclose(m.cell_weights[ci], 1.0)


# --------------------------------------------------------------------------- #
# 3. Per-cell matrix
# --------------------------------------------------------------------------- #
def test_cell_matrix_area_weighted_mean_and_shape():
    cells, coords = _one_cell_over_grid(foot_opt=10, gx=2, gy=2)
    # use distinct peak intensities per pixel so the weighted mean is checkable
    axis = np.linspace(100.0, 200.0, 64)
    j = 32
    mz = float(axis[j])
    amps = [1.0, 2.0, 3.0, 4.0]
    ints = []
    for a in amps:
        s = np.zeros(axis.size, dtype=np.float32)
        s[j] = float(a)                   # single bin spike
        ints.append(s)
    ds = MSIDataset.from_arrays(coords, [axis] * 4, ints, spec_mode="profile")
    ds.pixel_size_um = 10.0
    M = _grid_transform(10)
    m = sc.map_pixels_to_cells(ds, cells, M, optical_pixel_size_um=1.0)
    ds.build_features([mz], tol_ppm=300.0, reduce="sum")
    cm = sc.build_cell_matrix(ds, m, tol_ppm=300.0, reduce="sum",
                              pixel_norm="none", cell_norm="none", min_support=0.5)
    assert cm.matrix.shape == (1, 1)
    # area-weighted mean of {1,2,3,4} with equal 0.25 weights = 2.5
    assert np.isclose(cm.matrix[0, 0], 2.5, atol=1e-6)
    assert cm.peaks.tolist() == [mz]
    assert cm.cell_ids.tolist() == [1]


def test_cell_matrix_peaks_default_from_feature_peaks():
    cells, coords = _one_cell_over_grid(foot_opt=8, gx=2, gy=2)
    ds, axis = _make_ds(coords, [1.0, 1.0, 1.0, 1.0])
    ds.pixel_size_um = 8.0
    M = _grid_transform(8)
    m = sc.map_pixels_to_cells(ds, cells, M, optical_pixel_size_um=1.0)
    ds.build_features([120.0, 180.0], tol_ppm=400.0)
    # peaks=None -> defaults to ds.feature_peaks
    cm = sc.build_cell_matrix(ds, m, peaks=None, tol_ppm=400.0,
                              pixel_norm="none", min_support=0.5)
    assert cm.matrix.shape == (1, 2)
    assert np.allclose(cm.peaks, [120.0, 180.0])


def test_cell_matrix_missing_peaks_raises():
    cells, coords = _one_cell_over_grid(foot_opt=8, gx=2, gy=2)
    ds, _ = _make_ds(coords, [1.0, 1.0, 1.0, 1.0])
    ds.pixel_size_um = 8.0
    M = _grid_transform(8)
    m = sc.map_pixels_to_cells(ds, cells, M, optical_pixel_size_um=1.0)
    # no build_features and no explicit peaks -> ValueError
    with pytest.raises(ValueError):
        sc.build_cell_matrix(ds, m, peaks=None)


def test_cell_matrix_cell_norm_tic_rows_sum_to_one():
    cells, coords = _one_cell_over_grid(foot_opt=8, gx=2, gy=2)
    axis = np.linspace(100.0, 200.0, 64)
    ints = []
    for a in (1.0, 2.0, 3.0, 4.0):
        s = np.zeros(axis.size, dtype=np.float32)
        s[10] = a
        s[40] = 2 * a
        ints.append(s)
    ds = MSIDataset.from_arrays(coords, [axis] * 4, ints, spec_mode="profile")
    ds.pixel_size_um = 8.0
    M = _grid_transform(8)
    m = sc.map_pixels_to_cells(ds, cells, M, optical_pixel_size_um=1.0)
    ds.build_features([axis[10], axis[40]], tol_ppm=400.0)
    cm = sc.build_cell_matrix(ds, m, tol_ppm=400.0, pixel_norm="none",
                              cell_norm="tic", min_support=0.5)
    assert np.isclose(cm.matrix.sum(axis=1)[0], 1.0, atol=1e-6)


def test_cell_matrix_min_support_drops_barely_touched_cell():
    """A second cell that only nicks a corner (tiny overlap area) is dropped by
    min_support while the well-covered cell is kept."""
    foot = 20
    H, W = foot, 2 * foot
    lab = np.zeros((H, W), dtype=np.int32)
    lab[:, :foot] = 1                     # cell 1 fully covers pixel 0
    lab[0:2, foot:foot + 2] = 2           # cell 2: a 2x2 nick of pixel 1
    uniq, cen, ar = sc._label_props(lab)
    cells = sc.CellSet(label_image=lab, n_cells=2, centroids_opt=cen, areas_opt=ar,
                       labels=uniq, backend="watershed", params={})
    coords = [(1, 1), (2, 1)]
    ds, _ = _make_ds(coords, [1.0, 1.0])
    ds.pixel_size_um = float(foot)
    M = _grid_transform(foot)
    m = sc.map_pixels_to_cells(ds, cells, M, optical_pixel_size_um=1.0)
    ds.build_features([150.0], tol_ppm=400.0)
    cm = sc.build_cell_matrix(ds, m, tol_ppm=400.0, pixel_norm="none",
                              min_support=0.5)
    # cell 1 (support ~1.0) kept; cell 2 (support ~4/400=0.01) dropped
    assert cm.cell_ids.tolist() == [1]


# --------------------------------------------------------------------------- #
# 4. Orientation invariance + None pixel sizes
# --------------------------------------------------------------------------- #
def test_mapping_orientation_invariant():
    cells, coords = _one_cell_over_grid(foot_opt=10, gx=3, gy=2)
    ds, _ = _make_ds(coords, [float(i) for i in range(6)])
    ds.pixel_size_um = 10.0
    M = _grid_transform(10)
    m0 = sc.map_pixels_to_cells(ds, cells, M, optical_pixel_size_um=1.0)
    ds.set_orientation(1)                 # rotate the display
    m1 = sc.map_pixels_to_cells(ds, cells, M, optical_pixel_size_um=1.0)
    # cell->pixel assignment is identical (mapping never reads ds.orientation)
    assert m0.n_cells == m1.n_cells
    for a, b in zip(m0.cell_pixels, m1.cell_pixels):
        assert a.tolist() == b.tolist()
    for a, b in zip(m0.cell_weights, m1.cell_weights):
        assert np.allclose(a, b)


def test_mapping_pixel_size_none_falls_back_to_transform_scale():
    cells, coords = _one_cell_over_grid(foot_opt=10, gx=2, gy=2)
    ds, _ = _make_ds(coords, [1.0, 2.0, 3.0, 4.0])
    ds.pixel_size_um = None               # imzML without 'pixel size x'
    M = _grid_transform(10)
    # optical_pixel_size_um None too -> footprint derived from transform scale (10)
    m = sc.map_pixels_to_cells(ds, cells, M, optical_pixel_size_um=None,
                               weighting="area")
    assert m.n_cells == 1
    w = m.cell_weights[0]
    assert np.allclose(w.sum(), 1.0)
    assert np.allclose(w, 0.25, atol=1e-9)


def test_centroid_weighting_assigns_one_pixel_per_cell():
    foot = 10
    lab = np.zeros((foot, 2 * foot), dtype=np.int32)
    lab[:, : foot] = 1
    lab[:, foot:] = 2
    uniq, cen, ar = sc._label_props(lab)
    cells = sc.CellSet(label_image=lab, n_cells=2, centroids_opt=cen, areas_opt=ar,
                       labels=uniq, backend="watershed", params={})
    coords = [(1, 1), (2, 1)]
    ds, _ = _make_ds(coords, [1.0, 1.0])
    ds.pixel_size_um = float(foot)
    M = _grid_transform(foot)
    m = sc.map_pixels_to_cells(ds, cells, M, optical_pixel_size_um=1.0,
                               weighting="centroid")
    # cell 1 centroid in pixel 0, cell 2 centroid in pixel 1
    assert m.cell_pixels[0].tolist() == [0]
    assert m.cell_pixels[1].tolist() == [1]
    assert np.allclose(m.cell_weights[0], 1.0)


def test_map_bad_weighting_and_singular_transform_raise():
    cells, coords = _one_cell_over_grid(foot_opt=8, gx=2, gy=2)
    ds, _ = _make_ds(coords, [1.0, 1.0, 1.0, 1.0])
    M = _grid_transform(8)
    with pytest.raises(ValueError):
        sc.map_pixels_to_cells(ds, cells, M, weighting="bogus")
    singular = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        sc.map_pixels_to_cells(ds, cells, singular)


# --------------------------------------------------------------------------- #
# 5. Cell embedding + clustering
# --------------------------------------------------------------------------- #
def _cellmat(n_cells=24, n_peaks=6, seed=0):
    rng = np.random.default_rng(seed)
    # two blobs in feature space so clustering has something to find
    a = rng.normal(0.0, 0.1, size=(n_cells // 2, n_peaks)) + 1.0
    b = rng.normal(0.0, 0.1, size=(n_cells - n_cells // 2, n_peaks)) + 5.0
    X = np.vstack([a, b]).clip(0, None)
    return sc.CellMatrix(matrix=X, cell_ids=np.arange(n_cells),
                         peaks=np.linspace(100, 200, n_peaks),
                         centroids_opt=rng.random((n_cells, 2)) * 50,
                         support=np.ones(n_cells))


def test_cell_segments_shapes_and_determinism():
    cm = _cellmat(24, 6)
    coords1, labels1, used1 = sc.cell_segments(cm, method="umap", k=2,
                                               random_state=0)
    assert coords1.shape == (24, 2)
    assert labels1.shape == (24,)
    assert len(set(labels1.tolist())) >= 1
    coords2, labels2, used2 = sc.cell_segments(cm, method="umap", k=2,
                                               random_state=0)
    assert np.allclose(coords1, coords2)
    assert labels1.tolist() == labels2.tolist()
    assert used1 == used2


def test_cell_segments_empty_matrix():
    cm = sc.CellMatrix(matrix=np.zeros((0, 5)), cell_ids=np.zeros(0, dtype=int),
                       peaks=np.linspace(100, 200, 5),
                       centroids_opt=np.zeros((0, 2)), support=np.zeros(0))
    coords, labels, used = sc.cell_segments(cm)
    assert coords.shape == (0, 2)
    assert labels.shape == (0,)
