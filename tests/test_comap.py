"""Engine tests for :mod:`smile_msi.comap` — spatial multi-omics co-mapping.

Synthetic data only (no real ST / microscopy file): a tiny MSIDataset built via
``from_arrays`` plus a hand-built :class:`~smile_msi.comap.ModalityGrid` whose
features are exact functions of ion images, so the correlation / resample / region
arithmetic is checked against known ground truth.
"""
from __future__ import annotations

import numpy as np
import pytest

from smile_msi import comap
from smile_msi.msi import MSIDataset
from smile_msi.spatial import RegionCorrelation


# --------------------------------------------------------------------------- #
# Fixtures: a deterministic 6x6 grid with two ion images
# --------------------------------------------------------------------------- #
@pytest.fixture
def ds():
    """6x6 single-pixel-pitch MSIDataset with two peaks (m/z 100 and 200).

    Peak 100's intensity = a smooth horizontal ramp; peak 200's = a vertical ramp,
    so the two ions are spatially distinct.
    """
    W = H = 6
    coords = np.array([[x, y] for y in range(H) for x in range(W)], dtype=int)
    n = coords.shape[0]
    mzs = [np.array([100.0, 200.0]) for _ in range(n)]
    inten = []
    for x, y in coords:
        inten.append(np.array([1.0 + x, 1.0 + y], dtype=float))   # ramp_x, ramp_y
    d = MSIDataset.from_arrays(coords, mzs, inten)
    return d


def _ion_image_vector(ds, peaks):
    """The (n_pixels, n_peaks) feature matrix with no normalization, pixel order."""
    from smile_msi import spatial
    return spatial.feature_matrix(ds, peaks, norm="none")


# --------------------------------------------------------------------------- #
# resample_to_msi — same-grid identity round-trips values
# --------------------------------------------------------------------------- #
def test_resample_same_grid_identity_roundtrips(ds):
    coords = np.asarray(ds.coordinates, dtype=float)
    # one modality feature = a known per-pixel value (the x+y sum), same coords/order
    vals = (coords[:, 0] + coords[:, 1]).reshape(-1, 1)
    grid = comap.ModalityGrid(kind="generic", feature_names=["F"],
                              coords_um=coords.copy(), values=vals.copy())
    M, names = comap.resample_to_msi(grid, ds, transform=None)
    assert names == ["F"]
    assert M.shape == (ds.n_pixels, 1)
    # identity same-grid → each pixel gets its own location's value, no NaNs
    assert not np.isnan(M).any()
    np.testing.assert_allclose(M[:, 0], vals[:, 0])


def test_resample_offgrid_locations_are_nan(ds):
    coords = np.asarray(ds.coordinates, dtype=float)
    # cover only the first 10 pixels; the rest get NaN
    grid = comap.ModalityGrid(kind="generic", feature_names=["F"],
                              coords_um=coords[:10].copy(),
                              values=np.arange(10, dtype=float).reshape(-1, 1))
    M, _ = comap.resample_to_msi(grid, ds, transform=None)
    assert not np.isnan(M[:10, 0]).any()
    assert np.isnan(M[10:, 0]).all()


def test_resample_known_translation_lands_values(ds):
    """A modality grid shifted by (+2, +1) plus a transform that undoes the shift
    must land each value on the original pixel."""
    coords = np.asarray(ds.coordinates, dtype=float)
    shift = np.array([2.0, 1.0])
    vals = (coords[:, 0] * 10 + coords[:, 1]).reshape(-1, 1)
    grid = comap.ModalityGrid(kind="generic", feature_names=["F"],
                              coords_um=(coords + shift).copy(), values=vals.copy())
    # transform: subtract the shift (modality frame -> MSI frame)
    Tmat = np.eye(3)
    Tmat[0, 2] = -shift[0]
    Tmat[1, 2] = -shift[1]
    M, _ = comap.resample_to_msi(grid, ds, transform=Tmat)
    assert not np.isnan(M[:, 0]).any()
    np.testing.assert_allclose(M[:, 0], vals[:, 0])


def test_resample_accepts_registration_result(ds):
    """A registration.RegistrationResult (linear) is accepted as the transform."""
    from smile_msi import registration

    coords = np.asarray(ds.coordinates, dtype=float)
    shift = np.array([1.0, 0.0])
    src = (coords + shift)           # modality frame
    dst = coords                     # MSI frame
    res = registration.estimate_landmark_transform(src, dst, kind="affine")
    vals = (coords[:, 0] + 2 * coords[:, 1]).reshape(-1, 1)
    grid = comap.ModalityGrid(kind="generic", feature_names=["F"],
                              coords_um=src.copy(), values=vals.copy())
    M, _ = comap.resample_to_msi(grid, ds, transform=res)
    assert not np.isnan(M[:, 0]).any()
    np.testing.assert_allclose(M[:, 0], vals[:, 0], atol=1e-6)


def test_resample_orientation_invariant(ds):
    """Rotating the display (set_orientation) must not change the resampled values —
    binning is in the pre-rotation frame."""
    coords = np.asarray(ds.coordinates, dtype=float)
    vals = (coords[:, 0] - coords[:, 1]).reshape(-1, 1)
    grid = comap.ModalityGrid(kind="generic", feature_names=["F"],
                              coords_um=coords.copy(), values=vals.copy())
    M0, _ = comap.resample_to_msi(grid, ds, transform=None)
    ds.set_orientation(1)
    M1, _ = comap.resample_to_msi(grid, ds, transform=None)
    np.testing.assert_allclose(M0, M1)


def test_resample_agg_sum_and_max(ds):
    """Two modality locations in the same pixel aggregate by mean/sum/max."""
    coords = np.asarray(ds.coordinates, dtype=float)
    # put two locations exactly on pixel (0,0): values 2 and 4
    loc = np.array([[0.0, 0.0], [0.0, 0.0]])
    grid = comap.ModalityGrid(kind="generic", feature_names=["F"], coords_um=loc,
                              values=np.array([[2.0], [4.0]]))
    Mmean, _ = comap.resample_to_msi(grid, ds, agg="mean")
    Msum, _ = comap.resample_to_msi(grid, ds, agg="sum")
    Mmax, _ = comap.resample_to_msi(grid, ds, agg="max")
    p00 = int(np.flatnonzero((coords[:, 0] == 0) & (coords[:, 1] == 0))[0])
    assert Mmean[p00, 0] == pytest.approx(3.0)
    assert Msum[p00, 0] == pytest.approx(6.0)
    assert Mmax[p00, 0] == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# cross_modality_correlation — exact on same grid
# --------------------------------------------------------------------------- #
def test_cross_correlation_exact_same_grid(ds):
    peaks = [100.0, 200.0]
    X = _ion_image_vector(ds, peaks)         # cols: ramp_x, ramp_y
    coords = np.asarray(ds.coordinates, dtype=float)
    # modality feature 0 = exact linear function of ramp_x (peak 100) -> r=+1
    # modality feature 1 = exact linear function of ramp_y (peak 200) -> r=+1
    f0 = 3.0 * X[:, 0] + 7.0
    f1 = -2.0 * X[:, 1] + 5.0                 # negative slope -> r=-1 with peak 200
    vals = np.column_stack([f0, f1])
    grid = comap.ModalityGrid(kind="generic", feature_names=["g0", "g1"],
                              coords_um=coords.copy(), values=vals)
    cc = comap.cross_modality_correlation(ds, peaks, grid, mask=None, norm="none",
                                          min_pixels=5, method="pearson")
    assert isinstance(cc, comap.CrossCorr)
    assert cc.matrix.shape == (2, 2)
    # peak100 vs g0 ≈ +1 ; peak200 vs g1 ≈ -1
    assert cc.matrix[0, 0] == pytest.approx(1.0, abs=1e-9)
    assert cc.matrix[1, 1] == pytest.approx(-1.0, abs=1e-9)
    # cross terms (peak100 vs g1, peak200 vs g0) are weakly correlated (ramps independent)
    assert abs(cc.matrix[0, 0]) > abs(cc.matrix[0, 1])
    assert abs(cc.matrix[1, 1]) > abs(cc.matrix[1, 0])


def test_cross_correlation_spearman_monotone_invariant(ds):
    """A strictly-monotone nonlinear copy of an ion image -> Spearman ≈ 1 even though
    Pearson is < 1."""
    peaks = [100.0, 200.0]
    X = _ion_image_vector(ds, peaks)
    coords = np.asarray(ds.coordinates, dtype=float)
    # cube of the (shifted positive) ramp_x: strictly increasing, nonlinear
    f0 = (X[:, 0] + 1.0) ** 3
    vals = f0.reshape(-1, 1)
    grid = comap.ModalityGrid(kind="generic", feature_names=["g0"],
                              coords_um=coords.copy(), values=vals)
    cc_p = comap.cross_modality_correlation(ds, peaks, grid, mask=None, norm="none",
                                            min_pixels=5, method="pearson")
    cc_s = comap.cross_modality_correlation(ds, peaks, grid, mask=None, norm="none",
                                            min_pixels=5, method="spearman")
    assert cc_s.matrix[0, 0] == pytest.approx(1.0, abs=1e-9)      # rank-perfect
    assert cc_p.matrix[0, 0] < 0.999                              # nonlinear -> < 1


def test_cross_correlation_min_pixels_guard(ds):
    coords = np.asarray(ds.coordinates, dtype=float)
    grid = comap.ModalityGrid(kind="generic", feature_names=["g0"],
                              coords_um=coords[:3].copy(),
                              values=np.arange(3, dtype=float).reshape(-1, 1))
    with pytest.raises(ValueError, match="shared pixels"):
        comap.cross_modality_correlation(ds, [100.0], grid, mask=None, min_pixels=30)


def test_cross_correlation_top_pairs(ds):
    peaks = [100.0, 200.0]
    X = _ion_image_vector(ds, peaks)
    coords = np.asarray(ds.coordinates, dtype=float)
    vals = np.column_stack([X[:, 0], X[:, 1]])
    grid = comap.ModalityGrid(kind="generic", feature_names=["gx", "gy"],
                              coords_um=coords.copy(), values=vals)
    cc = comap.cross_modality_correlation(ds, peaks, grid, mask=None, norm="none",
                                          min_pixels=5)
    top = cc.top_pairs(k=1)
    assert len(top) == 1
    mz, feat, score = top[0]
    assert score == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# joint_embedding
# --------------------------------------------------------------------------- #
def test_joint_embedding_shape_and_determinism(ds):
    from smile_msi.multivariate import Embedding

    peaks = [100.0, 200.0]
    X = _ion_image_vector(ds, peaks)
    coords = np.asarray(ds.coordinates, dtype=float)
    vals = np.column_stack([X[:, 0] * 2.0, X[:, 1] * 0.5])
    grid = comap.ModalityGrid(kind="generic", feature_names=["g0", "g1"],
                              coords_um=coords.copy(), values=vals)
    emb1 = comap.joint_embedding(ds, peaks, grid, mask=None, random_state=0)
    emb2 = comap.joint_embedding(ds, peaks, grid, mask=None, random_state=0)
    assert isinstance(emb1, Embedding)
    assert emb1.coords.shape == (ds.n_pixels, 2)
    # index maps back to absolute ds pixel ids (0..n-1 here, all shared)
    assert set(emb1.index.tolist()) == set(range(ds.n_pixels))
    np.testing.assert_allclose(emb1.coords, emb2.coords)


def test_joint_embedding_weight_changes_embedding(ds):
    """The block-balance ``weight`` must actually affect the embedding. Regression: it was
    silently erased by a re-standardization (StandardScaler) inside multivariate.embedding,
    so every weight gave identical coords."""
    peaks = [100.0, 200.0]
    X = _ion_image_vector(ds, peaks)
    coords = np.asarray(ds.coordinates, dtype=float)
    vals = np.column_stack([X[:, 0] * 2.0, X[:, 1] * 0.5])
    grid = comap.ModalityGrid(kind="generic", feature_names=["g0", "g1"],
                              coords_um=coords.copy(), values=vals)
    lo = comap.joint_embedding(ds, peaks, grid, mask=None, weight=0.1, random_state=0)
    hi = comap.joint_embedding(ds, peaks, grid, mask=None, weight=0.9, random_state=0)
    assert not np.allclose(lo.coords, hi.coords)


# --------------------------------------------------------------------------- #
# cross_region_comparison
# --------------------------------------------------------------------------- #
def test_cross_region_comparison_returns_regioncorrelation(ds):
    peaks = [100.0, 200.0]
    X = _ion_image_vector(ds, peaks)
    coords = np.asarray(ds.coordinates, dtype=float)
    vals = np.column_stack([X[:, 0], X[:, 1]])
    grid = comap.ModalityGrid(kind="generic", feature_names=["g0", "g1"],
                              coords_um=coords.copy(), values=vals)
    # two regions: left half vs right half
    left = coords[:, 0] < 3
    right = coords[:, 0] >= 3
    rc = comap.cross_region_comparison(ds, peaks, grid,
                                       {"Left": left, "Right": right}, method="pearson")
    assert isinstance(rc, RegionCorrelation)
    assert rc.names == ["Left", "Right"]
    assert rc.matrix.shape == (2, 2)
    np.testing.assert_allclose(np.diag(rc.matrix), 1.0, atol=1e-9)
    bm = rc.best_match("Left")
    assert bm is not None and bm[0] == "Right"
    # the two halves differ -> off-diagonal similarity strictly below 1
    assert rc.matrix[0, 1] < 1.0


def test_cross_region_comparison_needs_two_regions(ds):
    coords = np.asarray(ds.coordinates, dtype=float)
    grid = comap.ModalityGrid(kind="generic", feature_names=["g0"],
                              coords_um=coords.copy(),
                              values=np.ones((ds.n_pixels, 1)))
    with pytest.raises(ValueError, match="two non-empty regions"):
        comap.cross_region_comparison(ds, [100.0], grid,
                                      {"only": coords[:, 0] < 3})


# --------------------------------------------------------------------------- #
# load_modality — generic CSV (core) + .h5ad lazy-import error
# --------------------------------------------------------------------------- #
def test_load_modality_generic_csv(tmp_path):
    import pandas as pd

    df = pd.DataFrame({
        "x_um": [0.0, 1.0, 2.0],
        "y_um": [0.0, 0.0, 1.0],
        "GeneA": [5.0, 6.0, 7.0],
        "GeneB": [1.0, 0.0, 2.0],
    })
    p = tmp_path / "mod.csv"
    df.to_csv(p, index=False)
    grid = comap.load_modality(str(p), kind="transcriptomics")
    assert grid.kind == "transcriptomics"
    assert grid.feature_names == ["GeneA", "GeneB"]
    assert grid.coords_um.shape == (3, 2)
    np.testing.assert_allclose(grid.feature_vector("GeneA"), [5.0, 6.0, 7.0])
    np.testing.assert_allclose(grid.coords_um[:, 0], [0.0, 1.0, 2.0])


def test_load_modality_tsv_separator(tmp_path):
    import pandas as pd

    df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0], "Prot1": [3.0, 4.0]})
    p = tmp_path / "mod.tsv"
    df.to_csv(p, sep="\t", index=False)
    grid = comap.load_modality(str(p), kind="proteomics")
    assert grid.feature_names == ["Prot1"]
    np.testing.assert_allclose(grid.feature_vector("Prot1"), [3.0, 4.0])


def test_load_modality_unknown_extension_raises(tmp_path):
    p = tmp_path / "mod.foo"
    p.write_text("nope")
    with pytest.raises(ValueError, match="Unrecognised"):
        comap.load_modality(str(p))


def test_load_modality_h5ad_missing_backend_raises(tmp_path, monkeypatch):
    """When anndata is absent, .h5ad import raises a clear ImportError naming comap."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "anndata":
            raise ImportError("no anndata")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    p = tmp_path / "x.h5ad"
    p.write_bytes(b"\x89HDF\r\n\x1a\n")          # not a real h5ad; import fails first
    with pytest.raises(ImportError, match="comap"):
        comap.load_modality(str(p))


# --------------------------------------------------------------------------- #
# sparse handling
# --------------------------------------------------------------------------- #
def test_modality_grid_accepts_scipy_sparse(ds):
    sparse = pytest.importorskip("scipy.sparse")
    coords = np.asarray(ds.coordinates, dtype=float)
    dense = np.zeros((ds.n_pixels, 2))
    dense[:, 0] = coords[:, 0]
    csr = sparse.csr_matrix(dense)
    grid = comap.ModalityGrid(kind="transcriptomics", feature_names=["g0", "g1"],
                              coords_um=coords.copy(), values=csr)
    # feature_vector densifies one column
    np.testing.assert_allclose(grid.feature_vector("g0"), coords[:, 0])
    M, _ = comap.resample_to_msi(grid, ds, transform=None)
    np.testing.assert_allclose(M[:, 0], coords[:, 0])


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def test_modality_grid_validates_shapes():
    with pytest.raises(ValueError, match="coords_um"):
        comap.ModalityGrid(kind="generic", feature_names=["a"],
                           coords_um=np.zeros((3,)), values=np.zeros((3, 1)))
    with pytest.raises(ValueError, match="feature_names"):
        comap.ModalityGrid(kind="generic", feature_names=["a"],
                           coords_um=np.zeros((3, 2)), values=np.zeros((3, 2)))
