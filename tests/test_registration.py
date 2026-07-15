"""Unit tests for the pure registration engine (smile_msi.registration).

Headless, synthetic data only. The core landmark/affine/similarity path uses only
numpy and must pass with the base stack; piecewise-affine and the SimpleITK
intensity/deformable paths are guarded with ``pytest.importorskip``.
"""

from __future__ import annotations

import numpy as np
import pytest

from smile_msi import registration as reg
from smile_msi.demo import make_synthetic


def _affine_matrix(angle_deg, scale, tx, ty, shear=0.0):
    a = np.radians(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    S = np.array([[scale, scale * shear], [0.0, scale]])
    M = np.eye(3)
    M[:2, :2] = R @ S
    M[0, 2] = tx
    M[1, 2] = ty
    return M


# --------------------------------------------------------------------------- #
# 1. Round-trip recovery of a known affine
# --------------------------------------------------------------------------- #
def test_recover_known_affine():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 100, size=(8, 2))
    M = _affine_matrix(23.0, 1.7, 12.0, -5.0, shear=0.3)
    dst = reg._apply_matrix(M, src)

    res = reg.estimate_landmark_transform(src, dst, kind="affine")
    assert res.kind == "affine"
    assert res.matrix is not None
    assert np.allclose(res.matrix, M, rtol=1e-6, atol=1e-6)
    assert res.rmse < 1e-9
    assert res.n_landmarks == 8
    assert res.residuals.shape == (8,)


# --------------------------------------------------------------------------- #
# 2. Similarity vs affine: similarity ground truth recovered, no shear
# --------------------------------------------------------------------------- #
def test_similarity_recovers_no_shear():
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 50, size=(6, 2))
    M = _affine_matrix(40.0, 0.8, -3.0, 7.0, shear=0.0)  # pure similarity
    dst = reg._apply_matrix(M, src)

    res_sim = reg.estimate_landmark_transform(src, dst, kind="similarity")
    assert np.allclose(res_sim.matrix, M, rtol=1e-6, atol=1e-6)
    assert res_sim.rmse < 1e-9
    # No shear: M = scale*R, so columns are orthogonal and equal length.
    A = res_sim.matrix[:2, :2]
    assert abs(np.dot(A[:, 0], A[:, 1])) < 1e-8
    assert abs(np.linalg.norm(A[:, 0]) - np.linalg.norm(A[:, 1])) < 1e-8

    # Affine kind also recovers the same similarity transform.
    res_aff = reg.estimate_landmark_transform(src, dst, kind="affine")
    assert np.allclose(res_aff.matrix, M, rtol=1e-6, atol=1e-6)


def test_rigid_no_scale():
    rng = np.random.default_rng(7)
    src = rng.uniform(0, 30, size=(5, 2))
    M = _affine_matrix(15.0, 1.0, 4.0, -2.0)  # rotation+translation only
    dst = reg._apply_matrix(M, src)
    res = reg.estimate_landmark_transform(src, dst, kind="rigid")
    assert np.allclose(res.matrix, M, rtol=1e-6, atol=1e-6)
    # scale ~ 1
    assert abs(res.params["scale"] - 1.0) < 1e-6


# --------------------------------------------------------------------------- #
# 3. Degenerate / too-few input
# --------------------------------------------------------------------------- #
def test_too_few_points_affine_raises():
    src = np.array([[0.0, 0.0], [1.0, 0.0]])
    dst = np.array([[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="at least 3 landmark"):
        reg.estimate_landmark_transform(src, dst, kind="affine")


def test_unknown_kind_raises():
    src = np.zeros((3, 2))
    dst = np.zeros((3, 2))
    with pytest.raises(ValueError, match="Unknown transform kind"):
        reg.estimate_landmark_transform(src, dst, kind="nope")


def test_mismatched_counts_raise():
    with pytest.raises(ValueError, match="same number of points"):
        reg.estimate_landmark_transform(np.zeros((3, 2)), np.zeros((4, 2)), kind="affine")


def test_bad_shape_raises():
    with pytest.raises(ValueError, match="must be an"):
        reg.estimate_landmark_transform(np.zeros((3, 3)), np.zeros((3, 3)), kind="affine")


# --------------------------------------------------------------------------- #
# 4. warp_points inverse round-trip
# --------------------------------------------------------------------------- #
def test_warp_points_inverse_roundtrip():
    rng = np.random.default_rng(2)
    src = rng.uniform(0, 80, size=(5, 2))
    M = _affine_matrix(33.0, 1.3, -8.0, 4.0, shear=0.15)
    dst = reg._apply_matrix(M, src)
    res = reg.estimate_landmark_transform(src, dst, kind="affine")

    pts = rng.uniform(0, 80, size=(10, 2))
    fwd = reg.warp_points(res, pts)                       # photo -> grid
    back = reg.warp_points(res, fwd, inverse=True)        # grid -> photo
    assert np.allclose(back, pts, atol=1e-7)


# --------------------------------------------------------------------------- #
# 5. compose two transforms
# --------------------------------------------------------------------------- #
def test_compose_two_transforms():
    rng = np.random.default_rng(3)
    src = rng.uniform(0, 60, size=(6, 2))

    M1 = _affine_matrix(10.0, 1.1, 3.0, 1.0)
    M2 = _affine_matrix(-25.0, 0.9, -2.0, 5.0, shear=0.2)
    inner = reg.estimate_landmark_transform(src, reg._apply_matrix(M1, src), kind="affine")
    mid = reg._apply_matrix(M1, src)
    outer = reg.estimate_landmark_transform(mid, reg._apply_matrix(M2, mid), kind="affine")

    comp = reg.compose(outer, inner)
    # Composed should equal applying M1 then M2.
    expected = reg._apply_matrix(M2, reg._apply_matrix(M1, src))
    got = reg.warp_points(comp, src)
    assert np.allclose(got, expected, atol=1e-7)
    assert np.allclose(comp.matrix, M2 @ M1, atol=1e-7)


def test_compose_requires_matrix():
    res = reg.RegistrationResult(kind="bspline", matrix=None)
    lin = reg.estimate_landmark_transform(
        np.zeros((3, 2)) + [[0, 0], [1, 0], [0, 1]],
        np.zeros((3, 2)) + [[0, 0], [1, 0], [0, 1]],
        kind="affine",
    )
    with pytest.raises(ValueError, match="3x3 matrix"):
        reg.compose(res, lin)


# --------------------------------------------------------------------------- #
# 6. invert
# --------------------------------------------------------------------------- #
def test_invert_roundtrip():
    rng = np.random.default_rng(4)
    src = rng.uniform(0, 70, size=(5, 2))
    M = _affine_matrix(18.0, 1.25, 6.0, -3.0, shear=0.1)
    dst = reg._apply_matrix(M, src)
    res = reg.estimate_landmark_transform(src, dst, kind="affine")

    inv = reg.invert(res)
    composed = reg.compose(inv, res)
    assert np.allclose(composed.matrix, np.eye(3), atol=1e-7)


def test_invert_singular_raises():
    # Map all points to a line -> singular affine.
    res = reg.RegistrationResult(
        kind="affine",
        matrix=np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    )
    with pytest.raises(ValueError, match="singular"):
        reg.invert(res)


# --------------------------------------------------------------------------- #
# 7. warp a small mask / image
# --------------------------------------------------------------------------- #
def test_warp_image_identity_nearest_and_bilinear():
    img = np.arange(16, dtype=float).reshape(4, 4)
    identity = reg.RegistrationResult(kind="affine", matrix=np.eye(3))
    out0 = reg.warp_image(identity, img, (4, 4), order=0)
    out1 = reg.warp_image(identity, img, (4, 4), order=1)
    assert np.allclose(out0, img)
    assert np.allclose(out1, img)


def test_warp_image_translation():
    img = np.zeros((5, 5), dtype=float)
    img[2, 2] = 1.0
    # grid -> photo inverse shifts by +1: matrix photo->grid translates by +1 in x.
    M = np.eye(3)
    M[0, 2] = 1.0  # photo col c -> grid x = c + 1; so grid x maps back to photo c = x - 1
    res = reg.RegistrationResult(kind="affine", matrix=M)
    out = reg.warp_image(res, img, (5, 5), order=0)
    # The bright photo pixel at (row2,col2) lands at grid x = 3.
    assert out[2, 3] == 1.0


def test_warp_mask_to_photo_identity():
    ds = make_synthetic(width=10, height=8)
    n = ds.n_pixels
    mask = np.zeros(n, dtype=bool)
    mask[0] = True
    mask[5] = True
    identity = reg.RegistrationResult(kind="affine", matrix=np.eye(3))

    # Photo raster sized to the base grid; under identity the masked pixels land
    # at their own base-grid (x, y).
    h0, w0 = ds._base_hw
    out = reg.warp_mask_to_photo(identity, ds, mask, (h0, w0))
    assert out.dtype == bool
    assert out.sum() == 2

    coords = np.asarray(ds.coordinates, dtype=int)
    x0 = coords[:, 0].min()
    y0 = coords[:, 1].min()
    for i in (0, 5):
        x = coords[i, 0] - x0
        y = coords[i, 1] - y0
        assert out[y, x]


def test_warp_mask_empty():
    ds = make_synthetic(width=6, height=6)
    mask = np.zeros(ds.n_pixels, dtype=bool)
    identity = reg.RegistrationResult(kind="affine", matrix=np.eye(3))
    out = reg.warp_mask_to_photo(identity, ds, mask, (6, 6))
    assert out.sum() == 0


def test_warp_mask_wrong_length_raises():
    ds = make_synthetic(width=6, height=6)
    identity = reg.RegistrationResult(kind="affine", matrix=np.eye(3))
    with pytest.raises(ValueError, match="!= n_pixels"):
        reg.warp_mask_to_photo(identity, ds, np.zeros(3, dtype=bool), (6, 6))


# --------------------------------------------------------------------------- #
# 8. Orientation composition: base frame is orientation-0 invariant
# --------------------------------------------------------------------------- #
def test_warp_mask_orientation_invariant_base_frame():
    """The engine works purely in the orientation-0 base frame: changing the
    dataset's display orientation must NOT change the warped base-frame raster,
    because base-grid (x, y) for a pixel is independent of ``ds.orientation``.
    This guards the highest-risk seam (orientation composition is applied only at
    render time by gui/optical.py, never inside the engine)."""
    ds = make_synthetic(width=12, height=9)
    mask = np.zeros(ds.n_pixels, dtype=bool)
    mask[3] = True
    mask[7] = True
    mask[20] = True
    identity = reg.RegistrationResult(kind="affine", matrix=np.eye(3))
    h0, w0 = ds._base_hw

    ds.set_orientation(0)
    out0 = reg.warp_mask_to_photo(identity, ds, mask, (h0, w0))
    ds.set_orientation(1)
    out1 = reg.warp_mask_to_photo(identity, ds, mask, (h0, w0))
    ds.set_orientation(3)
    out3 = reg.warp_mask_to_photo(identity, ds, mask, (h0, w0))

    assert np.array_equal(out0, out1)
    assert np.array_equal(out0, out3)
    assert out0.sum() == 3


# --------------------------------------------------------------------------- #
# 9. to_qtransform_coeffs decomposition
# --------------------------------------------------------------------------- #
def test_to_qtransform_coeffs_similarity_exact():
    rng = np.random.default_rng(5)
    src = rng.uniform(0, 40, size=(5, 2))
    M = _affine_matrix(30.0, 1.5, 9.0, -4.0)
    dst = reg._apply_matrix(M, src)
    res = reg.estimate_landmark_transform(src, dst, kind="similarity")
    coeffs = reg.to_qtransform_coeffs(res)
    assert abs(coeffs["scale"] - 1.5) < 1e-6
    assert abs(coeffs["angle"] - 30.0) < 1e-4
    assert abs(coeffs["tx"] - 9.0) < 1e-6
    assert abs(coeffs["ty"] + 4.0) < 1e-6
    assert coeffs["flipx"] is False


def test_to_qtransform_coeffs_requires_matrix():
    res = reg.RegistrationResult(kind="bspline", matrix=None)
    with pytest.raises(ValueError, match="linear transform"):
        reg.to_qtransform_coeffs(res)


def test_to_qtransform_coeffs_reflection_surfaces_as_flipy():
    """A reflected (det < 0) affine decomposes with flipx always False and the
    reflection captured by flipy — and the {scale, angle, flipx, flipy} readout
    rebuilds the original linear part exactly (the GUI uses the full matrix anyway).
    Guards the vestigial-flipx cleanup (KNOWN_ISSUES.md)."""
    rng = np.random.default_rng(7)
    src = rng.uniform(0, 40, size=(6, 2))
    # an affine with a negative determinant (a reflection across the x-axis + rotation)
    M = _affine_matrix(20.0, 1.3, 5.0, -2.0).copy()
    M[:, 1] *= -1.0                                  # flip the second linear column → det < 0
    dst = reg._apply_matrix(M, src)
    res = reg.estimate_landmark_transform(src, dst, kind="affine")
    coeffs = reg.to_qtransform_coeffs(res)
    assert coeffs["flipx"] is False                  # reflections never attributed to x here
    assert coeffs["flipy"] is True                   # ...always to y

    # the readout reconstructs the linear part: R(angle) · diag(±scale_x, ±scale_y)
    ang = np.radians(coeffs["angle"])
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    sx = coeffs["scale"] * (-1.0 if coeffs["flipx"] else 1.0)
    sy = coeffs["scale"] * (-1.0 if coeffs["flipy"] else 1.0)
    rebuilt = R @ np.diag([sx, sy])
    assert np.allclose(rebuilt, np.asarray(res.matrix)[:2, :2], atol=1e-6)


# --------------------------------------------------------------------------- #
# 10. Optional scikit-image path: piecewise-affine
# --------------------------------------------------------------------------- #
def test_piecewise_affine_with_skimage():
    pytest.importorskip("skimage")
    rng = np.random.default_rng(6)
    src = rng.uniform(0, 100, size=(9, 2))
    M = _affine_matrix(12.0, 1.2, 5.0, -2.0)
    dst = reg._apply_matrix(M, src)
    res = reg.estimate_landmark_transform(src, dst, kind="piecewise-affine")
    assert res.kind == "piecewise-affine"
    assert res.matrix is None
    assert res.transform is not None
    # Piecewise-affine interpolates the landmarks exactly -> ~zero residual at nodes.
    assert res.rmse < 1e-6


def test_piecewise_affine_missing_dep_errors_clearly(monkeypatch):
    """Without scikit-image installed, the piecewise path must raise a clear
    ImportError naming the install command. Simulate absence by blocking import."""
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("skimage"):
            raise ImportError("no skimage")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    dst = src.copy()
    with pytest.raises(ImportError, match="uv sync --extra register"):
        reg.estimate_landmark_transform(src, dst, kind="piecewise-affine")


# --------------------------------------------------------------------------- #
# 11. Optional SimpleITK paths
# --------------------------------------------------------------------------- #
def test_intensity_transform_missing_dep_errors_clearly(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "SimpleITK":
            raise ImportError("no sitk")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    fixed = np.zeros((8, 8))
    moving = np.zeros((8, 8))
    with pytest.raises(ImportError, match="uv sync --extra register"):
        reg.estimate_intensity_transform(fixed, moving)
    with pytest.raises(ImportError, match="uv sync --extra register"):
        reg.refine_bspline(fixed, moving, None)


def test_intensity_transform_with_simpleitk():
    pytest.importorskip("SimpleITK")
    # A structured, off-centre blob: smooth enough that mutual information is well-conditioned
    # and asymmetric enough to pin the rotation (random noise is pathological for MI and drifts
    # ~1.5 px even on identical input).
    yy, xx = np.mgrid[0:32, 0:32].astype(float)
    fixed = np.exp(-(((xx - 17.0) ** 2 + (yy - 13.0) ** 2) / 50.0))
    moving = fixed.copy()
    res = reg.estimate_intensity_transform(fixed, moving, kind="rigid", random_state=0)
    assert res.kind == "rigid"
    assert res.transform is not None
    assert np.isfinite(res.params["metric"])
    # moving == fixed, so the recovered transform must be ~identity: it should not invent a
    # shift/rotation. Map several points and require sub-pixel error. Asserting only
    # "transform is not None" would pass even if SITK returned a garbage transform.
    for pt in [(5.0, 5.0), (12.0, 18.0), (20.0, 3.0), (25.0, 25.0)]:
        out = res.transform.TransformPoint(pt)
        assert abs(out[0] - pt[0]) < 0.25 and abs(out[1] - pt[1]) < 0.25
