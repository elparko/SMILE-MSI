"""Pure multimodal image-registration engine (MSI <-> histology / microscopy).

This module estimates a geometric transform that maps an optical/histology photo
onto the MSI acquisition grid (and back), so that ROIs/Region masks can be warped
between the two frames with a real, quantitative transform instead of manual
alpha-blend nudging.

All functions operate in the **offset, 0-indexed, orientation-0 base tissue frame**:
the base grid is ``ds._base_hw`` with grid coordinates ``(x, y)`` where
``x = coordinates[:, 0] - x0`` and ``y = coordinates[:, 1] - y0`` (exactly the
orientation-0 case of :meth:`MSIDataset._pixel_rows_cols`). The estimated transform
therefore composes cleanly with the existing ``orient`` rotation matrices used by
``gui/optical.py`` ``_optical_transform()`` at render time. Photo points are
photo-pixel ``(col, row)``.

No PySide6 / pyqtgraph imports here; heavy/optional dependencies
(``scikit-image``, ``SimpleITK``) are lazy-imported inside the functions that need
them and raise a clear ``ImportError`` naming the ``uv sync --extra register``
install command when absent.

Literature
----------
- Umeyama, S. (1991). Least-squares estimation of transformation parameters
  between two point patterns. *IEEE TPAMI*, 13(4), 376-380.
  doi:10.1109/34.88573  (closed-form similarity fit)
- Bookstein, F.L. (1989). Principal warps: thin-plate splines and the
  decomposition of deformations. *IEEE TPAMI*, 11(6), 567-585.
  doi:10.1109/34.24792  (piecewise-affine / thin-plate landmark warping)
- van der Walt, S. et al. (2014). scikit-image: image processing in Python.
  *PeerJ*, 2:e453. doi:10.7717/peerj.453  (BSD-3; PiecewiseAffineTransform)
- Klein, S. et al. (2010). elastix: a toolbox for intensity-based medical image
  registration. *IEEE TMI*, 29(1), 196-205. doi:10.1109/TMI.2009.2035616
- Mattes, D. et al. (2003). PET-CT image registration in the chest using
  free-form deformations. *IEEE TMI*, 22(1), 120-128.
  doi:10.1109/TMI.2003.809072
- Lowekamp, B.C. et al. (2013). The design of SimpleITK.
  *Front. Neuroinform.*, 7:45. doi:10.3389/fninf.2013.00045  (Apache-2.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TRANSFORM_KINDS = ("similarity", "rigid", "affine", "piecewise-affine")

_REGISTER_EXTRA_HINT = (
    "Install the optional registration extra:\n"
    "    uv sync --extra register"
)


@dataclass
class RegistrationResult:
    """Outcome of a registration estimate.

    Attributes
    ----------
    kind:
        One of ``'similarity' | 'rigid' | 'affine' | 'piecewise-affine' | 'bspline'``.
    matrix:
        3x3 homogeneous transform mapping photo-px ``(col, row)`` -> base-grid-px
        ``(x, y)``. ``None`` for piecewise/bspline (non-linear) transforms, whose
        mapping lives on :attr:`transform`.
    params:
        Decomposed ``{tx, ty, scale, angle, ...}`` where available.
    landmarks_src:
        ``(N, 2)`` photo points ``(col, row)``.
    landmarks_dst:
        ``(N, 2)`` base-grid points ``(x, y)`` in orientation-0.
    residuals:
        ``(N,)`` per-landmark fit error, in base-grid pixels.
    rmse:
        Overall landmark RMSE, in base-grid pixels.
    n_landmarks:
        Number of landmark pairs.
    transform:
        Opaque scikit-image / SimpleITK transform handle (piecewise/bspline);
        not serialized.
    """

    kind: str
    matrix: np.ndarray | None
    params: dict = field(default_factory=dict)
    landmarks_src: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    landmarks_dst: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    residuals: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    rmse: float = float("nan")
    n_landmarks: int = 0
    transform: object = None


# --------------------------------------------------------------------------- #
# Core: least-squares landmark fitting (pure numpy, always available)
# --------------------------------------------------------------------------- #
_MIN_POINTS = {
    "similarity": 2,
    "rigid": 2,
    "affine": 3,
    "piecewise-affine": 4,
}


def _as_points(pts, name: str) -> np.ndarray:
    arr = np.asarray(pts, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must be an (N, 2) array of (col/x, row/y) points; "
                         f"got shape {arr.shape}.")
    return arr


def _fit_affine_ls(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares affine fit src -> dst, returned as a 3x3 homogeneous matrix.

    Solves ``dst = A @ src + t`` in the LS sense (the classic over-determined
    affine; van der Walt et al. 2014 / scikit-image ``estimate_transform('affine')``).
    """
    n = src.shape[0]
    # Design matrix: [x, y, 1] per point; solve for each output dimension.
    X = np.hstack([src, np.ones((n, 1))])          # (N, 3)
    # Solve X @ p = dst  (p is (3, 2)); least squares.
    p, *_ = np.linalg.lstsq(X, dst, rcond=None)    # (3, 2)
    M = np.eye(3)
    M[0, :] = p[:, 0]
    M[1, :] = p[:, 1]
    return M


def _fit_similarity_umeyama(src: np.ndarray, dst: np.ndarray,
                            *, with_scale: bool = True) -> np.ndarray:
    """Closed-form similarity (or rigid, ``with_scale=False``) fit src -> dst.

    Implements Umeyama (1991): the least-squares similarity transform between two
    point sets, returned as a 3x3 homogeneous matrix.
    """
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    # Covariance of the two centred sets.
    cov = (dst_c.T @ src_c) / n                    # (2, 2)
    U, D, Vt = np.linalg.svd(cov)

    # Sign-correction matrix S to ensure a proper rotation (det = +1).
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0

    R = U @ S @ Vt                                 # (2, 2) rotation

    if with_scale:
        var_src = (src_c ** 2).sum() / n
        scale = (D * np.diag(S)).sum() / var_src if var_src > 0 else 1.0
    else:
        scale = 1.0

    t = mu_dst - scale * (R @ mu_src)

    M = np.eye(3)
    M[:2, :2] = scale * R
    M[:2, 2] = t
    return M


def _apply_matrix(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homogeneous transform to ``(N, 2)`` points."""
    n = pts.shape[0]
    hom = np.hstack([pts, np.ones((n, 1))])        # (N, 3)
    out = hom @ M.T                                # (N, 3)
    w = out[:, 2:3]
    w = np.where(w == 0, 1.0, w)
    return out[:, :2] / w


def _decompose_matrix(M: np.ndarray) -> dict:
    """Best-effort decomposition of a 3x3 transform into align-dict coefficients.

    Returns ``{tx, ty, scale, scale_y, angle, shear}``. For a pure similarity the
    decomposition is exact; for a general affine ``scale``/``shear`` capture the
    non-rigid part. ``angle`` is in degrees.
    """
    a, b = M[0, 0], M[0, 1]
    c, d = M[1, 0], M[1, 1]
    tx, ty = M[0, 2], M[1, 2]

    sx = float(np.hypot(a, c))
    angle = float(np.degrees(np.arctan2(c, a)))
    # Shear: remove the rotation, read the residual cross term.
    if sx > 0:
        msr = (a * b + c * d) / (sx * sx)
        # remaining y-scale after de-shearing
        sy = float((a * d - b * c) / sx)
    else:
        msr, sy = 0.0, 0.0
    return {
        "tx": float(tx),
        "ty": float(ty),
        "scale": sx,
        "scale_y": sy,
        "angle": angle,
        "shear": float(msr),
    }


def estimate_landmark_transform(src_xy, dst_xy, kind: str = "affine") -> RegistrationResult:
    """Least-squares fit of photo->grid from paired landmarks.

    Parameters
    ----------
    src_xy:
        ``(N, 2)`` photo points ``(col, row)``.
    dst_xy:
        ``(N, 2)`` base-grid points ``(x, y)`` in the orientation-0 frame.
    kind:
        ``'similarity'`` (closed-form Umeyama 1991), ``'rigid'`` (Umeyama without
        scale), ``'affine'`` (least-squares fit; scikit-image / van der Walt 2014),
        or ``'piecewise-affine'`` (Bookstein 1989 thin-plate / piecewise warp; lazy
        ``scikit-image``).

    Returns
    -------
    RegistrationResult
        With the 3x3 ``matrix`` (None for piecewise-affine), per-point residuals,
        and overall RMSE in base-grid pixels.

    Raises
    ------
    ValueError
        If ``kind`` is unknown or there are too few point pairs.
    ImportError
        If ``kind == 'piecewise-affine'`` and scikit-image is not installed.
    """
    if kind not in _MIN_POINTS:
        raise ValueError(
            f"Unknown transform kind {kind!r}; expected one of {tuple(_MIN_POINTS)}."
        )
    src = _as_points(src_xy, "src_xy")
    dst = _as_points(dst_xy, "dst_xy")
    if src.shape[0] != dst.shape[0]:
        raise ValueError(
            f"src_xy and dst_xy must have the same number of points; "
            f"got {src.shape[0]} and {dst.shape[0]}."
        )
    n = src.shape[0]
    need = _MIN_POINTS[kind]
    if n < need:
        raise ValueError(
            f"{kind!r} registration needs at least {need} landmark pairs; got {n}. "
            f"Pick more matching points on the ion image and the photo."
        )

    if kind == "piecewise-affine":
        try:
            from skimage.transform import PiecewiseAffineTransform
        except ImportError as e:  # pragma: no cover - exercised via importorskip
            raise ImportError(
                "Piecewise-affine landmark registration needs scikit-image. "
                + _REGISTER_EXTRA_HINT
            ) from e
        tform = PiecewiseAffineTransform()
        # skimage estimates the map dst->src for inverse warping; we keep the
        # forward (src->dst) handle on the result and compute residuals from it.
        if not tform.estimate(src, dst):
            raise ValueError(
                "Piecewise-affine estimation failed (degenerate / collinear landmarks)."
            )
        pred = tform(src)
        residuals = np.linalg.norm(pred - dst, axis=1)
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        return RegistrationResult(
            kind=kind, matrix=None, params={},
            landmarks_src=src, landmarks_dst=dst,
            residuals=residuals, rmse=rmse, n_landmarks=n, transform=tform,
        )

    if kind == "similarity":
        M = _fit_similarity_umeyama(src, dst, with_scale=True)
    elif kind == "rigid":
        M = _fit_similarity_umeyama(src, dst, with_scale=False)
    else:  # affine
        M = _fit_affine_ls(src, dst)

    pred = _apply_matrix(M, src)
    residuals = np.linalg.norm(pred - dst, axis=1)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    return RegistrationResult(
        kind=kind, matrix=M, params=_decompose_matrix(M),
        landmarks_src=src, landmarks_dst=dst,
        residuals=residuals, rmse=rmse, n_landmarks=n,
    )


# --------------------------------------------------------------------------- #
# Compose / invert
# --------------------------------------------------------------------------- #
def compose(outer: RegistrationResult, inner: RegistrationResult,
            *, kind: str | None = None) -> RegistrationResult:
    """Compose two linear transforms: result maps inner-input -> outer-output.

    Given ``inner`` (A -> B) and ``outer`` (B -> C), returns the transform A -> C,
    i.e. ``outer.matrix @ inner.matrix``. Both must be linear (have a ``matrix``).

    Raises
    ------
    ValueError
        If either input lacks a 3x3 matrix (piecewise/bspline cannot be composed
        this way).
    """
    if inner.matrix is None or outer.matrix is None:
        raise ValueError(
            "compose() requires linear transforms with a 3x3 matrix; "
            "piecewise-affine / bspline results cannot be composed this way."
        )
    M = np.asarray(outer.matrix) @ np.asarray(inner.matrix)
    return RegistrationResult(
        kind=kind or "affine",
        matrix=M,
        params=_decompose_matrix(M),
        landmarks_src=inner.landmarks_src,
        landmarks_dst=outer.landmarks_dst,
        residuals=np.empty((0,)),
        rmse=float("nan"),
        n_landmarks=0,
    )


def invert(res: RegistrationResult, *, kind: str | None = None) -> RegistrationResult:
    """Return the inverse transform (grid -> photo for a photo -> grid input).

    Raises
    ------
    ValueError
        If ``res`` has no invertible 3x3 matrix.
    """
    if res.matrix is None:
        raise ValueError(
            "invert() requires a linear transform with a 3x3 matrix; "
            "piecewise-affine / bspline results are not invertible this way."
        )
    M = np.asarray(res.matrix)
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError as e:
        raise ValueError("Transform matrix is singular and cannot be inverted.") from e
    return RegistrationResult(
        kind=kind or res.kind,
        matrix=Minv,
        params=_decompose_matrix(Minv),
        landmarks_src=res.landmarks_dst,
        landmarks_dst=res.landmarks_src,
        residuals=np.empty((0,)),
        rmse=float("nan"),
        n_landmarks=0,
    )


# --------------------------------------------------------------------------- #
# Warp points / masks / images
# --------------------------------------------------------------------------- #
def warp_points(res: RegistrationResult, pts_xy, *, inverse: bool = False) -> np.ndarray:
    """Map points through the fitted transform.

    Forward (``inverse=False``) maps photo ``(col, row)`` -> base-grid ``(x, y)``;
    ``inverse=True`` maps base-grid -> photo. Used to transfer ROI/region geometry
    between the two frames.

    Raises
    ------
    ValueError
        If a non-linear (piecewise/bspline) inverse is requested without a handle.
    """
    pts = _as_points(pts_xy, "pts_xy")
    if res.matrix is not None:
        M = np.asarray(res.matrix)
        if inverse:
            M = np.linalg.inv(M)
        return _apply_matrix(M, pts)
    # Non-linear: use the skimage transform handle.
    tform = res.transform
    if tform is None:
        raise ValueError("Result has neither a matrix nor a transform handle to warp through.")
    if inverse:
        return np.asarray(tform.inverse(pts), dtype=float)
    return np.asarray(tform(pts), dtype=float)


def _grid_points_for_pixels(ds, mask_bool: np.ndarray) -> np.ndarray:
    """Base-grid ``(x, y)`` points (orientation-0) for the True entries of a
    per-pixel boolean mask over ``ds`` pixels."""
    mask_bool = np.asarray(mask_bool, dtype=bool).ravel()
    coords = np.asarray(ds.coordinates, dtype=int)
    if mask_bool.shape[0] != coords.shape[0]:
        raise ValueError(
            f"mask length {mask_bool.shape[0]} != n_pixels {coords.shape[0]}."
        )
    x0 = int(coords[:, 0].min())
    y0 = int(coords[:, 1].min())
    sel = coords[mask_bool]
    xs = sel[:, 0] - x0
    ys = sel[:, 1] - y0
    return np.column_stack([xs.astype(float), ys.astype(float)])


def warp_mask_to_photo(res: RegistrationResult, ds, mask_bool,
                       photo_hw: tuple[int, int]) -> np.ndarray:
    """Project an MSI per-pixel boolean Region mask into the photo's pixel raster.

    Resolves each masked MSI pixel to its base-grid ``(x, y)`` (orientation-0),
    warps grid -> photo via the inverse transform, and fills the nearest photo
    pixel. Returns a boolean ``(H, W)`` raster the shape of ``photo_hw``.

    Parameters
    ----------
    res:
        A photo -> grid registration result.
    ds:
        The :class:`~smile_msi.msi.MSIDataset` whose pixels the mask indexes.
    mask_bool:
        Per-pixel boolean mask, length ``ds.n_pixels``.
    photo_hw:
        ``(height, width)`` of the target photo raster.
    """
    H, W = int(photo_hw[0]), int(photo_hw[1])
    out = np.zeros((H, W), dtype=bool)
    grid_pts = _grid_points_for_pixels(ds, mask_bool)
    if grid_pts.shape[0] == 0:
        return out
    photo_pts = warp_points(res, grid_pts, inverse=True)   # grid -> photo (col, row)
    cols = np.rint(photo_pts[:, 0]).astype(int)
    rows = np.rint(photo_pts[:, 1]).astype(int)
    inside = (cols >= 0) & (cols < W) & (rows >= 0) & (rows < H)
    out[rows[inside], cols[inside]] = True
    return out


def warp_image(res: RegistrationResult, image: np.ndarray,
               out_hw: tuple[int, int], *, order: int = 1,
               cval: float = 0.0) -> np.ndarray:
    """Warp a 2-D ``image`` (photo, grayscale) onto the base grid raster.

    ``order=0`` nearest-neighbour, ``order=1`` bilinear. Pure-numpy inverse
    sampling; resamples each output base-grid pixel ``(x, y)`` from the photo via
    the inverse transform (grid -> photo).

    Parameters
    ----------
    res:
        A photo -> grid registration result (linear).
    image:
        2-D source photo array ``(Hp, Wp)``.
    out_hw:
        ``(height, width)`` of the output base-grid raster.
    order:
        0 (nearest) or 1 (bilinear).
    cval:
        Fill value for samples landing outside the photo.

    Raises
    ------
    ValueError
        For a non-linear result or unsupported ``order``.
    """
    if res.matrix is None:
        raise ValueError("warp_image requires a linear transform (3x3 matrix).")
    if order not in (0, 1):
        raise ValueError("order must be 0 (nearest) or 1 (bilinear).")
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"image must be 2-D grayscale; got shape {image.shape}.")
    Hp, Wp = image.shape
    Hg, Wg = int(out_hw[0]), int(out_hw[1])

    yy, xx = np.mgrid[0:Hg, 0:Wg]
    grid_pts = np.column_stack([xx.ravel().astype(float), yy.ravel().astype(float)])
    photo_pts = warp_points(res, grid_pts, inverse=True)   # grid -> photo (col, row)
    pc = photo_pts[:, 0].reshape(Hg, Wg)                   # photo cols
    pr = photo_pts[:, 1].reshape(Hg, Wg)                   # photo rows

    out = np.full((Hg, Wg), cval, dtype=float)
    if order == 0:
        c = np.rint(pc).astype(int)
        r = np.rint(pr).astype(int)
        valid = (c >= 0) & (c < Wp) & (r >= 0) & (r < Hp)
        out[valid] = image[r[valid], c[valid]]
        return out

    # bilinear
    c0 = np.floor(pc).astype(int)
    r0 = np.floor(pr).astype(int)
    c1 = c0 + 1
    r1 = r0 + 1
    fc = pc - c0
    fr = pr - r0
    # A sample is usable if its lower corner is in-range and the upper corner is
    # either in-range or unused (fraction ~0 at the right/bottom edge).
    valid = (
        (pc >= 0) & (pc <= Wp - 1) & (pr >= 0) & (pr <= Hp - 1)
    )
    cc0 = np.clip(c0, 0, Wp - 1)
    cc1 = np.clip(c1, 0, Wp - 1)
    rr0 = np.clip(r0, 0, Hp - 1)
    rr1 = np.clip(r1, 0, Hp - 1)
    Ia = image[rr0, cc0]
    Ib = image[rr0, cc1]
    Ic = image[rr1, cc0]
    Id = image[rr1, cc1]
    top = Ia * (1 - fc) + Ib * fc
    bot = Ic * (1 - fc) + Id * fc
    sampled = top * (1 - fr) + bot * fr
    out[valid] = sampled[valid]
    return out


# --------------------------------------------------------------------------- #
# QTransform coefficient helper (for the GUI spinboxes)
# --------------------------------------------------------------------------- #
def to_qtransform_coeffs(res: RegistrationResult) -> dict:
    """Decompose ``res.matrix`` into ``{tx, ty, scale, angle, flipx, flipy}``.

    Best-effort align-dict for the existing optical spinboxes. Similarity is exact;
    a general affine (with shear) is approximated — the GUI builds the backdrop
    QTransform from the full 3x3 matrix, not from this decomposition, so shear still
    renders correctly.

    Raises
    ------
    ValueError
        If ``res`` has no linear matrix.
    """
    if res.matrix is None:
        raise ValueError("to_qtransform_coeffs requires a linear transform (3x3 matrix).")
    d = _decompose_matrix(np.asarray(res.matrix))
    # ``_decompose_matrix`` returns ``scale = hypot(a, c) >= 0`` and folds any reflection
    # into the signed ``scale_y`` (rotation absorbs the rest), so an estimated reflection
    # always surfaces as ``flipy`` here — ``flipx`` from this path is always False. Which
    # axis a single reflection is attributed to is convention-dependent and the GUI rebuilds
    # the backdrop from the full 3x3 matrix (not this dict), so the reconstruction is exact
    # either way; ``flipx`` stays in the dict because it is a live user-set field elsewhere
    # (the optical flip-horizontal button). See KNOWN_ISSUES.md.
    flipy = d["scale_y"] < 0
    return {
        "tx": d["tx"],
        "ty": d["ty"],
        "scale": abs(d["scale"]),
        "angle": d["angle"],
        "flipx": False,
        "flipy": bool(flipy),
    }


# --------------------------------------------------------------------------- #
# Optional: intensity-based & deformable registration (lazy SimpleITK)
# --------------------------------------------------------------------------- #
def estimate_intensity_transform(fixed, moving, *, kind: str = "affine",
                                 random_state: int = 0) -> RegistrationResult:
    """Optional automatic intensity-based registration (SimpleITK, Mattes-MI).

    Aligns ``moving`` (grayscale photo, resampled to the grid) to ``fixed`` (an
    ion/TIC image) by Mattes mutual information + regular-step gradient descent
    (Mattes 2003; Klein 2010). CPU-only. ``random_state`` seeds the MI sampler.

    Raises
    ------
    ImportError
        If SimpleITK is not installed (the ``register`` extra is absent).
    ValueError
        For an unsupported ``kind``.
    """
    try:
        import SimpleITK as sitk
    except ImportError as e:  # pragma: no cover - exercised via importorskip
        raise ImportError(
            "Intensity-based registration needs SimpleITK. " + _REGISTER_EXTRA_HINT
        ) from e

    if kind not in ("rigid", "similarity", "affine"):
        raise ValueError(
            f"intensity registration kind must be rigid/similarity/affine; got {kind!r}."
        )

    fixed_img = sitk.GetImageFromArray(np.asarray(fixed, dtype=np.float32))
    moving_img = sitk.GetImageFromArray(np.asarray(moving, dtype=np.float32))

    if kind == "rigid":
        tx = sitk.Euler2DTransform()
    elif kind == "similarity":
        tx = sitk.Similarity2DTransform()
    else:
        tx = sitk.AffineTransform(2)
    initial = sitk.CenteredTransformInitializer(
        fixed_img, moving_img, tx,
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, int(random_state))
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=200,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(initial, inPlace=False)
    final = reg.Execute(fixed_img, moving_img)

    return RegistrationResult(
        kind=kind, matrix=None,
        params={"metric": float(reg.GetMetricValue())},
        landmarks_src=np.empty((0, 2)), landmarks_dst=np.empty((0, 2)),
        residuals=np.empty((0,)), rmse=float("nan"), n_landmarks=0,
        transform=final,
    )


def refine_bspline(fixed, moving, init: RegistrationResult, *,
                   grid_spacing_px: int = 64, random_state: int = 0) -> RegistrationResult:
    """Optional deformable B-spline FFD refinement (SimpleITK, Mattes-MI).

    Initialised from ``init``; returns a result whose ``.transform`` warps points
    (``matrix`` stays ``None``). Mattes mutual information metric (Mattes 2003;
    Klein 2010). CPU-only.

    Raises
    ------
    ImportError
        If SimpleITK is not installed (the ``register`` extra is absent).
    """
    try:
        import SimpleITK as sitk
    except ImportError as e:  # pragma: no cover - exercised via importorskip
        raise ImportError(
            "Deformable B-spline registration needs SimpleITK. " + _REGISTER_EXTRA_HINT
        ) from e

    fixed_img = sitk.GetImageFromArray(np.asarray(fixed, dtype=np.float32))
    moving_img = sitk.GetImageFromArray(np.asarray(moving, dtype=np.float32))

    size = fixed_img.GetSize()
    mesh = [max(1, int(round(s / float(grid_spacing_px)))) for s in size]
    bspline = sitk.BSplineTransformInitializer(fixed_img, mesh)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, int(random_state))
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100)
    if init is not None and init.transform is not None:
        reg.SetMovingInitialTransform(init.transform)
    reg.SetInitialTransform(bspline, inPlace=True)
    final = reg.Execute(fixed_img, moving_img)

    return RegistrationResult(
        kind="bspline", matrix=None,
        params={"metric": float(reg.GetMetricValue()),
                "grid_spacing_px": int(grid_spacing_px)},
        landmarks_src=getattr(init, "landmarks_src", np.empty((0, 2))),
        landmarks_dst=getattr(init, "landmarks_dst", np.empty((0, 2))),
        residuals=np.empty((0,)), rmse=float("nan"), n_landmarks=0,
        transform=final,
    )
