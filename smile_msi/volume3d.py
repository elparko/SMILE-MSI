"""Pure 3D serial-section reconstruction engine (stack -> register -> volume).

SMILE MSI treats every slide as an isolated 2D acquisition. This module layers an
**ordered, spacing-aware section stack** over the existing flat cohort roster,
registers each section into a common ``(x, y)`` frame at its depth ``z``, and
resamples one ion channel into a true ``(Z, H, W)`` molecular volume that can be
read as orthoslices (triplanar), a maximum-intensity projection, or an
alpha-composited volume render.

Scope is **spatial only** (x,y inter-section registration + z stacking) for one
ion channel at a time. There is no separate "resample onto a shared m/z axis"
step: querying the same ``mz ± tol_ppm`` across sections via
:meth:`MSIDataset.ion_image` *is* the shared-axis operation.

Design constraints
------------------
- Engine only; **no Qt / pyqtgraph imports** at module scope. The optional GL
  volume render is lazy-imported inside :func:`gl_volume_item` and raises a clear,
  actionable error naming the ``viz3d`` extra when absent.
- The core path (stack model, phase-correlation rigid registration, ion-channel
  resampling, numpy MIP / orthoslices / alpha-composite) uses **numpy + scipy +
  the in-repo** :mod:`smile_msi.registration` **engine only** -- no scikit-image,
  no SimpleITK required. Those heavier backends are reached only via
  ``registration.py``'s own optional paths (``affine``/``deformable`` modes that
  delegate to it) and stay lazy there.
- Determinism: every stochastic step takes ``random_state``. The default
  phase-correlation translation estimate is deterministic by construction.
- Generated entity names use ``" - "`` (never the middle dot ``"·"``).

This module builds **on the real** :mod:`smile_msi.registration` public API:
:func:`~smile_msi.registration.estimate_landmark_transform` (to express a fitted
transform as a :class:`~smile_msi.registration.RegistrationResult`),
:func:`~smile_msi.registration.estimate_intensity_transform` (SimpleITK affine,
opt-in), :func:`~smile_msi.registration.refine_bspline` (deformable, opt-in),
:func:`~smile_msi.registration.warp_image` (inverse-sampling resample),
:func:`~smile_msi.registration.compose`, :func:`~smile_msi.registration.invert`
and :data:`~smile_msi.registration.TRANSFORM_KINDS`.

Literature
----------
- Andersson, M., Groseclose, M.R., Deutch, A.Y. & Caprioli, R.M. (2008). Imaging
  mass spectrometry of proteins and peptides: 3D volume reconstruction.
  *Nature Methods*, 5(1), 101-108. doi:10.1038/nmeth1145
  (serial-section stacking + per-z 2D registration recipe we mirror)
- Crecelius, A.C., Cornett, D.S., Caprioli, R.M., et al. (2005). Three-dimensional
  visualization of protein expression in mouse brain structures using imaging mass
  spectrometry. *J. Am. Soc. Mass Spectrom.*, 16(7), 1093-1099.
  doi:10.1016/j.jasms.2005.02.026
- Reddy, B.S. & Chatterjee, B.N. (1996). An FFT-based technique for translation,
  rotation, and scale-invariant image registration. *IEEE Trans. Image Process.*,
  5(8), 1266-1271. doi:10.1109/83.506761
  (phase-correlation translation estimate -- the rigid-stack default)
- Levoy, M. (1988). Display of surfaces from volume data. *IEEE Computer Graphics
  and Applications*, 8(3), 29-37. doi:10.1109/38.511
  (front-to-back alpha-composited volume rendering)

All algorithms are reimplemented clean-room from the publications above.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from .constants import DEFAULT_TOL_PPM

from . import registration

# Registration modes this engine understands. "rigid" is the numpy
# phase-correlation default; "affine"/"deformable" delegate to registration.py's
# optional SimpleITK paths.
STACK_MODES = ("rigid", "affine", "deformable")

VERSION = 1


# --------------------------------------------------------------------------- #
# locations / persistence
# --------------------------------------------------------------------------- #
def stacks_dir() -> str:
    """Directory holding saved section stacks; created on demand (mirrors
    :func:`smile_msi.cohort.cohorts_dir`)."""
    from . import library
    d = os.path.join(library.home_dir(), "stacks")
    os.makedirs(d, exist_ok=True)
    return d


def stack_path(name: str) -> str:
    """Path of the named stack's JSON file under :func:`stacks_dir`."""
    from . import session
    return os.path.join(stacks_dir(), f"{session._sanitize(name)}.json")


def list_stacks() -> list[dict]:
    """``{name, path, n_sections, mtime}`` for every saved stack, newest first.
    Corrupt/partial files are skipped (mirrors :func:`smile_msi.cohort.list_cohorts`)."""
    out = []
    d = stacks_dir()
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(d, fn)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            mtime = os.stat(path).st_mtime
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        out.append({"name": data.get("name", os.path.splitext(fn)[0]), "path": path,
                    "n_sections": len(data.get("sections", []) or []), "mtime": mtime})
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


# --------------------------------------------------------------------------- #
# stack model (pure dataclasses, JSON round-trippable)
# --------------------------------------------------------------------------- #
@dataclass
class StackSection:
    """One slot in a serial-section stack.

    Attributes
    ----------
    ref_key:
        :meth:`smile_msi.cohort.SampleRef.key` -- ties this slot to a roster sample.
    z_index:
        Ordinal position in the stack (0 = first cut).
    z_um:
        Absolute depth in microns (``z_index * spacing_um`` unless measured).
    transform:
        ``3x3`` affine (row-major, homogeneous) mapping this section's pixel grid
        into the **reference** section's grid; ``None`` until registered. Identity
        for the reference section.
    """

    ref_key: str
    z_index: int
    z_um: float = 0.0
    transform: list | None = None

    def to_dict(self) -> dict:
        return {
            "ref_key": self.ref_key,
            "z_index": int(self.z_index),
            "z_um": float(self.z_um),
            "transform": (np.asarray(self.transform, dtype=float).tolist()
                          if self.transform is not None else None),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StackSection":
        t = d.get("transform")
        return cls(
            ref_key=str(d["ref_key"]),
            z_index=int(d.get("z_index", 0)),
            z_um=float(d.get("z_um", 0.0)),
            transform=(np.asarray(t, dtype=float).tolist() if t is not None else None),
        )

    def matrix(self) -> np.ndarray | None:
        """The stored transform as a ``(3, 3)`` array, or ``None`` if unregistered."""
        if self.transform is None:
            return None
        return np.asarray(self.transform, dtype=float)


@dataclass
class SectionStack:
    """An ordered, spacing-aware stack of serial sections layered over a cohort.

    Persisted as its own JSON file (see :func:`stack_path`) so it stays independent
    of any single cohort and needs no cohort-schema migration.
    """

    name: str
    sections: list = field(default_factory=list)   # list[StackSection], ordered by z_index
    spacing_um: float = 0.0       # nominal inter-section spacing (0 = irregular / per-section z_um)
    reference: str = "previous"   # "previous" | "first" | "centroid" | "<ref_key>"
    mode: str = "rigid"           # one of STACK_MODES
    version: int = VERSION

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "sections": [s.to_dict() for s in self.sections],
            "spacing_um": float(self.spacing_um),
            "reference": self.reference,
            "mode": self.mode,
            "version": int(self.version),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SectionStack":
        secs = [StackSection.from_dict(s) for s in (d.get("sections") or [])]
        return cls(
            name=str(d.get("name", "Stack")),
            sections=secs,
            spacing_um=float(d.get("spacing_um", 0.0)),
            reference=str(d.get("reference", "previous")),
            mode=str(d.get("mode", "rigid")),
            version=int(d.get("version", VERSION)),
        )

    def ordered(self) -> list:
        """Sections sorted by ``z_index`` (stable on ties via ``ref_key``)."""
        return sorted(self.sections, key=lambda s: (s.z_index, s.ref_key))

    def with_depths(self) -> "SectionStack":
        """Return a copy whose ``z_um`` is filled from ``z_index * spacing_um`` for
        any section that has no explicit (non-zero) depth -- bookkeeping only."""
        secs = []
        for s in self.ordered():
            z_um = s.z_um if s.z_um else float(s.z_index) * float(self.spacing_um)
            secs.append(StackSection(s.ref_key, s.z_index, z_um, s.transform))
        return SectionStack(self.name, secs, self.spacing_um, self.reference,
                            self.mode, self.version)

    def save(self, path: str | None = None) -> str:
        """Write to ``path`` (defaults to :func:`stack_path` of ``name``)."""
        p = path or stack_path(self.name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return p

    @classmethod
    def load(cls, path: str) -> "SectionStack":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def make_stack(name: str, ref_keys, *, spacing_um: float = 10.0,
               reference: str = "previous", mode: str = "rigid") -> SectionStack:
    """Build a stack from an ordered list of ``SampleRef.key()`` strings.

    ``z_index`` is the position in ``ref_keys``; ``z_um = z_index * spacing_um``.
    The generated ``name`` is used verbatim -- callers must already use ``" - "``
    (never the middle dot) for any composed entity name.
    """
    if mode not in STACK_MODES:
        raise ValueError(f"mode must be one of {STACK_MODES}; got {mode!r}.")
    secs = [StackSection(ref_key=str(k), z_index=i, z_um=float(i) * float(spacing_um))
            for i, k in enumerate(ref_keys)]
    return SectionStack(name=name, sections=secs, spacing_um=float(spacing_um),
                        reference=reference, mode=mode)


# --------------------------------------------------------------------------- #
# phase correlation (clean-room Reddy & Chatterjee 1996) -- pure numpy
# --------------------------------------------------------------------------- #
def _prep_for_phase_corr(img: np.ndarray) -> np.ndarray:
    """NaN-fill, de-mean and window an image for FFT phase correlation."""
    a = np.asarray(img, dtype=float)
    a = np.where(np.isfinite(a), a, 0.0)
    a = a - a.mean()
    # A separable Hann window suppresses the FFT's wrap-around edge response so the
    # correlation peak reflects content, not the image border (Reddy & Chatterjee 1996).
    h, w = a.shape
    if h > 1 and w > 1:
        wy = np.hanning(h)[:, None]
        wx = np.hanning(w)[None, :]
        a = a * (wy * wx)
    return a


def phase_correlation_shift(moving: np.ndarray, fixed: np.ndarray,
                            *, upsample: int = 1) -> tuple[float, float]:
    """Estimate the ``(dy, dx)`` translation that maps ``moving`` onto ``fixed``.

    FFT phase correlation (Reddy & Chatterjee 1996): the normalized cross-power
    spectrum of two translated images is a unit complex exponential whose inverse
    FFT is a delta at the shift. Returns the **integer-or-subpixel** ``(row, col)``
    shift to add to ``moving``'s coordinates to align it with ``fixed``.

    Parameters
    ----------
    moving, fixed:
        Same-shape 2D images (NaN tolerated -- treated as 0 after de-meaning).
    upsample:
        ``1`` -> integer-pixel peak; ``>1`` -> local 3-point parabolic sub-pixel
        refinement around the peak (cheap, deterministic).
    """
    a = _prep_for_phase_corr(moving)
    b = _prep_for_phase_corr(fixed)
    if a.shape != b.shape:
        raise ValueError(f"phase correlation needs same-shape images; got {a.shape} vs {b.shape}.")
    Fa = np.fft.fft2(a)
    Fb = np.fft.fft2(b)
    cross = Fa * np.conj(Fb)
    denom = np.abs(cross)
    denom[denom == 0] = 1.0
    r = np.fft.ifft2(cross / denom).real
    h, w = r.shape
    peak = np.unravel_index(int(np.argmax(r)), r.shape)
    py, px = int(peak[0]), int(peak[1])

    dy = float(py)
    dx = float(px)
    if upsample and upsample > 1 and h > 2 and w > 2:
        # 3-point parabolic interpolation along each axis around the integer peak.
        def _parab(c0, cm, cp):
            denom2 = (cm - 2 * c0 + cp)
            return 0.5 * (cm - cp) / denom2 if denom2 != 0 else 0.0
        ym = r[(py - 1) % h, px]
        yp = r[(py + 1) % h, px]
        xm = r[py, (px - 1) % w]
        xp = r[py, (px + 1) % w]
        dy += _parab(r[py, px], ym, yp)
        dx += _parab(r[py, px], xm, xp)

    # Wrap from [0, N) into the signed [-N/2, N/2) range.
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w
    # cross = Fa * conj(Fb) peaks at the shift that maps fixed onto moving; the shift
    # to apply to *moving* to reach fixed is its negation.
    return (-dy, -dx)


def _translation_result(dy: float, dx: float, hw: tuple[int, int]) -> registration.RegistrationResult:
    """Express a ``(dy, dx)`` translation as a rigid :class:`RegistrationResult`.

    Built through the real :func:`registration.estimate_landmark_transform` so the
    transform machinery (matrix, ``warp_image``, ``compose``, ``invert``) is shared
    with the histology-registration engine rather than re-derived here. The
    landmarks are three non-collinear grid points and their translated images, which
    a ``rigid`` (Umeyama, no scale) fit recovers exactly.
    """
    h, w = int(hw[0]), int(hw[1])
    # moving-pixel (x=col, y=row) -> reference-pixel (x, y): a pure translation.
    src = np.array([[0.0, 0.0], [w - 1.0, 0.0], [0.0, h - 1.0]], dtype=float)
    dst = src + np.array([dx, dy], dtype=float)[None, :]
    return registration.estimate_landmark_transform(src, dst, kind="rigid")


# --------------------------------------------------------------------------- #
# section image rendering
# --------------------------------------------------------------------------- #
def section_image(ds, mz: float | None, *, tol_ppm: float = DEFAULT_TOL_PPM, reduce: str = "sum",
                  norm: str = "tic") -> np.ndarray:
    """Render one section's 2D image in **display coordinates**.

    ``mz=None`` returns the TIC image (the registration anchor default); otherwise
    the ``ion_image(mz, tol_ppm, reduce, norm)``. Routes through ``ds.ion_image`` /
    ``ds.tic_image`` so the result is already post-orientation (the whole volume
    therefore lives in display coordinates). ``norm`` defaults to ``"tic"`` here --
    ``ion_image``'s own default is ``"none"``.
    """
    if mz is None:
        return np.asarray(ds.tic_image(), dtype=float)
    return np.asarray(ds.ion_image(float(mz), tol_ppm=tol_ppm, reduce=reduce, norm=norm),
                      dtype=float)


def _resolve_reference_index(stack: SectionStack, n: int) -> int:
    """Ordinal (0..n-1) of the fixed/reference section for a stack of ``n`` sections."""
    ref = (stack.reference or "previous")
    secs = stack.ordered()
    if ref == "first" or ref == "previous":
        # "previous" propagates pairwise but the *frame* anchor is the first cut.
        return 0
    if ref == "centroid":
        return n // 2
    # explicit ref_key
    for i, s in enumerate(secs):
        if s.ref_key == ref:
            return i
    return 0


# --------------------------------------------------------------------------- #
# registration orchestration
# --------------------------------------------------------------------------- #
def register_pair(moving_img, fixed_img, *, mode: str = "rigid",
                  random_state: int = 0) -> registration.RegistrationResult:
    """Register one section image (``moving``) onto another (``fixed``).

    Returns a :class:`smile_msi.registration.RegistrationResult` whose transform maps
    **moving-pixel ``(col, row)`` -> fixed-pixel ``(x, y)``** -- exactly the
    convention :func:`registration.warp_image` consumes, so the channel resample is
    ``registration.warp_image(res, moving_channel, fixed.shape)``.

    Modes
    -----
    ``"rigid"``:
        FFT phase correlation translation (Reddy & Chatterjee 1996), pure numpy /
        scipy. Deterministic; the parameter-free default for adjacent serial sections
        of one block.
    ``"affine"``:
        Delegates to :func:`registration.estimate_intensity_transform` (SimpleITK,
        Mattes-MI) -- raises a clear ``ImportError`` (its ``register`` extra) when
        SimpleITK is absent. ``random_state`` seeds the MI sampler.
    ``"deformable"``:
        Affine init then :func:`registration.refine_bspline` (SimpleITK B-spline
        FFD). Exploratory -- it can manufacture continuity across genuine biological
        change, so ``"rigid"`` is the default.
    """
    if mode not in STACK_MODES:
        raise ValueError(f"mode must be one of {STACK_MODES}; got {mode!r}.")
    moving = np.asarray(moving_img, dtype=float)
    fixed = np.asarray(fixed_img, dtype=float)
    if moving.shape != fixed.shape:
        raise ValueError(
            f"register_pair needs same-shape section images; got {moving.shape} vs {fixed.shape}.")

    if mode == "rigid":
        dy, dx = phase_correlation_shift(moving, fixed, upsample=10)
        return _translation_result(dy, dx, fixed.shape)

    # affine / deformable: intensity-based via registration.py's SimpleITK path.
    # fixed must be the registration "fixed" (target); moving is aligned to it.
    f = np.where(np.isfinite(fixed), fixed, 0.0)
    m = np.where(np.isfinite(moving), moving, 0.0)
    affine = registration.estimate_intensity_transform(f, m, kind="affine",
                                                       random_state=int(random_state))
    if mode == "affine":
        return affine
    return registration.refine_bspline(f, m, affine, random_state=int(random_state))


def register_stack(stack: SectionStack, loader, *, channel_mz: float | None = None,
                   mode: str | None = None, reference: str | None = None,
                   tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic",
                   random_state: int = 0, progress=None) -> SectionStack:
    """Populate every :attr:`StackSection.transform` and return a **new** stack.

    For each section a MOVING image is rendered -- the section TIC by default, or a
    chosen anatomical-anchor ion ``channel_mz`` -- and aligned to the FIXED
    reference section (per ``reference``) via :func:`register_pair`. The reference
    section's transform is identity. Sections the ``loader`` skips (``load`` returns
    ``None``: non-imzML / moved / lost region) keep their existing transform.

    Deterministic given ``random_state`` (the rigid default is deterministic by
    construction; only the SimpleITK paths consume the seed).

    Parameters
    ----------
    stack:
        The stack whose sections to register.
    loader:
        A :class:`smile_msi.cohort.SectionLoader` (or anything exposing ``load(ref)``
        -> ``(ds, pix) | None``). Sections are addressed by an object carrying a
        ``.source`` -- here the loader is keyed by ``ref_key`` via the ``refs`` map.
    channel_mz:
        Anatomical-anchor m/z to register on. ``None`` -> TIC.
    mode / reference:
        Override the stack's ``mode`` / ``reference`` for this run.
    """
    use_mode = mode or stack.mode
    if use_mode not in STACK_MODES:
        raise ValueError(f"mode must be one of {STACK_MODES}; got {use_mode!r}.")
    work = SectionStack(stack.name, [StackSection(s.ref_key, s.z_index, s.z_um, s.transform)
                                     for s in stack.ordered()],
                        stack.spacing_um, reference or stack.reference, use_mode, stack.version)
    secs = work.sections
    n = len(secs)
    if n == 0:
        return work
    ref_idx = _resolve_reference_index(work, n)

    # Render every section's moving image once.
    images: list[np.ndarray | None] = []
    for i, s in enumerate(secs):
        if progress is not None:
            progress(i, 2 * n, f"render {s.ref_key}")
        ds = _load_ds(loader, s)
        images.append(None if ds is None else section_image(
            ds, channel_mz, tol_ppm=tol_ppm, norm=norm))

    fixed = images[ref_idx]
    if fixed is None:
        raise ValueError("reference section could not be loaded (loader returned None).")

    for i, s in enumerate(secs):
        if progress is not None:
            progress(n + i, 2 * n, f"register {s.ref_key}")
        if i == ref_idx:
            s.transform = np.eye(3).tolist()
            continue
        mov = images[i]
        if mov is None:
            continue                          # un-loadable section keeps its prior transform
        # "previous" propagates: register to the neighbour towards the reference and
        # compose with that neighbour's already-solved transform; all others register
        # directly to the reference frame.
        if work.reference == "previous":
            neighbour = i - 1 if i > ref_idx else i + 1
            res = register_pair(mov, images[neighbour] if images[neighbour] is not None else fixed,
                                mode=use_mode, random_state=random_state)
            prev_M = secs[neighbour].matrix()
            if prev_M is not None and res.matrix is not None:
                s.transform = (np.asarray(prev_M) @ np.asarray(res.matrix)).tolist()
            elif res.matrix is not None:
                s.transform = res.matrix.tolist()
        else:
            res = register_pair(mov, fixed, mode=use_mode, random_state=random_state)
            if res.matrix is not None:
                s.transform = res.matrix.tolist()
    return work


def _load_ds(loader, section: StackSection):
    """Resolve a section to its ``MSIDataset`` via the loader, tolerating either a
    ``{ref_key: ref}`` map being attached or a loader that takes the section itself."""
    refmap = getattr(loader, "refs_by_key", None)
    ref = section
    if isinstance(refmap, dict):
        ref = refmap.get(section.ref_key, section)
    res = loader.load(ref)
    if res is None:
        return None
    ds, _pix = res
    return ds


# --------------------------------------------------------------------------- #
# volume build
# --------------------------------------------------------------------------- #
@dataclass
class IonVolume:
    """A single ion channel resampled into a ``(Z, H, W)`` molecular volume.

    Attributes
    ----------
    data:
        ``(Z, H, W)`` float32 array; ``NaN`` marks no-data pixels.
    z_um:
        ``(Z,)`` absolute section depths in microns.
    mz:
        The reconstructed ion's m/z.
    pixel_size_um:
        In-plane pixel size (microns), or ``None`` if unknown.
    """

    data: np.ndarray
    z_um: np.ndarray
    mz: float
    pixel_size_um: float | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(x) for x in self.data.shape)  # type: ignore[return-value]

    def orthoslices(self, z: int | None = None, y: int | None = None,
                    x: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Axial / coronal / sagittal 2D slices at the given indices.

        Returns ``(axial, coronal, sagittal)`` where ``axial = data[z]`` is ``(H, W)``,
        ``coronal = data[:, y, :]`` is ``(Z, W)`` and ``sagittal = data[:, :, x]`` is
        ``(Z, H)``. Indices default to the volume centre.
        """
        Z, H, W = self.shape
        zi = Z // 2 if z is None else int(z)
        yi = H // 2 if y is None else int(y)
        xi = W // 2 if x is None else int(x)
        axial = self.data[zi]
        coronal = self.data[:, yi, :]
        sagittal = self.data[:, :, xi]
        return axial, coronal, sagittal

    def mip(self, axis: int = 0) -> np.ndarray:
        """NaN-safe maximum-intensity projection along ``axis``.

        Columns that are entirely no-data stay ``NaN`` (rather than collapsing to a
        spurious value), so the MIP keeps the same no-data semantics as the volume.
        """
        data = self.data
        all_nan = np.all(np.isnan(data), axis=axis)
        with np.errstate(all="ignore"):
            out = np.nanmax(np.where(np.isnan(data), -np.inf, data), axis=axis)
        out = np.where(all_nan, np.nan, out)
        return out.astype(np.float32)

    def composite(self, axis: int = 0, alpha: float = 0.04) -> np.ndarray:
        """Front-to-back alpha-composited projection along ``axis`` (Levoy 1988).

        Each voxel contributes ``c * a * (1 - accumulated_opacity)`` where its local
        opacity ``a = alpha * normalized_intensity``. NaN voxels are transparent. The
        result is monotonically non-decreasing in ``alpha``. Output is normalized to
        the volume's finite max so projections are comparable across ``alpha``.
        """
        data = np.moveaxis(self.data, axis, 0)        # (N, ...) -- composite over axis 0
        finite = data[np.isfinite(data)]
        vmax = float(finite.max()) if finite.size else 0.0
        if vmax <= 0:
            return np.zeros(data.shape[1:], dtype=np.float32)
        a = float(np.clip(alpha, 0.0, 1.0))
        out = np.zeros(data.shape[1:], dtype=float)
        trans = np.ones(data.shape[1:], dtype=float)  # remaining transparency
        for k in range(data.shape[0]):
            c = data[k]
            valid = np.isfinite(c)
            cn = np.where(valid, np.clip(c, 0.0, None) / vmax, 0.0)
            opacity = a * cn                          # local per-voxel opacity
            contrib = cn * opacity * trans
            out += contrib
            trans *= (1.0 - opacity)
        return out.astype(np.float32)


def build_channel_volume(stack: SectionStack, mz: float, loader, *,
                         tol_ppm: float = DEFAULT_TOL_PPM, reduce: str = "sum", norm: str = "tic",
                         z_interp: str = "nearest", pixel_size_um: float | None = None,
                         random_state: int = 0, progress=None) -> IonVolume:
    """Resample one ion channel through a registered stack into a ``(Z, H, W)`` volume.

    Streams each section once via ``loader.load``, renders its
    ``ion_image(mz, tol_ppm, reduce, norm)`` in display coordinates, warps it by the
    section's stored transform onto the **reference** section's ``(H, W)`` grid (via
    :func:`registration.warp_image`), and stacks the registered images. ``NaN`` marks
    no-data. Sections the loader skips (``load`` -> ``None``) are dropped from the
    volume. The reference section defines the output ``(H, W)``.

    ``z_interp`` controls anisotropic-z handling for the returned ``z_um`` axis:
    ``"nearest"`` keeps one plane per section at its measured depth; ``"linear"`` is
    reserved for resampling onto an isotropic z grid by the caller (the data array is
    unchanged here -- z resampling is a display concern). ``norm`` defaults to
    ``"tic"`` (``ion_image``'s own default is ``"none"``).
    """
    if z_interp not in ("nearest", "linear"):
        raise ValueError(f"z_interp must be 'nearest' or 'linear'; got {z_interp!r}.")
    filled = stack.with_depths()
    secs = filled.ordered()
    n = len(secs)
    if n == 0:
        raise ValueError("stack has no sections.")
    ref_idx = _resolve_reference_index(filled, n)

    # The reference section fixes the output grid.
    ref_ds = _load_ds(loader, secs[ref_idx])
    if ref_ds is None:
        raise ValueError("reference section could not be loaded for the volume build.")
    ref_img = section_image(ref_ds, mz, tol_ppm=tol_ppm, reduce=reduce, norm=norm)
    out_hw = ref_img.shape

    planes: list[np.ndarray] = []
    depths: list[float] = []
    for i, s in enumerate(secs):
        if progress is not None:
            progress(i, n, f"stack {s.ref_key}")
        if i == ref_idx:
            planes.append(np.asarray(ref_img, dtype=np.float32))
            depths.append(float(s.z_um))
            continue
        ds = _load_ds(loader, s)
        if ds is None:
            continue
        img = section_image(ds, mz, tol_ppm=tol_ppm, reduce=reduce, norm=norm)
        M = s.matrix()
        if M is None:
            # Unregistered: assume identity but require matching grid; resample if not.
            warped = _warp_to_grid(img, np.eye(3), out_hw)
        else:
            warped = _warp_to_grid(img, M, out_hw)
        planes.append(warped.astype(np.float32))
        depths.append(float(s.z_um))

    data = np.stack(planes, axis=0).astype(np.float32)
    return IonVolume(data=data, z_um=np.asarray(depths, dtype=float), mz=float(mz),
                     pixel_size_um=pixel_size_um)


def _warp_to_grid(image: np.ndarray, matrix: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Resample ``image`` onto ``out_hw`` through a ``3x3`` transform, preserving
    NaN no-data.

    Builds a :class:`registration.RegistrationResult` around ``matrix`` and warps via
    :func:`registration.warp_image` (pure-numpy inverse sampling). ``warp_image`` has
    no NaN concept, so no-data is tracked with a companion mask warped at the same
    transform (``order=0``) and reapplied.
    """
    M = np.asarray(matrix, dtype=float)
    res = registration.RegistrationResult(kind="affine", matrix=M)
    img = np.asarray(image, dtype=float)
    valid = np.isfinite(img).astype(float)
    filled = np.where(np.isfinite(img), img, 0.0)
    warped = registration.warp_image(res, filled, out_hw, order=1, cval=0.0)
    mask = registration.warp_image(res, valid, out_hw, order=1, cval=0.0)
    out = np.where(mask > 0.5, warped, np.nan)
    return out


# --------------------------------------------------------------------------- #
# optional GL volume render (lazy; viz3d extra)
# --------------------------------------------------------------------------- #
def gl_volume_item(volume: IonVolume, *, cmap: str = "viridis"):
    """Build a ``pyqtgraph.opengl.GLVolumeItem`` for interactive 3D rendering.

    Lazy-imports ``pyqtgraph.opengl`` and raises a clear, actionable error naming the
    ``viz3d`` extra when it is missing -- the numpy :meth:`IonVolume.mip` /
    :meth:`IonVolume.composite` path is the offline, CPU-only default and needs no GL.

    Raises
    ------
    ImportError
        If ``pyqtgraph[opengl]`` (the ``viz3d`` extra) is not installed.
    """
    try:
        import pyqtgraph.opengl as gl
    except ImportError as e:  # pragma: no cover - exercised via importorskip / monkeypatch
        raise ImportError(
            "Interactive 3D volume rendering needs the 'viz3d' extra "
            "(pyqtgraph[opengl]). Install it with:\n"
            "    uv pip install 'pyqtgraph[opengl]'\n"
            "(or: uv sync --extra viz3d). The numpy MIP / alpha-composite path "
            "works without it."
        ) from e

    data = np.asarray(volume.data, dtype=float)
    finite = data[np.isfinite(data)]
    vmax = float(finite.max()) if finite.size else 1.0
    vmin = float(finite.min()) if finite.size else 0.0
    rng = (vmax - vmin) or 1.0
    norm = np.clip((np.where(np.isfinite(data), data, vmin) - vmin) / rng, 0.0, 1.0)

    rgba = np.zeros(data.shape + (4,), dtype=np.ubyte)
    grey = (norm * 255).astype(np.ubyte)
    rgba[..., 0] = grey
    rgba[..., 1] = grey
    rgba[..., 2] = grey
    rgba[..., 3] = (norm * 255).astype(np.ubyte)        # opacity tracks intensity
    return gl.GLVolumeItem(rgba, smooth=True)
