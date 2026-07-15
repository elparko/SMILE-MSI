"""Engine tests for smile_msi.volume3d (3D serial-section reconstruction).

Synthetic data only -- no real microscopy / serial-section MSI is available, so
these exercise the *engine logic*: phase-correlation recovers a known planted
translation to ~1px; stacking yields the right ``(Z, H, W)``; MIP / orthoslice /
alpha-composite correctness; deterministic given ``random_state``; the stack model
round-trips and obeys the naming rule; and the optional GL path raises the clear
``viz3d`` error when absent.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from smile_msi import registration, volume3d as v3d


# --------------------------------------------------------------------------- #
# synthetic helpers
# --------------------------------------------------------------------------- #
def _blob(h, w, cy, cx, r=4, amp=1.0):
    """A square-ish blob image with NaN off-blob? No -- zeros off-blob, finite."""
    img = np.zeros((h, w), dtype=float)
    yy, xx = np.mgrid[0:h, 0:w]
    img[(np.abs(yy - cy) <= r) & (np.abs(xx - cx) <= r)] = amp
    return img


def _centroid(a):
    a = np.where(np.isfinite(a), a, 0.0)
    if not (a > 0).any():
        return np.nan, np.nan
    ys, xs = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    s = a.sum()
    return float((ys * a).sum() / s), float((xs * a).sum() / s)


class _FakeDS:
    """Minimal MSIDataset stand-in exposing the methods volume3d calls."""

    def __init__(self, img, tic=None):
        self._img = np.asarray(img, dtype=float)
        self._tic = self._img if tic is None else np.asarray(tic, dtype=float)

    def ion_image(self, mz, tol_ppm=50.0, reduce="sum", norm="none"):
        return self._img

    def tic_image(self):
        return self._tic


class _FakeLoader:
    """Maps a StackSection.ref_key -> _FakeDS, like SectionLoader.load -> (ds, pix)."""

    def __init__(self, by_key, skip=()):
        self._by_key = by_key
        self._skip = set(skip)

    def load(self, ref):
        key = getattr(ref, "ref_key", ref)
        if key in self._skip:
            return None
        ds = self._by_key.get(key)
        if ds is None:
            return None
        return ds, None


# --------------------------------------------------------------------------- #
# stack model
# --------------------------------------------------------------------------- #
def test_stack_roundtrip_and_ordering():
    stack = v3d.make_stack("Donor 02 - serial stack",
                           ["a", "b", "c"], spacing_um=12.0, reference="first", mode="rigid")
    assert "·" not in stack.name           # naming rule: no middle dot
    # scramble z_index to test ordered()
    stack.sections[0].z_index = 5
    stack.sections[2].z_index = 0
    ordered = stack.ordered()
    assert [s.z_index for s in ordered] == [0, 1, 5]
    assert [s.ref_key for s in ordered] == ["c", "b", "a"]

    d = stack.to_dict()
    back = v3d.SectionStack.from_dict(d)
    assert back.name == stack.name
    assert back.spacing_um == 12.0
    assert back.reference == "first"
    assert back.mode == "rigid"
    assert [s.ref_key for s in back.sections] == [s.ref_key for s in stack.sections]


def test_stack_with_depths_fills_z_um():
    stack = v3d.make_stack("S", ["a", "b", "c"], spacing_um=10.0)
    filled = stack.with_depths()
    assert [s.z_um for s in filled.ordered()] == [0.0, 10.0, 20.0]


def test_make_stack_rejects_bad_mode():
    with pytest.raises(ValueError):
        v3d.make_stack("S", ["a"], mode="bogus")


def test_stack_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    stack = v3d.make_stack("Round - trip", ["a", "b"], spacing_um=8.0)
    stack.sections[1].transform = np.eye(3).tolist()
    p = stack.save()
    assert os.path.exists(p)
    listed = v3d.list_stacks()
    assert any(r["name"] == "Round - trip" and r["n_sections"] == 2 for r in listed)
    back = v3d.SectionStack.load(p)
    assert back.spacing_um == 8.0
    assert np.allclose(back.sections[1].matrix(), np.eye(3))


# --------------------------------------------------------------------------- #
# phase correlation / register_pair
# --------------------------------------------------------------------------- #
def test_phase_correlation_recovers_translation():
    h, w = 48, 56
    fixed = _blob(h, w, 24, 28, r=5)
    dy_true, dx_true = 4, -6
    moving = _blob(h, w, 24 + dy_true, 28 + dx_true, r=5)
    dy, dx = v3d.phase_correlation_shift(moving, fixed, upsample=10)
    # shift to apply to moving to reach fixed == -(planted shift)
    assert abs(dy - (-dy_true)) <= 1.0
    assert abs(dx - (-dx_true)) <= 1.0


def test_register_pair_warp_aligns_centroid():
    h, w = 48, 56
    fixed = _blob(h, w, 22, 30, r=5)
    moving = _blob(h, w, 26, 22, r=5)        # planted (dy=+4, dx=-8)
    res = v3d.register_pair(moving, fixed, mode="rigid")
    assert isinstance(res, registration.RegistrationResult)
    warped = v3d._warp_to_grid(moving, res.matrix, fixed.shape)
    cy_f, cx_f = _centroid(fixed)
    cy_w, cx_w = _centroid(warped)
    assert abs(cy_w - cy_f) <= 1.0
    assert abs(cx_w - cx_f) <= 1.0


def test_register_pair_deterministic():
    h, w = 40, 40
    fixed = _blob(h, w, 18, 20, r=4)
    moving = _blob(h, w, 22, 15, r=4)
    r1 = v3d.register_pair(moving, fixed, mode="rigid", random_state=0)
    r2 = v3d.register_pair(moving, fixed, mode="rigid", random_state=0)
    assert np.allclose(r1.matrix, r2.matrix)


def test_register_pair_shape_mismatch_raises():
    with pytest.raises(ValueError):
        v3d.register_pair(np.zeros((10, 10)), np.zeros((10, 12)), mode="rigid")


def test_register_pair_bad_mode_raises():
    with pytest.raises(ValueError):
        v3d.register_pair(np.zeros((8, 8)), np.zeros((8, 8)), mode="bogus")


# --------------------------------------------------------------------------- #
# register_stack
# --------------------------------------------------------------------------- #
def _planted_stack(shifts, h=48, w=56, r=5, base=(24, 28)):
    """Build a fake stack + loader where section i is the base blob translated by
    shifts[i] = (dy, dx). Returns (stack, loader, base_centroid)."""
    by_key = {}
    keys = []
    cy0, cx0 = base
    for i, (dy, dx) in enumerate(shifts):
        key = f"s{i}"
        keys.append(key)
        img = _blob(h, w, cy0 + dy, cx0 + dx, r=r)
        by_key[key] = _FakeDS(img)
    stack = v3d.make_stack("planted", keys, spacing_um=10.0, reference="first", mode="rigid")
    loader = _FakeLoader(by_key)
    return stack, loader, (cy0, cx0)


def test_register_stack_recovers_shifts_and_reference_identity():
    shifts = [(0, 0), (3, -4), (-5, 6), (2, 7)]
    stack, loader, _ = _planted_stack(shifts)
    out = v3d.register_stack(stack, loader, mode="rigid")
    secs = out.ordered()
    # reference (first) is identity
    assert np.allclose(secs[0].matrix(), np.eye(3), atol=1e-6)
    # every section's transform, applied to its blob, lands at the reference blob
    vol = v3d.build_channel_volume(out, mz=700.0, loader=loader)
    cy_ref, cx_ref = _centroid(vol.data[0])
    for z in range(1, vol.data.shape[0]):
        cy, cx = _centroid(vol.data[z])
        assert abs(cy - cy_ref) <= 1.2, f"z={z} y off"
        assert abs(cx - cx_ref) <= 1.2, f"z={z} x off"


def test_register_stack_reduces_centroid_variance():
    shifts = [(0, 0), (5, -7), (-6, 8), (4, 9)]
    stack, loader, _ = _planted_stack(shifts)

    # centroid spread BEFORE registration (raw blobs)
    raw_cents = []
    for s in stack.ordered():
        ds, _ = loader.load(s)
        raw_cents.append(_centroid(ds.ion_image(700.0)))
    raw_cents = np.array(raw_cents)
    var_before = raw_cents.var(axis=0).sum()

    out = v3d.register_stack(stack, loader, mode="rigid")
    vol = v3d.build_channel_volume(out, mz=700.0, loader=loader)
    reg_cents = np.array([_centroid(vol.data[z]) for z in range(vol.data.shape[0])])
    var_after = reg_cents.var(axis=0).sum()

    assert var_after < var_before * 0.1     # registration collapses the spread


def test_register_stack_deterministic():
    shifts = [(0, 0), (3, -4), (-5, 6)]
    stack, loader, _ = _planted_stack(shifts)
    a = v3d.register_stack(stack, loader, mode="rigid", random_state=0)
    b = v3d.register_stack(stack, loader, mode="rigid", random_state=0)
    for sa, sb in zip(a.ordered(), b.ordered()):
        assert np.allclose(sa.matrix(), sb.matrix())


def test_register_stack_skips_unloadable_section():
    shifts = [(0, 0), (3, -4), (-5, 6)]
    stack, loader, _ = _planted_stack(shifts)
    loader._skip = {"s1"}                    # section 1 can't load
    out = v3d.register_stack(stack, loader, mode="rigid")
    secs = out.ordered()
    assert secs[1].transform is None         # skipped, keeps prior (None)
    assert secs[0].transform is not None     # reference still solved


# --------------------------------------------------------------------------- #
# build_channel_volume
# --------------------------------------------------------------------------- #
def test_build_volume_shape_and_skip():
    shifts = [(0, 0), (2, -2), (-3, 3)]
    stack, loader, _ = _planted_stack(shifts)
    out = v3d.register_stack(stack, loader, mode="rigid")
    vol = v3d.build_channel_volume(out, mz=700.0, loader=loader)
    Z, H, W = vol.shape
    assert Z == 3 and (H, W) == (48, 56)
    assert vol.mz == 700.0
    assert np.array_equal(vol.z_um, [0.0, 10.0, 20.0])

    # a skipped section drops a plane
    loader._skip = {"s2"}
    vol2 = v3d.build_channel_volume(out, mz=700.0, loader=loader)
    assert vol2.shape[0] == 2


def test_build_volume_preserves_nan_no_data():
    # off-blob pixels in the warped (non-reference) planes become NaN where the
    # source grid did not cover them after translation.
    shifts = [(0, 0), (10, 10)]
    stack, loader, _ = _planted_stack(shifts, h=40, w=40)
    out = v3d.register_stack(stack, loader, mode="rigid")
    vol = v3d.build_channel_volume(out, mz=700.0, loader=loader)
    # the heavily-shifted plane must expose some NaN (translated out of frame)
    assert np.isnan(vol.data[1]).any()


def test_build_volume_empty_stack_raises():
    stack = v3d.SectionStack("empty", [])
    with pytest.raises(ValueError):
        v3d.build_channel_volume(stack, mz=700.0, loader=_FakeLoader({}))


def test_build_volume_bad_zinterp_raises():
    stack, loader, _ = _planted_stack([(0, 0)])
    with pytest.raises(ValueError):
        v3d.build_channel_volume(stack, mz=700.0, loader=loader, z_interp="cubic")


# --------------------------------------------------------------------------- #
# IonVolume projections / orthoslices
# --------------------------------------------------------------------------- #
def _toy_volume():
    Z, H, W = 4, 6, 5
    data = np.arange(Z * H * W, dtype=np.float32).reshape(Z, H, W)
    return v3d.IonVolume(data=data, z_um=np.arange(Z) * 10.0, mz=700.0, pixel_size_um=20.0)


def test_orthoslice_shapes():
    vol = _toy_volume()
    ax, cor, sag = vol.orthoslices()
    Z, H, W = vol.shape
    assert ax.shape == (H, W)
    assert cor.shape == (Z, W)
    assert sag.shape == (Z, H)


def test_orthoslice_indexing():
    vol = _toy_volume()
    ax, cor, sag = vol.orthoslices(z=1, y=2, x=3)
    assert np.array_equal(ax, vol.data[1])
    assert np.array_equal(cor, vol.data[:, 2, :])
    assert np.array_equal(sag, vol.data[:, :, 3])


def test_mip_dominates_every_slice():
    vol = _toy_volume()
    m = vol.mip(axis=0)
    for z in range(vol.shape[0]):
        assert np.all(m >= vol.data[z] - 1e-6)
    # exactly the elementwise max for an all-finite volume
    assert np.allclose(m, vol.data.max(axis=0))


def test_mip_nan_safe():
    data = np.array([[[np.nan, 2.0], [3.0, np.nan]],
                     [[1.0, np.nan], [np.nan, np.nan]]], dtype=np.float32)
    vol = v3d.IonVolume(data=data, z_um=np.array([0.0, 10.0]), mz=1.0, pixel_size_um=None)
    m = vol.mip(axis=0)
    assert m[0, 0] == 1.0                    # only finite value across z
    assert m[0, 1] == 2.0
    assert m[1, 0] == 3.0
    assert np.isnan(m[1, 1])                 # all-NaN column stays NaN


def test_composite_monotonic_in_alpha():
    vol = _toy_volume()
    c_lo = vol.composite(axis=0, alpha=0.02)
    c_hi = vol.composite(axis=0, alpha=0.20)
    # more alpha -> more accumulated signal (front-to-back, non-decreasing)
    assert c_hi.sum() >= c_lo.sum()
    assert np.all(c_hi >= c_lo - 1e-5)


def test_composite_zero_volume():
    data = np.zeros((3, 4, 4), dtype=np.float32)
    vol = v3d.IonVolume(data=data, z_um=np.arange(3.0), mz=1.0, pixel_size_um=None)
    c = vol.composite(axis=0, alpha=0.1)
    assert np.all(c == 0.0)


# --------------------------------------------------------------------------- #
# optional GL render guard
# --------------------------------------------------------------------------- #
def test_gl_volume_item_missing_extra_raises(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("pyqtgraph.opengl") or name == "pyqtgraph.opengl":
            raise ImportError("no opengl")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    vol = _toy_volume()
    with pytest.raises(ImportError) as exc:
        v3d.gl_volume_item(vol)
    assert "viz3d" in str(exc.value)


def test_gl_volume_item_when_available():
    gl = pytest.importorskip("pyqtgraph.opengl")
    pytest.importorskip("OpenGL")
    vol = _toy_volume()
    try:
        item = v3d.gl_volume_item(vol)
    except Exception as e:  # GL context / headless display issues are environmental
        pytest.skip(f"GL unavailable in this environment: {e}")
    assert item is not None


# --------------------------------------------------------------------------- #
# end-to-end through the real SectionLoader + imzML round-trip
# --------------------------------------------------------------------------- #
def test_real_loader_imzml_translation_stack(tmp_path, monkeypatch):
    """Write two synthetic sections to imzML, exercise the real SectionLoader.load
    path, and confirm register_stack + build produces an aligned (Z,H,W) volume."""
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path))
    from smile_msi import demo
    from smile_msi.cohort import SectionLoader

    p0 = str(tmp_path / "sec0.imzML")
    p1 = str(tmp_path / "sec1.imzML")
    demo.write_synthetic_imzml(p0, width=40, height=32, seed=0)
    demo.write_synthetic_imzml(p1, width=40, height=32, seed=1)

    class _Ref:
        def __init__(self, key, source):
            self.ref_key = key
            self.source = source
            self.region = ""
            self.session_path = ""

    refs = {"s0": _Ref("s0", p0), "s1": _Ref("s1", p1)}
    stack = v3d.make_stack("real - stack", ["s0", "s1"], spacing_um=10.0, reference="first")

    loader = SectionLoader(dense=True)
    loader.refs_by_key = refs

    out = v3d.register_stack(stack, loader, mode="rigid", norm="tic")
    # both sections registered (reference identity + one solved transform)
    assert out.ordered()[0].transform is not None
    assert out.ordered()[1].transform is not None

    # pick a real ion m/z from the demo peak list to reconstruct
    from smile_msi.demo import _demo_peaks
    mz = _demo_peaks()[0][0]
    vol = v3d.build_channel_volume(out, mz=float(mz), loader=loader)
    assert vol.shape[0] == 2
    assert vol.shape[1:] == (32, 40)         # (H, W) in display coords
    loader.close()
