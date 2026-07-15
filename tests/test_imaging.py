"""Tests for the ion-image display helpers (smile_msi.imaging) — contrast
controls and multi-channel overlays.

Pure/headless: small in-memory numpy fixtures, no GUI, no files, no network.
``apply_colormap`` touches matplotlib's object-oriented ``colormaps`` registry
only (no pyplot / display). ``relative_max`` is already exercised in
``test_export.py`` (% of clip value); here we extend coverage to its hotspot /
off-tissue / non-positive-anchor branches without duplicating that case.
"""
import numpy as np
import pytest

from smile_msi import imaging


# --------------------------------------------------------------------------- #
# quantile_clip
# --------------------------------------------------------------------------- #
def test_quantile_clip_drops_top_hotspots():
    """high=99 clips the saturating top pixels down to the 99th-percentile value
    while leaving the bulk untouched."""
    img = np.ones((10, 10))
    img[0, 0] = 1000.0            # single saturating hotspot
    clipped = imaging.quantile_clip(img, low=0.0, high=99.0)
    assert clipped.max() < 1000.0
    assert clipped.max() <= img.max()
    # the bulk (1.0) is below the clip ceiling, so it is unchanged
    assert clipped[5, 5] == pytest.approx(1.0)
    assert np.all(np.isfinite(clipped))


def test_quantile_clip_empty_returns_input():
    """An empty array has no finite values; the function returns it unchanged
    (and without raising)."""
    empty = np.empty((0,), float)
    out = imaging.quantile_clip(empty)
    assert out.shape == (0,)
    assert out.size == 0


def test_quantile_clip_all_nan_returns_input_untouched():
    """All-NaN input → no finite values → original array returned (NaNs preserved)."""
    nan_img = np.full((4, 5), np.nan)
    out = imaging.quantile_clip(nan_img)
    assert out.shape == (4, 5)
    assert np.all(np.isnan(out))


def test_quantile_clip_uniform_returns_input():
    """A uniform (constant) array has hi == lo, so the early-out returns it
    unchanged rather than collapsing it."""
    uniform = np.full((5, 6), 7.0)
    out = imaging.quantile_clip(uniform, high=99.0)
    assert np.all(out == 7.0)
    assert out.shape == (5, 6)


def test_quantile_clip_zeros_max_is_zero():
    """Regression mirror of test_msi: an all-zeros image stays all-zeros."""
    assert imaging.quantile_clip(np.zeros((5, 6)), high=99).max() == 0


def test_quantile_clip_ignores_nan_when_clipping():
    """Off-tissue NaNs do not skew the percentile and remain NaN in the output.

    The tissue spans a finite range so the [low, high] cut points differ (hi > lo)
    and the clip actually applies."""
    img = np.linspace(1.0, 20.0, 16).reshape(4, 4)   # spread tissue, no degenerate pct
    img[0, 0] = 500.0                                  # saturating hotspot
    img[1, 1] = np.nan                                 # off-tissue
    out = imaging.quantile_clip(img, low=0.0, high=90.0)
    assert np.isnan(out[1, 1])               # NaN preserved
    assert out[0, 0] < 500.0                  # hotspot clipped down to the ceiling
    assert np.nanmax(out) < 500.0


# --------------------------------------------------------------------------- #
# relative_max  (extends test_export.py, no duplication)
# --------------------------------------------------------------------------- #
def test_relative_max_hotspot_over_100():
    """A genuine hotspot above the clip anchor reads over 100%."""
    img = np.ones((10, 10))
    img[0, 0] = 2.0                           # 2x the ~1.0 anchor
    val = imaging.relative_max(img, high=99.0)
    assert val == pytest.approx(200.0, abs=2.0)
    assert np.isfinite(val)


def test_relative_max_flat_image_is_100():
    """When max equals the anchor (uniform tissue), the peak is exactly 100%."""
    img = np.full((8, 8), 3.0)
    assert imaging.relative_max(img, high=99.0) == pytest.approx(100.0)


def test_relative_max_offtissue_nan_ignored():
    """Off-tissue NaNs are dropped before computing both the anchor and the max."""
    img = np.full((6, 6), np.nan)
    img[2:4, 2:4] = 1.0
    img[2, 2] = 1.5
    val = imaging.relative_max(img, high=99.0)
    assert np.isfinite(val)
    assert val >= 100.0


def test_relative_max_empty_is_zero():
    """Empty array → 0.0 (safe, no raise)."""
    assert imaging.relative_max(np.empty((0,), float)) == 0.0


def test_relative_max_nonpositive_anchor_is_zero():
    """A non-positive anchor (e.g. all-zero tissue) yields 0.0 rather than dividing."""
    assert imaging.relative_max(np.zeros((5, 5))) == 0.0
    # negative bulk → anchor <= 0 → 0.0
    assert imaging.relative_max(np.full((5, 5), -2.0)) == 0.0


# --------------------------------------------------------------------------- #
# relative_window  (relative-intensity contrast mapping)
# --------------------------------------------------------------------------- #
def test_relative_window_maps_to_unit_interval():
    """Output is confined to [0, 1] and the anchor%-relative window controls the
    contrast: hi=100 puts the anchor (the clip-percentile value) at ~1.0."""
    img = np.ones((10, 10))
    img[0, 0] = 1.0                           # uniform → anchor ~= 1.0
    out = imaging.relative_window(img, lo=0.0, hi=100.0, clip=99.0)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out[5, 5] == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.isfinite(out))


def test_relative_window_contrast_lo_hi_stretch():
    """lo/hi are percentages of the anchor: a value below vlo maps to 0, a value
    at vhi maps to 1, and a midpoint lands in between."""
    # anchor (99th pct) ~ 100; window 20%..80% of anchor => vlo=20, vhi=80
    img = np.full((10, 10), 100.0)            # bulk + anchor at 100
    img[0, 0] = 10.0                           # below vlo
    img[0, 1] = 50.0                           # midpoint of [20, 80]
    img[0, 2] = 80.0                           # at vhi
    out = imaging.relative_window(img, lo=20.0, hi=80.0, clip=99.0)
    assert out[0, 0] == pytest.approx(0.0)                 # clamped low
    assert out[0, 1] == pytest.approx(0.5, abs=0.05)       # midpoint
    assert out[0, 2] == pytest.approx(1.0, abs=1e-6)       # at top of window


def test_relative_window_preserves_offtissue_nan():
    """NaN background passes through np.clip unchanged (stays transparent)."""
    img = np.full((5, 5), np.nan)
    img[1:4, 1:4] = 5.0
    img[2, 2] = 9.0
    out = imaging.relative_window(img, lo=0.0, hi=100.0, clip=99.0)
    assert np.isnan(out[0, 0])
    assert np.all((out[np.isfinite(out)] >= 0.0) & (out[np.isfinite(out)] <= 1.0))


def test_relative_window_empty_returns_input():
    """Empty array → returned as-is (no finite values to anchor on)."""
    empty = np.empty((0,), float)
    out = imaging.relative_window(empty, lo=0.0, hi=100.0, clip=99.0)
    assert out.size == 0


def test_relative_window_nonpositive_anchor_clamps_to_unit():
    """anchor <= 0 → fall back to a plain [0,1] clip (no division by the anchor)."""
    img = np.array([[-1.0, 0.0], [0.5, 2.0]])   # 99th pct ~ 2 -> positive; force <=0 below
    zeros = np.zeros((4, 4))                     # anchor == 0
    out = imaging.relative_window(zeros, lo=0.0, hi=100.0, clip=99.0)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.all(out == 0.0)                    # clip(0, 0, 1) == 0
    # a window where vhi <= vlo also falls back to the plain clip
    out2 = imaging.relative_window(img, lo=90.0, hi=10.0, clip=99.0)
    assert out2.min() >= 0.0 and out2.max() <= 1.0


# --------------------------------------------------------------------------- #
# rgb_overlay  (multi-channel)
# --------------------------------------------------------------------------- #
class _FakeDS:
    """Minimal dataset stand-in: exposes the height/width/ion_image surface that
    rgb_overlay touches. Each m/z maps to a stored (H, W) image."""

    def __init__(self, images: dict):
        self._images = images
        any_img = next(iter(images.values()))
        self.height, self.width = any_img.shape

    def ion_image(self, mz, tol_ppm=50.0, reduce="sum", norm="none"):
        # rgb_overlay calls with float(mz); keys are floats too
        return self._images[float(mz)]


def _gradient(h, w, scale):
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx + yy).astype(float) / max(1, (h + w - 2))) * scale


def test_rgb_overlay_three_channels_shape_dtype():
    h, w = 6, 8
    imgs = {
        700.0: _gradient(h, w, 5.0),
        560.0: _gradient(h, w, 2.0),
        430.0: _gradient(h, w, 1.0),
    }
    ds = _FakeDS(imgs)
    rgb = imaging.rgb_overlay(ds, 700.0, 560.0, 430.0)
    assert rgb.shape == (h, w, 3)
    assert rgb.dtype == np.uint8
    # each populated channel reaches near full scale somewhere (normalized
    # independently). _norm01 hotspot-clips at high=99 first, so the single
    # brightest pixel sits at the clip ceiling (~254), not a hard 255.
    assert rgb[..., 0].max() >= 250
    assert rgb[..., 1].max() >= 250
    assert rgb[..., 2].max() >= 250
    assert rgb[..., 0].min() == 0          # gradients start at 0


def test_rgb_overlay_none_channel_stays_zero():
    """A None channel is skipped → that plane is all zeros."""
    h, w = 5, 5
    imgs = {700.0: _gradient(h, w, 5.0), 560.0: _gradient(h, w, 3.0)}
    ds = _FakeDS(imgs)
    rgb = imaging.rgb_overlay(ds, 700.0, None, 560.0)
    assert rgb.shape == (h, w, 3)
    assert rgb[..., 1].max() == 0          # green channel left blank
    assert rgb[..., 0].max() >= 250        # red populated (near full scale)
    assert rgb[..., 2].max() >= 250        # blue populated


def test_rgb_overlay_all_none_is_black():
    h, w = 4, 4
    imgs = {700.0: _gradient(h, w, 1.0)}   # present but unused
    ds = _FakeDS(imgs)
    rgb = imaging.rgb_overlay(ds, None, None, None)
    assert rgb.shape == (h, w, 3)
    assert np.all(rgb == 0)


def test_rgb_overlay_handles_nan_channel():
    """A channel image full of NaNs normalizes to zeros (no finite values) and
    does not raise."""
    h, w = 4, 4
    nan_img = np.full((h, w), np.nan)
    imgs = {700.0: nan_img, 560.0: _gradient(h, w, 4.0)}
    ds = _FakeDS(imgs)
    rgb = imaging.rgb_overlay(ds, 700.0, 560.0, None)
    assert rgb.shape == (h, w, 3)
    assert rgb[..., 0].max() == 0          # all-NaN red → zeros
    assert np.all(np.isfinite(rgb))


# --------------------------------------------------------------------------- #
# apply_colormap  (matplotlib fallback)
# --------------------------------------------------------------------------- #
def test_apply_colormap_basic_rgba():
    img = _gradient(6, 7, 10.0)          # 0 at (0,0) rising to the max at (-1,-1)
    rgba = imaging.apply_colormap(img, cmap="viridis")
    assert rgba.shape == (6, 7, 4)
    assert rgba.dtype == np.uint8
    # viridis is actually applied (not a flat/garbage map): it runs dark blue-purple at the
    # low end -> bright yellow at the high end. A stub returning any finite uint8 array fails.
    lo, hi = rgba[0, 0], rgba[-1, -1]
    assert lo[2] > lo[0]                          # low end: more blue than red (dark purple)
    assert hi[0] > 200 and hi[1] > 200 and hi[2] < 120   # high end: yellow
    assert int(hi.sum()) > int(lo.sum())          # high end is far brighter
    assert rgba[..., 3].min() == 255              # fully opaque


def test_apply_colormap_unknown_name_falls_back_to_viridis():
    """An unknown colormap name does not crash export — it falls back to viridis
    and produces the same RGBA as viridis."""
    img = _gradient(5, 5, 3.0)
    fallback = imaging.apply_colormap(img, cmap="definitely-not-a-cmap")
    expected = imaging.apply_colormap(img, cmap="viridis")
    assert fallback.shape == (5, 5, 4)
    assert np.array_equal(fallback, expected)


def test_apply_colormap_capitalized_name_lenient_match():
    """A capitalized name is matched leniently via .lower() (Viridis → viridis)."""
    img = _gradient(5, 5, 3.0)
    cap = imaging.apply_colormap(img, cmap="Viridis")
    low = imaging.apply_colormap(img, cmap="viridis")
    assert np.array_equal(cap, low)


def test_apply_colormap_relative_clip_branch():
    """With clip set, low/high are relative-intensity %; the call still yields a
    valid RGBA array (relative_window path)."""
    img = np.ones((6, 6))
    img[0, 0] = 4.0
    rgba = imaging.apply_colormap(img, cmap="viridis", low=0.0, high=100.0, clip=99.0)
    assert rgba.shape == (6, 6, 4)
    assert rgba.dtype == np.uint8
    assert np.all(np.isfinite(rgba))


def test_apply_colormap_all_nan_input_safe():
    """An all-NaN image normalizes to zeros and still produces RGBA (no raise)."""
    img = np.full((4, 4), np.nan)
    rgba = imaging.apply_colormap(img, cmap="viridis")
    assert rgba.shape == (4, 4, 4)
    assert rgba.dtype == np.uint8
