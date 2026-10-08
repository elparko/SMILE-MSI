"""Atlas-style visuals (:mod:`smile_msi.atlasviz`) — dot mosaics and splitter movies."""
import numpy as np
import pytest
from matplotlib.figure import Figure

from smile_msi import atlasviz, demo, spatial

LEVELS = [2, 3, 5]
STEPS = 3
HOLD = atlasviz.DEFAULT_HOLD


@pytest.fixture(scope="module")
def tree():
    ds = demo.make_synthetic(width=24, height=20)
    mzs = [p[0] for p in demo._demo_peaks()]
    return ds, spatial.hierarchy(ds, mzs, n_micro=20)


def _hex_rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0


def _level_rgb(ds, hier, k):
    lab = spatial.cut(hier, k)
    pal = np.array([_hex_rgb(c) for c in spatial.segment_colors(hier, k)])
    return np.stack([ds.to_image(pal[lab][:, c], fill=0.0) for c in range(3)], axis=-1)


# ---------------------------------------------------------------- dot mosaic
def test_dot_mosaic_float_image_writes_png_and_skips_nan(tmp_path):
    img = np.arange(30, dtype=float).reshape(5, 6)
    img[0, :2] = np.nan
    img[3, 4] = np.nan
    out = tmp_path / "ion.png"
    fig = atlasviz.dot_mosaic(img, out, dpi=60, pixel_size_um=50, title="ion",
                              inset_outline=True)
    assert isinstance(fig, Figure)
    assert out.exists() and out.stat().st_size > 0
    coll = fig.axes[0].collections[0]
    assert len(coll.get_offsets()) == int(np.isfinite(img).sum())
    assert fig.get_facecolor()[:3] == (0.0, 0.0, 0.0)


def test_dot_mosaic_label_image_uses_colours_and_skips_negatives(tmp_path):
    lab = np.array([[0, 1, 2], [2, -1, 0], [1, 1, 2]], dtype=float)
    lab[0, 0] = np.nan
    cols = ["#ff0000", "#00ff00", "#0000ff"]
    out = tmp_path / "seg.png"
    fig = atlasviz.dot_mosaic(lab, out, colors=cols, dpi=60)
    coll = fig.axes[0].collections[0]
    assert len(coll.get_offsets()) == 7           # 9 pixels - 1 NaN - 1 negative
    face = np.asarray(coll.get_facecolor())[:, :3]
    assert {tuple(np.round(c)) for c in face} == {(1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                                   (0.0, 0.0, 1.0)}
    assert out.exists()


def test_dot_mosaic_dot_diameter_scales_with_pitch():
    fig = atlasviz.dot_mosaic(np.ones((4, 4)), dot=0.5, dpi=50)
    coll = fig.axes[0].collections[0]
    assert np.allclose(coll.get_widths(), 0.5) and np.allclose(coll.get_heights(), 0.5)


def test_dot_mosaic_vector_formats(tmp_path):
    img = np.random.default_rng(0).random((6, 6))
    for ext in ("pdf", "svg"):
        p = tmp_path / f"m.{ext}"
        atlasviz.dot_mosaic(img, p, dpi=50)
        assert p.stat().st_size > 0


def test_dot_mosaic_rejects_bad_input():
    with pytest.raises(ValueError):
        atlasviz.dot_mosaic(np.zeros((2, 2, 3)))
    with pytest.raises(ValueError):
        atlasviz.dot_mosaic(np.array([[0, 5]]), colors=["#ffffff"])


# ---------------------------------------------------------- splitter frames
def test_splitter_frames_count_levels_and_range(tree):
    ds, hier = tree
    frames = atlasviz.splitter_frames(hier, ds, LEVELS, steps_per_level=STEPS)
    assert len(frames) == len(LEVELS) * HOLD + (len(LEVELS) - 1) * STEPS
    for f in frames:
        assert f.shape == (ds.height, ds.width, 3)
        assert np.isfinite(f).all() and f.min() >= 0.0 and f.max() <= 1.0
    for i, k in enumerate(LEVELS):
        expect = _level_rgb(ds, hier, k)
        start = i * (HOLD + STEPS)
        for f in frames[start:start + HOLD]:
            assert np.allclose(f, expect, atol=1e-9)
    # a blend frame is neither endpoint
    mid = frames[HOLD]
    assert not np.allclose(mid, frames[0]) and not np.allclose(mid, frames[-1])


def test_splitter_frames_first_frame_is_level0_segment_colors(tree):
    ds, hier = tree
    frames = atlasviz.splitter_frames(hier, ds, LEVELS, steps_per_level=STEPS)
    cols = spatial.segment_colors(hier, LEVELS[0])
    lab = ds.to_image(spatial.cut(hier, LEVELS[0]).astype(float), fill=-1)
    for j, c in enumerate(cols):
        sel = lab == j
        if sel.any():
            assert np.allclose(frames[0][sel], _hex_rgb(c))


def test_splitter_frames_region_scoped_is_black_outside(tree):
    ds, _ = tree
    mzs = [p[0] for p in demo._demo_peaks()]
    mask = np.zeros(ds.n_pixels, dtype=bool)
    mask[: ds.n_pixels // 2] = True
    hier = spatial.hierarchy(ds, mzs, n_micro=10, pixel_mask=mask)
    frames = atlasviz.splitter_frames(hier, ds, [2, 3], steps_per_level=2)
    outside = ds.to_image(mask.astype(float), fill=0.0) == 0.0
    assert outside.any()
    for f in frames:
        assert np.isfinite(f).all()
        assert (f[outside] == 0.0).all()


def test_oklab_roundtrip():
    rng = np.random.default_rng(1)
    rgb = rng.random((50, 3))
    assert np.allclose(atlasviz._oklab_to_rgb(atlasviz._rgb_to_oklab(rgb)), rgb, atol=1e-6)


# ----------------------------------------------------------- splitter movie
def test_splitter_movie_gif_frame_count(tree, tmp_path):
    from PIL import Image

    ds, hier = tree
    out = tmp_path / "split.gif"
    assert atlasviz.splitter_movie(hier, ds, out, LEVELS, fps=10, steps_per_level=STEPS) == str(out)
    with Image.open(out) as im:
        # identical hold frames are stored once (longer duration): L + (L-1)*steps frames
        assert im.n_frames == len(LEVELS) + (len(LEVELS) - 1) * STEPS
        assert im.size == (ds.width * atlasviz.PIXEL_SCALE, ds.height * atlasviz.PIXEL_SCALE)
        total_ms = 0
        for i in range(im.n_frames):
            im.seek(i)
            total_ms += im.info.get("duration", 0)
    n_frames = len(LEVELS) * HOLD + (len(LEVELS) - 1) * STEPS
    assert abs(total_ms - n_frames * 100) <= n_frames


def test_splitter_movie_dot_mode(tree, tmp_path):
    from PIL import Image

    ds, hier = tree
    out = tmp_path / "dots.gif"
    atlasviz.splitter_movie(hier, ds, out, [2, 3], steps_per_level=1, dot=0.85)
    with Image.open(out) as im:
        assert im.n_frames == 3
        assert im.size == (ds.width * atlasviz.DOT_PX, ds.height * atlasviz.DOT_PX)


def test_splitter_movie_default_levels(tree):
    ds, hier = tree
    ks = atlasviz.default_k_levels(hier)
    assert ks[0] == 2 and ks[-1] == min(hier.max_clusters, 24)
    assert ks == sorted(set(ks))


def test_splitter_movie_format_errors(tree, tmp_path):
    ds, hier = tree
    with pytest.raises(ValueError):
        atlasviz.splitter_movie(hier, ds, tmp_path / "x.webm", [2, 3])
    try:
        import imageio_ffmpeg  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="imageio-ffmpeg"):
            atlasviz.splitter_movie(hier, ds, tmp_path / "x.mp4", [2, 3])
