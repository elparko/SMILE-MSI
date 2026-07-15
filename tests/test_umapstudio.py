"""Pure-engine tests for the UMAP Studio renderer (no Qt, no GUI).

Covers the density rasterizer (binning, eq_hist alpha, count-weighted colour blend,
background compositing), the palette / colour-key helpers, the serialisable spec round-trip,
and figure assembly (panel count, channel routing) — the parts that must stay correct
independently of the dialog."""
import numpy as np
import pytest

from smile_msi import umapstudio as us


# --------------------------------------------------------------------------- #
# Rasterizer
# --------------------------------------------------------------------------- #
def test_shade_categorical_shape_and_opaque_white_bg():
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(2000, 2))
    cats = np.array(["a", "b"] * 1000, dtype=object)
    key = us.build_color_key(cats, "tab10")
    rgba = us.shade_categorical(coords, cats, key, width=64, height=48, background="white")
    assert rgba.shape == (48, 64, 4)
    assert rgba.dtype == np.uint8
    # white background → fully opaque image
    assert (rgba[..., 3] == 255).all()
    # empty pixels stay white where no points landed
    assert rgba[..., :3].max() == 255


def test_shade_categorical_blends_overlapping_categories():
    """Two categories landing in the SAME output pixel blend to the count-weighted mean
    colour (the 'painterly' compositing) — red + blue → purple."""
    coords = np.array([[0.0, 0.0], [0.0, 0.0]])
    cats = np.array(["r", "b"], dtype=object)
    key = {"r": "#ff0000", "b": "#0000ff"}
    rgba = us.shade_categorical(coords, cats, key, width=1, height=1,
                                x_range=(-1, 1), y_range=(-1, 1), how=us.HOW_LINEAR,
                                min_alpha=1.0, background="white")
    r, g, b, a = rgba[0, 0]
    assert a == 255
    assert 100 <= r <= 160 and g < 30 and 100 <= b <= 160      # ~ (128, 0, 128)


def test_eq_hist_alpha_is_monotonic_in_count():
    total = np.array([0.0, 1.0, 5.0, 50.0])
    a = us._alpha_from_counts(total, us.HOW_EQ_HIST, min_alpha=0.2)
    assert a[0] == 0.0                                          # empty pixel stays transparent
    assert a[1] < a[2] < a[3] <= 1.0
    assert a[1] >= 0.2                                          # floored at min_alpha


def test_transparent_background_keeps_per_pixel_alpha():
    coords = np.array([[0.0, 0.0]])
    cats = np.array(["a"], dtype=object)
    key = {"a": "#00aa00"}
    rgba = us.shade_categorical(coords, cats, key, width=4, height=4,
                                x_range=(-1, 1), y_range=(-1, 1), background="transparent")
    assert rgba[..., 3].min() == 0                             # most pixels empty/transparent
    assert rgba[..., 3].max() > 0                              # the occupied pixel shows


def test_shade_continuous_returns_value_window():
    rng = np.random.default_rng(1)
    coords = rng.normal(size=(500, 2))
    vals = rng.normal(10, 2, size=500)
    rgba, (lo, hi) = us.shade_continuous(coords, vals, cmap="viridis", width=40, height=40)
    assert rgba.shape == (40, 40, 4)
    assert lo < hi                                             # a real percentile window


# --------------------------------------------------------------------------- #
# Palettes / colour keys
# --------------------------------------------------------------------------- #
def test_glasbey_palette_is_distinct_for_many_categories():
    cols = us.categorical_palette(28, "glasbey")
    assert len(cols) == 28
    assert len(set(cols)) == 28                                # all distinct (no recycling)


def test_auto_palette_switches_at_ten():
    assert us.categorical_palette(8, "auto") == us.TAB10[:8]   # few → tab10
    assert len(us.categorical_palette(20, "auto")) == 20       # many → glasbey-like


def test_build_color_key_first_seen_and_stable():
    cats = ["B", "A", "B", "C", "A"]
    key = us.build_color_key(cats, "tab10")
    assert list(key.keys()) == ["B", "A", "C"]                 # first-seen order
    assert us.build_color_key(cats, "tab10") == key            # deterministic


# --------------------------------------------------------------------------- #
# Spec + EmbeddingData
# --------------------------------------------------------------------------- #
def test_spec_round_trips_through_dict():
    spec = us.UMAPStudioSpec(
        panels=[us.PanelSpec(color_by="Sample", panel_letter="A"),
                us.PanelSpec(color_by="744.5 m/z", cmap="magma", panel_letter="B")],
        layout="two", palette="glasbey", how=us.HOW_LINEAR, background="black")
    spec.annotations.append(us.Annotation(text="cluster", x=1.0, y=2.0, panel=1, bold=True))
    back = us.UMAPStudioSpec.from_dict(spec.to_dict())
    assert back.layout == "two" and back.palette == "glasbey" and back.background == "black"
    assert [p.color_by for p in back.panels] == ["Sample", "744.5 m/z"]
    assert back.panels[1].cmap == "magma"
    assert back.annotations[0].text == "cluster" and back.annotations[0].panel == 1


def test_from_dict_ignores_unknown_keys():
    back = us.UMAPStudioSpec.from_dict({"layout": "single", "future_knob": 7})
    assert back.layout == "single"                             # forward-compatible


def test_embedding_data_set_feature_channel():
    rng = np.random.default_rng(2)
    data = us.EmbeddingData(coords=rng.normal(size=(50, 2)),
                            features=rng.normal(size=(50, 3)),
                            feature_mz=np.array([744.5, 810.5, 885.5]))
    name = data.set_feature_channel(1)
    assert "810.5" in name and "m/z" in name
    assert name in data.continuous and data.continuous[name].shape == (50,)
    assert not data.is_categorical(name)


# --------------------------------------------------------------------------- #
# Figure assembly
# --------------------------------------------------------------------------- #
@pytest.fixture
def data():
    rng = np.random.default_rng(3)
    n = 800
    coords = np.vstack([rng.normal([0, 0], 0.5, (n, 2)), rng.normal([3, 2], 0.5, (n, 2))])
    return us.EmbeddingData(
        coords=coords, method="UMAP",
        categorical={"Sample": np.array(["s1"] * n + ["s2"] * n, dtype=object),
                     "Group": np.array(["A"] * n + ["B"] * n, dtype=object)},
        continuous={"744.5 m/z": np.r_[rng.normal(5, 1, n), rng.normal(9, 1, n)]})


def test_render_single_panel(data):
    spec = us.UMAPStudioSpec(panels=[us.PanelSpec(color_by="Sample")], layout="single")
    fig = us.render_figure(data, spec)
    assert len(fig._umap_panel_axes) == 1
    assert "Sample" in fig._umap_color_keys


def test_render_four_panels_mixed_channels(data):
    spec = us.UMAPStudioSpec(
        panels=[us.PanelSpec(color_by="Sample", panel_letter="A"),
                us.PanelSpec(color_by="Group", panel_letter="B"),
                us.PanelSpec(color_by="744.5 m/z", cmap="magma", panel_letter="C"),
                us.PanelSpec(color_by="Group", panel_letter="D")],
        layout="four")
    spec.annotations.append(us.Annotation(text="PT", x=3.0, y=2.0, panel=1))
    fig = us.render_figure(data, spec)
    assert len(fig._umap_panel_axes) == 4
    # categorical channels get colour keys; the continuous one does not
    assert set(fig._umap_color_keys) == {"Sample", "Group"}
    assert len(fig._umap_annotation_artists) == 1


def test_render_locks_shared_range_across_panels(data):
    spec = us.UMAPStudioSpec(
        panels=[us.PanelSpec(color_by="Sample"), us.PanelSpec(color_by="Group")], layout="two")
    fig = us.render_figure(data, spec)
    xa, ya = fig._umap_panel_axes[0].get_xlim(), fig._umap_panel_axes[0].get_ylim()
    xb, yb = fig._umap_panel_axes[1].get_xlim(), fig._umap_panel_axes[1].get_ylim()
    assert xa == pytest.approx(xb) and ya == pytest.approx(yb)


def test_render_into_existing_figure_reuses_it(data):
    from matplotlib.figure import Figure
    fig = Figure()
    spec = us.UMAPStudioSpec(panels=[us.PanelSpec(color_by="Group")], layout="single")
    out = us.render_figure(data, spec, fig=fig)
    assert out is fig                                          # drew into the supplied figure
    assert len(out._umap_panel_axes) == 1
