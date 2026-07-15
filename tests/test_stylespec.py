"""Tests for the StyleSpec design-system bundle and its integration with the export token
layer (:func:`smile_msi.export.active_style` / ``make_style`` / ``type_pt``)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from smile_msi import export, palettes, stylespec
from smile_msi.stylespec import StyleSpec


# --------------------------------------------------------------------------- #
# StyleSpec — pure round-trip + resolvers
# --------------------------------------------------------------------------- #
def test_default_spec_is_default_and_minimal():
    s = StyleSpec()
    assert s.is_default()
    # a default spec serialises to just its name (a minimal diff)
    assert s.to_dict() == {"name": "Custom"}


def test_roundtrip_preserves_fields():
    s = StyleSpec(name="X", theme="light", type_gain=1.4, font_family=["Arial", "Helvetica"],
                  type_scale={"title": 14.0}, spine_w=2.0, category_palette="Okabe–Ito (CVD-safe)")
    d = s.to_dict()
    back = StyleSpec.from_dict(d)
    assert back.theme == "light"
    assert back.type_gain == 1.4
    assert back.type_scale == {"title": 14.0}
    assert back.spine_w == 2.0
    assert back.category_palette == "Okabe–Ito (CVD-safe)"
    # JSON round-trip is lossless too
    assert StyleSpec.from_json(s.to_json()).to_dict() == d


def test_from_dict_ignores_unknown_keys():
    # a recipe written by a newer version (or with a typo) must load, not crash
    s = StyleSpec.from_dict({"name": "future", "type_gain": 1.2, "brand_new_knob": 99,
                             "quantum_flux": "on"})
    assert s.name == "future"
    assert s.type_gain == 1.2


def test_to_dict_is_a_minimal_diff():
    s = StyleSpec(name="P", type_gain=1.6, spine_w=2.0)
    d = s.to_dict()
    assert d == {"name": "P", "type_gain": 1.6, "spine_w": 2.0}  # unchanged fields dropped


def test_resolved_category_palette_by_name_and_list():
    assert StyleSpec(category_palette="Okabe–Ito (CVD-safe)").resolved_category_palette() \
        == palettes.OKABE_ITO
    assert StyleSpec(category_palette=["#111111", "#222222"]).resolved_category_palette() \
        == ("#111111", "#222222")
    assert StyleSpec().resolved_category_palette() is None
    # tolerant name match (dashes/case/spaces)
    assert StyleSpec(category_palette="okabeito(cvd-safe)").resolved_category_palette() \
        == palettes.OKABE_ITO


def test_resolved_gain_is_clamped():
    assert StyleSpec(type_gain=99).resolved_gain() == stylespec.GAIN_MAX
    assert StyleSpec(type_gain=0.0).resolved_gain() == stylespec.GAIN_MIN or 1.0


def test_resolved_type_scale_merges_and_keeps_weight():
    base = {"title": (9.0, "bold"), "axis": (7.0, "normal")}
    out = StyleSpec(type_scale={"title": 20.0}).resolved_type_scale(base)
    assert out["title"] == (20.0, "bold")     # weight preserved
    assert out["axis"] == (7.0, "normal")     # untouched


def test_coerce_accepts_every_form():
    assert stylespec.coerce(None) is None
    assert isinstance(stylespec.coerce("Poster (large type)"), StyleSpec)
    assert stylespec.coerce("Poster (large type)").name == "Poster (large type)"
    assert isinstance(stylespec.coerce({"name": "d", "type_gain": 1.1}), StyleSpec)
    assert isinstance(stylespec.coerce(json.dumps({"name": "j", "spine_w": 3})), StyleSpec)
    assert isinstance(stylespec.coerce(StyleSpec(name="obj")), StyleSpec)
    assert stylespec.coerce("not a preset name") is None


def test_builtin_presets_exist_and_are_distinct():
    names = stylespec.preset_names()
    assert "App Default" in names
    assert "Poster (large type)" in names
    # every preset round-trips
    for name, spec in stylespec.PRESETS.items():
        assert StyleSpec.from_dict(spec.to_dict()).name == name
    # App Default really is a no-op look
    assert stylespec.PRESETS["App Default"].is_default()


# --------------------------------------------------------------------------- #
# Integration — active_style threads through the token layer (zero render churn)
# --------------------------------------------------------------------------- #
def test_active_style_scales_type_and_restores():
    base = export.type_pt("title", 3.4)
    with export.active_style("Poster (large type)"):     # type_gain 1.6
        big = export.type_pt("title", 3.4)
        assert big == pytest.approx(base * 1.6, rel=0.02)
    assert export.type_pt("title", 3.4) == base          # restored on exit
    assert export.current_style_spec() is None


def test_active_style_per_role_override():
    with export.active_style(StyleSpec(name="t", type_scale={"title": 30.0})):
        # at the reference width the width-factor is 1.0, so the size is the override
        assert export.type_pt("title", 3.4) == pytest.approx(30.0, abs=0.1)
        assert export.type_weight("title") == "bold"     # weight preserved


def test_active_style_forces_theme_and_palette_and_weights():
    spec = StyleSpec(name="t", theme="light", spine_w=2.5, hairline_w=1.7, outline_w=3.0,
                     category_palette="Okabe–Ito (CVD-safe)", accent="#ff0000")
    with export.active_style(spec):
        s = export.make_style("dark")                    # preset forces light despite "dark"
        assert s.name == "light"
        assert s.spine_w == 2.5
        assert s.hairline_w == 1.7
        assert s.outline_w == 3.0
        assert s.category_palette == palettes.OKABE_ITO
        assert s.accent == "#ff0000"


def test_explicit_args_win_over_preset():
    spec = StyleSpec(name="t", image_cmap="gray", accent="#ff0000")
    with export.active_style(spec):
        s = export.make_style("dark", "#00ff00", cmap="magma")
        assert s.cmap == "magma"        # explicit cmap beats preset image_cmap
        assert s.accent == "#00ff00"    # explicit accent beats preset accent


def test_active_style_none_is_noop():
    base = export.type_pt("title", 3.4)
    with export.active_style(None):
        assert export.type_pt("title", 3.4) == base
        assert export.current_style_spec() is None


def test_active_style_nests_and_restores():
    with export.active_style("Poster (large type)"):
        outer = export.type_pt("title", 3.4)
        with export.active_style(StyleSpec(name="inner", type_gain=0.5)):
            assert export.type_pt("title", 3.4) < outer
        assert export.type_pt("title", 3.4) == outer     # inner restored


def test_weight_tokens_clamped():
    with export.active_style(StyleSpec(name="t", spine_w=999)):
        s = export.make_style("light")
        assert s.spine_w == stylespec.WEIGHT_MAX


# --------------------------------------------------------------------------- #
# End-to-end — a preset really changes a rendered figure
# --------------------------------------------------------------------------- #
def _title_size(fig):
    # titles may be set at loc center/left/right (render_spectrum_figure uses loc="left")
    for ax in fig.axes:
        for artist in (getattr(ax, "title", None), getattr(ax, "_left_title", None),
                       getattr(ax, "_right_title", None)):
            if artist is not None and artist.get_text():
                return artist.get_fontsize()
    return None


def test_render_spectrum_title_grows_under_poster():
    mz = np.linspace(700, 900, 200)
    inten = np.exp(-((mz - 800) ** 2) / 50.0)
    spectra = [("mean", mz, inten)]
    base_fig = export.render_spectrum_figure(spectra, title="Spectrum", theme="light")
    base_pt = _title_size(base_fig)
    with export.active_style("Poster (large type)"):
        big_fig = export.render_spectrum_figure(spectra, title="Spectrum", theme="light")
        big_pt = _title_size(big_fig)
    assert base_pt is not None and big_pt is not None
    assert big_pt > base_pt * 1.2          # gain 1.6 clearly larger


def test_render_ion_panel_outline_weight_from_preset():
    img = np.random.default_rng(0).random((20, 20))
    mask = np.zeros((20, 20), bool)
    mask[5:15, 5:15] = True
    with export.active_style("Poster (large type)"):     # outline_w 2.2
        fig = export.render_ion_panel(image=img, mz=744.5, outline_mask=mask,
                                      show_scalebar=False, theme="dark")
    # the contour collection carries the preset's outline weight
    widths = []
    for ax in fig.axes:
        for coll in ax.collections:
            lw = coll.get_linewidths()
            if lw is not None and len(lw):
                widths.append(float(np.ravel(lw)[0]))
    assert any(abs(w - 2.2) < 0.3 for w in widths), widths


def test_render_under_preset_does_not_error_across_figures():
    # smoke: every simple figure renders under a non-default preset without raising
    rng = np.random.default_rng(1)
    img = rng.random((16, 16))
    with export.active_style("Slide (dark)"):
        export.render_ion_panel(image=img, mz=700.0, show_scalebar=False)
        export.render_spectrum_figure([("s", np.linspace(700, 900, 50), rng.random(50))])


def test_render_cell_applies_and_restores_style(tmp_path):
    # the Export Studio worker path: a StyleSpec dict rides in the picklable cell spec and is
    # applied per-render, then cleared (so a worker rendering many cells doesn't leak a look)
    img = np.random.default_rng(0).random((16, 16))
    path = str(tmp_path / "cell.png")
    spec = {"kw": {"mz": 744.5, "show_scalebar": False, "theme": "dark"},
            "dpi": 100, "fmt": "png",
            "style": {"name": "Poster", "type_gain": 1.6, "spine_w": 2.0}}
    out = export.render_cell(img, spec, path)
    assert out == path
    assert (tmp_path / "cell.png").stat().st_size > 0
    assert export.current_style_spec() is None      # no leak after the worker render


def test_render_cell_without_style_is_unchanged(tmp_path):
    img = np.random.default_rng(0).random((12, 12))
    path = str(tmp_path / "plain.png")
    export.render_cell(img, {"kw": {"mz": 700.0, "show_scalebar": False}, "dpi": 90, "fmt": "png"}, path)
    assert (tmp_path / "plain.png").stat().st_size > 0
