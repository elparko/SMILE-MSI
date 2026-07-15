"""Tests for the centralized colour-palette module (engine-side, Qt-free)."""
import re

from smile_msi import palettes


_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def _all_hex(seq):
    return all(_HEX.match(c) for c in seq)


def test_categorical_palettes_are_valid_hex():
    for name, cols in palettes.CATEGORICAL_PALETTES.items():
        assert cols, name
        assert _all_hex(cols), name


def test_overlay_presets_and_accents():
    for name, cols in palettes.OVERLAY_PRESETS.items():
        assert 2 <= len(cols) <= 3
        assert _all_hex(cols)
        assert name in palettes.OVERLAY_BG
        assert _HEX.match(palettes.accent_for(name))
    # an unknown preset still yields the safe default accent
    assert palettes.accent_for("nope") == palettes.ACCENT_FOR["default"]


def test_overlay_colors_cycles_to_n():
    cols = palettes.overlay_colors("Green / Magenta", 5)
    assert len(cols) == 5
    assert cols[0] == cols[2] == cols[4]    # 2-colour preset cycles
    assert palettes.overlay_colors("Green / Magenta") == ["#00E676", "#FF4DFF"]


def test_categorical_resolves_name_or_sequence():
    by_name = palettes.categorical("Tableau 10", 3)
    by_seq = palettes.categorical(palettes.TABLEAU10, 3)
    assert by_name == by_seq == list(palettes.TABLEAU10[:3])


def test_categorical_cycles_when_n_exceeds_palette():
    """Requesting more entries than the palette holds wraps around (the ``i % len`` branch),
    rather than truncating or raising -- so a 12-cluster map still gets 12 colours."""
    base = list(palettes.TABLEAU10)
    out = palettes.categorical("Tableau 10", len(base) + 2)
    assert len(out) == len(base) + 2
    assert out[: len(base)] == base
    assert out[len(base)] == base[0] and out[len(base) + 1] == base[1]


def test_picker_groups_nonempty_and_valid():
    groups = palettes.picker_swatch_groups()
    assert groups
    for name, cols in groups:
        assert cols and _all_hex(cols), name


def test_cmaps_present():
    assert "cividis" in palettes.SEQUENTIAL_CMAPS      # the most CVD-robust sequential map
    assert "bwr" in palettes.DIVERGING_CMAPS


def test_export_palette_is_centralized_here():
    from smile_msi import export
    assert tuple(export.DEFAULT_CATEGORY_PALETTE) == palettes.CATEGORY
