"""Centralized colour palettes — one source of truth for every categorical palette, CVD-safe
multichannel overlay preset, sequential/diverging colormap list, and accent the app and its
exports share.

Pure and dependency-free (no Qt, no numpy at import) so the headless export engine, the GUI,
and the colour picker all read the same definitions. Research basis for the overlay presets
(below) is colour-vision-deficiency safety: avoid red/green & rainbow/jet (fail for ~4% of
viewers); cyan/magenta and green/magenta are the most robust complementary pairs and overlap
to neutral white, so colocalization reads intuitively.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Categorical palettes — cycled per category (clusters, regions, donors, …)
# --------------------------------------------------------------------------- #
# Export-figure categories (UMAP / overlay / segmentation in export.py).
CATEGORY = (
    "#5ad1c4", "#f4a259", "#8ac926", "#e15759", "#b07aa1", "#76b7b2",
    "#ff9da7", "#9c755f", "#bab0ac", "#4e79a7", "#edc948", "#59a14f",
)
# Segmentation clusters in the GUI (was gui.common.PALETTE).
SEGMENT = (
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860",
    "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD", "#E15759", "#76B7B2",
)
# User regions / cohort categories in the GUI (was gui.common.REGION_PALETTE).
REGION = (
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
    "#A65628", "#F781BF", "#1B9E77", "#D95F02", "#666666",
)
# Okabe–Ito — the canonical colour-vision-deficiency-safe qualitative set.
OKABE_ITO = (
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000",
)
# Tableau 10 — familiar, well-separated qualitative palette.
TABLEAU10 = (
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
)

# Named categorical palettes offered in the picker / palette menus.
CATEGORICAL_PALETTES = {
    "Category": CATEGORY,
    "Okabe–Ito (CVD-safe)": OKABE_ITO,
    "Tableau 10": TABLEAU10,
    "Segment": SEGMENT,
    "Region": REGION,
}

# --------------------------------------------------------------------------- #
# Multichannel overlay presets — ordered channel colours (black → colour LUTs)
# --------------------------------------------------------------------------- #
OVERLAY_PRESETS = {
    "Green / Magenta":        ("#00E676", "#FF4DFF"),             # default, dark bg, CVD ***
    "Cyan / Magenta":         ("#1DE3E3", "#FF45E0"),             # max discriminability, dark bg ***
    "Red / Cyan":             ("#FF5252", "#1DE3E3"),             # intuitive red marker **
    "Yellow / Blue":          ("#FFD400", "#4070FF"),             # high brightness sep, dark bg ***
    "Orange / Blue (Okabe–Ito)": ("#E69F00", "#0072B2"),          # light / print ***
    "Green / Magenta / Yellow": ("#00E676", "#FF4DFF", "#FFD400"),  # 3-channel **
}
# Recommended background per overlay preset ("dark" spatial / "light" print).
OVERLAY_BG = {
    "Green / Magenta": "dark", "Cyan / Magenta": "dark", "Red / Cyan": "dark",
    "Yellow / Blue": "dark", "Orange / Blue (Okabe–Ito)": "light",
    "Green / Magenta / Yellow": "dark",
}
# Accent (spectrum active-ion / marker) chosen to NOT collide with each preset's channels.
ACCENT_FOR = {
    "Green / Magenta": "#1DE3E3",
    "Cyan / Magenta":  "#FFD400",
    "Red / Cyan":      "#FFD400",
    "Yellow / Blue":   "#FF4DFF",
    "Orange / Blue (Okabe–Ito)": "#11A579",
    "default":         "#5AD1C4",
}

# --------------------------------------------------------------------------- #
# Colormaps
# --------------------------------------------------------------------------- #
# Single-ion sequential maps. cividis is the most CVD-robust; viridis/magma strong too.
SEQUENTIAL_CMAPS = ("viridis", "magma", "inferno", "plasma", "cividis", "turbo", "gray")
# Diverging maps for signed scales (SHAP direction, difference images).
DIVERGING_CMAPS = ("bwr", "coolwarm", "RdBu_r", "seismic", "PiYG")

DEFAULTS = {
    "overlay": "Green / Magenta",
    "cmap": "viridis",
    "diverging": "bwr",
    "accessibility": {"overlay": "Cyan / Magenta", "cmap": "cividis"},
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def overlay_colors(preset: str = "Green / Magenta", n: int | None = None) -> list:
    """Channel colours for an overlay ``preset``, cycled/truncated to ``n`` channels."""
    cols = list(OVERLAY_PRESETS.get(preset) or OVERLAY_PRESETS["Green / Magenta"])
    if n is None:
        return cols
    return [cols[i % len(cols)] for i in range(int(n))]


def accent_for(preset: str) -> str:
    """The non-colliding accent colour suggested for an overlay ``preset``."""
    return ACCENT_FOR.get(preset, ACCENT_FOR["default"])


def categorical(name_or_seq, n: int | None = None) -> list:
    """Resolve a categorical palette (a name in :data:`CATEGORICAL_PALETTES` or a sequence of
    hex strings) to a list, cycled to ``n`` entries when given."""
    cols = list(CATEGORICAL_PALETTES.get(name_or_seq, name_or_seq))
    if n is None:
        return cols
    return [cols[i % len(cols)] for i in range(int(n))]


def picker_swatch_groups() -> list:
    """``[(group_label, [hex, …]), …]`` for the colour picker's preset rows — every
    categorical palette plus the overlay-preset channel colours flattened into a CVD row."""
    groups = [(name, list(cols)) for name, cols in CATEGORICAL_PALETTES.items()]
    cvd = []
    for cols in OVERLAY_PRESETS.values():
        for c in cols:
            if c not in cvd:
                cvd.append(c)
    groups.append(("CVD-safe channels", cvd))
    return groups
