"""StyleSpec — a serialisable **design-system bundle** shared by every exported figure.

The export engine (:mod:`smile_msi.export`) already centralises *colour* into a
:class:`~smile_msi.export.Style` and *type* into a named type scale, but those tokens were
fixed module constants: a user could pick a colormap per figure, yet could not change the
font, the type sizes, the line weights, the margins, or save a look and reuse it across
figures. That is the gap this module closes.

A :class:`StyleSpec` captures the *reusable look* of a figure — typography, theme colours,
category palette, and line weights — **independently of the data-specific choices** (which
ion, which contrast window, which title) that stay per-figure parameters. One StyleSpec
therefore restyles *every* figure the app renders, consistently, because
:func:`smile_msi.export.active_style` installs it as the active token bundle that
``make_style`` / ``type_pt`` / ``type_weight`` all consult — with **zero change to any
``render_*`` body**.

Because it round-trips losslessly to a plain dict / JSON (``to_dict`` / ``from_dict``, robust
to unknown future keys — the same contract as ``UMAPStudioSpec``), a StyleSpec *is* a figure
"recipe" you can:

* save to a named **preset** library and re-apply,
* export as JSON, hand to an assistant ("make this read like a Nature figure — bigger title,
  Arial, thinner rules — return the JSON"), and re-import so every figure updates, and
* embed in a session for exact, provenance-friendly reproduction.

Pure and dependency-free at import (no Qt, no numpy, no matplotlib) so the headless export
engine, the GUI, and tests all read the same definitions. It references
:mod:`smile_msi.palettes` by name rather than duplicating any colour list.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields

from . import palettes

# Type-scale roles a StyleSpec may override (must match export.TYPE_SCALE keys). Kept here
# so a preset editor can enumerate them without importing the (matplotlib-touching) engine.
TYPE_ROLES = ("title", "subtitle", "axis", "tick", "annotation", "legend")

# Sensible clamps so a hand-edited / assistant-authored recipe can't produce an unusable
# figure (billboard text, invisible hairlines). Applied when the spec is resolved.
GAIN_MIN, GAIN_MAX = 0.5, 3.0
WEIGHT_MIN, WEIGHT_MAX = 0.1, 6.0

# Named font stacks offered in the preset editor. The first that resolves on the machine
# wins; DejaVu Sans is always appended by the engine as a glyph-coverage fallback.
FONT_STACKS = {
    "Helvetica (default)": None,        # None ⇒ the engine's built-in Helvetica stack
    "Arial": ("Arial", "Helvetica", "Liberation Sans"),
    "Calibri": ("Calibri", "Carlito", "Helvetica"),
    "Times / serif": ("Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"),
    "Georgia / serif": ("Georgia", "DejaVu Serif"),
    "Courier / mono": ("Courier New", "Courier", "DejaVu Sans Mono"),
}

# Rough margin-density presets (a hint some renderers can honour); kept small + explicit.
MARGIN_DENSITY = ("tight", "normal", "roomy")


@dataclass
class StyleSpec:
    """A complete, serialisable description of a figure's *look* (not its data).

    Every field is optional (``None``/default = "keep the engine's built-in value"), so a
    minimal spec that sets only ``type_gain`` leaves everything else at the app default. This
    is what lets a preset be a small, legible diff rather than a full copy of every token.
    """
    name: str = "Custom"
    description: str = ""

    # ---- theme / colour (colour is already a parameter in the engine; here we make the
    # whole theme reusable, and optionally fully custom) --------------------------------- #
    theme: str | None = None            # None ⇒ keep each figure's own default; "dark"/"light" ⇒ force
    bg: str | None = None               # explicit theme-colour overrides (None ⇒ theme default)
    fg: str | None = None
    muted: str | None = None
    accent: str | None = None
    hairline: object | None = None      # RGBA list/tuple or hex; faint grid/guide colour
    grid: str | None = None             # solid spine colour
    image_cmap: str | None = None       # default ion-image colormap (a per-figure cmap still wins)
    diverging_cmap: str | None = None   # SHAP / direction colormap
    category_palette: object | None = None  # name in palettes.CATEGORICAL_PALETTES, or explicit list of hex

    # ---- typography (the biggest gap: previously fixed module constants) --------------- #
    font_family: object | None = None   # font stack (list of family names); None ⇒ default Helvetica stack
    type_gain: float = 1.0              # single global multiplier over the whole type scale
    type_scale: dict = field(default_factory=dict)   # per-role absolute pt overrides {role: pt}
    type_ref_in: float | None = None    # reference width the scale is calibrated to (default 3.4in)
    type_max_scale: float | None = None  # cap on the width-driven scale-up (default 1.45)

    # ---- line weight / spacing --------------------------------------------------------- #
    spine_w: float | None = None        # xy-plot spine weight (engine default 1.2)
    hairline_w: float | None = None     # grid / faint-guide weight (engine default 1.0)
    outline_w: float | None = None      # default ROI-outline weight (engine default 1.4)
    margin: str | None = None           # "tight" | "normal" | "roomy" density hint

    # ------------------------------------------------------------------ round-trip ---- #
    def to_dict(self) -> dict:
        """Plain-dict form for JSON / session persistence — a **minimal diff**: only fields that
        differ from the default are emitted (``name`` always), so a recipe reads as a short list
        of just what the look changes. Sequences normalise to lists for byte-stable JSON."""
        base = StyleSpec()
        out: dict = {"name": self.name}
        for f in fields(self):
            if f.name == "name":
                continue
            v = getattr(self, f.name)
            if v == getattr(base, f.name):
                continue
            if isinstance(v, tuple):
                v = list(v)
            out[f.name] = v
        return out

    @classmethod
    def from_dict(cls, d: dict | None) -> "StyleSpec":
        """Build from a dict, **ignoring unknown keys** so a recipe written by a newer version
        (or hand-edited with a typo) loads instead of crashing — same tolerance as
        ``UMAPStudioSpec.from_dict``."""
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in d.items() if k in known}
        if "type_scale" in clean and not isinstance(clean["type_scale"], dict):
            clean.pop("type_scale")
        return cls(**clean)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "StyleSpec":
        return cls.from_dict(json.loads(text))

    def copy(self, **changes) -> "StyleSpec":
        d = self.to_dict()
        d.update(changes)
        return StyleSpec.from_dict(d)

    # ------------------------------------------------------------------ resolvers ----- #
    # These turn the *declarative* spec into concrete values the engine applies. Kept here
    # (not in export) so they're testable without matplotlib and reusable by a preset editor.
    def is_default(self) -> bool:
        """True when the spec would not change any engine token (only name/description set)."""
        base = StyleSpec()
        for f in fields(self):
            if f.name in ("name", "description"):
                continue
            if getattr(self, f.name) != getattr(base, f.name):
                return False
        return True

    def resolved_gain(self) -> float:
        return _clamp(float(self.type_gain or 1.0), GAIN_MIN, GAIN_MAX)

    def resolved_category_palette(self) -> tuple | None:
        """Resolve ``category_palette`` (a palettes name *or* an explicit list) to a tuple of
        hex strings, or ``None`` to keep the engine default."""
        cp = self.category_palette
        if cp is None:
            return None
        if isinstance(cp, str):
            pal = palettes.CATEGORICAL_PALETTES.get(cp)
            if pal is None:                       # tolerant name match (ignore case/punctuation)
                key = _alnum(cp)
                for k, v in palettes.CATEGORICAL_PALETTES.items():
                    if _alnum(k) == key:
                        pal = v
                        break
            return tuple(pal) if pal else None
        seq = [str(c) for c in cp if c]
        return tuple(seq) or None

    def resolved_font_family(self) -> tuple | None:
        ff = self.font_family
        if ff is None:
            return None
        if isinstance(ff, str):                   # a FONT_STACKS label, or a single family name
            if ff in FONT_STACKS:
                stack = FONT_STACKS[ff]
                return tuple(stack) if stack else None
            return (ff,)
        return tuple(str(f) for f in ff) or None

    def resolved_type_scale(self, base: dict) -> dict:
        """Merge per-role pt overrides over ``base`` (``{role: (pt, weight)}``), keeping each
        role's weight. Roles the spec doesn't mention are untouched."""
        if not self.type_scale:
            return dict(base)
        out = dict(base)
        for role, pt in self.type_scale.items():
            if role in out and pt:
                try:
                    out[role] = (float(pt), out[role][1])
                except (TypeError, ValueError):
                    continue
        return out


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _alnum(s: str) -> str:
    """Lowercased alphanumerics only — for tolerant palette-name matching across
    dashes/en-dashes/spaces/case (e.g. 'Okabe–Ito (CVD-safe)' ≈ 'okabeito cvdsafe')."""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def coerce(obj) -> "StyleSpec | None":
    """Accept a StyleSpec, a dict, a JSON string, a preset name, or ``None`` → StyleSpec|None.
    The one entry point the GUI/engine use so any of those forms 'just works'."""
    if obj is None:
        return None
    if isinstance(obj, StyleSpec):
        return obj
    if isinstance(obj, dict):
        return StyleSpec.from_dict(obj)
    if isinstance(obj, str):
        if obj in PRESETS:
            return PRESETS[obj]
        s = obj.strip()
        if s.startswith("{"):
            try:
                return StyleSpec.from_json(s)
            except (ValueError, json.JSONDecodeError):
                return None
        return None
    return None


# --------------------------------------------------------------------------- #
# Built-in presets — a small, opinionated library of distinct, ready looks.
# Each is a legible diff from the app default; users can save their own alongside.
# --------------------------------------------------------------------------- #
def _preset(name, description, **kw) -> StyleSpec:
    return StyleSpec(name=name, description=description, **kw)


_PRESET_LIST = [
    _preset(
        "App Default",
        "The app's built-in look — Helvetica, compact type, dark spatial / light analytics.",
    ),
    _preset(
        "Journal (print)",
        "Light, compact, CVD-safe — tuned for a single-column print figure.",
        theme="light", font_family="Arial", type_gain=0.95,
        category_palette="Okabe–Ito (CVD-safe)", spine_w=1.0, hairline_w=0.8,
        margin="tight",
    ),
    _preset(
        "Poster (large type)",
        "Everything scaled up with heavier rules so text reads from across a room.",
        type_gain=1.6, spine_w=2.0, hairline_w=1.4, outline_w=2.2, margin="roomy",
    ),
    _preset(
        "Slide (dark)",
        "Dark background, bright type and rules — for projected presentation slides.",
        theme="dark", type_gain=1.3, accent="#5ad1c4", spine_w=1.6, hairline_w=1.2,
        outline_w=2.0,
    ),
    _preset(
        "Grayscale ion (print-safe)",
        "Grayscale ion images for a black-and-white print pipeline; light theme.",
        theme="light", image_cmap="gray", font_family="Arial", type_gain=0.95,
        spine_w=1.0,
    ),
    _preset(
        "High-contrast (accessible)",
        "Larger type, heavier rules, CVD-safe categories, high-contrast ink.",
        type_gain=1.25, category_palette="Okabe–Ito (CVD-safe)", spine_w=1.8,
        hairline_w=1.3, outline_w=2.0, fg="#000000", muted="#333333",
    ),
]

PRESETS: dict = {p.name: p for p in _PRESET_LIST}


def preset_names() -> list:
    return list(PRESETS.keys())
