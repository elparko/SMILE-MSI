"""Named figure-style **preset library** — the built-in looks plus any the user saves.

:mod:`smile_msi.stylespec` defines the :class:`StyleSpec` data model and a handful of
built-in :data:`~smile_msi.stylespec.PRESETS`. This module adds the *library* on top: a
user can save the current look under a name, and it sits alongside the built-ins in every
preset picker, survives restarts, and can be shared as JSON.

User presets live in one prefs slot (``figure_style_presets`` → ``{name: spec_dict}``) —
the same lightweight persistence the Export dialog and UMAP studio already use — rather than
a new on-disk format, so there is nothing extra to back up or migrate. Built-ins are always
present and cannot be deleted; a user preset that reuses a built-in's name shadows it.

Qt-free and pure, so the GUI, the scripting console, and tests all share one source of truth.
"""
from __future__ import annotations

from . import prefs, stylespec
from .stylespec import StyleSpec

STYLE_PREFS_KEY = "figure_style_presets"     # prefs slot: {name: StyleSpec.to_dict()}


def user_presets() -> dict:
    """``{name: StyleSpec}`` for the user-saved presets (empty if none)."""
    raw = prefs.get(STYLE_PREFS_KEY, {}) or {}
    out: dict = {}
    if isinstance(raw, dict):
        for name, d in raw.items():
            try:
                spec = StyleSpec.from_dict(d)
                spec.name = str(name)
                out[str(name)] = spec
            except Exception:  # noqa: BLE001 — a corrupt entry must not sink the whole library
                continue
    return out


def all_presets() -> dict:
    """Built-ins first, then user presets (a user name shadows a built-in of the same name).
    Insertion order is preserved so pickers list built-ins on top."""
    out = dict(stylespec.PRESETS)                # built-ins (App Default … High-contrast)
    out.update(user_presets())                   # user presets override by name
    return out


def names() -> list:
    return list(all_presets().keys())


def get(name: str) -> "StyleSpec | None":
    return all_presets().get(name)


def is_builtin(name: str) -> bool:
    return name in stylespec.PRESETS and name not in user_presets()


def save(spec: StyleSpec) -> StyleSpec:
    """Persist ``spec`` as a user preset under ``spec.name``. Returns the stored spec.

    A built-in's name may be reused — the user copy shadows it and remains deletable (which
    reveals the built-in again). Raises ``ValueError`` on an empty name."""
    name = (spec.name or "").strip()
    if not name:
        raise ValueError("a preset needs a name")
    store = dict(prefs.get(STYLE_PREFS_KEY, {}) or {})
    spec = spec.copy(name=name)
    store[name] = spec.to_dict()
    prefs.set(STYLE_PREFS_KEY, store)
    return spec


def delete(name: str) -> bool:
    """Remove a *user* preset. Built-ins can't be deleted (returns ``False``). Returns ``True``
    when a user preset was removed."""
    store = dict(prefs.get(STYLE_PREFS_KEY, {}) or {})
    if name in store:
        store.pop(name)
        prefs.set(STYLE_PREFS_KEY, store)
        return True
    return False


def export_json(name: str, *, indent: int = 2) -> str:
    """The JSON recipe for preset ``name`` (raises ``KeyError`` if unknown)."""
    spec = get(name)
    if spec is None:
        raise KeyError(name)
    return spec.to_json(indent=indent)


def import_json(text: str, *, name: str | None = None, save_it: bool = True) -> StyleSpec:
    """Parse a JSON recipe into a :class:`StyleSpec` (optionally rename and persist it).
    This is the receiving half of the assistant round-trip: paste an edited recipe, get a
    named preset back."""
    import json
    raw = json.loads(text)
    spec = StyleSpec.from_dict(raw)
    if name:
        spec.name = name
    elif not str(raw.get("name", "")).strip():   # recipe carried no explicit name
        spec.name = "Imported"
    if not (spec.name or "").strip():
        spec.name = "Imported"
    if save_it:
        spec = save(spec)
    return spec
