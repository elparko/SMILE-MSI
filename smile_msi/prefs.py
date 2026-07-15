"""Tiny persistent app-preferences store (``<home>/prefs.json``).

Holds cross-cutting UI preferences that aren't tied to a single dataset/session — the
last-used file-dialog folder (so every chooser reopens where you last browsed) and the
Export dialog's last-used selections. This is deliberately separate from the per-sample
session store (:mod:`smile_msi.session`): a preference applies to the *app*, not to one
dataset.

Loaded lazily and cached on first access; writes are immediate but best-effort — a failed
read or write degrades to defaults and never breaks the app.
"""
from __future__ import annotations

import json
import os

from . import library

_CACHE: dict | None = None


def _path() -> str:
    return os.path.join(library.home_dir(), "prefs.json")


def _load() -> dict:
    global _CACHE
    if _CACHE is None:
        try:
            with open(_path(), encoding="utf-8") as fh:
                data = json.load(fh)
            _CACHE = data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            _CACHE = {}
    return _CACHE


def _save() -> None:
    try:
        with open(_path(), "w", encoding="utf-8") as fh:
            json.dump(_load(), fh, indent=2)
    except OSError:
        pass


def get(key: str, default=None):
    """Value for ``key`` (a copy is *not* made — treat returned containers as read-only)."""
    return _load().get(key, default)


def set(key: str, value) -> None:  # noqa: A001 — mirrors QSettings.setValue naming
    """Store ``value`` under ``key`` and flush to disk."""
    _load()[key] = value
    _save()
