"""File choosers that remember the last-used folder.

Every GUI open/save dialog funnels through these wrappers so the folder you last browsed
to — whether opening or saving — becomes the default the next chooser opens in. The folder
is persisted via :mod:`smile_msi.prefs` (``<home>/prefs.json``), so it survives both
across dialogs within a session and across app restarts.

The wrappers mirror the ``QtWidgets.QFileDialog`` static-method signatures exactly
``(parent, caption, dir, filter)`` so they're drop-in replacements at every call site.
A caller that passes an explicit directory in ``default`` keeps it; a bare filename (or
empty string) is anchored in the remembered folder.
"""
from __future__ import annotations

import os

from PySide6 import QtWidgets

from .. import prefs

_KEY = "last_dir"


def _last_dir() -> str:
    d = prefs.get(_KEY, "") or ""
    return d if os.path.isdir(d) else ""


def _remember(path: str) -> None:
    if not path:
        return
    d = path if os.path.isdir(path) else os.path.dirname(path)
    if d and os.path.isdir(d):
        prefs.set(_KEY, d)


def _seed(default: str) -> str:
    """Anchor a bare filename in the remembered folder; respect an explicit directory."""
    if default and os.path.dirname(default):
        return default
    base = _last_dir()
    return os.path.join(base, default) if base else default


def get_save_file_name(parent, caption, default="", filter=""):  # noqa: A002 — Qt arg name
    path, sel = QtWidgets.QFileDialog.getSaveFileName(parent, caption, _seed(default), filter)
    _remember(path)
    return path, sel


def get_open_file_name(parent, caption, default="", filter=""):  # noqa: A002
    path, sel = QtWidgets.QFileDialog.getOpenFileName(parent, caption, _seed(default), filter)
    _remember(path)
    return path, sel


def get_open_file_names(parent, caption, default="", filter=""):  # noqa: A002
    paths, sel = QtWidgets.QFileDialog.getOpenFileNames(parent, caption, _seed(default), filter)
    if paths:
        _remember(paths[0])
    return paths, sel


def get_existing_directory(parent, caption, default=""):
    d = QtWidgets.QFileDialog.getExistingDirectory(parent, caption, default or _last_dir())
    _remember(d)
    return d
