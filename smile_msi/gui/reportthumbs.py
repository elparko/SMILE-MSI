"""Persisted preview thumbnails for Report-tab / Analyses-browser items.

Figure-bearing report items (ion images, overlays, spectra, segmentation maps) are
expensive to re-render and need the heavy m/z cube loaded. Caching a small PNG per item
beside the managed session lets the browser paint a preview instantly — and crucially,
*before* (or without) the dataset's cube has finished loading on reopen.

The cache is a sidecar **directory** next to the managed session file (mirroring the
``<stem>.cache.npz`` cube sidecar), so the session JSON stays small: each item persists
only a ``thumb`` filename, never the image bytes. These helpers are deliberately pure
(no GUI/MainWindow coupling) and entirely best-effort — a thumbnail is a cache, so any
read/write failure degrades to a live render and never breaks the app.
"""
from __future__ import annotations

import os

from PySide6 import QtCore, QtGui

THUMB_MAX_PX = 256


def thumb_dir_for(session_path: str | None) -> str | None:
    """Directory holding an item's cached thumbnails, derived from the managed session
    path (``<stem>__thumbs/``). Returns ``None`` when there's no session yet (a synthetic
    dataset, or before the first auto-save), so callers simply skip caching."""
    if not session_path:
        return None
    stem = os.path.splitext(str(session_path))[0]
    return stem + "__thumbs"


def thumb_path(thumb_dir: str | None, item_id: str | None) -> str | None:
    """Absolute PNG path for one item's thumbnail, or ``None`` when either part is missing."""
    if not thumb_dir or not item_id:
        return None
    return os.path.join(thumb_dir, f"{item_id}.png")


def write_pixmap(path: str | None, pixmap, *, max_px: int = THUMB_MAX_PX) -> bool:
    """Downscale ``pixmap`` to fit ``max_px`` (keeping aspect) and save it as PNG. Creates
    the sidecar directory on demand. Best-effort: returns ``True`` on success, ``False`` on
    any failure (missing path, null pixmap, I/O error) without raising."""
    if not path or pixmap is None or pixmap.isNull():
        return False
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        scaled = pixmap.scaled(max_px, max_px, QtCore.Qt.KeepAspectRatio,
                               QtCore.Qt.SmoothTransformation)
        return bool(scaled.save(path, "PNG"))
    except Exception:  # noqa: BLE001 — a cache write must never break a capture
        return False


def load_pixmap(path: str | None):
    """Load a cached thumbnail as a ``QPixmap``, or ``None`` if it's missing/unreadable."""
    if not path or not os.path.exists(path):
        return None
    try:
        pix = QtGui.QPixmap(path)
        return pix if not pix.isNull() else None
    except Exception:  # noqa: BLE001
        return None


def prune(thumb_dir: str | None, live_ids) -> int:
    """Delete cached PNGs whose item id is no longer present in ``live_ids`` (a set/iterable
    of the surviving items' ids). Returns how many files were removed. Best-effort."""
    if not thumb_dir or not os.path.isdir(thumb_dir):
        return 0
    keep = {str(i) for i in (live_ids or [])}
    removed = 0
    try:
        for fn in os.listdir(thumb_dir):
            if not fn.endswith(".png"):
                continue
            if os.path.splitext(fn)[0] not in keep:
                try:
                    os.remove(os.path.join(thumb_dir, fn))
                    removed += 1
                except OSError:
                    pass
    except OSError:
        return removed
    return removed
