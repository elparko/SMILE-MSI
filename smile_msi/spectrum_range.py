"""Pure-numpy helper for cropping a spectrum's m/z axis to where the signal is.

Kept Qt- and matplotlib-free so the live pyqtgraph views (via ``gui.common``) and the
headless export engine (``export.py``) share one definition of "the visible range"."""
from __future__ import annotations

import numpy as np


def signal_mz_range(axis, *specs, frac=0.001, pad_frac=0.02):
    """The m/z window that actually holds signal — for auto-cropping a spectrum's
    x-axis to its peaks instead of the full acquired range (which often trails a long
    empty tail past the last real peak; see the region-comparison butterfly plot).

    Scans every intensity array given — so region A *and* B together, or a lone mean
    spectrum — and returns ``(lo, hi)`` spanning every bin whose magnitude clears
    ``frac`` of the tallest peak, padded by ``pad_frac`` of that span on each side so
    edge peaks aren't flush against the frame. Magnitudes are absolute, so a signed
    difference spectrum crops to where *either* side has signal. Returns ``None`` when
    there's no usable signal, so callers fall back to the full axis."""
    axis = np.asarray(axis, dtype=float)
    if axis.size == 0:
        return None
    mag = None
    for spec in specs:
        s = np.abs(np.asarray(spec, dtype=float))
        if s.size != axis.size:
            continue
        mag = s if mag is None else np.maximum(mag, s)
    if mag is None or mag.size == 0:
        return None
    peak = float(np.nanmax(mag))
    if not peak > 0:
        return None
    idx = np.flatnonzero(mag >= peak * frac)
    if idx.size == 0:
        return None
    lo, hi = float(axis[idx[0]]), float(axis[idx[-1]])
    if not hi > lo:
        return None
    pad = (hi - lo) * pad_frac
    return (lo - pad, hi + pad)
