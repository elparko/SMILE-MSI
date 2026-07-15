"""Lasso primitives shared by the pixel-embedding scatters.

The interactive **Feature space** tab this module used to build was retired in the
plan-24 move to the Analyze gallery. Only the freehand-lasso helpers survive —
:func:`points_in_polygon` and :class:`_LassoViewBox` are still imported by the
embedding view in :mod:`.resultviews`. ``ScatterMixin`` is kept as an empty shell
until its slot is dropped from ``MainWindow``'s base list.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore


def points_in_polygon(xy, poly):
    """Even-odd ray-cast: boolean mask of which ``(N,2)`` points fall inside the
    polygon given by ``(M,2)`` vertices. Pure vectorised numpy (loops only over the
    M edges), so it scales to every pixel at once."""
    xy = np.asarray(xy, dtype=float)
    poly = np.asarray(poly, dtype=float)
    if xy.size == 0 or len(poly) < 3:
        return np.zeros(xy.shape[0], dtype=bool)
    x, y = xy[:, 0], xy[:, 1]
    inside = np.zeros(x.shape[0], dtype=bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        straddles = (yi > y) != (yj > y)
        denom = yj - yi
        denom = denom if denom != 0 else 1e-12
        x_cross = xi + (y - yi) / denom * (xj - xi)
        inside ^= straddles & (x < x_cross)
        j = i
    return inside


class _LassoViewBox(pg.ViewBox):
    """A ViewBox that, in lasso mode, captures a freehand drag into a polygon and
    emits it; otherwise it pans/zooms like a normal plot."""

    lassoFinished = QtCore.Signal(object)            # (M,2) ndarray of view-space vertices

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lasso = False
        self._pts = []
        self._curve = pg.PlotCurveItem(pen=pg.mkPen("y", width=1.5))
        self._curve.setZValue(20)
        self.addItem(self._curve)

    def setLasso(self, on):
        self._lasso = bool(on)
        self.setMouseEnabled(not on, not on)         # no pan/zoom while lassoing

    def mouseDragEvent(self, ev, axis=None):
        if not self._lasso:
            return super().mouseDragEvent(ev, axis)
        ev.accept()
        p = self.mapSceneToView(ev.scenePos())
        if ev.isStart():
            self._pts = []
        self._pts.append([p.x(), p.y()])
        if len(self._pts) >= 2:
            arr = np.asarray(self._pts)
            self._curve.setData(arr[:, 0], arr[:, 1])
        if ev.isFinish():
            poly = np.asarray(self._pts, dtype=float)
            self._pts = []
            self._curve.clear()
            if len(poly) >= 3:
                self.lassoFinished.emit(poly)


class ScatterMixin:
    """Retired Feature-space tab builder — see the module docstring. Kept as an
    empty base only so ``MainWindow`` still imports until its slot is dropped."""
