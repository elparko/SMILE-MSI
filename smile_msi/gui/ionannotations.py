"""Live, in-image annotation overlay for the ion view — the on-screen twin of the exported
legend + scale bar that :mod:`smile_msi.export` bakes into exported figures (WYSIWYG).

It is a transparent ``QWidget`` laid over the ion image's ``GraphicsView``. On every paint it
anchors the legend to the **bottom-right of the viewport** (and the scale bar to the
bottom-left) — pinned to the widget, not the tissue, so the annotations stay put when the
layout changes (expanding the spectrum, opening/closing side panels) instead of hopping to
whichever margin was emptiest. It paints — directly on the image, no box — one row per ion: a
right-aligned ``m/z ± ppm`` label, a transparency checkerboard, a gradient bar, a trailing
relative-max ``%``, and dim ``lo``/``hi`` contrast labels under the bar ends; plus a scale bar
(line + end-ticks + length label) in the opposite bottom corner.

The widget is mouse-transparent so it never blocks ROI drawing / panning; the label text is
shared with the export via :func:`smile_msi.annotations.format_ion_label`, and rename
hit-testing is exposed through :meth:`label_hit` for the view's double-click handler.
"""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .. import annotations as annot

# px layout (screen-fixed, so text stays readable at any zoom)
_FONT_LABEL = 11
_FONT_SMALL = 9
_BAR_W = 92
_BAR_H = 11
_CHECK_W = 13
_GAP_LABEL = 8
_GAP_CHECK = 4
_GAP_TRAIL = 6
_TRAIL_W = 40
_UNDER_H = 14
_ROW_GAP = 11
_MARGIN = 14
_WHITE = QtGui.QColor("#f2f4f8")        # primary ink on a dark backing
_DIM = QtGui.QColor("#cfd6e0")          # secondary ink (lo/hi, overflow) on a dark backing
_INK_DARK = QtGui.QColor("#101418")     # primary ink on a light backing
_DIM_DARK = QtGui.QColor("#3a4150")     # secondary ink on a light backing


class IonAnnotations(QtCore.QObject):
    """Owns the transparent overlay widget and keeps it in sync with the ion view."""

    def __init__(self, iv):
        super().__init__(iv)
        self.iv = iv
        self.gv = iv.ui.graphicsView                 # the QGraphicsView holding the image
        self.vb = iv.getView()                       # the ViewBox (data coords)
        self.w = _OverlayWidget(self)
        self._fit()
        self.gv.installEventFilter(self)
        try:
            self.vb.sigRangeChanged.connect(lambda *_: self.w.update())
        except Exception:  # noqa: BLE001
            pass
        # state
        self.rows = []                               # list of dicts (see set_data)
        self.subject = None                          # (r0, r1, c0, c1) in data/array coords
        self.pixel_size_um = None
        self.scale_bar_um = None
        self.label_mode = annot.LABEL_MZ_PPM
        self._label_rects = []                       # [(QRectF in widget px, mz)] for rename hits

    # ----- public API ------------------------------------------------------ #
    def set_data(self, rows, *, subject=None, pixel_size_um=None, scale_bar_um=None,
                 label_mode=annot.LABEL_MZ_PPM):
        """``rows`` is a list of dicts: ``{mz, color|None, cmap|None, lo, hi, peak, name,
        override}``. ``subject`` is ``(r0, r1, c0, c1)`` in data coords or None."""
        self.rows = list(rows or [])
        self.subject = subject
        self.pixel_size_um = pixel_size_um
        self.scale_bar_um = scale_bar_um
        self.label_mode = label_mode or annot.LABEL_MZ_PPM
        self.w.setVisible(bool(self.rows))
        self._fit()
        self.w.update()

    def clear(self):
        self.rows = []
        self.w.setVisible(False)

    def label_hit(self, scene_pos):
        """Return the row's m/z whose label was double-clicked (scene coords), or None."""
        gp = self.gv.mapFromScene(scene_pos)
        pt = QtCore.QPointF(gp)
        for rect, mz in self._label_rects:
            if rect.contains(pt):
                return mz
        return None

    # ----- internals ------------------------------------------------------- #
    def eventFilter(self, obj, ev):
        if obj is self.gv and ev.type() in (QtCore.QEvent.Resize, QtCore.QEvent.Move,
                                            QtCore.QEvent.Show):
            self._fit()
        return False

    def _fit(self):
        self.w.setGeometry(self.gv.viewport().geometry())
        self.w.raise_()

    def _to_widget(self, vx, vy):
        sp = self.vb.mapViewToScene(QtCore.QPointF(float(vx), float(vy)))
        return QtCore.QPointF(self.gv.mapFromScene(sp))

    def _corner(self):
        """Fixed bottom corner for the legend (the scale bar takes the opposite bottom
        corner). Pinned to the *viewport widget* — not the projected tissue — so the
        annotations stay anchored to the bottom of the view and don't hop to another margin
        when the layout changes (expanding the spectrum, opening/closing side panels). Legend
        bottom-right, scale bar bottom-left, matching the exported footer band."""
        return "right", "lower"

    def _scale_len_px(self):
        if not (self.pixel_size_um and self.scale_bar_um):
            return 0.0
        n = float(self.scale_bar_um) / float(self.pixel_size_um)   # data units (pixels)
        a = self._to_widget(0, 0)
        b = self._to_widget(n, 0)                                  # along view-x (isotropic)
        return float(np.hypot(b.x() - a.x(), b.y() - a.y()))


def _gradient_for(rect, *, color=None, cmap=None):
    """A QLinearGradient across ``rect``: black→``color`` for an overlay channel, or sampled
    from the named pyqtgraph ``cmap`` for a single-ion view."""
    g = QtGui.QLinearGradient(rect.left(), 0, rect.right(), 0)
    if cmap is not None:
        try:
            from .common import colormap
            arr = colormap(cmap).map(np.linspace(0, 1, 8), mode='byte')
            for i, c in enumerate(arr):
                g.setColorAt(i / 7.0, QtGui.QColor(int(c[0]), int(c[1]), int(c[2])))
            return g
        except Exception:  # noqa: BLE001
            pass
    col = QtGui.QColor(color or "#ffffff")
    g.setColorAt(0.0, QtGui.QColor(0, 0, 0))
    g.setColorAt(1.0, col)
    return g


def _checker(p, rect):
    """Paint a small grey transparency checkerboard into ``rect``."""
    n = 3
    cw, ch = rect.width() / n, rect.height() / 2
    light, dark = QtGui.QColor("#9aa0aa"), QtGui.QColor("#5b616c")
    for i in range(n):
        for j in range(2):
            c = light if (i + j) % 2 == 0 else dark
            p.fillRect(QtCore.QRectF(rect.left() + i * cw, rect.top() + j * ch, cw + 0.5, ch + 0.5), c)


class _OverlayWidget(QtWidgets.QWidget):
    def __init__(self, owner):
        super().__init__(owner.gv)
        self.owner = owner
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)

    def _bg_is_light(self, rect):
        """Mean luminance of the rendered image *under* ``rect`` (overlay-widget px): True
        when the backing is light enough to want dark ink. Falls back to dark (→ white ink)
        when there's nothing to sample. Mirrors annotations.pick_ink's 0.5 threshold."""
        img = getattr(self, "_bg_img", None)
        if img is None or img.isNull():
            return False
        s = getattr(self, "_bg_scale", 1.0) or 1.0
        x0 = max(0, int(rect.left() * s)); y0 = max(0, int(rect.top() * s))
        x1 = min(img.width() - 1, int(rect.right() * s))
        y1 = min(img.height() - 1, int(rect.bottom() * s))
        if x1 <= x0 or y1 <= y0:
            return False
        sx = max(1, (x1 - x0) // 6); sy = max(1, (y1 - y0) // 3)
        tot = 0.0; n = 0
        for yy in range(y0, y1 + 1, sy):
            for xx in range(x0, x1 + 1, sx):
                c = img.pixelColor(xx, yy)
                if c.alpha() < 16:
                    continue
                tot += 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
                n += 1
        return n > 0 and (tot / n) >= 128.0

    def _ink(self, rect, *, dim=False):
        """Legible ink for text/marks over ``rect``: white-on-dark, near-black-on-light."""
        if self._bg_is_light(rect):
            return _DIM_DARK if dim else _INK_DARK
        return _DIM if dim else _WHITE

    def _text(self, p, rect, flags, s, *, font, dim=False):
        # no shadow: pick an ink that contrasts with whatever sits under this label
        p.setFont(font)
        p.setPen(self._ink(rect, dim=dim))
        p.drawText(rect, flags, s)

    def paintEvent(self, _ev):
        o = self.owner
        o._label_rects = []
        if not o.rows:
            return
        # snapshot the rendered image *beneath* this overlay (the viewport, excluding our own
        # text) so every label and the scale bar can pick a legible ink from its own backing.
        # Downscale first: we only need coarse luminance, and scaling the pixmap keeps the
        # per-paint cost low during pan/zoom. _bg_scale maps logical overlay px → snapshot px.
        self._bg_img = None
        self._bg_scale = 1.0
        try:
            pm = o.gv.viewport().grab()
            if pm.width() > 240:
                pm = pm.scaledToWidth(240, QtCore.Qt.FastTransformation)
            self._bg_img = pm.toImage()
            self._bg_scale = self._bg_img.width() / max(1, self.width())
        except Exception:  # noqa: BLE001 — degrade to white ink if the grab fails
            self._bg_img = None
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        p.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
        f_lab = QtGui.QFont(); f_lab.setPointSize(_FONT_LABEL); f_lab.setBold(True)
        f_sm = QtGui.QFont(); f_sm.setPointSize(_FONT_SMALL)
        fm = QtGui.QFontMetricsF(f_lab)

        rows = o.rows[:annot._MAX_ROWS]
        overflow = len(o.rows) - len(rows)
        labels = [annot.format_ion_label(r.get("mz"), ppm=r.get("ppm"),
                                         name=r.get("name", ""), override=r.get("override"),
                                         mode=o.label_mode) for r in rows]
        label_w = max((fm.horizontalAdvance(s) for s in labels), default=60.0)
        has_trail = any(r.get("peak") for r in rows)
        trail_w = _TRAIL_W if has_trail else 0.0
        block_w = label_w + _GAP_LABEL + _CHECK_W + _GAP_CHECK + _BAR_W + \
            (_GAP_TRAIL + trail_w if trail_w else 0.0)
        row_pitch = _BAR_H + _UNDER_H + _ROW_GAP
        block_h = row_pitch * len(rows)

        W, H = self.width(), self.height()
        horiz, vert = o._corner()
        x0 = (W - _MARGIN - block_w) if horiz == "right" else _MARGIN
        y0 = (H - _MARGIN - block_h) if vert == "lower" else _MARGIN

        label_right = x0 + label_w
        check_x = label_right + _GAP_LABEL
        bar_x = check_x + _CHECK_W + _GAP_CHECK
        trail_x = bar_x + _BAR_W + _GAP_TRAIL

        for i, (r, s) in enumerate(zip(rows, labels)):
            bar_top = y0 + i * row_pitch
            bar = QtCore.QRectF(bar_x, bar_top, _BAR_W, _BAR_H)
            # label (right-aligned, vertically centred on the bar)
            lab_rect = QtCore.QRectF(x0, bar_top - 2, label_w, _BAR_H + 4)
            self._text(p, lab_rect, int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter),
                       s, font=f_lab)
            o._label_rects.append((QtCore.QRectF(lab_rect), r.get("mz")))
            # checkerboard
            if r.get("checker", True):
                _checker(p, QtCore.QRectF(check_x, bar_top, _CHECK_W, _BAR_H))
            # gradient bar, hairline border tinted to contrast with the backing
            p.fillRect(bar, _gradient_for(bar, color=r.get("color"), cmap=r.get("cmap")))
            edge = QtGui.QColor(_INK_DARK if self._bg_is_light(bar) else _WHITE)
            edge.setAlpha(70)
            p.setPen(QtGui.QPen(edge, 0.8))
            p.drawRect(bar)
            # trailing relative-max %
            if r.get("peak"):
                tr = QtCore.QRectF(trail_x, bar_top - 2, trail_w, _BAR_H + 4)
                self._text(p, tr, int(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter),
                           f"{float(r['peak']):.0f}%", font=f_sm)
            # lo / hi under the bar ends
            under = QtCore.QRectF(bar_x, bar_top + _BAR_H, _BAR_W, _UNDER_H)
            if r.get("lo") is not None:
                self._text(p, under, int(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop),
                           f"{float(r['lo']):.0f}%", font=f_sm, dim=True)
            if r.get("hi") is not None:
                self._text(p, under, int(QtCore.Qt.AlignRight | QtCore.Qt.AlignTop),
                           f"{float(r['hi']):.0f}%", font=f_sm, dim=True)
        if overflow > 0:
            ov = QtCore.QRectF(x0, y0 + block_h, block_w, _UNDER_H)
            self._text(p, ov, int(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop),
                       f"+{overflow} more", font=f_sm, dim=True)

        self._paint_scalebar(p, horiz, f_sm)
        p.end()

    def _paint_scalebar(self, p, legend_horiz, font):
        o = self.owner
        length = o._scale_len_px()
        if length < 6:
            return
        W, H = self.width(), self.height()
        bar_um = float(o.scale_bar_um)
        max_len = annot._SCALEBAR_MAX_FRAC * W
        if length > max_len and length > 0:
            # zoomed in so far the fixed bar would overrun the view: recompute a smaller
            # nice round length from the *visible* physical width and relabel — never clamp
            # the line while keeping the caption (that draws a bar that lies about its length).
            visible_um = W * bar_um / length
            fitted = annot.nice_scalebar_um(visible_um)
            if fitted:
                length *= fitted / bar_um
                bar_um = fitted
            else:
                length = max_len
        on_left = legend_horiz == "right"           # bar opposite the legend
        y = H - _MARGIN - 6
        x1 = (_MARGIN + length) if on_left else (W - _MARGIN)
        x0 = x1 - length
        # no shadow: ink the line + end ticks to contrast with the backing under the bar
        ink = self._ink(QtCore.QRectF(min(x0, x1), y - 6, abs(x1 - x0), 12))
        pen = QtGui.QPen(ink, 2.4); pen.setCapStyle(QtCore.Qt.FlatCap)
        p.setPen(pen)
        p.drawLine(QtCore.QPointF(x0, y), QtCore.QPointF(x1, y))
        p.drawLine(QtCore.QPointF(x0, y - 5), QtCore.QPointF(x0, y + 5))
        p.drawLine(QtCore.QPointF(x1, y - 5), QtCore.QPointF(x1, y + 5))
        rect = QtCore.QRectF(x0, y - 5 - _UNDER_H, length, _UNDER_H)
        self._text(p, rect, int(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom),
                   annot.format_scalebar_label(bar_um), font=font)
