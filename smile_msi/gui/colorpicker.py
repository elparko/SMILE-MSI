"""A richer, reusable colour picker — used everywhere the app lets you choose a colour.

Replaces the bare ``QColorDialog`` with one dialog that offers a **hue/saturation colour
wheel** + a brightness slider, live **RGB / hex** fields, a row of **recent colours**, and
clickable **palette-preset swatches** (the app's categorical palettes + the CVD-safe overlay
channels from :mod:`smile_msi.palettes`). Recents persist across sessions via
:mod:`smile_msi.prefs`.

Public surface:

* :func:`pick_color` — modal convenience: returns a ``"#rrggbb"`` hex string or ``None``.
* :class:`ColorPickerDialog` — the dialog itself (use directly for non-modal flows).
* :class:`ColorSwatchButton` — a button that shows its colour and opens the picker on click.
* :func:`color_icon` — a small filled-square :class:`QIcon` for a colour (toolbar swatches).

The wheel is painted off a cached NumPy image (rebuilt only when brightness changes), so it
stays responsive; NumPy is imported lazily so this module is cheap to import at GUI startup.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from .common import NoScrollSlider, NoScrollSpinBox

from .. import prefs, palettes

_RECENTS_KEY = "color_recents"
_MAX_RECENTS = 24


# --------------------------------------------------------------------------- #
# Recents (persisted)
# --------------------------------------------------------------------------- #
def recent_colors() -> list:
    """The most-recently-picked colours (newest first), as ``"#rrggbb"`` strings."""
    vals = prefs.get(_RECENTS_KEY, []) or []
    out = []
    for v in vals:
        h = _norm_hex(v)
        if h and h not in out:
            out.append(h)
    return out[:_MAX_RECENTS]


def remember_color(hex_color: str) -> None:
    """Record a freshly-picked colour at the front of the recents list."""
    h = _norm_hex(hex_color)
    if not h:
        return
    cur = [c for c in recent_colors() if c != h]
    prefs.set(_RECENTS_KEY, [h] + cur[: _MAX_RECENTS - 1])


def _norm_hex(value) -> str | None:
    """Coerce anything QColor accepts to a lowercase ``"#rrggbb"`` (or ``None``)."""
    try:
        c = QtGui.QColor(value)
        return c.name().lower() if c.isValid() else None
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Small filled-square icon for a colour (toolbar / list swatches)
# --------------------------------------------------------------------------- #
def color_icon(hex_color: str, size: int = 14) -> QtGui.QIcon:
    """A rounded filled-square icon of ``hex_color`` — for swatch buttons/lists."""
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing, True)
    p.setBrush(QtGui.QColor(hex_color))
    p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 70), 1))
    p.drawRoundedRect(0, 0, size - 1, size - 1, 3, 3)
    p.end()
    return QtGui.QIcon(pm)


# --------------------------------------------------------------------------- #
# The hue/saturation colour wheel
# --------------------------------------------------------------------------- #
class ColorWheel(QtWidgets.QWidget):
    """An HSV colour wheel: hue = angle, saturation = radius. Brightness (value) is set
    externally (a slider) and re-tints the wheel. Emits :attr:`colorChanged` on drag."""

    colorChanged = QtCore.Signal(QtGui.QColor)

    def __init__(self, parent=None, diameter=212):
        super().__init__(parent)
        self._d = int(diameter)
        self.setFixedSize(self._d, self._d)
        self._h, self._s, self._v = 0.0, 0.0, 1.0
        self._image = None
        self._image_v = None                    # value the cached image was built for
        self.setCursor(QtCore.Qt.CrossCursor)

    # ----- state ----------------------------------------------------------- #
    def set_color(self, color: QtGui.QColor):
        """Move the marker to ``color`` (hue/sat) and re-tint to its value, without emitting."""
        h, s, v = color.hueF(), color.saturationF(), color.valueF()
        self._h = 0.0 if h < 0 else h           # grey → hue undefined (-1); keep current angle 0
        self._s, self._v = s, v
        self.update()

    def value(self) -> float:
        return self._v

    def set_value(self, v: float):
        """Set brightness in [0, 1] (re-tints the wheel) and emit the new colour."""
        self._v = max(0.0, min(1.0, float(v)))
        self.update()
        self._emit()

    def current(self) -> QtGui.QColor:
        c = QtGui.QColor()
        c.setHsvF(max(0.0, self._h), max(0.0, self._s), max(0.0, self._v))
        return c

    # ----- painting -------------------------------------------------------- #
    def _ensure_image(self):
        if self._image is not None and self._image_v == self._v:
            return
        try:
            import numpy as np
            d = self._d
            yy, xx = np.mgrid[0:d, 0:d]
            cx = cy = (d - 1) / 2.0
            dx = (xx - cx) / cx
            dy = (yy - cy) / cy
            r = np.sqrt(dx * dx + dy * dy)
            hue = (np.arctan2(-dy, dx) / (2 * np.pi)) % 1.0
            sat = np.clip(r, 0.0, 1.0)
            v = float(self._v)
            # vectorised HSV→RGB (piecewise by 60° hue sector)
            hp = hue * 6.0
            c = v * sat
            x = c * (1 - np.abs(hp % 2 - 1))
            m = v - c
            z = np.zeros_like(hp)
            i = hp.astype(int) % 6
            conds = [i == k for k in range(6)]
            rr = np.select(conds, [c, x, z, z, x, c]) + m
            gg = np.select(conds, [x, c, c, x, z, z]) + m
            bb = np.select(conds, [z, z, x, c, c, x]) + m
            a = np.where(r <= 1.0, 255.0, 0.0)
            a = np.where((r > 0.985) & (r <= 1.0), (1.0 - (r - 0.985) / 0.015) * 255.0, a)
            arr = np.dstack([rr * 255.0, gg * 255.0, bb * 255.0, a]).clip(0, 255).astype(np.uint8)
            arr = np.ascontiguousarray(arr)
            self._image = QtGui.QImage(arr.data, d, d, 4 * d, QtGui.QImage.Format_RGBA8888).copy()
            self._image_v = self._v
        except Exception:  # noqa: BLE001 — fall back to no wheel image rather than crash
            self._image = None
            self._image_v = self._v

    def paintEvent(self, _e):
        import math
        self._ensure_image()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        if self._image is not None:
            p.drawImage(0, 0, self._image)
        # selection marker at (r=sat, angle=hue)
        d = self._d
        cx = cy = (d - 1) / 2.0
        mx = cx + self._s * cx * math.cos(self._h * 2 * math.pi)
        my = cy - self._s * cy * math.sin(self._h * 2 * math.pi)
        ring = 6
        p.setBrush(self.current())
        p.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 2))
        p.drawEllipse(QtCore.QPointF(mx, my), ring, ring)
        p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 150), 1))
        p.drawEllipse(QtCore.QPointF(mx, my), ring, ring)
        p.end()

    # ----- interaction ----------------------------------------------------- #
    def mousePressEvent(self, e):
        self._update_from_pos(e.position())

    def mouseMoveEvent(self, e):
        if e.buttons() & QtCore.Qt.LeftButton:
            self._update_from_pos(e.position())

    def _update_from_pos(self, pos):
        import math
        d = self._d
        cx = cy = (d - 1) / 2.0
        dx = (pos.x() - cx) / cx
        dy = (pos.y() - cy) / cy
        r = math.hypot(dx, dy)
        self._h = (math.atan2(-dy, dx) / (2 * math.pi)) % 1.0
        self._s = min(1.0, r)
        self.update()
        self._emit()

    def _emit(self):
        self.colorChanged.emit(self.current())


# --------------------------------------------------------------------------- #
# The dialog
# --------------------------------------------------------------------------- #
class ColorPickerDialog(QtWidgets.QDialog):
    """Modal colour chooser — wheel + brightness + RGB/hex + recents + palette swatches."""

    def __init__(self, parent=None, initial="#ffffff", title="Pick colour",
                 swatch_groups=None, show_recents=True):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self._color = QtGui.QColor(initial if QtGui.QColor(initial).isValid() else "#ffffff")
        self._updating = False

        groups = list(swatch_groups) if swatch_groups is not None else palettes.picker_swatch_groups()
        if show_recents:
            rc = recent_colors()
            if rc:
                groups = [("Recent", rc)] + groups

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # ---- left: wheel + brightness slider ----
        left = QtWidgets.QHBoxLayout()
        left.setSpacing(8)
        self.wheel = ColorWheel(self)
        self.wheel.colorChanged.connect(self._on_wheel)
        left.addWidget(self.wheel, 0, QtCore.Qt.AlignTop)
        self.value_slider = NoScrollSlider(QtCore.Qt.Vertical)
        self.value_slider.setRange(0, 100)
        self.value_slider.setFixedHeight(self.wheel.height())
        self.value_slider.setToolTip("Brightness")
        self.value_slider.valueChanged.connect(self._on_value_slider)
        left.addWidget(self.value_slider, 0, QtCore.Qt.AlignTop)
        root.addLayout(left)

        # ---- right: preview + fields + swatches ----
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(8)
        self.preview = QtWidgets.QFrame()
        self.preview.setMinimumHeight(40)
        self.preview.setFrameShape(QtWidgets.QFrame.StyledPanel)
        right.addWidget(self.preview)

        hexrow = QtWidgets.QHBoxLayout()
        hexrow.addWidget(QtWidgets.QLabel("Hex"))
        self.hex_edit = QtWidgets.QLineEdit()
        self.hex_edit.setMaxLength(7)
        self.hex_edit.setFixedWidth(96)
        self.hex_edit.editingFinished.connect(self._on_hex)
        hexrow.addWidget(self.hex_edit)
        hexrow.addStretch(1)
        right.addLayout(hexrow)

        self.rgb_spins = {}
        rgbrow = QtWidgets.QHBoxLayout()
        for ch in ("R", "G", "B"):
            rgbrow.addWidget(QtWidgets.QLabel(ch))
            sp = NoScrollSpinBox()
            sp.setRange(0, 255)
            sp.setFixedWidth(58)
            sp.valueChanged.connect(self._on_rgb)
            self.rgb_spins[ch] = sp
            rgbrow.addWidget(sp)
        rgbrow.addStretch(1)
        right.addLayout(rgbrow)

        # palette / recent swatches
        sw_holder = QtWidgets.QWidget()
        sw_v = QtWidgets.QVBoxLayout(sw_holder)
        sw_v.setContentsMargins(0, 0, 0, 0)
        sw_v.setSpacing(6)
        for name, cols in groups:
            if not cols:
                continue
            lbl = QtWidgets.QLabel(name)
            lbl.setStyleSheet("color: palette(mid); font-size: 11px;")
            sw_v.addWidget(lbl)
            sw_v.addLayout(self._swatch_grid(cols))
        sw_v.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(sw_holder)
        scroll.setMinimumWidth(228)
        scroll.setMinimumHeight(150)
        right.addWidget(scroll, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        right.addWidget(buttons)
        root.addLayout(right, 1)

        self._set_color(self._color, source="init")

    # ----- swatch grid ----------------------------------------------------- #
    def _swatch_grid(self, cols, per_row=12):
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(3)
        for i, hexc in enumerate(cols):
            b = QtWidgets.QToolButton()
            b.setFixedSize(18, 18)
            b.setAutoRaise(True)
            b.setStyleSheet(
                f"QToolButton{{background:{hexc}; border:1px solid palette(mid); border-radius:3px;}}"
                f"QToolButton:hover{{border:2px solid palette(highlight);}}")
            b.setToolTip(str(hexc))
            b.clicked.connect(lambda _=0, c=hexc: self._set_color(QtGui.QColor(c), source="swatch"))
            grid.addWidget(b, i // per_row, i % per_row)
        return grid

    # ----- the single state setter (keeps every control in sync) ----------- #
    def _set_color(self, color: QtGui.QColor, source=""):
        if not color.isValid():
            return
        self._color = QtGui.QColor(color)
        self._updating = True
        try:
            if source != "wheel":
                self.wheel.set_color(self._color)
            if source not in ("slider", "wheel"):
                self.value_slider.setValue(int(round(self._color.valueF() * 100)))
            if source != "hex":
                self.hex_edit.setText(self._color.name())
            if source != "rgb":
                self.rgb_spins["R"].setValue(self._color.red())
                self.rgb_spins["G"].setValue(self._color.green())
                self.rgb_spins["B"].setValue(self._color.blue())
        finally:
            self._updating = False
        self._refresh_preview_and_slider()

    def _refresh_preview_and_slider(self):
        self.preview.setStyleSheet(
            f"background:{self._color.name()}; border:1px solid palette(mid); border-radius:4px;")
        # tint the brightness groove black → the pure hue/sat at full value
        pure = QtGui.QColor()
        pure.setHsvF(max(0.0, self._color.hueF()), max(0.0, self._color.saturationF()), 1.0)
        self.value_slider.setStyleSheet(
            "QSlider::groove:vertical{width:16px;border:1px solid palette(mid);border-radius:3px;"
            f"background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 {pure.name()},stop:1 #000000);}}"
            "QSlider::handle:vertical{background:#ffffff;border:1px solid #444;height:8px;"
            "margin:0 -4px;border-radius:2px;}")

    # ----- control signals ------------------------------------------------- #
    def _on_wheel(self, color):
        if self._updating:
            return
        # the wheel keeps the slider's brightness; merge it back in
        c = QtGui.QColor()
        c.setHsvF(max(0.0, color.hueF()), max(0.0, color.saturationF()),
                  self.value_slider.value() / 100.0)
        self._set_color(c, source="wheel")

    def _on_value_slider(self, v):
        if self._updating:
            return
        c = QtGui.QColor()
        c.setHsvF(max(0.0, self._color.hueF()), max(0.0, self._color.saturationF()), v / 100.0)
        self.wheel.set_value(v / 100.0)
        self._set_color(c, source="slider")

    def _on_hex(self):
        if self._updating:
            return
        txt = self.hex_edit.text().strip()
        if not txt.startswith("#"):
            txt = "#" + txt
        c = QtGui.QColor(txt)
        if c.isValid():
            self._set_color(c, source="hex")

    def _on_rgb(self, _=0):
        if self._updating:
            return
        c = QtGui.QColor(self.rgb_spins["R"].value(), self.rgb_spins["G"].value(),
                         self.rgb_spins["B"].value())
        self._set_color(c, source="rgb")

    # ----- result ---------------------------------------------------------- #
    def color_hex(self) -> str:
        return self._color.name()


# --------------------------------------------------------------------------- #
# Convenience: modal pick → hex or None
# --------------------------------------------------------------------------- #
def pick_color(parent=None, initial="#ffffff", title="Pick colour", swatch_groups=None) -> str | None:
    """Open the picker modally; return the chosen ``"#rrggbb"`` (and record it as recent) or
    ``None`` if cancelled. Drop-in for ``QColorDialog.getColor(...).name()`` flows."""
    dlg = ColorPickerDialog(parent, initial=initial or "#ffffff", title=title,
                            swatch_groups=swatch_groups)
    if dlg.exec() == QtWidgets.QDialog.Accepted:
        hexc = dlg.color_hex()
        remember_color(hexc)
        return hexc
    return None


# --------------------------------------------------------------------------- #
# A button that shows its colour and opens the picker
# --------------------------------------------------------------------------- #
class ColorSwatchButton(QtWidgets.QPushButton):
    """A compact button showing its current colour; clicking opens :func:`pick_color`. Emits
    :attr:`colorChanged(str)` with the new hex when the user accepts a colour."""

    colorChanged = QtCore.Signal(str)

    def __init__(self, color="#ffffff", parent=None, *, title="Pick colour", swatch_groups=None):
        super().__init__(parent)
        self._color = _norm_hex(color) or "#ffffff"
        self._title = title
        self._swatch_groups = swatch_groups
        self.setFixedWidth(34)
        self.setIcon(color_icon(self._color))
        self.setToolTip("Click to change colour")
        self.clicked.connect(self._open)

    def color(self) -> str:
        return self._color

    def set_color(self, hex_color: str):
        h = _norm_hex(hex_color)
        if h:
            self._color = h
            self.setIcon(color_icon(h))

    def _open(self):
        hexc = pick_color(self, initial=self._color, title=self._title,
                          swatch_groups=self._swatch_groups)
        if hexc:
            self.set_color(hexc)
            self.colorChanged.emit(hexc)
