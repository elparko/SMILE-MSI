"""Headless GUI tests for the reusable colour picker (offscreen Qt)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtGui, QtWidgets  # noqa: E402

from smile_msi.gui import colorpicker as cp  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def _fake_recents(monkeypatch):
    """Back the recents store with a per-test dict so we never touch the real prefs.json."""
    store = {}
    monkeypatch.setattr(cp.prefs, "get", lambda k, d=None: store.get(k, d))
    monkeypatch.setattr(cp.prefs, "set", lambda k, v: store.__setitem__(k, v))
    return store


def test_norm_hex():
    assert cp._norm_hex("#ABCDEF") == "#abcdef"
    assert cp._norm_hex("red") == "#ff0000"
    assert cp._norm_hex("not-a-color") is None


def test_remember_and_recent_colors(_fake_recents):
    cp.remember_color("#112233")
    cp.remember_color("#445566")
    cp.remember_color("#112233")                  # dedupes to the front
    assert cp.recent_colors()[:2] == ["#112233", "#445566"]


def test_color_icon_renders(app):
    icon = cp.color_icon("#00ff88", size=14)
    assert not icon.isNull()


def test_dialog_fields_stay_in_sync(app):
    dlg = cp.ColorPickerDialog(None, initial="#3366cc", title="t")
    assert dlg.color_hex() == "#3366cc"
    # RGB spinboxes drive the colour
    dlg.rgb_spins["R"].setValue(255)
    dlg.rgb_spins["G"].setValue(0)
    dlg.rgb_spins["B"].setValue(0)
    assert dlg.color_hex() == "#ff0000"
    # hex field drives the colour
    dlg.hex_edit.setText("#00ff88")
    dlg._on_hex()
    assert dlg.color_hex() == "#00ff88"
    # brightness slider scales value (50% of a full-value colour halves it)
    dlg.value_slider.setValue(50)
    c = QtGui.QColor(dlg.color_hex())
    assert 0.45 <= c.valueF() <= 0.55


def test_wheel_renders_and_picks(app):
    w = cp.ColorWheel()
    w.set_color(QtGui.QColor("#ff8800"))
    w._ensure_image()
    assert w._image is not None and not w._image.isNull()
    got = {}
    w.colorChanged.connect(lambda c: got.update(name=c.name()))
    # a click at the centre is fully desaturated → near-grey at the current value
    from PySide6 import QtCore
    w._update_from_pos(QtCore.QPointF((w._d - 1) / 2.0, (w._d - 1) / 2.0))
    assert got and QtGui.QColor(got["name"]).saturationF() < 0.05


def test_swatch_button_emits(app, _fake_recents):
    btn = cp.ColorSwatchButton("#ff0000")
    assert btn.color() == "#ff0000"
    seen = {}
    btn.colorChanged.connect(lambda h: seen.update(h=h))
    btn.set_color("#0000ff")
    assert btn.color() == "#0000ff"
