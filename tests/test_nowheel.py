"""Wheel-safe value widgets — the guard against silently re-tuning an analysis.

Spinning the wheel while merely *hovering* over a plain QComboBox / QSpinBox / QSlider steps
its value. Over a feature-set picker that swaps the active set out from under you mid-scroll;
over ``ppm_spin`` or ``snr_spin`` it re-tunes the analysis and the next run answers a question
you never asked. Nothing announces it and the result still looks plausible, which is what
makes it worth a test rather than a code comment.

The contract: the wheel acts only once the widget has focus, and an unfocused widget *ignores*
the event rather than swallowing it, so the enclosing scroll area still scrolls.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from smile_msi.gui.common import (NoScrollComboBox, NoScrollDoubleSpinBox,  # noqa: E402
                                  NoScrollSlider, NoScrollSpinBox, _NoWheelMixin)


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def host(app):
    """A shown, activated window with somewhere for focus to park.

    Both matter: ``setFocus()`` is a no-op on a widget whose window was never shown, and
    activating a window focuses its first child — so a 'hover' test must move focus away
    explicitly or it silently becomes a 'focused' test.
    """
    win = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(win)
    park = QtWidgets.QLineEdit()
    lay.addWidget(park)
    win.show()
    win.activateWindow()
    app.processEvents()
    win._park, win._lay = park, lay
    yield win
    win.close()


def _wheel(w, dy=-120):
    ev = QtGui.QWheelEvent(QtCore.QPointF(5, 5), w.mapToGlobal(QtCore.QPoint(5, 5)),
                           QtCore.QPoint(0, 0), QtCore.QPoint(0, dy),
                           QtCore.Qt.NoButton, QtCore.Qt.NoModifier,
                           QtCore.Qt.NoScrollPhase, False)
    QtWidgets.QApplication.sendEvent(w, ev)
    return ev


def _hover_scroll(host, w):
    host._park.setFocus()
    QtWidgets.QApplication.processEvents()
    assert not w.hasFocus()
    return _wheel(w)


def _focused_scroll(host, w):
    w.setFocus()
    QtWidgets.QApplication.processEvents()
    assert w.hasFocus(), "the fixture window must be shown+activated for focus to take"
    return _wheel(w)


def _add(host, w):
    host._lay.addWidget(w)
    QtWidgets.QApplication.processEvents()
    return w


def test_plain_widgets_really_do_step_on_hover(host):
    """The bug this guards. If Qt ever stops stealing the wheel, the guard is dead weight and
    this test says so, rather than every other test in this file quietly passing for free."""
    c = _add(host, QtWidgets.QComboBox())
    c.addItems(["a", "b", "c"])
    c.setCurrentIndex(1)
    _hover_scroll(host, c)
    assert c.currentIndex() == 2, "a plain QComboBox steps on hover — that is the bug"

    s = _add(host, QtWidgets.QSpinBox())
    s.setRange(0, 100)
    s.setValue(10)
    _hover_scroll(host, s)
    assert s.value() == 9


def test_combo_wheel_needs_focus(host):
    c = _add(host, NoScrollComboBox())
    c.addItems(["a", "b", "c"])
    c.setCurrentIndex(1)
    _hover_scroll(host, c)
    assert c.currentIndex() == 1
    _focused_scroll(host, c)
    assert c.currentIndex() == 2


def test_spin_boxes_hold_their_value_on_hover(host):
    """`ppm_spin`, `id_ppm_spin`, `snr_spin` are all NoScrollDoubleSpinBox."""
    s = _add(host, NoScrollSpinBox())
    s.setRange(0, 100)
    s.setValue(10)
    _hover_scroll(host, s)
    assert s.value() == 10
    _focused_scroll(host, s)
    assert s.value() == 9

    d = _add(host, NoScrollDoubleSpinBox())
    d.setRange(0.0, 100.0)
    d.setSingleStep(0.5)
    d.setValue(10.0)
    _hover_scroll(host, d)
    assert d.value() == 10.0
    _focused_scroll(host, d)
    assert d.value() == 9.5


def test_slider_holds_its_value_on_hover(host):
    """A stray step on the segmentation Detail slider re-cuts the whole tree."""
    sl = _add(host, NoScrollSlider(QtCore.Qt.Horizontal))
    sl.setRange(0, 20)
    sl.setValue(10)
    _hover_scroll(host, sl)
    assert sl.value() == 10
    _focused_scroll(host, sl)
    assert sl.value() != 10


def test_lock_wheel_never_steps_even_when_focused(host):
    """For the feature-set selector, where each change rebuilds the whole feature list."""
    c = _add(host, NoScrollComboBox(lock_wheel=True))
    c.addItems(["a", "b", "c"])
    c.setCurrentIndex(1)
    _hover_scroll(host, c)
    assert c.currentIndex() == 1
    _focused_scroll(host, c)
    assert c.currentIndex() == 1


def test_unfocused_wheel_is_ignored_so_the_panel_can_scroll(host):
    """Ignored, not swallowed. Eating the event would leave a combo sitting in a scroll area
    as a dead zone the page refuses to scroll over — trading one annoyance for another."""
    c = _add(host, NoScrollComboBox())
    c.addItems(["a", "b", "c"])
    ev = _hover_scroll(host, c)
    assert not ev.isAccepted()


def test_focus_policy_drops_wheel_focus(host):
    """QComboBox/QAbstractSpinBox default to WheelFocus — the wheel itself grabs focus, which
    would make the very first hover-scroll focus the widget and the second one step it."""
    for w in (NoScrollComboBox(), NoScrollSpinBox(), NoScrollDoubleSpinBox(),
              NoScrollSlider(QtCore.Qt.Horizontal)):
        _add(host, w)
        assert w.focusPolicy() == QtCore.Qt.StrongFocus
        assert isinstance(w, _NoWheelMixin)


def test_two_hover_scrolls_in_a_row_still_do_nothing(host):
    """The WheelFocus regression, specifically: scroll twice without ever clicking."""
    c = _add(host, NoScrollComboBox())
    c.addItems(["a", "b", "c"])
    c.setCurrentIndex(1)
    _hover_scroll(host, c)
    _hover_scroll(host, c)
    assert c.currentIndex() == 1


def test_the_analysis_parameters_are_guarded():
    """The whole point: the widgets that decide what an analysis computes. A source-level check
    over every GUI module, with no exemptions — it fails the moment someone writes a raw
    QSpinBox back in. (common.py defines the guarded subclasses; `class NoScrollSlider(...,
    QtWidgets.QSlider):` doesn't match, because the pattern requires a call.)"""
    import pathlib
    import re

    pat = re.compile(r"QtWidgets\.(QComboBox\(\)|QSpinBox\(\)|QDoubleSpinBox\(\)|QSlider\()")
    gui = pathlib.Path(__file__).resolve().parent.parent / "smile_msi" / "gui"
    scanned = sorted(p.name for p in gui.glob("*.py"))
    assert len(scanned) > 40, f"expected the whole gui package, scanned {len(scanned)}"
    offenders = {p.name: len(pat.findall(p.read_text())) for p in gui.glob("*.py")}
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, f"raw wheel-stealing widgets reintroduced: {offenders}"


def test_the_analysis_parameter_forms_are_guarded(app):
    """`form_from_params` is the single place a parameter form is built from the analysis
    registry, so every analysis's Configure & Run popup inherits whatever it produces. A raw
    spin box here would put a hover-scroll between the user and every parameter in the app."""
    from smile_msi.gui.common import ParamForm, RangeSliderField, form_from_params
    from smile_msi.registry import ParamSpec

    specs = [ParamSpec("ppm", "Tolerance (ppm)", "float", 10.0, 1.0, 50.0, 0.5),
             ParamSpec("n", "Clusters", "int", 6, 2, 24, 1),
             ParamSpec("method", "Method", "choice", "ward", choices=["ward", "kmeans"]),
             ParamSpec("flag", "Align", "bool", True)]

    rows, getter = form_from_params(specs)
    for _label, w in rows:
        if isinstance(w, (QtWidgets.QComboBox, QtWidgets.QAbstractSpinBox)):
            assert isinstance(w, _NoWheelMixin), type(w).__name__
    assert getter() == {"ppm": 10.0, "n": 6, "method": "ward", "flag": True}

    form = ParamForm(specs, {})
    values = form.findChildren(QtWidgets.QComboBox) + form.findChildren(QtWidgets.QAbstractSpinBox)
    assert values, "the form must actually build value widgets"
    assert all(isinstance(w, _NoWheelMixin) for w in values)

    r = RangeSliderField()                       # the intensity window's typed low/high
    assert isinstance(r.lo_spin, _NoWheelMixin) and isinstance(r.hi_spin, _NoWheelMixin)


def test_feature_export_join_tolerance_is_wheel_safe(host):
    """`FeatureExportDialog._ppm` is the m/z tolerance every analysis is joined onto the
    feature list with. A hover-scroll over it would silently re-join the whole wide CSV at a
    tolerance nobody chose — and the file would look entirely reasonable."""
    import pandas as pd

    from smile_msi.gui.featureexport import FeatureExportDialog

    base = pd.DataFrame({"m/z": [700.0, 750.0], "lipid": ["PC 34:1", "PE 36:2"]})
    dlg = FeatureExportDialog(None, base, [])
    try:
        assert isinstance(dlg._ppm, _NoWheelMixin), f"_ppm is a raw {type(dlg._ppm).__name__}"
        assert dlg._ppm.focusPolicy() == QtCore.Qt.StrongFocus

        # and it really does hold its value under a hover-scroll
        host._lay.addWidget(dlg._ppm)
        QtWidgets.QApplication.processEvents()
        before = dlg._ppm.value()
        _hover_scroll(host, dlg._ppm)
        assert dlg._ppm.value() == before
    finally:
        dlg.deleteLater()
