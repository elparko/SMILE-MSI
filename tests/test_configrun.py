"""Configure & Run foundation — the shared ParamSpec→form builder and the standard
analysis configure-and-run popup, driven headless (offscreen Qt)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402

from smile_msi import registry as flowmod  # noqa: E402
from smile_msi.gui import common  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_form_from_params_defaults_round_trip(app):
    """A form built from a real StepDef's params returns the declared defaults."""
    sd = flowmod.REGISTRY["auto_segment"]
    rows, getter = common.form_from_params(sd.params)
    assert len(rows) == len(sd.params)
    vals = getter()
    for ps in sd.params:
        assert ps.name in vals
        if ps.kind in ("int", "float", "mz") and ps.default is not None:
            assert vals[ps.name] == pytest.approx(ps.default)
        elif ps.kind in ("bool", "choice") and ps.default is not None:
            assert vals[ps.name] == ps.default


def test_form_from_params_seed_values(app):
    """Seeded values override the declared defaults."""
    sd = flowmod.REGISTRY["auto_segment"]
    seed = {ps.name: ps.default for ps in sd.params}
    # bump the first numeric param so we can detect the override
    num = next(ps for ps in sd.params if ps.kind in ("int", "float"))
    bumped = (seed[num.name] or 0) + 1
    seed[num.name] = bumped
    _, getter = common.form_from_params(sd.params, values=seed)
    assert getter()[num.name] == pytest.approx(bumped)


def test_configure_run_dialog_values(app):
    """The dialog exposes its live values without needing to exec()."""
    sd = flowmod.REGISTRY["auto_segment"]
    dlg = common.ConfigureRunDialog(None, title="Segmentation", params=sd.params,
                                    help_text="Cluster pixels into regions.")
    vals = dlg.values()
    assert set(vals) == {ps.name for ps in sd.params}
    # accept/reject wiring exists and the run button is the default
    assert dlg.b_run.isDefault()
    dlg.accept()
    assert dlg.result() == QtWidgets.QDialog.DialogCode.Accepted


def test_form_from_params_kinds(app):
    """Each ParamSpec kind builds the expected widget type."""
    P = flowmod.ParamSpec
    params = [
        P("a_bool", "Bool", "bool", True),
        P("a_choice", "Choice", "choice", "x", choices=["x", "y"]),
        P("an_int", "Int", "int", 3, lo=0, hi=10),
        P("a_float", "Float", "float", 1.5, lo=0, hi=9, step=0.5),
    ]
    rows, getter = common.form_from_params(params)
    widgets = [field for _lbl, field in rows]
    assert isinstance(widgets[0], QtWidgets.QCheckBox)
    assert isinstance(widgets[1], QtWidgets.QComboBox)
    assert isinstance(widgets[2], QtWidgets.QSpinBox)
    assert isinstance(widgets[3], QtWidgets.QDoubleSpinBox)
    assert getter() == {"a_bool": True, "a_choice": "x", "an_int": 3, "a_float": 1.5}


# --------------------------------------------------------------------------- #
# common.guarded — the shared best-effort call wrapper that replaced the
# duplicated `def safe(fn, *a)` closures in main.py / ion.py.
# --------------------------------------------------------------------------- #
def test_guarded_returns_value_on_success_and_passes_args():
    assert common.guarded(lambda a, b: a + b, 2, 3) == 5
    assert common.guarded(lambda x, *, k: x * k, 4, k=10) == 40


def test_guarded_swallows_exception_and_returns_none():
    calls = []

    def boom():
        calls.append(1)
        raise RuntimeError("view not built yet")

    assert common.guarded(boom) is None      # exception swallowed, no raise
    assert calls == [1]                       # but the call really happened
