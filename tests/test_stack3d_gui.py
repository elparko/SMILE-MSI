"""GUI smoke test for the 3D reconstruction dialog (plan 09, headless).

Writes three small synthetic imzML sections, builds a cohort of SampleRefs over
them, and exercises ``Stack3DDialog``: populate from the cohort, register the
stack, build the ion volume, and toggle the MIP view. The real
``SectionLoader.load`` path is used (the engine streams the imzML files).
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402

from smile_msi import cohort as cohort_engine, demo  # noqa: E402
from smile_msi.gui import stack3d as s3d  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope="module")
def sections(tmp_path_factory):
    """Three small imzML sections + a probe m/z drawn from the demo peaks."""
    d = tmp_path_factory.mktemp("sections3d")
    paths = []
    for i in range(3):
        p = str(d / f"section_{i}.imzML")
        demo.write_synthetic_imzml(p, width=20, height=16, seed=i)
        paths.append(p)
    mz = float(demo._demo_peaks()[0][0])           # a real demo ion (sulfatide)
    return paths, mz


class _FakeMain(QtWidgets.QMainWindow):
    def __init__(self, paths, mz):
        super().__init__()
        self.cohort = cohort_engine.Cohort(name="Test cohort")
        for i, p in enumerate(paths):
            self.cohort.add(cohort_engine.SampleRef(
                name=f"section {i}", session_path="", source=p, group="A"))
        self.peaks = [{"mz": mz}, {"mz": mz + 28.0}]
        self.active_mz = mz
        self.steps = []

    def record_step(self, kind, *, label="", params=None, regions=None, now=None):
        self.steps.append((kind, params))
        return {}


def test_stack3d_dialog_populates(app, sections):
    paths, mz = sections
    main = _FakeMain(paths, mz)
    dlg = s3d.Stack3DDialog(main)
    dlg.load_from_main()
    assert dlg.order_list.count() == 3
    assert len(dlg._refs) == 3
    assert dlg.channel.count() == 2
    assert dlg.anchor.count() == 3                 # (TIC) + two peaks
    assert dlg.channel.currentText() != ""         # active_mz seeded the ion box


def test_stack3d_reorder_swaps_refs(app, sections):
    paths, mz = sections
    main = _FakeMain(paths, mz)
    dlg = s3d.Stack3DDialog(main)
    dlg.load_from_main()
    first, second = dlg._refs[0], dlg._refs[1]
    dlg.order_list.setCurrentRow(0)
    dlg._move_down()
    assert dlg._refs[0] is second
    assert dlg._refs[1] is first
    assert dlg._stack is None                       # ordering change invalidates registration


def test_stack3d_register_and_build_volume(app, sections):
    paths, mz = sections
    main = _FakeMain(paths, mz)
    dlg = s3d.Stack3DDialog(main)
    dlg.load_from_main()
    dlg.channel.setEditText(f"{mz:.4f}")

    dlg._register()
    assert dlg._stack is not None
    assert all(s.transform is not None for s in dlg._stack.sections)

    dlg._build()
    assert dlg._volume is not None
    Z, H, W = dlg._volume.shape
    assert Z == 3
    assert dlg.z_slider.maximum() == 2
    assert dlg.b_gl.isEnabled()
    # build records a volume3d audit step with the documented keys
    assert main.steps and main.steps[-1][0] == "volume3d"
    params = main.steps[-1][1]
    for key in ("section_keys", "spacing_um", "reference", "mode", "channel_mz"):
        assert key in params

    # MIP toggle swaps the displayed projection without error
    dlg.mip_toggle.setChecked(True)
    assert dlg._mip is True
    dlg.mip_toggle.setChecked(False)


def test_stack3d_build_without_ion_is_graceful(app, sections):
    paths, mz = sections
    main = _FakeMain(paths, mz)
    dlg = s3d.Stack3DDialog(main)
    dlg.load_from_main()
    dlg.channel.setEditText("")                     # no ion picked
    dlg._build()
    assert dlg._volume is None
    assert "ion m/z" in dlg.status.text().lower()


def test_stack3d_gl_render_without_extra_is_graceful(app, sections, monkeypatch):
    paths, mz = sections
    main = _FakeMain(paths, mz)
    dlg = s3d.Stack3DDialog(main)
    dlg.load_from_main()
    dlg.channel.setEditText(f"{mz:.4f}")
    dlg._build()

    from smile_msi import volume3d

    def _raise(*a, **k):
        raise ImportError("Interactive 3D volume rendering needs the 'viz3d' extra")

    monkeypatch.setattr(volume3d, "gl_volume_item", _raise)
    dlg._gl_render()                                # must not crash
    assert "viz3d" in dlg.status.text()
