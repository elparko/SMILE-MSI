"""GUI tests for Export Studio's standardised intensity-window control (headless, offscreen).

The Studio dialog now exposes an 'Override intensity window' checkbox + range slider (like the
Export hub), so a batch can pin one low/high contrast window across every rendered panel and the
matrix. These tests prove the window is resolved from the design dict and actually reaches the
per-cell render spec and the matrix figure — the plumbing the audit found was previously
hardcoded to the full (0, 100) range. Skipped when the desktop deps aren't installed."""
import os
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from smile_msi import studio  # noqa: E402
from smile_msi.gui import studiodialog  # noqa: E402
from smile_msi.studio import ContrastMode, OutputKind, Section, SelectionRecord, compile_plan  # noqa: E402


class _Host(studiodialog.StudioMixin):
    """A minimal StudioMixin host: the render/matrix helpers only touch ``self.ds`` and
    ``self._studio_section_crops``, so we can drive them without a real MainWindow."""
    def __init__(self, pixel_size_um=None):
        self.ds = types.SimpleNamespace(pixel_size_um=pixel_size_um)
        self._studio_section_crops = {}


def _plan(design, contrast=ContrastMode.PER_SECTION, outputs=(OutputKind.ION_IMAGES,)):
    secs = [Section(sid="s0", name="section_00", source="/d/s0.imzML", session_path="/d/s0.json")]
    sels = [SelectionRecord(mz=744.5548, label="PC 34:1", list_name="PC", source_kind="saved")]
    return compile_plan(secs, sels, list(outputs), design=design, contrast=contrast)


# --------------------------------------------------------------------------- #
# _studio_window — the single source of truth both render paths consult
# --------------------------------------------------------------------------- #
def test_studio_window_defaults_to_full_range_when_no_override():
    assert studiodialog.StudioMixin._studio_window({}) == (0.0, 100.0)
    # a window is present but the override is off → still the full range (carried, not applied)
    assert studiodialog.StudioMixin._studio_window(
        {"window": [10.0, 80.0], "window_override": False}) == (0.0, 100.0)


def test_studio_window_uses_override_window_when_on():
    assert studiodialog.StudioMixin._studio_window(
        {"window": [12.0, 88.0], "window_override": True}) == (12.0, 88.0)
    # override on but no / malformed window recorded → safe full-range fallback (a corrupted or
    # hand-edited session must not crash the background render job)
    assert studiodialog.StudioMixin._studio_window({"window_override": True}) == (0.0, 100.0)
    assert studiodialog.StudioMixin._studio_window(
        {"window_override": True, "window": [50.0]}) == (0.0, 100.0)


# --------------------------------------------------------------------------- #
# build_spec — the per-cell render kwargs the (parallel) render phase pickles
# --------------------------------------------------------------------------- #
def test_build_spec_hardcodes_full_window_without_override():
    host = _Host()
    plan = _plan({"clip": 99.0})
    _render, build_spec = host._studio_make_render(plan, {"ppm": 5.0})
    arr = np.full((4, 4), 2.0, float)
    kw = build_spec(arr, plan.selections[0], plan.sections[0], anchor=None)["kw"]
    assert kw["low"] == 0.0 and kw["high"] == 100.0 and kw["window"] == (0.0, 100.0)


def test_build_spec_carries_the_override_window_into_every_cell():
    host = _Host()
    plan = _plan({"clip": 99.0, "window_override": True, "window": [15.0, 85.0]})
    _render, build_spec = host._studio_make_render(plan, {"ppm": 5.0})
    arr = np.full((4, 4), 2.0, float)
    kw = build_spec(arr, plan.selections[0], plan.sections[0], anchor=None)["kw"]
    assert kw["low"] == 15.0 and kw["high"] == 85.0 and kw["window"] == (15.0, 85.0)
    # the shared-mode anchor still rides alongside the window (it sets the 100% reference)
    kw2 = build_spec(arr, plan.selections[0], plan.sections[0], anchor=123.0)["kw"]
    assert kw2["window"] == (15.0, 85.0) and kw2["anchor"] == 123.0


# --------------------------------------------------------------------------- #
# matrix figure — the contact sheet honours the same standardised window
# --------------------------------------------------------------------------- #
def test_matrix_figure_honours_the_override_window(tmp_path, monkeypatch):
    host = _Host()
    plan = _plan({"clip": 99.0, "window_override": True, "window": [20.0, 90.0]},
                 outputs=(OutputKind.ION_IMAGES, OutputKind.MATRIX))
    captured = {}

    def fake_render_matrix_figure(grid, **kw):
        captured.update(kw)
        return object()

    monkeypatch.setattr(studiodialog.export, "render_matrix_figure", fake_render_matrix_figure)
    monkeypatch.setattr(studiodialog.export, "save_figure", lambda *a, **k: None)

    sec, sel = plan.sections[0], plan.selections[0]
    cells = {(sec.sid, sel.uid()): np.full((4, 4), 2.0, float)}
    host._studio_write_matrix(plan, cells, {sel.uid(): None}, str(tmp_path))
    assert captured["low"] == 20.0 and captured["high"] == 90.0


# --------------------------------------------------------------------------- #
# colour overlay per section — the previously-dead OutputKind.OVERLAY
# --------------------------------------------------------------------------- #
def _overlay_plan(design):
    secs = [Section(sid=f"s{i}", name=f"sec_{i}", source=f"/d/s{i}.imzML",
                    session_path=f"/d/s{i}.json") for i in range(2)]
    sels = [SelectionRecord(mz=744.5, label="PC 34:1", list_name="L", source_kind="saved",
                            color="#ff0000"),                       # explicit channel colour
            SelectionRecord(mz=810.5, label="PE 40:6", list_name="L", source_kind="saved")]  # → fallback hue
    return compile_plan(secs, sels, [OutputKind.OVERLAY], design=design)


def test_studio_write_overlays_one_composite_per_section(tmp_path, monkeypatch):
    host = _Host(pixel_size_um=10.0)
    plan = _overlay_plan({"clip": 99.0, "scalebar_um": 0.0,
                          "window_override": True, "window": [10.0, 80.0]})
    captured = []

    def fake_overlay(rgb, chans=None, **kw):
        captured.append((np.asarray(rgb), chans))
        return object()

    monkeypatch.setattr(studiodialog.export, "render_overlay_panel", fake_overlay)
    monkeypatch.setattr(studiodialog.export, "save_figure", lambda *a, **k: None)

    cells = {(sec.sid, sel.uid()): np.full((3, 4), 2.0, float)
             for sec in plan.sections for sel in plan.selections}
    anchors = {sel.uid(): None for sel in plan.selections}
    folder = host._studio_write_overlays(plan, cells, anchors, str(tmp_path), {"ppm": 5.0})

    assert folder is not None and os.path.isdir(folder)
    assert len(captured) == 2                                # one additive composite per section
    rgb0, chans0 = captured[0]
    assert rgb0.dtype == np.uint8 and rgb0.shape == (3, 4, 3)
    assert len(chans0) == 2                                  # both ions became channels
    assert chans0[0]["color"] == "#ff0000"                  # explicit feature colour respected
    assert chans0[1]["color"] and chans0[1]["color"] != "#ff0000"   # fallback hue assigned
    # the standardised intensity window rides into every channel's contrast bounds
    assert chans0[0]["lo"] == 10.0 and chans0[0]["hi"] == 80.0


def test_studio_write_overlays_skips_sections_with_nothing_extracted(tmp_path, monkeypatch):
    host = _Host()
    plan = _overlay_plan({"clip": 99.0})
    seen = []
    monkeypatch.setattr(studiodialog.export, "render_overlay_panel",
                        lambda rgb, chans=None, **kw: seen.append(1) or object())
    monkeypatch.setattr(studiodialog.export, "save_figure", lambda *a, **k: None)

    s0 = plan.sections[0]                                    # only the first section extracted
    cells = {(s0.sid, sel.uid()): np.full((3, 4), 1.0, float) for sel in plan.selections}
    anchors = {sel.uid(): None for sel in plan.selections}
    host._studio_write_overlays(plan, cells, anchors, str(tmp_path), {"ppm": 5.0})
    assert len(seen) == 1                                    # the empty section is silently skipped
