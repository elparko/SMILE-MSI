"""Tests for the export engine (smile_msi.export) — the corner-overlay ion panel,
multi-format figure saving, table/spectra writers, segmentation maps, and the PDF book.

Pure/headless: matplotlib uses the object-oriented (Agg/PDF) API, so no display or
``pyplot`` global state is needed.
"""
import os

import numpy as np
import pandas as pd
import pytest

from smile_msi import export, imaging, provenance


def test_relative_max_is_pct_of_clip_value():
    """Peak intensity is reported as a % of the clip-percentile value, so a hotspot
    above the clip reads over 100%."""
    img = np.ones((10, 10))           # bulk at 1.0; the 99th pct stays ~1.0
    img[0, 0] = 1.58                  # a single hotspot at 1.58x the clip
    assert imaging.relative_max(img, high=99.0) == pytest.approx(158.0, abs=1.0)
    # off-tissue NaNs are ignored; an all-NaN / empty image is safe
    assert imaging.relative_max(np.full((4, 4), np.nan)) == 0.0


@pytest.fixture
def ion():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:40, 0:56]
    img = np.exp(-(((xx - 30) / 12) ** 2 + ((yy - 20) / 9) ** 2)) * 5 + rng.random((40, 56)) * 0.3
    axis = np.linspace(400, 900, 240)
    spec = np.exp(-((axis - 700) / 12) ** 2) * 8 + np.exp(-((axis - 560) / 8) ** 2) * 3
    return img, axis, spec


def _nonempty(path):
    return os.path.exists(path) and os.path.getsize(path) > 20


# --------------------------------------------------------------------------- #
# ion panel + corner overlay
# --------------------------------------------------------------------------- #
def test_render_ion_panel_all_formats(ion, tmp_path):
    img, axis, spec = ion
    for fmt in ("png", "tiff", "jpg", "pdf", "svg"):
        fig = export.render_ion_panel(img, mz=700.0, label="PE 36:1 [M-H]-", cmap="inferno",
                                      low=0, high=99, window=(0, 99), mean_spectrum=(axis, spec),
                                      pixel_size_um=40, scale_bar_um=500, dpi=90)
        out = tmp_path / f"ion.{fmt}"
        export.save_figure(fig, str(out), dpi=90)
        assert _nonempty(out), fmt


def test_render_cell_worker_renders_from_spec(ion, tmp_path):
    """export.render_cell — the process-pool worker for Export Studio's parallel render phase:
    a picklable spec (render kwargs minus image, + dpi/fmt) + the image array → a saved panel."""
    img, _axis, _spec = ion
    out = tmp_path / "cell.png"
    cell_spec = {"kw": dict(mz=700.0, label="PE 36:1", cmap="inferno", low=0, high=99,
                            window=(0, 99)),
                 "dpi": 80, "fmt": "png"}
    rv = export.render_cell(img, cell_spec, str(out))
    assert _nonempty(out)
    assert rv == str(out)


def test_save_figure_guarantees_extension(ion, tmp_path):
    """A filename whose trailing '.xxx' isn't a real image format — e.g. an ion label like
    '947.546PG46', where splitext misreads the m/z's decimal as the extension — still lands on
    disk as a proper .png, not a '.546pg46'-typed file. Real extensions are left untouched."""
    img, axis, spec = ion

    # dotted, extension-less label → the real extension is appended, and it's truly PNG bytes
    fig = export.render_ion_panel(img, mz=947.546, label="PG46", dpi=70)
    out = export.save_figure(fig, str(tmp_path / "947.546PG46"), dpi=70)
    assert out.endswith(".png") and _nonempty(out)
    assert not os.path.exists(str(tmp_path / "947.546PG46"))     # no bogus extension-less file
    with open(out, "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"                # genuine PNG signature

    # a correct extension is preserved verbatim (no "plain.tif.png" double-extension)
    for ext in ("png", "tif", "jpg", "pdf"):
        fig = export.render_ion_panel(img, mz=700.0, label="x", dpi=70)
        out = export.save_figure(fig, str(tmp_path / f"plain.{ext}"), dpi=70)
        assert out == str(tmp_path / f"plain.{ext}") and _nonempty(out)


def test_render_ion_panel_corners_and_themes(ion, tmp_path):
    img, axis, spec = ion
    for theme in ("dark", "light"):
        for corner in ("lower left", "lower right", "upper left", "upper right"):
            fig = export.render_ion_panel(img, mz=700.0, label="x", mean_spectrum=(axis, spec),
                                          theme=theme, overlay_corner=corner, dpi=80)
            out = tmp_path / f"{theme}_{corner.replace(' ', '_')}.png"
            export.save_figure(fig, str(out), dpi=80)
            assert _nonempty(out)


def test_ion_panel_overlay_toggles(ion, tmp_path):
    """Turning overlay elements off actually *removes* them (the side spectrum panel,
    colorbar/footer and scale-bar axes are gone) -- not merely 'renders without crashing'."""
    img, axis, spec = ion
    full = export.render_ion_panel(img, mz=700.0, mean_spectrum=(axis, spec), dpi=80)
    bare = export.render_ion_panel(img, mz=700.0, mean_spectrum=(axis, spec), show_spectrum=False,
                                   show_colorbar=False, show_title=False, show_scalebar=False, dpi=80)
    # the full panel reserves extra axes for the spectrum margin + footer legend/scale bar;
    # the bare panel collapses to essentially just the image axis.
    assert len(bare.axes) < len(full.axes)
    export.save_figure(full, str(tmp_path / "full.png"), dpi=80)
    export.save_figure(bare, str(tmp_path / "bare.png"), dpi=80)
    assert _nonempty(tmp_path / "full.png")
    assert _nonempty(tmp_path / "bare.png")


def test_render_ion_panel_from_rgb(ion, tmp_path):
    img, axis, spec = ion
    rgb = (np.dstack([img / img.max(), img * 0, img / img.max()]) * 255).astype(np.uint8)
    channels = [
        dict(mz=700.0, color="#DD8452", ppm=10.0, lo=0.0, hi=99.0, label="PE 36:1"),
        dict(mz=800.0, color="#4C72B0", ppm=10.0, lo=20.0, hi=90.0),
    ]
    # an ROI outline traced over the composite
    outline = np.zeros(img.shape, bool)
    outline[5:15, 5:15] = True
    fig = export.render_overlay_panel(rgb, channels, outlines=[(outline, "#ff3b30")],
                                      pixel_size_um=20.0, scale_bar_um=200.0, dpi=80)
    out = tmp_path / "overlay.png"
    export.save_figure(fig, str(out), dpi=80)
    assert _nonempty(out)


def test_render_overlay_legacy_tuples_and_toggles(ion, tmp_path):
    """Legacy (label, color) entries still render, and turning the legend off actually drops
    the legend/footer axes (not just 'no crash')."""
    img, _axis, _spec = ion
    rgb = (np.dstack([img / img.max(), img * 0, img * 0]) * 255).astype(np.uint8)
    entry = [("m/z 700.0123  PE 36:1", "#DD8452")]
    legged = export.render_overlay_panel(rgb, entry, show_legend=True, show_scalebar=False, dpi=80)
    bare = export.render_overlay_panel(rgb, entry, show_legend=False, show_scalebar=False, dpi=80)
    # the legend-on panel adds the footer band + channel-legend axes; legend-off drops them.
    assert len(bare.axes) < len(legged.axes)
    out = tmp_path / "overlay_bare.png"
    export.save_figure(bare, str(out), dpi=80)
    assert _nonempty(out)


def test_ion_panel_handles_empty_image(tmp_path):
    fig = export.render_ion_panel(np.zeros((20, 20)), mz=500.0, mean_spectrum=None, dpi=70)
    out = tmp_path / "zeros.png"
    export.save_figure(fig, str(out), dpi=70)
    assert _nonempty(out)


def test_export_orientation_matches_viewer():
    """The app forces ``imageAxisOrder == 'row-major'`` (gui/common.py), so the pyqtgraph
    ImageView displays a ``(rows, cols)`` image the same way matplotlib ``imshow`` does. The
    export must therefore **preserve** orientation — no transpose — or a wide tissue comes out
    tall (and vice-versa) relative to what the user sees and aligns to. Lock that contract here
    (regression: an earlier ``_view_orient`` wrongly swapped axes, flipping every export)."""
    img = np.zeros((3, 5), float)          # asymmetric: 3 rows × 5 cols
    img[0, 4] = 1.0                        # bright marker at (row 0, col 4)
    fig = export.render_ion_panel(img, mz=700.0, overlay=False, show_colorbar=False,
                                  show_scalebar=False, show_title=False, dpi=70)
    disp = np.asarray(fig.axes[0].images[0].get_array())
    assert disp.shape[:2] == (3, 5)        # preserved: rows×cols, matching the row-major viewer
    # the marker at (row 0, col 4) stays at display index (row 0, col 4)
    bright = np.unravel_index(int(np.argmax(disp[..., :3].sum(axis=2))), disp.shape[:2])
    assert bright == (0, 4)
    # the helper is an identity pass-through (preserves a trailing colour-channel axis)
    assert export._view_orient(np.zeros((3, 5, 4))).shape == (3, 5, 4)


def test_ion_panel_mismatched_spectrum_does_not_crash(tmp_path):
    """A malformed mean spectrum (m/z axis and intensity of unequal length — seen on a dense
    dataset whose shared axis and matrix width disagree) must not crash the export; the
    spectrum is trimmed to the shorter side."""
    img = np.random.default_rng(0).random((24, 32))
    axis = np.linspace(400, 900, 600)
    spec = np.linspace(0, 5, 2000)                       # deliberately longer than the axis
    fig = export.render_ion_panel(img, mz=700.0, label="x", mean_spectrum=(axis, spec), dpi=70)
    out = tmp_path / "mismatch.png"
    export.save_figure(fig, str(out), dpi=70)
    assert _nonempty(out)
    x, y = export._clip_xy(axis, spec)
    assert len(x) == len(y) == 600


# --------------------------------------------------------------------------- #
# spectra figure + accents
# --------------------------------------------------------------------------- #
def test_render_spectrum_figure(ion, tmp_path):
    _img, axis, spec = ion
    fig = export.render_spectrum_figure([("mean", axis, spec), ("roi", axis, spec * 0.6)],
                                        peaks=[560, 700], active_mz=700, dpi=80)
    out = tmp_path / "spectrum.pdf"
    export.save_figure(fig, str(out), dpi=80)
    assert _nonempty(out)


def test_render_shap_figure(tmp_path):
    """The SHAP bubble-plot export renders and saves — guards the colormap lookup, which
    broke when matplotlib 3.9 removed cm.get_cmap (now colormaps['bwr'])."""
    rng = np.random.default_rng(0)
    importance = rng.random((3, 6))                     # (n_cols, n_rows)
    direction = rng.uniform(-1.0, 1.0, (3, 6))
    fig = export.render_shap_figure(
        importance, direction,
        col_labels=["region A", "region B", "region C"],
        row_mz=[760.5, 788.6, 810.6, 834.6, 885.5, 904.6], dpi=80)
    out = tmp_path / "shap.png"
    export.save_figure(fig, str(out), dpi=80)
    assert _nonempty(out)


def test_render_shap_figure_widens_for_long_labels():
    """Long auto-annotated ion labels (m/z + lipid) must not spill past the frame: the left
    gutter and overall canvas grow with the longest row label."""
    rng = np.random.default_rng(0)
    imp = rng.random((2, 4))
    drc = rng.uniform(-1.0, 1.0, (2, 4))
    short = export.render_shap_figure(imp, drc, col_labels=["A", "B"],
                                      row_mz=[700.0, 800.0, 900.0, 1000.0], dpi=80)
    longl = export.render_shap_figure(
        imp, drc, col_labels=["A", "B"], row_mz=[700.0, 800.0, 900.0, 1000.0],
        row_labels=["812.5400  PC 36:1 a fairly long lipid name"] * 4, dpi=80)
    assert longl.get_size_inches()[0] > short.get_size_inches()[0]   # wider canvas
    # the plotting axes starts further right (bigger left gutter) to hold the labels
    assert longl.axes[0].get_position().x0 > short.axes[0].get_position().x0


def test_render_shap_figure_caps_bubbles_to_cell():
    """On a dense plot the largest bubble must fit within one grid cell (no overlap on
    export); every bubble scales down from there."""
    rng = np.random.default_rng(2)
    n_rows, n_cols = 30, 3
    imp = rng.random((n_cols, n_rows))
    fig = export.render_shap_figure(imp, -imp, col_labels=["A", "B", "C"],
                                    row_mz=list(range(700, 700 + n_rows)), dpi=80)
    ax = fig.axes[0]
    sizes = np.asarray(ax.collections[0].get_sizes())     # marker area in pt²
    max_diam_pt = float(np.sqrt(sizes.max()))             # circle diameter ≈ sqrt(area)
    bbox = ax.get_position()
    _fw, fh = fig.get_size_inches()
    cell_h_pt = (bbox.height * fh * 72.0) / (n_rows + 0.6)
    assert max_diam_pt <= cell_h_pt + 0.5                 # biggest bubble fits a row cell


def test_render_shap_figure_custom_cmap():
    """A custom colormap renders; an unknown name falls back to bwr rather than raising."""
    rng = np.random.default_rng(1)
    imp = rng.random((2, 3))
    assert export.render_shap_figure(imp, -imp, col_labels=["A", "B"], row_mz=[1, 2, 3],
                                     cmap="coolwarm", dpi=70) is not None
    assert export.render_shap_figure(imp, -imp, col_labels=["A", "B"], row_mz=[1, 2, 3],
                                     cmap="not-a-cmap", dpi=70) is not None


def test_render_shap_histogram_figure(tmp_path):
    """The per-category biomarker bar chart (after Farrow Fig. S153) renders + saves: vertical
    bars (species on the X axis), error bars, a direction colour bar; the canvas is taller for
    long species labels (rotated on the X axis → bottom gutter) and wider for more species."""
    rng = np.random.default_rng(0)
    mz = [760.5, 788.6, 810.6, 834.6, 885.5]
    mean = rng.random(5)
    std = rng.random(5) * 0.1
    direction = rng.uniform(-1.0, 1.0, 5)
    fig = export.render_shap_histogram_figure(
        mz, mean, importance_std=std, direction=direction, dpi=80,
        title="SHAP biomarker importance · cortex")
    assert len(fig.axes) == 2                                # bars + colour bar
    # vertical bars: species on the X axis (one tick per species), importance on Y from 0
    ax = fig.axes[0]
    assert len(ax.patches) == 5 and len(ax.get_xticks()) == 5
    assert ax.get_ylim()[0] == 0
    out = tmp_path / "shap_hist.png"
    export.save_figure(fig, str(out), dpi=80)
    assert _nonempty(out)
    longl = export.render_shap_histogram_figure(
        mz, mean, importance_std=std, direction=direction, dpi=80,
        row_labels=["812.5400  PC 36:1 a fairly long lipid name"] * 5)
    assert longl.get_size_inches()[1] > fig.get_size_inches()[1]   # taller for long X labels
    more = export.render_shap_histogram_figure(
        list(mz) * 4, list(mean) * 4, importance_std=list(std) * 4,
        direction=list(direction) * 4, dpi=80)
    assert more.get_size_inches()[0] > fig.get_size_inches()[0]    # wider for more species


def test_render_shap_histogram_figure_no_direction():
    """Without a direction series the histogram still renders — just no colour bar."""
    fig = export.render_shap_histogram_figure([700.0, 800.0], [0.4, 0.2], dpi=70)
    assert len(fig.axes) == 1                                # bars only, no colour bar


def test_render_shap_histogram_grid(tmp_path):
    """The 'all regions' multi-panel export renders one panel per region plus a single shared
    direction colour bar, and saves."""
    rng = np.random.default_rng(2)
    items = []
    for name in ("cortex", "medulla", "glomeruli"):
        mz = [700.0 + 10 * i for i in range(4)]
        items.append({"title": name, "mz": mz, "mean": rng.random(4),
                      "std": rng.random(4) * 0.1, "direction": rng.uniform(-1, 1, 4),
                      "row_labels": [f"{m:.2f}  PC {30+i}:1" for i, m in enumerate(mz)]})
    fig = export.render_shap_histogram_grid(items, suptitle="SHAP importance", dpi=80)
    # panel titles are left-aligned (DESIGN_SPEC §0), so read the "left" location
    titles = {a.get_title(loc="left") for a in fig.axes
              if a.get_visible() and a.get_title(loc="left")}
    assert {"cortex", "medulla", "glomeruli"} <= titles      # one titled panel per region
    out = tmp_path / "grid.png"
    export.save_figure(fig, str(out), dpi=80)
    assert _nonempty(out)
    # empty input is tolerated (renders an empty grid rather than raising)
    assert export.render_shap_histogram_grid([], dpi=70) is not None


def test_signal_mz_range_crops_empty_tail():
    from smile_msi.spectrum_range import signal_mz_range
    axis = np.linspace(200.0, 2300.0, 4000)
    spec = np.zeros_like(axis)
    for mz, h in [(382.0, 5e4), (560.0, 8e3), (1040.0, 6e3)]:   # signal only 382..1040
        spec += h * np.exp(-0.5 * ((axis - mz) / 0.4) ** 2)
    lo, hi = signal_mz_range(axis, spec)
    assert 330.0 < lo < 382.0          # opens just left of the first peak
    assert 1040.0 < hi < 1200.0        # crops the long empty acquired tail past the last peak
    # signed difference / multi-spectrum: spans where *either* side has signal
    lo2, hi2 = signal_mz_range(axis, spec, -np.flip(spec))
    assert lo2 < lo and hi2 > hi
    # degenerate inputs fall back to None so callers keep the full axis
    assert signal_mz_range(axis, np.zeros_like(axis)) is None
    assert signal_mz_range(np.array([]), np.array([])) is None


def test_render_spectrum_figure_auto_crops():
    axis = np.linspace(200.0, 2300.0, 3000)
    spec = 5e4 * np.exp(-0.5 * ((axis - 600.0) / 0.5) ** 2)    # one peak, long empty tail
    fig = export.render_spectrum_figure([("mean", axis, spec)], dpi=80)
    assert fig.axes[0].get_xlim()[1] < 1200.0                  # tail to 2300 cropped off
    explicit = export.render_spectrum_figure([("mean", axis, spec)], mz_range=(200, 2300), dpi=80)
    assert explicit.axes[0].get_xlim()[1] > 2000.0            # an explicit range still wins


def test_render_spectrum_figure_stops_past_last_peak():
    axis = np.linspace(200.0, 2300.0, 3000)
    spec = np.zeros_like(axis)
    for mz in (400.0, 600.0, 760.0):                          # last real peak at 760
        spec += 5e4 * np.exp(-0.5 * ((axis - mz) / 0.4) ** 2)
    peaks = [400.0, 600.0, 760.0]
    fig = export.render_spectrum_figure([("mean", axis, spec)], peaks=peaks, pad_right=50.0, dpi=80)
    hi = fig.axes[0].get_xlim()[1]
    assert 805.0 <= hi <= 815.0                               # ≈ last peak (760) + 50, tail dropped
    # pad past the acquired axis is clamped to the data extent
    clamped = export._spectrum_xlim([("mean", axis, spec)], [2295.0], pad_right=50.0)
    assert clamped[1] <= float(axis.max())
    # the y-axis keeps headroom above the tallest peak (not flush to the frame)
    ymax = float(np.nanmax(spec))
    assert fig.axes[0].get_ylim()[1] > ymax * 1.2


def test_render_spectrum_figure_peak_markers_below_axis():
    """Peak ticks render as upward triangles just below the x-axis (axes-fraction y < 0,
    unclipped), and pass ``peaks=None`` to omit them entirely."""
    axis = np.linspace(600.0, 900.0, 400)
    spec = np.abs(np.sin(axis / 10.0))
    fig = export.render_spectrum_figure([("mean", axis, spec)], peaks=[650.0, 720.0, 810.0], dpi=80)
    marks = [ln for ln in fig.axes[0].get_lines() if ln.get_marker() == "^"]
    assert len(marks) == 1
    assert float(marks[0].get_ydata()[0]) < 0.0          # below the axis, in axes-fraction coords
    assert marks[0].get_clip_on() is False               # allowed to sit in the bottom margin

    fig2 = export.render_spectrum_figure([("mean", axis, spec)], peaks=None, dpi=80)
    assert not [ln for ln in fig2.axes[0].get_lines() if ln.get_marker() == "^"]


def test_render_spectrum_figure_stick_style():
    """``draw_style='sticks'`` renders centroid-style impulses (a vlines LineCollection) from
    the baseline instead of a connected profile trace."""
    from matplotlib.collections import LineCollection
    axis = np.linspace(600.0, 900.0, 500)
    spec = np.abs(np.sin(axis / 7.0))
    fig = export.render_spectrum_figure([("mean", axis, spec)], draw_style="sticks", dpi=80)
    ax = fig.axes[0]
    assert any(isinstance(c, LineCollection) for c in ax.collections)
    assert not ax.get_lines()                       # no connected Line2D trace in stick mode
    # the default profile style still draws a connected line
    line_fig = export.render_spectrum_figure([("mean", axis, spec)], dpi=80)
    assert line_fig.axes[0].get_lines()


def test_render_spectrum_figure_difference_direction_labels():
    """A signed difference with ``diff_labels`` gets a zero baseline, ``↑/↓ higher in`` text for
    each region, and a ``Δ intensity`` y-label so the sign is unambiguous."""
    axis = np.linspace(600.0, 900.0, 400)
    diff = np.sin(axis / 5.0) * 1000.0              # crosses zero (signed)
    fig = export.render_spectrum_figure([("Endo − Peri", axis, diff, "#e15759")],
                                        diff_labels=("Endo", "Peri"), dpi=80)
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    assert any("higher in Endo" in t for t in texts)
    assert any("higher in Peri" in t for t in texts)
    assert ax.get_ylabel() == "Δ intensity"
    assert any(list(ln.get_ydata()) == [0.0, 0.0] for ln in ax.get_lines())   # zero baseline
    # a strictly-positive difference (region A higher everywhere) STILL gets the labels +
    # baseline — the annotation follows the declared diff, not the data sign (regression guard)
    pos = export.render_spectrum_figure([("A − B", axis, np.abs(diff) + 5.0, "#e15759")],
                                        diff_labels=("Endo", "Peri"), dpi=80)
    assert any("higher in Endo" in t.get_text() for t in pos.axes[0].texts)
    assert pos.axes[0].get_ylabel() == "Δ intensity"
    assert any(list(ln.get_ydata()) == [0.0, 0.0] for ln in pos.axes[0].get_lines())
    # a non-signed spectrum (or no labels) gets no direction text
    plain = export.render_spectrum_figure([("mean", axis, np.abs(diff))], dpi=80)
    assert not any("higher in" in t.get_text() for t in plain.axes[0].texts)


def test_spectra_ylim_shared_scale():
    """``spectra_ylim`` returns one (lo, hi) covering every spectrum so separate exports share a
    comparable scale; passing it as ``ylim`` forces that exact range."""
    axis = np.linspace(600.0, 900.0, 200)
    base = np.abs(np.sin(axis / 5.0))
    lo, hi = export.spectra_ylim([("tall", axis, base * 10.0), ("short", axis, base * 2.0)])
    assert lo == 0.0 and hi > 10.0                  # spans the taller spectrum
    fig = export.render_spectrum_figure([("short", axis, base * 2.0)], ylim=(lo, hi), dpi=80)
    assert fig.axes[0].get_ylim() == (lo, hi)
    lo2, _ = export.spectra_ylim([("signed", axis, base * 10.0 - 5.0)])
    assert lo2 < 0.0                                # signed differences drop below zero


def test_accent_for_cmap():
    for cmap in ("viridis", "inferno", "Magma", "not-a-real-cmap"):
        acc = export.accent_for_cmap(cmap)
        assert isinstance(acc, str) and acc.startswith("#") and len(acc) == 7


# --------------------------------------------------------------------------- #
# table + spectra writers
# --------------------------------------------------------------------------- #
def test_write_table_formats(tmp_path):
    df = pd.DataFrame({"mz": [700.1, 800.2], "lipid": ["PE 36:1", "PC 34:1"], "AUC": [0.8, 0.2]})
    for fmt in ("csv", "tsv", "xlsx", "json", "md"):
        out = tmp_path / f"t.{fmt}"
        export.write_table(df, str(out), fmt=fmt)
        assert _nonempty(out)
    # CSV is round-trippable
    back = pd.read_csv(tmp_path / "t.csv")
    assert list(back["lipid"]) == ["PE 36:1", "PC 34:1"]
    # Markdown is GitHub-flavoured and dependency-free
    assert "| mz | lipid | AUC |" in (tmp_path / "t.md").read_text()


def test_write_spectra(ion, tmp_path):
    _img, axis, spec = ion
    # single → m/z,intensity
    export.write_spectra([("mean", axis, spec)], str(tmp_path / "s1.csv"))
    one = pd.read_csv(tmp_path / "s1.csv")
    assert list(one.columns) == ["m/z", "intensity"] and len(one) == len(axis)
    # multiple on a shared axis → wide
    export.write_spectra([("mean", axis, spec), ("roi", axis, spec * 0.5)], str(tmp_path / "s2.csv"))
    wide = pd.read_csv(tmp_path / "s2.csv")
    assert {"m/z", "mean", "roi"} <= set(wide.columns)
    # mismatched axes → long form
    export.write_spectra([("a", axis, spec), ("b", axis[:100], spec[:100])], str(tmp_path / "s3.csv"))
    long = pd.read_csv(tmp_path / "s3.csv")
    assert {"spectrum", "m/z", "intensity"} <= set(long.columns)


# --------------------------------------------------------------------------- #
# segmentation map
# --------------------------------------------------------------------------- #
def test_segmentation(tmp_path):
    label = np.full((30, 40), np.nan)
    label[2:12, 2:18] = 0
    label[14:28, 6:34] = 1
    label[2:10, 22:38] = 2
    colors = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868"}
    rgba = export.segmentation_rgba(label, colors)
    assert rgba.shape == (30, 40, 4) and rgba[..., 3].max() == 255
    fig = export.render_segmentation_figure(label, colors,
                                            legend=[("#4C72B0", "epi"), ("#DD8452", "fascicle")],
                                            dpi=80)
    out = tmp_path / "seg.png"
    export.save_figure(fig, str(out), dpi=80)
    assert _nonempty(out)


# --------------------------------------------------------------------------- #
# the PDF book
# --------------------------------------------------------------------------- #
def _is_pdf(path):
    with open(path, "rb") as f:
        return f.read(4) == b"%PDF"


def _pdf_page_count(path):
    data = open(path, "rb").read()
    return data.count(b"/Type /Page") - data.count(b"/Type /Pages")


def test_build_book_full(ion, tmp_path):
    img, axis, spec = ion
    prov = provenance.Provenance(title="Test")
    prov.dataset = {"pixels": 2240, "width": 56, "height": 40, "mz_min": 400.0,
                    "mz_max": 900.0, "polarity": "negative"}
    prov.step("peak_picking", snr=3, norm="tic")
    prov.step("segmentation", method="kmeans", n_clusters=3, auto=True, spatial=True)
    prov.step("roi_comparison")
    label = np.full((40, 56), np.nan)
    label[2:18, 2:26] = 0
    label[20:38, 8:50] = 1
    stats = pd.DataFrame({"mz": [700.1, 800.2, 650.0], "best_lipid": ["PE 36:1", "PC 34:1", "LPA 18:1"],
                          "AUC": [0.82, 0.18, 0.71], "q_value": [1e-3, 2e-4, 3e-3],
                          "log2_fc": [1.07, -1.0, 0.85]})
    doc = {
        "title": "Test report", "subtitle": "Unit test", "cmap": "inferno",
        "meta": {"Dataset": "demo.imzML", "Pixels": "2,240"},
        "dataset_lines": ["2,240 pixels", "m/z 400–900", "negative ion mode"],
        "provenance": prov,
        "panels": [dict(image=img, mz=700.0, label="PE 36:1", cmap="inferno", low=0, high=99,
                        window=(0, 99), mean_spectrum=(axis, spec)),
                   dict(image=img * 0.6, mz=800.2, label="PC 34:1", mean_spectrum=(axis, spec))],
        "spectra": [("mean spectrum", axis, spec), ("skyline", axis, spec * 1.2)],
        "segmentation": {"image": label, "colors": {0: "#4C72B0", 1: "#DD8452"},
                         "legend": [("#4C72B0", "A"), ("#DD8452", "B")], "title": "Segmentation"},
        "stats": {"df": stats, "a_label": "Normal", "b_label": "Synkinetic", "title": "Discriminating"},
    }
    out = tmp_path / "book.pdf"
    export.build_book(doc, str(out), theme="dark", dpi=80)
    assert _is_pdf(out)
    assert _pdf_page_count(out) >= 6          # cover + summary + methods + gallery + spectra + seg + stats


def test_roc_report_pdf(tmp_path):
    """The ROC report writes a valid multi-page PDF: a volcano overview page plus one
    grid page per six ROC curves."""
    from smile_msi import spatial
    rng = np.random.default_rng(2)
    stats = pd.DataFrame({"mz": [700.1, 800.2, 650.0, 712.5],
                          "AUC": [0.82, 0.18, 0.71, 0.55],
                          "q_value": [1e-3, 2e-4, 3e-3, 0.2],
                          "log2_fc": [1.07, -1.0, 0.85, 0.14],
                          "best_lipid": ["PE 36:1", "PC 34:1", "LPA 18:1", "?"]})
    curves = []
    for _, r in stats.iterrows():
        fpr, tpr, auc = spatial.roc_curve(rng.normal(0, 1, 60), rng.normal(0.6, 1, 70))
        curves.append({"mz": float(r["mz"]), "lipid": r["best_lipid"], "fpr": fpr, "tpr": tpr,
                       "auc": auc, "q": float(r["q_value"]), "log2_fc": float(r["log2_fc"]),
                       "higher": "Synkinetic" if auc >= 0.5 else "Normal"})
    out = tmp_path / "roc.pdf"
    export.roc_report_pdf(str(out), stats, curves, a_label="Normal", b_label="Synkinetic",
                          note="60 vs 70 pixels", theme="dark", dpi=80)
    assert _is_pdf(out)
    assert _pdf_page_count(out) >= 2           # overview + at least one ROC grid page


def test_build_book_minimal(tmp_path):
    """A near-empty document (title only) still produces a valid one-page PDF."""
    out = tmp_path / "min.pdf"
    export.build_book({"title": "Bare"}, str(out), dpi=70)
    assert _is_pdf(out)


def test_build_book_section_toggle(ion, tmp_path):
    """Sections can be switched off."""
    img, axis, spec = ion
    doc = {"title": "Partial", "dataset_lines": ["a", "b"],
           "panels": [dict(image=img, mz=700.0, mean_spectrum=(axis, spec))],
           "sections": {"summary": False, "methods": False, "gallery": True,
                        "spectra": False, "segmentation": False, "stats": False}}
    out = tmp_path / "partial.pdf"
    export.build_book(doc, str(out), dpi=70)
    assert _is_pdf(out)


def test_build_book_from_curated_items(ion, tmp_path):
    """When the document carries an ordered ``items`` list (the curated Report tab), the
    book body is rendered from it in order — one path per item type, with overlay panels,
    captions, a feature table and a free-text note."""
    img, axis, spec = ion
    rgb = np.stack([img, img * 0.4, img * 0.1], axis=-1)
    rgb = (rgb / rgb.max() * 255).astype(np.uint8)
    label = np.full((40, 56), np.nan)
    label[2:18, 2:26] = 0
    label[20:38, 8:50] = 1
    stats = pd.DataFrame({"mz": [700.1, 800.2], "best_lipid": ["PE 36:1", "PC 34:1"],
                          "AUC": [0.82, 0.18], "q_value": [1e-3, 2e-4], "log2_fc": [1.07, -1.0]})
    feats = [{"mz": 700.1, "lipid": "PE 36:1", "ppm": 1.2, "snr": 12.0},
             {"mz": 800.2, "lipid": "PC 34:1", "ppm": 0.8, "snr": 9.0}]
    doc = {
        "title": "Curated report", "dataset_lines": ["2,240 pixels", "m/z 400–900"],
        "items": [
            {"type": "note", "title": "Overview", "text": "A curated report."},
            {"type": "ion", "title": "PE 36:1", "caption": "Enriched in the lesion.",
             "panel": dict(image=img, mz=700.0, label="PE 36:1", low=0, high=99,
                           window=(0, 99), mean_spectrum=(axis, spec))},
            {"type": "ion", "title": "PC 34:1",
             "panel": dict(image=img * 0.6, mz=800.2, label="PC 34:1", mean_spectrum=(axis, spec))},
            {"type": "overlay", "title": "Two-ion overlay", "caption": "Red/green composite.",
             "panel": {"kind": "overlay", "rgb": rgb,
                       "channels": [{"mz": 700.0, "color": "#ff3b30", "label": "PE"},
                                    {"mz": 800.2, "color": "#30d158", "label": "PC"}]}},
            {"type": "spectrum", "title": "Mean spectrum",
             "spectra": [("mean spectrum", axis, spec)]},
            {"type": "segmentation", "title": "Segments", "caption": "k=2.",
             "segmentation": {"image": label, "colors": {0: "#4C72B0", 1: "#DD8452"},
                              "legend": [("#4C72B0", "A"), ("#DD8452", "B")]}},
            {"type": "stats", "title": "A vs B", "caption": "Rank AUC.",
             "stats": {"df": stats, "a_label": "A", "b_label": "B"}},
            {"type": "features", "title": "Feature table", "records": feats},
        ],
    }
    out = tmp_path / "curated.pdf"
    export.build_book(doc, str(out), theme="light", dpi=80)
    assert _is_pdf(out)
    # cover + note + gallery(2 ion, 1 page) + overlay + spectrum + segmentation + stats + features
    assert _pdf_page_count(out) >= 7


# --------------------------------------------------------------------------- #
# the CSV data bundle (companion to the PDF book)
# --------------------------------------------------------------------------- #
def test_build_data_bundle(ion, tmp_path):
    """The reworked bundle is tight: SMART feature table, per-region intensities, statistics,
    one compact methods.md — and crucially NO provenance.json / README / dataset_summary, and
    raw spectra only when opted in."""
    img, axis, spec = ion
    prov = provenance.Provenance(title="Bundle")
    prov.step("peak_picking", snr=3, norm="tic")
    feat = pd.DataFrame({"mz": [700.1, 800.2], "lipid": ["PE 36:1", "PC 34:1"],
                         "adduct": ["[M-H]-", "[M+H]+"]})
    rstats = pd.DataFrame({"region": ["A", "B"], "mz": [700.1, 700.1], "lipid": ["PE 36:1"] * 2,
                           "n_px": [120, 80], "mean": [10.0, 4.0], "median": [9.0, 3.5],
                           "std": [2.0, 1.0]})
    stats = pd.DataFrame({"mz": [700.1], "AUC": [0.8], "q_value": [1e-3]})
    doc = {
        "title": "Bundle test", "meta": {"Dataset": "demo.imzML", "Pixels": "2,240"},
        "feature_table": feat, "region_stats": rstats,
        "spectra": [("mean spectrum", axis, spec), ("skyline", axis, spec * 1.2)],
        "stats": {"df": stats, "a_label": "A", "b_label": "B"}, "provenance": prov,
    }
    folder = tmp_path / "report_data"
    export.build_data_bundle(doc, str(folder))                 # default options
    for fn in ("features.csv", "region_intensities.csv", "statistics.csv", "methods.md"):
        assert (folder / fn).exists(), fn
    for gone in ("dataset_summary.csv", "ion_images.csv", "README.txt", "provenance.json", "spectra.csv"):
        assert not (folder / gone).exists(), gone          # the bloat is gone; spectra opt-in
    assert list(pd.read_csv(folder / "features.csv")["lipid"]) == ["PE 36:1", "PC 34:1"]
    assert {"region", "mean", "median"} <= set(pd.read_csv(folder / "region_intensities.csv").columns)
    methods = (folder / "methods.md").read_text()
    assert "Files in this bundle" in methods and "demo.imzML" in methods


def test_build_data_bundle_options(ion, tmp_path):
    """Explicit options gate each file; raw spectra ship only when asked for."""
    img, axis, spec = ion
    doc = {"title": "Opt", "meta": {"Dataset": "d"},
           "feature_table": pd.DataFrame({"mz": [1.0], "lipid": ["a"]}),
           "region_stats": pd.DataFrame({"region": ["A"], "mz": [1.0], "mean": [2.0]}),
           "spectra": [("mean spectrum", axis, spec)]}
    folder = tmp_path / "b"
    export.build_data_bundle(doc, str(folder), options={
        "features": True, "region_stats": False, "statistics": False,
        "spectra": True, "methods": False})
    assert (folder / "features.csv").exists()
    assert (folder / "spectra.csv").exists()               # opted in
    assert not (folder / "region_intensities.csv").exists()
    assert not (folder / "methods.md").exists()


def test_bundle_preview():
    """The preview lists the enabled files with mock column/row samples, and tracks toggles."""
    default = export.bundle_preview()
    assert "features.csv" in default and "region_intensities.csv" in default
    assert "spectra.csv" not in default                    # opt-in, off by default
    assert "adduct" in default                             # mock columns are shown
    withspec = export.bundle_preview({"spectra": True, "region_stats": False})
    assert "spectra.csv" in withspec and "region_intensities.csv" not in withspec
    assert "Nothing selected" in export.bundle_preview(
        {k: False for k in export.DEFAULT_BUNDLE_OPTIONS})


# --------------------------------------------------------------------------- #
# ROI crop (zoom an export in on a region)
# --------------------------------------------------------------------------- #
def test_render_ion_panel_crop_and_outline(ion, tmp_path):
    img, axis, spec = ion
    mask = np.zeros(img.shape, bool)
    mask[8:30, 14:46] = True
    for dim in (False, True):
        fig = export.render_ion_panel(img, mz=700.0, label="PE 36:1", crop=(5, 33, 11, 49),
                                      outline_mask=mask, dim_outside=dim, mean_spectrum=(axis, spec),
                                      pixel_size_um=40, scale_bar_um=300, dpi=80)
        out = tmp_path / f"crop_{dim}.png"
        export.save_figure(fig, str(out), dpi=80)
        assert _nonempty(out)


def test_crop_bounds_clamp_and_reject():
    assert export._crop_bounds((5, 30, 10, 40), 50, 60) == (5, 30, 10, 40)
    assert export._crop_bounds((-5, 999, -2, 999), 40, 56) == (0, 40, 0, 56)   # clamped
    assert export._crop_bounds((20, 20, 0, 10), 40, 56) is None                # empty → None
    assert export._crop_bounds(None, 40, 56) is None


def test_overlay_and_segmentation_crop(ion, tmp_path):
    img, _axis, _spec = ion
    rgb = (np.dstack([img / img.max(), img * 0, img / img.max()]) * 255).astype(np.uint8)
    fig = export.render_overlay_panel(rgb, [dict(mz=700.0, color="#f00", ppm=5, lo=0, hi=99)],
                                      crop=(5, 30, 10, 40), dpi=80)
    export.save_figure(fig, str(tmp_path / "ov_crop.png"), dpi=80)
    assert _nonempty(tmp_path / "ov_crop.png")
    label = np.full((40, 56), np.nan); label[6:20, 6:25] = 0; label[22:38, 8:50] = 1
    fig = export.render_segmentation_figure(label, {0: "#4C72B0", 1: "#DD8452"},
                                            crop=(0, 40, 0, 40), dpi=80)
    export.save_figure(fig, str(tmp_path / "seg_crop.png"), dpi=80)
    assert _nonempty(tmp_path / "seg_crop.png")


# --------------------------------------------------------------------------- #
# audit-trail header on exported tables (CSV comment block + XLSX provenance sheet)
# --------------------------------------------------------------------------- #
def _audit_header():
    p = provenance.Provenance(title="t", started="2026-06-19T00:00:00+00:00")
    p.step("roi_comparison", test="mwu", norm="tic", tol_ppm=10)
    return p.csv_header_lines(analysis="ROI comparison",
                              settings={"norm": "tic", "tol_ppm": 10},
                              now="2026-06-19T00:00:00+00:00")


def test_write_table_csv_prepends_comment_header_and_stays_parseable(tmp_path):
    df = pd.DataFrame({"mz": [788.5447], "AUC": [0.91]})
    path = tmp_path / "out.csv"
    export.write_table(df, str(path), fmt="csv", header_lines=_audit_header())
    text = path.read_text(encoding="utf-8-sig")
    assert text.startswith("# SMILE MSI")
    assert "# analysis: ROI comparison" in text
    # a standard comment-aware read recovers exactly the data, header skipped
    back = pd.read_csv(path, comment="#")
    assert list(back.columns) == ["mz", "AUC"] and len(back) == 1
    assert back.iloc[0]["mz"] == pytest.approx(788.5447)


def test_write_table_csv_without_header_is_unchanged(tmp_path):
    df = pd.DataFrame({"mz": [1.0]})
    path = tmp_path / "plain.csv"
    export.write_table(df, str(path), fmt="csv")
    assert not path.read_text(encoding="utf-8-sig").startswith("#")


def test_write_table_xlsx_adds_provenance_sheet(tmp_path):
    df = pd.DataFrame({"mz": [1.0, 2.0]})
    path = tmp_path / "out.xlsx"
    export.write_table(df, str(path), fmt="xlsx", header_lines=_audit_header())
    sheets = pd.read_excel(path, sheet_name=None)
    assert set(sheets) == {"Data", "Provenance"}
    assert len(sheets["Data"]) == 2
    assert any("ROI comparison" in str(v)
               for v in sheets["Provenance"].iloc[:, 0].tolist())


# --------------------------------------------------------------------------- #
# Export Studio matrix (ion-row × section-column contact sheet)
# --------------------------------------------------------------------------- #
def _cell(scale):
    yy, xx = np.mgrid[0:20, 0:28]
    return np.exp(-(((xx - 16) / 6.0) ** 2 + ((yy - 10) / 4.0) ** 2)) * scale


def test_render_matrix_figure_grid_and_absent_cell(tmp_path):
    # 3 ions × 2 sections, one absent cell — must render to a non-empty file without error
    grid = [[_cell(5.0), _cell(2.0)], [_cell(3.0), None], [_cell(1.0), _cell(4.0)]]
    fig = export.render_matrix_figure(
        grid, row_labels=["PC 34:1", "PE 40:6", "ST 24:1"],
        col_labels=["synk_02", "synk_04"], cmap="viridis", row_anchors=[5.0, 3.0, 4.0],
        theme="light", dpi=100, title="ions × sections")
    p = tmp_path / "matrix.png"
    export.save_figure(fig, str(p), dpi=100)
    assert _nonempty(str(p))


def test_render_matrix_figure_empty_is_safe():
    fig = export.render_matrix_figure([], row_labels=[], col_labels=[])
    assert fig is not None                      # a "nothing to render" placeholder, not a crash


def test_apply_colormap_shared_anchor_keeps_faint_dim():
    """A shared absolute anchor must NOT re-stretch a faint section to full range — that is the
    whole point of shared cross-section contrast."""
    bright = export.imaging.apply_colormap(_cell(5.0), "inferno", 0, 100, 99, anchor=5.0)
    faint = export.imaging.apply_colormap(_cell(2.0), "inferno", 0, 100, 99, anchor=5.0)
    assert faint.mean() < bright.mean()
    # with NO shared anchor each is auto-stretched to its own max → comparable means
    faint_auto = export.imaging.apply_colormap(_cell(2.0), "inferno", 0, 100, 99)
    assert faint_auto.mean() > faint.mean()
