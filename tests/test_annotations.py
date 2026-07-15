"""Unit tests for the shared annotation layout/format model (smile_msi.annotations)."""
import numpy as np

from smile_msi import annotations as A


def test_format_ion_label_modes():
    assert A.format_ion_label(463.2463, ppm=10, name="PC 34:1", mode=A.LABEL_MZ_PPM) == \
        "463.2463 m/z ± 10 ppm"
    assert A.format_ion_label(463.2463, ppm=10, name="PC 34:1", mode=A.LABEL_MZ) == "463.2463 m/z"
    assert A.format_ion_label(463.2463, ppm=10, name="PC 34:1", mode=A.LABEL_NAME) == \
        "463.2463 m/z · PC 34:1"
    # ppm mode without a tolerance gracefully drops the ± part
    assert A.format_ion_label(700.0, ppm=None, mode=A.LABEL_MZ_PPM) == "700.0000 m/z"
    # name mode without a name drops the dot
    assert A.format_ion_label(700.0, name="", mode=A.LABEL_NAME) == "700.0000 m/z"


def test_format_ion_label_override_wins():
    assert A.format_ion_label(463.2463, ppm=10, name="PC 34:1", override="Myelin") == "Myelin"
    # blank override falls back to the mode
    assert A.format_ion_label(463.0, ppm=5, override="  ", mode=A.LABEL_MZ) == "463.0000 m/z"
    assert A.format_ion_label(None, name="intensity") == "intensity"


def test_detect_subject_bbox_fractions():
    occ = np.zeros((100, 200), bool)
    occ[20:60, 50:150] = True                      # rows 20..59, cols 50..149
    r = A.detect_subject_bbox(occ)
    assert r is not None
    assert abs(r.x - 50 / 200) < 1e-9              # left = col0 / W
    assert abs(r.w - 100 / 200) < 1e-9
    # y is flipped (origin bottom-left): top row 20 -> y_top = 1 - 20/100
    assert abs((r.y + r.h) - (1 - 20 / 100)) < 1e-9
    assert A.detect_subject_bbox(np.zeros((10, 10), bool)) is None


def test_empty_corner_opposite_subject():
    # tissue tucked in the visual lower-left → emptiest corner is upper-right
    occ = np.zeros((100, 120), bool)
    occ[60:95, 5:40] = True
    assert A.empty_corner(occ) == "upper right"
    assert A.opposite_corner("upper right") == "lower left"
    assert A.opposite_corner("lower left") == "upper right"


def test_layout_legend_within_frame_and_clear():
    rows = [A.LegendRow(label="463.2463 m/z ± 10 ppm", color="#0a84ff", lo=20, hi=90, max_pct=194),
            A.LegendRow(label="607.3615 m/z ± 10 ppm", color="#ff453a", lo=20, hi=90, max_pct=163)]
    L = A.layout_legend(rows, corner="lower right")
    assert len(L.rows) == 2 and L.overflow == 0
    for g in L.rows:
        for x in (g.bar.x, g.bar.x + g.bar.w, g.label_xy[0], g.max_xy[0]):
            assert 0.0 <= x <= 1.0
        for y in (g.bar.y, g.bar.y + g.bar.h):
            assert 0.0 <= y <= 1.0
        assert g.label_xy[0] < g.bar.x            # label sits to the LEFT of the bar
        assert g.max_xy[0] > g.bar.x + g.bar.w    # trailing % sits to the RIGHT of the bar


def test_layout_legend_overflow_capped():
    rows = [A.LegendRow(label=f"{i}", color="#fff") for i in range(A._MAX_ROWS + 4)]
    L = A.layout_legend(rows)
    assert len(L.rows) == A._MAX_ROWS
    assert L.overflow == 4


def test_layout_scalebar():
    sb = A.layout_scalebar(pixel_size_um=50, scale_bar_um=500, img_w_px=200, corner="lower left")
    assert sb is not None and sb.label == "500µm"
    assert 0.0 <= sb.x0 < sb.x1 <= 1.0
    assert A.layout_scalebar(pixel_size_um=None, scale_bar_um=500, img_w_px=200) is None


def test_nice_scalebar_um():
    # ~20% of width, snapped to a 1/2/5×10ⁿ round number, never over 42%
    assert A.nice_scalebar_um(5000.0) == 1000.0        # 20% of 5mm → 1mm
    assert A.nice_scalebar_um(1000.0) == 200.0         # 20% of 1mm → 200µm
    for w in (137.0, 940.0, 6321.0, 25000.0):
        v = A.nice_scalebar_um(w)
        assert v is not None and v <= 0.42 * w         # always drawable without clamping
        assert str(v / 10 ** np.floor(np.log10(v)))[0] in "125"   # 1/2/5 lead digit
    assert A.nice_scalebar_um(0) is None and A.nice_scalebar_um(None) is None


def test_format_scalebar_label():
    assert A.format_scalebar_label(250) == "250µm"
    assert A.format_scalebar_label(1000) == "1mm"
    assert A.format_scalebar_label(1500) == "1.5mm"


def test_layout_scalebar_relabels_instead_of_lying():
    """An over-long requested bar (>42% of the image) is snapped to a nice fitting length
    AND relabelled, so the drawn bar always matches its caption — the old code clamped the
    line but kept the original label, drawing a bar that lied about its length."""
    # 500µm at 50µm/px on a 20px-wide image = 10px = 50% of the width → must NOT stay '500µm'
    sb = A.layout_scalebar(pixel_size_um=50, scale_bar_um=500, img_w_px=20, corner="lower left")
    assert sb is not None
    drawn_um = (sb.x1 - sb.x0) / 1.0 * (20 * 50)        # bar fraction × physical width
    assert abs(drawn_um - float(sb.label.rstrip("µm"))) < 1e-6   # caption matches the line
    assert (sb.x1 - sb.x0) <= 0.42 + 1e-9


def test_pick_ink_contrast_and_default():
    assert A.pick_ink(np.zeros((8, 8, 3)))[0] == "#ffffff"          # dark bg → white ink
    assert A.pick_ink(np.full((8, 8, 3), 240.0))[0] == "#101418"    # light bg → dark ink
    # fully transparent → no luminance → the provided default
    transparent = np.zeros((8, 8, 4))
    assert A.pick_ink(transparent, default=A.INK_LIGHT) == A.INK_LIGHT
    # a flat list of RGB pixels is accepted too
    assert A.pick_ink(np.zeros((20, 3)))[0] == "#ffffff"
