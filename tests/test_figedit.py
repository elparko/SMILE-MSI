"""Tests for the pure figure-edit model (smile_msi.figedit) — reorder / drop / relabel of
figure rows and columns, and its application to the SHAP render inputs."""
import numpy as np

from smile_msi import figedit as fe


def test_axis_identity_and_live_order():
    ax = fe.AxisEdit.identity(3)
    assert ax.live_indices() == [0, 1, 2]
    assert ax.resolve_labels(["a", "b", "c"]) == ["a", "b", "c"]


def test_axis_drop_reorder_relabel():
    ax = fe.AxisEdit.identity(3)
    ax.dropped.add(1)                                  # drop the middle item
    assert ax.live_indices() == [0, 2]
    # move operates on the full order (the list shows dropped items too), so item 2 must
    # step past both the dropped item 1 and item 0 to reach the front
    ax.move(2, -2)                                     # pull item 2 to the front
    assert ax.live_indices() == [2, 0]
    ax.labels[0] = "renamed"
    assert ax.resolve_labels(["a", "b", "c"]) == ["c", "renamed"]


def test_axis_sort_by():
    ax = fe.AxisEdit.identity(3)
    ax.sort_by([30.0, 10.0, 20.0])                     # ascending by value
    assert ax.live_indices() == [1, 2, 0]
    ax.sort_by([30.0, 10.0, 20.0], descending=True)
    assert ax.live_indices() == [0, 2, 1]
    ax.dropped.add(0)                                  # dropped item parks after sorted live
    ax.sort_by([5.0, 1.0, 3.0])
    assert ax.live_indices() == [1, 2]
    assert 0 in ax.order                              # still present, just excluded


def test_axis_from_dict_reconciles_stale_counts():
    # saved spec knew 3 items; the analysis now has 4 (one new) — and index 2 vanished
    d = {"order": [2, 0, 1], "labels": {"0": "x"}, "dropped": [1]}
    ax = fe.AxisEdit.from_dict(d, n=4)
    assert set(ax.order) == {0, 1, 2, 3}               # new item 3 appended
    assert ax.order[-1] == 3
    assert ax.labels == {0: "x"}
    assert ax.dropped == {1}
    # and against a shrunk count, unknown indices are dropped
    ax2 = fe.AxisEdit.from_dict(d, n=2)
    assert set(ax2.order) == {0, 1}


def test_apply_matrix_reorders_and_subsets():
    # values shape (n_cols=2, n_rows=3)
    vals = np.array([[10, 11, 12], [20, 21, 22]])
    rows = fe.AxisEdit.identity(3)
    cols = fe.AxisEdit.identity(2)
    rows.dropped.add(1)                                # drop row 1
    rows.move(2, -2)                                   # rows now [2, 0]
    out = fe.apply_matrix(vals, rows, cols)
    assert out.shape == (2, 2)
    # col 0, rows [2,0] -> [12, 10]; col 1 -> [22, 20]
    assert out.tolist() == [[12, 10], [22, 20]]


def test_apply_shap_full_roundtrip():
    importance = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])   # (2 cols, 3 rows)
    direction = -importance
    col_labels = ["regA", "regB"]
    row_mz = [700.1, 800.2, 900.3]
    spec = fe.FigEditSpec.identity(n_rows=3, n_cols=2)
    spec.cols.move(1, -1)                              # regB first
    spec.cols.labels[1] = "Region B!"                 # rename a column
    spec.rows.dropped.add(0)                           # drop the first ion
    spec.rows.labels[2] = "PC 34:1"                    # rename an ion row
    out = fe.apply_shap(spec, importance, direction, col_labels, row_mz)

    assert out["col_labels"] == ["Region B!", "regA"]  # reordered + relabeled
    assert out["row_mz"] == [800.2, 900.3]             # row 0 dropped, order kept
    assert out["row_labels"] == ["800.2000", "PC 34:1"]
    # importance[colB, rows 1&2] then [colA, ...]
    assert out["importance"].shape == (2, 2)
    assert out["importance"].tolist() == [[5.0, 6.0], [2.0, 3.0]]
    assert out["direction"].tolist() == [[-5.0, -6.0], [-2.0, -3.0]]


def test_apply_rows_writes_label_override():
    items = [{"mz": 700.1}, {"mz": 800.2, "label": "X"}, {"mz": 900.3}]
    spec = fe.FigEditSpec.identity(n_rows=3)
    spec.rows.dropped.add(1)
    spec.rows.move(2, -2)                              # [2, 0]
    spec.rows.labels[2] = "my ion"
    out = fe.apply_rows(spec, items)
    assert [e["mz"] for e in out] == [900.3, 700.1]
    assert out[0]["label_override"] == "my ion"
    assert "label_override" not in out[1]              # untouched row keeps its dict


def test_spec_to_from_dict_roundtrip():
    spec = fe.FigEditSpec.identity(n_rows=3, n_cols=2)
    spec.rows.labels[1] = "ion"
    spec.cols.dropped.add(0)
    spec.title = "My figure"
    d = spec.to_dict()
    back = fe.FigEditSpec.from_dict(d, n_rows=3, n_cols=2)
    assert back.title == "My figure"
    assert back.rows.labels == {1: "ion"}
    assert back.cols.dropped == {0}
