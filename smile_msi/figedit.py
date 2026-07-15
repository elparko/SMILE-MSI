"""Shared, pure edit model for figure annotations — reorder, relabel, and drop the rows
and columns of an exportable figure without touching the underlying analysis.

The WYSIWYG editor (:mod:`smile_msi.gui.figeditor`) edits a :class:`FigEditSpec` and the
*same* render functions in :mod:`smile_msi.export` draw both the live preview and the
saved file — so what you arrange is exactly what you get. Two figure shapes are covered
by one model:

* the **SHAP bubble plot** — a matrix with an *ion* axis (rows) and a *region/donor* axis
  (columns); both axes are reorderable / droppable / relabelable.
* an **ion-image legend** — a single list of ion rows (no column axis).

Everything here is pure (numpy in, numpy/dict out) so it is unit-tested without a GUI and
serialises to/from the session via :meth:`FigEditSpec.to_dict`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AxisEdit:
    """Editable ordering, label overrides, and inclusion for one figure axis.

    ``order`` lists the ORIGINAL item indices in display order; ``dropped`` are original
    indices the user excluded; ``labels`` maps an original index to override text. Keeping
    everything keyed by the *original* index means a dropped item can be restored to its
    place and a relabelled item keeps its label through reordering."""
    order: list                      # original indices, in display order
    labels: dict = field(default_factory=dict)   # original index -> override text
    dropped: set = field(default_factory=set)    # original indices excluded

    @classmethod
    def identity(cls, n: int) -> "AxisEdit":
        return cls(order=list(range(int(n))), labels={}, dropped=set())

    def live_indices(self) -> list:
        """Original indices to draw, in display order (dropped ones removed)."""
        return [i for i in self.order if i not in self.dropped]

    def resolve_labels(self, originals) -> list:
        """Display labels for the live indices: the override if set, else the original."""
        out = []
        for i in self.live_indices():
            ov = self.labels.get(i)
            out.append(str(ov) if ov is not None and str(ov) != "" else str(originals[i]))
        return out

    def move(self, orig_index: int, delta: int) -> None:
        """Shift ``orig_index`` by ``delta`` places within ``order`` (clamped)."""
        if orig_index not in self.order:
            return
        pos = self.order.index(orig_index)
        new = max(0, min(len(self.order) - 1, pos + delta))
        self.order.insert(new, self.order.pop(pos))

    def sort_by(self, values, descending: bool = False) -> None:
        """Reorder the live items by ``values`` (indexed by *original* item index); dropped
        items keep their identities, parked after the sorted live ones."""
        live = sorted(self.live_indices(), key=lambda i: values[i], reverse=descending)
        self.order = live + [i for i in self.order if i in self.dropped]

    def to_dict(self) -> dict:
        return {"order": list(self.order),
                "labels": {str(k): v for k, v in self.labels.items()},
                "dropped": sorted(self.dropped)}

    @classmethod
    def from_dict(cls, d: dict, n: int) -> "AxisEdit":
        """Rebuild, reconciling against the current item count ``n`` so a stale spec (the
        analysis was re-run with more/fewer items) never crashes: unknown indices are
        dropped from ``order`` and any new items are appended at the end."""
        if not d:
            return cls.identity(n)
        order = [int(i) for i in d.get("order", []) if 0 <= int(i) < n]
        for i in range(n):                       # append items the saved order didn't know
            if i not in order:
                order.append(i)
        labels = {int(k): str(v) for k, v in (d.get("labels") or {}).items()
                  if 0 <= int(k) < n}
        dropped = {int(i) for i in (d.get("dropped") or []) if 0 <= int(i) < n}
        return cls(order=order, labels=labels, dropped=dropped)


@dataclass
class FigEditSpec:
    """Edit state for one figure: a row axis, an optional column axis, a title override, and
    a free-form ``options`` dict for figure-type-specific style choices (e.g. the SHAP
    bubble colormap) the caller's render callback reads."""
    rows: AxisEdit
    cols: AxisEdit | None = None
    title: str | None = None
    options: dict = field(default_factory=dict)

    @classmethod
    def identity(cls, n_rows: int, n_cols: int | None = None) -> "FigEditSpec":
        return cls(rows=AxisEdit.identity(n_rows),
                   cols=AxisEdit.identity(n_cols) if n_cols is not None else None,
                   title=None, options={})

    def to_dict(self) -> dict:
        d = {"rows": self.rows.to_dict(), "title": self.title, "options": dict(self.options)}
        if self.cols is not None:
            d["cols"] = self.cols.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict, n_rows: int, n_cols: int | None = None) -> "FigEditSpec":
        if not d:
            return cls.identity(n_rows, n_cols)
        cols = (AxisEdit.from_dict(d.get("cols"), n_cols) if n_cols is not None else None)
        return cls(rows=AxisEdit.from_dict(d.get("rows"), n_rows), cols=cols,
                   title=d.get("title"), options=dict(d.get("options") or {}))


def apply_matrix(values, row_axis: AxisEdit, col_axis: AxisEdit):
    """Reorder + subset a ``(n_cols, n_rows)`` matrix (the SHAP importance/direction shape)
    to the live columns × live rows the spec selects."""
    values = np.asarray(values)
    rows = row_axis.live_indices()
    cols = col_axis.live_indices()
    if not rows or not cols:
        return values[:0, :0]
    return values[np.ix_(cols, rows)]


def apply_shap(spec: FigEditSpec, importance, direction, col_labels, row_mz):
    """Transform the four ``render_shap_figure`` inputs through ``spec`` (reorder / drop /
    relabel rows and columns). Returns a dict ready to splat into the renderer."""
    rows, cols = spec.rows, spec.cols
    return {
        "importance": apply_matrix(importance, rows, cols),
        "direction": apply_matrix(direction, rows, cols),
        "col_labels": cols.resolve_labels(list(col_labels)),
        "row_mz": [row_mz[i] for i in rows.live_indices()],
        "row_labels": rows.resolve_labels([_mz_text(m) for m in row_mz]),
    }


def apply_rows(spec: FigEditSpec, items: list) -> list:
    """Reorder/drop/relabel a single-axis list (ion-image legend entries). ``items`` is a
    list of dicts; the row override, when set, is written to ``label_override`` so the
    existing legend renderer picks it up."""
    out = []
    originals = [_entry_text(e) for e in items]
    for disp_pos, i in enumerate(spec.rows.live_indices()):
        e = dict(items[i])
        ov = spec.rows.labels.get(i)
        if ov is not None and str(ov) != "":
            e["label_override"] = str(ov)
        out.append(e)
    return out


def _mz_text(mz) -> str:
    return f"{float(mz):.4f}"


def _entry_text(e) -> str:
    if e.get("label_override"):
        return str(e["label_override"])
    if e.get("label"):
        return str(e["label"])
    mz = e.get("mz")
    return _mz_text(mz) if mz is not None else "ion"
