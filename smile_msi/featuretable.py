"""Feature-list export — every analysis you have run, folded into one wide CSV.

A feature list is a set of m/z, plus whatever the annotator identified for each. Every
registry analysis that scores *per ion* — a region comparison's AUC and q-value, an ROI
localization, a PCA loading, a co-localization score — produces a table keyed by that same
m/z. This module folds them together: the feature list is the spine, and each analysis
contributes a namespaced block of columns aligned to it by m/z. The result is one table
that opens in R / Excel / pandas with no further joining, so the analysis can continue in
whatever tool the user actually writes their paper in.

Three problems this has to solve, and the choices made:

* **Not every analysis is per-feature.** A lipid-class comparison is keyed by class, a
  cross-validation result is a confusion matrix, a per-ion segmentation describes exactly
  one ion. :func:`attach` inspects the table and *refuses* the ones that cannot attach,
  recording a reason the picker can show — better than a block of NaN columns that looks
  like a failed analysis rather than an inapplicable one.

* **Some per-feature analyses are long, not wide.** Discriminating features emit one row
  per (region, m/z); PCA emits one row per (component, m/z). Those pivot to one column per
  key value — ``…endo.AUC``, ``…peri.AUC`` — so a feature stays exactly one row, which is
  the whole point of the export.

* **Two analyses can name a column the same thing.** Nearly every statistics step emits
  ``AUC`` and ``q_value``. Each block is namespaced with the analysis's registry id
  (``roi_comparison.AUC``), and repeated runs of one analysis take an ordinal
  (``roi_comparison_2.AUC``) assigned over the *whole* run store rather than over the
  current selection — so a column keeps its name whether or not you ticked its neighbour.

The join is by **nearest m/z within a ppm tolerance**, never float equality: an analysis
may have run against a feature set that was since re-picked or recalibrated, and an
equality join would silently drop every one of its rows.

Pure module — no Qt, and pandas/numpy are imported lazily inside the functions that need
them, so importing it costs nothing at GUI startup.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

#: Default join window. Feature m/z normally match to the bit, but a recalibration between
#: the analysis and the export shifts them by a ppm or two. Wide enough to survive that,
#: far tighter than the spacing between neighbouring features.
DEFAULT_JOIN_PPM = 5.0

#: Column names that identify WHICH sub-result a long-form row belongs to, tried in this
#: order before any other column. One row per (key, m/z) becomes one column per key value.
PIVOT_KEYS = ("region", "pair", "component", "module", "cluster", "group", "level", "class")

#: A pivot key with more distinct values than this would explode the table — a key column
#: names a handful of regions or components, it is not a second feature axis.
MAX_PIVOT_VALUES = 40

#: pandas writes an unnamed index column into ``result.csv`` when a table was saved with
#: one. It is never data.
_JUNK_COL = re.compile(r"^(Unnamed: \d+|)$")


def _pandas():
    import pandas as pd
    return pd


def _numpy():
    import numpy as np
    return np


def slug(text, *, fallback="analysis") -> str:
    """A machine-friendly column token: lowercase, non-alphanumerics collapsed to ``_``.

    Column names travel to other tools, so they stay ASCII and free of the spaces, dots and
    parentheses that R's ``read.csv`` mangles and SQL needs quoting for.
    """
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(text if text is not None else "")).strip("_").lower()
    return s or fallback


# --------------------------------------------------------------------------- #
# one analysis, reduced to columns
# --------------------------------------------------------------------------- #
@dataclass
class Attachment:
    """One analysis, reduced to the columns it can contribute to a feature list.

    ``table`` is ``None`` when the analysis is not per-feature; ``reason`` then says why,
    and the picker greys the row out instead of silently dropping it. When the analysis *is*
    attachable, ``reason`` may still carry a note about how its table was reshaped.
    """

    key: str = ""
    prefix: str = "analysis"
    title: str = ""
    subtitle: str = ""
    table: object = None            # wide DataFrame, index = unique sorted float m/z
    reason: str = ""

    @property
    def usable(self) -> bool:
        return self.table is not None and len(self.table) > 0 and len(self.table.columns) > 0

    @property
    def columns(self) -> list:
        return list(self.table.columns) if self.table is not None else []

    @property
    def n_features(self) -> int:
        return int(len(self.table)) if self.table is not None else 0


def attach(table, *, key="", prefix="analysis", title="", subtitle="") -> Attachment:
    """Reduce one analysis result table to an :class:`Attachment`."""
    wide, reason = _widen(table, prefix)
    return Attachment(key=key, prefix=prefix, title=(title or prefix), subtitle=subtitle,
                      table=wide, reason=reason)


def _widen(table, prefix):
    """``(wide DataFrame indexed by m/z, note)`` or ``(None, reason it cannot attach)``."""
    pd = _pandas()
    if table is None:
        return None, "no result table"
    if not isinstance(table, pd.DataFrame):
        # an image / spectra / classifier-map result persists as JSON, not a table
        return None, "result is not a table (an image or spectra)"
    if not len(table):
        return None, "the result table is empty"
    if "mz" not in table.columns:
        return None, "not a per-feature result (no m/z column)"

    d = table.copy()
    d["mz"] = pd.to_numeric(d["mz"], errors="coerce")
    d = d[d["mz"].notna()]
    if d.empty:
        return None, "no usable m/z values"

    values = [c for c in d.columns if c != "mz" and not _JUNK_COL.match(str(c))]
    if not values:
        return None, "no columns besides m/z"
    # A per-ion segmentation tabulates ONE ion's intensity zones: every row shares an m/z.
    # It describes that ion, not the feature set, so it has nothing to attach to.
    if len(d) > 1 and d["mz"].nunique() == 1:
        return None, "describes a single ion, not the feature set"

    note = ""
    if d["mz"].duplicated().any():
        key_col = _pivot_key(d, values)
        vals = [c for c in values if c != key_col]
        if key_col is None or not vals:
            # Nothing keys the repeats apart — keep the first row per m/z rather than pick
            # arbitrarily among the rest, and say so.
            d = d.drop_duplicates("mz")
            wide = d.set_index("mz")[values]
            note = "several rows per m/z — kept the first of each"
        else:
            d = d.drop_duplicates(["mz", key_col])
            d[key_col] = d[key_col].astype(str)
            wide = d.pivot(index="mz", columns=key_col, values=vals)
            wide = _flatten_pivot(wide, vals, key_col)
            note = f"one row per ({key_col}, m/z) — pivoted to one column per {key_col}"
    else:
        wide = d.set_index("mz")[values]

    wide = wide[~wide.index.duplicated()].sort_index()
    wide.columns = [f"{prefix}.{c}" for c in wide.columns]
    return wide, note


def _flatten_pivot(wide, vals, key_col):
    """MultiIndex ``(value, key)`` columns → flat ``key.value``, grouped by key value.

    Grouping by key keeps one region's columns adjacent (``endo.AUC, endo.q_value,
    peri.AUC, …``), which is how a human reads the sheet; pandas' native order interleaves
    them by statistic instead.
    """
    present = set(wide.columns)
    keyvals = sorted({kv for _, kv in wide.columns}, key=_natural_key)
    order = [(vc, kv) for kv in keyvals for vc in vals if (vc, kv) in present]
    wide = wide[order]
    tokens, used = {}, set()
    for kv in keyvals:
        t = slug(kv, fallback="value")
        # A bare number names nothing: PCA's components are 0…4, so 'pca.0.loading' would
        # leave the reader guessing. Borrow the key column's own name for those.
        if t[0].isdigit():
            t = f"{slug(key_col)}{t}"
        # Two key values can slug to one token ('endo ' and 'endo'); a duplicate header
        # would make the CSV unreadable by name, so disambiguate by rank.
        while t in used:
            t += "_"
        used.add(t)
        tokens[kv] = t
    wide.columns = [f"{tokens[kv]}.{vc}" for vc, kv in order]
    return wide


def _natural_key(s):
    """'PC10' sorts after 'PC2', not before."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def _pivot_key(d, values):
    """The column that keys a long-form table apart, or ``None``.

    A key column makes ``(m/z, key)`` unique, takes a handful of distinct values, and is not
    continuous — a float statistic is never a key, however unique it happens to be.
    """
    pd = _pandas()
    preferred = [c for c in PIVOT_KEYS if c in values]
    for c in preferred + [c for c in values if c not in preferred]:
        col = d[c]
        if pd.api.types.is_float_dtype(col):
            continue
        if not (2 <= col.nunique(dropna=False) <= MAX_PIVOT_VALUES):
            continue
        if d.duplicated(["mz", c]).any():
            continue
        return c
    return None


# --------------------------------------------------------------------------- #
# runs → attachments
# --------------------------------------------------------------------------- #
#: Separates an analysis's name from what makes one run of it different, in ``AnalysisRun.title``
#: (``AnalysisDialog._run_title``). Everything after it names the run: ``endo vs peri``.
TITLE_SEP = " — "


def _distinguisher(run) -> str:
    """The slugged tail of a run's title — what makes this run of the analysis different."""
    _, sep, tail = (getattr(run, "title", "") or "").partition(TITLE_SEP)
    return slug(tail, fallback="") if sep else ""


def run_prefixes(runs) -> dict:
    """``{run_id: column prefix}`` over the *whole* run list.

    Prefixed by registry id, so the columns read as what the analysis was
    (``roi_comparison.AUC``). When one analysis was run more than once, the runs are told apart
    by what made them different — ``roi_comparison_endo_vs_peri.AUC`` — falling back to the
    order they were run (``roi_comparison_2``) when a run has nothing distinguishing to say.
    A bare ordinal in a CSV header is a trap: nothing in the file says which contrast it was.

    Assigning over every run — not just the ones being exported — means a prefix names the same
    run in every export, whatever else was ticked.
    """
    by_step: dict = {}
    for r in sorted(runs, key=lambda r: (getattr(r, "created", "") or "",
                                         getattr(r, "run_id", "") or "")):
        by_step.setdefault(getattr(r, "step_id", "") or "analysis", []).append(r)
    out = {}
    for step, rs in by_step.items():
        base = slug(step)
        if len(rs) == 1:
            out[getattr(rs[0], "run_id", "")] = base
            continue
        tags = [_distinguisher(r) for r in rs]
        # only trust the titles when every one of them says something, and says it uniquely
        useful = all(tags) and len(set(tags)) == len(tags)
        for i, r in enumerate(rs, 1):
            out[getattr(r, "run_id", "")] = f"{base}_{tags[i - 1] if useful else i}"
    return out


def attachments_from_runs(runs, load_result, *, name_of=None) -> list:
    """Every completed run, reduced to an :class:`Attachment` (usable or not).

    ``load_result(run)`` rehydrates the stored payload (``RunStore.load_result``); ``name_of``
    maps a run to its human analysis name (the registry's, in the app). Unfinished runs are
    skipped entirely — a failed or still-running analysis has no columns to offer and does
    not belong in the picker.
    """
    prefixes = run_prefixes(runs)
    out = []
    for r in runs:
        if getattr(r, "status", "") != "done":
            continue
        rid = getattr(r, "run_id", "")
        prefix = prefixes.get(rid) or slug(getattr(r, "step_id", ""))
        # the run's own title only when it says something the analysis name doesn't ('… — endo
        # vs peri'); otherwise the registry's current name, which survives a step being renamed
        stored = getattr(r, "title", "") or ""
        title = stored if TITLE_SEP in stored else (
            (name_of(r) if name_of is not None else None) or getattr(r, "step_id", ""))
        sub = " · ".join(x for x in (getattr(r, "created", ""), getattr(r, "summary", "")) if x)
        try:
            payload = load_result(r)
        except Exception as exc:                       # noqa: BLE001 — a bad payload greys out
            out.append(Attachment(key=rid, prefix=prefix, title=title, subtitle=sub,
                                  reason=f"result could not be read ({type(exc).__name__})"))
            continue
        out.append(attach(payload, key=rid, prefix=prefix, title=title, subtitle=sub))
    return out


# --------------------------------------------------------------------------- #
# the join
# --------------------------------------------------------------------------- #
def nearest_within(query, ref, tol_ppm):
    """Index into ``ref`` of the nearest value to each ``query``, or ``-1`` past ``tol_ppm``."""
    np = _numpy()
    query = np.asarray(query, dtype=float)
    ref = np.asarray(ref, dtype=float)
    out = np.full(query.shape, -1, dtype=int)
    if ref.size == 0 or query.size == 0:
        return out
    order = np.argsort(ref, kind="stable")
    rs = ref[order]
    pos = np.searchsorted(rs, query)
    lo = np.clip(pos - 1, 0, rs.size - 1)
    hi = np.clip(pos, 0, rs.size - 1)
    dlo, dhi = np.abs(rs[lo] - query), np.abs(rs[hi] - query)
    pick = np.where(dlo <= dhi, lo, hi)
    dist = np.minimum(dlo, dhi)
    # NaN queries: searchsorted parks them at the end and every distance is NaN, so the
    # comparison below is already False — the isfinite guard just makes that intent explicit.
    ok = np.isfinite(query) & (dist <= np.abs(query) * (float(tol_ppm) * 1e-6))
    out[ok] = order[pick][ok]
    return out


def join_by_mz(base, attachments, *, tol_ppm=DEFAULT_JOIN_PPM):
    """``(wide DataFrame, per-analysis notes)`` — the feature list plus every attachment.

    ``base`` is the feature list (one row per feature, an ``mz`` column). Unusable
    attachments are skipped; a feature an analysis never scored gets NaN, not a dropped row —
    the exported list is always exactly the feature list you were looking at.
    """
    np, pd = _numpy(), _pandas()
    if base is None or "mz" not in getattr(base, "columns", []):
        raise ValueError("the feature table needs an 'mz' column to join analyses onto")
    out = base.copy()
    q = pd.to_numeric(out["mz"], errors="coerce").to_numpy(dtype=float)
    blocks, notes = [], []
    for a in attachments:
        if not a.usable:
            continue
        idx = nearest_within(q, a.table.index.to_numpy(dtype=float), tol_ppm)
        hit = idx >= 0
        take = np.where(hit, idx, 0)                      # placeholder row for the misses…
        hit_s = pd.Series(hit, index=out.index)
        # …which `.where` then blanks to NaN. A feature no analysis scored keeps its row.
        blocks.append(pd.DataFrame(
            {col: pd.Series(a.table[col].to_numpy()[take], index=out.index).where(hit_s)
             for col in a.table.columns}, index=out.index))
        notes.append(f"{a.title} → {a.prefix}.*: {int(hit.sum())} of {len(out)} features matched"
                     + (f" ({a.reason})" if a.reason else ""))
    # One concat, not a column at a time: a dozen analyses on a long list would otherwise
    # re-copy the whole frame for every column they contribute.
    return (pd.concat([out, *blocks], axis=1) if blocks else out), notes
