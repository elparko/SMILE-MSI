"""Export Studio engine — a pure, Qt-free batch compiler + runner that renders a chosen
set of ions across a chosen set of sections into an organised folder tree (plus an optional
combined ion×section *matrix* figure, the feature-list CSVs and the analyses bundle), all in
one action.

This module is the sibling of :mod:`smile_msi.export` (rendering) and
:mod:`smile_msi.cohort` (multi-sample tables). It owns **only the orchestration**:

* the cross-product of *(sections × selected ions × outputs)* compiled into an ordered plan,
* the **source-outer load → extract → evict** loop that bounds peak RAM to one data cube even
  across a large cohort (a slide shared by several region-samples is loaded once),
* the **shared-vs-per-section** contrast aggregation (one honest intensity window per ion
  across all sections, or each section auto-contrasted), and
* the on-disk layout (``<list>/<section>/ion_<mz>_<label>.<ext>``) + ``manifest.csv``.

Every heavyweight step — loading a dataset, extracting an ion image, rendering a styled panel,
writing a table — is an **injected callback**, so the engine imports no Qt, no
:class:`~smile_msi.msi.MSIDataset` and no matplotlib, and is fully unit-testable with fakes
(see ``tests/test_studio.py``). The GUI layer (``gui/studiodialog.py``) supplies the real
callbacks that close over the live app state and the pure :mod:`smile_msi.export` engine.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from enum import Enum


# --------------------------------------------------------------------------- #
# Filesystem-safe names (shared so the GUI and engine agree on every path)
# --------------------------------------------------------------------------- #
def safe_name(name: str) -> str:
    """Filesystem-safe stem: spaces → ``_`` and every character Windows forbids
    (``\\ / : * ? " < > |``) → ``-`` so a label like ``PG 46:2`` can't produce an
    unwritable path on the lab's Windows machines. Trailing dots/spaces stripped."""
    s = str(name).replace(" ", "_")
    for ch in '\\/:*?"<>|':
        s = s.replace(ch, "-")
    return s.strip("_- .") or "item"


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
class OutputKind(str, Enum):
    """The artifacts a single Studio run can emit (user ticks any subset)."""
    ION_IMAGES = "ion_images"   # one styled ion panel per (section, selection) → folder tree
    OVERLAY = "overlay"         # additive colour composite per section (all picks together)
    MATRIX = "matrix"           # combined ion-row × section-column figure
    LIST_CSV = "list_csv"       # the feature-list CSVs (one per picked list)
    ANALYSES = "analyses"       # the analyses / data bundle (region intensities, stats, methods)


class ContrastMode(str, Enum):
    SHARED = "shared"           # one intensity window per ion across all sections (comparable)
    PER_SECTION = "per_section"  # each section auto-contrasted to look its best individually


# --------------------------------------------------------------------------- #
# Selection (what to render) and Section (where to render it)
# --------------------------------------------------------------------------- #
@dataclass
class SelectionRecord:
    """One ion — or one lipid-class composite — the user picked, tagged with the list it
    came from so every output is traceable back to its source feature list."""
    mz: float | None = None         # representative m/z (None only for an unresolved composite)
    label: str = ""                 # display + file label (lipid name, or the class name)
    list_name: str = ""             # the feature list / scope / lipid-list it was picked from
    kind: str = "ion"               # "ion" | "composite" (a ◆ class summed into one image)
    source_kind: str = "saved"      # "scope" | "saved" | "lipid_class" | "working"
    lipid_class: str = ""           # set when kind == "composite"
    member_mzs: tuple = ()          # member m/z of a composite (summed image / expanded ions)
    color: str = ""                 # optional explicit channel colour (used by the overlay)

    def uid(self) -> str:
        """Stable id for the manifest + de-duplication, unique within a list."""
        tag = self.lipid_class if self.kind == "composite" else (
            f"{self.mz:.4f}" if self.mz is not None else self.label)
        return f"{safe_name(self.list_name)}::{self.kind}::{safe_name(tag)}"

    def file_stem(self) -> str:
        """The ``ion_<mz>_<label>`` (or ``class_<name>``) filename stem."""
        if self.kind == "composite":
            return f"class_{safe_name(self.lipid_class or self.label or 'composite')}"
        mz = f"{self.mz:.4f}" if self.mz is not None else "feature"
        lab = safe_name(self.label) if self.label else ""
        return f"ion_{mz}" + (f"_{lab}" if lab else "")

    def to_dict(self) -> dict:
        return {"mz": self.mz, "label": self.label, "list_name": self.list_name,
                "kind": self.kind, "source_kind": self.source_kind,
                "lipid_class": self.lipid_class, "member_mzs": list(self.member_mzs),
                "color": self.color}

    @classmethod
    def from_dict(cls, d: dict) -> "SelectionRecord":
        return cls(mz=(float(d["mz"]) if d.get("mz") is not None else None),
                   label=str(d.get("label", "")), list_name=str(d.get("list_name", "")),
                   kind=str(d.get("kind", "ion")), source_kind=str(d.get("source_kind", "saved")),
                   lipid_class=str(d.get("lipid_class", "")),
                   member_mzs=tuple(float(m) for m in (d.get("member_mzs") or [])),
                   color=str(d.get("color", "")))


@dataclass
class Section:
    """A renderable section: a whole slide or a named region within one. ``source`` is the
    load key — sections sharing a source load their cube exactly once."""
    sid: str = ""                   # stable id (caller-assigned)
    name: str = ""                  # display + folder name
    source: str = ""                # imzML / source path; "" for the live in-memory slide only
    session_path: str = ""          # saved session (region mask + frozen settings live here)
    region: str = ""                # region name within the slide; "" = whole slide
    is_live: bool = False           # the currently-loaded slide (reuse self.ds, never reload)

    def load_key(self) -> str:
        """What the loader caches on. The live slide is its own key so it's never evicted by
        a same-source disk section."""
        return "<live>" if self.is_live else (self.source or self.session_path or self.sid)

    def to_dict(self) -> dict:
        return {"sid": self.sid, "name": self.name, "source": self.source,
                "session_path": self.session_path, "region": self.region, "is_live": self.is_live}

    @classmethod
    def from_dict(cls, d: dict) -> "Section":
        return cls(sid=str(d.get("sid", "")), name=str(d.get("name", "")),
                   source=str(d.get("source", "")), session_path=str(d.get("session_path", "")),
                   region=str(d.get("region", "")), is_live=bool(d.get("is_live", False)))


@dataclass
class StudioJob:
    """One ion-image render unit: a (section, selection) cell and the file it writes."""
    section: Section
    selection: SelectionRecord
    path: str                       # path relative to the run's output folder
    status: str = "planned"         # planned | done | skipped | error
    note: str = ""

    def manifest_row(self) -> dict:
        sel = self.selection
        return {"section": self.section.name, "region": self.section.region,
                "list": sel.list_name, "kind": sel.kind,
                "mz": ("" if sel.mz is None else f"{sel.mz:.4f}"),
                "label": sel.label or sel.lipid_class, "file": self.path,
                "status": self.status, "note": self.note}


# --------------------------------------------------------------------------- #
# The compiled plan
# --------------------------------------------------------------------------- #
@dataclass
class StudioPlan:
    """A fully-resolved export plan: the ordered sections, the resolved selections, the chosen
    outputs + design, and the precomputed ion-image jobs with their on-disk paths."""
    sections: list = field(default_factory=list)        # list[Section]
    selections: list = field(default_factory=list)      # list[SelectionRecord]
    outputs: list = field(default_factory=list)         # list[OutputKind]
    design: dict = field(default_factory=dict)
    layout: str = "list_outer"                          # "list_outer" | "section_outer"
    contrast: str = ContrastMode.PER_SECTION.value
    ext: str = "png"
    jobs: list = field(default_factory=list)            # list[StudioJob] (ion images)

    # ----- serialization (persisted in the session beside report_items) ----- #
    def to_dict(self) -> dict:
        return {"sections": [s.to_dict() for s in self.sections],
                "selections": [s.to_dict() for s in self.selections],
                "outputs": [OutputKind(o).value for o in self.outputs],
                "design": dict(self.design), "layout": self.layout,
                "contrast": ContrastMode(self.contrast).value, "ext": self.ext}

    @classmethod
    def from_dict(cls, d: dict) -> "StudioPlan":
        return compile_plan(
            sections=[Section.from_dict(s) for s in (d.get("sections") or [])],
            selections=[SelectionRecord.from_dict(s) for s in (d.get("selections") or [])],
            outputs=[OutputKind(o) for o in (d.get("outputs") or [])],
            design=dict(d.get("design") or {}), layout=str(d.get("layout", "list_outer")),
            contrast=str(d.get("contrast", ContrastMode.PER_SECTION.value)),
            ext=str(d.get("ext", "png")))

    # ----- convenience ----------------------------------------------------- #
    def wants(self, kind: OutputKind) -> bool:
        return OutputKind(kind) in {OutputKind(o) for o in self.outputs}

    def job_count(self) -> int:
        return len(self.jobs)


def _job_path(section: Section, sel: SelectionRecord, layout: str, ext: str) -> str:
    """The relative path for one (section, selection) image under the output folder.

    The RENDER loop is always source-outer for cube efficiency; this only controls the
    *written* layout. ``list_outer`` (default) groups a feature list together so the user can
    flip through one list across every section in one folder."""
    sec = safe_name(section.name)
    lst = safe_name(sel.list_name) if sel.list_name else "features"
    stem = sel.file_stem() + "." + ext.lstrip(".")
    return os.path.join(lst, sec, stem) if layout == "list_outer" else os.path.join(sec, lst, stem)


def compile_plan(sections, selections, outputs, *, design=None, layout="list_outer",
                 contrast=ContrastMode.PER_SECTION, ext="png") -> StudioPlan:
    """Compile a selection cross-product into an ordered :class:`StudioPlan`.

    The ion-image ``jobs`` are the full *(sections × selections)* product with their planned
    relative paths; duplicate (section, selection) cells are dropped so a feature appearing in
    two lists doesn't render twice for the same list. ``design`` carries the styling kwargs
    (cmap/theme/dpi/scalebar…); ``contrast`` is shared (comparable) or per-section.
    """
    sections = list(sections)
    selections = list(selections)
    outputs = [OutputKind(o) for o in outputs]
    ext = str(ext).lstrip(".") or "png"
    layout = layout if layout in ("list_outer", "section_outer") else "list_outer"
    contrast = ContrastMode(contrast).value

    jobs: list[StudioJob] = []
    if OutputKind.ION_IMAGES in outputs:
        seen: set[tuple[str, str]] = set()
        for sec in sections:                          # section-outer order = render order
            for sel in selections:
                key = (sec.sid, sel.uid())
                if key in seen:
                    continue
                seen.add(key)
                jobs.append(StudioJob(section=sec, selection=sel,
                                      path=_job_path(sec, sel, layout, ext)))
    return StudioPlan(sections=sections, selections=selections, outputs=outputs,
                      design=dict(design or {}), layout=layout, contrast=contrast,
                      ext=ext, jobs=jobs)


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
@dataclass
class RunResult:
    out_dir: str
    jobs: list                       # list[StudioJob] with final status
    written: list = field(default_factory=list)        # absolute paths written (files + folders)
    skipped: int = 0
    errors: int = 0
    extra: dict = field(default_factory=dict)          # output-kind → path (matrix, csv folder…)
    cancelled: bool = False
    contrast_anchors: dict = field(default_factory=dict)  # sel_uid → shared 100% anchor (SHARED mode)
    contrast_note: str | None = None                      # human description of the shared window

    def manifest_rows(self) -> list:
        return [j.manifest_row() for j in self.jobs]


def _group_by_source(sections):
    """Sections grouped by ``load_key`` in first-seen order, so a shared slide loads once and
    each cube can be evicted the moment its last section is extracted (peak RAM ≈ one cube)."""
    order: list[str] = []
    groups: dict[str, list] = {}
    for s in sections:
        k = s.load_key()
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(s)
    return [(k, groups[k]) for k in order]


# Below this many renderable cells, a spawn process pool's per-worker startup (re-import +
# matplotlib init, ~0.5–1s each on Windows/macOS spawn) costs more than it saves — render serially.
_PARALLEL_MIN_JOBS = 8


def _make_render_executor(max_workers):
    """A **spawn** ProcessPoolExecutor for the render phase. Spawn (never fork) because the
    caller holds Qt threads — a forked child of a threaded process can deadlock. Factored out
    so tests can inject a synchronous/in-process executor (real subprocess spawning is flaky to
    drive deterministically from pytest)."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    return ProcessPoolExecutor(max_workers=max_workers,
                               mp_context=multiprocessing.get_context("spawn"))


def _render_parallel(renderable, build_spec, render_cell, anchors, out_dir, max_workers,
                     result, on_path, progress, total, done, cancelled) -> bool:
    """Render the already-extracted cells across a **spawn** process pool. Returns True if the
    run was cancelled. Extraction already ran serially (one cube in RAM at a time); here we only
    ship each 2-D array + its picklable ``spec`` to a worker, then collect results as they land
    and update the manifest/progress. Spawn (never fork) because the caller holds Qt threads — a
    forked child of a threaded process can deadlock."""
    from concurrent.futures import as_completed

    # Build specs + create output dirs on the main thread (workers only write files).
    specs = []
    for job, arr in renderable:
        abspath = os.path.join(out_dir, job.path)
        os.makedirs(os.path.dirname(abspath), exist_ok=True)
        spec = build_spec(arr, job.selection, job.section, anchors.get(job.selection.uid()))
        specs.append((job, arr, spec, abspath))

    ex = _make_render_executor(max_workers)
    was_cancelled = False
    n = done
    try:
        futs = {ex.submit(render_cell, arr, spec, path): (job, path)
                for (job, arr, spec, path) in specs}
        for fut in as_completed(futs):
            job, path = futs[fut]
            try:
                fut.result()
                job.status = "done"
                result.written.append(path)
                if on_path:
                    on_path(path)
            except Exception as e:  # noqa: BLE001 — one bad cell must not sink the batch
                job.status = "error"
                job.note = f"{type(e).__name__}: {e}"
                result.errors += 1
            n += 1
            if progress:
                # A progress hook that raises (the GUI worker's cancel signal) propagates out;
                # the finally below tears the pool down without waiting.
                progress(n, total, f"Rendering {job.section.name} · {_sel_label(job.selection)}")
            if cancelled():
                was_cancelled = True
                break
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    return was_cancelled


def _shared_anchor(arrays, percentile, measure):
    """The shared per-ion contrast anchor across sections — the value mapped to 100% brightness
    in every section so an ion is comparable slide-to-slide. With ``percentile`` set it is the
    **pooled** percentile of all sections' finite tissue pixels (robust to a single hot pixel:
    e.g. the 99th percentile across both nerves combined, in whatever normalization the images
    were extracted with). Falls back to ``max`` of the per-section ``measure`` scalars when no
    percentile is given (legacy behaviour). Returns None when there is no positive signal."""
    if not arrays:
        return None
    if percentile is not None:
        import numpy as np
        pooled = []
        for a in arrays:
            fa = np.asarray(a, dtype=float).ravel()
            fa = fa[np.isfinite(fa)]
            if fa.size:
                pooled.append(fa)
        if not pooled:
            return None
        v = float(np.percentile(np.concatenate(pooled), float(percentile)))
        return v if v > 0 else None
    vals = []
    for a in arrays:
        if measure is None:
            continue
        try:
            m = measure(a)
            if m is not None and m > 0:
                vals.append(float(m))
        except Exception:  # noqa: BLE001
            pass
    return max(vals) if vals else None


def run_plan(plan: StudioPlan, *, loader, extract, render_image,
             measure=None, contrast_percentile=None, contrast_note=None,
             write_outputs=None, release=None,
             out_dir, progress=None, cancel=None, on_path=None,
             render_cell=None, build_spec=None, max_workers=None) -> RunResult:
    """Execute a compiled plan against injected callbacks.

    Callbacks (the GUI supplies real implementations; tests supply fakes):

    * ``loader(section) -> ds`` — return an opaque dataset handle for ``section`` (it may
      cache by ``section.load_key()`` internally), or ``None`` to skip the whole section. The
      engine calls ``release(load_key)`` once the source's last section is extracted.
    * ``extract(ds, section, selection) -> ndarray | None`` — the 2-D ion image for this cell,
      or ``None`` when the ion is absent / out of range (recorded ``status="skipped"``).
    * ``measure(array) -> float`` — per-section anchor used only when ``plan.contrast`` is
      ``SHARED`` (the shared per-ion window is ``max`` of these). Optional.
    * ``render_image(array, *, selection, section, anchor, path)`` — render + save one ion
      panel to ``path`` (anchor = shared per-ion value, or ``None`` for per-section auto).
    * ``write_outputs(plan, cells, anchors, out_dir) -> dict`` — write the non-per-cell
      outputs (matrix figure, list CSVs, analyses bundle); returns ``{label: path}``. Optional.
    * ``release(load_key)`` — free a cached cube. Optional (no-op for the live slide).

    **Parallel render (optional).** Rendering each cell is independent and CPU-bound (matplotlib
    rasterise + encode), so it's the batch's bottleneck. When ``max_workers > 1`` AND a picklable
    ``render_cell(image, spec, path)`` worker AND a main-thread ``build_spec(image, selection,
    section, anchor) -> picklable dict`` are supplied, the render phase fans out across a
    **spawn** process pool (spawn, not fork — the parent holds Qt threads). Extraction stays
    serial (one cube in RAM at a time); only the already-extracted 2-D cells are shipped to
    workers. Falls back to the serial ``render_image`` path for small batches or when these
    aren't provided, so existing callers are unchanged.

    The loop is **extract-all (source-outer, evicting each cube) → aggregate shared contrast →
    render every cell → write the other outputs → manifest**. Cancellation (via ``cancel()``)
    leaves a valid partial tree; the manifest records exactly what completed.
    """
    os.makedirs(out_dir, exist_ok=True)
    result = RunResult(out_dir=out_dir, jobs=list(plan.jobs))

    def _cancelled() -> bool:
        return bool(cancel and cancel())

    # ----- Phase A: extract every cell, one cube at a time ----------------- #
    # cells[(sid, sel_uid)] = ndarray (or absent → that cell is skipped at render time)
    cells: dict[tuple[str, str], object] = {}
    section_loaded: dict[str, bool] = {}              # sid → did its source load OK
    n_sel = max(len(plan.selections), 1)
    total_extract = len(plan.sections) * n_sel
    done = 0
    for _key, group in _group_by_source(plan.sections):
        if _cancelled():
            result.cancelled = True
            break
        for sec in group:
            ds = None
            try:
                ds = loader(sec)
            except Exception as e:  # noqa: BLE001 — a bad section must not abort the batch
                ds = None
                _note_section(result, sec, f"load failed: {type(e).__name__}: {e}")
            section_loaded[sec.sid] = ds is not None
            for sel in plan.selections:
                if _cancelled():
                    result.cancelled = True
                    break
                arr = None
                if ds is not None:
                    try:
                        arr = extract(ds, sec, sel)
                    except Exception as e:  # noqa: BLE001
                        arr = None
                        _note_section(result, sec, f"extract {sel.label}: {type(e).__name__}")
                if arr is not None:
                    cells[(sec.sid, sel.uid())] = arr
                done += 1
                if progress:
                    progress(done, total_extract, f"Reading {sec.name} · {_sel_label(sel)}")
            if result.cancelled:
                break
        if release:                                   # evict this source's cube before the next
            try:
                release(_key)
            except Exception:  # noqa: BLE001
                pass
        if result.cancelled:
            break

    # ----- Phase B: shared per-ion contrast anchor ------------------------- #
    # SHARED mode maps one absolute intensity to 100% for every section of an ion, so a faint
    # section can't be brightened to look like a strong one. The anchor is the pooled percentile
    # across all sections (hot-pixel-robust) — not each section's own max — so the window is a
    # true cross-section reference (see _shared_anchor).
    anchors: dict[str, float | None] = {}
    if ContrastMode(plan.contrast) == ContrastMode.SHARED:
        for sel in plan.selections:
            arrs = [cells.get((sec.sid, sel.uid())) for sec in plan.sections]
            anchors[sel.uid()] = _shared_anchor([a for a in arrs if a is not None],
                                                contrast_percentile, measure)
    else:
        anchors = {sel.uid(): None for sel in plan.selections}   # per-section auto
    result.contrast_anchors = anchors
    result.contrast_note = contrast_note

    # ----- Phase C: render every ion-image cell ---------------------------- #
    if not result.cancelled and OutputKind.ION_IMAGES in {OutputKind(o) for o in plan.outputs}:
        total = len(result.jobs)
        # Mark cells with no extracted image as skipped up front; collect the rest to render.
        renderable = []
        for job in result.jobs:
            arr = cells.get((job.section.sid, job.selection.uid()))
            if arr is None:
                job.status = "skipped"
                job.note = job.note or ("source unavailable"
                                        if not section_loaded.get(job.section.sid, False)
                                        else "ion absent / out of range")
                result.skipped += 1
            else:
                renderable.append((job, arr))
        done = result.skipped
        use_parallel = (render_cell is not None and build_spec is not None
                        and max_workers and int(max_workers) > 1
                        and len(renderable) >= _PARALLEL_MIN_JOBS)
        if use_parallel:
            result.cancelled = _render_parallel(
                renderable, build_spec, render_cell, anchors, out_dir, int(max_workers),
                result, on_path, progress, total, done, _cancelled)
        else:
            for job, arr in renderable:
                if _cancelled():
                    result.cancelled = True
                    break
                done += 1
                abspath = os.path.join(out_dir, job.path)
                os.makedirs(os.path.dirname(abspath), exist_ok=True)
                try:
                    render_image(arr, selection=job.selection, section=job.section,
                                 anchor=anchors.get(job.selection.uid()), path=abspath)
                    job.status = "done"
                    result.written.append(abspath)
                    if on_path:
                        on_path(abspath)
                except Exception as e:  # noqa: BLE001
                    job.status = "error"
                    job.note = f"{type(e).__name__}: {e}"
                    result.errors += 1
                if progress:
                    progress(done, total, f"Rendering {job.section.name} · {_sel_label(job.selection)}")

    # ----- Phase D: the non-per-cell outputs (matrix / CSV / analyses) ----- #
    if not result.cancelled and write_outputs is not None:
        try:
            extra = write_outputs(plan, cells, anchors, out_dir) or {}
            for label, path in extra.items():
                result.extra[label] = path
                if path and os.path.exists(path):
                    result.written.append(path)
        except Exception as e:  # noqa: BLE001
            result.extra["_error"] = f"{type(e).__name__}: {e}"

    # ----- Phase E: manifest + README -------------------------------------- #
    write_manifest(result, plan, out_dir)
    return result


def _sel_label(sel: SelectionRecord) -> str:
    if sel.kind == "composite":
        return sel.lipid_class or sel.label or "class"
    return sel.label or (f"m/z {sel.mz:.4f}" if sel.mz is not None else "feature")


def _note_section(result: RunResult, section: Section, note: str) -> None:
    """Attach a note to every planned job of ``section`` whose note is still empty."""
    for j in result.jobs:
        if j.section.sid == section.sid and not j.note:
            j.note = note


# --------------------------------------------------------------------------- #
# Manifest (pure stdlib — provenance for the whole batch)
# --------------------------------------------------------------------------- #
MANIFEST_FIELDS = ["section", "region", "list", "kind", "mz", "label", "file", "status", "note"]


def write_manifest(result: RunResult, plan: StudioPlan, out_dir: str) -> str:
    """Write ``manifest.csv`` (one row per planned cell + its status) and a short
    ``README.md`` describing the run, so the folder is self-documenting."""
    path = os.path.join(out_dir, "manifest.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for row in result.manifest_rows():
            w.writerow(row)
    result.written.append(path)

    done = sum(1 for j in result.jobs if j.status == "done")
    readme = os.path.join(out_dir, "README.md")
    contrast_desc = ContrastMode(plan.contrast).value.replace('_', ' ')
    if result.contrast_note:                          # honest record of the shared 100% reference
        contrast_desc += f" — {result.contrast_note}"
    lines = [
        "# Export Studio output", "",
        f"- Sections: {len(plan.sections)}",
        f"- Feature lists: {len({s.list_name for s in plan.selections})}",
        f"- Ions / composites: {len(plan.selections)}",
        f"- Contrast: {contrast_desc}",
        f"- Layout: {plan.layout.replace('_', '-')}",
        f"- Outputs: {', '.join(OutputKind(o).value for o in plan.outputs)}",
        "",
        f"Ion images: {done} written, {result.skipped} skipped, {result.errors} error(s).",
    ]
    if result.extra:
        lines += ["", "Other outputs:"] + [f"- {k}: {os.path.basename(str(v))}"
                                            for k, v in result.extra.items() if not k.startswith("_")]
    lines += ["", "See `manifest.csv` for the per-image status of every section × ion cell."]
    with open(readme, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    result.written.append(readme)
    return path
