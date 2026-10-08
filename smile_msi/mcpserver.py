"""Model Context Protocol server — drive the analysis engine from an AI assistant.

Run it with ``python -m smile_msi.mcpserver`` (stdio). An assistant that speaks the Model
Context Protocol (MCP) — a desktop assistant, say — then gets a small set of tools over the same
engine the desktop app runs: list the slides you have worked on, see the regions and groups
you drew, run any registered analysis against them, render an ion image, read back what an
earlier analysis found.

Two design rules the tools follow:

* **The app's state is the input.** Every tool reads the managed session the app auto-saves
  (:mod:`smile_msi.headless`), so an analysis runs on the regions, groups, features and
  extraction settings you actually left behind — not on a fresh guess at them.
* **The app's state is not the output.** Nothing here writes into your session files.
  Results land in their own folder (``~/.smile-msi/mcp-results``, override with
  ``$SMILE_MSI_MCP_OUT``) and the tool returns the path, so an assistant can never corrupt
  a session the app has open. The one hand-off is :func:`send_regions`, which drops regions
  into the session's inbox (:mod:`smile_msi.regioninbox`) for the app itself to add.

Result payloads are trimmed on purpose. A stats table comes back as a short preview plus
the path of the full CSV: an assistant pays for every row it is handed, and a 700-ion table
is worth exactly one glance and a file path.

The MCP SDK is imported lazily inside :func:`build_server`, so this module (and its tool
functions, which are plain Python) can be imported and tested without it installed.
"""
from __future__ import annotations

import os

from . import headless, library, registry

#: Rows of a result table returned inline; the rest lives in the CSV the tool points at.
PREVIEW_ROWS = 12

_open: dict = {"slide": None, "ref": ""}

#: References that open the synthetic demo slide (which has no session file).
_DEMO_REFS = ("demo", "demo dataset", "synthetic")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def results_dir() -> str:
    """Where result CSVs and PNGs are written. Never inside a session directory."""
    d = os.environ.get("SMILE_MSI_MCP_OUT") or os.path.join(library.home_dir(), "mcp-results")
    os.makedirs(d, exist_ok=True)
    return d


def _out_path(name: str, ext: str) -> str:
    import re

    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "result"
    base = os.path.join(results_dir(), stem)
    path, n = f"{base}{ext}", 1
    while os.path.exists(path):
        path, n = f"{base}-{n}{ext}", n + 1
    return path


def _round(value, digits: int = 6):
    """JSON-safe, token-cheap scalars: NaN/inf become None, floats lose their noise tail."""
    import math

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, digits)
    if hasattr(value, "item"):                       # numpy scalar
        try:
            return _round(value.item(), digits)
        except Exception:  # noqa: BLE001
            return str(value)
    return value


def _table_out(df, name: str, *, preview: int = PREVIEW_ROWS, sort: str = "") -> dict:
    """A DataFrame as {columns, n_rows, preview rows, csv path}. The CSV is the whole table;
    the preview is what an assistant should actually read."""
    if df is None:
        return {}
    if sort and sort in df.columns:
        df = df.sort_values(sort)
    path = _out_path(name, ".csv")
    df.to_csv(path, index=False)
    head = df.head(int(preview))
    return {
        "columns": [str(c) for c in df.columns],
        "n_rows": len(df),
        "preview": [{str(k): _round(v) for k, v in row.items()}
                    for row in head.to_dict("records")],
        "truncated": bool(len(df) > len(head)),
        "csv": path,
    }


def _slide():
    """The open slide, or a message saying how to get one."""
    if _open["slide"] is None:
        raise RuntimeError("no slide open — call open_slide first (open_slide('demo') needs "
                           "no data), or pass a slide reference.")
    return _open["slide"]


# --------------------------------------------------------------------------- #
# tools — what is there
# --------------------------------------------------------------------------- #
def list_slides() -> list[dict]:
    """Every slide the app has saved analysis state for, newest first.

    ``source_exists`` false means the imzML has moved or its drive is unplugged: the saved
    state is still readable, but the slide cannot be reopened until the file is back."""
    return [{"name": r.get("name"), "ref": os.path.basename(str(r["path"])).removesuffix(".json"),
             "source": r.get("source"), "source_exists": r.get("source_exists"),
             "n_pixels": r.get("n_pixels"), "n_features": r.get("n_features")}
            for r in headless.managed_sessions()]


def slide_state(ref: str) -> dict:
    """What has been done on a slide: extraction settings, preprocessing, named regions and
    their group tags, saved feature lists, the segmentation, and every analysis already run
    on it. Reads files only — instant, and it works while the slide's drive is unplugged.

    The analysis list comes from the run store on disk, which is the truth: the copy inside
    the session file is a snapshot from the last autosave and can miss a run that finished
    after it."""
    if str(ref).lower() in _DEMO_REFS:
        raise ValueError("'demo' is the synthetic slide: it has no saved session on disk. "
                         "Open it with open_slide('demo'), then call state() for its regions, "
                         "groups and settings.")
    path = headless.resolve_session_ref(ref)
    out = headless.session_summary(headless.read_session(path), path=path)
    out["analysis_runs"] = [
        {"run_id": r.run_id, "step_id": r.step_id, "title": r.title, "status": r.status,
         "created": r.created, "summary": r.summary, "params": r.params,
         "has_result": bool(r.result_ref)}
        for r in headless.run_store(path).list_runs()]
    return out


def analysis_catalog(category: str = "") -> list[dict]:
    """The analyses that can be run, with the inputs each needs and the parameters it takes.

    ``needs`` is what must exist before a step will run: ``feature_set`` (pick peaks first),
    ``groups`` / ``ab`` (regions tagged into groups), ``region``, ``target_mz``. Pass a step's
    ``id`` to :func:`run_analysis`."""
    out = []
    for step in registry.REGISTRY.values():
        if category and step.category.lower() != category.lower():
            continue
        out.append({
            "id": step.id, "name": step.name, "category": step.category,
            "needs": sorted(step.needs), "result": step.result_kind,
            "params": {p.name: {"kind": p.kind, "default": p.default,
                                **({"choices": p.choices} if p.choices else {})}
                       for p in step.params},
            "help": step.help,
        })
    return out


def scripting_guide() -> str:
    """The full scripting reference for the analysis engine — every bare-name function, every
    registered step, and worked examples. Read this before writing a script for
    :func:`run_script`. It is generated from the live API, so it never drifts."""
    from . import scripting

    slide = _open["slide"]
    return scripting.capabilities_doc(slide.api if slide is not None else None)


def run_result(ref: str, run_id: str) -> dict:
    """What an analysis already run in the app found — its saved result table, straight off
    disk. Nothing is recomputed. Run ids come from ``slide_state(...)['analysis_runs']``."""
    result = headless.run_result(ref, run_id)
    if result is None:
        return {"run_id": run_id, "result": None,
                "note": "this run kept no result payload"}
    if hasattr(result, "columns"):
        return {"run_id": run_id, "table": _table_out(result, f"{run_id}-{ref}")}
    if hasattr(result, "shape"):
        return {"run_id": run_id, "array_shape": list(result.shape)}
    return {"run_id": run_id, "result": result}


# --------------------------------------------------------------------------- #
# tools — open a slide and work on it
# --------------------------------------------------------------------------- #
def open_slide(ref: str = "demo", stride: int = 1) -> dict:
    """Open a slide for analysis and keep it open for later tool calls.

    ``ref`` is a slide from :func:`list_slides` (its ``ref`` or name), a path to an imzML or
    a session file, or ``'demo'`` for the synthetic slide that needs no data at all. The
    saved preprocessing is re-applied and the fast cube is restored from the session sidecar
    when one matches, so reopening a slide the app has worked on is quick; a slide with no
    sidecar pays one pass over the .ibd here instead.

    ``stride`` loads every Nth pixel for a quick look. Leave it 1 for real analysis — a
    strided slide has different pixel indices, so saved regions will not line up."""
    close_slide()
    if str(ref).lower() in _DEMO_REFS:
        slide = headless.open_demo()
    else:
        slide = headless.open_slide(ref, stride=int(stride))
    _open["slide"], _open["ref"] = slide, str(ref)
    return state()


def close_slide() -> dict:
    """Release the open slide and the memory its spectra hold."""
    was = _open["ref"]
    _open["slide"], _open["ref"] = None, ""
    return {"closed": was or None}


def state() -> dict:
    """The open slide's live working state — which regions and groups resolved to how many
    pixels, how many features are in the working set, what the extraction settings are."""
    slide = _slide()
    out = slide.summary()
    out["open"] = {"ref": _open["ref"], "source": slide.source,
                   "n_pixels": int(slide.ds.n_pixels),
                   "ppm": slide.api.ppm, "norm": slide.api.norm, "reduce": slide.api.reduce}
    return out


def find_peaks(snr: float = 3.0, max_peaks: int = 0, spatial: bool = False,
               min_morans: float = 0.0, region: str = "",
               max_candidates: int = 2000) -> dict:
    """Pick the working feature set — the m/z every later step defaults to.

    ``spatial=True`` runs the spatially-aware detector instead (signal-to-noise, then
    reproducibility across pixels, then Moran's I autocorrelation), which keeps ions that
    are actually structured in tissue rather than merely intense.

    ``region`` detects over one region's pixels only (the features are still extracted over
    the whole slide). ``max_peaks`` / ``max_candidates`` cap the list by descending
    intensity; ``0`` means no cap. The reply always reports ``n_detected`` — how many peaks
    were found before any cap — because a capped count is the cap, not a property of the
    data, and two regions that both report it have not been shown to agree."""
    api = _slide().api
    mask = region or None
    if spatial:
        res = api.find_spatial_features(snr=float(snr), min_morans=float(min_morans),
                                        max_candidates=int(max_candidates), mask=mask)
        peaks, n_detected = res.peaks, int(res.n_detected)
        funnel = {"n_candidates": int(res.n_candidates),
                  "n_after_frequency": int(res.n_after_frequency),
                  "n_after_morans": int(res.n_after_morans),
                  "candidates_capped": bool(res.candidates_capped)}
    else:
        peaks = api.find_peaks(snr=float(snr), max_peaks=int(max_peaks), mask=mask)
        n_detected = int(getattr(peaks, "n_detected", len(peaks)))
        funnel = {}
    mzs = [_round(p["mz"], 4) for p in peaks]
    return {"n_features": len(mzs), "n_detected": n_detected,
            "capped": n_detected > len(mzs), "region": region or "whole slide",
            "mz": mzs[:60], "truncated": len(mzs) > 60, **funnel,
            "note": "this is now the working feature set for later steps"}


def run_analysis(step_id: str, params: dict | None = None, features: list | None = None,
                 a: str = "", b: str = "", region: str = "", groups: list | None = None,
                 mask: str = "", target_mz: float = 0.0) -> dict:
    """Run one registered analysis on the open slide — the same step, the same engine, the
    same inputs the desktop app would use.

    Inputs default to the slide's state: the working feature set, the tagged groups (``a``
    and ``b`` default to the first two). Name regions or groups to override — ``a='Peri'``,
    ``b='Endo'``, ``mask='ROI 1'`` to restrict a step to one region. ``groups`` takes a list
    of region **or** group names (``groups=['core', 'rim', 'lesion']``) for multi-group
    steps; a pixel in two of them counts for the first. ``params`` are the step's own
    tunables; :func:`analysis_catalog` lists them with their defaults.

    Returns the step's one-line outcome plus a preview of its result table and the path to
    the full CSV. ``warning`` is set when the result carries a caveat — most often that a
    test ran on pixels, so its p/q values describe pixels rather than replicates. ``mask``
    echoes the region a step was restricted to and its pixel count."""
    slide = _slide()
    step = registry.REGISTRY.get(step_id)
    if step is None:
        raise ValueError(f"unknown analysis {step_id!r} — see analysis_catalog() "
                         f"({len(registry.REGISTRY)} steps).")
    kwargs = dict(params or {})
    if features:
        kwargs["features"] = [float(f) for f in features]
    if a and b:
        kwargs["a"], kwargs["b"] = a, b
    if region:
        kwargs["region"] = region
    if groups:
        kwargs["groups"] = list(groups)
    if mask:
        kwargs["mask"] = mask
    if target_mz:
        kwargs["target_mz"] = float(target_mz)

    result = slide.api.run_analysis(step_id, **kwargs)
    out = {"step_id": step_id, "name": step.name}
    try:
        out["summary"] = step.summary(result) if step.summary else ""
    except Exception:  # noqa: BLE001 — a missing summary hook must not lose the result
        out["summary"] = ""
    table = None
    try:
        table = step.to_table(result) if step.to_table else None
    except Exception:  # noqa: BLE001
        table = None
    if table is None and hasattr(result, "columns"):
        table = result
    if table is not None:
        out["table"] = _table_out(table, f"{step_id}", sort=_sort_column(table))
    warning = _result_warning(result, table)
    if warning:
        out["warning"] = warning
    if mask:
        m = slide.api._mask(mask)
        out["mask"] = {"name": mask, "n_pixels": int(m.sum())}
    if getattr(result, "labels", None) is not None:
        out["segmentation"] = _segmentation_out(slide, result, masked=bool(mask))
    out["working_features"] = len(slide.api.get_features())
    return out


def _result_warning(result, table) -> str:
    """The caveat a step attached to its result (``attrs['warning']`` on a table, or a
    ``warning`` attribute) — dropped by the CSV round-trip, so it is lifted out here."""
    for obj in (result, table):
        attrs = getattr(obj, "attrs", None)
        w = (attrs.get("warning") if isinstance(attrs, dict) else None) or getattr(obj, "warning", None)
        if isinstance(w, str) and w:
            return w
    return ""


def _segmentation_out(slide, seg, *, masked: bool) -> dict:
    """Cluster count, silhouette, how many pixels were clustered — and, for a run without a
    mask, which clusters look like off-tissue background, because a 2-cluster cut of a slide
    with background usually just separates tissue from background."""
    import numpy as np

    from . import spatial

    labels = np.asarray(seg.labels)
    out = {"n_clusters": int(getattr(seg, "n_clusters", 0)),
           "silhouette": _round(float(getattr(seg, "silhouette", float("nan"))), 4),
           "n_pixels_clustered": int((labels >= 0).sum())}
    if not masked:
        bg = spatial.background_clusters(slide.ds, seg, tol_ppm=slide.api.ppm)
        if bg:
            out["background_clusters"] = {str(c): n for c, n in bg.items()}
            out["note"] = (f"clusters {sorted(bg)} ({sum(bg.values())} px) look like off-tissue "
                           "background: no mask was given, so background pixels were "
                           "clustered too — pass mask='<tissue region>' to cluster tissue only")
    return out


def _sort_column(df) -> str:
    """Sort a result table by the column that makes its top rows the interesting ones."""
    for col in ("q_value", "p_value"):
        if col in df.columns:
            return col
    return ""


def run_script(code: str) -> dict:
    """Run a Python analysis script against the open slide — the escape hatch for anything
    the single-step tools do not cover (chained steps, custom masks, arithmetic on results).

    The script runs in the same namespace the app's Script Console gives you: ``ds`` is the
    slide, the analysis functions are bare names, and ``log`` / ``table`` / ``image`` /
    ``record`` surface results. Call :func:`scripting_guide` first — it is the reference for
    what is bound. Errors come back with your line numbers, not a swallowed traceback."""
    from . import scripting

    slide = _slide()
    result = scripting.run_script(code, slide.api)
    out = {
        "ok": bool(result.ok),
        "summary": result.summary(),
        "logs": list(result.logs)[-80:],
        "values": {k: _round(v) if not hasattr(v, "shape") else f"<array {v.shape}>"
                   for k, v in (result.values or {}).items()},
        "tables": [_table_out(df, title or "script-table")
                   for title, df in (result.tables or [])],
    }
    if getattr(result, "stdout", ""):                  # print() output
        out["stdout"] = result.stdout[-4000:]
    if result.error:
        out["error"] = result.error
    if result.images:
        out["images"] = [_save_image(img, title or "script-image", title=title)
                         for title, img, _meta in result.images]
    if slide.api.new_regions:
        out["staged_regions"] = [r["name"] for r in slide.api.new_regions]
    return out


def send_regions(names: list | None = None, groups: dict | None = None,
                 parents: dict | None = None) -> dict:
    """Hand regions staged with ``add_region(name, mask)`` in :func:`run_script` to the app.

    The app adds them to the slide (one undo step) within a few seconds if the slide is
    open there, or the next time it is opened. ``names`` picks which staged regions to send
    (default: all); ``groups`` tags them for comparisons, e.g. ``{"Sample 1 tissue": "Group A"}``;
    ``parents`` nests them under an existing region, e.g. ``{"S01 core": "Sample 01"}`` (a
    parent the slide does not have leaves the region top-level).
    A sent region is no longer staged, so sending twice does not add it twice."""
    from . import regioninbox

    slide = _slide()
    if not slide.session_path:
        raise RuntimeError("this slide has no saved session (the demo, or an imzML the app "
                           "has never opened) — there is nowhere to deliver regions to.")
    staged = slide.api.new_regions
    want = [str(n) for n in names] if names else [r["name"] for r in staged]
    by_name = {r["name"]: r for r in staged}
    missing = [n for n in want if n not in by_name]
    if missing:
        raise ValueError(f"not staged: {missing}. Staged: {sorted(by_name)} — stage a region "
                         "with add_region(name, mask) in run_script first.")
    groups = {str(k): str(v) for k, v in (groups or {}).items()}
    parents = {str(k): str(v) for k, v in (parents or {}).items()}
    send = [dict(by_name[n], group=groups.get(n, ""), parent=parents.get(n)) for n in want]
    path = regioninbox.post_regions(slide.session_path, send, n_pixels=int(slide.ds.n_pixels),
                                    source=slide.source)
    slide.api.new_regions = [r for r in staged if r["name"] not in want]
    return {"sent": [{"name": r["name"], "n_pixels": int(r["mask"].sum()),
                      "group": r["group"], "parent": r["parent"]} for r in send],
            "inbox_file": path,
            "note": "The app adds these within a few seconds if the slide is open, else on "
                    "next open. Names that already exist on the slide get a numeric suffix."}


# --------------------------------------------------------------------------- #
# tools — look at the data
# --------------------------------------------------------------------------- #
def _save_image(img, name: str, *, cmap: str = "viridis", title: str = "") -> str:
    from . import imaging

    path = _out_path(name, ".png")
    imaging.export_ion_figure(img, path, title=title or name, cmap=cmap)
    return path


def ion_image(mz: float, cmap: str = "viridis", mask: str = "") -> dict:
    """Render one ion's spatial distribution to a PNG and report where its signal sits.

    The image uses the slide's current tolerance, normalization and window reducer. The
    ``mean`` / ``max`` / ``nonzero_fraction`` statistics are over **every acquired pixel**,
    off-tissue background included, so a small tissue on a large raster reads low. ``mask``
    adds the same statistics inside a named region (and outside it) without changing the
    image, which is how you ask "how much of this ion is in the endoneurium?"."""
    import numpy as np

    slide = _slide()
    mz = float(mz)
    img = slide.api.ion_image(mz)
    path = _save_image(img, f"ion-{mz:.4f}", cmap=cmap, title=f"m/z {mz:.4f}")
    vec = slide.api.ion_vector(mz)
    out = {"mz": _round(mz, 4), "png": path,
           "shape": [int(img.shape[0]), int(img.shape[1])],
           "mean": _round(float(np.nanmean(vec))), "max": _round(float(np.nanmax(vec))),
           "nonzero_fraction": _round(float(np.mean(np.asarray(vec) > 0)), 4)}
    if mask:
        m = slide.api._mask(mask)
        if m is not None and m.any():
            inside = float(np.nanmean(np.asarray(vec)[m]))
            outside = float(np.nanmean(np.asarray(vec)[~m]))
            out["region"] = {"name": mask, "n_pixels": int(m.sum()),
                             "mean_inside": _round(inside), "mean_outside": _round(outside),
                             "ratio": _round(inside / outside if outside else float("nan"))}
    return out


def mean_spectrum(region: str = "", top_n: int = 25) -> dict:
    """The mean spectrum of the whole slide, or of one named region, as its strongest peaks.

    Returns the ``top_n`` most intense local maxima, **listed in m/z order** (not by
    intensity), with their m/z and relative intensity, plus a PNG of the spectrum — enough to
    see what is in the tissue before picking features. Each m/z is the same 3-point centroid
    :func:`find_peaks` reports, so a peak reads the same in both tools."""
    import numpy as np

    from .msi import _centroid

    slide = _slide()
    mask = slide.api._mask(region) if region else None
    axis, spec = slide.api.mean_spectrum(mask=mask)
    axis, spec = np.asarray(axis, float), np.asarray(spec, float)
    interior = np.arange(1, spec.size - 1)
    local = interior[(spec[1:-1] > spec[:-2]) & (spec[1:-1] >= spec[2:])]
    order = local[np.argsort(spec[local])[::-1][:int(top_n)]]
    top = sorted(order.tolist())
    peak = float(spec.max()) or 1.0
    return {
        "region": region or "whole slide",
        "n_points": int(spec.size),
        "mz_range": [_round(float(axis[0]), 4), _round(float(axis[-1]), 4)],
        "top_peaks": [{"mz": _round(_centroid(axis, spec, i), 4),
                       "rel_intensity": _round(float(spec[i]) / peak, 4)} for i in top],
        "png": _spectrum_png(axis, spec, region),
    }


def _spectrum_png(axis, spec, region: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = _out_path(f"spectrum-{region or 'slide'}", ".png")
    fig, ax = plt.subplots(figsize=(9, 3), dpi=140)
    ax.plot(axis, spec, lw=0.6, color="#222")
    ax.set_xlabel("m/z")
    ax.set_ylabel("mean intensity")
    ax.set_title(f"Mean spectrum — {region or 'whole slide'}")
    ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def annotate(mz: list | None = None, mode: str = "negative", match_ppm: float = 5.0) -> dict:
    """Identify m/z as lipids against the in-silico database (sum-composition level).

    Defaults to the working feature set; pass ``mz`` to identify specific ions. The table
    carries the best match per ion plus its adduct, ppm error, isotope-pattern and adduct
    corroboration, a folded confidence label, and the annotation-FDR estimate."""
    slide = _slide()
    df = slide.api.annotate(features=([float(m) for m in mz] if mz else None),
                            mode=str(mode), match_ppm=float(match_ppm))
    return {"mode": mode, "match_ppm": match_ppm,
            "table": _table_out(df, f"annotate-{mode}")}


def export_feature_table(ref: str = "", tol_ppm: float = 5.0) -> dict:
    """One wide CSV: the slide's feature list as the spine, with every analysis already run
    on it joined on by m/z as its own namespaced block of columns. This is the table to hand
    to R, Excel or pandas — no further joining needed."""
    from . import featuretable

    path = headless.resolve_session_ref(ref or _open["ref"])
    data = headless.read_session(path)
    store = headless.run_store(path)
    records = store.list_runs()
    if not records:
        return {"note": "no analyses have been logged on this slide yet", "csv": None}

    attachments = featuretable.attachments_from_runs(records, store.load_result)
    joined, notes = featuretable.join_by_mz(_feature_spine(data), attachments,
                                            tol_ppm=float(tol_ppm))
    out = _table_out(joined, "feature-table", preview=6)
    out["attached"] = notes
    out["skipped"] = [{"title": a.title, "why": a.reason}
                      for a in attachments if not a.usable]
    return out


def _feature_spine(data: dict):
    import pandas as pd

    peaks = data.get("peaks") or []
    return pd.DataFrame({"mz": [float(p.get("mz", 0.0)) for p in peaks],
                         "rel_intensity": [float(p.get("rel_intensity", 0.0)) for p in peaks]})


#: The tool surface: ``(function, human title, read-only)``. Read-only tools touch nothing —
#: they read session files or compute from the open slide; the rest open datasets, write
#: result files under :func:`results_dir`, or run arbitrary user code. None of them write
#: into a session, so no tool is ever marked destructive.
TOOLS = (
    (list_slides, "List slides", True),
    (slide_state, "Slide state", True),
    (analysis_catalog, "Analysis catalogue", True),
    (scripting_guide, "Scripting guide", True),
    (run_result, "Read a past result", True),
    (open_slide, "Open slide", False),
    (state, "Open slide state", True),
    (find_peaks, "Find peaks", False),
    (run_analysis, "Run analysis", False),
    (run_script, "Run analysis script", False),
    (send_regions, "Send regions to the app", False),
    (ion_image, "Ion image", False),
    (mean_spectrum, "Mean spectrum", False),
    (annotate, "Identify lipids", False),
    (export_feature_table, "Export feature table", False),
    (close_slide, "Close slide", False),
)

INSTRUCTIONS = """Drive SMILE MSI, a MALDI mass-spectrometry-imaging workspace, over the same
engine its desktop app uses. Start with list_slides to see the slides the user has analysed
and slide_state to see the regions, groups and analyses already on one — both read the app's
saved session files and need no data loaded. open_slide('demo') gives a synthetic slide for
trying anything out. Then find_peaks to establish a feature set, run_analysis for any step in
analysis_catalog, and run_script (read scripting_guide first) for anything else. Tables come
back as a short preview plus a CSV path; read the CSV when you need the rest. Nothing here
writes into the user's session files; to give the user a region, stage it with
add_region(name, mask) in run_script, then send_regions — the app adds it to the slide."""


def build_server():
    """The MCP server with every tool in :data:`TOOLS` registered.

    The SDK is imported here, not at module scope, so the tool functions above stay
    importable (and testable) with no ``mcp`` installed."""
    try:
        from mcp.server import MCPServer
        from mcp.types import ToolAnnotations
    except ImportError as exc:  # pragma: no cover — install-time error path
        raise SystemExit("the MCP server needs the 'mcp' package:  "
                         "uv pip install -e '.[mcp]'") from exc

    from . import __version__

    server = MCPServer("smile-msi", title="SMILE MSI", version=__version__,
                       instructions=INSTRUCTIONS)
    import inspect

    for fn, title, read_only in TOOLS:
        server.add_tool(_guarded(fn), title=title,
                        # the SDK passes __doc__ through verbatim, so an indented
                        # continuation line reaches the assistant with its indentation
                        description=inspect.cleandoc(fn.__doc__ or ""),
                        annotations=ToolAnnotations(read_only_hint=read_only,
                                                    destructive_hint=False))
    return server


def _guarded(fn):
    """Report a failure as the SDK's ``ToolError``, which is the only failure the assistant
    is shown. Any other exception reaches it as "Error executing tool <name>" with the real
    reason left in the server log — a message nothing can act on. These tools fail for
    ordinary, correctable reasons (no slide open, unknown region, features not picked yet),
    so the message *is* the useful part."""
    import functools

    from mcp.server.mcpserver.exceptions import ToolError

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001 — every failure is reported, none swallowed
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc

    return wrapper


def main() -> None:
    """stdio entry point: ``python -m smile_msi.mcpserver``.

    stdout *is* the protocol on stdio, so nothing may print to it. The SDK diverts flushed
    stray output to stderr, but an import-time print that drains at exit would still land on
    the wire and drop the connection."""
    build_server().run("stdio")


if __name__ == "__main__":
    main()
