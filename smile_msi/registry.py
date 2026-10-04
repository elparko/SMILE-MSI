"""Analysis registry — the catalogue of analyses the app can run.

This module is the **pure** core (dict/array in, dataclass/DataFrame out — no Qt) that both
the Analyze gallery / :class:`~smile_msi.gui.analysisdialog.AnalysisDialog` and the scripting
API (:mod:`smile_msi.scripting`) drive:

* :data:`REGISTRY` — the catalogue of analyses. Each :class:`StepDef` declares its parameters
  (so the GUI can auto-build a form), what data inputs it needs, and small pure callables that
  wire the existing engine functions and shape their results (``run`` / ``peaks`` /
  ``produce`` / ``to_table`` / ``rep_ions`` / ``summary``).
* :func:`registry_by_category`, :func:`default_params`, :func:`unmet_needs`,
  :func:`resolve_inputs` — the pure helpers the gallery and dialog read.

Engine modules (spatial / multivariate / annotate / imaging) are imported **lazily inside
the step callables**, never at module top, so importing this module (which the GUI does at
startup) stays cheap and never pulls scipy/sklearn.

The Flow *designer* that once wrapped this registry into a saveable pipeline was retired in
plan 24; batch/replay lives on in the scripting API. A caller resolves each step's data inputs
into an ``inputs`` dict and calls ``step_def.run(ds, inputs, params)``; the ``inputs`` contract:

    mzs        list[float]   feature set (m/z) the step operates on
    mask       bool[n_pix]   optional region scope (None = whole slide)
    labels     int[n_pix]    per-pixel group ids (>=0; -1 = unassigned) — groups steps
    names      list[str]     group names aligned to label ids
    mask_a/b   bool[n_pix]   the two group masks — A/B steps
    a_label/b_label  str     the two group names
    samples    int[n_pix]    per-pixel replicate ids for ROI-as-replicate stats
    target_mz  float         the co-localization target ion
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Data-input tokens a step can require (drive Setup gating + input resolution in the GUI).
#   feature_set — an m/z set;  groups — per-pixel group labels (>=2 groups);
#   ab — two group masks A/B (+ per-ROI replicate ids);  region — a single group's mask;
#   ab_multi — a list of (A vs B) region pairs (the Region marker panel);
#   target_mz — the co-localization target ion;
#   stats — a stats table (AUC column) produced by an earlier step in the same flow.
NEEDS = {"feature_set", "groups", "ab", "ab_multi", "region", "target_mz", "stats"}


# --------------------------------------------------------------------------- #
# parameter + step-type schema
# --------------------------------------------------------------------------- #
@dataclass
class ParamSpec:
    """One field in a step's auto-generated parameter form."""
    name: str
    label: str
    kind: str                       # "float" | "int" | "choice" | "bool" | "mz"
    default: object = None
    lo: float | None = None
    hi: float | None = None
    step: float | None = None
    choices: list | None = None
    help: str = ""


@dataclass
class StepDef:
    """A kind of analysis a flow step can be — the single source of truth driving the
    palette menu, the parameter form, the runner, and result routing."""
    id: str
    name: str
    category: str
    needs: set = field(default_factory=set)
    produces: set = field(default_factory=set)
    params: list = field(default_factory=list)
    targets: set = field(default_factory=lambda: {"slide"})
    result_kind: str = "table"      # features|segmentation|stats|table|components|classifier|embedding
    view: str = "Ion image"         # reveal_view() label to surface the result live
    help: str = ""
    uses_mask: bool = False         # run() honours inp["mask"] → offer a per-step "Restrict to ROI"
    description: str = ""           # long-form blurb for the Analyze gallery card (falls back to help)
    presentation: str = "dialog"    # "dialog" (transactional ConfigureRunDialog) | "screen" (own tab)
    # pure wiring callables (engine imports happen lazily inside each)
    run: object = None              # (ds, inputs, params) -> raw engine result
    peaks: object = None            # (result) -> list[peak dict] | None  (working-set commit)
    produce: object = None          # (result) -> dict   (extra ctx keys for later steps)
    to_table: object = None         # (result) -> DataFrame | None  (logging + bundle CSV)
    rep_ions: object = None         # (result, n, ds=None) -> [(mz, label, score)]  (bundle images)
    summary: object = None          # (result) -> str   (one-line outcome)
    lists: object = None            # (result) -> {list_name: [peak dict]}  (named ★ lists to save)
    regions_out: object = None      # (result, ds) -> {region_name: bool[n_pix]}  (regions to create)

    def __post_init__(self):
        if not self.description:
            self.description = self.help
        if self.peaks is None:
            self.peaks = lambda r: None
        if self.produce is None:
            self.produce = lambda r: {}
        if self.to_table is None:
            self.to_table = lambda r: None
        if self.rep_ions is None:
            self.rep_ions = lambda r, n, ds=None: []
        if self.summary is None:
            self.summary = lambda r: ""
        if self.lists is None:
            self.lists = lambda r: {}
        if self.regions_out is None:
            self.regions_out = lambda r, ds=None: {}


# --------------------------------------------------------------------------- #
# result shapers — pure helpers over engine return values
# --------------------------------------------------------------------------- #
def _mz_list(peaks):
    """m/z floats from a peak list (dicts or bare floats)."""
    out = []
    for p in peaks:
        out.append(float(p["mz"]) if isinstance(p, dict) else float(p))
    return out


def _df(rows):
    import pandas as pd
    return pd.DataFrame(rows)


def _auc_effect(s):
    """Effect size of an AUC column: ``|AUC - 0.5|`` (distance from the chance level 0.5);
    higher = more discriminating. Shared by the AUC-based representative-ion pickers."""
    return (s.astype(float) - 0.5).abs()


def _rep_by_auc(df, n, col="AUC"):
    if df is None or len(df) == 0:
        return []
    d = df.copy()
    d = d[d[col].notna()]                       # degenerate contrast → all-NaN; emit no images
    if d.empty:
        return []
    d["__s"] = _auc_effect(d[col])
    d = d.sort_values("__s", ascending=False).head(int(n))
    return [(float(r["mz"]), "", float(r[col])) for _, r in d.iterrows()]


def _rep_by_pvalue(df, n, ds=None):
    if df is None or len(df) == 0:
        return []
    d = df.sort_values("p_value", kind="stable").head(int(n))
    score = "q_value" if "q_value" in d.columns else "p_value"
    return [(float(r["mz"]), "", float(r[score])) for _, r in d.iterrows()]


def _region_name(dct, cl):
    """Human label for a per-cluster key — the group name the dict carries (e.g. 'Trt'),
    else 'region <n>'. ``dct`` may be a plain dict or a :class:`_DiscResult`."""
    names = getattr(dct, "names", None)
    try:
        if names and 0 <= int(cl) < len(names):
            return str(names[int(cl)])
    except (TypeError, ValueError):
        pass
    return f"region {int(cl)}" if isinstance(cl, (int, float)) else str(cl)


def _dgmm_table(r):
    """The per-ion DGMM result as a small zone→mean table (for the report + flow bundle CSV).
    The labelled image itself is rendered by ``resultviews`` — this is only the tabular view."""
    import pandas as pd
    means = r.get("means") or []
    return pd.DataFrame({"level": list(range(len(means))),
                         "mz": [float(r.get("mz", 0.0))] * len(means),
                         "mean_intensity": [float(m) for m in means]})


def _per_cluster_table(dct):
    import pandas as pd
    frames = []
    for cl, df in dct.items():
        d = df.copy()
        d.insert(0, "region", _region_name(dct, cl))
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else None


def _per_cluster_rep(dct, n, ds=None):
    rows = []
    k = max(1, len(dct))
    per = max(1, int(n) // k)
    for cl, df in dct.items():
        d = df.copy()
        if "AUC" in d.columns:
            d["__s"] = _auc_effect(d["AUC"])
            col = "AUC"
        elif "shrunken" in d.columns:
            d["__s"] = d["shrunken"].astype(float).abs()
            col = "shrunken"
        else:
            d["__s"] = 0.0
            col = d.columns[1]
        label = _region_name(dct, cl)
        for _, r in d.sort_values("__s", ascending=False).head(per).iterrows():
            rows.append((float(r["mz"]), label, float(r.get(col, 0.0) or 0.0)))
    return rows[:int(n)]


class _NamedGroups(dict):
    """A ``{cluster_id: DataFrame}`` per-region result that remembers the group ``names``
    aligned to each label id, so every per-region table/list reads the group name (e.g.
    'Trt') rather than 'region 0'. ``names`` stays None on the segmentation fallback (no
    user-assigned groups), where ``_region_name`` labels the clusters 'region <n>'."""
    names = None


class _DiscResult(_NamedGroups):
    """A discriminating :class:`_NamedGroups` result that also remembers the enrichment
    filter used (AUC / q-value cutoffs) so the kept-marker save mirrors the Stats tab."""
    min_auc = 0.5
    max_q = 1.0
    enriched_only = True
    _save_lists = False


def _disc_lists(r):
    """Per-region marker lists ``{f'{name} markers': [{mz, lipid, note}]}`` from a filtered
    discriminating result — mirrors the Stats tab's 'Build feature list' save."""
    out = {}
    for cl, df in r.items():
        if df is None or not len(df):
            continue
        feats = []
        for _, x in df.iterrows():
            tag = (f"AUC {float(x['AUC']):.2f} · q {float(x['q_value']):.1e} · "
                   f"log2 FC {float(x['log2_fc']):+.2f}")
            feats.append({"mz": float(x["mz"]), "lipid": "", "note": tag})
        out[f"{_region_name(r, cl)} markers"] = feats
    return out


def _disc_survivors(r):
    """Union of every region's surviving m/z → the working set the next step runs on."""
    mzs = sorted({float(z) for df in r.values() if df is not None for z in df["mz"]})
    return {"peaks": mzs} if mzs else {}


# --------------------------------------------------------------------------- #
# step run/wiring callables  (engine imports are lazy, inside each)
# --------------------------------------------------------------------------- #
def _run_find_spatial(ds, inp, p):
    from . import spatial
    # ``inputs["progress"]`` is the worker's reporter when the caller has one (AnalysisDialog);
    # absent for scripted runs. Without it the finder's long tail is silent.
    return spatial.find_spatial_features(
        ds, snr=p["snr"], min_rel_intensity=p.get("min_rel_intensity", 0.0),
        min_frequency=p["min_frequency"], min_morans=p["min_morans"],
        max_candidates=int(p.get("max_candidates", 2000)),
        tol_ppm=p["tol_ppm"], norm=p["norm"], mask=inp.get("mask"),
        progress=inp.get("progress"))


def _run_find_coherent(ds, inp, p):
    from . import spatial
    return spatial.find_coherent_features(
        ds, snr=p["snr"], min_rel_intensity=p.get("min_rel_intensity", 0.0),
        min_frequency=p["min_frequency"], min_morans=p.get("min_morans", 0.0),
        min_quality=p["min_quality"], max_hotspot=p["max_hotspot"],
        max_candidates=int(p.get("max_candidates", 2000)),
        tol_ppm=p["tol_ppm"], norm=p["norm"], mask=inp.get("mask"),
        progress=inp.get("progress"))


def _run_find_peaks(ds, inp, p):
    return ds.pick_peaks(snr=p["snr"], min_rel_intensity=p.get("min_rel_intensity", 0.0),
                         max_peaks=int(p.get("max_peaks", 0)),   # 0 = no cap
                         prominence=p.get("prominence", 1.0), mask=inp.get("mask"))


def _run_segment(ds, inp, p):
    n = int(p.get("n_clusters", 0) or 0)
    nc = n if n >= 2 else None
    mzs = inp["mzs"]
    mask = inp.get("mask")
    if p.get("spatial", True):
        from . import multivariate
        return multivariate.spatial_segment(ds, mzs, n_clusters=nc, tol_ppm=p["tol_ppm"],
                                             norm=p["norm"], spatial_sigma=p.get("spatial_sigma", 1.0),
                                             mask=mask)
    from . import spatial
    if nc is None:
        return spatial.auto_segment(ds, mzs, tol_ppm=p["tol_ppm"], norm=p["norm"], mask=mask)
    return spatial.segment(ds, mzs, n_clusters=nc, tol_ppm=p["tol_ppm"], norm=p["norm"], mask=mask)


def _run_roi_comparison(ds, inp, p):
    from . import spatial
    df = spatial.roi_comparison(
        ds, inp["mask_a"], inp["mask_b"], inp["mzs"], tol_ppm=p["tol_ppm"],
        norm=p.get("norm", "tic"), a_label=inp.get("a_label", "Group A"),
        b_label=inp.get("b_label", "Group B"), samples=inp.get("samples"),
        method=p.get("method", "mwu"))
    df.attrs["_save_lists"] = bool(p.get("save_lists", True))
    df.attrs["_save_q"] = float(p.get("max_q", 0.05))
    return df


def _run_region_comparison(ds, inp, p):
    """Two regions' mean spectra (full profile) + their difference — the Region-comparison
    overlay/diff view, as a render-ready dict."""
    import numpy as np
    a, b = inp["mask_a"], inp["mask_b"]
    fa = ds.cube_mean_spectrum(a)
    axis, spec_a = fa if fa is not None else ds.mean_spectrum(mask=a)
    fb = ds.cube_mean_spectrum(b)
    _, spec_b = fb if fb is not None else ds.mean_spectrum(mask=b)
    return {"axis": np.asarray(axis, dtype=float), "spec_a": np.asarray(spec_a, dtype=float),
            "spec_b": np.asarray(spec_b, dtype=float),
            "a_label": inp.get("a_label", "A"), "b_label": inp.get("b_label", "B")}


def _roi_cmp_lists(r):
    """Per-side distinguishing-ion lists from an A-vs-B comparison: ``{a_label} distinguishing
    ions`` (AUC < 0.5, higher in A) and ``{b_label} distinguishing ions`` (AUC > 0.5), each
    filtered to q ≤ the step's cutoff. Mirrors the Stats-tab right-click 'Build feature list'."""
    if r is None or not getattr(r, "attrs", None) or not r.attrs.get("_save_lists"):
        return {}
    a = r.attrs.get("a_label", "Group A")
    b = r.attrs.get("b_label", "Group B")
    maxq = float(r.attrs.get("_save_q", 0.05))
    d = r[r["q_value"].astype(float) <= maxq] if maxq < 1.0 else r
    out = {}
    for label, keep in ((a, d["AUC"].astype(float) < 0.5), (b, d["AUC"].astype(float) > 0.5)):
        side = d[keep]
        feats = [{"mz": float(x["mz"]), "lipid": "",
                  "note": (f"distinguishes {a} vs {b} · AUC {float(x['AUC']):.2f} · "
                           f"log2 FC {float(x['log2_fc']):+.2f} · q {float(x['q_value']):.1e}")}
                 for _, x in side.iterrows()]
        if feats:
            out[f"{label} distinguishing ions"] = feats
    return out


def _classes_for_flow(mzs, p):
    """Per-peak lipid classes for the class roll-ups, via a fresh Annotator built from the
    step's identification params (mode + tolerance) — the same labels the GUI's annotator
    assigns. Lazy import keeps the lipid database out of flow import time."""
    from .match import Annotator
    ann = Annotator(mode=p.get("mode", "negative"), ppm_tol=float(p.get("id_ppm", 5.0)))
    return ann.classes_for(mzs)


def _run_class_comparison(ds, inp, p):
    from . import spatial
    mzs = inp["mzs"]
    return spatial.class_comparison(
        ds, inp["mask_a"], inp["mask_b"], mzs, _classes_for_flow(mzs, p),
        tol_ppm=p["tol_ppm"], norm=p.get("norm", "tic"),
        a_label=inp.get("a_label", "Group A"), b_label=inp.get("b_label", "Group B"),
        samples=inp.get("samples"), method=p.get("method", "mwu"))


def _run_class_composition(ds, inp, p):
    from . import spatial
    mzs = inp["mzs"]
    return spatial.class_composition(
        ds, [inp["mask_a"], inp["mask_b"]], mzs, _classes_for_flow(mzs, p),
        names=[inp.get("a_label", "Group A"), inp.get("b_label", "Group B")],
        tol_ppm=p["tol_ppm"], norm=p.get("norm", "tic"))


def _run_discriminating(ds, inp, p):
    from . import spatial
    raw = spatial.discriminating_features(ds, inp["labels"], inp["mzs"], tol_ppm=p["tol_ppm"],
                                          norm=p["norm"], top_n=int(p.get("top_n", 50)))
    res = _DiscResult(raw)
    res.names = inp.get("names")
    res.enriched_only = bool(p.get("enriched_only", True))
    res.min_auc = float(p.get("min_auc", 0.5))
    res.max_q = float(p.get("max_q", 0.05))
    res._save_lists = bool(p.get("save_lists", False))
    for cl, df in list(res.items()):                      # keep markers: enriched + significant
        d = df
        if res.enriched_only:
            d = d[d["AUC"].astype(float) >= res.min_auc]
        if res.max_q < 1.0:
            d = d[d["q_value"].astype(float) <= res.max_q]
        res[cl] = d.reset_index(drop=True)
    return res


def _run_multigroup(ds, inp, p):
    from . import spatial
    return spatial.multigroup_features(ds, inp["labels"], inp["mzs"], tol_ppm=p["tol_ppm"],
                                       norm=p["norm"], method=p.get("method", "kruskal"),
                                       names=inp.get("names"))


def _run_roi_localization(ds, inp, p):
    from . import spatial
    return spatial.roi_localization(ds, inp["mask_a"], inp["mzs"], tol_ppm=p["tol_ppm"],
                                    norm=p["norm"])


def _run_shrunken(ds, inp, p):
    from . import spatial
    raw = spatial.shrunken_centroids(ds, inp["labels"], inp["mzs"], shrink=p.get("shrink", 2.0),
                                     tol_ppm=p["tol_ppm"], norm=p["norm"], top_n=int(p.get("top_n", 15)))
    res = _NamedGroups(raw)                          # carry the group names into the per-group table
    res.names = inp.get("names")
    return res


def _run_dgmm(ds, inp, p):
    """Spatially-aware Gaussian-mixture segmentation of ONE ion image into intensity zones.

    The ion is the active m/z (``inp["target_mz"]``, resolved from ``state["target_mz"]`` — the
    ``target_mz`` need); levels + normalization are step params. Wraps
    ``multivariate.spatial_dgmm`` directly, returning a render-ready payload (the labelled
    image, the sorted component means, and the m/z)."""
    from . import multivariate
    mz = float(inp["target_mz"])
    labels, means = multivariate.spatial_dgmm(ds, mz, k=int(p.get("k", 3)), norm=p["norm"])
    return {"image": ds.to_image(labels.astype(float)),
            "means": [float(m) for m in means], "mz": mz, "k": int(p.get("k", 3))}


def _run_membership(ds, inp, p):
    from . import spatial
    return spatial.region_membership(ds, inp["masks"], inp["mzs"], names=inp.get("names"),
                                     tol_ppm=p["tol_ppm"], norm=p["norm"],
                                     min_prevalence=p.get("min_prevalence", 0.5))


def _run_colocalize(ds, inp, p):
    from . import spatial
    return spatial.colocalize(ds, float(p["target_mz"]), inp["mzs"], tol_ppm=p["tol_ppm"],
                              norm=p["norm"], method=p.get("method", "pearson"), mask=inp.get("mask"))


def _run_coloc_modules(ds, inp, p):
    from . import spatial
    nm = int(p.get("n_modules", 0) or 0)
    return spatial.coloc_modules(ds, inp["mzs"], tol_ppm=p["tol_ppm"], norm=p["norm"],
                                 method=p.get("method", "pearson"),
                                 n_modules=(nm if nm >= 2 else None),
                                 threshold=p.get("threshold", 0.5), mask=inp.get("mask"))


def _run_region_correlation(ds, inp, p):
    from . import spatial
    res = spatial.region_correlation(ds, inp["region_masks"], inp["mzs"],
                                     tol_ppm=p["tol_ppm"], norm=p["norm"],
                                     method=p.get("method", "pearson"))
    try:
        res.region_masks = dict(inp["region_masks"])       # kept for the create-matched-regions action
    except Exception:                                       # noqa: BLE001
        pass
    return res


def _region_corr_table(res):
    """Each region's single closest other region — the 'which regions look alike' takeaway
    that folds the old Region-match view into Region correlation."""
    rows = []
    for name in res.names:
        bm = res.best_match(name)
        rows.append({"region": name,
                     "closest match": (bm[0] if bm else "—"),
                     "similarity": (round(float(bm[1]), 4) if bm else float("nan"))})
    return _df(rows)


def _run_pca(ds, inp, p):
    from . import multivariate
    return multivariate.pca_images(ds, inp["mzs"], n_components=int(p.get("n_components", 5)),
                                   tol_ppm=p["tol_ppm"], norm=p["norm"], mask=inp.get("mask"))


def _run_nmf(ds, inp, p):
    from . import multivariate
    return multivariate.nmf_images(ds, inp["mzs"], n_components=int(p.get("n_components", 5)),
                                   tol_ppm=p["tol_ppm"], norm=p["norm"], mask=inp.get("mask"))


def _run_embedding(ds, inp, p):
    from . import multivariate
    return multivariate.embedding(ds, inp["mzs"], method=p.get("method", "umap"),
                                  tol_ppm=p["tol_ppm"], norm=p["norm"])


def _run_plsda(ds, inp, p):
    from . import multivariate
    return multivariate.plsda(ds, inp["mzs"], inp["labels"], n_components=int(p.get("n_components", 2)),
                              tol_ppm=p["tol_ppm"], norm=p["norm"], orthogonal=p.get("orthogonal", False))


def _run_classify_cv(ds, inp, p):
    from . import multivariate, spatial
    samples = spatial.sample_labels(ds)                    # leave-one-tissue-out when resolvable
    if len({int(s) for s in samples if s >= 0}) < 2:
        samples = None                                     # else stratified CV (engine flags it optimistic)
    r = multivariate.cross_validate(
        ds, inp["mzs"], inp["labels"], samples=samples,
        n_components=int(p.get("n_components", 2)), tol_ppm=p["tol_ppm"], norm=p["norm"],
        orthogonal=p.get("orthogonal", False), n_folds=int(p.get("n_folds", 5)))
    # Class ids are integer group labels; relabel with the group names so the confusion-matrix
    # rows/columns read 'Normal'/'Trt' not '0'/'1'. The matrix stays index-aligned to this list.
    names = inp.get("names")
    if names is not None:
        r["classes"] = [names[int(c)] if 0 <= int(c) < len(names) else str(c)
                        for c in (r.get("classes") or [])]
    return r


def _cv_table(r):
    """The confusion matrix (true rows × predicted cols) as a table; accuracy/fold detail go
    to the summary line."""
    classes = [str(c) for c in (r.get("classes") or [])]
    conf = r.get("confusion")
    rows = []
    if conf is not None:
        import numpy as np
        conf = np.asarray(conf)
        for i, c in enumerate(classes):
            row = {"true \\ pred": c}
            row.update({classes[j]: int(conf[i, j]) for j in range(len(classes))})
            rows.append(row)
    return _df(rows)


def _run_classify_map(ds, inp, p):
    from . import multivariate, spatial
    import numpy as np
    clf = multivariate.plsda(ds, inp["mzs"], inp["labels"], n_components=int(p.get("n_components", 2)),
                             tol_ppm=p["tol_ppm"], norm=p["norm"], orthogonal=p.get("orthogonal", False))
    X = spatial.feature_matrix(ds, clf.peaks, tol_ppm=p["tol_ppm"], norm=p["norm"])
    pred = clf.predict(X)                                  # a class label per pixel
    idx = {c: i for i, c in enumerate(clf.classes)}
    lab = np.array([idx.get(c, -1) for c in pred], dtype=int)
    # Display the group name per class ('predicted Normal') not the raw id ('predicted 0'). Only
    # the label strings change — `lab`/`idx` stay keyed on clf.classes positions, so the written-
    # back regions still map to the right pixels.
    names = inp.get("names")
    disp = [str(names[int(c)]) if names is not None and 0 <= int(c) < len(names) else str(c)
            for c in clf.classes]
    return {"image": ds.to_image(lab.astype(float)), "means": [], "mz": 0.0,
            "classes": disp,
            "labels": lab.tolist()}                        # per-pixel class index → save-as-regions


def _classify_map_regions(r, ds=None):
    """One region per predicted class, from the per-pixel class map."""
    import numpy as np
    lab = np.asarray(r.get("labels") if isinstance(r, dict) else [], dtype=int)
    classes = (r.get("classes") if isinstance(r, dict) else []) or []
    out = {}
    for i, c in enumerate(classes):
        mask = (lab == i)
        if mask.any():
            out[f"predicted {c}"] = mask
    return out


def _region_correlation_regions(r, ds=None):
    """Merge each region with its single closest match (similarity ≥ 0.5) into a '{a} + {b}'
    region — the create-matched-regions write-back, deduped by unordered pair."""
    out, seen = {}, set()
    for name in getattr(r, "names", []):
        bm = r.best_match(name)
        if not bm or float(bm[1]) < 0.5:
            continue
        pair = tuple(sorted((str(name), str(bm[0]))))
        if pair in seen:
            continue
        seen.add(pair)
        masks = getattr(r, "region_masks", None)
        if not masks:
            continue
        import numpy as np
        a, b = masks.get(pair[0]), masks.get(pair[1])
        if a is None or b is None:
            continue
        out[f"{pair[0]} + {pair[1]}"] = (np.asarray(a, dtype=bool) | np.asarray(b, dtype=bool))
    return out


def _run_shap(ds, inp, p):
    from . import explain
    return explain.shap_importance(
        ds, inp["mzs"], inp["labels"], names=inp.get("names"),
        n_estimators=int(p.get("n_estimators", 300)),
        max_pixels=int(p.get("max_pixels", 20000)),
        tol_ppm=p["tol_ppm"], norm=p["norm"])


def _shap_rep_ions(r, n, ds=None):
    """Top biomarker ions across all regions (best per-ion importance) for the bundle."""
    import numpy as np
    best = np.max(r.importance, axis=0)
    order = np.argsort(-best)[:int(n)]
    return [(float(r.peaks[j]), "", float(best[j])) for j in order]


def _run_annotate(ds, inp, p):
    from . import annotate
    return annotate.build_feature_list(ds, inp["mzs"], mode=p.get("mode", "negative"),
                                       match_ppm=p.get("match_ppm", 5.0),
                                       image_ppm=p.get("image_ppm", 10.0), norm=p["norm"])


def _auc_keep(auc, direction, cut):
    """Boolean keep-mask over an AUC Series for a ``direction`` + cutoff — shared by the
    'Keep features by AUC' step and the Region marker panel. AUC 0.5 = no discrimination,
    >0.5 = higher in B, <0.5 = higher in A; 'either side' keeps |AUC−0.5| ≥ |cut−0.5|.
    Degenerate (NaN) AUC is dropped."""
    direction = str(direction)
    if direction.startswith("higher in A"):
        keep = auc <= cut
    elif direction.startswith("higher in B"):
        keep = auc >= cut
    else:                                                   # "either side" — symmetric about 0.5
        keep = (auc - 0.5).abs() >= abs(cut - 0.5)
    return keep.fillna(False)


def _run_filter_auc(ds, inp, p):
    """Trim the upstream stats table by ROC AUC, keeping the full (untrimmed) list too.

    AUC 0.5 = no discrimination; >0.5 = higher in group B; <0.5 = higher in group A
    (i.e. more confined to the first group's ROI). Returns both lists so the bundle
    can export the full table (with a ``kept`` flag) and the trimmed working set."""
    df = inp.get("stats_df")
    if df is None or not len(df):
        raise ValueError("no upstream statistics — add a Region comparison (A vs B) or "
                         "ROI localization step before this filter.")
    col = inp.get("auc_col")
    if col is None:
        col = "AUC" if "AUC" in df.columns else ("roi_auc" if "roi_auc" in df.columns else None)
    if col is None:
        raise ValueError("the upstream step has no AUC column to filter on.")
    cut = float(p.get("threshold", 0.7))
    direction = str(p.get("direction", "either side"))
    keep = _auc_keep(df[col].astype(float), direction, cut)
    full = df.copy()
    full.insert(0, "kept", keep.to_numpy())
    full.attrs = dict(df.attrs)
    trimmed = full[full["kept"]].drop(columns=["kept"]).reset_index(drop=True)
    trimmed.attrs = dict(df.attrs)
    peaks = []
    for rec in trimmed.to_dict("records"):
        rec["mz"] = float(rec["mz"])
        peaks.append(rec)
    return {"full": full, "trimmed": trimmed, "peaks": peaks,
            "col": col, "cut": cut, "direction": direction}


def _run_marker_panel(ds, inp, p):
    """Sweep several (A vs B) region pairs, keep each pair's AUC-distinctive ions, and union
    the survivors into ONE marker panel.

    Built for 'sub-region vs its parent' contrasts: run each sub-ROI against its whole region,
    keep the distinctive ions per pair, then merge every pair's markers into a single feature
    list that drives downstream analyses. Uses per-pixel AUC as the effect size (candidate
    selection) — chain a Region comparison / cohort step afterwards for replicate-level
    p-values. ``inp['pairs']`` is the GUI-resolved list of
    ``{mask_a, mask_b, a_label, b_label}`` dicts (overlap between A and B is allowed)."""
    from . import spatial
    pairs = inp.get("pairs") or []
    if not pairs:
        raise ValueError("no region pairs configured — use 'Auto-pair sub→parent' or 'Add pair…'.")
    direction = str(p.get("direction", "either side"))
    cut = float(p.get("threshold", 0.6))
    maxq = float(p.get("max_q", 0.05))
    method = p.get("method", "mwu")
    tol, norm = p["tol_ppm"], p.get("norm", "tic")
    out_pairs, merged, seen = [], [], set()
    for pr in pairs:
        la, lb = pr["a_label"], pr["b_label"]
        try:
            df = spatial.roi_comparison(ds, pr["mask_a"], pr["mask_b"], inp["mzs"],
                                        tol_ppm=tol, norm=norm, a_label=la, b_label=lb,
                                        method=method)
        except Exception as exc:                            # degenerate/empty pair → skip, record
            out_pairs.append({"a_label": la, "b_label": lb, "full": None, "kept": None,
                              "error": f"{type(exc).__name__}: {exc}"})
            continue
        d = df
        if maxq < 1.0 and "q_value" in d.columns:
            d = d[d["q_value"].astype(float) <= maxq]
        kept = d[_auc_keep(d["AUC"].astype(float), direction, cut).to_numpy()].reset_index(drop=True)
        out_pairs.append({"a_label": la, "b_label": lb, "full": df, "kept": kept})
        has_q = "q_value" in kept.columns
        for _, x in kept.iterrows():                        # union across pairs, first note wins
            mz = float(x["mz"])
            key = round(mz, 4)
            if key in seen:
                continue
            seen.add(key)
            note = (f"marker · {la} vs {lb} · AUC {float(x['AUC']):.2f}"
                    + (f" · q {float(x['q_value']):.1e}" if has_q else ""))
            merged.append({"mz": mz, "lipid": "", "note": note})
    if not any(op.get("full") is not None for op in out_pairs):
        errs = "; ".join(op["error"] for op in out_pairs if op.get("error"))
        raise ValueError(f"every region pair failed ({errs})" if errs else "no pairs produced results")
    return {"pairs": out_pairs, "merged": merged, "cut": cut, "direction": direction,
            "name": "Region marker panel", "save_each": bool(p.get("save_each", False))}


def _marker_panel_table(r):
    """One stacked table across all pairs — each pair's full comparison with a ``pair`` column
    and a ``kept`` flag (bundle CSV + Report)."""
    import pandas as pd
    frames = []
    for op in r["pairs"]:
        full = op.get("full")
        if full is None:
            continue
        d = full.copy()
        kept = op.get("kept")
        kept_mz = {round(float(z), 4) for z in (kept["mz"] if kept is not None else [])}
        d.insert(0, "pair", f"{op['a_label']} vs {op['b_label']}")
        d.insert(1, "kept", [round(float(z), 4) in kept_mz for z in d["mz"]])
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else None


def _marker_panel_rep(r, n, ds=None):
    """Representative ions for the bundle: kept ions across pairs, most distinctive first."""
    rows = []
    for op in r["pairs"]:
        kept = op.get("kept")
        if kept is None:
            continue
        lbl = f"{op['a_label']} vs {op['b_label']}"
        for _, x in kept.iterrows():
            rows.append((float(x["mz"]), lbl, float(x["AUC"]), abs(float(x["AUC"]) - 0.5)))
    rows.sort(key=lambda t: -t[3])
    return [(mz, lbl, auc) for mz, lbl, auc, _ in rows[:int(n)]]


def _marker_panel_lists(r):
    """The merged panel as a named ★ list, plus each pair's own list when ``save_each``."""
    out = {}
    if r.get("merged"):
        out[r["name"]] = r["merged"]
    if r.get("save_each"):
        for op in r["pairs"]:
            kept = op.get("kept")
            if kept is None or not len(kept):
                continue
            has_q = "q_value" in kept.columns
            feats = [{"mz": float(x["mz"]), "lipid": "",
                      "note": f"AUC {float(x['AUC']):.2f}"
                              + (f" · q {float(x['q_value']):.1e}" if has_q else "")}
                     for _, x in kept.iterrows()]
            out[f"{op['a_label']} vs {op['b_label']} markers"] = feats
    return out


# common ParamSpecs
def _p_tol(default=50.0):
    return ParamSpec("tol_ppm", "Extraction tol (ppm)", "float", default, lo=1, hi=200, step=1)


def _p_norm():
    return ParamSpec("norm", "Normalization", "choice", "tic",
                     choices=["tic", "rms", "median", "none"],
                     help="Per-pixel scaling applied before the analysis.")


# --------------------------------------------------------------------------- #
# THE REGISTRY
# --------------------------------------------------------------------------- #
REGISTRY: dict = {}


def _register(sd: StepDef):
    REGISTRY[sd.id] = sd


_register(StepDef(
    id="find_spatial_features", name="Find spatial features", category="Peaks",
    needs=set(), produces={"peaks"}, targets={"slide"}, uses_mask=True,
    result_kind="features", view="Ion image",
    help="Spatially-aware peak detection (S/N → reproducibility → Moran's I denoise). Produces the working feature set later steps use.",
    params=[ParamSpec("snr", "Signal-to-noise", "float", 3.0, lo=0, hi=50, step=0.5),
            ParamSpec("min_rel_intensity", "Min rel. intensity", "float", 0.002, lo=0, hi=0.1, step=0.001),
            ParamSpec("min_frequency", "Min pixel frequency", "float", 0.01, lo=0, hi=1, step=0.01),
            ParamSpec("min_morans", "Min Moran's I", "float", 0.05, lo=0, hi=1, step=0.01),
            ParamSpec("max_candidates", "Max candidates (0 = no limit)", "int", 2000, lo=0, hi=200000, step=500),
            _p_tol(10.0), _p_norm()],
    run=_run_find_spatial,
    peaks=lambda r: list(r.peaks),
    produce=lambda r: {"peaks": _mz_list(r.peaks)},
    to_table=lambda r: _df(r.peaks),
    rep_ions=lambda r, n, ds=None: [(float(p["mz"]), str(p.get("label", "")),
                                     float(p.get("morans_i", 0.0) or 0.0)) for p in r.peaks[:int(n)]],
    summary=lambda r: (f"{len(r.peaks)} spatial features (of {r.n_candidates} candidates"
                       + (f", capped from {r.n_detected} detected" if r.candidates_capped else "") + ")")))

_register(StepDef(
    id="find_coherent_features", name="Coherent feature extraction", category="Peaks",
    needs=set(), produces={"peaks"}, targets={"slide"}, uses_mask=True,
    result_kind="features", view="Ion image",
    help="Spatial feature detection (S/N → reproducibility → Moran's I) plus an artifact-rejection quality gate (spatial-chaos morphology + hotspot concentration) that screens out delocalization / matrix-crystal ions which pass autocorrelation but aren't real. Produces a working feature set.",
    params=[ParamSpec("snr", "Signal-to-noise", "float", 3.0, lo=0, hi=50, step=0.5),
            ParamSpec("min_rel_intensity", "Min rel. intensity", "float", 0.002, lo=0, hi=0.1, step=0.001),
            ParamSpec("min_frequency", "Min pixel frequency", "float", 0.01, lo=0, hi=1, step=0.01),
            ParamSpec("min_morans", "Min Moran's I", "float", 0.0, lo=0, hi=1, step=0.01),
            ParamSpec("min_quality", "Min quality", "float", 0.15, lo=0, hi=1, step=0.01),
            ParamSpec("max_hotspot", "Max hotspot fraction", "float", 0.80, lo=0, hi=1, step=0.05),
            ParamSpec("max_candidates", "Max candidates (0 = no limit)", "int", 2000, lo=0, hi=200000, step=500),
            _p_tol(10.0), _p_norm()],
    run=_run_find_coherent,
    peaks=lambda r: list(r.peaks),
    produce=lambda r: {"peaks": _mz_list(r.peaks)},
    to_table=lambda r: _df(r.peaks),
    rep_ions=lambda r, n, ds=None: [(float(p["mz"]), str(p.get("label", "")),
                                     float(p.get("quality", 0.0) or 0.0)) for p in r.peaks[:int(n)]],
    summary=lambda r: (f"{len(r.peaks)} coherent features (of {r.n_candidates} candidates"
                       + (f", capped from {r.n_detected} detected" if r.candidates_capped else "")
                       + f"; {r.n_after_spatial} spatial)")))

_register(StepDef(
    id="find_peaks", name="Find peaks", category="Peaks",
    needs=set(), produces={"peaks"}, targets={"slide"}, uses_mask=True,
    result_kind="features", view="Ion image",
    help="Detect peaks in the mean spectrum (S/N + prominence). Produces the working feature set.",
    params=[ParamSpec("snr", "Signal-to-noise", "float", 3.0, lo=0, hi=50, step=0.5),
            ParamSpec("min_rel_intensity", "Min rel. intensity", "float", 0.002, lo=0, hi=0.1, step=0.001),
            ParamSpec("prominence", "Prominence", "float", 1.0, lo=0, hi=10, step=0.5),
            ParamSpec("max_peaks", "Max peaks (0 = no limit)", "int", 0, lo=0, hi=200000, step=100)],
    run=_run_find_peaks,
    peaks=lambda r: list(r),
    produce=lambda r: {"peaks": _mz_list(r)},
    to_table=lambda r: _df(r),
    rep_ions=lambda r, n, ds=None: [(float(p["mz"]), "", float(p.get("intensity", 0.0) or 0.0))
                                    for p in sorted(r, key=lambda q: -(q.get("intensity", 0.0) or 0.0))[:int(n)]],
    summary=lambda r: (f"{len(r)} peaks"
                       + (f" (capped from {r.n_detected} detected)"
                          if getattr(r, "n_detected", len(r)) > len(r) else ""))))

_register(StepDef(
    id="auto_segment", name="Segmentation", category="Segmentation",
    needs={"feature_set"}, produces={"segmentation"}, targets={"slide"}, uses_mask=True,
    result_kind="segmentation", view="Segmentation",
    help="Cluster pixels into regions. 0 clusters = auto-pick by silhouette.",
    params=[ParamSpec("n_clusters", "Clusters (0 = auto)", "int", 0, lo=0, hi=30, step=1),
            ParamSpec("spatial", "Spatially-aware", "bool", True,
                      help="Smooth before clustering for coherent regions (no speckle)."),
            ParamSpec("spatial_sigma", "Smoothing radius (px)", "float", 1.0, lo=0, hi=5, step=0.5),
            _p_tol(), _p_norm()],
    run=_run_segment,
    produce=lambda r: {"segmentation": r},
    summary=lambda r: f"{int(r.n_clusters)} clusters (silhouette {r.silhouette:.2f})"))

_register(StepDef(
    id="roi_comparison", name="Region comparison (A vs B)", category="Statistics",
    needs={"ab", "feature_set"}, produces={"stats_df"}, targets={"slide", "cohort"},
    result_kind="stats", view="Region comparison",
    help="Per-ion comparison of two groups (AUC + fold-change + test). Each named sub-ROI is a replicate.",
    params=[ParamSpec("method", "Statistical test", "choice", "mwu",
                      choices=["mwu", "welch", "student"]),
            ParamSpec("save_lists", "Save distinguishing ions as ★ lists", "bool", True,
                      help="After running, save each region's distinguishing ions (q ≤ max-q) "
                           "as a named feature list — one for A, one for B."),
            ParamSpec("max_q", "Max q-value (for saved lists)", "float", 0.05, lo=0.0, hi=1.0, step=0.01,
                      help="Significance cutoff for the saved lists; 1.0 keeps every ion."),
            _p_tol(10.0), _p_norm()],
    run=_run_roi_comparison,
    produce=lambda r: {"stats_df": r},
    to_table=lambda r: r,
    rep_ions=lambda r, n, ds=None: _rep_by_auc(r, n, "AUC"),
    lists=_roi_cmp_lists,
    summary=lambda r: f"{int((r['q_value'] < 0.05).sum())} ions q<0.05 of {len(r)}"))

_register(StepDef(
    id="region_comparison", name="Region spectra (A vs B overlay)", category="Statistics",
    needs={"ab"}, produces=set(), targets={"slide"},
    result_kind="spectra", view="Region comparison",
    help="Overlay two regions' whole mean spectra (or their difference) — the visual companion "
         "to Region comparison (A vs B). Toggle overlay ⇄ difference in the result.",
    description="Compare two regions by their full mean spectra: overlay both, or show their "
                "A−B difference. Pairs with 'Region comparison (A vs B)' (the per-ion stats) as "
                "the two halves of 'compare two regions'.",
    params=[_p_norm()],
    run=_run_region_comparison,
    summary=lambda r: f"mean-spectrum overlay: {r.get('a_label', 'A')} vs {r.get('b_label', 'B')}"))

_register(StepDef(
    id="class_comparison", name="Lipid class comparison (A vs B)", category="Statistics",
    needs={"ab", "feature_set"}, produces=set(), targets={"slide", "cohort"},
    result_kind="table", view="Discriminating && ROI stats",
    help="Roll the per-ion identifications up to whole lipid classes (PE, PC, SM, …) and compare "
         "two groups: each class's summed signal gets the same exact rank-based AUC + fold-change "
         "+ test as the per-ion Region comparison. Identification mode/tolerance set how peaks map "
         "to classes.",
    params=[ParamSpec("method", "Statistical test", "choice", "mwu",
                      choices=["mwu", "welch", "student"]),
            ParamSpec("mode", "Adduct mode", "choice", "negative", choices=["negative", "positive"]),
            ParamSpec("id_ppm", "Identification tol (ppm)", "float", 5.0, lo=1, hi=50, step=1),
            _p_tol(10.0), _p_norm()],
    run=_run_class_comparison,
    to_table=lambda r: r,
    summary=lambda r: f"{int((r['q_value'] < 0.05).sum())} of {len(r)} classes differ q<0.05"))

_register(StepDef(
    id="class_composition", name="Lipid class composition (A vs B)", category="Statistics",
    needs={"ab", "feature_set"}, produces=set(), targets={"slide", "cohort"},
    result_kind="table", view="Discriminating && ROI stats",
    help="Each group's lipid-class composition: the share (%) of annotated-lipid signal in every "
         "class, for group A and group B side by side (each column sums to 100%). Identification "
         "mode/tolerance set how peaks map to classes.",
    params=[ParamSpec("mode", "Adduct mode", "choice", "negative", choices=["negative", "positive"]),
            ParamSpec("id_ppm", "Identification tol (ppm)", "float", 5.0, lo=1, hi=50, step=1),
            _p_tol(10.0), _p_norm()],
    run=_run_class_composition,
    to_table=lambda r: r.reset_index(),
    summary=lambda r: f"{len(r)} classes across {len(r.columns)} regions"))

_register(StepDef(
    id="marker_panel", name="Region marker panel", category="Statistics",
    needs={"ab_multi", "feature_set"}, produces={"peaks"}, targets={"slide"},
    result_kind="features", view="Feature lists",
    help="Sweep several region pairs (each sub-region vs its parent by default), keep every "
         "pair's AUC-distinctive ions, and merge them into ONE feature list for downstream "
         "analyses. Click 'Auto-pair sub→parent' to pair each sub-ROI with its parent region, "
         "or 'Add pair…' to build A-vs-B pairs by hand. Per-pixel AUC effect size for candidate "
         "selection — follow with a Region comparison / cohort step for replicate-level p-values.",
    params=[ParamSpec("direction", "Keep ions", "choice", "either side",
                      choices=["either side", "higher in A (AUC ≤ cutoff)",
                               "higher in B (AUC ≥ cutoff)"],
                      help="Which side of each pair to keep. 'either side' keeps strong "
                           "discriminators in both directions (|AUC−0.5| ≥ |cutoff−0.5|)."),
            ParamSpec("threshold", "AUC cutoff", "float", 0.6, lo=0.0, hi=1.0, step=0.01,
                      help="0.6 keeps AUC ≥ 0.6 or ≤ 0.4 ('either side'); 0.5 keeps any enrichment."),
            ParamSpec("max_q", "Max q-value", "float", 0.05, lo=0.0, hi=1.0, step=0.01,
                      help="Significance cutoff per pair; 1.0 keeps ions on AUC effect size alone."),
            ParamSpec("subtract_a_from_b", "Exclude A's pixels from B", "bool", False,
                      help="Compare A against B with A's pixels removed (sub vs the REST of its "
                           "parent) instead of against the whole parent. Off = literal sub vs whole."),
            ParamSpec("save_each", "Also save each pair's own ★ list", "bool", False,
                      help="Besides the merged panel, save each pair's markers as its own list."),
            _p_tol(10.0), _p_norm()],
    run=_run_marker_panel,
    peaks=lambda r: list(r["merged"]),
    produce=lambda r: {"peaks": _mz_list(r["merged"])} if r.get("merged") else {},
    to_table=_marker_panel_table,
    rep_ions=_marker_panel_rep,
    lists=_marker_panel_lists,
    summary=lambda r: (f"{len(r['merged'])} marker ions from "
                       f"{sum(1 for op in r['pairs'] if op.get('full') is not None)} pair(s)")))

_register(StepDef(
    id="discriminating_features", name="Discriminating features (per group)", category="Statistics",
    needs={"groups", "feature_set"}, produces={"peaks"}, targets={"slide"},
    result_kind="table", view="Discriminating && ROI stats",
    help="Characteristic ions for each group vs the rest (one-vs-rest AUC + FDR). With exactly "
         "two tagged groups this is A-vs-B. Keep only the enriched, significant markers, save "
         "them as a per-group ★ list, and pass the survivors to the next step (e.g. a Venn). "
         "Ungrouped/background pixels are excluded from 'the rest'.",
    params=[ParamSpec("enriched_only", "Enriched markers only", "bool", True,
                      help="Keep ions a group is HIGH in (AUC ≥ min AUC). Untick to also keep "
                           "ions it is depleted in."),
            ParamSpec("min_auc", "Min AUC", "float", 0.5, lo=0.5, hi=1.0, step=0.01,
                      help="0.5 = any enrichment; raise to demand a stronger marker."),
            ParamSpec("max_q", "Max q-value", "float", 0.05, lo=0.0, hi=1.0, step=0.01,
                      help="FDR cutoff; 1.0 keeps everything regardless of significance."),
            ParamSpec("top_n", "Candidates / group", "int", 50, lo=3, hi=500, step=5,
                      help="Rank by discrimination strength first, then apply the filters above."),
            ParamSpec("save_lists", "Save per-group ★ marker lists", "bool", False,
                      help="Save each group's kept ions as a named feature list you can reuse."),
            _p_tol(), _p_norm()],
    run=_run_discriminating,
    produce=_disc_survivors,
    to_table=_per_cluster_table,
    rep_ions=_per_cluster_rep,
    lists=lambda r: _disc_lists(r) if getattr(r, "_save_lists", True) else {},
    summary=lambda r: (f"{len(r)} groups · kept {sum(len(d) for d in r.values())} markers "
                       f"(AUC ≥ {r.min_auc:.2f}, q ≤ {r.max_q:g})")))

_register(StepDef(
    id="multigroup_features", name="Multi-group features", category="Statistics",
    needs={"groups", "feature_set"}, produces=set(), targets={"slide"},
    result_kind="table", view="Discriminating && ROI stats",
    help="Per-ion test across 3+ groups (Kruskal-Wallis or ANOVA).",
    params=[ParamSpec("method", "Statistical test", "choice", "kruskal",
                      choices=["kruskal", "anova"]),
            _p_tol(), _p_norm()],
    run=_run_multigroup,
    to_table=lambda r: r,
    rep_ions=_rep_by_pvalue,
    summary=lambda r: f"{int((r['q_value'] < 0.05).sum())} ions q<0.05 of {len(r)}"))

_register(StepDef(
    id="roi_localization", name="ROI localization", category="Statistics",
    needs={"region", "feature_set"}, produces={"stats_df"}, targets={"slide"},
    result_kind="table", view="Discriminating && ROI stats",
    help="How confined each ion is to the first group's ROI vs the rest of the tissue "
         "(one group is enough).",
    params=[_p_tol(), _p_norm()],
    run=_run_roi_localization,
    produce=lambda r: {"stats_df": r},
    to_table=lambda r: r,
    rep_ions=lambda r, n, ds=None: _rep_by_auc(r, n, "roi_auc"),
    summary=lambda r: f"{len(r)} ions ranked by ROI enrichment"))

_register(StepDef(
    id="filter_auc", name="Keep features by AUC", category="Statistics",
    needs={"stats"}, produces={"stats_df"}, targets={"slide"},
    result_kind="features", view="Ion image",
    help="Trim the previous comparison's ions by ROC AUC and make the survivors the working "
         "feature set. AUC 0.5 = no discrimination; >0.5 = higher in group B; <0.5 = higher in "
         "group A (more confined to the first ROI). The exported table keeps every ion with a "
         "'kept' flag, so you get both the untrimmed and trimmed lists. Add after a Region "
         "comparison (A vs B) or ROI localization step.",
    params=[ParamSpec("direction", "Keep ions", "choice", "either side",
                      choices=["either side", "higher in A (AUC ≤ cutoff)",
                               "higher in B (AUC ≥ cutoff)"],
                      help="Which side of the contrast to keep. 'either side' keeps strong "
                           "discriminators in both directions (|AUC−0.5| ≥ |cutoff−0.5|)."),
            ParamSpec("threshold", "AUC cutoff", "float", 0.7, lo=0.0, hi=1.0, step=0.01,
                      help="For 'either side', a cutoff of 0.7 keeps AUC ≥ 0.7 or ≤ 0.3.")],
    run=_run_filter_auc,
    peaks=lambda r: list(r["peaks"]),
    produce=lambda r: {"stats_df": r["trimmed"]},
    to_table=lambda r: r["full"],
    rep_ions=lambda r, n, ds=None: _rep_by_auc(r["trimmed"], n, r["col"]),
    summary=lambda r: f"kept {len(r['trimmed'])} of {len(r['full'])} ions "
                      f"({r['direction']}, cutoff {r['cut']:.2f})"))

_register(StepDef(
    id="shrunken_centroids", name="Markers (shrunken centroids)", category="Statistics",
    needs={"groups", "feature_set"}, produces=set(), targets={"slide"},
    result_kind="table", view="Markers (SSC)",
    help="Nearest shrunken centroids — automatic marker selection per group.",
    description="Nearest shrunken centroids: per labelled group, keep only the ions whose mean "
                "clearly departs from the overall mean — a sparse marker shortlist per group.",
    params=[ParamSpec("shrink", "Shrinkage", "float", 2.0, lo=0, hi=10, step=0.5),
            ParamSpec("top_n", "Top markers / group", "int", 15, lo=3, hi=100, step=1),
            _p_tol(), _p_norm()],
    run=_run_shrunken,
    to_table=_per_cluster_table,
    rep_ions=_per_cluster_rep,
    summary=lambda r: f"{len(r)} groups, {sum(len(d) for d in r.values())} markers"))

_register(StepDef(
    id="dgmm", name="Per-ion segmentation (DGMM)", category="Multivariate",
    needs={"target_mz"}, produces=set(), targets={"slide"},
    result_kind="ion_segmentation", view="Ion image",
    help="Spatially-aware Gaussian-mixture segmentation of a single ion image into "
         "intensity zones (background → bright).",
    description="Split the active ion's image into spatially-coherent intensity zones "
                "(background → dim → bright) with a spatially-aware Gaussian mixture. Use to "
                "threshold one ion's hotspots; for a between-sample differential use the Cohort tab.",
    params=[ParamSpec("k", "Levels", "int", 3, lo=2, hi=6, step=1,
                      help="Number of intensity zones to split the ion image into."),
            _p_norm()],
    run=_run_dgmm,
    to_table=_dgmm_table,
    summary=lambda r: f"m/z {r['mz']:.4f} → {len(r['means'])} intensity zones "
                      f"(means {', '.join(f'{m:.3g}' for m in r['means'])})"))

_register(StepDef(
    id="region_membership", name="Distinct & shared ions (Venn)", category="Statistics",
    needs={"groups", "feature_set"}, produces=set(), targets={"slide"},
    result_kind="table", view="Discriminating && ROI stats",
    help="Partition ions by which groups they are present in (Venn compartments).",
    params=[ParamSpec("min_prevalence", "Min prevalence", "float", 0.5, lo=0, hi=1, step=0.05),
            _p_tol(), _p_norm()],
    run=_run_membership,
    to_table=lambda r: r.get("peaks"),
    rep_ions=lambda r, n, ds=None: [(float(mz), c.get("label", ""), 0.0)
                                    for c in r.get("compartments", []) for mz in c.get("mzs", [])][:int(n)],
    summary=lambda r: f"{len(r.get('compartments', []))} compartments across {len(r.get('names', []))} groups"))

_register(StepDef(
    id="colocalize", name="Co-localization with ion", category="Co-localization",
    needs={"target_mz", "feature_set"}, produces=set(), targets={"slide"}, uses_mask=True,
    result_kind="table", view="Co-localization",
    help="Rank all ions by spatial similarity to a target m/z.",
    params=[ParamSpec("target_mz", "Target m/z", "mz", None, lo=0, hi=5000, step=0.0001),
            ParamSpec("method", "Similarity", "choice", "pearson",
                      choices=["pearson", "cosine", "moc", "m1", "m2", "dice"]),
            _p_tol(), _p_norm()],
    run=_run_colocalize,
    to_table=lambda r: _df(r),
    rep_ions=lambda r, n, ds=None: [(float(d["mz"]), "", float(d["score"])) for d in r[:int(n)]],
    lists=lambda r: {"co-localized peaks": [{"mz": float(d["mz"])} for d in r[:30]]},
    summary=lambda r: f"{len(r)} ions ranked by co-localization"))

_register(StepDef(
    id="coloc_modules", name="Co-localization modules", category="Co-localization",
    needs={"feature_set"}, produces=set(), targets={"slide"}, uses_mask=True,
    result_kind="table", view="Co-localization",
    help="Group ions into co-localized modules by hierarchical clustering.",
    params=[ParamSpec("n_modules", "Modules (0 = auto)", "int", 0, lo=0, hi=30, step=1),
            ParamSpec("threshold", "Merge threshold", "float", 0.5, lo=0, hi=1, step=0.05),
            ParamSpec("method", "Similarity", "choice", "pearson",
                      choices=["pearson", "cosine", "moc", "m1", "m2", "dice"]),
            _p_tol(), _p_norm()],
    run=_run_coloc_modules,
    to_table=lambda r: _df([{"module": int(lab), "mz": float(mz)}
                            for mz, lab in zip(r.peaks, r.labels)]),
    rep_ions=lambda r, n, ds=None: [(float(mz), f"module {int(lab)}", 0.0)
                                    for mz, lab in zip(r.peaks, r.labels)][:int(n)],
    summary=lambda r: f"{len(set(int(x) for x in r.labels))} modules"))

_register(StepDef(
    id="region_correlation", name="Region correlation & match", category="Co-localization",
    needs={"feature_set"}, produces=set(), targets={"slide"},
    result_kind="matrix", view="Co-localization",
    help="Molecular-fingerprint similarity of every named region to every other — a "
         "region×region heatmap plus each region's closest match (the old Region-match view).",
    description="Compare named regions by their mean molecular fingerprint: a region×region "
                "similarity heatmap and, below it, each region's single closest match. Use to "
                "ask 'which of my regions look chemically alike?' Needs at least two named "
                "regions (draw them on the Ion image tab).",
    params=[ParamSpec("method", "Measure", "choice", "pearson", choices=["pearson", "cosine"]),
            _p_tol(), _p_norm()],
    run=_run_region_correlation,
    to_table=_region_corr_table,
    regions_out=_region_correlation_regions,
    summary=lambda r: f"{len(r.names)} regions correlated"))

_register(StepDef(
    id="pca", name="PCA images", category="Multivariate",
    needs={"feature_set"}, produces=set(), targets={"slide"}, uses_mask=True,
    result_kind="components", view="Components",
    help="Principal-component score images + loadings.",
    params=[ParamSpec("n_components", "Components", "int", 5, lo=2, hi=20, step=1), _p_tol(), _p_norm()],
    run=_run_pca,
    to_table=lambda r: _df([{"component": k, "mz": mz, "loading": load}
                            for k in range(len(r.images)) for mz, load in r.top_peaks(k, 10)]),
    rep_ions=lambda r, n, ds=None: [(float(mz), f"PC{k + 1}", float(load))
                                    for k in range(len(r.images))
                                    for mz, load in r.top_peaks(k, max(1, int(n) // max(1, len(r.images))))][:int(n)],
    summary=lambda r: f"{len(r.images)} components"))

_register(StepDef(
    id="nmf", name="NMF images", category="Multivariate",
    needs={"feature_set"}, produces=set(), targets={"slide"}, uses_mask=True,
    result_kind="components", view="Components",
    help="Non-negative matrix factorization (additive parts) score images + loadings.",
    params=[ParamSpec("n_components", "Components", "int", 5, lo=2, hi=20, step=1), _p_tol(), _p_norm()],
    run=_run_nmf,
    to_table=lambda r: _df([{"component": k, "mz": mz, "loading": load}
                            for k in range(len(r.images)) for mz, load in r.top_peaks(k, 10)]),
    rep_ions=lambda r, n, ds=None: [(float(mz), f"NMF{k + 1}", float(load))
                                    for k in range(len(r.images))
                                    for mz, load in r.top_peaks(k, max(1, int(n) // max(1, len(r.images))))][:int(n)],
    summary=lambda r: f"{len(r.images)} components"))

_register(StepDef(
    id="plsda", name="PLS-DA / OPLS-DA", category="Multivariate",
    needs={"groups", "feature_set"}, produces=set(), targets={"slide"},
    result_kind="classifier", view="Classify",
    help="Supervised classification; ranks ions by VIP (variable importance).",
    params=[ParamSpec("n_components", "Latent components", "int", 2, lo=1, hi=10, step=1),
            ParamSpec("orthogonal", "OPLS-DA (binary)", "bool", False), _p_tol(), _p_norm()],
    run=_run_plsda,
    to_table=lambda r: _df([{"mz": float(mz), "vip": float(v)} for mz, v in zip(r.peaks, r.vip)]),
    rep_ions=lambda r, n, ds=None: [(float(mz), "", float(v)) for mz, v in r.top_peaks(int(n))],
    summary=lambda r: f"{r.method}: {len(r.classes)} classes"))

_register(StepDef(
    id="classify_cv", name="Classifier cross-validation", category="Multivariate",
    needs={"groups", "feature_set"}, produces=set(), targets={"slide"},
    result_kind="table", view="Classify",
    help="Cross-validated accuracy + confusion matrix for the PLS-DA classifier. Uses "
         "leave-one-tissue-out folds when the slide has ≥2 detected tissue pieces, else "
         "stratified k-fold (the engine flags an optimistic, leakage-prone estimate).",
    params=[ParamSpec("n_components", "Latent components", "int", 2, lo=1, hi=10, step=1),
            ParamSpec("orthogonal", "OPLS-DA (binary)", "bool", False),
            ParamSpec("n_folds", "Folds (stratified fallback)", "int", 5, lo=2, hi=10, step=1),
            _p_tol(), _p_norm()],
    run=_run_classify_cv,
    to_table=_cv_table,
    summary=lambda r: (f"accuracy {r.get('accuracy', float('nan')):.2f} over "
                       f"{r.get('n_folds', 0)} folds"
                       + ("" if r.get("leakage_safe", True) else " (optimistic — see warning)"))))

_register(StepDef(
    id="classify_map", name="Classifier prediction map", category="Multivariate",
    needs={"groups", "feature_set"}, produces=set(), targets={"slide"}, uses_mask=True,
    result_kind="ion_segmentation", view="Classify",
    help="Fit the PLS-DA classifier, then paint every pixel with its predicted class — a "
         "class-label map over the whole slide.",
    params=[ParamSpec("n_components", "Latent components", "int", 2, lo=1, hi=10, step=1),
            ParamSpec("orthogonal", "OPLS-DA (binary)", "bool", False), _p_tol(), _p_norm()],
    run=_run_classify_map,
    regions_out=_classify_map_regions,
    summary=lambda r: f"predicted map over {len(r.get('classes', []))} classes"))

_register(StepDef(
    id="shap_biomarkers", name="SHAP biomarkers", category="Multivariate",
    needs={"groups", "feature_set"}, produces=set(), targets={"slide"},
    result_kind="table", view="Discriminating && ROI stats",
    help="Per-region biomarker ions ranked by SHAP importance (RandomForest + TreeExplainer); "
         "each ion's direction is the Spearman sign of intensity vs its SHAP value. "
         "Needs the optional 'shap' package.",
    params=[ParamSpec("n_estimators", "Trees", "int", 300, lo=50, hi=1000, step=50),
            ParamSpec("max_pixels", "Max pixels (subsample)", "int", 20000, lo=1000, hi=200000, step=1000),
            _p_tol(), _p_norm()],
    run=_run_shap,
    to_table=lambda r: r.to_long_df(),
    rep_ions=_shap_rep_ions,
    summary=lambda r: f"SHAP biomarkers: {len(r.classes)} regions (train acc {r.accuracy:.2f})"))

_register(StepDef(
    id="embedding", name="2-D embedding (UMAP/t-SNE)", category="Multivariate",
    needs={"feature_set"}, produces=set(), targets={"slide"},
    result_kind="embedding", view="Feature space",
    help="2-D pixel embedding + molecular-similarity tissue map.",
    params=[ParamSpec("method", "Method", "choice", "umap", choices=["umap", "tsne"]), _p_tol(), _p_norm()],
    run=_run_embedding,
    summary=lambda r: f"{r.method} embedding of {r.coords.shape[0]} pixels"))

_register(StepDef(
    id="annotate", name="Annotate lipids", category="Annotation",
    needs={"feature_set"}, produces=set(), targets={"slide"},
    result_kind="features", view="Feature lists",
    help="Match the feature set to the lipid database (isotopes + adducts + confidence).",
    params=[ParamSpec("mode", "Adduct mode", "choice", "negative", choices=["negative", "positive"]),
            ParamSpec("match_ppm", "Match tol (ppm)", "float", 10.0, lo=1, hi=50, step=1),
            ParamSpec("image_ppm", "Image tol (ppm)", "float", 10.0, lo=1, hi=50, step=1),
            _p_norm()],
    run=_run_annotate,
    to_table=lambda r: r,
    rep_ions=lambda r, n, ds=None: ([(float(x["mz"]), str(x.get("lipid", "")),
                                      float(x.get("confidence_score", 0.0) or 0.0))
                                     for _, x in r.sort_values("confidence_score", ascending=False).head(int(n)).iterrows()]
                                    if "confidence_score" in getattr(r, "columns", []) else []),
    summary=lambda r: f"{int((r['lipid'].astype(str) != '').sum()) if 'lipid' in r.columns else 0} annotated of {len(r)}"))


# --------------------------------------------------------------------------- #
# Cohort (cross-subject) analyses — surfaced as gallery cards in the Cohort scope.
# These run across every sample in the loaded cohort, not a single slide, so they keep their
# own full screen (``presentation="screen"``): a card opens the Cohort screen named by ``view``
# rather than the single-slide AnalysisDialog. ``run`` is intentionally a screen-launch stub —
# a headless caller drives the cohort through the Cohort object / cohortview, not run_analysis.
# --------------------------------------------------------------------------- #
def _cohort_screen_only(ds, inp, p):
    raise RuntimeError(
        "This is a cohort analysis — it runs across the whole loaded cohort, not a single "
        "slide. Open it from the Analyze gallery's Cohort scope (or the Cohort screen).")


for _cid, _name, _view, _blurb in [
    ("cohort_group_comparison", "Group comparison (A vs B)", "Cohort",
     "Compare two subject groups across the whole cohort: each sample is one replicate, so "
     "the per-ion test (fold-change + q) is across subjects — the statistically defensible "
     "answer that the single-slide Region comparison only approximates per-pixel."),
    ("cohort_nested_comparison", "Nested stats (subject × compartment)", "Cohort nested stats",
     "Per-ion mixed model y ~ group * compartment + (1 | subject) across the cohort — the "
     "correct multi-group-across-subjects test when each subject contributes several "
     "compartments/regions."),
    ("cohort_embedding", "Pooled embedding (UMAP/t-SNE)", "Cohort UMAP",
     "One UMAP/t-SNE over every sample's pixels on a shared feature axis — see how samples and "
     "groups separate in molecular space across the cohort."),
    ("cohort_segmentation", "Joint segmentation (shared)", "Cohort segmentation",
     "One segmentation tree learned jointly and applied to every sample, so the same cluster "
     "labels mean the same thing across slides."),
]:
    _register(StepDef(
        id=_cid, name=_name, category="Cohort",
        needs=set(), produces=set(), targets={"cohort"},
        result_kind="table", view=_view, presentation="screen",
        help=_blurb, description=_blurb,
        run=_cohort_screen_only))


# --------------------------------------------------------------------------- #
# accessors
# --------------------------------------------------------------------------- #
# Category display order for the "+ Add step" menu.
CATEGORIES = ["Peaks", "Segmentation", "Statistics", "Co-localization", "Multivariate",
              "Annotation", "Cohort"]


def step_def(type_id: str) -> StepDef | None:
    return REGISTRY.get(type_id)


def default_params(type_id: str) -> dict:
    """A fresh params dict for a step type, from its registry defaults."""
    sd = REGISTRY.get(type_id)
    if sd is None:
        return {}
    return {ps.name: ps.default for ps in sd.params}


def registry_by_category() -> dict:
    """``{category: [StepDef, …]}`` in :data:`CATEGORIES` order, for building the palette."""
    out = {c: [] for c in CATEGORIES}
    for sd in REGISTRY.values():
        out.setdefault(sd.category, []).append(sd)
    return {c: out[c] for c in out if out[c]}


# --------------------------------------------------------------------------- #
# data-input resolution — pure (no Qt), mirroring gui/flowdialog.py
# --------------------------------------------------------------------------- #
# These two functions lift the Flow runner's input gating out of the Qt dialog so the
# Analyze surface (and any headless caller) can decide "is this step runnable?" and build
# the ``inputs`` dict handed to ``StepDef.run(ds, inp, params)`` from a plain, Qt-free
# ``state`` snapshot. The GUI's helper methods (``_groups_map`` / ``_union`` /
# ``_group_labels`` / ``_ab_masks`` / ``_ab_from_scope`` / ``_pairs_from_scope`` /
# ``_step_mask`` / ``_resolve_mzs``) are collapsed here; ``state`` carries their already-
# resolved primitives (masks, m/z list, stats table) rather than the live window.
#
# ``state`` schema — every key optional; supply only what your steps need:
#   n_pixels    int                         pixel count (to size labels/samples arrays)
#   mask        bool[n_pix] | None          step/flow region scope; None = whole slide
#                                           (empty masks must already be coerced to None)
#   mzs         list[float]                 resolved feature set (already picked; may be [])
#   groups      dict{name: [bool[n_pix]…]}  ORDERED group→ROI masks (== _groups_map, but each
#                                           ROI pre-resolved to a pixel mask); order sets the
#                                           A/B/label order, exactly like the Setup table
#   seg_labels  int[n_pix] | None           segmentation labels; the <2-group fallback for the
#                                           general ``groups`` token (NOT region_membership)
#   seg_names   list[str] | None            display name per segmentation cluster id (parallel to
#                                           seg_labels' 0..k-1); the seg fallback's group names.
#                                           None → synthesized 'Segment N'
#   regions     dict{name: bool[n_pix]}     every region by name (for scope a/b/pairs lookups)
#   scope_a     list[str]                   per-step Group-A region names (st.scope["a"])
#   scope_b     list[str]                   per-step Group-B region names (st.scope["b"])
#   scope_pairs list[[list[str], list[str]]]  marker-panel (A,B) region-name pairs
#   subtract_a_from_b  bool                 pair mode: remove A's pixels from B
#   target_mz   float | None                the co-localization target ion (st.params)
#   stats_df    DataFrame | None            a prior step's stats table (needs an AUC column)


def _union(masks):
    """OR a list of boolean pixel masks; ``None`` entries are skipped; returns ``None`` when
    every entry is ``None`` (verbatim port of flowdialog's ``_union``)."""
    m = None
    for mk in masks or []:
        if mk is None:
            continue
        m = mk.copy() if m is None else (m | mk)
    return m


def _resolve_group_labels(state):
    """(labels, names) or (None, reason). <2 tagged groups falls back to ``seg_labels``
    (returning ``(seg_labels, None)``); mirrors ``_group_labels``."""
    import numpy as np
    groups = state.get("groups") or {}
    names = list(groups)
    if len(names) < 2:
        seg = state.get("seg_labels")
        if seg is not None:
            # Segmentation clusters stand in for groups. Name them so per-cluster tables read
            # 'Segment 0' (or the region a cluster was assigned to) rather than 'region 0'.
            seg_names = state.get("seg_names")
            if seg_names is None:
                s = np.asarray(seg)
                k = int(s.max()) + 1 if s.size and s.max() >= 0 else 0
                seg_names = [f"Segment {i}" for i in range(k)]
            return seg, seg_names
        return None, "Needs at least two groups (or a segmentation step first)."
    labels = np.full(int(state.get("n_pixels", 0)), -1, dtype=int)
    for gi, g in enumerate(names):
        mk = _union(groups[g])
        if mk is None:
            continue
        labels[(labels < 0) & mk] = gi          # first-seen group wins each pixel
    if int((labels >= 0).sum()) == 0:
        return None, "The assigned groups have no pixels."
    return labels, names


def _resolve_group_masks_named(state):
    """(masks, names) or (None, reason); >=2 groups, each non-empty, NO seg fallback —
    the ``region_membership`` special case (mirrors ``_group_masks_named``)."""
    groups = state.get("groups") or {}
    names = list(groups)
    if len(names) < 2:
        return None, "Needs at least two groups."
    masks = []
    for g in names:
        mk = _union(groups[g])
        if mk is None or not mk.any():
            return None, f"Group '{g}' has no pixels."
        masks.append(mk)
    return masks, names


def _resolve_ab_masks(state):
    """First two Setup groups → (ma, mb, a_label, b_label, samples) or
    (None, reason, None, None, None). Overlap is REJECTED (mirrors ``_ab_masks``)."""
    import numpy as np
    groups = state.get("groups") or {}
    names = list(groups)
    if len(names) < 2:
        return None, "Needs two groups (A and B).", None, None, None
    ga, gb = names[0], names[1]
    ma, mb = _union(groups[ga]), _union(groups[gb])
    if ma is None or mb is None or not ma.any() or not mb.any():
        return None, "The first two groups have no pixels.", None, None, None
    if bool((ma & mb).any()):
        return None, f"Groups '{ga}' and '{gb}' overlap — give each pixel one group.", None, None, None
    samples = np.full(int(state.get("n_pixels", 0)), -1, dtype=int)
    rid = 0
    for g in (ga, gb):                          # each ROI is one replicate id
        for mk in groups[g]:
            if mk is None:
                continue
            samples[mk] = rid
            rid += 1
    return ma, mb, ga, gb, samples


def _resolve_ab_from_scope(state):
    """Per-step A/B from ``scope_a`` / ``scope_b`` region names. Overlap ALLOWED (a sub-ROI
    vs its parent is a deliberate contrast); mirrors ``_ab_from_scope``."""
    import numpy as np
    a_names = list(state.get("scope_a") or [])
    b_names = list(state.get("scope_b") or [])
    if not a_names or not b_names:
        return None, "Needs at least one region for each of Group A and Group B.", None, None, None
    regions = state.get("regions") or {}
    a_masks = [regions[n] for n in a_names if n in regions]
    b_masks = [regions[n] for n in b_names if n in regions]
    ma, mb = _union(a_masks), _union(b_masks)
    if ma is None or mb is None or not ma.any() or not mb.any():
        return None, "The chosen A/B regions have no pixels.", None, None, None
    samples = np.full(int(state.get("n_pixels", 0)), -1, dtype=int)
    rid = 0
    for masks in (a_masks, b_masks):
        for mk in masks:
            if mk is None:
                continue
            samples[mk] = rid
            rid += 1
    return ma, mb, " + ".join(a_names), " + ".join(b_names), samples


def _resolve_ab(state):
    """Dispatch: explicit per-step ``scope_a``/``scope_b`` → ``_resolve_ab_from_scope``
    (overlap ok); else the Setup A/B fallback ``_resolve_ab_masks`` (overlap rejected)."""
    if state.get("scope_a") or state.get("scope_b"):
        return _resolve_ab_from_scope(state)
    return _resolve_ab_masks(state)


def _resolve_pairs(state, subtract=False):
    """``scope_pairs`` → list of ``{mask_a, mask_b, a_label, b_label}`` or (None, reason).
    Pairs whose regions are gone / empty are dropped; ``subtract`` removes A from B
    (mirrors ``_pairs_from_scope``)."""
    raw = state.get("scope_pairs") or []
    if not raw:
        return None, "Needs at least one region pair."
    regions = state.get("regions") or {}
    out = []
    for a_names, b_names in raw:
        a_masks = [regions[n] for n in a_names if n in regions]
        b_masks = [regions[n] for n in b_names if n in regions]
        if not a_masks or not b_masks:
            continue
        ma, mb = _union(a_masks), _union(b_masks)
        if ma is None or mb is None:
            continue
        if subtract:
            mb = mb & ~ma
        if not ma.any() or not mb.any():
            continue
        out.append({"mask_a": ma, "mask_b": mb,
                    "a_label": " + ".join(a_names), "b_label": " + ".join(b_names)})
    if not out:
        return None, "None of the region pairs have pixels on this slide."
    return out, None


def _resolve(state: dict, sd: StepDef):
    """Single source of truth for both public functions: walk ``sd.needs`` in the exact order
    flowdialog's ``_resolve_inputs`` does, assembling ``inp`` and collecting every unmet
    reason. Returns ``(inp, errors)``; ``errors == []`` iff the step is runnable."""
    errors: list[str] = []
    inp: dict = {"mask": state.get("mask")}                     # always first, like flowdialog
    mzs = [float(m) for m in (state.get("mzs") or [])]
    if "feature_set" in sd.needs and not mzs:
        errors.append("Needs a feature list.")
    inp["mzs"] = mzs                                            # set unconditionally

    # 'groups' — general case is SKIPPED for region_membership (its own branch runs below).
    if "groups" in sd.needs and sd.id != "region_membership":
        labels, names = _resolve_group_labels(state)
        if labels is None:
            errors.append(names)                               # reason is in slot 2
        else:
            inp["labels"], inp["names"] = labels, names

    if sd.id == "region_membership":                           # hard-coded special case
        masks, names = _resolve_group_masks_named(state)
        if masks is None:
            errors.append(names)
        else:
            inp["masks"], inp["names"] = masks, names

    if sd.id == "region_correlation":                          # every named region, not groups
        import numpy as np
        regions = state.get("regions") or {}
        named = {n: m for n, m in regions.items()
                 if m is not None and bool(np.asarray(m, dtype=bool).any())}
        if len(named) < 2:
            errors.append("Needs at least two named regions.")
        else:
            inp["region_masks"] = named

    if "region" in sd.needs:                                   # first labelled group's mask
        groups = state.get("groups") or {}
        names = list(groups)
        if not names:
            errors.append("Needs a region assigned to a group.")
        else:
            mk = _union(groups[names[0]])
            if mk is None or not mk.any():
                errors.append(f"Group '{names[0]}' has no pixels.")
            else:
                inp["mask_a"] = mk

    if "ab" in sd.needs:
        ma, mb, la, lb, samples = _resolve_ab(state)
        if ma is None:
            errors.append(mb)                                  # reason is in slot 2
        else:
            inp.update(mask_a=ma, mask_b=mb, a_label=la, b_label=lb, samples=samples)

    if "ab_multi" in sd.needs:
        pairs, err = _resolve_pairs(state, subtract=bool(state.get("subtract_a_from_b")))
        if pairs is None:
            errors.append(err)
        else:
            inp["pairs"] = pairs

    if "target_mz" in sd.needs:
        t = state.get("target_mz")
        if t is None:              # is-None, NOT falsiness — 0.0 is falsy but a valid m/z
            errors.append("Needs a target m/z.")
        else:
            inp["target_mz"] = float(t)

    if "stats" in sd.needs:                                    # chained from an earlier stats step
        sdf = state.get("stats_df")
        if sdf is None or not len(sdf):
            errors.append("Needs a stats table (AUC) from an earlier Region-comparison / "
                          "ROI-localization step.")
        else:
            auc_col = ("AUC" if "AUC" in sdf.columns
                       else "roi_auc" if "roi_auc" in sdf.columns else None)
            if auc_col is None:
                errors.append("The earlier step produced no AUC column to filter on.")
            else:
                inp["stats_df"], inp["auc_col"] = sdf, auc_col

    return inp, errors


def unmet_needs(state: dict, sd: StepDef) -> list[str]:
    """Short, user-facing reasons a step can't run yet against ``state`` — ``[]`` means ready.

    Pure mirror of the gating in ``gui/flowdialog.py`` ``_resolve_inputs`` (see the ``state``
    schema above the helpers). Every token in :data:`NEEDS` is covered:

    * ``feature_set`` — needs a non-empty ``mzs``.
    * ``groups`` — needs >=2 tagged groups, OR falls back to ``seg_labels`` when fewer;
      the ``region_membership`` step is the exception (below) and takes NO seg fallback.
    * ``region_membership`` (a ``groups`` step by id) — needs >=2 groups, each non-empty.
    * ``region`` — needs >=1 tagged group with pixels (uses the first).
    * ``ab`` — with per-step ``scope_a``/``scope_b`` set: >=1 region each side, overlap ok;
      else the first two Setup groups, both non-empty and NON-overlapping.
    * ``ab_multi`` — needs >=1 region pair that survives (regions present, non-empty).
    * ``target_mz`` — tested with ``is None`` (NOT falsiness): ``0.0`` is a valid m/z and
      counts as present; only a missing value is unmet.
    * ``stats`` — needs a non-empty stats table carrying an ``AUC`` / ``roi_auc`` column.
    """
    return _resolve(state, sd)[1]


def resolve_inputs(state: dict, sd: StepDef) -> dict:
    """Assemble the ``inputs`` dict handed to ``sd.run(ds, inp, params)`` from a Qt-free
    ``state`` (schema documented above the helpers) — the pure counterpart of flowdialog's
    ``_resolve_inputs``. Keys produced per token match the runner exactly:

        always            mask, mzs
        groups            labels, names           (names is None on the seg fallback)
        region_membership masks, names
        region            mask_a
        ab                mask_a, mask_b, a_label, b_label, samples
        ab_multi          pairs
        target_mz         target_mz
        stats             stats_df, auc_col

    Call :func:`unmet_needs` first; if any need is unmet this raises ``ValueError`` with the
    first reason (0.0 is a valid ``target_mz``, unlike flowdialog's falsiness check)."""
    inp, errors = _resolve(state, sd)
    if errors:
        raise ValueError(errors[0])
    return inp
