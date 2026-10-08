"""Programmatic analysis API + saveable Python *workflows*.

This module is the **imperative** analysis surface: write a short Python *script* that drives
the analysis engine directly, run it against the loaded slide, and save the script as a
reusable *workflow* preset. Since the Flow *designer* was retired (plan 24), this scripting
API is the batch/replay surface — it runs the same :mod:`smile_msi.registry` steps a GUI
analysis does. :func:`capabilities_doc` emits a self-describing guide so an AI assistant can
write correct scripts without reading the source.

Three pieces, all **pure** (no Qt — testable headless, mirroring :mod:`smile_msi.registry`):

* :class:`ScriptAPI` — a thin, friendly facade bound to one ``MSIDataset``. Its analysis
  methods (``find_peaks``, ``segment``, ``compare``, ``colocalize``, …) delegate to the
  **same** :data:`smile_msi.registry.REGISTRY` callables a GUI analysis uses, so behaviour
  never drifts between the two. It also resolves the script's notion of *features*,
  *regions* and *groups* into the array arguments the engine wants.
* :func:`run_script` / :func:`build_namespace` — execute user code in a namespace where the
  API methods are bare names (``find_peaks(...)``) plus a handful of output helpers
  (``log`` / ``table`` / ``image`` / ``record``). Captures stdout, tables, images, named
  values and any traceback into a :class:`ScriptResult` the GUI renders.
* :class:`Workflow` + the on-disk preset store (``~/.smile-msi/workflows/``) — save / load /
  list, mirroring the :mod:`smile_msi.cohort` preset pattern.

Engine modules (spatial / multivariate / annotate) stay lazily imported (inside the registry
step callables), so importing this module — which the GUI does at startup — stays cheap.
"""
from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
import traceback
from dataclasses import dataclass, field

import numpy as np

from . import registry, library, session

VERSION = 1

#: threshold_mask() logs a warning when filling holes grows the mask by more than this
#: fraction — speckle holes are a few pixels; a filled annulus is its whole interior.
FILL_WARN_FRAC = 0.10


class ScriptError(Exception):
    """Raised for *user* mistakes (no features yet, unknown region, …) so the message is a
    plain sentence in the console, not a wall of internal traceback."""


# --------------------------------------------------------------------------- #
# the API facade
# --------------------------------------------------------------------------- #
def _mz_list(features) -> list[float]:
    """Normalize a feature set (peak dicts, m/z floats, or None) to a list of m/z floats."""
    out = []
    for f in features or []:
        out.append(float(f["mz"]) if isinstance(f, dict) else float(f))
    return out


class ScriptAPI:
    """The scripting surface for one loaded slide.

    Construct it bound to an :class:`~smile_msi.msi.MSIDataset` plus the analysis defaults
    (``ppm`` extraction tolerance, ``norm`` per-pixel normalization, ``reduce`` m/z-window
    reducer) and, optionally, the named region masks and group→mask map the GUI builds from
    the current ROIs. Every analysis method returns the engine's native result (a list of
    peak dicts, a ``DataFrame``, a ``Segmentation``, …) and most also update ``self.features``
    / ``self.segmentation`` so later calls can default to "the working set".

    Pure: tests construct one with ``masks=`` / ``groups=`` dicts directly; the GUI fills
    those from the live regions before a run.
    """

    #: Methods exposed as bare names in a workflow's namespace (single source of truth for
    #: :func:`build_namespace` *and* the generated guide, so the two never disagree).
    NAMESPACE_FUNCS = (
        "mean_spectrum", "ion_image", "ion_vector", "feature_matrix",
        "find_peaks", "find_spatial_features", "set_features", "get_features",
        "segment", "compare", "discriminating", "multigroup", "markers",
        "roi_localization", "region_membership", "colocalize", "coloc_modules",
        "pca", "nmf", "plsda", "embedding", "annotate", "run",
        "region", "group", "region_names", "group_names",
        "composite", "threshold_mask", "ring", "invert", "add_region",
    )

    def __init__(self, ds, *, ppm: float = 10.0, norm: str = "tic", reduce: str = "sum",
                 masks: dict | None = None, groups: dict | None = None,
                 features=None, active_mz: float | None = None):
        self.ds = ds
        self.ppm = float(ppm)
        self.norm = str(norm)
        self.reduce = str(reduce)
        self.masks = dict(masks or {})              # region name -> bool[n_pixels]
        self.groups = dict(groups or {})            # group label -> bool[n_pixels]
        self.features = _mz_list(features)          # working m/z set later steps default to
        self.active_mz = (float(active_mz) if active_mz is not None else None)
        self.segmentation = None                    # last segment() result
        self.last_peaks = []                        # last find_*() peak dicts (for "apply to app")
        self.new_regions = []                       # add_region() stagings: {name, mask, color}
        self._result = None                         # set by run_script for log routing

    # ---- logging -------------------------------------------------------- #
    def _log(self, msg: str):
        if self._result is not None:
            self._result.logs.append(str(msg))

    # ---- input resolution ----------------------------------------------- #
    def _features(self, features):
        """The m/z list a step runs on: the explicit ``features=`` arg, else the working set
        from the last find_* call. Raises a friendly error if neither is available."""
        mzs = _mz_list(features) if features is not None else list(self.features)
        if not mzs:
            raise ScriptError("no features yet — call find_peaks() (or find_spatial_features()), "
                              "or pass features=[...m/z...].")
        return mzs

    def _mask(self, m):
        """Resolve ``None`` | name | bool-array | index-list → a bool[n_pixels] or None.
        A name is matched against the regions first, then the tagged groups."""
        if m is None:
            return None
        if isinstance(m, str):
            src = self.masks.get(m, self.groups.get(m))
            if src is None:
                avail = sorted(set(self.masks) | set(self.groups))
                raise ScriptError(f"unknown region/group {m!r}. Available: {avail or '—'}.")
            return np.asarray(src, dtype=bool)
        a = np.asarray(m)
        if a.dtype == bool:
            return a
        out = np.zeros(self.ds.n_pixels, dtype=bool)        # treat as a pixel-index list
        idx = a.astype(int).ravel()
        out[idx[(idx >= 0) & (idx < self.ds.n_pixels)]] = True
        return out

    def _labels(self, groups):
        """Resolve a group spec → ``(labels int[n_pixels], names)`` for multi-group steps.

        ``groups`` may be: ``None`` (use the tagged groups, else the last segmentation),
        a ``{name: mask}`` dict, a list of region/group names, or an int label array.
        """
        if groups is None:
            gmap = self.groups
            if len(gmap) < 2:
                if self.segmentation is not None:
                    return np.asarray(self.segmentation.labels, dtype=int), None
                raise ScriptError("need ≥2 groups — pass groups={'A': maskA, 'B': maskB}, tag "
                                  "regions into groups, or segment() first.")
        elif isinstance(groups, dict):
            gmap = {str(k): self._mask(v) for k, v in groups.items()}
        elif isinstance(groups, (list, tuple)):
            gmap = {str(n): self._mask(n) for n in groups}
        else:                                               # an explicit label array
            return np.asarray(groups, dtype=int), None
        names = list(gmap)
        labels = np.full(self.ds.n_pixels, -1, dtype=int)
        for gi, n in enumerate(names):
            mk = gmap[n]
            if mk is not None:
                labels[(labels < 0) & mk] = gi
        if (labels >= 0).sum() == 0:
            raise ScriptError("the chosen groups cover no pixels.")
        return labels, names

    def _grouped_samples(self, ma, mb):
        """Per-pixel replicate IDs for an across-ROI A/B comparison, derived from tissue
        detection (:func:`spatial.sample_labels`) — the scripting reconstruction of the Flow
        runner's per-ROI replicate array, which restores :func:`spatial.roi_comparison`'s
        across-replicate (``unit=='sample'``) inference for the ``@grouped`` scope.

        Returns ``None`` when the slide has no resolvable replicate structure — a single
        tissue piece (``detect_samples`` returns ``[]`` → all ``-1``), a group whose pixels
        land off detected tissue, or fewer than two distinct replicates across the two sides —
        so the caller stays on the honest per-pixel path instead of crashing on an empty
        per-sample summary or mislabelling one tissue piece as replicated."""
        from . import spatial
        smp = spatial.sample_labels(self.ds)
        if smp is None:
            return None
        smp = np.asarray(smp, dtype=int)
        a = np.asarray(ma, dtype=bool)
        b = np.asarray(mb, dtype=bool)
        ids_a = {int(s) for s in np.unique(smp[a]) if s >= 0}
        ids_b = {int(s) for s in np.unique(smp[b]) if s >= 0}
        if not ids_a or not ids_b or len(ids_a | ids_b) < 2:
            return None
        return smp

    def _run_step(self, step_id, inputs, params):
        """Dispatch through the Flow registry so scripted + designed analyses share one code
        path. Backfills registry-default params so partial kwargs never KeyError mid-engine."""
        sd = registry.REGISTRY.get(step_id)
        if sd is None:
            raise ScriptError(f"unknown analysis {step_id!r}. "
                              f"Known: {sorted(registry.REGISTRY)}.")
        p = {**registry.default_params(step_id), **(params or {})}
        inputs.setdefault("mask", None)
        return sd.run(self.ds, inputs, p)

    # ---- region / group accessors --------------------------------------- #
    def region(self, name):
        """The boolean pixel mask of a named region/ROI."""
        return self._mask(name)

    def group(self, name):
        """The boolean pixel mask of a named group (the union of its tagged ROIs)."""
        if name not in self.groups:
            raise ScriptError(f"unknown group {name!r}. Available: {sorted(self.groups) or '—'}.")
        return np.asarray(self.groups[name], dtype=bool)

    def region_names(self):
        """Names of the regions/ROIs available to this script."""
        return sorted(self.masks)

    def group_names(self):
        """Names of the groups (A/B/…) tagged on this slide's ROIs."""
        return sorted(self.groups)

    # ---- regions by construction ---------------------------------------- #
    def composite(self, features, weight: str = "raw"):
        """Per-pixel summed intensity of several m/z (a lipid class, the sulfatides, …)."""
        mzs = _mz_list(features)
        if not mzs:
            raise ScriptError("composite() needs at least one m/z.")
        return self.ds.composite_vector(mzs, tol_ppm=self.ppm, reduce=self.reduce,
                                        norm=self.norm, weight=weight)

    def threshold_mask(self, values, cut, percentile: bool = True, fill_holes: bool = True,
                       min_pixels: int = 0):
        """Mask of pixels where a signal (per-pixel vector, or one m/z) reaches ``cut``.
        ``cut`` is a percentile of every pixel with signal > 0 by default — off-tissue pixels
        with any noise count too (``percentile=False`` makes it an absolute intensity).
        **Holes are filled by default**: a rim/annulus becomes a solid disc, so pass
        ``fill_holes=False`` for one. Islands under ``min_pixels`` are dropped."""
        from . import spatial
        v = self.ion_vector(float(values)) if np.ndim(values) == 0 else np.asarray(values, float)
        if v.shape[0] != self.ds.n_pixels:
            raise ScriptError(f"threshold_mask() wants one value per pixel ({self.ds.n_pixels}), "
                              f"got {v.shape[0]}.")
        mask = spatial.threshold_mask(self.ds, v, cut, percentile=percentile,
                                      fill_holes=fill_holes, min_pixels=min_pixels)
        if fill_holes:
            raw = spatial.threshold_mask(self.ds, v, cut, percentile=percentile,
                                         fill_holes=False, min_pixels=min_pixels)
            added = int(mask.sum()) - int(raw.sum())
            if added > FILL_WARN_FRAC * max(1, int(raw.sum())):
                self._log(f"threshold_mask: filling holes added {added:,} px to "
                          f"{int(raw.sum()):,} — if the signal is a rim or ring, pass "
                          "fill_holes=False")
        return mask

    def ring(self, mask, width_px=None, width_um=None, mode: str = "outer"):
        """A rim (inner) / collar (outer) / band of the given width around a mask's boundary.
        Euclidean distance transform on the tissue grid; ``width_um`` needs the slide's
        pixel size, ``width_px`` always works."""
        from . import spatial
        if width_px is None:
            if width_um is None:
                raise ScriptError("ring() needs width_px= or width_um=.")
            px = getattr(self.ds, "pixel_size_um", None)
            if not px:
                raise ScriptError("this slide records no pixel size — pass width_px= instead.")
            width_px = float(width_um) / float(px)
        if mode not in ("inner", "outer", "band"):
            raise ScriptError("ring() mode must be 'inner', 'outer' or 'band'.")
        m = self._mask(mask)
        if m is None:
            raise ScriptError("ring() needs a mask or region name.")
        return spatial.ring_mask(self.ds, m, float(width_px), mode=mode)

    def invert(self, mask):
        """Every acquired pixel NOT in ``mask`` (the 'everything else' region)."""
        from . import spatial
        m = self._mask(mask)
        if m is None:
            raise ScriptError("invert() needs a mask or region name.")
        return spatial.invert_mask(self.ds, m)

    def add_region(self, name, mask, color=None):
        """Stage a named region from a mask (usable by name from here on; pushable into the app).
        The console's Apply to app ▸ Add regions to the slide creates the staged regions."""
        m = self._mask(mask)
        if m is None or not m.any():
            raise ScriptError(f"region {name!r} has no pixels.")
        name = str(name)
        self.masks[name] = m
        self.new_regions = [r for r in self.new_regions if r["name"] != name]
        self.new_regions.append({"name": name, "mask": m, "color": color})
        self._log(f"add_region {name!r} → {int(m.sum()):,} px")
        return m

    # ---- raw signal ----------------------------------------------------- #
    def mean_spectrum(self, mask=None):
        """``(mz_axis, intensities)`` mean spectrum over the whole slide or a region/mask."""
        return self.ds.mean_spectrum(mask=self._mask(mask))

    def ion_image(self, mz):
        """2-D ion image (H×W array) for one m/z at the current ppm / reduce / norm."""
        return self.ds.ion_image(float(mz), tol_ppm=self.ppm, reduce=self.reduce, norm=self.norm)

    def ion_vector(self, mz):
        """Per-pixel intensity vector (length n_pixels) for one m/z."""
        return self.ds.ion_vector(float(mz), tol_ppm=self.ppm, reduce=self.reduce, norm=self.norm)

    def feature_matrix(self, features=None):
        """The pixels×features intensity matrix for the working (or given) feature set."""
        from . import spatial
        return spatial.feature_matrix(self.ds, self._features(features), tol_ppm=self.ppm, norm=self.norm)

    # ---- features ------------------------------------------------------- #
    def set_features(self, features):
        """Set the working feature set later steps default to (m/z floats or peak dicts)."""
        self.features = _mz_list(features)
        return self.features

    def get_features(self):
        """The current working feature set (list of m/z floats)."""
        return list(self.features)

    def find_peaks(self, snr: float = 3.0, min_rel_intensity: float = 0.0,
                   max_peaks: int = 0, prominence: float = 1.0, mask=None):
        """Detect peaks in the mean spectrum → set + return the working feature set.
        ``max_peaks=0`` (the default) keeps every peak that clears the gates; the result's
        ``.n_detected`` reports the count before any cap."""
        peaks = self._run_step("find_peaks", {"mask": self._mask(mask)},
                               dict(snr=snr, min_rel_intensity=min_rel_intensity,
                                    max_peaks=max_peaks, prominence=prominence))
        self.features = _mz_list(peaks)
        self.last_peaks = list(peaks)
        n_det = int(getattr(peaks, "n_detected", len(peaks)))
        capped = f" (capped from {n_det} detected)" if n_det > len(peaks) else ""
        self._log(f"find_peaks → {len(peaks)} peaks{capped}")
        return peaks

    def find_spatial_features(self, snr: float = 3.0, min_rel_intensity: float = 0.0,
                              min_frequency: float = 0.0, min_morans: float = 0.0, mask=None,
                              max_candidates: int = 2000):
        """Spatially-aware peak detection (S/N → reproducibility → Moran's I) → working set.
        ``max_candidates=0`` lifts the candidate-pool cap (one Moran's I pass per extra
        candidate); ``.candidates_capped`` says whether the cap bound."""
        res = self._run_step("find_spatial_features", {"mask": self._mask(mask)},
                             dict(snr=snr, min_rel_intensity=min_rel_intensity,
                                  min_frequency=min_frequency, min_morans=min_morans,
                                  max_candidates=max_candidates,
                                  tol_ppm=self.ppm, norm=self.norm))
        self.features = [float(p["mz"]) for p in res.peaks]
        self.last_peaks = list(res.peaks)
        capped = f" (capped from {res.n_detected} detected)" if res.candidates_capped else ""
        self._log(f"find_spatial_features → {len(res.peaks)} of "
                  f"{res.n_candidates} candidates{capped}")
        return res

    # ---- mass calibration ----------------------------------------------- #
    def measure_calibration(self, region=None, mode: str = "negative", tol_ppm: float = 30.0):
        """Measure the absolute m/z offset against known anchor ions. ``region`` (name, bool
        mask or pixel-index list) measures one region instead of the whole slide — samples in
        different embedding media drift by different amounts, and a slide-wide number is the
        average of those populations. Returns the :func:`intake.measure_calibration_offset`
        dict."""
        from . import intake
        res = intake.measure_calibration_offset(self.ds, mode=mode, tol_ppm=tol_ppm,
                                                mask=self._mask(region))
        where = region if isinstance(region, str) else ("whole slide" if region is None
                                                        else "region")
        self._log(f"measure_calibration[{where}] → median {res['median_ppm']:+.2f} ppm "
                  f"(n={res['n']}, slope {res['slope_ppm_per_da']:+.3f} ppm/Da)")
        return res

    def calibrate_regions(self, regions=None, mode: str = "negative", tol_ppm: float = 30.0):
        """Measure every named region separately and apply a **per-region** lock-mass
        recalibration to the dataset. ``regions`` defaults to every region the script can
        see. Pixels in no region are left uncorrected. The correction is axis-preserving, so
        all pixels stay on one shared m/z axis. Returns ``{region: measurement}``."""
        from . import intake
        from . import preprocess as pp
        # names only: the config is persisted and re-resolved by name on reopen, so a raw
        # mask array has nothing to resolve back to
        names = [str(n) for n in (regions or self.masks)]
        if not names:
            raise ScriptError("no regions to calibrate — draw or load regions first.")
        from .provenance import mask_fingerprint
        out, specs = {}, []
        for n in names:
            m = self._mask(n)
            res = intake.measure_calibration_offset(self.ds, mode=mode, tol_ppm=tol_ppm, mask=m)
            out[n] = res
            if res["anchors"]:
                # the window must exceed this region's own offset so recalibrate locks onto
                # the right apex
                specs.append((n, m, [round(float(a["ref_mz"]), 4) for a in res["anchors"]],
                              round(abs(float(res["max_abs_ppm"])) * 3.0 + 10.0, 2)))
        if not specs:
            raise ScriptError("no anchor ions matched in any region — check mode/polarity.")
        # merge into the pipeline already on the slide — a session's baseline / smoothing /
        # normalization must survive adding a calibration, not be replaced by it
        cfg = dict(self.ds.preprocessing or {})
        rc = dict(cfg.get("recalibrate") or {})
        rc.setdefault("refs", [])
        rc.setdefault("tol_ppm", 200.0)
        rc["n_pixels"] = int(self.ds.n_pixels)
        rc["regions"] = [{"name": n, "refs": r, "tol_ppm": t, "mask_sha1": mask_fingerprint(m)}
                         for n, m, r, t in specs]
        cfg["recalibrate"] = rc
        self.ds.set_preprocessing(
            pp.build_pipeline(cfg, masks={n: m for n, m, _, _ in specs}), config=cfg)
        self._log("calibrate_regions → " + ", ".join(
            f"{n} {out[n]['median_ppm']:+.2f} ppm" for n in names if out[n]["n"]))
        return out

    # ---- segmentation --------------------------------------------------- #
    def segment(self, features=None, n_clusters: int = 0, spatial: bool = True,
                spatial_sigma: float = 1.0, mask=None):
        """Cluster pixels into regions (``n_clusters=0`` → auto by silhouette). Sets
        ``self.segmentation``; ``.labels`` is the per-pixel cluster array (``-1`` outside
        ``mask``). Without ``mask`` every acquired pixel is clustered, off-tissue background
        included — pass a tissue region to cluster the tissue alone."""
        m = self._mask(mask)
        res = self._run_step("auto_segment", {"mzs": self._features(features), "mask": m},
                             dict(n_clusters=n_clusters, spatial=spatial,
                                  spatial_sigma=spatial_sigma, tol_ppm=self.ppm, norm=self.norm))
        self.segmentation = res
        where = "whole slide" if m is None else f"{int(m.sum()):,} px in mask"
        self._log(f"segment → {res.n_clusters} clusters (silhouette {res.silhouette:.2f}; {where})")
        if m is None:
            from . import spatial as _spatial
            bg = _spatial.background_clusters(self.ds, res, tol_ppm=self.ppm)
            if bg:
                self._log(f"segment: clusters {sorted(bg)} look like off-tissue background — "
                          "pass mask= to cluster tissue only")
        return res

    # ---- statistics ----------------------------------------------------- #
    def compare(self, a, b, features=None, method: str = "mwu", samples=None):
        """Per-ion comparison of two groups A vs B (AUC + signed log2_fc + test) → DataFrame.
        ``a`` / ``b`` are group/region names or boolean masks."""
        ma, mb = self._mask(a), self._mask(b)
        if ma is None or mb is None or not ma.any() or not mb.any():
            raise ScriptError("both groups must resolve to pixels.")
        inp = {"mzs": self._features(features), "mask_a": ma, "mask_b": mb,
               "a_label": a if isinstance(a, str) else "Group A",
               "b_label": b if isinstance(b, str) else "Group B",
               "samples": (self._mask(samples) if isinstance(samples, str) else samples)}
        df = self._run_step("roi_comparison", inp, dict(method=method, tol_ppm=self.ppm, norm=self.norm))
        self._log(f"compare {inp['a_label']} vs {inp['b_label']} → {len(df)} ions")
        return df

    def discriminating(self, features=None, groups=None, top_n: int = 15):
        """Top ions enriched in each group vs the rest (one-vs-rest AUC + FDR) → {group: df}."""
        labels, _ = self._labels(groups)
        return self._run_step("discriminating_features", {"mzs": self._features(features), "labels": labels},
                              dict(top_n=top_n, tol_ppm=self.ppm, norm=self.norm))

    def multigroup(self, features=None, groups=None, method: str = "kruskal"):
        """Per-ion test across 3+ groups (Kruskal-Wallis or ANOVA) → DataFrame."""
        labels, _ = self._labels(groups)
        return self._run_step("multigroup_features", {"mzs": self._features(features), "labels": labels},
                              dict(method=method, tol_ppm=self.ppm, norm=self.norm))

    def markers(self, features=None, groups=None, shrink: float = 2.0, top_n: int = 15):
        """Nearest shrunken centroids — automatic marker ions per group → {group: df}."""
        labels, _ = self._labels(groups)
        return self._run_step("shrunken_centroids", {"mzs": self._features(features), "labels": labels},
                              dict(shrink=shrink, top_n=top_n, tol_ppm=self.ppm, norm=self.norm))

    def roi_localization(self, region, features=None):
        """How confined each ion is to one region vs the rest of the tissue → DataFrame."""
        mk = self._mask(region)
        if mk is None or not mk.any():
            raise ScriptError("the region must resolve to pixels.")
        return self._run_step("roi_localization", {"mzs": self._features(features), "mask_a": mk},
                              dict(tol_ppm=self.ppm, norm=self.norm))

    def region_membership(self, groups=None, features=None, min_prevalence: float = 0.5):
        """Partition ions by which groups they're present in (Venn compartments) → dict."""
        if groups is None:
            groups = self.group_names()
        if isinstance(groups, dict):
            names, masks = list(groups), [self._mask(v) for v in groups.values()]
        else:
            names, masks = list(groups), [self._mask(n) for n in groups]
        if len(masks) < 2:
            raise ScriptError("need ≥2 groups for region_membership.")
        return self._run_step("region_membership", {"mzs": self._features(features), "masks": masks, "names": names},
                              dict(min_prevalence=min_prevalence, tol_ppm=self.ppm, norm=self.norm))

    # ---- co-localization ------------------------------------------------ #
    def colocalize(self, target_mz=None, features=None, method: str = "pearson", mask=None):
        """Rank all features by spatial similarity to a target m/z (defaults to active ion)."""
        t = target_mz if target_mz is not None else self.active_mz
        if not t:
            raise ScriptError("pass target_mz=… (no active ion set).")
        return self._run_step("colocalize", {"mzs": self._features(features), "mask": self._mask(mask)},
                              dict(target_mz=float(t), method=method, tol_ppm=self.ppm, norm=self.norm))

    def coloc_modules(self, features=None, n_modules: int = 0, threshold: float = 0.5,
                      method: str = "pearson", mask=None):
        """Group features into co-localized modules (``n_modules=0`` → auto)."""
        return self._run_step("coloc_modules", {"mzs": self._features(features), "mask": self._mask(mask)},
                              dict(n_modules=n_modules, threshold=threshold, method=method,
                                   tol_ppm=self.ppm, norm=self.norm))

    # ---- multivariate --------------------------------------------------- #
    def pca(self, features=None, n_components: int = 5):
        """Principal-component score images + loadings."""
        return self._run_step("pca", {"mzs": self._features(features)},
                              dict(n_components=n_components, tol_ppm=self.ppm, norm=self.norm))

    def nmf(self, features=None, n_components: int = 5):
        """Non-negative matrix factorization (additive parts) score images + loadings."""
        return self._run_step("nmf", {"mzs": self._features(features)},
                              dict(n_components=n_components, tol_ppm=self.ppm, norm=self.norm))

    def plsda(self, features=None, groups=None, n_components: int = 2, orthogonal: bool = False):
        """Supervised PLS-DA / OPLS-DA classification; ranks ions by VIP."""
        labels, _ = self._labels(groups)
        return self._run_step("plsda", {"mzs": self._features(features), "labels": labels},
                              dict(n_components=n_components, orthogonal=orthogonal,
                                   tol_ppm=self.ppm, norm=self.norm))

    def embedding(self, features=None, method: str = "umap"):
        """2-D pixel embedding (UMAP / t-SNE) + molecular-similarity tissue map."""
        return self._run_step("embedding", {"mzs": self._features(features)},
                              dict(method=method, tol_ppm=self.ppm, norm=self.norm))

    # ---- annotation ----------------------------------------------------- #
    def annotate(self, features=None, mode: str = "negative", match_ppm: float = 5.0,
                 image_ppm: float = 10.0):
        """Match the feature set to the lipid database (isotopes + adducts + confidence) → DataFrame."""
        return self._run_step("annotate", {"mzs": self._peak_records(self._features(features))},
                              dict(mode=mode, match_ppm=match_ppm, image_ppm=image_ppm, norm=self.norm))

    def _peak_records(self, mzs, tol_ppm: float = 2.0):
        """``mzs`` as the last find_*() peak dicts where one sits within ``tol_ppm`` (so the
        intensity / S/N columns are filled and isotopologues can be told from their parent),
        else the bare m/z."""
        if not self.last_peaks:
            return mzs
        known = np.array([float(p["mz"]) for p in self.last_peaks])
        order = np.argsort(known)
        known = known[order]
        out = []
        for mz in mzs:
            i = int(np.searchsorted(known, mz))
            j = min((k for k in (i - 1, i) if 0 <= k < len(known)), key=lambda k: abs(known[k] - mz))
            hit = abs(known[j] - mz) <= mz * tol_ppm / 1e6
            out.append(dict(self.last_peaks[order[j]], mz=float(mz)) if hit else mz)
        return out

    # ---- generic escape hatch ------------------------------------------- #
    def run(self, step_id, features=None, groups=None, mask=None, target_mz=None, **params):
        """Run any registered analysis by id (see the guide's step table) with raw params —
        the forward-compatible escape hatch for steps without a named wrapper."""
        sd = registry.REGISTRY.get(step_id)
        if sd is None:
            raise ScriptError(f"unknown analysis {step_id!r}. Known: {sorted(registry.REGISTRY)}.")
        inp = {"mask": self._mask(mask)}
        if "feature_set" in sd.needs or sd.produces == {"peaks"}:
            inp["mzs"] = self._features(features) if (features is not None or "feature_set" in sd.needs) else list(self.features)
        if "groups" in sd.needs and step_id != "region_membership":
            inp["labels"], inp["names"] = self._labels(groups)
        if target_mz is not None:
            params.setdefault("target_mz", float(target_mz))
        # the session's extraction settings, as the named wrappers pass them — the registry
        # default tolerance would extract a different window than the session's ion images
        step_defaults = registry.default_params(step_id)
        for key, value in (("tol_ppm", self.ppm), ("norm", self.norm)):
            if key in step_defaults:
                params.setdefault(key, value)
        return self._run_step(step_id, inp, params)

    def _commit(self, sd, res):
        """Chain a step's outputs into the session's working state so a *following*
        ``run_analysis`` picks them up with no explicit hand-off — the scripting analogue of
        the Flow runner's ``ctx`` (peaks) and the ``segment()`` wrapper's ``self.segmentation``.
        A step that yields neither leaves the working state untouched."""
        try:
            pk = sd.peaks(res)
        except Exception:  # noqa: BLE001 — a step with no peaks hook just chains nothing forward
            pk = None
        if pk:
            self.features = _mz_list(pk)
            self.last_peaks = list(pk)
        if getattr(res, "labels", None) is not None and "segmentation" in sd.produces:
            self.segmentation = res
        return res

    def run_analysis(self, step_id: str, **params):
        """Public, documented alias for :meth:`_run_step`: run any registered analysis by id,
        dispatched through the **same** :data:`smile_msi.registry.REGISTRY` (registry defaults
        backfilled) that both the named wrappers above and the retired Flow designer use.

        This is the stable entry point a saved *scripting* preset calls
        (``s.run_analysis("auto_segment", n_clusters=2)``) — the one that makes migrating a
        Flow to a preset lossless rather than destructive: anything the Flow designer could
        run, a preset can now reproduce through this method against the same engine.

        Inputs are assembled from the session's current features / regions / groups exactly as
        the named wrappers (``find_peaks``, ``segment``, ``compare``, ``discriminating``,
        ``markers``, ``roi_localization`` …) do, keyed off the step's declared ``needs``.
        Override any of them via ``**params`` (popped before the rest go to the engine):

        * ``features=[…m/z…]`` — feature set (else the working set from the last find_*);
        * ``mask=`` — region/group name or bool array to restrict a mask-aware step to; the
          analogue of a Flow's region scope, and (like the Flow runner) attached to *every*
          step's inputs whether or not the engine consumes it;
        * ``a=`` / ``b=`` — the two groups of an ``ab`` (two-group) comparison; default = the
          first two tagged groups, mirroring the Flow Setup table's first-two-groups rule;
        * ``region=`` — the single group/region a ``region``-need step localizes (default =
          the first tagged group);
        * ``groups=`` — ``{name: mask}`` / list of names / label array for multi-group steps;
        * ``target_mz=`` — the co-localization target.

        Every other keyword is forwarded to the engine as a tunable param.
        """
        sd = registry.REGISTRY.get(step_id)
        if sd is None:
            raise ScriptError(f"unknown analysis {step_id!r}. Known: {sorted(registry.REGISTRY)}.")
        features = params.pop("features", None)
        groups = params.pop("groups", None)
        mask = params.pop("mask", None)
        target_mz = params.pop("target_mz", None)
        a, b = params.pop("a", None), params.pop("b", None)
        region = params.pop("region", None)
        samples = params.pop("samples", None)

        inp = {"mask": self._mask(mask)}
        if "feature_set" in sd.needs or sd.produces == {"peaks"}:
            want = features is not None or "feature_set" in sd.needs
            inp["mzs"] = self._features(features) if want else list(self.features)
        if "groups" in sd.needs and step_id != "region_membership":
            inp["labels"], inp["names"] = self._labels(groups)
        if step_id == "region_membership":
            if groups is None:
                names = self.group_names()
                masks = [self._mask(n) for n in names]
            elif isinstance(groups, dict):
                names, masks = list(groups), [self._mask(v) for v in groups.values()]
            else:
                names, masks = list(groups), [self._mask(n) for n in groups]
            if len(masks) < 2:
                raise ScriptError("need ≥2 groups for region_membership.")
            inp["masks"], inp["names"] = masks, names
        if "region" in sd.needs:                            # a single labelled group's mask
            if region is not None:
                mk = self._mask(region)
            else:
                names = list(self.groups)                    # first-seen order, like the Flow Setup
                if not names:
                    raise ScriptError("tag at least one region into a group, or pass region=….")
                mk = self.group(names[0])
            if mk is None or not mk.any():
                raise ScriptError("the region must resolve to pixels.")
            inp["mask_a"] = mk
        if "ab" in sd.needs:
            if a is not None and b is not None:
                ma, mb = self._mask(a), self._mask(b)
                la = a if isinstance(a, str) else "Group A"
                lb = b if isinstance(b, str) else "Group B"
                from_groups = False
            else:                                            # first two tagged groups (Flow's rule)
                names = list(self.groups)
                if len(names) < 2:
                    raise ScriptError("need ≥2 tagged groups for a two-group comparison "
                                      "(tag ROIs into groups, or pass a=…, b=…).")
                la, lb = names[0], names[1]
                ma, mb = self.group(la), self.group(lb)
                from_groups = True
            if ma is None or mb is None or not ma.any() or not mb.any():
                raise ScriptError("both groups must resolve to pixels.")
            if samples is not None:
                smp = self._mask(samples) if isinstance(samples, str) else samples
            elif from_groups:
                # No explicit replicate labels, and the two sides come from grouped ROIs (the
                # Flow designer's @grouped scope). The Flow runner treated each ROI as an
                # independent replicate so roi_comparison tested *across* replicates rather than
                # pseudoreplicating over spatially-correlated pixels. Scripting keeps only the
                # groups' union masks, so reconstruct the replicate structure from tissue
                # detection (spatial.sample_labels) — restoring df.attrs['unit']=='sample'.
                smp = self._grouped_samples(ma, mb)
            else:
                smp = None
            inp.update(mask_a=ma, mask_b=mb, a_label=la, b_label=lb, samples=smp)
        if target_mz is not None:
            params.setdefault("target_mz", float(target_mz))
        return self._commit(sd, self._run_step(step_id, inp, params))


# --------------------------------------------------------------------------- #
# the script result + namespace
# --------------------------------------------------------------------------- #
@dataclass
class ScriptResult:
    """Everything a workflow run produced, for the console to render: captured ``stdout``,
    ``log()`` lines, surfaced ``tables`` / ``images``, named ``values``, and a ``error``
    traceback string (None on success). ``features`` / ``segmentation`` carry the API's
    end-state so the console can offer to apply them to the app."""
    stdout: str = ""
    logs: list = field(default_factory=list)
    tables: list = field(default_factory=list)      # [(title, DataFrame)]
    images: list = field(default_factory=list)      # [(title, ndarray2d, meta dict)]
    values: dict = field(default_factory=dict)
    error: str | None = None
    features: list = field(default_factory=list)
    peaks: list = field(default_factory=list)        # last find_*() peak dicts (apply-to-app)
    segmentation: object = None
    regions: list = field(default_factory=list)      # add_region() stagings (apply-to-app)

    @property
    def ok(self) -> bool:
        return self.error is None

    def summary(self) -> str:
        if self.error:
            return self.error.strip().splitlines()[-1] if self.error.strip() else "error"
        bits = []
        if self.tables:
            bits.append(f"{len(self.tables)} table(s)")
        if self.images:
            bits.append(f"{len(self.images)} image(s)")
        if self.values:
            bits.append(f"{len(self.values)} value(s)")
        if self.features:
            bits.append(f"{len(self.features)} features")
        if self.regions:
            bits.append(f"{len(self.regions)} region(s)")
        return ", ".join(bits) or "ran (no output surfaced)"


def build_namespace(api: ScriptAPI, result: ScriptResult) -> dict:
    """The globals a workflow script runs in: the API methods as bare names, the session
    handle (``s`` / ``api``, the same :class:`ScriptAPI`), the dataset (``ds``) and ``np``,
    plus the output helpers ``log`` / ``table`` / ``image`` / ``record``."""
    api._result = result

    def log(*args):
        """Append a line to the run log (also returns the string)."""
        msg = " ".join(str(a) for a in args)
        result.logs.append(msg)
        return msg

    def table(df, title=""):
        """Surface a DataFrame (or list-of-dicts) as a result table."""
        import pandas as pd
        if df is None:
            return None
        d = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
        result.tables.append((str(title), d))
        return d

    def image(x, title=""):
        """Surface an ion image. ``x`` may be an m/z (float), a per-pixel vector, or a 2-D array."""
        arr = np.asarray(x)
        if arr.ndim == 0:                                  # an m/z scalar
            arr = api.ion_image(float(x))
        elif arr.ndim == 1:                                # a per-pixel vector
            arr = api.ds.to_image(arr)
        meta = {"mz": float(x)} if np.asarray(x).ndim == 0 else {}
        result.images.append((str(title), np.asarray(arr, dtype=float), meta))
        return arr

    def record(name, value):
        """Stash a named value to inspect after the run."""
        result.values[str(name)] = value
        return value

    ns = {"__builtins__": __builtins__, "np": np, "numpy": np,
          "api": api, "s": api, "ds": api.ds, "active_mz": api.active_mz,
          "log": log, "table": table, "image": image, "record": record}
    for name in ScriptAPI.NAMESPACE_FUNCS:
        ns[name] = getattr(api, name)
    return ns


def run_script(code: str, api: ScriptAPI, *, extra_globals: dict | None = None,
               filename: str = "<workflow>") -> ScriptResult:
    """Execute ``code`` against ``api`` and return a :class:`ScriptResult`. stdout/stderr are
    captured; a :class:`ScriptError` is reported as its message alone, any other exception as
    a traceback trimmed to the user's own frames. Never raises — failures land in
    ``result.error`` so the console always has something to show."""
    result = ScriptResult()
    ns = build_namespace(api, result)
    if extra_globals:
        ns.update(extra_globals)
    try:
        compiled = compile(code, filename, "exec")
    except SyntaxError as exc:
        result.error = "".join(traceback.format_exception_only(type(exc), exc))
        return result
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(compiled, ns, ns)
    except ScriptError as exc:
        result.error = str(exc)
    except Exception:                                       # noqa: BLE001 — user code; report, don't crash
        result.error = _user_traceback(filename)
    finally:
        result.stdout = buf.getvalue()
        result.features = list(api.features)
        result.peaks = list(api.last_peaks)
        result.segmentation = api.segmentation
        result.regions = list(api.new_regions)
    return result


def _user_traceback(filename: str) -> str:
    """Format the current exception, dropping the harness frames above the user's ``exec``
    so the reported line numbers point at the workflow, not this module."""
    import sys
    exc_type, exc, tb = sys.exc_info()
    frames = traceback.extract_tb(tb)
    user = [f for f in frames if f.filename == filename]
    head = "Traceback (most recent call last):\n"
    body = "".join(traceback.format_list(user)) if user else "".join(traceback.format_list(frames))
    tail = "".join(traceback.format_exception_only(exc_type, exc))
    return head + body + tail


# --------------------------------------------------------------------------- #
# saved workflows  (~/.smile-msi/workflows/)
# --------------------------------------------------------------------------- #
@dataclass
class Workflow:
    """A named, saveable Python workflow: the script ``code`` + a one-line ``description``.
    Holds no data, so a preset is portable across slides (mirrors :class:`smile_msi.cohort.Cohort`)."""
    name: str = "Workflow"
    code: str = ""
    description: str = ""

    def to_dict(self) -> dict:
        return {"version": VERSION, "name": self.name, "code": self.code,
                "description": self.description}

    @classmethod
    def from_dict(cls, d: dict) -> "Workflow":
        return cls(name=str(d.get("name", "Workflow")), code=str(d.get("code", "")),
                   description=str(d.get("description", "")))

    def save(self, path: str | None = None) -> str:
        """Atomic write (temp + fsync + replace), like :func:`smile_msi.session.save_session`."""
        path = path or workflow_path(self.name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: str) -> "Workflow":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def workflows_dir() -> str:
    """Directory holding saved workflow presets; created on demand."""
    d = os.path.join(library.home_dir(), "workflows")
    os.makedirs(d, exist_ok=True)
    return d


def workflow_path(name: str) -> str:
    """Path of the named workflow's JSON file under :func:`workflows_dir`."""
    return os.path.join(workflows_dir(), f"{session._sanitize(name)}.json")


def list_workflows() -> list[dict]:
    """``{name, path, description, mtime}`` for every saved workflow, newest first; corrupt
    files skipped (mirrors :func:`smile_msi.cohort.list_cohorts`)."""
    out = []
    d = workflows_dir()
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(d, fn)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            mtime = os.stat(path).st_mtime
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        out.append({"name": data.get("name", os.path.splitext(fn)[0]), "path": path,
                    "description": data.get("description", ""), "mtime": mtime})
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


# --------------------------------------------------------------------------- #
# starter examples
# --------------------------------------------------------------------------- #
EXAMPLES: dict = {
    "Find peaks & annotate": '''\
# Detect the working feature set, annotate it against the lipid database,
# and surface the confident hits as a table.
peaks = find_peaks(snr=5)
log(f"{len(peaks)} peaks")
hits = annotate(mode="negative", match_ppm=5)
named = hits[hits["lipid"].astype(str) != ""].copy()
named["confidence_score"] = np.asarray(named["confidence_score"], dtype=float)
table(named.sort_values("confidence_score", ascending=False).head(25), "Top lipid IDs")
log(f"{len(named)} of {len(hits)} features annotated")
''',
    "Segment, then find markers": '''\
# Cluster the tissue, then pull automatic marker ions for each region.
find_peaks(snr=5)
seg = segment(n_clusters=0)          # 0 = auto-pick the cluster count
log(f"{seg.n_clusters} regions, silhouette {seg.silhouette:.2f}")
for region_id, df in markers(top_n=10).items():
    table(df, f"Markers · region {region_id}")
''',
    "Compare two groups (A vs B)": '''\
# Requires two tagged groups (or two named ROIs). Lists ions that differ,
# then renders the most discriminating one as an image.
find_peaks(snr=5)
res = compare("Group A", "Group B", method="mwu")
sig = res[res["q_value"] < 0.05].sort_values("AUC", ascending=False)
table(sig, "Significant ions (q<0.05)")
if len(sig):
    image(float(sig.iloc[0]["mz"]), "Most enriched in A")
''',
    "Co-localization with an ion": '''\
# Rank features by spatial similarity to a target m/z (here the active ion).
find_peaks(snr=5)
ranked = colocalize(target_mz=active_mz or 888.62, method="pearson")
table(ranked[:25], "Most co-localized")
''',
    "Compartments by construction (endo → peri → epi)": '''\
# Nerve compartments from the signal, not by eye: endoneurium = where the sulfatides
# are (60th percentile of signal pixels, holes filled); perineurium = a collar grown
# outside it; epineurium = every other acquired pixel. Then test across the three.
find_peaks(snr=5)
sulf = [885.5499, 888.6236]
endo = threshold_mask(composite(sulf), 60)
peri = ring(endo, width_px=6, mode="outer")      # 6 px = 30 µm on a 5 µm slide; or width_um=30
epi = invert(endo | peri)
add_region("endo", endo); add_region("peri", peri); add_region("epi", epi)
log(f"endo {int(endo.sum())} px · peri {int(peri.sum())} px · epi {int(epi.sum())} px")
table(multigroup(groups={"endo": endo, "peri": peri, "epi": epi}), "KW across compartments")
# Apply to app ▸ Add regions to the slide pushes the three in as named regions.
''',
}


# --------------------------------------------------------------------------- #
# the AI / capabilities guide  (the "instructions for an AI assistant")
# --------------------------------------------------------------------------- #
def _api_reference() -> str:
    """A bulleted reference of the bare-name functions, introspected from
    :class:`ScriptAPI` so the guide can never drift from the actual signatures."""
    import re
    lines = []
    for name in ScriptAPI.NAMESPACE_FUNCS:
        fn = getattr(ScriptAPI, name)
        try:
            sig = str(inspect.signature(fn))
            sig = sig.replace("(self, ", "(").replace("(self)", "()")
            sig = re.sub(r": '([^']+)'", r": \1", sig)      # unquote PEP-563 string annotations
        except (TypeError, ValueError):
            sig = "(...)"
        doc = (fn.__doc__ or "").strip()
        first = doc.splitlines()[0] if doc else ""
        lines.append(f"- `{name}{sig}` — {first}")
    return "\n".join(lines)


def _step_reference() -> str:
    """The full registry of analysis ids (for the generic `run(...)`), with what each needs
    and its tunable params — generated from :data:`smile_msi.registry.REGISTRY`."""
    lines = []
    for cat, defs in registry.registry_by_category().items():
        lines.append(f"\n**{cat}**")
        for sd in defs:
            needs = ", ".join(sorted(sd.needs)) or "—"
            params = ", ".join(p.name for p in sd.params) or "—"
            lines.append(f"- `{sd.id}` — {sd.help} _(needs: {needs}; params: {params})_")
    return "\n".join(lines)


def _context_section(api: ScriptAPI) -> str:
    ds = api.ds
    lo, hi = ds.mz_range
    parts = [
        "## This slide (runtime context)",
        f"- Dataset: `{getattr(ds, 'source', '?')}` — {ds.n_pixels:,} pixels, "
        f"{ds.width}×{ds.height} grid, m/z {lo:.1f}–{hi:.1f}.",
        f"- Defaults: ppm={api.ppm:g}, norm={api.norm!r}, reduce={api.reduce!r}.",
        f"- Working features: {len(api.features)} m/z" + (" (none yet — call find_peaks())" if not api.features else "."),
        f"- Regions available: {api.region_names() or '— none drawn'}.",
        f"- Groups tagged: {api.group_names() or '— none tagged'}.",
        f"- Active ion: {api.active_mz if api.active_mz else '— none selected'}.",
    ]
    return "\n".join(parts)


def capabilities_doc(api: ScriptAPI | None = None) -> str:
    """The self-describing guide handed to an LLM (or shown to a user) so it can write a
    correct workflow script. Static API + step references are generated from the code;
    when ``api`` is given, a runtime-context section describes the loaded slide."""
    out = [
        "# SMILE MSI — Analysis Workflow Scripting Guide",
        "",
        "You are writing a short **Python script** that analyses one loaded mass-spectrometry "
        "imaging (MSI) slide and surfaces results. It runs inside SMILE MSI's Script Console "
        "(Data ▸ Analysis script…) against the slide already open in the app.",
        "",
        "## Execution model",
        "- The script is ordinary Python 3. `numpy` is available as `np`. No imports are "
        "needed for the analysis functions — they are pre-bound bare names.",
        "- It runs **against the open slide**, exposed as `ds` (an `MSIDataset`). You do not "
        "load data; it is already there.",
        "- There is no value to `return`. You surface results with the output helpers below. "
        "`print()` output is captured separately from the run log, as the run's `stdout` (the "
        "Script Console shows it under *stdout*; the MCP `run_script` result returns it as "
        "`stdout`). Use `log()` for lines that belong in the run log.",
        "- Runs may take seconds (peak picking, segmentation, multivariate). That is normal.",
        "- The script is sandbox-free (full Python on the user's machine) but should avoid "
        "network access and writing files — use the console's *Export* for artifacts.",
        "",
        "## Output helpers (how results become visible)",
        "- `log(*args)` — add a line to the run log.",
        "- `table(df, title='')` — show a pandas DataFrame (or list-of-dicts) as a table.",
        "- `image(x, title='')` — show an ion image. `x` is an m/z (float), a per-pixel "
        "vector, or a 2-D array.",
        "- `record(name, value)` — stash a named value to inspect after the run.",
        "- `ds.ratio_image(num_mz, den_mz, tol_ppm=…, norm='none', eps=1.0)` — an H×W image of "
        "one ion over another (e.g. sulfatide/PC); show it with `image(...)`. It takes its own "
        "`tol_ppm` (pass `api.ppm` to match the session) and `eps` keeps a ~0 denominator finite.",
        "",
        "## Core concepts",
        "- **Features** = the working set of m/z the analyses operate on. Produce it with "
        "`find_peaks()` / `find_spatial_features()` (they set it as a side effect), or pass "
        "`features=[...]` explicitly. Most steps default to the working set.",
        "- **Regions** = named ROIs drawn in the app; resolve a mask with `region('name')`.",
        "- **Groups** = ROIs tagged Group A/B/… for comparisons; `group('A')` or pass names "
        "to `compare` / `discriminating` / `markers`. With no groups tagged, group-based "
        "steps fall back to the latest `segment()` clusters.",
        "- **ppm / norm / reduce** are the extraction defaults; set `api.ppm = 5`, "
        "`api.norm = 'rms'` etc. once at the top to change them for the whole script.",
        "- **Regions by construction** — build masks from the signal instead of drawing: "
        "`threshold_mask(composite([...m/z...]), 60)`, `ring(mask, width_px=6, mode='outer')` "
        "(a collar outside it), `invert(mask)` (everything else). `add_region('name', mask)` "
        "stages a region the console can push into the app (Apply to app ▸ Add regions to the "
        "slide) and makes it usable by name.",
        "- **`threshold_mask` fills holes by default** (`fill_holes=True`): it is built for "
        "solid compartments, so a **rim or ring** signal comes back as a solid disc covering "
        "the core it encloses — pass `fill_holes=False` for one (the run log warns when "
        "filling grew the mask a lot). The cut is a **percentile of every pixel with signal "
        "> 0**, and off-tissue pixels with any noise count, so `60` is not the 60th "
        "percentile of the tissue; for that, threshold `values * tissue_mask`, or pass "
        "`percentile=False` and an absolute intensity.",
        "- **Segmentation clusters every acquired pixel** unless you pass `mask=`: with "
        "off-tissue background on the slide, `segment(n_clusters=2)` usually splits tissue "
        "from background. Use `segment(mask=tissue)` to cluster the tissue alone.",
        "",
        "## Analysis functions (bare names)",
        _api_reference(),
        "",
        "## All analyses via `run(step_id, ...)`",
        "Every analysis is also reachable by id through `run()` (forward-compatible escape "
        "hatch). `run()` resolves `features` / `groups` / `mask` / `target_mz` for you:",
        _step_reference(),
        "",
        "## The session handle: `s.run_analysis(step_id, **params)`",
        "The loaded session is also bound as `s` (an alias of `api`). "
        "`s.run_analysis(step_id, **params)` is the documented public form of `run()`: it "
        "dispatches through the **same** registry with the registry defaults backfilled, and "
        "assembles the step's inputs (`features` / groups / region masks / `target_mz`) from "
        "the session's current state exactly as the named wrappers do. Override any of them "
        "with `features=`, `mask=`, `a=`/`b=` (the two comparison groups; default = the first "
        "two tagged groups), `region=`, `groups=` or `target_mz=`; every other keyword is a "
        "tunable engine param. This is the call a migrated Flow preset is written in, so a "
        "saved recipe reproduces a Flow losslessly:",
        "```python",
        's.run_analysis("find_spatial_features", snr=3.0, min_morans=0.05)',
        's.run_analysis("auto_segment", n_clusters=2)',
        's.run_analysis("roi_comparison", method="mwu")   # A/B = first two tagged groups',
        "```",
        "",
        "## Best practices",
        "1. Start by establishing features: `find_peaks(snr=5)` (or `find_spatial_features`).",
        "2. Prefer the named functions; reach for `run(id, ...)` only for steps without one.",
        "3. Surface *something* — at least one `table`, `image`, `log`, or `record` — so the "
        "run isn't silent.",
        "4. Guard group/region steps: check `group_names()` / `region_names()` first and "
        "`log()` a clear message if the slide isn't set up for them.",
        "5. Keep it deterministic and reasonably small; this is interactive, not a batch job.",
        "6. Errors are reported with your line numbers — read the traceback and fix in place.",
        "",
        "## Example",
        "```python",
        EXAMPLES["Compare two groups (A vs B)"].rstrip(),
        "```",
    ]
    if api is not None:
        out += ["", _context_section(api)]
    return "\n".join(out)
