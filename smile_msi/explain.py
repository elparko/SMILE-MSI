"""SHAP-based biomarker discovery over an :class:`~smile_msi.msi.MSIDataset`.

A supervised, model-explanation workflow for finding the molecular species that drive
a tissue-region (e.g. functional-tissue-unit, "FTU") classification, and *which
direction* each one pushes — the analysis behind the donor × molecule bubble plots of
Farrow et al. 2025 (*Sci. Adv.*, the human-kidney MSI atlas).

Pipeline (built from published primitives, lazy-imported so importing this module is
cheap and ``shap`` stays an optional dependency):

1. **Feature matrix** — per-pixel TIC-normalized intensities for a chosen feature set
   (:func:`spatial.feature_matrix`), with the unassigned (label ``< 0``) pixels dropped
   and an optional class-stratified pixel subsample for tractability.
2. **Classifier** — a :class:`~sklearn.ensemble.RandomForestClassifier` fit one-vs-rest
   over the region labels. A tree ensemble is what makes step 3 exact and fast.
3. **SHAP attribution** — :class:`shap.TreeExplainer` gives every pixel a per-feature
   Shapley value *per class*: the additive contribution of that ion's intensity to the
   model's score for that region (Lundberg & Lee 2017; TreeExplainer, Lundberg et al.
   2020).
4. **Per-molecule summaries** for each region:
   * **importance** — ``mean(|SHAP|)`` over *all* explained pixels (Lundberg's global
     importance — the standard summary, computed over every subsampled pixel, not only the
     region's own); the *magnitude* of an ion's influence → bubble **size**.
   * **direction** — Spearman rank correlation between the ion's (mean-centered)
     intensity and its own per-pixel SHAP value; ``+`` = high intensity drives *toward*
     the region, ``−`` = away → bubble **colour**.

The single-slide entry point is :func:`shap_importance`; :func:`shap_panel` runs it
across a cohort of donors (each with optional age/sex/BMI) and stacks the results into a
:class:`ShapPanel` the bubble-plot view consumes.

**Cross-donor normalization.** Each donor is explained by its *own* RandomForest, so its
SHAP values live on that model's scale: TreeSHAP for a probability classifier makes each
pixel's per-class values sum to ``proba_c(x) − E[proba_c]``, whose magnitude grows with how
confidently that donor's forest separates its regions (and shifts with the donor's class
count). Pooling raw ``mean(|SHAP|)`` across donors therefore makes a cleanly-separating
donor look bigger for *every* ion — a model artifact, not biology. So every cross-donor
aggregate (:func:`importance_bars`, :meth:`ShapPanel.bubble_frame`) first rescales each
donor's per-region importance vector onto a common scale via :func:`normalize_importance`;
the mode is customizable (see ``IMPORTANCE_NORMS``) and defaults to L1 (fraction of that
donor's total importance), the standard way to combine SHAP across independently-fit models.

Refs: Lundberg, S.M. & Lee, S.-I. (2017). A unified approach to interpreting model
predictions. *NeurIPS 30*. arXiv:1705.07874. — Lundberg, S.M. et al. (2020). From local
explanations to global understanding with explainable AI for trees. *Nature Machine
Intelligence* 2, 56–67. doi:10.1038/s42256-019-0138-9. — Breiman, L. (2001). Random
forests. *Machine Learning* 45, 5–32. doi:10.1023/A:1010933404324. — Spearman, C. (1904).
The proof and measurement of association between two things. *Am. J. Psychol.* 15, 72–101.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .constants import DEFAULT_TOL_PPM

from . import spatial

# Row-chunk parallelism for the SHAP step: shap's sklearn TreeSHAP path runs single-
# threaded, so on a multi-core machine the explainer — not the (already n_jobs=-1) forest
# fit — is the bottleneck. Below this many pixels, or with a single worker, the serial call
# wins because process startup + model pickling would dominate; each worker gets at least
# ``_MIN_ROWS_PER_CHUNK`` rows. There is no GPU path here: GPU TreeSHAP is CUDA/NVIDIA-only
# (no Apple-Silicon/Metal backend), so multicore CPU is the portable acceleration.
_PARALLEL_MIN_ROWS = 4000
_MIN_ROWS_PER_CHUNK = 1000

# Cross-donor / cross-region importance normalization modes (see normalize_importance).
# "l1"  — each donor's ion importances become a fraction of that donor's total (sum to 1);
#         the standard, most interpretable way to pool SHAP across independently-fit models.
# "max" — rescale so the donor's strongest ion is 1 (relative-to-top).
# "none"— raw magnitudes (not comparable across separately-fit models; pre-fix behaviour).
IMPORTANCE_NORMS = ("l1", "max", "none")
DEFAULT_IMPORTANCE_NORM = "l1"


@dataclass
class ShapResult:
    """SHAP biomarker summary for **one** dataset (one slide / one donor).

    ``importance`` and ``direction`` are ``(n_classes, n_peaks)`` matrices aligned to
    :attr:`classes` (rows) and :attr:`peaks` (columns)."""
    classes: list                    # region / FTU labels, in row order
    peaks: np.ndarray                # (n_peaks,) m/z per column
    importance: np.ndarray           # (n_classes, n_peaks) mean(|SHAP|) — bubble size
    direction: np.ndarray            # (n_classes, n_peaks) Spearman(intensity, SHAP) ∈ [-1, 1]
    accuracy: float = float("nan")   # train-set classification accuracy (sanity, not CV)
    n_pixels: int = 0                # pixels actually used (after drop / subsample)
    donor: str = ""                  # donor / sample label (cohort mode); "" for a lone slide
    meta: dict = field(default_factory=dict)   # optional {age, sex, bmi, group, ...}

    def top(self, cls, n: int = 10):
        """The ``n`` strongest biomarkers for region ``cls`` as
        ``[(mz, importance, direction), ...]`` ordered by importance."""
        i = self.classes.index(cls)
        order = np.argsort(-self.importance[i])[:n]
        return [(float(self.peaks[j]), float(self.importance[i, j]),
                 float(self.direction[i, j])) for j in order]

    def to_long_df(self):
        """Tidy one-row-per-(region, ion) table — the report/CSV form."""
        import pandas as pd
        rows = []
        for i, c in enumerate(self.classes):
            for j, mz in enumerate(self.peaks):
                rows.append({"donor": self.donor, "ftu": c, "mz": float(mz),
                             "importance": float(self.importance[i, j]),
                             "direction": float(self.direction[i, j])})
        df = pd.DataFrame(rows)
        df.attrs["unit"] = "ion×region"
        return df


@dataclass
class ShapPanel:
    """A cohort's worth of :class:`ShapResult` (one per donor) on a shared feature axis —
    the data behind a donor × molecule bubble plot.

    ``donor_norm`` (see :func:`normalize_importance`) is the cross-donor importance
    normalization the aggregating views (:meth:`bubble_frame`, :meth:`histogram_bars`,
    :meth:`histogram_frame`) apply by default so donors explained by separately-fit models
    are comparable; it defaults to L1 and can be overridden per call."""
    results: list                    # list[ShapResult]
    peaks: np.ndarray                # shared (n_peaks,) m/z axis
    classes: list                    # union of region labels seen across donors
    donor_norm: str = DEFAULT_IMPORTANCE_NORM   # cross-donor importance scaling for aggregates

    def donors(self):
        return [r.donor or f"sample {i+1}" for i, r in enumerate(self.results)]

    def bubble_frame(self, cls, *, norm: str | None = None):
        """Long-form ``donor × ion`` table for region ``cls`` — size=importance,
        colour=direction — i.e. one panel of the Farrow-style bubble plot. ``importance`` is
        cross-donor normalized (``norm``, default :attr:`donor_norm`) so it matches the
        rendered bubbles; direction is left as the raw Spearman correlation."""
        import pandas as pd
        norm = self.donor_norm if norm is None else norm
        rows = []
        for r in self.results:
            if cls not in r.classes:
                continue
            i = r.classes.index(cls)
            imp = normalize_importance(r.importance[i], norm)
            for j, mz in enumerate(r.peaks):
                rows.append({"donor": r.donor, "mz": float(mz),
                             "importance": float(imp[j]),
                             "direction": float(r.direction[i, j]),
                             **{k: r.meta.get(k) for k in ("age", "sex", "bmi")}})
        return pd.DataFrame(rows)

    def to_long_df(self):
        """Tidy one-row-per-(donor, region, ion) table stacking every donor — the
        report/CSV form for a whole cohort panel."""
        import pandas as pd
        frames = [r.to_long_df() for r in self.results]
        if not frames:
            return pd.DataFrame(columns=["donor", "ftu", "mz", "importance", "direction"])
        df = pd.concat(frames, ignore_index=True)
        df.attrs["unit"] = "donor×region×ion"
        return df

    def histogram_bars(self, cls, n=None, *, norm: str | None = None):
        """Cohort-aggregated biomarker bars for region ``cls`` — see
        :func:`importance_bars`. Uses :attr:`donor_norm` unless ``norm`` overrides it."""
        return importance_bars(self.results, cls, self.peaks, n,
                               norm=self.donor_norm if norm is None else norm)

    def histogram_frame(self, cls, n=None, *, norm: str | None = None):
        """The :meth:`histogram_bars` aggregation as a tidy table (the data behind the
        per-category importance histogram), strongest species first."""
        import pandas as pd
        cols = ["mz", "importance_mean", "importance_std", "direction_mean"]
        bars = importance_bars(self.results, cls, self.peaks, n,
                               norm=self.donor_norm if norm is None else norm)
        if bars is None:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame({"mz": bars["mz"], "importance_mean": bars["mean"],
                           "importance_std": bars["std"],
                           "direction_mean": bars["direction"]})
        df.attrs["unit"] = f"ion (region={cls}, n_donors={bars['n_donors']})"
        return df


def normalize_importance(imp, mode: str = DEFAULT_IMPORTANCE_NORM):
    """Rescale SHAP importance onto a common, cross-model-comparable scale, normalizing each
    donor/region independently along the ion (last) axis.

    SHAP importances from *separately-fit* models (one RandomForest per donor) are not
    directly comparable: a donor whose regions separate cleanly yields larger ``|SHAP|``
    across *every* ion than a fuzzier donor, purely as a model-confidence artifact, and the
    per-donor scale also shifts with the donor's class count. Normalizing each donor's
    per-region importance vector before pooling/plotting removes that confound so the
    bubbles/bars reflect biology rather than model scale.

    ``mode`` is one of :data:`IMPORTANCE_NORMS`: ``"l1"`` (default) → fraction of that
    donor's total importance (the vector sums to 1); ``"max"`` → relative to the donor's
    strongest ion (top ion = 1); ``"none"`` → raw magnitudes, unchanged. Works on a 1-D
    ``(n_ions,)`` vector or a 2-D ``(n_donors, n_ions)`` matrix (each row normalized on its
    own). Empty and all-zero vectors pass through untouched (no divide-by-zero)."""
    imp = np.asarray(imp, float)
    if mode in (None, "none", "") or imp.size == 0:
        return imp
    ax = imp.ndim - 1
    if mode == "l1":
        denom = np.abs(imp).sum(axis=ax, keepdims=True)
    elif mode == "max":
        denom = np.abs(imp).max(axis=ax, keepdims=True)
    else:
        raise ValueError(f"unknown importance norm {mode!r}; use one of {IMPORTANCE_NORMS}")
    return np.divide(imp, denom, out=np.zeros_like(imp), where=denom > 0)


def importance_bars(results, cls, peaks, n=None, *, norm: str = DEFAULT_IMPORTANCE_NORM):
    """Cohort-aggregated top-``n`` biomarker bars for region/class ``cls`` — the data
    behind a Farrow-style per-category importance histogram (their Fig. S151).

    For every ion, aggregates across the donor samples that contain ``cls``:

    * **mean** — mean across donors of the per-donor SHAP importance (``mean(|SHAP|)``);
      the bar length.
    * **std** — standard deviation across donors of that importance; the error bar (the
      "SD across all donor samples" the paper's caption reports).
    * **direction** — mean across donors of the Spearman direction (intensity vs SHAP);
      the bar colour (``+`` = high intensity marks the region).

    ``norm`` (see :func:`normalize_importance`, default L1) rescales each donor's per-region
    importance vector onto a common scale *before* the cross-donor mean/std, so a
    more-separable donor's larger raw ``|SHAP|`` can't dominate the bar or inflate the SD
    with model-scale variance instead of biological variability. Direction is a correlation
    (already comparable) and is left unscaled.

    ``results`` is a list of :class:`ShapResult` (a lone slide passes ``[result]`` →
    zero-width error bars). Returns a dict of arrays ordered by ``mean`` descending and
    truncated to ``n`` (``None`` keeps every ion), or ``None`` if no donor has ``cls``."""
    donors = [r for r in results if cls in r.classes]
    if not donors:
        return None
    imp = np.vstack([r.importance[r.classes.index(cls)] for r in donors])
    drc = np.vstack([r.direction[r.classes.index(cls)] for r in donors])
    imp = normalize_importance(imp, norm)           # per-donor rescale before pooling
    mean = imp.mean(axis=0)
    std = imp.std(axis=0)
    direction = drc.mean(axis=0)
    order = np.argsort(-mean)
    if n is not None:
        order = order[:int(n)]
    peaks = np.asarray(peaks, float)
    return {"mz": peaks[order], "mean": mean[order], "std": std[order],
            "direction": direction[order], "n_donors": len(donors)}


def _ensure_shap(progress=None):
    """Import ``shap``, auto-installing it into the running interpreter's environment on
    first use if it is missing.

    SHAP is a heavy optional extra (pulls in numba/llvmlite). Rather than dead-ending the
    analysis with an install hint, we ``pip install shap`` on demand — the same pattern the
    in-app updater uses for dependency bumps. Raises ``ImportError`` only if the install or
    the post-install import genuinely fails.

    In a frozen (PyInstaller) build we must NOT try to self-install: ``sys.executable`` is
    the app itself (e.g. ``SMILE MSI.exe``), so ``sys.executable -m pip install`` would just
    launch a second copy of the GUI and block the worker on that phantom window. shap is
    bundled into the packaged build instead (see ``smile_msi.spec``); if it is somehow
    missing there we fail with an honest message rather than spawning a process."""
    import sys
    try:
        import shap
        return shap
    except ImportError:
        pass
    if getattr(sys, "frozen", False):
        raise ImportError(
            "SHAP biomarker analysis isn't available in this packaged build. "
            "Run SMILE MSI from source (`pip install \"smile_msi[gui,shap]\"`) to use it."
        )
    import subprocess
    if progress:
        progress("installing the 'shap' package (one-time, this can take a minute)…")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "shap"],
                       check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, OSError) as e:
        log = (getattr(e, "stderr", "") or str(e))[-800:]
        raise ImportError(
            "SHAP biomarker analysis needs the 'shap' package and the automatic install "
            f"failed. Install it manually with `pip install shap`.\n\n{log}"
        ) from e
    import importlib
    try:
        return importlib.import_module("shap")
    except ImportError as e:                        # pragma: no cover - installed but unimportable
        raise ImportError(
            "Installed 'shap' but it could not be imported — restart the app and try again."
        ) from e


def _stratified_subsample(y: np.ndarray, max_pixels: int, rng) -> np.ndarray:
    """Indices keeping at most ``max_pixels`` pixels, balanced across classes so a small
    region isn't swamped. Returns all indices if already under the cap."""
    n = len(y)
    if max_pixels <= 0 or n <= max_pixels:
        return np.arange(n)
    classes = np.unique(y)
    per = max(1, max_pixels // len(classes))
    keep = []
    for c in classes:
        idx = np.flatnonzero(y == c)
        if len(idx) > per:
            idx = rng.choice(idx, per, replace=False)
        keep.append(idx)
    return np.sort(np.concatenate(keep))


def _normalize_shap(values, n_classes: int) -> list:
    """Coerce the several shap return shapes into a list ``sv[c] -> (n_samples,
    n_features)`` for ``c`` in class order.

    shap's TreeExplainer has returned, across versions: a list of ``n_classes`` arrays;
    a ``(n_samples, n_features, n_classes)`` array; or a bare ``(n_samples, n_features)``
    for a single/binary output."""
    if isinstance(values, list):
        sv = [np.asarray(v) for v in values]
    else:
        arr = np.asarray(values)
        if arr.ndim == 3:                       # (samples, features, classes)
            sv = [arr[:, :, c] for c in range(arr.shape[2])]
        else:                                   # (samples, features)
            sv = [arr]
    if len(sv) == 1 and n_classes == 2:         # binary: +class explains both directions
        sv = [-sv[0], sv[0]]
    return sv


def _shap_chunk_worker(model, X_chunk):
    """Explain one row-chunk in its own process. Module-level so joblib's process backend
    can pickle it; rebuilds the (cheap) TreeExplainer per worker from the picklable model."""
    import shap
    return shap.TreeExplainer(model).shap_values(X_chunk)


def _concat_shap(parts: list):
    """Recombine per-chunk ``shap_values`` outputs along the sample (row) axis, preserving
    whichever shape shap returned — a list of per-class ``(n, f)`` arrays, a 3-D
    ``(n, f, classes)`` array, or a bare 2-D ``(n, f)`` array — so :func:`_normalize_shap`
    sees exactly what a single ``shap_values(X)`` call would have produced."""
    first = parts[0]
    if isinstance(first, list):                 # list[(n_chunk, n_features)] per class
        return [np.concatenate([np.asarray(p[c]) for p in parts], axis=0)
                for c in range(len(first))]
    return np.concatenate([np.asarray(p) for p in parts], axis=0)


def _parallel_shap_values(rf, X, *, n_jobs: int = -1, progress=None):
    """TreeSHAP values for ``rf`` over ``X``, computed across CPU cores in row-chunks.

    Path-dependent TreeSHAP is independent per pixel, so splitting the rows across worker
    processes and concatenating is numerically identical to a single
    ``shap.TreeExplainer(rf).shap_values(X)`` call — but uses every core instead of the one
    thread shap's sklearn path runs on. ``n_jobs=-1`` uses all cores. Falls back to a serial
    call for small inputs, a single worker, or if the process pool cannot start (e.g. a
    PyInstaller-frozen build)."""
    import shap
    n = X.shape[0]
    jobs = (os.cpu_count() or 1) if n_jobs in (-1, None) else max(1, int(n_jobs))
    jobs = min(jobs, max(1, n // _MIN_ROWS_PER_CHUNK))
    if jobs <= 1 or n < _PARALLEL_MIN_ROWS:
        if progress:
            progress("computing SHAP values…")
        return shap.TreeExplainer(rf).shap_values(X)
    from joblib import Parallel, delayed
    chunks = np.array_split(np.arange(n), jobs)
    if progress:
        progress(f"computing SHAP values across {jobs} cores…")
    try:
        parts = Parallel(n_jobs=jobs)(delayed(_shap_chunk_worker)(rf, X[idx]) for idx in chunks)
        return _concat_shap(parts)
    except Exception:                           # pool unavailable (frozen build) → serial
        if progress:
            progress("parallel SHAP unavailable — falling back to a single core…")
        return shap.TreeExplainer(rf).shap_values(X)


def _spearman_direction(X: np.ndarray, shap_c: np.ndarray) -> np.ndarray:
    """Per-feature Spearman corr between intensity column and its SHAP column.
    Constant columns (no rank variation) yield 0 rather than NaN."""
    from scipy.stats import spearmanr
    n_features = X.shape[1]
    out = np.zeros(n_features)
    for j in range(n_features):
        xj, sj = X[:, j], shap_c[:, j]
        if np.ptp(xj) == 0 or np.ptp(sj) == 0:
            continue
        rho = spearmanr(xj, sj)[0]              # [0]=correlation across scipy versions
        out[j] = 0.0 if np.isnan(rho) else float(rho)
    return out


def shap_importance(ds, peaks, labels, names=None, *, n_estimators: int = 300,
                    max_depth=8, max_pixels: int = 20000, tol_ppm: float = DEFAULT_TOL_PPM,
                    norm: str = "tic", random_state: int = 0, n_jobs: int = -1,
                    donor: str = "", meta: dict | None = None, progress=None) -> ShapResult:
    """SHAP biomarker importance + direction for one dataset's region ``labels``.

    ``labels`` is a per-pixel integer/text array; pixels labelled ``< 0`` (numeric
    "unassigned") are dropped. ``names`` optionally maps the label values to display
    names (region/FTU names). A RandomForest is fit one-vs-rest over the classes and
    explained with :class:`shap.TreeExplainer`; see the module docstring for the maths.

    Raises ``ImportError`` with an install hint if ``shap`` is not available, and
    ``ValueError`` if fewer than two classes survive the drop.

    ``max_depth`` is capped (8) by default: exact TreeSHAP costs ``O(trees · leaves ·
    depth²)`` per pixel, and fully-grown trees (``max_depth=None``) dominate the runtime —
    a shallow forest is both far faster and a milder, less overfit explainer. Pass
    ``max_depth=None`` to restore unbounded trees.

    ``n_jobs`` controls CPU parallelism of the SHAP step (default ``-1`` = all cores): for
    large pixel sets the explanation is computed in independent row-chunks across worker
    processes, which is numerically identical to a single-thread run but far faster. Pass
    ``n_jobs=1`` to force the serial path."""
    from sklearn.ensemble import RandomForestClassifier
    _ensure_shap(progress)

    labels = np.asarray(labels)
    peaks = np.asarray(peaks, float)
    X = spatial.feature_matrix(ds, peaks, tol_ppm=tol_ppm, norm=norm)

    if np.issubdtype(labels.dtype, np.number):
        keep = labels >= 0
        X, labels = X[keep], labels[keep]
    if X.shape[0] == 0:
        raise ValueError("no labelled pixels to explain")

    rng = np.random.default_rng(random_state)
    sub = _stratified_subsample(labels, max_pixels, rng)
    X, labels = X[sub], labels[sub]

    classes_raw = list(np.unique(labels))
    if len(classes_raw) < 2:
        raise ValueError("SHAP biomarkers need at least two regions/classes to compare")
    if progress:
        progress(f"fitting RandomForest on {X.shape[0]} px × {X.shape[1]} ions…")

    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                random_state=random_state, n_jobs=-1)
    rf.fit(X, labels)
    accuracy = float(rf.score(X, labels))
    # column order RF used for its outputs / shap classes
    model_classes = list(rf.classes_)

    sv = _normalize_shap(_parallel_shap_values(rf, X, n_jobs=n_jobs, progress=progress),
                         len(model_classes))

    n_peaks = X.shape[1]
    importance = np.zeros((len(model_classes), n_peaks))
    direction = np.zeros((len(model_classes), n_peaks))
    for c in range(len(model_classes)):
        shap_c = sv[c]
        importance[c] = np.mean(np.abs(shap_c), axis=0)
        direction[c] = _spearman_direction(X, shap_c)

    def _name(v):
        if names is not None:
            try:
                return names[int(v)]
            except (KeyError, IndexError, TypeError, ValueError):
                return str(v)
        return str(v)

    return ShapResult(classes=[_name(c) for c in model_classes], peaks=peaks,
                      importance=importance, direction=direction, accuracy=accuracy,
                      n_pixels=int(X.shape[0]), donor=donor, meta=dict(meta or {}))


def shap_panel(loader, peaks, *, n_estimators: int = 300, max_pixels: int = 20000,
               tol_ppm: float = DEFAULT_TOL_PPM, norm: str = "tic", random_state: int = 0,
               n_jobs: int = -1, donor_norm: str = DEFAULT_IMPORTANCE_NORM,
               progress=None) -> ShapPanel:
    """Run :func:`shap_importance` across a cohort and stack into a :class:`ShapPanel`.

    ``loader`` is an iterable of ``(donor, ds, labels, names, meta)`` tuples — flexible
    by design so callers can feed slides from a cohort roster, regions, segmentation
    labels, or any per-pixel grouping, each with optional ``meta`` demographics. Donors
    that error (too few classes, load failure) are skipped; if none survive, raises
    ``ValueError``.

    ``norm`` is the per-pixel intensity normalization fed to each donor's model (TIC by
    default); ``donor_norm`` is the *cross-donor* importance rescaling the resulting panel's
    aggregates use so separately-fit donors are comparable (see :func:`normalize_importance`,
    default L1)."""
    results = []
    all_peaks = np.asarray(peaks, float)
    classes: list = []
    for item in loader:
        donor, ds, labels, names, meta = item
        if progress:
            progress(f"donor {donor}…")
        try:
            r = shap_importance(ds, all_peaks, labels, names=names,
                                n_estimators=n_estimators, max_pixels=max_pixels,
                                tol_ppm=tol_ppm, norm=norm, random_state=random_state,
                                n_jobs=n_jobs, donor=str(donor), meta=meta, progress=progress)
        except (ValueError, ImportError):
            if progress:
                progress(f"  skipped {donor}")
            continue
        results.append(r)
        for c in r.classes:
            if c not in classes:
                classes.append(c)
    if not results:
        raise ValueError("no donors yielded a SHAP result (need ≥2 regions each)")
    return ShapPanel(results=results, peaks=all_peaks, classes=classes, donor_norm=donor_norm)
