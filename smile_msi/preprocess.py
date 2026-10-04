"""Per-spectrum preprocessing — the steps every MSI pipeline applies before
analysis (baseline correction, smoothing, m/z recalibration, normalization).

Each factory returns a transform ``f(mz, intensity) -> (mz, intensity)`` that
**preserves the m/z axis** (so a continuous dataset stays continuous). Compose a
pipeline and attach it with :meth:`MSIDataset.set_preprocessing`::

    from smile_msi import preprocess as pp
    ds.set_preprocessing([pp.baseline_snip(), pp.smooth_savgol(window=9),
                          pp.recalibrate([255.2330, 885.5499], tol_ppm=200)])

``build_pipeline`` maps a config dict (from the CLI/GUI) to such a list.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Baseline correction
# --------------------------------------------------------------------------- #
def _snip(y: np.ndarray, iterations: int) -> np.ndarray:
    """SNIP baseline estimate (statistics-sensitive non-linear iterative peak
    clipping) on the LLS-transformed signal — the standard MS baseline method.

    Refs: Ryan et al. (1988), doi:10.1016/0168-583X(88)90063-8; LLS operator —
    Morháč & Matoušek (2008), doi:10.1366/000370208783412762.
    """
    v = np.log(np.log(np.sqrt(np.clip(y, 0, None) + 1) + 1) + 1)
    n = len(v)
    for p in range(1, iterations + 1):
        if n - 2 * p <= 0:
            break
        a = v[p:n - p]
        b = (v[:n - 2 * p] + v[2 * p:]) / 2.0
        v[p:n - p] = np.minimum(a, b)
    base = (np.exp(np.exp(v) - 1) - 1) ** 2 - 1
    return np.clip(base, 0, None)


def baseline_snip(iterations: int = 40):
    """Subtract a SNIP-estimated baseline (``iterations`` ~ half the widest peak)."""
    def f(mz, inten):
        return mz, np.clip(inten - _snip(np.asarray(inten, float), iterations), 0, None)
    return f


def baseline_locmin(window: int = 50):
    """Local-minimum baseline — Cardinal's default ``reduceBaseline`` method: a running
    minimum over ``window`` points, smoothed, then subtracted. Cheap and robust for the
    slowly-varying chemical/matrix background of MALDI spectra."""
    from scipy.ndimage import minimum_filter1d, gaussian_filter1d

    def f(mz, inten):
        y = np.asarray(inten, float)
        w = max(3, int(window))
        base = gaussian_filter1d(minimum_filter1d(y, size=w, mode="nearest"), sigma=w / 4.0)
        return mz, np.clip(y - base, 0, None)
    return f


def baseline_median(window: int = 50):
    """Rolling-median baseline (Cardinal ``median``) — subtract a windowed median."""
    from scipy.ndimage import median_filter

    def f(mz, inten):
        y = np.asarray(inten, float)
        base = median_filter(y, size=max(3, int(window)), mode="nearest")
        return mz, np.clip(y - base, 0, None)
    return f


def _cross(x, y, o, a, b) -> float:
    return (x[a] - x[o]) * (y[b] - y[o]) - (y[a] - y[o]) * (x[b] - x[o])


def baseline_hull():
    """Convex-hull baseline (Cardinal ``hull``): the lower convex hull of the spectrum,
    interpolated across the m/z axis and subtracted — a parameter-free estimate of a
    broad, monotone background."""
    def f(mz, inten):
        x = np.asarray(mz, float)
        y = np.asarray(inten, float)
        n = len(y)
        if n < 3:
            return mz, np.clip(y, 0, None)
        hull = []
        for i in range(n):
            while len(hull) >= 2 and _cross(x, y, hull[-2], hull[-1], i) <= 0:
                hull.pop()
            hull.append(i)
        base = np.interp(x, x[hull], y[hull])
        return mz, np.clip(y - base, 0, None)
    return f


def reduce_baseline(method: str = "locmin", **kw):
    """Baseline-reduction dispatcher mirroring Cardinal's ``reduceBaseline(method=)``:
    ``locmin`` (default), ``hull``, ``snip``, ``median``."""
    method = (method or "locmin").lower()
    if method == "snip":
        return baseline_snip(int(kw.get("iterations", 40)))
    if method == "hull":
        return baseline_hull()
    if method == "median":
        return baseline_median(int(kw.get("window", 50)))
    return baseline_locmin(int(kw.get("window", 50)))


# --------------------------------------------------------------------------- #
# Smoothing
# --------------------------------------------------------------------------- #
def smooth_savgol(window: int = 9, poly: int = 3):
    """Savitzky-Golay smoothing (window forced odd and >= poly+2).

    Reference: Savitzky & Golay (1964), Anal. Chem. 36(8):1627–1639, doi:10.1021/ac60214a047.
    """
    from scipy.signal import savgol_filter

    def f(mz, inten):
        n = len(inten)
        w = min(window if window % 2 == 1 else window + 1, n if n % 2 == 1 else n - 1)
        if w <= poly + 1 or w < 3:
            return mz, inten
        return mz, np.clip(savgol_filter(inten, w, poly), 0, None)
    return f


def smooth_gaussian(sigma: float = 1.0):
    from scipy.ndimage import gaussian_filter1d

    def f(mz, inten):
        return mz, np.clip(gaussian_filter1d(np.asarray(inten, float), sigma), 0, None)
    return f


def smooth(method: str = "gaussian", **kw):
    """Smoothing dispatcher mirroring Cardinal's ``smooth(method=)``: ``gaussian``
    (default) or ``sgolay`` (Savitzky-Golay)."""
    method = (method or "gaussian").lower()
    if method in ("sgolay", "savgol"):
        return smooth_savgol(int(kw.get("window", 9)), int(kw.get("poly", 3)))
    return smooth_gaussian(float(kw.get("sigma", 1.0)))


# --------------------------------------------------------------------------- #
# m/z recalibration (lock-mass alignment)
# --------------------------------------------------------------------------- #
def recalibrate(reference_mzs, tol_ppm: float = 200.0):
    """Align each spectrum to known reference m/z (lock masses).

    For each reference, the nearest local-max within ``tol_ppm`` gives an observed
    m/z; the median ppm offset across references is removed and the intensities are
    resampled back onto the original axis (so the shared axis is preserved). Falls
    back to a no-op for spectra where no references are found.
    """
    refs = np.asarray(reference_mzs, dtype=float)

    def f(mz, inten):
        mz = np.asarray(mz, float)
        inten = np.asarray(inten, float)
        shifts = []
        for r in refs:
            win = r * tol_ppm / 1e6
            lo = np.searchsorted(mz, r - win, "left")
            hi = np.searchsorted(mz, r + win, "right")
            if hi > lo:
                k = lo + int(np.argmax(inten[lo:hi]))
                shifts.append((mz[k] - r) / r)        # fractional offset
        if not shifts:
            return mz, inten
        shift = float(np.median(shifts))              # observed = true*(1+shift)
        mz_cal = mz / (1.0 + shift)                   # map observed back to true
        return mz, np.interp(mz, mz_cal, inten, left=0.0, right=0.0)
    return f


def recalibrate_regions(specs, default_refs=None, default_tol_ppm: float = 200.0,
                        n_pixels: int | None = None):
    """Lock-mass recalibration with a **different reference set per region** — for a slide
    carrying several samples in different embedding media, where each medium drifts by its
    own constant ppm offset and one slide-wide lock mass splits the difference.

    ``specs`` is ``[(mask, reference_mzs, tol_ppm), ...]`` with ``mask`` a boolean per-pixel
    array. Pixels outside every mask fall back to ``default_refs`` (pass through unchanged
    when that is empty). Earlier specs win where masks overlap.

    The returned transform takes the pixel index as a third argument and carries
    ``needs_index = True``; :meth:`MSIDataset._read` passes the index only to transforms that
    ask for it, so every other transform keeps the two-argument form. Each region still gets
    the axis-preserving correction of :func:`recalibrate`, so all pixels stay on one shared
    m/z axis and remain directly comparable bin for bin.
    """
    fns, masks = [], []
    for mask, refs, tol in specs:
        m = np.asarray(mask, dtype=bool)
        if refs is None or len(refs) == 0 or not m.any():
            continue
        fns.append(recalibrate(refs, float(tol)))
        masks.append(m)
    fallback = recalibrate(default_refs, float(default_tol_ppm)) if default_refs else None
    if not fns:
        return fallback if fallback is not None else (lambda mz, inten: (mz, inten))

    n = int(n_pixels or max(m.size for m in masks))
    assign = np.full(n, -1, dtype=np.int32)
    for k in range(len(masks) - 1, -1, -1):       # reverse fill → the earlier spec wins
        m = masks[k][:n]
        assign[:m.size][m] = k

    def f(mz, inten, i):
        k = int(assign[i]) if 0 <= i < n else -1
        if k < 0:
            return fallback(mz, inten) if fallback is not None else (mz, inten)
        return fns[k](mz, inten)
    f.needs_index = True
    return f


def auto_recalibrate(ds, n_refs: int = 3, min_rel_intensity: float = 0.05,
                     tol_ppm: float = 100.0):
    """Lock-mass recalibration with **auto-selected internal references** — the
    dataset's own strongest, most prevalent mean-spectrum peaks become the lock masses,
    so each spectrum is aligned to their consensus position without the user typing in
    reference m/z. This is spectrum-to-spectrum alignment (it reduces pixel-to-pixel
    spread; it does not fix absolute mass error — for that, supply true lock masses to
    :func:`recalibrate`). Returns a ``(mz, inten) -> (mz, inten)`` transform; apply via
    ``MSIDataset.set_preprocessing([...])``. Pair with :func:`intake.mass_drift` to decide
    whether it's needed at all."""
    ds.prime()
    refs = [float(p["mz"]) for p in
            ds.pick_peaks(min_rel_intensity=min_rel_intensity, max_peaks=n_refs)]
    if not refs:
        return lambda mz, inten: (mz, inten)
    return recalibrate(refs, tol_ppm=tol_ppm)


# --------------------------------------------------------------------------- #
# Per-spectrum normalization (axis-preserving)
# --------------------------------------------------------------------------- #
def normalize_vector():
    """L2 (unit-vector) normalization of each spectrum."""
    def f(mz, inten):
        n = np.linalg.norm(inten)
        return mz, (inten / n if n > 0 else inten)
    return f


def normalize_reference(ref_mz: float, tol_ppm: float = 100.0):
    """Divide each spectrum by the intensity at a reference m/z (e.g. an internal
    standard) — quantitative normalization to a spiked lock mass."""
    def f(mz, inten):
        mz = np.asarray(mz, float)
        win = ref_mz * tol_ppm / 1e6
        lo = np.searchsorted(mz, ref_mz - win, "left")
        hi = np.searchsorted(mz, ref_mz + win, "right")
        ref = inten[lo:hi].max() if hi > lo else 0.0
        return mz, (inten / ref if ref > 0 else inten)
    return f


# --------------------------------------------------------------------------- #
# Config -> pipeline (for CLI / GUI)
# --------------------------------------------------------------------------- #
def build_pipeline(config: dict, masks: dict | None = None):
    """Map a config dict to a transform list. Recognized keys:
    ``baseline`` (int iters), ``smooth`` ({'savgol': window} or {'gaussian': sigma}),
    ``recalibrate`` ({'refs': [...], 'tol_ppm': ...}), ``normalize`` ('vector' or
    {'reference': mz}).

    ``recalibrate`` may also carry ``regions``: ``[{'name', 'refs', 'tol_ppm'}, ...]`` for a
    per-region lock mass (:func:`recalibrate_regions`). Region names are resolved against
    ``masks`` (``{name: bool[n_pixels]}``) — a caller reopening a session should pass the
    session's region masks (:func:`headless.build_preprocessing` does this), since a region
    that does not resolve falls back to the slide-wide ``refs``."""
    steps = []
    bl = config.get("baseline")
    if isinstance(bl, dict):                  # Cardinal-style {'method': 'locmin'|'snip'|...}
        steps.append(reduce_baseline(bl.get("method", "locmin"), **{k: v for k, v
                                     in bl.items() if k != "method"}))
    elif bl:                                   # back-compat: an int is SNIP iterations
        steps.append(baseline_snip(int(bl)))
    sm = config.get("smooth")
    if sm:
        if isinstance(sm, dict) and "gaussian" in sm:
            steps.append(smooth_gaussian(float(sm["gaussian"])))
        else:
            w = sm.get("savgol", 9) if isinstance(sm, dict) else int(sm)
            steps.append(smooth_savgol(int(w)))
    rc = config.get("recalibrate")
    if rc:                                    # need reference masses; skip (don't crash) without them
        tol = rc.get("tol_ppm", 200.0)
        specs = [((masks or {}).get(r.get("name")), r.get("refs"), r.get("tol_ppm", tol))
                 for r in (rc.get("regions") or [])]
        specs = [s for s in specs if s[0] is not None and s[1]]
        if specs:
            steps.append(recalibrate_regions(specs, rc.get("refs"), tol,
                                             n_pixels=rc.get("n_pixels")))
        elif rc.get("refs"):
            steps.append(recalibrate(rc["refs"], tol))
    nm = config.get("normalize")
    if nm == "vector":
        steps.append(normalize_vector())
    elif isinstance(nm, dict) and "reference" in nm:
        steps.append(normalize_reference(float(nm["reference"]), nm.get("tol_ppm", 100.0)))
    return steps
