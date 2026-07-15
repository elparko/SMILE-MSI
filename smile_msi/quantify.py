"""Absolute quantification via on-tissue calibration.

Pure engine (numpy/scipy only). Turns relative MSI ion intensities into absolute
concentration estimates by fitting a per-analyte response curve from user-defined
standard-spot regions (each a Region tagged with a known concentration level), then
back-calculating per-pixel concentration by inverse prediction. Supports ordinary and
weighted (1/x, 1/x^2) least squares, an optional through-origin (zero-intercept) model,
and an optional internal-standard (IS) ratio response.

Algorithms reimplemented clean-room from the literature (no third-party calibration
package is copied; all formulas are textbook closed form computed with numpy):

* On-tissue calibration curve + concentration back-calculation (mimetic tissue model):
  Groseclose, M.R. & Castellino, S. A mimetic tissue model for the quantification of drug
  distributions by MALDI imaging mass spectrometry. Anal. Chem. (2016).
  doi:10.1021/acs.analchem.5b04409
* Quantitative MSI on-tissue calibration-geometry protocols and matrix-effect correction
  in heterogeneous tissue. PMC (2025). PMC12138873.
* Weighted (1/x, 1/x^2) least squares for heteroscedastic calibration:
  Almeida, A.M., Castel-Branco, M.M. & Falcao, A.C. (2002). Linear regression for
  calibration lines revisited: weighting schemes for bioanalytical methods.
  J. Chromatogr. B, 774(1), 215-222. doi:10.1016/S0378-4347(01)00549-9
* LOD / LOQ definitions: ICH Harmonised Tripartite Guideline Q2(R1), Validation of
  Analytical Procedures. LOD = 3.3 * sigma / S, LOQ = 10 * sigma / S, with sigma the
  residual standard error of the fit and S the slope.

The fit is fully deterministic (no stochastic step); ``random_state`` is accepted by the
public entry points only for API symmetry with the rest of the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from .constants import QUANT_TOL_PPM

__all__ = [
    "CalLevel",
    "CalibrationModel",
    "region_response",
    "fit_calibration",
    "apply_calibration",
    "calibration_report",
]

_WEIGHTINGS = ("none", "1/x", "1/x2")
_RESPONSES = ("intensity", "is_ratio")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class CalLevel:
    """One calibration standard: a region with a known concentration.

    ``mask`` is a boolean ``[n_pixels]`` array resolved by the caller (e.g. via
    ``session.mask_from_indices`` / ``SegmentTabMixin._region_pixel_mask``)."""

    region_name: str
    concentration: float
    mask: np.ndarray


@dataclass
class CalibrationModel:
    """A fitted per-analyte response curve plus its quality/limit metrics.

    The model maps response (analyte intensity, or analyte/IS ratio) to concentration by
    inverse prediction ``c = (response - intercept) / slope``."""

    analyte_mz: float
    is_mz: Optional[float]
    slope: float
    intercept: float
    r2: float
    n_levels: int
    residual_se: float
    lod: Optional[float]
    loq: Optional[float]
    conc_min: float
    conc_max: float
    units: str
    weighting: str
    through_origin: bool
    response: str

    def to_dict(self) -> dict:
        d = asdict(self)
        # JSON-friendly scalars (no numpy types leak into the session file).
        for k, v in list(d.items()):
            if isinstance(v, (np.floating,)):
                d[k] = float(v)
            elif isinstance(v, (np.integer,)):
                d[k] = int(v)
            elif isinstance(v, (np.bool_,)):
                d[k] = bool(v)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationModel":
        is_mz = d.get("is_mz")
        lod = d.get("lod")
        loq = d.get("loq")
        return cls(
            analyte_mz=float(d["analyte_mz"]),
            is_mz=None if is_mz is None else float(is_mz),
            slope=float(d["slope"]),
            intercept=float(d["intercept"]),
            r2=float(d["r2"]),
            n_levels=int(d["n_levels"]),
            residual_se=float(d["residual_se"]),
            lod=None if lod is None else float(lod),
            loq=None if loq is None else float(loq),
            conc_min=float(d["conc_min"]),
            conc_max=float(d["conc_max"]),
            units=str(d.get("units", "a.u.")),
            weighting=str(d.get("weighting", "none")),
            through_origin=bool(d.get("through_origin", False)),
            response=str(d.get("response", "intensity")),
        )


# --------------------------------------------------------------------------- #
# Response extraction
# --------------------------------------------------------------------------- #
def _region_mean_intensity(ds, mask, mz, *, tol_ppm, norm, reduce, agg):
    """Region-aggregated raw intensity at a single peak m/z.

    Extracts only the masked pixel rows via ``ds.features_for_rows`` (the efficient
    region-only path) and aggregates the per-pixel column with ``agg`` (mean/median)."""
    rows = np.flatnonzero(np.asarray(mask, dtype=bool))
    if rows.size == 0:
        raise ValueError("calibration region resolves to zero pixels")
    col = ds.features_for_rows([mz], rows, tol_ppm=tol_ppm, reduce=reduce, norm=norm)[:, 0]
    if agg == "median":
        return float(np.median(col))
    if agg == "mean":
        return float(np.mean(col))
    raise ValueError(f"unknown agg {agg!r} (use 'mean' or 'median')")


def region_response(ds, mask, analyte_mz, *, is_mz=None, tol_ppm=QUANT_TOL_PPM,
                    norm="tic", reduce="sum", agg="mean") -> float:
    """Mean (or median) analyte response over a region.

    Without ``is_mz`` this is the region-aggregated analyte intensity. With ``is_mz`` set,
    the response is the per-region analyte/IS intensity *ratio* (ratio of the region-mean
    analyte intensity to the region-mean IS intensity) — the internal-standard
    normalization that cancels per-region ionization/matrix-effect variation
    (Groseclose & Castellino 2016, doi:10.1021/acs.analchem.5b04409).

    Returns a scalar response. Raises ``ValueError`` on an empty region or a zero IS
    intensity (the ratio is undefined; surfacing it rather than silently no-oping)."""
    analyte = _region_mean_intensity(
        ds, mask, float(analyte_mz), tol_ppm=tol_ppm, norm=norm, reduce=reduce, agg=agg)
    if is_mz is None:
        return analyte
    is_int = _region_mean_intensity(
        ds, mask, float(is_mz), tol_ppm=tol_ppm, norm=norm, reduce=reduce, agg=agg)
    if is_int <= 0:
        raise ValueError(
            "internal-standard intensity is zero/negative in a calibration region; "
            "the analyte/IS ratio is undefined (check is_mz and tol_ppm)")
    return analyte / is_int


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def _weights(conc, weighting):
    """Per-point weights for WLS. 1/x and 1/x^2 down-weight high-concentration points so
    the low end (where relative error matters) drives the fit (Almeida et al. 2002,
    doi:10.1016/S0378-4347(01)00549-9). Zero-concentration points get weight 0 under a
    1/x scheme (their weight is undefined)."""
    conc = np.asarray(conc, dtype=float)
    if weighting in (None, "none", ""):
        return np.ones_like(conc)
    with np.errstate(divide="ignore"):
        if weighting == "1/x":
            w = np.where(conc > 0, 1.0 / conc, 0.0)
        elif weighting == "1/x2":
            w = np.where(conc > 0, 1.0 / (conc * conc), 0.0)
        else:
            raise ValueError(
                f"unknown weighting {weighting!r} (use {' / '.join(_WEIGHTINGS)})")
    return w


def _fit_line(conc, resp, weights, through_origin):
    """Weighted least-squares straight line. Returns (slope, intercept, residual_se, r2).

    ``through_origin`` fits ``resp = slope * conc`` (single-column design via lstsq);
    otherwise fits ``resp = slope * conc + intercept``. residual_se is the (weighted)
    residual standard error sqrt(SS_res / (n - p)); r2 is the weighted coefficient of
    determination 1 - SS_res / SS_tot."""
    conc = np.asarray(conc, dtype=float)
    resp = np.asarray(resp, dtype=float)
    w = np.asarray(weights, dtype=float)
    n = conc.size

    sw = np.sqrt(w)
    if through_origin:
        A = (conc * sw)[:, None]
        p = 1
    else:
        A = np.column_stack([conc, np.ones_like(conc)]) * sw[:, None]
        p = 2
    b = resp * sw

    if n - p < 1:
        raise ValueError(
            f"need more calibration levels than fit parameters (got {n} for {p} params)")

    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    if through_origin:
        slope = float(coef[0])
        intercept = 0.0
        pred = slope * conc
    else:
        slope = float(coef[0])
        intercept = float(coef[1])
        pred = slope * conc + intercept

    resid = resp - pred
    ss_res = float(np.sum(w * resid ** 2))
    residual_se = float(np.sqrt(ss_res / (n - p)))

    # Weighted R^2. For the through-origin model the conventional total sum of squares is
    # taken about zero (no mean term), matching the no-intercept design.
    if through_origin:
        ss_tot = float(np.sum(w * resp ** 2))
    else:
        wmean = float(np.sum(w * resp) / np.sum(w))
        ss_tot = float(np.sum(w * (resp - wmean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, intercept, residual_se, r2


def fit_calibration(ds, levels, analyte_mz, *, is_mz=None, tol_ppm=QUANT_TOL_PPM,
                    norm="tic", reduce="sum", units="a.u.", weighting="none",
                    through_origin=False, agg="mean",
                    random_state=None) -> CalibrationModel:
    """Fit a per-analyte on-tissue calibration curve from standard-spot regions.

    For each ``CalLevel`` the region-mean response is extracted via :func:`region_response`
    (analyte intensity, or analyte/IS ratio when ``is_mz`` is given), then a (optionally
    weighted, optionally through-origin) least-squares straight line response=f(concentration)
    is fit. R^2 comes from the (weighted) residuals; the residual standard error gives the
    IUPAC/ICH LOD = 3.3*sigma/slope and LOQ = 10*sigma/slope.

    Citations: linear/WLS calibration + concentration back-calculation
    (doi:10.1021/acs.analchem.5b04409, PMC12138873); 1/x, 1/x^2 weighting
    (doi:10.1016/S0378-4347(01)00549-9); LOD/LOQ (ICH Q2(R1)).

    Raises ``ValueError`` if fewer than 2 distinct concentration levels are supplied or the
    design is degenerate (zero concentration variance)."""
    if weighting not in _WEIGHTINGS:
        raise ValueError(f"unknown weighting {weighting!r} (use {' / '.join(_WEIGHTINGS)})")
    levels = list(levels)
    if len(levels) < 2:
        raise ValueError("calibration needs at least 2 standard levels")

    conc = np.array([float(lv.concentration) for lv in levels], dtype=float)
    if np.unique(conc).size < 2:
        raise ValueError(
            "calibration needs at least 2 distinct concentration levels "
            "(zero-variance design)")

    resp = np.array(
        [region_response(ds, lv.mask, analyte_mz, is_mz=is_mz, tol_ppm=tol_ppm,
                         norm=norm, reduce=reduce, agg=agg)
         for lv in levels],
        dtype=float)

    w = _weights(conc, weighting)
    if not np.any(w > 0):
        raise ValueError(
            "all calibration weights are zero (a 1/x scheme requires "
            "non-zero concentrations)")

    slope, intercept, residual_se, r2 = _fit_line(conc, resp, w, through_origin)

    if slope > 0 and residual_se > 0:
        lod = float(3.3 * residual_se / slope)
        loq = float(10.0 * residual_se / slope)
    else:
        # A non-positive slope makes the IUPAC limits meaningless; surface that as None
        # rather than emit a negative/garbage limit.
        lod = None
        loq = None

    return CalibrationModel(
        analyte_mz=float(analyte_mz),
        is_mz=None if is_mz is None else float(is_mz),
        slope=float(slope),
        intercept=float(intercept),
        r2=float(r2),
        n_levels=int(len(levels)),
        residual_se=float(residual_se),
        lod=lod,
        loq=loq,
        conc_min=float(conc.min()),
        conc_max=float(conc.max()),
        units=str(units),
        weighting=str(weighting),
        through_origin=bool(through_origin),
        response="is_ratio" if is_mz is not None else "intensity",
    )


# --------------------------------------------------------------------------- #
# Inverse prediction (concentration map)
# --------------------------------------------------------------------------- #
def apply_calibration(ds, model, *, tol_ppm=QUANT_TOL_PPM, norm="tic", reduce="sum"):
    """Inverse-predict per-pixel concentration from a fitted model.

    Computes the per-pixel response (analyte intensity, or analyte/IS ratio when
    ``model.is_mz`` is set) and back-calculates concentration c = (response - intercept)/slope
    (Groseclose & Castellino 2016, doi:10.1021/acs.analchem.5b04409).

    Returns ``(conc[n_pixels] float, in_range[n_pixels] bool)``. ``in_range`` marks pixels
    whose back-calculated concentration falls inside the calibrated range
    [conc_min, conc_max]; extrapolated pixels are flagged ``False`` and are **never silently
    clamped** (the raw extrapolated value is returned so the caller can show + flag it)."""
    if model.slope == 0:
        raise ValueError("model slope is zero; inverse prediction is undefined")

    peaks = [model.analyte_mz]
    if model.is_mz is not None:
        peaks.append(model.is_mz)
    X = ds.ensure_features(np.asarray(peaks, dtype=float), tol_ppm=tol_ppm, reduce=reduce)
    Xn = ds.feature_matrix(norm)
    analyte = Xn[:, 0]
    if model.is_mz is not None:
        is_int = Xn[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            response = np.where(is_int > 0, analyte / is_int, np.nan)
    else:
        response = analyte

    conc = (response - model.intercept) / model.slope
    # Tolerant boundary so a pixel sitting exactly at the calibrated min/max is not
    # flagged out-of-range by float round-off; genuine extrapolation is still flagged.
    span = model.conc_max - model.conc_min
    eps = 1e-9 * (abs(span) if span else max(abs(model.conc_max), 1.0))
    in_range = (conc >= model.conc_min - eps) & (conc <= model.conc_max + eps)
    in_range = np.asarray(in_range) & np.isfinite(conc)
    return conc.astype(float), in_range.astype(bool)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def calibration_report(model) -> dict:
    """Flat audit/report dict for a fitted model.

    Keys are exactly what the GUI passes to ``record_step`` (the provenance/Report/CSV
    header): analyte_mz, is_mz, slope, intercept, r2, residual_se, lod, loq, units,
    n_levels, conc_min, conc_max, weighting, through_origin, response."""
    return {
        "analyte_mz": float(model.analyte_mz),
        "is_mz": None if model.is_mz is None else float(model.is_mz),
        "slope": float(model.slope),
        "intercept": float(model.intercept),
        "r2": float(model.r2),
        "residual_se": float(model.residual_se),
        "lod": None if model.lod is None else float(model.lod),
        "loq": None if model.loq is None else float(model.loq),
        "units": str(model.units),
        "n_levels": int(model.n_levels),
        "conc_min": float(model.conc_min),
        "conc_max": float(model.conc_max),
        "weighting": str(model.weighting),
        "through_origin": bool(model.through_origin),
        "response": str(model.response),
    }
