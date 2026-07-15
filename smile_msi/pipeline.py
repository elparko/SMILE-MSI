"""End-to-end pipeline shared by the CLI, the Streamlit UI, and scripts.

Flexible-input flow:
  read_table(file)            -> DataFrame (handles SCiLS '#' headers, ;/, /tab, xlsx)
  annotate_and_score(df, ...) -> per-peak AUC (two-group Gaussian) + lipid annotation
  build_report(df, out, ...)  -> formatted 2-sheet Excel (identified / unidentified)

The caller chooses which columns map to m/z and to each group's mean & SD, so the
tool is not tied to SCiLS' specific column names.
"""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

from .match import Annotator

# --------------------------------------------------------------------------- #
# Reference tables (display only)
# --------------------------------------------------------------------------- #
CLASS_FULL = {
    "PA": "Phosphatidic acid", "PC": "Phosphatidylcholine",
    "PE": "Phosphatidylethanolamine", "PS": "Phosphatidylserine",
    "PG": "Phosphatidylglycerol", "PI": "Phosphatidylinositol",
    "PE-O": "Ether/plasmalogen PE", "PC-O": "Ether/plasmalogen PC",
    "LPA": "Lyso-phosphatidic acid", "LPC": "Lyso-phosphatidylcholine",
    "LPE": "Lyso-phosphatidylethanolamine", "LPI": "Lyso-phosphatidylinositol",
    "LPS": "Lyso-phosphatidylserine", "LPG": "Lyso-phosphatidylglycerol",
    "SM": "Sphingomyelin", "Cer": "Ceramide",
    "HexCer": "Hexosylceramide (Gal/Glc-Cer)", "Sulfatide": "Sulfatide (sulfated HexCer)",
    "FA": "Fatty acid",
    "GM3": "Ganglioside GM3", "GM2": "Ganglioside GM2", "GM1": "Ganglioside GM1",
    "GD3": "Ganglioside GD3", "GD1": "Ganglioside GD1", "GT1": "Ganglioside GT1",
    "CE": "Cholesteryl ester", "ST": "Sterol",
    "CAR": "Acylcarnitine", "Metab": "Metabolite",
}
CLASS_CAT = {
    **{c: "Sphingolipids" for c in ["SM", "Cer", "HexCer", "Sulfatide"]},
    **{c: "Gangliosides" for c in ["GM3", "GM2", "GM1", "GD3", "GD1", "GT1"]},
    **{c: "Ether lipids / plasmalogens" for c in ["PE-O", "PC-O"]},
    **{c: "Glycerophospholipids" for c in ["PA", "PC", "PE", "PS", "PG", "PI"]},
    **{c: "Lysophospholipids" for c in ["LPA", "LPC", "LPE", "LPI", "LPS", "LPG"]},
    **{c: "Sterol lipids" for c in ["CE", "ST"]},
    "CAR": "Acylcarnitines", "Metab": "Metabolites",
    "FA": "Fatty acids",
}
CAT_ORDER = ["Sphingolipids", "Gangliosides", "Ether lipids / plasmalogens",
             "Glycerophospholipids", "Lysophospholipids", "Sterol lipids", "Acylcarnitines",
             "Fatty acids", "Metabolites"]
ADDUCT_DESCS = [
    ("[M-H]-", "Molecule minus a proton - the usual negative-mode ion"),
    ("[M-CH3]-", "Loss of a methyl group - typical for PC and SM"),
    ("[M+Cl]-", "Chloride adduct"),
    ("[M+HCOO]-", "Formate adduct"),
    ("[M+CH3COO]-", "Acetate adduct"),
]


# --------------------------------------------------------------------------- #
# 1. Reading
# --------------------------------------------------------------------------- #
def _is_number(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _detect_header(raw: pd.DataFrame) -> int:
    """First row with >=3 non-numeric string cells = the column-header row."""
    for i in range(min(25, len(raw))):
        strs = [c for c in raw.iloc[i].tolist() if isinstance(c, str) and not _is_number(c)]
        if len(strs) >= 3:
            return i
    return 0


def read_table(source, header_row: int | None = None) -> pd.DataFrame:
    """Read a peak table from a path or file-like object.

    Supports CSV/TSV and XLSX. Tolerates SCiLS exports: leading '#' comment lines,
    ';' or tab delimiters, and '#hexcolor' cells mid-row. header_row (0-based) forces
    the header row for spreadsheets; None auto-detects.
    """
    if hasattr(source, "seek"):
        try:
            source.seek(0)
        except Exception:  # noqa: BLE001 — non-seekable stream: just read from where it is
            pass
    name = getattr(source, "name", source if isinstance(source, str) else "")
    if str(name).lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(source, header=None, dtype=object)
        hdr = header_row if header_row is not None else _detect_header(raw)
        df = raw.iloc[hdr + 1:].copy()
        df.columns = [str(c) for c in raw.iloc[hdr].tolist()]
        return df.reset_index(drop=True)

    if hasattr(source, "read"):
        data = source.read()
        text = data.decode("utf-8-sig", errors="replace") if isinstance(data, (bytes, bytearray)) else data
    else:
        with open(source, encoding="utf-8-sig") as fh:
            text = fh.read()
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    head = lines[0] if lines else ""
    counts = {d: head.count(d) for d in [";", "\t", ","]}
    delim = max(counts, key=counts.get) if max(counts.values()) > 0 else ","
    return pd.read_csv(io.StringIO("\n".join(lines)), delimiter=delim)


def guess_columns(columns) -> dict:
    """Best-guess column assignment to help pre-fill the UI. Returns a dict with
    keys mz, a_mean, a_sd, b_mean, b_sd (values may be None)."""
    cols = list(columns)
    low = {c: str(c).lower() for c in cols}
    def find(*subs, exclude=()):
        for c in cols:
            s = low[c]
            if all(x in s for x in subs) and not any(x in s for x in exclude):
                yield c
    def find_token(tok):
        # tok as a whole word, treating _ - . as separators (so "Control_sd" matches "sd")
        return [c for c in cols if re.search(rf"\b{tok}\b", re.sub(r"[\s_\-.]+", " ", low[c]))]
    mz = next((c for c in cols if low[c] in ("m/z", "mz") or "m/z" in low[c] or low[c].startswith("mz")), None)
    means = list(find("average")) or list(find("mean")) or find_token("mean")
    sds = list(find("deviation")) or list(find("std")) or find_token("sd")
    return {
        "mz": mz,
        "a_mean": means[0] if len(means) > 0 else None,
        "b_mean": means[1] if len(means) > 1 else None,
        "a_sd": sds[0] if len(sds) > 0 else None,
        "b_sd": sds[1] if len(sds) > 1 else None,
    }


# Stat-suffix tokens stripped when deriving a group name from a column header.
_STAT_TOKENS = r"mean|average|avg|median|std(?:ev)?|sd|deviation|stdev|intensity|value|signal"


def label_from_column(col, fallback: str = "") -> str:
    """Derive a human group name from an intensity column header.

    Strips common statistic suffixes/prefixes so e.g. ``"Control_mean"`` -> ``"Control"``,
    ``"Group A average intensity"`` -> ``"Group A"``. Returns ``fallback`` if nothing
    meaningful is left (or the column is None)."""
    if col is None:
        return fallback
    s = re.sub(r"[\s_\-.]+", " ", str(col))                  # separators -> space first
    s = re.sub(rf"(?i)\b({_STAT_TOKENS})s?\b", " ", s)       # strip statistic words
    s = re.sub(r"\s+", " ", s).strip()
    return s or fallback


# --------------------------------------------------------------------------- #
# 2. AUC + annotation
# --------------------------------------------------------------------------- #
def compute_auc(df, mz_col, a_mean, a_sd, b_mean, b_sd) -> pd.DataFrame:
    """Two-group ROC AUC per peak via the Gaussian approximation
    AUC = P(intensity_B > intensity_A) = Phi((mB-mA)/sqrt(sA^2+sB^2)).
    AUC>0.5 => higher in group B. Returns tidy df (mz, mean/sd per group, AUC, log2_fc)."""
    mz = pd.to_numeric(df[mz_col], errors="coerce")
    mA = pd.to_numeric(df[a_mean], errors="coerce")
    mB = pd.to_numeric(df[b_mean], errors="coerce")
    sA = pd.to_numeric(df[a_sd], errors="coerce")
    sB = pd.to_numeric(df[b_sd], errors="coerce")
    from scipy.stats import norm  # lazy: keeps scipy.stats out of GUI startup
    denom = np.sqrt(sA ** 2 + sB ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        auc = norm.cdf((mB - mA) / denom)
    # fall back to the mean-difference sign when SDs are zero/missing
    bad = ~np.isfinite(auc)
    auc = np.where(bad, np.where(mB > mA, 1.0, np.where(mB < mA, 0.0, 0.5)), auc)
    # signed log2 fold-change (B over A, matching AUC's ">0.5 ⇒ higher in B") with a
    # data-scaled pseudocount τ = 5th percentile of the positive group means, so a pair of
    # near-noise means gives log2 FC ≈ 0 instead of exploding (a fixed eps=1.0 is on the
    # wrong scale for these integrated intensities; KNOWN_ISSUES.md artifact #2)
    pooled = np.concatenate([np.asarray(mA, float), np.asarray(mB, float)])
    pooled = pooled[np.isfinite(pooled) & (pooled > 0)]
    tau = float(np.percentile(pooled, 5)) if pooled.size else 1.0
    log2_fc = np.log2((mB + tau) / (mA + tau))
    out = pd.DataFrame({
        "mz": mz, "mean_A": mA, "sd_A": sA, "mean_B": mB, "sd_B": sB,
        "AUC": auc, "log2_fc": log2_fc,
    })
    out = out[out["mz"].notna()].reset_index(drop=True)
    # mark provenance of the AUC so downstream/readers know it's the approximation,
    # not the exact rank-based AUC that spatial.roi_comparison computes from pixels
    out.attrs["auc_method"] = "Gaussian approximation (group mean+/-SD)"
    return out


def annotate_df(df, mz_col="mz", mode="negative", ppm=5.0, db=None,
                recalibrate=True) -> pd.DataFrame:
    """Add lipid annotation columns to a per-feature table (e.g. an A-vs-B comparison):
    ``best_lipid/best_class/best_adduct/best_ppm/n_candidates/other_candidates`` plus an MS1
    ``id_confidence`` (0–100).

    ``id_confidence`` blends mass accuracy, sibling-adduct corroboration (using this table's
    own m/z list as the peak list) and how decisively the top candidate beats the runner-up —
    so a *statistically significant* hit in a volcano/ROC table also carries **how trustworthy
    its lipid ID is**, not just a name. It is deliberately image-free (no dataset here), so it
    reads on the same 0–100 scale as, but is weaker than, the Features-tab confidence that adds
    isotope + spatial evidence; NaN for an unidentified m/z.

    ``recalibrate`` (default on) applies the same gated per-dataset mass self-calibration as
    :func:`smile_msi.annotate.build_feature_list` — measuring the table's systematic m/z offset
    from its own confident matches and matching at the corrected m/z — so a region/cohort
    comparison over a slightly mis-calibrated dataset labels the *same* extra ions the Features
    tab does, not fewer. Gated (needs enough matches / a real offset), records the applied
    offset on ``df.attrs['mass_offset_ppm']``; the reported ``best_ppm`` is the residual."""
    from . import isotopes
    from .annotate import estimate_mass_offset
    ann = Annotator(mode=mode, ppm_tol=ppm, db=db)
    peak_mzs = [float(m) for m in df[mz_col] if pd.notna(m)]
    offset_ppm, _n = (estimate_mass_offset(peak_mzs, mode=mode, db=db)
                      if (recalibrate and peak_mzs) else (0.0, 0))
    cal = 1.0 / (1.0 + offset_ppm * 1e-6) if offset_ppm else 1.0
    peak_mzs_q = [m * cal for m in peak_mzs] if offset_ppm else peak_mzs
    rows = []
    for mz in df[mz_col]:
        cs = ann.annotate_mz(float(mz) * cal) if pd.notna(mz) else []
        if cs:
            b = cs[0]
            gap = (b.score - cs[1].score) if len(cs) > 1 else None
            add = isotopes.corroborating_adducts(b.lipid.neutral_mass, mode, peak_mzs_q)
            conf = isotopes.confidence_detail(b.ppm, False, len(add), ppm_tol=ppm,
                                              score_gap=gap)["score"]
            rows.append((b.lipid.name, b.lipid.lipid_class, b.adduct, round(b.ppm, 2),
                         len(cs), " | ".join(f"{c.lipid.name} {c.adduct} ({c.ppm:+.1f})" for c in cs[1:3]),
                         int(conf)))
        else:
            rows.append(("", "", "", np.nan, 0, "", np.nan))
    cols = ["best_lipid", "best_class", "best_adduct", "best_ppm", "n_candidates",
            "other_candidates", "id_confidence"]
    out = df.copy()
    out[cols] = pd.DataFrame(rows, index=out.index)
    out.attrs["mass_offset_ppm"] = float(offset_ppm)
    return out


def annotate_and_score(df, mz_col, a_mean, a_sd, b_mean, b_sd, mode="negative", ppm=5.0):
    """One call: AUC + lipid annotation. Returns the merged per-peak DataFrame."""
    scored = compute_auc(df, mz_col, a_mean, a_sd, b_mean, b_sd)
    return annotate_df(scored, "mz", mode=mode, ppm=ppm)


# --------------------------------------------------------------------------- #
# 3. Excel report
# --------------------------------------------------------------------------- #
def _build_overview(wb, d, a_label, b_label):
    """Add a leading 'Overview' sheet with four native (editable) Excel charts,
    backed by a hidden '_data' sheet that holds the chart source ranges:
      1. Top discriminating lipids  (diverging horizontal bars)
      2. Per-peak mean intensity    (A vs B log-log scatter, colored by band)
      3. AUC distribution           (histogram)
      4. By lipid category          (peak count bars + median-AUC line)
    """
    from openpyxl.styles import Font
    from openpyxl.chart import BarChart, LineChart, ScatterChart, Reference, Series
    from openpyxl.chart.marker import Marker
    from openpyxl.chart.shapes import GraphicalProperties
    from openpyxl.drawing.line import LineProperties

    A_FILL, B_FILL = "ED7D31", "2E75B6"  # orange = group A side, blue = group B side
    over = wb.create_sheet("Overview", 0)
    dat = wb.create_sheet("_data"); dat.sheet_state = "hidden"
    over["A1"] = f"Overview  -  {a_label} vs {b_label}"; over["A1"].font = Font(bold=True, size=14)
    over["A2"] = (f"Native, editable charts. AUC > 0.5 favors {b_label}; < 0.5 favors {a_label}. "
                  "Source data is on the hidden '_data' sheet.")
    over["A2"].font = Font(italic=True, color="808080")

    def fill(series, rgb):  # give a chart series a solid fill, robustly
        series.graphicalProperties = GraphicalProperties(solidFill=rgb)

    # ---- 1. Top discriminating lipids: diverging bars of (AUC - 0.5) ----
    w = d.copy()
    w["signed"] = w["AUC"] - 0.5
    top = w.reindex(w["signed"].abs().sort_values(ascending=False).index).head(15).sort_values("signed")
    dat["A1"] = "label"; dat["B1"] = f"favors {b_label}"; dat["C1"] = f"favors {a_label}"
    for i, (_, x) in enumerate(top.iterrows(), start=2):
        dat.cell(i, 1, x.best_lipid if x.best_lipid else f"m/z {x['mz']:.3f}")
        dat.cell(i, 2, round(x.signed, 3) if x.signed >= 0 else None)
        dat.cell(i, 3, round(x.signed, 3) if x.signed < 0 else None)
    n1 = len(top) + 1
    bar = BarChart(); bar.type = "bar"; bar.grouping = "stacked"; bar.overlap = 100
    bar.title = "Top discriminating lipids"
    bar.add_data(Reference(dat, min_col=2, max_col=3, min_row=1, max_row=n1), titles_from_data=True)
    bar.set_categories(Reference(dat, min_col=1, min_row=2, max_row=n1))
    fill(bar.series[0], B_FILL); fill(bar.series[1], A_FILL)
    bar.x_axis.title = "AUC - 0.5 (signed separation)"; bar.x_axis.delete = False; bar.y_axis.delete = False
    bar.height = 9.5; bar.width = 17
    over.add_chart(bar, "A4")

    # ---- 2. Mean A vs mean B scatter, one colored series per band ----
    sca = d.copy()
    sca["xa"] = pd.to_numeric(sca["mean_A"], errors="coerce").fillna(0).clip(lower=1)
    sca["yb"] = pd.to_numeric(sca["mean_B"], errors="coerce").fillna(0).clip(lower=1)
    bands = [("Bs", f"{b_label} strong", "1F4E79"), ("Bl", f"{b_label} loose", "2E75B6"),
             ("Al", f"{a_label} loose", "F4B183"), ("As", f"{a_label} strong", "C55A11")]
    dat.cell(1, 5, "meanA")
    for j, (_b, name, _c) in enumerate(bands):
        dat.cell(1, 6 + j, name)
    for i, (_, x) in enumerate(sca.iterrows(), start=2):
        dat.cell(i, 5, round(x.xa, 1))
        for j, (b, _n, _c) in enumerate(bands):
            dat.cell(i, 6 + j, round(x.yb, 1) if x.band == b else None)
    n2 = len(sca) + 1
    sc = ScatterChart(); sc.scatterStyle = "marker"; sc.title = f"Per-peak mean: {a_label} vs {b_label}"
    sc.x_axis.title = f"Mean {a_label}"; sc.y_axis.title = f"Mean {b_label}"
    sc.x_axis.scaling.logBase = 10; sc.y_axis.scaling.logBase = 10
    sc.x_axis.delete = False; sc.y_axis.delete = False
    xref = Reference(dat, min_col=5, min_row=2, max_row=n2)
    for j, (_b, _n, c) in enumerate(bands):
        ser = Series(Reference(dat, min_col=6 + j, min_row=1, max_row=n2), xref, title_from_data=True)
        ser.marker = Marker(symbol="circle", size=7)
        ser.marker.graphicalProperties = GraphicalProperties(solidFill=c)
        ser.graphicalProperties = GraphicalProperties()
        ser.graphicalProperties.line = LineProperties(noFill=True)  # markers only, no connecting line
        sc.series.append(ser)
    sc.height = 9.5; sc.width = 17
    over.add_chart(sc, "K4")

    # ---- 3. AUC distribution histogram (10 fixed bins) ----
    edges = [round(i / 10, 1) for i in range(11)]
    counts, _ = np.histogram(pd.to_numeric(d["AUC"], errors="coerce").dropna(), bins=edges)
    dat.cell(1, 13, "AUC bin"); dat.cell(1, 14, "peaks")
    for kk in range(10):
        dat.cell(2 + kk, 13, f"{edges[kk]:.1f}-{edges[kk + 1]:.1f}")
        dat.cell(2 + kk, 14, int(counts[kk]))
    hist = BarChart(); hist.type = "col"; hist.title = "AUC distribution (all peaks)"; hist.legend = None
    hist.add_data(Reference(dat, min_col=14, min_row=1, max_row=11), titles_from_data=True)
    hist.set_categories(Reference(dat, min_col=13, min_row=2, max_row=11))
    hist.x_axis.title = f"AUC  ({a_label} <- 0.5 -> {b_label})"; hist.y_axis.title = "peaks"
    hist.x_axis.delete = False; hist.y_axis.delete = False
    fill(hist.series[0], "8EAADB")
    hist.height = 9.5; hist.width = 17
    over.add_chart(hist, "A24")

    # ---- 4. By lipid category: count bars + median-AUC line on a 2nd axis ----
    dat.cell(1, 16, "category"); dat.cell(1, 17, "peaks"); dat.cell(1, 18, "median AUC")
    r = 2
    for cat in CAT_ORDER + ["Unidentified"]:
        sub = d[d.category == cat]
        if not len(sub):
            continue
        dat.cell(r, 16, cat); dat.cell(r, 17, len(sub)); dat.cell(r, 18, round(sub.AUC.median(), 3))
        r += 1
    last = r - 1
    cbar = BarChart(); cbar.type = "col"; cbar.title = "By lipid category"
    cbar.add_data(Reference(dat, min_col=17, min_row=1, max_row=last), titles_from_data=True)
    cbar.set_categories(Reference(dat, min_col=16, min_row=2, max_row=last))
    cbar.y_axis.title = "peaks"; cbar.x_axis.delete = False; cbar.y_axis.delete = False
    fill(cbar.series[0], "A9D08E")
    cline = LineChart()
    cline.add_data(Reference(dat, min_col=18, min_row=1, max_row=last), titles_from_data=True)
    cline.y_axis.axId = 200; cline.y_axis.title = "median AUC"; cline.y_axis.crosses = "max"
    cline.series[0].graphicalProperties = GraphicalProperties()
    cline.series[0].graphicalProperties.line = LineProperties(solidFill="C00000", w=20000)
    cbar += cline
    cbar.height = 9.5; cbar.width = 17
    over.add_chart(cbar, "K24")


def _provenance_sheet(wb, provenance):
    """Append a 'Provenance & methods' sheet documenting how the report was made."""
    from openpyxl.styles import Font

    ws = wb.create_sheet("Provenance & methods")
    r = 1

    def put(col, val, bold=False, italic=False):
        c = ws.cell(row=r, column=col, value=val)
        if bold or italic:
            c.font = Font(bold=bold, italic=italic)

    p = provenance.to_dict()
    put(1, "PROVENANCE & METHODS", bold=True); r += 1
    put(1, f"Generated {p['started']}", italic=True); r += 2
    if p["inputs"]:
        put(1, "Input files", bold=True); r += 1
        for i in p["inputs"]:
            put(1, i["name"]); put(2, f"{i['bytes']:,} bytes"); put(3, f"sha256 {i['sha256'][:16]}…")
            r += 1
        r += 1
    d = p["dataset"]
    if d:
        put(1, "Dataset", bold=True); r += 1
        put(1, f"{d['pixels']:,} pixels, {d['width']}x{d['height']}, m/z {d['mz_min']}–{d['mz_max']}, "
               f"{d.get('polarity') or '?'} mode"); r += 2
    put(1, "Processing steps", bold=True); r += 1
    for k, s in enumerate(p["steps"], 1):
        params = ", ".join(f"{a}={b}" for a, b in s["params"].items())
        put(1, f"{k}. {s['step']}"); put(2, params); r += 1
    r += 1
    put(1, "Software", bold=True); r += 1
    for key, val in p["environment"].items():
        put(1, key); put(2, str(val)); r += 1
    r += 1
    put(1, "Methods (draft)", bold=True); r += 1
    cell = ws.cell(row=r, column=1, value=provenance.methods_paragraph())
    from openpyxl.styles import Alignment
    cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells(start_row=r, start_column=1, end_row=r + 8, end_column=6)
    for col, w in zip("ABC", (26, 60, 30)):
        ws.column_dimensions[col].width = w


def build_report(df, out, a_label="Group A", b_label="Group B", mode="negative", note=None,
                 provenance=None):
    """Write the formatted 2-sheet workbook. `out` may be a path or a file-like
    (e.g. BytesIO) so the UI can stream it for download.

    df must contain: mz, AUC, log2_fc, mean_A, mean_B, best_lipid, best_class,
    best_adduct, best_ppm.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    d = df.copy()
    d["best_lipid"] = d["best_lipid"].fillna("")
    d["best_class"] = d["best_class"].fillna("")
    d["category"] = d["best_class"].map(CLASS_CAT).fillna("Unidentified")
    d["class_full"] = d["best_class"].map(CLASS_FULL).fillna("-")
    d["discrimination"] = ((d["AUC"] - 0.5).abs() * 2).round(2)

    P_BS, P_BL = f"{b_label} (strong)", f"{b_label} (loose)"
    P_AL, P_AS = f"{a_label} (loose)", f"{a_label} (strong)"

    def preference(a):
        if a >= 0.70: return P_BS
        if a >= 0.50: return P_BL
        if a > 0.30:  return P_AL
        return P_AS

    def band(a):
        if a >= 0.70: return "Bs"
        if a >= 0.50: return "Bl"
        if a > 0.30:  return "Al"
        return "As"

    d["preference"] = d["AUC"].map(preference)
    d["band"] = d["AUC"].map(band)

    def confidence(rw):
        # Mass-match confidence: mass accuracy + adduct plausibility ONLY. This is a
        # distinct (and deliberately differently-named) metric from the Features-tab
        # "confidence" (isotopes.confidence), which additionally folds in isotope-pattern
        # consistency and adduct corroboration — signals the report/CSV path often lacks.
        # The two can legitimately differ for the same ion; they are not the same number.
        if rw.best_lipid == "":
            return "unidentified"
        p = abs(rw.best_ppm) if pd.notna(rw.best_ppm) else 99
        odd = rw.best_adduct == "[M+Cl]-" and str(rw.best_class).startswith("L")
        if p <= 1.5 and not odd:
            return "High"
        return "Medium" if p <= 3 else "Low"
    d["confidence"] = d.apply(confidence, axis=1)

    MCOLS = ["m/z", "Lipid", "Lipid class (full)", "Adduct", "Mass error (ppm)", "AUC",
             "Present more in", "Discrim.", "log2 FC", f"Mean {a_label}", f"Mean {b_label}",
             "Mass-match confidence"]
    PREF_COL = 7
    COLKEY = [
        ("m/z", "Mass-to-charge ratio of the detected ion"),
        ("Lipid", "Sum-composition ID = class + total carbons:double bonds (e.g. PE 36:1). ';O2/;O3' = backbone OH count"),
        ("Lipid class (full)", "Full name of the lipid class (e.g. Phosphatidylethanolamine for PE)"),
        ("Adduct", "How the ion formed (see adduct key)"),
        ("Mass error (ppm)", "Measured vs theoretical m/z; smaller = more confident ID"),
        ("AUC", f"Separation 0-1: higher = more in {b_label}, lower = more in {a_label}, 0.5 = equal. "
                "Exact rank-based (Mann-Whitney) when computed from pixel data (ROI comparison); "
                "a Gaussian approximation from group mean+/-SD when computed from a peak table."),
        ("Present more in", f"4-tier: {b_label} strong (>=0.70)/loose(0.50-0.70), {a_label} loose(0.30-0.50)/strong(<=0.30)"),
        ("Discrimination", "|AUC-0.5| x 2: 0 = none, 1 = perfect separation"),
        ("log2 FC", f"Signed log2 fold-change ({b_label}/{a_label}): + = higher in "
                    f"{b_label}, - = higher in {a_label}, 0 = no change (pseudocount-regularized)"),
        (f"Mean {a_label} / {b_label}", "Average pixel intensity in each group"),
        ("Mass-match confidence", "Based on mass accuracy + adduct plausibility only: "
         "High <=1.5 ppm | Medium <=3 ppm | Low >3 ppm or unusual adduct | unidentified = no match. "
         "(The Features tab's 'confidence' also uses isotope pattern + adduct corroboration, "
         "so the two can differ for the same ion.)"),
    ]

    TITLE = Font(bold=True, size=14)
    SECTION = Font(bold=True, size=12, color="FFFFFF"); SECTION_FILL = PatternFill("solid", fgColor="2F5496")
    CATBAND = Font(bold=True, color="1F3864"); CATFILL = PatternFill("solid", fgColor="D9E1F2")
    HEAD = Font(bold=True, color="FFFFFF"); HEADFILL = PatternFill("solid", fgColor="8EAADB")
    thin = Side(style="thin", color="BFBFBF"); BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)
    CONF_FILL = {"High": "C6EFCE", "Medium": "FFEB9C", "Low": "F8CBAD", "unidentified": "EDEDED"}
    PREF_FILL = {P_BS: ("2E75B6", "FFFFFF"), P_BL: ("BDD7EE", "1F3864"),
                 P_AL: ("FCE4D6", "843C0C"), P_AS: ("ED7D31", "FFFFFF")}

    wb = Workbook()

    class Page:
        def __init__(self, ws):
            self.ws = ws; self.r = 1
        def put(self, col, val, font=None, fill=None, align=None, border=False, num=None):
            c = self.ws.cell(row=self.r, column=col, value=val)
            if font: c.font = font
            if fill: c.fill = fill
            if align: c.alignment = Alignment(horizontal=align, vertical="center", wrap_text=(align == "left"))
            if border: c.border = BORDER
            if num: c.number_format = num
            return c
        def section(self, title):
            self.put(1, title, SECTION, SECTION_FILL)
            for c in range(2, len(MCOLS) + 1): self.put(c, "", fill=SECTION_FILL)
            self.r += 1
        def header(self, cols):
            for i, name in enumerate(cols, 1):
                self.put(i, name, HEAD, HEADFILL, align="left", border=True)
            self.r += 1
        def row(self, x):
            lipid = x.best_lipid if x.best_lipid else "(unidentified)"
            vals = [round(x["mz"], 4), lipid, x.class_full, x.best_adduct,
                    (round(x.best_ppm, 2) if pd.notna(x.best_ppm) else ""),
                    round(x.AUC, 3), x.preference, x.discrimination, round(x.log2_fc, 2),
                    round(x.mean_A, 1), round(x.mean_B, 1), x.confidence]
            for i, v in enumerate(vals, 1):
                self.put(i, v, align="left", border=True)
            pf, pfont = PREF_FILL[x.preference]
            pc = self.ws.cell(row=self.r, column=PREF_COL)
            pc.fill = PatternFill("solid", fgColor=pf); pc.font = Font(bold=True, color=pfont)
            self.ws.cell(row=self.r, column=len(MCOLS)).fill = PatternFill("solid", fgColor=CONF_FILL.get(x.confidence, "FFFFFF"))
            self.r += 1
        def marker_table(self, title, subset, ascending, by_category):
            self.section(title)
            if not len(subset):
                self.put(1, "(none)", Font(italic=True, color="808080")); self.r += 2; return
            self.header(MCOLS)
            if by_category:
                sub = subset.copy()
                sub["catrank"] = sub.category.map({c: i for i, c in enumerate(CAT_ORDER)})
                sub = sub.sort_values(["catrank", "AUC"], ascending=[True, ascending])
                for cat in CAT_ORDER:
                    block = sub[sub.category == cat]
                    if not len(block): continue
                    self.put(1, cat, CATBAND, CATFILL)
                    for c in range(2, len(MCOLS) + 1): self.put(c, "", fill=CATFILL)
                    self.r += 1
                    for _, x in block.iterrows(): self.row(x)
            else:
                for _, x in subset.sort_values("AUC", ascending=ascending).iterrows():
                    self.row(x)
            self.r += 1

    def four_tier(page, src, by_category):
        page.marker_table(f"{b_label.upper()} (STRONG)  -  AUC >= 0.70  -  {len(src[src.band=='Bs'])} peaks", src[src.band == "Bs"], False, by_category)
        page.marker_table(f"{b_label.upper()} (LOOSE)  -  AUC 0.50-0.70  -  {len(src[src.band=='Bl'])} peaks", src[src.band == "Bl"], False, by_category)
        page.marker_table(f"{a_label.upper()} (LOOSE)  -  AUC 0.30-0.50  -  {len(src[src.band=='Al'])} peaks", src[src.band == "Al"], True, by_category)
        page.marker_table(f"{a_label.upper()} (STRONG)  -  AUC <= 0.30  -  {len(src[src.band=='As'])} peaks", src[src.band == "As"], True, by_category)

    # ---- Page 1: identified ----
    p = Page(wb.active); p.ws.title = "Identified lipids"
    p.put(1, f"Lipid MSI - {a_label} vs {b_label}  |  IDENTIFIED LIPIDS", TITLE); p.r += 1
    p.put(1, note or f"{mode.title()}-mode. AUC approximated from per-group mean/SD. IDs are sum-composition level (confirm isobars with MS/MS). Column definitions & abbreviations: see the 'Key & abbreviations' sheet. Unidentified peaks: see the 'Unidentified peaks' sheet. Charts: see the 'Overview' sheet.",
          Font(italic=True, color="808080")); p.r += 2

    p.section("SUMMARY BY CATEGORY (identified peaks)"); p.header(["Category", "# peaks", "Median AUC", "Present more in"])
    ident = d[d.category != "Unidentified"]
    for cat in CAT_ORDER:
        sub = ident[ident.category == cat]
        if not len(sub): continue
        med = sub.AUC.median()
        p.put(1, cat, align="left", border=True); p.put(2, len(sub), align="left", border=True)
        p.put(3, round(med, 2), align="left", border=True, num="0.00")
        p.put(4, preference(med), Font(bold=True), align="left", border=True); p.r += 1
    p.r += 1
    four_tier(p, ident, by_category=True)

    # ---- Page 2: unidentified ----
    q = Page(wb.create_sheet("Unidentified peaks"))
    q.put(1, "UNIDENTIFIED PEAKS  -  no membrane-lipid match (likely metabolites / oxylipins / sulfated species)", TITLE); q.r += 1
    q.put(1, "Column meanings: see the 'Key & abbreviations' sheet. Crack these with MS/MS or a metabolite database.", Font(italic=True, color="808080")); q.r += 2
    four_tier(q, d[d.category == "Unidentified"], by_category=False)

    # ---- Reference sheet: column key / adduct key / lipid abbreviations ----
    k = Page(wb.create_sheet("Key & abbreviations"))
    k.put(1, f"KEY & ABBREVIATIONS  -  reference for the {a_label} vs {b_label} report", TITLE); k.r += 2
    k.section("HOW TO READ THIS  -  column key"); k.header(["Column", "What it means"])
    for name, desc in COLKEY:
        k.put(1, name, Font(bold=True), align="left", border=True); k.put(2, desc, align="left", border=True); k.r += 1
    k.r += 1
    k.section("ADDUCT KEY"); k.header(["Adduct", "Meaning"])
    for a, desc in ADDUCT_DESCS:
        k.put(1, a, Font(bold=True), align="left", border=True); k.put(2, desc, align="left", border=True); k.r += 1
    k.r += 1
    k.section("LIPID ABBREVIATIONS  -  grouped by category"); k.header(["Category", "Abbrev.", "Full name"])
    for cat in CAT_ORDER:
        for j, c in enumerate([c for c in CLASS_FULL if CLASS_CAT.get(c) == cat]):
            k.put(1, cat if j == 0 else "", align="left", border=True)
            k.put(2, c, Font(bold=True), align="left", border=True)
            k.put(3, CLASS_FULL[c], align="left", border=True); k.r += 1

    widths = [11, 22, 30, 13, 15, 11, 20, 9, 7, 16, 16, 13]
    for ws in (wb["Identified lipids"], wb["Unidentified peaks"]):
        for i, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = w
        ws.freeze_panes = "A1"
    for col, w in zip("ABC", (22, 74, 36)):
        k.ws.column_dimensions[col].width = w
    k.ws.freeze_panes = "A1"

    if provenance is not None:
        _provenance_sheet(wb, provenance)

    _build_overview(wb, d, a_label, b_label)
    wb.active = 0  # open on the Overview sheet
    wb.save(out)
    return out
