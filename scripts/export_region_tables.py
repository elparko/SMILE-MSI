#!/usr/bin/env python3
"""Per-region feature tables for comparing regions on one slide — mounting media, replicate
ROIs, off-tissue background — with every metric computed inside each region.

The app's feature-table export mixes scopes: S/N and intensity come from the region a list was
picked on, but Moran's I is always whole-slide, there is no per-region TIC readout, and a saved
★ feature list drops S/N and regenerates it slide-wide on reload. This script picks and measures
each region on its own and writes one CSV per region plus a summary.

Per region it:
  1. picks peaks on the region's own mean spectrum             -> snr, intensity
  2. extracts every region's candidates in ONE pass over the slide
  3. measures each ion over the region's pixels only           -> mean_intensity, frequency, morans_i
  4. drops ions below --min-frequency (and --min-morans if set), collapses M+1/M+2 isotopes
  5. identifies lipids with the database the app loads         -> lipid, class, adduct, ppm, ...

Difference from the app's Spatial finder: its frequency and Moran's I gates run over the whole
slide; here they run over the region. Moran's I uses the app's formula (4-neighbour links, the
session's normalization) restricted to links whose two pixels are both inside the region.

OUTPUT  (in --out)
  <region>.csv          one row per feature
  regions_summary.csv   one row per region: pixels, area, mean/median TIC, feature counts
  settings.json         slide, session, every parameter, lipid database, software version

  Region CSV columns
    mz                    observed m/z
    snr                   peak height / noise on the region's mean spectrum (no baseline subtraction)
    intensity             region mean-spectrum height at the peak (absolute, per-pixel mean)
    mean_intensity        mean ion-image value over the region's pixels (absolute, +/-ppm window)
    frequency             fraction of region pixels where the ion exceeds 5% of its region maximum
    morans_i              Moran's I over the region's pixels only
    lipid, class, adduct, ppm, ...   lipid identification, same columns as the app
    morans_i_whole_slide  the app's whole-slide Moran's I, for reference only

USAGE  (run with the project venv; the app does not need to be open)

  # sessions saved on this machine
  .venv/bin/python scripts/export_region_tables.py --list

  # regions on one slide, with pixel counts
  .venv/bin/python scripts/export_region_tables.py --slide "slide_pos.imzML" --list-regions

  # QC: tissue ROI + off-tissue ROI per embedding medium
  .venv/bin/python scripts/export_region_tables.py --slide "slide_pos.imzML" --mode positive \\
      --region "OCT tissue" --region "CMC tissue" --region "CMC24 tissue" \\
      --region "OCT bg" --region "CMC bg" --region "CMC24 bg" --out media_qc_pos

  # synthetic demo slide
  .venv/bin/python scripts/export_region_tables.py --demo --all-regions --out /tmp/demo_regions

  Windows: .venv\\Scripts\\python scripts\\export_region_tables.py ...

Defaults are the positive-mode media QC settings: S/N 1.5, prominence 1.0, min rel. intensity 0,
min frequency 1%, no Moran's I gate, no candidate cap, isotopes collapsed. Each is a flag.
--slide takes a session path or stem, the sample name shown in the app, or the .imzML path.
Draw and name the regions in the app first; the script reads them from the saved session.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smile_msi import __version__, annotate, headless, isotopes, prefs  # noqa: E402

LEAD_COLUMNS = ["mz", "snr", "intensity", "mean_intensity", "frequency", "morans_i"]
URL_COLUMNS = ["pubmed_url", "europepmc_url", "scholar_url"]
PRESENT_FRAC = 0.05          # the app's per-ion presence floor (spatial.feature_frequency)
MORANS_CHUNK_BYTES = 256e6


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_name(name):
    return re.sub(r"[^\w.-]+", "_", str(name)).strip("_") or "region"


def load_lipid_db(use_builtin):
    if use_builtin:
        return None, "built-in"
    cfg = prefs.get("lipid_db")
    if not isinstance(cfg, dict) or not cfg.get("path"):
        return None, "built-in"
    path, mode = cfg["path"], cfg.get("mode", "merge")
    if not os.path.exists(path):
        say(f"saved lipid database not found ({path}); using the built-in database")
        return None, "built-in (saved database missing)"
    from smile_msi import lipiddb
    say(f"loading lipid database {os.path.basename(path)}")
    if path.lower().endswith(".sdf"):
        ext = lipiddb.load_sdf(path, categories=cfg.get("categories"),
                               min_mass=cfg.get("min_mass"), max_mass=cfg.get("max_mass"))
        db = lipiddb.merge_lipids(lipiddb.build_database(), ext)[0] if mode == "merge" else ext
    else:
        db = (lipiddb.merge_external_db(path)[0] if mode == "merge"
              else lipiddb.load_external_db(path))
    return db, f"{path} ({mode})"


def resolve_mode(requested, slide):
    saved = str((slide.data.get("settings") or {}).get("mode") or "")
    declared = str(getattr(slide.ds, "polarity", "") or "").lower()
    mode = requested or saved or ("positive" if declared.startswith("pos") else "negative")
    if declared and not declared.startswith(mode[:3]):
        say(f"WARNING: identifying lipids as {mode}, but the imzML declares polarity {declared!r}")
    return mode


def region_links(ds, idx):
    rows, cols = ds._pixel_rows_cols()
    grid = np.full((ds.height, ds.width), -1, dtype=np.int64)
    grid[rows[idx], cols[idx]] = np.arange(len(idx))
    right = (grid[:, :-1] >= 0) & (grid[:, 1:] >= 0)
    down = (grid[:-1, :] >= 0) & (grid[1:, :] >= 0)
    ei = np.concatenate([grid[:, :-1][right], grid[:-1, :][down]])
    ej = np.concatenate([grid[:, 1:][right], grid[1:, :][down]])
    return ei, ej


def region_morans(region_raw, factors, ei, ej):
    n, p = region_raw.shape
    out = np.zeros(p)
    if n < 2 or len(ei) == 0 or p == 0:
        return out
    chunk = max(1, int(MORANS_CHUNK_BYTES // max(1, len(ei) * 8 * 3)))
    for s in range(0, p, chunk):
        x = region_raw[:, s:s + chunk].astype(np.float64) / factors[:, None]
        z = x - x.mean(axis=0, keepdims=True)
        denom = (z * z).sum(axis=0)
        num = (z[ei] * z[ej]).sum(axis=0)
        out[s:s + chunk] = np.divide(n * num, len(ei) * denom,
                                     out=np.zeros(z.shape[1]), where=denom > 0)
    return out


def finite_median(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return round(float(np.median(a)), 3) if a.size else ""


def export_regions(slide, names, args):
    ds, api = slide.ds, slide.api
    os.makedirs(args.out, exist_ok=True)
    mode = resolve_mode(args.mode, slide)
    id_ppm = float((slide.data.get("settings") or {}).get("id_ppm", 5.0))
    masks = {n: api.region(n) for n in names}
    tic = ds.tic()
    factors = ds.norm_factors(api.norm)

    picks = {}
    for name in names:
        say(f"{name}: picking peaks on {int(masks[name].sum()):,} px")
        pk = ds.pick_peaks(snr=args.snr, min_rel_intensity=args.min_rel,
                           max_peaks=args.max_candidates, mask=masks[name],
                           projection=args.projection, prominence=args.prominence)
        picks[name] = pk
        capped = pk.n_detected > len(pk)
        say(f"{name}: {len(pk):,} candidates"
            + (f" (capped from {pk.n_detected:,})" if capped else ""))

    union = np.asarray(sorted({float(p["mz"]) for pk in picks.values() for p in pk}), dtype=float)
    gb = ds.n_pixels * len(union) * 4 / 1e9
    if gb > args.max_gb:
        raise SystemExit(f"extracting {len(union):,} ions over {ds.n_pixels:,} px needs ~{gb:.1f} GB "
                         f"(over --max-gb {args.max_gb}). Raise --snr, set --max-candidates, "
                         "or pass a larger --max-gb.")
    say(f"extracting {len(union):,} ions over {ds.n_pixels:,} px (~{gb:.2f} GB), one pass")
    if len(union):
        raw = ds.ensure_features(union, tol_ppm=api.ppm, reduce=api.reduce)
        union_cache = ds._feat
    else:
        raw, union_cache = np.empty((ds.n_pixels, 0), dtype=np.float32), None
    col = {m: j for j, m in enumerate(union.tolist())}

    db, db_label = (None, "not used") if args.skip_annotation else load_lipid_db(args.builtin_db)
    summary = []
    for name in names:
        idx = np.flatnonzero(masks[name])
        peaks = list(picks[name])
        cols = [col[float(p["mz"])] for p in peaks]
        region_raw = raw[np.ix_(idx, cols)]
        cmax = region_raw.max(axis=0) if region_raw.size else np.zeros(len(cols))
        frequency = ((region_raw > PRESENT_FRAC * cmax[None, :]) & (cmax[None, :] > 0)).mean(axis=0)
        ei, ej = region_links(ds, idx)
        morans = region_morans(region_raw, factors[idx], ei, ej)
        for p, f, mi, mean in zip(peaks, frequency, morans, region_raw.mean(axis=0)):
            p.update(frequency=round(float(f), 4), morans_i=round(float(mi), 4),
                     mean_intensity=round(float(mean), 2))

        n_candidates = len(peaks)
        peaks = [p for p in peaks if p["frequency"] >= args.min_frequency]
        n_after_frequency = len(peaks)
        if args.min_morans is not None:
            peaks = [p for p in peaks if p["morans_i"] >= args.min_morans]
        n_after_morans = len(peaks)
        if args.collapse_isotopes and peaks:
            peaks, _ = isotopes.deisotope(peaks, charge=1, tol_ppm=min(api.ppm, 15.0))

        table = pd.DataFrame({c: [p[c] for p in peaks] for c in LEAD_COLUMNS})
        n_identified = 0
        if not args.skip_annotation and peaks:
            say(f"{name}: identifying lipids for {len(peaks):,} features ({mode})")
            # Annotation shrinks the dataset's cached feature matrix to this region's ions.
            # Restoring the all-region cache makes the next region a column copy, not a new
            # pass over the slide.
            ds._feat = union_cache
            ann = annotate.build_feature_list(ds, peaks, mode=mode, match_ppm=id_ppm,
                                              image_ppm=api.ppm, norm=api.norm, db=db)
            ann = (ann.drop(columns=["mz", "intensity", "snr", *URL_COLUMNS], errors="ignore")
                      .rename(columns={"spatial_morans_i": "morans_i_whole_slide"}))
            table = pd.concat([table, ann.reset_index(drop=True)], axis=1)
            n_identified = int((table["lipid"].astype(str).str.strip() != "").sum())

        path = os.path.join(args.out, f"{safe_name(name)}.csv")
        table.to_csv(path, index=False, encoding="utf-8-sig")
        say(f"{name}: wrote {os.path.basename(path)} — {len(peaks):,} features, "
            f"{n_identified:,} identified")

        px_x = ds.pixel_size_um
        px_y = getattr(ds, "pixel_size_y_um", None) or px_x
        summary.append({
            "region": name, "n_px": len(idx),
            "area_mm2": round(len(idx) * px_x * px_y / 1e6, 4) if px_x else "",
            "mean_tic": round(float(tic[idx].mean()), 2) if len(idx) else "",
            "median_tic": round(float(np.median(tic[idx])), 2) if len(idx) else "",
            "n_detected": picks[name].n_detected, "n_candidates": n_candidates,
            "candidates_capped": picks[name].n_detected > len(picks[name]),
            "n_after_frequency": n_after_frequency, "n_after_morans": n_after_morans,
            "n_features": len(peaks), "n_identified": n_identified,
            "median_snr": finite_median([p["snr"] for p in peaks]),
            "median_morans_i": finite_median([p["morans_i"] for p in peaks]),
            "csv": os.path.basename(path),
        })

    pd.DataFrame(summary).to_csv(os.path.join(args.out, "regions_summary.csv"),
                                 index=False, encoding="utf-8-sig")
    info = {
        "software": f"SMILE MSI {__version__}",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "slide": slide.source, "session": slide.session_path,
        "n_pixels": int(ds.n_pixels), "pixel_size_um": ds.pixel_size_um,
        "polarity_declared": getattr(ds, "polarity", None), "identification_mode": mode,
        "extraction": {"ppm": api.ppm, "reduce": api.reduce, "norm_for_morans_i": api.norm},
        "picking": {"snr": args.snr, "prominence": args.prominence,
                    "min_rel_intensity": args.min_rel, "projection": args.projection,
                    "max_candidates": args.max_candidates, "baseline_subtracted": False},
        "gates": {"min_frequency": args.min_frequency, "presence_floor": PRESENT_FRAC,
                  "min_morans_i": args.min_morans, "collapse_isotopes": args.collapse_isotopes,
                  "isotope_tol_ppm": min(api.ppm, 15.0), "computed_over": "region pixels"},
        "identification": {"skipped": args.skip_annotation, "match_ppm": id_ppm,
                           "database": db_label},
        "fast_cube_loaded": getattr(ds, "_cube", None) is not None,
        "preprocessing": slide.data.get("preprocessing"),
        "regions": {n: int(masks[n].sum()) for n in names},
    }
    with open(os.path.join(args.out, "settings.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=str)
    say(f"done — {len(names)} region table(s), regions_summary.csv and settings.json in {args.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Per-region feature tables (S/N, intensity, frequency, Moran's I, lipids) "
                    "for comparing regions on one slide. See the top of this file for details.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--slide", help="session path or stem, sample name, or .imzML path")
    src.add_argument("--demo", action="store_true", help="use the synthetic demo slide")
    ap.add_argument("--list", action="store_true", help="list saved sessions and exit")
    ap.add_argument("--list-regions", action="store_true", help="list the slide's regions and exit")
    ap.add_argument("--region", action="append", default=[], help="region name; repeat for several")
    ap.add_argument("--all-regions", action="store_true", help="every region on the slide")
    ap.add_argument("--out", default="region_tables", help="output folder")
    ap.add_argument("--mode", choices=["positive", "negative"],
                    help="polarity for lipid identification (default: the session's setting)")
    ap.add_argument("--snr", type=float, default=1.5, help="min S/N (default 1.5)")
    ap.add_argument("--prominence", type=float, default=1.0, help="min prominence in S/N units; 0 = off")
    ap.add_argument("--min-rel", type=float, default=0.0,
                    help="min intensity as a fraction of the base peak (default 0)")
    ap.add_argument("--projection", choices=["mean", "max"], default="mean",
                    help="pick on the mean or the per-m/z maximum spectrum")
    ap.add_argument("--max-candidates", type=int, default=0, help="cap on candidates; 0 = no cap")
    ap.add_argument("--min-frequency", type=float, default=0.01,
                    help="fraction of region pixels an ion must be present in (0.01 = 1%%)")
    ap.add_argument("--min-morans", type=float, default=None,
                    help="drop ions whose region Moran's I is below this (default: no gate)")
    ap.add_argument("--no-collapse-isotopes", dest="collapse_isotopes", action="store_false",
                    help="keep M+1/M+2 isotope peaks")
    ap.add_argument("--skip-annotation", action="store_true",
                    help="detection metrics only, no lipid identification (faster)")
    ap.add_argument("--builtin-db", action="store_true",
                    help="use the built-in lipid database, not the one saved in the app's setup")
    ap.add_argument("--max-gb", type=float, default=8.0,
                    help="refuse an extraction larger than this many GB (default 8)")
    args = ap.parse_args(argv)

    if args.list:
        for row in headless.managed_sessions():
            missing = "" if row["source_exists"] else "   [data file not found]"
            print(f"{row.get('name') or os.path.basename(row['path'])}{missing}\n    {row.get('source')}")
        return 0
    if not (args.slide or args.demo):
        ap.error("pass --slide, --demo or --list")

    say("opening slide")
    try:
        slide = headless.open_demo() if args.demo else headless.open_slide(args.slide)
    except FileNotFoundError as e:
        raise SystemExit(str(e))
    available = slide.api.region_names()
    if args.list_regions:
        for n in available:
            print(f"{n}\t{int(slide.api.region(n).sum()):,} px")
        return 0

    names = available if args.all_regions else args.region
    if not names:
        ap.error("pass --region NAME (repeatable) or --all-regions; --list-regions shows the names")
    unknown = [n for n in names if n not in available]
    if unknown:
        raise SystemExit(f"unknown region(s) {unknown}. Available: {available}")
    export_regions(slide, names, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
