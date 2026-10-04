#!/usr/bin/env python3
"""Export a cohort's **per-sample** feature-intensity matrix — one column (or row) per
individual sample, not the group means.

The Cohort tab's "Comparison table (CSV)" writes the *collapsed* differential once you've
run a Group-A-vs-B comparison: one ``mean_A`` / ``mean_B`` per feature, no per-sample
values. That group-mean table is exactly ``cohort.group_comparison(tbl, ...)``. The thing
it collapses — ``tbl`` — is the per-sample matrix, and that's what this script writes out.

It reads only the saved **session peak lists** (the same source the Cohort tab uses), so it
needs the per-sample session JSONs but NOT the raw .imzML cubes — it runs fast and works
even on a machine where the raw data paths are stale.

Defaults mirror the Cohort tab exactly (value=rel_intensity, normalize=median, match=20 ppm,
consensus feature axis at min-prevalence=0.5, apex centres, deisotoped), so the per-sample
values reconcile with a differential exported from the GUI with those same settings. Change
any of them with the flags below to match whatever settings your differential used.

------------------------------------------------------------------------------------------
USAGE  (run with the project venv so `import smile_msi` works)

  # 1. List the cohorts saved on this machine
  .venv/bin/python scripts/export_sample_matrix.py --list
  #   Windows:  .venv\\Scripts\\python scripts\\export_sample_matrix.py --list

  # 2. Export a cohort's per-sample matrix (defaults = Cohort-tab defaults)
  .venv/bin/python scripts/export_sample_matrix.py --cohort "My Cohort"

  # 3. Match the settings your differential actually used, e.g. no normalization + raw intensity
  .venv/bin/python scripts/export_sample_matrix.py --cohort "My Cohort" \
      --value intensity --normalize none --tol 20 --min-prevalence 0.5

  # 4. Use a fixed feature list (one m/z per line, or a saved list .json/.csv) instead of consensus
  .venv/bin/python scripts/export_sample_matrix.py --cohort "My Cohort" --features my_targets.csv

Two files are written next to --out (default: <cohort>_matrix):
  <out>_by_sample.csv          tidy — one ROW per sample: sample, group, then one column per
                               feature (mz_xxxx). The format every stats tool (R/pandas/Prism)
                               ingests directly; group-by and per-sample ratios are one step.
  <out>_features_by_sample.csv literal "features (rows) x samples (columns)": first columns are
                               mz (and lipid, with --annotate), then one column per sample. Row 1
                               of the header carries each sample's normal/trt group label.

The console prints per-group n and, for the first few features, each group's mean — so you can
eyeball that the group means equal the differential table you already shared.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def home_dir() -> str:
    return os.environ.get("SMILE_MSI_HOME") or os.path.join(os.path.expanduser("~"), ".smile-msi")


def cohorts_dir() -> str:
    return os.path.join(home_dir(), "cohorts")


def list_cohorts() -> None:
    d = cohorts_dir()
    if not os.path.isdir(d):
        print(f"No cohorts dir at {d}  (set SMILE_MSI_HOME if your home is elsewhere)")
        return
    print(f"Cohorts in {d}:\n")
    for fn in sorted(os.listdir(d)):
        if not fn.lower().endswith(".json"):
            continue
        p = os.path.join(d, fn)
        try:
            data = json.load(open(p, encoding="utf-8"))
            samples = data.get("samples") or []
            groups: dict[str, int] = {}
            for s in samples:
                g = (s.get("group", "") or "(ungrouped)")
                groups[g] = groups.get(g, 0) + 1
            gtxt = ", ".join(f"{g}={n}" for g, n in groups.items())
            print(f"  {data.get('name', fn)!r:34} {len(samples)} samples   [{gtxt}]")
        except Exception as e:  # noqa: BLE001
            print(f"  {fn} <unreadable: {e}>")


def resolve_cohort_path(name: str) -> str:
    if os.path.isfile(name):
        return name
    direct = os.path.join(cohorts_dir(), name if name.lower().endswith(".json") else name + ".json")
    if os.path.isfile(direct):
        return direct
    d = cohorts_dir()
    if os.path.isdir(d):
        for fn in os.listdir(d):
            if fn.lower().endswith(".json"):
                try:
                    if str(json.load(open(os.path.join(d, fn), encoding="utf-8")).get("name", "")).lower() == name.lower():
                        return os.path.join(d, fn)
                except Exception:  # noqa: BLE001
                    continue
    sys.exit(f"Could not find a cohort named {name!r}. Try --list or pass the full .json path.")


def load_target_mzs(path: str) -> list[float]:
    """Read a fixed feature axis: a saved feature-list .json (list of m/z, or peaks with an
    'mz'/'m/z' field), or a plain text/CSV with one m/z per line (first numeric token per row)."""
    if not os.path.isfile(path):
        sys.exit(f"--features file not found: {path}")
    mzs: list[float] = []
    if path.lower().endswith(".json"):
        data = json.load(open(path, encoding="utf-8"))
        seq = data.get("peaks", data) if isinstance(data, dict) else data
        for item in seq:
            if isinstance(item, dict):
                v = item.get("mz", item.get("m/z"))
            else:
                v = item
            try:
                mzs.append(float(v))
            except (TypeError, ValueError):
                continue
    else:
        import re
        for line in open(path, encoding="utf-8"):
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
            if m:
                mzs.append(float(m.group(0)))
    mzs = sorted({round(m, 4) for m in mzs if m > 0})
    if not mzs:
        sys.exit(f"No usable m/z values parsed from {path}")
    return mzs


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a cohort's per-sample feature matrix.")
    ap.add_argument("--list", action="store_true", help="list cohorts on this machine and exit")
    ap.add_argument("--cohort", help="cohort name or full path to its .json")
    ap.add_argument("--out", help="output path prefix (default: <cohort>_matrix)")
    ap.add_argument("--value", choices=["rel_intensity", "intensity"], default="rel_intensity",
                    help="per-peak value to read (Cohort-tab default: rel_intensity)")
    ap.add_argument("--normalize", choices=["median", "sum", "none"], default="median",
                    help="cross-slide per-sample normalization (Cohort-tab default: median)")
    ap.add_argument("--tol", type=float, default=20.0, help="feature match tolerance, ppm (default 20)")
    ap.add_argument("--min-prevalence", type=float, default=0.5,
                    help="consensus only: keep ions seen in >= this fraction of samples (default 0.5)")
    ap.add_argument("--missing", choices=["nan", "zero"], default="nan",
                    help="absent ion -> NaN (default, matches the GUI) or 0.0 (below-detection)")
    ap.add_argument("--features", help="fixed feature list (.json / one-m/z-per-line) instead of consensus")
    ap.add_argument("--annotate", action="store_true",
                    help="best-effort lipid IDs on the features-by-sample file (needs the lipid DB)")
    args = ap.parse_args()

    if args.list or not args.cohort:
        list_cohorts()
        if not args.cohort:
            print('\nRe-run with --cohort "<name>".')
        return

    try:
        from smile_msi import cohort as cohort_engine
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Could not import smile_msi ({e}). Run with the project venv, e.g.: "
                 ".venv/bin/python scripts/export_sample_matrix.py ...")
    import numpy as np

    path = resolve_cohort_path(args.cohort)
    c = cohort_engine.Cohort.load(path)
    refs = list(c.samples)
    print(f"Cohort: {path}")
    print(f"Samples: {len(refs)}")
    if len(refs) < 1:
        sys.exit("Cohort has no samples.")

    sessions: dict = {}   # one parse per session JSON, shared by consensus + the table build
    if args.features:
        targets = load_target_mzs(args.features)
        print(f"Feature axis: {len(targets)} fixed m/z from {args.features}")
    else:
        targets = cohort_engine.consensus_targets(
            refs, tol_ppm=cohort_engine.CONSENSUS_TOL_PPM, min_prevalence=args.min_prevalence,
            value=args.value, sessions=sessions)
        print(f"Feature axis: {len(targets)} consensus ions "
              f"(min-prevalence {args.min_prevalence}, apex, deisotoped)")
    if not targets:
        sys.exit("No features on the axis — lower --min-prevalence or pass --features.")

    tbl = cohort_engine.batch_feature_table(
        refs, targets, tol_ppm=args.tol, value=args.value,
        normalize=args.normalize, missing=args.missing, sessions=sessions)

    skipped = list(tbl.attrs.get("skipped") or [])
    if skipped:
        print(f"\nSkipped {len(skipped)} sample(s):")
        for s in skipped:
            print(f"  - {s.get('name')}: {s.get('reason')}")
    if tbl.empty:
        sys.exit("No samples with saved peaks — nothing to export. (See skipped list above.)")

    feat_cols = [col for col in tbl.columns if col != "group"]
    out_prefix = args.out or os.path.splitext(os.path.basename(path))[0] + "_matrix"

    # --- (1) tidy: one row per sample ------------------------------------------------- #
    tidy = tbl.reset_index().rename(columns={"index": "sample"})
    tidy_path = out_prefix + "_by_sample.csv"
    tidy.to_csv(tidy_path, index=False, encoding="utf-8-sig")

    # --- (2) features as rows, samples as columns ------------------------------------- #
    wide = tbl[feat_cols].T.copy()                 # index = mz_xxxx, columns = sample names
    wide.insert(0, "mz", [float(x.split("_", 1)[1]) for x in wide.index])
    if args.annotate:
        try:
            from smile_msi import pipeline
            ann = pipeline.annotate_df(wide.reset_index(drop=True), "mz", ppm=args.tol)
            for label_col in ("annotation", "lipid", "label", "name"):
                if label_col in ann.columns:
                    wide.insert(1, "lipid", list(ann[label_col]))
                    break
        except Exception as e:  # noqa: BLE001
            print(f"(annotation skipped: {e})")
    # header row 0 = group label for each sample column ("" for the mz/lipid label columns)
    lead = [col for col in wide.columns if col not in tbl.index]   # mz (+ lipid)
    group_row = {col: "" for col in lead}
    group_row.update({nm: (tbl.loc[nm, "group"] or "") for nm in tbl.index})
    import pandas as pd
    wide_out = pd.concat([pd.DataFrame([group_row], index=["group"]), wide])
    wide_path = out_prefix + "_features_by_sample.csv"
    wide_out.to_csv(wide_path, index=False, encoding="utf-8-sig")

    # --- console cross-check: per-group n + first few feature means -------------------- #
    print(f"\nWrote:\n  {tidy_path}\n  {wide_path}")
    by_group: dict[str, int] = {}
    for g in tbl["group"]:
        key = g or "(ungrouped)"
        by_group[key] = by_group.get(key, 0) + 1
    print(f"\nSamples per group: " + ", ".join(f"{g}: {n}" for g, n in by_group.items()))
    groups = [g for g in dict.fromkeys(tbl["group"]) if g]
    if len(groups) >= 2:
        print(f"\nCross-check (should match your differential's means) — first 8 features:")
        head = f"  {'m/z':>10}  " + "  ".join(f"mean[{g}]" for g in groups)
        print(head)
        M = tbl[feat_cols].to_numpy(dtype=float)
        gvec = tbl["group"].to_numpy()
        for j, col in enumerate(feat_cols[:8]):
            mz = float(col.split("_", 1)[1])
            means = []
            for g in groups:
                v = M[gvec == g, j]
                v = v[~np.isnan(v)]
                means.append(f"{v.mean():.4g}" if v.size else "n/a")
            print(f"  {mz:>10.4f}  " + "  ".join(f"{m:>8}" for m in means))
    else:
        print("\n(Only one group present — assign normal/trt labels in the Samples panel "
              "for a group cross-check.)")


if __name__ == "__main__":
    main()
