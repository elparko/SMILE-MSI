#!/usr/bin/env python3
"""Convert a LIPID MAPS Structure Database (LMSD) ``.sdf`` download into an import-ready CSV.

LIPID MAPS distributes the LMSD as an SDF (structure file). SMILE MSI's *Data ▸ Import lipid
database…* wants a simple table with a **name**, a **FORMULA**, and (optionally) a class column,
so this script pulls just those data fields out of the SDF — no chemistry toolkit needed — and
writes ``ABBREVIATION,FORMULA,MAIN_CLASS`` columns that the importer auto-detects.

It can also **filter** while converting, which is worth doing: the full LMSD is ~48k structures
and a smaller, tissue-relevant slice annotates with far less isobaric ambiguity.

  * ``--categories`` keeps only rows whose CATEGORY / MAIN_CLASS matches (comma-separated,
    case-insensitive substring), e.g. ``--categories "Glycerophospholipids,Sphingolipids,Sterol,Fatty acyls"``.
  * ``--min-mass`` / ``--max-mass`` keep only rows in a monoisotopic-mass window (e.g. the m/z
    span your data covers), computed from the formula.
  * By default only rows whose formula uses the elements this engine can mass
    (H, C, N, O, P, S, Na, K, Cl) are kept — pass ``--keep-all-elements`` to keep everything
    (those extra rows are skipped again at import time, so it only bloats the CSV).

------------------------------------------------------------------------------------------
USAGE  (use your venv's python)

  # whole LMSD → CSV
  .venv/bin/python scripts/lmsd_to_csv.py LMSD.sdf -o lmsd.csv

  # only the lipid categories you expect, in your data's mass range (recommended)
  .venv/bin/python scripts/lmsd_to_csv.py LMSD.sdf -o lmsd_neg.csv \
      --categories "Glycerophospholipids,Sphingolipids,Sterol Lipids,Fatty Acyls" \
      --min-mass 250 --max-mass 1100

Then in the app: Data ▸ Import lipid database… → pick the CSV → choose Merge → rebuild the
feature list.
------------------------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import csv
import sys

from smile_msi.lipiddb import parse_sdf   # shared SDF field parser (also used by the setup wizard)

# Which SDF data field to use as the display name, best first (LMSD carries several).
_NAME_FIELDS = ["ABBREVIATION", "COMMON_NAME", "NAME", "SYSTEMATIC_NAME", "LM_ID"]
_FORMULA_FIELDS = ["FORMULA", "MOLECULAR_FORMULA"]
_CLASS_FIELDS = ["MAIN_CLASS", "CATEGORY", "SUB_CLASS"]


def _first(rec: dict, keys) -> str:
    for k in keys:
        v = (rec.get(k) or "").strip()
        if v and v.lower() not in ("n/a", "na", "none", "-"):
            return v
    return ""


def to_rows(records, *, categories=None, min_mass=None, max_mass=None,
            keep_all_elements=False) -> tuple[list[dict], dict]:
    """Turn parsed SDF records into import-ready ``{ABBREVIATION, FORMULA, MAIN_CLASS}`` rows,
    applying the optional category / mass-window / element filters. Returns ``(rows, counts)``
    where ``counts`` tallies what was dropped and why, so the CLI can report coverage."""
    cats = [c.strip().lower() for c in categories] if categories else None
    need_mass = (min_mass is not None) or (max_mass is not None) or (not keep_all_elements)
    formula_mass = elements = parse_formula = None
    if need_mass:
        # only import the engine's chemistry when a filter needs it (keeps the script usable
        # for a no-filter dump even outside the repo venv)
        from smile_msi.lipiddb import parse_formula
        from smile_msi.masses import ELEMENTS as elements
        from smile_msi.masses import formula_mass

    rows: list[dict] = []
    counts = {"records": len(records), "no_name_or_formula": 0, "bad_formula": 0,
              "off_element": 0, "off_mass": 0, "off_category": 0, "kept": 0}
    for rec in records:
        name = _first(rec, _NAME_FIELDS)
        formula = _first(rec, _FORMULA_FIELDS)
        cls = _first(rec, _CLASS_FIELDS)
        if not name or not formula:
            counts["no_name_or_formula"] += 1
            continue
        if cats is not None:
            hay = f"{rec.get('CATEGORY', '')} {rec.get('MAIN_CLASS', '')} {rec.get('SUB_CLASS', '')}".lower()
            if not any(c in hay for c in cats):
                counts["off_category"] += 1
                continue
        if need_mass:
            f = parse_formula(formula)
            if not f:
                counts["bad_formula"] += 1
                continue
            if any(el not in elements for el in f):
                if not keep_all_elements:
                    counts["off_element"] += 1
                    continue
            elif (min_mass is not None) or (max_mass is not None):
                m = formula_mass(f)
                if (min_mass is not None and m < min_mass) or (max_mass is not None and m > max_mass):
                    counts["off_mass"] += 1
                    continue
        rows.append({"ABBREVIATION": name, "FORMULA": formula, "MAIN_CLASS": cls})
        counts["kept"] += 1
    return rows, counts


def write_csv(rows, path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["ABBREVIATION", "FORMULA", "MAIN_CLASS"])
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Convert a LIPID MAPS LMSD .sdf into an import-ready CSV.")
    ap.add_argument("sdf", help="path to the LMSD .sdf download")
    ap.add_argument("-o", "--out", required=True, help="output CSV path")
    ap.add_argument("--categories", default="",
                    help="comma-separated CATEGORY/MAIN_CLASS substrings to keep (case-insensitive)")
    ap.add_argument("--min-mass", type=float, default=None, help="min monoisotopic mass (Da)")
    ap.add_argument("--max-mass", type=float, default=None, help="max monoisotopic mass (Da)")
    ap.add_argument("--keep-all-elements", action="store_true",
                    help="keep rows whose formula has elements outside H,C,N,O,P,S,Na,K,Cl "
                         "(they are skipped again at import time)")
    args = ap.parse_args(argv)

    try:
        with open(args.sdf, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError as e:
        print(f"error: cannot read {args.sdf}: {e}", file=sys.stderr)
        return 2

    records = parse_sdf(text)
    cats = [c for c in (s.strip() for s in args.categories.split(",")) if c] or None
    rows, counts = to_rows(records, categories=cats, min_mass=args.min_mass,
                           max_mass=args.max_mass, keep_all_elements=args.keep_all_elements)
    if not rows:
        print(f"no rows kept from {counts['records']} records "
              f"(dropped: {counts}) — check the SDF / filters.", file=sys.stderr)
        return 1
    write_csv(rows, args.out)
    print(f"wrote {counts['kept']} lipids to {args.out} "
          f"(from {counts['records']} records; dropped "
          f"{counts['off_category']} off-category, {counts['off_mass']} off-mass, "
          f"{counts['off_element']} off-element, "
          f"{counts['no_name_or_formula'] + counts['bad_formula']} missing name/formula).")
    print("Import it in the app: Data ▸ Import lipid database… → Merge → rebuild the feature list.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
