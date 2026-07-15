#!/usr/bin/env python3
"""Explain, per sample, EXACTLY why a cohort sample does or doesn't load for a pooled
embedding — replays SectionLoader.load's checks so "None of the N samples could be loaded"
stops being a black box.

Run with the project's venv (needs `import smile_msi`):
  .venv\\Scripts\\python scripts\\diagnose_cohort.py --list
  .venv\\Scripts\\python scripts\\diagnose_cohort.py --cohort "<name>"

For each sample it prints one of:
  OK whole-slide / OK region 'X' -> N px
  FAIL: source not an imzML / source MISSING / region session MISSING / region session unreadable
  FAIL: region 'X' not in session.named_regions   (region never saved into the session)
  FAIL: region 'X' present but 0 px (mask=…, segments=…, seg.labels=…)
"""
from __future__ import annotations

import argparse
import os
import sys


def home_dir() -> str:
    return os.environ.get("SMILE_MSI_HOME") or os.path.join(os.path.expanduser("~"), ".smile-msi")


def cohorts_dir() -> str:
    return os.path.join(home_dir(), "cohorts")


def list_cohorts() -> None:
    import json
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
            print(f"  {data.get('name', fn)!r:40} {len(data.get('samples') or [])} samples   {p}")
        except Exception as e:  # noqa: BLE001
            print(f"  {fn} <unreadable: {e}>")


def resolve_cohort_path(name: str) -> str:
    import json
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


def diagnose_sample(s, session, region_pixels) -> str:
    """Mirror SectionLoader.load + region_pixels to report why a sample loads or not."""
    src = (getattr(s, "source", "") or "")
    if not src.lower().endswith(".imzml"):
        return f"FAIL: source is not an imzML ({src or '<empty>'}) — can't pool pixels"
    if not os.path.exists(src):
        return f"FAIL: source MISSING on this machine: {src}"
    region = (getattr(s, "region", "") or "")
    if not region:
        return "OK  whole-slide"
    sp = (getattr(s, "session_path", "") or "")
    if not sp or not os.path.exists(sp):
        return f"FAIL: region '{region}' — session_path MISSING: {sp or '<empty>'}"
    try:
        data = session.load_session(sp)
    except Exception as e:  # noqa: BLE001
        return f"FAIL: region '{region}' — session unreadable: {e}"
    pix = region_pixels(data, region)
    if pix is not None and len(pix) > 0:
        return f"OK  region '{region}' -> {len(pix):,} px"
    nr = next((r for r in (data.get("named_regions") or []) if r.get("name") == region), None)
    if nr is None:
        names = [r.get("name") for r in (data.get("named_regions") or [])]
        return (f"FAIL: region '{region}' NOT in session.named_regions "
                f"(session has: {names or 'none'}) — region never saved into the session")
    has_mask = bool(nr.get("mask"))
    has_segs = bool(nr.get("segments"))
    has_labels = bool((data.get("segmentation") or {}).get("labels"))
    return (f"FAIL: region '{region}' present but 0 px "
            f"(mask={has_mask}, segments={has_segs}, seg.labels={has_labels}) — pixels unrecoverable")


def main() -> None:
    ap = argparse.ArgumentParser(description="Explain why each cohort sample loads (or not).")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--cohort", help="cohort name or full path to its .json")
    args = ap.parse_args()

    if args.list or not args.cohort:
        list_cohorts()
        if not args.cohort:
            print('\nRe-run with --cohort "<name>".')
        return

    try:
        from smile_msi import cohort as cohort_engine, session
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Could not import smile_msi ({e}). Run with the project venv: "
                 ".venv\\Scripts\\python scripts\\diagnose_cohort.py ...")

    path = resolve_cohort_path(args.cohort)
    c = cohort_engine.Cohort.load(path)
    print(f"Cohort: {path}")
    print(f"Home:   {home_dir()}")
    print(f"Samples: {len(c.samples)}\n")

    n_ok = n_fail = 0
    region_seen: dict[str, int] = {}
    for s in c.samples:
        verdict = diagnose_sample(s, session, cohort_engine.region_pixels)
        if verdict.startswith("OK"):
            n_ok += 1
        else:
            n_fail += 1
        print(f"  [{getattr(s, 'name', '?')}]")
        print(f"     region={getattr(s, 'region', '') or '(whole slide)'}")
        print(f"     source={getattr(s, 'source', '') or '<empty>'}")
        print(f"     session_path={getattr(s, 'session_path', '') or '<empty>'}")
        print(f"     -> {verdict}\n")

    print("-" * 70)
    print(f"{n_ok} load OK, {n_fail} fail.  ({'pooled embedding will work' if n_ok >= 2 else 'embedding will raise None-loaded'})")


if __name__ == "__main__":
    main()
