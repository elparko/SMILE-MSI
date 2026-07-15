#!/usr/bin/env python3
"""Re-snap the m/z of already-saved ★ feature lists onto real peak apexes.

Cohort comparisons built before 2026-07 stored each feature's m/z as an *intensity-
weighted consensus centroid* (``cohort.consensus_targets``). That centroid is a synthetic
number that need not sit on any slide's real peak — so an annotated ion can land *between*
two peaks and its ion image renders blank (the "epi blanks" report). New comparisons now
emit a real apex, but lists saved earlier still carry the old centroids.

This script rewrites those stored m/z. For every named list in a session's
``feature_lists``, each feature's m/z is snapped to the nearest **real detected peak** in
that same session's saved ``peaks`` (its true local maxima), provided one lies within
``--tol-ppm``. Ions with no peak within tolerance are LEFT UNCHANGED and reported — those
are genuinely absent on that slide (e.g. depletion markers), not off-apex, and must not be
snapped into a neighbour. ``--deisotope`` additionally drops ¹³C M+1/M+2 satellites within
each list (same rule as ``isotopes.deisotope``: a peak is a satellite when a strictly more
intense peak sits n·1.00336 Da below it within tolerance).

It is a DRY RUN by default: it reports what it would change and writes nothing. Add --apply
to rewrite (a .bak copy of each session JSON is made first).

------------------------------------------------------------------------------------------
USAGE  (use your venv's python)

  # every session referenced by a cohort (dry run)
  .venv/bin/python scripts/resnap_feature_lists.py --cohort "My Cohort"

  # apply, also collapsing isotope satellites
  .venv/bin/python scripts/resnap_feature_lists.py --cohort "My Cohort" --deisotope --apply

  # specific session files, or every managed session on this machine
  .venv/bin/python scripts/resnap_feature_lists.py session_a.json session_b.json
  .venv/bin/python scripts/resnap_feature_lists.py --all --apply
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

import numpy as np

DELTA_C13 = 1.0033548   # 13C − 12C mass difference (mirrors isotopes.DELTA_C13)


def home_dir() -> str:
    return os.environ.get("SMILE_MSI_HOME") or os.path.join(os.path.expanduser("~"), ".smile-msi")


def sessions_dir() -> str:
    return os.path.join(home_dir(), "sessions")


def cohorts_dir() -> str:
    return os.path.join(home_dir(), "cohorts")


def base_name(p: str) -> str:
    """Basename that works for both Windows (\\) and POSIX (/) stored paths."""
    return re.split(r"[\\/]", p)[-1] if p else p


def resolve_cohort_path(cohort: str) -> str:
    if os.path.isfile(cohort):
        return cohort
    direct = os.path.join(cohorts_dir(), cohort if cohort.lower().endswith(".json") else cohort + ".json")
    if os.path.isfile(direct):
        return direct
    d = cohorts_dir()
    if os.path.isdir(d):
        for fn in os.listdir(d):
            if fn.lower().endswith(".json"):
                try:
                    data = json.load(open(os.path.join(d, fn), encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    continue
                if str(data.get("name", "")).lower() == cohort.lower():
                    return os.path.join(d, fn)
    sys.exit(f"Could not find a cohort named {cohort!r}. Pass the full .json path, or use --all.")


def sessions_from_cohort(path: str) -> list[str]:
    """The managed-session JSONs a cohort references, repointed to this machine's sessions
    dir by basename when the stored absolute path is stale (mirrors rebase_cohort.py)."""
    data = json.load(open(path, encoding="utf-8"))
    out = []
    for s in data.get("samples") or []:
        sp = s.get("session_path") or ""
        if sp and os.path.isfile(sp):
            out.append(sp)
        elif sp:
            cand = os.path.join(sessions_dir(), base_name(sp))
            if os.path.isfile(cand):
                out.append(cand)
    return out


def _peak_axis(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """(mz_sorted, intensity_sorted) of a session's real detected peaks — the snap targets."""
    peaks = data.get("peaks") or []
    if not peaks:
        return np.empty(0), np.empty(0)
    mz = np.array([float(p.get("mz", 0.0)) for p in peaks], dtype=float)
    inten = np.array([float(p.get("intensity", p.get("rel_intensity", 1.0))) for p in peaks], dtype=float)
    order = np.argsort(mz)
    return mz[order], inten[order]


def _drop_isotopes(feats: list[dict], inten_of, tol_ppm: float, max_iso: int = 2) -> list[dict]:
    """Drop ¹³C M+1/M+2 satellites within one list (same rule as isotopes.deisotope)."""
    mzs = [float(f.get("mz", 0.0)) for f in feats]
    ints = [inten_of(m) for m in mzs]
    keep = []
    for i, f in enumerate(feats):
        m, v, iso = mzs[i], ints[i], False
        win = m * tol_ppm / 1e6
        for n in range(1, max_iso + 1):
            below = m - n * DELTA_C13
            if any(abs(mm - below) <= win and ints[k] > v for k, mm in enumerate(mzs)):
                iso = True
                break
        if not iso:
            keep.append(f)
    return keep


def resnap_session(data: dict, tol_ppm: float, deisotope: bool) -> dict:
    """Snap every feature-list m/z to the nearest real apex; optionally drop isotopes.
    Mutates ``data`` in place. Returns per-session stats."""
    pmz, pint = _peak_axis(data)
    lists = data.get("feature_lists") or {}
    st = {"lists": len(lists), "snapped": 0, "unchanged": 0, "no_apex": 0,
          "dropped_iso": 0, "max_shift_ppm": 0.0, "no_peaks": pmz.size == 0}

    def inten_of(mz: float) -> float:
        if pmz.size == 0:
            return 1.0
        j = int(np.argmin(np.abs(pmz - mz)))
        return float(pint[j]) if abs(pmz[j] - mz) / mz * 1e6 <= tol_ppm else 0.0

    for name, feats in lists.items():
        for f in feats:
            mz = float(f.get("mz", 0.0))
            if pmz.size == 0 or mz <= 0:
                st["no_apex"] += 1
                continue
            j = int(np.argmin(np.abs(pmz - mz)))
            apex = float(pmz[j])
            ppm = abs(apex - mz) / mz * 1e6
            if ppm > tol_ppm:                       # no real peak in range — a true absence; leave it
                st["no_apex"] += 1
                continue
            new = round(apex, 4)
            if abs(new - round(mz, 4)) > 1e-9:
                f["mz"] = new
                st["snapped"] += 1
                st["max_shift_ppm"] = max(st["max_shift_ppm"], ppm)
            else:
                st["unchanged"] += 1
        if deisotope:
            kept = _drop_isotopes(feats, inten_of, tol_ppm)
            st["dropped_iso"] += len(feats) - len(kept)
            lists[name] = kept
    return st


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-snap saved ★ feature-list m/z onto real apexes.")
    ap.add_argument("sessions", nargs="*", help="session JSON file(s) to fix")
    ap.add_argument("--cohort", help="cohort name or .json path — fix every session it references")
    ap.add_argument("--all", action="store_true", help="fix every session under <home>/sessions")
    ap.add_argument("--tol-ppm", type=float, default=20.0,
                    help="snap radius = cohort clustering tol (default 20, = CONSENSUS_TOL_PPM)")
    ap.add_argument("--deisotope", action="store_true", help="also drop 13C M+1/M+2 satellites per list")
    ap.add_argument("--apply", action="store_true", help="write changes (default is a dry run)")
    args = ap.parse_args()

    paths = list(args.sessions)
    if args.cohort:
        paths += sessions_from_cohort(resolve_cohort_path(args.cohort))
    if args.all:
        d = sessions_dir()
        if os.path.isdir(d):
            paths += [os.path.join(d, fn) for fn in sorted(os.listdir(d)) if fn.lower().endswith(".json")]
    paths = list(dict.fromkeys(os.path.abspath(p) for p in paths))   # de-dup, keep order
    if not paths:
        sys.exit("No sessions given. Pass session .json paths, --cohort <name>, or --all.")

    print(f"Snap radius: {args.tol_ppm:g} ppm | deisotope: {args.deisotope} | "
          f"{'APPLY' if args.apply else 'DRY RUN'}\n")
    tot = {"snapped": 0, "no_apex": 0, "dropped_iso": 0, "changed_sessions": 0}
    for p in paths:
        if not os.path.isfile(p):
            print(f"  ! missing: {p}")
            continue
        try:
            data = json.load(open(p, encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"  ! unreadable ({e}): {p}")
            continue
        st = resnap_session(data, args.tol_ppm, args.deisotope)
        touched = st["snapped"] + st["dropped_iso"]
        flag = "no peaks in session" if st["no_peaks"] else ""
        print(f"[{'FIX' if touched else 'ok '}] {base_name(p)}  "
              f"lists={st['lists']} snapped={st['snapped']} "
              f"(max {st['max_shift_ppm']:.1f} ppm) left-as-absent={st['no_apex']} "
              f"dropped-iso={st['dropped_iso']} {flag}")
        tot["snapped"] += st["snapped"]
        tot["no_apex"] += st["no_apex"]
        tot["dropped_iso"] += st["dropped_iso"]
        if touched and args.apply:
            shutil.copy2(p, p + ".bak")
            tmp = p + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, p)
            tot["changed_sessions"] += 1

    print("\n" + "-" * 70)
    print(f"total: {tot['snapped']} snapped to apex, {tot['dropped_iso']} isotope(s) dropped, "
          f"{tot['no_apex']} left as true absence")
    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to save (a .bak is kept per session).")
    else:
        print(f"\nWrote {tot['changed_sessions']} session(s); a .bak was kept for each.")


if __name__ == "__main__":
    main()
