#!/usr/bin/env python3
"""Rebase stale absolute paths in a SMILE cohort after the data / project moved to a
new folder or a new machine.

A cohort JSON stores, per sample, two absolute paths captured on the machine where the
cohort was built:

  * source        -> the raw .imzML the embedding actually reads
  * session_path  -> the managed session JSON (under <home>/sessions)

When the project or the data is copied to another folder/PC those paths no longer exist,
so every sample fails to load and a pooled embedding dies with
"None of the N samples could be loaded ...". This script rewrites those paths to point at
the files on THIS device.

It is a DRY RUN by default: it shows what it would change and writes nothing. Add --apply
to actually rewrite (a .bak copy of the cohort JSON is made first).

------------------------------------------------------------------------------------------
USAGE  (run from anywhere; Windows examples — use your venv's python)

  # 1. See which cohorts exist on this machine
  .venv\\Scripts\\python rebase_cohort.py --list

  # 2. Dry-run a fix, searching a data folder for the raw .imzML files by name
  .venv\\Scripts\\python rebase_cohort.py --cohort "My Cohort" --data-root "D:\\MSI\\data"

  # 3. Apply it (multiple --data-root allowed; subfolders are searched recursively)
  .venv\\Scripts\\python rebase_cohort.py --cohort "My Cohort" --data-root "D:\\MSI\\data" --apply

  # ALTERNATIVE: if everything moved together under one root, a straight prefix swap
  .venv\\Scripts\\python rebase_cohort.py --cohort "My Cohort" ^
      --old "C:\\Users\\OldUser\\Documents\\SMILE-MSI\\data" ^
      --new "C:\\Users\\YourName\\Documents\\SMILE-MSI\\data" --apply

--cohort accepts a cohort *name* (resolved under <home>/cohorts) or a full path to the JSON.
session_path values are repointed automatically to this machine's <home>/sessions by
basename — you usually do NOT need --data-root for those, only for the raw .imzML sources.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys


def base_name(p: str) -> str:
    """Basename that works for both Windows (\\) and POSIX (/) stored paths, regardless of
    the OS running this script — os.path.basename only splits on the *local* separator."""
    return re.split(r"[\\/]", p)[-1] if p else p


def home_dir() -> str:
    base = os.environ.get("SMILE_MSI_HOME") or os.path.join(os.path.expanduser("~"), ".smile-msi")
    return base


def cohorts_dir() -> str:
    return os.path.join(home_dir(), "cohorts")


def sessions_dir() -> str:
    return os.path.join(home_dir(), "sessions")


def list_cohorts() -> None:
    d = cohorts_dir()
    if not os.path.isdir(d):
        print(f"No cohorts dir at {d}  (set SMILE_MSI_HOME if your home is elsewhere)")
        return
    rows = []
    for fn in sorted(os.listdir(d)):
        if not fn.lower().endswith(".json"):
            continue
        p = os.path.join(d, fn)
        try:
            data = json.load(open(p, encoding="utf-8"))
            n = len(data.get("samples") or [])
            name = data.get("name", os.path.splitext(fn)[0])
        except Exception as e:  # noqa: BLE001
            name, n = f"<unreadable: {e}>", "?"
        rows.append((name, n, p))
    if not rows:
        print(f"No cohort JSON files in {d}")
        return
    print(f"Cohorts in {d}:\n")
    for name, n, p in rows:
        print(f"  {name!r:40}  {n} samples   {p}")


def resolve_cohort_path(cohort: str) -> str:
    if os.path.isfile(cohort):
        return cohort
    # treat as a name -> <home>/cohorts/<sanitized>.json, else first case-insensitive match
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
    sys.exit(f"Could not find a cohort named {cohort!r}. Try --list, or pass the full .json path.")


def build_basename_index(roots: list[str], exts: tuple[str, ...]) -> dict[str, list[str]]:
    """basename(lower) -> [full paths] found anywhere under the given roots."""
    index: dict[str, list[str]] = {}
    for root in roots:
        if not os.path.isdir(root):
            print(f"  ! data-root not found, skipping: {root}")
            continue
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if f.lower().endswith(exts):
                    index.setdefault(f.lower(), []).append(os.path.join(dirpath, f))
    return index


def prefix_swap(path: str, old: str, new: str) -> str | None:
    """Case-insensitive prefix swap (Windows paths). Returns the new path or None."""
    if not path:
        return None
    op = os.path.normpath(old)
    pp = os.path.normpath(path)
    if pp.lower().startswith(op.lower()):
        tail = pp[len(op):].lstrip("\\/")
        return os.path.normpath(os.path.join(new, tail))
    return None


def relocate_source(src: str, args, src_index: dict[str, list[str]]) -> tuple[str | None, str]:
    """Return (new_path_or_None, note)."""
    if src and os.path.exists(src):
        return None, "ok (exists)"
    # 1) explicit prefix swap
    if args.old and args.new:
        cand = prefix_swap(src, args.old, args.new)
        if cand and os.path.exists(cand):
            return cand, "prefix-swap"
    # 2) basename search under --data-root
    if src:
        hits = src_index.get(base_name(src).lower(), [])
        if len(hits) == 1:
            return hits[0], "found by name"
        if len(hits) > 1:
            return None, f"AMBIGUOUS — {len(hits)} files named {base_name(src)!r} under data-root(s); pass --old/--new instead"
    return None, "NOT FOUND (give --data-root pointing at the .imzML, or --old/--new)"


def relocate_session(sp: str) -> tuple[str | None, str]:
    """Repoint a managed-session path to this machine's <home>/sessions by basename."""
    if sp and os.path.exists(sp):
        return None, "ok (exists)"
    if not sp:
        return None, "none"
    cand = os.path.join(sessions_dir(), base_name(sp))
    if os.path.exists(cand):
        return cand, "repointed to local sessions"
    return None, "NOT FOUND in local sessions (sample may be region-only)"


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebase stale paths in a SMILE cohort.")
    ap.add_argument("--list", action="store_true", help="list cohorts on this machine and exit")
    ap.add_argument("--cohort", help="cohort name or full path to its .json")
    ap.add_argument("--data-root", action="append", default=[],
                    help="folder to search recursively for the raw .imzML files (repeatable)")
    ap.add_argument("--old", help="old path prefix to swap (use with --new)")
    ap.add_argument("--new", help="new path prefix to swap in (use with --old)")
    ap.add_argument("--apply", action="store_true", help="write changes (default is a dry run)")
    ap.add_argument("--fix-sessions", action="store_true",
                    help="also rewrite the 'source' inside each referenced session JSON")
    args = ap.parse_args()

    if args.list or not args.cohort:
        list_cohorts()
        if not args.cohort:
            print("\nNow re-run with --cohort \"<name>\" and --data-root \"<folder with your imzML>\".")
        return

    path = resolve_cohort_path(args.cohort)
    print(f"Cohort: {path}")
    print(f"Home:   {home_dir()}")
    data = json.load(open(path, encoding="utf-8"))
    samples = data.get("samples") or []
    print(f"Samples: {len(samples)}\n")

    src_index = build_basename_index(args.data_root, (".imzml",)) if args.data_root else {}

    n_src_fixed = n_src_bad = n_sess_fixed = 0
    session_rewrites: list[tuple[str, str]] = []  # (session_json_path, new_source)

    for i, s in enumerate(samples):
        name = s.get("name") or f"sample {i}"
        src = s.get("source") or ""
        sp = s.get("session_path") or ""

        new_src, note = relocate_source(src, args, src_index)
        new_sp, snote = relocate_session(sp)

        tag = "OK" if (new_src is None and "ok" in note) else ("FIX" if new_src else "MISS")
        print(f"[{tag}] {name}")
        print(f"      source : {src or '<empty>'}")
        if new_src:
            print(f"           -> {new_src}   ({note})")
            n_src_fixed += 1
            if args.apply:
                s["source"] = new_src
            if args.fix_sessions:
                # remember to rewrite the session JSON's own 'source' to match
                sess_for_rewrite = new_sp or sp
                if sess_for_rewrite:
                    session_rewrites.append((sess_for_rewrite, new_src))
        elif "ok" not in note:
            print(f"           !! {note}")
            n_src_bad += 1
        if new_sp:
            print(f"      session-> {new_sp}   ({snote})")
            n_sess_fixed += 1
            if args.apply:
                s["session_path"] = new_sp
        print()

    print("-" * 70)
    print(f"sources: {n_src_fixed} to fix, {n_src_bad} still unresolved | "
          f"session_path: {n_sess_fixed} to repoint")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to save (a .bak is kept).")
        return

    if n_src_fixed == 0 and n_sess_fixed == 0:
        print("\nNothing to change.")
        return

    bak = path + ".bak"
    shutil.copy2(path, bak)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    print(f"\nWrote {path}\nBackup at {bak}")

    if args.fix_sessions and session_rewrites:
        print(f"\nRewriting 'source' in {len(session_rewrites)} session file(s)...")
        for sess_path, new_source in session_rewrites:
            if not os.path.isfile(sess_path):
                print(f"  ! missing session file: {sess_path}")
                continue
            try:
                sd = json.load(open(sess_path, encoding="utf-8"))
                sd["source"] = new_source
                shutil.copy2(sess_path, sess_path + ".bak")
                with open(sess_path + ".tmp", "w", encoding="utf-8") as f:
                    json.dump(sd, f, indent=2)
                os.replace(sess_path + ".tmp", sess_path)
                print(f"  fixed {sess_path}")
            except Exception as e:  # noqa: BLE001
                print(f"  ! failed {sess_path}: {e}")

    if n_src_bad:
        print(f"\n{n_src_bad} source(s) still unresolved — point --data-root at the folder that "
              "holds those .imzML files, or use --old/--new for a prefix swap.")


if __name__ == "__main__":
    main()
