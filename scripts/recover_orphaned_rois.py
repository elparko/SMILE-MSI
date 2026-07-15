#!/usr/bin/env python3
"""Recover ROIs stranded by the managed-session fingerprint-drift bug.

Before the stable-key fix, a slide's managed session was named after a hash of its FULL
content fingerprint, which included the m/z range and grid width/height. Both can shift
between loads of the same slide (a misread spectrum flips the m/z bounds; an orientation
change transposes width/height), so the same slide got re-keyed into a *new* session file
on such a load and its ROIs were stranded in the old file — the "I added/removed a sample
and had to redraw all my ROIs" symptom.

This tool finds every group of managed sessions that describe the SAME slide (same pixel
count + coordinate hash), and consolidates each group into its newest file with the UNION of
all their named regions, so no ROI is lost and the most recent analysis is kept as the base.
The other files in the group are moved aside into a backup folder. Everything is backed up
first, and the tool is dry-run by default — pass --apply to actually write.

    # see what would change (safe):
    python scripts/recover_orphaned_rois.py
    # do it (backs up to ~/.smile-msi/sessions/_roi_recovery_backup first):
    python scripts/recover_orphaned_rois.py --apply
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

# allow running straight from a checkout without installing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smile_msi import library, session  # noqa: E402


def _load(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _identity(data: dict) -> tuple:
    """Stable identity of the slide a session belongs to: (basename, stable-fingerprint).
    Falls back to (basename, 'n<pixels>') when the session predates fingerprints."""
    src = data.get("source", "")
    base = library.dataset_key(src)
    fp = data.get("dataset_fingerprint")
    if fp:
        return (base, session.stable_fingerprint(fp))
    return (base, f"n{data.get('n_pixels')}")


def _region_names(data: dict) -> list[str]:
    return [str(r.get("name", "")) for r in (data.get("named_regions") or [])]


def _sidecars(json_path: str) -> list[str]:
    """Existing cube/parse sidecars that belong beside a managed session file."""
    stem = json_path[:-5] if json_path.endswith(".json") else json_path
    cands = [stem + ".cache.npz", stem + ".cube.zarr",
             session.cube_sidecar_path(json_path), session.cube_zarr_path(json_path)]
    seen, out = set(), []
    for c in cands:
        if c not in seen and os.path.exists(c):
            seen.add(c)
            out.append(c)
    return out


def _merge_regions(files: list[dict]) -> tuple[list[dict], list[tuple[str, str]]]:
    """Union of named_regions across ``files`` (already ordered newest→oldest). Keeps the
    newest version of a region on a name collision; adds older-only regions. Returns the
    merged list and a list of (region_name, from_filename) recovered from older files."""
    merged, seen, recovered = [], set(), []
    for f in files:
        for r in (f["data"].get("named_regions") or []):
            name = str(r.get("name", ""))
            if name in seen:
                continue
            seen.add(name)
            merged.append(r)
            if f is not files[0]:
                recovered.append((name, os.path.basename(f["path"])))
    return merged, recovered


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="actually consolidate (default: dry-run report only)")
    ap.add_argument("--sessions-dir", default=None,
                    help="managed-sessions directory (default: the app's own)")
    args = ap.parse_args()

    sdir = args.sessions_dir or session.sessions_dir()
    backup = os.path.join(sdir, "_roi_recovery_backup")
    print(f"Scanning managed sessions in: {sdir}\n")

    files = []
    for fn in sorted(os.listdir(sdir)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(sdir, fn)
        data = _load(path)
        if data is None:
            continue
        files.append({"path": path, "data": data, "mtime": os.stat(path).st_mtime})

    groups: dict[tuple, list[dict]] = {}
    for f in files:
        groups.setdefault(_identity(f["data"]), []).append(f)

    consolidations = [g for g in groups.values() if len(g) > 1]
    if not consolidations:
        print("No multi-file slide groups found — nothing to consolidate. "
              "Every slide already has a single session file.")
        return 0

    total_recovered = 0
    for group in consolidations:
        group.sort(key=lambda f: f["mtime"], reverse=True)   # newest first = base/keeper
        keeper = group[0]
        merged, recovered = _merge_regions(group)
        base_n = len(keeper["data"].get("named_regions") or [])
        src = keeper["data"].get("source", "")
        print(f"● {os.path.basename(src) or '(unknown source)'}")
        for f in group:
            tag = "  KEEP (newest)" if f is keeper else "  archive"
            print(f"    {os.path.basename(f['path']):40s} regions={len(_region_names(f['data'])):3d}{tag}")
        print(f"    → merged region set: {base_n} in keeper + {len(recovered)} recovered "
              f"= {len(merged)} total")
        for name, frm in recovered:
            print(f"        + '{name}'  (recovered from {frm})")
        total_recovered += len(recovered)

        if args.apply:
            os.makedirs(backup, exist_ok=True)
            # back up every original file (and archived sidecars) before touching anything
            for f in group:
                shutil.copy2(f["path"], os.path.join(backup, os.path.basename(f["path"])))
            keeper["data"]["named_regions"] = merged
            session.save_session(keeper["path"], keeper["data"])
            for f in group[1:]:
                for extra in _sidecars(f["path"]):
                    shutil.move(extra, os.path.join(backup, os.path.basename(extra)))
                shutil.move(f["path"], os.path.join(backup, os.path.basename(f["path"])))
        print()

    if args.apply:
        print(f"Done. Recovered {total_recovered} region(s) across {len(consolidations)} slide(s).")
        print(f"Originals backed up to: {backup}")
    else:
        print(f"DRY RUN — would recover {total_recovered} region(s) across "
              f"{len(consolidations)} slide(s). Re-run with --apply to write "
              f"(originals are backed up first).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
