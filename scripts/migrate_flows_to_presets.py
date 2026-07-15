#!/usr/bin/env python
"""Convert saved **Flow designer** presets (``~/.smile-msi/flows/*.json``) into **scripting**
presets (``~/.smile-msi/workflows/*.json``) so retiring the Flow designer is lossless.

For every flow it emits a :class:`smile_msi.scripting.Workflow` whose ``code`` reproduces the
recipe as a sequence of ``s.run_analysis("<step_id>", **params)`` calls — one per *enabled*
step, in order, carrying that step's EXACT stored params (nothing silently re-defaulted; the
registry backfills any omitted param at run time, identically to the Flow runner). The flow's
region scope is honoured the way ``flowdialog``'s runner interprets it:

    ""/None  → whole slide          (no mask passed)
    "@grouped" → the UNION of every grouped ROI's pixels, applied to every step
                 (this is exactly what flowdialog ``_scope_mask()`` builds — a single union
                 mask, NOT a per-ROI loop)
    "<name>" → restrict to that one named region

Non-destructive and idempotent: it never deletes or edits a source ``.json``, and it refuses
to overwrite an existing workflow preset of the same name unless ``--force`` is given.

    python scripts/migrate_flows_to_presets.py [--force] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smile_msi import library, scripting  # noqa: E402

# Self-contained flow reader: the Flow designer + smile_msi.flow's Flow/flows_dir have been
# retired (plan 24), so this migration parses the saved .json presets directly rather than
# depending on the deleted API. A saved flow is a plain dict — {name, scope_region, steps:[{type,
# params, enabled}, …]} — so a shallow SimpleNamespace view is all the emitter needs.


def _flows_dir() -> str:
    return os.path.join(library.home_dir(), "flows")


def _load_flow(path: str) -> SimpleNamespace:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    steps = [SimpleNamespace(type=str(s.get("type", "")), params=dict(s.get("params") or {}),
                             enabled=bool(s.get("enabled", True)))
             for s in (d.get("steps") or [])]
    return SimpleNamespace(name=str(d.get("name", os.path.splitext(os.path.basename(path))[0])),
                           scope_region=str(d.get("scope_region", "")), steps=steps)


def _emit_code(fl: SimpleNamespace, src_path: str) -> str:
    """The Python script body reproducing ``fl`` as ``s.run_analysis(...)`` calls."""
    enabled = [st for st in fl.steps if st.enabled]
    scope = (fl.scope_region or "").strip()

    lines: list[str] = [
        f"# Migrated from Flow {fl.name!r}",
        f"# Source: {src_path}",
    ]

    mask_kw = ""                                   # appended to every run_analysis call
    if scope == "@grouped":
        lines += [
            "# Region scope '@grouped': restrict every step to the union of all grouped ROIs",
            "# (exactly what the Flow runner's _scope_mask() builds — one union mask).",
            "_scope = None",
            "_grouped = [s.group(g) for g in s.group_names()]",
            "if _grouped:",
            "    _scope = np.zeros(s.ds.n_pixels, dtype=bool)",
            "    for _m in _grouped:",
            "        _scope |= np.asarray(_m, dtype=bool)",
            "",
        ]
        mask_kw = ", mask=_scope"
    elif scope:
        lines += [
            f"# Region scope: restrict every step to region {scope!r}.",
            f"_scope = s.region({scope!r})",
            "",
        ]
        mask_kw = ", mask=_scope"
    else:
        lines.append("# Region scope: whole slide.")

    for st in enabled:
        kwargs = ", ".join(f"{k}={v!r}" for k, v in st.params.items())
        head = f's.run_analysis("{st.type}"'
        parts = [head]
        if kwargs:
            parts.append(kwargs)
        call = ", ".join(parts) + mask_kw + ")"
        lines.append(call)

    return "\n".join(lines) + "\n"


def migrate_one(src_path: str, *, force: bool, dry_run: bool) -> str:
    """Convert one flow file → workflow preset. Returns a one-line status for printing."""
    fl = _load_flow(src_path)
    n_enabled = sum(1 for st in fl.steps if st.enabled)
    code = _emit_code(fl, src_path)
    wf = scripting.Workflow(
        name=fl.name, code=code,
        description=f"migrated from Flow {fl.name!r} ({n_enabled} steps)")
    dest = scripting.workflow_path(fl.name)
    existed = os.path.exists(dest)

    if existed and not force:
        return f"SKIP  {fl.name!r}: preset already exists at {dest} (use --force to overwrite)"
    if dry_run:
        return f"DRY   {fl.name!r}: would write {dest} ({n_enabled} steps)"
    wf.save(dest)
    verb = "OVERWROTE" if existed else "WROTE"
    return f"{verb} {fl.name!r} → {dest} ({n_enabled} steps)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing workflow preset of the same name")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be written without writing anything")
    args = ap.parse_args()

    src_dir = _flows_dir()
    files = sorted(fn for fn in os.listdir(src_dir) if fn.endswith(".json"))
    if not files:
        print(f"No flow presets found in {src_dir}.")
        return 0

    print(f"Migrating {len(files)} flow(s) from {src_dir}")
    print(f"          into workflow presets under {scripting.workflows_dir()}\n")
    for fn in files:
        print("  " + migrate_one(os.path.join(src_dir, fn),
                                 force=args.force, dry_run=args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
