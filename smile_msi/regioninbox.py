"""Regions handed to the app from outside it — the MCP server's write path.

The app owns a slide's managed session file and auto-saves over it, so nothing else may
write regions into that file directly. Instead a sender drops a small JSON file into the
session's inbox (``<session>.inbox/`` beside the session, the ``.runs/`` sidecar idiom) and
the app takes it: on its next inbox poll when the slide is open, or when the slide is next
opened. Taking a file deletes it, so each region is added exactly once.

Masks travel as pixel-index lists, the same encoding the session file uses. Each file
carries the pixel count it was built against; the app refuses (and sets aside) a file whose
count does not match the open slide, since its indices would land on the wrong pixels.
"""
from __future__ import annotations

import json
import os
import time
import uuid

import numpy as np

from . import session


def inbox_dir(session_path: str) -> str:
    return os.path.splitext(str(session_path))[0] + ".inbox"


def post_regions(session_path: str, regions: list[dict], *, n_pixels: int,
                 source: str = "", sender: str = "mcp") -> str:
    """Queue ``regions`` (``{name, mask, color?, group?, parent?}``) for the app. Returns the file."""
    if not session_path:
        raise ValueError("this slide has no saved session to deliver regions to")
    payload = {
        "sender": sender, "created": time.time(), "source": source, "n_pixels": int(n_pixels),
        "regions": [{"name": str(r["name"]), "color": r.get("color"),
                     "group": str(r.get("group") or ""), "parent": r.get("parent") or None,
                     "mask": np.flatnonzero(np.asarray(r["mask"], dtype=bool)).tolist()}
                    for r in regions],
    }
    d = inbox_dir(session_path)
    os.makedirs(d, exist_ok=True)
    stem = f"regions-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    tmp, final = os.path.join(d, stem + ".tmp"), os.path.join(d, stem + ".json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, final)          # the app never sees a half-written file
    return final


def pending(session_path: str) -> list[str]:
    d = inbox_dir(session_path)
    if not session_path or not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, fn) for fn in os.listdir(d) if fn.endswith(".json"))


def take_regions(session_path: str, n_pixels: int) -> tuple[list[dict], list[str]]:
    """Every queued region for this slide, with boolean masks, removing the files.

    Returns ``(regions, problems)``. A file built against a different pixel count, or one
    that cannot be parsed, is renamed ``*.rejected`` and reported instead of applied."""
    regions, problems = [], []
    for path in pending(session_path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if int(data.get("n_pixels", -1)) != int(n_pixels):
                raise ValueError(f"built for {data.get('n_pixels')} pixels, slide has {n_pixels}")
            for r in data.get("regions") or []:
                regions.append({"name": str(r["name"]), "color": r.get("color"),
                                "group": str(r.get("group") or ""),
                                "parent": r.get("parent") or None,
                                "mask": session.mask_from_indices(r["mask"], n_pixels)})
            os.remove(path)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            problems.append(f"{os.path.basename(path)}: {exc}")
            try:
                os.replace(path, path[:-5] + ".rejected")
            except OSError:
                pass
    return regions, problems
