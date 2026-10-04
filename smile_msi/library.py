"""Dataset utilities shared by the session store and the GUI.

This module used to back a global ``library.json`` of reusable feature lists and saved
regions. That global store has been retired: a sample's whole analysis — feature lists,
working scopes, regions, segmentation — now lives in its per-sample session (see
:mod:`smile_msi.session`), which the app auto-saves under ``<home>/sessions/``. What
remains here are the dataset-identity and target-list helpers those sessions (and the GUI)
still rely on.

The home directory is ``~/.smile-msi`` (override with ``$SMILE_MSI_HOME``).
"""
from __future__ import annotations

import os
import re


# --------------------------------------------------------------------------- #
# locations
# --------------------------------------------------------------------------- #
def home_dir() -> str:
    """Writable app directory (holds the managed ``sessions/`` store); created on demand.

    Defaults to ``~/.smile-msi`` (override with ``$SMILE_MSI_HOME``)."""
    base = os.environ.get("SMILE_MSI_HOME")
    if not base:
        base = home_redirect() or default_home_dir()
    os.makedirs(base, exist_ok=True)
    return base


def default_home_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".smile-msi")


def _redirect_file() -> str:
    """``<default home>/home`` — one line naming the folder that actually holds the app
    store (sessions, cubes, prefs). Lets the caches live on a bigger disk without an
    environment variable; ``$SMILE_MSI_HOME`` still wins when set."""
    return os.path.join(default_home_dir(), "home")


def home_redirect() -> str:
    """The redirected app folder named by the redirect file, or ``""``."""
    try:
        with open(_redirect_file(), encoding="utf-8") as f:
            path = f.readline().strip()
    except OSError:
        return ""
    return os.path.expanduser(path) if path else ""


def set_home_redirect(path) -> None:
    """Point the app store at ``path`` (``None``/empty clears the redirect). Takes effect
    on the next :func:`home_dir` call; callers move the existing store themselves."""
    os.makedirs(default_home_dir(), exist_ok=True)
    rf = _redirect_file()
    if not path:
        try:
            os.remove(rf)
        except OSError:
            pass
        return
    with open(rf, "w", encoding="utf-8") as f:
        f.write(os.path.abspath(os.path.expanduser(str(path))) + "\n")


def dataset_key(source: str) -> str:
    """Stable id for a dataset. Uses the file's base name (so moving the project
    doesn't orphan its session); falls back to the raw string for synthetic /
    in-memory datasets."""
    if not source:
        return "unknown"
    if source == "synthetic":
        return "synthetic"
    return os.path.basename(str(source)) or str(source)


def dataset_fingerprint(ds) -> str:
    """Cheap, deterministic content id for a *loaded* dataset, memoised for its lifetime.

    The region/segmentation indices are aligned to a dataset's pixel ordering and
    count, so two different slides that happen to share a file basename could otherwise
    load one slide's analysis onto another. This captures the things that must match for
    those indices to mean anything — pixel count, grid size, m/z range, and a hash of the
    acquisition coordinates — so a mismatch can be detected and warned about. Returns
    ``""`` if it can't be computed.

    It is an *identity*, so it is cached on the dataset and must not drift while you work:
    re-deriving the m/z range scans every spectrum (slow — and this runs on every
    auto-save), and that scan can shift as preprocessing/reduce changes the live range. A
    drifting fingerprint re-keys the managed session and mints a *duplicate* cohort
    sample on each change, so lock it to the first (load-time) value."""
    cached = getattr(ds, "_fingerprint", None)
    if cached:
        return cached

    import hashlib

    import numpy as np

    try:
        coords = np.ascontiguousarray(np.asarray(ds.coordinates))
        h = hashlib.sha1(coords.tobytes()).hexdigest()[:12]
        lo, hi = ds.mz_range
        # A drifting / garbage m/z range (non-finite or absurd bounds from a misread
        # spectrum) must NOT re-key the slide — that mints a duplicate managed session and
        # cohort sample on every load. Fall back to a fixed token; the pixel count, grid
        # size and coordinate hash already identify the slide.
        if (np.isfinite(lo) and np.isfinite(hi) and hi > lo
                and abs(lo) < 1e6 and abs(hi) < 1e6):
            mz = f"{lo:.2f}-{hi:.2f}"
        else:
            mz = "NA"
        fp = f"n{int(ds.n_pixels)}:w{int(ds.width)}:h{int(ds.height)}:mz{mz}:{h}"
    except Exception:  # noqa: BLE001 — fingerprint is best-effort, never fatal
        return ""
    try:
        ds._fingerprint = fp          # memoise: identity must stay stable for the ds lifetime
    except Exception:  # noqa: BLE001 — an exotic ds may reject attribute writes
        pass
    return fp


# --------------------------------------------------------------------------- #
# target-list import  (plain m/z files, for interop)
# --------------------------------------------------------------------------- #
def parse_target_mzs(text: str) -> list[float]:
    """Pull m/z values out of a pasted/loaded target list. Accepts CSV/TSV/lines;
    if a header row names an m/z column it is honored, else the first numeric token
    on each line is used. Ignores blanks and comment (#) lines."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()
             and not ln.lstrip().startswith("#")]
    if not lines:
        return []
    # locate an m/z column from a header, if present
    head = re.split(r"[,\t;]", lines[0])
    mz_col = None
    if not _is_number(head[0]):
        for i, h in enumerate(head):
            if h.strip().lower() in ("mz", "m/z", "m_z", "mass"):
                mz_col = i
                break
        lines = lines[1:]                     # drop the header row
    out, seen = [], set()
    for ln in lines:
        cells = re.split(r"[,\t;]", ln)
        val = None
        if mz_col is not None and mz_col < len(cells) and _is_number(cells[mz_col]):
            val = float(cells[mz_col])
        else:
            for c in cells:                   # first numeric token on the line
                if _is_number(c):
                    val = float(c)
                    break
        if val is not None and val > 0:
            key = round(val, 6)
            if key not in seen:
                seen.add(key)
                out.append(val)
    return out


def _is_number(s: str) -> bool:
    try:
        float(str(s).strip())
        return True
    except ValueError:
        return False
