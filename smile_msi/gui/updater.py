"""In-place update for a run-from-source (git checkout) install.

A frozen Windows/macOS bundle can't safely overwrite its own running files — that's why
:mod:`smile_msi.gui.update_check` only points those users at the releases download page.
A *source* checkout, however, can update itself: this module runs ``git pull --ff-only``,
reinstalls dependencies when the project metadata changed in the pull, and relaunches.

Detecting a git checkout (a ``.git`` above the package) is what gates the feature — on a
frozen bundle :func:`is_git_checkout` returns ``False`` and the caller falls back to the
download page. The slow work (network pull, pip) runs on a daemon thread; results come
back on the GUI thread via a queued signal, mirroring :mod:`update_check`.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PySide6 import QtCore

# Files whose change in a pull means the environment must be reinstalled before relaunch.
_DEP_FILES = ("pyproject.toml", "uv.lock", "requirements.txt")


def repo_root() -> Path | None:
    """Top of the git checkout containing this package, or ``None`` if not a checkout
    (e.g. a frozen bundle, or a pip-installed wheel with no working tree)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists():
            return parent
    return None


def is_git_checkout() -> bool:
    return repo_root() is not None


def _run(args, cwd):
    return subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, timeout=300)


def _git(args, cwd):
    return _run(["git", *args], cwd)


def pull(root: Path) -> dict:
    """``git pull --ff-only`` in *root*. Returns a result dict:

    ``ok`` (bool), ``message`` (str — git output or the error to show), and on success also
    ``changed`` (did HEAD move), ``deps_changed`` (did a dependency file change),
    ``before`` / ``after`` (short SHAs)."""
    head = _git(["rev-parse", "HEAD"], root)
    if head.returncode != 0:
        return {"ok": False, "message": (head.stderr or head.stdout).strip() or "Not a git checkout."}
    before = head.stdout.strip()

    pulled = _git(["pull", "--ff-only"], root)
    if pulled.returncode != 0:
        return {"ok": False, "message": (pulled.stderr or pulled.stdout).strip()
                or "git pull failed (local changes or a diverged branch?)."}

    after = _git(["rev-parse", "HEAD"], root).stdout.strip()
    changed = bool(after) and after != before
    deps_changed = False
    if changed:
        diff = _git(["diff", "--name-only", before, after], root).stdout
        deps_changed = any(f in diff for f in _DEP_FILES)
    return {"ok": True, "changed": changed, "deps_changed": deps_changed,
            "before": before[:7], "after": after[:7],
            "message": (pulled.stdout or "").strip()}


def remote_status(root: Path) -> dict:
    """``git fetch`` then report how far this checkout is **behind its upstream** — the
    commit-based 'is there an update?' check for a source install, so *every push* counts
    (no GitHub release needed). Returns ``ok`` (bool), ``behind`` (int new commits on the
    remote), ``log`` (their short subjects, newest first), ``upstream`` (the ref compared
    against), and ``message`` (error text when not ok)."""
    head = _git(["rev-parse", "HEAD"], root)
    if head.returncode != 0:
        return {"ok": False, "behind": 0, "log": [],
                "message": (head.stderr or head.stdout).strip() or "Not a git checkout."}
    fetched = _git(["fetch", "--quiet"], root)
    if fetched.returncode != 0:
        return {"ok": False, "behind": 0, "log": [],
                "message": (fetched.stderr or fetched.stdout).strip() or "git fetch failed."}
    up = _git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], root)
    upstream = up.stdout.strip() if up.returncode == 0 else ""
    if not upstream:                                # no tracking branch → origin/<branch>, else origin/main
        br = _git(["rev-parse", "--abbrev-ref", "HEAD"], root).stdout.strip()
        for cand in ((f"origin/{br}" if br and br != "HEAD" else ""), "origin/main"):
            if cand and _git(["rev-parse", "--verify", "--quiet", cand], root).returncode == 0:
                upstream = cand
                break
    if not upstream:
        return {"ok": False, "behind": 0, "log": [],
                "message": "No remote branch to compare against."}
    cnt = _git(["rev-list", "--count", f"HEAD..{upstream}"], root)
    behind = int(cnt.stdout.strip() or 0) if cnt.returncode == 0 else 0
    log = []
    if behind:
        lg = _git(["log", "--oneline", "--no-decorate", "--no-color", f"HEAD..{upstream}"], root)
        if lg.returncode == 0:
            log = lg.stdout.strip().splitlines()[:10]
    return {"ok": True, "behind": behind, "log": log, "upstream": upstream, "message": ""}


class CheckWorker(QtCore.QObject):
    """Runs :func:`remote_status` (a network fetch) off the GUI thread; emits
    ``done(result, error)`` where *error* is ``""`` on success."""

    done = QtCore.Signal(dict, str)

    def run(self, root: Path):
        try:
            res = remote_status(root)
            self.done.emit(res, "" if res.get("ok") else res.get("message", "check failed"))
        except Exception as exc:                    # noqa: BLE001 — best-effort
            self.done.emit({"ok": False, "behind": 0, "log": [], "message": str(exc)}, str(exc))


def reinstall(root: Path) -> tuple[bool, str]:
    """``pip install -e .[gui]`` using the *running* interpreter's environment (the venv the
    app launched from), so a dependency bump in the pull lands before we relaunch."""
    r = _run([sys.executable, "-m", "pip", "install", "-e", ".[gui]"], root)
    return r.returncode == 0, (r.stdout + r.stderr)


def restart(root: Path) -> None:
    """Relaunch the app with the same interpreter, then the caller quits the old process."""
    QtCore.QProcess.startDetached(sys.executable, ["-m", "smile_msi.gui"], str(root))


class Worker(QtCore.QObject):
    """Runs pull (+ optional reinstall) off the GUI thread. Emits ``done(result, error)``
    where *result* is the :func:`pull` dict (with an added ``reinstall_log`` when deps were
    reinstalled) and *error* is ``""`` on success."""

    progress = QtCore.Signal(str)
    done = QtCore.Signal(dict, str)

    def run(self, root: Path):
        try:
            self.progress.emit("Pulling latest changes from GitHub…")
            res = pull(root)
            if not res["ok"]:
                self.done.emit(res, res["message"])
                return
            if res.get("changed") and res.get("deps_changed"):
                self.progress.emit("Dependencies changed — reinstalling (this can take a minute)…")
                ok, log = reinstall(root)
                res["reinstall_log"] = log
                if not ok:
                    self.done.emit(res, "Dependency reinstall failed:\n\n" + log[-2000:])
                    return
            self.done.emit(res, "")
        except Exception as exc:                # noqa: BLE001 — update is best-effort
            self.done.emit({"ok": False, "message": str(exc)}, str(exc))
