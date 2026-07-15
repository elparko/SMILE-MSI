"""Lightweight "check for updates" against GitHub Releases.

A frozen Windows/macOS bundle can't safely overwrite its own running files, so this does
the simple, robust thing: ask the GitHub Releases API for the latest version and, if it's
newer than the running one, point the user at the download page. No background polling and
no telemetry — it runs only when the user picks Help → Check for updates. Network work
happens on a daemon thread so the UI never blocks; the result is delivered back on the GUI
thread via a queued signal.
"""
from __future__ import annotations

import json
import os
import threading
import urllib.request

from PySide6 import QtCore, QtGui, QtWidgets

REPO = "elparko/SMILE-MSI"          # owner/name the releases API is queried against
_API = "https://api.github.com/repos/{}/releases/latest"
_RELEASES_PAGE = "https://github.com/{}/releases/latest"

def _token() -> str:
    """GitHub token used to reach the releases API — required while the repo is **private**.

    Reuses the *same* token the feedback / agent-fix flow stores (keyring, or a 0600 file
    fallback — see :mod:`smile_msi.gui.feedback`), so one token set via *GitHub token…*
    powers both filing feature requests and checking for updates. An env var
    (``$SMILE_MSI_GITHUB_TOKEN`` / ``$GITHUB_TOKEN``) overrides the stored token. Returns
    ``""`` when none is set, in which case the request goes out anonymously (fine once the
    repo is public). For a private repo the token needs *Contents: Read* on it."""
    tok = os.environ.get("SMILE_MSI_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not tok:
        try:
            from . import feedback
            tok = feedback.get_token()
        except Exception:                       # noqa: BLE001 — token store is best-effort
            tok = ""
    return str(tok or "").strip()


def _parse_version(tag):
    """Dotted version tuple from a tag like ``v0.3.1`` → ``(0, 3, 1)``. Stops at the first
    non-numeric component so pre-release suffixes don't crash the compare."""
    nums = []
    for part in str(tag).lstrip("vV").split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        nums.append(int(digits))
    return tuple(nums)


def _fetch_latest(repo):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "SMILE-MSI"}
    tok = _token()
    if tok:                                     # authenticated read — needed for a private repo
        headers["Authorization"] = f"Bearer {tok}"
    req = urllib.request.Request(_API.format(repo), headers=headers)
    with urllib.request.urlopen(req, timeout=6) as resp:
        data = json.load(resp)
    return data.get("tag_name") or "", data.get("html_url") or _RELEASES_PAGE.format(repo)


class _Worker(QtCore.QObject):
    done = QtCore.Signal(str, str, str)        # tag, url, error ("" on success)

    def run(self, repo):
        try:
            tag, url = _fetch_latest(repo)
            self.done.emit(tag, url, "")
        except Exception as exc:               # noqa: BLE001 — network is best-effort
            self.done.emit("", "", str(exc))


def check_for_updates(parent, current_version, repo=REPO, silent=False):
    """Query GitHub for the latest release; offer the download page if it's newer.

    ``silent`` suppresses the "you're up to date" / error dialogs (for an automatic check);
    a manual Help-menu check passes ``silent=False`` so it always gives feedback.
    """
    worker = _Worker()

    def _on_done(tag, url, error):
        try:
            if error:
                if not silent:
                    QtWidgets.QMessageBox.information(
                        parent, "Check for updates",
                        f"Couldn't reach GitHub to check for updates.\n\n{error}")
                return
            latest, current = _parse_version(tag), _parse_version(current_version)
            if latest and latest > current:
                box = QtWidgets.QMessageBox(parent)
                box.setWindowTitle("Update available")
                box.setText(f"SMILE MSI {tag} is available (you have {current_version}).")
                box.setInformativeText("Open the download page to get the latest build?")
                box.setStandardButtons(
                    QtWidgets.QMessageBox.Open | QtWidgets.QMessageBox.Cancel)
                box.setDefaultButton(QtWidgets.QMessageBox.Open)
                if box.exec() == QtWidgets.QMessageBox.Open:
                    QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))
            elif not silent:
                QtWidgets.QMessageBox.information(
                    parent, "Check for updates",
                    f"You're on the latest version ({current_version}).")
        finally:
            worker.deleteLater()
            if getattr(parent, "_update_worker", None) is worker:
                parent._update_worker = None

    worker.done.connect(_on_done)               # cross-thread → delivered on the GUI thread
    parent._update_worker = worker              # keep a ref so it isn't GC'd before it fires
    threading.Thread(target=worker.run, args=(repo,), daemon=True).start()
