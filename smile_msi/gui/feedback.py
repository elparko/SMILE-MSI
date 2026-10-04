"""In-app feature requests / bug reports → GitHub issues.

Help → "Send feedback / Feature request…" opens a small form. On submit it POSTs a new
issue to the project repo via the GitHub REST API, on a daemon thread so the UI never
blocks (the result comes back on the GUI thread via a queued signal — same shape as
``update_check``). A "Have an automated agent attempt this" checkbox adds the ``agent-fix``
label so an agent (the ``agent-fix`` GitHub Action) can pick the issue up and open a PR;
left unchecked it's a plain ``enhancement`` / ``bug`` for a human to look at later.

Auth: the user's own GitHub token. We store it with :mod:`keyring` when that's importable
(macOS Keychain / Windows Credential Manager), and otherwise fall back to a ``0600`` JSON
file under the per-OS app config dir — the same approach the ``gh`` and ``aws`` CLIs take.
``keyring`` is deliberately *not* a declared dependency: it's fragile inside the frozen
PyInstaller bundle, so the file path is the one that always works in a shipped build.

Network uses ``urllib`` (not ``requests``) so nothing new has to be bundled.
"""
from __future__ import annotations

import json
import os
import platform
import re
import threading
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .common import NoScrollComboBox, install_table_export

from .update_check import REPO            # single source of truth for owner/name

AGENT_LABEL = "agent-fix"                 # marks issues the automation should attempt
_KIND_LABELS = {"Feature request": "enhancement", "Bug report": "bug"}
_TOKEN_PAGE = "https://github.com/settings/tokens?type=beta"   # fine-grained PAT page
_ISSUES_PAGE = f"https://github.com/{REPO}/issues"

_SERVICE = "SMILE MSI"                     # keyring service / app config folder name
_TOKEN_KEY = "github-token"


# --------------------------------------------------------------------------- #
# Token storage (keyring if available, else a 0600 file)
# --------------------------------------------------------------------------- #
def _config_dir() -> Path:
    """Per-OS app config directory (created on demand). Overridable via
    ``SMILE_MSI_CONFIG_DIR`` so tests can point it at a tmp path."""
    override = os.environ.get("SMILE_MSI_CONFIG_DIR")
    if override:
        base = Path(override)
    elif os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home())) / _SERVICE
    elif platform.system() == "Darwin":
        base = Path.home() / "Library" / "Application Support" / _SERVICE
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        base = (Path(xdg) if xdg else Path.home() / ".config") / "smile-msi"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _token_file() -> Path:
    return _config_dir() / "github_token.json"


def _keyring():
    """The :mod:`keyring` module if importable and not disabled, else ``None``.
    ``SMILE_MSI_NO_KEYRING`` forces the file path (used by tests, and an escape hatch
    on machines where the keyring backend misbehaves)."""
    if os.environ.get("SMILE_MSI_NO_KEYRING"):
        return None
    try:
        import keyring
        return keyring
    except Exception:                      # noqa: BLE001 — keyring is best-effort
        return None


def get_token() -> str | None:
    kr = _keyring()
    if kr is not None:
        try:
            tok = kr.get_password(_SERVICE, _TOKEN_KEY)
            if tok:
                return tok
        except Exception:                  # noqa: BLE001 — fall back to the file
            pass
    p = _token_file()
    if p.exists():
        try:
            return json.loads(p.read_text()).get("github_token") or None
        except (OSError, ValueError):
            return None
    return None


def set_token(token: str) -> None:
    token = (token or "").strip()
    kr = _keyring()
    if kr is not None:
        try:
            kr.set_password(_SERVICE, _TOKEN_KEY, token)
            _token_file().unlink(missing_ok=True)   # don't keep a plaintext copy too
            return
        except Exception:                  # noqa: BLE001 — fall back to the file
            pass
    p = _token_file()
    p.write_text(json.dumps({"github_token": token}))
    try:
        os.chmod(p, 0o600)                 # readable only by the user
    except OSError:
        pass


def clear_token() -> None:
    kr = _keyring()
    if kr is not None:
        try:
            kr.delete_password(_SERVICE, _TOKEN_KEY)
        except Exception:                  # noqa: BLE001 — best-effort
            pass
    _token_file().unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Pure helpers (no Qt / no network — unit-tested)
# --------------------------------------------------------------------------- #
def labels_for(kind: str, agent: bool) -> list[str]:
    """Labels for a new issue: the kind (``enhancement``/``bug``) plus ``agent-fix``
    when the user asked the automation to handle it. GitHub auto-creates any label the
    repo doesn't have yet (the token has push access)."""
    labels = [_KIND_LABELS.get(kind, "enhancement")]
    if agent:
        labels.append(AGENT_LABEL)
    return labels


def build_body(detail: str, *, version: str, env: str) -> str:
    """Issue body: the user's text plus a footer recording the app version and platform
    (handy when triaging a bug)."""
    detail = (detail or "").strip() or "_(no description provided)_"
    return f"{detail}\n\n---\n_Filed from SMILE MSI {version} · {env}_"


def _env_string() -> str:
    return (f"{platform.system()} {platform.release()} · "
            f"Python {platform.python_version()}")


# --------------------------------------------------------------------------- #
# Submitted-request log (so "My requests" knows what *this* app filed) + status
# --------------------------------------------------------------------------- #
def _submissions_file() -> Path:
    return _config_dir() / "submitted_requests.json"


def _issue_number_from_url(url: str) -> int | None:
    """Pull the issue number out of an ``…/issues/<n>`` html_url."""
    m = re.search(r"/issues/(\d+)", url or "")
    return int(m.group(1)) if m else None


def read_submissions() -> list[dict]:
    """Newest-first list of requests filed from this app: number, title, kind, agent, ts."""
    p = _submissions_file()
    if not p.exists():
        return []
    try:
        recs = json.loads(p.read_text())
        return recs if isinstance(recs, list) else []
    except (OSError, ValueError):
        return []


def record_submission(number: int, title: str, kind: str, agent: bool) -> None:
    """Append (or refresh) a submitted issue at the front of the local log, capped at 50."""
    recs = [r for r in read_submissions() if r.get("number") != number]
    recs.insert(0, {"number": int(number), "title": title, "kind": kind,
                    "agent": bool(agent),
                    "ts": datetime.now().isoformat(timespec="seconds")})
    try:
        _submissions_file().write_text(json.dumps(recs[:50], indent=2))
    except OSError:                            # logging is best-effort
        pass


def status_label(state: str, state_reason, labels) -> tuple[str, str]:
    """Map raw GitHub issue fields to (display text, kind). ``kind`` drives the row colour
    and is the stable value tests assert on."""
    labels = labels or []
    if state == "open":
        if AGENT_LABEL in labels:
            return ("Agent working", "working")
        return ("Open", "open")
    if state == "closed":
        if state_reason == "completed":
            return ("Done", "done")
        if state_reason == "not_planned":
            return ("Won’t fix", "wontfix")
        return ("Closed", "closed")
    return ("Unknown", "unknown")


def _friendly_ts(ts: str) -> str:
    return (ts or "")[:10]                     # ISO timestamp → just the date


# --------------------------------------------------------------------------- #
# GitHub REST call
# --------------------------------------------------------------------------- #
def create_issue(repo: str, token: str, title: str, body: str, labels) -> str:
    """POST a new issue; return its ``html_url``. Raises on any HTTP/network error."""
    payload = json.dumps({"title": title, "body": body, "labels": list(labels)}).encode()
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues",
        data=payload, method="POST",
        headers={"Accept": "application/vnd.github+json",
                 "Authorization": f"Bearer {token}",
                 "User-Agent": "SMILE-MSI",
                 "Content-Type": "application/json",
                 "X-GitHub-Api-Version": "2022-11-28"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.load(resp)
    return data.get("html_url") or _ISSUES_PAGE


def _describe_error(exc: Exception) -> str:
    """Turn a raw HTTP/network error into a one-paragraph, fixable message."""
    if isinstance(exc, urllib.error.HTTPError):
        try:
            detail = json.loads(exc.read().decode()).get("message", "")
        except Exception:                  # noqa: BLE001
            detail = ""
        code = exc.code
        if code == 401:
            return ("GitHub rejected the token (401). It's probably wrong or expired — "
                    "use “GitHub token…” to set a new one.")
        if code in (403, 404):
            return (f"GitHub returned {code}. The token likely lacks “Issues: read and "
                    f"write” permission on {repo_label()} (or the repo name is wrong)."
                    + (f"\n\n{detail}" if detail else ""))
        if code == 422:
            return f"GitHub couldn’t process the request (422). {detail}".strip()
        return f"GitHub error {code}. {detail}".strip()
    return f"Couldn’t reach GitHub.\n\n{exc}"


def repo_label() -> str:
    return REPO


# --------------------------------------------------------------------------- #
# Worker (mirrors update_check: network off-thread, result via queued signal)
# --------------------------------------------------------------------------- #
class _Worker(QtCore.QObject):
    done = QtCore.Signal(str, str)         # html_url, error ("" on success)

    def run(self, repo, token, title, body, labels):
        try:
            url = create_issue(repo, token, title, body, labels)
            self.done.emit(url, "")
        except Exception as exc:           # noqa: BLE001 — network is best-effort
            self.done.emit("", _describe_error(exc))


def fetch_issue_status(repo: str, token: str, number: int) -> dict:
    """GET one issue; return the fields the status view needs. Raises on HTTP/network error."""
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues/{number}",
        headers={"Accept": "application/vnd.github+json",
                 "Authorization": f"Bearer {token}",
                 "User-Agent": "SMILE-MSI",
                 "X-GitHub-Api-Version": "2022-11-28"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        d = json.load(resp)
    return {"number": number,
            "title": d.get("title", ""),
            "state": d.get("state", ""),
            "state_reason": d.get("state_reason"),
            "labels": [lab.get("name") for lab in d.get("labels", [])],
            "html_url": d.get("html_url", f"https://github.com/{repo}/issues/{number}")}


class _StatusWorker(QtCore.QObject):
    """Fetch status for many issues off-thread. Tolerant: a single bad issue (deleted, etc.)
    becomes an ``unknown`` row rather than failing the batch; the first error is reported so
    a global problem (e.g. a 401 token) still surfaces."""
    done = QtCore.Signal(list, str)        # [status dict…], error ("" on success)

    def run(self, repo, token, numbers):
        out, err = [], ""
        for n in numbers:
            try:
                out.append(fetch_issue_status(repo, token, n))
            except Exception as exc:       # noqa: BLE001 — per-issue, keep going
                if not err:
                    err = _describe_error(exc)
                out.append({"number": n, "title": None, "state": "unknown",
                            "state_reason": None, "labels": [],
                            "html_url": f"https://github.com/{repo}/issues/{n}"})
        self.done.emit(out, err)


# --------------------------------------------------------------------------- #
# Token entry
# --------------------------------------------------------------------------- #
def _prompt_for_token(parent) -> str | None:
    """Modal dialog explaining the required token and collecting it. Returns the token
    (already saved) or ``None`` if cancelled / empty."""
    from .common import icon, note, section_title

    dlg = QtWidgets.QDialog(parent)
    dlg.setWindowTitle("GitHub token")
    v = QtWidgets.QVBoxLayout(dlg)
    v.addWidget(section_title("Connect your GitHub account"))
    v.addWidget(note(
        f"Feedback is filed as an issue on {REPO}, and the same token lets the app check "
        "for updates. Create a fine-grained personal access token on that repository with "
        "“Issues: Read and write” and “Contents: Read”, then paste it below. It's stored "
        "locally on this machine only."))

    open_btn = QtWidgets.QPushButton("Open GitHub token page…")
    open_btn.setIcon(icon("open"))
    open_btn.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(_TOKEN_PAGE)))
    v.addWidget(open_btn)

    field = QtWidgets.QLineEdit()
    field.setEchoMode(QtWidgets.QLineEdit.Password)
    field.setPlaceholderText("github_pat_… or ghp_…")
    v.addWidget(field)

    bb = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
    bb.accepted.connect(dlg.accept)
    bb.rejected.connect(dlg.reject)
    v.addWidget(bb)

    if dlg.exec() != QtWidgets.QDialog.Accepted:
        return None
    token = field.text().strip()
    if not token:
        return None
    set_token(token)
    return token


# --------------------------------------------------------------------------- #
# Feedback dialog
# --------------------------------------------------------------------------- #
class FeedbackDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        from .common import icon, note, section_title
        from .. import __version__

        self._version = __version__
        self._worker = None

        self.setWindowTitle("Send feedback")
        self.setMinimumWidth(460)
        v = QtWidgets.QVBoxLayout(self)

        v.addWidget(section_title("Send feedback / feature request"))
        v.addWidget(note(f"Files an issue on {REPO}. Be specific — for a bug, include what "
                         "you did, what you expected, and what happened."))

        form = QtWidgets.QFormLayout()
        self.kind_combo = NoScrollComboBox()
        self.kind_combo.addItems(list(_KIND_LABELS))
        form.addRow("Type:", self.kind_combo)

        self.title_edit = QtWidgets.QLineEdit()
        self.title_edit.setPlaceholderText("Short summary")
        form.addRow("Title:", self.title_edit)
        v.addLayout(form)

        self.detail = QtWidgets.QPlainTextEdit()
        self.detail.setPlaceholderText(
            "Details — steps to reproduce, what you expected, what happened, screenshots…")
        self.detail.setMinimumHeight(130)
        v.addWidget(self.detail)

        self.agent_check = QtWidgets.QCheckBox("Have an automated agent attempt this")
        self.agent_check.setToolTip(
            f"Adds the “{AGENT_LABEL}” label so the automation picks this up and opens "
            "a pull request for review")
        v.addWidget(self.agent_check)
        v.addWidget(note("Checked → an agent attempts it and opens a pull request for you "
                         "to review. Unchecked → filed for a person to look at later."))

        # Token status row + change button.
        row = QtWidgets.QHBoxLayout()
        self.token_status = note("")
        row.addWidget(self.token_status, 1)
        tok_btn = QtWidgets.QPushButton("GitHub token…")
        tok_btn.setIcon(icon("settings"))
        tok_btn.setToolTip("Set or replace the token used to file issues")
        tok_btn.clicked.connect(self._change_token)
        row.addWidget(tok_btn)
        v.addLayout(row)

        self.bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Cancel)
        self.submit_btn = self.bb.addButton("Submit", QtWidgets.QDialogButtonBox.AcceptRole)
        self.submit_btn.setIcon(icon("save"))
        self.submit_btn.setDefault(True)
        self.bb.accepted.connect(self._submit)
        self.bb.rejected.connect(self.reject)
        v.addWidget(self.bb)

        self._refresh_token_status()

    # -- token status ------------------------------------------------------- #
    def _refresh_token_status(self):
        if get_token():
            self.token_status.setText("GitHub: token saved ✓")
        else:
            self.token_status.setText("GitHub: no token set — required to submit")

    def _change_token(self):
        if _prompt_for_token(self):
            self._refresh_token_status()

    # -- submit ------------------------------------------------------------- #
    def _set_busy(self, busy):
        self.submit_btn.setEnabled(not busy)
        self.submit_btn.setText("Sending…" if busy else "Submit")

    def _submit(self):
        title = self.title_edit.text().strip()
        if not title:
            QtWidgets.QMessageBox.information(self, "Send feedback", "Please enter a title.")
            self.title_edit.setFocus()
            return
        token = get_token() or _prompt_for_token(self)
        if not token:
            self._refresh_token_status()
            return
        self._refresh_token_status()
        body = build_body(self.detail.toPlainText(), version=self._version, env=_env_string())
        labels = labels_for(self.kind_combo.currentText(), self.agent_check.isChecked())

        self._set_busy(True)
        self._worker = _Worker()
        self._worker.done.connect(self._on_done)
        threading.Thread(target=self._worker.run,
                         args=(REPO, token, title, body, labels), daemon=True).start()

    def _on_done(self, url, error):
        self._set_busy(False)
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.deleteLater()
        if error:
            QtWidgets.QMessageBox.warning(self, "Couldn’t file the issue", error)
            return
        num = _issue_number_from_url(url)
        if num is not None:                    # log it so Help → My requests can track status
            record_submission(num, self.title_edit.text().strip(),
                               self.kind_combo.currentText(), self.agent_check.isChecked())
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Thanks!")
        box.setText("Your feedback was filed as a GitHub issue.")
        box.setInformativeText("Open it in your browser?")
        box.setStandardButtons(QtWidgets.QMessageBox.Open | QtWidgets.QMessageBox.Close)
        box.setDefaultButton(QtWidgets.QMessageBox.Open)
        if box.exec() == QtWidgets.QMessageBox.Open:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))
        self.accept()


def open_feedback_dialog(parent):
    """Help-menu entry point. Modal so the network call's result stays scoped to it."""
    dlg = FeedbackDialog(parent)
    dlg.exec()
    return dlg


# --------------------------------------------------------------------------- #
# "My requests" — status of everything filed from this app
# --------------------------------------------------------------------------- #
_STATUS_COLORS = {                             # by status_label() kind; "" = theme default
    "working": "#d29922",                      # amber — agent is on it
    "open": "",
    "done": "#3fb950",                         # green
    "wontfix": "#8b949e",
    "closed": "#8b949e",
    "unknown": "#8b949e",
    "loading": "#8b949e",
}


class MyRequestsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        from .common import icon, note, section_title

        self.setWindowTitle("My requests")
        self.setMinimumSize(560, 380)
        self._worker = None
        self._records = []

        v = QtWidgets.QVBoxLayout(self)
        v.addWidget(section_title("My requests"))
        v.addWidget(note(f"Feature requests and bug reports you've filed from this app, with "
                         f"their current status on {REPO}. Double-click a row to open it on "
                         "GitHub."))

        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Status", "Request", "Submitted"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.table.cellDoubleClicked.connect(self._open_row)
        install_table_export(self.table, self, stem="my_requests", title="Export requests")
        v.addWidget(self.table, 1)

        self.status = note("")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.status, 1)
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.setIcon(icon("refresh"))
        self.refresh_btn.clicked.connect(self.reload)
        row.addWidget(self.refresh_btn)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        row.addWidget(close_btn)
        v.addLayout(row)

        self.reload()

    def reload(self):
        self._records = read_submissions()
        # Show titles immediately (status "…"), then fetch live state off-thread.
        self._populate([dict(r, state=None) for r in self._records])
        if not self._records:
            self.status.setText("No requests submitted from this app yet.")
            return
        token = get_token()
        if not token:
            self.status.setText("No GitHub token set — can't fetch status. "
                                "Set one from Help → Send feedback.")
            return
        self.status.setText("Loading status…")
        self.refresh_btn.setEnabled(False)
        self._worker = _StatusWorker()
        self._worker.done.connect(self._on_status)
        threading.Thread(target=self._worker.run,
                         args=(REPO, token, [r["number"] for r in self._records]),
                         daemon=True).start()

    def _on_status(self, results, error):
        self.refresh_btn.setEnabled(True)
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.deleteLater()
        self._populate(results)
        if error:
            self.status.setText(error)
        else:
            self.status.setText(f"{len(results)} request(s) · status updated")

    def _populate(self, results):
        meta = {r["number"]: r for r in self._records}
        self.table.setRowCount(len(results))
        for i, res in enumerate(results):
            num = res.get("number")
            m = meta.get(num, {})
            if res.get("state") is None:       # still loading
                text, kind = ("…", "loading")
            else:
                text, kind = status_label(res.get("state"), res.get("state_reason"),
                                          res.get("labels"))
            sitem = QtWidgets.QTableWidgetItem(text)
            color = _STATUS_COLORS.get(kind, "")
            if color:
                sitem.setForeground(QtGui.QBrush(QtGui.QColor(color)))
            self.table.setItem(i, 0, sitem)

            title = res.get("title") or m.get("title") or f"#{num}"
            titem = QtWidgets.QTableWidgetItem(f"{title}  (#{num})")
            titem.setData(QtCore.Qt.UserRole, res.get("html_url") or m.get("html_url"))
            self.table.setItem(i, 1, titem)

            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(_friendly_ts(m.get("ts", ""))))

    def _open_row(self, row, _col):
        item = self.table.item(row, 1)
        url = item.data(QtCore.Qt.UserRole) if item else None
        if url:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))


def open_my_requests_dialog(parent):
    """Help-menu entry point for the request-status window."""
    dlg = MyRequestsDialog(parent)
    dlg.exec()
    return dlg
