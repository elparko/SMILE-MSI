"""Feedback module: label/body helpers and the file-based token store.

The GUI dialog and the network POST aren't exercised here (no display, no live token);
these cover the logic that decides what gets filed and how the token is persisted."""
import importlib

import pytest

# The feedback module pulls in the desktop GUI stack (PySide6/pyqtgraph) on import, so
# skip these on the core-only CI run that installs no GUI extras — the `[gui]` job still
# exercises them. Mirrors the guard in test_gui.py.
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")


@pytest.fixture()
def fb(tmp_path, monkeypatch):
    """The feedback module with token storage pointed at a tmp dir and keyring disabled,
    so get/set/clear exercise the 0600-file fallback deterministically."""
    monkeypatch.setenv("SMILE_MSI_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("SMILE_MSI_NO_KEYRING", "1")
    mod = importlib.import_module("smile_msi.gui.feedback")
    return mod


def test_labels_plain_vs_agent(fb):
    assert fb.labels_for("Feature request", False) == ["enhancement"]
    assert fb.labels_for("Bug report", False) == ["bug"]
    assert fb.labels_for("Feature request", True) == ["enhancement", fb.AGENT_LABEL]


def test_build_body_appends_footer(fb):
    body = fb.build_body("It crashed.", version="9.9.9", env="TestOS")
    assert body.startswith("It crashed.")
    assert "SMILE MSI 9.9.9" in body and "TestOS" in body


def test_build_body_handles_empty_detail(fb):
    body = fb.build_body("   ", version="1.0", env="X")
    assert "no description" in body.lower()


def test_token_roundtrip_file(fb, tmp_path):
    assert fb.get_token() is None
    fb.set_token("ghp_secret123")
    assert fb.get_token() == "ghp_secret123"
    # stored with owner-only permissions
    tok_file = tmp_path / "github_token.json"
    assert tok_file.exists()
    assert (tok_file.stat().st_mode & 0o077) == 0  # no group/other access
    fb.clear_token()
    assert fb.get_token() is None


# --- "My requests" status helpers ------------------------------------------ #
def test_status_label_kinds(fb):
    assert fb.status_label("open", None, [])[1] == "open"
    assert fb.status_label("open", None, ["enhancement", fb.AGENT_LABEL])[1] == "working"
    assert fb.status_label("closed", "completed", [])[1] == "done"
    assert fb.status_label("closed", "not_planned", [])[1] == "wontfix"
    assert fb.status_label("closed", None, [])[1] == "closed"
    assert fb.status_label("weird", None, [])[1] == "unknown"


def test_issue_number_from_url(fb):
    assert fb._issue_number_from_url("https://github.com/o/r/issues/42") == 42
    assert fb._issue_number_from_url("https://github.com/o/r/pull/9") is None
    assert fb._issue_number_from_url("garbage") is None


def test_submissions_log_roundtrip_newest_first(fb):
    assert fb.read_submissions() == []
    fb.record_submission(7, "Add X", "Feature request", True)
    fb.record_submission(8, "Fix Y", "Bug report", False)
    recs = fb.read_submissions()
    assert [r["number"] for r in recs] == [8, 7]        # newest first
    assert recs[1]["agent"] is True and recs[1]["title"] == "Add X"


def test_submissions_log_dedupes_on_number(fb):
    fb.record_submission(7, "first", "Feature request", False)
    fb.record_submission(7, "updated", "Feature request", True)
    recs = fb.read_submissions()
    assert len(recs) == 1
    assert recs[0]["title"] == "updated" and recs[0]["agent"] is True
