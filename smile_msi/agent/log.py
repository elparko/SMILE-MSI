"""The agent's audit log — an append-only record of everything a chat session does.

One JSON object per line in ``<home>/agent/logs/<started>-<id>.jsonl``, written (and flushed)
as each event happens, so a crash mid-analysis still leaves the full trail. Every record has
``seq``, ``t`` (ISO time) and ``type``; the types are:

``session``      provider, model, settings (never the API key), smile_msi version
``user``         what the user typed
``assistant``    the model's text
``thinking``     the model's reasoning summary (when the provider returns one)
``tool_call``    tool name, arguments, call id
``tool_result``  ok/error, duration, the result text the model read, images and files made
``approval``     a create/edit/delete-tool request and the user's decision (with code sha256)
``dataset``      fingerprint of a slide when one is opened (ties results to the exact data)
``usage``        tokens per model step
``error``        anything that went wrong outside a tool

:meth:`SessionLog.markdown` renders the same record as a readable report, ending with the
ordered tool calls as a replayable flow — the reproducibility appendix of the analysis.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import threading
import uuid


#: Characters of each tool result quoted in the Markdown report (the JSONL has all of it).
EXCERPT = 1500
#: Agent tools that manage tools/flows — not part of a replayable analysis.
_NOT_REPLAYABLE = {"create_tool", "edit_tool", "delete_tool", "save_flow", "list_flows",
                   "get_tool_code", "run_flow"}


def _quote(text: str) -> str:
    """User/model text as a Markdown blockquote, so nothing in it can pose as a log heading."""
    return "\n".join("> " + ln for ln in str(text).splitlines()) or ">"


def _inline(text) -> str:
    return " ".join(str(text).split())


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="milliseconds")


class SessionLog:
    def __init__(self, directory: str | None = None):
        if directory is None:
            from .custom import agent_dir

            directory = agent_dir("logs")
        self.id = uuid.uuid4().hex[:8]
        self.started = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = os.path.join(directory, f"{self.started}-{self.id}.jsonl")
        self.records: list[dict] = []
        self._lock = threading.Lock()
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, type_: str, **data) -> dict:
        with self._lock:
            rec = {"seq": len(self.records) + 1, "t": _now(), "type": type_, **data}
            try:
                line = json.dumps(rec, default=str)
            except (TypeError, ValueError, RecursionError):
                rec = {k: (v if isinstance(v, (str, int, float, bool, type(None))) else repr(v))
                       for k, v in rec.items()}
                line = json.dumps(rec, default=str)
            if self._fh.closed:              # an abandoned conversation's late event
                return rec
            self.records.append(rec)
            self._fh.write(line + "\n")
            self._fh.flush()
            return rec

    def close(self):
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    # ------------------------------------------------------------------ #
    def tool_calls(self) -> list[dict]:
        """The successful tool calls in order, as flow steps (``{"tool", "args"}``)."""
        ok = {r["call_id"] for r in self.records if r["type"] == "tool_result" and r.get("ok")}
        return [{"tool": r["name"], "args": r.get("args", {})} for r in self.records
                if r["type"] == "tool_call" and r["call_id"] in ok
                and r["name"] not in _NOT_REPLAYABLE]

    def markdown(self) -> str:
        """The session as a readable report: conversation, every tool call with its
        arguments and outcome, approvals, datasets, token use — and a replayable flow."""
        lines = [f"# SMILE MSI analysis log — {self.started}", ""]
        tokens_in = tokens_out = 0
        for r in self.records:
            t = r["t"][11:19]
            kind = r["type"]
            if kind == "session":
                lines += [f"**Session** `{self.id}` · provider **{r.get('provider')}** · model "
                          f"`{r.get('model')}` · smile_msi {r.get('version')}", ""]
            elif kind == "user":
                lines += [f"### {t} · You", "", _quote(r.get("text", "")), ""]
            elif kind == "assistant":
                lines += [f"### {t} · Assistant", "", _quote(r.get("text", "")), ""]
            elif kind == "thinking":
                lines += [f"<details><summary>{t} · reasoning</summary>", "",
                          _quote(r.get("text", "")), "", "</details>", ""]
            elif kind == "tool_call":
                args = json.dumps(r.get("args", {}), default=str)
                lines += [f"- `{t}` → **{r['name']}** `{args}`"]
            elif kind == "tool_result":
                status = "ok" if r.get("ok") else f"**error** — {_inline(r.get('error', ''))}"
                extra = "".join(f"\n  - image: `{p}`" for p in r.get("images", []))
                extra += "".join(f"\n  - file: `{p}`" for p in r.get("files", []))
                excerpt = (r.get("text") or "")[:EXCERPT]
                if excerpt and r.get("ok"):
                    more = " …" if len(r.get("text") or "") > EXCERPT else ""
                    extra += "\n\n    ```\n" + "\n".join(
                        "    " + ln.replace("```", "ʼʼʼ") for ln in excerpt.splitlines()) \
                        + more + "\n    ```"
                lines += [f"  - result ({r.get('duration_s', 0):.2f}s): {status}{extra}"]
            elif kind == "approval":
                lines += [f"- `{t}` **approval** {r.get('tool')} `{r.get('target', '')}` → "
                          f"**{r.get('decision')}**"
                          + (f" (code sha256 `{r['sha256'][:12]}`)" if r.get("sha256") else "")]
            elif kind == "dataset":
                lines += [f"- `{t}` dataset **{r.get('ref')}** · fingerprint "
                          f"`{r.get('fingerprint')}` · {r.get('n_pixels')} pixels"]
            elif kind == "usage":
                tokens_in += int(r.get("input_tokens") or 0)
                tokens_out += int(r.get("output_tokens") or 0)
            elif kind == "notice":
                lines += [f"- `{t}` _{r.get('text')}_"]
            elif kind == "error":
                lines += [f"- `{t}` **error**: {r.get('message')}"]
        lines += ["", "## Token use", "", f"{tokens_in:,} input · {tokens_out:,} output", "",
                  "## Reproduce", "",
                  "The successful tool calls, in order — save as a flow to replay:", "",
                  "```json", json.dumps(self.tool_calls(), indent=1, default=str), "```", ""]
        return "\n".join(lines)
