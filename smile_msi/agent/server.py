"""``smile-msi chat`` — the plain-language analysis chat, served to your browser from this
machine.

A small standard-library HTTP server (no web framework to install) that hosts one chat page
and runs the :class:`~smile_msi.agent.core.Agent` in its own process — separate from the
desktop app, so the chat never competes with the app's open slide or ion cube.

Security, because the agent can run code: the server binds to ``127.0.0.1`` only, checks the
``Host`` header (blocks DNS-rebinding pages), and every API call must carry a random token
minted at launch and handed to the browser in the URL it opens — another web page can't
drive the agent. The API key typed into Settings lives in this process's memory only; it is
never written to the log or to disk.

Endpoints (all but ``/`` need the token, ``?t=`` or the ``X-SMILE-Token`` header):

    GET  /                  the chat page
    GET  /api/events        server-sent events: the session's events so far, then live
    POST /api/send          {"text"}  start a turn
    POST /api/stop          stop the running turn after its current step
    POST /api/approve       {"call_id", "approve", "note"}  answer an approval request
    GET  /api/config        current settings (``has_key``, never the key)
    POST /api/config        new settings → a new conversation (and a new log)
    POST /api/reset         new conversation, same settings
    GET  /api/library       user-created tools and saved flows
    GET  /api/file?p=…      an image/CSV the session produced (only those)
    GET  /api/log.md        the session log as Markdown; /api/log.jsonl raw
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import queue
import secrets
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .core import Agent
from .providers import DEFAULT_CLAUDE_MODEL, ProviderError, make_provider

_STATIC = os.path.join(os.path.dirname(__file__), "static")


class ChatState:
    """The one live conversation plus everything the HTTP handlers share."""

    def __init__(self, config: dict):
        self.config = dict(config)
        self.lock = threading.Lock()
        self.events: list[dict] = []
        self.listeners: list[queue.Queue] = []
        self.files: set[str] = set()
        self.pending: dict[str, dict] = {}
        self.agent: Agent | None = None
        self.config_error = ""
        self.new_conversation()

    # ------------------------------------------------------------------ #
    def publish(self, ev: dict):
        for key in ("images", "files"):
            for p in ev.get(key) or ():
                self.files.add(os.path.abspath(p))
        with self.lock:
            self.events.append(ev)
            listeners = list(self.listeners)
        for q in listeners:
            q.put(ev)

    def approve(self, req: dict):
        slot = {"event": threading.Event(), "answer": (False, "")}
        self.pending[req["call_id"]] = slot
        while not slot["event"].wait(0.5):
            if self.agent is not None and self.agent._stop.is_set():
                self.pending.pop(req["call_id"], None)
                return False, "stopped by the user"
        self.pending.pop(req["call_id"], None)
        return slot["answer"]

    def answer(self, call_id: str, ok: bool, note: str) -> bool:
        slot = self.pending.get(call_id)
        if slot is None:
            return False
        slot["answer"] = (bool(ok), str(note or ""))
        slot["event"].set()
        return True

    def new_conversation(self):
        if self.agent is not None:
            self.agent.stop()
            for slot in list(self.pending.values()):
                slot["event"].set()
            self.agent.log.close()
        with self.lock:
            self.events = []
            listeners = list(self.listeners)
        for q in listeners:                      # open pages clear their transcript
            q.put({"type": "reset"})
        self.agent, self.config_error = None, ""
        try:
            provider = make_provider(self.config)
            self.agent = Agent(provider, emit=self.publish, approve=self.approve)
        except ProviderError as exc:
            self.config_error = str(exc)
            self.publish({"type": "error", "message": str(exc), "seq": 0})

    def public_config(self) -> dict:
        cfg = {k: v for k, v in self.config.items() if k != "api_key"}
        cfg["has_key"] = bool(self.config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY"))
        cfg["error"] = self.config_error
        if self.agent is not None:
            cfg["log_path"] = self.agent.log.path
        return cfg


def make_handler(state: ChatState, token: str, port: int):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "SMILE-MSI-chat"

        def log_message(self, *a):          # keep the terminal quiet
            pass

        # -------------------------------------------------------------- #
        def _deny(self, code=HTTPStatus.FORBIDDEN, msg="forbidden"):
            self._send(code, {"error": msg})

        def _send(self, code, obj=None, *, body: bytes | None = None, ctype="application/json"):
            data = body if body is not None else json.dumps(obj, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _ok_origin(self, q) -> bool:
            if self.headers.get("Host", "") not in allowed_hosts:
                return False
            supplied = self.headers.get("X-SMILE-Token") or (q.get("t") or [""])[0]
            return secrets.compare_digest(supplied, token)

        def _json_body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 2_000_000:
                raise ValueError("request too large")
            raw = self.rfile.read(n) if n else b"{}"
            return json.loads(raw.decode("utf-8") or "{}")

        # -------------------------------------------------------------- #
        def do_GET(self):  # noqa: N802 — http.server API
            url = urlparse(self.path)
            q = parse_qs(url.query)
            if url.path in ("/", "/index.html"):
                if self.headers.get("Host", "") not in allowed_hosts:
                    return self._deny()
                with open(os.path.join(_STATIC, "chat.html"), "rb") as fh:
                    return self._send(HTTPStatus.OK, body=fh.read(),
                                      ctype="text/html; charset=utf-8")
            if not self._ok_origin(q):
                return self._deny()
            if url.path == "/api/events":
                return self._stream()
            if url.path == "/api/config":
                return self._send(HTTPStatus.OK, state.public_config())
            if url.path == "/api/library":
                return self._library()
            if url.path == "/api/file":
                path = os.path.abspath((q.get("p") or [""])[0])
                if path not in state.files or not os.path.isfile(path):
                    return self._deny(HTTPStatus.NOT_FOUND, "not a file from this session")
                with open(path, "rb") as fh:
                    ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
                    return self._send(HTTPStatus.OK, body=fh.read(), ctype=ctype)
            if url.path in ("/api/log.md", "/api/log.jsonl"):
                if state.agent is None:
                    return self._deny(HTTPStatus.CONFLICT, "no conversation")
                log = state.agent.log
                if url.path.endswith(".md"):
                    return self._send(HTTPStatus.OK, body=log.markdown().encode("utf-8"),
                                      ctype="text/markdown; charset=utf-8")
                with open(log.path, "rb") as fh:
                    return self._send(HTTPStatus.OK, body=fh.read(),
                                      ctype="application/x-ndjson")
            return self._deny(HTTPStatus.NOT_FOUND, "unknown endpoint")

        def do_POST(self):  # noqa: N802
            url = urlparse(self.path)
            if not self._ok_origin(parse_qs(url.query)):
                return self._deny()
            try:
                body = self._json_body()
            except (ValueError, json.JSONDecodeError) as exc:
                return self._deny(HTTPStatus.BAD_REQUEST, str(exc))
            agent = state.agent
            if url.path == "/api/send":
                text = str(body.get("text") or "").strip()
                if agent is None:
                    return self._deny(HTTPStatus.CONFLICT,
                                      state.config_error or "set up a model in Settings first")
                if not text:
                    return self._deny(HTTPStatus.BAD_REQUEST, "empty message")
                if agent.busy:
                    return self._deny(HTTPStatus.CONFLICT, "still working on the last message")
                agent.busy = True
                threading.Thread(target=agent.send, args=(text,), daemon=True).start()
                return self._send(HTTPStatus.ACCEPTED, {"ok": True})
            if url.path == "/api/stop":
                if agent is not None:
                    agent.stop()
                return self._send(HTTPStatus.OK, {"ok": True})
            if url.path == "/api/approve":
                ok = state.answer(str(body.get("call_id")), bool(body.get("approve")),
                                  body.get("note", ""))
                return self._send(HTTPStatus.OK if ok else HTTPStatus.NOT_FOUND, {"ok": ok})
            if url.path == "/api/config":
                if agent is not None and agent.busy:
                    return self._deny(HTTPStatus.CONFLICT, "stop the running turn first")
                new = {k: body[k] for k in ("provider", "model", "base_url", "effort", "vision")
                       if k in body}
                if body.get("api_key"):
                    new["api_key"] = str(body["api_key"]).strip()
                if new.get("provider") and new["provider"] != state.config.get("provider"):
                    state.config.pop("model", None)
                state.config.update(new)
                state.new_conversation()
                return self._send(HTTPStatus.OK, state.public_config())
            if url.path == "/api/reset":
                if agent is not None and agent.busy:
                    return self._deny(HTTPStatus.CONFLICT, "stop the running turn first")
                state.new_conversation()
                return self._send(HTTPStatus.OK, {"ok": True})
            return self._deny(HTTPStatus.NOT_FOUND, "unknown endpoint")

        # -------------------------------------------------------------- #
        def _library(self):
            agent = state.agent
            tools_ = [{"name": s.name, "description": s.description.split("\n")[0],
                       "version": s.meta.get("version")}
                      for s in (agent.registry if agent else []) if s.source == "custom"]
            flows = agent.toolbox.list_flows() if agent else []
            builtin = [{"name": s.name, "title": s.title}
                       for s in (agent.registry if agent else []) if s.source == "builtin"]
            return self._send(HTTPStatus.OK, {"tools": tools_, "flows": flows,
                                              "builtin": builtin})

        def _stream(self):
            q: queue.Queue = queue.Queue()
            with state.lock:
                backlog = list(state.events)
                state.listeners.append(q)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                self.wfile.write(b"event: reset\ndata: {}\n\n")
                for ev in backlog:
                    self.wfile.write(f"data: {json.dumps(ev, default=str)}\n\n".encode())
                self.wfile.flush()
                while True:
                    try:
                        ev = q.get(timeout=15)
                        self.wfile.write(f"data: {json.dumps(ev, default=str)}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with state.lock:
                    if q in state.listeners:
                        state.listeners.remove(q)

    return Handler


def build_server(config: dict, port: int = 8765):
    """The configured server (not yet serving) → ``(httpd, state, url)``. ``port=0`` picks a
    free port."""
    token = secrets.token_urlsafe(24)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), None)
    port = httpd.server_address[1]
    state = ChatState(config)
    state.token = token
    httpd.RequestHandlerClass = make_handler(state, token, port)
    return httpd, state, f"http://127.0.0.1:{port}/?t={token}"


def serve(config: dict, port: int = 8765, open_browser: bool = True):
    """Run the chat server until Ctrl-C, printing the URL to open."""
    httpd, state, url = build_server(config, port)
    print(f"SMILE MSI chat running — open:\n  {url}\nPress Ctrl-C to stop.", flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        if state.agent is not None:
            state.agent.log.close()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="smile-msi chat",
                                 description="Plain-language MSI analysis in your browser.")
    ap.add_argument("--provider", choices=["claude", "local"], default="claude",
                    help="claude (API key or ant login) or local (OpenAI-compatible server)")
    ap.add_argument("--model", default="", help=f"model name (Claude default: "
                                                f"{DEFAULT_CLAUDE_MODEL})")
    ap.add_argument("--base-url", default="", help="local server URL "
                                                   "(default http://localhost:11434/v1)")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"],
                    help="Claude reasoning effort")
    ap.add_argument("--vision", action="store_true", help="local model can read images")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    cfg = {"provider": a.provider, "model": a.model, "base_url": a.base_url,
           "effort": a.effort, "vision": a.vision}
    serve(cfg, port=a.port, open_browser=not a.no_browser)


if __name__ == "__main__":
    main()
