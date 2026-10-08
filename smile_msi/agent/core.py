"""The agent loop — plain-language requests in, tool calls and figures out, everything logged.

:class:`Agent` sends the user's message to the provider, runs every tool call the model
makes (all results for one step go back together, as the providers require), and repeats
until the model answers without calling a tool, hits ``max_steps``, or the user presses Stop.
Each thing that happens is written to the :class:`~smile_msi.agent.log.SessionLog` *and*
passed to ``emit`` so a front end can show it live — the model's text, its reasoning summary,
each tool call as it starts and finishes, and every image a tool produced.

**Every tool call gets exactly one result.** Model APIs reject a conversation in which a
tool call was never answered, and every later request would fail — so whatever goes wrong
between the model asking and the result going back (a tool raising anything, even
``SystemExit``; a result that can't be rendered; a malformed approval request) becomes an
error *result* for that call, and a turn that dies mid-step still answers every call first.

Tools flagged ``approval`` (create / edit / delete a tool, and ``run_script`` when the user
turns that on) block on ``approve(request)`` after a pre-check — a request that would fail
anyway (bad name, duplicate, syntax error) is refused without bothering the user. A rejection
goes back to the model as the tool's error so it can adjust.

The approval gate is a review step, not a sandbox: ``run_script`` runs ordinary Python with
the user's permissions unless the user requires approval for it too.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time

from . import custom, tools
from . import setup as setup_mod
from .log import SessionLog
from .providers import ProviderError, ToolOutput

SYSTEM_PROMPT = """You are the analysis assistant inside SMILE MSI, a desktop workspace for \
MALDI mass-spectrometry imaging (MSI) of lipids. You work with a scientist in plain \
language: they say what they want to learn; you run the analysis with your tools, show \
them what you did, and explain the result.

How to work
- Use the tools for every number, image and identification. Never state a measured value, \
statistic or lipid assignment that did not come from a tool result in this conversation.
- Show, don't just tell: when an image or table helps the scientist decide something, \
produce it (ion_image, run_script with image()/table(), show_file). Keep figures purposeful.
- Before a long or consequential analysis (many steps, choosing regions or groups to \
compare, statistics that will be reported), say in a sentence or two what you plan to do \
and why, then do it. Ask when a choice is genuinely the scientist's (which regions are \
which tissue, which comparison matters) rather than guessing.
- Keep the scientist oriented: short progress notes between steps, a clear summary at the \
end with what was done, what was found, and what is still uncertain.
- A tool result with "ok": false or an "error" failed — say so; don't build on it.

Scientific standards (a reviewer will read this analysis)
- Separate what was measured from what you infer. Mark interpretations and biological \
context as such, and say how confident you are.
- Lipid identities from m/z alone are putative and at sum-composition level (e.g. \
"putatively PC 34:1, [M+H]+, 1.8 ppm"); never present them as confirmed without MS/MS.
- Report effect sizes and the test used, not just p-values; correct for multiple testing. \
Pixels are not independent replicates: with one region per group, p-values describe pixels, \
not biological replication — say so, and lean on effect sizes.
- Flag likely artefacts: matrix peaks, isotopes/adducts of another feature, \
normalisation-driven differences, edge effects, off-tissue background, low signal.

Tools
- Start with list_slides / slide_state to see the scientist's saved work, or \
open_slide('demo') for a synthetic slide. Read scripting_guide before writing run_script \
code. Tables come back as a preview plus a CSV path. Save figures with image() rather than \
writing files yourself.
- If no tool fits and the need will recur, you may create one with create_tool (the \
scientist approves the code first). Prefer existing tools; write new ones general \
(parameters, not hard-coded values) and test them right after approval.
- When an analysis is worth repeating, offer to save it with save_flow.

Background on the engine: """


def system_prompt() -> str:
    from .. import mcpserver

    return SYSTEM_PROMPT + mcpserver.INSTRUCTIONS


class ToolInterrupted(BaseException):
    """Raised inside a running tool when the user stops or abandons the turn. A
    BaseException so analysis code (and scripts) that catch ``Exception`` can't swallow it."""


def _interrupt_thread(thread_id: int) -> bool:
    """Raise :class:`ToolInterrupted` in another thread at its next Python instruction (a
    pure-Python loop stops at once; a long C call stops when it returns)."""
    import ctypes

    n = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread_id),
                                                   ctypes.py_object(ToolInterrupted))
    if n > 1:                                   # should never happen: undo, report failure
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread_id), None)
        return False
    return n == 1


#: Seconds a tool gets to finish on its own after Stop before it is interrupted.
STOP_GRACE_S = 1.5


def _failed(result) -> str:
    """The error message of a tool result that reports its own failure, else ``""``."""
    if isinstance(result, dict):
        if result.get("ok") is False:
            return str(result.get("error") or "the tool reported ok: false")
        if result.get("completed") is False:                      # a flow that stopped
            bad = [s for s in result.get("steps", []) if not s.get("ok")]
            return f"flow stopped at step {bad[0]['step']}: {bad[0].get('error')}" if bad \
                else "flow did not complete"
    return ""


class Agent:
    """One conversation. ``emit(event: dict)`` receives live events; ``approve(request)``
    returns ``(approved: bool, note: str)`` and may block until the user decides."""

    def __init__(self, provider, *, emit=None, approve=None, log: SessionLog | None = None,
                 registry: tools.Registry | None = None, max_steps: int = 40,
                 approve_scripts: bool = False, setup: dict | None = None):
        self.provider = provider
        self.emit = emit or (lambda ev: None)
        self.approve = approve or (lambda req: (False, "no approver connected"))
        self.log = log or SessionLog()
        self.registry = registry or tools.Registry(tools.builtin_tools())
        if approve_scripts and "run_script" in self.registry:
            self.registry.get("run_script").approval = True
        self.toolbox = custom.Toolbox(self.registry)
        self.max_steps = int(max_steps)
        self._stop = threading.Event()
        self.busy = False
        self._tool_lock = threading.Lock()
        self._tool_thread: int | None = None      # thread running a tool right now
        self._tool_sig = self._signature()
        self._dataset_seen = None
        self.setup: dict | None = None
        self._model_note = ""                # setup brief/diff prefixed to the next message
        provider.start(system_prompt(), self.registry.api_defs())
        from .. import __version__

        self._event("session", provider=provider.name, version=__version__,
                    approve_scripts=bool(approve_scripts), **provider.settings())
        if setup is not None:
            self.set_setup(setup)

    # ------------------------------------------------------------------ #
    def set_setup(self, setup: dict, how: str = "confirmed by the scientist"):
        """Lock (or change) the analysis setup: log it, apply it to the open slide, and
        tell the model with the next message."""
        old = self.setup
        self.setup = setup
        self._event("setup", how=how, **setup)
        if old is None:
            self._model_note = setup_mod.brief(setup)
        else:
            note = setup_mod.diff_brief(old, setup)
            self._model_note = (self._model_note + "\n\n" + note).strip() if note \
                else self._model_note
        self._apply_setup_to_slide()

    def _apply_setup_to_slide(self):
        from .. import mcpserver

        slide = mcpserver._open.get("slide")
        if slide is None or self.setup is None:
            return
        changes = setup_mod.apply_to_slide(self.setup, slide)
        if changes:
            self._event("setup_applied", ref=str(mcpserver._open.get("ref", "")),
                        changes={k: {"was": a, "now": b} for k, (a, b) in changes.items()})

    # ------------------------------------------------------------------ #
    def _event(self, type_: str, **data) -> dict:
        rec = self.log.write(type_, **data)
        self.emit(rec)
        return rec

    def stop(self):
        """Stop the running turn: it ends after the current step, and a tool still running
        after a short grace period is interrupted (so a runaway script can't keep the
        process busy)."""
        self._stop.set()
        with self._tool_lock:
            running = self._tool_thread
        if running is not None:
            def later():
                with self._tool_lock:
                    if self._tool_thread == running:
                        _interrupt_thread(running)
            t = threading.Timer(STOP_GRACE_S, later)
            t.daemon = True
            t.start()

    def send(self, text: str):
        """Run one user turn to completion (blocking — call from a worker thread)."""
        self.busy = True
        self._stop.clear()
        try:
            if self.setup is None:                 # never analyse without a recorded method
                self.set_setup(setup_mod.resolve(),
                               how="defaults accepted without review (the setup card was "
                                   "skipped)")
            self._event("user", text=text)
            note, self._model_note = self._model_note, ""
            self.provider.add_user(f"{note}\n\n{text}" if note else text)
            for _ in range(self.max_steps):
                self._sync_tools()
                self._event("status", state="thinking")
                step = self.provider.step()
                if self._answer_step(step):
                    break
            else:
                self._event("notice", text=f"Paused after {self.max_steps} steps. "
                                           "Say 'continue' to keep going.")
        except ProviderError as exc:
            self._event("error", message=str(exc))
        except BaseException as exc:  # noqa: BLE001 — surface, never kill the server thread
            self._event("error", message=f"{type(exc).__name__}: {exc}")
            if isinstance(exc, KeyboardInterrupt):
                raise
        finally:
            self.busy = False
            self._event("done")

    def _answer_step(self, step) -> bool:
        """Report one model step and run its tool calls. Returns True when the turn ends.
        Whatever happens, every call the model made is answered before this returns."""
        outputs: dict[str, ToolOutput] = {}
        try:
            self._event("usage", **step.usage)
            if step.thinking:
                self._event("thinking", text=step.thinking)
            if step.text:
                self._event("assistant", text=step.text)
            if step.note:
                self._event("notice", text=step.note)
            for call in step.calls:
                outputs[call.id] = self._run_call(call)
        finally:
            if step.calls:
                for call in step.calls:                # a dying step still answers every call
                    if call.id not in outputs:
                        outputs[call.id] = ToolOutput(call.id, "ERROR: not run — the turn "
                                                      "was interrupted", is_error=True)
                self.provider.add_tool_outputs([outputs[c.id] for c in step.calls])
        if not step.calls:
            return True
        if self._stop.is_set():
            self._event("notice", text="Stopped — the results so far are kept.")
            return True
        return False

    def _signature(self) -> str:
        return json.dumps(self.registry.api_defs(), sort_keys=True, default=str)

    def _sync_tools(self):
        """A tool created, edited or deleted last step changes what the model can call."""
        sig = self._signature()
        if sig != self._tool_sig:
            self._tool_sig = sig
            self.provider.set_tools(self.registry.api_defs())

    # ------------------------------------------------------------------ #
    def _run_call(self, call) -> ToolOutput:
        t0 = time.perf_counter()
        try:
            self._event("tool_call", call_id=call.id, name=call.name,
                        args=call.args if isinstance(call.args, dict) else repr(call.args))
        except BaseException:  # noqa: BLE001 — logging must not lose the call
            pass

        def fail(msg):
            try:
                self._event("tool_result", call_id=call.id, name=call.name, ok=False,
                            error=msg, duration_s=time.perf_counter() - t0, text=msg)
            except BaseException:  # noqa: BLE001
                pass
            return ToolOutput(call.id, f"ERROR: {msg}", is_error=True)

        try:
            spec = self.registry.get(call.name)
            if call.parse_error:
                return fail(call.parse_error)
            if spec is None:
                return fail(f"no tool named {call.name!r}")
            if not isinstance(call.args, dict):
                return fail(f"arguments must be a JSON object, got {type(call.args).__name__}")
            if spec.approval:
                problem = self.toolbox.precheck(call.name, call.args)
                if problem:
                    return fail(problem)
                req = self._approval_request(spec, call)
                self._event("approval_request", **req)
                ok, note = self.approve(req)
                self._event("approval", call_id=call.id, tool=call.name, target=req["target"],
                            decision="approved" if ok else "rejected", note=note,
                            sha256=req.get("sha256", ""))
                if not ok:
                    return fail("the scientist rejected this" + (f": {note}" if note else "")
                                + ". Ask what they would like instead.")
            args, deviations = (setup_mod.fill_defaults(self.setup, call.name, call.args)
                                if self.setup else (call.args, []))
            if deviations:
                self._event("deviation", call_id=call.id, name=call.name,
                            deviations=deviations)
            with self._tool_lock:
                self._tool_thread = threading.get_ident()
            try:
                result = spec.fn(**args)
            finally:
                with self._tool_lock:
                    self._tool_thread = None
            rendered = tools.render(result)
            err = _failed(result)
            flow_steps = None
            if call.name == "run_flow" and isinstance(result, dict):   # for the Reproduce list
                flow_steps = [{"tool": st["tool"], "args": st.get("args", {})}
                              for st in result.get("steps", []) if st.get("ok")]
            self._event("tool_result", call_id=call.id, name=call.name, ok=not err,
                        error=err or None, text=rendered.text, images=rendered.images,
                        files=rendered.files, duration_s=time.perf_counter() - t0,
                        effective_args=args if args != call.args else None,
                        flow_steps=flow_steps)
            self._check_dataset()
            return ToolOutput(call.id, rendered.text, images=rendered.images, is_error=bool(err))
        except KeyboardInterrupt:
            raise
        except ToolInterrupted:
            self._check_dataset()
            return fail("interrupted — the scientist stopped this tool")
        except BaseException as exc:  # noqa: BLE001 — incl. SystemExit from user scripts
            msg = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            self._check_dataset()
            return fail(msg)

    @staticmethod
    def _approval_request(spec, call) -> dict:
        a = call.args
        code = a.get("code") if isinstance(a.get("code"), str) else ""
        return {"call_id": call.id, "tool": call.name, "title": spec.title,
                "target": str(a.get("name", "")),
                "description": str(a.get("description", "")),
                "parameters": a.get("parameters"), "note": str(a.get("note", "")),
                "code": code,
                "sha256": hashlib.sha256(code.encode()).hexdigest() if code else ""}

    def _check_dataset(self):
        """Log the open slide's fingerprint whenever it changes — however it was opened
        (a tool, a flow, a script) and also when a conversation starts on an open slide."""
        from .. import library, mcpserver

        slide = mcpserver._open.get("slide")
        if slide is None or slide is self._dataset_seen:
            return
        self._dataset_seen = slide
        self._apply_setup_to_slide()
        try:
            fp = library.dataset_fingerprint(slide.ds)
        except Exception:  # noqa: BLE001 — a fingerprint is provenance, not a blocker
            fp = "unavailable"
        self._event("dataset", ref=str(mcpserver._open.get("ref", "")), source=slide.source,
                    fingerprint=fp, n_pixels=int(slide.ds.n_pixels),
                    cwd=os.getcwd())
