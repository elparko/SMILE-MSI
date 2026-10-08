"""The agent loop — plain-language requests in, tool calls and figures out, everything logged.

:class:`Agent` sends the user's message to the provider, runs every tool call the model
makes (all results for one step go back together, as the providers require), and repeats
until the model answers without calling a tool, hits ``max_steps``, or the user presses Stop.
Each thing that happens is written to the :class:`~smile_msi.agent.log.SessionLog` *and*
passed to ``emit`` so a front end can show it live — the model's text, its reasoning summary,
each tool call as it starts and finishes, and every image a tool produced.

Tools flagged ``approval`` (create / edit / delete a tool) block on ``approve(request)`` — the
front end shows the code and the user decides; a rejection goes back to the model as the
tool's error so it can adjust.
"""
from __future__ import annotations

import threading
import time

from . import custom, tools
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

Scientific standards (a reviewer will read this analysis)
- Separate what was measured from what you infer. Mark interpretations and biological \
context as such, and say how confident you are.
- Lipid identities from m/z alone are putative and at sum-composition level (e.g. \
"putatively PC 34:1, [M+H]+, 1.8 ppm"); never present them as confirmed without MS/MS.
- Report effect sizes and the test used, not just p-values; correct for multiple testing; \
remember that pixels are not independent replicates — say so when n is pixels.
- Flag likely artefacts: matrix peaks, isotopes/adducts of another feature, \
normalisation-driven differences, edge effects, low signal.

Tools
- Start with list_slides / slide_state to see the scientist's saved work, or \
open_slide('demo') for a synthetic slide. Read scripting_guide before writing run_script \
code. Tables come back as a preview plus a CSV path.
- If no tool fits and the need will recur, you may create one with create_tool (the \
scientist approves the code first). Prefer existing tools; write new ones general \
(parameters, not hard-coded values) and test them right after approval.
- When an analysis is worth repeating, offer to save it with save_flow.

Background on the engine: """


def system_prompt() -> str:
    from .. import mcpserver

    return SYSTEM_PROMPT + mcpserver.INSTRUCTIONS


class Agent:
    """One conversation. ``emit(event: dict)`` receives live events; ``approve(request)``
    returns ``(approved: bool, note: str)`` and may block until the user decides."""

    def __init__(self, provider, *, emit=None, approve=None, log: SessionLog | None = None,
                 registry: tools.Registry | None = None, max_steps: int = 40):
        self.provider = provider
        self.emit = emit or (lambda ev: None)
        self.approve = approve or (lambda req: (False, "no approver connected"))
        self.log = log or SessionLog()
        self.registry = registry or tools.Registry(tools.builtin_tools())
        self.toolbox = custom.Toolbox(self.registry)
        self.max_steps = int(max_steps)
        self._stop = threading.Event()
        self.busy = False
        self._tool_names = tuple(self.registry.names())
        provider.start(system_prompt(), self.registry.api_defs())
        from .. import __version__

        self._event("session", provider=provider.name, version=__version__,
                    **provider.settings())

    # ------------------------------------------------------------------ #
    def _event(self, type_: str, **data) -> dict:
        rec = self.log.write(type_, **data)
        self.emit(rec)
        return rec

    def stop(self):
        """Ask the running turn to stop after the current step."""
        self._stop.set()

    def send(self, text: str):
        """Run one user turn to completion (blocking — call from a worker thread)."""
        self.busy = True
        self._stop.clear()
        try:
            self._event("user", text=text)
            self.provider.add_user(text)
            for _ in range(self.max_steps):
                self._sync_tools()
                self._event("status", state="thinking")
                step = self.provider.step()
                self._event("usage", **step.usage)
                if step.thinking:
                    self._event("thinking", text=step.thinking)
                if step.text:
                    self._event("assistant", text=step.text)
                if step.note:
                    self._event("notice", text=step.note)
                if not step.calls:
                    break
                outputs = [self._run_call(c) for c in step.calls]
                self.provider.add_tool_outputs(outputs)
                if self._stop.is_set():
                    self._event("notice", text="Stopped — the results so far are kept.")
                    break
            else:
                self._event("notice", text=f"Paused after {self.max_steps} steps. "
                                           "Say 'continue' to keep going.")
        except ProviderError as exc:
            self._event("error", message=str(exc))
        except Exception as exc:  # noqa: BLE001 — surface, never kill the server thread
            self._event("error", message=f"{type(exc).__name__}: {exc}")
        finally:
            self.busy = False
            self._event("done")

    def _sync_tools(self):
        """A tool created or deleted last step changes what the model can call next."""
        names = tuple(self.registry.names())
        if names != self._tool_names:
            self._tool_names = names
            self.provider.set_tools(self.registry.api_defs())

    # ------------------------------------------------------------------ #
    def _run_call(self, call) -> ToolOutput:
        self._event("tool_call", call_id=call.id, name=call.name, args=call.args)
        spec = self.registry.get(call.name)
        t0 = time.perf_counter()

        def fail(msg):
            self._event("tool_result", call_id=call.id, name=call.name, ok=False, error=msg,
                        duration_s=time.perf_counter() - t0, text=msg)
            return ToolOutput(call.id, f"ERROR: {msg}", is_error=True)

        if call.parse_error:
            return fail(call.parse_error)
        if spec is None:
            return fail(f"no tool named {call.name!r}")
        if spec.approval:
            req = self._approval_request(spec, call)
            self._event("approval_request", **req)
            ok, note = self.approve(req)
            self._event("approval", call_id=call.id, tool=call.name, target=req["target"],
                        decision="approved" if ok else "rejected", note=note,
                        sha256=req.get("sha256", ""))
            if not ok:
                return fail("the scientist rejected this" + (f": {note}" if note else "") +
                            ". Ask what they would like instead.")
        try:
            result = spec.fn(**call.args)
        except Exception as exc:  # noqa: BLE001 — tool errors go back to the model
            return fail(f"{type(exc).__name__}: {exc}")
        rendered = tools.render(result)
        self._event("tool_result", call_id=call.id, name=call.name, ok=True, text=rendered.text,
                    images=rendered.images, files=rendered.files,
                    duration_s=time.perf_counter() - t0)
        if call.name == "open_slide":
            self._log_dataset(call.args.get("ref", "demo"))
        return ToolOutput(call.id, rendered.text, images=rendered.images)

    @staticmethod
    def _approval_request(spec, call) -> dict:
        import hashlib

        code = call.args.get("code") or ""
        return {"call_id": call.id, "tool": call.name, "title": spec.title,
                "target": call.args.get("name", ""),
                "description": call.args.get("description", ""),
                "parameters": call.args.get("parameters"),
                "note": call.args.get("note", ""), "code": code,
                "sha256": hashlib.sha256(code.encode()).hexdigest() if code else ""}

    def _log_dataset(self, ref):
        from .. import library, mcpserver

        slide = mcpserver._open.get("slide")
        if slide is None:
            return
        try:
            fp = library.dataset_fingerprint(slide.ds)
        except Exception:  # noqa: BLE001 — a fingerprint is provenance, not a blocker
            fp = "unavailable"
        self._event("dataset", ref=str(ref), source=slide.source, fingerprint=fp,
                    n_pixels=int(slide.ds.n_pixels))
