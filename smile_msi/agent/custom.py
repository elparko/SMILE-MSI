"""User-created tools and saved flows — how the agent grows new capabilities on the fly.

**Tools.** A model (or the user) writes a short analysis script plus a declared parameter
list; :meth:`Toolbox.create_tool` saves it as a named, versioned tool and registers it at once,
so the very next turn can call it like any built-in. The script runs in the same namespace as
``run_script`` / the app's Script Console (``ds``, the analysis functions, ``log`` / ``table``
/ ``image`` / ``record``), with each parameter bound as a variable (and all of them as the
dict ``params``). Creating, editing and deleting a tool **require the user's approval** — they
are how the model adds code to its toolbox; running a tool once approved does not.

The approval gate is a review step for the model's code, not a sandbox: ``run_script``
(unless the user requires approval for it) runs ordinary Python with the user's permissions
and could write files anywhere — so tools found on disk never replace a built-in or agent
tool, and the code of every version is hashed into the session log when it runs.

Each tool lives in ``<home>/agent/tools/<name>/``: ``tool.json`` (description, parameters,
current version, a sha256 per version) and ``v1.py``, ``v2.py``, … — every version is kept,
so a result logged against ``v2`` can always be re-run with exactly that code.

**Flows.** A flow is a saved list of tool calls (``[{"tool": …, "args": {…}}, …]``) with
optional ``{{name}}`` placeholders — a recorded analysis that replays deterministically on
another slide. Flows only compose existing tools, so saving one needs no approval.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re

from .tools import Registry, ToolSpec

_NAME = re.compile(r"^[a-z][a-z0-9_]{2,47}$")
_JSON_TYPES = {"number", "integer", "string", "boolean", "array", "object"}
_PLACEHOLDER = re.compile(r"^\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}$")
_EMBEDDED = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
#: Tools a flow may not contain: they change tools/flows or would recurse.
_NOT_IN_FLOWS = {"create_tool", "edit_tool", "delete_tool", "save_flow", "run_flow"}


def _check_name(name, what="tool") -> str:
    name = str(name).strip()
    if not _NAME.match(name):
        raise ValueError(f"{what} name must be snake_case, 3–48 chars, starting with a letter")
    return name


def _coerce(pname: str, value, prop: dict):
    """Check ``value`` against a declared parameter type (local models often send numbers
    as strings — those are converted; anything that doesn't fit is an error)."""
    t = prop.get("type", "string")
    if t == "number":
        if isinstance(value, bool):
            raise ValueError(f"parameter {pname!r} must be a number, got {value!r}")
        if isinstance(value, (int, float)):
            return value
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(f"parameter {pname!r} must be a number, got {value!r}") from None
    if t == "integer":
        if isinstance(value, bool):
            raise ValueError(f"parameter {pname!r} must be an integer, got {value!r}")
        try:
            f = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"parameter {pname!r} must be an integer, got {value!r}") from None
        if f != int(f):
            raise ValueError(f"parameter {pname!r} must be an integer, got {value!r}")
        return int(f)
    if t == "boolean":
        if isinstance(value, bool):
            return value
        if str(value).lower() in ("true", "false"):
            return str(value).lower() == "true"
        raise ValueError(f"parameter {pname!r} must be true or false, got {value!r}")
    if t == "string":
        if isinstance(value, (dict, list)):
            raise ValueError(f"parameter {pname!r} must be text, got {type(value).__name__}")
        return str(value)
    if t == "array":
        if not isinstance(value, list):
            raise ValueError(f"parameter {pname!r} must be a list, got {type(value).__name__}")
        return value
    if t == "object":
        if not isinstance(value, dict):
            raise ValueError(f"parameter {pname!r} must be an object, got "
                             f"{type(value).__name__}")
        return value
    return value


def _substitute(value, params: dict):
    """Fill ``{{name}}`` placeholders: a whole-value placeholder keeps the parameter's type;
    one embedded in text (e.g. inside run_script code) is replaced by its text form."""
    if isinstance(value, str):
        m = _PLACEHOLDER.match(value)
        if m:
            return params[m.group(1)]
        return _EMBEDDED.sub(lambda mm: str(params[mm.group(1)]), value)
    if isinstance(value, dict):
        return {k: _substitute(v, params) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, params) for v in value]
    return value


def _placeholders_in(value, out: set):
    if isinstance(value, str):
        out.update(_EMBEDDED.findall(value))
    elif isinstance(value, dict):
        for v in value.values():
            _placeholders_in(v, out)
    elif isinstance(value, list):
        for v in value:
            _placeholders_in(v, out)


def agent_dir(*parts) -> str:
    from .. import library

    d = os.path.join(library.home_dir(), "agent", *parts)
    os.makedirs(d, exist_ok=True)
    return d


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _sha(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _param_schema(parameters: dict | None) -> dict:
    """``{"name": {"type", "description", "default"?}}`` → a JSON schema (no default ⇒
    required). Rejects unknown types so a typo fails at creation, not at first use."""
    props, required = {}, []
    for pname, spec in (parameters or {}).items():
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(pname)):
            raise ValueError(f"parameter name {pname!r} is not a valid Python identifier")
        spec = dict(spec or {})
        t = spec.get("type", "string")
        if t not in _JSON_TYPES:
            raise ValueError(f"parameter {pname!r}: type must be one of {sorted(_JSON_TYPES)}")
        prop = {"type": t}
        if spec.get("description"):
            prop["description"] = str(spec["description"])
        if "default" in spec:
            prop["default"] = spec["default"]
        else:
            required.append(pname)
        props[pname] = prop
    out = {"type": "object", "properties": props}
    if required:
        out["required"] = required
    return out


class Toolbox:
    """Owns the user-created tools and flows on disk and keeps ``registry`` in sync."""

    def __init__(self, registry: Registry):
        self.registry = registry
        self.tools_dir = agent_dir("tools")
        self.flows_dir = agent_dir("flows")
        for spec in self.agent_specs():
            registry.add(spec)
        for name in sorted(os.listdir(self.tools_dir)):      # skips <name>.deleted-<time>
            if not (_NAME.match(name) and os.path.isfile(os.path.join(self.tools_dir, name,
                                                                      "tool.json"))):
                continue
            if name in registry:                 # never let a file on disk shadow a built-in
                continue
            try:
                registry.add(self._spec_for(self._meta(name)))
            except (OSError, ValueError, KeyError):  # a damaged tool folder: skip, don't crash
                continue

    # ------------------------------------------------------------------ #
    # custom tools
    # ------------------------------------------------------------------ #
    def _meta(self, name: str) -> dict:
        with open(os.path.join(self.tools_dir, name, "tool.json"), encoding="utf-8") as fh:
            return json.load(fh)

    def _code(self, name: str, version: int) -> str:
        with open(os.path.join(self.tools_dir, name, f"v{version}.py"), encoding="utf-8") as fh:
            return fh.read()

    def _write(self, meta: dict, code: str):
        d = os.path.join(self.tools_dir, meta["name"])
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"v{meta['version']}.py"), "w", encoding="utf-8") as fh:
            fh.write(code)
        with open(os.path.join(d, "tool.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    def _spec_for(self, meta: dict) -> ToolSpec:
        name = meta["name"]

        def run(**args):
            return self.run_custom(name, args)

        desc = (f"{meta['description']}\n\n(User-created tool, v{meta['version']}. "
                "Read its code with get_tool_code.)")
        return ToolSpec(name=name, title=name.replace("_", " "), description=desc,
                        schema=meta["schema"], fn=run, source="custom",
                        meta={"version": meta["version"]})

    def run_custom(self, name: str, args: dict) -> dict:
        from .. import mcpserver, scripting

        meta = self._meta(name)
        props = meta["schema"].get("properties", {})
        unknown = set(args) - set(props)
        if unknown:
            raise ValueError(f"{name} has no parameter(s) {sorted(unknown)}; "
                             f"it takes {sorted(props)}")
        missing = [k for k in meta["schema"].get("required", []) if k not in args]
        if missing:
            raise ValueError(f"{name} needs {missing}")
        params = {k: (_coerce(k, args[k], p) if k in args else p.get("default"))
                  for k, p in props.items()}
        slide = mcpserver._slide()
        code = self._code(name, meta["version"])
        result = scripting.run_script(code, slide.api,
                                      extra_globals={"params": dict(params), **params},
                                      filename=f"<tool {name} v{meta['version']}>")
        out = mcpserver.script_output(result, slide)
        out["tool"] = {"name": name, "version": meta["version"],
                       "sha256": meta["versions"][str(meta["version"])]["sha256"][:12]}
        return out

    def precheck(self, tool: str, args: dict) -> str:
        """Why an approval-gated call would fail anyway (so the user isn't asked to approve
        it), or ``""``. Mirrors the checks the tool itself makes."""
        try:
            if tool == "run_script":
                if not isinstance(args.get("code"), str):
                    return "run_script needs code (text)"
                compile(args["code"], "<run_script>", "exec")
                return ""
            name = _check_name(args.get("name", ""))
            if tool == "create_tool":
                if name in self.registry:
                    return f"a tool named {name!r} already exists — use edit_tool to change it"
                if not isinstance(args.get("code"), str) or not args["code"].strip():
                    return "create_tool needs code (text)"
                if not isinstance(args.get("description"), str):
                    return "create_tool needs a description (text)"
            elif tool in ("edit_tool", "delete_tool"):
                spec = self.registry.get(name)
                if spec is None or spec.source != "custom":
                    return f"{name!r} is not a user-created tool"
                if tool == "edit_tool":
                    if not any(args.get(k) for k in ("code", "description")) \
                            and args.get("parameters") is None:
                        return "edit_tool needs a change (code, description or parameters)"
                    if args.get("code") is not None and not isinstance(args["code"], str):
                        return "code must be text"
            if isinstance(args.get("code"), str) and args["code"]:
                compile(args["code"], f"<tool {name}>", "exec")
            if args.get("parameters") is not None:
                if not isinstance(args["parameters"], dict):
                    return "parameters must be an object {name: {type, description, default}}"
                _param_schema(args["parameters"])
        except (SyntaxError, ValueError) as exc:
            return f"{type(exc).__name__}: {exc}"
        return ""

    def create_tool(self, name: str, description: str, code: str,
                    parameters: dict | None = None) -> dict:
        """Create a new analysis tool from a script. Needs the user's approval.

        ``code`` is a script in the run_script namespace (read scripting_guide): ``ds`` is the
        slide, the analysis functions are bare names, and ``log`` / ``table`` / ``image`` /
        ``record`` surface results. Each entry of ``parameters`` —
        ``{"name": {"type": "number|integer|string|boolean|array|object", "description": …,
        "default": …}}`` — is bound as a variable of that name (all of them also as the dict
        ``params``); a parameter without ``default`` is required. Write it general (no
        hard-coded m/z or region names that should be parameters) and test it on the open
        slide right after it is approved. Prefer an existing tool when one already fits."""
        name = _check_name(name)
        if name in self.registry:
            raise ValueError(f"a tool named {name!r} already exists — use edit_tool to change it")
        compile(code, f"<tool {name}>", "exec")                 # syntax check before saving
        schema = _param_schema(parameters)
        meta = {"name": name, "description": str(description).strip(), "schema": schema,
                "version": 1, "created": _now(), "updated": _now(),
                "versions": {"1": {"sha256": _sha(code), "saved": _now(), "note": "created"}}}
        self._write(meta, code)
        self.registry.add(self._spec_for(meta))
        return {"created": name, "version": 1, "sha256": _sha(code)[:12],
                "parameters": list(schema["properties"]),
                "next": f"call {name}(...) to test it on the open slide"}

    def edit_tool(self, name: str, code: str = "", description: str = "",
                  parameters: dict | None = None, note: str = "") -> dict:
        """Change a user-created tool (new code, description and/or parameters). Needs the
        user's approval. The previous version is kept on disk; the tool's version number goes
        up by one. ``note`` says what changed and why (it goes into the tool's history)."""
        spec = self.registry.get(name)
        if spec is None or spec.source != "custom":
            raise ValueError(f"{name!r} is not a user-created tool (built-in tools can't be "
                             "edited — create a new tool instead)")
        if not code and not description and parameters is None:
            raise ValueError("nothing to change — give new code, description or parameters")
        meta = self._meta(name)
        new_code = code or self._code(name, meta["version"])
        compile(new_code, f"<tool {name}>", "exec")
        if parameters is not None:
            meta["schema"] = _param_schema(parameters)
        if description:
            meta["description"] = str(description).strip()
        meta["version"] += 1
        meta["updated"] = _now()
        meta["versions"][str(meta["version"])] = {"sha256": _sha(new_code), "saved": _now(),
                                                  "note": note or "edited"}
        self._write(meta, new_code)
        self.registry.add(self._spec_for(meta))
        return {"edited": name, "version": meta["version"], "sha256": _sha(new_code)[:12]}

    def delete_tool(self, name: str) -> dict:
        """Remove a user-created tool from the tool list. Needs the user's approval. Its files
        are kept (renamed ``<name>.deleted-<time>``) so logged results stay reproducible."""
        spec = self.registry.get(name)
        if spec is None or spec.source != "custom":
            raise ValueError(f"{name!r} is not a user-created tool")
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        os.replace(os.path.join(self.tools_dir, name),
                   os.path.join(self.tools_dir, f"{name}.deleted-{stamp}"))
        self.registry.remove(name)
        return {"deleted": name}

    def get_tool_code(self, name: str, version: int = 0) -> dict:
        """The code, parameters and version history of a user-created tool (``version`` 0 =
        current)."""
        name = _check_name(name)
        if self.registry.get(name) is None or self.registry.get(name).source != "custom":
            raise ValueError(f"{name!r} is not a user-created tool")
        meta = self._meta(name)
        v = int(version) or meta["version"]
        return {"name": name, "version": v, "description": meta["description"],
                "parameters": meta["schema"], "history": meta["versions"],
                "code": self._code(name, v)}

    # ------------------------------------------------------------------ #
    # flows
    # ------------------------------------------------------------------ #
    def _flow_path(self, name: str) -> str:
        return os.path.join(self.flows_dir, f"{_check_name(name, 'flow')}.json")

    def save_flow(self, name: str, description: str, steps: list) -> dict:
        """Save a reusable analysis flow: an ordered list of tool calls,
        ``[{"tool": "find_peaks", "args": {"snr": 3}}, …]``. An argument written as
        ``"{{name}}"`` becomes a parameter filled in by run_flow(params={…}). Use it to keep
        an analysis the user liked so it can be replayed on other slides."""
        name = _check_name(name, "flow")
        clean = []
        for i, st in enumerate(steps or []):
            tool = st.get("tool") if isinstance(st, dict) else None
            spec = self.registry.get(tool)
            if spec is None:
                raise ValueError(f"step {i + 1}: unknown tool {tool!r}")
            if spec.approval or tool in _NOT_IN_FLOWS:
                raise ValueError(f"step {i + 1}: {tool} can't be part of a flow")
            args = st.get("args") or {}
            if not isinstance(args, dict):
                raise ValueError(f"step {i + 1}: args must be an object")
            clean.append({"tool": tool, "args": dict(args)})
        if not clean:
            raise ValueError("a flow needs at least one step")
        flow = {"name": name, "description": str(description), "steps": clean,
                "saved": _now(), "params": sorted(self._placeholders(clean))}
        with open(self._flow_path(name), "w", encoding="utf-8") as fh:
            json.dump(flow, fh, indent=2)
        return {"saved": name, "steps": len(clean), "params": flow["params"]}

    @staticmethod
    def _placeholders(steps) -> set:
        out: set = set()
        for st in steps:
            _placeholders_in(st["args"], out)
        return out

    def list_flows(self) -> list:
        """The saved flows: name, description, steps and the parameters each one takes."""
        out = []
        for fn in sorted(os.listdir(self.flows_dir)):
            if fn.endswith(".json"):
                with open(os.path.join(self.flows_dir, fn), encoding="utf-8") as fh:
                    f = json.load(fh)
                out.append({"name": f["name"], "description": f["description"],
                            "steps": [s["tool"] for s in f["steps"]],
                            "params": f.get("params", [])})
        return out

    def run_flow(self, name: str, params: dict | None = None) -> dict:
        """Run a saved flow step by step on the open slide. ``params`` fills its ``{{name}}``
        placeholders. Stops at the first failing step and reports it."""
        with open(self._flow_path(name), encoding="utf-8") as fh:
            flow = json.load(fh)
        params = dict(params or {})
        needed = self._placeholders(flow["steps"])
        missing = sorted(needed - set(params))
        if missing:
            raise ValueError(f"flow {name} needs params {missing}")
        unknown = sorted(set(params) - needed)
        if unknown:
            raise ValueError(f"flow {name} has no params {unknown}; it takes {sorted(needed)}")
        from .core import _failed

        results = []
        for i, st in enumerate(flow["steps"]):
            args = _substitute(st["args"], params)
            spec = self.registry.get(st["tool"])
            if spec is None or spec.approval or st["tool"] in _NOT_IN_FLOWS:
                results.append({"step": i + 1, "tool": st["tool"], "ok": False,
                                "error": "tool missing or not allowed in a flow"})
                break
            try:
                res = spec.fn(**args)
            except Exception as exc:  # noqa: BLE001 — report the failing step, stop the flow
                results.append({"step": i + 1, "tool": st["tool"], "args": args, "ok": False,
                                "error": f"{type(exc).__name__}: {exc}"})
                break
            err = _failed(res)                    # e.g. run_script returning ok: false
            results.append({"step": i + 1, "tool": st["tool"], "args": args, "ok": not err,
                            "result": res, **({"error": err} if err else {})})
            if err:
                break
        return {"flow": name, "completed": all(r["ok"] for r in results), "steps": results}

    # ------------------------------------------------------------------ #
    @staticmethod
    def show_file(path: str, caption: str = "") -> dict:
        """Show an image (PNG/JPG) or table (CSV) that an analysis saved to the chat — e.g. a
        figure a script wrote earlier. Only files under the SMILE MSI data folder (results,
        sessions) can be shown. Results of other tools already show their images."""
        from .tools import shareable

        path = os.path.realpath(os.path.expanduser(str(path)))
        if not os.path.isfile(path):
            raise FileNotFoundError(os.path.basename(path))
        if not shareable(path):
            raise PermissionError("only PNG/JPG/CSV files under the SMILE MSI data folder "
                                  "can be shown")
        return {"path": path, "caption": caption}

    def agent_specs(self) -> list[ToolSpec]:
        """The agent's own tools (tool/flow management), wrapped like any other tool."""
        from .tools import schema_for

        rows = [
            (self.create_tool, "Create a tool", True, False),
            (self.edit_tool, "Edit a tool", True, False),
            (self.delete_tool, "Delete a tool", True, False),
            (self.get_tool_code, "Read a tool's code", False, True),
            (self.save_flow, "Save a flow", False, False),
            (self.list_flows, "List flows", False, True),
            (self.run_flow, "Run a flow", False, False),
            (self.show_file, "Show a file", False, True),
        ]
        import inspect

        return [ToolSpec(name=fn.__name__, title=title, approval=approval, read_only=ro,
                         description=inspect.cleandoc(fn.__doc__ or title),
                         schema=schema_for(fn), fn=fn, source="agent")
                for fn, title, approval, ro in rows]
