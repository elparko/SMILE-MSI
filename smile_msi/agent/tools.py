"""The agent's tool registry — every MCP tool, plus the user's own tools and flows, as one
list of :class:`ToolSpec` a model can call.

The built-in tools are the plain functions in :mod:`smile_msi.mcpserver` (``TOOLS``), so the
chat agent and an external MCP client drive exactly the same engine with the same docs. A
function's signature becomes its JSON schema (:func:`schema_for`) and its docstring its
description, so adding a tool to ``mcpserver.TOOLS`` adds it here with no second definition.

A tool result is whatever the function returns (a JSON-able dict, usually). :func:`render`
turns it into what the model reads — compact JSON text, plus the PNGs the result points at as
images — and lists those images separately so the chat can show them as they are made.
"""
from __future__ import annotations

import base64
import inspect
import json
import os
import types
import typing
from dataclasses import dataclass, field

#: Longest JSON text handed back to the model for one tool result (the rest is in the files
#: the result points at — tables already come back as a preview + CSV path).
MAX_RESULT_CHARS = 20_000
#: Most images attached to one tool result for the model to look at.
MAX_MODEL_IMAGES = 4

_IMAGE_EXT = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
_JSON_TYPE = {float: "number", int: "integer", bool: "boolean", str: "string",
              list: "array", tuple: "array", dict: "object"}


@dataclass
class ToolSpec:
    """One callable tool. ``fn(**args)`` runs it; ``approval`` marks tools that need the user's
    OK before every call (creating or editing a tool — the only actions that let a model put
    new code on the user's machine)."""
    name: str
    description: str
    schema: dict
    fn: typing.Callable
    title: str = ""
    read_only: bool = False
    approval: bool = False
    source: str = "builtin"          # builtin | agent | custom | flow
    meta: dict = field(default_factory=dict)

    def api_def(self) -> dict:
        """The provider-neutral definition: name, description, JSON-schema input."""
        return {"name": self.name, "description": self.description, "input_schema": self.schema}


def _json_type(tp) -> dict:
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        return _json_type(args[0]) if len(args) == 1 else {}
    base = origin or tp
    t = _JSON_TYPE.get(base)
    return {"type": t} if t else {}


def schema_for(fn) -> dict:
    """JSON schema of ``fn``'s keyword parameters, from its type hints and defaults: a
    parameter without a default is required; ``X | None`` is an optional ``X``."""
    sig = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # noqa: BLE001 — unresolvable hint: fall back to "any"
        hints = {}
    props, required = {}, []
    for name, p in sig.parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        prop = _json_type(hints.get(name, p.annotation if p.annotation is not p.empty else None))
        if p.default is p.empty:
            required.append(name)
        elif p.default is not None and isinstance(p.default, (int, float, str, bool)):
            prop["default"] = p.default
        props[name] = prop
    out = {"type": "object", "properties": props}
    if required:
        out["required"] = required
    return out


def builtin_tools() -> list[ToolSpec]:
    """Every tool the MCP server exposes, wrapped for the agent."""
    from .. import mcpserver

    specs = []
    for fn, title, read_only in mcpserver.TOOLS:
        specs.append(ToolSpec(name=fn.__name__, title=title, read_only=read_only,
                              description=inspect.cleandoc(fn.__doc__ or title),
                              schema=schema_for(fn), fn=fn))
    return specs


class Registry:
    """Name → :class:`ToolSpec`, in a stable order (built-ins first) so the tool list a model
    sees — and therefore the prompt-cache prefix — only changes when a tool is added."""

    def __init__(self, specs=()):
        self._specs: dict[str, ToolSpec] = {}
        for s in specs:
            self.add(s)

    def add(self, spec: ToolSpec):
        self._specs[spec.name] = spec

    def remove(self, name: str):
        self._specs.pop(name, None)

    def get(self, name: str):
        return self._specs.get(name)

    def __contains__(self, name):
        return name in self._specs

    def __iter__(self):
        return iter(self._specs.values())

    def names(self) -> list[str]:
        return list(self._specs)

    def api_defs(self) -> list[dict]:
        return [s.api_def() for s in self._specs.values()]


# --------------------------------------------------------------------------- #
# results → what the model reads + what the chat shows
# --------------------------------------------------------------------------- #
@dataclass
class Rendered:
    """A tool result ready for both audiences: ``text`` for the model (and the log),
    ``images`` the image files it produced (shown in the chat; the first few are also sent
    to the model), ``files`` other files it points at (CSV tables, …)."""
    text: str
    images: list = field(default_factory=list)
    files: list = field(default_factory=list)
    is_error: bool = False


def _walk_paths(obj, out):
    if isinstance(obj, dict):
        for v in obj.values():
            _walk_paths(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk_paths(v, out)
    elif isinstance(obj, str) and len(obj) < 1024 and os.path.isabs(obj) and os.path.isfile(obj):
        out.append(obj)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "tolist"):
        return _jsonable(obj.tolist())
    return str(obj)


def render(result, *, is_error: bool = False) -> Rendered:
    """Turn a tool's return value into a :class:`Rendered` (text + the files it points at)."""
    paths: list[str] = []
    _walk_paths(result, paths)
    images = [p for p in dict.fromkeys(paths) if os.path.splitext(p)[1].lower() in _IMAGE_EXT]
    files = [p for p in dict.fromkeys(paths) if p not in images]
    text = result if isinstance(result, str) else json.dumps(_jsonable(result), indent=1)
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + f"\n… [truncated {len(text) - MAX_RESULT_CHARS} chars]"
    return Rendered(text=text, images=images, files=files, is_error=is_error)


def image_b64(path: str) -> tuple[str, str]:
    """``(media_type, base64 data)`` of an image file, for a model's image block."""
    media = _IMAGE_EXT.get(os.path.splitext(path)[1].lower(), "image/png")
    with open(path, "rb") as fh:
        return media, base64.standard_b64encode(fh.read()).decode("ascii")
