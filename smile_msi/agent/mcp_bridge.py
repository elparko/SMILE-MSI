"""Expose the chat's user-created tools and flows to external MCP clients (e.g. Claude Desktop).

MCP clients cache the server's tool list, so the bridge adds a **fixed** set of tools that
never changes as users create their own: ``list_user_tools`` to discover them and
``run_user_tool`` to call one by name, plus the same flow and tool-management calls the chat
agent has. Creating, editing and deleting a tool are annotated ``destructive`` so the MCP
client asks the user before running them — the counterpart of the chat's approval prompt.

The :class:`~smile_msi.agent.custom.Toolbox` is built on first use, not at import, and shared
by every call. It reads the same ``<home>/agent/`` folders as the chat, so a tool made there
shows up here (and vice versa) without a restart — :func:`toolbox` re-syncs it from disk.
"""
from __future__ import annotations

import inspect
import os

_cache: dict = {"home": None, "box": None}

#: ``(function name, human title, read-only, destructive)`` — the fixed tool list.
_ROWS = (
    ("list_user_tools", "List user tools", True, False),
    ("run_user_tool", "Run a user tool", False, False),
    ("get_tool_code", "Read a tool's code", True, False),
    ("save_flow", "Save a flow", False, False),
    ("list_flows", "List flows", True, False),
    ("run_flow", "Run a flow", False, False),
    ("create_tool", "Create a tool", False, True),
    ("edit_tool", "Edit a tool", False, True),
    ("delete_tool", "Delete a tool", False, True),
)


def toolbox():
    """The shared Toolbox for the current app home, synced with the tools on disk."""
    from .. import library
    from .custom import Toolbox
    from .tools import Registry, builtin_tools

    home = library.home_dir()
    if _cache["box"] is None or _cache["home"] != home:      # first use (or a new home)
        _cache["box"], _cache["home"] = Toolbox(Registry(builtin_tools())), home
    box = _cache["box"]
    _sync(box)
    return box


def _sync(box):
    """Pick up tools the chat created, edited or deleted since the Toolbox was built."""
    from .custom import _NAME

    seen = set()
    for name in os.listdir(box.tools_dir):       # skips <name>.deleted-<time>
        if not (_NAME.match(name)
                and os.path.isfile(os.path.join(box.tools_dir, name, "tool.json"))):
            continue
        cur = box.registry.get(name)
        if cur is not None and cur.source != "custom":     # a file never shadows a built-in
            continue
        try:
            meta = box._meta(name)
            if cur is None or cur.meta.get("version") != meta["version"]:
                box.registry.add(box._spec_for(meta))
        except (OSError, ValueError, KeyError):            # a damaged tool folder: skip it
            continue
        seen.add(name)
    for spec in list(box.registry):
        if spec.source == "custom" and spec.name not in seen:
            box.registry.remove(spec.name)


def _user_tool(box, name: str) -> str:
    from .custom import _check_name

    name = _check_name(name)
    spec = box.registry.get(name)
    if spec is None or spec.source != "custom":
        raise ValueError(f"{name!r} is not a user-created tool — call list_user_tools to see "
                         "them (built-in tools are separate MCP tools)")
    return name


# --------------------------------------------------------------------------- #
# the MCP tools (docstrings are the descriptions the client shows)
# --------------------------------------------------------------------------- #
def list_user_tools() -> dict:
    """List the analysis tools and flows the user created in the SMILE MSI chat.

    Returns ``tools`` (name, description, version, ``parameters`` as a JSON schema) and
    ``flows`` (name, description, step tools, parameters). Call a tool with
    ``run_user_tool(name, args)`` and a flow with ``run_flow(name, params)``. This list changes
    as the user adds tools; the MCP tool list itself does not."""
    box = toolbox()
    tools = []
    for spec in box.registry:
        if spec.source == "custom":
            meta = box._meta(spec.name)
            tools.append({"name": spec.name, "description": meta["description"],
                          "version": meta["version"], "parameters": meta["schema"]})
    return {"tools": tools, "flows": box.list_flows()}


def run_user_tool(name: str, args: dict | None = None) -> dict:
    """Run a user-created analysis tool on the open slide (open one with open_slide first).

    ``name`` is from list_user_tools; ``args`` are its declared parameters, e.g.
    ``{"mz": 885.55}`` (a parameter without a default is required). Returns the same result a
    run_script call would — logs, values, table previews with CSV paths, image paths — plus
    the tool's name, version and a code hash."""
    box = toolbox()
    return box.run_custom(_user_tool(box, name), dict(args or {}))


def get_tool_code(name: str, version: int = 0) -> dict:
    """The code, parameters and version history of a user-created tool.

    ``version`` 0 means the current one; every earlier version is kept."""
    return toolbox().get_tool_code(name, version)


def save_flow(name: str, description: str, steps: list[dict]) -> dict:
    """Save a reusable analysis flow: an ordered list of tool calls, e.g.
    ``[{"tool": "find_peaks", "args": {"snr": 3}}, {"tool": "ion_image", "args": {"mz":
    "{{mz}}"}}]``.

    A value written ``"{{name}}"`` becomes a parameter filled in by run_flow. Steps may be any
    SMILE MSI tool or user tool except create/edit/delete_tool and flow tools."""
    return toolbox().save_flow(name, description, steps)


def list_flows() -> dict:
    """The saved flows as ``{"flows": [...]}``: name, description, the tools each calls and
    the parameters it takes."""
    return {"flows": toolbox().list_flows()}


def run_flow(name: str, params: dict | None = None) -> dict:
    """Run a saved flow step by step on the open slide.

    ``params`` fills the flow's ``{{name}}`` placeholders (see list_flows). Stops at the
    first failing step and reports it; ``completed`` says whether every step succeeded."""
    return toolbox().run_flow(name, params)


def create_tool(name: str, description: str, code: str,
                parameters: dict | None = None) -> dict:
    """Create a new analysis tool from a script; the user approves this call in their MCP
    client. The tool is saved (versioned) and callable at once with run_user_tool.

    ``name`` is snake_case, 3-48 characters. ``code`` runs like run_script (read
    scripting_guide): ``ds`` is the slide, analysis functions are bare names, and ``log`` /
    ``table`` / ``image`` / ``record`` surface results. ``parameters`` maps a name to
    ``{"type": "number|integer|string|boolean|array|object", "description": ..., "default":
    ...}``; each is bound as a variable (all also as the dict ``params``) and one without a
    default is required. Write it general, with no hard-coded m/z or region names, and test it
    right after creating it. Prefer an existing tool when one fits."""
    return toolbox().create_tool(name, description, code, parameters)


def edit_tool(name: str, code: str = "", description: str = "",
              parameters: dict | None = None, note: str = "") -> dict:
    """Change a user-created tool's code, description and/or parameters; the user approves
    this call in their MCP client. The previous version is kept and the version number goes up
    by one. ``note`` says what changed and why (it goes into the tool's history)."""
    return toolbox().edit_tool(name, code, description, parameters, note)


def delete_tool(name: str) -> dict:
    """Remove a user-created tool; the user approves this call in their MCP client. Its files
    are kept (renamed ``<name>.deleted-<time>``) so logged results stay reproducible."""
    return toolbox().delete_tool(name)


_FUNCS = {f.__name__: f for f in (list_user_tools, run_user_tool, get_tool_code, save_flow,
                                  list_flows, run_flow, create_tool, edit_tool, delete_tool)}


def register(server, ToolAnnotations, guard):
    """Add the fixed bridge tools to an MCP ``server``. ``guard`` is mcpserver's ``_guarded``
    (reports failures as the client-visible ``ToolError``). Touches no disk."""
    for name, title, read_only, destructive in _ROWS:
        fn = _FUNCS[name]
        server.add_tool(guard(fn), title=title, description=inspect.cleandoc(fn.__doc__),
                        annotations=ToolAnnotations(read_only_hint=read_only,
                                                    destructive_hint=destructive))
