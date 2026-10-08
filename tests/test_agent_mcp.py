"""The MCP bridge: user-created tools and flows reachable from an external MCP client."""
import asyncio
import json

import pytest

from smile_msi import mcpserver
from smile_msi.agent import mcp_bridge

mcp = pytest.importorskip("mcp", reason="the MCP server needs the optional 'mcp' extra")

BRIDGE = {"list_user_tools", "run_user_tool", "get_tool_code", "save_flow", "list_flows",
          "run_flow", "create_tool", "edit_tool", "delete_tool"}


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SMILE_MSI_MCP_OUT", str(tmp_path / "out"))
    yield
    mcpserver.close_slide()


def call(tool: str, args: dict | None = None):
    """One tool call over the real protocol, in process."""
    from mcp.client import Client

    async def go():
        async with Client(mcpserver.build_server()) as client:
            return await client.call_tool(tool, args or {})

    return asyncio.run(go())


def payload(result):
    assert result.is_error is False, result.content[0].text
    return json.loads(result.content[0].text)


def list_tools():
    from mcp.client import Client

    async def go():
        async with Client(mcpserver.build_server()) as client:
            return (await client.list_tools()).tools

    return asyncio.run(go())


def _make(name="count_one", **kw):
    args = {"name": name, "description": "records n = 1", "code": "record('n', 1)", **kw}
    return payload(call("create_tool", args))


def test_server_lists_the_bridge_tools_with_annotations():
    tools = {t.name: t for t in list_tools()}
    assert BRIDGE <= set(tools)
    assert len(tools) == len(mcpserver.TOOLS) + len(BRIDGE)
    for name in BRIDGE:
        assert tools[name].title and tools[name].description
    for name in ("create_tool", "edit_tool", "delete_tool"):
        assert tools[name].annotations.destructive_hint is True
        assert "approves" in tools[name].description          # the user, in their client
    for name in ("list_user_tools", "get_tool_code", "list_flows"):
        assert tools[name].annotations.read_only_hint is True
    assert tools["run_user_tool"].annotations.read_only_hint is False
    assert tools["run_user_tool"].annotations.destructive_hint is False


def test_importing_and_registering_does_no_disk_work(tmp_path):
    mcpserver.build_server()
    assert not (tmp_path / "home").exists()                   # nothing until first use


def test_the_tool_list_is_stable_when_users_create_tools():
    before = sorted(t.name for t in list_tools())
    _make()
    assert sorted(t.name for t in list_tools()) == before


def test_created_tool_is_listed_and_runs_on_the_demo_slide():
    created = _make(parameters={"k": {"type": "integer", "description": "unused", "default": 2}})
    assert created["created"] == "count_one" and created["version"] == 1
    listing = payload(call("list_user_tools"))
    (tool,) = listing["tools"]
    assert tool["name"] == "count_one" and tool["version"] == 1
    assert tool["description"] == "records n = 1"             # not the chat's decorated text
    assert tool["parameters"]["properties"]["k"]["type"] == "integer"
    mcpserver.open_slide("demo")
    out = payload(call("run_user_tool", {"name": "count_one", "args": {"k": "3"}}))
    assert out["ok"] is True and out["values"] == {"n": 1}
    assert out["tool"]["name"] == "count_one" and out["tool"]["version"] == 1
    assert payload(call("get_tool_code", {"name": "count_one"}))["code"] == "record('n', 1)"


def test_edit_and_delete_through_the_bridge():
    _make()
    edited = payload(call("edit_tool", {"name": "count_one", "code": "record('n', 2)",
                                        "note": "two"}))
    assert edited["version"] == 2
    mcpserver.open_slide("demo")
    assert payload(call("run_user_tool", {"name": "count_one"}))["values"] == {"n": 2}
    assert payload(call("get_tool_code", {"name": "count_one", "version": 1}))["code"] \
        == "record('n', 1)"
    assert payload(call("delete_tool", {"name": "count_one"})) == {"deleted": "count_one"}
    assert payload(call("list_user_tools"))["tools"] == []


def test_tools_made_elsewhere_on_disk_are_picked_up():
    mcpserver.open_slide("demo")
    assert payload(call("list_user_tools"))["tools"] == []    # toolbox now cached
    from smile_msi.agent.custom import Toolbox
    from smile_msi.agent.tools import Registry, builtin_tools

    Toolbox(Registry(builtin_tools())).create_tool("from_chat", "made in the chat",
                                                   "record('n', 7)")
    assert payload(call("run_user_tool", {"name": "from_chat"}))["values"] == {"n": 7}


def test_flows_save_list_and_run():
    _make()
    flow = payload(call("save_flow", {
        "name": "count_flow", "description": "run the counter",
        "steps": [{"tool": "count_one", "args": {}},
                  {"tool": "ion_image", "args": {"mz": "{{mz}}"}}]}))
    assert flow["saved"] == "count_flow" and flow["params"] == ["mz"]
    assert payload(call("list_flows"))["flows"][0]["steps"] == ["count_one", "ion_image"]
    assert [f["name"] for f in payload(call("list_user_tools"))["flows"]] == ["count_flow"]
    mcpserver.open_slide("demo")
    out = payload(call("run_flow", {"name": "count_flow", "params": {"mz": 885.55}}))
    assert out["completed"] is True
    assert out["steps"][0]["result"]["values"] == {"n": 1}
    assert out["steps"][1]["result"]["png"]
    missing = call("run_flow", {"name": "count_flow"})
    assert missing.is_error and "mz" in missing.content[0].text


@pytest.mark.parametrize("tool, args", [
    ("run_user_tool", {"name": "Bad Name!"}),
    ("run_user_tool", {"name": "no_such_tool"}),
    ("run_user_tool", {"name": "find_peaks"}),                # a built-in is not a user tool
    ("get_tool_code", {"name": "no_such_tool"}),
    ("delete_tool", {"name": "find_peaks"}),
    ("create_tool", {"name": "X", "description": "d", "code": "pass"}),
    ("create_tool", {"name": "bad_syntax", "description": "d", "code": "def ("}),
    ("save_flow", {"name": "empty_flow", "description": "d", "steps": []}),
])
def test_bad_requests_give_a_clean_error(tool, args):
    result = call(tool, args)
    assert result.is_error is True
    text = result.content[0].text
    assert ("ValueError:" in text or "SyntaxError:" in text) and "Traceback" not in text


def test_a_tool_that_fails_reports_it_instead_of_crashing():
    _make("broken_tool", code="raise RuntimeError('boom')")
    mcpserver.open_slide("demo")
    out = payload(call("run_user_tool", {"name": "broken_tool"}))
    assert out["ok"] is False and "boom" in out["error"]


def test_a_tool_needs_an_open_slide():
    _make()
    result = call("run_user_tool", {"name": "count_one"})
    assert result.is_error and "open_slide" in result.content[0].text


def test_toolbox_is_cached_per_home(tmp_path, monkeypatch):
    first = mcp_bridge.toolbox()
    assert mcp_bridge.toolbox() is first
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path / "other"))
    assert mcp_bridge.toolbox() is not first
