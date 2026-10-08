"""The plain-language agent (:mod:`smile_msi.agent`) — driven by a scripted fake model so the
whole loop (real tools on the demo slide, approvals, user-created tools, flows, the audit log
and the local web server) runs with no API key and no network."""
import json
import os
import threading
import urllib.error
import urllib.request

import pytest

from smile_msi import mcpserver
from smile_msi.agent import tools
from smile_msi.agent.core import Agent
from smile_msi.agent.log import SessionLog
from smile_msi.agent.providers import Step, ToolCall


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SMILE_MSI_MCP_OUT", str(tmp_path / "out"))
    yield
    mcpserver.close_slide()


class FakeProvider:
    """Plays back a script of steps; records what the agent fed it."""
    name = "fake"

    def __init__(self, script):
        self.script = list(script)
        self.outputs, self.users, self.tool_sets = [], [], []

    def settings(self):
        return {"model": "scripted"}

    def start(self, system, tools_):
        self.system = system
        self.tool_sets.append([t["name"] for t in tools_])

    def set_tools(self, tools_):
        self.tool_sets.append([t["name"] for t in tools_])

    def add_user(self, text):
        self.users.append(text)

    def step(self):
        item = self.script.pop(0)
        return item(self) if callable(item) else item

    def add_tool_outputs(self, outputs):
        self.outputs.append(outputs)


def _call(tool, /, **args):
    return Step(calls=[ToolCall(f"c-{tool}", tool, args)],
                usage={"input_tokens": 10, "output_tokens": 5})


def _agent(script, approve=None):
    events = []
    p = FakeProvider(script)
    a = Agent(p, emit=events.append, approve=approve or (lambda req: (True, "")))
    return a, p, events


def _by_type(events, kind):
    return [e for e in events if e["type"] == kind]


# --------------------------------------------------------------------------- #
def test_every_mcp_tool_gets_a_schema():
    specs = {s.name: s for s in tools.builtin_tools()}
    assert set(specs) == {fn.__name__ for fn, _t, _r in mcpserver.TOOLS}
    ion = specs["ion_image"].schema
    assert ion["properties"]["mz"] == {"type": "number"}
    assert ion["required"] == ["mz"]
    assert ion["properties"]["cmap"]["default"] == "viridis"
    send = specs["send_regions"].schema
    assert send["properties"]["names"] == {"type": "array"} and "required" not in send


def test_render_finds_images_and_truncates(tmp_path):
    png = tmp_path / "a.png"
    png.write_bytes(b"\x89PNG\r\n")
    csv = tmp_path / "t.csv"
    csv.write_text("x\n1\n")
    r = tools.render({"png": str(png), "tables": [{"csv": str(csv)}], "big": "x" * 30000})
    assert r.images == [str(png)] and r.files == [str(csv)]
    assert "truncated" in r.text


def test_a_turn_runs_real_tools_and_logs_everything():
    mz = None

    def pick_ion(p):
        nonlocal mz
        res = json.loads(p.outputs[-1][0].text)          # find_peaks result
        mz = res["mz"][0]
        return _call("ion_image", mz=mz)

    a, p, ev = _agent([_call("open_slide", ref="demo"), _call("find_peaks", snr=3.0),
                       pick_ion, Step(text="Here is the strongest ion.")])
    a.send("show me the strongest ion")
    assert not _by_type(ev, "error"), _by_type(ev, "error")
    results = _by_type(ev, "tool_result")
    assert [r["name"] for r in results] == ["open_slide", "find_peaks", "ion_image"]
    assert all(r["ok"] for r in results)
    assert results[-1]["images"] and os.path.isfile(results[-1]["images"][0])
    assert p.outputs[-1][0].images == results[-1]["images"]       # the model saw the figure
    assert _by_type(ev, "dataset")[0]["n_pixels"] > 0             # provenance of the data
    assert _by_type(ev, "assistant")[-1]["text"] == "Here is the strongest ion."
    # the log on disk has the same record, and the report ends in a replayable flow
    with open(a.log.path, encoding="utf-8") as fh:
        on_disk = [json.loads(line) for line in fh]
    assert [r["type"] for r in on_disk] == [e["type"] for e in ev]
    md = a.log.markdown()
    assert "ion_image" in md and "## Reproduce" in md
    assert [s["tool"] for s in a.log.tool_calls()] == ["open_slide", "find_peaks", "ion_image"]


def test_tool_errors_go_back_to_the_model():
    a, p, ev = _agent([_call("ion_image", mz=500.0), Step(text="no slide yet")])
    a.send("image 500")
    res = _by_type(ev, "tool_result")[0]
    assert not res["ok"] and "open_slide" in res["error"]
    assert p.outputs[0][0].is_error


CODE = "v = ion_vector(target)\nrecord('mean', float(v.mean()))\nimage(ion_image(target), 'it')\n"


def test_created_tool_needs_approval_then_runs_like_a_builtin():
    asked = []

    def approve(req):
        asked.append(req)
        return True, "looks fine"

    a, p, ev = _agent([
        _call("open_slide", ref="demo"),
        _call("create_tool", name="mean_of_ion", description="Mean intensity of one ion.",
              code=CODE, parameters={"target": {"type": "number", "description": "m/z"}}),
        _call("mean_of_ion", target=885.55),
        Step(text="done")], approve=approve)
    a.send("make a tool")
    assert len(asked) == 1 and asked[0]["code"] == CODE and asked[0]["sha256"]
    assert _by_type(ev, "approval")[0]["decision"] == "approved"
    assert "mean_of_ion" in p.tool_sets[-1]                       # model sees the new tool
    run = [r for r in _by_type(ev, "tool_result") if r["name"] == "mean_of_ion"][0]
    assert run["ok"], run.get("error")
    out = json.loads(run["text"])
    assert out["tool"]["version"] == 1 and "mean" in out["values"] and run["images"]
    # editing bumps the version and keeps v1 on disk
    tb = a.toolbox
    tb.edit_tool("mean_of_ion", code=CODE + "log('v2')\n", note="add a log line")
    assert tb.get_tool_code("mean_of_ion")["version"] == 2
    assert tb.get_tool_code("mean_of_ion", version=1)["code"] == CODE
    # a deleted tool stays gone in the next conversation (its files are kept, renamed)
    tb.delete_tool("mean_of_ion")
    a2, _p2, _ev2 = _agent([])
    assert "mean_of_ion" not in a2.registry


def test_rejected_tool_is_not_created():
    a, p, ev = _agent([
        _call("create_tool", name="sneaky_tool", description="x", code="print(1)"),
        Step(text="ok, what instead?")], approve=lambda req: (False, "not needed"))
    a.send("make a tool")
    assert "sneaky_tool" not in a.registry
    res = _by_type(ev, "tool_result")[0]
    assert not res["ok"] and "not needed" in res["error"]


def test_bad_tool_definitions_fail_before_saving():
    a, _p, _ev = _agent([])
    with pytest.raises(SyntaxError):
        a.toolbox.create_tool("broken_tool", "x", "def (:")
    with pytest.raises(ValueError):
        a.toolbox.create_tool("ion_image", "clash", "print(1)")      # built-in name
    with pytest.raises(ValueError):
        a.toolbox.create_tool("ok_name", "x", "print(1)", parameters={"p": {"type": "float"}})
    with pytest.raises(ValueError):
        a.toolbox.edit_tool("ion_image", code="print(1)")            # built-ins are read-only


def test_flows_save_and_replay_with_params():
    a, _p, _ev = _agent([])
    tb = a.toolbox
    tb.save_flow("quick_look", "open and image one ion",
                 [{"tool": "open_slide", "args": {"ref": "demo"}},
                  {"tool": "ion_image", "args": {"mz": "{{mz}}"}}])
    assert tb.list_flows()[0]["params"] == ["mz"]
    with pytest.raises(ValueError):
        tb.run_flow("quick_look")                                    # missing param
    out = tb.run_flow("quick_look", params={"mz": 885.55})
    assert out["completed"] and out["steps"][1]["result"]["png"]
    with pytest.raises(ValueError):                                  # no code via flows
        tb.save_flow("evil_flow", "x", [{"tool": "create_tool", "args": {}}])


def test_log_survives_on_disk(tmp_path):
    log = SessionLog(str(tmp_path))
    log.write("user", text="hi")
    log.write("tool_call", call_id="1", name="state", args={})
    log.write("tool_result", call_id="1", name="state", ok=True)
    log.close()
    lines = open(log.path, encoding="utf-8").read().splitlines()
    assert [json.loads(x)["seq"] for x in lines] == [1, 2, 3]
    assert log.tool_calls() == [{"tool": "state", "args": {}}]


# --------------------------------------------------------------------------- #
# the local web server
# --------------------------------------------------------------------------- #
@pytest.fixture
def server(monkeypatch):
    from smile_msi.agent import server as srv

    monkeypatch.setattr(srv, "make_provider", lambda cfg: FakeProvider([Step(text="hello")]))
    httpd, state, url = srv.build_server({"provider": "claude"}, port=0)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    yield httpd, state, url
    httpd.shutdown()
    httpd.server_close()


def _req(url, *, token=None, host=None, body=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-SMILE-Token"] = token
    if host:
        headers["Host"] = host
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_server_requires_the_token_and_a_local_host(server):
    _httpd, state, url = server
    base = url.split("/?")[0]
    assert _req(base + "/")[0] == 200                                 # the page itself
    assert _req(base + "/api/config")[0] == 403                       # no token
    assert _req(base + "/api/config", token="wrong")[0] == 403
    assert _req(base + "/api/config", token=state.token,
                host="evil.example:80")[0] == 403                     # DNS-rebinding guard
    code, body = _req(base + "/api/config", token=state.token)
    assert code == 200 and "api_key" not in json.loads(body)


def test_server_runs_a_turn_and_serves_only_session_files(server, tmp_path):
    _httpd, state, url = server
    base = url.split("/?")[0]
    code, _ = _req(base + "/api/send", token=state.token, body={"text": "hi"})
    assert code == 202
    for _ in range(100):
        if any(e["type"] == "done" for e in state.events):
            break
        threading.Event().wait(0.05)
    assert [e["text"] for e in state.events if e["type"] == "assistant"] == ["hello"]
    secret = tmp_path / "secret.txt"
    secret.write_text("no")
    assert _req(f"{base}/api/file?p={secret}", token=state.token)[0] == 404
    code, md = _req(base + "/api/log.md", token=state.token)
    assert code == 200 and b"hello" in md
