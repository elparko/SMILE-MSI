"""The analysis Reviewer (:mod:`smile_msi.agent.review`): deterministic rule checks over the
session log, a one-shot model report, the ``review`` log event / Markdown section, the two
providers' ``complete()`` and ``POST /api/review`` — all with a scripted fake model."""
import json
import threading
import time
import types
import urllib.error
import urllib.request

import pytest

from smile_msi import mcpserver
from smile_msi.agent import review as rv
from smile_msi.agent import setup as setup_mod
from smile_msi.agent.core import Agent
from smile_msi.agent.providers import (AnthropicProvider, OpenAICompatProvider, ProviderError,
                                       Step, ToolCall)

REPORT = "## Major issues\n- Pixels are not replicates.\n## Minor issues\nNone found.\n"


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("SMILE_MSI_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SMILE_MSI_MCP_OUT", str(tmp_path / "out"))
    yield
    mcpserver.close_slide()


class FakeProvider:
    """Plays back a script of steps; ``complete()`` returns a canned report."""
    name = "fake"

    def __init__(self, script=(), report=REPORT):
        self.script = list(script)
        self.report = report
        self.users, self.completions = [], []

    def settings(self):
        return {"model": "scripted"}

    def start(self, system, tools_):
        pass

    def set_tools(self, tools_):
        pass

    def add_user(self, text):
        self.users.append(text)

    def step(self):
        return self.script.pop(0)

    def add_tool_outputs(self, outputs):
        pass

    def complete(self, system, text):
        self.completions.append((system, text))
        if isinstance(self.report, Exception):
            raise self.report
        return self.report


def _call(tool, /, **args):
    return Step(calls=[ToolCall(f"c-{tool}", tool, args)],
                usage={"input_tokens": 10, "output_tokens": 5})


# --------------------------------------------------------------------------- #
# synthetic log records
# --------------------------------------------------------------------------- #
def _setup(replicate="regions / sections", question="Which lipids mark the lesion?", max_q=0.05):
    return {"study": {"question": question, "replicate": replicate, "max_q": max_q,
                      "min_auc": 0.7}, "params": {}}


class Log:
    """Builds a list of log records with increasing ``seq``."""

    def __init__(self):
        self.records = []

    def add(self, type_, **kw):
        self.records.append({"seq": len(self.records) + 1, "t": "2026-01-01T10:00:00.000",
                             "type": type_, **kw})
        return self

    def user(self, text="go"):
        return self.add("user", text=text)

    def say(self, text):
        return self.add("assistant", text=text)

    def tool(self, name, ok=True, text="done", error=None, **args):
        cid = f"c{len(self.records) + 1}"
        self.add("tool_call", call_id=cid, name=name, args=args)
        return self.add("tool_result", call_id=cid, name=name, ok=ok,
                        error=error or (None if ok else "it broke"), text=text)


def _titles(findings):
    return [f.title for f in findings]


def _has(findings, fragment):
    return [f for f in findings if fragment in f.title]


# --------------------------------------------------------------------------- #
# a. pixel replication
# --------------------------------------------------------------------------- #
def test_pixel_replication_fires_on_a_stats_step():
    lg = Log().user().tool("run_analysis", step_id="discriminating_features",
                           text="ok: 12 features")
    hit = _has(rv.rule_checks(lg.records, _setup("pixels (exploratory)")), "pixels")
    assert len(hit) == 1 and hit[0].severity == "major"
    assert 3 in hit[0].evidence and "c2" in hit[0].evidence      # the result's seq and call id


def test_pixel_replication_detects_p_columns_in_any_result():
    lg = Log().user().tool("run_script", text="mz  AUC  p_value  q_value\n760.58 0.9 1e-8 1e-6")
    assert _has(rv.rule_checks(lg.records, _setup("pixels (exploratory)")), "pixels")


def test_pixel_replication_quiet_when_replicates_are_real_or_no_stats():
    stats = Log().user().tool("run_analysis", step_id="multigroup_features", text="ok")
    assert not _has(rv.rule_checks(stats.records, _setup("animals / patients")), "pixels")
    plain = Log().user().tool("ion_image", mz=760.58, text="image made")
    assert not _has(rv.rule_checks(plain.records, _setup("pixels (exploratory)")), "pixels")
    failed = Log().user().tool("run_analysis", ok=False, step_id="roi_comparison")
    assert not _has(rv.rule_checks(failed.records, _setup("pixels (exploratory)")), "pixels")


# --------------------------------------------------------------------------- #
# b. ignored failure
# --------------------------------------------------------------------------- #
def test_ignored_failure_fires_and_acknowledgement_silences_it():
    bad = Log().user().tool("ion_image", ok=False, mz=500.0).say("Here is a summary of the slide.")
    hit = _has(rv.rule_checks(bad.records, _setup()), "never acknowledged")
    assert len(hit) == 1 and hit[0].severity == "major" and "ion_image" in hit[0].title

    said = Log().user().tool("ion_image", ok=False, mz=500.0) \
        .say("That ion image failed — no slide was open, so I'll open one first.")
    assert not _has(rv.rule_checks(said.records, _setup()), "never acknowledged")


def test_ignored_failure_ack_must_come_after_and_in_the_same_turn():
    early = Log().user().say("There was an error earlier.").tool("x", ok=False) \
        .say("All done, here are the results.")
    assert _has(rv.rule_checks(early.records, _setup()), "never acknowledged")
    nxt = Log().user().tool("x", ok=False).say("Moving on.").user("again") \
        .say("Sorry, that failed.")
    assert _has(rv.rule_checks(nxt.records, _setup()), "never acknowledged")


def test_ignored_failure_skips_user_decisions_and_softens_silent_retries():
    rejected = Log().user().tool("create_tool", ok=False,
                                 error="the scientist rejected this: no").say("Okay.")
    assert not _has(rv.rule_checks(rejected.records, _setup()), "never acknowledged")
    retried = Log().user().tool("find_peaks", ok=False).tool("find_peaks").say("Found peaks.")
    hit = _has(rv.rule_checks(retried.records, _setup()), "never acknowledged")
    assert hit and hit[0].severity == "minor"


# --------------------------------------------------------------------------- #
# c. unhedged identifications
# --------------------------------------------------------------------------- #
def test_unhedged_lipid_names_fire_and_hedges_silence_them():
    bare = Log().user().say("The lesion is rich in PC 34:1 and Cer d18:1/24:0.")
    hit = _has(rv.rule_checks(bare.records, _setup()), "identities")
    assert len(hit) == 1 and hit[0].severity == "minor" and hit[0].evidence == [2]
    assert "PC 34:1" in hit[0].detail

    for text in ("Putatively PC 34:1 ([M+H]+, 1.8 ppm).",
                 "PC 34:1 at sum composition level, to be confirmed.",
                 "A candidate for PE 38:4 pending MS/MS."):
        lg = Log().user().say(text)
        assert not _has(rv.rule_checks(lg.records, _setup()), "identities"), text


def test_lipid_regex_ignores_prose_that_only_looks_like_lipids():
    for text in ("PCA separates the groups.", "Compare PC 1:1 against the SM 2:1 ratio.",
                 "The PC1 axis explains 40% of 12:30 variance.", "Sulfatide is abundant here."):
        lg = Log().user().say(text)
        assert not _has(rv.rule_checks(lg.records, _setup()), "identities"), text


# --------------------------------------------------------------------------- #
# d. deviations
# --------------------------------------------------------------------------- #
def test_deviations_are_listed():
    lg = Log().user().tool("find_peaks", snr=1.7)
    lg.add("deviation", call_id="c2", name="find_peaks",
           deviations=[{"arg": "snr", "setup": 5.0, "used": 1.7}])
    hit = _has(rv.rule_checks(lg.records, _setup()), "departed")
    assert len(hit) == 1 and hit[0].severity == "minor" and hit[0].evidence == [4]
    assert "snr = 1.7 (setup 5.0)" in hit[0].detail
    assert not _has(rv.rule_checks(Log().user().records, _setup()), "departed")


# --------------------------------------------------------------------------- #
# e. unrecorded setup, g. no question
# --------------------------------------------------------------------------- #
def test_setup_accepted_without_review_is_noted():
    lg = Log().add("setup", how="defaults accepted without review (the setup card was skipped)",
                   **_setup())
    hit = _has(rv.rule_checks(lg.records, None), "without review")
    assert len(hit) == 1 and hit[0].severity == "note" and hit[0].evidence == [1]
    ok = Log().add("setup", how="confirmed by the scientist", **_setup())
    assert not _has(rv.rule_checks(ok.records, None), "without review")


def test_missing_question_is_noted():
    assert _has(rv.rule_checks([], _setup(question="  ")), "question")
    assert not _has(rv.rule_checks([], _setup()), "question")
    assert not _has(rv.rule_checks([], None), "question")        # no setup at all: can't say


# --------------------------------------------------------------------------- #
# f. bare p-values
# --------------------------------------------------------------------------- #
def test_bare_pvalues_fire_unless_an_effect_size_is_near():
    bare = Log().user().say("m/z 760.58 is higher in the lesion (q = 0.003).")
    hit = _has(rv.rule_checks(bare.records, _setup()), "effect size")
    assert len(hit) == 1 and hit[0].severity == "minor" and hit[0].evidence == [2]

    for text in ("m/z 760.58 is higher in the lesion (AUC 0.91, q = 0.003).",
                 "q < 1e-5 with a 2.1-fold change in the lesion.",
                 "The log2 fold change is 1.1 (p = 0.002).",
                 "Cohen's d = 1.3, p < 0.001."):
        lg = Log().user().say(text)
        assert not _has(rv.rule_checks(lg.records, _setup()), "effect size"), text


def test_restating_the_threshold_or_naming_p_values_is_not_a_report():
    for text in ("Features are significant at q ≤ 0.05.",
                 "p-values describe pixels here, so lean on effect sizes.",
                 "I used a Benjamini–Hochberg correction."):
        lg = Log().user().say(text)
        assert not _has(rv.rule_checks(lg.records, _setup()), "effect size"), text


# --------------------------------------------------------------------------- #
# a clean analysis raises nothing; ordering
# --------------------------------------------------------------------------- #
def test_a_clean_analysis_has_no_findings():
    lg = (Log().add("setup", how="confirmed by the scientist", **_setup("animals / patients"))
          .user("compare lesion and control")
          .say("I'll compare the two groups with a per-animal test.")
          .tool("run_analysis", step_id="roi_comparison", text="ok")
          .say("m/z 760.58, putatively PC 34:1 (sum composition, [M+H]+), is higher in the "
               "lesion: AUC 0.92, q = 0.004 across 6 animals."))
    assert rv.rule_checks(lg.records, None) == []


def test_findings_come_most_severe_first_and_serialise():
    lg = (Log().user().tool("x", ok=False).say("The lesion has PC 34:1 (q = 0.01).")
          .tool("run_analysis", step_id="roi_comparison"))
    found = rv.rule_checks(lg.records, _setup("pixels (exploratory)", question=""))
    sev = [f.severity for f in found]
    assert sev == sorted(sev, key=rv.SEVERITIES.index) and sev[0] == "major" and sev[-1] == "note"
    json.dumps([f.to_dict() for f in found])


# --------------------------------------------------------------------------- #
# review(agent): a scripted turn, the event, the Markdown section
# --------------------------------------------------------------------------- #
def _scripted_agent(report=REPORT):
    events = []
    p = FakeProvider([
        _call("ion_image", mz=500.0),                      # no slide open: fails, unacknowledged
        _call("open_slide", ref="demo"),
        _call("find_peaks", snr=1.7),                      # not the setup's snr: a deviation
        Step(text="The top feature is PC 34:1 (q = 0.001).")], report=report)
    a = Agent(p, emit=events.append, setup=setup_mod.resolve({"question": ""}))
    a.send("what is the top lipid?")
    return a, p, events


def test_review_logs_an_event_with_findings_and_the_model_report():
    a, p, events = _scripted_agent()
    assert not [e for e in events if e["type"] == "error"], events
    before_users = list(p.users)
    ev = rv.review(a, use_llm=True)

    assert ev["type"] == "review" and a.log.records[-1] is ev and events[-1] is ev
    assert ev["llm"] == REPORT.strip() and ev["model"] == "scripted" and ev["llm_error"] == ""
    titles = [f["title"] for f in ev["findings"]]
    assert any("never acknowledged" in t for t in titles)
    assert any("identities" in t for t in titles)
    assert any("departed" in t for t in titles)
    assert any("effect size" in t for t in titles)
    assert any("question" in t for t in titles)
    assert [f["severity"] for f in ev["findings"]][0] == "major"
    assert [e["state"] for e in events if e["type"] == "status"][-1] == "reviewing"

    system, text = p.completions[0]                          # a reviewer prompt, with the log
    assert "peer reviewer" in system and "Major issues" in system
    assert "ion_image" in text and "PC 34:1" in text and "Analysis setup" in text
    assert p.users == before_users                           # the conversation was not touched

    md = a.log.markdown()
    assert "## Review" in md and "Pixels are not replicates." in md
    assert "> ## Major issues" in md                         # the report is quoted
    assert "**major**" in md
    assert "## Review" not in a.log.markdown(include_reviews=False)

    rv.review(a, use_llm=False)                              # a second review, rules only
    assert "## Review" not in p.completions[-1][1]
    assert len(p.completions) == 1 and a.log.records[-1]["llm"] == ""
    with open(a.log.path, encoding="utf-8") as fh:
        assert [json.loads(x)["type"] for x in fh].count("review") == 2


def test_a_failed_model_call_keeps_the_rule_findings():
    a, _p, _ev = _scripted_agent(report=ProviderError("Rate limited by the API"))
    ev = rv.review(a, use_llm=True)
    assert ev["llm"] == "" and "Rate limited" in ev["llm_error"] and ev["findings"]
    assert "The model review failed: Rate limited" in a.log.markdown()


def test_a_provider_without_complete_is_reported_not_raised():
    a, p, _ev = _scripted_agent()
    a.provider = types.SimpleNamespace(settings=p.settings)          # no complete()
    ev = rv.review(a, use_llm=True)
    assert "can't write a review" in ev["llm_error"] and ev["findings"]


def test_long_logs_are_clipped_from_the_middle():
    text = "A" * 1000 + "B" * 500_000 + "C" * 1000
    out = rv._clip(text)
    assert len(out) < rv.MAX_LOG_CHARS + 100 and out.startswith("AAA") and out.endswith("CCC")
    assert "omitted" in out


# --------------------------------------------------------------------------- #
# providers: complete() is one-shot and touches no history
# --------------------------------------------------------------------------- #
def _block(type_, **kw):
    return types.SimpleNamespace(type=type_, **kw)


def test_anthropic_complete_is_a_one_shot_call():
    p = AnthropicProvider(model="m", api_key="sk-test", effort="medium")
    p.add_user("hello")
    seen = {}

    def create(**kw):
        seen.update(kw)
        return types.SimpleNamespace(stop_reason="end_turn", content=[
            _block("thinking", thinking="hmm"), _block("text", text="Part one."),
            _block("text", text="Part two.")])

    p.client = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
    assert p.complete("SYS", "TEXT") == "Part one.\n\nPart two."
    assert seen == {"model": "m", "max_tokens": 16000, "system": "SYS",
                    "messages": [{"role": "user", "content": "TEXT"}],
                    "thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}}
    assert p.messages == [{"role": "user", "content": "hello"}]        # history untouched

    p.client = types.SimpleNamespace(messages=types.SimpleNamespace(
        create=lambda **kw: types.SimpleNamespace(stop_reason="refusal", content=[])))
    assert "declined" in p.complete("S", "T")


def test_anthropic_complete_maps_sdk_errors():
    import anthropic
    httpx = pytest.importorskip("httpx2")           # the SDK's HTTP client

    p = AnthropicProvider(api_key="sk-test")
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

    def boom(**kw):
        raise anthropic.RateLimitError("slow down", response=httpx.Response(429, request=req),
                                       body=None)

    p.client = types.SimpleNamespace(messages=types.SimpleNamespace(create=boom))
    with pytest.raises(ProviderError, match="Rate limited"):
        p.complete("S", "T")


def test_openai_compat_complete_posts_without_tools():
    p = OpenAICompatProvider(model="local-m")
    p.start("chat system", [])
    sent = []
    p._post = lambda payload: (sent.append(payload),
                               {"choices": [{"message": {"content": "A report."}}]})[1]
    assert p.complete("REVIEW SYS", "TEXT") == "A report."
    assert "tools" not in sent[0] and sent[0]["messages"] == [
        {"role": "system", "content": "REVIEW SYS"}, {"role": "user", "content": "TEXT"}]
    assert p.messages == [{"role": "system", "content": "chat system"}]
    p._post = lambda payload: {"nope": 1}
    with pytest.raises(ProviderError):
        p.complete("S", "T")


# --------------------------------------------------------------------------- #
# POST /api/review
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


def _post(url, body, token):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "X-SMILE-Token": token or ""})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _wait(pred, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_api_review_runs_on_a_worker_and_streams_the_event(server):
    _httpd, state, url = server
    api = url.split("/?")[0] + "/api/review"
    assert _post(api, {"llm": True}, None)[0] == 403                  # same token checks
    assert _post(api, {"llm": True}, "wrong")[0] == 403

    state.agent.busy = True                                           # a turn is running
    code, body = _post(api, {"llm": True}, state.token)
    assert code == 409 and "still working" in body["error"]
    state.agent.busy = False

    code, body = _post(api, {"llm": True}, state.token)
    assert code == 202 and body == {"ok": True}
    assert _wait(lambda: any(e["type"] == "done" for e in state.events))
    reviews = [e for e in state.events if e["type"] == "review"]
    assert len(reviews) == 1 and reviews[0]["llm"] == REPORT.strip()
    assert not state.agent.busy
    assert "## Review" in state.agent.log.markdown()

    assert _post(api, {"llm": False}, state.token)[0] == 202          # rules only
    assert _wait(lambda: len([e for e in state.events if e["type"] == "review"]) == 2)
    assert state.events and [e for e in state.events if e["type"] == "review"][-1]["llm"] == ""
    assert len(state.agent.provider.completions) == 1
