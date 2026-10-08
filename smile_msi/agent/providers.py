"""Model providers — bring your own model.

Each provider keeps the conversation in its **own native format** (so nothing is lost in
translation — Claude's thinking blocks are echoed back untouched, a local model's tool-call
ids stay its own) and exposes the same four calls to the agent loop:

    start(system, tools)       the stable system prompt + tool definitions
    add_user(text)             a user message
    step() -> Step             one model call: text, reasoning summary, tool calls, usage
    add_tool_outputs(outputs)  every result for the previous step's calls, in one message

:class:`AnthropicProvider` talks to Claude through the official ``anthropic`` SDK (API key
from the settings, or ``ANTHROPIC_API_KEY`` / an ``ant auth login`` profile).
:class:`OpenAICompatProvider` talks to any server with an OpenAI-style
``/chat/completions`` endpoint — Ollama, LM Studio, llama.cpp ``server``, vLLM — over plain
HTTP from the standard library, so a fully local setup needs no extra package. Tool use on a
local model is only as good as that model's function calling; pick one trained for it.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .tools import MAX_MODEL_IMAGES, image_b64, model_image_ok

DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
DEFAULT_LOCAL_URL = "http://localhost:11434/v1"      # Ollama's OpenAI-compatible endpoint
#: Images sent to the model over a whole conversation (the API allows 100 per request, and
#: the history is resent every step). Past this, results say an image exists but don't attach.
IMAGE_BUDGET = 50


def _pick_images(paths, sent: int) -> tuple[list, str]:
    """The images of one tool result that can go to the model, and a note for the rest."""
    usable = [p for p in paths if model_image_ok(p)]
    skipped = len(paths) - len(usable)
    room = max(0, min(MAX_MODEL_IMAGES, IMAGE_BUDGET - sent))
    take = usable[:room]
    notes = []
    if skipped:
        notes.append(f"{skipped} image(s) not attached (not PNG/JPEG or too large)")
    if len(usable) > len(take):
        notes.append(f"{len(usable) - len(take)} image(s) not attached (image limit for this "
                     "conversation) — the user still sees them")
    return take, "; ".join(notes)


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict
    parse_error: str = ""


@dataclass
class ToolOutput:
    call_id: str
    text: str
    images: list = field(default_factory=list)       # image file paths
    is_error: bool = False


@dataclass
class Step:
    text: str = ""
    thinking: str = ""
    calls: list = field(default_factory=list)
    stop: str = ""
    usage: dict = field(default_factory=dict)
    note: str = ""                                   # refusal / truncation explanation


class ProviderError(RuntimeError):
    """A model call failed in a way the user should see (bad key, server down, …)."""


# --------------------------------------------------------------------------- #
# Claude
# --------------------------------------------------------------------------- #
class AnthropicProvider:
    name = "claude"

    def __init__(self, model: str = DEFAULT_CLAUDE_MODEL, api_key: str = "",
                 effort: str = "high", max_tokens: int = 16000):
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderError("Claude needs the 'anthropic' package:  "
                                "uv pip install -e '.[agent]'") from exc
        self._anthropic = anthropic
        try:
            self.client = (anthropic.Anthropic(api_key=api_key) if api_key
                           else anthropic.Anthropic())
        except anthropic.AnthropicError as exc:
            raise ProviderError("No Claude credentials found — paste an API key in Settings, "
                                "set ANTHROPIC_API_KEY, or run `ant auth login`.") from exc
        self.model = model or DEFAULT_CLAUDE_MODEL
        self.effort = effort
        self.max_tokens = int(max_tokens)
        self.messages: list = []
        self.system = ""
        self.tools: list = []
        self.images_sent = 0

    def settings(self) -> dict:
        return {"model": self.model, "effort": self.effort, "max_tokens": self.max_tokens}

    def start(self, system: str, tools: list):
        self.system, self.tools = system, list(tools)

    def set_tools(self, tools: list):
        self.tools = list(tools)

    def add_user(self, text: str):
        self.messages.append({"role": "user", "content": text})

    def _create(self):
        return self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            tools=self.tools,
            messages=self.messages,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
            # a safety-classifier decline is re-run on Anthropic's recommended model
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )

    def _drop_last_images(self) -> bool:
        """The request was rejected over an image: replace the images in the tool results
        just added (not yet seen by the model) with a note. Earlier turns are never edited."""
        last = self.messages[-1] if self.messages else None
        if not last or last["role"] != "user" or not isinstance(last["content"], list):
            return False
        changed = False
        for block in last["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                kept = [c for c in block["content"] if c.get("type") != "image"]
                if len(kept) != len(block["content"]):
                    kept.append({"type": "text", "text": "(image not attached: the API "
                                                         "rejected it)"})
                    block["content"] = kept
                    changed = True
        return changed

    def step(self) -> Step:
        a = self._anthropic
        try:
            try:
                resp = self._create()
            except a.BadRequestError as exc:
                if "image" in str(exc.message).lower() and self._drop_last_images():
                    resp = self._create()
                else:
                    raise
        except TypeError as exc:                 # the SDK found no API key / token / profile
            if "auth" in str(exc).lower():
                raise ProviderError("No Claude credentials found — paste an API key in "
                                    "Settings, set ANTHROPIC_API_KEY, or run "
                                    "`ant auth login`.") from exc
            raise
        except a.AuthenticationError as exc:
            raise ProviderError("Claude rejected the API key — check it in Settings.") from exc
        except a.PermissionDeniedError as exc:
            raise ProviderError(f"This key can't use {self.model}: {exc.message}") from exc
        except a.NotFoundError as exc:
            raise ProviderError(f"Unknown model {self.model!r}: {exc.message}") from exc
        except a.RateLimitError as exc:
            raise ProviderError("Rate limited by the API — wait a moment and retry.") from exc
        except a.BadRequestError as exc:
            raise ProviderError(f"Claude rejected the request: {exc.message}") from exc
        except a.APIStatusError as exc:
            raise ProviderError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        except a.APIConnectionError as exc:
            raise ProviderError("Can't reach the Claude API — check the network.") from exc

        # Append-only history: the content (thinking, fallback blocks) goes back as is — except
        # tool calls we will not run (a refusal, or a reply cut off mid-call), which would
        # otherwise sit unanswered and make every later request fail.
        content = list(resp.content)
        if resp.stop_reason in ("refusal", "max_tokens"):
            content = [b for b in content if b.type != "tool_use"]
            if not content:
                content = [{"type": "text", "text": "(no reply)"}]
        self.messages.append({"role": "assistant", "content": content})
        out = Step(stop=resp.stop_reason or "")
        texts, thoughts = [], []
        for block in resp.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "thinking" and getattr(block, "thinking", ""):
                thoughts.append(block.thinking)
            elif block.type == "tool_use":
                out.calls.append(ToolCall(block.id, block.name, dict(block.input or {})))
        out.text, out.thinking = "\n\n".join(texts), "\n\n".join(thoughts)
        u = resp.usage
        out.usage = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                     "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
                     "model": resp.model}
        if resp.stop_reason == "refusal":
            cat = getattr(getattr(resp, "stop_details", None), "category", None)
            out.note = f"The model declined this request (category: {cat or 'unspecified'})."
            out.calls = []
        elif resp.stop_reason == "max_tokens":
            out.note = ("The reply hit the output limit and was cut short"
                        + (" before its tool call was complete." if out.calls else "."))
            out.calls = []
        return out

    def add_tool_outputs(self, outputs: list):
        blocks = []
        for o in outputs:
            take, note = _pick_images(o.images, self.images_sent)
            content = [{"type": "text", "text": (o.text or "(no output)")
                        + (f"\n[{note}]" if note else "")}]
            for path in take:
                try:
                    media, data = image_b64(path)
                except OSError:
                    continue
                content.append({"type": "image",
                                "source": {"type": "base64", "media_type": media, "data": data}})
                self.images_sent += 1
            blocks.append({"type": "tool_result", "tool_use_id": o.call_id,
                           "content": content, "is_error": bool(o.is_error)})
        self.messages.append({"role": "user", "content": blocks})


# --------------------------------------------------------------------------- #
# local / OpenAI-compatible
# --------------------------------------------------------------------------- #
class OpenAICompatProvider:
    name = "local"

    def __init__(self, model: str, base_url: str = DEFAULT_LOCAL_URL, api_key: str = "",
                 vision: bool = False, timeout: float = 600.0):
        if not model:
            raise ProviderError("Name the local model to use (e.g. one listed by `ollama list`).")
        self.model = model
        self.base_url = (base_url or DEFAULT_LOCAL_URL).rstrip("/")
        self.api_key = api_key
        self.vision = bool(vision)
        self.timeout = float(timeout)
        self.messages: list = []
        self.tools: list = []
        self.images_sent = 0

    def settings(self) -> dict:
        return {"model": self.model, "base_url": self.base_url, "vision": self.vision}

    def start(self, system: str, tools: list):
        self.messages = [{"role": "system", "content": system}]
        self.set_tools(tools)

    def set_tools(self, tools: list):
        self.tools = [{"type": "function",
                       "function": {"name": t["name"], "description": t["description"],
                                    "parameters": t["input_schema"]}} for t in tools]

    def add_user(self, text: str):
        self.messages.append({"role": "user", "content": text})

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read().decode("utf-8", "replace")
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ProviderError(f"The model server sent a reply that isn't JSON: "
                                    f"{raw[:200]!r}") from exc
        except TimeoutError as exc:
            raise ProviderError(f"The model server took longer than {self.timeout:.0f}s to "
                                "reply.") from exc
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:500]
            raise ProviderError(f"Local model server returned {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"Can't reach {self.base_url} — is the model server running? "
                                f"({exc.reason})") from exc

    def step(self) -> Step:
        data = self._post({"model": self.model, "messages": self.messages,
                           "tools": self.tools, "tool_choice": "auto"})
        try:
            choice = data["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Unexpected reply from the model server: {str(data)[:300]}") \
                from exc
        if not isinstance(msg, dict):
            raise ProviderError(f"Unexpected reply from the model server: {str(data)[:300]}")
        text = msg.get("content") if isinstance(msg.get("content"), str) else ""
        out = Step(text=text, stop=choice.get("finish_reason") or "",
                   thinking=msg.get("reasoning_content") or msg.get("reasoning") or "")
        stored_calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            tc = tc if isinstance(tc, dict) else {}
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            cid = tc.get("id") or f"call_{len(self.messages)}_{i}"
            raw = fn.get("arguments")
            err = ""
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(args, str):                       # double-encoded
                    args = json.loads(args)
            except json.JSONDecodeError as exc:
                args, err = {}, f"tool arguments were not valid JSON: {exc}"
            if args is None:
                args = {}
            if not isinstance(args, dict) and not err:
                args, err = {}, "tool arguments must be a JSON object"
            name = fn.get("name") or ""
            # the stored call carries the same id the tool result will answer
            stored_calls.append({"id": cid, "type": "function",
                                 "function": {"name": name, "arguments": json.dumps(args)}})
            out.calls.append(ToolCall(cid, name, args, err))
        keep = {"role": "assistant", "content": text}
        if stored_calls:
            keep["tool_calls"] = stored_calls
        self.messages.append(keep)
        u = data.get("usage") or {}
        out.usage = {"input_tokens": u.get("prompt_tokens", 0),
                     "output_tokens": u.get("completion_tokens", 0), "model": self.model}
        if out.stop == "length":
            out.note = "The reply hit the output limit and was cut short."
        return out

    def add_tool_outputs(self, outputs: list):
        images = []
        for o in outputs:
            text = ("ERROR: " if o.is_error and not o.text.startswith("ERROR") else "") \
                + (o.text or "(no output)")
            self.messages.append({"role": "tool", "tool_call_id": o.call_id, "content": text})
            images += o.images
        if self.vision and images:          # tool messages can't carry images on most servers
            take, _note = _pick_images(images, self.images_sent)
            parts = [{"type": "text", "text": "Images produced by the tool calls above:"}]
            for path in take:
                try:
                    media, data = image_b64(path)
                except OSError:
                    continue
                parts.append({"type": "image_url",
                              "image_url": {"url": f"data:{media};base64,{data}"}})
                self.images_sent += 1
            if len(parts) > 1:
                self.messages.append({"role": "user", "content": parts})


def make_provider(cfg: dict):
    """Build a provider from a settings dict (``provider``: ``claude`` | ``local``)."""
    kind = (cfg.get("provider") or "claude").lower()
    if kind == "local":
        return OpenAICompatProvider(model=cfg.get("model", ""),
                                    base_url=cfg.get("base_url") or DEFAULT_LOCAL_URL,
                                    api_key=cfg.get("api_key", ""),
                                    vision=bool(cfg.get("vision")))
    return AnthropicProvider(model=cfg.get("model") or DEFAULT_CLAUDE_MODEL,
                             api_key=cfg.get("api_key", ""),
                             effort=cfg.get("effort") or "high")
