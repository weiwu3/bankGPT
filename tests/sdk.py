"""A stubbed API client for kb.transport.AnthropicBackend, and the SDK's errors and replies to script it with.

backend() gives the transport a client that plays a script and a clock that moves only when the
backend sleeps or a scripted call says it took time, so a question's 40-second budget is spent in
no time at all. Needs the `anthropic` package: a test module that imports this is skipped without it.
"""
import types

import pytest

from kb import transport

anthropic = pytest.importorskip("anthropic")
try:
    import httpx2 as http
except ImportError:
    import httpx as http

A, B, C = "claude-made-up-a", "claude-made-up-b", "claude-made-up-c"     # tried in this order; no table lists them
REQUEST = http.Request("POST", "https://api.anthropic.com/v1/messages")
TOKENS = types.SimpleNamespace(input_tokens=10, output_tokens=5)


def status(code, retry_after=None, cls=None):
    """What the SDK raises for an HTTP error status: a plain APIStatusError unless cls is given, so
    the backend has to read the code and not the class."""
    headers = {} if retry_after is None else {"retry-after": str(retry_after)}
    response = http.Response(code, headers=headers, request=REQUEST)
    return (cls or anthropic.APIStatusError)("stub", response=response, body=None)


def not_found():
    """What the API raises for a model the key may not use."""
    return status(404, cls=anthropic.NotFoundError)


def message(text="ok", stop_reason="end_turn", usage=TOKENS):
    """A reply as the SDK returns it."""
    return types.SimpleNamespace(stop_reason=stop_reason, content=[types.SimpleNamespace(type="text", text=text)],
                                 usage=usage)


class Takes:
    """A scripted step that takes this many seconds by the test clock before it happens."""

    def __init__(self, seconds, then):
        self.seconds, self.then = seconds, then


class Clock:
    """The backend's clock and sleep: time moves only when the backend sleeps or a call takes time."""

    def __init__(self):
        self.now, self.slept = 1000.0, []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def backend(monkeypatch, script, choices=(A, B, C), **limits):
    """An AnthropicBackend on a test clock that tries the models in choices and whose client plays
    script, one step per request: a list of steps, or a function of (call number, request fields)
    that returns one. A step is raised if it is an exception, takes time first if it is Takes, and is
    the response otherwise. limits replace class attributes for this test. Returns the backend, its
    clock and the requests sent."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    monkeypatch.setattr(transport.AnthropicBackend, "CHOICES", list(choices))
    for name, value in limits.items():
        monkeypatch.setattr(transport.AnthropicBackend, name, value)
    b, clock, sent = transport.AnthropicBackend(), Clock(), []
    b._now, b._sleep = clock, clock.sleep
    steps = list(script) if not callable(script) else None

    def create(**fields):
        sent.append(fields)
        step = script(len(sent), fields) if steps is None else steps.pop(0)
        if isinstance(step, Takes):
            clock.now += step.seconds
            step = step.then
        if isinstance(step, BaseException):
            raise step
        return step
    b.client = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
    return b, clock, sent
