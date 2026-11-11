"""The fake Anthropic API in fake_api.py, checked with the real SDK: a reply is read like a real one,
and each failure surfaces as the error the SDK raises for the real API.

No API key and no network beyond 127.0.0.1. Skipped when the `anthropic` package is not installed.
"""
import time

import pytest

anthropic = pytest.importorskip("anthropic")
from fake_api import FakeAnthropic  # noqa: E402


def ask(client, model="claude-test-model"):
    return client.messages.create(model=model, max_tokens=50, messages=[{"role": "user", "content": "Hi"}])


def test_a_reply_is_read_by_the_sdk_found_through_anthropic_base_url(monkeypatch):
    with FakeAnthropic(text="Opening hours are made up here.") as fake:
        for name, value in fake.env().items():
            monkeypatch.setenv(name, value)              # the way the scorer's process finds it
        reply = ask(anthropic.Anthropic(max_retries=0))
    assert reply.content[0].text == "Opening hours are made up here."
    assert (reply.model, reply.stop_reason, reply.usage.input_tokens) == ("claude-test-model", "end_turn", 10)
    assert [(r["call"], r["model"], r["action"], r["status"]) for r in fake.requests] == [
        (1, "claude-test-model", "reply", 200)]


def test_a_busy_api_raises_the_sdk_errors_with_the_retry_after_header():
    rules = [{"calls": 1, "do": "status", "status": 429, "retry_after": 7},
             {"model": "claude-other-model", "do": "status", "status": 529}]
    with FakeAnthropic(rules) as fake:
        client = anthropic.Anthropic(base_url=fake.url, api_key="fake-key", max_retries=0)
        with pytest.raises(anthropic.RateLimitError) as busy:
            ask(client)
        assert busy.value.response.headers["retry-after"] == "7"
        # A 529 is an InternalServerError in some SDK releases and an OverloadedError, which is not
        # one, in others (0.76 to 2.0): only its status code is the same in all.
        with pytest.raises(anthropic.APIStatusError) as overloaded:
            ask(client, "claude-other-model")
        assert overloaded.value.status_code == 529
        assert ask(client).content[0].text                                   # call 3: no rule, a reply
    assert [(r["call"], r["status"]) for r in fake.requests] == [(1, 429), (2, 529), (3, 200)]


def test_a_hang_ends_at_the_client_timeout_a_closed_call_is_a_connection_error_and_stop_does_not_wait():
    with FakeAnthropic([{"calls": 1, "do": "hang", "seconds": 30}, {"calls": 2, "do": "close"}]) as fake:
        client = anthropic.Anthropic(base_url=fake.url, api_key="fake-key", max_retries=0, timeout=0.5)
        start = time.monotonic()
        with pytest.raises(anthropic.APITimeoutError):
            ask(client)
        assert 0.4 < time.monotonic() - start < 2.5
        with pytest.raises(anthropic.APIConnectionError) as closed:
            ask(client)
        assert not isinstance(closed.value, anthropic.APITimeoutError)
        start = time.monotonic()
    assert time.monotonic() - start < 1.0                # call 1's handler was still hanging
    assert [(r["action"], r["status"]) for r in fake.requests] == [("hang", None), ("close", None)]


def test_a_rule_can_cover_the_calls_that_arrive_in_a_window_of_time_after_the_first():
    with FakeAnthropic([{"from_s": 0.3, "to_s": 1.0, "do": "status", "status": 503}]) as fake:
        client = anthropic.Anthropic(base_url=fake.url, api_key="fake-key", max_retries=0)
        ask(client)                                      # the first call starts the clock
        time.sleep(0.4)
        with pytest.raises(anthropic.APIStatusError):
            ask(client)
        time.sleep(0.7)
        ask(client)
    assert [r["status"] for r in fake.requests] == [200, 503, 200]
