"""What the API transport (kb.transport.AnthropicBackend) sends, what it does when the API fails, and
what it counts, checked without waiting.

Most tests give the backend the stub client of sdk.py, on a clock that moves only when the backend
sleeps or a scripted call says it took time. The watchdog's tests and the tests that use the
installed SDK against the local fake API in fake_api.py run on the real clock with the time limits
shrunk through the class attributes. No API key and no network beyond 127.0.0.1; skipped when the
`anthropic` package is not installed.
"""
import itertools
import json
import os
import subprocess
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

anthropic = pytest.importorskip("anthropic")

from fake_api import FakeAnthropic  # noqa: E402
from helpers import GONE, NOW, REFUSAL, ROOT, SMALL_SITE, SPRING, build  # noqa: E402
from kb import reader, transport  # noqa: E402
from scoring import score_speed  # noqa: E402
from sdk import REQUEST, A, B, C, Takes, backend, message, not_found, status  # noqa: E402

PLAIN = {"model", "max_tokens", "system", "messages", "timeout"}           # what every request carries
TESTS = str(ROOT / "tests")


def closed():
    return anthropic.APIConnectionError(request=REQUEST)


def timed_out():
    return anthropic.APITimeoutError(request=REQUEST)


def models(sent):
    return [r["model"] for r in sent]


def lines(capsys):
    return [x for x in capsys.readouterr().err.splitlines() if x.startswith("[bankgpt]")]


# --------------------------------------------------------------------------- a question that is answered


def test_a_question_is_one_request_with_the_fields_it_always_had_plus_a_timeout_and_no_wait(monkeypatch, capsys):
    b, clock, sent = backend(monkeypatch, [message("hello")], choices=["claude-sonnet-5-5", A])
    assert b.complete("the instructions", "the question") == "hello"
    assert len(sent) == 1 and clock.slept == [] and lines(capsys) == []
    assert sent[0] == dict(
        model="claude-sonnet-5-5", max_tokens=600, system="the instructions",
        messages=[{"role": "user", "content": "the question"}], timeout=anthropic.Timeout(25.0, connect=5.0),
        **transport.SETTINGS["claude-sonnet-5-5"])
    assert transport.USAGE.models["claude-sonnet-5-5"]["calls"] == 1 and transport.USAGE.unanswered == 0


@pytest.mark.parametrize("dated", ["claude-sonnet-5-5-20260915", "claude-haiku-5-5-20261001", "claude-haiku-4-5-20251001"])
def test_a_dated_model_name_is_sent_the_settings_of_its_alias(monkeypatch, dated):
    alias = dated[:-len("-20260915")]
    assert transport.settings(dated) == transport.SETTINGS[alias]
    b, _, sent = backend(monkeypatch, [message()], choices=[dated])
    assert b.complete("s", "u") == "ok"
    assert {k: v for k, v in sent[0].items() if k not in PLAIN} == transport.SETTINGS[alias]
    assert transport.settings(A) == {} and transport.settings("claude-sonnet-5-5-latest") == {}    # only an 8-digit date is cut


def test_a_refusal_is_a_reply_it_is_counted_and_gives_none(monkeypatch):
    b, _, sent = backend(monkeypatch, [message("no", stop_reason="refusal"), message("", stop_reason="end_turn")])
    assert b.complete("s", "u") is None and b.complete("s", "u") is None and len(sent) == 2
    assert transport.USAGE.models[A]["calls"] == 2 and transport.USAGE.unanswered == 0 and b.misses == 0


# --------------------------------------------------------------------------- what a failed request means


@pytest.mark.parametrize("rejection", [status(400), status(403), status(404), status(404, cls=anthropic.NotFoundError),
                                     TypeError("create() got an unexpected keyword argument 'output_config'")])
def test_a_model_that_rejects_before_it_ever_answered_is_dropped_and_the_next_one_asked_at_once(monkeypatch, capsys,
                                                                                                rejection):
    b, clock, sent = backend(monkeypatch, [rejection, message("from b"), message("b again")])
    assert b.complete("s", "u") == "from b" and clock.now == 1000.0 and clock.slept == []
    assert b.complete("s", "u") == "b again"
    assert models(sent) == [A, B, B] and b.dropped == {A}
    said = lines(capsys)
    assert len(said) == 1 and A in said[0]
    assert ("pip install -U -r requirements.txt" in said[0]) is isinstance(rejection, TypeError)


@pytest.mark.parametrize("code", [400, 403, 404])
def test_a_rejection_from_a_model_that_has_answered_costs_that_question_only(monkeypatch, code):
    b, clock, sent = backend(monkeypatch, [message("first"), status(code), message("third")])
    assert [b.complete("s", "u") for _ in range(3)] == ["first", None, "third"]
    assert models(sent) == [A, A, A] and b.dropped == set() and clock.slept == []


@pytest.mark.parametrize("code", [413, 422, 302])
def test_any_other_status_gives_up_on_that_question_only(monkeypatch, code):
    b, clock, sent = backend(monkeypatch, [status(code), message("next")])
    assert b.complete("s", "u") is None and len(sent) == 1 and clock.slept == []
    assert b.complete("s", "u") == "next" and models(sent) == [A, A]


def test_a_rejected_key_switches_the_reader_off_after_one_call(monkeypatch, capsys):
    b, _, sent = backend(monkeypatch, lambda n, fields: status(401))
    assert [b.complete("s", "u") for _ in range(5)] == [None] * 5
    assert len(sent) == 1 and b.off and transport.USAGE.unanswered == 5
    assert len(lines(capsys)) == 1


@pytest.mark.parametrize("busy", [status(408), status(409), status(429), status(500), status(503), status(529),
                                  status(529, cls=anthropic.InternalServerError), status(429, cls=anthropic.RateLimitError)])
def test_a_busy_model_is_left_alone_while_the_next_one_answers_the_same_question(monkeypatch, busy):
    b, clock, sent = backend(monkeypatch, [busy, message("from b"), message("b again"), message("a again")])
    assert b.complete("s", "u") == "from b" and clock.slept == []       # at once, without waiting
    assert b.complete("s", "u") == "b again"                            # A is still busy (2 s)
    clock.now += 2.0
    assert b.complete("s", "u") == "a again"                            # free again: the first choice again
    assert models(sent) == [A, B, B, A] and b.busy == {}                # its reply cleared its mark


# A header under one second counts as none: the API sends whole seconds, and 0.001 s would send
# thousands of requests in one question.
@pytest.mark.parametrize("header, pause", [("7", 7.0), ("1.5", 1.5), ("1", 1.0), ("120", 30.0), ("inf", 30.0),
                                           ("0.5", 2.0), ("0.001", 2.0), ("0", 2.0), ("-3", 2.0), ("nan", 2.0),
                                           ("soon", 2.0), (None, 2.0), ("Wed, 21 Oct 2026 07:28:00 GMT", 2.0)])
def test_the_pause_is_the_retry_after_header_up_to_30_s_or_else_2_s(monkeypatch, header, pause):
    b, _, _ = backend(monkeypatch, [status(429, retry_after=header), message()])
    assert b.complete("s", "u") == "ok"
    assert b.busy == {A: (1000.0 + pause, 1)}


def test_without_a_retry_after_the_pause_doubles_to_15_s_and_a_reply_starts_it_again(monkeypatch):
    b, clock, sent = backend(monkeypatch, [status(529) for _ in range(5)] + [message("back"), status(529),
                                                                            message("again")], choices=[A])
    assert b.complete("s", "u") is None
    # 2, 4, 8 and 15 s; after the fifth busy reply a wait of 15 s would end too close to the budget.
    assert clock.slept == [2.0, 4.0, 8.0, 15.0] and len(sent) == 5 and clock.now - 1000.0 == 29.0
    assert b.complete("s", "u") == "back" and clock.slept[-1] == 15.0  # the next question waits for it
    assert b.complete("s", "u") == "again" and clock.slept[-1] == 2.0  # its reply reset the doubling


def test_a_model_busy_a_thousand_times_over_is_still_left_alone_for_15_s(monkeypatch):
    # 2.0 * 2 ** 1024 overflows a float: the doubling must stop well before that.
    b, _, sent = backend(monkeypatch, [status(529), message("from b")])
    b.busy[A] = (0.0, 1024)                                             # 1,024 busy replies, the last long ago
    assert b.complete("s", "u") == "from b" and models(sent) == [A, B]
    assert b.busy[A] == (1000.0 + 15.0, 1025)


def test_a_model_that_sends_nothing_in_all_its_time_is_left_alone_while_the_next_one_answers(monkeypatch, capsys):
    # A sends nothing until its 25 s for a reply are up: it is slow, which is not the network's fault.
    # B answers the same question at once, and the next ones for 60 s; then A is asked again, and
    # being slow again, it is left alone for 120 s.
    def script(n, fields):
        if fields["model"] == A:
            return Takes(fields["timeout"].read, timed_out())
        return message("from b")
    b, clock, sent = backend(monkeypatch, script)
    assert b.complete("s", "u") == "from b" and models(sent) == [A, B] and clock.slept == []
    assert clock.now - 1000.0 == 25.0 and b.busy[A] == (1000.0 + 25.0 + 60.0, 1)
    assert b.complete("s", "u") == "from b" and models(sent)[2:] == [B]
    clock.now = 1085.0
    assert b.complete("s", "u") == "from b" and models(sent)[3:] == [A, B]
    assert b.busy[A] == (1085.0 + 25.0 + 120.0, 2)
    said = lines(capsys)
    assert len(said) == 1 and said[0].startswith(f"[bankgpt] no reply from {A} in time")


def test_when_every_model_is_busy_the_question_waits_for_the_first_one_free(monkeypatch):
    b, clock, sent = backend(monkeypatch, [status(429, retry_after=3)] * 3 + [message("after the wait")])
    assert b.complete("s", "u") == "after the wait"
    assert models(sent) == [A, B, C, A] and clock.slept == [3.0]


def test_a_question_waits_for_a_busy_api_only_while_a_request_can_still_follow(monkeypatch, capsys):
    b, clock, sent = backend(monkeypatch, lambda n, fields: status(429, retry_after=3))
    assert b.complete("s", "u") is None
    # Every 3 s the three models are asked again, until a wait would end less than 2 s before the
    # 40 s are up: 13 rounds, the last one at 36 s.
    assert clock.now - 1000.0 == 36.0 and clock.slept == [3.0] * 12 and models(sent) == [A, B, C] * 13
    assert len(lines(capsys)) == 1                                      # one line for the first busy reply


NO_REPLY = {
    "connection closed": closed,
    "timed out at once (connecting)": timed_out,
    "an HTML page from a proxy": lambda: "<html><body>Sign in to use this network</body></html>",
    "a message without content": lambda: types.SimpleNamespace(stop_reason="end_turn", content=None),
    "a body cut off": lambda: ValueError("Unterminated string starting at: line 1 column 25"),
    "any other exception": lambda: RuntimeError("something else went wrong"),
    "a TypeError of another kind": lambda: TypeError("'NoneType' object is not iterable"),
}


@pytest.mark.parametrize("failure", list(NO_REPLY), ids=list(NO_REPLY))
def test_no_reply_is_asked_again_of_the_same_model_long_by_the_first_question_then_twice(monkeypatch, capsys,
                                                                                        failure):
    b, clock, sent = backend(monkeypatch, [NO_REPLY[failure]() for _ in range(10)] + [message("next")])
    # The first question keeps asking until a wait would end less than 2 s before its 40 s are up:
    # seven tries, the last at 33 s.
    assert b.complete("s", "u") is None
    assert models(sent) == [A] * 7 and clock.slept == [1.0, 3.0, 5.0, 8.0, 8.0, 8.0] and transport.USAGE.models == {}
    # After a question without a reply, the next one asks again after 1 s and 3 s, and no more.
    assert b.complete("s", "u") is None
    assert models(sent) == [A] * 10 and clock.slept[6:] == [1.0, 3.0]
    assert b.complete("s", "u") == "next" and len(lines(capsys)) == 1


def test_a_second_try_that_answers_ends_the_waiting(monkeypatch):
    b, clock, sent = backend(monkeypatch, [closed(), message("second try")])
    assert b.complete("s", "u") == "second try" and models(sent) == [A, A] and clock.slept == [1.0]


def test_a_short_outage_costs_the_first_question_time_but_not_its_answer(monkeypatch):
    b, clock, sent = backend(monkeypatch, [closed()] * 4 + [message("after the outage")])
    assert b.complete("s", "u") == "after the outage" and b.misses == 0
    assert models(sent) == [A] * 5 and clock.slept == [1.0, 3.0, 5.0, 8.0] and clock.now - 1000.0 == 17.0


def test_the_second_try_goes_to_the_same_model_even_when_an_earlier_one_is_free_again(monkeypatch):
    # A is busy for 1 s, so B is asked; B cannot be reached. After the 1 s wait A is free again, but
    # a network fault is not the model's: B is asked again.
    def script(n, fields):
        if fields["model"] == A:
            return status(429, retry_after=1) if n == 1 else message("from a")
        return closed() if n == 2 else message("from b")
    b, clock, sent = backend(monkeypatch, script, choices=[A, B])
    assert b.complete("s", "u") == "from b" and models(sent) == [A, B, B] and clock.slept == [1.0]


def test_each_request_is_given_no_more_time_than_the_question_has_left(monkeypatch):
    # The first request fails after 20 s (the connection closes); after a wait of 1 s the second has
    # the 19 s that are left and sends nothing in them, so the slow model is left alone too.
    b, clock, sent = backend(monkeypatch, [Takes(20.0, closed()), Takes(19.0, timed_out())], choices=[A])
    assert b.complete("s", "u") is None
    assert [r["timeout"] for r in sent] == [anthropic.Timeout(25.0, connect=5.0), anthropic.Timeout(19.0, connect=5.0)]
    assert clock.slept == [1.0] and clock.now - 1000.0 == 40.0 and b.busy[A] == (1040.0 + 60.0, 1)


def test_a_slow_model_hands_the_next_one_only_the_time_that_is_left(monkeypatch):
    # Every request sends nothing until its time is up: A has 25 s, B the 15 s left, and C none.
    b, clock, sent = backend(monkeypatch, lambda n, fields: Takes(fields["timeout"].read, timed_out()))
    assert b.complete("s", "u") is None and clock.slept == [] and clock.now - 1000.0 == 40.0
    assert [(r["model"], r["timeout"]) for r in sent] == [(A, anthropic.Timeout(25.0, connect=5.0)),
                                                         (B, anthropic.Timeout(15.0, connect=5.0))]


def test_no_request_is_started_with_less_than_2_s_left(monkeypatch):
    # A takes 38.5 s to reject the request (it has never answered, so it is dropped). B would answer, but only
    # 1.5 s are left: it is not asked.
    b, _, sent = backend(monkeypatch, [Takes(38.5, status(404)), message("too late")])
    assert b.complete("s", "u") is None and models(sent) == [A]


def test_the_watchdog_never_waits_past_the_question_s_time(monkeypatch):
    # Real time: a request that never comes back, a reply time (5 s) longer than the whole question
    # (1 s). The watchdog gives up when the question's time does.
    release = threading.Event()
    b, _, sent = backend(monkeypatch, lambda n, fields: release.wait(30) and message(), choices=[A],
                         ANSWER_BUDGET_S=1.0, MIN_LEFT_S=0.1, REPLY_S=5.0)
    b._now, b._sleep = time.monotonic, time.sleep
    start = time.monotonic()
    assert b.complete("s", "u") is None
    took = time.monotonic() - start
    release.set()
    assert 1.0 <= took < 1.5 and len(sent) == 1


def test_the_watchdog_waits_on_the_monotonic_clock_even_when_an_event_wait_ends_early(monkeypatch):
    # Python builds that time Event.wait on the wall clock end it early when that clock jumps. Here
    # every wait of the request's Event ends after 10 ms, as if it had: the request must still get
    # its 0.3 s and the watchdog's 0.2 s before it is left behind.
    class Jumpy(threading.Event):
        def wait(self, timeout=None):
            return super().wait(0.01 if timeout is None else min(timeout, 0.01))
    release = threading.Event()
    b, _, sent = backend(monkeypatch, lambda n, fields: release.wait(30) and message(), choices=[A],
                         REPLY_S=0.3, WATCHDOG_S=0.2, FIRST_WAITS=(), RETRY_WAITS=())
    b._now, b._sleep = time.monotonic, time.sleep
    monkeypatch.setattr(transport, "threading", types.SimpleNamespace(Event=Jumpy, Thread=threading.Thread))
    start = time.monotonic()
    assert b.complete("s", "u") is None
    took = time.monotonic() - start
    release.set()
    assert 0.5 <= took < 1.5 and len(sent) == 1


@pytest.mark.parametrize("fault", [KeyError("a bug"), RuntimeError("another bug"), ZeroDivisionError("and one more")])
def test_a_fault_in_the_backend_itself_never_raises(monkeypatch, capsys, fault):
    b, _, sent = backend(monkeypatch, [message()])

    def broken(*args):
        raise fault
    b._pick = broken
    assert b.complete("s", "u") is None and b.complete("s", "u") is None and sent == []
    assert len(lines(capsys)) == 1 and transport.USAGE.unanswered == 2


# --------------------------------------------------------------------------- misses, rests and probes


def asked(b, sent, questions):
    """How many requests each of this many questions sent."""
    counts = []
    for _ in range(questions):
        before = len(sent)
        b.complete("s", "u")
        counts.append(len(sent) - before)
    return counts


def test_three_misses_in_a_row_rest_the_reader_and_each_failed_probe_doubles_the_rest(monkeypatch, capsys):
    b, clock, sent = backend(monkeypatch, lambda n, fields: closed())
    rests = [5, 10, 20, 40, 80, 80]
    got = asked(b, sent, 3 + sum(rests) + len(rests))
    # Three questions of seven, three and three tries, then every rest, each followed by a probe that
    # asks each model once, without waiting.
    want = [7, 3, 3]
    for rest in rests:
        want += [0] * rest + [3]
    assert got == want
    assert clock.slept == [1.0, 3.0, 5.0, 8.0, 8.0, 8.0] + [1.0, 3.0] * 2      # the probes never wait
    assert [r["model"] for r in sent[13:16]] == [A, B, C]
    said = lines(capsys)
    assert len(said) == 2 and "no reply from the API" in said[0] and "questions in a row" in said[1]
    assert transport.USAGE.unanswered == len(want)


def test_a_reply_to_the_probe_puts_everything_back_to_normal(monkeypatch, capsys):
    down = [True]
    b, _, sent = backend(monkeypatch, lambda n, fields: closed() if down[0] else message("back"))
    assert asked(b, sent, 3 + 5 + 1 + 10) == [7, 3, 3] + [0] * 5 + [3] + [0] * 10
    down[0] = False
    assert b.complete("s", "u") == "back" and b.complete("s", "u") == "back"
    down[0] = True
    assert asked(b, sent, 3 + 5 + 1) == [7, 3, 3] + [0] * 5 + [3]      # the rest is 5 again, not 20
    said = lines(capsys)
    assert len(said) == 3 and "back to normal" in said[2]


def test_a_probe_asks_a_busy_model_too_and_never_waits(monkeypatch):
    # A says it is busy for 30 s, and B cannot be reached: three misses (with the first question's
    # waits made short, so that they all fall within those 30 s). After the rest A has not waited
    # its 30 s, but the probe asks it anyway, and it answers.
    def script(n, fields):
        if fields["model"] == A:
            return status(429, retry_after=30) if n == 1 else message("from a")
        return closed()
    b, clock, sent = backend(monkeypatch, script, choices=[A, B], FIRST_WAITS=(1.0, 3.0))
    assert asked(b, sent, 3 + 5) == [4, 3, 3] + [0] * 5
    assert clock.now - 1000.0 < 30.0 and b.busy[A][0] > clock.now
    before, waits = len(sent), len(clock.slept)
    assert b.complete("s", "u") == "from a" and models(sent[before:]) == [A] and len(clock.slept) == waits
    assert b.busy == {}


@pytest.mark.parametrize("rejection", [status(403), status(422)], ids=["403", "422"])
def test_a_probe_asks_the_next_model_when_one_that_has_answered_rejects(monkeypatch, rejection):
    # A answers once, then rejects every request: each question is a miss, the reader rests, and the
    # probe goes on from A's rejection to B, which answers.
    def script(n, fields):
        if fields["model"] == A:
            return message("from a") if n == 1 else rejection
        return message("from b")
    b, _, sent = backend(monkeypatch, script)
    assert asked(b, sent, 1 + 3 + 5) == [1, 1, 1, 1] + [0] * 5
    assert b.complete("s", "u") == "from b" and models(sent)[4:] == [A, B]
    assert (b.misses, b.resting, b.probing, b.dropped) == (0, 0, False, set())


def test_a_miss_between_replies_does_not_start_a_rest(monkeypatch):
    # After a reply the next question without one is the first again, with the long waits.
    b, _, sent = backend(monkeypatch, [message("a")] + [closed()] * 7 + [message("b")] + [closed()] * 10
                         + [message("c")])
    assert [b.complete("s", "u") for _ in range(5)] == ["a", None, "b", None, None]
    assert [len(sent), b.misses, b.resting] == [19, 2, 0]
    assert b.complete("s", "u") == "c" and b.misses == 0 and b.resting == 0


def test_a_400_from_every_model_before_any_answered_is_taken_as_the_request_s_fault(monkeypatch, capsys):
    b, clock, sent = backend(monkeypatch, [status(400)] * 3 + [message("the next question")])
    assert b.complete("s", "u") is None and models(sent) == [A, B, C] and clock.slept == []
    assert b.dropped == set() and b.misses == 1
    assert b.complete("s", "u") == "the next question" and models(sent)[3:] == [A]
    said = lines(capsys)
    assert len(said) == 4 and "the request is taken to be at fault" in said[3]


@pytest.mark.parametrize("rejection", [status(403), status(404), TypeError("create() got an unexpected keyword argument 'x'")],
                         ids=["403", "404", "unsendable"])
def test_other_rejections_from_every_model_drop_them_all(monkeypatch, rejection):
    b, _, sent = backend(monkeypatch, [rejection] * 3)
    assert b.complete("s", "u") is None and b.complete("s", "u") is None
    assert len(sent) == 3 and b.dropped == {A, B, C}


def test_after_a_400_from_every_model_only_the_ones_it_dropped_come_back(monkeypatch):
    b, _, sent = backend(monkeypatch, [status(404), status(400), status(400), message("from b")])
    assert b.complete("s", "u") is None and b.dropped == {A}
    assert b.complete("s", "u") == "from b" and models(sent) == [A, B, C, B]


# --------------------------------------------------------------------------- the watchdog and threads


def test_a_request_that_does_not_come_back_is_left_behind_and_its_late_reply_counted_once(monkeypatch):
    release = threading.Event()

    def script(n, fields):
        if n == 1:
            release.wait(10)                  # a hang the request's own timeout does not end
            return message("late")
        return message("on time")
    b, clock, sent = backend(monkeypatch, script, REPLY_S=0.2, WATCHDOG_S=0.1)
    start = time.monotonic()
    assert b.complete("s", "u") == "on time"
    assert time.monotonic() - start < 2 and clock.slept == [1.0]
    assert sent[0]["timeout"] == anthropic.Timeout(0.2, connect=5.0) and transport.USAGE.models[A]["calls"] == 1
    release.set()
    end = time.monotonic() + 5
    while transport.USAGE.models[A]["calls"] < 2 and time.monotonic() < end:
        time.sleep(0.01)
    time.sleep(0.05)
    assert transport.USAGE.models[A]["calls"] == 2                            # paid for: counted, and only once


def test_a_hanging_request_in_one_thread_never_delays_another(monkeypatch):
    release = threading.Event()

    def script(n, fields):
        if fields["messages"][0]["content"] == "stuck":
            release.wait(10)
            return message("late")
        return message("quick")
    b, _, sent = backend(monkeypatch, script)
    stuck = threading.Thread(target=b.complete, args=("s", "stuck"), daemon=True)
    stuck.start()
    end = time.monotonic() + 5
    while not sent and time.monotonic() < end:
        time.sleep(0.01)
    start = time.monotonic()
    assert b.complete("s", "quick") == "quick" and time.monotonic() - start < 1
    release.set()
    stuck.join(5)
    assert not stuck.is_alive()


def test_threads_that_see_the_same_rejection_or_busy_reply_say_so_once(monkeypatch, capsys):
    threads = 4
    together = threading.Barrier(threads, timeout=10)

    def script(n, fields):
        if fields["model"] == A:
            together.wait()                   # every thread has sent its request before any is answered
            return status(404) if fields["messages"][0]["content"] == "rejected" else status(429)
        return message("from b")
    for question in ("busy", "rejected"):
        b, _, _ = backend(monkeypatch, script)
        results = []
        workers = [threading.Thread(target=lambda: results.append(b.complete("s", question))) for _ in range(threads)]
        for w in workers:
            w.start()
        for w in workers:
            w.join(10)
        assert results == ["from b"] * threads
        assert len(lines(capsys)) == 1


# --------------------------------------------------------------------------- what is said on stderr


def test_at_exit_one_line_says_how_many_questions_got_no_reply(monkeypatch, capsys):
    b, _, _ = backend(monkeypatch, [message("a")] + [closed()] * 7 + [message("b")])
    assert [b.complete("s", "u") for _ in range(3)] == ["a", None, "b"]
    capsys.readouterr()
    transport._at_exit()
    assert lines(capsys) == [
        "[bankgpt] model usage: 2 calls to claude-made-up-a, 20 input + 10 output tokens, cost unknown: no price for"
        " this model in kb/transport.py; calls took a median of 0.0 s, 95% under 0.0 s, longest 0.0 s",
        "[bankgpt] 1 question got no reply from the model and was answered from the rules"]


# --------------------------------------------------------------------------- the installed SDK against the fake API


def real(monkeypatch, fake, choices=(A, B, C), **limits):
    """An AnthropicBackend whose own SDK client talks to the fake API, with limits shrunk."""
    for name, value in dict(fake.env(), NO_PROXY="127.0.0.1", no_proxy="127.0.0.1").items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(transport.AnthropicBackend, "CHOICES", list(choices))
    for name, value in limits.items():
        monkeypatch.setattr(transport.AnthropicBackend, name, value)
    return transport.AnthropicBackend()


def test_sdk_one_529_is_answered_by_the_next_model_at_once(monkeypatch):
    with FakeAnthropic([{"calls": 1, "do": "status", "status": 529}], text="from the next model") as fake:
        b = real(monkeypatch, fake)
        start = time.monotonic()
        assert b.complete("s", "u") == "from the next model"
        assert time.monotonic() - start < 1
    assert [(r["model"], r["status"]) for r in fake.calls] == [(A, 529), (B, 200)]


def test_sdk_a_429_with_retry_after_from_every_model_then_a_reply(monkeypatch):
    with FakeAnthropic([{"calls": [1, 3], "do": "status", "status": 429, "retry_after": 1}]) as fake:
        b = real(monkeypatch, fake)
        start = time.monotonic()
        assert b.complete("s", "u") == fake.text
        assert 0.9 < time.monotonic() - start < 2
    assert [(r["model"], r["status"]) for r in fake.calls] == [(A, 429), (B, 429), (C, 429), (A, 200)]


def test_sdk_two_closed_connections_then_a_reply(monkeypatch):
    with FakeAnthropic([{"calls": [1, 2], "do": "close"}]) as fake:
        b = real(monkeypatch, fake, FIRST_WAITS=(0.1, 0.3))
        start = time.monotonic()
        assert b.complete("s", "u") == fake.text
        assert 0.4 <= time.monotonic() - start < 2
    assert [(r["model"], r["action"]) for r in fake.calls] == [(A, "close"), (A, "close"), (A, "reply")]


def test_sdk_a_hang_is_ended_by_the_request_timeout_and_a_trickle_by_the_watchdog(monkeypatch):
    for action in ("hang", "trickle"):
        with FakeAnthropic([{"calls": 1, "do": action, "seconds": 30}]) as fake:
            b = real(monkeypatch, fake, REPLY_S=0.5, WATCHDOG_S=0.3)
            start = time.monotonic()
            assert b.complete("s", "u") == fake.text
            took = time.monotonic() - start
        # The hang ends at 0.5 s, the SDK's timeout. The trickle sends a byte about every 0.12 s, so
        # the SDK would wait the whole 30 s: the watchdog ends it at 0.5 + 0.3 s. Either way the
        # model sent nothing in all its time: the next model is asked at once, and A is left alone.
        assert (0.5 < took < 1.5) if action == "hang" else (0.8 < took < 1.8), (action, took)
        assert [(r["model"], r["action"]) for r in fake.calls] == [(A, action), (B, "reply")]
        assert list(b.busy) == [A]


def test_sdk_a_200_html_page_is_no_reply_and_raises_nothing(monkeypatch):
    portal = {"do": "reply", "body": "<html><body>Sign in to use this network</body></html>",
              "headers": {"Content-Type": "text/html"}}
    with FakeAnthropic([portal]) as fake:
        b = real(monkeypatch, fake, FIRST_WAITS=(0.05, 0.05))
        assert b.complete("s", "u") is None
    assert [r["model"] for r in fake.calls] == [A, A, A] and transport.USAGE.models == {}


def test_sdk_a_401_switches_the_reader_off_after_one_call(monkeypatch):
    with FakeAnthropic([{"do": "status", "status": 401}]) as fake:
        b = real(monkeypatch, fake)
        assert [b.complete("s", "u") for _ in range(3)] == [None] * 3
    assert len(fake.calls) == 1


def test_sdk_a_400_after_the_model_has_answered_keeps_the_model(monkeypatch):
    with FakeAnthropic([{"calls": 2, "do": "status", "status": 400}]) as fake:
        b = real(monkeypatch, fake)
        assert [b.complete("s", "u") for _ in range(3)] == [fake.text, None, fake.text]
    assert [(r["model"], r["status"]) for r in fake.calls] == [(A, 200), (A, 400), (A, 200)]


# A separate process whose one request trickles in for a minute: the watchdog gives up on it, and the
# request's thread, still reading, must not keep the process from exiting.
EXIT_CHILD = """
import sys
sys.path.insert(0, sys.argv[1])
from kb import transport
B = transport.AnthropicBackend
B.CHOICES, B.REPLY_S, B.WATCHDOG_S, B.FIRST_WAITS, B.RETRY_WAITS = ["claude-made-up-a"], 0.3, 0.2, (), ()
print(B().complete("s", "u"))
"""


def test_sdk_a_request_left_behind_does_not_delay_the_exit_of_the_process():
    with FakeAnthropic([{"do": "trickle", "seconds": 60}]) as fake:
        env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "BANKGPT_"))}
        env.update(fake.env(), NO_PROXY="127.0.0.1", no_proxy="127.0.0.1")
        start = time.monotonic()
        out = subprocess.run([sys.executable, "-c", EXIT_CHILD, str(ROOT / "solution")], env=env,
                             capture_output=True, text=True, timeout=60)
        took = time.monotonic() - start
    assert out.returncode == 0 and out.stdout.strip() == "None", out.stderr
    assert took < 20 and [r["action"] for r in fake.calls] == ["trickle"]


# --------------------------------------------------------------------------- the request each model is sent

# The fields each model must be sent beyond the plain request, as checked with live calls on 2026-11-09.
EXPECTED = {
    "claude-haiku-4-5": {},
    "claude-haiku-5-5": {"thinking": {"type": "disabled"}, "output_config": {"effort": "low"}},
    "claude-sonnet-5-5": {"thinking": {"type": "between_tools"}, "output_config": {"effort": "low"}},
}


def extra(request):
    """The fields of a request beyond the plain ones."""
    return {k: v for k, v in request.items() if k not in PLAIN}


@pytest.mark.parametrize("order", list(itertools.permutations(EXPECTED)))
def test_each_model_is_sent_its_own_settings_wherever_it_stands_in_the_order(monkeypatch, order):
    b, _, calls = backend(monkeypatch, [not_found(), not_found(), message("ok"), message("again")], order)
    assert b.complete("s", "u") == "ok"                          # the first two are rejected, the third answers
    assert b.complete("s", "u") == "again"                       # and is kept for the next question
    assert [c["model"] for c in calls] == list(order) + [order[-1]]
    assert all(PLAIN <= set(c) for c in calls)
    assert [extra(c) for c in calls] == [EXPECTED[c["model"]] for c in calls]


def test_a_model_the_table_does_not_list_is_sent_the_plain_request(monkeypatch):
    b, _, calls = backend(monkeypatch, [not_found(), message("ok")], ["claude-made-up-test", "claude-haiku-5-5"])
    assert b.complete("s", "u") == "ok"
    assert calls[0]["model"] == "claude-made-up-test" and set(calls[0]) == PLAIN
    assert extra(calls[1]) == EXPECTED["claude-haiku-5-5"]

    b, _, calls = backend(monkeypatch, [not_found(), message("ok")], ["claude-sonnet-5-5", "claude-made-up-test"])
    assert b.complete("s", "u") == "ok"                          # the same as a fallback
    assert calls[1]["model"] == "claude-made-up-test" and set(calls[1]) == PLAIN


def old_sdk(monkeypatch, order):
    """An AnthropicBackend that tries the models in order, with a client that knows only the plain
    fields and the timeout every SDK takes, like an SDK from before extended thinking: any other
    field is a TypeError, raised before a request is sent. Returns the backend and the requests it
    tried to send."""
    def plain(*, model, max_tokens, system, messages, timeout):
        return message()
    b, _, sent = backend(monkeypatch, lambda n, fields: plain(**fields), order)
    return b, sent


def test_a_model_whose_settings_the_sdk_does_not_know_is_skipped_like_one_the_key_may_not_use(monkeypatch, capsys):
    b, sent = old_sdk(monkeypatch, ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-haiku-4-5"])
    assert b.complete("s", "u") == "ok" and b.complete("s", "u") == "ok"
    assert models(sent) == ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-haiku-4-5", "claude-haiku-4-5"]   # then kept
    said = capsys.readouterr().err
    assert all(f"cannot send {m}'s request" in said for m in ("claude-haiku-5-5", "claude-sonnet-5-5"))


def test_an_sdk_that_can_send_no_model_leaves_the_answer_to_the_rules(monkeypatch):
    b, sent = old_sdk(monkeypatch, ["claude-haiku-5-5", "claude-sonnet-5-5"])
    ks = build(SPRING, NOW)
    rules = ks.answer(GONE)                                      # what the rules alone say
    ks.engine.reader = reader.Reader(b)
    ks.engine._answers.clear()
    answer = ks.answer(GONE)
    assert models(sent) == ["claude-haiku-5-5", "claude-sonnet-5-5"]
    assert answer.confident and answer.text == rules.text != REFUSAL


@pytest.mark.parametrize("rejected", ["by the API", "by the SDK"])
def test_threads_that_see_the_same_rejection_move_on_by_one_model(monkeypatch, rejected):
    """dev/run.py --jobs answers from several threads. When all of them see the first model
    reject the request, the reader moves on to the second one once, not once per thread past the end of the list."""
    threads = 4
    order = ["claude-made-up-test", "claude-haiku-4-5", "claude-haiku-5-5"]
    b, _, _ = backend(monkeypatch, [], order)
    together = threading.Barrier(threads, timeout=30)
    sent = []

    def create(**fields):
        sent.append(fields["model"])
        if fields["model"] != order[0]:
            return message(fields["model"])
        together.wait()                       # every thread has sent its request before any is rejected
        raise not_found() if rejected == "by the API" else TypeError("unexpected keyword argument 'thinking'")
    b.client = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))

    with ThreadPoolExecutor(threads) as pool:
        answers = list(pool.map(lambda _: b.complete("s", "u"), range(threads)))
    assert answers == [order[1]] * threads and b.dropped == {order[0]}
    assert sent.count(order[0]) == threads and order[2] not in sent
    assert b.complete("s", "u") == order[1]                      # and the reader is still on for the next question


# Run in a separate process, because the order is fixed when kb.transport is first imported: every model
# rejects the request, so the backend sends one request to each in turn. Prints the requests' fields.
CHILD = """
import json, sys, types
sys.path[:0] = sys.argv[1:]
from kb import transport
from sdk import not_found

sent = []

def create(**fields):
    sent.append({k: v for k, v in fields.items() if k not in ("max_tokens", "system", "messages", "timeout")})
    raise not_found()

b = transport.AnthropicBackend()
b.client = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
assert b.complete("s", "u") is None
print(json.dumps(sent))
"""


@pytest.mark.parametrize("first", [None, "", "claude-haiku-5-5", "claude-sonnet-5-5", "claude-made-up-test"])
def test_bankgpt_model_is_tried_first_and_the_others_follow_with_their_own_settings(first):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "BANKGPT_"))}
    env["ANTHROPIC_API_KEY"] = "not-a-real-key"
    if first is not None:
        env["BANKGPT_MODEL"] = first
    out = subprocess.run([sys.executable, "-c", CHILD, str(ROOT / "solution"), TESTS], env=env, capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    if not first:                                                # unset or empty: the default, then its two fallbacks
        order = ["claude-sonnet-5-5", "claude-haiku-4-5", "claude-haiku-5-5"]
    else:
        order = [first] + [m for m in EXPECTED if m != first]
    assert json.loads(out.stdout) == [dict(EXPECTED.get(m, {}), model=m) for m in order]


# The installed SDK itself, in a separate process with the default order, building the backend's own
# client: only its HTTP transport is replaced, by one that answers every request on this machine.
SDK_CHILD = """
import json, sys
sys.path[:0] = sys.argv[1:]
import anthropic
from kb import transport
from sdk import http

sent = []

def answer(request):
    body = json.loads(request.content)
    sent.append({k: v for k, v in body.items() if k not in ("max_tokens", "system", "messages")})
    return http.Response(200, json={"id": "msg_stub", "type": "message", "role": "assistant", "model": body["model"],
                                    "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                                    "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 2}})

sdk_client = anthropic.Anthropic
anthropic.Anthropic = lambda **kw: sdk_client(http_client=http.Client(transport=http.MockTransport(answer)), **kw)
b = transport.AnthropicBackend()
print(json.dumps({"reply": b.complete("s", "u"), "sent": sent}))
"""


def test_the_default_model_is_sent_by_an_sdk_that_knows_its_fields_and_skipped_by_an_older_one():
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "BANKGPT_"))}
    env["ANTHROPIC_API_KEY"] = "not-a-real-key"
    out = subprocess.run([sys.executable, "-c", SDK_CHILD, str(ROOT / "solution"), TESTS], env=env, capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)
    assert got["reply"] == "ok"
    if tuple(int(x) for x in anthropic.__version__.split(".")[:2]) >= (0, 77):      # what requirements.txt asks for
        assert got["sent"] == [dict(EXPECTED["claude-sonnet-5-5"], model="claude-sonnet-5-5")]
        assert "cannot send" not in out.stderr
    else:                       # it does not know output_config: the call is rejected before it is sent, and said so
        assert got["sent"] == [{"model": "claude-haiku-4-5"}]
        assert "this anthropic SDK cannot send claude-sonnet-5-5's request" in out.stderr
    usage = [x for x in out.stderr.splitlines() if x.startswith("[bankgpt] model usage")]
    assert len(usage) == 1 and "10 input + 2 output tokens" in usage[0] and "; the call took " in usage[0]


# --------------------------------------------------------------------------- what the API costs


def test_usage_is_counted_per_model_across_calls_and_priced(monkeypatch):
    usage = anthropic.types.Usage
    # Each call takes this long by the test clock; the rejected call's 9 s is no call time.
    b, _, calls = backend(monkeypatch, [
        Takes(0.5, message("a", usage=usage(input_tokens=1000, output_tokens=100))),
        Takes(1.5, message("b", usage=usage(input_tokens=500, output_tokens=50,
                                            cache_read_input_tokens=2000, cache_creation_input_tokens=400))),
        Takes(9.0, not_found()),                                       # rejected: tried, not counted
        Takes(2.3, message("c", usage=usage(input_tokens=10_000, output_tokens=1_000))),
        Takes(0.04, message("d", usage=usage(input_tokens=7, output_tokens=3))),
    ], ["claude-haiku-4-5", "claude-sonnet-5-5", "claude-unpriced-test"])
    assert b.budget is None                                            # no cap unless one is asked for
    # claude-haiku-4-5 has answered, so its rejection costs that one question: the rules answer it.
    assert [b.complete("s", "u") for _ in range(3)] == ["a", "b", None]
    b.dropped.add("claude-haiku-4-5")                                  # as if it had rejected one before answering
    assert b.complete("s", "u") == "c"
    b.dropped.add("claude-sonnet-5-5")
    assert b.complete("s", "u") == "d" and len(calls) == 5

    zero = {"cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    assert transport.USAGE.models == {
        "claude-haiku-4-5": {"calls": 2, "input_tokens": 1500, "output_tokens": 150,
                             "cache_read_input_tokens": 2000, "cache_creation_input_tokens": 400},
        "claude-sonnet-5-5": dict(zero, calls=1, input_tokens=10_000, output_tokens=1_000),
        "claude-unpriced-test": dict(zero, calls=1, input_tokens=7, output_tokens=3)}
    assert transport.USAGE.seconds == {"claude-haiku-4-5": [0.5, 1.5], "claude-sonnet-5-5": [pytest.approx(2.3)],
                                 "claude-unpriced-test": [pytest.approx(0.04)]}
    # haiku-4-5: (1,500 + 0.1 x 2,000 cache read + 1.25 x 400 cache write) x $1 + 150 x $5 per million = $0.00295
    # sonnet-5-5: 10,000 x $2 + 1,000 x $10 per million = $0.03; the unpriced model adds nothing it can know
    assert transport.USAGE.spent() == pytest.approx(0.00295 + 0.03)
    assert transport.USAGE.summary() == [
        "[bankgpt] model usage: 2 calls to claude-haiku-4-5, 1,500 input + 150 output + 2,000 cache-read"
        " + 400 cache-write tokens, under $0.01; calls took a median of 1.0 s, 95% under 1.5 s, longest 1.5 s",
        "[bankgpt] model usage: 1 call to claude-sonnet-5-5, 10,000 input + 1,000 output tokens, about $0.03;"
        " the call took 2.3 s",
        "[bankgpt] model usage: 1 call to claude-unpriced-test, 7 input + 3 output tokens,"
        " cost unknown: no price for this model in kb/transport.py; the call took 0.0 s"]
    assert transport.price("claude-haiku-4-5-20251001") == transport.PRICES["claude-haiku-4-5"]


def test_the_95th_percentile_of_call_times_is_taken_as_the_scorer_takes_it_of_answer_times():
    times = list(range(40, 0, -1))                                     # 40 calls: 40 s down to 1 s
    for t in times:
        transport.USAGE.add("claude-haiku-4-5", message("x", usage=None), t)
    assert transport._spread(times) == {"median": 20.5, "p95": 39, "max": 40}
    assert score_speed(times)["p95_s"] == 39                          # the organisers' p95 of the same times
    assert transport.USAGE.summary() == ["[bankgpt] model usage: 40 calls to claude-haiku-4-5, 0 input + 0 output tokens, "
                                   "under $0.01; calls took a median of 20.5 s, 95% under 39.0 s, longest 40.0 s"]
    assert transport._spread([]) is None


def test_a_spent_budget_stops_the_model_and_the_rules_answer(monkeypatch, capsys):
    monkeypatch.setenv("BANKGPT_BUDGET_USD", "0.01")
    said = "QUOTES: none\nVERDICT: answer\nSOURCES: 1\nANSWER: It was founded in 1957 by twelve teachers."
    costly = anthropic.types.Usage(input_tokens=8_000, output_tokens=500)       # $0.0105 on claude-haiku-4-5
    b, _, calls = backend(monkeypatch, [message(said, usage=costly) for _ in range(3)], ["claude-haiku-4-5"])
    ks = build(SMALL_SITE, reader=reader.Reader(b))

    question = "When was the credit union founded?"
    first = ks.engine.answer(question)
    assert first.path == "reader" and "1957" in first.text and len(calls) == 1   # the call that crosses it answers
    for _ in range(2):                                   # asked again: the reader is consulted, the model is not
        ks.engine._answers.clear()
        again = ks.engine.answer(question)
        assert len(calls) == 1 and again.path != "reader"
        assert again.text == build(SMALL_SITE).engine.answer(question).text          # exactly what the rules alone say
    assert capsys.readouterr().err.count("BANKGPT_BUDGET_USD=0.01") == 1           # said once

    for cap in ("0", "fifty cents"):                     # a cap of nothing, or one that cannot be read: no call
        monkeypatch.setenv("BANKGPT_BUDGET_USD", cap)
        transport.USAGE.reset()
        b, _, calls = backend(monkeypatch, [message(said, usage=costly)], ["claude-haiku-4-5"])
        assert b.complete("s", "u") is None and calls == []


def test_a_response_without_usage_never_breaks_an_answer(monkeypatch):
    class Unreadable:
        stop_reason = "end_turn"
        content = [types.SimpleNamespace(type="text", text="unreadable usage")]

        @property
        def usage(self):
            raise RuntimeError("no usage here")

    b, _, calls = backend(monkeypatch, [
        types.SimpleNamespace(stop_reason="end_turn", content=[types.SimpleNamespace(type="text", text="no usage")]),
        message("usage is None", usage=None),
        message("fields are None",
                usage=types.SimpleNamespace(input_tokens=None, output_tokens=None, cache_read_input_tokens=None)),
        message("odd values", usage=types.SimpleNamespace(input_tokens="12", output_tokens=-3)),
        Unreadable(),
        message("normal", usage=anthropic.types.Usage(input_tokens=9, output_tokens=2)),
    ], ["claude-haiku-4-5"])
    assert [b.complete("s", "u") for _ in range(6)] == [
        "no usage", "usage is None", "fields are None", "odd values", "unreadable usage", "normal"]
    assert transport.USAGE.models == {"claude-haiku-4-5": {"calls": 6, "input_tokens": 9, "output_tokens": 2,
                                                     "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}}


def test_at_exit_one_summary_line_and_one_ledger_line(monkeypatch, tmp_path, capsys):
    ledger = tmp_path / "not" / "there" / "spend.jsonl"                     # the folders are created
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "BANKGPT_"))}
    env.update(ANTHROPIC_API_KEY="not-a-real-key", BANKGPT_MODEL="claude-haiku-4-5",
               BANKGPT_SPEND_LOG=str(ledger), BANKGPT_SPEND_LABEL="unit test")
    # A separate process, so the summary and the ledger line come from the real exit hook. Its clock
    # moves only during the calls: the first takes 1 s by it, the second 3 s.
    child = (f"import sys, time, types; sys.path.insert(0, {str(ROOT / 'solution')!r})\n"
             "from kb import transport\n"
             "now, steps = [100.0], iter([1.0, 3.0])\n"
             "b = transport.AnthropicBackend()\n"
             "b._now = lambda: now[0]\n"
             "usage = types.SimpleNamespace(input_tokens=12000, output_tokens=800)\n"
             "done = types.SimpleNamespace(stop_reason='end_turn', usage=usage,\n"
             "                             content=[types.SimpleNamespace(type='text', text='ok')])\n"
             "def create(**fields):\n"
             "    now[0] += next(steps)\n"
             "    return done\n"
             "b.client = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))\n"
             "print([b.complete('s', 'u') for _ in range(int(sys.argv[1]))])\n")

    start = int(time.time())
    out = subprocess.run([sys.executable, "-c", child, "2"], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == "['ok', 'ok']"
    assert [x for x in out.stderr.splitlines() if x.startswith("[bankgpt]")] == [
        "[bankgpt] model usage: 2 calls to claude-haiku-4-5, 24,000 input + 1,600 output tokens, about $0.03;"
        " calls took a median of 2.0 s, 95% under 3.0 s, longest 3.0 s"]
    entry = json.loads(ledger.read_text())                                  # one line, and only one
    assert start <= entry["time"] <= time.time() and entry["label"] == "unit test"
    assert entry["models"] == {"claude-haiku-4-5": {"calls": 2, "input_tokens": 24_000, "output_tokens": 1_600,
                                                    "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
                                                    "usd": 0.032, "seconds": {"median": 2.0, "p95": 3.0, "max": 3.0}}}

    monkeypatch.setenv("BANKGPT_SPEND_LOG", str(ledger))
    transport._at_exit()                                         # this process has made no call: nothing printed or logged
    assert capsys.readouterr().err == "" and len(ledger.read_text().splitlines()) == 1

    transport.USAGE.add("claude-haiku-5-5", message("x", usage=None))
    transport.USAGE.log(str(ledger), "")                                          # appended, never overwritten
    assert [json.loads(x)["label"] for x in ledger.read_text().splitlines()] == ["unit test", ""]
    (tmp_path / "a-file").write_text("")
    transport.USAGE.log(str(tmp_path / "a-file" / "spend.jsonl"), "")               # cannot be written: said, not raised
    assert "could not add this run to the spend log" in capsys.readouterr().err
