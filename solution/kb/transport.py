"""The Anthropic API: the transport the reader's messages go through, and what they cost.

The `anthropic` package reads the key from ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN); whether
there is a reader at all is reader.make_reader's choice. The model is claude-sonnet-5-5 with thinking
off (anthropic 0.77 or newer); claude-haiku-4-5 and then claude-haiku-5-5 stand in when the key or
the SDK cannot use it. BANKGPT_MODEL changes the model; SETTINGS holds each model's request fields.

Every call is counted, priced (PRICES) and timed; at exit one stderr line per model says what it
cost and how long its calls took. For development, off unless set: BANKGPT_BUDGET_USD=0.50 answers
from rules once the process has spent that much; BANKGPT_SPEND_LOG=path appends the run's spend to
a JSON-lines ledger, labelled BANKGPT_SPEND_LABEL=name (dev/spend.py adds the ledger up).
"""
from __future__ import annotations

import atexit
import json
import os
import re
import statistics
import sys
import threading
import time
from typing import Optional

_MODEL = os.environ.get("BANKGPT_MODEL", "").strip() or "claude-sonnet-5-5"     # set to nothing is not set

# List prices in USD per million tokens (input, output), checked on 2026-11-09. Every cost this
# module reports comes from here; a model missing from the table is counted in tokens only.
# claude-haiku-5-5 costs this for prompts of up to 100K tokens (0.50 / 2.50 above that); the
# reader's prompts are a few thousand tokens.
PRICES = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
}
# Prompt-cache tokens are billed against the input price: a read at 0.1 times it (the Haiku rate;
# claude-sonnet-5-5's is half that), a (five-minute) write at 1.25 times it. One rate serves every
# model because the reader asks for no caching, so both normally stay at zero.
CACHE_READ, CACHE_WRITE = 0.10, 1.25

# Extra request fields per model, sent with that model wherever it stands in the order of models
# tried. Each keeps extended thinking off: with it on, the reader's answers are slower and less
# accurate. Checked with live calls on 2026-11-09 (SDK 2.7.0): claude-haiku-4-5 does not think
# unless asked; claude-sonnet-5-5 rejects {"type": "disabled"} and turns thinking off with
# "between_tools". A model the table lists neither by its name nor by its name without a date
# (see settings()) is sent the plain request. An SDK older than 0.77 does not know output_config
# and rejects those requests before sending them; the backend then moves on to the next model
# (claude-haiku-4-5 asks for no extra fields).
SETTINGS = {
    "claude-haiku-4-5": {},
    "claude-haiku-5-5": {"thinking": {"type": "disabled"}, "output_config": {"effort": "low"}},
    "claude-sonnet-5-5": {"thinking": {"type": "between_tools"}, "output_config": {"effort": "low"}},
}


# --------------------------------------------------------------------------- what the API costs

_TOKENS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


def _alias(model: str) -> str:
    """A dated model name without its date: claude-haiku-4-5-20251001 -> claude-haiku-4-5."""
    return re.sub(r"-\d{8}$", "", model)


def price(model: str) -> Optional[tuple]:
    """(input, output) USD per million tokens, or None for a model PRICES does not list. A dated
    model name (claude-haiku-4-5-20251001) costs what its alias costs."""
    return PRICES.get(model) or PRICES.get(_alias(model))


def _count(usage, field: str) -> int:
    value = getattr(usage, field, None)
    return value if isinstance(value, int) and value > 0 else 0      # missing or None counts as 0


def _spread(times: list) -> Optional[dict]:
    """The median, 95th percentile and longest of call times in seconds, or None when there are
    none. The 95th percentile is taken the way the organisers' scorer takes the p95 of its answer
    times (the value at place int(0.95 n) in sorted order), so at least 95% of the calls took no
    longer and the two figures can be compared."""
    if not times:
        return None
    ordered = sorted(times)
    return {"median": statistics.median(ordered), "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
            "max": ordered[-1]}


class Usage:
    """Calls, tokens and call times per model since the process started. One for the whole
    process, because the budget and the exit summary are about what this run spent."""

    def __init__(self) -> None:
        self._lock = threading.Lock()          # dev/run.py --jobs answers from several threads
        self.reset()

    def reset(self) -> None:
        """Forget every count: back to the state of a new process."""
        with self._lock:
            self.models: dict = {}             # model -> {"calls": n, "input_tokens": n, ...}
            self.seconds: dict = {}            # model -> [how long each timed call took, in seconds]
            self.budget_said = False
            self.unanswered = 0                # questions the model gave no reply to: the rules answered them

    def no_reply(self) -> None:
        """Count one question that got no reply from the model."""
        with self._lock:
            self.unanswered += 1

    def add(self, model: str, response, seconds: Optional[float] = None) -> None:
        """Count one call the API answered, and how long it took when that was measured. Never
        raises: a response without usage, or with usage this code cannot read, is a call with no
        tokens."""
        try:
            tokens = [_count(response.usage, f) for f in _TOKENS]
        except Exception:
            tokens = [0] * len(_TOKENS)
        with self._lock:
            counts = self.models.setdefault(model, dict.fromkeys(("calls",) + _TOKENS, 0))
            counts["calls"] += 1
            for field, n in zip(_TOKENS, tokens):
                counts[field] += n
            if seconds is not None:
                self.seconds.setdefault(model, []).append(seconds)

    @staticmethod
    def cost(model: str, counts: dict) -> Optional[float]:
        """USD for one model's counts, or None when its price is not known."""
        per_million = price(model)
        if per_million is None:
            return None
        per_input, per_output = per_million
        cached = CACHE_READ * counts["cache_read_input_tokens"] + CACHE_WRITE * counts["cache_creation_input_tokens"]
        return (per_input * (counts["input_tokens"] + cached) + per_output * counts["output_tokens"]) / 1e6

    def _copy(self) -> dict:
        with self._lock:
            return {model: dict(m) for model, m in self.models.items()}

    def _times(self, model: str) -> list:
        with self._lock:
            return list(self.seconds.get(model, ()))

    def spent(self) -> float:
        """USD so far, over the models whose price is known."""
        costs = [self.cost(model, m) for model, m in self._copy().items()]
        return sum(c for c in costs if c is not None)

    def reached(self, budget: float) -> bool:
        """True once the spend has reached budget; the first time, says so."""
        spent = self.spent()
        if spent < budget:
            return False
        with self._lock:
            first, self.budget_said = not self.budget_said, True
        if first:
            print(f"[bankgpt] model spend ${spent:.4f} has reached BANKGPT_BUDGET_USD={budget:g}: no more model "
                  "calls in this run, the rest is answered from rules", file=sys.stderr)
        return True

    def summary(self) -> list:
        """The exit lines, one per model; none when no call was made."""
        lines = []
        for model, counts in sorted(self._copy().items()):
            usd = self.cost(model, counts)
            read, written = counts["cache_read_input_tokens"], counts["cache_creation_input_tokens"]
            cache = f" + {read:,} cache-read + {written:,} cache-write" if read or written else ""
            cost = ("cost unknown: no price for this model in kb/transport.py" if usd is None
                    else f"about ${usd:.2f}" if usd >= 0.005 else "under $0.01")
            times = self._times(model)
            spread = _spread(times)
            took = ("" if spread is None else f"; the call took {spread['max']:.1f} s" if len(times) == 1
                    else f"; calls took a median of {spread['median']:.1f} s, 95% under {spread['p95']:.1f} s, "
                         f"longest {spread['max']:.1f} s")
            calls = counts["calls"]
            lines.append(f"[bankgpt] model usage: {calls} call{'' if calls == 1 else 's'} to {model}, "
                         f"{counts['input_tokens']:,} input + {counts['output_tokens']:,} output{cache} tokens, {cost}{took}")
        return lines

    def log(self, path: str, label: str) -> None:
        """Append this run's spend to the ledger at path as one JSON line, with each model's median,
        95th percentile and longest call time in seconds when its calls were timed. A ledger that
        cannot be written is reported and skipped: it never fails the run."""
        models = self._copy()
        for model, counts in models.items():
            usd = self.cost(model, counts)
            counts["usd"] = None if usd is None else round(usd, 6)
            spread = _spread(self._times(model))
            if spread is not None:
                counts["seconds"] = {k: round(v, 3) for k, v in spread.items()}
        entry = {"time": int(time.time()), "label": label, "models": models}
        try:
            path = os.path.abspath(os.path.expanduser(path))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[bankgpt] could not add this run to the spend log {path} ({type(e).__name__})", file=sys.stderr)


USAGE = Usage()


def _at_exit() -> None:
    """The ledger line (when BANKGPT_SPEND_LOG is set) and one stderr line per model, then one line
    more when some questions got no reply. Nothing when no question reached the reader."""
    if USAGE.models and os.environ.get("BANKGPT_SPEND_LOG"):
        USAGE.log(os.environ["BANKGPT_SPEND_LOG"], os.environ.get("BANKGPT_SPEND_LABEL", ""))
    for text in USAGE.summary():
        print(text, file=sys.stderr)
    n = USAGE.unanswered
    if n:
        print(f"[bankgpt] {n} question{'' if n == 1 else 's'} got no reply from the model and "
              f"{'was' if n == 1 else 'were'} answered from the rules", file=sys.stderr)


atexit.register(_at_exit)


def _budget() -> Optional[float]:
    """BANKGPT_BUDGET_USD in dollars, or None when it is not set (no cap)."""
    raw = os.environ.get("BANKGPT_BUDGET_USD", "").strip()
    if not raw:
        return None
    try:
        budget = float(raw)
    except ValueError:
        budget = float("nan")
    if not budget >= 0:                   # negative, unreadable or NaN (for which `budget < 0` is false)
        # A cap that cannot be read must not turn into no cap: spend nothing, and say why.
        print(f"[bankgpt] BANKGPT_BUDGET_USD={raw!r} is not an amount in USD; no model calls in this run",
              file=sys.stderr)
        return 0.0
    return budget


# --------------------------------------------------------------------------- the transport


def settings(model: str) -> dict:
    """The fields SETTINGS adds to model's request. A dated model name (claude-sonnet-5-5-20260915)
    gets those of its alias, as price() prices it."""
    return SETTINGS.get(model) or SETTINGS.get(_alias(model)) or {}


def _kind(error) -> str:
    """What a failed request means, from its HTTP status code when it has one (the SDK's error
    classes differ between versions; the codes do not): rejected, key, busy, unreachable or other.
    error is None when the request did not come back in time."""
    if isinstance(error, TypeError) and "unexpected keyword argument" in str(error):
        return "rejected"                 # an SDK that does not know one of the fields sends nothing
    status = getattr(error, "status_code", None)
    if not isinstance(status, int):
        return "unreachable"              # no HTTP reply, a reply that is not a message, or any other fault
    if status in (400, 403, 404):
        return "rejected"
    if status == 401:
        return "key"
    if status in (408, 409, 429) or 500 <= status <= 599:
        return "busy"
    return "other"


def _why(error) -> str:
    """What went wrong, in a few words for stderr."""
    if error is None:
        return "no reply in time"
    status = getattr(error, "status_code", None)
    return f"{type(error).__name__} {status}" if isinstance(status, int) else type(error).__name__


def _retry_after(error) -> Optional[float]:
    """The seconds a busy reply's retry-after header asks for, or None when it gives no number of one
    second or more. The API sends whole seconds. Asking again sooner would only be busy again, and
    a tiny value would send thousands of requests in one question."""
    try:
        seconds = float(error.response.headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        return None
    return seconds if seconds >= 1 else None         # also None for NaN


def _text(response) -> Optional[str]:
    """The text of a message, or None when the model declines (stop reason "refusal") or gives no
    text. Raises ValueError for anything that is not a message: a proxy's HTML page comes back from
    the SDK as a plain string."""
    content = getattr(response, "content", None)
    if not isinstance(content, list):
        raise ValueError(f"the reply is not a message ({type(response).__name__})")
    if getattr(response, "stop_reason", None) == "refusal":
        return None
    return "".join(b.text for b in content if getattr(b, "type", None) == "text") or None


def _wait(event: threading.Event, seconds: float) -> bool:
    """event.wait(seconds), timed on the monotonic clock: with Python builds that time a wait on the
    wall clock, a wait ends early when the wall clock jumps forward."""
    end = time.monotonic() + seconds
    while not event.is_set():
        left = end - time.monotonic()
        if left <= 0:
            return False
        event.wait(left)
    return True


class AnthropicBackend:
    """The Anthropic API through the official SDK: claude-sonnet-5-5 unless BANKGPT_MODEL names
    another model, then the others in CHOICES. The request is the four fields every model accepts
    plus the model's own SETTINGS.

    complete() returns the reply's text or None, never raises, and spends at most ANSWER_BUDGET_S
    on one question. With None the question is answered without the model: kb.quote quotes the
    site or declines. Each request has its own timeout, and a watchdog for hangs the timeout
    misses. A failed request is read by its status code:
    - A model that rejects a request before it has ever answered (400, 403, 404, or a field this
      SDK does not know) is dropped for the run, and the next one is asked at once. When 400s
      leave no model at all, the request is taken to be at fault, and those models stay.
    - A model that has answered and then rejects one request, or any other status, costs that
      question only.
    - A busy model (408, 409, 429, 5xx) is left alone for its retry-after, or for 2 s doubling to
      15 s. A model that sends nothing back in all the time it had for its reply is left alone
      for 60 s doubling to 240 s. Either way the next model is asked at once.
    - An API that cannot be reached is asked again after 1 s and 3 s. The first question to find
      it so keeps asking for most of its time, so a short outage costs one slow answer.
    - A rejected key (401) switches the reader off.
    After three questions in a row without a reply the reader rests for five questions, then tries
    each model once; a failed try doubles the rest, up to 80. Rests are counted in questions, not
    seconds: rule answers are instant, so a pause in seconds would cover the rest of a run.

    Every reply is counted in USAGE, with how long the call took, even one that comes back after
    the watchdog gave up on it: it was paid for."""
    # The models tried in turn: BANKGPT_MODEL (or claude-sonnet-5-5), then the others in SETTINGS order.
    CHOICES = [_MODEL] + [m for m in SETTINGS if m != _MODEL]
    # Seconds, and rests in questions. Class attributes, so a test can shrink them.
    ANSWER_BUDGET_S = 40.0               # one question, whatever the API does (the scorer allows 60)
    CONNECT_S, REPLY_S = 5.0, 25.0       # one request: to connect, and for its reply
    WATCHDOG_S = 1.0                     # a request is waited for this long past its own timeout
    MIN_LEFT_S = 2.0                     # no request is started with less time than this left
    # Waits before asking an unreachable API again: the first question to find it so, then the later ones.
    FIRST_WAITS = (1.0, 3.0, 5.0, 8.0, 8.0, 8.0, 8.0)
    RETRY_WAITS = (1.0, 3.0)
    MAX_RETRY_AFTER_S = 30.0             # the longest retry-after taken as it is given
    BUSY_S, MAX_BUSY_S = 2.0, 15.0       # without one: 2 s, doubled while the model stays busy
    SLOW_S, MAX_SLOW_S = 60.0, 240.0     # a model that sent nothing in all its time: 60 s, doubled likewise
    MISSES = 3                           # questions in a row without a reply before a rest
    REST, MAX_REST = 5, 80               # a rest's length, doubled after each failed try

    def __init__(self) -> None:
        import anthropic
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(timeout=self.REPLY_S, max_retries=0)
        self._sleep, self._now = time.sleep, time.monotonic    # tests replace both, so nothing waits
        self._lock = threading.Lock()     # dev/run.py --jobs calls complete() from several threads
        self.dropped: set = set()         # models that rejected a request before they ever answered: not asked again
        self.answered: set = set()        # models that have replied in this run
        self.busy: dict = {}              # model -> (when it may be asked again, busy or slow replies in a row)
        self.off = False                  # the key was rejected: no more calls in this run
        self.misses = 0                   # questions in a row that got no reply
        self.rest = self.REST             # how many questions the next rest lasts
        self.resting = 0                  # questions left in this rest
        self.probing = False              # the question after a rest asks each model once
        self._said: set = set()           # the kinds of event already reported on stderr
        self.budget = _budget()           # development only: None, the default, is no cap
        if self.budget is not None and price(_MODEL) is None:
            print(f"[bankgpt] {_MODEL} has no price in kb/transport.py, so BANKGPT_BUDGET_USD cannot limit what it spends",
                  file=sys.stderr)

    def complete(self, system: str, user: str) -> Optional[str]:
        """The reply's text for one question, or None; never raises (see the class docstring)."""
        try:
            text, replied = self._ask(system, user)
        except Exception as e:                # a fault in this code must not cost the answer either
            self._say("fault", f"the reader failed ({type(e).__name__}: {e}); answering from rules")
            text, replied = None, False
        if not replied:
            USAGE.no_reply()
        return text

    def _ask(self, system: str, user: str) -> tuple:
        """(the reply's text or None, whether the model replied) for one question."""
        with self._lock:
            if self.off or all(m in self.dropped for m in self.CHOICES):     # the key or every model was rejected
                return None, False
            if self.resting:
                self.resting -= 1
                return None, False
            probe = self.probing
            waits = self.FIRST_WAITS if self.misses == 0 and not probe else self.RETRY_WAITS
        # Checked before each question, not before each call, so the question that crosses the budget
        # still finishes: the spend can go over by that question's calls (one such question per
        # thread when several answer at once), and by late replies the watchdog had given up on.
        if self.budget is not None and USAGE.reached(self.budget):
            return None, False
        deadline = self._now() + self.ANSWER_BUDGET_S
        model, tried, fails, bad = None, set(), 0, set()
        while True:
            left = deadline - self._now()
            if left < self.MIN_LEFT_S:
                break
            if model is None:
                model, wait = self._pick(probe, tried)
                if model is None:
                    # Every usable model is busy: wait for the first to be free, if a request can follow.
                    if wait is None or left - wait < self.MIN_LEFT_S:
                        break
                    self._sleep(wait)
                    continue
            tried.add(model)
            start = self._now()
            out = self._send(model, system, user, left)
            if "reply" in out:
                self._replied(model, probe)
                return out["reply"], True
            error = out.get("error")
            kind = _kind(error)
            if kind == "unreachable" and self._now() - start >= min(self.REPLY_S, left):
                kind = "slow"                 # it had all its time and sent nothing: the model, not the network
            if kind == "rejected":
                with self._lock:
                    if model not in self.answered:      # it never answered: the fault is the model's
                        self.dropped.add(model)
                        kind = "dropped"
            if kind == "key":
                with self._lock:
                    self.off = True
                self._say("key", f"the API did not accept the key ({_why(error)}): no more model calls in this "
                                 "run, the rest is answered from rules")
                return None, False
            if kind == "dropped":
                if getattr(error, "status_code", None) == 400:
                    bad.add(model)
                if isinstance(error, TypeError):
                    self._say("drop " + model, f"this anthropic SDK cannot send {model}'s request ({error}); trying "
                                               "the next model. To send it: pip install -U -r requirements.txt")
                else:
                    self._say("drop " + model, f"{model} was not accepted ({_why(error)}); trying the next model")
                model = None
            elif kind in ("busy", "slow"):
                with self._lock:
                    count = self.busy.get(model, (0.0, 0))[1] + 1
                    self.busy[model] = (self._now() + self._pause(kind, error, count), count)
                if kind == "busy":
                    self._say("busy", f"the API is busy ({_why(error)} from {model}): asking another model, or "
                                      "waiting for one to be free")
                else:
                    self._say("slow", f"no reply from {model} in time ({_why(error)}): asking another model, and "
                                      "leaving this one alone for a while")
                model = None
            elif kind == "unreachable":
                fails += 1
                self._say("unreachable", f"no reply from the API ({_why(error)}): asking again after a short wait")
                if probe:
                    model = None              # a probe asks each model once, without waiting
                elif fails > len(waits):
                    break
                else:                         # the same model again: another one would not help
                    wait = waits[fails - 1]
                    if deadline - self._now() - wait < self.MIN_LEFT_S:
                        break
                    self._sleep(wait)
            else:                             # rejected by a model that has answered, or any other status:
                self._say("rejected", f"{model} did not accept one question's request ({_why(error)}); it stays "
                                     "in use for the others")
                if not probe:                 # the fault is in this one request
                    break
                model = None                  # a probe asks the next model all the same
        if bad:
            with self._lock:
                every = all(m in self.dropped for m in self.CHOICES)
                if every:                     # one request rejected by all: more likely its fault than theirs
                    self.dropped -= bad
            if every:
                self._say("bad request", "no model accepted one question's request (400): the request is taken to "
                                         "be at fault, so the models stay in use")
        self._missed(probe)
        return None, False

    def _pause(self, kind: str, error, count: int) -> float:
        """How long a busy or slow model is left alone: a busy reply's retry-after, up to 30 s; or else
        2 s when busy and 60 s when slow, doubled each time it is so again before it answers (up to
        15 s and 240 s)."""
        after = _retry_after(error) if kind == "busy" else None
        if after is not None:
            return min(after, self.MAX_RETRY_AFTER_S)
        first, most = (self.BUSY_S, self.MAX_BUSY_S) if kind == "busy" else (self.SLOW_S, self.MAX_SLOW_S)
        return min(first * 2 ** min(count - 1, 8), most)      # a bounded power: 2.0 * 2 ** 1024 overflows

    def _pick(self, probe: bool, tried: set) -> tuple:
        """(the model to ask next, None), or (None, seconds until the first busy model is free), or
        (None, None) when no model is left. A probe asks each model that is not dropped once, busy
        or not: what made it busy is older than the rest."""
        now = self._now()
        with self._lock:
            usable = [m for m in self.CHOICES if m not in self.dropped and not (probe and m in tried)]
            free = [m for m in usable if probe or self.busy.get(m, (0.0, 0))[0] <= now]
            if free:
                return free[0], None
            return None, (min(self.busy[m][0] for m in usable) - now if usable else None)

    def _send(self, model: str, system: str, user: str, left: float) -> dict:
        """One request, on a thread of its own, so a hang its timeout misses (a name lookup, a reply
        that trickles in) is cut off too. {"reply": text or None}, {"error": what was raised}, or {}
        when it did not come back within its timeout plus WATCHDOG_S."""
        request = dict(model=model, max_tokens=600, system=system, messages=[{"role": "user", "content": user}],
                       timeout=self._anthropic.Timeout(min(self.REPLY_S, left), connect=min(self.CONNECT_S, left)),
                       **settings(model))
        out, done = {}, threading.Event()

        def send() -> None:
            start = self._now()               # the wall clock can jump; this one never does
            try:
                response = self.client.messages.create(**request)
                text = _text(response)
                USAGE.add(model, response, self._now() - start)   # here, so a late reply is counted too, once
                out["reply"] = text
            except Exception as e:
                out["error"] = e
            finally:
                done.set()

        threading.Thread(target=send, daemon=True).start()     # a daemon: a hung request never delays the exit
        _wait(done, min(self.REPLY_S + self.WATCHDOG_S, left))
        return dict(out)

    def _replied(self, model: str, probe: bool) -> None:
        with self._lock:
            self.answered.add(model)
            self.busy.pop(model, None)
            self.misses, self.rest, self.probing = 0, self.REST, False
        if probe:
            self._say("back", "the model replies again: back to normal")

    def _missed(self, probe: bool) -> None:
        """A question got no reply. The third in a row starts a rest; a failed try after a rest
        doubles it."""
        with self._lock:
            if probe:
                self.rest = min(2 * self.rest, self.MAX_REST)
            else:
                self.misses += 1
                if self.misses < self.MISSES:
                    return
                self.misses, self.probing = 0, True
            self.resting = rest = self.rest
        self._say("rest", f"{self.MISSES} questions in a row got no reply from the model: the next {rest} are "
                          "answered from rules, then each model is tried once (a longer rest each time that fails)")

    def _say(self, kind: str, text: str) -> None:
        """One stderr line per kind of event in a run, however many calls or threads see it."""
        with self._lock:
            if kind in self._said:
                return
            self._said.add(kind)
        print(f"[bankgpt] {text}", file=sys.stderr)
