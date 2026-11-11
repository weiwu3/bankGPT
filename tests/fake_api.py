"""A local stand-in for the Anthropic Messages API (POST /v1/messages), for tests and dev scripts.

Standard library only. It listens on a free port of 127.0.0.1; point the SDK at it with
ANTHROPIC_BASE_URL=<fake.url> and any non-empty ANTHROPIC_API_KEY.

Each call is answered by a list of rules. The first rule that covers the call is used; a call no
rule covers gets `default` (a reply). Calls are numbered from 1 in the order they arrive, retries
included. What a rule covers:

    {"calls": 3, ...}              the third call only
    {"calls": [3, 5], ...}         calls 3 to 5
    {"calls": [5, None], ...}      call 5 and every later one
    {"model": "claude-x", ...}     calls that ask for this model
    {"from_s": 1, "to_s": 16, ...} calls that arrive from 1 s to 16 s after the first call
    none of these                  every call

What a rule does:

    {"do": "reply"}                          HTTP 200 with a valid message whose text is `text`
    {"do": "reply", "text": "..."}           the same with this text
    {"do": "reply", "body": "<html>..."}     HTTP 200 with this body as it is: not a message at all
    {"do": "status", "status": 529}          an error status with an error body (429, 500, 401 ...)
    {"do": "status", "status": 429, "retry_after": 1}    with a retry-after header
    {"do": "close"}                          read the request, then close without answering
    {"do": "hang", "seconds": 30}            read the request, send nothing for 30 s, then close
    {"do": "trickle", "seconds": 60}         HTTP 200 with a valid message, sent a byte at a time over
                                             60 s: the SDK's read timeout never fires (it restarts
                                             with each byte), so only a watchdog stops the wait
    "delay": 0.3                             with any of them: wait this long first
    "headers": {"name": "value"}             with "reply" or "status": extra response headers

    with FakeAnthropic([{"calls": 3, "do": "status", "status": 529}]) as fake:
        os.environ.update(fake.env())
        ...
    fake.requests      # one dict per request: call, t and done (time.monotonic()), model, path,
                       # action, status (None when nothing was sent), prompt (the last message's text)

From a shell, for trying the solution by hand:

    python tests/fake_api.py '{"calls": [2, 3], "do": "close"}'
"""
from __future__ import annotations

import json
import socket
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

# The error type the API puts in the body for each status; the messages are short stand-ins.
ERRORS = {
    400: ("invalid_request_error", "Invalid request"),
    401: ("authentication_error", "invalid x-api-key"),
    403: ("permission_error", "Your API key does not have permission to use the specified resource."),
    404: ("not_found_error", "model: {model}"),
    413: ("request_too_large", "Request exceeds the maximum allowed number of bytes."),
    429: ("rate_limit_error", "Number of requests has exceeded your rate limit."),
    500: ("api_error", "Internal server error"),
    503: ("api_error", "Service unavailable"),
    529: ("overloaded_error", "Overloaded"),
}
ACTIONS = ("reply", "status", "close", "hang", "trickle")
KEYS = {"calls", "model", "from_s", "to_s", "do", "status", "retry_after", "delay", "seconds", "text", "body",
        "headers"}


def _check(rule: dict) -> dict:
    """The rule itself, or ValueError for a misspelt key or action, so a typo fails at once."""
    unknown = set(rule) - KEYS
    if unknown:
        raise ValueError(f"unknown key(s) {sorted(unknown)} in rule {rule}")
    if rule.get("do", "reply") not in ACTIONS:
        raise ValueError(f"unknown action {rule.get('do')!r} in rule {rule}; use one of {ACTIONS}")
    if rule.get("do") == "status" and not isinstance(rule.get("status"), int):
        raise ValueError(f"rule {rule} needs an integer status")
    calls = rule.get("calls")
    if calls is not None and not isinstance(calls, int) and not (isinstance(calls, (list, tuple)) and len(calls) == 2):
        raise ValueError(f"calls must be a number or [first, last] in rule {rule}")
    return rule


def _prompt(request) -> Optional[str]:
    """The text of the request's last message: what was asked."""
    try:
        content = request["messages"][-1]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content if isinstance(content, str) else None


def _covers(rule: dict, call: int, model: Optional[str], since: float) -> bool:
    """Whether the rule covers this call; since is how many seconds after the first call it came."""
    calls = rule.get("calls")
    if calls is not None:
        first, last = (calls, calls) if isinstance(calls, int) else calls
        if call < first or (last is not None and call > last):
            return False
    if since < rule.get("from_s", 0) or since >= rule.get("to_s", float("inf")):
        return False
    return rule.get("model") is None or rule["model"] == model


class FakeAnthropic:
    """The fake API: start() it, or use it in a with block, and point the SDK at its url."""

    def __init__(self, rules=(), default: Optional[dict] = None, text: str = "Hello from the fake API.") -> None:
        self.rules = [_check(dict(r)) for r in rules]
        self.default = _check(dict(default or {"do": "reply"}))
        self.text = text
        self.requests: list = []           # one dict per request, in the order they arrived
        self._lock = threading.Lock()
        self._calls = 0
        self._first = None                  # when the first call came (time.monotonic())
        self._stopping = threading.Event()  # wakes every delay and hang at once when the fake stops
        self._open: set = set()             # client connections, closed by stop()
        self._server: Optional[_Server] = None
        self.port = 0

    # ------------------------------------------------------------------ start and stop

    def start(self) -> "FakeAnthropic":
        self._stopping.clear()
        self._server = _Server(("127.0.0.1", 0), _Handler)
        self._server.fake = self
        self.port = self._server.server_port
        threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        return self

    def stop(self) -> None:
        """Stops at once, even while a handler hangs or a client keeps its connection open."""
        if self._server is None:
            return
        self._stopping.set()
        self._server.shutdown()
        with self._lock:
            conns = list(self._open)
        for conn in conns:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self._server.server_close()
        self._server = None

    def __enter__(self) -> "FakeAnthropic":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def env(self) -> dict:
        """The variables that point the SDK here."""
        return {"ANTHROPIC_BASE_URL": self.url, "ANTHROPIC_API_KEY": "fake-key"}

    @property
    def calls(self) -> list:
        """The requests to /v1/messages, the ones the rules answer."""
        with self._lock:
            return [r for r in self.requests if r["call"] is not None]

    # ------------------------------------------------------------------ one request

    def _pause(self, seconds: float) -> None:
        """Waits this long on the monotonic clock, or until stop(). Event.wait alone can end early
        when the wall clock steps forward, as it does every half minute or so on a WSL2 machine."""
        end = time.monotonic() + seconds
        while not self._stopping.is_set() and time.monotonic() < end:
            self._stopping.wait(min(end - time.monotonic(), 0.25))

    def _take(self, path: str, request) -> tuple:
        """Numbers the call, picks its rule and records it. Returns (call, rule, record)."""
        model = request.get("model") if isinstance(request, dict) else None
        with self._lock:
            if path == "/v1/messages":
                self._calls += 1
                call = self._calls
                if self._first is None:
                    self._first = time.monotonic()
                since = time.monotonic() - self._first
                rule = next((r for r in self.rules if _covers(r, call, model, since)), self.default)
            else:
                call, rule = None, {"do": "status", "status": 404}
            record = {"call": call, "t": time.monotonic(), "done": None, "model": model, "path": path,
                      "action": rule.get("do", "reply"), "status": None, "prompt": _prompt(request)}
            self.requests.append(record)
        return call, rule, record


class _Server(ThreadingHTTPServer):
    daemon_threads = True        # a hanging handler never keeps the process alive
    block_on_close = False       # and server_close() does not wait for it

    def server_bind(self) -> None:
        # HTTPServer.server_bind looks the host name up, which can take seconds; nothing here needs it.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    def handle_error(self, request, client_address) -> None:
        if not isinstance(sys.exc_info()[1], OSError):     # a client that went away is not an error
            super().handle_error(request, client_address)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"          # connections are kept open between calls, as the real API does
    disable_nagle_algorithm = True         # headers and body go out at once, not 40 ms apart

    def setup(self) -> None:
        super().setup()
        with self.server.fake._lock:
            self.server.fake._open.add(self.connection)

    def finish(self) -> None:
        with self.server.fake._lock:
            self.server.fake._open.discard(self.connection)
        super().finish()

    def log_message(self, *args) -> None:
        pass                               # quiet: the requests are recorded in fake.requests

    def do_POST(self) -> None:
        fake = self.server.fake
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        try:
            request = json.loads(body or b"{}")
        except ValueError:
            request = {}
        call, rule, record = fake._take(self.path.split("?")[0], request)
        try:
            self._act(fake, call, rule, record["model"], record)
        finally:
            record["done"] = time.monotonic()

    def _act(self, fake: FakeAnthropic, call, rule: dict, model, record: dict) -> None:
        do = rule.get("do", "reply")
        if rule.get("delay"):
            fake._pause(float(rule["delay"]))
        if do == "reply" and "body" in rule:
            self._send(200, rule["body"], call, rule, record)
        elif do in ("reply", "trickle"):
            text = rule.get("text", fake.text)
            self._send(200, {"id": f"msg_fake_{call}", "type": "message", "role": "assistant", "model": model,
                             "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
                             "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5}},
                       call, rule, record, over=float(rule.get("seconds", 60)) if do == "trickle" else 0.0)
        elif do == "status":
            kind, message = ERRORS.get(rule["status"], ("api_error", "Error"))
            self._send(rule["status"], {"type": "error", "error": {"type": kind, "message": message.format(model=model)},
                                        "request_id": f"req_fake_{call}"}, call, rule, record)
        else:
            if do == "hang":
                fake._pause(float(rule.get("seconds", 60)))
            self.close_connection = True   # "close", or the end of a hang: no answer at all

    def _send(self, status: int, body, call, rule: dict, record: dict, over: float = 0.0) -> None:
        """Sends the reply; with over, its body a byte at a time, spread over that many seconds."""
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        headers = {"Content-Type": "application/json", "Content-Length": str(len(data)),
                   "request-id": f"req_fake_{call}"}
        if rule.get("retry_after") is not None:
            headers["retry-after"] = str(rule["retry_after"])
        headers.update(rule.get("headers") or {})
        record["status"] = status
        try:
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            if over:
                for i in range(len(data)):
                    if self.server.fake._stopping.is_set():
                        break
                    self.wfile.write(data[i:i + 1])
                    self.server.fake._pause(over / len(data))
            else:
                self.wfile.write(data)
            self.wfile.flush()
        except OSError:                    # the client gave up first (its own timeout)
            self.close_connection = True


if __name__ == "__main__":
    fake = FakeAnthropic([json.loads(a) for a in sys.argv[1:]]).start()
    print(f"export ANTHROPIC_BASE_URL={fake.url} ANTHROPIC_API_KEY=fake-key    (Ctrl-C stops it)", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        fake.stop()
        for r in fake.requests:
            print(json.dumps(r))
