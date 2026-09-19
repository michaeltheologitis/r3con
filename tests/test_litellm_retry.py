"""Empirical check: does ``litellm.completion(num_retries=N)`` actually retry, and
would it have rescued our transient-failure aborts?

Our big vLLM run died because per-task litellm calls failed with transient/server
errors — surfaced as ``litellm.exceptions.InternalServerError`` (the ``[Errno 111]
Connection refused`` was wrapped as InternalServerError), ``Timeout``, and
``Connection reset``. One failed schema/parsing call aborts the whole run.
Question: does adding ``num_retries`` turn those into successes?

This is **hermetic** — it spins a local mock OpenAI-compatible HTTP server (no
creds, no real LLM) that fails the first K requests, then succeeds. It drives
``litellm.completion`` against it and counts how many requests actually hit the
server. Failure modes covered:
  - HTTP 500  → litellm InternalServerError  (our run's dominant class)
  - mid-request socket close → connection error (our resets / refused family)

Takes ~20-30s (litellm sleeps between retries with exponential backoff).

Run:  uv run python tests/test_litellm_retry.py
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import litellm

# Mutable config the handler reads; reset per scenario.
STATE = {"fail_first": 0, "mode": "500", "count": 0}

_OK_COMPLETION = {
    "id": "chatcmpl-mock",
    "object": "chat.completion",
    "created": 0,
    "model": "mock-model",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_a):  # silence the default request logging
        pass

    def do_GET(self):  # tolerate any health probe
        self._json(200, {"status": "ok"})

    def do_POST(self):
        STATE["count"] += 1
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)  # drain body

        if STATE["count"] <= STATE["fail_first"]:
            if STATE["mode"] == "disconnect":
                # Close the connection without responding → httpx RemoteProtocolError
                # → litellm connection error (our "reset by peer" / "refused" family).
                try:
                    self.close_connection = True
                    self.connection.close()
                except Exception:
                    pass
                return
            self._json(500, {"error": {"message": "mock 500", "type": "server_error"}})
            return

        self._json(200, _OK_COMPLETION)

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _call(base_url: str, num_retries: int):
    return litellm.completion(
        model="hosted_vllm/mock-model",          # same provider path as the real run
        messages=[{"role": "user", "content": "hi"}],
        api_base=base_url,
        api_key="sk-mock",
        num_retries=num_retries,
        timeout=10,                              # test only: fail fast so it stays quick
    )


def main() -> int:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/v1"

    results: list[tuple[str, bool, str]] = []

    def record(name, passed, detail):
        results.append((name, passed, detail))
        print(f"  [{'PASS' if passed else 'FAIL'}] {name} — {detail}")

    print("=== litellm num_retries behavior (mock server) ===")

    # 1) Default (no retries): a single failure must abort — our CURRENT behavior.
    STATE.update(fail_first=1, mode="500", count=0)
    try:
        _call(base, 0)
        record("default (num_retries=0): 1×500 → aborts", False, f"unexpected SUCCESS, count={STATE['count']}")
    except Exception as e:
        record("default (num_retries=0): 1×500 → aborts", True, f"raised {type(e).__name__}, requests={STATE['count']}")

    # 2) THE key one: transient — fail twice, then succeed. num_retries=3 should recover.
    STATE.update(fail_first=2, mode="500", count=0)
    try:
        r = _call(base, 3)
        ok = r.choices[0].message.content == "OK"
        record("num_retries=3: 2×500 then 200 → RECOVERS", ok, f"success={ok}, requests={STATE['count']} (expect 3)")
    except Exception as e:
        record("num_retries=3: 2×500 then 200 → RECOVERS", False, f"raised {type(e).__name__}, requests={STATE['count']}")

    # 3) Sustained outage: always fail. Retries should NOT save it (but should retry N×).
    STATE.update(fail_first=999, mode="500", count=0)
    try:
        _call(base, 2)
        record("num_retries=2: sustained 500 → still fails", False, f"unexpected SUCCESS, count={STATE['count']}")
    except Exception as e:
        retried = STATE["count"] >= 3  # 1 try + 2 retries
        record("num_retries=2: sustained 500 → still fails", retried,
               f"raised {type(e).__name__} after {STATE['count']} requests (expect ≥3)")

    # 4) Connection-error family (socket close) — transient, num_retries=3 should recover.
    STATE.update(fail_first=2, mode="disconnect", count=0)
    try:
        r = _call(base, 3)
        ok = r.choices[0].message.content == "OK"
        record("num_retries=3: 2×conn-drop then 200 → RECOVERS", ok, f"success={ok}, requests={STATE['count']}")
    except Exception as e:
        record("num_retries=3: 2×conn-drop then 200 → RECOVERS", False, f"raised {type(e).__name__}, requests={STATE['count']}")

    srv.shutdown()

    passed = sum(1 for _, p, _ in results if p)
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
