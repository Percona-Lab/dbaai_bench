"""Run every offline suite and report one line per suite.

    uv run python run_tests.py
    uv run python run_tests.py guard wrap   # just those

Run it through uv: paramiko, rich and the OpenAI SDK come from this project's
environment.

tests/test_dba_live.py is deliberately not in this list. It calls a hosted model
with a real key and costs real money, so it stays a thing you run on purpose.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = HERE / "tests"
CHILD_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

# This process prints what the suites printed, so it needs the same treatment as
# they do - otherwise the runner itself dies on the output of a failing suite,
# which is the one moment it has to work.
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8", errors="replace")

# Each suite prints its own detail and exits non-zero on failure, so the runner
# only has to report the exit code - except that a suite can also fail by
# printing "FAIL" while still exiting 0 if it is ever run under a broken shell,
# so the output is checked too.
SUITES = [
    ("guard", "test_dba_guard.py", False),
    ("offline", "test_dba_offline.py", False),
    ("prompts", "test_dba_prompts.py", False),
    ("engines", "test_dba_engines.py", False),
    ("fleet", "test_dba_fleet.py", False),
    ("secrets", "test_dba_secrets.py", False),
    ("wrap", "test_dba_wrap.py", False),
    ("providers", "test_dba_providers.py", False),
    ("openrouter-wire", "test_dba_openrouter_wire.py", False),
    ("selfhosted-wire", "test_dba_selfhosted_wire.py", False),
    # This one talks to mock_do_server.py on 127.0.0.1:8899, which the runner
    # starts for it; on its own it fails with a connection error.
    ("client", "test_dba_client.py", True),
]
MOCK_PORT = 8899


def port_open(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def start_mock() -> subprocess.Popen | None:
    """The mock inference server, unless something is already on its port."""
    if port_open(MOCK_PORT):
        print(f"  using the server already listening on {MOCK_PORT}")
        return None
    process = subprocess.Popen(
        [sys.executable, str(HERE / "mock_do_server.py")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(50):  # 5s, in case the interpreter is cold
        if port_open(MOCK_PORT):
            return process
        time.sleep(0.1)
    process.terminate()
    raise SystemExit(f"mock_do_server.py did not come up on {MOCK_PORT}")


def main(argv: list[str]) -> int:
    wanted = set(argv[1:])
    suites = [s for s in SUITES if not wanted or s[0] in wanted]
    if not suites:
        raise SystemExit(f"no suite matches {sorted(wanted)}; have {[s[0] for s in SUITES]}")

    mock = None
    results: list[tuple[str, bool, str]] = []
    try:
        for name, script, needs_mock in suites:
            if needs_mock and mock is None:
                mock = start_mock()
            print(f"--- {name}")
            done = subprocess.run(
                [sys.executable, str(TESTS / script)],
                cwd=HERE, capture_output=True, text=True, timeout=900,
                # Both ends of the pipe, explicitly. The default on this platform
                # is cp1252, and real command output in the transcripts is full of
                # box drawing and quotes it cannot represent: the child dies
                # writing them, and the parent dies reading them.
                encoding="utf-8", errors="replace", env=CHILD_ENV,
            )
            tail = (done.stdout or "").strip().splitlines()
            summary = next((line for line in reversed(tail) if line.strip()), "(no output)")
            ok = done.returncode == 0 and "FAIL" not in (done.stdout or "")
            results.append((name, ok, summary.strip()))
            if not ok:
                # The whole thing, not the summary: a failure is why you ran this.
                print((done.stdout or "").rstrip())
                if done.stderr.strip():
                    print(done.stderr.rstrip(), file=sys.stderr)
    finally:
        if mock is not None:
            mock.terminate()

    print()
    width = max(len(name) for name, _, _ in results)
    for name, ok, summary in results:
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:<{width}}  {summary}")
    failed = [name for name, ok, _ in results if not ok]
    print()
    print(f"{len(results) - len(failed)}/{len(results)} suites passed"
          + (f" - failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
