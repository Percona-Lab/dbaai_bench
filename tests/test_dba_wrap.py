"""What the far end's shell actually does with a wrapped command.

The wrapping is a string built on this side and parsed on the other, so the only
honest test is to hand it to a real shell. bash is local here, and the properties
that matter are shell properties, not remote ones: a broken script must not run
its first lines, a working one must keep its exit code, its stdout, and pipefail.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]  # the suites sit in tests/, the harness above it
# Ahead of anything installed on purpose: the point is to test this tree.
sys.path.insert(0, str(PROJECT))

from do_dba.agent import SIGPIPE_EXIT, filter_matched_nothing, pipe_closed_early
from do_dba.ssh import CommandResult, SYNTAX_ERROR_NOTE, wrap_command

# Created up front: if the directory were missing, `touch` on the far end would
# fail and every "nothing ran" assertion below would pass without proving a thing.
SCRATCH = PROJECT / "_scratch"
SCRATCH.mkdir(parents=True, exist_ok=True)
MARKER = SCRATCH / "dba_wrap_marker.txt"


def shell(wrapped: str) -> subprocess.CompletedProcess:
    """Run as sshd would: hand the whole string to a shell."""
    return subprocess.run(["bash", "-c", wrapped], capture_output=True, text=True, timeout=60)


def check(failures: list[str], condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    failures: list[str] = []

    # ------------------------------------------------- a syntax error runs nothing
    MARKER.unlink(missing_ok=True)
    broken = (
        f'echo "=== first ==="\n'
        f'touch {MARKER.as_posix()}\n'
        f'mariadb -N -e "SELECT 1 ORDER BY n | head -100'
    )
    result = shell(wrap_command(broken))
    check(failures, not MARKER.exists(),
          "the lines before the broken one were executed - the step was half-applied")
    check(failures, "=== first ===" not in result.stdout,
          f"output was produced by a command that should never have run: {result.stdout!r}")
    check(failures, result.returncode == 2, f"expected exit 2, got {result.returncode}")
    check(failures, SYNTAX_ERROR_NOTE.split(":")[1].strip()[:30] in result.stderr,
          f"the harness did not say nothing ran:\n{result.stderr}")
    check(failures, "unexpected EOF" in result.stderr,
          "bash's own diagnosis should still reach the model")

    # ---------------------------------------------------- a good command is intact
    ok = shell(wrap_command('echo one; echo two >&2; printf "three\\n"'))
    check(failures, ok.returncode == 0, f"a valid command exited {ok.returncode}: {ok.stderr!r}")
    check(failures, ok.stdout.splitlines()[-2:] == ["one", "three"],
          f"stdout was altered: {ok.stdout!r}")
    check(failures, "two" in ok.stderr, "stderr was lost")

    exit_code = shell(wrap_command("exit 42"))
    check(failures, exit_code.returncode == 42, f"the exit code was not preserved: {exit_code.returncode}")

    # pipefail: a failure behind a pipe must not be reported as success.
    piped = shell(wrap_command("false | tail -n 1"))
    check(failures, piped.returncode != 0, "pipefail was lost in the wrapping")

    # The other edge of pipefail: a reader that stops reading. `head` closes the pipe
    # by design, the shell kills the writer for it, and pipefail hands that back as
    # the pipeline's code - so a command that did exactly what was asked comes back
    # 141. Nine steps across the recorded runs ended this way, so the number is
    # pinned against a real bash here rather than taken on trust in agent.py.
    for command in ("yes ok | head -n 3", "yes ok | grep -q ok"):
        closed = shell(wrap_command(command))
        check(failures, closed.returncode == SIGPIPE_EXIT,
              f"`{command}` exited {closed.returncode}, not {SIGPIPE_EXIT}")
        check(failures, pipe_closed_early(CommandResult(
            command=command, exit_code=closed.returncode, stdout=closed.stdout,
            stderr=closed.stderr, duration=0.1)),
            f"the harness did not recognise the closed pipe in `{command}`")
    check(failures, shell(wrap_command("yes ok | head -n 3")).stdout.splitlines()[-3:]
          == ["ok", "ok", "ok"], "the reader's own output was lost")

    # `tail` reads to the end, so nothing is killed and rule 6's advice stays safe -
    # and a real failure behind a truncating pipe still comes back as that failure,
    # which is what stops this from excusing every non-zero code near a `| head`.
    tailed = shell(wrap_command("printf 'a\\nb\\nc\\n' | tail -n 1"))
    check(failures, tailed.returncode == 0, f"`| tail` exited {tailed.returncode}")
    real = shell(wrap_command("apt-get-does-not-exist | head -n 3"))
    check(failures, real.returncode not in (0, SIGPIPE_EXIT),
          f"a command that really failed behind `| head` exited {real.returncode}")
    check(failures, not pipe_closed_early(CommandResult(
        command="apt-get-does-not-exist | head -n 3", exit_code=real.returncode,
        stdout=real.stdout, stderr=real.stderr, duration=0.1)),
        "a real failure behind a truncating pipe was excused as a closed pipe")

    # The same pipefail edge from the other side: a filter that matches nothing exits
    # 1, and there is nothing left in the output to say whether the command before it
    # worked. Real commands, because the whole point is which exit code bash produces.
    filtered = shell(wrap_command("echo 'mysql: [Warning] using a password' | grep -v Warning"))
    check(failures, filtered.returncode == 1 and not filtered.stdout.strip(),
          f"a filter that removed everything gave {filtered.returncode}: {filtered.stdout!r}")
    result = CommandResult(command="echo x | grep -v Warning", exit_code=filtered.returncode,
                           stdout=filtered.stdout, stderr=filtered.stderr, duration=0.1)
    check(failures, filter_matched_nothing(result),
          "the harness did not recognise a filter that matched nothing")
    check(failures, not pipe_closed_early(result),
          "an empty filter was mistaken for a closed pipe")

    # And where it must stay quiet: output got through, so the 1 came from upstream and
    # is the command's own; and `grep -q` was asked a question, where 1 is the answer.
    upstream = shell(wrap_command("(echo 'ERROR 1045: denied'; exit 1) | grep -v Warning"))
    check(failures, upstream.returncode == 1, f"expected exit 1, got {upstream.returncode}")
    check(failures, not filter_matched_nothing(CommandResult(
        command="mysql -e 'SELECT 1' | grep -v Warning", exit_code=1,
        stdout=upstream.stdout, stderr=upstream.stderr, duration=0.1)),
        "a real failure was explained away as an empty filter")
    asked = shell(wrap_command("echo present | grep -q absent"))
    check(failures, asked.returncode == 1, f"`grep -q` with no match gave {asked.returncode}")
    check(failures, not filter_matched_nothing(CommandResult(
        command="echo present | grep -q absent", exit_code=asked.returncode,
        stdout=asked.stdout, stderr=asked.stderr, duration=0.1)),
        "a test the model wrote on purpose was treated as a lost exit code")

    # Quoting survives being wrapped twice: single quotes, double quotes, $, and
    # backslashes are exactly where a hand-rolled wrapper goes wrong.
    tricky = """printf '%s\\n' "it's \\$HOME" 'a "b" c' "back\\\\slash" """
    quoting = shell(wrap_command(tricky))
    check(failures, quoting.returncode == 0, f"a quoted command failed: {quoting.stderr!r}")
    lines = quoting.stdout.splitlines()[-3:]
    check(failures, lines == ["it's $HOME", 'a "b" c', "back\\slash"],
          f"quoting did not survive the wrapping: {lines!r}")

    # `mysql -e "SHOW REPLICA STATUS\G"` is refused by the client on every recorded run,
    # and RESULT_HINTS tells the model that this is not a quoting problem - that the
    # backslash reaches the client exactly as written. That claim is about this wrapping,
    # so it is checked here, against a real shell, one byte at a time.
    survived = shell(wrap_command(r"""printf '%s' "SHOW REPLICA STATUS\G" | od -An -c"""))
    passed_on = "".join(survived.stdout.split())
    check(failures, passed_on.endswith("STATUS\\G"),
          f"the wrapping altered what the program receives: {survived.stdout!r}")

    # A heredoc is a heredoc on the far end too.
    heredoc = shell(wrap_command("cat <<'EOF'\n# don't touch\nmax_connections = 100\nEOF"))
    check(failures, heredoc.stdout.splitlines()[-2:] == ["# don't touch", "max_connections = 100"],
          f"a heredoc body was mangled: {heredoc.stdout!r}")

    # An unterminated heredoc is only a warning to bash - the body ends at EOF
    # and the command does what the model meant - so the check must let it
    # through rather than inventing a stricter shell than the real one.
    MARKER.unlink(missing_ok=True)
    unterminated = shell(wrap_command(f"touch {MARKER.as_posix()}\ncat <<EOF\nno terminator here\n"))
    check(failures, MARKER.exists(), "a command bash accepts was refused")
    check(failures, unterminated.returncode == 0, f"expected exit 0, got {unterminated.returncode}")
    check(failures, "no terminator here" in unterminated.stdout, "the heredoc body was lost")
    check(failures, "delimited by end-of-file" in unterminated.stderr,
          "bash's warning should still reach the model")

    # Structure the guard does not parse at all is caught on the far end.
    MARKER.unlink(missing_ok=True)
    unclosed = shell(wrap_command(f"touch {MARKER.as_posix()}\nif systemctl is-active mysql; then echo up"))
    check(failures, not MARKER.exists(), "an unclosed if still half-applied the step")

    # A model that carries on past the end of its own command leaves its reasoning
    # inside it. Quotes stay balanced, so the guard has nothing to catch; the bare
    # `(noble)` is what the shell refuses. Verbatim from a real run.
    MARKER.unlink(missing_ok=True)
    prose = shell(wrap_command(
        f"grep -i -E '^(Suite|Codename):' /etc/os-release; touch {MARKER.as_posix()}; "
        "curl -fsSL http://repo.percona.com/pdmdb-8.0/apt/dists/noble/Release 2>&1 | tail -n 80"
        "It appears there is no PDMDB 8.0 package from Percona for Ubuntu 24.04 (noble). "
        "Need explore alternatives. Let's investigate."
    ))
    check(failures, not MARKER.exists(), "prose glued to a command still half-applied the step")
    check(failures, prose.returncode == 2, f"expected exit 2, got {prose.returncode}")
    check(failures, "no commentary" in prose.stderr,
          f"the harness did not say what to fix:\n{prose.stderr}")

    # A chat-template token the parser somehow let through is a bare `|` to the
    # shell. The step is refused whole rather than run up to the leak - this is
    # the second layer under strip_control_tokens, not a substitute for it.
    MARKER.unlink(missing_ok=True)
    leaked = shell(wrap_command(
        f"touch {MARKER.as_posix()} ; true<|close|>argument<|sep|><|close|>call<|sep|>"
    ))
    check(failures, not MARKER.exists(), "a leaked control token half-applied the step")
    check(failures, leaked.returncode == 2, f"expected exit 2, got {leaked.returncode}")
    check(failures, "syntax error" in leaked.stderr,
          f"bash's diagnosis of the leak was lost:\n{leaked.stderr}")

    MARKER.unlink(missing_ok=True)
    print("FAILURES" if failures else "all checks passed")
    for failure in failures:
        print(f"  FAIL {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
