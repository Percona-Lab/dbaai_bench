"""Who answers the questions: the operator, --yes, or --mode unattended.

No network and no server. Four things stop a run to ask: an unknown host key, an
account without root, the plan gate, and every step the guard flags. This pins
which switch answers which, that a question answered by a switch still says on
screen what was asked and who answered it, and - the point of the whole thing -
that answering every question does not widen what the guard allows.
"""

from __future__ import annotations

import builtins
import contextlib
import io
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]  # the suites sit in tests/, the harness above it
# Ahead of anything installed on purpose: the point is to test this tree.
sys.path.insert(0, str(PROJECT))

from rich.console import Console

from do_dba.agent import MODE_AUTO, MODE_PLAN, MODE_STEP, MODE_UNATTENDED, Limits
from do_dba.cli import (
    Approver,
    Screen,
    answers_everything,
    answers_steps,
    ask_yes_no,
    build_parser,
    host_key_asker,
    show_settings,
    unanswerable_prompt,
)
from do_dba.cli import main as cli_main
from do_dba.inference import providers
from do_dba.inference.pricing import Price, PriceBook
from do_dba.secrets import SecretStore
from do_dba.term import Glyphs
from fake_droplet import FakeDroplet
# The offline suite's builder: a scripted model, the fake droplet and a run
# directory, so this one only has to vary who answers the questions.
from test_dba_offline import build, check

RUNS = PROJECT / "_scratch" / "dba_prompt_runs"
UNATTENDED = f"--mode {MODE_UNATTENDED}"


# --------------------------------------------------------------- test scaffolding


def buffered_screen() -> tuple[Screen, io.StringIO]:
    """A Screen whose output can be read back, with ASCII glyphs so it compares."""
    buffer = io.StringIO()
    console = Console(file=buffer, no_color=True, soft_wrap=True, width=200, legacy_windows=False)
    return Screen(console, Glyphs(fancy=False)), buffer


class FakeStdin:
    """Only isatty() is ever asked of stdin here."""

    def __init__(self, tty: bool):
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty


class Typed:
    """Stands in for a person at the keyboard; runs out of patience with EOF."""

    def __init__(self, *answers: str):
        self.answers = list(answers)
        self.asked: list[str] = []

    def __call__(self, prompt: str = "") -> str:
        self.asked.append(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)


def never(prompt: str = "") -> str:
    """An input() that must not be reached: a switch already said yes."""
    raise AssertionError(f"the operator was asked {prompt!r} even though a switch had answered")


@contextlib.contextmanager
def keyboard(input_fn, tty: bool = True):
    """Swap in a terminal and a scripted operator for the duration."""
    saved_stdin, saved_input = sys.stdin, builtins.input
    sys.stdin, builtins.input = FakeStdin(tty), input_fn
    try:
        yield
    finally:
        sys.stdin, builtins.input = saved_stdin, saved_input


class FakeKey:
    """The two things host_key_asker asks of a paramiko key."""

    def get_name(self) -> str:
        return "ssh-ed25519"

    def asbytes(self) -> bytes:
        return b"not a real key"


def parse(*argv: str):
    return build_parser().parse_args(["--host", "203.0.113.10", *argv])


def main() -> int:
    shutil.rmtree(RUNS, ignore_errors=True)
    failures: list[str] = []

    # ------------------------------------------------------------- ask_yes_no
    # A switch answers before the terminal is even consulted, and says so: the
    # session's own output is the record of what was asked, so an approval that
    # nobody gave has to be visible as one.
    screen, buffer = buffered_screen()
    with keyboard(never, tty=True):
        answered = ask_yes_no(screen, "proceed?", default=False, answered_by=UNATTENDED)
    printed = buffer.getvalue()
    check(failures, answered, "a switch-answered question came back no")
    check(failures, "proceed?" in printed and UNATTENDED in printed,
          f"the answered question did not name the question and the switch: {printed!r}")

    # Nothing answering and no terminal to ask on: no, not yes. stdin is /dev/null
    # for the model's sake, so a prompt here would hang a run forever.
    screen, buffer = buffered_screen()
    with keyboard(never, tty=False):
        piped = ask_yes_no(screen, "proceed?", default=True)
    check(failures, piped is False, "a question with no terminal must answer no, whatever the default")
    check(failures, "no terminal" in buffer.getvalue(), "the operator was not told why it said no")

    # And with a person there, the answer is theirs.
    screen, _ = buffered_screen()
    for typed, want in [("y", True), ("yes", True), ("n", False), ("no", False), ("N", False)]:
        with keyboard(Typed(typed)):
            got = ask_yes_no(screen, "run it?", default=False)
        check(failures, got is want, f"input {typed!r} was read as {got}")
    with keyboard(Typed("")):  # enter takes the default, either way
        check(failures, ask_yes_no(screen, "run it?", default=True) is True, "enter did not take a yes default")
    with keyboard(Typed("")):
        check(failures, ask_yes_no(screen, "run it?", default=False) is False, "enter did not take a no default")
    with keyboard(Typed("maybe", "sort of", "y")):  # rubbish is re-asked, not guessed at
        asker = builtins.input
        check(failures, ask_yes_no(screen, "run it?", default=False) is True, "a re-asked question lost its yes")
        check(failures, len(asker.asked) == 3, f"an unreadable answer was not re-asked: {asker.asked}")
    with keyboard(Typed()):  # ctrl-D
        check(failures, ask_yes_no(screen, "run it?", default=True) is False, "EOF must not be taken as yes")

    # --------------------------------------------------- which switch answers what
    check(failures, answers_everything(parse("--mode", MODE_UNATTENDED)) == UNATTENDED,
          "unattended does not answer everything")
    check(failures, answers_steps(parse("--mode", MODE_UNATTENDED)) == UNATTENDED,
          "unattended does not answer the steps")
    check(failures, answers_everything(parse("--yes")) == "",
          "--yes must not answer the host key question")
    check(failures, answers_steps(parse("--yes")) == "--yes", "--yes does not answer the steps")
    check(failures, answers_steps(parse("--mode", MODE_AUTO)) == "" and answers_everything(parse()) == "",
          "a plain run should answer nothing by itself")
    # Both given: the wider one names itself, since it is the one that would have
    # answered anyway. --accept-host-key is read at its own call site, not here.
    both = parse("--mode", MODE_UNATTENDED, "--yes", "--accept-host-key")
    check(failures, answers_steps(both) == UNATTENDED and answers_everything(both) == UNATTENDED,
          f"with both switches the report named {answers_steps(both)!r}")

    parser = build_parser()
    check(failures, parser.parse_args(["--host", "x"]).mode == MODE_PLAN, "the default mode moved")
    for mode in (MODE_PLAN, MODE_STEP, MODE_AUTO, MODE_UNATTENDED):
        check(failures, parser.parse_args(["--host", "x", "--mode", mode]).mode == mode,
              f"--mode {mode} was not accepted")
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            parser.parse_args(["--host", "x", "--mode", "nobody-home"])
            failures.append("an unknown mode was accepted")
        except SystemExit as exc:
            check(failures, exc.code == 2, f"a bad mode exited {exc.code}, want 2")

    # ------------------------------------- the one question a switch cannot answer
    # --ask-password wants a credential typed at the console, which is not a yes.
    # Better refused at the front door than left to wait for typing that is never
    # coming, or to authenticate with the empty string it would otherwise read.
    check(failures, unanswerable_prompt(parse("--mode", MODE_UNATTENDED)) == "",
          "an ordinary unattended run was refused")
    check(failures, unanswerable_prompt(parse("--ask-password")) == "",
          "--ask-password is fine when there is somebody to ask")
    refused = unanswerable_prompt(parse("--mode", MODE_UNATTENDED, "--ask-password"))
    check(failures, "--ask-password" in refused and "DBA_SSH_PASSWORD" in refused,
          f"an unanswerable password prompt was not refused with a way out: {refused!r}")
    refused = unanswerable_prompt(parse("--mode", MODE_UNATTENDED, "--ask-key-passphrase"))
    check(failures, "DBA_SSH_KEY_PASSPHRASE" in refused,
          f"the passphrase prompt was not refused with a way out: {refused!r}")
    with contextlib.redirect_stderr(io.StringIO()) as complaint:
        try:
            cli_main(["--host", "203.0.113.10", "--mode", MODE_UNATTENDED, "--ask-password",
                      "--task", "install MySQL"])
            failures.append("an unanswerable prompt was accepted by main()")
        except SystemExit as exc:
            check(failures, exc.code == 2, f"the refusal exited {exc.code}, want 2")
    check(failures, "--ask-password" in complaint.getvalue(),
          f"main() refused it without saying which flag: {complaint.getvalue()!r}")

    # ------------------------------------------------ the settings block says so
    # Before any work starts, in the same place the mode is printed: an unattended
    # run has nobody to notice a warning later on.
    limits = Limits(max_steps=40, command_timeout=300.0)
    prices = PriceBook(prices={"llama-4-maverick": Price(0.20, 0.696)}, warning=None)
    openrouter = providers.get("openrouter")
    settings: dict[str, str] = {}
    for label, argv in [("unattended", ["--mode", MODE_UNATTENDED]),
                        ("dry", ["--mode", MODE_UNATTENDED, "--dry-run"]),
                        ("auto", ["--mode", MODE_AUTO])]:
        screen, buffer = buffered_screen()
        show_settings(screen, "llama-4-maverick", openrouter, prices, parse(*argv), limits)
        settings[label] = buffer.getvalue()
    check(failures, "nothing will be asked" in settings["unattended"],
          f"an unattended run does not warn that nothing will be asked:\n{settings['unattended']}")
    check(failures, MODE_UNATTENDED in settings["dry"] and "nothing will be asked" not in settings["dry"],
          "a dry run has nothing to warn about - it executes nothing either way")
    check(failures, "nothing will be asked" not in settings["auto"],
          "a mode that does ask was warned about anyway")

    # ---------------------------------------------------------------- Approver
    screen, buffer = buffered_screen()
    approver = Approver(screen, UNATTENDED)
    with keyboard(never, tty=True):
        allowed = approver("run", "rm -rf /var/lib/mysql", "deletes a data directory")
    printed = buffer.getvalue()
    check(failures, allowed, "a flagged step was refused by the switch that should have approved it")
    check(failures, (approver.approved, approver.declined) == (1, 0),
          f"the counters read {approver.approved}/{approver.declined}, want 1/0")
    # The step, the guard's reason and who approved it: enough to reconstruct
    # afterwards what ran unwatched and why it was allowed to.
    for wanted in ("rm -rf /var/lib/mysql", "deletes a data directory", "approved automatically", UNATTENDED):
        check(failures, wanted in printed, f"the approval note is missing {wanted!r}: {printed!r}")

    screen, buffer = buffered_screen()
    asking = Approver(screen)
    with keyboard(Typed("n", "y")):
        first = asking("run", 'mysql -e "DROP DATABASE app"', "drops a database")
        second = asking("run", "rm -rf /var/lib/postgresql", "deletes a data directory")
    check(failures, first is False and second is True, "an asked approval did not follow the answer")
    check(failures, (asking.approved, asking.declined) == (1, 1),
          f"asked counters read {asking.approved}/{asking.declined}, want 1/1")
    check(failures, "approved automatically" not in buffer.getvalue(),
          "a step the operator answered was reported as automatic")

    # ----------------------------------------------------------- host key asker
    # The one answer here that cannot be taken back, so the fingerprint is printed
    # whether or not anyone is going to look at it.
    screen, buffer = buffered_screen()
    with keyboard(never, tty=True):
        trusted = host_key_asker(screen, UNATTENDED)("203.0.113.10", FakeKey())
    printed = buffer.getvalue()
    check(failures, trusted, "an unknown host key was not accepted by the switch")
    check(failures, "SHA256:" in printed and "ssh-ed25519" in printed,
          f"the fingerprint was not printed before trusting the key: {printed!r}")
    check(failures, UNATTENDED in printed, "the accepted key does not say what accepted it")

    screen, buffer = buffered_screen()
    with keyboard(Typed("y")):
        check(failures, host_key_asker(screen, "")("203.0.113.10", FakeKey()) is True,
              "the operator's yes on a host key was lost")
    with keyboard(never, tty=False):
        check(failures, host_key_asker(screen, "")("203.0.113.10", FakeKey()) is False,
              "an unknown host key must not be trusted with nobody to ask")

    # ------------------------------------- a whole unattended run against the droplet
    # The switch answers the guard's CONFIRM, and the run goes ahead without a
    # person. What it does not do is answer a BLOCK: `mysql` on its own opens a
    # REPL against a /dev/null stdin and hangs the run until the timeout, so it is
    # refused here exactly as it would be with someone watching.
    droplet = FakeDroplet()
    screen, buffer = buffered_screen()
    approver = Approver(screen, UNATTENDED)
    script = [
        "ACTION: run\nCOMMAND: apt-get update",
        "ACTION: run\nCOMMAND: apt-get install -y mysql-server",
        "THOUGHT: start from a clean data directory\nACTION: run\nCOMMAND: rm -rf /var/lib/mysql",
        "THOUGHT: check the server\nACTION: run\nCOMMAND: mysql",
        "ACTION: done\nVERIFY: systemctl is-active mysql\nSUMMARY: MySQL is installed and running",
    ]
    agent, record, _, events = build(
        droplet, SecretStore(), script, approve=approver,
        directory=RUNS / "unattended", mode=MODE_UNATTENDED,
    )
    with keyboard(never, tty=True):  # a terminal is there; nothing may use it
        outcome = agent.run()
    report_path = record.write_report()

    check(failures, outcome.status == "done", f"the unattended run ended {outcome.status}, want done")
    check(failures, any("rm -rf /var/lib/mysql" in c for c in droplet.commands),
          f"the flagged step did not run unattended: {droplet.commands}")
    check(failures, approver.approved == 1 and approver.declined == 0,
          f"the approver counted {approver.approved}/{approver.declined}, want 1/0")
    check(failures, not any(c.strip() == "mysql" for c in droplet.commands),
          "a blocked command ran because nobody was watching")
    blocked = [message for kind, message in events if kind == "blocked"]
    check(failures, len(blocked) == 1, f"expected the bare REPL to be blocked once, saw {blocked}")
    report = report_path.read_text(encoding="utf-8")
    check(failures, MODE_UNATTENDED in report, "the report does not say the run was unattended")

    # The same script with nobody answering: the flagged step is declined, and the
    # run has to find another way. Same guard, different answer - which is the
    # whole difference the switch makes.
    watched_droplet = FakeDroplet()
    screen, _ = buffered_screen()
    watched_approver = Approver(screen)
    watched_agent, _, _, _ = build(
        watched_droplet, SecretStore(), script, approve=watched_approver,
        directory=RUNS / "watched", mode=MODE_AUTO,
    )
    with keyboard(never, tty=False):  # no terminal: every flagged step is a no
        watched_agent.run()
    check(failures, not any("rm -rf" in c for c in watched_droplet.commands),
          f"a flagged step ran with nobody to approve it: {watched_droplet.commands}")
    check(failures, watched_approver.declined == 1 and watched_approver.approved == 0,
          f"the unattended approver counted {watched_approver.approved}/{watched_approver.declined}, want 0/1")

    print(f"{'FAILURES' if failures else 'all checks passed'}")
    for failure in failures:
        print(f"  FAIL {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
