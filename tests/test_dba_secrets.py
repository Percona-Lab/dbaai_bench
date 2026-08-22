"""Credentials that outlive one run.

The story this suite is about happened: a run installed MySQL and set the root
password to a generated value, and the next run against the same two servers could
not log in. It opened with a passwordless `mysql`, searched /etc, the error log and
root's shell history for a password the harness had never written there, and
finally reset root through skip-grant-tables - ninety-one steps, and the operator's
note of the old password was quietly wrong afterwards.

The fix is that a server keeps its own credentials, in a root-only file under
/etc/profile.d that later runs read back before the model is asked anything. So
this suite runs a task twice against the same droplet: the first run creates a
password, the second is a fresh process with an empty store, and what is checked is
that the second one logs in with the first one's value, is told the credential
exists, and never sees the value itself.

/etc/profile.d matters for a reason worth one assertion of its own: every command
already runs through `bash -lc`, so the file is sourced and $DBA_SECRET_MYSQL_ROOT
is in the environment of every step without the wrapper doing anything.
"""

from __future__ import annotations

import io
import json
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]  # the suites sit in tests/, the harness above it
# Ahead of anything installed on purpose: the point is to test this tree.
sys.path.insert(0, str(PROJECT))

from rich.console import Console

from do_dba.cli import Screen, adopt_secrets, show_outcome
from do_dba.fleet import Fleet, Target
from do_dba.secrets import (
    ENV_PREFIX,
    KEEPER_MODE,
    KEEPER_PATH,
    SecretStore,
    canonical,
    env_name,
    parse_keeper,
    read_keeper,
    write_keeper,
)
from do_dba.ssh import wrap_command
from do_dba.term import Glyphs
from fake_droplet import FakeDroplet
from test_dba_offline import build, check

RUNS = PROJECT / "_scratch" / "dba_secret_runs"

# Two runs against one server. The first sets a password, the second has to use it -
# and writes the name in a different case, which is how the recorded failure spelled
# it the second time round.
FIRST = [
    "THOUGHT: set the root password\nACTION: run\n"
    "COMMAND: mysql -e \"ALTER USER 'root'@'localhost' IDENTIFIED BY "
    "'{{DBA_SECRET:mysql_root}}'\"",
    "ACTION: done\nVERIFY: systemctl is-active ssh\n"
    "SUMMARY: the root password is {{DBA_SECRET:mysql_root}}",
]
SECOND = [
    "THOUGHT: the credential is listed as already in place, so log in with it\n"
    "ACTION: run\nCOMMAND: mysql -uroot -p'{{DBA_SECRET:MYSQL_ROOT}}' -e 'SELECT 1'",
    "ACTION: done\nVERIFY: systemctl is-active ssh\nSUMMARY: nothing needed doing",
]


class Recorder:
    """A droplet with the harness's own writes noted, so the mode can be checked."""

    def __init__(self, droplet: FakeDroplet):
        self.droplet = droplet
        self.writes: list[tuple[str, str]] = []

    def run(self, command: str, timeout: float = 300.0):
        return self.droplet.run(command, timeout=timeout)

    def write_file(self, path: str, content: str, mode: str = "0644"):
        self.writes.append((path, mode))
        return self.droplet.write_file(path, content, mode=mode)

    def __getattr__(self, name):  # state(), commands, packages - whatever a test asks
        return getattr(self.droplet, name)


def check_names(failures: list[str]) -> None:
    """One credential, however it is spelled.

    The recorded failure asked for mysql_root in one run and MYSQL_ROOT_PASSWORD in
    the next. The case and the punctuation are not a distinction worth keeping - a
    second value for a second spelling of the same name is a locked-out run - so
    they fold together, and the environment variable follows from the same rule.
    """
    check(failures, canonical("MYSQL_ROOT") == canonical("mysql-root")
          == canonical(" mysql.root ") == "mysql_root",
          f"the spellings do not fold together: {canonical('MYSQL_ROOT')}, {canonical('mysql-root')}")
    check(failures, canonical("mysql_root_password") != "mysql_root",
          "two genuinely different names were folded into one")
    check(failures, env_name("mysql-root") == f"{ENV_PREFIX}MYSQL_ROOT",
          f"the variable name is {env_name('mysql-root')}")
    check(failures, canonical("") == "secret" and canonical("!!") == "secret",
          "a name with nothing usable in it should still resolve to something")

    store = SecretStore()
    value = store.resolve("{{DBA_SECRET:mysql_root}}")
    check(failures, store.resolve("{{DBA_SECRET:MYSQL_ROOT}}") == value,
          "the same credential under another spelling got a second value")
    check(failures, len(value) > 16 and all(c not in value for c in "'\"\\ "),
          f"a generated value has to survive a shell round trip: {value!r}")
    check(failures, store.names == ["mysql_root"], f"the store holds {store.names}")
    check(failures, store.redact(f"password is {value}") == "password is {{DBA_SECRET:mysql_root}}",
          f"the value was not redacted: {store.redact(value)!r}")
    check(failures, store.unsaved, "a freshly generated credential is not marked unsaved")
    store.mark_saved()
    check(failures, not store.unsaved, "the store still reports unsaved work after saving")

    # The file the servers keep, and reading it back: whatever went in comes out.
    script = store.env_script()
    check(failures, f"export {ENV_PREFIX}MYSQL_ROOT=" in script,
          f"the exports are missing from the keeper file:\n{script}")
    check(failures, parse_keeper(script) == {"mysql_root": value},
          f"the keeper file does not round-trip: {parse_keeper(script)}")
    # An operator's own line, a comment, and a value with a quote in it: the parser
    # reads the file as a shell script, because that is what it is.
    hand_edited = (
        "# a comment\n"
        "export UNRELATED=keepme\n"
        f"export {ENV_PREFIX}PG_APP='awkward'\"'\"'value'\n"
        f"export {ENV_PREFIX}BROKEN='unbalanced\n"
        f"export {ENV_PREFIX}Repl_Pass=plain\n"
    )
    read = parse_keeper(hand_edited)
    check(failures, read == {"pg_app": "awkward'value", "repl_pass": "plain"},
          f"a hand-edited keeper file was not read as a shell script: {read}")


def check_store_adoption(failures: list[str]) -> None:
    """Taking in what a server already has, including when two disagree."""
    store = SecretStore()
    learned, clashed = store.adopt({"MYSQL_ROOT": "from-the-server", "empty": ""})
    check(failures, (learned, clashed) == (["mysql_root"], []),
          f"adopting reported {learned}, {clashed}")
    check(failures, store.resolve("{{DBA_SECRET:mysql_root}}") == "from-the-server",
          "an adopted credential was not used for the placeholder")
    check(failures, store.inherited == ["mysql_root"],
          f"the inherited names are {store.inherited}")
    check(failures, not store.unsaved,
          "reading a credential off a server is not a change to write back")

    # The second server disagrees. The first value read wins - changing it under a
    # step that has already run is worse - and the disagreement is reported.
    learned, clashed = store.adopt({"mysql_root": "different", "repl": "shared"})
    check(failures, (learned, clashed) == (["repl"], ["mysql_root"]),
          f"the clash was not reported: {learned}, {clashed}")
    check(failures, store.resolve("{{DBA_SECRET:mysql_root}}") == "from-the-server",
          "a credential already in use this run was overwritten")

    # A generated value is not "inherited": the model is only told about the ones
    # that already worked when the run started.
    store.resolve("{{DBA_SECRET:brand_new}}")
    check(failures, store.inherited == ["mysql_root", "repl"] and "brand_new" in store.names,
          f"a newly generated credential was reported as pre-existing: {store.inherited}")


def check_keeper_on_server(failures: list[str]) -> None:
    """The file goes onto the server root-only, and comes back off it."""
    runner = Recorder(FakeDroplet())
    check(failures, read_keeper(runner) == {},
          "a server that has never been touched reported credentials")

    store = SecretStore()
    value = store.resolve("{{DBA_SECRET:mysql_root}}")
    check(failures, write_keeper(runner, store) == "",
          "writing the credentials to the server failed")
    check(failures, runner.writes == [(KEEPER_PATH, KEEPER_MODE)],
          f"the keeper file was written as {runner.writes}, want {KEEPER_PATH} at {KEEPER_MODE}")
    check(failures, KEEPER_MODE == "0600" and KEEPER_PATH.startswith("/etc/profile.d/"),
          f"the keeper must be root-only and sourced by login shells: "
          f"{KEEPER_PATH} at {KEEPER_MODE}")
    check(failures, read_keeper(runner) == {"mysql_root": value},
          f"what was written did not come back: {read_keeper(runner)}")

    # Why /etc/profile.d and not somewhere the harness has to source itself: every
    # command already runs through a login shell, so the values are in the
    # environment of every step. If that ever changes, this assertion is the notice.
    check(failures, "bash -lc" in wrap_command("mysql -e 'SELECT 1'"),
          "commands no longer run through a login shell, so /etc/profile.d is not read")

    # An empty store writes nothing: a run that generated no credential should not
    # leave a file, or truncate the one that is there.
    fresh = Recorder(FakeDroplet())
    check(failures, write_keeper(fresh, SecretStore()) == "" and not fresh.writes,
          "an empty store still wrote to the server")


def screen_text(call) -> str:
    """What the operator reads, plain enough to compare: no colour, no wrapping."""
    buffer = io.StringIO()
    console = Console(file=buffer, no_color=True, soft_wrap=True, width=200, legacy_windows=False)
    call(Screen(console, Glyphs(fancy=False)))
    return buffer.getvalue()


def check_fleet_adoption(failures: list[str]) -> None:
    """Two servers, read before the model is asked anything.

    Names on screen and never values: the operator has the values in the run
    directory, and the point of the file is that nobody has to go and look. When the
    two machines disagree about a shared password that is a fact about the fleet, so
    it is said out loud rather than quietly resolved.
    """
    primary = FakeDroplet(hostname="primary")
    replica = FakeDroplet(hostname="replica", address="10.116.0.3", public="203.0.113.11")
    primary.files[KEEPER_PATH] = (
        f"export {ENV_PREFIX}MYSQL_ROOT=from-primary\n"
        f"export {ENV_PREFIX}REPL=agreed\n"
    )
    replica.files[KEEPER_PATH] = (
        f"export {ENV_PREFIX}MYSQL_ROOT=from-replica\n"      # the disagreement
        f"export {ENV_PREFIX}REPL=agreed\n"
        f"export {ENV_PREFIX}PG_APP=only-here\n"             # and one the primary lacks
    )
    fleet = Fleet([
        Target(name=droplet.hostname, host=droplet.address, runner=droplet, named=True)
        for droplet in (primary, replica)
    ])
    store = SecretStore()
    shown = screen_text(lambda screen: adopt_secrets(screen, fleet, store))

    check(failures, store.inherited == ["mysql_root", "pg_app", "repl"],
          f"the fleet's credentials were not all picked up: {store.inherited}")
    check(failures, store.resolve("{{DBA_SECRET:mysql_root}}") == "from-primary",
          "the value read first did not win the clash")
    for phrase in ("credentials already in place on primary: mysql_root, repl",
                   "credentials already in place on replica: pg_app",
                   "replica has a different value for mysql_root"):
        check(failures, phrase in shown, f"the operator was not told {phrase!r}:\n{shown}")
    for value in ("from-primary", "from-replica", "agreed", "only-here"):
        check(failures, value not in shown, f"a credential's value was printed: {value}")

    # And the closing line that says where they now live, since that is the whole
    # difference between the next run logging in and the next run resetting the
    # password - and it is plaintext on a server, which is worth saying every time.
    agent, record, _, _ = build(FakeDroplet(), SecretStore(), FIRST, directory=RUNS / "outcome")
    outcome = agent.run()
    report = record.write_report()
    closing = screen_text(lambda screen: show_outcome(
        screen, outcome, record, report, record.directory / "secrets.json", on_servers=True))
    check(failures, KEEPER_PATH in closing and "root only" in closing,
          f"the outcome does not say the credentials are on the servers:\n{closing}")
    quiet = screen_text(lambda screen: show_outcome(
        screen, outcome, record, report, record.directory / "secrets.json", on_servers=False))
    check(failures, KEEPER_PATH not in quiet,
          "--no-server-secrets still claims the credentials are on the servers")


def check_across_runs(failures: list[str]) -> None:
    """The recorded failure, in two runs against one droplet.

    Run one generates the root password. Run two is what a follow-up task is: a new
    process, an empty store, the same machine. It has to arrive at the same value
    without the operator passing anything, and without the value entering its own
    context.
    """
    droplet = FakeDroplet()
    runner = Recorder(droplet)

    first = SecretStore()
    agent, record, _, _ = build(runner, first, FIRST, directory=RUNS / "first")
    persisted: list[str] = []

    def persist() -> None:
        persisted.append(write_keeper(runner, first))
        first.mark_saved()

    agent.persist = persist
    outcome = agent.run()
    check(failures, outcome.status == "done", f"the first run ended {outcome.status}")
    password = first.resolve("{{DBA_SECRET:mysql_root}}")
    check(failures, any(password in command for command in droplet.commands),
          "the generated password never reached the server")
    # Written during the run, not after it: a run that dies mid-way has still
    # changed the password, and nobody could look that one up.
    check(failures, persisted == [""],
          f"the credential was not stored on the server as it was created: {persisted}")

    # ---- the follow-up run: a new process, so nothing but the server remembers
    second = SecretStore()
    learned, clashed = second.adopt(read_keeper(runner))
    check(failures, (learned, clashed) == (["mysql_root"], []),
          f"the follow-up run did not find the credential: {learned}, {clashed}")

    agent2, record2, client2, _ = build(runner, second, SECOND, directory=RUNS / "second")
    outcome2 = agent2.run()
    check(failures, outcome2.status == "done", f"the second run ended {outcome2.status}")
    check(failures, second.resolve("{{DBA_SECRET:mysql_root}}") == password,
          "the second run generated a new password instead of using the one in place")

    # The model was told the credential exists, by name and not by value, and the
    # spelling it used - a different case - resolved to the same secret.
    prompt = client2.prompts[0][0]["content"]
    for needle in ("CREDENTIALS ALREADY ON THESE SERVERS", "  mysql_root",
                   "{{DBA_SECRET:mysql_root}}", f"${env_name('mysql_root')}",
                   "Do not reset a password listed here"):
        check(failures, needle in prompt, f"the prompt is missing {needle!r}")
    check(failures, "already on\n   these servers is a password that works now" in prompt.lower(),
          "rule 4 does not say to use a credential that already works")
    check(failures, password not in json.dumps(client2.prompts),
          "the follow-up run leaked the real password into the model's context")
    check(failures, any(f"-p'{password}'" in command for command in droplet.commands),
          f"the adopted credential was not used on the server: {droplet.commands[-2:]}")

    # And nothing leaked into what an operator or a git repository ends up with.
    report = record2.write_report().read_text(encoding="utf-8")
    logged = (record2.directory / "transcript.jsonl").read_text(encoding="utf-8")
    check(failures, password not in report, "the follow-up report leaked the adopted password")
    check(failures, password not in logged, "the follow-up transcript leaked the adopted password")
    # Case-insensitively: what is logged is the step as the model wrote it, and it
    # wrote the name in caps. The value is what must not be there.
    check(failures, "{{dba_secret:mysql_root}}" in logged.lower(),
          "the adopted credential was not recorded as its placeholder")
    # Redaction covers an adopted value as well as a generated one - the run that
    # reads a password off a server did not choose it, and can still print it.
    check(failures, second.redact(f"Access denied for {password}").endswith(
        "{{DBA_SECRET:mysql_root}}"), "an adopted value is not redacted from output")

    # A run told not to touch the servers has no persist hook at all, so the third
    # run against this droplet would be back to guessing - which is the trade
    # --no-server-secrets makes, stated as a test so it stays deliberate.
    third = SecretStore()
    agent3, _, _, _ = build(runner, third, FIRST, directory=RUNS / "third")
    check(failures, agent3.persist is None,
          "the agent writes to the servers by default even without a hook")
    check(failures, "CREDENTIALS ALREADY ON" not in agent3.messages[0]["content"],
          "a run that adopted nothing still claims credentials are in place")
    print(f"first run: {record.status}   follow-up: {record2.status}   "
          f"credentials on the server: {sorted(read_keeper(runner))}")


def main() -> int:
    shutil.rmtree(RUNS, ignore_errors=True)
    failures: list[str] = []
    check_names(failures)
    check_store_adoption(failures)
    check_keeper_on_server(failures)
    check_fleet_adoption(failures)
    check_across_runs(failures)

    print()
    if failures:
        for failure in failures:
            print(f"  FAIL {failure}")
        print(f"\n{len(failures)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
