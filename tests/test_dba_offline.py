"""The whole loop, driven by a scripted model against the fake droplet.

No network, no server: this checks the harness's own behaviour - guard blocks fed
back to the model, operator refusals honoured, secrets substituted outward and
redacted inward, verification run independently, and the report written.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]  # the suites sit in tests/, the harness above it
# Ahead of anything installed on purpose: the point is to test this tree.
sys.path.insert(0, str(PROJECT))

from do_dba.agent import (
    FILTER_EMPTY_NOTE,
    PIPE_CLOSED_NOTE,
    PIPE_CLOSED_VERIFY,
    RESULT_HINTS,
    SIGPIPE_EXIT,
    DBAAgent,
    Limits,
)
from do_dba.fleet import Fleet, Target
from do_dba.inference.client import Completion
from do_dba.inference.pricing import PriceBook
from do_dba.report import HostInfo, RunRecord
from do_dba.secrets import SecretStore
from do_dba.ssh import CommandResult
from fake_droplet import FakeDroplet

RUNS = PROJECT / "_scratch" / "dba_test_runs"
MODEL = "llama-4-maverick"  # priced, so the cost path is exercised


class ScriptedClient:
    """Returns canned replies in order, counting tokens like the real service.

    costs stands in for a gateway that reports what it charged for each reply, as
    OpenRouter does when asked: one entry per reply, None where it says nothing.
    Short lists run out and the rest report nothing, which is the mixed case.
    """

    def __init__(self, replies: list[str], costs: list[float | None] | None = None):
        self.replies = list(replies)
        self.costs = list(costs or [])
        self.calls = 0
        self.prompts: list[list[dict[str, str]]] = []

    def complete(self, *, model, messages, temperature=None, max_tokens=None, on_note=None) -> Completion:
        self.calls += 1
        self.prompts.append([dict(m) for m in messages])
        reply = self.replies.pop(0) if self.replies else "ACTION: abort\nSUMMARY: out of script"
        # A reply may be given as (text, finish_reason) to model one that the
        # service cut off at the output limit.
        text, finish = reply if isinstance(reply, tuple) else (reply, "stop")
        prompt_tokens = sum(len(m["content"]) for m in messages) // 4
        return Completion(
            text=text,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(text) // 4,
                "total_tokens": prompt_tokens + len(text) // 4,
                "cached_tokens": 0,
            },
            model=model,
            finish_reason=finish,
            cost=self.costs.pop(0) if self.costs else None,
            id=f"gen-{self.calls}",
        )


SCRIPT = [
    # 1. installs before updating: the droplet rejects it, as a real one would
    "THOUGHT: install the servers\nACTION: run\nCOMMAND: apt-get install -y mysql-server",
    # 2. reads the error and fixes it
    "THOUGHT: the package lists are empty, refresh them\nACTION: run\nCOMMAND: apt-get update",
    # 3. both servers at once
    "ACTION: run\nCOMMAND: apt-get install -y mysql-server postgresql postgresql-contrib",
    # 4. risky: the operator will decline this one
    "THOUGHT: start from a clean data directory\nACTION: run\nCOMMAND: rm -rf /var/lib/mysql",
    # 5. blocked outright: bare mysql would hang
    "THOUGHT: check the server\nACTION: run\nCOMMAND: mysql",
    # 6. a config file the guard flags on content, which the operator allows
    "ACTION: write_file\nPATH: /etc/mysql/mysql.conf.d/zz-harness.cnf\nMODE: 0644\nCONTENT_BEGIN\n"
    "[mysqld]\nbind-address = 0.0.0.0\nCONTENT_END",
    # 7. ordinary SQL
    "ACTION: run\nCOMMAND: mysql -e \"CREATE DATABASE app CHARACTER SET utf8mb4\"",
    # 8. a credential the model never sees
    "ACTION: run\nCOMMAND: mysql -e \"CREATE USER 'app'@'localhost' IDENTIFIED BY "
    "'{{DBA_SECRET:mysql_app}}'; GRANT ALL ON app.* TO 'app'@'localhost'\"",
    # 9. a reply in the wrong format: the harness must correct it, not die
    "I think the next thing to do is check that PostgreSQL accepts connections.",
    # 10. the corrected step
    "ACTION: run\nCOMMAND: sudo -u postgres psql -c \"SELECT version()\"",
    # 11. finished
    "ACTION: done\nVERIFY: systemctl is-active mysql\nVERIFY: systemctl is-active postgresql\n"
    "SUMMARY: MySQL 8.0 and PostgreSQL 16 are installed, enabled and listening on localhost. "
    "The app database and its user exist; the password is {{DBA_SECRET:mysql_app}}.",
]


class SigpipeDroplet(FakeDroplet):
    """A server that answers like a real one when the reader stops reading.

    The fake's pipeline handling gives back the reader's own exit code, and `head`
    always succeeds. A real server runs the harness's `set -o pipefail`, so what comes
    back is the writer being killed by the closed pipe: 141, for a command that did
    what was asked. Nine steps across the recorded runs came back that way.

    Modelled here rather than inside FakeDroplet because whether the writer is still
    writing when the reader closes depends on how much it had left and on the pipe
    buffer - `printf 'a\\nb\\n' | head -1` really is 0 - and a fake that always said
    141 would be wrong more often than it was right.
    """

    def run(self, command: str, timeout: float = 300.0) -> CommandResult:
        result = super().run(command, timeout=timeout)
        if "| head" in command and result.exit_code == 0:
            return replace(result, exit_code=SIGPIPE_EXIT)
        return result


class FilteringDroplet(FakeDroplet):
    """A server whose work succeeds and whose filter then eats the evidence.

    Verbatim in shape from a recorded run: `mysql -e "DROP TABLE ...; DROP DATABASE
    ..." 2>&1 | grep -v Warning` dropped both, printed only the password warning, and
    the filter removed it - leaving exit 1 and silence, which the next step read as the
    work not having happened. The fake pipes to nothing, so the case is modelled here.
    """

    def run(self, command: str, timeout: float = 300.0) -> CommandResult:
        result = super().run(command, timeout=timeout)
        if "| grep -v Warning" in command:
            return replace(result, exit_code=1, stdout="", stderr="")
        return result


class RefusingDroplet(FakeDroplet):
    """A 9.7 server: its client refuses `\\G`, its parser refuses `SHOW MASTER STATUS`.

    Both branches are verbatim in shape from recorded runs, and both answer on stdout,
    behind the step's own `2>&1` - which the rules ask for - where the hint table used
    not to look. `\\G` is refused in eight recorded steps and `MASTER STATUS` in eight
    more, and in each case the model went looking for the difference in its own
    quoting. The `${` branch keeps a diagnostic on stderr, so widening the table cannot
    quietly break the case it was written for.
    """

    def run(self, command: str, timeout: float = 300.0) -> CommandResult:
        result = super().run(command, timeout=timeout)
        if "\\G" in command:
            return replace(result, exit_code=1,
                           stdout="ERROR at line 1: Unknown command '\\G'.\n", stderr="")
        if "MASTER STATUS" in command:
            return replace(result, exit_code=1, stderr="", stdout=(
                "ERROR 1064 (42000) at line 1: You have an error in your SQL syntax; "
                "check the manual that corresponds to your MySQL server version for the "
                "right syntax to use near 'MASTER STATUS' at line 1\n"))
        if "${" in command:
            return replace(result, exit_code=1, stdout="",
                           stderr="bash: line 1: ${x:-'y'}: bad substitution\n")
        return result


def fleet_of(*droplets) -> Fleet:
    """A surveyed fleet around one droplet, or several named primary/replica/...

    Named as if the operator had named them (named=True), because these cases are
    about what the steps and the report carry once the names exist. The other path -
    a bare list of addresses the harness labels node1, node2, ... - is the fleet
    suite's business.
    """
    if len(droplets) == 1:
        fleet = Fleet.of(droplets[0], name="fake.droplet")
    else:
        fleet = Fleet([
            Target(name=droplet.hostname, host=droplet.address, runner=droplet, named=True)
            for droplet in droplets
        ])
    fleet.survey()
    return fleet


def build(droplet, store, replies, dry_run=False, approve=None, directory=None, mode="auto",
          fleet=None, task="Install MySQL and PostgreSQL", costs=None):
    fleet = fleet if fleet is not None else fleet_of(droplet)
    record = RunRecord(
        directory=directory or (RUNS / ("dry" if dry_run else "main")),
        task=task,
        hosts=[HostInfo(name=name, label=label, facts=facts)
               for name, label, facts in fleet.host_lines()],
        model=MODEL,
        mode=mode,
        dry_run=dry_run,
        redact=store.redact,
    )
    client = ScriptedClient(replies, costs=costs)
    events: list[tuple[str, str]] = []
    agent = DBAAgent(
        client=client,
        model=MODEL,
        fleet=fleet,
        task=task,
        record=record,
        store=store,
        prices=PriceBook(),
        emit=lambda kind, message: events.append((kind, message)),
        approve=approve or (lambda action, detail, reason: True),
        mode=mode,
        dry_run=dry_run,
        limits=Limits(max_steps=20, command_timeout=300.0),
    )
    return agent, record, client, events


def check(failures: list[str], condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    shutil.rmtree(RUNS, ignore_errors=True)
    failures: list[str] = []

    # ---------------------------------------------------------- the main run
    droplet = FakeDroplet()
    store = SecretStore()
    declined: list[str] = []

    def approve(action, detail, reason):
        if "rm -rf" in detail:
            declined.append(detail)
            return False
        return True

    agent, record, client, events = build(droplet, store, SCRIPT, approve=approve)
    outcome = agent.run()
    secrets_path = store.save(record.directory / "secrets.json")
    report = record.write_report()

    print(f"status: {outcome.status}  steps: {outcome.steps} proposed, {outcome.executed} executed")
    print(f"model calls: {client.calls}   cost: ${outcome.cost:.6f}")
    print("\ndroplet state after the run:")
    print("  " + droplet.state().replace("\n", "\n  "))

    check(failures, outcome.status == "done", f"status is {outcome.status}, want done")
    check(failures, outcome.ok, "outcome should be ok")

    # the work actually happened
    check(failures, "mysql-server" in droplet.packages, "mysql-server was not installed")
    check(failures, "postgresql" in droplet.packages, "postgresql was not installed")
    check(failures, droplet.services.get("mysql") is True, "mysql is not running")
    check(failures, droplet.services.get("postgresql") is True, "postgresql is not running")
    check(failures, "app" in droplet.mysql.databases, "the app database was not created")
    check(failures, "app@localhost" in droplet.mysql.users, "the app user was not created")

    # the failed first step was retried rather than repeated
    installs = [c for c in droplet.commands if c.startswith("apt-get install")]
    check(failures, len(installs) == 2, f"expected 2 install attempts, saw {len(installs)}")

    # refusals and blocks never reached the server
    check(failures, declined == ["rm -rf /var/lib/mysql"], f"the rm should have been offered once, got {declined}")
    check(failures, not any(c.strip() == "rm -rf /var/lib/mysql" for c in droplet.commands),
          "a declined command was executed anyway")
    check(failures, not any(c.strip() == "mysql" for c in droplet.commands),
          "a blocked command was executed anyway")
    blocked = [message for kind, message in events if kind == "blocked"]
    check(failures, len(blocked) == 1, f"expected one block, saw {blocked}")

    # the block was explained back to the model
    last_prompt = client.prompts[-1]
    fed_back = [m["content"] for m in last_prompt if "BLOCKED BY THE SAFETY GUARD" in m["content"]]
    check(failures, len(fed_back) == 1, "the guard's reason was not fed back to the model")
    check(failures, any("DECLINED BY THE OPERATOR" in m["content"] for m in last_prompt),
          "the operator's refusal was not fed back to the model")
    check(failures, any("could not be read" in m["content"] for m in last_prompt),
          "the protocol error was not fed back to the model")

    # the operator-approved file was written, contents intact
    written = droplet.files.get("/etc/mysql/mysql.conf.d/zz-harness.cnf", "")
    check(failures, "bind-address = 0.0.0.0" in written, "the approved file was not written")

    # secrets: the real value reached the server, and nothing else
    password = store.resolve("{{DBA_SECRET:mysql_app}}")
    check(failures, len(password) > 16, f"generated password looks weak: {password!r}")
    check(failures, any(password in c for c in droplet.commands),
          "the real password never reached the server")
    report_text = report.read_text(encoding="utf-8")
    transcript = (record.directory / "transcript.jsonl").read_text(encoding="utf-8")
    check(failures, password not in report_text, "the report leaked the real password")
    check(failures, password not in transcript, "the transcript leaked the real password")
    check(failures, "{{DBA_SECRET:mysql_app}}" in report_text, "the report lost the placeholder")
    model_saw = "\n".join(m["content"] for m in last_prompt)
    check(failures, password not in model_saw, "the model was shown the real password")
    check(failures, secrets_path is not None and password in secrets_path.read_text(encoding="utf-8"),
          "secrets.json does not hold the password")

    # verification ran on the harness's terms
    commands_verified = [check_.command for check_ in record.verifications]
    check(failures, any("list-units" in c for c in commands_verified),
          "the harness did not run its own service check")
    check(failures, "systemctl is-active mysql" in commands_verified,
          "the model's verify command was not run")
    check(failures, all(v.exit_code == 0 for v in record.verifications if "is-active" in v.command),
          "a verification reported failure")

    # accounting and the report
    check(failures, record.prompt_tokens > 0 and record.completion_tokens > 0, "no tokens were counted")
    check(failures, outcome.cost > 0 and outcome.cost_complete, "cost was not accounted")
    check(failures, "## Steps" in report_text and "## Independent verification" in report_text,
          "the report is missing sections")
    check(failures, "blocked" in report_text, "the report does not mention the blocked step")
    logged = [json.loads(line) for line in transcript.splitlines()]
    kinds = {event["kind"] for event in logged}
    check(failures, {"run_started", "step", "verification", "run_finished", "protocol_error"} <= kinds,
          f"transcript is missing event kinds, has {sorted(kinds)}")
    # The reply, not just the complaint: without it there is no telling afterwards
    # whether the model wrote prose or the parser missed a step it should have found.
    errors = [event for event in logged if event["kind"] == "protocol_error"]
    check(failures, errors and all(event.get("reply") for event in errors),
          f"a protocol error was recorded without the reply that caused it: {errors}")

    # A failure the model has been seen not to work out for itself gets a hint
    # appended to the observation. Keyed on bash's own wording, so an unrelated
    # failure is handed back exactly as it was.
    def observed(stderr: str) -> str:
        return agent._format_result(1, CommandResult(
            command="x", exit_code=1, stdout="", stderr=stderr, duration=0.1), agent.fleet.only)

    hinted = observed("bash: line 1: sed 's/[${}]//g': bad substitution")
    check(failures, "single quotes inside double quotes" in hinted,
          f"the bad-substitution hint was not offered:\n{hinted}")
    check(failures, "bad substitution" in hinted, "the shell's own diagnosis was replaced")
    plain = observed("mysql: unknown variable 'foo'")
    check(failures, "single quotes inside double quotes" not in plain,
          f"a hint was attached to an unrelated failure:\n{plain}")

    # ------------------------------------------- what the gateway says it cost
    # A cost the gateway reports is what the account was charged: it knows about
    # cached prompt tokens and about which upstream provider served the reply, and
    # a rate table knows neither. So it wins, and the run says which figure it is
    # showing - a bill and an estimate added together silently is a number that
    # cannot be reconciled with anything.
    billed_script = [
        "ACTION: run\nCOMMAND: apt-get update",
        "ACTION: done\nVERIFY: systemctl is-active ssh\nSUMMARY: the package lists are current",
    ]
    reported = [0.021, 0.0004]
    billed_agent, billed_record, _, _ = build(
        FakeDroplet(), SecretStore(), list(billed_script), costs=list(reported),
        directory=RUNS / "billed",
    )
    billed = billed_agent.run()
    billed_report = billed_record.write_report().read_text(encoding="utf-8")
    estimate = PriceBook().cost(MODEL, billed_record.prompt_tokens, billed_record.completion_tokens)
    check(failures, abs(billed.cost - sum(reported)) < 1e-12,
          f"the reported cost was not the one kept: {billed.cost} for {sum(reported)}")
    check(failures, estimate is not None and abs(estimate - billed.cost) > 1e-9,
          "the rate table happens to agree, so this proves nothing - change the reported figures")
    check(failures, (billed_record.billed_replies, billed_record.estimated_replies) == (2, 0),
          f"the replies were counted as {billed_record.billed_replies} billed, "
          f"{billed_record.estimated_replies} estimated")
    check(failures, billed_record.cost_note == " (billed by the gateway)",
          f"the cost note reads {billed_record.cost_note!r}")
    check(failures, "billed by the gateway" in billed_report,
          "the report does not say the cost is the billed one")

    # One line per reply, with the gateway's own id for it: that is what makes the
    # total checkable against the gateway's activity page rather than believed.
    logged_usage = [json.loads(line) for line in
                    (billed_record.directory / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
                    if '"usage"' in line]
    check(failures, len(logged_usage) == 2, f"{len(logged_usage)} usage events for two replies")
    check(failures, [event["reply"] for event in logged_usage] == ["gen-1", "gen-2"],
          f"the gateway's reply ids were not recorded: {[e.get('reply') for e in logged_usage]}")
    check(failures, all(event["cost_source"] == "gateway" for event in logged_usage),
          f"a billed reply was logged as an estimate: {logged_usage}")
    check(failures, abs(sum(event["cost"] for event in logged_usage) - sum(reported)) < 1e-9,
          "the logged per-reply costs do not add up to the total")

    # Half and half, which is what a gateway that reports only sometimes gives.
    mixed_agent, mixed_record, _, _ = build(
        FakeDroplet(), SecretStore(), list(billed_script), costs=[reported[0]],
        directory=RUNS / "billed-mixed",
    )
    mixed = mixed_agent.run()
    mixed_report = mixed_record.write_report().read_text(encoding="utf-8")
    check(failures, (mixed_record.billed_replies, mixed_record.estimated_replies) == (1, 1),
          f"a mixed run counted {mixed_record.billed_replies} billed, "
          f"{mixed_record.estimated_replies} estimated")
    check(failures, mixed.cost > reported[0],
          "the estimated reply was left out of the total instead of being added and labelled")
    check(failures, mixed_record.cost_note == " (part billed by the gateway, part from published rates)"
          and "part billed by the gateway" in mixed_report,
          f"a mixed run's cost note reads {mixed_record.cost_note!r}")

    # And with no gateway figure at all, the table is the answer and says so.
    check(failures, record.cost_note == " (estimated from published rates)",
          f"the main run's cost note reads {record.cost_note!r}")
    check(failures, "estimated from published rates" in report_text,
          "the report does not say the cost is an estimate when it is one")

    # ------------------------------------------------------------- dry run
    dry_droplet = FakeDroplet()
    dry_store = SecretStore()
    dry_agent, dry_record, _, _ = build(dry_droplet, dry_store, SCRIPT, dry_run=True)
    probes = len(dry_droplet.commands)  # fact gathering is read-only and still happens
    dry_outcome = dry_agent.run()
    dry_record.write_report()
    check(failures, len(dry_droplet.commands) == probes,
          f"dry run touched the server: {dry_droplet.commands[probes:]}")
    check(failures, not dry_droplet.packages, "dry run installed something")
    check(failures, dry_outcome.status == "done", f"dry run status is {dry_outcome.status}")

    # --------------------------------------------------------- stuck on blocks
    stuck_droplet = FakeDroplet()
    stuck_agent, stuck_record, _, _ = build(
        stuck_droplet, SecretStore(),
        ["ACTION: run\nCOMMAND: rm -rf /"] * 5,
        directory=RUNS / "stuck",
    )
    stuck = stuck_agent.run()
    check(failures, stuck.status == "stuck", f"repeated blocks gave {stuck.status}, want stuck")
    check(failures, stuck.executed == 0, "a blocked command ran")

    # --------------------------------- a reader that closes the pipe (exit 141)
    # `apt-cache search x | head -30` did its job and came back 141, nine times over
    # six runs. Read as a failure it costs a step at best; a run of them in a row would
    # have ended a healthy run as `failed`, and rule 7 tells the model to change a
    # command that failed - so it would have rewritten one that was fine.
    #
    # Counted off the real limit rather than a number written here, so that raising it
    # cannot quietly turn these two cases into a pair that proves nothing: the first
    # run has to survive a whole limit's worth of closed pipes, the second has to stop
    # on a whole limit's worth of real failures with a closed pipe sitting in them.
    allowed = Limits().max_consecutive_failures
    pipe_droplet = SigpipeDroplet()
    pipe_agent, pipe_record, _, pipe_events = build(
        pipe_droplet, SecretStore(),
        ["ACTION: run\nCOMMAND: cat /etc/os-release | head -n 2"] * allowed
        + ["ACTION: done\nVERIFY: systemctl is-active ssh\nSUMMARY: had a look at the release"],
        directory=RUNS / "closed-pipe",
    )
    pipe = pipe_agent.run()
    pipe_record.write_report()
    check(failures, pipe.status == "done",
          f"{allowed} closed pipes in a row gave {pipe.status}, want done")
    truncated = [s for s in pipe_record.steps if s.exit_code == SIGPIPE_EXIT]
    check(failures, len(truncated) == allowed,
          f"{len(truncated)} steps came back 141, want {allowed}")
    check(failures, all("reader closed the pipe" in (s.note or "") for s in truncated),
          f"the record does not say what 141 was: {[s.note for s in truncated]}")
    check(failures, all(kind != "fail" for kind, _ in pipe_events),
          f"a step that worked was shown to the operator as a failure: {pipe_events}")
    check(failures, any("reader closed the pipe" in message for kind, message in pipe_events
                        if kind == "ok"),
          f"the operator saw exit 141 with nothing to explain it: {pipe_events}")
    told = [m["content"] for m in pipe_agent.messages if PIPE_CLOSED_NOTE[:40] in m["content"]]
    check(failures, len(told) == allowed,
          f"the model was told about the pipe {len(told)} times, want {allowed}")
    # The most recent one, not the first: older observations are trimmed to 400
    # characters, which keeps the note - it sits just under the exit code - and drops
    # the output below it. So this is also where that ordering gets checked.
    check(failures, told and 'PRETTY_NAME="Ubuntu 24.04.1 LTS"' in told[-1],
          "the output the reader did get was not handed back with it")

    # It must not launder a thrashing run either: a closed pipe does not clear the
    # count, so real failures around one still stop the run.
    half = allowed // 2
    thrash_agent, _, _, _ = build(
        SigpipeDroplet(), SecretStore(),
        ["ACTION: run\nCOMMAND: systemctl is-active mysql"] * half
        + ["ACTION: run\nCOMMAND: cat /etc/os-release | head -n 2"]
        + ["ACTION: run\nCOMMAND: systemctl is-active mysql"] * (allowed - half)
        + ["ACTION: done\nVERIFY: systemctl is-active ssh\nSUMMARY: nothing works"],
        directory=RUNS / "closed-pipe-thrash",
    )
    thrash = thrash_agent.run()
    check(failures, thrash.status == "failed",
          f"{allowed} real failures around a closed pipe gave {thrash.status}, want failed")
    check(failures, thrash.executed == allowed + 1,
          f"the run stopped after {thrash.executed} steps, want {allowed + 1} "
          f"(the pipe step is not a failure)")

    # A VERIFY is a different question: 141 there means the check cannot say whether
    # the work is present, so it is neither passed nor counted as the work failing -
    # it goes back to be rewritten.
    verify_agent, verify_record, _, _ = build(
        SigpipeDroplet(), SecretStore(),
        ["ACTION: done\nVERIFY: systemctl is-active ssh | head -n 1\nSUMMARY: ssh is up"] * 4,
        directory=RUNS / "closed-pipe-verify",
    )
    verified = verify_agent.run()
    verify_record.write_report()
    check(failures, verified.status == "unverified",
          f"an unusable check gave {verified.status}, want unverified")
    handed_back = [m["content"] for m in verify_agent.messages
                   if PIPE_CLOSED_VERIFY[:40] in m["content"]]
    check(failures, handed_back, "the model was not told why its check could not be read")
    check(failures, any("unusable check" in (v.output or "") and v.exit_code == SIGPIPE_EXIT
                        for v in verify_record.verifications),
          f"the report does not mark the check unusable: "
          f"{[(v.command, v.exit_code, (v.output or '')[:40]) for v in verify_record.verifications]}")

    # --------------------------------- a filter that ate the evidence (exit 1)
    # The other half of pipefail, and the harder half: exit 1 is both what a grep with
    # no match reports and what the command itself would report, so this one cannot be
    # excused - only explained. What the recorded runs show is the model deciding the
    # work did not happen (a dropped database that had been dropped, a reconfigured
    # replica it then undid), so the explanation is what has to be there.
    filter_agent, filter_record, _, filter_events = build(
        FilteringDroplet(), SecretStore(),
        ["ACTION: run\nCOMMAND: mysql -e \"DROP DATABASE clustertest;\" 2>&1 | grep -v Warning",
         "ACTION: done\nVERIFY: systemctl is-active ssh\nSUMMARY: the test database is gone"],
        directory=RUNS / "empty-filter",
    )
    filtered = filter_agent.run()
    filter_record.write_report()
    check(failures, filtered.status == "done", f"the filtered run ended {filtered.status}")
    step = filter_record.steps[0]
    check(failures, step.exit_code == 1 and "nothing matched the filter" in (step.note or ""),
          f"the record does not say what the 1 was: exit {step.exit_code}, note {step.note!r}")
    told = [m["content"] for m in filter_agent.messages if FILTER_EMPTY_NOTE[:40] in m["content"]]
    check(failures, len(told) == 1, f"the model was told about the filter {len(told)} times, want 1")
    check(failures, told and "(no output)" in told[0],
          "the note replaced the silence instead of explaining it")
    # A failure it still is - the harness does not know the work happened, and saying
    # otherwise would be the same mistake in the other direction.
    check(failures, any(kind == "fail" and "nothing matched the filter" in message
                        for kind, message in filter_events),
          f"the operator was not shown the step as an unexplained failure: {filter_events}")
    thrash_filter, _, _, _ = build(
        FilteringDroplet(), SecretStore(),
        ["ACTION: run\nCOMMAND: mysql -e \"SELECT 1;\" 2>&1 | grep -v Warning"] * allowed
        + ["ACTION: done\nVERIFY: systemctl is-active ssh\nSUMMARY: nothing to report"],
        directory=RUNS / "empty-filter-thrash",
    )
    check(failures, thrash_filter.run().status == "failed",
          f"{allowed} steps in a row that said nothing at all did not stop the run")

    # And the prompt says not to write the command that way in the first place.
    check(failures, "Do not end a step" in filter_agent.messages[0]["content"]
          and "|| true" in filter_agent.messages[0]["content"],
          "the rules do not warn against ending a step with a filter")

    # ------------------------------------------- a hint on stdout, not stderr
    # 57% of the executed steps in the recorded runs redirect stderr into stdout and only
    # 76 of 935 produce any stderr at all, so a hint table keyed on stderr was blind on
    # nine steps in ten. The `\G` refusal is what proved it: eight times recorded, never
    # once on stderr, and one run spent five steps re-quoting a command that no quoting
    # would fix. `SHOW MASTER STATUS` is the same shape - renamed in 8.4, and the model
    # reads the 1064 as its own typo - so both are checked here on one run.
    hints = dict(RESULT_HINTS)
    check(failures, "Unknown command '\\G'" in hints,
          "the hint table says nothing about a client that refused `\\G`")
    check(failures, "near 'MASTER STATUS'" in hints,
          "the hint table says nothing about the statement 8.4 renamed")
    refusal = hints.get("Unknown command '\\G'", "\0 no such hint")
    renamed = hints.get("near 'MASTER STATUS'", "\0 no such hint")
    hint_agent, hint_record, _, _ = build(
        RefusingDroplet(), SecretStore(),
        ['ACTION: run\nCOMMAND: mysql -e "SHOW REPLICA STATUS\\G" 2>&1 | tail -n 40',
         'ACTION: run\nCOMMAND: mysql -e "SHOW MASTER STATUS;" 2>&1 | tail -n 30',
         "ACTION: run\nCOMMAND: echo \"${x:-'y'}\"",
         "ACTION: done\nVERIFY: systemctl is-active ssh\nSUMMARY: replication is up"],
        directory=RUNS / "hint-on-stdout",
    )
    hint_agent.run()
    hint_record.write_report()
    said = [m["content"] for m in hint_agent.messages if m["role"] == "user"]
    on_stdout = [text for text in said if refusal in text]
    check(failures, len(on_stdout) == 1,
          f"a diagnostic on stdout got its hint {len(on_stdout)} times, want 1")
    check(failures, on_stdout and "stderr" not in on_stdout[0],
          "the hint arrived on a step that had stderr, so stdout was not what matched")
    check(failures, on_stdout and "--vertical" in on_stdout[0],
          "the hint does not say what to write instead")
    # `-N` is the other half of the advice and it silently removes the field names, which
    # cost one recorded run two failed checks and five steps. Recommend it, say that.
    check(failures, "-N" not in refusal or "field names" in refusal,
          "the hint offers `-N` without saying it drops the field names")
    # The rename, on its own step: the model is told the server parsed what it sent, so
    # the semicolon and the quotes are not where the mistake is.
    said_renamed = [text for text in said if renamed in text]
    check(failures, len(said_renamed) == 1,
          f"the renamed statement got its hint {len(said_renamed)} times, want 1")
    check(failures, said_renamed and "SHOW BINARY LOG STATUS" in said_renamed[0],
          "the hint does not name the statement that replaced it")
    check(failures, said_renamed and refusal not in said_renamed[0],
          "the `\\G` hint was appended to the step about the renamed statement")
    # The case the table was written for still works - this widened where it looks, not
    # what it looks for - and neither hint is offered to a step it has nothing to do with.
    check(failures, any(hints["bad substitution"] in text for text in said),
          "widening the match lost the hint that only ever appears on stderr")
    check(failures, not any(hints["bad substitution"] in text for text in on_stdout),
          "both hints were appended to one step")

    # ------------------------------------------- a "done" that does not hold up
    early_droplet = FakeDroplet()
    early_agent, early_record, _, early_events = build(
        early_droplet, SecretStore(),
        [
            "ACTION: done\nVERIFY: systemctl is-active mysql\nSUMMARY: MySQL is installed and running",
            "ACTION: run\nCOMMAND: apt-get update",
            "ACTION: run\nCOMMAND: apt-get install -y mysql-server",
            "ACTION: done\nVERIFY: systemctl is-active mysql\nSUMMARY: MySQL is installed and running",
        ],
        directory=RUNS / "unverified-then-fixed",
    )
    early = early_agent.run()
    early_record.write_report()
    check(failures, early.status == "done", f"a corrected done gave {early.status}, want done")
    check(failures, "mysql-server" in early_droplet.packages,
          "the rejected done did not lead to the work being done")
    check(failures, all(v.exit_code == 0 for v in early_record.verifications),
          "the final report kept a failing check")

    liar_droplet = FakeDroplet()
    liar_agent, liar_record, _, _ = build(
        liar_droplet, SecretStore(),
        ["ACTION: done\nVERIFY: systemctl is-active mysql\nSUMMARY: MySQL is running"] * 5,
        directory=RUNS / "unverified",
    )
    liar = liar_agent.run()
    liar_record.write_report()
    check(failures, liar.status == "unverified", f"an unfixed done gave {liar.status}, want unverified")
    check(failures, not liar.ok, "an unverified run must not report ok")

    # ------------------------------- a "done" with nothing to prove it
    # The standing checks cannot fail on their own, so without this rule a bare
    # claim would be reported as done and the report would assert work never done.
    bare_droplet = FakeDroplet()
    bare_agent, bare_record, _, bare_events = build(
        bare_droplet, SecretStore(),
        [
            "ACTION: done\nSUMMARY: MySQL and PostgreSQL are installed with the app database",
            "ACTION: run\nCOMMAND: apt-get update",
            "ACTION: done\nVERIFY: systemctl is-active mysql\nSUMMARY: gave up and checked",
            "ACTION: done\nVERIFY: systemctl is-active mysql\nSUMMARY: still claiming it",
        ],
        directory=RUNS / "unproven",
    )
    bare = bare_agent.run()
    bare_record.write_report()
    check(failures, bare.status == "unverified",
          f"a done with no VERIFY, then a failing one, gave {bare.status}, want unverified")
    asked = [m for kind, m in bare_events if "no check" in m or "carried no check" in m]
    check(failures, len(asked) == 1, f"the missing check was not reported once: {asked}")
    handed_back = [m["content"] for m in bare_agent.messages
                   if "WITHOUT A SINGLE VERIFY LINE" in m["content"]]
    check(failures, len(handed_back) == 1, "the model was not asked for proof")

    proven_droplet = FakeDroplet()
    proven_agent, proven_record, _, _ = build(
        proven_droplet, SecretStore(),
        [
            "ACTION: done\nSUMMARY: nothing to do, it is all there",
            "ACTION: done\nVERIFY: systemctl is-active ssh\nSUMMARY: ssh is up, which is all I claimed",
        ],
        directory=RUNS / "proven-on-retry",
    )
    proven = proven_agent.run()
    check(failures, proven.status == "done",
          f"a done proven on the retry gave {proven.status}, want done")

    # ------------------------------------------------------- a cut-off reply
    # A command chopped in half by the output limit still parses as a command.
    # It must not reach the server; the model is asked for a smaller step.
    cut_droplet = FakeDroplet()
    cut_agent, cut_record, _, cut_events = build(
        cut_droplet, SecretStore(),
        [
            ("THOUGHT: read the release list\nACTION: run\nCOMMAND: curl -s https://x/ | python3 -c \"",
             "length"),
            "ACTION: run\nCOMMAND: systemctl is-active ssh",
            "ACTION: done\nVERIFY: systemctl is-active ssh\nSUMMARY: ssh is up",
        ],
        directory=RUNS / "cut-off",
    )
    cut = cut_agent.run()
    check(failures, not any("python3" in command for command in cut_droplet.commands),
          f"a truncated command was executed: {cut_droplet.commands}")
    check(failures, cut.status == "done", f"the run did not recover from a cut-off reply: {cut.status}")
    check(failures, any("cut off" in message for kind, message in cut_events),
          "the operator was not told the reply was cut off")
    asked_smaller = [m["content"] for m in cut_agent.messages if "stopped at the output limit" in m["content"]]
    check(failures, len(asked_smaller) == 1, "the model was not asked for a smaller step")
    kinds = {json.loads(line)["kind"]
             for line in (cut_record.directory / "transcript.jsonl").read_text(encoding="utf-8").splitlines()}
    check(failures, "truncated_reply" in kinds, "the truncation was not recorded in the transcript")

    # ------------------------------------------- replies that leak end-of-turn markers
    # kimi-k3 appends `<|close|>argument<|sep|>` to the command and deepseek-v4-flash
    # appends `</antml>`; a shell reads either as a syntax error. The step in front
    # of the leak is usable. The second one ends the command without ending the
    # reply, so only the parser sees it - both paths must report and record.
    leak_droplet = FakeDroplet()
    leak_agent, leak_record, _, leak_events = build(
        leak_droplet, SecretStore(),
        [
            "THOUGHT: check the service\nACTION: run\nCOMMAND: systemctl is-active ssh"
            "<|close|>argument<|sep|><|close|>call<|sep|>",
            "ACTION: run\nCOMMAND_BEGIN\napt-get update</antml>\nCOMMAND_END\nThat is the next step.",
            "ACTION: done\nVERIFY: systemctl is-active ssh\nSUMMARY: ssh is up",
        ],
        directory=RUNS / "control-tokens",
    )
    leak = leak_agent.run()
    leak_record.write_report()
    check(failures, leak.status == "done", f"a leaked-marker reply gave {leak.status}, want done")
    check(failures, not any("<|" in command or "</antml>" in command for command in leak_droplet.commands),
          f"a leaked marker reached the server: {leak_droplet.commands}")
    ran = [c.strip() for c in leak_droplet.commands]
    check(failures, "systemctl is-active ssh" in ran and "apt-get update" in ran,
          f"the cleaned commands were not run: {leak_droplet.commands}")
    stripped = [m for kind, m in leak_events if "end-of-turn" in m]
    check(failures, len(stripped) == 2, f"the stripping was not reported twice: {stripped}")
    check(failures, any("<|close|>" in m for m in stripped) and any("</antml>" in m for m in stripped),
          f"the operator was not told which markers were removed: {stripped}")
    events = [json.loads(line) for line in
              (leak_record.directory / "transcript.jsonl").read_text(encoding="utf-8").splitlines()]
    recorded = [e for e in events if e["kind"] == "control_tokens"]
    check(failures, len(recorded) == 2 and all(e.get("markers") for e in recorded),
          f"the markers were not recorded in the transcript: {recorded}")

    # ------------------------------- a reply that writes the harness's next message
    # `... | tail -n 5STEP 21 RESULT`: the model ran past the end of its turn and
    # glued the harness's own header onto its command. Trimming it would be a guess
    # at what the number was, so the reply is refused and asked for again.
    echo_droplet = FakeDroplet()
    echo_agent, echo_record, _, echo_events = build(
        echo_droplet, SecretStore(),
        [
            "ACTION: run\nCOMMAND: journalctl -u ssh --no-pager -n 5STEP 1 RESULT\nexit code: 0",
            "ACTION: run\nCOMMAND: journalctl -u ssh --no-pager -n 50",
            "ACTION: done\nVERIFY: systemctl is-active ssh\nSUMMARY: ssh is up",
        ],
        directory=RUNS / "framing-echo",
    )
    echo = echo_agent.run()
    check(failures, echo.status == "done", f"the run did not recover from a framing echo: {echo.status}")
    check(failures, not any("RESULT" in command for command in echo_droplet.commands),
          f"a command carrying the harness's header was executed: {echo_droplet.commands}")
    check(failures, any("'STEP 1 RESULT'" in m["content"] and "step format" in m["content"]
                        for m in echo_agent.messages),
          "the model was not told which text was refused, or why")

    # ------------------------------- a reply that carries on into its own next step
    # `... | head -n 20THOUGHT: Clean up ...`, verbatim in shape from a real run.
    # Worse than the header echo, because `20THOUGHT:` is a valid shell word: the
    # far end's `bash -n` sees nothing wrong and the destructive first half of the
    # command runs before the last one fails. So it has to be caught here.
    glue_droplet = FakeDroplet()
    glue_agent, _, _, _ = build(
        glue_droplet, SecretStore(),
        [
            "THOUGHT: clear the stale unit and look at what is left\nACTION: run\n"
            "COMMAND: mysql -e 'DROP DATABASE staging'; systemctl is-active ssh | "
            "head -n 20THOUGHT: Clean up the stale database before recreating it.",
            "ACTION: run\nCOMMAND: systemctl is-active ssh | head -n 20",
            "ACTION: done\nVERIFY: systemctl is-active ssh\nSUMMARY: ssh is up",
        ],
        directory=RUNS / "step-glue",
    )
    glue = glue_agent.run()
    check(failures, glue.status == "done", f"the run did not recover from a glued step: {glue.status}")
    check(failures, not any("THOUGHT" in command for command in glue_droplet.commands),
          f"a command carrying the next step's key was executed: {glue_droplet.commands}")
    # The point of catching it in the parser: the half of the command in front of
    # the leak is destructive and must not run either.
    check(failures, not any("DROP DATABASE" in command for command in glue_droplet.commands),
          f"the destructive half of a glued command still ran: {glue_droplet.commands}")
    check(failures, any("'THOUGHT:'" in m["content"] for m in glue_agent.messages),
          "the model was not told which text was refused")

    # ------------------------------- a command written over several lines under COMMAND:
    # `cat > gr.cnf <<EOF` with the body on the lines below it. Those lines used to
    # be dropped without a word: the command ran on its own, wrote an empty file,
    # and came back exit 0, which cost a real run six steps of fixing a config it
    # had never written. So a first line that cannot stand alone takes the lines
    # below it, a complete one with a script hard against it is refused rather than
    # half-run, and commentary set off by a blank line is recorded as not run.
    body_droplet = FakeDroplet()
    body_agent, body_record, _, body_events = build(
        body_droplet, SecretStore(),
        [
            "THOUGHT: write the tuning drop-in\nACTION: run\n"
            "COMMAND: cat > /etc/mysql/mysql.conf.d/zz-probe.cnf <<'EOF'\n[mysqld]\n"
            "server_id = 7\nEOF",
            "ACTION: run\nCOMMAND: apt-get update\napt-get install -y mysql-server\n"
            "systemctl enable --now mysql",
            "ACTION: run\nCOMMAND: systemctl is-active ssh\n\nThat proves ssh survived it.",
            "ACTION: done\nVERIFY: systemctl is-active ssh\nSUMMARY: the drop-in is in place",
        ],
        directory=RUNS / "command-body",
    )
    body = body_agent.run()
    body_record.write_report()
    check(failures, body.status == "done",
          f"a multi-line command under COMMAND: gave {body.status}, want done")
    written = body_droplet.files.get("/etc/mysql/mysql.conf.d/zz-probe.cnf", "")
    check(failures, "server_id = 7" in written,
          f"the heredoc body never reached the file: {written!r}")
    # The refusal, not a first line run on its own: nothing may be installed here.
    check(failures, not any("mysql-server" in command for command in body_droplet.commands),
          f"half of a script under COMMAND: was executed: {body_droplet.commands}")
    check(failures, any("COMMAND: takes a single line" in m["content"]
                        and "COMMAND_BEGIN" in m["content"] for m in body_agent.messages),
          "the model was not told why the script was refused, or what to send instead")
    check(failures, any("did not run" in message for kind, message in body_events),
          "the operator was not told that lines below the command were left out")
    logged = [json.loads(line) for line in
              (body_record.directory / "transcript.jsonl").read_text(encoding="utf-8").splitlines()]
    continued = [e for e in logged if e["kind"] == "command_continued"]
    dropped = [e for e in logged if e["kind"] == "dropped_lines"]
    check(failures, len(continued) == 1 and continued[0]["lines"] == 3,
          f"the absorbed lines were not recorded: {continued}")
    check(failures, len(dropped) == 1 and dropped[0]["lines"] == ["That proves ssh survived it."],
          f"the lines left out of the command were not recorded: {dropped}")

    # ------------------------------------------------- unparseable model output
    junk_droplet = FakeDroplet()
    junk_agent, junk_record, _, _ = build(
        junk_droplet, SecretStore(), ["no step here"] * 4, directory=RUNS / "junk"
    )
    junk = junk_agent.run()
    check(failures, junk.status == "failed", f"unreadable replies gave {junk.status}, want failed")

    print(f"\n{'FAILURES' if failures else 'all checks passed'}")
    for failure in failures:
        print(f"  FAIL {failure}")
    if droplet.unhandled:
        print("\ncommands the simulator did not model:")
        for command in droplet.unhandled:
            print(f"  {command}")
    print(f"\nreport: {report}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
