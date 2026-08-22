"""The loop: the model proposes one step, the harness judges and runs it, repeat."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from . import guard
from .fleet import Fleet, Target
from .inference.client import InferenceClient, InferenceError
from .inference.pricing import PriceBook, format_cost
from .protocol import (
    ABORT,
    DONE,
    RESULT_HEADER,
    RUN,
    Check,
    ProtocolError,
    Step,
    control_markers,
    parse,
    spec,
    strip_control_tokens,
)
from .report import RunRecord, StepRecord, Verification
from .secrets import SecretStore, env_name
from .ssh import SSHError

# Independent of anything the model claims, these run at the end.
DEFAULT_VERIFICATIONS = [
    "systemctl list-units --type=service --state=active --no-pager --plain "
    "| grep -Ei 'mysql|maria|postgres|mongo|valkey|redis' || echo 'no database service active'",
    "ss -tlnH | awk '{print $4}' | sort -u | paste -sd' ' -",
]

# What the harness can add to a failure the model has been seen not to work out for
# itself. `bad substitution` cost three steps in a row in a real run - the model kept
# rewriting the backslashes and never spotted that its single quotes were inside
# double quotes, where they are literal characters rather than quoting. The hint is
# appended to the result; it never changes what runs. Add an entry only when a
# recorded run shows the same confusion more than once.
RESULT_HINTS = [
    (
        "bad substitution",
        "harness: `${...}` inside a double-quoted string is expanded by the shell before "
        "the command runs, and single quotes inside double quotes are literal characters, "
        "not quoting. Wrap the whole argument in single quotes instead, or write `\\${...}`.",
    ),
    (
        # `\G` is in thirty-three steps across eight recorded runs. Nothing on this side
        # touches it - test_dba_wrap.py checks against a real shell that the backslash
        # arrives exactly as written - and the client's answer depends on which client it
        # is: every step where `\G` was honoured was an 8.x client, and all eight visible
        # `ERROR at line 1: Unknown command '\G'.` came from 9.7.2. One run has both,
        # which is the trap: the same command worked at step 8 against 8.4.11 and was
        # refused at step 27 once that server had been upgraded, so a model with a
        # working `\G` behind it looks for the difference in its own quoting and re-sends
        # the same statement in new quotes. One did that five times before trying
        # `--vertical`, which worked first time and for the rest of the run; another wrote
        # off `SHOW BINARY LOG STATUS` entirely. It is also the reason the needles are
        # matched against stdout: not one of the eight refusals reached stderr - they were
        # all behind the step's own `2>&1` - and only 76 of 935 recorded steps produce any
        # stderr at all.
        #
        # The `-N` caveat is from the same corpus: `mysql -N -e "SHOW REPLICA STATUS\G"`
        # prints the values without their field names, so two of one run's final checks
        # grepped for `Replica_IO_Running: Yes` against output that could not contain it,
        # and that run spent five steps working out why a healthy replica read as broken.
        "Unknown command '\\G'",
        "harness: the mysql client refused `\\G` instead of taking it as the "
        "vertical-output terminator, so that statement did not run - anything after it in "
        "the same -e did not either. This is not a quoting problem: the backslash reaches "
        "the client exactly as you wrote it, so re-quoting will be refused the same way, "
        "and an older client on another server may well have taken it. Use "
        "`mysql --vertical -e 'SHOW REPLICA STATUS'` for one field per line. `-N` gives "
        "plain tab-separated values but drops the field names, so do not add it to a "
        "check that greps for `Replica_IO_Running:`.",
    ),
    (
        # Eight steps across four recorded runs, four of them re-sends of a statement the
        # server had already refused. 8.4 renamed it, and the 1064 quotes only the part
        # the parser choked on, which reads like a typo: one run re-sent it with a
        # semicolon added, then in single quotes, then with the semicolon removed, its
        # thoughts going "the semicolon might be causing shell parsing issues" and then
        # "the semicolon is being stripped by the shell" - six steps, and two of them
        # spent doubting the server was MySQL, before it reached the new name. That
        # belief is what the hint is for: a model that thinks the harness edits its SQL
        # has no reason to trust any later step either.
        "near 'MASTER STATUS'",
        "harness: `SHOW MASTER STATUS` was removed in MySQL 8.4; it is `SHOW BINARY LOG "
        "STATUS` now (and `SHOW SLAVE STATUS` -> `SHOW REPLICA STATUS`, `SHOW SLAVE "
        "HOSTS` -> `SHOW REPLICAS`). Error 1064 comes from the server's parser, so what "
        "you sent arrived intact: nothing on this side strips semicolons or rewrites "
        "quotes, and sending the same statement quoted differently will be refused the "
        "same way.",
    ),
]

# 128 + SIGPIPE, which is what the shell reports for a program killed because
# whatever was reading its output stopped reading. `head` stops by design, and
# `set -o pipefail` (see ssh.wrap_command) then makes that the whole pipeline's exit
# code - so `apt-cache search percona | head -30` comes back a failure having done
# exactly what was asked. curl notices the write error itself and exits 23 instead.
#
# Nine steps across six recorded runs ended this way and every one of them had
# worked: package searches, `docker logs | head`, a dump verified with
# `zcat ... | head -5`. One model spent its next thought explaining the exit code to
# itself; a weaker one would have taken rule 7 ("never send a command that just
# failed unchanged") and rewritten a command that was fine. A run of them in a row
# would end the run as `failed` (max_consecutive_failures) having broken nothing.
SIGPIPE_EXIT = 141
CURL_WRITE_EXIT = 23
# The readers that stop early. `tail` is not one of them - it has to read to the end
# to know what the end is - which is why rule 6 recommends it.
PIPE_READER = re.compile(
    r"\|\s*(?:sudo\s+)?(?:"
    r"head\b"
    r"|(?:z?e?grep)\b[^|]*(?:\s-\w*q|\s-m\s*\d|--quiet|--silent|--max-count)"
    r")"
)
PIPE_CLOSED_NOTE = (
    "harness: that exit code is the pipe closing, not the command failing. `head` (or "
    "grep with a match limit) stopped reading, the shell killed the writer for it, and "
    "`set -o pipefail` made that the pipeline's exit code. The output above is "
    "everything the reader asked for, and the work itself is not affected. Do not "
    "re-run it for this reason; if you need the writer's own exit code, drop the "
    "truncating pipe or send the output to a file and read the file."
)
# Handed back for a VERIFY, where the same exit code is a different problem: the
# writer's status is gone, so the check cannot say whether the work is there.
PIPE_CLOSED_VERIFY = (
    "the truncating pipe (head, or grep with a match limit) closed early and the exit "
    "code above is that, not an answer - it says nothing about whether the work is "
    "there. Send the check again without the truncating pipe: a VERIFY is judged on "
    "its exit code."
)


def pipe_closed_early(result) -> bool:
    """Did this step only "fail" because something stopped reading its output?

    A program killed by SIGPIPE never got to report on its own work, so the exit code
    is not a verdict on it either way. The harness says so rather than leaving the
    model to read a failure into it, and does not count it toward the consecutive-
    failure limit - but it does not rewrite the code, which stays in the record.
    """
    command = getattr(result, "command", "") or ""
    if getattr(result, "timed_out", False) or not PIPE_READER.search(command):
        return False
    # 23 means something else entirely to another program, so it is curl's only.
    return result.exit_code == SIGPIPE_EXIT or (
        result.exit_code == CURL_WRITE_EXIT and "curl" in command
    )


# The other end of the same problem. `grep` exits 1 when nothing matched, and under
# pipefail that becomes the step's exit code - so a step whose work succeeded reads
# as a failure, with nothing in the output to say otherwise because the filter is
# what removed it. Four recorded runs show the model reading the wrong thing into it:
# `my_print_defaults --mysqld | grep -E '^(bind-address|...)'` matched nothing because
# the real output is `--bind-address=`, and the model concluded "the defaults tool
# failed"; `mysql -e "DROP TABLE ...; DROP DATABASE ..." 2>&1 | grep -v Warning`
# dropped both and came back 1 because the password warning was the only line to
# filter; and in the worst of them a replica really was reconfigured, the trailing
# grep found none of the status lines it wanted, and the model spent three steps
# undoing and redoing work that was already done.
#
# Unlike a closed pipe this cannot be excused: with pipefail, exit 1 is also what the
# command itself would report, so the harness genuinely does not know whether the work
# happened. What it can do is say that, and say which part of the step is unaccounted
# for - so the note is an explanation, not a verdict, and the step stays a failure.
GREP_STAGE = re.compile(r"\|\s*(?:sudo\s+)?z?e?grep\b((?:\s+-[-\w]+)*)")
# `grep -q`/`-m1` are asking a question, and 1 is the answer to it. Only the filters
# that were meant to pass output through are worth explaining.
QUIET_GREP = re.compile(r"(?:^|\s)-\w*q|--quiet|--silent|--max-count|(?:^|\s)-m\b")
FILTER_EMPTY_NOTE = (
    "harness: exit 1 with no output, from a step that ends in a `grep` filter, is "
    "ambiguous: `grep` exits 1 when no line matched, and `set -o pipefail` makes that "
    "the step's exit code, so this is either the command failing or the filter "
    "matching nothing while the command worked. The filter has also removed whatever "
    "would tell you which. Do not assume the work did not happen - check its actual "
    "state, and run the command again without the filter (or add `|| true`) if you "
    "need its own exit code."
)


def filter_matched_nothing(result) -> bool:
    """Exit 1 and silence, from a step whose last act was to filter its own output."""
    command = getattr(result, "command", "") or ""
    if result.exit_code != 1 or getattr(result, "timed_out", False):
        return False
    if result.stdout.strip() or result.stderr.strip():
        return False  # something got through the filter, so 1 came from further up
    return any(not QUIET_GREP.search(flags) for flags in GREP_STAGE.findall(command))

# How much the operator is asked, from most to least. The loop below only cares
# about MODE_STEP, which adds a prompt to steps the guard did not flag; the rest
# is decided in cli.py, where the questions are. MODE_UNATTENDED answers all of
# them - it does not widen what the guard allows, so a blocked step is still
# blocked with nobody watching.
MODE_STEP = "step"
MODE_PLAN = "plan"
MODE_AUTO = "auto"
MODE_UNATTENDED = "unattended"


@dataclass
class Outcome:
    status: str  # done | unverified | aborted | exhausted | failed | cancelled | stuck
    summary: str = ""
    steps: int = 0
    executed: int = 0
    cost: float = 0.0
    cost_complete: bool = True

    @property
    def ok(self) -> bool:
        return self.status == "done"


@dataclass
class Limits:
    max_steps: int = 40
    command_timeout: float = 300.0
    max_output_chars: int = 3000
    max_cost: float | None = None
    # Enough rope to work through a genuinely awkward server - an expired repository
    # key, a package that is not where the model expected - without a run being ended
    # by a bad patch it was about to climb out of. It is still a thrash stop, not a
    # budget: max_steps and max_cost are what bound the run.
    max_consecutive_failures: int = 10
    max_protocol_retries: int = 2
    # How many times a "done" whose checks fail is handed back to be fixed.
    max_done_rejections: int = 2
    # Older observations are trimmed to keep the prompt (and its cost) bounded.
    keep_full_observations: int = 6


class DBAAgent:
    def __init__(
        self,
        *,
        client: InferenceClient,
        model: str,
        fleet: Fleet,
        task: str,
        record: RunRecord,
        store: SecretStore,
        prices: PriceBook,
        emit: Callable[[str, str], None],
        approve: Callable[[str, str, str], bool],
        mode: str = MODE_PLAN,
        dry_run: bool = False,
        limits: Limits | None = None,
        temperature: float | None = 0.2,
        verifications: list[str] | None = None,
        persist: Callable[[], None] | None = None,
    ):
        self.client = client
        self.model = model
        self.fleet = fleet
        self.task = task
        self.record = record
        self.store = store
        self.prices = prices
        self.emit = emit
        self.approve = approve
        self.mode = mode
        self.dry_run = dry_run
        self.limits = limits or Limits()
        self.temperature = temperature
        self.verifications = DEFAULT_VERIFICATIONS if verifications is None else verifications
        # Called when a step has just brought a new credential into being, so it can
        # be put somewhere it survives this process. See secrets.write_keeper.
        self.persist = persist
        # The format instructions name the servers, so the model is told what a
        # HOST: line may say in the same place it is told to write one.
        self.spec = spec(fleet.names if fleet.many else ())

        self.messages: list[dict[str, str]] = [{"role": "system", "content": self._system_prompt()}]
        self.cost = 0.0
        self.cost_complete = True
        self.last_finish = ""  # why the last reply ended; "length" means cut off
        self._observation_indices: list[int] = []

    # ---------------------------------------------------------------- prompts

    def _system_prompt(self) -> str:
        where = (
            f"a shell on each of {len(self.fleet)} remote Linux servers"
            if self.fleet.many else "a shell on a remote Linux server"
        )
        return f"""You are a senior database administrator with {where}.
A harness relays your steps over SSH: you send one step, it runs it and sends back
the result. You never see a terminal, only these results.

TASK
{self.task}

{self.fleet.brief()}

{self._credentials()}HOW TO ANSWER
{self.spec}

RULES
1. One step per reply, then wait for the result before choosing the next.
2. Nothing interactive: stdin is /dev/null. Use -y with apt-get, --no-pager with
   systemctl, mysql -e '...', psql -c '...', mongosh --eval '...' and valkey-cli
   with the command on the same line. Editors, pagers, bare REPLs, a server started
   in the foreground, and anything that streams until interrupted (MONITOR,
   SUBSCRIBE) are rejected by the harness.
3. Be idempotent. Check state before changing it, so re-running a step is safe.
4. Never invent a password. Write {{{{DBA_SECRET:name}}}} where one belongs and the
   harness substitutes a strong generated value you will never see. Reuse the same
   name for the same credential.{self._reuse_rule()}
5. Keep databases reachable only from localhost unless the task says otherwise: no
   bind-address 0.0.0.0, no 0.0.0.0/0 in pg_hba.conf, no trust authentication, no
   `bind 0.0.0.0` or `protected-mode no` in valkey.conf, and mongod stays on
   bindIp 127.0.0.1 with authorization enabled.
6. Keep output small - append `| tail -n 30` to anything chatty. Output is truncated
   at {self.limits.max_output_chars} characters and long output wastes your budget. Do not end a step
   with a `grep` filter: no match is exit 1, so a step that worked comes back looking
   failed with nothing left to show you why. Add `|| true`, or put `| tail` last. Keep
   stderr for the same reason - it carries the diagnosis - so no `2>/dev/null` on a
   step you may have to debug, and `curl -sS`/`wget` without `-q` when a download
   might fail.
7. Read failures and adapt. Never send a command that just failed unchanged.
8. Prefer distribution packages and systemd units over source builds or containers
   unless the task asks for them. MongoDB is the one exception: it is not in
   Ubuntu's archive, so add the vendor repository with its key in
   /usr/share/keyrings and signed-by in the sources line, then apt-get update
   before installing - apt cannot see a repository it has not fetched.
9. Each command is killed after {self.limits.command_timeout:.0f}s. A slow apt install is fine;
   design steps that finish inside that.
10. Finish with ACTION: done, one VERIFY: line per thing the task asked for, and a
    SUMMARY covering what changed, the state of each service, and where credentials
    went. Each VERIFY command must exit non-zero if the work is not really there;
    the harness re-runs them all, and a done step with none is handed straight back.
{self._fleet_rules()}
A safety guard inspects every step first. Destructive commands are blocked - you are
told why and should choose another route - and risky ones pause for a human. You have
at most {self.limits.max_steps} steps."""

    def _credentials(self) -> str:
        """The credentials these servers already hold, by name and never by value.

        Without this a run has no way to know a password exists. It opens with a
        passwordless login, and when that is refused its only remaining route in is
        to reset the password - which works, takes dozens of steps, and quietly
        invalidates whatever the operator wrote down. The names are enough: the
        harness substitutes the value, so the model needs to know that mysql_root is
        a thing it may use, not what it is.
        """
        names = self.store.inherited
        if not names:
            return ""
        example = names[0]
        lines = ["CREDENTIALS ALREADY ON THESE SERVERS"]
        lines += [f"  {name}" for name in names]
        lines += [
            "An earlier run set these and they are still in place. Write the placeholder to",
            f"use one - {{{{DBA_SECRET:{example}}}}} becomes the value the harness set, so a",
            f"command with it in logs in - or use ${env_name(example)} directly, which the",
            "shell running your commands already has. Do not reset a password listed here,",
            "and do not invent a second name for one.",
        ]
        return "\n".join(lines) + "\n\n"

    def _reuse_rule(self) -> str:
        """The half of rule 4 that only applies when a credential already exists."""
        if not self.store.inherited:
            return ""
        return (" A name listed under CREDENTIALS ALREADY ON\n"
                "   THESE SERVERS is a password that works now: use it rather than resetting it.")

    def _fleet_rules(self) -> str:
        """The rules that only exist because there is more than one server.

        Not a plain f-string: the placeholder syntax below has braces of its own.
        """
        if not self.fleet.many:
            return ""
        # The last rule exists only when the operator gave a bare list of servers.
        # With names they have already said what each one is for, and inviting the
        # model to decide would invite it to overrule them.
        roles = "" if self.fleet.assigned else """15. The names above are labels, not roles: which server takes which role is
    yours to decide and part of the task. Say the assignment in one line in your
    first step and hold to it for the rest of the run - the servers are alike, so
    any assignment that satisfies the task is a good one, and changing your mind
    halfway is what breaks these runs.
"""
        return """11. Every run and write_file step says which server it is for on a HOST: line.
    Nothing is broadcast: one step runs on one server and you get that server's
    result back before the next step. Work through the servers one at a time.
12. The PRIVATE NETWORK note above says what the servers can reach each other on;
    the harness checked it before this run started, so trust it over any assumption.
    Where a private address is listed, that is the address for everything the servers
    say to each other: bind the database to it, point replicas at it, and never at
    0.0.0.0 or at the public address. Where the note says there is no private path,
    the public address is all there is - use it, and keep the scoping just as tight.
    Either way, grants, pg_hba.conf lines and firewall rules name the peer's exact
    address, never a range and never %.
13. The same {{DBA_SECRET:name}} placeholder resolves to the same value on every
    server, which is exactly how a credential two of them share is set up: write the
    same placeholder on both sides and the harness makes them match.
14. Verify from both ends. A replica that reports itself healthy is not proof; check
    on the primary that the replica is connected, and confirm on the replica that
    data written to the primary has arrived. Scope a check to one server with
    VERIFY: [name] command.
""" + roles

    def plan(self) -> str | None:
        """One non-executing call, so a human can read the intent before anything runs."""
        prompt = (
            "Before you touch the server, outline your plan as a short numbered list of "
            "the commands or files you intend to use, then one line on how you will "
            "verify the result. Do not use the step format for this reply."
        )
        if self.fleet.many:
            prompt += (
                f" Name the server each numbered item runs on ({', '.join(self.fleet.names)}), "
                "and say which addresses the servers will use to reach each other."
            )
            if not self.fleet.assigned:
                prompt += (
                    " The names are only labels, so start with one line saying which server "
                    "takes which role - that is yours to decide, and the rest of the plan "
                    "depends on it."
                )
        reply = self._ask([*self.messages, {"role": "user", "content": prompt}])
        return reply

    # ------------------------------------------------------------------- loop

    def run(self) -> Outcome:
        self.messages.append({
            "role": "user",
            "content": "Begin. Send your first step.",
        })

        consecutive_failures = 0
        consecutive_rejections = 0
        done_rejections = 0
        executed = 0

        for index in range(1, self.limits.max_steps + 1):
            if self.limits.max_cost is not None and self.cost >= self.limits.max_cost:
                return self._finish("exhausted", f"stopped at the {format_cost(self.limits.max_cost)} cost cap", index - 1, executed)

            step = self._next_step(index)
            if step is None:
                return self._finish("failed", "the model never produced a usable step", index - 1, executed)

            if step.extra_actions:
                self.emit("note", "the reply held more than one step; only the first was used")

            if step.action in (DONE, ABORT):
                record = StepRecord(index=index, action=step.action, detail=step.summary, thought=step.thought)
                if step.action == ABORT:
                    self.record.add_step(record)
                    return self._finish("aborted", step.summary, index, executed)

                # "done" is a claim, not a fact: check it before believing it.
                self.emit("note", "checking the work")
                broken = self._verify(step.verify)
                # A done step with no VERIFY line leaves nothing to re-run, so the
                # claim would be taken on trust - the one thing this harness will
                # not do. A dry run is exempt: no check was ever going to run.
                unproven = not step.verify and not self.dry_run
                if (broken or unproven) and done_rejections < self.limits.max_done_rejections:
                    done_rejections += 1
                    if broken:
                        record.note = f"claimed done, but {len(broken)} check(s) failed"
                        self.record.add_step(record)
                        self.emit("fail", f"{len(broken)} of the checks failed; handing it back")
                        self._observe(
                            "YOU REPORTED THE TASK COMPLETE, BUT THE HARNESS RE-RAN THE CHECKS AND "
                            "THESE FAILED:\n\n" + "\n\n".join(broken) +
                            "\n\nThe task is not finished. Fix what these show and continue."
                        )
                    else:
                        record.note = "claimed done with nothing to prove it"
                        self.record.add_step(record)
                        self.emit("fail", "the done step carried no check; asking for proof")
                        self._observe(
                            "YOU REPORTED THE TASK COMPLETE WITHOUT A SINGLE VERIFY LINE.\n"
                            "A completion claim is not taken on trust here. Send the done step "
                            "again with one VERIFY line for each thing the task asked for: a "
                            "read-only command that exits non-zero if the work is not actually "
                            "there. For a database task that usually means one query for the "
                            "database, one for the login user, and one check that the service "
                            "is enabled at boot."
                        )
                    continue
                self.record.add_step(record)
                if broken:
                    return self._finish(
                        "unverified",
                        f"the model reported success but {len(broken)} check(s) still fail. "
                        f"{step.summary}",
                        index,
                        executed,
                    )
                if unproven:
                    return self._finish(
                        "unverified",
                        "the model reported success but never gave a check to prove it. "
                        f"{step.summary}",
                        index,
                        executed,
                    )
                return self._finish("done", step.summary, index, executed)

            # Which server, decided once. parse() has already turned the HOST: line
            # into one of the fleet's own names, or refused the step outright, so
            # the lookup cannot miss unless there is only one server to hit.
            target = self.fleet.find(step.host) or self.fleet.only
            verdict, detail = self._judge(step)
            record = StepRecord(
                index=index,
                action=step.action,
                detail=detail,
                thought=step.thought,
                verdict=verdict.level,
                verdict_reason=verdict.reason,
                host=target.name if self.fleet.many else "",
            )

            if verdict.blocked:
                consecutive_rejections += 1
                self.emit("blocked", f"step {index} blocked: {verdict.reason}")
                record.note = "blocked by the safety guard"
                self.record.add_step(record)
                self._observe(
                    f"STEP {index} WAS BLOCKED BY THE SAFETY GUARD\n"
                    f"reason: {verdict.reason}\n"
                    "It was not executed. Choose a different approach that avoids this."
                )
                if consecutive_rejections >= 3:
                    return self._finish("stuck", "three steps in a row were blocked by the guard", index, executed)
                continue
            consecutive_rejections = 0

            # The server is part of what is being approved: on a pair, the same
            # command is routine on one of them and a disaster on the other.
            asked = self._where(target) + detail
            if verdict.needs_approval and not self.approve(step.action, asked, verdict.reason):
                record.note = "the operator declined this step"
                self.record.add_step(record)
                self._observe(
                    f"STEP {index} WAS DECLINED BY THE OPERATOR\n"
                    f"the guard flagged it: {verdict.reason}\n"
                    "It was not executed. Find a way that does not need it, or ACTION: abort."
                )
                continue

            if self.mode == MODE_STEP and not verdict.needs_approval:
                if not self.approve(step.action, asked, "step-by-step mode"):
                    record.note = "the operator declined this step"
                    self.record.add_step(record)
                    self._observe(f"STEP {index} WAS DECLINED BY THE OPERATOR. Try something else or abort.")
                    continue

            if self.dry_run:
                record.note = "dry run - not executed"
                self.record.add_step(record)
                self.emit("dry", f"would run: {asked}")
                self._observe(
                    f"STEP {index} WAS NOT EXECUTED (dry run).\n"
                    "Assume it would have succeeded and continue planning the remaining steps."
                )
                continue

            result = self._execute(step, index, target)
            if result is None:
                return self._finish("failed", "the SSH connection failed mid-run", index, executed)

            executed += 1
            record.executed = True
            record.exit_code = result.exit_code
            record.duration = result.duration
            record.stdout = self.store.redact(result.stdout)
            record.stderr = self.store.redact(result.stderr)
            closed_pipe = pipe_closed_early(result)
            if result.timed_out:
                record.note = f"killed after {self.limits.command_timeout:.0f}s"
            elif closed_pipe:
                record.note = "the reader closed the pipe; the writer's own exit code is lost"
            elif filter_matched_nothing(result):
                record.note = "nothing matched the filter; the step's own exit code is lost in it"
            self.record.add_step(record)
            if self.store.unsaved and self.persist is not None:
                # A credential came into being in that step. Put it somewhere it
                # outlives this process now, not at the end of the run: the server's
                # password has already changed, and a run that dies here would leave
                # one nobody can look up - which is the failure this exists for.
                self.persist()
            self.emit("ok" if result.ok or closed_pipe else "fail", self._describe_result(result))

            # A closed pipe is neither a success nor a failure of the work, so it
            # neither clears the count nor adds to it: a run of real failures around
            # one `| head` should still stop the run, and a run of `| head`s should not.
            if result.ok:
                consecutive_failures = 0
            elif not closed_pipe:
                consecutive_failures += 1
            self._observe(self._format_result(index, result, target))

            if consecutive_failures >= self.limits.max_consecutive_failures:
                return self._finish(
                    "failed",
                    f"{consecutive_failures} steps in a row failed; stopping rather than thrashing",
                    index,
                    executed,
                )

        return self._finish("exhausted", f"hit the {self.limits.max_steps}-step limit", self.limits.max_steps, executed)

    # ------------------------------------------------------------- mechanics

    def _next_step(self, index: int) -> Step | None:
        """Ask for a step, correcting the model if the format is wrong."""
        for attempt in range(self.limits.max_protocol_retries + 1):
            raw = self._ask(self.messages)
            if raw is None:
                return None
            # Some models leak their own end-of-turn markers into the text. The
            # step in front of the leak is usable, so it is cleaned rather than
            # rejected - but recorded, because it says something about the model.
            reply = strip_control_tokens(raw)
            if reply != raw:
                self._note_markers(index, control_markers(raw))
            self.messages.append({"role": "assistant", "content": reply})
            # A reply that hit the output limit stops mid-sentence, and a command
            # cut in half still parses as a command - which is how `python3 -c "`
            # once reached a server. An extra round trip is the cheaper mistake.
            if self.last_finish == "length":
                self.emit("note", f"step {index}: the reply was cut off at the output limit; asking again")
                self.record.event(
                    "truncated_reply", index=index, attempt=attempt + 1, tail=reply[-200:]
                )
                self._observe(
                    "Your reply stopped at the output limit before the step was complete, so "
                    "nothing was run. Send the step again, smaller: shorten the command, drop "
                    "the commentary, or split the work across two steps."
                )
                continue
            try:
                step = parse(reply, self.fleet)
                # A marker can end the command without ending the reply, in which
                # case only the parser sees it.
                self._note_markers(index, step.cleaned_markers)
                self._note_command_lines(index, step)
                return step
            except ProtocolError as exc:
                self.emit("note", f"step {index}: {exc}")
                # The reply itself, not just the complaint: without it there is no
                # telling afterwards whether the model wrote prose or the parser
                # missed a step it should have found.
                self.record.event(
                    "protocol_error",
                    index=index,
                    attempt=attempt + 1,
                    error=str(exc),
                    reply=reply[:800],
                )
                self._observe(f"Your reply could not be read. {exc}\n\n{self.spec}")
        return None

    def _note_markers(self, index: int, markers: list[str]) -> None:
        """Say which end-of-turn markers were removed, and record them.

        Which markers a model leaks is the diagnosis when a step arrives mangled,
        and the cleaned reply no longer shows them.
        """
        if not markers:
            return
        self.emit("note", f"step {index}: stripped end-of-turn markers from the reply "
                          f"({' '.join(markers)})")
        self.record.event("control_tokens", index=index, markers=markers)

    def _note_command_lines(self, index: int, step: Step) -> None:
        """Record what happened to the lines under a single-line COMMAND:.

        Both halves are worth a transcript entry. Lines taken as part of the
        command say the model used the wrong form and the harness repaired it -
        which is fine, and worth knowing if the command then misbehaves. Lines not
        taken say the step ran on less than the model wrote, and that is the kind
        of thing that looks like a success at the time and like a mystery later, so
        it is said out loud as well.
        """
        if step.continued_lines:
            self.record.event(
                "command_continued", index=index, lines=len(step.continued_lines)
            )
        if step.dropped_lines:
            self.emit("note", f"step {index}: {len(step.dropped_lines)} line(s) below the "
                              "command were not part of it and did not run")
            self.record.event(
                "dropped_lines", index=index, lines=[line[:200] for line in step.dropped_lines]
            )

    def _ask(self, messages: list[dict[str, str]]) -> str | None:
        try:
            completion = self.client.complete(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                on_note=lambda note: self.emit("note", note),
            )
        except InferenceError as exc:
            self.emit("error", str(exc))
            self.record.event("model_error", error=str(exc))
            return None

        self.last_finish = completion.finish_reason
        if completion.usage:
            # The gateway's own figure where it gives one: it is what the account
            # was charged, and it accounts for cached prompt tokens and for which
            # upstream provider served the reply, neither of which a rate table
            # knows. The table is the fallback, and an estimate is labelled as one.
            cost, billed = completion.cost, completion.cost is not None
            if not billed:
                priced = completion.model if self.prices.has(completion.model) else self.model
                cost = self.prices.cost(
                    priced,
                    completion.usage.get("prompt_tokens", 0),
                    completion.usage.get("completion_tokens", 0),
                )
            if cost is None:
                self.cost_complete = False
            else:
                self.cost += cost
            self.record.add_usage(completion.usage, cost, billed=billed, reply_id=completion.id)
        return completion.text

    def _judge(self, step: Step) -> tuple[guard.Verdict, str]:
        if step.action == RUN:
            return guard.classify(step.command), step.command
        detail = f"{step.path} (mode {step.mode}, {len(step.content)} bytes)"
        verdict = guard.classify_file_write(step.path)
        if verdict.level == guard.ALLOW:
            verdict = guard.classify_file_content(step.content, step.path)
        return verdict, detail

    def _where(self, target: Target) -> str:
        """`[replica] `, or nothing at all when there is only one server."""
        return f"[{target.name}] " if self.fleet.many else ""

    def _execute(self, step: Step, index: int, target: Target):
        # Placeholders become real credentials only here, on the way out.
        try:
            if step.action == RUN:
                self.emit("run", self._where(target) + step.command)
                return target.runner.run(
                    self.store.resolve(step.command), timeout=self.limits.command_timeout
                )
            self.emit("run", f"{self._where(target)}write {step.path}")
            return target.runner.write_file(
                step.path, self.store.resolve(step.content), mode=step.mode
            )
        except SSHError as exc:
            self.emit("error", f"{self._where(target)}{exc}")
            self.record.event("ssh_error", index=index, host=target.name, error=str(exc))
            return None

    def _describe_result(self, result) -> str:
        """One line for the operator watching the run."""
        parts = [f"exit {result.exit_code} in {result.duration:.1f}s"]
        if result.timed_out:
            parts.append(f"killed after {self.limits.command_timeout:.0f}s")
        if result.output_truncated:
            parts.append("output truncated")
        if pipe_closed_early(result):
            # Said on the line, because the step is shown as fine and a non-zero code
            # next to a tick is the sort of thing that teaches an operator to
            # distrust the ticks.
            parts.append("the reader closed the pipe")
        elif filter_matched_nothing(result):
            # Still shown as a failure - it may well be one - but the operator is
            # spared hunting for an error message that the filter removed.
            parts.append("nothing matched the filter")
        elif not result.ok:
            tail = [line for line in (result.stderr or result.stdout or "").splitlines() if line.strip()]
            if tail:
                parts.append(self.store.redact(tail[-1].strip())[:160])
        return " - ".join(parts)

    def _format_result(self, index: int, result, target: Target) -> str:
        limit = self.limits.max_output_chars
        parts = [RESULT_HEADER.format(index=index)]
        if self.fleet.many:
            # Named on its own line, so a model reading back through the
            # observations can see which server each result came from.
            parts.append(f"server: {target.name}")
        parts.append(f"exit code: {result.exit_code} ({result.duration:.1f}s)")
        if pipe_closed_early(result):
            # Straight after the exit code, because it is the exit code that needs
            # explaining and the output below it is fine.
            parts.append(PIPE_CLOSED_NOTE)
        if result.timed_out:
            parts.append(f"the command was killed after {self.limits.command_timeout:.0f}s")
        if result.output_truncated:
            parts.append("the server produced more output than was captured")

        for label, body in (("stdout", result.stdout), ("stderr", result.stderr)):
            text = self.store.redact(body or "").strip()
            if not text:
                continue
            if len(text) > limit:
                text = text[: limit // 3] + f"\n... [{len(text) - limit} characters cut] ...\n" + text[-(2 * limit // 3):]
            parts.append(f"{label}:\n{text}")
        if not result.stdout.strip() and not result.stderr.strip():
            parts.append("(no output)")
            # After the silence rather than before it: the note is about what the
            # silence and the exit code together do and do not say.
            if filter_matched_nothing(result):
                parts.append(FILTER_EMPTY_NOTE)
        # Both streams. 533 of the 935 executed steps in the recorded runs - 57% - redirect
        # with `2>&1`, which is the harness's own advice and puts every diagnostic on
        # stdout; only 76 of the 935 produce any stderr at all. Keyed on stderr alone the
        # table was blind on nine steps in ten, the `\G` refusal included. The
        # cost of reading stdout too is a hint on a step that merely printed the phrase -
        # a paragraph of explanation nobody needed, against a wrong turn nobody caught.
        streams = (result.stdout or "") + "\n" + (result.stderr or "")
        parts.extend(hint for needle, hint in RESULT_HINTS if needle in streams)
        return "\n".join(parts)

    def _observe(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        self._observation_indices.append(len(self.messages) - 1)
        self._compact()

    def _compact(self) -> None:
        """Trim old observations so the prompt does not grow without bound."""
        stale = self._observation_indices[: -self.limits.keep_full_observations]
        for position in stale:
            message = self.messages[position]
            body = message["content"]
            if len(body) > 400 and not body.endswith("[earlier output trimmed]"):
                message["content"] = body[:400] + "\n... [earlier output trimmed]"

    def _verify(self, model_checks: list[Check]) -> list[str]:
        """Re-check the work with the harness's own commands, not the model's word.

        Returns one description per check that failed, ready to hand back. An
        unscoped check runs on every server: half a cluster working is not the task,
        and a check only ever asked of the primary would not notice.
        """
        if self.dry_run:
            return []
        plan: list[tuple[Target, str]] = [
            (target, command) for target in self.fleet for command in self.verifications
        ]
        for check in model_checks:
            scoped = [self.fleet.find(check.host)] if check.host else list(self.fleet)
            for target in scoped:
                if target is not None and (target, check.command) not in plan:
                    plan.append((target, check.command))

        # Only the latest attempt belongs in the report; the transcript keeps
        # every attempt, so nothing is lost.
        self.record.verifications = []
        broken: list[str] = []

        for target, command in plan:
            where = self._where(target)
            verdict = guard.classify(command)
            if verdict.level != guard.ALLOW:
                # Not a failure of the work: the check itself was not safe to run.
                self.record.verifications.append(
                    Verification(target.name, command, -1, f"skipped: {verdict.reason}"))
                continue
            try:
                # A check may carry a placeholder too, e.g. mysql -u app -p'...'.
                result = target.runner.run(self.store.resolve(command), timeout=60)
            except SSHError as exc:
                self.record.verifications.append(
                    Verification(target.name, command, -1, f"could not run: {exc}"))
                broken.append(f"$ {where}{command}\ncould not run: {exc}")
                continue
            output = self.store.redact((result.stdout or "") + (result.stderr or ""))
            if pipe_closed_early(result):
                # Not a pass and not a failure of the work: the writer was killed by
                # its own reader, so nothing here says whether the work is there. A
                # check whose result is unknowable is not a check, so it goes back to
                # be rewritten rather than being counted either way.
                self.record.verifications.append(Verification(
                    target.name, command, result.exit_code, f"unusable check: {output[:1600]}"))
                self.record.event("verification", host=target.name, command=command,
                                  exit_code=result.exit_code, output=output[:2000],
                                  note="unusable: the reader closed the pipe")
                broken.append(f"$ {where}{command}\nexit {result.exit_code}: {PIPE_CLOSED_VERIFY}")
                continue
            self.record.verifications.append(
                Verification(target.name, command, result.exit_code, output[:2000]))
            self.record.event("verification", host=target.name, command=command,
                              exit_code=result.exit_code, output=output[:2000])
            if result.exit_code != 0 or result.timed_out:
                broken.append(
                    f"$ {where}{command}\nexit {result.exit_code}\n"
                    f"{output.strip()[:600] or '(no output)'}")
        return broken

    def _finish(self, status: str, summary: str, steps: int, executed: int) -> Outcome:
        self.record.status = status
        self.record.summary = summary
        return Outcome(
            status=status,
            summary=summary,
            steps=steps,
            executed=executed,
            cost=self.cost,
            cost_complete=self.cost_complete,
        )
