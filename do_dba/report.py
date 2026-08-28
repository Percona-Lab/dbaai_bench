"""The record of a run: a JSONL audit log written as it happens, plus a report.

The log is appended after every event rather than assembled at the end, so a
crash, a dropped connection, or a Ctrl+C still leaves a full account of what was
run on the server.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from .inference.pricing import format_cost


@dataclass
class StepRecord:
    index: int
    action: str
    detail: str
    thought: str = ""
    verdict: str = "allow"
    verdict_reason: str = ""
    executed: bool = False
    exit_code: int | None = None
    duration: float = 0.0
    stdout: str = ""
    stderr: str = ""
    note: str = ""
    # Which server it ran on, left empty when the run has only one and the
    # question does not arise.
    host: str = ""


@dataclass
class HostInfo:
    """One server in the run, as the report and the log describe it."""

    name: str
    label: str  # user@host[:port]
    facts: dict[str, str] = field(default_factory=dict)


@dataclass
class Verification:
    """One check the harness re-ran for itself, and where it ran it."""

    host: str
    command: str
    exit_code: int
    output: str


@dataclass
class RunRecord:
    directory: Path
    task: str
    hosts: list[HostInfo]
    model: str
    mode: str
    dry_run: bool
    provider: str = ""  # which gateway served the model
    # What the model's context window was spent on, where the gateway reported one.
    # Recorded because it decides what the model was shown: a run whose results were
    # cut at 3,000 characters is not comparable with one cut at 8,000. See agent.py.
    context: str = ""
    started: datetime = field(default_factory=datetime.now)
    steps: list[StepRecord] = field(default_factory=list)
    verifications: list[Verification] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    cost_complete: bool = True
    # How the cost was arrived at, counted per reply. A figure the gateway
    # reported is what the account was charged; one worked out from published
    # rates is an estimate that cannot know about cached prompt tokens or which
    # upstream provider served the request, so the two are never added up
    # silently - the report says which it is looking at.
    billed_replies: int = 0
    estimated_replies: int = 0
    # Whether the gateway bills per token at all. A self-hosted server does not,
    # so its zero is a fact about the run rather than a rate looked up in a table.
    metered: bool = True
    status: str = "incomplete"
    summary: str = ""
    redact: Callable[[str], str] = lambda text: text

    # ------------------------------------------------------------------ log

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.log_path = self.directory / "transcript.jsonl"
        self.event(
            "run_started",
            task=self.task,
            hosts=[{"name": host.name, "address": host.label} for host in self.hosts],
            model=self.model,
            provider=self.provider,
            mode=self.mode,
            dry_run=self.dry_run,
            context=self.context,
            at=self.started.isoformat(timespec="seconds"),
        )

    def event(self, kind: str, **payload) -> None:
        def scrub(value):
            if isinstance(value, str):
                return self.redact(value)
            if isinstance(value, list):
                return [scrub(item) for item in value]
            if isinstance(value, dict):
                return {key: scrub(item) for key, item in value.items()}
            return value

        # Redacted before serializing, not the serialized line afterwards: a value
        # holding a quote, a backslash or a newline comes out of json.dumps escaped,
        # and a string replace would no longer recognise it there. An adopted
        # credential, read off a keeper file an operator may have edited, is exactly
        # the kind of value that can hold one - and a transcript that silently keeps
        # it in the clear is the one thing this log must not do.
        record = {"kind": kind, **{key: scrub(value) for key, value in payload.items()}}
        cleaned = json.dumps(record, ensure_ascii=False, default=str)
        try:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(cleaned + "\n")
        except OSError:
            pass  # losing the log must not stop the work

    def add_step(self, step: StepRecord) -> None:
        self.steps.append(step)
        self.event(
            "step",
            index=step.index,
            host=step.host,
            action=step.action,
            detail=step.detail,
            thought=step.thought,
            verdict=step.verdict,
            verdict_reason=step.verdict_reason,
            executed=step.executed,
            exit_code=step.exit_code,
            duration=round(step.duration, 2),
            stdout=step.stdout[-4000:],
            stderr=step.stderr[-2000:],
            note=step.note,
        )

    def add_usage(self, usage: dict[str, int], cost: float | None,
                  billed: bool = False, reply_id: str = "") -> None:
        """Account for one reply, and log it so the total can be taken apart.

        A line per reply, with the gateway's own id for it where there is one:
        that is what makes a run's cost line checkable against the gateway's
        activity page, rather than a single number to be believed or not.
        """
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        if cost is None:
            self.cost_complete = False
        else:
            self.cost += cost
            if billed:
                self.billed_replies += 1
            else:
                self.estimated_replies += 1
        self.event(
            "usage",
            reply=reply_id,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            cached_tokens=usage.get("cached_tokens", 0),
            cost=None if cost is None else round(cost, 8),
            cost_source="gateway" if billed else ("rates" if cost is not None else "none"),
        )

    @property
    def cost_note(self) -> str:
        """Where the cost figure came from, in parentheses, or nothing to add."""
        if not self.metered:
            return " (self-hosted - no per-token bill)"
        if self.billed_replies and self.estimated_replies:
            return " (part billed by the gateway, part from published rates)"
        if self.billed_replies:
            return " (billed by the gateway)"
        if self.estimated_replies:
            return " (estimated from published rates)"
        return ""

    # --------------------------------------------------------------- report

    def write_report(self) -> Path:
        path = self.directory / "report.md"
        elapsed = (datetime.now() - self.started).total_seconds()
        executed = [step for step in self.steps if step.executed]
        failed = [step for step in executed if step.exit_code not in (0, None)]

        lines: list[str] = [
            f"# DBA run: {self.status}",
            "",
            f"- **Task:** {self.task}",
            *self._host_lines(),
            f"- **Model:** {self.model}{f' (via {self.provider})' if self.provider else ''}",
            f"- **Mode:** {self.mode}{' (dry run - nothing was executed)' if self.dry_run else ''}",
            *([f"- **Context:** {self.context}"] if self.context else []),
            f"- **Started:** {self.started.isoformat(timespec='seconds')}  ({elapsed:.0f}s elapsed)",
            f"- **Steps:** {len(self.steps)} proposed, {len(executed)} executed, {len(failed)} non-zero exit",
            f"- **Tokens:** {self.prompt_tokens:,} in / {self.completion_tokens:,} out",
            f"- **Model cost:** {format_cost(self.cost)}{self.cost_note}"
            f"{'' if self.cost_complete else ' (excludes unpriced replies)'}",
            "",
            "## Outcome",
            "",
            self.summary or "_no summary was produced_",
            "",
        ]

        described = [host for host in self.hosts if any(host.facts.values())]
        if described:
            lines += ["## Servers as found" if len(self.hosts) > 1 else "## Server as found", ""]
            for host in described:
                if len(self.hosts) > 1:
                    lines += [f"**{host.name}** ({host.label})", ""]
                lines += ["```"]
                lines += [f"{key}: {value}" for key, value in host.facts.items() if value]
                lines += ["```", ""]

        if self.verifications:
            lines += ["## Independent verification", "", "Run by the harness, not by the model.", ""]
            for check in self.verifications:
                where = f"[{check.host}] " if len(self.hosts) > 1 else ""
                lines += [
                    f"`{where}{check.command}` -> exit {check.exit_code}", "",
                    "```", check.output.strip() or "(no output)", "```", "",
                ]

        lines += ["## Steps", ""]
        for step in self.steps:
            status = self._step_status(step)
            where = f" on {step.host}" if step.host else ""
            lines += [f"### {step.index}. {status} - {step.action}{where}", ""]
            if step.thought:
                lines += [f"_{step.thought}_", ""]
            lines += ["```", step.detail.strip() or "(nothing)", "```", ""]
            if step.verdict != "allow":
                lines += [f"- guard: **{step.verdict}** - {step.verdict_reason}"]
            if step.note:
                lines += [f"- {step.note}"]
            if step.executed:
                lines += [f"- exit {step.exit_code} in {step.duration:.1f}s"]
                body = (step.stdout or "").strip()
                if body:
                    lines += ["", "```", body[-3000:], "```"]
                err = (step.stderr or "").strip()
                if err:
                    lines += ["", "stderr:", "```", err[-1500:], "```"]
            lines.append("")

        text = self.redact("\n".join(lines))
        path.write_text(text, encoding="utf-8")
        self.event("run_finished", status=self.status, steps=len(self.steps), cost=round(self.cost, 6))
        return path

    def _host_lines(self) -> list[str]:
        """The header bullet for one server, or a bullet plus a list for several."""
        if len(self.hosts) == 1:
            return [f"- **Host:** {self.hosts[0].label}"]
        return [f"- **Hosts:** {len(self.hosts)}"] + [
            f"  - **{host.name}:** {host.label}" for host in self.hosts
        ]

    @staticmethod
    def _step_status(step: StepRecord) -> str:
        if step.verdict == "block":
            return "blocked"
        if not step.executed:
            return "not executed"
        if step.exit_code == 0:
            return "ok"
        return f"exit {step.exit_code}"


def run_directory(base: Path, host: str, when: datetime | None = None) -> Path:
    stamp = (when or datetime.now()).strftime("%Y%m%d-%H%M%S")
    safe_host = "".join(char if char.isalnum() or char in ".-_" else "-" for char in host)
    return base / f"{stamp}-{safe_host}"
