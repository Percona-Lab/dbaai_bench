"""The one-step-per-reply format the model answers in.

Line-based rather than JSON on purpose: file contents and shell quoting are
exactly where JSON string escaping goes wrong, and a mangled step costs a round
trip. Keys may appear in any order; anything unrecognised is ignored, so a model
that adds a sentence of prose still parses.

    THOUGHT: why this step
    ACTION: run
    COMMAND: apt-get install -y postgresql

    ACTION: write_file
    PATH: /etc/mysql/mysql.conf.d/bind.cnf
    MODE: 0644
    CONTENT_BEGIN
    [mysqld]
    bind-address = 127.0.0.1
    CONTENT_END

    ACTION: script
    INTERPRETER: bash
    SCRIPT_BEGIN
    #!/bin/bash
    set -euo pipefail
    for db in app logs; do mysql -e "CREATE DATABASE IF NOT EXISTS $db"; done
    SCRIPT_END

    ACTION: done
    VERIFY: systemctl is-active mysql
    SUMMARY: what changed and how it was checked

A run with more than one server adds one key, HOST:, and then insists on it: a
step that does not say which server it is for is refused rather than guessed at.

COMMAND: is one line; COMMAND_BEGIN / COMMAND_END is the form for more. Models
write loops and heredocs under the single-line key anyway, so the lines below it
are read as part of the command when its first line cannot stand alone, and a
whole script sent that way is refused. What is neither is recorded as not run:
dropping a command's second half in silence is the one thing that must not happen
here, because the shell reports success on the half it received.

A script is its own action rather than a command that happens to be long, because
what has to happen to it is different: it is copied to the server as a file and
run there by name, so INTERPRETER: decides which interpreter reads it and the
guard has to judge the whole body before any of it is copied. bash and python3
are the two, spelled however the model spells them; with no INTERPRETER: line the
shebang decides, and with neither it is bash.
"""

from __future__ import annotations

import itertools
import re
import textwrap
from dataclasses import dataclass, field

RUN = "run"
WRITE_FILE = "write_file"
SCRIPT = "script"
DONE = "done"
ABORT = "abort"
ACTIONS = {RUN, WRITE_FILE, SCRIPT, DONE, ABORT}

# The two interpreters a script step may ask for, and the spellings that reach
# them. Models write `sh`, `shell`, `python`, `py` and `python3.11` for these two
# things; refusing a spelling costs a round trip and teaches nothing, so the
# aliases resolve and only a third language is an error.
BASH = "bash"
PYTHON = "python3"
_INTERPRETER_ALIASES = {
    "bash": BASH, "sh": BASH, "shell": BASH, "zsh": BASH, "dash": BASH, "ksh": BASH,
    "/bin/bash": BASH, "/bin/sh": BASH, "/usr/bin/bash": BASH,
    "python": PYTHON, "python3": PYTHON, "py": PYTHON, "python2": PYTHON,
    "/usr/bin/python3": PYTHON, "/usr/bin/env python3": PYTHON,
}
# `python3.11`, `python3.12` and so on, which no alias table can enumerate.
_PYTHON_VERSIONED = re.compile(r"^/?(?:usr/bin/)?python\s*3(?:\.\d+)*$")
_SCRIPT_EXTENSIONS = {".sh": BASH, ".bash": BASH, ".zsh": BASH, ".py": PYTHON}

_SPEC = """Reply with exactly one step, in this format and nothing else:

THOUGHT: one line on why this step comes next
{host}ACTION: run
COMMAND: the shell command  (one line - the block form is below)

Other actions:

ACTION: run                 (a command over several lines: a loop, a heredoc)
{write_host}COMMAND_BEGIN
for i in 1 2 3; do
  echo "$i"
done
COMMAND_END

ACTION: write_file          (create or replace a file)
{write_host}PATH: /absolute/path
MODE: 0644                  (optional, default 0644)
CONTENT_BEGIN
file contents
CONTENT_END

ACTION: script              (a script, copied to the server and run there)
{write_host}INTERPRETER: bash           (bash or python3; optional, default bash)
SCRIPT_BEGIN
#!/bin/bash
set -euo pipefail
for db in app logs; do
  mysql -e "CREATE DATABASE IF NOT EXISTS $db"
done
SCRIPT_END

ACTION: done                (the task is complete and verified)
VERIFY: a read-only command proving it - at least one, repeat the line for more
{scoped}SUMMARY: what you changed, what state things are in now, where credentials went

ACTION: abort               (the task cannot be completed)
SUMMARY: why, and what you tried

COMMAND: holds one line. A loop, a heredoc or anything else that spans lines goes
between COMMAND_BEGIN and COMMAND_END, and a file's contents are better written
with write_file than with a heredoc. Several commands that belong together, or
anything needing real control flow, belong in ACTION: script - it is copied to the
server and run there in one step, and you get its exit code, stdout and stderr
back exactly as for a command. Never send two steps in one reply.{tail}"""


def spec(hosts=()) -> str:
    """The format instructions, naming the servers when there is more than one.

    One server needs no HOST: line and is not told about one - a key that can only
    be answered one way is a key a model gets wrong. Several servers turn it into
    the most important line in the step, so it goes first and is described twice.
    """
    names = [name for name in hosts if name]
    if len(names) < 2:
        return _SPEC.format(host="", write_host="", scoped="", tail="")
    listed = ", ".join(names)
    return _SPEC.format(
        # Padded to the column the other comments sit in, so the block reads as one.
        host=f"HOST: {names[0]}{' ' * max(1, 22 - len(names[0]))}(which server: {listed})\n",
        write_host=f"HOST: {names[-1]}\n",
        scoped=f"VERIFY: [{names[-1]}] a check to run on that server only\n",
        tail=(
            "\n\nEvery run, script and write_file step needs a HOST: line naming the server it\n"
            f"is for - one of: {listed}. A step without one, or with any other name, is\n"
            "refused and comes back to you unrun; nothing is guessed.\n"
            "A VERIFY: line with no [name] runs on every server."
        ),
    )


# Kept for the single-server case, which is most of them.
SPEC = spec()

_KEYS = {
    "thought", "action", "command", "content", "path", "mode", "summary", "verify",
    "explanation", "reason", "host", "script", "interpreter", "lang", "language",
}
_BLOCKS = {
    "command_begin": ("command", "command_end"),
    "content_begin": ("content", "content_end"),
    "summary_begin": ("summary", "summary_end"),
    "script_begin": ("script", "script_end"),
}
_KEY_LINE = re.compile(r"^\s{0,4}([A-Za-z_]{3,20})\s*:\s*(.*)$")
# The two keys a step opens with, and so the only ones that can end a command
# written over several lines under a single-line COMMAND:. Not all of _KEYS: a
# heredoc body is often YAML or a config file, and `mode: 0644` inside one is a
# line of the file, not a key of the step.
_STEP_OPENERS = {"action", "thought"}
_FENCE = re.compile(r"^\s*```")
# Models fall into YAML habits and write `COMMAND: |` with the body indented
# below. Taken literally that runs a bare pipe, so the block is read instead.
_YAML_BLOCK = re.compile(r"[|>][-+]?\d*$")
_MULTILINE_KEYS = {"command", "content", "summary", "script"}

# A single-line COMMAND: whose one line cannot be the whole command. Models write
# loops and heredocs under it constantly, and every line but the first used to be
# dropped without a word: `cat > gr.cnf <<EOF` ran on its own, wrote an empty
# file, and came back exit 0. So the lines below are taken as part of the command
# when the first line is one of these three shapes.
#
# A heredoc, whose body can only be on the following lines. `<<<` is a here-string
# and complete on its own, hence the lookarounds; `grep '<<EOF' f` is a quoted
# literal, so a match inside quotes does not count.
_HEREDOC = re.compile(r"(?<!<)<<(?!<)-?\s*['\"]?[A-Za-z_]")
_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")
# A line the shell would read as unfinished: a continuation, an operator or a
# keyword with nothing after it. The keywords need whitespace in front so that
# `ls /var/spool/in` is not mistaken for the `in` of a `case`. No trailing `;`,
# which ends a command perfectly well.
_CONTINUES = re.compile(r"(?:\\|&&|\|\||\||\{|\()$|(?:^|\s)(?:do|then|else|elif|in)$")


# Chat-template control tokens that leaked into the content instead of ending the
# turn. kimi-k3 ends replies `...; true<|close|>argument<|sep|><|close|>call<|sep|>`,
# which a shell reads as a syntax error near `|`. Other families leak `<|im_end|>`,
# `<|eot_id|>`, `<|end|>`. None of them are ever part of a step.
_CONTROL_TOKEN = re.compile(r"<\|[^|<>\s]{1,40}\|>")
# The same mistake in markup rather than pipes: deepseek-v4-flash ends commands
# `ls /etc/apt/sources.list.d/</antml>`, other families close with `<end_of_turn>`
# or `</s>`. Only a run of them at the very end of the reply or of the command
# counts, because that is the only place they cannot be part of the step: a shell
# reads a trailing `<tag>` as a redirect from `tag` and then a `>` with nothing to
# write to, which is a syntax error. Quoted - `grep '</div>' x` - the text does not
# end with the `>`, so nothing matches and nothing is touched.
_TRAILING_TAG = re.compile(r"(?:\s*</?[A-Za-z_][A-Za-z0-9_:-]{0,40}>)+\s*$")
_ACTION_LINE = re.compile(r"(?im)^\s{0,4}action\s*:")

# The header on every observation the harness sends back; DBAAgent._format_result
# builds it from this. A model that runs past the end of its turn writes the next
# message itself, and this is the first thing it writes - deepseek-v4-pro once
# finished a command `| tail -n 5STEP 21 RESULT`, with no space in between. There
# is no way to tell what the number was meant to be, so such a reply is refused
# rather than trimmed into something plausible.
RESULT_HEADER = "STEP {index} RESULT"
_FRAMING_ECHO = re.compile(r"STEP\s+\d+\s+RESULT")
# The other half of the same mistake: instead of the harness's reply the model
# begins its own next step, and the command swallows the key that opens it -
# `... | grep -E 'pxc' | head -n 20THOUGHT: Clean up test/old PXC containers`,
# again with no space. That one is worse than a leaked marker: `20THOUGHT:` is a
# perfectly good shell word, so `bash -n` on the far end has nothing to object to,
# and the real run got as far as removing three containers before `head` rejected
# its argument - a step half-applied after being approved whole.
#
# Only THOUGHT and ACTION, the two keys a step opens with. The rest of _KEYS
# would catch `echo "PATH: $PATH"` and `MODE: 0644` in an ordinary diagnostic,
# and no run has ever shown a step restarting at one of them. Add a key here when
# one does, not before.
#
# Both patterns are asked of a command and not of a SCRIPT_BEGIN body, for the
# same reason _inside_content leaves control markers there alone: the body is a
# file, so an overrunning model's prose arrives as extra lines under the script
# rather than glued into the middle of the command it was writing, the lines that
# were approved are still the lines that run, and the interpreter's parse pass
# refuses whatever does not parse. A forty-line script also has honest reason to
# write these words - `echo "ACTION: failover done"` in a report - where a
# one-line command does not. A recorded run whose script overran is what would
# justify widening this; there is none yet.
_STEP_RESTART = re.compile(r"(THOUGHT|ACTION)\s*:")


class ProtocolError(ValueError):
    """The reply could not be read as a step; the message is fed back verbatim."""


@dataclass(frozen=True)
class Check:
    """One VERIFY: command, and the server it is for.

    An empty host means every server in the run, which is what an unscoped check
    should mean: `systemctl is-active mysql` proves nothing about a pair if it is
    only ever asked of one of them.
    """

    command: str
    host: str = ""


# `VERIFY: [replica] mysql -e "SHOW REPLICA STATUS\\G"` - a check for one server.
# The name has to look like a name for this to fire, because `[ -f /etc/my.cnf ]`
# is a perfectly ordinary check and starts with the same bracket.
_VERIFY_HOST = re.compile(r"^\[\s*([A-Za-z][A-Za-z0-9._-]{0,31})\s*\]\s*(\S.*)$")


@dataclass
class Step:
    action: str
    thought: str = ""
    command: str = ""
    path: str = ""
    mode: str = "0644"
    content: str = ""
    script: str = ""       # the body of an ACTION: script step
    interpreter: str = ""  # bash or python3, resolved; empty for every other action
    summary: str = ""
    verify: list[Check] = field(default_factory=list)
    host: str = ""  # the server this step is for, as that server is named
    extra_actions: int = 0  # more than one ACTION: line was present
    cleaned_markers: list[str] = field(default_factory=list)  # end-of-turn junk removed
    # Lines below a single-line COMMAND: - the ones taken as part of the command,
    # and the ones that were not part of it and did not run. Both are recorded so
    # that a step which succeeded on less than the model wrote can still be seen.
    continued_lines: list[str] = field(default_factory=list)
    dropped_lines: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if self.action == RUN:
            return self.command
        if self.action == WRITE_FILE:
            return f"write {self.path} ({len(self.content)} bytes, mode {self.mode})"
        if self.action == SCRIPT:
            lines = len(self.script.splitlines())
            return f"{self.interpreter} script ({lines} lines, {len(self.script)} bytes)"
        return self.action


def strip_control_tokens(reply: str) -> str:
    """The reply with leaked end-of-turn markers removed.

    Two shapes, so two treatments. A leak that trails a finished step is cut off
    with everything after it: the step in front of it is whole, and appending
    `<|close|>argument` to a command only breaks it. A leak that wraps the step
    instead - `<|start|>assistant<|message|>THOUGHT: ...` - is unwrapped by
    turning each token into a newline, which keeps the keys at the start of their
    lines where the parser looks for them.

    Markers inside a CONTENT_BEGIN or SCRIPT_BEGIN block are left alone. There the
    text is a file body or a script, not shell the harness is about to run word for
    word, and a model asked to write a prompt template means them.
    """
    cleaned = _strip_trailing_tag(reply)
    first = _CONTROL_TOKEN.search(cleaned)
    if not first:
        return cleaned
    head = cleaned[: first.start()]
    if _inside_content(head):
        return cleaned
    if _ACTION_LINE.search(head):
        return head.rstrip()
    return _CONTROL_TOKEN.sub("\n", cleaned)


def control_markers(reply: str) -> list[str]:
    """The distinct end-of-turn markers in the reply, for the transcript.

    Recorded rather than only counted: which markers a model leaks is the whole
    diagnosis when a step arrives mangled, and the cleaned reply no longer shows.
    """
    found = [match.group(0) for match in _CONTROL_TOKEN.finditer(reply)]
    tail = _TRAILING_TAG.search(reply)
    if tail:
        found.extend(re.findall(r"</?[A-Za-z_][A-Za-z0-9_:-]{0,40}>", tail.group(0)))
    return sorted(set(found))


def _strip_trailing_tag(reply: str) -> str:
    match = _TRAILING_TAG.search(reply)
    if not match or _inside_content(reply[: match.start()]):
        return reply
    return reply[: match.start()].rstrip()


def _inside_content(head: str) -> bool:
    """Whether the text ends inside an open CONTENT_BEGIN or SCRIPT_BEGIN block.

    A script body gets the same treatment as a file body for the same reason: the
    text there is not shell the harness is about to run word for word, so a marker
    in it may well be something the model meant to write.
    """
    lowered = head.lower()
    return (lowered.rfind("content_begin") > lowered.rfind("content_end")
            or lowered.rfind("script_begin") > lowered.rfind("script_end"))


def parse(reply: str, fleet=None) -> Step:
    """Read one step. With a fleet, resolve which of its servers the step is for.

    The fleet is only ever asked for `find`, `names`, `only` and its length, so a
    test can pass anything that answers those. Without one the HOST: line is kept
    as written and nothing is checked, which is what a single-server caller wants.
    """
    if not reply or not reply.strip():
        raise ProtocolError("Your reply was empty. Send one step in the required format.")
    reply = strip_control_tokens(reply)

    fields: dict[str, str] = {}
    verify: list[str] = []
    actions: list[str] = []
    continued: list[str] = []  # lines taken as the rest of a single-line COMMAND:
    trailing: list[str] = []  # lines below one that were not; judged after the loop

    lines = reply.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if _FENCE.match(line):
            continue  # models like to wrap the whole step in a code fence

        # Block openers carry no value, so they are matched before the key
        # pattern (which insists on a colon). A stray trailing colon is fine.
        opener = line.strip().lower().rstrip(":")
        if opener in _BLOCKS:
            target, terminator = _BLOCKS[opener]
            body, index = _read_block(lines, index, terminator)
            fields[target] = body
            continue

        match = _KEY_LINE.match(line)
        if not match:
            continue
        key = match.group(1).lower()
        value = match.group(2).strip()

        if key not in _KEYS:
            continue

        if key == "action":
            actions.append(value.lower())
            continue
        if key == "verify":
            if value:
                verify.append(value)
            continue
        if key in _MULTILINE_KEYS and (not value or _YAML_BLOCK.fullmatch(value)):
            # "SUMMARY:" or "COMMAND: |" with the text on the following lines.
            body, index = _read_indented(lines, index)
            fields.setdefault(key, body)
            continue
        if key == "command" and "command" not in fields:
            below = _lines_below(lines, index)
            if _incomplete(value):
                # Verbatim, not dedented: the indentation of a heredoc body is part
                # of the file being written, and a terminator moved off column one
                # by a dedent is a terminator the shell never finds.
                continued = _strip_blank(below)
                fields["command"] = "\n".join([value, *continued]) if continued else value
                index += len(continued)
            else:
                fields["command"] = value
                trailing = below
            continue
        fields.setdefault(key, value)

    if not actions:
        raise ProtocolError(
            "No 'ACTION:' line found. Every reply needs one, with a value of "
            "run, write_file, done or abort."
        )

    action = actions[0]
    if action not in ACTIONS:
        raise ProtocolError(
            f"ACTION: {action!r} is not one of run, write_file, done, abort."
        )

    # Lines under a single-line COMMAND: that were not taken as part of it. Hard
    # against the command, with no blank line between, they read as the rest of
    # something the model wrote as one piece; prose set off by a blank line is what
    # it looks like. Worked out here rather than in the loop because what to do
    # about it depends on the action, and the keys may arrive in any order.
    dangling = list(itertools.takewhile(lambda line: line.strip(), trailing))
    dropped = [line for line in trailing if line.strip()]

    # The leak may end the command without ending the reply - a tag glued to the
    # last line inside COMMAND_BEGIN, say - so the command is checked in its own
    # right. A command ending in a markup tag is a shell syntax error either way.
    command = fields.get("command", "").strip()
    trimmed = _strip_trailing_tag(command)

    script_body = fields.get("script", "")
    step = Step(
        action=action,
        thought=fields.get("thought", ""),
        command=trimmed,
        script=script_body,
        interpreter=_interpreter(
            fields.get("interpreter") or fields.get("lang") or fields.get("language", ""),
            script_body,
            fields.get("path", ""),
        ) if action == SCRIPT else "",
        # Quotes around a path belong to shell syntax, not to the name; left in
        # they create a file whose name really does contain them.
        path=fields.get("path", "").strip().strip("'\"`").strip(),
        mode=(fields.get("mode") or "0644").strip(),
        content=fields.get("content", ""),
        summary=fields.get("summary") or fields.get("reason", ""),
        verify=[_check(line, fleet) for line in verify],
        host=_resolve_host(fields.get("host", ""), action, fleet),
        extra_actions=max(0, len(actions) - 1),
        cleaned_markers=[] if trimmed == command else control_markers(command),
        continued_lines=continued,
        dropped_lines=dropped,
    )

    if action == RUN and not step.command:
        raise ProtocolError("ACTION: run needs a COMMAND: line with the command to execute.")
    if action == RUN:
        # The dangling lines are searched along with the command: a reply that ran
        # past the end of its turn glues the next message on either mid-line or on
        # the line below, and the two deserve the same answer. Asked before the
        # dangling refusal below, which would otherwise send the model looking at
        # its quoting when the real mistake was carrying on writing.
        overreach = "\n".join([step.command, *dangling])
        overrun = _FRAMING_ECHO.search(overreach) or _STEP_RESTART.search(overreach)
        if overrun:
            raise ProtocolError(
                f"Your command contains {overrun.group(0)!r}, which belongs to the step format "
                "rather than to a shell command, so the reply carried on past the end of the "
                "command and what arrived is not what you meant to run. Nothing was executed. "
                "Send the step again and stop as soon as the command ends - the result comes "
                "back from the harness, and the next step comes after it."
            )
        if dangling:
            # Running only the first line is the worst of the outcomes available:
            # the step is half-applied, and the exit code says it worked.
            shown = " / ".join(line.strip() for line in dangling[:3])[:200]
            raise ProtocolError(
                f"COMMAND: takes a single line, so the {len(dangling)} line(s) directly below it "
                f"were not part of the command: {shown!r}. Nothing was run. If they belong to "
                "the command - a loop, a heredoc, a script - send it between COMMAND_BEGIN and "
                "COMMAND_END instead, and use ACTION: write_file for a file's contents. If they "
                "were commentary, send the step again without them."
            )
    if action == SCRIPT and not step.script.strip():
        raise ProtocolError(
            "ACTION: script needs the script itself between SCRIPT_BEGIN and SCRIPT_END. "
            "Nothing was run. Send the step again with the body in that block, and an "
            "INTERPRETER: line of bash or python3."
        )
    if action == WRITE_FILE:
        if not step.path:
            raise ProtocolError("ACTION: write_file needs a PATH: line with an absolute path.")
        if not step.content:
            raise ProtocolError(
                "ACTION: write_file needs the file body between CONTENT_BEGIN and CONTENT_END."
            )
    if action in (DONE, ABORT) and not step.summary:
        raise ProtocolError(f"ACTION: {action} needs a SUMMARY: describing the outcome.")
    if not re.fullmatch(r"[0-7]{3,4}", step.mode):
        step.mode = "0644"

    return step


def _interpreter(spelling: str, body: str, path: str = "") -> str:
    """Which interpreter reads this script: bash or python3.

    Asked in three places in turn, because a model that has said which language
    this is has usually said it somewhere. The INTERPRETER: line is the answer
    when there is one; a shebang is the model saying the same thing in the script
    itself, and it is what the far end would honour if the file were executable;
    a PATH:-style extension is the last hint. With none of them it is bash, which
    is what a step full of shell commands almost always is.

    A third language is refused rather than run as one of these two: `perl` read
    as bash is a screenful of syntax errors, and the round trip that says so is
    cheaper than the one that does not explain itself.
    """
    said = (spelling or "").strip().strip("`'\"").lower()
    if said:
        resolved = _INTERPRETER_ALIASES.get(said)
        if resolved is None and _PYTHON_VERSIONED.match(said):
            resolved = PYTHON
        if resolved is None:
            raise ProtocolError(
                f"INTERPRETER: {spelling.strip()!r} is not one this harness can run, so nothing "
                f"was run. It is {BASH} or {PYTHON}. Send the step again with one of those, or "
                "write the work as shell commands."
            )
        return resolved

    shebang = body.lstrip().splitlines()[0] if body.strip() else ""
    if shebang.startswith("#!"):
        return PYTHON if "python" in shebang.lower() else BASH
    for extension, interpreter in _SCRIPT_EXTENSIONS.items():
        if path.strip().lower().endswith(extension):
            return interpreter
    return BASH


def _resolve_host(spelling: str, action: str, fleet) -> str:
    """The server this step is for, spelled the way the fleet spells it.

    Refusing an unnamed step costs one round trip. Guessing costs whichever server
    was wrong, and on a replicated pair the two are configured differently on
    purpose - which is the entire reason there are two of them.
    """
    text = (spelling or "").strip()
    if fleet is None:
        return text
    target = fleet.find(text) if text else None
    if target is not None:
        return target.name
    if action in (DONE, ABORT):
        return ""  # nothing is run, so nothing needs a server
    if len(fleet) == 1:
        # One server: there is nowhere else the step could go, so a HOST: line
        # that names the machine's own hostname, or none at all, is not worth a
        # round trip.
        return fleet.only.name
    names = ", ".join(fleet.names)
    if not text:
        raise ProtocolError(
            f"This step has no HOST: line, and this run has {len(fleet)} servers. Nothing was "
            f"run. Add a HOST: line naming the server this step is for - one of: {names} - "
            "and send the step again."
        )
    raise ProtocolError(
        f"HOST: {text} is not a server in this run, so nothing was run. The servers are: "
        f"{names}. Send the step again with one of those names, spelled exactly."
    )


def _check(line: str, fleet) -> Check:
    """One VERIFY: line, with its optional [name] scope resolved."""
    match = _VERIFY_HOST.match(line)
    if not match:
        return Check(command=line)
    spelling, command = match.group(1), match.group(2).strip()
    if fleet is None:
        return Check(command=command, host=spelling)
    target = fleet.find(spelling)
    if target is None:
        raise ProtocolError(
            f"VERIFY: [{spelling}] names no server in this run. The servers are: "
            f"{', '.join(fleet.names)}. Send the step again, or leave the [name] off to "
            "check every server."
        )
    return Check(command=command, host=target.name)


def _read_block(lines: list[str], index: int, terminator: str) -> tuple[str, int]:
    body: list[str] = []
    while index < len(lines):
        line = lines[index]
        # startswith, not equality: models write "COMMAND_END \n" or "CONTENT_END:"
        # and an exact match would swallow the whole rest of the reply.
        if line.strip().lower().startswith(terminator):
            index += 1
            break
        # A terminator the model forgot entirely must not eat the next step either.
        match = _KEY_LINE.match(line)
        if match and match.group(1).lower() == "action":
            break
        body.append(line)
        index += 1
    return "\n".join(body).strip("\n"), index


def _lines_below(lines: list[str], index: int) -> list[str]:
    """The lines under a key line, up to the next thing the parser recognises.

    Read without consuming: whether they belong to the command is the caller's
    decision, and the ones it does not take have to stay where they are so the
    rest of the parse still sees them.
    """
    body: list[str] = []
    while index < len(lines):
        line = lines[index]
        if _FENCE.match(line) or line.strip().lower().rstrip(":") in _BLOCKS:
            break
        match = _KEY_LINE.match(line)
        if match and match.group(1).lower() in _STEP_OPENERS:
            break
        body.append(line)
        index += 1
    return body


def _strip_blank(body: list[str]) -> list[str]:
    """The lines with the blank ones at the end removed."""
    while body and not body[-1].strip():
        body = body[:-1]
    return body


def _incomplete(command: str) -> bool:
    """Whether this one line cannot be the whole command.

    Three shapes, each of them something a shell reads as unfinished, so each of
    them a reason to believe the lines below are the rest of the command rather
    than commentary about it.
    """
    text = command.rstrip()
    if _CONTINUES.search(text):
        return True
    quoted = [match.span() for match in _QUOTED.finditer(text)]
    if any(not any(start < match.start() < end for start, end in quoted)
           for match in _HEREDOC.finditer(text)):
        return True
    return _unbalanced(text)


def _unbalanced(text: str) -> bool:
    """Whether a quote is left open - `mysql -e "SELECT` with the query below it.

    Each kind of quote is counted with the other kind's pairs removed, so that the
    apostrophe in `echo "it's fine"` is read as the text it is.
    """
    return (re.sub(r'"[^"]*"', "", text).count("'") % 2 == 1
            or re.sub(r"'[^']*'", "", text).count('"') % 2 == 1)


def _read_indented(lines: list[str], index: int) -> tuple[str, int]:
    """The lines up to the next recognised key, dedented.

    Dedented because a YAML-style body arrives indented, and indentation is
    meaningful once the text is handed to a shell as a script.
    """
    body: list[str] = []
    while index < len(lines):
        line = lines[index]
        if _FENCE.match(line) or line.strip().lower().rstrip(":") in _BLOCKS:
            break
        match = _KEY_LINE.match(line)
        if match and match.group(1).lower() in _KEYS:
            break
        body.append(line)
        index += 1
    return textwrap.dedent("\n".join(body)).strip("\n").strip(), index
