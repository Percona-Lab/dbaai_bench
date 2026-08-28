"""Classifies a proposed command before it runs.

Three outcomes: ALLOW runs it, CONFIRM needs a human to say yes, BLOCK never
runs and is fed back to the model so it can pick another approach.

A file being written and a script being run go through the same rules. A shell
script is judged line by line, so `rm -rf /var/lib/mysql` is the same event
wherever it is written down; a python script is parsed and judged by its calls,
which is a separate set of rules further down because read as shell a python file
is line after line of unknown program names.

This is a safety net over a well-known set of footguns, not a sandbox. The model
is driving a root shell; a determined or unlucky sequence of individually
harmless commands can still break a server. Read the plan.
"""

from __future__ import annotations

import ast
import re
import shlex
from dataclasses import dataclass

ALLOW = "allow"
CONFIRM = "confirm"
BLOCK = "block"


@dataclass
class Verdict:
    level: str
    reason: str = ""

    @property
    def blocked(self) -> bool:
        return self.level == BLOCK

    @property
    def needs_approval(self) -> bool:
        return self.level == CONFIRM


# Removing any of these outright leaves an unbootable or unusable machine.
CRITICAL_PATHS = {
    "/", "/*", "/bin", "/boot", "/dev", "/etc", "/home", "/lib", "/lib32", "/lib64",
    "/opt", "/proc", "/root", "/sbin", "/srv", "/sys", "/usr", "/var",
}
# Removing these destroys data but is sometimes exactly what a reinstall needs.
DATA_PATHS = {
    "/var/lib/mysql", "/var/lib/postgresql", "/var/lib/mongodb", "/var/lib/redis",
    "/var/lib/valkey", "/etc/mysql", "/etc/postgresql", "/etc/mongod.conf",
    "/etc/valkey", "/etc/redis", "/var/log",
}
# Programs that open a UI or a pager and would hang forever on a closed stdin.
ALWAYS_INTERACTIVE = {
    "vi", "vim", "nano", "emacs", "pico", "joe", "less", "more", "top", "htop",
    "man", "mysql_secure_installation", "mariadb-secure-installation", "visudo",
    "ftp", "telnet",
}
# Shells that take a script as an argument; their -c payload is classified too.
SHELLS = {"bash", "sh", "dash", "zsh", "ksh"}
# These are only a problem with no arguments, when they drop into a REPL.
REPL_WITHOUT_ARGS = {
    "python", "python3", "node", "irb", "bash", "sh", "screen", "tmux",
    "sqlite3", "redis-cli", "valkey-cli", "mongosh", "mongo",
}
# Database servers that stay in the foreground unless told otherwise: started by
# hand they hold the connection until the command timeout kills them, and the
# service never ends up under systemd where the next step expects to find it.
# Daemon names only, never a name that is also an account: `sudo -iu postgres psql`
# resolves to `postgres` here, and reading that as a server start would refuse one
# of the ordinary ways to reach the client.
FOREGROUND_SERVERS = {
    "mysqld", "mariadbd", "mongod", "valkey-server", "redis-server",
}
# The flags that make one of those return instead: print something and exit, or
# detach and leave the server running. `--print-defaults` prints the options the
# config files actually set and stops before the server starts, which is how a step
# finds out that its drop-in never landed - a recorded run asked three times, was
# refused three times, and went looking for the setting the long way round. Two more
# runs used it against mariadbd, and were allowed only because this rule did not exist
# yet: it is ordinary practice, not a way of starting a server.
BACKGROUNDING_FLAGS = {"--version", "-V", "--help", "--fork", "--daemonize", "--print-defaults"}
# And the flags that hand the daemon one job and end when it is done: these never
# open the port or hold the terminal, so the reason above does not apply to them.
# `--initialize`/`--initialize-insecure` build a data directory and exit, which is
# how a server gets started over data it cannot use - a recorded run hit
# `Invalid MySQL server upgrade: Cannot upgrade from 80046 to 90702` and this was the
# way out of it - and refuse outright on a directory that already has files in it, so
# the destructive half is the `rm`/`mv` that comes first and is classified on its own.
# `--validate-config` reads the config, says what is wrong with it and stops, which is
# the one way to find a bad setting without restarting the service to discover it.
ONE_SHOT_FLAGS = {"--initialize", "--initialize-insecure", "--validate-config"}
# `mongod --outputConfig` belongs with them: it resolves the config file and the command
# line, prints the result as YAML and exits, which is how a step finds out what the server
# will actually read. A version that does not have the flag rejects it and exits too, so
# either way nothing starts.
ONE_SHOT_FLAGS |= {"--outputConfig"}
# Some of those listings are asked for with a value rather than a flag, and a value is two
# tokens, which no set of flag names can match. `--setParameter help` is the recorded one:
# mongod answers it with the server parameters it accepts and stops, and a build that does
# not treat `help` as a listing rejects it as a parameter name and stops as well - what it
# never does is open a port. Two recorded blocks, both piping that listing into grep to
# find which knob a server that would not start might have (`tcmalloc`, `rseq`, `kernel`),
# and both were refused a listing on the grounds that it was a server start.
LISTING_VALUES = {("--setParameter", "help")}
# A flag is not the only way one of them returns. `/usr/bin/ldd` is a shell script that
# sets this variable and runs the binary: the dynamic loader prints the shared libraries
# it would load and exits before main, so `LD_TRACE_LOADED_OBJECTS=1 /path/to/mysqld` -
# what ldd runs, and what a step writes by hand when ldd is not installed - opens no
# port and holds nothing. Reading it as a server start refuses the ordinary way to find
# out why a tarball build will not run: which library it is missing.
LOADER_TRACE = "LD_TRACE_LOADED_OBJECTS"
# valkey-cli / redis-cli options that consume the next token, so what follows is
# not the command being sent to the server.
CLI_VALUE_FLAGS = {
    "-h", "-p", "-a", "-n", "-u", "-s", "-t", "-r", "-i", "-d",
    "--user", "--pass", "--socket", "--timeout", "--pattern", "--rdb",
}
# Cache commands that stream until interrupted. There is no interrupt here, so
# they run to the command timeout and the step comes back with nothing useful.
STREAMING_CLI = {"monitor", "subscribe", "psubscribe", "ssubscribe"}
# Prefixes to strip before looking at what is actually being run. `runuser` and
# `setpriv` are here because they are how a step drops to the database account -
# `runuser -u mysql -- /usr/sbin/mariadbd --validate-config` is recorded - and while
# they were unknown the program on the far side of them was never classified at all,
# which made either of them a way past every rule below.
WRAPPERS = {"sudo", "doas", "nohup", "time", "nice", "ionice", "eatmydata", "command",
            "exec", "env", "runuser", "setpriv"}
# Wrapper options that consume the next token, so it is not the program name.
WRAPPER_VALUE_FLAGS = {"-u", "--user", "-g", "--group", "-U", "-C", "-p", "-D", "--chdir", "-n20"}
# The two that take a shell script instead of a command, and start an interactive
# shell when given neither. `su` is deliberately not a wrapper: `su postgres -c ...`
# names the account where a wrapper would name the program, and reading `postgres` as
# the program is how a client session gets waved through.
SCRIPT_WRAPPERS = {"su", "runuser"}
# Files that carry the account database or the sudo rules: replacing one wholesale,
# by any route, locks the machine or hands it over.
PROTECTED_FILES = {"/etc/passwd", "/etc/shadow", "/etc/group", "/etc/gshadow", "/etc/sudoers"}

_OPERATORS = ";|&\n"

# (pattern, level, reason). Checked in order, first match wins.
_RULES: list[tuple[re.Pattern[str], str, str]] = [
    # ---- never, under any circumstances -----------------------------------
    (re.compile(r":\s*\(\s*\)\s*\{.*\|.*&.*\}\s*;?\s*:"), BLOCK, "fork bomb"),
    (re.compile(r"\bmkfs(\.\w+)?\b|\bwipefs\b|\bfdisk\b|\bparted\b|\bsgdisk\b"), BLOCK,
     "formats or repartitions a disk"),
    (re.compile(r"\bdd\b[^|;&]*\bof=/dev/(sd|vd|nvme|xvd|hd)"), BLOCK,
     "writes raw bytes over a block device"),
    (re.compile(r"\bshred\b[^|;&]*\s/dev/"), BLOCK, "destroys a block device"),
    (re.compile(r">\s*/dev/(sd|vd|nvme|xvd|hd)\w*"), BLOCK, "redirects output over a block device"),
    (re.compile(r">\s*/etc/(passwd|shadow|group|gshadow)\b"), BLOCK,
     "overwrites the account database"),
    (re.compile(r"\bsystemctl\b[^|;&]*\b(stop|disable|mask)\b[^|;&]*\bssh(d)?\b"), BLOCK,
     "stopping sshd would end this session and lock the harness out"),
    (re.compile(r"\bkill(all)?\b[^|;&]*\bsshd\b"), BLOCK,
     "killing sshd would end this session"),
    (re.compile(r"\bnc\b[^|;&]*\s-\w*e|\bncat\b[^|;&]*--exec"), BLOCK,
     "netcat with command execution is a reverse shell"),
    (re.compile(r"\brm\b[^|;&]*\s--no-preserve-root"), BLOCK, "explicitly targets /"),

    # ---- needs a human ----------------------------------------------------
    (re.compile(r"\b(reboot|shutdown|poweroff|halt)\b|\binit\s+[06]\b|\bsystemctl\s+(reboot|poweroff)\b"),
     CONFIRM, "restarts or powers off the server, ending this run"),
    (re.compile(r"\bsystemctl\b[^|;&]*\brestart\b[^|;&]*\bssh(d)?\b"), CONFIRM,
     "restarting sshd risks the connection"),
    (re.compile(r"\bufw\b[^|;&]*\b(disable|reset)\b|\biptables\b[^|;&]*\s-[FX]\b"
                r"|\bnft\b[^|;&]*flush\s+ruleset|\biptables\b[^|;&]*-P\s+INPUT\s+DROP"),
     CONFIRM, "changes the firewall wholesale and could cut off SSH"),
    (re.compile(r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(ba|z|d)?sh\b"), CONFIRM,
     "pipes a downloaded script straight into a shell"),
    (re.compile(r"\bapt(-get)?\b[^|;&]*\b(purge|remove|autoremove)\b|\bdpkg\b[^|;&]*--purge"
                r"|\b(yum|dnf)\b[^|;&]*\bremove\b"), CONFIRM, "removes installed packages"),
    (re.compile(r"(?i)\bdrop\s+(database|schema|table|user|role)\b"), CONFIRM,
     "drops a database object"),
    (re.compile(r"(?i)\btruncate\s+table\b"), CONFIRM, "empties a table"),
    # MongoDB destroys through method calls rather than statements, so none of the
    # SQL rules above see any of this.
    (re.compile(r"(?i)\.\s*(dropDatabase|dropUser|dropAllUsers(?:FromDatabase)?|dropIndexe?s?)\s*\("),
     CONFIRM, "drops a MongoDB database, user or index"),
    (re.compile(r"(?i)\.\s*drop\s*\(\s*\)"), CONFIRM, "drops a MongoDB collection"),
    (re.compile(r"(?i)\.\s*(deleteMany|remove)\s*\(\s*\{\s*\}"), CONFIRM,
     "deletes every document in a MongoDB collection"),
    (re.compile(r"(?i)\.\s*shutdownServer\s*\(|\badminCommand\s*\(\s*\{\s*['\"]?shutdown"),
     CONFIRM, "shuts the MongoDB server down"),
    # Valkey and Redis take the same shape: one word to the client wipes the keyspace.
    (re.compile(r"(?i)\b(?:valkey-cli|redis-cli)\b[^|;&]*\b(?:flushall|flushdb)\b"), CONFIRM,
     "empties the keyspace"),
    (re.compile(r"(?i)\b(?:valkey-cli|redis-cli)\b[^|;&]*\bshutdown\b"), CONFIRM,
     "shuts the cache server down, and NOSAVE discards everything not yet on disk"),
    (re.compile(r"(?i)\b(?:valkey-cli|redis-cli)\b[^|;&]*\bdebug\s+(?:segfault|panic)\b"), CONFIRM,
     "deliberately crashes the server"),
    # The closing quote of `mysql -e '...'` counts as the end of the statement.
    (re.compile(r"(?i)\bdelete\s+from\s+[\w.`\"\[\]]+\s*['\"`]?\s*(;|$)"), CONFIRM,
     "DELETE with no WHERE clause"),
    (re.compile(r"(?i)bind-address\s*=\s*(0\.0\.0\.0|\*)|listen_addresses\s*=\s*'?\s*\*"),
     CONFIRM, "exposes the database on every interface"),
    # The same setting under three more names: `bind 0.0.0.0` is valkey.conf and
    # redis.conf, bindIp/bindIpAll is mongod.conf, and each is written by echo or a
    # heredoc as often as by an editor.
    # The lookahead stops at a longer number and not at any non-space, so the
    # `s/^bind 127.0.0.1/bind 0.0.0.0/` form is caught too. The cost is that reading
    # the setting - `grep 'bind 0.0.0.0' valkey.conf` - asks for approval as well,
    # which is a cheap question next to a cache left open to the internet.
    (re.compile(r"(?i)\bbind\s+(?:\S+\s+)*(?:0\.0\.0\.0|\*)(?![\w.])"), CONFIRM,
     "binds the cache to every interface"),
    (re.compile(r"(?i)\bbindIp(?:All)?\s*[:=]\s*['\"]?(?:0\.0\.0\.0|\*|true)"), CONFIRM,
     "exposes MongoDB on every interface"),
    (re.compile(r"(?i)\bprotected-mode\s+no\b|\bconfig\s+set\s+protected-mode\s+no\b"), CONFIRM,
     "turns off the guard that keeps an unconfigured cache from answering the network"),
    (re.compile(r"(?i)\bauthorization\s*:\s*['\"]?disabled|\bmongod\b[^|;&]*--noauth\b"), CONFIRM,
     "runs MongoDB with access control off"),
    (re.compile(r"(?i)\bconfig\s+set\s+requirepass\s*(?:''|\"\")(?![^\s;&|])"), CONFIRM,
     "removes the cache password"),
    (re.compile(r"(?i)\b0\.0\.0\.0/0\b|\b::/0\b"), CONFIRM, "grants access from any address"),
    (re.compile(r"(?i)\b(host|hostssl)\s+all\s+all\s+\S+\s+trust\b|\bmethod\s*=\s*trust\b"),
     CONFIRM, "configures password-less database access"),
    (re.compile(r"(?i)@\s*'%'|@\"%\"|identified\s+by\s+''"), CONFIRM,
     "grants a database account from any host, or with an empty password"),
    (re.compile(r"authorized_keys"), CONFIRM, "changes who can log in over SSH"),
    # `passwd`/`chpasswd` are matched by program name below instead: as a word this
    # pattern also caught `cat /etc/passwd`, `getent passwd mysql` and every `awk -F:`
    # over the account file - read-only steps, asked about as if they were setting a
    # password, and the CONFIRM returned here hid what the rest of the command was doing.
    (re.compile(r"\busermod\b[^|;&]*\s-p\b"), CONFIRM, "changes a system account password"),
    (re.compile(r"\bvisudo\b|>\s*/etc/sudoers|\bsudoers\.d/"), CONFIRM, "changes sudo rules"),
    (re.compile(r"\bchmod\b[^|;&]*\s777\b|\bchmod\b[^|;&]*-R[^|;&]*\s/(etc|var|usr)\b"),
     CONFIRM, "opens up permissions broadly"),
    (re.compile(r"\bchown\b[^|;&]*-R[^|;&]*\s/(\s|$|etc|usr|var\s)"), CONFIRM,
     "recursive ownership change high up the tree"),
    (re.compile(r"\bjournalctl\b[^|;&]*--vacuum|\bhistory\s+-c\b|>\s*/var/log/"), CONFIRM,
     "erases logs"),
    (re.compile(r"\b(scp|rsync|mysqldump|mariadb-dump|pg_dump(all)?|mongodump)\b[^|;&]*\s\S+@"
                r"|\b(nc|curl|wget)\b[^|;&]*--upload-file"), CONFIRM, "sends data off this server"),
    # ssh to somewhere else, whether given a command or fed by a pipe. Written
    # this way so sshd and ssh-keygen do not match.
    (re.compile(r"(?<![-\w])ssh(?![-\w])[^|;&]*\s\S+@\S+"), CONFIRM,
     "reaches out to another server"),
    (re.compile(r"\bsetenforce\s+0\b|\baa-disable\b|SELINUX=disabled"), CONFIRM,
     "disables a mandatory access control system"),
    (re.compile(r"\bkill\b[^|;&]*-9[^|;&]*\b(mysqld|postgres|mariadbd|mongod|valkey-server"
                r"|redis-server)\b"), CONFIRM,
     "hard-kills a running database, risking an unclean shutdown"),
]


def classify(command: str, _depth: int = 0, _quotes: bool = True, _executed: bool = True,
             _background: bool = False) -> Verdict:
    """Judge a shell command about to be run as root.

    `_executed` is false for text that is only being written down - the body of a
    heredoc that lands in a file, a file body handed to write_file - where the few
    rules about what would hang this step do not apply, because nothing here runs it.

    `_background` is true when the caller already knows this whole text runs behind an
    `&`: `bash -c 'mysqld' &` backgrounds the server just as surely as `mysqld &` does,
    and the payload is judged by recursing here, where the `&` is no longer in sight.
    """
    text = command.strip()
    if not text:
        return Verdict(BLOCK, "empty command")

    for pattern, level, reason in _RULES:
        if pattern.search(text):
            return Verdict(level, reason)

    # Everything below reasons about shell structure, so the text has to be
    # structure the shell would agree with. An unclosed quote means it is not:
    # bash refuses the command, but only after running whatever came before the
    # broken line, and the guard's own view of where one command ends and the
    # next begins stops matching bash's. Refuse instead of guessing.
    if _quotes:
        dangling = _unterminated_quote(text)
        if dangling:
            return Verdict(BLOCK, f"unclosed {dangling} quote - the shell cannot parse this and would "
                                  "fail partway through; fix the quoting and send it again")

    saw_command = False
    for segment, background in _segments(text):
        verdict = _judge_segment(segment, _depth, _executed, _background or background)
        if verdict.level != ALLOW:
            return verdict
        saw_command = saw_command or bool(segment.strip())

    # A reply of "|" or "&&" alone: nothing to run, and worth saying so rather
    # than letting the far end answer with a syntax error.
    if not saw_command:
        return Verdict(BLOCK, "that is shell punctuation, not a command")

    return Verdict(ALLOW)


def _segments(text: str) -> list[tuple[str, bool]]:
    """The separate commands on a line, split on the shell's own operators.

    Quote-aware, because a plain split is not: `grep -E 'mariadb|mysql'` comes
    apart into three pieces, one of which reads exactly like a bare `mysql`
    client session - and a read-only inspection command gets blocked for a pipe
    that is not a pipe.

    Each segment comes back with the one thing the split would otherwise destroy:
    whether the operator that ended it was a single `&`, which is the difference
    between a command that holds this step and one that returns immediately.
    """
    segments: list[tuple[str, bool]] = []
    buffer: list[str] = []
    quote = ""
    index = 0
    while index < len(text):
        char = text[index]
        if quote:
            buffer.append(char)
            if char == quote:
                quote = ""
            index += 1
            continue
        if char == "\\" and index + 1 < len(text):
            buffer.append(text[index:index + 2])  # escaped character, operators included
            index += 2
            continue
        if char in "'\"":
            quote = char
            buffer.append(char)
            index += 1
            continue
        # `2>&1` and `&>log` are redirections, not the background operator.
        if char == "&" and text[index + 1:index + 2] != "&":
            previous = "".join(buffer).rstrip()[-1:]
            if previous == ">" or text[index + 1:index + 2] == ">":
                buffer.append(char)
                index += 1
                continue
        if char in _OPERATORS:
            # A lone `&` backgrounds what came before it. `&&` is a conditional and
            # backgrounds nothing; the redirection cases were taken above.
            background = char == "&" and text[index + 1:index + 2] != "&"
            segments.append(("".join(buffer), background))
            buffer = []
            while index < len(text) and text[index] in _OPERATORS:
                index += 1  # `||`, `&&` and a run of `;` all separate the same way
            continue
        buffer.append(char)
        index += 1
    segments.append(("".join(buffer), False))  # nothing followed it, so nothing backgrounded it
    return segments


# `<<EOF`, `<<-EOF`, `<<'EOF'`: the body that follows is data, not shell text. The
# lookarounds are the ones protocol.py uses for the same job: `<<<word` is a here-string,
# complete on its own, and read as a heredoc named `word` it would swallow the rest of a
# script as that heredoc's body.
_HEREDOC = re.compile(r"(?<!<)<<(?!<)-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")


def _unterminated_quote(text: str) -> str:
    """The quote character left open at the end of the command, or "".

    Heredoc bodies are skipped first: an apostrophe in a config comment is not
    an unclosed quote, and my.cnf bodies are full of prose.
    """
    body = _without_heredoc_bodies(text)
    quote = ""
    index = 0
    while index < len(body):
        char = body[index]
        if quote:
            if char == quote:
                quote = ""
            elif char == "\\" and quote == '"':
                index += 1  # inside '' a backslash is literal; inside "" it escapes
            index += 1
            continue
        if char == "\\":
            index += 2
            continue
        if char in "'\"":
            quote = char
            index += 1
            continue
        if char == "#" and (index == 0 or body[index - 1] in " \t\n;&|("):
            # A comment runs to the end of the line, quotes in it included.
            newline = body.find("\n", index)
            if newline < 0:
                break
            index = newline + 1
            continue
        index += 1
    return quote


def _without_heredoc_bodies(text: str) -> str:
    """The command with the contents of any heredoc removed."""
    if "<<" not in text:
        return text
    kept: list[str] = []
    pending: list[str] = []
    for line in text.splitlines():
        if pending:
            if line.strip() == pending[0]:
                pending.pop(0)
            continue  # body line: data, so it says nothing about quoting
        kept.append(line)
        pending.extend(match.group(2) for match in _HEREDOC.finditer(line))
    return "\n".join(kept)


def _judge_segment(segment: str, depth: int = 0, executed: bool = True,
                   background: bool = False) -> Verdict:
    tokens = _tokens(segment)
    if not tokens:
        return Verdict(ALLOW)

    program = tokens[0]
    args = tokens[1:]

    if program in ALWAYS_INTERACTIVE:
        return Verdict(BLOCK, f"{program} waits for input and would hang; write the file or use a flag instead")

    # `bash -c '<script>'` would otherwise hide whatever it runs from every rule
    # above, so the payload is judged on its own.
    if program in SHELLS and depth < 3:
        inner = _shell_payload(args)
        if inner:
            return classify(inner, depth + 1, _executed=executed, _background=background)

    # `su postgres -c '<script>'` is `bash -c` in another hat, and it was a way past
    # every rule here: the script was never looked at, so `su -c 'rm -rf /'` classified
    # as one unknown program. The `--` form resolves through the wrapper stripping
    # instead; this is the string form. With neither, both of these open an interactive
    # shell as that user and sit there until the command timeout.
    if program in SCRIPT_WRAPPERS:
        inner = _shell_payload(args)
        if not inner:
            return Verdict(BLOCK, f"{program} without a command opens an interactive shell "
                                  f"and would hang; use {program} ... -c '<command>'")
        if depth < 3:
            return classify(inner, depth + 1, _executed=executed, _background=background)

    if program in REPL_WITHOUT_ARGS and not args:
        return Verdict(BLOCK, f"bare {program} opens a REPL and would hang; pass it something to run")

    # adduser is interactive by default - it asks for a password and the GECOS
    # fields - but is perfectly scriptable once told not to.
    if program == "adduser":
        quiet = _any_flag(args, {"--disabled-password", "--disabled-login", "--system"})
        if not (quiet and _any_flag(args, {"--gecos", "--system"})):
            return Verdict(BLOCK, "adduser prompts for a password and full name; use "
                                  "adduser --system, or --disabled-password --gecos '', or useradd")

    if program == "crontab" and _any_flag(args, {"-e"}):
        return Verdict(BLOCK, "crontab -e opens an editor; write to /etc/cron.d instead")

    if program in {"passwd", "chpasswd"}:
        return Verdict(CONFIRM, "changes a system account password")

    if program == "rm":
        return _judge_rm(args)

    if program == "mv":
        return _judge_move(args)

    # A package install that was not told to assume yes will stop at the first
    # prompt, and stdin is closed, so it fails in a confusing way.
    if program in {"apt", "apt-get", "aptitude"} and _has_word(args, "install"):
        # Unless it is not really an install. -s prints what would happen, and
        # --print-uris prints the .deb URLs it would fetch; both change nothing and
        # both return before apt reaches its prompt, so there is nothing to stall
        # on, and blocking them refuses the cheapest way to look ahead - which
        # repository a version would come from, what else it would pull in. A
        # recorded run asked --print-uris where percona-server-server lived, was
        # refused, and spent the next step downloading 134MB of .deb to find out.
        lookahead = {"-s", "--simulate", "--dry-run", "--just-print", "--no-act", "--recon",
                     "--print-uris"}
        if not _any_flag(args, {"-y", "--yes", "--assume-yes", "-q", "-qq"} | lookahead):
            return Verdict(BLOCK, "apt install without -y will stall on a prompt")

    # A client asked for help prints its options and exits like the server does, and
    # `mysql --help` prints the config files it read on the way - which is the cheapest
    # answer to "is this client even reading my.cnf". Recorded once, refused as a client
    # session. `-?` is the same flag; `-h` is deliberately absent, it means host here.
    if program in {"mysql", "mariadb"} and not _any_flag(
            args, {"-e", "--execute", "-f", "--version", "-V", "--help", "-?", "--print-defaults"}):
        if "<" not in segment and "<<" not in segment:
            return Verdict(BLOCK, "bare mysql opens a client session; use mysql -e '<sql>'")

    if program == "psql" and not _any_flag(args, {"-c", "--command", "-f", "--file", "-l", "--list",
                                                 "--version", "-V", "--help", "-?"}):
        if "<" not in segment and "<<" not in segment:
            return Verdict(BLOCK, "bare psql opens a client session; use psql -c '<sql>'")

    # mongosh takes a script instead of a statement, and a .js path counts as one.
    if program in {"mongosh", "mongo"} and not _any_flag(args, {"--eval", "-e", "-f", "--file",
                                                                "--version", "--help"}):
        script = any(arg.endswith((".js", ".mongodb")) for arg in args)
        if not script and "<" not in segment:
            return Verdict(BLOCK, f"bare {program} opens a shell session; use {program} --eval '<js>'")

    # `-v` is the cache client's version flag, not verbosity, and it prints and
    # exits like the server's - so the no-command rule below would otherwise block
    # the one call that answers "is the client even there".
    if program in {"valkey-cli", "redis-cli"} and not _any_flag(args, {"--version", "-v", "--help"}):
        words = _cli_words(args)
        if not words:
            return Verdict(BLOCK, f"{program} with no command opens a REPL; pass the command, "
                                  f"e.g. {program} PING")
        if words[0].lower() in STREAMING_CLI or _any_flag(args, {"--stat"}):
            return Verdict(BLOCK, f"{words[0]} streams until interrupted and nothing here can "
                                  "interrupt it; query the state instead")

    # Only where it would actually hang. In a file body the same line is the point of the
    # file: a unit's ExecStart and a wrapper script's `exec mongod` have to stay in the
    # foreground, that is how systemd supervises a service, and refusing them refused the
    # very thing the rule asks for. Three recorded blocks, and one model left a comment
    # saying the wrapper existed "to bypass the safety guard's detection of mongod in
    # ExecStart" - a rule that teaches models to hide from it is worse than no rule.
    # A trailing `&` answers the reason above - the step returns, nothing hangs - and what
    # is left is real but smaller: the server is running outside systemd, so `systemctl
    # status` disagrees with `ps` for the rest of the run and nothing restarts it. Three
    # recorded blocks were this shape, and two of them were the documented way out of a
    # lost root password: a `--skip-grant-tables --skip-networking` server, which is
    # exactly what systemd will not start for you. So it goes to an operator instead of
    # being refused, and an unattended run with --yes proceeds.
    if executed and program in FOREGROUND_SERVERS \
            and not _any_flag(args, BACKGROUNDING_FLAGS | ONE_SHOT_FLAGS) \
            and not _asks_for_a_listing(args) \
            and not _traces_libraries(segment):
        if background:
            return Verdict(CONFIRM, f"{program} in the background will not hold this step, but it "
                                    f"runs outside systemd, where the next step looks for it")
        # The reason names the bounded form as well as the supervised one, because the
        # recorded steps that hit this were not trying to run a service by hand: they were
        # trying to see why the service would not start, after systemctl had already failed
        # and journalctl had not said enough. `timeout 20 mongod --config ...` answers that
        # and returns on its own, which is why it is allowed - and a reason that only says
        # "use systemctl" to a model whose systemctl is broken teaches it to hide instead.
        return Verdict(BLOCK, f"{program} started this way stays in the foreground until the "
                              f"command timeout; start it with systemctl, or to see why it will "
                              f"not start, bound the direct run: timeout 20 {program} ...")

    if program == "dpkg-reconfigure" and not _any_flag(args, {"-f", "--frontend"}):
        return Verdict(BLOCK, "dpkg-reconfigure is interactive without -f noninteractive")

    return Verdict(ALLOW)


def _judge_rm(args: list[str]) -> Verdict:
    flags = [a for a in args if a.startswith("-")]
    targets = [a for a in args if not a.startswith("-")]
    recursive = any("r" in flag.lower() for flag in flags)

    for target in targets:
        normalised = target.rstrip("/") or "/"
        if target in CRITICAL_PATHS or normalised in CRITICAL_PATHS:
            if recursive or target in {"/", "/*"}:
                return Verdict(BLOCK, f"recursive delete of {target}")
            return Verdict(CONFIRM, f"deletes {target}")
        if normalised in DATA_PATHS or any(normalised.startswith(p + "/") for p in DATA_PATHS):
            return Verdict(CONFIRM, f"deletes database or log data under {target}")
    return Verdict(ALLOW)


def _judge_move(args: list[str]) -> Verdict:
    """A move is a delete of the source, and sometimes a write over the destination.

    For the service that was using it, `mv /var/lib/mysql /var/lib/mysql.bak` is the
    same event as `rm -rf /var/lib/mysql`: the data directory is gone. So it is judged
    the same way, which is also what makes the reinstall path honest - `mysqld
    --initialize-insecure` is allowed because it only builds a directory and exits, and
    the step that empties the old one is where the operator gets asked. Recorded five
    times across three runs, always this move, twice in the same step as the initialise.

    Only what is written here can be judged: `mv "$DATA" "${DATA}.old"`, also recorded,
    names its path in a variable the guard cannot expand, and is not caught.
    """
    paths = [arg for arg in args if not arg.startswith("-")]
    if not paths:
        return Verdict(ALLOW)
    # With -t the destination came from the flag, so every path left is a source.
    into_directory = _any_flag(args, {"-t", "--target-directory"})
    sources = paths if into_directory else paths[:-1]
    destination = "" if into_directory else paths[-1]

    for source in sources:
        normalised = source.rstrip("/") or "/"
        if source in CRITICAL_PATHS or normalised in CRITICAL_PATHS:
            return Verdict(BLOCK, f"moves {source} aside, which leaves the machine unusable")
        if normalised in PROTECTED_FILES:
            return Verdict(BLOCK, f"moves {normalised} away, which locks every account out")
        if normalised in DATA_PATHS or any(normalised.startswith(p + "/") for p in DATA_PATHS):
            return Verdict(CONFIRM, f"moves database or log data out from under {source}")
    if destination.rstrip("/") in PROTECTED_FILES:
        return Verdict(BLOCK, f"{destination.rstrip('/')} must not be replaced wholesale")
    return Verdict(ALLOW)


def classify_file_write(path: str) -> Verdict:
    """Judge a file the model wants to create or overwrite."""
    clean = path.strip()
    if not clean.startswith("/"):
        return Verdict(BLOCK, "write_file needs an absolute path")
    if clean in PROTECTED_FILES:
        return Verdict(BLOCK, f"{clean} must not be overwritten wholesale")
    if clean.startswith(("/dev/", "/proc/", "/sys/", "/boot/")):
        return Verdict(BLOCK, f"{clean} is not a normal configuration file")
    if "authorized_keys" in clean:
        return Verdict(CONFIRM, "changes who can log in over SSH")
    if clean.startswith("/etc/ssh/"):
        return Verdict(CONFIRM, "changes the SSH server configuration")
    if clean.startswith("/etc/sudoers.d/"):
        return Verdict(CONFIRM, "changes sudo rules")
    return Verdict(ALLOW)


# Config text that changes who can reach the database. Checked against file
# bodies, where the command rules above do not apply.
_CONFIG_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?im)^\s*bind-address\s*=\s*(0\.0\.0\.0|\*|::)"),
     "binds the database to every interface"),
    (re.compile(r"(?im)^\s*listen_addresses\s*=\s*'?\s*(\*|0\.0\.0\.0)"),
     "makes PostgreSQL listen on every interface"),
    (re.compile(r"(?im)^\s*(host|hostssl|hostnossl)\s+\S+\s+\S+\s+\S+\s+trust\b"),
     "grants password-less database access in pg_hba.conf"),
    (re.compile(r"(?m)\b0\.0\.0\.0/0\b|\b::/0\b"), "allows connections from any address"),
    (re.compile(r"(?im)^\s*skip-grant-tables"), "disables MySQL authentication entirely"),
    # valkey.conf and redis.conf: space-separated, and one bind line can carry
    # several addresses.
    (re.compile(r"(?im)^\s*bind\s+(?:\S+\s+)*(?:0\.0\.0\.0|\*)(?!\S)"),
     "binds the cache to every interface"),
    (re.compile(r"(?im)^\s*protected-mode\s+no\b"),
     "turns off the guard that keeps an unconfigured cache from answering the network"),
    (re.compile(r"(?im)^\s*requirepass\s*(?:''|\"\")?\s*$"), "leaves the cache with no password"),
    # mongod.conf is YAML, so these sit indented under net: and security:.
    (re.compile(r"(?im)^\s*bindIp(?:All)?\s*:\s*['\"]?(?:0\.0\.0\.0|\*|true)"),
     "exposes MongoDB on every interface"),
    (re.compile(r"(?im)^\s*authorization\s*:\s*['\"]?disabled"),
     "runs MongoDB with access control off"),
    (re.compile(r"(?im)^\s*PermitRootLogin\s+yes|^\s*PasswordAuthentication\s+yes"),
     "loosens SSH authentication"),
]


def classify_file_content(content: str, path: str = "") -> Verdict:
    """Judge a file body for settings that expose the server."""
    for pattern, reason in _CONFIG_RULES:
        if pattern.search(content):
            return Verdict(CONFIRM, reason)
    # Writing a script and then running it would otherwise slip every command
    # rule, since the write is just bytes and the run is just a path. Judged as
    # text that is not running yet, though: see classify.
    kind = script_kind(content, path)
    if kind:
        return classify_script_body(content, kind, executed=False)
    return Verdict(ALLOW)


def script_kind(content: str, path: str = "") -> str:
    """Which interpreter would read this file: "python3", "bash", or "" for neither.

    Asked of a file body because a script reaching the server as bytes and then
    being run by name is the same event as a command, and has to be judged like
    one. Python is named separately rather than lumped in with shell: judged as
    shell a Python script reads as line after line of unknown programs, which is
    to say it is not judged at all.
    """
    first = content.lstrip().splitlines()[0] if content.strip() else ""
    if first.startswith("#!"):
        return "python3" if "python" in first.lower() else "bash"
    clean = path.strip().lower()
    if clean.endswith(".py"):
        return "python3"
    if clean.endswith((".sh", ".bash", ".zsh")):
        return "bash"
    return ""


def looks_like_script(content: str, path: str = "") -> bool:
    return bool(script_kind(content, path))


def classify_script_body(body: str, interpreter: str = "bash", executed: bool = True) -> Verdict:
    """Judge a whole script, by the interpreter that will read it."""
    if "python" in (interpreter or "").lower():
        return classify_python_script(body)
    return classify_script(body, executed)


# Lines that are shell structure rather than a command to judge.
_SCRIPT_NOISE = re.compile(
    r"^\s*(#|$|\}|\{\s*$|fi\b|done\b|esac\b|else\b|elif\b|then\b|do\b|"
    r"\w+\s*\(\s*\)\s*\{?\s*$)"
)


# Where a heredoc body ends up when the line opening it writes it to a file. The
# character class keeps `2>&1` out of it, and the leading `-` keeps tee's own flags out.
_REDIRECT_TARGET = re.compile(r">>?\s*[\"']?([^\s\"'|;&<>]+)")
_TEE_TARGET = re.compile(r"\btee\b\s+(?:(?:-a|--append)\s+)*[\"']?([^\s\"'|;&<>-][^\s\"'|;&<>]*)")


def _written_file(line: str) -> str:
    """The path this line writes to, or "" if its output is not going to a file.

    The first one, so a line opening two heredocs names the first file for both. What
    that costs is a path in a reason, not a verdict: both bodies are still judged.
    """
    for pattern in (_REDIRECT_TARGET, _TEE_TARGET):
        found = pattern.search(line)
        if found:
            return found.group(1)
    return ""


def _continues(line: str) -> bool:
    """Does a backslash at the end of this line join the next one to it?

    An even run of backslashes is escaped backslashes, and ends the line.
    """
    stripped = line.rstrip()
    return (len(stripped) - len(stripped.rstrip("\\"))) % 2 == 1


def classify_script(body: str, executed: bool = True) -> Verdict:
    """Judge a shell script line by line and keep the strictest verdict.

    Line by line rather than whole-body, because several rules are written with
    `[^|;&]*` and would otherwise match across unrelated lines. A line means a
    logical one: a command split over several physical lines with backslashes is
    joined back up first, because taken apart the halves are two commands the script
    never runs. Seven recorded blocks were exactly that, and every one of them was
    wrong - `docker run -d \\` ... `mongod --config x` read as a bare mongod, `gdb
    -batch ... \\` ... `/usr/bin/mongod core` read as a server start when the program
    was gdb, and `mysql ... \\` ... `-e "ALTER USER ..."` read as a client session
    because the statement was on the next line.

    The line number is in the reason because a script is not a command: told only
    that something in sixty lines drops a database, an operator has to find it,
    and the model has to guess which line to rewrite.
    """
    worst = Verdict(ALLOW)
    lines = body.splitlines()
    index = 0
    while index < len(lines):
        number = index + 1
        line = lines[index]
        index += 1
        if _SCRIPT_NOISE.match(line):
            continue
        while _continues(line) and index < len(lines):
            line = line.rstrip()[:-1] + lines[index]  # what the shell does with the pair
            index += 1

        found: list[Verdict] = []
        # No quote check here: a string in a script may legitimately open on one
        # line and close on another, and judging lines in isolation would read
        # every such string as broken.
        verdict = classify(line, _quotes=False, _executed=executed)
        if verdict.level != ALLOW:
            shown = " ".join(line.split())[:80]
            found.append(Verdict(verdict.level,
                                 f"line {number} of the script, {shown!r}: {verdict.reason}"))

        # A heredoc whose body goes to a file is file content, not shell text. Judged
        # as commands, `exec mongod` in the wrapper script a step writes reads as a
        # server start, and a my.cnf reads as whatever its settings happen to look
        # like; judged as a file, the config rules get it right. A heredoc with no
        # file behind it - `mysql <<EOF`, `bash <<EOF` - is text a program will run,
        # so it is left to the rules above exactly as it was.
        terminators = [match.group(2) for match in _HEREDOC.finditer(line)]
        path = _written_file(line) if terminators else ""
        if path:
            for terminator in terminators:
                collected: list[str] = []
                while index < len(lines) and lines[index].strip() != terminator:
                    collected.append(lines[index])
                    index += 1
                index += 1  # step over the terminator, which is not content
                verdict = classify_file_content("\n".join(collected) + "\n", path)
                if verdict.level != ALLOW:
                    # A line number from inside the body counts lines of that file, not
                    # of the script, and one reason saying "script" twice helps nobody.
                    inner = re.sub(r"^line (\d+) of the script,",
                                   r"line \1 of the file,", verdict.reason)
                    found.append(Verdict(verdict.level,
                                         f"line {number} of the script writes {path}: {inner}"))

        for verdict in found:
            if verdict.level == BLOCK:
                return verdict
            if worst.level == ALLOW:
                worst = verdict
    return worst


# ------------------------------------------------------------------- python
#
# Every set below is a dotted name as it resolves after imports, so `sp.run` and
# `from subprocess import run` both arrive here as subprocess.run.

# Calls that hand a string to a shell. What they are given is classified as the
# command it is - the same treatment `bash -c` already gets - because without it
# python is a way past every rule above, and the script action would be a hole
# rather than a feature.
_PY_SHELL = {
    "os.system", "os.popen",
    "os.execl", "os.execlp", "os.execv", "os.execvp", "os.execve",
    "os.spawnl", "os.spawnv", "os.spawnlp", "os.spawnvp",
    "subprocess.run", "subprocess.call", "subprocess.check_call",
    "subprocess.check_output", "subprocess.Popen",
    "subprocess.getoutput", "subprocess.getstatusoutput",
}
# Deletes and moves, mapped onto the very tables the rm and mv rules use, so that
# a python script and a shell script are told the same thing about /var/lib/mysql.
_PY_REMOVE = {"os.remove", "os.unlink", "os.rmdir", "os.removedirs"}
_PY_REMOVE_TREE = {"shutil.rmtree"}
_PY_MOVE = {"shutil.move", "os.rename", "os.renames", "os.replace"}
# Sends data somewhere else. Deliberately not every network call: fetching a
# package or a repository key is ordinary work, and requests.get is the same event
# as the curl this harness recommends. What is asked about is a body leaving.
_PY_SEND = {
    "requests.post", "requests.put", "requests.patch",
    "smtplib.SMTP", "ftplib.FTP", "paramiko.SSHClient", "paramiko.Transport",
}
# Runs text the guard cannot see, so nothing below can judge it. A DBA script has
# no need of any of these, and `exec(base64.b64decode(...))` is the shape this
# stops being able to reason about at all.
_PY_OPAQUE = {"eval", "exec", "compile", "__import__",
              "builtins.eval", "builtins.exec", "builtins.compile"}
# Asks the operator something. These do not hang the way an editor does - with
# stdin on /dev/null, input() raises EOFError on the spot - but the script still
# cannot do what it was written to do, and the traceback is a poor way to find out.
_PY_ASKS = {"input", "raw_input", "getpass.getpass"}
_PY_INTERACTIVE = {"pty.spawn", "os.forkpty"}
# Where a database driver is handed SQL. The receiver is a cursor or a connection
# whose name cannot be known, so these are matched on the method name alone - the
# SQL is what gets judged, exactly as it is inside `mysql -e`.
_PY_SQL = {"execute", "executemany", "executescript"}
# Stands in for a part of a string the guard cannot work out. A bare word, chosen
# so that `rm -rf /var/lib/mysql/DBAUNKNOWN` still reads as a delete under the
# data directory while `rm -rf DBAUNKNOWN` does not read as a path at all.
_PY_UNKNOWN = "DBAUNKNOWN"
_PY_PERCENT = re.compile(r"%[-#0 +]*\d*(?:\.\d+)?[sdrfgxi]")
_PY_BRACES = re.compile(r"\{[^{}]*\}")


def classify_python_script(body: str) -> Verdict:
    """Judge a python script by what it asks the system to do.

    Parsed rather than pattern-matched, and the parse is itself the first
    judgement: a script this cannot read is a script whose danger it cannot
    assess, which is the same reason an unclosed quote is refused above.
    ast.parse builds a tree and executes nothing.

    What is judged is the calls, each mapped onto a rule that already exists: a
    string on its way to a shell is classified as a command, a delete against the
    same path tables as `rm`, a move as `mv`, a file opened for writing by
    classify_file_write, and SQL handed to a driver as the SQL in `mysql -e`. The
    strictest verdict wins, and the line number comes with it.

    Two blind spots, both deliberate. A path the parser cannot fold -
    os.remove(target), where target was built earlier - is not caught, which is
    the same blindness `mv "$DATA" "${DATA}.old"` has in a shell script and is
    documented on _judge_move. A whole command that cannot be read -
    subprocess.run(cmd) - asks the operator instead of passing quietly: that is
    an instruction stream the guard cannot see, which is what `curl | sh` is, and
    it is treated the same way. A fragment it can partly read is not enough to
    ask about, so subprocess.run(["mysql", "-e", sql]) goes straight through.

    pathlib is covered for writes - Path("/etc/passwd").write_text(...) - but not
    for deletes: Path(x).unlink() names its path on an object, and the object is
    what the parser cannot follow.
    """
    try:
        tree = ast.parse(body)
    except SyntaxError as exc:
        where = f" at line {exc.lineno}" if exc.lineno else ""
        return Verdict(BLOCK, f"this is not valid python{where}: {exc.msg}. None of it could be "
                              "judged, so none of it was copied to the server - fix the syntax "
                              "and send it again")
    except (ValueError, RecursionError) as exc:
        return Verdict(BLOCK, f"this python could not be parsed, so it could not be judged: {exc}")

    bound = _python_bindings(tree)
    worst = Verdict(ALLOW)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        verdict = _judge_python_call(node, bound)
        if verdict.level == ALLOW:
            continue
        reason = f"line {node.lineno} of the script, {verdict.reason}"
        if verdict.level == BLOCK:
            return Verdict(BLOCK, reason)
        if worst.level == ALLOW:
            worst = Verdict(CONFIRM, reason)
    return worst


def _python_bindings(tree: ast.AST) -> dict[str, str]:
    """The names the script binds to modules, so `sp.run` resolves to subprocess.run."""
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                # `import os.path` binds os; `import subprocess as sp` binds sp.
                bound[alias.asname or root] = alias.name if alias.asname else root
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                bound[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return bound


def _python_dotted(func: ast.expr, bound: dict[str, str]) -> str:
    """The dotted name being called, with imported aliases resolved."""
    parts: list[str] = []
    node = func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""  # called on a value - connect().cursor() - so there is no module to name
    parts.append(node.id)
    parts.reverse()
    return ".".join([bound.get(parts[0], parts[0]), *parts[1:]])


def _judge_python_call(call: ast.Call, bound: dict[str, str]) -> Verdict:
    dotted = _python_dotted(call.func, bound)
    # The method name on its own, which is all there is when the receiver is a
    # value: cursor.execute(...) and connect().cursor().execute(...) both land here.
    if isinstance(call.func, ast.Attribute):
        tail = call.func.attr
    elif isinstance(call.func, ast.Name):
        tail = call.func.id
    else:
        tail = ""

    if dotted in _PY_OPAQUE:
        return Verdict(BLOCK, f"{dotted}() runs text the guard cannot read, so what it would do "
                              "on the server cannot be judged before it does it; write the work "
                              "out in full instead")
    if dotted in _PY_ASKS:
        return Verdict(BLOCK, f"{dotted}() waits for an answer, and stdin is /dev/null here, so "
                              "it raises EOFError instead of ever getting one; take the value "
                              "from the script itself")
    if dotted in _PY_INTERACTIVE:
        return Verdict(BLOCK, f"{dotted}() starts an interactive session and would hang until "
                              "the step timeout")
    if dotted in _PY_SHELL:
        return _judge_python_shell(dotted, call)
    if dotted in _PY_REMOVE or dotted in _PY_REMOVE_TREE:
        return _judge_python_delete(dotted, call, recursive=dotted in _PY_REMOVE_TREE)
    if dotted in _PY_MOVE:
        return _judge_python_move(dotted, call)
    if dotted in _PY_SEND:
        return Verdict(CONFIRM, f"{dotted}() sends data off this server")
    if dotted in {"open", "io.open", "codecs.open"}:
        return _judge_python_open(call)
    if tail in {"write_text", "write_bytes"}:
        return _judge_python_pathlib_write(call, bound)
    if tail in _PY_SQL:
        return _judge_python_sql(call)
    return Verdict(ALLOW)


def _judge_python_shell(dotted: str, call: ast.Call) -> Verdict:
    """A string on its way to a shell, classified as the command it is."""
    if not call.args:
        return Verdict(ALLOW)
    text, _ = _python_text(call.args[0])
    if not text.replace(_PY_UNKNOWN, "").strip():
        # Nothing of the command is written down, so there is nothing to judge and
        # no way to know what will run. The operator is asked rather than told
        # afterwards; see the note on classify_python_script.
        return Verdict(CONFIRM, f"{dotted}() is handed a command that is built rather than "
                                "written out, so the guard cannot see what will run on the "
                                "server; read the script and decide")
    verdict = classify(text, _quotes=False)
    if verdict.level == ALLOW:
        return Verdict(ALLOW)
    return Verdict(verdict.level, f"{dotted}() runs {text.strip()[:80]!r}: {verdict.reason}")


def _judge_python_delete(dotted: str, call: ast.Call, recursive: bool) -> Verdict:
    """A delete, judged against the same path tables as `rm`."""
    if not call.args:
        return Verdict(ALLOW)
    path, _ = _python_text(call.args[0])
    if not path.strip():
        return Verdict(ALLOW)
    verdict = _judge_rm(["-rf", path] if recursive else [path])
    if verdict.level == ALLOW:
        return Verdict(ALLOW)
    return Verdict(verdict.level, f"{dotted}({path!r}): {verdict.reason}")


def _judge_python_move(dotted: str, call: ast.Call) -> Verdict:
    """A move, judged as `mv` is: a delete of the source, a write of the target."""
    paths = [_python_text(argument)[0] or _PY_UNKNOWN for argument in call.args[:2]]
    if len(paths) < 2 or not any(path != _PY_UNKNOWN for path in paths):
        return Verdict(ALLOW)
    verdict = _judge_move(paths)
    if verdict.level == ALLOW:
        return Verdict(ALLOW)
    return Verdict(verdict.level, f"{dotted}({paths[0]!r}, {paths[1]!r}): {verdict.reason}")


def _judge_python_open(call: ast.Call) -> Verdict:
    """A file opened for writing, judged exactly as a write_file step would be."""
    if not call.args:
        return Verdict(ALLOW)
    path, _ = _python_text(call.args[0])
    mode = ""
    if len(call.args) > 1:
        mode, _ = _python_text(call.args[1])
    for keyword in call.keywords:
        if keyword.arg == "mode":
            mode, _ = _python_text(keyword.value)
    if not any(letter in mode for letter in "wax+"):
        return Verdict(ALLOW)  # opened for reading, which changes nothing
    return _judge_written_path(path, f"opens {path.strip()} for writing")


def _judge_python_pathlib_write(call: ast.Call, bound: dict[str, str]) -> Verdict:
    """Path("/etc/passwd").write_text(...), which no rule above would see."""
    receiver = call.func.value if isinstance(call.func, ast.Attribute) else None
    if not isinstance(receiver, ast.Call):
        return Verdict(ALLOW)
    if _python_dotted(receiver.func, bound) not in {"pathlib.Path", "Path", "pathlib.PurePath"}:
        return Verdict(ALLOW)
    if not receiver.args:
        return Verdict(ALLOW)
    path, _ = _python_text(receiver.args[0])
    return _judge_written_path(path, f"writes {path.strip()}")


def _judge_written_path(path: str, description: str) -> Verdict:
    """One absolute path about to be written, put through the write_file rules."""
    if not path.strip().startswith("/"):
        # Relative, or built from values the parser cannot fold. classify_file_write
        # would refuse it for not being absolute, which is not what is being asked.
        return Verdict(ALLOW)
    verdict = classify_file_write(path)
    if verdict.level == ALLOW:
        return Verdict(ALLOW)
    return Verdict(verdict.level, f"{description}: {verdict.reason}")


def _judge_python_sql(call: ast.Call) -> Verdict:
    """SQL handed to a driver, judged as the SQL inside `mysql -e` already is."""
    if not call.args:
        return Verdict(ALLOW)
    text, _ = _python_text(call.args[0])
    if not text.replace(_PY_UNKNOWN, "").strip():
        return Verdict(ALLOW)
    verdict = classify(text, _quotes=False)
    if verdict.level == ALLOW:
        return Verdict(ALLOW)
    return Verdict(verdict.level, f"executes {text.strip()[:80]!r}: {verdict.reason}")


def _python_text(node: ast.expr) -> tuple[str, bool]:
    """The string this expression stands for, and whether all of it could be read.

    A part the parser cannot fold - a name, a call, an f-string's `{db}` - becomes
    _PY_UNKNOWN rather than abandoning the whole string, so that
    `f"rm -rf /var/lib/mysql/{db}"` is still recognisably a delete under the data
    directory. Nothing is evaluated here: this reads the tree, it does not run it.
    """
    if isinstance(node, ast.Constant):
        return (node.value, True) if isinstance(node.value, str) else (str(node.value), True)
    if isinstance(node, ast.JoinedStr):  # an f-string
        parts, whole = [], True
        for value in node.values:
            text, known = _python_text(value)
            parts.append(text)
            whole = whole and known
        return "".join(parts), whole
    if isinstance(node, ast.FormattedValue):  # the {...} of an f-string
        return _PY_UNKNOWN, False
    if isinstance(node, (ast.List, ast.Tuple)):
        # An argv list, which is the command with its words already separated.
        words, whole = [], True
        for element in node.elts:
            text, known = _python_text(element)
            words.append(text)
            whole = whole and known
        return " ".join(words), whole
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, left_known = _python_text(node.left)
        right, right_known = _python_text(node.right)
        return left + right, left_known and right_known
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):  # "rm -rf %s" % path
        values = node.right.elts if isinstance(node.right, ast.Tuple) else [node.right]
        return _fill(_PY_PERCENT, _python_text(node.left), values)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
            and node.func.attr == "format":  # "rm -rf {}".format(path)
        return _fill(_PY_BRACES, _python_text(node.func.value), node.args)
    if isinstance(node, ast.Name):
        return _PY_UNKNOWN, False
    return "", False


def _fill(holes: re.Pattern, template: tuple[str, bool], values: list[ast.expr]) -> tuple[str, bool]:
    """A template with its arguments put back into it, in order.

    `"rm -rf %s" % path` and `"rm -rf {}".format(path)` are the same sentence as the
    f-string, and a literal argument is as readable as a literal in the template
    itself - so the substitution is done rather than every hole written off as
    unknown. An argument that cannot be read leaves _PY_UNKNOWN behind, and so does
    a hole with no argument to fill it: `%(name)s` against a dict has none.
    """
    text, whole = template
    supplied = iter(_python_text(value) for value in values)

    def one(_match: re.Match) -> str:
        nonlocal whole
        filled, known = next(supplied, (_PY_UNKNOWN, False))
        whole = whole and known
        return filled

    return holes.sub(one, text), whole


# `name=$(prog ...)`, and the backtick spelling of the same thing: an assignment whose
# value is the output of a command. Group 1 is the first word inside the substitution,
# and empty when a space follows the paren. `$((` is arithmetic, not a command.
_ASSIGNED_SUBSTITUTION = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=[\"']?(?:\$\((?!\()|`)\s*(\S*)")


def _tokens(segment: str) -> list[str]:
    """Split a segment and skip past wrappers to the program actually invoked."""
    try:
        tokens = shlex.split(segment, comments=True)
    except ValueError:
        # Unbalanced quotes: fall back to whitespace so the caller still gets
        # a rough look at the command rather than nothing at all.
        tokens = segment.split()

    index = 0
    while index < len(tokens):
        token = tokens[index]
        # The program being run is the first word inside the substitution, not the token
        # after it. Read as a plain assignment and stepped over, ldd's argument was taken
        # for the program instead, so `missing=$(ldd "$PSM/bin/mysqld" | awk ...)` came
        # back as a server start - twice in one recorded run, refusing the ordinary way to
        # find which library a tarball build is missing. The model's answer was to glob the
        # bin directory so that no command line named mysqld directly.
        opens = _ASSIGNED_SUBSTITUTION.match(token)
        if opens and opens.group(1):
            tokens = [opens.group(1)] + tokens[index + 1:]
            if tokens[-1].endswith(")"):
                tokens[-1] = tokens[-1][:-1]  # the substitution's own closing paren
            index = 0
            continue
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", token):
            index += 1  # leading VAR=value assignment
            continue
        wrapper = token.rsplit("/", 1)[-1]
        if wrapper not in WRAPPERS:
            break
        start = index
        index += 1
        # Skip the wrapper's own options. `sudo -u postgres psql` must resolve
        # to psql, not to postgres.
        flags = []
        while index < len(tokens) and tokens[index].startswith("-"):
            flag = tokens[index]
            flags.append(flag)
            index += 1
            if flag in WRAPPER_VALUE_FLAGS and index < len(tokens):
                index += 1
        # `command -v mysql` asks where mysql lives; it does not start it. Strip
        # the wrapper and the lookup reads as a bare client session instead.
        if wrapper == "command" and any(flag in {"-v", "-V"} for flag in flags):
            return tokens[start:]
        # `runuser -u mysql -c 'mysqld'` passes a script, not a program, and one token
        # of shell is not a command name; `runuser -u mysql` with nothing after it opens
        # a shell as that user. Both are the wrapper's own business, so it stays.
        if wrapper in SCRIPT_WRAPPERS and (
                index >= len(tokens) or any(flag in {"-c", "--command"} for flag in flags)):
            return tokens[start:]

    tokens = tokens[index:]
    if tokens:
        tokens[0] = tokens[0].rsplit("/", 1)[-1]  # /usr/bin/vim -> vim
    return tokens


def _shell_payload(args: list[str]) -> str:
    """The script given to `bash -c`, if there is one."""
    for index, arg in enumerate(args):
        if arg in {"-c", "-lc", "-cl", "--command"} and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith("-") and not arg.startswith("--") and "c" in arg[1:] and index + 1 < len(args):
            return args[index + 1]
    return ""


def _cli_words(args: list[str]) -> list[str]:
    """What a valkey-cli/redis-cli line actually asks the server to do.

    Flags and the values they consume are dropped, so `valkey-cli -h 127.0.0.1 -p
    6379` comes back empty - which is a REPL, not a query - while `valkey-cli -n 0
    DBSIZE` comes back as the one word that matters.
    """
    words: list[str] = []
    skip = False
    for arg in args:
        if skip:
            skip = False
            continue
        if arg in CLI_VALUE_FLAGS:
            skip = True
            continue
        if arg.startswith("-"):
            continue
        words.append(arg)
    return words


def _traces_libraries(segment: str) -> bool:
    """Is the loader being asked to list a binary's libraries rather than run it?

    Only the assignments in front of the program count, which is where the shell puts
    them into its environment: `mysqld --a=LD_TRACE_LOADED_OBJECTS=1` is still a server
    start. An empty value is not tracing either - the loader wants a value there - so
    that form keeps the block, which is the safe way round.
    """
    try:
        tokens = shlex.split(segment, comments=True)
    except ValueError:
        tokens = segment.split()
    for token in tokens:
        name, sep, value = token.partition("=")
        if sep and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            if name == LOADER_TRACE and value:
                return True
            continue  # some other assignment, still in front of the program
        if token.rsplit("/", 1)[-1] not in WRAPPERS:
            return False  # the program itself, reached without one
    return False


def _asks_for_a_listing(args: list[str]) -> bool:
    """Is one of LISTING_VALUES here - an option whose value asks for a list, not a start?

    `--setParameter help` and `--setParameter=help` are the same request. A value that
    only begins with `help` is not one: `--setParameter helpText=on` sets a parameter.
    """
    for flag, value in LISTING_VALUES:
        for index, arg in enumerate(args):
            if arg == flag and args[index + 1:index + 2] == [value]:
                return True
            if arg == f"{flag}={value}":
                return True
    return False


def _has_word(args: list[str], word: str) -> bool:
    return word in args


def _any_flag(args: list[str], flags: set[str]) -> bool:
    for arg in args:
        if arg in flags:
            return True
        # --gecos="" is the same flag as --gecos ""
        if arg.startswith("--") and arg.split("=", 1)[0] in flags:
            return True
        # Clustered short flags, e.g. -yq.
        if arg.startswith("-") and not arg.startswith("--"):
            letters = set(arg[1:])
            if any(flag.startswith("-") and len(flag) == 2 and flag[1] in letters for flag in flags):
                return True
    return False
