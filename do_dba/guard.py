"""Classifies a proposed command before it runs.

Three outcomes: ALLOW runs it, CONFIRM needs a human to say yes, BLOCK never
runs and is fed back to the model so it can pick another approach.

This is a safety net over a well-known set of footguns, not a sandbox. The model
is driving a root shell; a determined or unlucky sequence of individually
harmless commands can still break a server. Read the plan.
"""

from __future__ import annotations

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
# detach and leave the server running.
BACKGROUNDING_FLAGS = {"--version", "-V", "--help", "--fork", "--daemonize"}
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


def classify(command: str, _depth: int = 0, _quotes: bool = True) -> Verdict:
    """Judge a shell command about to be run as root."""
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
    for segment in _segments(text):
        verdict = _judge_segment(segment, _depth)
        if verdict.level != ALLOW:
            return verdict
        saw_command = saw_command or bool(segment.strip())

    # A reply of "|" or "&&" alone: nothing to run, and worth saying so rather
    # than letting the far end answer with a syntax error.
    if not saw_command:
        return Verdict(BLOCK, "that is shell punctuation, not a command")

    return Verdict(ALLOW)


def _segments(text: str) -> list[str]:
    """The separate commands on a line, split on the shell's own operators.

    Quote-aware, because a plain split is not: `grep -E 'mariadb|mysql'` comes
    apart into three pieces, one of which reads exactly like a bare `mysql`
    client session - and a read-only inspection command gets blocked for a pipe
    that is not a pipe.
    """
    segments: list[str] = []
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
            segments.append("".join(buffer))
            buffer = []
            while index < len(text) and text[index] in _OPERATORS:
                index += 1  # `||`, `&&` and a run of `;` all separate the same way
            continue
        buffer.append(char)
        index += 1
    segments.append("".join(buffer))
    return segments


# `<<EOF`, `<<-EOF`, `<<'EOF'`: the body that follows is data, not shell text.
_HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")


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


def _judge_segment(segment: str, depth: int = 0) -> Verdict:
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
            return classify(inner, depth + 1)

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
            return classify(inner, depth + 1)

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
        # -s prints what would happen and changes nothing, so there is no prompt
        # to stall on: blocking it would refuse the safest way to look ahead.
        simulate = {"-s", "--simulate", "--dry-run", "--just-print", "--no-act", "--recon"}
        if not _any_flag(args, {"-y", "--yes", "--assume-yes", "-q", "-qq"} | simulate):
            return Verdict(BLOCK, "apt install without -y will stall on a prompt")

    if program in {"mysql", "mariadb"} and not _any_flag(args, {"-e", "--execute", "-f", "--version", "-V"}):
        if "<" not in segment and "<<" not in segment:
            return Verdict(BLOCK, "bare mysql opens a client session; use mysql -e '<sql>'")

    if program == "psql" and not _any_flag(args, {"-c", "--command", "-f", "--file", "-l", "--list", "--version", "-V"}):
        if "<" not in segment and "<<" not in segment:
            return Verdict(BLOCK, "bare psql opens a client session; use psql -c '<sql>'")

    # mongosh takes a script instead of a statement, and a .js path counts as one.
    if program in {"mongosh", "mongo"} and not _any_flag(args, {"--eval", "-e", "-f", "--file", "--version"}):
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

    if program in FOREGROUND_SERVERS and not _any_flag(args, BACKGROUNDING_FLAGS | ONE_SHOT_FLAGS):
        return Verdict(BLOCK, f"{program} started this way stays in the foreground until the "
                              f"command timeout; start it with systemctl instead")

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
    # rule, since the write is just bytes and the run is just a path.
    if looks_like_script(content, path):
        return classify_script(content)
    return Verdict(ALLOW)


def looks_like_script(content: str, path: str = "") -> bool:
    if content.lstrip().startswith("#!"):
        return True
    return path.strip().endswith((".sh", ".bash", ".zsh"))


# Lines that are shell structure rather than a command to judge.
_SCRIPT_NOISE = re.compile(
    r"^\s*(#|$|\}|\{\s*$|fi\b|done\b|esac\b|else\b|elif\b|then\b|do\b|"
    r"\w+\s*\(\s*\)\s*\{?\s*$)"
)


def classify_script(body: str) -> Verdict:
    """Judge a shell script line by line and keep the strictest verdict.

    Line by line rather than whole-body, because several rules are written with
    `[^|;&]*` and would otherwise match across unrelated lines.
    """
    worst = Verdict(ALLOW)
    for line in body.splitlines():
        if _SCRIPT_NOISE.match(line):
            continue
        # No quote check here: a string in a script may legitimately open on one
        # line and close on another, and judging lines in isolation would read
        # every such string as broken.
        verdict = classify(line, _quotes=False)
        if verdict.level == ALLOW:
            continue
        reason = f"line {line.strip()[:80]!r} in the script: {verdict.reason}"
        if verdict.level == BLOCK:
            return Verdict(BLOCK, reason)
        if worst.level == ALLOW:
            worst = Verdict(CONFIRM, reason)
    return worst


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
