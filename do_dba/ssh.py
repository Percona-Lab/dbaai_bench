"""SSH transport: one connection, one command at a time, nothing interactive.

Every command is wrapped so it cannot sit waiting for a human — stdin is
/dev/null and DEBIAN_FRONTEND is set — because a prompt on the far end would
otherwise hang the run until the timeout.

A script is a file rather than a command: staged over SFTP, parsed on the far end
without being run, then run by naming its interpreter. It is still one step, and
its result comes back in the same CommandResult a command's does.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import shlex
import socket
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import paramiko

DEFAULT_COMMAND_TIMEOUT = 300.0
DEFAULT_CONNECT_TIMEOUT = 30.0
# Enough to see what a package manager did, small enough that a runaway command
# cannot exhaust memory. Anything past this is dropped and flagged.
MAX_OUTPUT_BYTES = 256 * 1024
# Said in the harness's own voice, because bash's "unexpected EOF" alone does not
# make clear that nothing ran. Quoting is the commonest cause but not the only
# one: a model that carries on past the end of its own command leaves prose in it
# - `| tail -n 80It appears there is no package for noble.` - and telling it to
# check the quoting sends it looking in the wrong place. The other is a command
# that arrived with its body missing, so the block form is named too: "here-
# document delimited by end-of-file" is bash's way of saying the heredoc had
# nothing under it.
SYNTAX_ERROR_NOTE = (
    "harness: the shell could not parse this command, so none of it was run. Send the "
    "step again with the COMMAND holding nothing but the command - no commentary before "
    "or after it - and check that every quote closes. A command that spans lines, such "
    "as a loop or a heredoc, belongs between COMMAND_BEGIN and COMMAND_END rather than "
    "on a COMMAND: line, which holds one line only."
)
# Where scripts land. Under /tmp because it is writable without sudo on every
# image this runs against, and one directory rather than scattered files so that a
# run leaves one thing to look at afterwards.
SCRIPT_DIR = "/tmp/dba-harness"
SCRIPT_SYNTAX_ERROR_NOTE = (
    "harness: {interpreter} could not parse the script, so none of it ran. The script is "
    "on the server at {path} if you want to look at it. Send the step again with the "
    "syntax fixed."
)
_SCRIPT_SUFFIX = {"python3": ".py"}


def script_path(index: int, interpreter: str) -> str:
    """Where step number `index` will be written on the server.

    Numbered by step rather than hashed or made unique, so that the path in the
    model's reply, the path in the transcript and the path on the server are one
    and the same, and a re-sent step overwrites its own file instead of leaving a
    directory of near-identical scripts to tell apart.
    """
    return f"{SCRIPT_DIR}/step{index:02d}{_SCRIPT_SUFFIX.get(interpreter, '.sh')}"


def script_check(path: str, interpreter: str) -> str:
    """The command that parses the script without running any of it.

    PYTHONDONTWRITEBYTECODE so py_compile checks the syntax without leaving a
    __pycache__ beside the script for whoever looks at the directory afterwards.
    """
    check = (
        f"python3 -m py_compile {shlex.quote(path)}"
        if interpreter == "python3"
        else f"bash -n {shlex.quote(path)}"
    )
    return f"PYTHONDONTWRITEBYTECODE=1 {check}"


class SSHError(RuntimeError):
    """A connection or transport failure to report without a traceback."""


@dataclass
class CommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False
    output_truncated: bool = False

    @property
    def ok(self) -> bool:
        return not self.timed_out and self.exit_code == 0


def wrap_command(command: str) -> str:
    """The command as it is handed to the far end's shell.

    bash -lc, not the account's default shell: the model writes bash, and
    /bin/sh on Debian is dash. Redirecting stdin from /dev/null turns any
    interactive prompt into an immediate EOF instead of a hang.

    pipefail matters more than it looks: models are told to keep output small
    with `| tail -n 30`, and without it a failed apt-get behind a pipe reports
    the exit code of tail, i.e. success.

    The `bash -n` pass parses the whole script before any of it runs. bash reads
    a `-c` string one command at a time, so a quote left open on line 4 runs
    lines 1 to 3 and only then fails - half-applying a step that was approved as
    a whole. `-n` executes nothing, so a malformed step costs a round trip
    instead of a partial change.
    """
    script = (
        "set -o pipefail; export DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a; "
        f"{command}"
    )
    # Through a variable rather than inline twice: the script can be kilobytes of
    # SQL or config, and one copy keeps the request the size it always was.
    return (
        f"__dba_script={shlex.quote(script)}; "
        f'bash -n -c "$__dba_script" || {{ echo {shlex.quote(SYNTAX_ERROR_NOTE)} >&2; exit 2; }}; '
        'bash -lc "$__dba_script" </dev/null'
    )


def key_fingerprint(key: paramiko.PKey) -> str:
    """The SHA256 fingerprint in the same form ssh(1) prints."""
    digest = hashlib.sha256(key.asbytes()).digest()
    return "SHA256:" + base64.b64encode(digest).decode().rstrip("=")


class _AskPolicy(paramiko.MissingHostKeyPolicy):
    """Trust on first use, but only after the operator says so."""

    def __init__(self, ask: Callable[[str, paramiko.PKey], bool]):
        self.ask = ask
        self.accepted: paramiko.PKey | None = None

    def missing_host_key(self, client, hostname, key):
        if not self.ask(hostname, key):
            raise SSHError(f"host key for {hostname} was not accepted")
        self.accepted = key
        client.get_host_keys().add(hostname, key.get_name(), key)


class SSHRunner:
    """Runs shell commands on one remote host over a single SSH connection."""

    def __init__(
        self,
        host: str,
        user: str = "root",
        port: int = 22,
        key_path: str | None = None,
        password: str | None = None,
        passphrase: str | None = None,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        known_hosts: Path | None = None,
        ask_host_key: Callable[[str, paramiko.PKey], bool] | None = None,
    ):
        self.host = host
        self.user = user
        self.port = port
        self.key_path = key_path
        self.password = password
        self.passphrase = passphrase
        self.connect_timeout = connect_timeout
        self.known_hosts = known_hosts or (Path.home() / ".ssh" / "known_hosts")
        self.ask_host_key = ask_host_key or (lambda hostname, key: False)
        self._client: paramiko.SSHClient | None = None

    # ------------------------------------------------------------- connection

    def connect(self) -> None:
        client = paramiko.SSHClient()
        if self.known_hosts.is_file():
            try:
                client.load_host_keys(str(self.known_hosts))
            except OSError:
                pass  # unreadable known_hosts is not fatal; we just ask instead

        policy = _AskPolicy(self.ask_host_key)
        client.set_missing_host_key_policy(policy)

        try:
            client.connect(
                hostname=self.host,
                port=self.port,
                username=self.user,
                key_filename=self.key_path,
                password=self.password,
                passphrase=self.passphrase,
                timeout=self.connect_timeout,
                auth_timeout=self.connect_timeout,
                banner_timeout=self.connect_timeout,
                allow_agent=True,
                look_for_keys=True,
            )
        except paramiko.PasswordRequiredException as exc:
            raise SSHError(
                "the private key is encrypted - pass --key-passphrase, or load it into your ssh agent"
            ) from exc
        except paramiko.AuthenticationException as exc:
            raise SSHError(
                f"{self.user}@{self.host} rejected the credentials. Check --user, --key, "
                "and that the key is in the droplet's authorized_keys."
            ) from exc
        except SSHError:
            raise
        except (paramiko.SSHException, OSError) as exc:
            raise SSHError(f"could not connect to {self.host}:{self.port} - {exc}") from exc

        self._client = client
        if policy.accepted is not None:
            self._remember_host_key(policy.accepted)

    def _remember_host_key(self, key: paramiko.PKey) -> None:
        """Append the accepted key rather than rewriting the user's known_hosts."""
        target = self.host if self.port == 22 else f"[{self.host}]:{self.port}"
        line = f"{target} {key.get_name()} {key.get_base64()}\n"
        try:
            self.known_hosts.parent.mkdir(parents=True, exist_ok=True)
            with self.known_hosts.open("a", encoding="utf-8") as handle:
                handle.write(line)
        except OSError:
            pass  # not being able to remember it only costs another prompt later

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "SSHRunner":
        self.connect()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ---------------------------------------------------------------- running

    def run(self, command: str, timeout: float = DEFAULT_COMMAND_TIMEOUT) -> CommandResult:
        transport = self._client.get_transport() if self._client else None
        if transport is None or not transport.is_active():
            raise SSHError("the SSH connection is not open")

        wrapped = wrap_command(command)

        started = time.monotonic()
        try:
            channel = transport.open_session(timeout=self.connect_timeout)
            channel.settimeout(1.0)  # per-recv, so the deadline loop stays responsive
            channel.exec_command(wrapped)
        except (paramiko.SSHException, OSError) as exc:
            raise SSHError(f"could not start a command on {self.host}: {exc}") from exc

        stdout, stderr, truncated, timed_out = self._drain(channel, started + timeout)
        exit_code = channel.recv_exit_status() if not timed_out else -1
        channel.close()

        return CommandResult(
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration=time.monotonic() - started,
            timed_out=timed_out,
            output_truncated=truncated,
        )

    def _drain(self, channel, deadline: float) -> tuple[str, str, bool, bool]:
        out, err = bytearray(), bytearray()
        truncated = False

        while True:
            moved = False
            for ready, recv, sink in (
                (channel.recv_ready, channel.recv, out),
                (channel.recv_stderr_ready, channel.recv_stderr, err),
            ):
                while ready():
                    try:
                        data = recv(65536)
                    except socket.timeout:
                        break
                    if not data:
                        break
                    moved = True
                    if len(sink) < MAX_OUTPUT_BYTES:
                        sink.extend(data)
                    else:
                        truncated = True  # keep draining so the command can finish

            done = channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready()
            if done:
                return out.decode("utf-8", "replace"), err.decode("utf-8", "replace"), truncated, False
            if time.monotonic() > deadline:
                return out.decode("utf-8", "replace"), err.decode("utf-8", "replace"), truncated, True
            if not moved:
                time.sleep(0.05)

    def run_script(
        self,
        body: str,
        interpreter: str = "bash",
        index: int = 1,
        timeout: float = DEFAULT_COMMAND_TIMEOUT,
    ) -> CommandResult:
        """Copy a script to the far end and run it there, as one step.

        Run by naming the interpreter - `bash /tmp/dba-harness/step03.sh` - rather
        than by making the file executable and calling it. That way the execute bit
        is never needed, which matters because /tmp is mounted noexec on hardened
        images, and the interpreter is the one the guard judged the script as
        rather than whatever the shebang happens to say.

        Mode 0700, set before the body is written and not after: a script may carry
        a password that a {{DBA_SECRET:...}} placeholder stood for until this
        moment, and it sits in a world-readable directory for as long as it takes
        to run.

        The parse pass first, for the reason wrap_command gives: an interpreter
        reads a script as it goes, so a quote left open at the bottom runs
        everything above it and only then fails, half-applying a step that was
        approved whole. Neither `bash -n` nor py_compile executes anything.

        Nothing of the harness's own is injected into the body - no pipefail, no
        `set -e`. What runs on the server is byte for byte what was judged, and
        what the model wants of its shell it writes at the top itself.
        """
        started = time.monotonic()
        path = script_path(index, interpreter)
        payload = body.encode("utf-8")

        try:
            sftp = self._client.open_sftp()  # type: ignore[union-attr]
        except (paramiko.SSHException, OSError) as exc:
            raise SSHError(f"could not open SFTP on {self.host}: {exc}") from exc
        try:
            try:
                sftp.mkdir(SCRIPT_DIR, 0o700)
            except OSError:
                pass  # already there, which is the usual case after the first script
            # The mode is verified rather than trusted, whatever the mkdir did.
            # On a shared box a local user can create /tmp/dba-harness first, and
            # sticky-bit /tmp then lets them swap a script between the parse pass
            # below and the run that follows it - a script which may carry a
            # generated credential besides. A directory that is not owner-only is
            # refused rather than written into; the one residual case is a 0700
            # directory owned by someone else, which only root can write through
            # and which SFTP st_uid is too unreliable to check for.
            mode = stat.S_IMODE(sftp.stat(SCRIPT_DIR).st_mode)
            if mode != 0o700:
                return CommandResult(
                    command=f"{interpreter} {path}",
                    exit_code=1,
                    stdout="",
                    stderr=(
                        f"harness: {SCRIPT_DIR} on the server exists with mode {mode:04o}, not 0700, "
                        "so the scripts written there would not be private to the account this "
                        f"harness connects as. Remove or rename {SCRIPT_DIR} on the server and send "
                        "the step again."
                    ),
                    duration=time.monotonic() - started,
                )
            with sftp.open(path, "wb") as handle:
                handle.write(payload)
            sftp.chmod(path, 0o700)
        except OSError as exc:
            return CommandResult(
                command=f"{interpreter} {path}",
                exit_code=1,
                stdout="",
                stderr=f"harness: could not copy the script to {path} on the server: {exc}",
                duration=time.monotonic() - started,
            )
        finally:
            sftp.close()

        parsed = self.run(script_check(path, interpreter), timeout=60)
        if parsed.exit_code != 0 and not parsed.timed_out:
            note = SCRIPT_SYNTAX_ERROR_NOTE.format(interpreter=interpreter, path=path)
            return CommandResult(
                command=f"{interpreter} {path}",
                exit_code=2,
                stdout="",
                stderr=f"{parsed.stderr.strip()}\n{note}".strip(),
                duration=time.monotonic() - started,
            )

        result = self.run(f"{interpreter} {shlex.quote(path)}", timeout=timeout)
        return CommandResult(
            command=f"{interpreter} {path}",
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            duration=time.monotonic() - started,
            timed_out=result.timed_out,
            output_truncated=result.output_truncated,
        )

    # ----------------------------------------------------------------- files

    def write_file(self, path: str, content: str, mode: str = "0644") -> CommandResult:
        """Put a file on the far end, via sudo if the account cannot write it."""
        started = time.monotonic()
        payload = content.encode("utf-8")
        try:
            sftp = self._client.open_sftp()  # type: ignore[union-attr]
        except (paramiko.SSHException, OSError) as exc:
            raise SSHError(f"could not open SFTP on {self.host}: {exc}") from exc

        # A random name, not one derived from the target path: two runs working the
        # same server at once must not stage onto each other's file, and a name a
        # local user can predict in advance is one they can create first.
        staging = f"/tmp/.dba-harness-{secrets.token_hex(8)}"
        try:
            with sftp.open(staging, "wb") as handle:
                handle.write(payload)
            # Before it sits in /tmp for the length of a round trip: a config file
            # written here can hold a password, and the umask that made it is not
            # this harness's to trust.
            sftp.chmod(staging, 0o600)
        except OSError as exc:
            sftp.close()
            return CommandResult(
                command=f"write_file {path}",
                exit_code=1,
                stdout="",
                stderr=f"could not stage the file on the server: {exc}",
                duration=time.monotonic() - started,
            )
        sftp.close()

        # install(1) does the move, the mode, and the parent directory in one
        # step, and picks up sudo only when the account actually needs it.
        sudo = "" if self.user == "root" else "sudo -n "
        install = (
            f"{sudo}install -D -m {shlex.quote(mode)} {shlex.quote(staging)} {shlex.quote(path)} "
            f"&& rm -f {shlex.quote(staging)}"
        )
        result = self.run(install, timeout=60)
        return CommandResult(
            command=f"write_file {path} (mode {mode}, {len(payload)} bytes)",
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            duration=time.monotonic() - started,
            timed_out=result.timed_out,
        )
