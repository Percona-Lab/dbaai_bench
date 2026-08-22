"""A pretend Ubuntu 24.04 droplet, good enough to drive the DBA harness against.

Duck-types SSHRunner: .run(command, timeout) and .write_file(path, content, mode).
It models packages, systemd units, listening ports, files, and just enough SQL to
answer the questions a model actually asks while installing a database server.

Several droplets can share a Network, which is what a replication or cluster task
needs: they see each other's addresses, refuse connections to a port bound to
loopback, and a replica really does read its primary's state through a link that
only works if the credentials, the grant, the server ids and the bind address are
all right.

Commands it does not recognise return exit 0 with no output and are collected in
.unhandled, so a test can report exactly where the simulation stopped being real.
"""

from __future__ import annotations

import fnmatch
import re
import shlex
from dataclasses import dataclass, field

from do_dba.facts import PROBES
from do_dba.ssh import CommandResult

# Answers to the harness's own probes, keyed by probe name.
PROBE_ANSWERS = {
    "os": "Ubuntu 24.04.1 LTS",
    "kernel": "6.8.0-45-generic",
    "arch": "x86_64",
    "account": "root",
    "privilege": "root",
    "cpu": "2",
    "memory": "3924 MB total, 3455 MB available",
    "disk_root": "48G total, 44G free, 8% used",
    "package_manager": "apt-get",
    "init": "systemd",
    "package_lock": "free",
}

PACKAGE_SERVICES = {
    "mysql-server": ("mysql", 3306),
    "mysql-server-8.0": ("mysql", 3306),
    "mariadb-server": ("mariadb", 3306),
    "mariadb-server-10.11": ("mariadb", 3306),
    "postgresql": ("postgresql", 5432),
    "postgresql-16": ("postgresql", 5432),
    "postgresql-contrib": (None, None),
    "mysql-client": (None, None),
    "mariadb-client": (None, None),
    "postgresql-client": (None, None),
    # Valkey comes from the distribution, and the unit is valkey-server, as in
    # Debian's packaging.
    "valkey-server": ("valkey-server", 6379),
    "valkey-tools": (None, None),
    # MongoDB does not: these are the vendor's packages, and apt cannot see any of
    # them until the mongodb repository has been added - see _mongo_repo.
    "mongodb-org": ("mongod", 27017),
    "mongodb-org-server": ("mongod", 27017),
    "mongodb-org-shell": (None, None),
    "mongodb-org-tools": (None, None),
    "mongodb-mongosh": (None, None),
}
# The packages Ubuntu's own archive does not carry.
VENDOR_ONLY = {name for name in PACKAGE_SERVICES if name.startswith("mongodb-")}
# mysql-server and mariadb-server are two implementations of one service on one
# port, and apt will not have both.
SQL_SERVICES = {"mysql", "mariadb"}
# Unit names that resolve to a different unit. MariaDB ships mysql.service and
# mysqld.service as aliases of mariadb.service, and models write all three.
UNIT_ALIASES = {
    "mysqld": ("mysql", "mariadb"),
    "mysql": ("mariadb",),
    "mariadbd": ("mariadb",),
    "mongodb": ("mongod",),
    "valkey": ("valkey-server",),
    "redis": ("redis-server",),
}
# What may sit in front of the program on a command line: VAR=value assignments
# and wrappers with their own flags. Purely non-capturing, so a route's own
# groups keep their numbers.
PREFIX = (r"^\s*(?:[A-Za-z_]\w*=(?:'[^']*'|\"[^\"]*\"|\S*)\s+)*"
          r"(?:(?:sudo|doas|env|time|nohup|ionice|nice)\s+"
          r"(?:-u\s+\S+\s+|-\w+\s+|--[\w-]+(?:=\S+)?\s+)*)*"
          # timeout takes a duration of its own, so it needs its own clause: a
          # `timeout 30 apt-get install ...` is the same install with a deadline.
          r"(?:timeout\s+(?:-\w+\s+|--[\w-]+(?:=\S+)?\s+)*[\d.]+[smhd]?\s+)?")

CLIENT_BINARIES = {
    "mysql-server": ["mysql", "mysqladmin", "mysqldump", "mysqld"],
    "mysql-server-8.0": ["mysql", "mysqladmin", "mysqldump", "mysqld"],
    "mysql-client": ["mysql", "mysqldump"],
    # MariaDB installs its tools under both names, which is why a model that only
    # knows the mysql ones gets away with it.
    "mariadb-server": ["mariadb", "mysql", "mariadb-admin", "mysqladmin",
                       "mariadb-dump", "mysqldump", "mariadbd"],
    "mariadb-server-10.11": ["mariadb", "mysql", "mariadb-admin", "mysqladmin",
                             "mariadb-dump", "mysqldump", "mariadbd"],
    "mariadb-client": ["mariadb", "mysql", "mariadb-dump", "mysqldump"],
    "postgresql": ["psql", "createdb", "createuser", "pg_isready", "pg_dump"],
    "postgresql-16": ["psql", "createdb", "createuser", "pg_isready", "pg_dump"],
    "postgresql-client": ["psql", "pg_isready"],
    "valkey-server": ["valkey-server", "valkey-cli"],
    "valkey-tools": ["valkey-cli"],
    # No redis-cli: on a Valkey box it is not there unless redis-tools is, and a
    # model reaching for it should be told so rather than quietly served.
    "mongodb-org": ["mongod", "mongosh", "mongodump", "mongorestore"],
    "mongodb-org-server": ["mongod"],
    "mongodb-org-shell": ["mongosh"],
    "mongodb-mongosh": ["mongosh"],
    "mongodb-org-tools": ["mongodump", "mongorestore"],
}

# The stock config files, as the packages ship them. Valkey and MongoDB are
# configured by editing these rather than by SQL, so a run that gets no file to
# edit is not exercising the interesting half of the work - and `sed -i` on a file
# that does not exist fails.
VALKEY_CONF = """\
# Valkey configuration file example.
bind 127.0.0.1 -::1
protected-mode yes
port 6379
dir /var/lib/valkey
appendonly no
save 3600 1 300 100 60 10000
# requirepass foobared
"""
MONGOD_CONF = """\
# mongod.conf, see https://docs.mongodb.com/manual/reference/configuration-options/
storage:
  dbPath: /var/lib/mongodb
systemLog:
  destination: file
  logAppend: true
  path: /var/log/mongodb/mongod.log
net:
  port: 27017
  bindIp: 127.0.0.1
"""
# Ubuntu's own mysqld.cnf, trimmed to the lines that matter here. It ships bound to
# localhost, which is the line a replication task has to change - and changing it
# without restarting the service changes nothing, as on a real machine.
MYSQLD_CNF = """\
[mysqld]
user            = mysql
datadir         = /var/lib/mysql
log_error       = /var/log/mysql/error.log
bind-address    = 127.0.0.1
mysqlx-bind-address = 127.0.0.1
"""
PACKAGE_FILES = {
    "mysql-server": {"/etc/mysql/mysql.conf.d/mysqld.cnf": MYSQLD_CNF},
    "mysql-server-8.0": {"/etc/mysql/mysql.conf.d/mysqld.cnf": MYSQLD_CNF},
    "mariadb-server": {"/etc/mysql/mariadb.conf.d/50-server.cnf": MYSQLD_CNF},
    "mariadb-server-10.11": {"/etc/mysql/mariadb.conf.d/50-server.cnf": MYSQLD_CNF},
    "valkey-server": {"/etc/valkey/valkey.conf": VALKEY_CONF},
    "mongodb-org": {"/etc/mongod.conf": MONGOD_CONF},
    "mongodb-org-server": {"/etc/mongod.conf": MONGOD_CONF},
}
# Where a MySQL or MariaDB server reads its configuration from. Everything under
# these is re-read when the unit restarts, and only then.
MYSQL_CONF_PATHS = ("/etc/mysql/", "/etc/my.cnf")

# Options that consume the next token, so what follows is not a positional.
VALKEY_VALUE_FLAGS = {"-h", "-p", "-a", "-n", "-s", "-u", "-t", "-i", "-r",
                      "--user", "--pass", "--timeout", "--socket"}
MONGO_VALUE_FLAGS = {"-u", "--username", "-p", "--password", "--authenticationDatabase",
                     "--authenticationMechanism", "-h", "--host", "--port", "--eval", "-e",
                     "--file", "-f", "--apiVersion", "--tlsCAFile"}
# Commands this simulator answers for real are handled in _valkey_command; these
# exist on a real server but are not modelled, and go to .unhandled rather than
# coming back as "unknown command", which would be a lie about the server.
VALKEY_UNMODELLED = {
    "LPUSH", "RPUSH", "LPOP", "RPOP", "LRANGE", "LLEN", "SADD", "SREM", "SMEMBERS",
    "SCARD", "HSET", "HGET", "HGETALL", "HDEL", "ZADD", "ZRANGE", "ZSCORE",
    "PUBLISH", "SETEX", "SETNX", "GETSET", "MSET", "MGET", "PERSIST", "RENAME",
    "SLOWLOG", "MEMORY", "LATENCY", "CLUSTER", "REPLICAOF", "SLAVEOF", "WAIT",
}

# `GRANT REPLICATION SLAVE ON *.* TO 'repl'@'10.116.0.3'`. _user_key cannot be used
# for this: it looks for the words `user` or `role`, which a GRANT has not got.
_GRANT_TO = re.compile(r"\bto\s+['\"`]?([\w.-]+)['\"`]?\s*(?:@\s*['\"`]?([\w%.-]+))?", re.I)
# The quoted options of CHANGE REPLICATION SOURCE TO / CHANGE MASTER TO. Unquoted
# ones (SOURCE_PORT=3306, SOURCE_AUTO_POSITION=1) are skipped: nothing here needs
# them, and a value read wrong would be worse than a value left at its default.
_REPL_OPTION = re.compile(r"([A-Za-z_]+)\s*=\s*['\"]([^'\"]*)['\"]")


class Network:
    """The private network a set of droplets share.

    Reachability only, which is the one property of a network that decides whether a
    cluster can be built at all: a droplet answers on an address if it is on this
    network under that address and the port asked for is bound to something other
    than loopback. connected=False is a network that was never enabled - the
    droplets exist, know their own addresses, and no packet gets between them.

    blocked names addresses that answer nobody while the rest of the network works,
    which is the shape of two droplets in different VPCs: each has a private address
    of its own, neither can reach the other's, and the public ones work fine.

    filtered names addresses that answer ping but drop the SSH port, which is what a
    firewall in front of a working network looks like. The distinction is not
    pedantry: a refused port and a dropped packet reach the harness as different
    findings, and only one of them means there is no network.
    """

    def __init__(self, connected: bool = True, blocked: tuple[str, ...] = (),
                 filtered: tuple[str, ...] = ()):
        self.hosts: dict[str, FakeDroplet] = {}
        self.connected = connected
        self.blocked = set(blocked)
        self.filtered = set(filtered)

    def add(self, droplet: "FakeDroplet") -> None:
        for address in droplet.addresses:
            self.hosts[address] = droplet

    def host(self, address: str) -> "FakeDroplet | None":
        return self.hosts.get(self._clean(address))

    def verdict(self, address: str, port: int) -> str:
        """open, refused or dropped - what a connection attempt would come back as.

        An address nobody holds drops the packets, as the internet does: the anchor
        address a cloud puts on the public interface and a docker bridge on another
        machine are both this case, and reporting them as "refused" would say the
        packets got somewhere when they went nowhere.
        """
        clean = self._clean(address)
        droplet = self.host(clean)
        if droplet is None or not self.connected or clean in self.blocked | self.filtered:
            return "dropped"
        return "open" if droplet.listens(port, clean) else "refused"

    def pings(self, address: str) -> bool:
        """Whether ICMP comes back, which a filtered port does not stop."""
        clean = self._clean(address)
        return bool(self.host(clean) is not None and self.connected
                    and clean not in self.blocked)

    def reachable(self, address: str, port: int) -> bool:
        return self.verdict(address, port) == "open"

    @staticmethod
    def _clean(address: str) -> str:
        return (address or "").strip("[]'\"")


@dataclass
class Engine:
    """A database server's state, shared by the mysql and psql simulators."""

    databases: set[str] = field(default_factory=set)
    users: set[str] = field(default_factory=set)
    passwords: dict[str, str] = field(default_factory=dict)
    version: str = ""
    # Replication, as far as driving the harness needs it. server_id and log_bin
    # come from the config files at each restart; the rest from CHANGE REPLICATION
    # SOURCE TO and START REPLICA. A replica whose id matches its source, or whose
    # user has no REPLICATION SLAVE grant, connects and then stops - which is how it
    # goes wrong on a real pair, and only SHOW REPLICA STATUS says so.
    server_id: int = 1
    log_bin: bool = True  # MySQL 8 ships with the binary log on
    repl_grants: set[str] = field(default_factory=set)
    source: str = ""
    source_user: str = ""
    source_password: str = ""
    replicating: bool = False

    def bare_users(self) -> set[str]:
        return {user.split("@")[0] for user in self.users}


@dataclass
class Mongo:
    """A mongod's state.

    Databases and collections are not created by a statement: they appear the first
    time something is written to them and disappear when the last document goes, so
    they are counted rather than listed. Users belong to the database they were
    created in, which is the one a later --authenticationDatabase has to name.
    """

    databases: dict[str, dict[str, int]] = field(default_factory=dict)
    users: dict[str, str] = field(default_factory=dict)  # user -> the database it lives in
    passwords: dict[str, str] = field(default_factory=dict)
    auth: bool = False  # whether authorization is enabled in mongod.conf
    version: str = "8.0.4"

    def collections(self, database: str) -> dict[str, int]:
        return self.databases.setdefault(database, {})


@dataclass
class Valkey:
    """A valkey-server's keyspace and the part of its config models change."""

    keys: dict[str, str] = field(default_factory=dict)
    config: dict[str, str] = field(default_factory=dict)
    version: str = "7.2.5"

    @property
    def requirepass(self) -> str:
        return self.config.get("requirepass", "")


class FakeDroplet:
    def __init__(
        self,
        verbose: bool = False,
        hostname: str = "ubuntu-dba-01",
        address: str = "10.116.0.2",
        public: str = "203.0.113.10",
        network: Network | None = None,
        extra: tuple[tuple[str, str], ...] = (),
    ):
        self.verbose = verbose
        # Its own identity, because a task spanning servers is largely a matter of
        # telling one of them the other's address. `address` is the private one -
        # the one to bind to and to point a peer at - and `public` is the one the
        # harness itself connected on.
        self.hostname = hostname
        self.address = address
        self.public = public
        # What `ip addr` shows, interface by interface. Only `address` and `public`
        # are on the network; `extra` is for the addresses a real cloud host also
        # reports and no peer can reach - a provider's anchor address beside the
        # public one on eth0, a docker bridge - given as (interface, address). They
        # go between the two, which is where the kernel lists them.
        self.interfaces: list[tuple[str, str]] = (
            ([("eth0", public)] if public else [])
            + [(name, value) for name, value in extra]
            + ([("eth1", address)] if address else [])
        )
        self.network = network if network is not None else Network()
        self.network.add(self)
        self.packages: set[str] = set()
        self.binaries: set[str] = {"bash", "systemctl", "apt-get", "ss", "grep", "awk"}
        self.services: dict[str, bool] = {"ssh": True}
        self.enabled: set[str] = {"ssh"}
        self.ports: dict[int, str] = {22: "0.0.0.0"}
        self.files: dict[str, str] = {
            "/etc/os-release": 'PRETTY_NAME="Ubuntu 24.04.1 LTS"\nID=ubuntu\n',
        }
        self.apt_updated = False
        # The sources files apt had seen the last time it fetched package lists. A
        # repository added after that is not usable yet, which is the step models
        # skip when they add the MongoDB repo - see _mongo_repo.
        self.repos_fetched: set[str] = set()
        # Config files a package brought with it, kept apart from the ones the model
        # wrote so state() still reports only the model's own work.
        self.provisioned: set[str] = set()
        self.vars: dict[str, str] = {}
        self.env: dict[str, str] = {}
        self.commands: list[str] = []
        self.unhandled: list[str] = []
        self._depth = 0  # nesting of bash -c payloads
        # One engine for MySQL and MariaDB: same protocol, same port, same data
        # directory, and apt will not install both. The flavour decides which unit
        # name and version string the client reports, and is set by the install.
        self.sql_flavour = "mysql"
        self.mysql = Engine(databases={"information_schema", "mysql", "performance_schema", "sys"},
                            users={"root@localhost"}, version="8.0.39-0ubuntu0.24.04.2")
        self.postgres = Engine(databases={"postgres", "template0", "template1"},
                               users={"postgres"}, version="16.4 (Ubuntu 16.4-0ubuntu0.24.04.2)")
        self.mongo = Mongo(databases={"admin": {}, "config": {}, "local": {"startup_log": 1}})
        # requirepass is present and empty, as it is on a fresh install: CONFIG GET
        # requirepass answers with a blank line rather than with nothing at all.
        self.valkey = Valkey(config={"bind": "127.0.0.1 -::1", "protected-mode": "yes",
                                     "requirepass": "", "maxmemory": "0",
                                     "appendonly": "no", "save": "3600 1"})
        self._probe_by_command = {command: name for name, command in PROBES}

    # --------------------------------------------------------------- identity

    @property
    def addresses(self) -> list[str]:
        return [a for a in (self.address, self.public) if a]

    def listens(self, port: int, address: str = "") -> bool:
        """Whether something on this machine would answer a peer on that port.

        Bound to loopback is the same as not listening as far as another server is
        concerned, and that distinction is the whole of what goes wrong when a
        replica cannot reach its primary.

        With an address, whether it would answer *there*: a service bound to one of
        the machine's addresses refuses connections to the others, which is what an
        sshd with a ListenAddress looks like to a peer on the private network. Asked
        without one, the question is only whether anything is listening at all.
        """
        bound = self.ports.get(port)
        if bound is None or bound in {"127.0.0.1", "::1", "localhost"}:
            return False
        return not address or bound in {"0.0.0.0", "::", "*", address}

    # ------------------------------------------------------------------ shell

    def run(self, command: str, timeout: float = 300.0) -> CommandResult:
        self.commands.append(command)
        code, out, err = self._dispatch(command.strip())
        if self.verbose:
            print(f"    [droplet] {command.splitlines()[0][:100]} -> {code}")
        return CommandResult(command=command, exit_code=code, stdout=out, stderr=err, duration=0.4)

    def write_file(self, path: str, content: str, mode: str = "0644") -> CommandResult:
        self.files[path] = content
        return CommandResult(
            command=f"write_file {path} (mode {mode}, {len(content)} bytes)",
            exit_code=0, stdout="", stderr="", duration=0.2,
        )

    def _dispatch(self, command: str) -> tuple[int, str, str]:
        probe = self._probe_by_command.get(command)
        if probe is not None:
            return 0, self._probe(probe), ""

        # A multi-line script: newline is a sequencer like ; and `set -e` decides
        # whether a failure stops the rest. Heredocs are left whole.
        if "\n" in command and "<<" not in command:
            return self._script(command)
        # `if cond; then cmd; fi` on one line needs the same treatment, or the
        # condition is split on ; and run as though it were a command itself.
        if re.match(r"^\s*if\s", command):
            return self._script(command)
        return self._logical(command)

    def _logical(self, command: str) -> tuple[int, str, str]:
        """One logical line - no more newline splitting, however it is quoted."""
        # A reply cut off mid-string leaves a quote open. Say what bash says, so
        # the model fixes the quoting instead of chasing a client-side error.
        quote = self._open_quote(command)
        if quote and "<<" not in command:
            return 2, "", (f"bash: -c: line 1: unexpected EOF while looking for "
                           f"matching `{quote}'\nbash: -c: line 2: syntax error: "
                           f"unexpected end of file\n")
        command = self._expand(command)
        command = self._take_env(command)

        # A compound command: run each part, stopping where the operator says to.
        parts = self._split(command)
        if len(parts) > 1:
            code, out, err = 0, [], []
            for operator, part in parts:
                # A short-circuited part is skipped, not the rest of the line:
                # in `a || b; c` a success skips b and still runs c.
                if operator == "&&" and code != 0:
                    continue
                if operator == "||" and code == 0:
                    continue
                code, part_out, part_err = self._dispatch(part)
                out.append(part_out)
                err.append(part_err)
            return code, "".join(out), "".join(err)

        return self._pipeline(command)

    def _script(self, command: str) -> tuple[int, str, str]:
        """Run a multi-line script line by line, honouring set -e and if blocks."""
        strict = bool(re.search(r"^\s*set\s+-\w*e", command, re.M))
        code, out, err = 0, [], []
        # if/elif/else/fi is how a model writes an idempotency guard, and without
        # it the whole block would run - or worse, look like it did.
        chain: list[dict] = []
        lines = self._logical_lines(command)
        number = 0
        while number < len(lines):
            line = lines[number].strip()
            number += 1
            if not line or line.startswith("#") or re.match(r"^set\s+[-+]", line):
                continue
            # A line that *begins* with && is a syntax error in bash, which
            # reports it only after running everything above it.
            leading = re.match(r"^(&&|\|\||\||;)", line)
            if leading:
                return 2, "".join(out), (
                    "".join(err)
                    + f"bash: line {number}: syntax error near unexpected token `{leading.group(1)}'\n"
                )

            word = re.match(r"[A-Za-z]+", line)
            word = word.group(0) if word else ""

            if word in {"if", "elif"}:
                if word == "if":
                    chain.append({"running": False, "taken": False})
                elif not chain:
                    return 2, "".join(out), (
                        "".join(err)
                        + f"bash: line {number}: syntax error near unexpected token `elif'\n"
                    )
                entry = chain[-1]
                condition, rest = self._split_then(line)
                outer = all(other["running"] for other in chain[:-1])
                take = outer and not entry["taken"] and self._condition(condition)
                entry["running"] = take
                entry["taken"] = entry["taken"] or take
                # `if cond; then cmd; fi` on one line: queue what follows `then`.
                for extra in reversed(self._after_then(rest)):
                    lines.insert(number, extra)
                continue
            if word == "else":
                if chain:
                    entry = chain[-1]
                    outer = all(other["running"] for other in chain[:-1])
                    entry["running"] = outer and not entry["taken"]
                    entry["taken"] = True
                continue
            if word == "fi":
                if chain:
                    chain.pop()
                continue
            if word == "then":
                continue
            if not all(entry["running"] for entry in chain):
                continue  # inside a branch that was not taken

            # _logical, not _dispatch: a logical line may still hold newlines
            # (a quoted SQL block), and dispatching it again would recurse.
            code, line_out, line_err = self._logical(line)
            out.append(line_out)
            err.append(line_err)
            if code != 0 and strict:
                break
        return code, "".join(out), "".join(err)

    @staticmethod
    def _split_then(line: str) -> tuple[str, str]:
        """An if/elif header split into its condition and whatever follows `then`."""
        body = line.split(None, 1)[1] if " " in line else ""
        match = re.search(r"(?:;|\s)\s*then\b", body)
        if not match:
            return body.strip(), ""
        return body[: match.start()].strip(), body[match.end():].strip()

    @staticmethod
    def _after_then(rest: str) -> list[str]:
        """The inline body of a one-line if, with a trailing `fi` as its own line."""
        if not rest:
            return []
        body = re.sub(r";?\s*\bfi\b\s*$", "", rest).strip()
        closed = body != rest.strip()
        return [piece for piece in (body, "fi" if closed else "") if piece]

    def _condition(self, text: str) -> bool:
        """Run an `if` condition and report whether it succeeded.

        Its output is discarded: conditions here are tests and --quiet probes,
        and letting them print would put noise in the output the model reads.
        """
        text = text.strip()
        negated = False
        while text.startswith("!"):
            negated = not negated
            text = text[1:].strip()
        if not text:
            return not negated
        code, _, _ = self._logical(text)
        return code != 0 if negated else code == 0

    @classmethod
    def _logical_lines(cls, command: str) -> list[str]:
        """Group physical lines the way bash does before it runs anything.

        A trailing \\, && , || or | continues the line, and so does an unclosed
        quote - which is how a model writes multi-line SQL as `mysql -e "` plus
        the statements below it.
        """
        lines: list[str] = []
        current: str | None = None
        for raw in command.splitlines():
            if current is None:
                current = raw.strip()
            elif cls._open_quote(current) == '"' and current.endswith("\\"):
                # Inside "..." a trailing backslash is a line continuation and
                # bash removes it, which is how models lay out long SQL.
                current = current[:-1] + raw.strip()
            elif cls._open_quote(current):
                current = f"{current}\n{raw}"       # inside a quote: keep the line
            elif current.endswith("\\"):
                current = f"{current[:-1].rstrip()} {raw.strip()}"
            else:
                current = f"{current} {raw.strip()}"
            if (current.endswith("\\") or re.search(r"(&&|\|\||\|)$", current)
                    or cls._open_quote(current)):
                continue
            lines.append(current)
            current = None
        if current is not None:
            lines.append(current)
        return lines

    @staticmethod
    def _open_quote(text: str) -> str:
        """The quote character still waiting to be closed, or ''."""
        quote = ""
        index = 0
        while index < len(text):
            char = text[index]
            if char == "\\" and quote != "'":
                index += 2
                continue
            if quote:
                if char == quote:
                    quote = ""
            elif char in "'\"":
                quote = char
            index += 1
        return quote

    def _expand(self, command: str) -> str:
        """Substitute $(...) and known variables, the way the shell would."""
        for _ in range(4):  # innermost first, a few levels deep
            match = re.search(r"\$\(([^()]*)\)", command)
            if not match:
                break
            _, out, _ = self._dispatch(match.group(1).strip())
            command = command[: match.start()] + " ".join(out.split()) + command[match.end():]

        assignment = re.match(r"^([A-Za-z_]\w*)=(\S*)$", command)
        if assignment:
            self.vars[assignment.group(1)] = assignment.group(2).strip("'\"")
            return "true"
        for name, value in self.vars.items():
            command = command.replace(f"${{{name}}}", value).replace(f"${name}", value)
        return command

    def _take_env(self, command: str) -> str:
        """Strip a `VAR=value command` prefix, remembering the variables."""
        self.env = {}
        pattern = re.compile(r"^([A-Za-z_]\w*)=('[^']*'|\"[^\"]*\"|\S*)\s+(?=\S)")
        while True:
            match = pattern.match(command)
            if not match:
                return command
            self.env[match.group(1)] = match.group(2).strip("'\"")
            command = command[match.end():]

    def _pipeline(self, command: str) -> tuple[int, str, str]:
        """Run the head of a pipeline, then filter its output through the tail.

        Exit codes follow `set -o pipefail`, which is what the harness runs with:
        the first non-zero stage wins, not the last stage.
        """
        segments = self._split_pipes(command)
        if len(segments) == 1:
            return self._single(command)
        if not segments or not segments[0].strip():
            return 2, "", "bash: line 1: syntax error near unexpected token `|'\n"

        code, out, err = self._single(segments[0])
        for segment in segments[1:]:
            out, filter_code, handled = self._filter(out, segment)
            if not handled:
                self.unhandled.append(segment)
            code = code or filter_code
        return code, out, err

    @staticmethod
    def _split_pipes(command: str) -> list[str]:
        """Split on a single | , leaving || and quoted pipes alone."""
        if "<<" in command or "\n" in command:
            return [command]
        segments, buffer, quote = [], [], ""
        index = 0
        while index < len(command):
            char = command[index]
            if char == "\\" and quote != "'" and index + 1 < len(command):
                buffer.append(command[index:index + 2])  # escaped: never a quote
                index += 2
                continue
            if quote:
                if char == quote:
                    quote = ""
                buffer.append(char)
                index += 1
                continue
            if char in "'\"":
                quote = char
                buffer.append(char)
                index += 1
                continue
            if char == "|" and command[index:index + 2] != "||":
                segments.append("".join(buffer).strip())
                buffer = []
                index += 1
                continue
            buffer.append(char)
            index += 1
        segments.append("".join(buffer).strip())
        return [segment for segment in segments if segment]

    def _filter(self, text: str, segment: str) -> tuple[str, int, bool]:
        """Apply one downstream stage of a pipeline to the text flowing through."""
        lines = text.splitlines()
        tokens = self._tokens(segment)
        if not tokens:
            return text, 0, True
        # `... | sudo tee /etc/apt/...` is one wrapper deep, and reading the wrapper
        # as the program would send the text nowhere.
        while len(tokens) > 1 and tokens[0] in {"sudo", "env", "nohup", "command"}:
            tokens = tokens[1:]
        program = tokens[0]
        args = tokens[1:]

        def count(default: int) -> int:
            for index, token in enumerate(args):
                if token in {"-n", "-c"} and index + 1 < len(args):
                    return int(re.sub(r"\D", "", args[index + 1]) or default)
                if re.fullmatch(r"-\d+", token):
                    return int(token[1:])
            return default

        # `echo "SQL" | mysql -u root` is a common idiom: the pipe is the input,
        # not a filter, so the text becomes the statements to run.
        if program in {"mysql", "mariadb", "psql"}:
            engine = self.postgres if program == "psql" else self.mysql
            dialect = "postgres" if program == "psql" else "mysql"
            _, out, err = self._sql(engine, text, dialect=dialect,
                                    user="root" if dialect == "mysql" else "postgres")
            return out or err, 0 if not err else 1, True

        # The same idiom for the other two clients: one cache command per line, or a
        # whole script for mongosh.
        if program in {"valkey-cli", "redis-cli"}:
            password = self._flag_value(segment, ("-a", "--pass"))
            code, out, err = 0, [], []
            for line in lines:
                one, text_out, error = self._valkey_command(self._tokens(line), password)
                out.append(text_out)
                err.append(error)
                code = code or one
            return ("".join(out) or "".join(err)), code, True

        if program in {"mongosh", "mongo"}:
            code, out, err = self._mongo_js(text, self._mongo_database(segment, program))
            return (out or err), code, True

        # A downloaded signing key on its way to /usr/share/keyrings.
        if program == "gpg":
            target = self._flag_value(segment, ("-o", "--output"))
            if target:
                self.files[target] = text
                return "", 0, True
            return text, 0, True

        if program == "tee":
            for path in (a for a in args if a.startswith("/") and a != "/dev/null"):
                self.files[path] = (self.files.get(path, "") if "-a" in args else "") + text
            return text, 0, True

        if program == "tail":
            return "\n".join(lines[-count(10):]) + ("\n" if lines else ""), 0, True
        if program == "head":
            return "\n".join(lines[: count(10)]) + ("\n" if lines else ""), 0, True
        if program in {"grep", "egrep"}:
            patterns = [a for a in args if not a.startswith("-")]
            needle = patterns[0] if patterns else ""
            short = [a for a in args if a.startswith("-") and not a.startswith("--")]
            invert = any("v" in a for a in short)
            ignore = any("i" in a for a in short)
            extended = program == "egrep" or any("E" in a for a in short)
            if not extended:
                needle = self._basic_regex(needle)
            flags = re.IGNORECASE if ignore else 0
            try:
                hits = [line for line in lines if bool(re.search(needle, line, flags)) != invert]
            except re.error:
                hits = [line for line in lines if (needle in line) != invert]
            return ("\n".join(hits) + "\n" if hits else ""), (0 if hits else 1), True
        if program == "wc":
            return f"{len(lines)}\n", 0, True
        if program == "sort":
            unique = any("u" in a for a in args if a.startswith("-"))
            ordered = sorted(set(lines) if unique else lines)
            return "\n".join(ordered) + ("\n" if ordered else ""), 0, True
        if program == "uniq":
            out, previous = [], None
            for line in lines:
                if line != previous:
                    out.append(line)
                previous = line
            return "\n".join(out) + ("\n" if out else ""), 0, True
        if program == "awk":
            script = next((a for a in args if "print" in a), "")
            field = re.search(r"\$(\d+)", script)
            if field:
                column = int(field.group(1))
                out = [parts[column - 1] for line in lines
                       if (parts := line.split()) and len(parts) >= column]
                return "\n".join(out) + ("\n" if out else ""), 0, True
            return text, 0, True
        if program == "cut":
            match = re.search(r"-c(\d+)-(\d+)", segment)
            if match:
                start, end = int(match.group(1)), int(match.group(2))
                return "\n".join(line[start - 1:end] for line in lines) + ("\n" if lines else ""), 0, True
            return text, 0, True
        if program == "paste":
            separator = " "
            match = re.search(r"-s?d\s*'?(.)'?", segment)
            if match:
                separator = match.group(1)
            return separator.join(lines) + "\n", 0, True
        if program == "tr":
            return re.sub(r"\s+", " ", text), 0, True
        if program in {"cat", "xargs", "column", "less", "more"}:
            return text, 0, True
        if program == "sed":
            script = next((a for a in args if a.startswith("s")), "")
            parts = script.split(script[1]) if len(script) > 2 else []
            if len(parts) >= 3:
                return re.sub(parts[1], parts[2], text), 0, True
            return text, 0, True
        return text, 0, False

    @staticmethod
    def _split(command: str) -> list[tuple[str, str]]:
        """Split on && || ; but not inside quotes or a heredoc."""
        if "<<" in command or "\n" in command:
            return [("", command)]
        parts: list[tuple[str, str]] = []
        operator, buffer, quote = "", [], ""
        index = 0
        while index < len(command):
            char = command[index]
            # A backslash-escaped character is literal, quote marks included:
            # `bash -c "psql -c \"...\""` has one string, not three.
            if char == "\\" and quote != "'" and index + 1 < len(command):
                buffer.append(command[index:index + 2])
                index += 2
                continue
            if quote:
                if char == quote:
                    quote = ""
                buffer.append(char)
                index += 1
                continue
            if char in "'\"":
                quote = char
                buffer.append(char)
                index += 1
                continue
            two = command[index:index + 2]
            if two in {"&&", "||"}:
                parts.append((operator, "".join(buffer).strip()))
                operator = two
                buffer = []
                index += 2
                continue
            if char == ";":
                parts.append((operator, "".join(buffer).strip()))
                operator = ";"
                buffer = []
                index += 1
                continue
            buffer.append(char)
            index += 1
        parts.append((operator, "".join(buffer).strip()))
        return [(op, text) for op, text in parts if text]

    # --------------------------------------------------------------- handlers

    def _single(self, command: str) -> tuple[int, str, str]:
        command, stderr_to = self._stderr_redirect(command)
        for pattern, handler in self._routes():
            match = re.search(pattern, command, re.DOTALL)
            if match:
                code, out, err = handler(match, command)
                if stderr_to == "/dev/null":
                    return code, out, ""
                if stderr_to == "&1":  # merged, and in that order for a real shell
                    return code, out + err, ""
                return code, out, err
        self.unhandled.append(command)
        return 0, "", ""

    _STDERR_REDIRECT = re.compile(r"\s*2>\s*(&1|/dev/null)\s*$")

    @classmethod
    def _stderr_redirect(cls, command: str) -> tuple[str, str]:
        """Take a trailing `2>/dev/null` or `2>&1` off, and say which it was.

        Off before routing, because the routes match on the whole line: with the
        redirect still attached, `cat /etc/foo 2>/dev/null` matches nothing and comes
        back empty and successful, which reads as a file that exists and is empty.
        """
        match = cls._STDERR_REDIRECT.search(command)
        if not match or cls._open_quote(command[: match.start()]):
            return command, ""  # inside a quoted string it is just text
        return command[: match.start()], match.group(1)

    def _routes(self):
        # Anchored on the program, not matched anywhere in the line: an
        # unanchored route sends `echo "mysql active"` to the MySQL client, and
        # the resulting error is a false failure the model then chases.
        head = PREFIX
        return [
            # Ahead of the shell route, and matched anywhere in the line: a
            # connection test is written `timeout 3 bash -c '</dev/tcp/host/port'`,
            # and the payload is the thing being asked about rather than a script to
            # descend into. Nothing else on a server mentions /dev/tcp.
            (r"/dev/tcp/([\w.:-]+)/(\d+)", self._dev_tcp),
            # The other two halves of the same question, asked when a port says
            # nothing: does anything answer at that address, and is there a route to
            # it from here.
            (head + r"ping\b", self._ping),
            (head + r"ip\s+(?:-\d\s+)?route\s+get\s+([\d.]+)", self._ip_route),
            (r"^\s*(?:sudo\s+(?:-u\s+(\S+)\s+)?)?(?:bash|sh|dash|zsh)\s+-\S*c\b", self._shell_c),
            (head + r"(?:bash|sh)?\s*(/\S+\.(?:sh|bash))\b", self._run_script_file),
            (head + r"apt-get\s+(?:-\w+\s+)*update", self._apt_update),
            (head + r"apt(?:-get)?\s+.*\binstall\b", self._apt_install),
            (head + r"apt(?:-get)?\s+.*\b(?:purge|remove)\b", self._apt_remove),
            (head + r"systemctl\s+", self._systemctl),
            (head + r"service\s+(\w+)\s+(\w+)", self._service),
            (head + r"command\s+-v\s+(\S+)", self._command_v),
            (head + r"pg_isready\b", self._pg_isready),
            (head + r"(mysqladmin|mariadb-admin)\b", self._mysqladmin),
            (head + r"psql\b", self._psql),
            (head + r"mongod(?![\w-])", self._mongod),
            (head + r"(mongosh|mongo)(?![\w-])", self._mongo_client),
            (head + r"valkey-server\b", self._valkey_server),
            (head + r"(valkey-cli|redis-cli)\b", self._valkey_client),
            # (?![\w-]) and not \b: `mariadb-dump` is a different program, and a
            # word boundary sits between the b and the hyphen, so \b routes it to
            # the client and the model gets a SQL syntax error for a backup.
            (head + r"(mysql|mariadb)(?![\w-])", self._mysql_client),
            (head + r"createdb\b", self._createdb),
            (head + r"ss\s+-\w*t", self._ss),
            (head + r"(?:curl|wget)\b", self._curl),
            (head + r"(cat|tee)\b[^|]*<<[-']?\s*['\"]?(\w+)", self._heredoc),
            (r"^\s*(echo|printf)\s+(.*?)\s*(>>?)\s*(\S+)\s*$", self._echo_redirect),
            (head + r"sed\s+-i", self._sed),
            (head + r"(cat|head|tail)\s+(\S+)\s*$", self._cat),
            (head + r"grep\b", self._grep),
            (head + r"(test|\[)\s+(?:!\s+)?-[fdersznw]\s+", self._test_file),
            (head + r"ls\b", self._ls),
            (head + r"(mkdir|chown|chmod|install|rm|cp|mv|touch)\b", self._filesystem),
            (head + r"dpkg(?:-query)?\b", self._dpkg),
            (head + r"journalctl\b", self._journalctl),
            (head + r"ufw\b", self._ufw),
            (head + r"(id|whoami)\b", self._id),
            (head + r"(echo|printf)\s+(.*)$", self._echo),
            (head + r"(?:true|:)\s*$", lambda m, c: (0, "", "")),
            (head + r"sleep\b", lambda m, c: (0, "", "")),
        ]

    def _dev_tcp(self, match, command):
        """Bash's own port test: opening /dev/tcp/host/port succeeds or it does not.

        The three ways it fails are not interchangeable. A refused connection comes
        back at once and proves the packets crossed; dropped packets hang until
        something kills the attempt, which is exit 124 when `timeout` wrapped it and
        a connect error much later when nothing did. The harness reads these apart to
        decide whether there is a private network at all, so the simulator has to
        keep them apart too.
        """
        address, port = match.group(1), int(match.group(2))
        verdict = self.network.verdict(address, port)
        if verdict == "open":
            return 0, "", ""
        if verdict == "refused":
            return 1, "", (f"bash: connect: Connection refused\n"
                           f"bash: /dev/tcp/{address}/{port}: Connection refused\n")
        if re.match(r"^\s*timeout\s", command):
            return 124, "", ""
        return 1, "", (f"bash: connect: Connection timed out\n"
                       f"bash: /dev/tcp/{address}/{port}: Connection timed out\n")

    def _ping(self, match, command):
        """ICMP, which a firewall that drops a port often still lets through.

        The address is the last dotted quad on the line rather than a capture: the
        flags carry numbers of their own (`-W 2`), and the redirections a caller adds
        sit after the address.
        """
        quads = [word for word in command.split() if re.fullmatch(r"\d+(?:\.\d+){3}", word)]
        address = quads[-1] if quads else ""
        if self.network.pings(address):
            return 0, (f"PING {address} ({address}) 56(84) bytes of data.\n"
                       f"64 bytes from {address}: icmp_seq=1 ttl=64 time=0.401 ms\n"), ""
        return 1, f"PING {address} ({address}) 56(84) bytes of data.\n", ""

    def _ip_route(self, match, command):
        """`ip route get ADDR`: whether this machine has a way to send to it at all.

        Same subnet as one of its own addresses, in this simulator: a route exists
        because the address is on a network this machine is attached to. Anything
        else is what a droplet says about an address in a VPC it is not in.
        """
        address = match.group(1)
        for own in self.addresses:
            if own.split(".")[:3] == address.split(".")[:3]:
                device = "eth1" if own == self.address else "eth0"
                return 0, f"{address} dev {device} src {own} uid 0 \n    cache\n", ""
        return 2, "", "RTNETLINK answers: Network is unreachable\n"

    def _shell_c(self, match, command):
        """`bash -c '<script>'`: run the payload, the way a real shell would.

        Without this the whole quoted string falls through to the psql route and
        comes back as `syntax error at or near "psql"`, which reads like a model
        mistake when it is only a gap in this simulator.
        """
        payload = (self._extract_sql(command, ("-c", "--command")) or "").strip()
        if not payload:
            return 2, "", "bash: -c: option requires an argument\n"
        user = match.group(1)
        if user and user != "root":
            # `sudo -u postgres bash -c 'psql ...'` runs psql as postgres, and the
            # psql simulator reads the role off the command line it is handed.
            payload = re.sub(r"(?<![-\w])(psql|createdb|dropdb|pg_dump(?:all)?)\b",
                             rf"sudo -u {user} \1", payload)
        if self._depth >= 4:
            return 2, "", "bash: fork: retry: Resource temporarily unavailable\n"
        self._depth += 1
        try:
            return self._dispatch(payload)
        finally:
            self._depth -= 1

    def _probe(self, name: str) -> str:
        if name in PROBE_ANSWERS:
            return PROBE_ANSWERS[name] + "\n"
        if name == "hostname":
            return self.hostname + "\n"
        if name == "addresses":
            # interface=address/CIDR, as the probe's awk prints it: the harness reads
            # the interface to tell a VPC address from a bridge, and strips the mask
            # - a model that copies either wholesale should be seen doing it.
            return " ".join(f"{iface}={a}/20" for iface, a in self.interfaces) + "\n"
        if name == "db_clients":
            # Same order as the probe's own loop, so the answer reads like the one a
            # real shell would give.
            return " ".join(b for b in ("mysql", "mariadb", "psql", "mongosh", "mongo",
                                        "valkey-cli", "redis-cli")
                            if b in self.binaries) + "\n"
        if name == "db_services":
            return " ".join(f"{svc}.service" for svc, up in sorted(self.services.items())
                            if up and svc != "ssh") + "\n"
        if name == "db_packages":
            keys = ("mysql", "mariadb", "postgresql", "mongodb", "valkey", "redis")
            return " ".join(sorted(p for p in self.packages
                                   if any(k in p for k in keys))) + "\n"
        if name == "listening":
            return self._listening() + "\n"
        return "\n"

    def _listening(self) -> str:
        return " ".join(f"{addr}:{port}" for port, addr in sorted(self.ports.items()))

    def _apt_update(self, match, command):
        self.apt_updated = True
        self.repos_fetched = {p for p in self.files if p.startswith("/etc/apt/sources.list")}
        out = "Hit:1 http://archive.ubuntu.com/ubuntu noble InRelease\nReading package lists...\n"
        # A third-party repository is unusable until its key is on the machine, and
        # apt fails the update rather than trusting it. The Ubuntu archive is not
        # checked because its keys ship with the distribution.
        for path in sorted(self.repos_fetched):
            text = self.files[path]
            if "http" not in text or "ubuntu.com" in text or "debian.org" in text:
                continue
            key = re.search(r"signed-by=([^\]\s]+)", text)
            if key and key.group(1) in self.files:
                continue
            line = re.sub(r"^deb\s*(?:\[[^\]]*\]\s*)?", "", text.strip().splitlines()[0])
            return 100, out, (
                f"E: The repository '{line} Release' is not signed.\n"
                "N: Updating from such a repository can't be done securely, and is therefore "
                "disabled by default.\n"
            )
        return 0, out, ""

    def _mongo_repo(self) -> bool:
        """Whether apt can see the vendor's MongoDB packages.

        Three things have to line up, and missing any of them gets the same "unable
        to locate package" a real machine gives: a sources file pointing at
        repo.mongodb.org, the keyring it says it is signed by, and an apt-get update
        run after that file appeared.
        """
        for path in self.repos_fetched:
            text = self.files.get(path, "")
            if "repo.mongodb.org" not in text:
                continue
            key = re.search(r"signed-by=([^\]\s]+)", text)
            if key and key.group(1) in self.files:
                return True
        return False

    def _apt_install(self, match, command):
        if not self.apt_updated:
            return 100, "", ("E: Unable to locate package - the package lists have never been "
                             "downloaded on this machine. Run apt-get update first.\n")
        tokens = self._tokens(command)
        requested = [t for t in tokens if not t.startswith("-") and t not in
                     {"apt-get", "apt", "sudo", "install", "-y"}]
        installed, unknown = [], []
        for package in requested:
            # A vendor package is invisible until its repository is configured, and
            # apt says so in exactly the same words as for a name that does not
            # exist. This is the wall a model hits when it installs MongoDB the way
            # it installs everything else.
            if package in PACKAGE_SERVICES and not (package in VENDOR_ONLY and not self._mongo_repo()):
                installed.append(package)
            else:
                unknown.append(package)
        if unknown:
            return 100, "", f"E: Unable to locate package {unknown[0]}\n"

        lines = []
        for package in installed:
            if package in self.packages:
                lines.append(f"{package} is already the newest version.")
                continue
            service, port = PACKAGE_SERVICES[package]
            if service in SQL_SERVICES:
                rival = (SQL_SERVICES - {service}).pop()
                if rival in self.services:
                    return 100, "", (
                        f"E: Unable to correct problems: {package} conflicts with the installed "
                        f"{rival}-server - both provide the same service on port 3306. Remove "
                        f"the other one first.\n"
                    )
            self.packages.add(package)
            self.binaries.update(CLIENT_BINARIES.get(package, []))
            for path, text in PACKAGE_FILES.get(package, {}).items():
                if path not in self.files:
                    self.files[path] = text
                    self.provisioned.add(path)
            if service == "mariadb":
                self.sql_flavour = "mariadb"
                self.mysql.version = "10.11.8-MariaDB-0ubuntu0.24.04.1"
            if service:
                self.services[service] = True
                self.enabled.add(service)
                self.ports[port] = "127.0.0.1"
                lines.append(f"Setting up {package} ... Created symlink /etc/systemd/system/"
                             f"multi-user.target.wants/{service}.service.")
            else:
                lines.append(f"Setting up {package} ...")
        return 0, "Reading package lists...\n" + "\n".join(lines) + "\n", ""

    def _apt_remove(self, match, command):
        for package in self._tokens(command):
            if package in self.packages:
                self.packages.discard(package)
                service, port = PACKAGE_SERVICES.get(package, (None, None))
                if service:
                    self.services.pop(service, None)
                    self.enabled.discard(service)
                    self.ports.pop(port, None)
        return 0, "Removing packages...\n", ""

    def _systemctl(self, match, command):
        tokens = [t for t in self._tokens(command) if t not in {"sudo", "systemctl"}]
        flags = [t for t in tokens if t.startswith("-")]
        # -p takes a value, which must not be mistaken for a unit name.
        words, skip = [], False
        for token in tokens:
            if skip:
                skip = False
                continue
            if token in {"-p", "--property"}:
                skip = True
                continue
            if not token.startswith("-"):
                words.append(token)
        verb = words[0] if words else "status"
        units = [w.removesuffix(".service") for w in words[1:]]

        if verb == "list-units":
            active = [f"{svc}.service loaded active running" for svc, up in sorted(self.services.items()) if up]
            return 0, "\n".join(active) + "\n", ""
        if verb == "daemon-reload":
            return 0, "", ""

        if not units:
            return 1, "", "Too few arguments.\n"
        code, out = 0, []
        for unit in units:
            unit = self._resolve_unit(unit)
            if verb in {"start", "restart", "reload"}:
                if unit not in self.services:
                    return 5, "", f"Failed to {verb} {unit}.service: Unit {unit}.service not found.\n"
                self.services[unit] = True
                self._apply_config(unit)
            elif verb == "stop":
                if unit in self.services:
                    self.services[unit] = False
            elif verb in {"enable", "disable"}:
                if unit not in self.services:
                    return 1, "", f"Failed to enable unit: Unit {unit}.service does not exist.\n"
                self.enabled.add(unit) if verb == "enable" else self.enabled.discard(unit)
                if "--now" in flags:
                    self.services[unit] = verb == "enable"
                    if verb == "enable":
                        self._apply_config(unit)
            elif verb == "is-active":
                up = self.services.get(unit, False)
                out.append("active" if up else "inactive")
                code = code or (0 if up else 3)
            elif verb == "is-enabled":
                on = unit in self.enabled
                out.append("enabled" if on else "disabled")
                code = code or (0 if on else 1)
            elif verb == "status":
                if unit not in self.services:
                    return 4, "", f"Unit {unit}.service could not be found.\n"
                up = self.services[unit]
                out.append(
                    f"* {unit}.service - {unit} database server\n"
                    f"     Loaded: loaded (/lib/systemd/system/{unit}.service; "
                    f"{'enabled' if unit in self.enabled else 'disabled'})\n"
                    f"     Active: {'active (running)' if up else 'inactive (dead)'}"
                )
                code = code or (0 if up else 3)
            elif verb == "show":
                # Property=value lines, which models grep for ActiveState=active.
                up = self.services.get(unit, False)
                properties = {
                    "Id": f"{unit}.service",
                    "LoadState": "loaded" if unit in self.services else "not-found",
                    "ActiveState": "active" if up else "inactive",
                    "SubState": "running" if up else "dead",
                    "UnitFileState": "enabled" if unit in self.enabled else "disabled",
                    "MainPID": "1234" if up else "0",
                }
                wanted = self._flag_value(command, ("-p", "--property"))
                names = wanted.split(",") if wanted else list(properties)
                if "--value" in flags:
                    out.append("\n".join(properties.get(n, "") for n in names))
                else:
                    out.append("\n".join(f"{n}={properties.get(n, '')}" for n in names))
            elif verb == "cat":
                out.append(f"# /lib/systemd/system/{unit}.service\n[Unit]\n"
                           f"Description={unit} database server\n[Service]\nType=notify\n")
        # --quiet means the exit code is the whole answer; models rely on that in
        # `systemctl is-active --quiet mysql && echo ...`.
        if self._any(flags, {"--quiet", "-q"}):
            out = []
        return code, ("\n".join(out) + "\n" if out else ""), ""

    def _resolve_unit(self, unit: str) -> str:
        """The unit a name actually refers to on this machine.

        MariaDB ships mysql.service and mysqld.service as aliases of mariadb.service
        and models write all three; the same goes for mongodb vs mongod and valkey vs
        valkey-server. An unknown name is returned unchanged, so the caller still
        reports it the way systemd would.
        """
        if unit in self.services:
            return unit
        for alias in UNIT_ALIASES.get(unit, ()):
            if alias in self.services:
                return alias
        return unit

    def _apply_config(self, unit: str) -> None:
        """Re-read the config file a unit was started with.

        Only here, and never at the moment the file is written: editing valkey.conf
        or mongod.conf changes nothing until the service restarts, and that is the
        step a model most often leaves out. Until it does, `ss` keeps reporting the
        old address and the client keeps accepting the old password.
        """
        if unit in SQL_SERVICES:
            text = "\n".join(
                body for path, body in sorted(self.files.items())
                if path.startswith(MYSQL_CONF_PATHS)
            )
            ids = re.findall(r"^\s*server[-_]id\s*=\s*(\d+)", text, re.M | re.I)
            self.mysql.server_id = int(ids[-1]) if ids else 1
            self.mysql.log_bin = not re.search(
                r"^\s*(?:skip[-_]log[-_]bin|disable[-_]log[-_]bin)\b", text, re.M | re.I)
            bound = re.findall(r"^\s*bind[-_]address\s*=\s*(\S+)", text, re.M | re.I)
            self.ports[3306] = bound[-1].strip("'\"") if bound else "127.0.0.1"
        if unit == "valkey-server":
            for line in self.files.get("/etc/valkey/valkey.conf", "").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                name, _, value = line.partition(" ")
                self.valkey.config[name.lower()] = value.strip().strip("'\"")
            bind = self.valkey.config.get("bind", "127.0.0.1")
            self.ports.pop(6379, None)
            port = int(re.sub(r"\D", "", self.valkey.config.get("port", "6379")) or 6379)
            self.ports[port] = "0.0.0.0" if ("0.0.0.0" in bind or "*" in bind) else "127.0.0.1"
        if unit == "mongod":
            text = self.files.get("/etc/mongod.conf", "")
            self.mongo.auth = bool(re.search(r"^\s*authorization\s*:\s*['\"]?enabled",
                                             text, re.M | re.I))
            bind = re.search(r"^\s*bindIp\s*:\s*(.+)$", text, re.M)
            address = bind.group(1).strip().strip("'\"") if bind else "127.0.0.1"
            everywhere = ("0.0.0.0" in address
                          or bool(re.search(r"^\s*bindIpAll\s*:\s*true", text, re.M)))
            self.ports[27017] = "0.0.0.0" if everywhere else "127.0.0.1"

    @staticmethod
    def _any(flags: list[str], wanted: set[str]) -> bool:
        return any(flag in wanted for flag in flags)

    def _service(self, match, command):
        return self._systemctl(match, f"systemctl {match.group(2)} {match.group(1)}")

    def _pg_isready(self, match, command):
        if "pg_isready" not in self.binaries:
            return 127, "", "bash: line 1: pg_isready: command not found\n"
        if self.services.get("postgresql"):
            return 0, "/var/run/postgresql:5432 - accepting connections\n", ""
        return 2, "/var/run/postgresql:5432 - no response\n", ""

    def _mysqladmin(self, match, command):
        binary = match.group(1)
        if binary not in self.binaries:
            return 127, "", f"bash: line 1: {binary}: command not found\n"
        if not self._sql_up():
            return 1, "", f"{binary}: connect to server at 'localhost' failed\n"
        if re.search(r"\bversion\b", command):
            return 0, (f"Server version\t{self.mysql.version}\nUptime:\t\t42 sec\n"), ""
        return 0, "Uptime: 42  Threads: 2  Questions: 10\n", ""

    def _sql_up(self) -> bool:
        """Whether the one MySQL-protocol server on this machine is running."""
        return any(self.services.get(unit) for unit in SQL_SERVICES)

    # ------------------------------------------------------------------- SQL

    def _mysql_client(self, match, command):
        binary = match.group(1)
        if binary not in self.binaries:
            return 127, "", f"bash: line 1: {binary}: command not found\n"
        if "--version" in command or re.search(r"\s-V\b", command):
            return 0, f"{binary}  Ver {self.mysql.version} for Linux\n", ""
        peer = self._flag_value(command, ("-h", "--host"))
        if peer and peer not in {"localhost", "127.0.0.1", "::1", self.address, self.public}:
            return self._remote_sql(peer, command)
        if not self._sql_up():
            server = "MySQL server" if self.sql_flavour == "mysql" else "server"
            return 1, "", (f"ERROR 2002 (HY000): Can't connect to local {server} through "
                           "socket '/var/run/mysqld/mysqld.sock' (2)\n")
        user = self._flag_value(command, ("-u", "--user")) or "root"
        given = self._flag_value(command, ("-p", "--password"))
        denied = (f"ERROR 1045 (28000): Access denied for user '{user}'@'localhost' "
                  f"(using password: {'YES' if given else 'NO'})\n")
        if f"{user}@localhost" not in self.mysql.users:
            return 1, "", denied
        expected = self.mysql.passwords.get(f"{user}@localhost")
        if expected and given is not None and given != expected:
            return 1, "", denied
        sql = self._extract_sql(command, ("-e", "--execute"))
        if sql is None:
            return 1, "", "ERROR: no statement given\n"
        database = self._flag_value(command, ("-D", "--database")) or ""
        if database and database not in self.mysql.databases:
            return 1, "", f"ERROR 1049 (42000): Unknown database '{database}'\n"
        return self._sql(self.mysql, sql, dialect="mysql", user=user, database=database)

    def _remote_sql(self, address: str, command: str):
        """`mysql -h <peer>`: the connection one server makes to another.

        Everything a real cross-server login needs has to be right here, because
        each of them is something a replication task gets wrong: the peer has to be
        on the network, listening on something other than loopback, running, and
        holding an account whose host part matches the address this connection comes
        from.
        """
        peer = self.network.host(address)
        if peer is None or not self.network.connected or not peer.listens(3306) or not peer._sql_up():
            return 2003, "", (f"ERROR 2003 (HY000): Can't connect to MySQL server on '{address}:3306' "
                              "(111 Connection refused)\n")
        user = self._flag_value(command, ("-u", "--user")) or "root"
        account = peer._account(user, self.address)
        if account is None:
            return 1, "", (f"ERROR 1130 (HY000): Host '{self.address}' is not allowed to connect to "
                           "this MySQL server\n")
        expected = peer.mysql.passwords.get(account)
        given = self._flag_value(command, ("-p", "--password"))
        if expected and given != expected:
            return 1, "", (f"ERROR 1045 (28000): Access denied for user '{user}'@'{self.address}' "
                           f"(using password: {'YES' if given else 'NO'})\n")
        sql = self._extract_sql(command, ("-e", "--execute"))
        if sql is None:
            return 1, "", "ERROR: no statement given\n"
        database = self._flag_value(command, ("-D", "--database")) or ""
        if database and database not in peer.mysql.databases:
            return 1, "", f"ERROR 1049 (42000): Unknown database '{database}'\n"
        return peer._sql(peer.mysql, sql, dialect="mysql", user=user, database=database)

    def _account(self, user: str, address: str) -> str | None:
        """The account a login by `user` from `address` matches, or None.

        MySQL's host part is the point: 'repl'@'localhost' does not let a replica in,
        and that is the commonest reason a pair that looks configured does nothing.
        """
        for candidate in (f"{user}@{address}", f"{user}@%", f"{user}@{self.hostname}"):
            if candidate in self.mysql.users:
                return candidate
        for key in sorted(self.mysql.users):  # 10.116.0.% and the like
            name, _, host = key.partition("@")
            if name == user and "%" in host and fnmatch.fnmatch(address, host.replace("%", "*")):
                return key
        return None

    def _psql(self, match, command):
        if "psql" not in self.binaries:
            return 127, "", "bash: line 1: psql: command not found\n"
        if "--version" in command:
            return 0, f"psql (PostgreSQL) {self.postgres.version}\n", ""
        if not self.services.get("postgresql"):
            return 2, "", ("psql: error: connection to server on socket \"/var/run/postgresql/"
                           ".s.PGSQL.5432\" failed: No such file or directory\n")
        role = self._flag_value(command, ("-U", "--username"))
        if role is None:
            # No -U: the role is the OS account, which is postgres via sudo -u.
            role = "postgres" if "sudo -u postgres" in command else "root"
        if role not in self.postgres.users:
            return 2, "", f'psql: error: FATAL:  role "{role}" does not exist\n'
        expected = self.postgres.passwords.get(role)
        given = self.env.get("PGPASSWORD")
        if expected and given is not None and given != expected:
            return 2, "", f'psql: error: FATAL:  password authentication failed for user "{role}"\n'
        database = self._flag_value(command, ("-d", "--dbname"))
        if database and database not in self.postgres.databases:
            return 2, "", f'psql: error: FATAL:  database "{database}" does not exist\n'
        if re.search(r"\s-l\b|--list", command):
            return 0, "\n".join(sorted(self.postgres.databases)) + "\n", ""
        sql = self._extract_sql(command, ("-c", "--command"))
        if sql is None:
            return 1, "", "psql: error: no command given\n"
        # With no -d, psql connects to a database named after the role.
        return self._sql(self.postgres, sql, dialect="postgres", user=role,
                         database=database or role)

    def _createdb(self, match, command):
        if "createdb" not in self.binaries:
            return 127, "", "bash: line 1: createdb: command not found\n"
        # -O app_user names the owner, not another database to create.
        value_flags = {"-u", "-O", "--owner", "-T", "--template", "-E", "--encoding",
                       "-U", "--username", "-h", "--host", "-p", "--port", "-l", "--locale"}
        names, skip = [], False
        for token in self._tokens(command):
            if skip:
                skip = False
                continue
            if token in value_flags:
                skip = True
                continue
            if token.startswith("-") or token in {"sudo", "createdb", "postgres"}:
                continue
            names.append(token)
        for name in names:
            if name in self.postgres.databases:
                return 1, "", f'createdb: error: database creation failed: ERROR:  database "{name}" already exists\n'
            self.postgres.databases.add(name)
        return 0, "", ""

    @staticmethod
    def _flag_value(command: str, flags: tuple[str, ...]) -> str | None:
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        for index, token in enumerate(tokens):
            if token in flags and index + 1 < len(tokens):
                return tokens[index + 1]
            for flag in flags:
                if len(flag) == 2 and token.startswith(flag) and len(token) > 2:
                    return token[2:]
                if flag.startswith("--") and token.startswith(f"{flag}="):
                    return token.split("=", 1)[1]
        return None

    @classmethod
    def _extract_sql(cls, command: str, flags: tuple[str, ...]) -> str | None:
        """Every -e / -c value, in order: both clients accept the flag repeatedly."""
        try:
            tokens = shlex.split(command)
        except ValueError:
            return None
        short = {flag[1] for flag in flags if len(flag) == 2}
        found: list[str] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token in flags and index + 1 < len(tokens):
                found.append(tokens[index + 1])
                index += 2
                continue
            attached = next(
                (token[2:] for flag in flags
                 if len(flag) == 2 and token.startswith(flag) and len(token) > 2),
                None,
            )
            if attached is not None:
                found.append(attached)
                index += 1
                continue
            # Clustered short options: psql -Atc "SELECT 1" puts the value next.
            if (token.startswith("-") and not token.startswith("--")
                    and token[-1] in short and index + 1 < len(tokens)):
                found.append(tokens[index + 1])
                index += 2
                continue
            index += 1
        if found:
            return ";\n".join(cls._unescape(value) for value in found)
        if "<<" in command:  # heredoc-fed SQL
            body = command.split("<<", 1)[1]
            return "\n".join(body.splitlines()[1:-1])
        return None

    @staticmethod
    def _unescape(value: str) -> str:
        """Finish the job shlex leaves half done inside double quotes.

        bash turns \\$ into $ there - which is how a model writes a $$ ... $$
        block on a shell command line - but Python's shlex keeps the backslash.
        """
        return value.replace("\\$", "$").replace("\\`", "`")

    def _sql(self, engine: Engine, sql: str, dialect: str,
             user: str = "", database: str = "") -> tuple[int, str, str]:
        out: list[str] = []
        for statement in self._statements(sql):
            lowered = statement.lower()

            # DO $$ ... $$ is how you write an idempotent CREATE ROLE, so it has
            # to work; CREATE DATABASE inside one does not, as in the real thing.
            if lowered.startswith("do"):
                if "create database" in lowered:
                    return 1, "".join(out), (
                        "ERROR:  CREATE DATABASE cannot be executed from a function\n"
                    )
                self._do_block(engine, statement)
                continue

            # PostgreSQL has no IF NOT EXISTS on CREATE DATABASE/ROLE, and models
            # reach for it out of MySQL habit.
            if dialect == "postgres" and re.match(r"create\s+(database|role|user)\s+if\s+not\s+exists", lowered):
                return 1, "".join(out), 'ERROR:  syntax error at or near "not"\nLINE 1: ...\n'

            create_db = re.match(r"create\s+database\s+(?:if\s+not\s+exists\s+)?[`\"']?(\w+)", lowered)
            if create_db:
                name = create_db.group(1)
                if name in engine.databases and "if not exists" not in lowered:
                    code = 1064 if dialect == "mysql" else 1
                    message = (f"ERROR 1007 (HY000) at line 1: Can't create database '{name}'; "
                               "database exists\n") if dialect == "mysql" else \
                              f'ERROR:  database "{name}" already exists\n'
                    return 1, "".join(out), message
                engine.databases.add(name)
                self._replicate_out(engine)
                continue

            create_user = re.match(r"create\s+(?:user|role)\s+(?:if\s+not\s+exists\s+)?[`\"']?([\w.-]+)", lowered)
            if create_user:
                name = create_user.group(1)
                host = re.search(r"@\s*['\"`]?([\w%.-]+)", lowered)
                key = f"{name}@{host.group(1)}" if host else name
                if key in engine.users and "if not exists" not in lowered:
                    message = (f"ERROR 1396 (HY000) at line 1: Operation CREATE USER failed for "
                               f"'{key}'\n") if dialect == "mysql" else f'ERROR:  role "{name}" already exists\n'
                    return 1, "".join(out), message
                engine.users.add(key)
                self._remember_password(engine, key, statement)
                self._replicate_out(engine)
                continue

            if re.match(r"alter\s+(user|role)\b", lowered):
                key = self._user_key(statement)
                if key:
                    self._remember_password(engine, key, statement)
                continue

            # Replication, before the generic show/select handling: SHOW REPLICA
            # STATUS is the one query whose answer the whole task turns on.
            if dialect == "mysql":
                answer = self._replication(engine, statement, lowered)
                if answer is not None:
                    code, text, error = answer
                    out.append(text)
                    if code:
                        return code, "".join(out), error
                    continue

            if lowered.startswith("grant"):
                self._grant(engine, statement, lowered)
                continue
            if lowered.startswith(("revoke", "flush", "set", "\\c", "comment")):
                continue
            if lowered.startswith("show databases") or lowered.startswith("\\l"):
                like = re.search(r"like\s+['\"]([%\w]+)['\"]", lowered)
                names = sorted(engine.databases)
                if like:
                    needle = like.group(1).strip("%")
                    names = [n for n in names if needle in n]
                out.append(("\n".join(names) + "\n") if names else "")
                continue
            if "version()" in lowered or lowered.startswith("select @@version"):
                out.append(f"{engine.version}\n")
                continue
            if lowered.startswith("select 1"):
                out.append("1\n")
                continue
            # The catalogue queries models write to prove their work.
            if "pg_database" in lowered or "information_schema.schemata" in lowered:
                out.append(self._rows(engine.databases, lowered, ("datname", "schema_name")))
                continue
            if "pg_roles" in lowered or "pg_user" in lowered or "pg_catalog.pg_user" in lowered:
                out.append(self._rows(engine.users, lowered, ("rolname", "usename")))
                continue
            if "mysql.user" in lowered or lowered.startswith("\\du"):
                out.append(self._rows(engine.bare_users(), lowered, ("user",)))
                continue
            # Connection-scoped functions: a model that has just created a user
            # proves it by connecting as them and asking where it landed.
            scalar = self._scalar(lowered, user, database, dialect)
            if scalar is not None:
                out.append(scalar)
                continue
            if lowered.startswith(("select", "show", "\\")):
                out.append("(simulated query, no rows)\n")
                continue
            if lowered.startswith(("drop", "truncate", "delete", "use", "commit", "begin")):
                continue
            # A real parser rejects anything it does not recognise, and models do
            # produce mangled SQL (stray \n from a quoted line continuation).
            snippet = statement[:60].replace("\n", " ")
            if dialect == "mysql":
                return 1, "".join(out), (
                    f"ERROR 1064 (42000) at line 1: You have an error in your SQL syntax; "
                    f"check the manual - near '{snippet}'\n"
                )
            return 1, "".join(out), f'ERROR:  syntax error at or near "{snippet.split()[0]}"\n'
        return 0, "".join(out), ""

    @staticmethod
    def _statements(sql: str) -> list[str]:
        """Split on ; but not inside quotes or a $$ ... $$ body."""
        parts: list[str] = []
        buffer: list[str] = []
        quote, dollar = "", False
        index = 0
        while index < len(sql):
            char = sql[index]
            if sql.startswith("$$", index) and not quote:
                dollar = not dollar
                buffer.append("$$")
                index += 2
                continue
            if dollar:
                buffer.append(char)
            elif quote:
                buffer.append(char)
                if char == quote:
                    quote = ""
            elif char in "'\"":
                quote = char
                buffer.append(char)
            elif char == ";":
                parts.append("".join(buffer))
                buffer = []
            else:
                buffer.append(char)
            index += 1
        parts.append("".join(buffer))
        return [part.strip() for part in parts if part.strip()]

    def _do_block(self, engine: Engine, statement: str) -> None:
        """Apply a PL/pgSQL block as far as create-if-missing goes."""
        body = statement.split("$$")[1] if statement.count("$$") >= 2 else statement
        for inner in re.split(r";\s*", body):
            # search, not match: the statement sits inside IF ... THEN.
            create_user = re.search(r"create\s+(?:user|role)\s+[`\"']?([\w.-]+)",
                                    inner, re.I)
            if create_user:
                key = create_user.group(1)
                engine.users.add(key)
                self._remember_password(engine, key, inner)

    @staticmethod
    def _user_key(statement: str) -> str:
        name = re.search(r"(?:user|role)\s+[`\"']?([\w.-]+)", statement, re.I)
        if not name:
            return ""
        host = re.search(r"@\s*['\"`]?([\w%.-]+)", statement)
        return f"{name.group(1)}@{host.group(1)}" if host else name.group(1)

    @staticmethod
    def _remember_password(engine: Engine, key: str, statement: str) -> None:
        match = re.search(r"(?:identified\s+by|password)\s+['\"]([^'\"]+)['\"]", statement, re.I)
        if match:
            engine.passwords[key] = match.group(1)

    @staticmethod
    def _scalar(lowered: str, user: str, database: str, dialect: str) -> str | None:
        """Answer the one-value functions, or None if this is not one of them."""
        if not lowered.startswith("select"):
            return None
        answers = {
            "current_database()": database,
            "database()": database,
            "current_schema()": "public" if database else "",
            "current_user": user,
            "current_user()": user,
            "user()": f"{user}@localhost" if dialect == "mysql" else user,
            "session_user": user,
            "now()": "2026-08-20 12:00:00+00",
            "current_timestamp": "2026-08-20 12:00:00+00",
        }
        for needle, value in answers.items():
            if needle in lowered:
                # Postgres reports no database as an empty field, not an error.
                return f"{value}\n"
        return None

    @staticmethod
    def _rows(values: set[str], lowered: str, columns: tuple[str, ...]) -> str:
        """Answer a catalogue query, honouring a `WHERE col = 'x'` filter."""
        names = sorted(values)
        for column in columns:
            match = re.search(rf"{column}\s*(?:=|like)\s*['\"]([%\w.-]+)['\"]", lowered)
            if match:
                needle = match.group(1).strip("%")
                names = [name for name in names if needle == name or needle in name]
                break
        # count(*) wants a number, and a model checking for '1' must not be
        # handed the row itself.
        if re.search(r"count\s*\(", lowered):
            return f"{len(names)}\n"
        return ("\n".join(names) + "\n") if names else ""

    # ------------------------------------------------------ MySQL replication

    def _grant(self, engine: Engine, statement: str, lowered: str) -> None:
        """Track the one privilege replication needs; ignore the rest.

        A replica connects with an ordinary login and is then refused at the point it
        asks for the binary log, so the grant has to be remembered separately from the
        account. `GRANT ALL ON *.*` carries REPLICATION SLAVE with it, as it does on a
        real server, and models do reach for it.
        """
        match = _GRANT_TO.search(statement)
        if not match:
            return
        key = f"{match.group(1)}@{match.group(2)}" if match.group(2) else match.group(1)
        everything = "all privileges" in lowered or re.search(r"grant\s+all\b", lowered)
        if "replication slave" in lowered or (everything and "*.*" in lowered):
            engine.repl_grants.add(key)

    def _replication(self, engine: Engine, statement: str, lowered: str):
        """The replication statements, or None if this is not one of them.

        Returns (exit_code, output, stderr) so a caller can append the output and
        stop on an error, the same shape the rest of _sql works in.
        """
        vertical = "\\g" in lowered  # mysql -e 'SHOW REPLICA STATUS\G'
        stripped = lowered.rstrip("\\g").strip()

        source = re.match(r"change\s+(?:replication\s+source|master)\s+to\b", stripped)
        if source:
            if engine.replicating:
                return 1, "", ("ERROR 3021 (HY000) at line 1: This operation cannot be performed "
                               "with a running replica io thread; run STOP REPLICA IO_THREAD "
                               "FOR CHANNEL '' first.\n")
            options = dict(_REPL_OPTION.findall(statement))
            named = {key.lower().replace("master_", "source_"): value
                     for key, value in options.items()}
            engine.source = named.get("source_host", engine.source)
            engine.source_user = named.get("source_user", engine.source_user)
            engine.source_password = named.get("source_password", engine.source_password)
            return 0, "", ""

        if re.match(r"(start|stop)\s+(replica|slave)\b", stripped):
            starting = stripped.startswith("start")
            if starting and not engine.source:
                return 1, "", ("ERROR 1200 (HY000) at line 1: The server is not configured as "
                               "replica; fix in config file or with CHANGE REPLICATION SOURCE TO.\n")
            engine.replicating = starting
            if starting:
                self._replicate_in(engine)
            return 0, "", ""

        if re.match(r"reset\s+(replica|slave)\b", stripped):
            engine.replicating = False
            if "all" in stripped:
                engine.source = engine.source_user = engine.source_password = ""
            return 0, "", ""

        if re.match(r"show\s+(replica|slave)\s+status", stripped):
            return 0, self._replica_status(engine, slave="slave" in stripped, vertical=vertical), ""

        if re.match(r"show\s+(master\s+status|binary\s+log\s+status)", stripped):
            if not engine.log_bin:
                return 0, "", ""  # binlog off: the statement succeeds with no rows
            columns = ("File", "Position", "Binlog_Do_DB", "Binlog_Ignore_DB", "Executed_Gtid_Set")
            values = ("binlog.000002", "1417", "", "", "")
            return 0, self._result(columns, values, vertical), ""

        if re.match(r"show\s+(replicas|slave\s+hosts)", stripped):
            rows = [
                f"{peer.mysql.server_id}\t{peer.address}\t3306\t{engine.server_id}\t"
                for peer in self._replicas()
            ]
            return 0, ("\n".join(rows) + "\n") if rows else "", ""

        variable = re.match(r"show\s+(?:global\s+)?variables\s+like\s+['\"]([\w%]+)", stripped)
        if variable:
            name = variable.group(1).strip("%").replace("-", "_")
            known = {"server_id": str(engine.server_id),
                     "log_bin": "ON" if engine.log_bin else "OFF",
                     "bind_address": self.ports.get(3306, "127.0.0.1"),
                     "read_only": "OFF", "gtid_mode": "OFF"}
            if name in known:
                return 0, f"{name}\t{known[name]}\n", ""
            return None

        at = re.match(r"select\s+@@(?:global\.|session\.)?(server_id|log_bin|read_only)\b", stripped)
        if at:
            name = at.group(1)
            value = {"server_id": str(engine.server_id),
                     "log_bin": "1" if engine.log_bin else "0",
                     "read_only": "0"}[name]
            return 0, f"{value}\n", ""

        assignment = re.match(r"set\s+(?:global|persist|persist_only)\s+server[-_]id\s*=\s*(\d+)",
                              stripped)
        if assignment:
            # Dynamic only: a restart re-reads the config files, so an id set this
            # way and never written down goes back to what the file says.
            engine.server_id = int(assignment.group(1))
            return 0, "", ""

        grants = re.match(r"show\s+grants\s+for\s+['\"`]?([\w.-]+)", stripped)
        if grants:
            user = grants.group(1)
            host = re.search(r"@\s*['\"`]?([\w%.-]+)", statement)
            key = f"{user}@{host.group(1)}" if host else user
            if key not in engine.users:
                return 1, "", (f"ERROR 1141 (42000) at line 1: There is no such grant defined for "
                               f"user '{user}' on host '{host.group(1) if host else '%'}'\n")
            privilege = "REPLICATION SLAVE" if key in engine.repl_grants else "USAGE"
            name, _, where = key.partition("@")
            return 0, f"GRANT {privilege} ON *.* TO `{name}`@`{where or '%'}`\n", ""

        return None

    def _replicas(self) -> list["FakeDroplet"]:
        """The droplets on the network replicating from this one."""
        seen: list[FakeDroplet] = []
        for peer in self.network.hosts.values():
            if peer is self or peer in seen:
                continue
            if peer.mysql.replicating and peer.mysql.source in self.addresses:
                seen.append(peer)
        return seen

    def _replica_status(self, engine: Engine, slave: bool, vertical: bool) -> str:
        """SHOW REPLICA STATUS, with the health of the link worked out afresh.

        Nothing is cached: the answer is computed from the two servers as they are
        now, so fixing a bind-address or adding the grant and asking again shows the
        change - which is exactly the loop a model works in.
        """
        if not engine.source:
            return ""  # never configured: the statement succeeds and prints nothing
        io, error = self._replica_health(engine)
        if not engine.replicating:
            io, error = "No", ""
        end = "master" if slave else "source"
        fields = {
            "Replica_IO_State": (f"Waiting for {end} to send event" if io == "Yes"
                                 else f"Connecting to {end}"),
            "Source_Host": engine.source,
            "Source_User": engine.source_user,
            "Source_Port": "3306",
            "Source_Log_File": "binlog.000002" if io == "Yes" else "",
            "Read_Source_Log_Pos": "1417" if io == "Yes" else "4",
            "Replica_IO_Running": io,
            "Replica_SQL_Running": "Yes" if engine.replicating else "No",
            "Last_IO_Errno": error.split()[0] if error else "0",
            "Last_IO_Error": error.partition(" ")[2] if error else "",
            "Last_SQL_Errno": "0",
            "Last_SQL_Error": "",
            "Seconds_Behind_Source": "0" if io == "Yes" else "NULL",
        }
        if slave:  # SHOW SLAVE STATUS answers in the old spelling, as MariaDB does
            fields = {key.replace("Replica", "Slave").replace("Source", "Master"): value
                      for key, value in fields.items()}
        return self._result(tuple(fields), tuple(fields.values()), vertical)

    def _replica_health(self, engine: Engine) -> tuple[str, str]:
        """(Replica_IO_Running, "errno message") for the link as it stands.

        Every branch here is a real way a configured-looking pair does nothing, and
        each one is invisible except through this status: the CHANGE REPLICATION
        SOURCE TO succeeded, START REPLICA succeeded, and no data moves.
        """
        peer = self.network.host(engine.source)
        if peer is None or not self.network.connected or not peer.listens(3306) or not peer._sql_up():
            return "Connecting", (
                f"2003 error connecting to source '{engine.source_user}@{engine.source}:3306' - "
                "retry-time: 60 retries: 1 message: Can't connect to MySQL server on "
                f"'{engine.source}:3306' (111)"
            )
        account = peer._account(engine.source_user, self.address)
        if account is None:
            return "Connecting", (
                f"1130 error connecting to source '{engine.source_user}@{engine.source}:3306' - "
                f"message: Host '{self.address}' is not allowed to connect to this MySQL server"
            )
        expected = peer.mysql.passwords.get(account)
        if expected and expected != engine.source_password:
            return "Connecting", (
                f"1045 error connecting to source '{engine.source_user}@{engine.source}:3306' - "
                f"message: Access denied for user '{engine.source_user}'@'{self.address}' "
                "(using password: YES)"
            )
        if account not in peer.mysql.repl_grants:
            return "Connecting", (
                "1227 Access denied; you need (at least one of) the REPLICATION SLAVE "
                f"privilege(s) for this operation - grant it to '{engine.source_user}' on "
                f"{peer.hostname} and START REPLICA again"
            )
        if not peer.mysql.log_bin:
            return "Connecting", (
                "1236 Got fatal error 1236 from source when reading data from binary log: "
                f"'Binary log is not open' - the binary log is disabled on {peer.hostname}"
            )
        if peer.mysql.server_id == engine.server_id:
            return "No", (
                f"1593 Fatal error: The replica I/O thread stops because source and replica have "
                f"equal MySQL server ids; these ids must be different for replication to work "
                f"(both are {engine.server_id})"
            )
        return "Yes", ""

    def _replicate_in(self, engine: Engine) -> None:
        """Copy what the source has, if the link is healthy.

        Lazily, at START REPLICA and at nothing else: a replica that is only checked
        for its status is not worth simulating a binlog for, but a model that proves
        the pair by creating a database on one and looking for it on the other needs
        it to be there.
        """
        if self._replica_health(engine)[0] != "Yes":
            return
        peer = self.network.host(engine.source)
        if peer is None:
            return
        engine.databases |= peer.mysql.databases
        engine.users |= peer.mysql.users
        engine.passwords.update(peer.mysql.passwords)

    def _replicate_out(self, engine: Engine) -> None:
        """Carry a write on this server to every replica whose link is healthy.

        The other half of _replicate_in, and the half a model actually tests with:
        it creates a database on the source and looks for it on the replica. Only
        while the link is up, so a broken pair shows as data that never arrives
        rather than only as a status field the model may not have read.
        """
        if engine is not self.mysql:
            return
        for peer in self._replicas():
            if peer._replica_health(peer.mysql)[0] != "Yes":
                continue
            peer.mysql.databases |= engine.databases
            peer.mysql.users |= engine.users
            peer.mysql.passwords.update(engine.passwords)

    @staticmethod
    def _result(columns: tuple[str, ...], values: tuple[str, ...], vertical: bool) -> str:
        """One row, the way the mysql client prints it: \\G vertical or tab-separated."""
        if vertical:
            width = max(len(column) for column in columns)
            body = "\n".join(f"{column:>{width}}: {value}"
                             for column, value in zip(columns, values))
            return f"*************************** 1. row ***************************\n{body}\n"
        return "\t".join(columns) + "\n" + "\t".join(values) + "\n"

    # --------------------------------------------------------------- MongoDB

    def _mongod(self, match, command):
        if "mongod" not in self.binaries:
            return 127, "", "bash: line 1: mongod: command not found\n"
        if "--version" in command:
            return 0, f"db version v{self.mongo.version}\n", ""
        if "--fork" in command:
            self.services["mongod"] = True
            self._apply_config("mongod")
            return 0, ("about to fork child process, waiting until server is ready for "
                       "connections.\nforked process: 4242\nchild process started successfully, "
                       "parent exiting\n"), ""
        # Without --fork, mongod holds the terminal until something kills it, and over
        # SSH with no tty that is the command timeout. Reported as the kill rather
        # than as a start, because a run that does this has lost the step: the server
        # is not under systemd, which is where the next step will look for it.
        return 124, "", ("mongod ran in the foreground and never returned; the command was killed "
                         "at the timeout. Start it with systemctl instead.\n")

    def _mongo_client(self, match, command):
        binary = match.group(1)
        if binary not in self.binaries:
            extra = "" if binary == "mongosh" else " (the legacy mongo shell is gone, use mongosh)"
            return 127, "", f"bash: line 1: {binary}: command not found{extra}\n"
        if "--version" in command:
            return 0, f"{self.mongo.version}\n", ""
        if not self.services.get("mongod"):
            return 1, "", "MongoNetworkError: connect ECONNREFUSED 127.0.0.1:27017\n"

        script = self._extract_sql(command, ("--eval", "-e"))
        if script is None:
            path = self._flag_value(command, ("--file", "-f"))
            if path and path not in self.files:
                return 1, "", f"MongoshInvalidInputError: File {path} does not exist\n"
            if path:
                script = self.files[path]

        user = self._flag_value(command, ("-u", "--username"))
        given = self._flag_value(command, ("-p", "--password"))
        auth_db = self._flag_value(command, ("--authenticationDatabase",)) or "admin"
        # The localhost exception: with authorization on but no user created yet, a
        # local connection may still create the first one. Without it there would be
        # no way to bootstrap, and a run would be locked out the moment it turned
        # access control on.
        if self.mongo.auth and self.mongo.users:
            if not user:
                return 1, "", "MongoServerError[Unauthorized]: Command requires authentication\n"
            expected = self.mongo.passwords.get(user)
            if user not in self.mongo.users or (expected and given != expected):
                return 1, "", "MongoServerError[AuthenticationFailed]: Authentication failed.\n"
            if self.mongo.users[user] != auth_db:
                return 1, "", ("MongoServerError[AuthenticationFailed]: Authentication failed - "
                               f"user '{user}' was created in the {self.mongo.users[user]} "
                               f"database, not {auth_db}\n")

        database = self._mongo_database(command, binary)
        if script is None:
            # A bare shell. stdin is /dev/null here, so it connects, says hello and
            # exits; the guard refuses it before this on the strength of the tty it
            # would want on a real terminal.
            return 0, f"Current Mongosh Log ID: 0\nUsing MongoDB: {self.mongo.version}\n", ""
        return self._mongo_js(script, database)

    def _mongo_database(self, command: str, binary: str) -> str:
        """Which database a mongosh invocation lands in: a name, a URI path, or test."""
        words = self._positionals(command, MONGO_VALUE_FLAGS, {binary, "sudo", "env"})
        target = words[0] if words else ""
        if target.startswith(("mongodb://", "mongodb+srv://")):
            path = re.sub(r"^mongodb(?:\+srv)?://[^/]*/?", "", target)
            return path.split("?")[0] or "test"
        if target and not target.endswith((".js", ".mongodb")):
            return target
        return "test"

    def _mongo_js(self, script: str, database: str) -> tuple[int, str, str]:
        """Interpret the JavaScript a model puts after --eval."""
        out: list[str] = []
        for statement in self._js_statements(script):
            switch = re.match(r"use\s+([\w.-]+)$", statement)
            if switch:
                database = switch.group(1)
                out.append(f"switched to db {database}\n")
                continue
            # printjson(x) and print(x) only decide whether the value reaches stdout,
            # which --eval does anyway; the value inside is the statement.
            wrapper = re.match(r"(?:printjson|print|JSON\.stringify)\s*\((.*)\)$", statement, re.S)
            if wrapper:
                statement = wrapper.group(1).strip()
            # db.getSiblingDB('other').x points one statement at another database
            # without switching to it.
            sibling = re.match(r"db\.getSiblingDB\(\s*['\"]([\w.-]+)['\"]\s*\)\.?(.*)$",
                               statement, re.S)
            target = database
            if sibling:
                target = sibling.group(1)
                statement = f"db.{sibling.group(2)}" if sibling.group(2) else "db"
            if statement in {"", "db", "quit()", "exit"}:
                continue
            if re.match(r"show\s+(?:dbs|databases)$", statement):
                out.append("".join(f"{name}  {40 + 8 * index}.00 KiB\n" for index, name
                                   in enumerate(sorted(self.mongo.databases))))
                continue
            if re.match(r"show\s+(?:collections|tables)$", statement):
                out.append("".join(f"{name}\n" for name
                                   in sorted(self.mongo.databases.get(target, {}))))
                continue
            if re.match(r"show\s+users$", statement):
                out.append("".join(f"{{ user: '{name}', db: '{home}' }}\n" for name, home
                                   in sorted(self.mongo.users.items()) if home == target))
                continue
            code, text, error = self._mongo_call(statement, target)
            out.append(text)
            if code:
                return code, "".join(out), error
        return 0, "".join(out), ""

    def _mongo_call(self, statement: str, database: str) -> tuple[int, str, str]:
        """One db.* call. Reads must not create the database they look in."""
        mongo = self.mongo
        present = mongo.databases.get(database, {})

        if re.match(r"db\.version\(\)$", statement):
            return 0, f"{mongo.version}\n", ""
        if re.search(r"(?:runCommand|adminCommand)\s*\(\s*\{\s*['\"]?ping", statement):
            return 0, "{ ok: 1 }\n", ""
        if (re.search(r"(?:runCommand|adminCommand)\s*\(\s*\{\s*['\"]?shutdown", statement)
                or ".shutdownServer(" in statement):
            self.services["mongod"] = False
            return 0, "", ""
        if ".serverStatus(" in statement:
            return 0, f"{{ host: 'droplet', version: '{mongo.version}', ok: 1 }}\n", ""
        if re.match(r"db\.stats\(", statement):
            return 0, (f"{{ db: '{database}', collections: {len(present)}, "
                       f"objects: {sum(present.values())}, ok: 1 }}\n"), ""
        if re.match(r"db\.auth\(", statement):
            return 0, "{ ok: 1 }\n", ""

        create_user = re.match(r"db\.createUser\s*\((.*)\)$", statement, re.S)
        if create_user:
            name = self._js_field(create_user.group(1), "user")
            if not name:
                return 1, "", "MongoServerError: createUser needs a user field\n"
            if name in mongo.users:
                return 1, "", (f'MongoServerError[Location51003]: User "{name}@'
                               f'{mongo.users[name]}" already exists\n')
            mongo.users[name] = database
            password = self._js_field(create_user.group(1), "pwd")
            if password:
                mongo.passwords[name] = password
            return 0, "{ ok: 1 }\n", ""

        update_user = re.match(r"db\.(?:updateUser|changeUserPassword)\s*\((.*)\)$", statement, re.S)
        if update_user:
            body = update_user.group(1)
            name = self._js_field(body, "user") or self._js_first_string(body)
            if name not in mongo.users:
                return 1, "", f"MongoServerError[UserNotFound]: User {name}@{database} not found\n"
            password = self._js_field(body, "pwd")
            if not password:
                strings = re.findall(r"['\"]([^'\"]+)['\"]", body)
                password = strings[1] if len(strings) > 1 else ""
            if password:
                mongo.passwords[name] = password
            return 0, "{ ok: 1 }\n", ""

        drop_user = re.match(r"db\.dropUser\s*\(\s*['\"]([\w.@-]+)['\"]", statement)
        if drop_user:
            name = drop_user.group(1)
            if name not in mongo.users:
                return 1, "", f"MongoServerError[UserNotFound]: User '{name}@{database}' not found\n"
            mongo.users.pop(name)
            mongo.passwords.pop(name, None)
            return 0, "{ ok: 1 }\n", ""

        if re.match(r"db\.getUsers?\s*\(", statement):
            listed = [f"{{ user: '{name}', db: '{home}' }}"
                      for name, home in sorted(mongo.users.items()) if home == database]
            return 0, "{ users: [ " + ", ".join(listed) + " ], ok: 1 }\n", ""

        if re.match(r"db\.dropDatabase\s*\(", statement):
            mongo.databases.pop(database, None)
            return 0, f"{{ ok: 1, dropped: '{database}' }}\n", ""

        create_collection = re.match(r"db\.createCollection\s*\(\s*['\"]([\w.-]+)['\"]", statement)
        if create_collection:
            name = create_collection.group(1)
            if name in present:
                return 1, "", ("MongoServerError[NamespaceExists]: Collection "
                               f"{database}.{name} already exists.\n")
            mongo.collections(database)[name] = 0
            return 0, "{ ok: 1 }\n", ""

        if re.match(r"db\.getCollectionNames\s*\(", statement):
            return 0, "[ " + ", ".join(f"'{name}'" for name in sorted(present)) + " ]\n", ""

        call = re.match(r"db\.(?:getCollection\(\s*['\"]([\w.-]+)['\"]\s*\)|([\w.-]+))"
                        r"\.(\w+)\s*\((.*)\)\s*[\w.()]*$", statement, re.S)
        if call:
            name = call.group(1) or call.group(2)
            return self._mongo_collection(database, name, call.group(3), call.group(4))

        self.unhandled.append(statement)
        if not statement.startswith("db"):
            first = re.match(r"[\w.]+", statement)
            return 1, "", f"ReferenceError: {first.group(0) if first else statement} is not defined\n"
        return 0, "", ""

    def _mongo_collection(self, database: str, name: str, method: str,
                          body: str) -> tuple[int, str, str]:
        """A db.<collection>.<method>(...) call: documents are counted, not stored."""
        method = method.lower()
        present = self.mongo.databases.get(database, {})
        count = present.get(name, 0)

        if method in {"insertone", "save"}:
            self.mongo.collections(database)[name] = count + 1
            return 0, "{ acknowledged: true, insertedId: ObjectId('66c0f00d0000000000000001') }\n", ""
        if method == "insertmany":
            added = self._js_documents(body)
            self.mongo.collections(database)[name] = count + added
            return 0, f"{{ acknowledged: true, insertedCount: {added} }}\n", ""
        if method in {"countdocuments", "count", "estimateddocumentcount"}:
            return 0, f"{count}\n", ""
        if method == "drop":
            if name not in present:
                return 0, "false\n", ""
            present.pop(name)
            return 0, "true\n", ""
        if method == "deletemany":
            if re.fullmatch(r"\s*\{\s*\}\s*", body or "{}"):
                if name in present:
                    present[name] = 0
                return 0, f"{{ acknowledged: true, deletedCount: {count} }}\n", ""
            return 0, "{ acknowledged: true, deletedCount: 0 }\n", ""
        if method == "deleteone":
            if name in present:
                present[name] = max(0, count - 1)
            return 0, f"{{ acknowledged: true, deletedCount: {1 if count else 0} }}\n", ""
        if method in {"updateone", "updatemany", "replaceone"}:
            return 0, (f"{{ acknowledged: true, matchedCount: {min(count, 1)}, "
                       f"modifiedCount: {min(count, 1)} }}\n"), ""
        if method == "createindex":
            field = re.search(r"['\"]?([\w.]+)['\"]?\s*:\s*-?1", body)
            return 0, f"{field.group(1) if field else 'field'}_1\n", ""
        if method == "getindexes":
            return 0, "[ { v: 2, key: { _id: 1 }, name: '_id_' } ]\n", ""
        if method in {"find", "findone", "aggregate", "distinct"}:
            if not count:
                return 0, ("null\n" if method == "findone" else ""), ""
            return 0, f"(simulated {method}: {database}.{name} holds {count} documents)\n", ""
        self.unhandled.append(f"db.{name}.{method}({body})")
        return 0, "", ""

    @staticmethod
    def _js_statements(script: str) -> list[str]:
        """Split JavaScript on ; and newlines, outside quotes and brackets."""
        parts: list[str] = []
        buffer: list[str] = []
        quote, depth = "", 0
        index = 0
        while index < len(script):
            char = script[index]
            if quote:
                buffer.append(char)
                if char == "\\" and index + 1 < len(script):
                    buffer.append(script[index + 1])
                    index += 2
                    continue
                if char == quote:
                    quote = ""
                index += 1
                continue
            if char in "'\"`":
                quote = char
                buffer.append(char)
                index += 1
                continue
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth = max(0, depth - 1)
            elif depth == 0 and char in ";\n":
                parts.append("".join(buffer))
                buffer = []
                index += 1
                continue
            buffer.append(char)
            index += 1
        parts.append("".join(buffer))
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _js_field(body: str, field: str) -> str:
        match = re.search(rf"['\"]?\b{field}\b['\"]?\s*:\s*['\"]([^'\"]*)['\"]", body)
        return match.group(1) if match else ""

    @staticmethod
    def _js_first_string(body: str) -> str:
        match = re.search(r"['\"]([^'\"]+)['\"]", body)
        return match.group(1) if match else ""

    @staticmethod
    def _js_documents(body: str) -> int:
        """How many documents an insertMany array holds, nested ones not counted."""
        depth, documents, quote = 0, 0, ""
        for char in body:
            if quote:
                if char == quote:
                    quote = ""
                continue
            if char in "'\"":
                quote = char
            elif char == "{":
                documents += 1 if depth == 1 else 0
                depth += 1
            elif char == "[":
                depth += 1
            elif char in "}]":
                depth -= 1
        return max(documents, 1)

    # ---------------------------------------------------------------- Valkey

    def _valkey_server(self, match, command):
        if "valkey-server" not in self.binaries:
            return 127, "", "bash: line 1: valkey-server: command not found\n"
        if "--version" in command or re.search(r"\s-v\b", command):
            return 0, f"Valkey server v={self.valkey.version} bits=64\n", ""
        if re.search(r"--daemonize\s+yes", command):
            self.services["valkey-server"] = True
            self._apply_config("valkey-server")
            return 0, "", ""
        return 124, "", ("valkey-server ran in the foreground and never returned; the command was "
                         "killed at the timeout. Start it with systemctl instead.\n")

    def _valkey_client(self, match, command):
        binary = match.group(1)
        if binary not in self.binaries:
            extra = ("" if binary == "valkey-cli" else
                     " (redis-cli comes from redis-tools; on a Valkey server use valkey-cli)")
            return 127, "", f"bash: line 1: {binary}: command not found{extra}\n"
        if "--version" in command:
            return 0, f"{binary} {self.valkey.version}\n", ""
        host = self._flag_value(command, ("-h", "--host")) or "127.0.0.1"
        port = int(re.sub(r"\D", "", self._flag_value(command, ("-p", "--port")) or "6379") or 6379)
        refused = f"Could not connect to Valkey at {host}:{port}: Connection refused\n"
        if not any(self.services.get(unit) for unit in ("valkey-server", "redis-server")):
            return 1, "", refused
        # A port the server is not listening on is refused too, which is how a model
        # finds out that its edit to valkey.conf moved the server.
        if port not in self.ports:
            return 1, "", refused
        words = self._positionals(command, VALKEY_VALUE_FLAGS, {binary, "sudo", "env"})
        return self._valkey_command(words, self._flag_value(command, ("-a", "--pass")))

    def _valkey_command(self, words: list[str], given: str | None = None) -> tuple[int, str, str]:
        """One command sent to the cache, printed the way a cli with no tty prints it.

        Raw: no `(error)` prefix, no quoted strings, no line numbers - the harness
        runs over SSH without a terminal, and this is what the model actually reads.
        An error reply goes to stderr and the client exits 1.
        """
        keys = self.valkey.keys
        if not words:
            return 0, "", ""  # nothing to send: the client reads stdin and finds none
        verb = words[0].upper()
        rest = words[1:]
        needed = self.valkey.requirepass
        if needed and verb != "AUTH" and given != needed:
            if given is None:
                return 1, "", "NOAUTH Authentication required.\n"
            return 1, "", "WRONGPASS invalid username-password pair or user is disabled.\n"

        if verb == "PING":
            return 0, (f"{rest[0]}\n" if rest else "PONG\n"), ""
        if verb == "AUTH":
            if not needed:
                return 1, "", "ERR Client sent AUTH, but no password is set.\n"
            if rest and rest[-1] == needed:
                return 0, "OK\n", ""
            return 1, "", "WRONGPASS invalid username-password pair or user is disabled.\n"
        if verb == "SET" and len(rest) >= 2:
            keys[rest[0]] = rest[1]
            return 0, "OK\n", ""
        if verb == "GET" and rest:
            value = keys.get(rest[0])
            return 0, ("\n" if value is None else f"{value}\n"), ""
        if verb == "DEL" and rest:
            return 0, f"{sum(1 for key in rest if keys.pop(key, None) is not None)}\n", ""
        if verb == "EXISTS" and rest:
            return 0, f"{sum(1 for key in rest if key in keys)}\n", ""
        if verb == "TYPE" and rest:
            return 0, ("string\n" if rest[0] in keys else "none\n"), ""
        if verb == "KEYS" and rest:
            return 0, "".join(f"{key}\n" for key in sorted(keys)
                              if fnmatch.fnmatchcase(key, rest[0])), ""
        if verb == "DBSIZE":
            return 0, f"{len(keys)}\n", ""
        if verb in {"INCR", "DECR"} and rest:
            current = keys.get(rest[0], "0")
            if not re.fullmatch(r"-?\d+", current):
                return 1, "", "ERR value is not an integer or out of range\n"
            keys[rest[0]] = str(int(current) + (1 if verb == "INCR" else -1))
            return 0, f"{keys[rest[0]]}\n", ""
        if verb == "EXPIRE" and rest:
            # There is no clock here, so nothing ever actually expires; the answer is
            # whether there was a key to put a deadline on.
            return 0, ("1\n" if rest[0] in keys else "0\n"), ""
        if verb == "TTL" and rest:
            return 0, ("-1\n" if rest[0] in keys else "-2\n"), ""
        if verb in {"FLUSHALL", "FLUSHDB"}:
            keys.clear()
            return 0, "OK\n", ""
        if verb in {"SELECT", "SAVE"}:
            return 0, "OK\n", ""
        if verb == "BGSAVE":
            return 0, "Background saving started\n", ""
        if verb == "LASTSAVE":
            return 0, "1771200000\n", ""
        if verb == "SHUTDOWN":
            # The server closes the connection as it goes, so there is nothing to
            # print - and the client does not treat that as an error.
            for unit in ("valkey-server", "redis-server"):
                if unit in self.services:
                    self.services[unit] = False
            return 0, "", ""
        if verb == "INFO":
            return 0, self._valkey_info(rest[0].lower() if rest else ""), ""
        if verb == "CONFIG" and rest:
            return self._valkey_config(rest)
        if verb == "ACL" and rest and rest[0].upper() == "WHOAMI":
            return 0, "default\n", ""
        if verb == "COMMAND" and rest and rest[0].upper() == "COUNT":
            return 0, "240\n", ""
        if verb in VALKEY_UNMODELLED:
            self.unhandled.append(" ".join(words))
            return 0, "", ""
        arguments = ", ".join(f"'{word}'" for word in rest)
        return 1, "", (f"ERR unknown command '{words[0]}', with args beginning with: "
                       f"{arguments}\n")

    def _valkey_config(self, rest: list[str]) -> tuple[int, str, str]:
        action = rest[0].upper()
        config = self.valkey.config
        if action == "GET" and len(rest) > 1:
            return 0, "".join(f"{name}\n{config[name]}\n" for name in sorted(config)
                              if fnmatch.fnmatchcase(name, rest[1].lower())), ""
        if action == "SET" and len(rest) > 2:
            config[rest[1].lower()] = rest[2]
            return 0, "OK\n", ""
        if action == "REWRITE":
            # Writes the running configuration back over the file, which is how a
            # CONFIG SET survives the next restart.
            path = "/etc/valkey/valkey.conf"
            if path not in self.files:
                return 1, "", "ERR The server is running without a config file\n"
            self.files[path] = "".join(f"{name} {value}\n" for name, value
                                      in sorted(config.items()) if value != "")
            return 0, "OK\n", ""
        return 1, "", (f"ERR Unknown CONFIG subcommand or wrong number of arguments for "
                       f"'{rest[0]}'\n")

    def _valkey_info(self, section: str) -> str:
        keyspace = (f"db0:keys={len(self.valkey.keys)},expires=0,avg_ttl=0\n"
                    if self.valkey.keys else "")
        blocks = {
            "server": ("# Server\n"
                       f"valkey_version:{self.valkey.version}\nredis_version:7.2.4\n"
                       "server_name:valkey\nos:Linux 6.8.0-45-generic x86_64\n"
                       "config_file:/etc/valkey/valkey.conf\n"),
            "clients": "# Clients\nconnected_clients:1\n",
            "memory": "# Memory\nused_memory_human:1.02M\nmaxmemory_human:0B\n",
            "persistence": ("# Persistence\naof_enabled:"
                            f"{1 if self.valkey.config.get('appendonly') == 'yes' else 0}\n"
                            "rdb_last_bgsave_status:ok\n"),
            "keyspace": "# Keyspace\n" + keyspace,
        }
        if section in blocks:
            return blocks[section]
        return "".join(blocks.values())

    # ------------------------------------------------------------------ files

    def _ss(self, match, command):
        rows = [f"LISTEN 0 128 {addr}:{port} 0.0.0.0:*" for port, addr in sorted(self.ports.items())]
        return 0, "\n".join(rows) + "\n", ""

    def _heredoc(self, match, command):
        terminator = match.group(2)
        lines = command.splitlines()
        body: list[str] = []
        collecting = False
        for line in lines:
            if not collecting:
                if "<<" in line:
                    collecting = True
                continue
            if line.strip() == terminator:
                break
            body.append(line)
        target = re.search(r"(?:>>?|tee(?:\s+-a)?)\s+(\S+)", command)
        if not target:
            return 0, "\n".join(body) + "\n", ""
        path = target.group(1)
        text = "\n".join(body) + "\n"
        if ">>" in command or re.search(r"tee\s+-a", command):
            self.files[path] = self.files.get(path, "") + text
        else:
            self.files[path] = text
        return 0, "", ""

    def _echo_redirect(self, match, command):
        text = match.group(2).strip().strip("'\"")
        append, path = match.group(3) == ">>", match.group(4)
        self.files[path] = (self.files.get(path, "") if append else "") + text + "\n"
        return 0, "", ""

    def _sed(self, match, command):
        tokens = self._tokens(command)
        path = tokens[-1] if tokens else ""
        script = next((t for t in tokens if t.startswith("s/") or t.startswith("s|")), "")
        if path not in self.files:
            return 4, "", f"sed: can't read {path}: No such file or directory\n"
        parts = script.split(script[1]) if len(script) > 2 else []
        if len(parts) >= 3:
            # \n and \t in the replacement are a newline and a tab to sed, and that is
            # how a model appends a stanza to mongod.conf without rewriting the file.
            replacement = parts[2].replace("\\n", "\n").replace("\\t", "\t")
            self.files[path] = self.files[path].replace(parts[1], replacement)
        return 0, "", ""

    def _cat(self, match, command):
        path = match.group(2)
        if path in self.files:
            return 0, self.files[path], ""
        return 1, "", f"cat: {path}: No such file or directory\n"

    def _grep(self, match, command):
        tokens = [t for t in self._tokens(command) if t not in {"sudo", "grep"}]
        paths = [t for t in tokens if t.startswith("/")]
        patterns = [t for t in tokens if not t.startswith("-") and not t.startswith("/")]
        if not paths:
            return 1, "", ""  # a pipeline; the upstream handler already answered
        needle = patterns[0].strip("'\"") if patterns else ""
        if not any("E" in t for t in tokens if t.startswith("-") and not t.startswith("--")):
            needle = self._basic_regex(needle)
        hits: list[str] = []
        for path in paths:
            for number, line in enumerate(self.files.get(path, "").splitlines(), start=1):
                try:
                    if re.search(needle, line):
                        hits.append(f"{path}:{number}:{line}" if len(paths) > 1 else f"{number}:{line}")
                except re.error:
                    if needle in line:
                        hits.append(line)
        return (0, "\n".join(hits) + "\n", "") if hits else (1, "", "")

    @staticmethod
    def _basic_regex(pattern: str) -> str:
        """Translate grep's default BRE into the regex Python speaks.

        `grep "active (running)"` matches literal parentheses; read as a Python
        regex it would not, and a correct check would look like a failure.
        """
        specials = "(){}+?|"
        out: list[str] = []
        index = 0
        while index < len(pattern):
            char = pattern[index]
            if char == "\\" and index + 1 < len(pattern):
                following = pattern[index + 1]
                out.append(following if following in specials else char + following)
                index += 2
                continue
            out.append("\\" + char if char in specials else char)
            index += 1
        return "".join(out)

    def _test_file(self, match, command):
        probe = re.search(r"(?:test|\[)\s+(?:!\s+)?-([a-z])\s+(.*?)\s*(?:\]\s*)?$", command)
        flag = probe.group(1) if probe else "e"
        operand = probe.group(2).strip().strip("'\"") if probe else ""
        if flag in {"z", "n"}:
            # -z/-n test a string, which is how a model checks whether a variable
            # it just set came back empty.
            true = (operand == "") if flag == "z" else bool(operand)
        else:
            path = operand or next((t for t in self._tokens(command) if t.startswith("/")), "")
            true = path in self.files or any(
                existing.startswith(path.rstrip("/") + "/") for existing in self.files  # a directory
            )
        # `[ ! -f /x ]` is the negation, and answering it the wrong way round
        # would send the model down a branch it did not ask for.
        if re.search(r"(?:test|\[)\s+!\s+-", command):
            true = not true
        return (0 if true else 1), "", ""

    def _ls(self, match, command):
        path = next((t for t in self._tokens(command) if t.startswith("/")), "/")
        children = sorted({
            existing[len(path.rstrip("/")) + 1:].split("/")[0]
            for existing in self.files if existing.startswith(path.rstrip("/") + "/")
        })
        if not children and path not in self.files:
            return 2, "", f"ls: cannot access '{path}': No such file or directory\n"
        return 0, "\n".join(children or [path]) + "\n", ""

    def _filesystem(self, match, command):
        program = match.group(1)
        if program == "rm":
            for token in self._tokens(command):
                self.files.pop(token, None)
        if program == "touch":
            for token in self._tokens(command):
                if token.startswith("/"):
                    self.files.setdefault(token, "")
        return 0, "", ""

    def _run_script_file(self, match, command):
        path = match.group(1)
        if path not in self.files:
            return 127, "", f"bash: {path}: No such file or directory\n"
        return self._script(self.files[path])

    def _echo(self, match, command):
        text = match.group(2).strip()
        # Only the outer shell quoting comes off. Quotes inside survive, which
        # matters when the text is SQL on its way into a client over a pipe.
        if len(text) > 1 and text[0] in "'\"" and text[-1] == text[0]:
            text = text[1:-1]
        return 0, text + "\n", ""

    def _command_v(self, match, command):
        name = match.group(1)
        if name in self.binaries:
            return 0, f"/usr/bin/{name}\n", ""
        return 1, "", ""

    def _dpkg(self, match, command):
        found = sorted(self.packages)
        if not found:
            return 1, "", "dpkg-query: no packages found matching the pattern\n"
        return 0, " ".join(found) + "\n", ""

    def _curl(self, match, command):
        """Fetch a URL. Only the signing keys a vendor repository needs are modelled.

        Adding the MongoDB repository starts with downloading its key, so this has to
        answer; anything else is recorded as unmodelled rather than pretended at.
        """
        url = next((t for t in self._tokens(command) if t.startswith(("http://", "https://"))), "")
        if not url:
            return 2, "", "curl: no URL specified\n"
        if re.search(r"\.(asc|gpg|key)$|/pgp/|pgp\.", url):
            name = url.rstrip("/").rsplit("/", 1)[-1]
            return 0, (f"-----BEGIN PGP PUBLIC KEY BLOCK-----\n\n"
                       f"mQINBGP{name[:8]}simulated/key/material\n"
                       f"-----END PGP PUBLIC KEY BLOCK-----\n"), ""
        # -o writes to a file; a download this simulator knows nothing about still
        # leaves something behind, so a later `test -f` is not answered wrongly.
        target = self._flag_value(command, ("-o", "--output", "-O"))
        if target:
            self.files[target] = f"# downloaded from {url} (simulated)\n"
        self.unhandled.append(command)
        return 0, "", ""

    def _journalctl(self, match, command):
        return 0, "-- No entries --\n", ""

    def _ufw(self, match, command):
        return 0, "Status: inactive\n", ""

    def _id(self, match, command):
        return 0, "root\n" if "-un" in command or match.group(1) == "whoami" else "uid=0(root) gid=0(root)\n", ""

    @staticmethod
    def _tokens(command: str) -> list[str]:
        try:
            return shlex.split(command)
        except ValueError:
            return command.split()

    @classmethod
    def _positionals(cls, command: str, value_flags: set[str], drop: set[str]) -> list[str]:
        """The words on a command line that are neither flags nor flag values.

        `valkey-cli -a hunter2 -n 0 DBSIZE` comes back as ["DBSIZE"], and `mongosh
        admin --eval 'db.version()'` as ["admin"].
        """
        words: list[str] = []
        skip = False
        for token in cls._tokens(command):
            if skip:
                skip = False
                continue
            if token in value_flags:
                skip = True
                continue
            if token.startswith("-") or token in drop:
                continue
            words.append(token)
        return words

    # ------------------------------------------------------------------ checks

    def state(self) -> str:
        written = sorted(p for p in self.files
                         if p != "/etc/os-release" and p not in self.provisioned)
        mongo = ", ".join(f"{name}({sum(collections.values())} docs)"
                          for name, collections in sorted(self.mongo.databases.items()))
        return (
            f"packages: {', '.join(sorted(self.packages)) or 'none'}\n"
            f"running:  {', '.join(sorted(s for s, up in self.services.items() if up))}\n"
            f"enabled:  {', '.join(sorted(self.enabled))}\n"
            f"ports:    {self._listening()}\n"
            f"{self.sql_flavour} databases:    {', '.join(sorted(self.mysql.databases))}\n"
            f"{self.sql_flavour} users:        {', '.join(sorted(self.mysql.users))}\n"
            f"postgres databases: {', '.join(sorted(self.postgres.databases))}\n"
            f"postgres users:     {', '.join(sorted(self.postgres.users))}\n"
            f"mongo databases:    {mongo or 'none'}\n"
            f"mongo users:        {', '.join(sorted(self.mongo.users)) or 'none'}"
            f"{' (auth on)' if self.mongo.auth else ''}\n"
            f"valkey keys:        {', '.join(sorted(self.valkey.keys)) or 'none'}"
            f"{' (password set)' if self.valkey.requirepass else ''}\n"
            f"files written:      {', '.join(written) or 'none'}"
        )
