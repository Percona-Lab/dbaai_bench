"""The three newer engines, driven end to end against the fake droplet.

test_dba_offline.py covers the harness's own behaviour on MySQL and PostgreSQL.
This suite covers what is different about MariaDB, Valkey and MongoDB: the
compatibility names MariaDB answers to, a cache whose password is set at runtime
and only persists if it is written back, and a server that is not in the
distribution at all and so has to be reached through a vendor repository.

Each engine is one scripted run, and each run walks into the trap a model walks
into on a real box - installing over the rival SQL server, restarting before
rewriting the config, installing before the repository is usable - so both the
recovery and the guard's part in it are exercised.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from do_dba.agent import DBAAgent, Limits
from do_dba.inference.pricing import PriceBook
from do_dba.report import HostInfo, RunRecord
from do_dba.secrets import SecretStore
from fake_droplet import FakeDroplet
from test_dba_offline import ScriptedClient, check, fleet_of

RUNS = PROJECT / "_scratch" / "dba_engine_runs"
MODEL = "llama-4-maverick"


def build(droplet, store, task, replies, directory, approve=None):
    fleet = fleet_of(droplet)
    record = RunRecord(
        directory=directory,
        task=task,
        hosts=[HostInfo(name=name, label=label, facts=facts)
               for name, label, facts in fleet.host_lines()],
        model=MODEL,
        mode="auto",
        dry_run=False,
        redact=store.redact,
    )
    client = ScriptedClient(replies)
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
        mode="auto",
        dry_run=False,
        limits=Limits(max_steps=30, command_timeout=300.0),
    )
    return agent, record, client, events


def said(agent, text: str) -> bool:
    """Whether the model was shown `text` - a result, a block or a refusal."""
    return any(text in message["content"] for message in agent.messages)


# --------------------------------------------------------------------- MariaDB
# The interesting part is that nothing here says "mysql" and everything still
# works: one server on port 3306, reached by either name, with two spellings of
# every tool.
MARIADB = [
    "THOUGHT: refresh the package lists first\nACTION: run\nCOMMAND: apt-get update",
    "ACTION: run\nCOMMAND: apt-get install -y mariadb-server mariadb-client",
    # The mistake worth modelling: the client is already there, and asking for
    # MySQL's would uninstall the server that was just put in.
    "THOUGHT: I also want the mysql client\nACTION: run\nCOMMAND: apt-get install -y mysql-server",
    "THOUGHT: read the version through the compatibility name\nACTION: run\n"
    "COMMAND: mysql --version",
    # Blocked: a bare client waits on a prompt that will never come.
    "THOUGHT: look around the server\nACTION: run\nCOMMAND: mariadb",
    "ACTION: run\nCOMMAND: mariadb -e \"CREATE DATABASE app CHARACTER SET utf8mb4\"",
    "ACTION: run\nCOMMAND: mariadb -e \"CREATE USER 'app'@'localhost' IDENTIFIED BY "
    "'{{DBA_SECRET:mariadb_app}}'; GRANT ALL ON app.* TO 'app'@'localhost'\"",
    "ACTION: run\nCOMMAND: mariadb-admin status",
    "ACTION: done\nVERIFY: systemctl is-active mariadb\n"
    "VERIFY: mariadb -e \"SHOW DATABASES\"\n"
    "SUMMARY: MariaDB 10.11 is installed and running on localhost:3306. The app database "
    "and its user exist; the password is {{DBA_SECRET:mariadb_app}}.",
]

# ---------------------------------------------------------------------- Valkey
# A cache has no users and no schema: the whole of its security is the bind
# address and one password, and the password is set on the running server, which
# forgets it at the next restart unless the config is rewritten.
VALKEY = [
    "ACTION: run\nCOMMAND: apt-get update",
    "ACTION: run\nCOMMAND: apt-get install -y valkey-server valkey-tools",
    "ACTION: run\nCOMMAND: valkey-cli PING",
    # Right idea, wrong box: Valkey does not ship redis-cli.
    "THOUGHT: check the redis client too\nACTION: run\nCOMMAND: redis-cli PING",
    "ACTION: run\nCOMMAND: valkey-cli SET session:1 ok",
    # Blocked: MONITOR streams until interrupted and nothing here interrupts.
    "THOUGHT: watch the traffic\nACTION: run\nCOMMAND: valkey-cli MONITOR",
    "ACTION: run\nCOMMAND: valkey-cli CONFIG SET requirepass '{{DBA_SECRET:valkey_pass}}'",
    "THOUGHT: confirm the password took effect\nACTION: run\nCOMMAND: valkey-cli PING",
    "ACTION: run\nCOMMAND: valkey-cli -a '{{DBA_SECRET:valkey_pass}}' PING",
    # Declined by the operator: the keyspace is not ours to empty.
    "THOUGHT: start from an empty keyspace\nACTION: run\n"
    "COMMAND: valkey-cli -a '{{DBA_SECRET:valkey_pass}}' FLUSHALL",
    "ACTION: run\nCOMMAND: valkey-cli -a '{{DBA_SECRET:valkey_pass}}' CONFIG REWRITE",
    "ACTION: run\nCOMMAND: systemctl restart valkey-server",
    "ACTION: run\nCOMMAND: valkey-cli -a '{{DBA_SECRET:valkey_pass}}' DBSIZE",
    "ACTION: done\nVERIFY: systemctl is-active valkey-server\n"
    "VERIFY: valkey-cli -a '{{DBA_SECRET:valkey_pass}}' PING\n"
    "SUMMARY: Valkey 7.2 is running on 127.0.0.1:6379 with a password, kept across "
    "restarts by CONFIG REWRITE. The password is {{DBA_SECRET:valkey_pass}}.",
]

# --------------------------------------------------------------------- MongoDB
# Three walls in a row: the package does not exist, then the repository is not
# signed, then it is signed but apt has not fetched it yet. Then authorization is
# off, and turning it on means editing YAML and restarting - and the user has to
# be created first, because after the restart there is no way in.
MONGODB = [
    "ACTION: run\nCOMMAND: apt-get update",
    "THOUGHT: try the archive first\nACTION: run\nCOMMAND: apt-get install -y mongodb-org",
    "THOUGHT: not in the archive, add the vendor key\nACTION: run\n"
    "COMMAND: curl -fsSL https://pgp.mongodb.com/server-8.0.asc | gpg --dearmor -o "
    "/usr/share/keyrings/mongodb-server-8.0.gpg",
    "ACTION: run\nCOMMAND: echo \"deb [ arch=amd64 "
    "signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg ] "
    "https://repo.mongodb.org/apt/ubuntu noble/mongodb-org/8.0 multiverse\" | "
    "tee /etc/apt/sources.list.d/mongodb-org-8.0.list",
    "THOUGHT: the repository is there now\nACTION: run\n"
    "COMMAND: apt-get install -y mongodb-org",
    "THOUGHT: apt has not read the new list yet\nACTION: run\nCOMMAND: apt-get update",
    "ACTION: run\nCOMMAND: apt-get install -y mongodb-org",
    "ACTION: run\nCOMMAND: systemctl enable --now mongod",
    # Blocked: mongosh with nothing to run is a shell session.
    "THOUGHT: connect and look\nACTION: run\nCOMMAND: mongosh",
    "ACTION: run\nCOMMAND: mongosh --eval 'db.version()'",
    "ACTION: run\nCOMMAND: mongosh app --eval 'db.widgets.insertMany([{name: \"a\"}, "
    "{name: \"b\"}]); db.widgets.countDocuments()'",
    # The administrator has to exist before authorization is turned on: the
    # localhost exception is the only way in on a fresh server.
    "ACTION: run\nCOMMAND: mongosh admin --eval 'db.createUser({user: \"admin\", "
    "pwd: \"{{DBA_SECRET:mongo_admin}}\", roles: [{role: \"root\", db: \"admin\"}]})'",
    "THOUGHT: now turn authorization on\nACTION: run\n"
    "COMMAND: sed -i 's/  bindIp: 127.0.0.1/  bindIp: 127.0.0.1\\n\\nsecurity:\\n"
    "  authorization: enabled/' /etc/mongod.conf",
    # Nothing has changed yet: mongod is still running the old config.
    "THOUGHT: check whether auth is on\nACTION: run\nCOMMAND: mongosh --eval 'db.version()'",
    "ACTION: run\nCOMMAND: systemctl restart mongod",
    "THOUGHT: unauthenticated should fail now\nACTION: run\n"
    "COMMAND: mongosh --eval 'db.version()'",
    "ACTION: run\nCOMMAND: mongosh -u admin -p '{{DBA_SECRET:mongo_admin}}' "
    "--authenticationDatabase admin --eval 'db.version()'",
    "ACTION: done\nVERIFY: systemctl is-active mongod\n"
    "VERIFY: mongosh -u admin -p '{{DBA_SECRET:mongo_admin}}' "
    "--authenticationDatabase admin --eval 'db.adminCommand({ping: 1})'\n"
    "SUMMARY: MongoDB 8.0 is installed from the vendor repository, listening on "
    "127.0.0.1:27017 with authorization enabled. The admin user's password is "
    "{{DBA_SECRET:mongo_admin}}.",
]


def report(name: str, droplet, outcome) -> None:
    print(f"--- {name}: {outcome.status}, "
          f"{outcome.steps} proposed, {outcome.executed} executed")
    print("  " + droplet.state().replace("\n", "\n  "))
    if droplet.unhandled:
        print("  commands the simulator did not model:")
        for command in droplet.unhandled:
            print(f"    {command}")


def mariadb_run(failures: list[str]) -> None:
    droplet, store = FakeDroplet(), SecretStore()
    agent, record, _, events = build(
        droplet, store, "Install MariaDB and create the app database",
        MARIADB, RUNS / "mariadb")
    outcome = agent.run()
    written = record.write_report()
    report("mariadb", droplet, outcome)

    check(failures, outcome.status == "done", f"mariadb run ended {outcome.status}, want done")
    check(failures, "mariadb-server" in droplet.packages, "mariadb-server was not installed")
    check(failures, "mysql-server" not in droplet.packages,
          "mysql-server was installed alongside mariadb-server")
    check(failures, said(agent, "conflicts with the installed"),
          "the rival-server conflict was not explained back to the model")
    check(failures, droplet.sql_flavour == "mariadb",
          f"the droplet is running {droplet.sql_flavour}, want mariadb")
    check(failures, droplet.services.get("mariadb") is True, "mariadb is not running")

    # The compatibility names: one server, three unit names, two tool spellings.
    check(failures, said(agent, "MariaDB"), "the model was never shown that this is MariaDB")
    check(failures, droplet.run("systemctl is-active mysql").stdout.strip() == "active",
          "mysql.service does not resolve to mariadb.service")
    check(failures, droplet.run("mariadb-admin ping").exit_code == 0,
          "mariadb-admin cannot reach the server")

    check(failures, "app" in droplet.mysql.databases, "the app database was not created")
    check(failures, "app@localhost" in droplet.mysql.users, "the app user was not created")
    password = store.resolve("{{DBA_SECRET:mariadb_app}}")
    check(failures, any(password in command for command in droplet.commands),
          "the real password never reached the server")
    report_text = written.read_text(encoding="utf-8")
    check(failures, password not in report_text, "the report leaked the mariadb password")
    check(failures, "{{DBA_SECRET:mariadb_app}}" in report_text,
          "the report lost the placeholder that stands in for it")

    blocked = [message for kind, message in events if kind == "blocked"]
    check(failures, len(blocked) == 1 and "bare mysql" in blocked[0],
          f"bare mariadb should have been blocked once: {blocked}")
    check(failures, not any(c.strip() == "mariadb" for c in droplet.commands),
          "a bare client session was opened on the server")


def valkey_run(failures: list[str]) -> None:
    droplet, store = FakeDroplet(), SecretStore()
    declined: list[str] = []

    def approve(action, detail, reason):
        if "FLUSHALL" in detail:
            declined.append(detail)
            return False
        return True

    agent, record, _, events = build(
        droplet, store, "Install Valkey and put a password on it",
        VALKEY, RUNS / "valkey", approve=approve)
    outcome = agent.run()
    record.write_report()
    report("valkey", droplet, outcome)
    password = store.resolve("{{DBA_SECRET:valkey_pass}}")

    check(failures, outcome.status == "done", f"valkey run ended {outcome.status}, want done")
    check(failures, "valkey-server" in droplet.packages, "valkey-server was not installed")
    check(failures, droplet.services.get("valkey-server") is True, "valkey-server is not running")
    check(failures, droplet.ports.get(6379) == "127.0.0.1",
          f"valkey is not listening on localhost: {droplet.ports}")

    # A Valkey box has no redis-cli, and the model has to be told so rather than
    # shown an empty success.
    check(failures, said(agent, "redis-cli: command not found"),
          "the missing redis client was not reported")

    check(failures, droplet.valkey.requirepass == password,
          "the password was not set on the running server")
    # CONFIG SET alone would be lost at the restart; the rewrite is what makes it
    # survive, and the restart is what proves it did.
    check(failures, password in droplet.files.get("/etc/valkey/valkey.conf", ""),
          "CONFIG REWRITE did not put the password in valkey.conf")
    restarted = droplet.commands.index("systemctl restart valkey-server")
    check(failures, any("DBSIZE" in c for c in droplet.commands[restarted:]),
          "nothing was asked of the cache after the restart")
    check(failures, droplet.run(f"valkey-cli -a {password} PING").stdout.strip() == "PONG",
          "the password does not work after the restart")
    check(failures, droplet.run("valkey-cli PING").exit_code != 0,
          "the cache still answers without a password")
    check(failures, said(agent, "NOAUTH"),
          "the model was not shown that an unauthenticated call now fails")

    check(failures, declined == [c for c in declined if "FLUSHALL" in c] and len(declined) == 1,
          f"the flush should have been offered once: {declined}")
    check(failures, droplet.valkey.keys.get("session:1") == "ok",
          "the declined FLUSHALL emptied the keyspace anyway")
    blocked = [message for kind, message in events if kind == "blocked"]
    check(failures, len(blocked) == 1 and "streams until interrupted" in blocked[0],
          f"MONITOR should have been blocked once: {blocked}")

    transcript = (record.directory / "transcript.jsonl").read_text(encoding="utf-8")
    check(failures, password not in transcript, "the transcript leaked the valkey password")
    check(failures, all(v.exit_code == 0 for v in record.verifications),
          f"a verification failed: {record.verifications}")


def mongodb_run(failures: list[str]) -> None:
    droplet, store = FakeDroplet(), SecretStore()
    agent, record, _, events = build(
        droplet, store, "Install MongoDB with authorization enabled",
        MONGODB, RUNS / "mongodb")
    outcome = agent.run()
    record.write_report()
    report("mongodb", droplet, outcome)
    password = store.resolve("{{DBA_SECRET:mongo_admin}}")

    check(failures, outcome.status == "done", f"mongodb run ended {outcome.status}, want done")
    check(failures, "mongodb-org" in droplet.packages, "mongodb-org was not installed")

    # Each wall in turn, and each one explained rather than silently passed.
    check(failures, said(agent, "Unable to locate package mongodb-org"),
          "the missing vendor package was not reported")
    check(failures, "/usr/share/keyrings/mongodb-server-8.0.gpg" in droplet.files,
          "the vendor keyring was not written")
    check(failures, "repo.mongodb.org" in
          droplet.files.get("/etc/apt/sources.list.d/mongodb-org-8.0.list", ""),
          "the vendor repository was not added")
    installs = [c for c in droplet.commands if c.startswith("apt-get install")]
    check(failures, len(installs) == 3,
          f"expected three install attempts before it worked, saw {len(installs)}")
    updates = [c for c in droplet.commands if c.strip() == "apt-get update"]
    check(failures, len(updates) == 2,
          f"the repository has to be fetched after it is added, saw {len(updates)} updates")

    check(failures, droplet.services.get("mongod") is True, "mongod is not running")
    check(failures, "mongod" in droplet.enabled, "mongod was not enabled at boot")
    check(failures, droplet.ports.get(27017) == "127.0.0.1",
          f"mongod is not listening on localhost: {droplet.ports}")
    check(failures, droplet.mongo.databases.get("app", {}).get("widgets") == 2,
          f"the two documents were not inserted: {droplet.mongo.databases}")

    # The user before the restart, the restart before the check.
    check(failures, droplet.mongo.users.get("admin") == "admin",
          f"the admin user is not in the admin database: {droplet.mongo.users}")
    check(failures, droplet.mongo.auth, "authorization was not turned on")
    check(failures, "security:\n  authorization: enabled" in droplet.files["/etc/mongod.conf"],
          "the authorization stanza is not in mongod.conf")
    check(failures, said(agent, "Unauthorized"),
          "the model was not shown that unauthenticated access now fails")
    # It only became true at the restart: the edit on its own changed nothing,
    # which is the part a model gets wrong.
    restarted = droplet.commands.index("systemctl restart mongod")
    before = [c for c in droplet.commands[:restarted] if "mongosh --eval 'db.version()'" in c]
    check(failures, len(before) == 2, f"the pre-restart check is missing: {before}")

    check(failures, droplet.run(
        f"mongosh -u admin -p {password} --authenticationDatabase admin "
        "--eval 'db.version()'").exit_code == 0,
        "the admin user cannot log in")
    check(failures, droplet.run("mongosh --eval 'db.version()'").exit_code != 0,
          "mongod still answers unauthenticated calls")

    blocked = [message for kind, message in events if kind == "blocked"]
    check(failures, len(blocked) == 1 and "bare mongosh" in blocked[0],
          f"bare mongosh should have been blocked once: {blocked}")
    transcript = (record.directory / "transcript.jsonl").read_text(encoding="utf-8")
    check(failures, password not in transcript, "the transcript leaked the mongo password")
    check(failures, all(v.exit_code == 0 for v in record.verifications),
          f"a verification failed: {record.verifications}")


def main() -> int:
    shutil.rmtree(RUNS, ignore_errors=True)
    failures: list[str] = []
    mariadb_run(failures)
    valkey_run(failures)
    mongodb_run(failures)
    print(f"\n{'FAILURES' if failures else 'all checks passed'}")
    for failure in failures:
        print(f"  FAIL {failure}")
    print(f"\nreports: {RUNS}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
