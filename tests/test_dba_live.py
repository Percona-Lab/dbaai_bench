"""A real DigitalOcean-hosted model driving the harness against the fake droplet.

Nothing here touches a real server, but everything else is real: the model, the
protocol, the guard, the secret substitution, the report.

    uv run python test_dba_live.py [model] [--task "..."] [--steps N]
    uv run python test_dba_live.py [model] --pair      (two servers, replication)

--pair is the multi-server case: two droplets on one private network, passed the
way an operator passes them - a bare list, with no roles assigned. The model has
to work out which server becomes the source, say so, name the server on every
step, and get the two configured differently. It is the harder test by a distance,
so it is opt-in and given a larger step budget. The verdicts below do not care
which of the two it picked, only that it picked one and stayed with it.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]  # the suites sit in tests/, the harness above it
# Ahead of anything installed on purpose: the point is to test this tree.
sys.path.insert(0, str(PROJECT))

from do_dba.agent import DBAAgent, Limits
from do_dba.fleet import Fleet, parse_target
from do_dba.inference.client import InferenceClient
from do_dba.inference.config import base_url, find_api_key, load_dotenv
from do_dba.inference.pricing import PriceBook, format_cost
from do_dba.report import HostInfo, RunRecord
from do_dba.secrets import SecretStore
from fake_droplet import FakeDroplet, Network

RUNS = PROJECT / "_scratch" / "dba_live_runs"
TASK = (
    "Install MySQL and PostgreSQL, make sure both start on boot, and create a database "
    "called app with its own login user on each of them."
)
PAIR_TASK = (
    "Set up MySQL replication between these two servers over the private network: install "
    "MySQL on both, decide which one is the source and which reads from it, and prove the "
    "replica is connected and receiving."
)


def one_fleet() -> Fleet:
    droplet = FakeDroplet(hostname="ubuntu-dba-01", address="10.116.0.2")
    return Fleet.of(droplet, name="fake.droplet")


def pair_fleet() -> Fleet:
    """Two droplets that can see each other, handed over as a bare list.

    No names, because deciding which server is the source is part of what this
    tests: the harness labels them node1 and node2 and the model chooses. Built
    through parse_target so the path is the one --host takes.
    """
    network = Network()
    runners = [
        FakeDroplet(hostname="db-1", address="10.116.0.2", public="203.0.113.10", network=network),
        FakeDroplet(hostname="db-2", address="10.116.0.3", public="203.0.113.11", network=network),
    ]
    targets = [parse_target(runner.public) for runner in runners]
    for target, runner in zip(targets, runners):
        target.runner = runner
    return Fleet(targets)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="llama-4-maverick")
    parser.add_argument("--task", default="")
    parser.add_argument("--pair", action="store_true",
                        help="two servers on a private network, replicating")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--max-cost", type=float, default=0.50)
    args = parser.parse_args()
    task = args.task or (PAIR_TASK if args.pair else TASK)
    steps = args.steps or (30 if args.pair else 18)

    load_dotenv()
    client = InferenceClient(api_key=find_api_key(), base_url=base_url(), label="DigitalOcean")
    fleet = pair_fleet() if args.pair else one_fleet()
    fleet.survey()
    droplet = fleet.only.runner if not args.pair else fleet.targets[0].runner
    store = SecretStore()

    directory = RUNS / args.model.replace(":", "-").replace("/", "-")
    shutil.rmtree(directory, ignore_errors=True)
    record = RunRecord(
        directory=directory,
        task=task,
        hosts=[HostInfo(name=name, label=label, facts=facts)
               for name, label, facts in fleet.host_lines()],
        model=args.model,
        mode="auto",
        dry_run=False,
        redact=store.redact,
    )

    def emit(kind: str, message: str) -> None:
        mark = {"run": "  ->", "ok": "     +", "fail": "     x", "blocked": "  !!", "error": "  !!"}
        first, *rest = message.splitlines() or [""]
        print(f"{mark.get(kind, '   .')} {first}")
        for line in rest[:6]:
            print(f"        {line}")

    def approve(action: str, detail: str, reason: str) -> bool:
        print(f"  ?? flagged ({reason}): {detail.splitlines()[0][:120]}")
        print("     -> approving automatically for this test")
        return True

    agent = DBAAgent(
        client=client,
        model=args.model,
        fleet=fleet,
        task=task,
        record=record,
        store=store,
        prices=PriceBook(),
        emit=emit,
        approve=approve,
        mode="auto",
        limits=Limits(max_steps=steps, command_timeout=300.0, max_cost=args.max_cost),
    )

    print(f"model: {args.model}\ntask:  {task}\nservers: {fleet.label}\n")
    outcome = agent.run()
    store.save(directory / "secrets.json")
    report = record.write_report()

    print(f"\nstatus: {outcome.status}")
    print(f"summary: {outcome.summary[:400]}")
    print(f"\n{outcome.executed} of {outcome.steps} steps executed  "
          f"{record.prompt_tokens:,} in / {record.completion_tokens:,} out  "
          f"{format_cost(outcome.cost)}")
    for target in fleet:
        print(f"\n{target.name} state:")
        print("  " + target.runner.state().replace("\n", "\n  "))

    if args.pair:
        both = [target.runner for target in fleet]
        # Which server took which role is the model's decision, so it is read back
        # off the servers rather than assumed: the one pointed at a source is the
        # replica, and there has to be exactly one of it.
        replicas = [droplet for droplet in both if droplet.mysql.source]
        replica = replicas[0] if len(replicas) == 1 else None
        source = next((d for d in both if d is not replica), None) if replica else None
        roles = (f"{_name(fleet, source)} is the source, {_name(fleet, replica)} the replica"
                 if replica else "no clear source and replica")
        print(f"\nroles the model chose: {roles}")
        verdicts = [
            ("mysql installed on both", all(_sql_installed(d) for d in both)),
            ("mysql running on both", all(d._sql_up() for d in both)),
            ("exactly one server was made a replica", replica is not None),
            ("the replica points at the other's private address",
             replica is not None and replica.mysql.source == source.address),
            ("the source listens off loopback", source is not None and source.listens(3306)),
            ("the server ids differ", both[0].mysql.server_id != both[1].mysql.server_id),
            ("a replication user with the grant exists on the source",
             source is not None and bool(source.mysql.repl_grants)),
            ("the replica's io thread is running",
             replica is not None and replica.mysql.replicating
             and replica._replica_health(replica.mysql)[0] == "Yes"),
            ("the summary says which server took which role",
             all(name in outcome.summary for name in fleet.names)),
            ("no plaintext password in the report", not _leaked(store, report)),
        ]
    else:
        verdicts = [
            ("mysql installed", _sql_installed(droplet)),
            ("postgres installed", any(p.startswith("postgresql") for p in droplet.packages)),
            ("mysql running", droplet.services.get("mysql") or droplet.services.get("mariadb")),
            ("postgres running", droplet.services.get("postgresql")),
            ("mysql enabled at boot", "mysql" in droplet.enabled or "mariadb" in droplet.enabled),
            ("postgres enabled at boot", "postgresql" in droplet.enabled),
            ("mysql app database", "app" in droplet.mysql.databases),
            ("postgres app database", "app" in droplet.postgres.databases),
            ("mysql app user", any(u.startswith("app") for u in droplet.mysql.users)),
            ("postgres app user", any(u.startswith("app") for u in droplet.postgres.users)),
            ("no plaintext password in the report", not _leaked(store, report)),
        ]
    verdicts = [(name, bool(ok)) for name, ok in verdicts]

    print()
    for name, ok in verdicts:
        print(f"  {'+' if ok else 'x'} {name}")

    unhandled = [(target.name, command) for target in fleet for command in target.runner.unhandled]
    if unhandled:
        print(f"\n{len(unhandled)} commands the simulator did not model:")
        for name, command in unhandled[:15]:
            print(f"  [{name}] {command[:140]}")

    print(f"\nreport: {report}")
    return 0 if all(ok for _, ok in verdicts) else 1


def _name(fleet: Fleet, runner) -> str:
    """What the run called this server: its own name if named, else node1/node2."""
    return next((t.name for t in fleet if t.runner is runner), "?")


def _sql_installed(droplet: FakeDroplet) -> bool:
    return any(p.startswith(("mysql-server", "mariadb-server")) for p in droplet.packages)


def _leaked(store: SecretStore, report: Path) -> bool:
    text = report.read_text(encoding="utf-8")
    return any(store.resolve(f"{{{{DBA_SECRET:{name}}}}}") in text for name in store.names)


if __name__ == "__main__":
    raise SystemExit(main())
