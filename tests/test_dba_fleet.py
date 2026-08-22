"""More than one server: naming them, routing steps to them, and a replication run.

No network and no server. Two fake droplets on one fake private network, which is
enough to exercise everything the multi-host path adds: reading --host, labelling
the servers the operator did not name, resolving what a HOST: line refers to,
refusing a step that does not say, scoping a check to one server, and carrying the
name through the transcript and the report.

Both ways of passing servers are covered, because they mean different things: a
bare list of addresses is labelled node1, node2, ... and the model is told to work
out the roles, while `--host primary=...` says the operator has decided and the
model follows.

Which address the servers say things to each other on is checked in every shape a
real pair comes in: a working private network, two machines in different VPCs, a
pair with no private interface at all, and a pair that answers on nothing. --host
is nearly always a public address, so the private one is discovered here or not at
all, and those four cases have three different answers - use it, fall back to the
public address with the scoping unchanged, or abort.

Then the shapes that made a live run get it wrong. A cloud host reports several
private addresses and only one of them reaches the peer, so the anchor address on
the public interface and the docker bridge are ordered behind the VPC address and
more than one is tried before the private network is written off. And a private
network can carry traffic while port 22 stays silent - a firewall in front of the
pair, an sshd bound to the public address - which is a mesh with something to fix,
not a missing network.

The run at the end is a real MySQL replication setup - source config, a
replication user, CHANGE REPLICATION SOURCE TO, START REPLICA - against a
simulator that fails the way a real pair does: bound to loopback it cannot be
reached, without the grant the replica connects and stops, and with matching
server ids it stops as well. So a script that gets the order wrong does not pass.
"""

from __future__ import annotations

import io
import json
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]  # the suites sit in tests/, the harness above it
# Ahead of anything installed on purpose: the point is to test this tree.
sys.path.insert(0, str(PROJECT))

from rich.console import Console

from do_dba.cli import Screen, build_parser, show_network
from do_dba.fleet import Fleet, Target, is_private, parse_addresses, parse_target
from do_dba.protocol import SPEC, ProtocolError, parse, spec
from do_dba.secrets import SecretStore
from do_dba.term import Glyphs
from fake_droplet import FakeDroplet, Network
from test_dba_offline import build, check

RUNS = PROJECT / "_scratch" / "dba_fleet_runs"

PRIMARY, REPLICA = "10.116.0.2", "10.116.0.3"

# The addresses a cloud host reports besides the one its peers share, which is why
# the private address has to be found by probing rather than by reading the first
# line of `ip addr`. ANCHOR is a provider's internal address on the public
# interface, as DigitalOcean's is; BRIDGE is docker's, the same address on every
# machine that runs one; STALE is an interface from a network these two are not
# both on. None of them reaches the peer, and all three look private.
ANCHOR = (("eth0", "10.19.0.5"), ("eth0", "10.19.0.6"))
BRIDGE = ("docker0", "172.17.0.1")
STALE = (("ens5", "192.168.77.5"), ("ens5", "192.168.88.6"))

# The done step's checks, named here because the assertions have to match them
# character for character. \G because the vertical form is the readable one, and it
# is the only one where a field and its value end up on the same line.
IS_ACTIVE = "systemctl is-active mysql"
SHOW_REPLICAS = 'mysql -e "SHOW REPLICAS"'
REPLICA_STATUS = 'mysql -e "SHOW REPLICA STATUS\\G"'


# ------------------------------------------------------------------- --host values

# (spec, name, host, user, port)
TARGETS = [
    ("10.0.0.2", "10.0.0.2", "10.0.0.2", "root", 22),
    ("db.example.com", "db.example.com", "db.example.com", "root", 22),
    ("primary=10.0.0.2", "primary", "10.0.0.2", "root", 22),
    ("  replica = 10.0.0.3  ", "replica", "10.0.0.3", "root", 22),
    ("ubuntu@10.0.0.2", "10.0.0.2", "10.0.0.2", "ubuntu", 22),
    ("10.0.0.2:2222", "10.0.0.2", "10.0.0.2", "root", 2222),
    ("node1=deploy@db2.example.com:2200", "node1", "db2.example.com", "deploy", 2200),
    ("[2001:db8::1]:2222", "2001:db8::1", "2001:db8::1", "root", 2222),
    # Unbracketed IPv6: a trailing :2222 cannot be told from the address, so the
    # whole thing is the address and the port stays the default.
    ("2001:db8::1", "2001:db8::1", "2001:db8::1", "root", 22),
    ("shard-1.a_b=10.0.0.9", "shard-1.a_b", "10.0.0.9", "root", 22),
]

BAD_TARGETS = [
    "",
    "   ",
    "=10.0.0.2",          # named nothing
    "primary=",           # named a server but not where it is
    "9lives=10.0.0.2",    # a name has to start with a letter, or it reads as an address
    "pri mary=10.0.0.2",  # a name has to be one shell-safe word
    "@10.0.0.2",          # no user
    "root@",              # no host
    "10.0.0.2:notaport",
    "10.0.0.2:0",
    "10.0.0.2:70000",
    "[2001:db8::1:2222",  # bracket never closed
]

# What a HOST: line may say, and which server it means. None means no server: the
# step is refused rather than guessed at.
LOOKUPS = [
    ("primary", "primary"),
    ("PRIMARY", "primary"),
    ("replica", "replica"),
    ("[replica]", "replica"),           # models bracket the name
    ("`primary`", "primary"),
    ("\"replica\"", "replica"),
    ("primary,", "primary"),
    ("10.0.0.3", "replica"),            # the address is as good as the name
    ("root@10.0.0.2", "primary"),
    ("primary (10.0.0.2)", "primary"),  # named and then explained
    ("replica - the standby", "replica"),
    ("prim", "primary"),                # an unambiguous prefix
    ("", None),
    ("both", None),
    ("all", None),
    ("10.0.0.", None),                  # matches both, so it answers nothing
    ("db", None),
]


def fleet_pair() -> Fleet:
    """A pair the operator named, which is what the lookup cases are about.

    named=True throughout: an unnamed pair is relabelled node1, node2 by the
    harness, and that path has its own checks in check_labels.
    """
    return Fleet([Target(name="primary", host="10.0.0.2", named=True),
                  Target(name="replica", host="10.0.0.3", named=True)])


def check_targets(failures: list[str]) -> None:
    for text, name, host, user, port in TARGETS:
        try:
            target = parse_target(text)
        except ValueError as exc:
            failures.append(f"parse_target({text!r}) raised {exc}")
            continue
        got = (target.name, target.host, target.user, target.port)
        if got != (name, host, user, port):
            failures.append(f"parse_target({text!r}) -> {got}, want {(name, host, user, port)}")

    for text in BAD_TARGETS:
        try:
            target = parse_target(text)
        except ValueError:
            continue
        failures.append(f"parse_target({text!r}) should have failed, gave {target}")

    # The defaults from -u and -p apply only where the value does not say.
    target = parse_target("10.0.0.2", user="ubuntu", port=2222)
    check(failures, (target.user, target.port) == ("ubuntu", 2222),
          f"the -u/-p defaults were not applied: {target}")
    target = parse_target("deploy@10.0.0.2:22", user="ubuntu", port=2222)
    check(failures, (target.user, target.port) == ("deploy", 22),
          f"the value should win over the defaults: {target}")

    # Whether the operator named the server, which decides both the label it ends
    # up with and what the model is told about who chose the roles.
    check(failures, parse_target("primary=10.0.0.2").named
          and not parse_target("10.0.0.2").named
          and not parse_target("ubuntu@10.0.0.2:2222").named,
          "parse_target does not record whether the name came from the operator")

    check(failures, is_private("10.116.0.2") and is_private("192.168.1.4/24")
          and is_private("172.20.0.5") and not is_private("203.0.113.10")
          and not is_private("172.32.0.1"),
          "private addresses are not being told from public ones")

    # The addresses fact, which carries the interface because that is the only thing
    # in it that tells a VPC address from a bridge or a provider's anchor address.
    found = parse_addresses("eth0=203.0.113.10/20 docker0=172.17.0.1/16 eth1=10.116.0.2/16")
    check(failures, [(a.interface, a.address, a.mask) for a in found]
          == [("eth0", "203.0.113.10", "20"), ("docker0", "172.17.0.1", "16"),
              ("eth1", "10.116.0.2", "16")],
          f"the addresses fact was not read interface by interface: {found}")
    check(failures, [a.private for a in found] == [False, True, True]
          and [a.virtual for a in found] == [False, True, False],
          f"a bridge address is not being told from a VPC one: {found}")
    check(failures, [a.text for a in found][-1] == "10.116.0.2/16",
          f"the mask was dropped from what the model is shown: {found[-1].text!r}")
    # An older server answers with `hostname -I`, which has no interfaces in it. An
    # address with an unknown interface is worth less, not nothing.
    bare = parse_addresses("10.116.0.2 203.0.113.10")
    check(failures, [(a.interface, a.address) for a in bare]
          == [("", "10.116.0.2"), ("", "203.0.113.10")]
          and bare[0].private and not bare[0].virtual,
          f"an address with no interface was not read: {bare}")


def check_fleet(failures: list[str]) -> None:
    pair = fleet_pair()
    for spelling, want in LOOKUPS:
        found = pair.find(spelling)
        got = found.name if found else None
        if got != want:
            failures.append(f"find({spelling!r}) -> {got}, want {want}")

    check(failures, pair.many and not Fleet([Target(name="a", host="10.0.0.2")]).many,
          "many is wrong for one server or for two")
    check(failures, pair.names == ["primary", "replica"], f"names are {pair.names}")
    check(failures, pair.slug == "10.0.0.2-plus1", f"the run directory slug is {pair.slug}")
    check(failures, Fleet([Target(name="a", host="10.0.0.2")]).slug == "10.0.0.2",
          "a single server's slug should be its address")
    check(failures, "primary (root@10.0.0.2)" in pair.label, f"the fleet label is {pair.label}")

    # An unnamed --host takes its address for a name, which is not a clash with
    # itself - the commonest single-server invocation of all.
    try:
        Fleet([parse_target("10.0.0.2")])
    except ValueError as exc:
        failures.append(f"one unnamed host was rejected: {exc}")

    for label, targets in (
        ("two servers with one name",
         [Target(name="db", host="10.0.0.2", named=True),
          Target(name="db", host="10.0.0.3", named=True)]),
        ("the same server twice",
         [Target(name="a", host="10.0.0.2", named=True),
          Target(name="b", host="10.0.0.2", named=True)]),
        ("a name that is another server's address",
         [Target(name="10.0.0.3", host="10.0.0.2", named=True),
          Target(name="replica", host="10.0.0.3", named=True)]),
        ("no servers at all", []),
    ):
        try:
            Fleet(targets)
        except ValueError:
            continue
        failures.append(f"Fleet accepted {label}")

    # Two servers behind one address on different ports is a real arrangement, so
    # it is allowed - but then the address alone cannot say which one is meant.
    ports = Fleet([Target(name="a", host="10.0.0.2", port=2222, named=True),
                   Target(name="b", host="10.0.0.2", port=2223, named=True)])
    check(failures, ports.find("10.0.0.2") is None,
          "an address shared by two servers should resolve to neither")
    check(failures, ports.find("root@10.0.0.2:2223") is ports.targets[1],
          "the full user@host:port should still pick one of them out")
    check(failures, ports.find("a") is ports.targets[0], "a name should still work")

    parser = build_parser()
    args = parser.parse_args(["--host", "primary=10.0.0.2", "--host", "replica=10.0.0.3",
                              "--task", "x"])
    check(failures, args.host == ["primary=10.0.0.2", "replica=10.0.0.3"],
          f"--host did not collect both servers: {args.host}")
    args = parser.parse_args(["--host", "10.0.0.2", "--task", "x"])
    check(failures, args.host == ["10.0.0.2"], f"one --host gave {args.host}")
    args = parser.parse_args(["--host", "10.0.0.2", "--host", "10.0.0.3", "--task", "x"])
    check(failures, args.host == ["10.0.0.2", "10.0.0.3"],
          f"a bare list of servers gave {args.host}")


# ------------------------------------------------------------- names and labels

def check_labels(failures: list[str]) -> None:
    """A bare list of servers: the harness labels them, the model assigns the roles.

    This is the ordinary way to ask for a cluster - the operator has the machines
    and no opinion about which is which - so the labels have to be stable, the
    addresses have to survive being labelled, and the model has to be told that
    working out the roles is part of its job.
    """
    bare = Fleet([parse_target("10.0.0.2"), parse_target("10.0.0.3")])
    check(failures, bare.names == ["node1", "node2"], f"a bare pair was labelled {bare.names}")
    check(failures, not bare.assigned, "a bare list should leave the roles to the model")
    check(failures, [t.host for t in bare] == ["10.0.0.2", "10.0.0.3"],
          f"labelling moved the servers: {[t.host for t in bare]}")
    check(failures, bare.slug == "10.0.0.2-plus1",
          f"the run directory should still be named after the address, not the label: {bare.slug}")
    found = bare.find("10.0.0.3")
    check(failures, found is not None and found.name == "node2",
          f"an address should resolve to the server the harness labelled: {found}")
    check(failures, bare.find("node2") is bare.targets[1], "the label itself does not resolve")
    check(failures, "node2 (root@10.0.0.3)" in bare.label, f"the fleet label is {bare.label}")

    # One name given and the rest left open: the operator's name stands, the others
    # are labelled around it, and the model follows what it is told.
    mixed = Fleet([parse_target("web=10.0.0.2"), parse_target("10.0.0.3")])
    check(failures, mixed.names == ["web", "node1"], f"a mixed list was labelled {mixed.names}")
    check(failures, mixed.assigned, "a list with a name in it should count as assigned")

    # An operator who uses node1 or node2 themselves: the labels go around the names
    # rather than colliding with them, which Fleet would refuse outright.
    clash = Fleet([parse_target("node1=10.0.0.2"), parse_target("10.0.0.3")])
    check(failures, clash.names == ["node1", "node2"], f"a taken label gave {clash.names}")
    three = Fleet([parse_target("node2=10.0.0.2"), parse_target("10.0.0.3"),
                   parse_target("10.0.0.4")])
    check(failures, three.names == ["node2", "node1", "node3"],
          f"labelling around a taken node2 gave {three.names}")

    # Alone there is nothing to tell apart: the name never reaches the model, and
    # relabelling would only change what a single-server transcript has always said.
    lone = Fleet([parse_target("10.0.0.2")])
    check(failures, lone.names == ["10.0.0.2"] and not lone.assigned,
          f"a single unnamed server was renamed to {lone.names}")

    # What the model is actually told, which is the whole point of the distinction.
    _, _, unnamed_pair = droplet_pair(named=False)
    check(failures, unnamed_pair.names == ["node1", "node2"],
          f"the surveyed bare pair was labelled {unnamed_pair.names}")
    open_brief, fixed_brief = unnamed_pair.brief(), fleet_pair().brief()
    for needle in ("labels the harness assigned", "which server takes",
                   "say which in your first step"):
        check(failures, needle in open_brief, f"the brief for a bare list is missing {needle!r}")
    check(failures, "chosen by the operator" not in open_brief,
          "the brief claims the operator named servers they did not name")
    check(failures, "chosen by the operator" in fixed_brief and "Follow them." in fixed_brief,
          "the brief for named servers does not say to follow the names")
    check(failures, "labels the harness assigned" not in fixed_brief,
          "the brief calls the operator's own names labels")

    open_rules = system_prompt(unnamed_pair, "prompt-open")
    fixed_rules = system_prompt(fleet_pair(), "prompt-fixed")
    check(failures, "15. The names above are labels, not roles" in open_rules,
          "a bare list of servers is not told the roles are its own to decide")
    check(failures, "labels, not roles" not in fixed_rules,
          "named servers are told to decide roles the operator already decided")
    check(failures, "14. Verify from both ends" in open_rules
          and "14. Verify from both ends" in fixed_rules,
          "the multi-server rules are not all there")
    # Rule 13 is a placeholder, and a placeholder written wrong is a password the
    # model invents itself.
    for rules in (open_rules, fixed_rules):
        check(failures, "The same {{DBA_SECRET:name}} placeholder" in rules,
              "the multi-server rules mangled the secret placeholder")


def system_prompt(fleet: Fleet, name: str) -> str:
    """The system prompt an agent for this fleet would send. Nothing is executed."""
    agent, _, _, _ = build(fleet.targets[0].runner, SecretStore(), [], fleet=fleet,
                           task=TASK, directory=RUNS / name)
    return agent.messages[0]["content"]


# --------------------------------------------------------------------- the spec

def check_spec(failures: list[str]) -> None:
    check(failures, spec() == SPEC and spec(()) == SPEC and spec(["only"]) == SPEC,
          "the single-server spec changed; it should be untouched by this feature")
    check(failures, "HOST:" not in SPEC, "the single-server spec should not mention HOST")

    text = spec(["primary", "replica"])
    for needle in (
        "HOST: primary",
        "(which server: primary, replica)",
        "VERIFY: [replica] a check to run on that server only",
        "one of: primary, replica",
        "A VERIFY: line with no [name]",
    ):
        check(failures, needle in text, f"the multi-server spec is missing {needle!r}")
    # The generated comment has to line up with the ones already in the block, or
    # the format instructions read as two half-formatted lists.
    columns = {line.index("(") for line in text.splitlines()
               if "(" in line and line.startswith(("HOST:", "ACTION:", "MODE:", "COMMAND: the"))}
    check(failures, len(columns) == 1, f"the spec's comments do not line up: {sorted(columns)}")


# ------------------------------------------------------------------- the protocol

def check_protocol(failures: list[str]) -> None:
    pair = fleet_pair()

    step = parse("ACTION: run\nHOST: replica\nCOMMAND: systemctl restart mysql", pair)
    check(failures, step.host == "replica", f"HOST: replica gave {step.host!r}")
    step = parse("ACTION: run\nHOST: [10.0.0.2]\nCOMMAND: uptime", pair)
    check(failures, step.host == "primary",
          f"an address in HOST should resolve to the server's own name, gave {step.host!r}")

    for label, reply in (
        ("no HOST line at all", "ACTION: run\nCOMMAND: apt-get update"),
        ("a name that is no server", "ACTION: run\nHOST: both\nCOMMAND: apt-get update"),
        ("an empty HOST line", "ACTION: run\nHOST:\nCOMMAND: apt-get update"),
        ("a write_file with no HOST",
         "ACTION: write_file\nPATH: /etc/my.cnf\nCONTENT_BEGIN\n[mysqld]\nCONTENT_END"),
    ):
        try:
            step = parse(reply, pair)
        except ProtocolError as exc:
            message = str(exc)
            check(failures, "primary, replica" in message,
                  f"the refusal for {label} does not list the servers: {message}")
            check(failures, "nothing was run" in message.lower(),
                  f"the refusal for {label} does not say nothing ran: {message}")
            continue
        failures.append(f"parse({label}) was accepted as {step.host!r}")

    # done and abort run nothing, so they need no server.
    step = parse("ACTION: done\nVERIFY: systemctl is-active mysql\nSUMMARY: it is up", pair)
    check(failures, step.host == "" and step.verify[0].host == "",
          f"a done step should need no server, gave {step.host!r}")

    # One server: a HOST line is neither required nor believed over the fleet.
    one = Fleet([Target(name="only", host="10.0.0.2")])
    step = parse("ACTION: run\nCOMMAND: apt-get update", one)
    check(failures, step.host == "only", f"one server should need no HOST, gave {step.host!r}")
    step = parse("ACTION: run\nHOST: ubuntu-dba-01\nCOMMAND: apt-get update", one)
    check(failures, step.host == "only",
          f"one server should absorb any HOST line, gave {step.host!r}")

    # Scoped checks.
    step = parse(
        "ACTION: done\n"
        "VERIFY: systemctl is-active mysql\n"
        "VERIFY: [replica] mysql -e 'SHOW REPLICA STATUS'\n"
        "VERIFY: [ -f /etc/mysql/my.cnf ] && echo present\n"
        "SUMMARY: replication is running from primary to replica",
        pair,
    )
    got = [(c.host, c.command) for c in step.verify]
    want = [
        ("", "systemctl is-active mysql"),
        ("replica", "mysql -e 'SHOW REPLICA STATUS'"),
        # A test command starts with the same bracket and is not a scope.
        ("", "[ -f /etc/mysql/my.cnf ] && echo present"),
    ]
    check(failures, got == want, f"the verify lines parsed as {got}")

    try:
        parse("ACTION: done\nVERIFY: [arbiter] mysql -e 'SELECT 1'\nSUMMARY: done", pair)
        failures.append("a VERIFY scoped to no server was accepted")
    except ProtocolError as exc:
        check(failures, "primary, replica" in str(exc),
              f"the scoped-verify refusal does not list the servers: {exc}")

    # Without a fleet nothing is checked and nothing is rewritten: the single-server
    # callers and the guard suite depend on that.
    step = parse("ACTION: run\nHOST: whatever\nCOMMAND: uptime")
    check(failures, step.host == "whatever", f"parse with no fleet rewrote the host: {step.host!r}")


# -------------------------------------------------------------------- the network

def droplet_pair(connected: bool = True,
                 named: bool = True,
                 blocked: tuple[str, ...] = (),
                 private: bool = True,
                 filtered: tuple[str, ...] = (),
                 extra: tuple[tuple[tuple[str, str], ...], ...] = ((), ()),
                 ssh_on_public: bool = False) -> tuple[FakeDroplet, FakeDroplet, Fleet]:
    """Two droplets, built out of --host values as the CLI builds them.

    The --host values are the public addresses, because that is what an operator
    has to hand; the private ones exist only on the droplets, which is what makes
    discovering them the harness's job.

    The ways the network can be other than fine: blocked=(PRIMARY, REPLICA) is two
    droplets in different VPCs, each with a private address neither can reach;
    private=False is a pair with no private interface at all; connected=False is a
    pair that answers on nothing; filtered=(PRIMARY, REPLICA) is a firewall in
    front of a network that works, where the port is silent and ping is not.

    extra gives each droplet, in order, the addresses a real cloud host reports
    besides the two that are on the network - a provider's anchor address, a docker
    bridge, an interface from a VPC it has left - as (interface, address) pairs.
    Nothing holds them, so probing one is a timeout, which is the point: they are
    what makes picking a peer's private address a question rather than a lookup.

    ssh_on_public binds sshd to the public address alone, so a peer's connection to
    the private one is refused: the packets cross, and there is a private network.

    named=False is the bare list an operator normally passes, where the harness
    labels the servers node1 and node2 and the roles are the model's to choose.
    """
    network = Network(connected=connected, blocked=blocked, filtered=filtered)
    primary = FakeDroplet(hostname="db-primary", address=PRIMARY if private else "",
                          public="203.0.113.10", network=network, extra=extra[0])
    replica = FakeDroplet(hostname="db-replica", address=REPLICA if private else "",
                          public="203.0.113.11", network=network, extra=extra[1])
    if ssh_on_public:
        for droplet in (primary, replica):
            droplet.ports[22] = droplet.public
    specs = (["primary=203.0.113.10", "replica=203.0.113.11"] if named
             else ["203.0.113.10", "203.0.113.11"])
    targets = [parse_target(text) for text in specs]
    for target, runner in zip(targets, (primary, replica)):
        target.runner = runner
    fleet = Fleet(targets)
    fleet.survey()
    return primary, replica, fleet


def network_screen(fleet: Fleet) -> str:
    """What show_network prints for this fleet, as the operator reads it.

    ASCII glyphs and a wide console, so the lines can be compared instead of
    guessed at through colour codes and wrapping.
    """
    buffer = io.StringIO()
    console = Console(file=buffer, no_color=True, soft_wrap=True, width=200, legacy_windows=False)
    show_network(Screen(console, Glyphs(fancy=False)), fleet)
    return buffer.getvalue()


def check_peers(failures: list[str]) -> None:
    primary, replica, fleet = droplet_pair()
    peers = {target.name: target.facts.values.get("peers", "") for target in fleet}
    check(failures, peers["primary"] == f"replica private {REPLICA}:22 reachable",
          f"the primary does not see the replica: {peers['primary']!r}")
    check(failures, peers["replica"] == f"primary private {PRIMARY}:22 reachable",
          f"the replica does not see the primary: {peers['replica']!r}")
    # The private address is the one a peer is pointed at, never the public one -
    # both in what is probed and in what the server is said to have.
    check(failures, fleet.targets[0].addresses[0].startswith(PRIMARY),
          f"the private address is not first: {fleet.targets[0].addresses}")
    check(failures, (fleet.targets[0].private, fleet.targets[1].private) == (PRIMARY, REPLICA),
          f"the private addresses read as {[t.private for t in fleet]}")
    check(failures, any(f"/dev/tcp/{REPLICA}/22" in command for command in primary.commands),
          f"the primary never probed the replica's private address: {primary.commands[-3:]}")
    check(failures, not any("/dev/tcp/203.0.113" in command
                            for command in primary.commands + replica.commands),
          "a public address was probed although the private one answered")

    # A single server is not asked the question at all.
    lone = FakeDroplet()
    solo = Fleet([Target(name="only", host="10.0.0.2", runner=lone)])
    solo.survey()
    check(failures, "peers" not in solo.only.facts.values,
          "a single-server run should not have a peers fact")
    check(failures, not solo.paths and not solo.private_mesh,
          "a single server should have no peer paths and no mesh")
    check(failures, not solo.network_note(),
          f"a single server was told about a private network: {solo.network_note()!r}")
    check(failures, not any("/dev/tcp" in command for command in lone.commands),
          "a single-server run probed for peers anyway")

    brief = fleet.brief()
    for needle in ("THE SERVERS (2)", "primary", "replica", PRIMARY, REPLICA, "reachable"):
        check(failures, needle in brief, f"the prompt's server list is missing {needle!r}")
    check(failures, "THIS SERVER" in solo.brief() and "THE SERVERS" not in solo.brief(),
          "a single server should still be described as this server")


def check_private_candidates(failures: list[str]) -> None:
    """A cloud host reports several private addresses; only one reaches the peer.

    This is the case a live run got wrong. Both servers were on a working VPC and
    the harness reported "private 10.10.0.6:22 unreachable, public reachable",
    because it probed the first private address `ip addr` happened to list - the
    provider's internal anchor address, which sits on the public interface and
    routes nowhere - and read that one failure as "there is no private network".

    So the addresses are ordered by what they are (the interface says), and more
    than one is tried before the private network is written off.
    """
    # 1. An anchor address on the public interface and a docker bridge, listed ahead
    # of the VPC address. The VPC address is still the one used, and the two dead
    # ones cost nothing: an address beside the public one is a provider's, and a
    # bridge is on this machine too.
    primary, replica, fleet = droplet_pair(extra=((ANCHOR[0], BRIDGE), (ANCHOR[1], BRIDGE)))
    check(failures, fleet.private_mesh and not fleet.unproven_paths,
          f"a VPC behind a cloud host's other addresses was missed: "
          f"{[p.describe() for p in fleet.paths]}")
    check(failures, (fleet.targets[0].private, fleet.targets[1].private) == (PRIMARY, REPLICA),
          f"the private address read as {[t.private for t in fleet]}")
    check(failures, fleet.targets[1].private_candidates == [REPLICA, ANCHOR[1][1]],
          f"the addresses are tried in the wrong order: {fleet.targets[1].private_candidates}")
    probed = [c for c in primary.commands + replica.commands if "/dev/tcp" in c]
    check(failures, not any("10.19.0." in command for command in probed),
          f"an anchor address was probed although the VPC answered first: {probed}")
    check(failures, not any(BRIDGE[1] in command for command in probed),
          f"a docker bridge - this machine's own address - was probed as a peer's: {probed}")
    # Ordered, not filtered: the model still sees everything the server reported.
    listed = fleet.targets[0].addresses
    check(failures, listed[0].startswith(PRIMARY) and any(BRIDGE[1] in a for a in listed)
          and any("203.0.113.10" in a for a in listed),
          f"the addresses were reordered into something incomplete: {listed}")

    # 2. The dead address on an interface of its own, so ordering cannot help and the
    # probe has to try the next one. Without this, a working VPC reads as no VPC.
    primary, replica, fleet = droplet_pair(extra=((STALE[0],), (STALE[1],)))
    check(failures, fleet.private_mesh,
          f"the second private address was never tried: {[p.describe() for p in fleet.paths]}")
    check(failures, any(f"/dev/tcp/{STALE[1][1]}/22" in c for c in primary.commands),
          "the first private address was not probed at all")
    check(failures, (fleet.targets[0].private_confirmed,
                     fleet.targets[1].private_confirmed) == (PRIMARY, REPLICA),
          f"the address that answered was not recorded: "
          f"{[t.private_confirmed for t in fleet]}")
    peers = fleet.targets[0].facts.values.get("peers", "")
    check(failures, peers == f"replica private {REPLICA}:22 reachable",
          f"the peer line does not report the address that worked: {peers!r}")

    # 3. Every private address dead, which is case 2 of check_network_states with the
    # cloud's noise on top: the fallback is the same, and each address tried is named
    # so an operator can see it was not one address that was written off.
    _, _, vpcs = droplet_pair(blocked=(PRIMARY, REPLICA),
                             extra=((ANCHOR[0], BRIDGE), (ANCHOR[1], BRIDGE)))
    check(failures, not vpcs.private_mesh and len(vpcs.broken_paths) == 2,
          "unreachable private addresses were accepted as a mesh")
    peers = vpcs.targets[0].facts.values.get("peers", "")
    check(failures, peers == f"replica private {REPLICA}:22 unreachable "
                             f"(also tried {ANCHOR[1][1]}), public 203.0.113.11:22 reachable",
          f"the peer line does not name every address tried: {peers!r}")
    check(failures, "Use the public address in each label" in vpcs.network_note(),
          f"the public fallback was not offered: {vpcs.network_note()}")

    # 4. An address in a network this server is not on at all. `ip route get` says so
    # at once, which is a different fault from a packet that goes out and dies, and
    # the one an operator can act on: the interface is up and on the wrong network.
    _, _, elsewhere = droplet_pair(blocked=(PRIMARY, REPLICA), extra=((STALE[0],), (STALE[1],)))
    peers = elsewhere.targets[0].facts.values.get("peers", "")
    check(failures, peers == f"replica private {STALE[1][1]}:22 unreachable (this server has no "
                             f"route to it) (also tried {REPLICA}), "
                             "public 203.0.113.11:22 reachable",
          f"a missing route reads the same as a dropped packet: {peers!r}")


def check_partial_paths(failures: list[str]) -> None:
    """A private network that carries traffic while port 22 stays silent.

    Two shapes of it, and neither means there is no private network: a firewall in
    front of the pair that drops the port and passes ping, and an sshd bound to the
    public address alone. Both used to read as "unreachable" and send a cluster onto
    the public internet - so both are meshes here, and both are said out loud,
    because the database port will need opening and the model has to know that.
    """
    for label, pair in (("a filtered port", droplet_pair(filtered=(PRIMARY, REPLICA))),
                        ("sshd on the public address", droplet_pair(ssh_on_public=True))):
        primary, replica, fleet = pair
        check(failures, fleet.private_mesh,
              f"{label}: a private network that carries traffic was written off: "
              f"{[p.describe() for p in fleet.paths]}")
        check(failures, len(fleet.unproven_paths) == 2 and not fleet.broken_paths,
              f"{label}: the paths are not reported as unproven: "
              f"{[p.describe() for p in fleet.paths]}")
        check(failures, all(not path.private_ok and path.private_works for path in fleet.paths),
              f"{label}: port 22 answering and packets crossing were confused")
        check(failures, (fleet.targets[0].private, fleet.targets[1].private) == (PRIMARY, REPLICA),
              f"{label}: the private addresses read as {[t.private for t in fleet]}")
        check(failures, not any("/dev/tcp/203.0.113" in c
                                for c in primary.commands + replica.commands),
              f"{label}: the public address was probed although the private one carries traffic")
        note = fleet.network_note()
        for needle in ("every server reaches every other on the private address",
                       "On these pairs the packets get through but port 22 did not answer:",
                       "open the database port to the peer's",
                       "prove the connection with a client"):
            check(failures, needle in note, f"{label}: the note is missing {needle!r}: {note}")
        check(failures, "not usable as it stands" not in note and "public address" not in note,
              f"{label}: a working private network was sent to the public one: {note}")
        check(failures, f"private: {PRIMARY}" in fleet.brief(),
              f"{label}: the brief does not name the private address:\n{fleet.brief()}")
        shown = network_screen(fleet)
        check(failures, "every pair reachable" in shown
              and f"primary -> replica private {REPLICA}:22" in shown,
              f"{label}: the operator was not shown which pairs are unproven: {shown!r}")

    # The wording, which is the whole of what the model has to act on: a refused
    # connection and a silent port are different faults with the same fix.
    _, _, quiet = droplet_pair(filtered=(PRIMARY, REPLICA))
    check(failures, quiet.targets[0].facts.values.get("peers", "")
          == f"replica private {REPLICA}:22 no answer on the port, but it replies to ping "
             "(the packets get through)",
          f"a filtered port is not described as one: "
          f"{quiet.targets[0].facts.values.get('peers', '')!r}")
    _, _, refused = droplet_pair(ssh_on_public=True)
    check(failures, refused.targets[0].facts.values.get("peers", "")
          == f"replica private {REPLICA}:22 refused (nothing listening there, but the packets "
             "get through)",
          f"a refused connection is not described as one: "
          f"{refused.targets[0].facts.values.get('peers', '')!r}")


def check_network_states(failures: list[str]) -> None:
    """The four plain shapes a pair's network comes in, and what each is told to do.

    One private address each and nothing in the way, so this is the mapping from
    finding to instruction; which address gets probed and how a failure is read are
    check_private_candidates and check_partial_paths.

    They are told apart because the answers differ. A private mesh is what a
    cluster should be built over. Private addresses that do not route, and no
    private addresses at all, are the same answer - the public address is the only
    path, so use it and keep the scoping exact - but they are different findings,
    and an operator reading the run needs to know which one it was. A pair that
    answers on nothing cannot be configured into working and is aborted.
    """
    # 1. A private network as a VPC-joined pair has it.
    _, _, mesh = droplet_pair()
    check(failures, mesh.private_mesh and not mesh.broken_paths,
          f"a working private network was not read as one: {[p.describe() for p in mesh.paths]}")
    note = mesh.network_note()
    for needle in ("every server reaches every other on the private address",
                   "bind to them, point replicas at them",
                   "The address in the label is only how the harness connects."):
        check(failures, needle in note, f"the private-mesh note is missing {needle!r}: {note}")
    brief = mesh.brief()
    check(failures, f"private: {PRIMARY}" in brief and f"private: {REPLICA}" in brief,
          f"the brief does not label which address is the private one:\n{brief}")
    shown = network_screen(mesh)
    check(failures, f"private network: primary {PRIMARY}, replica {REPLICA}" in shown
          and "every pair reachable" in shown,
          f"the operator was not shown the private addresses: {shown!r}")

    # 2. Two droplets in different VPCs: each has a private address, neither can
    # reach the other's, and the public ones are fine. The distinction the probe
    # exists for - without the second address tried, this looks like case 4.
    _, _, vpcs = droplet_pair(blocked=(PRIMARY, REPLICA))
    check(failures, not vpcs.private_mesh and len(vpcs.broken_paths) == 2,
          f"private addresses that do not route were accepted: {[p.describe() for p in vpcs.paths]}")
    check(failures, all(path.any_path for path in vpcs.paths),
          "the public path was not found although it works")
    peers = {target.name: target.facts.values.get("peers", "") for target in vpcs}
    check(failures, peers["primary"] == f"replica private {REPLICA}:22 unreachable, "
                                       "public 203.0.113.11:22 reachable",
          f"the primary's peer line does not say which address failed: {peers['primary']!r}")
    note = vpcs.network_note()
    for needle in ("PRIVATE NETWORK: not usable as it stands.",
                   f"primary -> replica private {REPLICA}:22 unreachable",
                   "Use the public address in each label",
                   "never a range, never %, never",
                   "crosses the public"):
        check(failures, needle in note, f"the public-fallback note is missing {needle!r}: {note}")
    check(failures, "abort" not in note,
          f"a pair that can talk over the public network was told to abort: {note}")
    shown = network_screen(vpcs)
    check(failures, "only have their public addresses to talk on" in shown
          and "cannot reach each other at all" not in shown,
          f"the operator was not told the traffic goes over the public network: {shown!r}")

    # 3. No private interface at all, which is a different finding and the same
    # instruction: nothing to enable, so the public address is all there is.
    _, _, public_only = droplet_pair(private=False)
    check(failures, not public_only.private_mesh and len(public_only.broken_paths) == 2,
          "a pair with no private addresses was read as a private mesh")
    check(failures, all(not target.private for target in public_only),
          f"a public address was labelled private: {[t.private for t in public_only]}")
    check(failures, "private: none reported" in public_only.brief(),
          f"the brief does not say the servers reported no private address:\n{public_only.brief()}")
    note = public_only.network_note()
    check(failures, "reports no private address" in note
          and "Use the public address in each label" in note,
          f"the no-private-address note is wrong: {note}")
    check(failures, "unreachable" not in note,
          f"an address that was never offered was reported unreachable: {note}")

    # 4. Nothing routes: no configuration makes this a cluster, so it is not one
    # to try. The only case the model is told to stop.
    _, _, apart = droplet_pair(connected=False)
    unreachable = [target.facts.values.get("peers", "") for target in apart]
    check(failures, all("unreachable" in line for line in unreachable),
          f"a disconnected network still reported reachable peers: {unreachable}")
    check(failures, not any(path.any_path for path in apart.paths),
          "a path was found across a network where nothing answers")
    note = apart.network_note()
    check(failures, "cannot reach each other at all, on any address" in note
          and "report it and abort" in note,
          f"a pair with no path between them was not told to stop: {note}")
    check(failures, "Use the public address" not in note,
          f"a public address that does not answer was offered as the fallback: {note}")
    shown = network_screen(apart)
    check(failures, "cannot reach each other at all" in shown,
          f"the operator was not warned the servers are isolated: {shown!r}")

    # What the model actually reads: the note reaches it through the brief, and
    # rule 12 covers both outcomes, since the same prompt text serves all four.
    for fleet, name, needle in ((mesh, "network-mesh", "on the private address"),
                                (apart, "network-apart", "not usable as it stands")):
        rules = system_prompt(fleet, f"prompt-{name}")
        check(failures, needle in rules,
              f"the {name} network note never reached the model's prompt")
        for line in ("12. The PRIVATE NETWORK note above says what the servers can reach",
                     "bind the database to it, point replicas at it, and never at",
                     "the public address is all there is - use it, and keep the scoping"):
            check(failures, line in rules, f"rule 12 is missing {line!r}")


# ------------------------------------------------------------- the replication run

# The order matters, and the simulator enforces it: the replica cannot reach a
# source bound to loopback, a source that was reconfigured but not restarted is
# still bound to loopback, a login with no REPLICATION SLAVE grant connects and
# stops, and two servers sharing a server id stop as well.
REPLICATION = [
    "THOUGHT: refresh the package lists on the source first\n"
    "HOST: primary\nACTION: run\nCOMMAND: apt-get update",
    "HOST: replica\nACTION: run\nCOMMAND: apt-get update",
    # Forgets to say where, and is sent back for it.
    "THOUGHT: install the server\nACTION: run\nCOMMAND: apt-get install -y mysql-server",
    # Tries to do both at once, which is not a thing.
    "HOST: both\nACTION: run\nCOMMAND: apt-get install -y mysql-server",
    "HOST: primary\nACTION: run\nCOMMAND: apt-get install -y mysql-server",
    "HOST: replica\nACTION: run\nCOMMAND: apt-get install -y mysql-server",
    # The source: an id of its own, the binary log, and bound to the private
    # address so the replica can reach it.
    "THOUGHT: give the source an id and let it listen on the private network\n"
    "HOST: primary\nACTION: write_file\nPATH: /etc/mysql/mysql.conf.d/zz-replication.cnf\n"
    f"MODE: 0644\nCONTENT_BEGIN\n[mysqld]\nserver-id = 1\nbind-address = {PRIMARY}\n"
    "log_bin = /var/log/mysql/mysql-bin.log\nCONTENT_END",
    "THOUGHT: the config is only read at startup\n"
    "HOST: primary\nACTION: run\nCOMMAND: systemctl restart mysql",
    "HOST: replica\nACTION: write_file\nPATH: /etc/mysql/mysql.conf.d/zz-replication.cnf\n"
    "MODE: 0644\nCONTENT_BEGIN\n[mysqld]\nserver-id = 2\nread_only = ON\nCONTENT_END",
    "HOST: replica\nACTION: run\nCOMMAND: systemctl restart mysql",
    # The replication account, scoped to the replica's private address alone.
    f"HOST: primary\nACTION: run\nCOMMAND: mysql -e \"CREATE USER 'repl'@'{REPLICA}' "
    "IDENTIFIED BY '{{DBA_SECRET:mysql_repl}}'; GRANT REPLICATION SLAVE ON *.* TO "
    f"'repl'@'{REPLICA}'\"",
    # Same placeholder on the other side: the harness makes the two match.
    "HOST: replica\nACTION: run\nCOMMAND: mysql -e \"CHANGE REPLICATION SOURCE TO "
    f"SOURCE_HOST='{PRIMARY}', SOURCE_USER='repl', "
    "SOURCE_PASSWORD='{{DBA_SECRET:mysql_repl}}', SOURCE_AUTO_POSITION=1\"",
    "HOST: replica\nACTION: run\nCOMMAND: mysql -e 'START REPLICA'",
    "THOUGHT: write something on the source and look for it on the replica\n"
    "HOST: primary\nACTION: run\nCOMMAND: mysql -e \"CREATE DATABASE shop\"",
    "HOST: replica\nACTION: run\nCOMMAND: mysql -e \"SHOW DATABASES LIKE 'shop'\"",
    "ACTION: done\n"
    f"VERIFY: {IS_ACTIVE}\n"
    f"VERIFY: [primary] {SHOW_REPLICAS}\n"
    f"VERIFY: [replica] {REPLICA_STATUS}\n"
    f"SUMMARY: MySQL is installed on both servers. primary is the source, bound to {PRIMARY} "
    f"with server-id 1 and the binary log on; replica reads from it as 'repl'@'{REPLICA}' with "
    "server-id 2, and the shop database created on the source arrived on the replica. The "
    "replication password is {{DBA_SECRET:mysql_repl}}.",
]

TASK = "Set up MySQL replication from primary to replica over the private network."


def check_run(failures: list[str]) -> None:
    primary, replica, fleet = droplet_pair()
    store = SecretStore()
    agent, record, client, events = build(
        primary, store, REPLICATION, fleet=fleet, task=TASK,
        directory=RUNS / "replication",
    )
    outcome = agent.run()
    report_text = record.write_report().read_text(encoding="utf-8")
    logged = [json.loads(line) for line in
              (record.directory / "transcript.jsonl").read_text(encoding="utf-8").splitlines()]

    print(f"status: {outcome.status}  steps: {outcome.steps} proposed, {outcome.executed} executed")
    for target in fleet:
        print(f"\n{target.name}:")
        print("  " + target.runner.state().replace("\n", "\n  "))

    check(failures, outcome.status == "done", f"status is {outcome.status}: {outcome.summary}")

    # Both servers were actually built, and differently.
    check(failures, "mysql-server" in primary.packages and "mysql-server" in replica.packages,
          "mysql was not installed on both servers")
    check(failures, primary.mysql.server_id == 1 and replica.mysql.server_id == 2,
          f"the server ids are {primary.mysql.server_id} and {replica.mysql.server_id}")
    check(failures, primary.listens(3306) and not replica.listens(3306),
          "the source should listen on the private network and the replica need not")

    # The link itself, judged by the simulator rather than by the model's word.
    check(failures, replica.mysql.source == PRIMARY,
          f"the replica points at {replica.mysql.source!r}, want the private {PRIMARY}")
    check(failures, replica.mysql.replicating, "the replica's threads were never started")
    io_running, error = replica._replica_health(replica.mysql)
    check(failures, io_running == "Yes", f"the replica is not replicating: {io_running} {error}")
    check(failures, f"repl@{REPLICA}" in primary.mysql.repl_grants,
          f"the replication grant is missing: {sorted(primary.mysql.repl_grants)}")
    check(failures, "shop" in replica.mysql.databases,
          "data written on the source never reached the replica")

    # One secret, the same value on both sides, and never in the clear afterwards.
    password = store.resolve("{{DBA_SECRET:mysql_repl}}")
    check(failures, any(password in c for c in primary.commands)
          and any(password in c for c in replica.commands),
          "the shared credential did not reach both servers")
    check(failures, password not in report_text, "the report leaked the replication password")
    check(failures, all(password not in json.dumps(event) for event in logged),
          "the transcript leaked the replication password")

    # Nothing was broadcast: each command ran on the one server it named.
    check(failures, not any("CREATE USER" in c for c in replica.commands),
          f"a step for the primary ran on the replica: {replica.commands[-3:]}")
    check(failures, not any("START REPLICA" in c for c in primary.commands),
          f"a step for the replica ran on the primary: {primary.commands[-3:]}")

    # The two refusals cost a round trip each and never touched a server.
    errors = [event for event in logged if event["kind"] == "protocol_error"]
    check(failures, len(errors) == 2, f"expected two refused steps, got {len(errors)}")
    check(failures, any("no HOST" in event["error"] for event in errors),
          f"the unnamed step was not refused for want of a HOST line: {errors}")
    check(failures, any("not a server in this run" in event["error"] for event in errors),
          f"HOST: both was not refused: {errors}")
    check(failures, sum(1 for c in primary.commands + replica.commands
                        if c.startswith("apt-get install")) == 2,
          "the refused install steps still reached a server")

    # Every executed step says which server it ran on, in the record and on screen.
    hosts = {step.host for step in record.steps if step.executed}
    check(failures, hosts == {"primary", "replica"}, f"the steps recorded hosts {hosts}")
    check(failures, all(step.host for step in record.steps if step.executed),
          "an executed step was recorded with no server")
    ran = [message for kind, message in events if kind == "run"]
    check(failures, all(m.startswith(("[primary] ", "[replica] ")) for m in ran),
          f"a step was announced without its server: {[m for m in ran if not m.startswith('[')]}")

    # Verification: the unscoped check ran on both, each scoped one on its own server.
    verified = {(check_.host, check_.command) for check_ in record.verifications}
    check(failures, ("primary", IS_ACTIVE) in verified and ("replica", IS_ACTIVE) in verified,
          f"the unscoped check did not run on both servers: {sorted(verified)}")
    check(failures, ("replica", REPLICA_STATUS) in verified,
          f"the replica's own check did not run there: {sorted(verified)}")
    check(failures, ("primary", REPLICA_STATUS) not in verified,
          "a check scoped to the replica was run on the primary as well")
    check(failures, ("primary", SHOW_REPLICAS) in verified
          and ("replica", SHOW_REPLICAS) not in verified,
          f"the primary's own check was not scoped to it: {sorted(verified)}")
    check(failures, all(check_.exit_code == 0 for check_ in record.verifications),
          f"a verification failed: {[c for c in record.verifications if c.exit_code]}")
    # Both ends, which is what proves a pair: the replica says its io thread is up
    # and the primary lists the replica as connected.
    outputs = {(check_.host, check_.command): check_.output for check_ in record.verifications}
    check(failures, "Replica_IO_Running: Yes" in outputs.get(("replica", REPLICA_STATUS), ""),
          f"the replica's status does not show it running: "
          f"{outputs.get(('replica', REPLICA_STATUS), '')!r}")
    check(failures, REPLICA in outputs.get(("primary", SHOW_REPLICAS), ""),
          f"the primary does not list the replica: {outputs.get(('primary', SHOW_REPLICAS))!r}")

    # The report and the transcript are readable afterwards as a two-server run.
    for needle in ("- **Hosts:** 2", "- **primary:** root@203.0.113.10",
                   "## Servers as found", "**replica** (root@203.0.113.11)",
                   "on primary", "on replica", "`[replica] mysql -e"):
        check(failures, needle in report_text, f"the report is missing {needle!r}")
    started = next(event for event in logged if event["kind"] == "run_started")
    check(failures, [host["name"] for host in started["hosts"]] == ["primary", "replica"],
          f"the transcript's server list is {started.get('hosts')}")
    steps = [event for event in logged if event["kind"] == "step" and event["executed"]]
    check(failures, all(event["host"] for event in steps),
          "a step was logged without the server it ran on")
    verifications = [event for event in logged if event["kind"] == "verification"]
    check(failures, {event["host"] for event in verifications} == {"primary", "replica"},
          "the logged verifications do not name both servers")

    # Nothing the simulator could not read: an unmodelled command would mean the
    # run only looked right.
    unmodelled = primary.unhandled + replica.unhandled
    check(failures, not unmodelled, f"the simulator did not model: {unmodelled}")
    check(failures, client.calls == len(REPLICATION),
          f"the model was called {client.calls} times for {len(REPLICATION)} replies")


def check_broken_run(failures: list[str]) -> None:
    """The same task on a bare pair whose private network was never enabled.

    Unnamed servers, so the steps are routed by the labels the harness gave out and
    the model picks the roles: node1 becomes the source and node2 the replica,
    which is the shape of every run started from a plain list of addresses.

    Every step succeeds - CHANGE REPLICATION SOURCE TO and START REPLICA both exit
    0 - and the model reports success. The check that catches it is the one rule 14
    asks for: a connection made from the replica to the source, which is the thing
    that does not work. Three done steps, because the first two are handed back.
    """
    source, standby, fleet = droplet_pair(connected=False, named=False)
    check(failures, fleet.names == ["node1", "node2"],
          f"the run was not driven by the harness's labels: {fleet.names}")
    store = SecretStore()
    from_standby = f"mysql -h {PRIMARY} -u repl -p'{{{{DBA_SECRET:repl}}}}' -e \"SELECT 1\""
    done = ("ACTION: done\n"
            f"VERIFY: [node2] {from_standby}\n"
            "SUMMARY: node1 is the source and node2 the replica; replication is configured "
            "and started")
    agent, record, _, _ = build(
        source, store,
        [
            "THOUGHT: node1 will be the source and node2 the replica\n"
            "HOST: node1\nACTION: run\nCOMMAND: apt-get update",
            "HOST: node2\nACTION: run\nCOMMAND: apt-get update",
            "HOST: node1\nACTION: run\nCOMMAND: apt-get install -y mysql-server",
            "HOST: node2\nACTION: run\nCOMMAND: apt-get install -y mysql-server",
            "HOST: node2\nACTION: run\nCOMMAND: mysql -e \"CHANGE REPLICATION SOURCE TO "
            f"SOURCE_HOST='{PRIMARY}', SOURCE_USER='repl', "
            "SOURCE_PASSWORD='{{DBA_SECRET:repl}}'\"",
            "HOST: node2\nACTION: run\nCOMMAND: mysql -e 'START REPLICA'",
            # Would have arrived on the replica if the link were up, as it does in
            # the run above; here it stays on the source.
            "HOST: node1\nACTION: run\nCOMMAND: mysql -e \"CREATE DATABASE shop\"",
            done, done, done,
        ],
        fleet=fleet, task=TASK, directory=RUNS / "unreachable",
    )
    outcome = agent.run()
    report_text = record.write_report().read_text(encoding="utf-8")

    check(failures, outcome.status == "unverified",
          f"a pair that cannot talk reported {outcome.status}, want unverified")
    check(failures, standby._replica_health(standby.mysql)[0] != "Yes",
          "the replica reported healthy across a disconnected network")
    check(failures, standby.mysql.replicating and standby.mysql.source == PRIMARY,
          "the replica was never configured, so this proves nothing about the network")
    check(failures, "shop" not in standby.mysql.databases,
          "a disconnected replica received data anyway")
    # The labels routed the steps: nothing meant for one server ran on the other.
    check(failures, not any("START REPLICA" in command for command in source.commands),
          f"a step for node2 ran on node1: {source.commands[-3:]}")
    check(failures, {step.host for step in record.steps if step.executed} == {"node1", "node2"},
          f"the steps recorded hosts {[s.host for s in record.steps]}")
    # Handed back naming the server, and twice, before the harness gives up on it.
    handed_back = [message["content"] for message in agent.messages
                   if "THESE FAILED" in message["content"]]
    check(failures, len(handed_back) == 2,
          f"the failing done step should have been handed back twice: {len(handed_back)}")
    check(failures, handed_back and "[node2]" in handed_back[0],
          f"the failing check was not handed back naming its server: {handed_back[:1]}")
    check(failures, store.resolve("{{DBA_SECRET:repl}}") not in report_text,
          "the failed run's report leaked the replication password")
    check(failures, "- **node1:** root@203.0.113.10" in report_text,
          "the report does not say which address the harness labelled node1")


def main() -> int:
    shutil.rmtree(RUNS, ignore_errors=True)
    failures: list[str] = []
    check_targets(failures)
    check_fleet(failures)
    check_labels(failures)
    check_spec(failures)
    check_protocol(failures)
    check_peers(failures)
    check_private_candidates(failures)
    check_partial_paths(failures)
    check_network_states(failures)
    check_run(failures)
    check_broken_run(failures)

    print()
    if failures:
        for failure in failures:
            print(f"  FAIL {failure}")
        print(f"\n{len(failures)} check(s) failed")
        return 1
    print("all checks passed")
    print(f"\nreports: {RUNS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
