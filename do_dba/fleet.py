"""The servers a run has, and how a step says which one it is for.

One server is the common case and stays the simple one: the step needs no HOST
line and the prompt says "this server". More than one is what replication,
clustering and load-balanced pairs need, and then every step has to name its
target. The harness refuses to guess which server a step meant, because that is
the one mistake the guard cannot catch - `DROP DATABASE app` is allowed on the
node being rebuilt and catastrophic on the one serving traffic, and the two
commands are identical.

A fleet also carries what the servers know about each other: their private
addresses and whether they can actually open a connection to one another. A
cluster that cannot be built because private networking was never enabled is the
commonest way these tasks fail, and it is much cheaper to find out here than
after the model has installed and configured everything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .facts import Facts, gather

# Waiting for a peer's SSH port. A refused connection comes back at once; this
# only bounds the case where the packets are dropped by a firewall, which is the
# answer the operator most needs and the one that would otherwise take the
# kernel's full SYN timeout to arrive at.
PEER_TIMEOUT = 4
PEER_PROBE = "timeout {timeout} bash -c '</dev/tcp/{host}/{port}'"
# Only when the port did not answer, and both questions in one round trip. A ping
# reply proves the packets cross, which is all a database needs; the route says
# whether this machine has any way to send them, which is a different fault with a
# different fix. Without these two, a firewalled port and a network that was never
# built are the same silent timeout - and telling a cluster task to fall back to
# the public internet because sshd was not listening on eth1 is a bad trade.
PEER_DIAGNOSE = (
    "ip route get {host} 2>&1 | head -1; "
    "timeout {timeout} ping -n -c 1 -W 2 {host} >/dev/null 2>&1 "
    "&& echo dba-ping-ok || echo dba-ping-none"
)
# Port 22, not the database port: at survey time no database is listening yet, and
# 22 is the one port known to be open, since the harness is talking to it. It
# answers the question that actually blocks a cluster - whether there is any
# private path between these machines at all.
PEER_PORT = 22
# How many of a peer's private addresses to try. More than one is the normal case
# on a cloud host and the reason this list exists at all, but each dead one costs a
# timeout, and a machine with three private addresses that answer nothing has told
# us what we needed to know.
PEER_CANDIDATES = 3
# Interfaces whose addresses are not a way to reach this machine: a container or VM
# bridge has the same address on every host that runs one, so probing a peer's
# 172.17.0.1 is probing your own docker0 and proves nothing either way.
_VIRTUAL = re.compile(
    r"^(lo|docker|br-|bridge|virbr|veth|tun|tap|wg|zt|cni|flannel|cali|kube|tailscale|utun)"
)


@dataclass
class Reach:
    """What one address answered when a server tried to open a port on it.

    More than reachable/not, because the difference decides what the model is told.
    A refused connection means the packets made the round trip and only sshd was
    not there; a port that times out while the address answers ping means a
    firewall in front of an network that works. Both are private networks a cluster
    can be built on, and both used to be reported as "unreachable" - which sent the
    run onto the public internet for no reason.
    """

    state: str = "none"  # none | open | refused | filtered | no-route | dropped | unknown
    detail: str = ""  # what the shell said, kept for the transcript

    @property
    def open(self) -> bool:
        """Whether the port answered - the only state that needs no explaining."""
        return self.state == "open"

    @property
    def probed(self) -> bool:
        return self.state != "none"

    @property
    def works(self) -> bool:
        """Whether the packets get through, whatever is or is not listening."""
        return self.state in {"open", "refused", "filtered"}

    @property
    def word(self) -> str:
        return _WORDS.get(self.state, self.state)


_WORDS = {
    "none": "",
    "open": "reachable",
    "refused": "refused (nothing listening there, but the packets get through)",
    "filtered": "no answer on the port, but it replies to ping (the packets get through)",
    "no-route": "unreachable (this server has no route to it)",
    "dropped": "unreachable",
    "unknown": "could not be probed",
}


@dataclass
class PeerPath:
    """One server's view of one peer: the addresses tried, and what answered.

    Both are worth knowing and they are not the same finding. A private address
    that does not answer means the private network was never enabled, or the two
    machines are in different ones - the task can still be described but not built
    the way it should be. No private address at all means the servers only have the
    public internet between them, which is a decision for the operator rather than
    something to configure around quietly.
    """

    source: str
    peer: str
    private: str = ""  # the peer's private address this path is about, empty if it has none
    public: str = ""  # where the harness itself reaches the peer
    private_reach: Reach = field(default_factory=Reach)
    public_reach: Reach = field(default_factory=Reach)  # only probed when nothing private got through
    tried: list[str] = field(default_factory=list)  # every private address probed, in order

    @property
    def private_ok(self) -> bool:
        """Port 22 answered on the private address: nothing left to explain."""
        return self.private_reach.open

    @property
    def private_works(self) -> bool:
        """The private network carries traffic between these two, port aside."""
        return self.private_reach.works

    @property
    def any_path(self) -> bool:
        return self.private_works or self.public_reach.works

    def describe(self) -> str:
        """The line the model and the operator read, in the `peers` fact."""
        if self.private_ok:
            return f"{self.peer} private {self.private}:{PEER_PORT} reachable"
        if not self.private:
            head = f"{self.peer} reports no private address"
        else:
            head = f"{self.peer} private {self.private}:{PEER_PORT} {self.private_reach.word}"
            # Named, not summarised: an address that was tried and failed is the
            # first thing an operator checks against what they think is configured.
            others = [address for address in self.tried if address != self.private]
            if others:
                head += f" (also tried {', '.join(others)})"
        if not self.public_reach.probed:
            return head
        return f"{head}, public {self.public}:{PEER_PORT} {self.public_reach.word}"


# A name is for the model to type, so it stays short and shell-safe. Starting with
# a letter keeps it apart from an address, which is what the same field may hold.
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{0,31}$")
# Punctuation a model wraps a name in: `HOST: [primary]`, `HOST: "primary"`.
_WRAPPERS = "[](){}<>\"'`"


@dataclass
class Address:
    """One address a server reports for itself, and the interface carrying it."""

    interface: str
    address: str  # without the mask: a peer connects to an address, not to a subnet
    mask: str = ""

    @property
    def private(self) -> bool:
        return is_private(self.address)

    @property
    def virtual(self) -> bool:
        return bool(_VIRTUAL.match(self.interface))

    @property
    def text(self) -> str:
        return f"{self.address}/{self.mask}" if self.mask else self.address


def parse_addresses(reported: str) -> list[Address]:
    """Read the addresses fact: `eth0=203.0.113.10/20 eth1=10.116.0.2/16`.

    The interface is what the probe adds and what `hostname -I` cannot, so a token
    without one is still an address - just one whose interface is unknown, which
    is worse but not useless.
    """
    found = []
    for token in reported.split():
        interface, _, rest = token.rpartition("=")
        address, _, mask = rest.partition("/")
        if address:
            found.append(Address(interface=interface, address=address, mask=mask))
    return found


@dataclass
class Target:
    """One server: how to reach it, what it is called, and what it looks like."""

    name: str
    host: str
    user: str = "root"
    port: int = 22
    runner: object | None = None
    facts: Facts = field(default_factory=Facts)
    # Whether the name came from the operator (`--host primary=...`) or from the
    # harness. An operator's name says what the server is for and the model should
    # respect it; a harness label says nothing, and then the roles are the model's
    # to work out.
    named: bool = False
    # The private address a peer actually got through to, filled in by the survey.
    # A machine can report several and only one of them is the one its peers share,
    # so which it is is a matter of evidence rather than of picking the first.
    private_confirmed: str = ""

    @property
    def label(self) -> str:
        return f"{self.user}@{self.host}" + ("" if self.port == 22 else f":{self.port}")

    @property
    def reported(self) -> list[Address]:
        """Every address the server reported, in the order the kernel listed them."""
        return parse_addresses(self.facts.values.get("addresses", ""))

    @property
    def addresses(self) -> list[str]:
        """The addresses this server reports for itself, private ones first.

        A replica has to be pointed at an address the primary answers on, and on a
        cloud host that is never the public one: the private interface is what the
        peers share and what the database should be bound to.
        """
        found = self.reported
        confirmed = [a.text for a in found if a.address == self.private_confirmed]
        # A bridge address is private and is not a way in, so it sorts with the
        # rest rather than ahead of the address a peer is meant to use.
        private = [a.text for a in found
                   if a.private and not a.virtual and a.text not in confirmed]
        listed = set(confirmed) | set(private)
        return confirmed + private + [a.text for a in found if a.text not in listed]

    @property
    def private_candidates(self) -> list[str]:
        """The addresses a peer should try for this server, best first.

        Several private addresses is the normal case on a cloud host, and only one
        of them is the one the peers share: DigitalOcean puts an internal anchor
        address on the public interface, docker adds 172.17.0.1, and either looks
        exactly like a VPC address. So bridges are dropped - their address is the
        same on every machine that has one - and anything sharing an interface with
        the address the harness itself connected on goes last, because that is where
        a provider's internal address lives and the VPC is on its own interface.

        More than one is returned because the only way to know which of them a peer
        can reach is to try it, and picking the first and reporting "no private
        network" when it failed is how a working VPC gets missed.
        """
        found = [a for a in self.reported if a.private and not a.virtual]
        beside_public = {a.interface for a in self.reported if a.address == self.host}
        preferred = [a.address for a in found if a.interface not in beside_public]
        beside = [a.address for a in found if a.interface in beside_public]
        ordered = list(dict.fromkeys(preferred + beside))
        if self.private_confirmed in ordered:  # a re-probe starts from what worked
            ordered.remove(self.private_confirmed)
            ordered.insert(0, self.private_confirmed)
        return ordered[:PEER_CANDIDATES]

    @property
    def private(self) -> str:
        """The address the other servers should use for this one, if it has one.

        --host almost always holds a public address, because that is what an
        operator has to hand, and the private interface is both the one that
        matters for replication and the one nothing outside the server can
        discover. So it is read off the server itself, and once the survey has
        found which of several a peer can actually reach, that is the answer.
        Empty when the server reports none, which is a different problem and gets a
        different answer.
        """
        if self.private_confirmed:
            return self.private_confirmed
        candidates = self.private_candidates
        return candidates[0] if candidates else ""


def is_private(address: str) -> bool:
    """Whether an IPv4 address is on a private range (RFC 1918 or the DO 10.x)."""
    parts = address.split("/")[0].split(".")
    if len(parts) != 4 or not all(part.isdigit() for part in parts):
        return False
    first, second = int(parts[0]), int(parts[1])
    return first == 10 or (first == 172 and 16 <= second <= 31) or (first == 192 and second == 168)


def parse_target(spec: str, user: str = "root", port: int = 22) -> Target:
    """Read one --host value: `[name=][user@]host[:port]`.

    The name is optional: pass a bare list of servers and the harness labels them
    node1, node2, ... and leaves it to the model to work out which one takes which
    role. Give a name when the roles are yours to decide rather than the model's -
    `--host primary=10.0.0.2 --host replica=10.0.0.3` - and from then on the model,
    the transcript and the report all talk about the primary rather than about
    10.0.0.2, and a step aimed at the wrong one is obvious to read.
    """
    text = spec.strip()
    if not text:
        raise ValueError("a --host value cannot be empty")

    name = ""
    if "=" in text:
        name, _, text = text.partition("=")
        name, text = name.strip(), text.strip()
        if not _NAME.match(name):
            raise ValueError(
                f"{name!r} is not a usable name: start with a letter and use letters, "
                "digits, dot, dash or underscore (up to 32 characters)"
            )
        if not text:
            raise ValueError(f"--host {spec!r} names a server but does not say where it is")

    if "@" in text:
        # rpartition, because a password is never in here but an IPv6 literal has
        # colons and an @ in the host part would be a malformed address anyway.
        user_part, _, text = text.rpartition("@")
        if not user_part.strip() or not text.strip():
            raise ValueError(f"--host {spec!r} is not [name=][user@]host[:port]")
        user = user_part.strip()

    host, port = _split_port(text, port)
    if not host:
        raise ValueError(f"--host {spec!r} has no hostname")
    return Target(name=name or host, host=host, user=user, port=port, named=bool(name))


def _split_port(text: str, default: int) -> tuple[str, int]:
    text = text.strip()
    if text.startswith("["):  # [2001:db8::1]:2222 - a bracketed IPv6 literal
        closing = text.find("]")
        if closing < 0:
            raise ValueError(f"{text!r} opens a bracket it never closes")
        host, rest = text[1:closing], text[closing + 1:]
        if rest.startswith(":"):
            return host, _port(rest[1:], text)
        return host, default
    # More than one colon and no brackets: an IPv6 literal, where a trailing
    # `:2222` cannot be told from part of the address. Take it whole.
    if text.count(":") > 1:
        return text, default
    if ":" in text:
        host, _, port_text = text.partition(":")
        return host, _port(port_text, text)
    return text, default


def _port(text: str, spec: str) -> int:
    if not text.isdigit() or not 1 <= int(text) <= 65535:
        raise ValueError(f"{spec!r} does not end in a port number")
    return int(text)


class Fleet:
    """Every server in the run, and the lookup from what a reply says to one of them."""

    def __init__(self, targets: list[Target]):
        if not targets:
            raise ValueError("a run needs at least one host")
        self.targets = list(targets)
        # Filled by the survey, one entry per ordered pair. Empty until then, and
        # empty for a single server, where there is no peer to have a path to.
        self.paths: list[PeerPath] = []
        self._label_unnamed()
        names: set[str] = set()
        for target in self.targets:
            if target.name.lower() in names:
                raise ValueError(
                    f"{target.name!r} is the name of two servers; give each one a distinct "
                    "name, e.g. --host primary=... --host replica=..."
                )
            names.add(target.name.lower())
        endpoints = [(t.user.lower(), t.host.lower(), t.port) for t in self.targets]
        for target, endpoint in zip(self.targets, endpoints):
            if endpoints.count(endpoint) > 1:
                raise ValueError(f"{target.label} is listed twice")
        # A name that is another server's address is the one ambiguity worth
        # refusing outright: `HOST: 10.0.0.3` would then be a name pointing at one
        # server and an address pointing at another, and no reading of it is safe.
        for target in self.targets:
            clash = [t for t in self.targets
                     if t is not target and t.host.lower() == target.name.lower()]
            if clash:
                raise ValueError(
                    f"{target.name!r} names one server and is the address of another "
                    f"({clash[0].name}); rename one of them"
                )

    def _label_unnamed(self) -> None:
        """Label the servers the operator did not name: node1, node2, ...

        Two addresses a digit apart are the worst thing to route steps by, and a
        list of servers with no names is the normal way to ask for a cluster: the
        operator has the machines and the model is the one deciding what each is
        for. So the harness supplies a label to talk about them with, and the
        addresses stay next to it everywhere it appears.

        Only for more than one server. Alone there is nothing to tell apart, the
        name never reaches the model, and changing it would only alter what the
        transcript of a single-server run has always said.
        """
        if len(self.targets) < 2:
            return
        taken = {t.name.lower() for t in self.targets} | {t.host.lower() for t in self.targets}
        number = 1
        for target in self.targets:
            if target.named:
                continue
            while f"node{number}" in taken:  # an operator may have used node2 themselves
                number += 1
            target.name = f"node{number}"
            taken.add(f"node{number}")
            number += 1

    @classmethod
    def of(cls, runner, name: str = "", facts: Facts | None = None) -> "Fleet":
        """A fleet of one around a runner that already exists.

        The single-server path and the many-server path are the same code from
        here on, so there is only ever one loop to reason about.
        """
        host = name or getattr(runner, "host", "") or "server"
        return cls([Target(
            name=host,
            host=getattr(runner, "host", host),
            user=getattr(runner, "user", "root"),
            port=getattr(runner, "port", 22),
            runner=runner,
            facts=facts or Facts(),
            named=bool(name),
        )])

    # ------------------------------------------------------------------ lookup

    def __len__(self) -> int:
        return len(self.targets)

    def __iter__(self):
        return iter(self.targets)

    @property
    def many(self) -> bool:
        return len(self.targets) > 1

    @property
    def only(self) -> Target:
        """The single server, for a run that has one. Ask first with .many."""
        return self.targets[0]

    @property
    def names(self) -> list[str]:
        return [target.name for target in self.targets]

    @property
    def assigned(self) -> bool:
        """Whether the operator named the servers, and so said what they are for.

        The difference the model is told about: named servers come with their roles
        decided, and a bare list of addresses does not.
        """
        return any(target.named for target in self.targets)

    def find(self, spelling: str) -> Target | None:
        """The server a reply's HOST line refers to, or None if it names no server.

        Deliberately forgiving about how it is written and unforgiving about which
        server it is: a name that could mean two of them is not resolved to
        either. Fuzziness here costs a round trip; a wrong guess costs a server.
        """
        text = (spelling or "").strip().strip(_WRAPPERS).strip().rstrip(".,;:").strip()
        if not text:
            return None
        lowered = text.lower()
        exact = [target for target in self.targets
                 if lowered in {target.name.lower(), target.host.lower(), target.label.lower()}]
        if len(exact) == 1:
            return exact[0]
        if exact:
            # Two servers behind one address, on different ports. The address is not
            # an answer to which of them, so only a name will do.
            return None
        matches = [
            target for target in self.targets
            if target.name.lower().startswith(lowered) or target.host.lower().startswith(lowered)
        ]
        if len(matches) == 1:
            return matches[0]
        # `HOST: primary (10.0.0.2)` and `HOST: primary - the source` both happen;
        # the first word is the answer if it names a server on its own.
        first = re.split(r"[\s,(/]+", text)[0]
        return self.find(first) if first and first != text else None

    @property
    def slug(self) -> str:
        """A short name for the run directory."""
        first = self.targets[0].host
        return first if len(self.targets) == 1 else f"{first}-plus{len(self.targets) - 1}"

    @property
    def label(self) -> str:
        if len(self.targets) == 1:
            return self.only.label
        return f"{len(self.targets)} servers: " + ", ".join(
            f"{target.name} ({target.label})" for target in self.targets
        )

    @property
    def without_root(self) -> list[Target]:
        return [target for target in self.targets if not target.facts.has_root_access]

    # -------------------------------------------------------------- connecting

    def connect(self, before=None) -> None:
        """Open every connection, closing the ones already open if any fails."""
        try:
            for target in self.targets:
                if before is not None:
                    before(target)
                target.runner.connect()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        for target in self.targets:
            runner = target.runner
            if runner is not None and hasattr(runner, "close"):
                try:
                    runner.close()
                except Exception:
                    pass  # closing is cleanup; a failure here must not mask the real one

    # ----------------------------------------------------------------- survey

    def survey(self, on_host=None) -> None:
        """Gather facts for every server, then ask each what it can reach.

        The peer check is skipped for a single server, where there is nothing to
        reach and the question is meaningless.
        """
        for target in self.targets:
            if on_host is not None:
                on_host(target)
            target.facts = gather(target.runner)
        if self.many:
            self._probe_peers()

    def _probe_peers(self) -> None:
        """Ask each server what it can reach the others on, private addresses first.

        Every private address the peer reported is tried, not just the first: a
        cloud host has several and only one of them is the one its peers share, so
        the first is a guess and the one that answers is an answer.

        The public address is only tried when nothing private got through, and then
        only to tell the two failures apart: a private network that is not enabled
        still leaves the servers able to see each other, and a pair that answers on
        neither is a firewall or a network nobody has built yet.
        """
        self.paths = []
        for target in self.targets:
            # Its own addresses, so they are never probed as if they were a peer's:
            # docker0 is 172.17.0.1 on every host that runs docker, and "reachable"
            # there would mean this machine reached itself.
            mine = {address.address for address in target.reported}
            seen = []
            for peer in self.targets:
                if peer is target:
                    continue
                path = PeerPath(source=target.name, peer=peer.name, public=peer.host)
                for candidate in peer.private_candidates:
                    if candidate in mine:
                        continue
                    path.tried.append(candidate)
                    reach = self._reach(target, candidate)
                    if reach.works or not path.private:
                        # The one that got through, or else the first one tried:
                        # the rest are listed beside it either way.
                        path.private, path.private_reach = candidate, reach
                    if reach.works:
                        peer.private_confirmed = candidate
                        break
                if not path.private_works and path.public and path.public not in path.tried:
                    path.public_reach = self._reach(target, path.public)
                self.paths.append(path)
                seen.append(path.describe())
            target.facts.values["peers"] = " | ".join(seen)

    @property
    def private_mesh(self) -> bool:
        """Whether every server can get to every other over the private network.

        The question a replication or cluster task turns on: a private path that
        does not exist cannot be configured into existence, and finding that out
        here costs one round trip instead of twenty.

        Traffic getting through is the test, not port 22 answering. sshd not
        listening on the private interface, or a firewall that allows the database
        port and not 22, both leave a private network a cluster can be built on -
        and treating either as a missing network sends the run onto the public
        internet for nothing.
        """
        return bool(self.paths) and all(path.private_works for path in self.paths)

    @property
    def unproven_paths(self) -> list[PeerPath]:
        """Private paths that carry traffic without answering on port 22."""
        return [path for path in self.paths if path.private_works and not path.private_ok]

    @property
    def broken_paths(self) -> list[PeerPath]:
        """The pairs with no private path between them, in the order probed."""
        return [path for path in self.paths if not path.private_works]

    def network_note(self) -> str:
        """What the servers can reach each other on, as the system prompt puts it.

        Said in the brief rather than left to the model to work out from a list of
        addresses: which of two addresses is the private one is obvious to a human
        and a coin toss to a model in a hurry, and binding a database to the wrong
        one of them is how a run ends up serving the internet.
        """
        if not self.paths:
            return ""
        if self.private_mesh:
            lines = [
                "PRIVATE NETWORK: every server reaches every other on the private address",
                "listed above. Use those addresses for everything the servers say to each",
                "other - bind to them, point replicas at them, scope grants and firewall",
                "rules to them. The address in the label is only how the harness connects.",
            ]
            if self.unproven_paths:
                # Said plainly, because the model would otherwise read a bare
                # "reaches" as "nothing to configure" and skip the firewall step
                # that is the whole reason port 22 did not answer.
                lines.append("")
                lines.append("On these pairs the packets get through but port 22 did not answer:")
                lines += [f"  {path.source} -> {path.describe()}" for path in self.unproven_paths]
                lines.append(
                    "That is a firewall or an sshd that is not listening there, not a missing\n"
                    "network. Keep to the private addresses, open the database port to the peer's\n"
                    "private address as part of the work, and prove the connection with a client\n"
                    "before you rely on it."
                )
            return "\n".join(lines)
        lines = ["PRIVATE NETWORK: not usable as it stands."]
        for path in self.broken_paths:
            lines.append(f"  {path.source} -> {path.describe()}")
        if any(not path.any_path for path in self.broken_paths):
            lines.append(
                "Those servers cannot reach each other at all, on any address. No database\n"
                "configuration will change that: report it and abort."
            )
        else:
            lines.append(
                "Use the public address in each label for what the servers say to each other,\n"
                "since it is the only path they have. Scope every grant, pg_hba.conf line and\n"
                "firewall rule to the peer's exact address - never a range, never %, never\n"
                "0.0.0.0 - and say in your summary that this traffic crosses the public\n"
                "network."
            )
        return "\n".join(lines)

    def _reach(self, target: Target, address: str) -> Reach:
        """Whether this server can open the peer's SSH port on that address.

        The exit code and what the shell said are both read, because "it did not
        answer" is three different findings: refused (the packets crossed and
        nothing was listening), no route (this machine cannot even send them), and
        silence (something in between is dropping them). Only the last needs a
        second question asked.
        """
        command = PEER_PROBE.format(timeout=PEER_TIMEOUT, host=address, port=PEER_PORT)
        try:
            result = target.runner.run(command, timeout=PEER_TIMEOUT + 6)
        except Exception as exc:  # a probe must never sink the run
            return Reach("unknown", str(exc)[:160])
        if result.exit_code == 0 and not result.timed_out:
            return Reach("open")
        said = " ".join(f"{result.stderr} {result.stdout}".split())
        lowered = said.lower()
        if "refused" in lowered:
            return Reach("refused", said[:160])
        if "no route" in lowered or "network is unreachable" in lowered:
            return Reach("no-route", said[:160])
        if "not found" in lowered or "permission denied" in lowered:
            # No bash, no /dev/tcp support, or a shell that will not open sockets.
            # Nothing has been learned about the network, and calling that
            # "unreachable" would blame it for the probe's own failure.
            return Reach("unknown", said[:160])
        return self._diagnose(target, address, said)

    def _diagnose(self, target: Target, address: str, said: str) -> Reach:
        """Nothing came back: is it filtered, unroutable, or a dead address?"""
        command = PEER_DIAGNOSE.format(timeout=PEER_TIMEOUT, host=address)
        try:
            result = target.runner.run(command, timeout=2 * PEER_TIMEOUT + 6)
        except Exception as exc:
            return Reach("dropped", f"{said} {exc}".strip()[:160])
        # Both streams: `ip route` writes its failure to stderr, and the 2>&1 in the
        # command only helps if the far end honoured it.
        out = " ".join(f"{result.stdout} {result.stderr}".split())
        lowered = out.lower()
        if "dba-ping-ok" in lowered:
            return Reach("filtered", out[:160])
        if "unreachable" in lowered or "no route" in lowered:
            return Reach("no-route", out[:160])
        return Reach("dropped", out[:160])

    # ----------------------------------------------------------------- prompts

    def brief(self) -> str:
        """The servers, as the system prompt describes them."""
        if not self.many:
            return f"THIS SERVER\n{self.only.facts.summary()}"

        lines = [
            f"THE SERVERS ({len(self.targets)})",
            "Every step must say which one it runs on with a HOST: line. The harness will",
            "not guess: a step with no HOST, or a name that is not below, is refused and",
            "sent back to you.",
            "",
            *([
                "The names below were chosen by the operator and say what each server is",
                "for. Follow them.",
            ] if self.assigned else [
                "The names below are labels the harness assigned. Nothing has been decided",
                "about what each server is for: read the task, work out which server takes",
                "which role, say which in your first step, and keep to it for the whole run.",
                "The servers are equivalent, so any assignment that satisfies the task is a",
                "good one - what matters is that you do not change your mind halfway.",
            ]),
            "",
        ]
        width = max(len(target.name) for target in self.targets)
        for target in self.targets:
            # Named rather than listed: the model is about to choose one address to
            # bind a database to, and "private" is the whole of what it needs to know
            # about which. Everything the server reported is in its facts below.
            private = target.private or "none reported"
            lines.append(f"  {target.name:<{width}}  {target.label:<28}  private: {private}")
        note = self.network_note()
        if note:
            lines += ["", note]
        for target in self.targets:
            lines += ["", f"--- {target.name} ({target.label})", target.facts.summary()]
        return "\n".join(lines)

    def host_lines(self) -> list[tuple[str, str, dict[str, str]]]:
        """(name, label, facts) per server, for the run record."""
        return [(t.name, t.label, dict(t.facts.values)) for t in self.targets]
