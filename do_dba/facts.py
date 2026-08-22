"""What the server looks like before the model touches it.

Read-only probes, run by the harness rather than by the model: they cost one
round trip instead of ten, and they stop the model from opening with "let me
check what distribution this is".
"""

from __future__ import annotations

from dataclasses import dataclass, field

PROBES: list[tuple[str, str]] = [
    ("hostname", "hostname -f 2>/dev/null || hostname"),
    ("os", ". /etc/os-release 2>/dev/null && echo \"$PRETTY_NAME\" || uname -s"),
    ("kernel", "uname -r"),
    ("arch", "uname -m"),
    # Every address the server answers on, each with the interface carrying it, so
    # a second server can be pointed at this one. On a cloud host the private
    # interface is the one that matters and the one nothing else can tell you
    # about: `--host` holds a public address, and a database bound to that is a
    # database open to the internet.
    #
    # With the interface, because a machine usually has more than one private
    # address and only one of them is the one its peers share. DigitalOcean puts an
    # internal anchor address on the public interface, docker adds 172.17.0.1, and
    # both look exactly like the VPC address on eth1 while reaching nobody. The
    # interface name is the only thing in the output that tells them apart.
    ("addresses",
     "ip -4 -o addr show scope global 2>/dev/null | awk '{print $2\"=\"$4}' | paste -sd' ' - "
     "| grep . || hostname -I 2>/dev/null"),
    ("account", "id -un"),
    ("privilege",
     "if [ \"$(id -u)\" = 0 ]; then echo 'root'; "
     "elif sudo -n true 2>/dev/null; then echo 'passwordless sudo'; "
     "else echo 'NO ROOT - sudo needs a password'; fi"),
    ("cpu", "nproc"),
    ("memory", "free -m 2>/dev/null | awk '/^Mem:/ {print $2\" MB total, \"$7\" MB available\"}'"),
    ("disk_root", "df -h / 2>/dev/null | awk 'NR==2 {print $2\" total, \"$4\" free, \"$5\" used\"}'"),
    ("package_manager",
     "for m in apt-get dnf yum zypper apk; do command -v $m >/dev/null 2>&1 && { echo $m; break; }; done"),
    ("init", "command -v systemctl >/dev/null 2>&1 && echo systemd || echo 'no systemd'"),
    ("db_clients",
     "for b in mysql mariadb psql mongosh mongo valkey-cli redis-cli; do "
     "command -v $b >/dev/null 2>&1 && printf '%s ' \"$b\"; done; echo"),
    ("db_services",
     "systemctl list-units --type=service --state=active --no-pager --plain 2>/dev/null "
     "| awk '{print $1}' | grep -Ei 'mysql|maria|postgres|mongo|valkey|redis' | paste -sd' ' -"),
    ("db_packages",
     "dpkg-query -W -f='${Package} ' '*mysql*' '*mariadb*' '*postgresql*' '*mongodb*' "
     "'*valkey*' '*redis*' 2>/dev/null | tr -s ' ' | cut -c1-200"),
    ("listening", "ss -tlnH 2>/dev/null | awk '{print $4}' | sort -u | paste -sd' ' - | cut -c1-200"),
    ("package_lock",
     "if fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; then echo 'HELD - another install is running'; "
     "else echo free; fi"),
]


@dataclass
class Facts:
    values: dict[str, str] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    @property
    def is_root(self) -> bool:
        return self.values.get("privilege", "").startswith("root")

    @property
    def has_root_access(self) -> bool:
        return "NO ROOT" not in self.values.get("privilege", "")

    @property
    def os_name(self) -> str:
        return self.values.get("os", "unknown")

    def summary(self) -> str:
        lines = [f"{key}: {value}" for key, value in self.values.items() if value]
        empty = [key for key, value in self.values.items() if not value]
        if empty:
            lines.append(f"(nothing reported for: {', '.join(empty)})")
        return "\n".join(lines)


def gather(runner, timeout: float = 30.0) -> Facts:
    """Run every probe, tolerating the ones this server cannot answer."""
    facts = Facts()
    for name, command in PROBES:
        try:
            result = runner.run(command, timeout=timeout)
        except Exception as exc:  # a probe must never sink the whole run
            facts.failures.append(f"{name}: {exc}")
            facts.values[name] = ""
            continue
        value = " ".join(result.stdout.split())
        if not value and result.stderr.strip():
            facts.failures.append(f"{name}: {result.stderr.strip()[:120]}")
        facts.values[name] = value
    return facts
