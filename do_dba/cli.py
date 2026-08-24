"""Command line front end: connect, probe, plan, run, report."""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from . import PROJECT_DIR, __version__
from .agent import MODE_AUTO, MODE_PLAN, MODE_STEP, MODE_UNATTENDED, DBAAgent, Limits
from .fleet import Fleet, parse_target
from .inference import details, providers
from .inference.catalog import Catalog
from .inference.client import EFFORTS, InferenceClient, InferenceError
from .inference.config import ConfigError, load_dotenv
from .inference.pricing import Price, PriceBook, format_cost, format_rate, from_records
from .report import HostInfo, RunRecord, run_directory
from .secrets import KEEPER_PATH, SecretStore, read_keeper, write_keeper
from .ssh import SSHError, SSHRunner, key_fingerprint
from .term import Glyphs, prepare_streams, supports_fancy_glyphs

# Runs land in output/, beside the ones already recorded there - one directory per
# run, created on demand. A run directory holds a transcript of everything that was
# done to a live server and, for some tasks, the database credentials the model
# generated, so output/ has to stay out of version control - see this project's
# .gitignore. DBA_RUNS_DIR overrides this, and is taken relative to the working
# directory.
DEFAULT_RUNS_DIR = PROJECT_DIR / "output"
RUNS_DIR_ENV = "DBA_RUNS_DIR"
PASSWORD_ENV = "DBA_SSH_PASSWORD"
PASSPHRASE_ENV = "DBA_SSH_KEY_PASSPHRASE"
PROVIDER_ENV = "DBA_PROVIDER"
MODEL_ENV = "DO_DBA_MODEL"


class Screen:
    """Everything the operator sees, in one place."""

    def __init__(self, console: Console, glyphs: Glyphs, quiet: bool = False):
        self.console = console
        self.glyphs = glyphs
        self.quiet = quiet

    def line(self, text: str = "", style: str = "") -> None:
        self.console.print(Text(text, style=style) if text else "", markup=False)

    def heading(self, text: str) -> None:
        self.line()
        self.line(text, "bold cyan")

    def note(self, text: str) -> None:
        self.line(f"  {text}", "dim yellow")

    def warn(self, text: str) -> None:
        self.line(f"  {text}", "yellow")

    def error(self, text: str) -> None:
        self.line(f"  {text}", "bold red")

    def emit(self, kind: str, message: str) -> None:
        """The agent's channel into the terminal."""
        if kind == "run":
            body = message if len(message) < 2000 else message[:2000] + self.glyphs.cont
            for index, part in enumerate(body.splitlines() or [""]):
                prefix = f"  {self.glyphs.reply} " if index == 0 else "      "
                self.line(f"{prefix}{part}", "white")
        elif kind == "ok":
            self.line(f"     {self.glyphs.check} {message}", "green")
        elif kind == "fail":
            self.line(f"     {'x' if not self.glyphs.fancy else '×'} {message}", "red")
        elif kind in {"blocked", "error"}:
            self.error(f"  {message}")
        elif kind == "dry":
            self.line(f"  {self.glyphs.reply} [dry run] {message}", "dim")
        else:
            self.note(f"  {message}")


def ask_yes_no(screen: Screen, question: str, default: bool = False, answered_by: str = "") -> bool:
    """A y/n prompt that refuses rather than blocks when there is no terminal.

    answered_by names the switch that already said yes, if one did; it is printed
    with the question so the transcript of the session still shows what was asked
    and who answered. Empty means ask the operator.
    """
    if answered_by:
        screen.note(f"{question} yes ({answered_by})")
        return True
    if not sys.stdin.isatty():
        screen.warn(f"{question} - no terminal to ask on, assuming no")
        return False
    suffix = "[y/N]" if not default else "[Y/n]"
    while True:
        try:
            answer = input(f"  {question} {suffix} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False


# ------------------------------------------------------------------- approvals

# Four things stop a run to ask: an unknown host key, an account without root, the
# plan gate, and every step the guard flags. --yes answers the last two, which are
# the ones about the work; --mode unattended answers all four, for a run nobody is
# sitting in front of. Neither widens what the guard allows - a blocked step is
# still refused, and a flagged one still says in the transcript that it was
# approved by a switch rather than by a person.


def answers_everything(args) -> str:
    """The switch answering every question in the run, or "" if there is none."""
    return f"--mode {MODE_UNATTENDED}" if args.mode == MODE_UNATTENDED else ""


def answers_steps(args) -> str:
    """The switch answering the questions about steps and the plan."""
    return answers_everything(args) or ("--yes" if args.yes else "")


def unanswerable_prompt(args) -> str:
    """Why this run cannot keep its promise not to ask, or "" if it can.

    The four questions above are all yes-or-no, so a switch can answer them. A
    credential is not: getpass reads the console directly, so on a machine that
    has one it waits for typing that is never coming, and on a machine that does
    not it hands back an empty password and fails at authentication instead. Both
    are worse than saying now what to set.
    """
    if not answers_everything(args):
        return ""
    asked = [flag for flag, given in (("--ask-password", args.ask_password),
                                      ("--ask-key-passphrase", args.ask_key_passphrase)) if given]
    if not asked:
        return ""
    variable = PASSWORD_ENV if asked[0] == "--ask-password" else PASSPHRASE_ENV
    return (f"{' and '.join(asked)} cannot be answered by --mode {MODE_UNATTENDED}; "
            f"put the credential in ${variable} instead")


class Approver:
    """Decides whether a flagged step goes ahead."""

    def __init__(self, screen: Screen, answered_by: str = ""):
        self.screen = screen
        self.answered_by = answered_by
        self.approved = 0
        self.declined = 0

    def __call__(self, action: str, detail: str, reason: str) -> bool:
        self.screen.line()
        self.screen.line(f"  proposed {action}:", "bold yellow")
        for part in detail.splitlines():
            self.screen.line(f"      {part}", "white")
        self.screen.line(f"  {reason}", "yellow")
        if self.answered_by:
            self.screen.note(f"  approved automatically ({self.answered_by})")
            self.approved += 1
            return True
        ok = ask_yes_no(self.screen, "run it?", default=False)
        self.approved += int(ok)
        self.declined += int(not ok)
        return ok


def host_key_asker(screen: Screen, accept: str):
    def ask(hostname: str, key) -> bool:
        fingerprint = key_fingerprint(key)
        screen.line()
        screen.warn(f"{hostname} is not in known_hosts")
        screen.line(f"  {key.get_name()} {fingerprint}", "white")
        if accept:
            # The fingerprint is printed either way: trusting a key unseen is the
            # one answer here that cannot be taken back, so it stays on the record.
            screen.note(f"accepted because {accept} was given")
            return True
        return ask_yes_no(screen, "trust this host key and remember it?", default=False)

    return ask


# ------------------------------------------------------------------------ task


def read_task(args, screen: Screen) -> str | None:
    if args.task_file:
        try:
            text = Path(args.task_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            screen.error(f"could not read {args.task_file}: {exc}")
            return None
        if text:
            return text
        screen.error(f"{args.task_file} is empty")
        return None
    if args.task:
        return " ".join(args.task).strip()
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped
    screen.error("no task given - pass --task \"install MySQL and PostgreSQL\" or --task-file")
    return None


def choose_model(catalog: Catalog, args, provider, screen: Screen) -> str | None:
    """The model to drive the run: what was asked for, or the provider's default."""
    requested = args.model or os.environ.get(MODEL_ENV, "").strip() or provider.model_from_env()
    if requested:
        return resolve_model(catalog, requested, screen)

    default = provider.choose_default(catalog)
    if not default:
        if not provider.default_model:
            # A self-hosted server with several models loaded, and no pinned
            # default to fall back on: which one runs the task is the operator's
            # call, so the choice is named rather than guessed.
            screen.error(
                f"{provider.label} serves {len(catalog.chat)} chat models and pins no "
                "default - name one with -m (see --list-models)"
            )
        else:
            screen.error(
                f"{provider.label} does not list {provider.default_model} and no fallback "
                "family matched - name a model with -m (see --list-models)"
            )
        return None
    if default != provider.default_model:
        screen.note(f"{provider.default_model} is not available; using {default}")
    return default


def resolve_model(catalog: Catalog, requested: str, screen: Screen) -> str | None:
    match, candidates = catalog.resolve(requested)
    if match is not None:
        if not match.is_chat:
            screen.error(f"{match.id} is not a chat model")
            return None
        return match.id
    if candidates:
        screen.error(f"'{requested}' matches several models:")
        for model in candidates[:12]:
            screen.line(f"      {model.id}", "white")
        return None
    screen.error(f"no model matches '{requested}' - try --list-models")
    return None


# --------------------------------------------------------------------- screens


def show_facts(screen: Screen, facts, name: str = "") -> None:
    screen.heading(f"server {name}".strip())
    table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 2, 0, 2))
    table.add_column(style="dim", no_wrap=True)
    table.add_column(style="white", overflow="fold")
    for key, value in facts.values.items():
        table.add_row(key.replace("_", " "), value or "-")
    screen.console.print(table)
    for failure in facts.failures[:5]:
        screen.note(failure)


def show_network(screen: Screen, fleet: Fleet) -> None:
    """What the servers can reach each other on, in a line or a short warning.

    Read at the top of the run rather than deduced from step 30: private networking
    not being enabled is the commonest reason a replication or cluster task cannot
    be done the way it should be, and the operator is the only one who can turn it
    on. Nothing is asked here - the model is told the same thing and works with
    what there is.
    """
    if fleet.private_mesh:
        listed = ", ".join(f"{target.name} {target.private}" for target in fleet)
        screen.note(f"private network: {listed}  (every pair reachable)")
        # Shown rather than folded into "reachable": the operator is the one who
        # knows whether a firewall in front of these machines is deliberate.
        for path in fleet.unproven_paths:
            screen.note(f"         {path.source} -> {path.describe()}")
        return
    for path in fleet.broken_paths:
        screen.warn(f"private network: {path.source} -> {path.describe()}")
    if all(path.any_path for path in fleet.broken_paths):
        screen.warn("         the servers only have their public addresses to talk on, so "
                    "traffic between them will cross the public network")
    else:
        screen.warn("         some of these servers cannot reach each other at all")


def show_settings(screen: Screen, model: str, provider, prices: PriceBook, args, limits: Limits,
                  info=None) -> None:
    price = prices.get(model)
    if not provider.metered:
        # The rate is zero and saying so twice per line adds nothing: what the
        # operator wants to know is that this run is not on anybody's bill.
        rate = "no per-token bill"
    elif price:
        rate = f"${format_rate(price.input)}/M in {screen.glyphs.sep} ${format_rate(price.output)}/M out"
    else:
        rate = "no published price"
    screen.heading("run")
    # The context length only where the gateway reports one: it is what the growing
    # prompt is measured against, and on a server that loads whatever it was asked
    # for it is a property of this session rather than of the model.
    facts = [provider.label, rate]
    if info is not None and info.context_label:
        facts.append(info.context_label)
    separator = f" {screen.glyphs.sep} "
    screen.line(f"  model    {model}  ({separator.join(facts)})", "white")
    if info is not None and info.loaded is False:
        # Said before the run rather than discovered as a stall on step 1. This is
        # the case first_token_wait exists for; see inference/providers.py.
        screen.note(f"         {provider.label} does not have this model in memory, so the "
                    "first step waits while the weights are read off disk")
    if provider.usage_accounting:
        # Worth saying, because it is the difference between a cost line that can
        # be reconciled with the gateway's bill and one that only ought to be.
        screen.note(f"         {provider.label} reports what each reply was charged, "
                    "so the cost below is the billed figure rather than the rate above")
    mode = args.mode + (" + dry run" if args.dry_run else "")
    screen.line(f"  mode     {mode}", "white")
    if args.mode == MODE_UNATTENDED and not args.dry_run:
        screen.warn("         nothing will be asked - the guard's blocks are all that "
                    "will stop a step, including destructive ones")
    cap = format_cost(limits.max_cost) if limits.max_cost else "none"
    screen.line(
        f"  limits   {limits.max_steps} steps {screen.glyphs.sep} "
        f"{limits.command_timeout:.0f}s per command {screen.glyphs.sep} cost cap {cap}",
        "white",
    )
    parts = limits.context_parts()
    if parts:
        # What was made of the window, since it is the operator's business how much
        # of a result the model gets to see and what a run will cost to keep asking.
        # Without a reported window this line is absent and the fixed limits apply.
        screen.line(f"  context  {separator.join(parts)}", "white")


def show_outcome(screen: Screen, outcome, record: RunRecord, report: Path,
                 secrets_path: Path | None, on_servers: bool = False) -> None:
    style = {"done": "bold green", "aborted": "yellow", "cancelled": "yellow"}.get(outcome.status, "bold red")
    screen.heading("result")
    screen.line(f"  {outcome.status}", style)
    if outcome.summary:
        for part in outcome.summary.splitlines():
            screen.line(f"  {part}", "white")

    if record.verifications:
        screen.line()
        screen.line("  verified independently:", "bold")
        many = len(record.hosts) > 1
        for check in record.verifications:
            head = (check.output or "").strip().splitlines()
            where = f"[{check.host}] " if many else ""
            screen.line(f"    {where}{check.command}", "dim")
            screen.line(f"      exit {check.exit_code}: {head[0] if head else '(no output)'}", "white")

    cost = (format_cost(outcome.cost) + record.cost_note
            + ("" if outcome.cost_complete else " (some replies unpriced)"))
    screen.line()
    screen.line(
        f"  {outcome.executed} of {outcome.steps} steps executed {screen.glyphs.sep} "
        f"{record.prompt_tokens:,} in / {record.completion_tokens:,} out {screen.glyphs.sep} {cost}",
        "dim",
    )
    screen.line(f"  report   {report}", "cyan")
    if secrets_path:
        screen.line(f"  secrets  {secrets_path}  (generated credentials, keep private)", "cyan")
    if on_servers:
        # Said every time: it is the difference between the next run logging in and
        # the next run resetting the password, and it is plaintext on a server.
        screen.line(f"           and on each server in {KEEPER_PATH}, root only, "
                    "so the next run can use them", "dim")
    screen.line()


# ------------------------------------------------------------------------ main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dba.py",
        description="Let a hosted model carry out a DBA task on a server over SSH.",
        epilog="examples:\n"
               "  dba.py --host 203.0.113.10 --task \"install MySQL and PostgreSQL\"\n"
               "  dba.py --host 10.0.0.2 --host 10.0.0.3 "
               "--task \"set up MySQL replication\"\n"
               "  dba.py --host primary=10.0.0.2 --host replica=10.0.0.3 "
               "--task \"set up MySQL replication\"   # roles fixed by you",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ssh = parser.add_argument_group("servers")
    ssh.add_argument("--host", action="append", default=None, metavar="[NAME=][USER@]HOST[:PORT]",
                     help="the target server (required, except with --list-models). Repeat it for a "
                          "task that spans servers - the harness labels them node1, node2, ... and "
                          "the model works out which one takes which role. Give a name "
                          "(--host primary=10.0.0.2) to decide the roles yourself; then the model "
                          "follows the names you chose")
    ssh.add_argument("-u", "--user", default="root", help="SSH user for hosts that name none (default: root)")
    ssh.add_argument("-p", "--port", type=int, default=22, help="SSH port for hosts that name none (default: 22)")
    ssh.add_argument("-i", "--key", default=None, metavar="PATH", help="private key file")
    ssh.add_argument("--ask-key-passphrase", action="store_true", help="prompt for the key passphrase")
    ssh.add_argument("--ask-password", action="store_true", help="prompt for a password instead of a key")
    ssh.add_argument("--no-server-secrets", action="store_true",
                     help=f"do not keep generated credentials on the servers in {KEEPER_PATH}; "
                          "later runs then cannot log in with them")
    ssh.add_argument("--accept-host-key", action="store_true",
                     help="trust an unknown host key without asking (only on a host you just created)")

    work = parser.add_argument_group("task")
    work.add_argument("--task", nargs="+", default=None, help="what to do, in plain English")
    work.add_argument("--task-file", default=None, metavar="PATH", help="read the task from a file")
    work.add_argument("-m", "--model", default=None, help="model id (partial names work)")
    work.add_argument("--provider", default=os.environ.get(PROVIDER_ENV, "").strip() or providers.OPENROUTER,
                      metavar="NAME",
                      help=f"where the model is hosted: {', '.join(providers.NAMES)} "
                           f"(default: {providers.OPENROUTER})")
    work.add_argument("--mode", choices=[MODE_PLAN, MODE_STEP, MODE_AUTO, MODE_UNATTENDED],
                      default=MODE_PLAN,
                      help="plan: show a plan and ask once (default); step: ask before every step; "
                           "auto: only ask about risky steps; unattended: ask nothing - every "
                           "question is answered yes, including an unknown host key")
    work.add_argument("--dry-run", action="store_true", help="show the steps, execute nothing")
    work.add_argument("--yes", action="store_true",
                      help="approve the plan and every flagged step without asking, but still ask "
                           "about an unknown host key and a missing sudo")
    work.add_argument("--probe", action="store_true", help="print what the server looks like and exit")

    limits = parser.add_argument_group("limits")
    limits.add_argument("--max-steps", type=int, default=40, help="give up after this many steps (default: 40)")
    limits.add_argument("--timeout", type=float, default=300.0, help="seconds per command (default: 300)")
    limits.add_argument("--max-cost", type=float, default=None, metavar="USD",
                        help="stop once model spend reaches this")
    limits.add_argument("--temperature", type=float, default=0.2, help="sampling temperature (default: 0.2)")
    limits.add_argument("--effort", choices=EFFORTS, default=None,
                        help="ask the model to think before each step, where it can be "
                             "asked - the gateway turns this into whatever the model wants. "
                             "Thinking is billed as output, so a high effort costs "
                             "several times a step that did none. Unset asks for nothing, "
                             "which leaves a reasoning model reasoning as it always does")

    output = parser.add_argument_group("output")
    output.add_argument("--runs-dir",
                        default=os.environ.get(RUNS_DIR_ENV, "").strip() or str(DEFAULT_RUNS_DIR),
                        metavar="PATH",
                        help=f"where to write the transcript and report (default: {DEFAULT_RUNS_DIR.name}/ "
                             f"in the {PROJECT_DIR.name} project, or ${RUNS_DIR_ENV})")
    output.add_argument("--list-models", action="store_true", help="print available models and exit")
    output.add_argument("--no-color", action="store_true", help="disable styling")
    output.add_argument("--version", action="version", version=f"do-dba {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # --list-models only talks to the inference API, so demanding a server for it
    # would be asking for something the command never uses.
    if not args.host and not args.list_models:
        parser.error("the following arguments are required: --host")
    conflict = unanswerable_prompt(args)
    if conflict:
        parser.error(conflict)

    prepare_streams()
    console = Console(
        no_color=args.no_color,
        soft_wrap=True,
        legacy_windows=False if not sys.stdout.isatty() else None,
    )
    screen = Screen(console, Glyphs(supports_fancy_glyphs()))

    # The .env next to dba.py, if there is one. It uses setdefault, so a variable
    # already exported in the environment wins over the file.
    load_dotenv()
    prices = PriceBook()
    if prices.warning:
        screen.warn(prices.warning)

    try:
        provider = providers.get(args.provider)
        api_key = provider.api_key()
    except ConfigError as exc:
        screen.line()
        screen.warn(str(exc))
        return 2

    client = InferenceClient(
        api_key=api_key,
        base_url=provider.base(),
        headers=provider.headers,
        label=provider.label,
        usage_accounting=provider.usage_accounting,
        key_help=provider.key_help,
        read_timeout=provider.read_timeout(),
    )
    try:
        records = client.list_models()
    except InferenceError as exc:
        screen.error(str(exc))
        return 2
    # A gateway with more to say about its models than the OpenAI listing allows is
    # asked; one without is not, and one that has stopped answering is not waited on.
    records = details.described(records, provider.detail_url(), api_key)
    catalog = Catalog(records)
    # OpenRouter publishes its rates in the same response, so every model it
    # serves can be priced exactly instead of falling back to "cost n/a".
    prices.learn(from_records(records))
    if not provider.metered:
        # A server the operator already owns sends no bill per token, so its
        # models are priced at zero rather than left unpriced - "cost n/a" on
        # every line of a run that cost nothing reads like a failure to work it
        # out. learn() keeps rates already known, so a model id that also appears
        # in pricing.json or the hand-kept table is still priced by it.
        prices.learn({model.id: Price(0.0, 0.0) for model in catalog.all})

    if args.list_models:
        for model in catalog.chat:
            price = prices.get(model.id)
            if not provider.metered:
                rate = "no per-token bill"
            else:
                rate = (f"${format_rate(price.input)}/${format_rate(price.output)} per M"
                        if price else "price unpublished")
            # Only the model that is loaded is marked. On a server that holds one
            # at a time that is the useful half of the fact - it answers at once,
            # where any other name means waiting for weights to be read off disk -
            # and marking the nine cold ones would say the same thing nine times.
            warm = "  loaded" if model.loaded else ""
            screen.line(f"{model.id:44} {model.context_label:9} {rate}{warm}", "white")
        return 0

    task = None if args.probe else read_task(args, screen)
    if task is None and not args.probe:
        return 2

    model = choose_model(catalog, args, provider, screen)
    if model is None:
        return 2

    try:
        fleet = Fleet([parse_target(spec, args.user, args.port) for spec in args.host])
    except ValueError as exc:
        screen.line()
        screen.error(str(exc))
        return 2

    # One credential set for the whole fleet. Servers built to work together are
    # built the same way, and a run needing a different key per server is a run
    # better done one server at a time.
    password = os.environ.get(PASSWORD_ENV) or None
    if args.ask_password:
        password = getpass.getpass(f"  password for {fleet.label}: ")
    passphrase = os.environ.get(PASSPHRASE_ENV) or None
    if args.ask_key_passphrase:
        passphrase = getpass.getpass("  passphrase for the private key: ")

    ask_host_key = host_key_asker(
        screen, "--accept-host-key" if args.accept_host_key else answers_everything(args)
    )
    for target in fleet:
        target.runner = SSHRunner(
            host=target.host,
            user=target.user,
            port=target.port,
            key_path=args.key,
            password=password,
            passphrase=passphrase,
            ask_host_key=ask_host_key,
        )

    screen.heading("connecting")
    if fleet.many and not fleet.assigned:
        # Said once, up front: the operator passed addresses and gets told what the
        # model will call them, and that choosing the roles is the model's job.
        screen.note("labelled " + ", ".join(f"{t.name} = {t.host}" for t in fleet)
                    + " - the model decides which server takes which role")
    try:
        fleet.connect(before=lambda target: screen.line(
            f"  {f'{target.name}  ' if fleet.many else ''}"
            f"{target.user}@{target.host}:{target.port}", "white"))
    except SSHError as exc:
        # Named, because with several servers the message alone does not say which
        # one refused - and the ones already open have been closed again.
        screen.error(str(exc))
        return 2

    try:
        return _run(args, screen, fleet, client, catalog, prices, provider, model, task or "")
    except KeyboardInterrupt:
        screen.line()
        screen.warn("interrupted - the transcript up to this point was written")
        return 130
    finally:
        fleet.close()


def adopt_secrets(screen: Screen, fleet: Fleet, store: SecretStore) -> None:
    """Read back the credentials earlier runs left on these servers.

    Before the model is asked anything, because it changes what the first step can
    be: a run that does not know the root password exists opens by trying to log in
    without one, and the only way in from there is to reset it.

    Names on screen, never values - the operator has them in the run directory, and
    the point of the file is that nobody has to go and look.
    """
    for target in fleet:
        learned, clashed = store.adopt(read_keeper(target.runner))
        where = f" on {target.name}" if fleet.many else ""
        if learned:
            screen.note(f"credentials already in place{where}: {', '.join(learned)}")
        if clashed:
            # Two servers disagreeing about a shared password is worth saying out
            # loud: the run will use the first value it read, and if the other server
            # is the one that matters, that is an operator's decision to make.
            screen.warn(f"{target.name} has a different value for {', '.join(clashed)}; "
                        "this run will use the one read first")


def _run(args, screen: Screen, fleet: Fleet, client, catalog, prices, provider, model: str, task: str) -> int:
    fleet.survey(on_host=lambda target: screen.note(f"probing {target.name}") if fleet.many else None)
    for target in fleet:
        show_facts(screen, target.facts, target.name if fleet.many else "")
    if fleet.many:
        show_network(screen, fleet)

    if args.probe:
        return 0

    starved = fleet.without_root
    if starved:
        who = ("this account" if not fleet.many
               else f"the account on {', '.join(target.name for target in starved)}")
        screen.error(f"{who} has neither root nor passwordless sudo; almost every DBA step will fail")
        if not ask_yes_no(screen, "carry on anyway?", default=False, answered_by=answers_everything(args)):
            return 2

    # Sized to the model where the gateway said how large its context window is:
    # how much of a result the model is shown, how many results stay whole, and how
    # long a single reply may run. Where it said nothing this is the fixed set of
    # limits the harness has always used. See agent.py's context budget.
    info = catalog.get(model)
    limits = Limits.for_window(
        info.context_window if info else 0,
        max_steps=max(1, args.max_steps),
        command_timeout=max(5.0, args.timeout),
        max_cost=args.max_cost,
    )

    screen.heading("task")
    for part in task.splitlines():
        screen.line(f"  {part}", "white")
    show_settings(screen, model, provider, prices, args, limits, info)

    store = SecretStore()
    keep_on_servers = not args.no_server_secrets
    if keep_on_servers:
        adopt_secrets(screen, fleet, store)

    def persist_secrets() -> None:
        """Put every credential back on every server, as soon as there is a new one."""
        for target in fleet:
            error = write_keeper(target.runner, store)
            if error:
                screen.warn(f"could not store the credentials on {target.name}: {error}")
        store.mark_saved()

    directory = run_directory(Path(args.runs_dir), fleet.slug)
    record = RunRecord(
        directory=directory,
        task=task,
        hosts=[HostInfo(name=name, label=label, facts=values)
               for name, label, values in fleet.host_lines()],
        model=model,
        provider=provider.label,
        metered=provider.metered,
        mode=args.mode,
        dry_run=args.dry_run,
        context=", ".join(limits.context_parts()),
        redact=store.redact,
    )

    agent = DBAAgent(
        client=client,
        model=model,
        fleet=fleet,
        task=task,
        record=record,
        store=store,
        prices=prices,
        emit=screen.emit,
        approve=Approver(screen, answers_steps(args)),
        mode=args.mode,
        dry_run=args.dry_run,
        limits=limits,
        temperature=args.temperature,
        effort=args.effort,
        persist=persist_secrets if keep_on_servers else None,
    )

    if args.mode == MODE_PLAN:
        screen.heading("plan")
        plan = agent.plan()
        if plan is None:
            screen.error("the model did not return a plan")
            record.status = "failed"
            record.summary = "the model did not return a plan"
            screen.line(f"  report   {record.write_report()}", "cyan")
            return 1
        record.event("plan", text=plan)
        for part in plan.strip().splitlines():
            screen.line(f"  {part}", "white")
        screen.line()
        if not ask_yes_no(screen, "proceed?", default=False, answered_by=answers_steps(args)):
            record.status = "cancelled"
            record.summary = "the operator did not approve the plan"
            screen.warn("nothing was run")
            screen.line(f"  report   {record.write_report()}", "cyan")
            return 1

    screen.heading("working")
    try:
        outcome = agent.run()
    except KeyboardInterrupt:
        record.status = "cancelled"
        record.summary = "interrupted by the operator"
        screen.line()
        screen.warn("interrupted")
        if keep_on_servers and store.unsaved:
            persist_secrets()
        secrets_path = store.save(directory / "secrets.json")
        screen.line(f"  report   {record.write_report()}", "cyan")
        if secrets_path:
            screen.line(f"  secrets  {secrets_path}", "cyan")
        return 130

    # After the run, not only during it: a credential first used by a VERIFY command
    # is generated after the last step.
    if keep_on_servers and store.unsaved:
        persist_secrets()
    secrets_path = store.save(directory / "secrets.json")
    report = record.write_report()
    show_outcome(screen, outcome, record, report, secrets_path,
                 on_servers=keep_on_servers and bool(store.names))
    return 0 if outcome.ok else 1
