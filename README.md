# do-dba

Give a hosted model an SSH login and a DBA task in plain English, and it does the
work on the server: one command — or one script — per reply, each classified before
it runs, and nothing reported as finished until the harness has re-run the checks
itself.

```bash
uv run dba.py --host 203.0.113.10 --task \
  "install MySQL and PostgreSQL, both enabled at boot, each with an app database and its own login user"

uv run do-dba --host 203.0.113.10 --task-file task.md     # installed script
uv run dba.py --host 203.0.113.10 --probe                 # just look at the server
echo "restore the nightly dump into staging" | uv run dba.py --host db1   # piped task

uv run dba.py --host 10.0.0.2 --host 10.0.0.3 \
  --task "set up MySQL replication between these servers over the private network"
```

Hand over several servers and the model works out what to do with them: the harness
labels them `node1`, `node2`, … and the model says in its first step which one takes
which role. Name them yourself — `--host primary=10.0.0.2` — when the roles are
already decided and the model should follow them.

Five engines are known to it — MySQL, MariaDB, PostgreSQL, MongoDB and
Valkey/Redis. That means the read-only probes look for all five, the guard reads
each one's clients, config syntax and destructive verbs, the standing checks at
the end cover their services and ports, and the model is told which client takes
a statement and which takes a script.

Nothing outside this repository is needed to run it: the gateway client, the model
catalog, the price table and the credential discovery live in
[do_dba/inference/](do_dba/inference/), so the only dependencies are the OpenAI
SDK, paramiko and rich.

SSH is [paramiko](https://www.paramiko.org/): it uses your agent and `~/.ssh` keys
by default, or `-i PATH`, `--ask-key-passphrase`, `--ask-password`. Unknown host
keys are shown with their fingerprint and confirmed once, then remembered in
`known_hosts` (`--accept-host-key`, or `--mode unattended`, to trust one on a host
you just created without being asked).

## Setup

Managed with [uv](https://docs.astral.sh/uv/) — no manual venv, and `uv.lock` pins
the exact dependency versions.

```bash
uv sync                    # create .venv and install from the lockfile
cp .env.example .env       # then paste your key into .env
```

`uv run` syncs before it runs, so `uv sync` above is only needed if you want it
done first. A key already exported in your environment beats the `.env` file.

## Where the model comes from

Three gateways, all OpenAI-compatible, so the client, catalog and cost accounting
are shared and only the base URL, the key and the spelling of model ids differ:

| `--provider` | Default model | Key | Model ids |
| --- | --- | --- | --- |
| `openrouter` (default) | `anthropic/claude-sonnet-4.5` | `OPENROUTER_API_KEY` | `vendor/model` |
| `digitalocean` | `anthropic-claude-opus-5` | `DIGITALOCEAN_INFERENCE_KEY` | flat, e.g. `anthropic-claude-opus-5` |
| `selfhosted` | whatever is loaded (`-m`) | none | whatever the server calls them |

`do`, `or`, `gradient`, `local`, `lmstudio`, `vllm`, `ollama` and any unambiguous
prefix work as provider names. Keys come from the environment or the `.env` next
to `dba.py`; set `DBA_PROVIDER=digitalocean` to change the default for good. Pick
a model with `-m` (`-m claude-opus-4.5`, `-m gpt-oss-120b` — partial names work,
and an ambiguous one prints its matches) or `DO_DBA_MODEL`, and see what your key
can reach with `uv run dba.py --list-models`, which needs no `--host` since it
never opens one.

### A server you run yourself

`--provider local` points the same harness at any OpenAI-compatible server — LM
Studio, vLLM, llama.cpp, Ollama — and defaults to the Mac Studio at
`https://mac-studio-lm.int.percona.com`:

```bash
uv run dba.py --provider local --list-models
uv run dba.py --provider local -m qwen3.8-27b --host 203.0.113.10 \
  --task "install MySQL with an app database and its own login user"

DBA_SELFHOSTED_BASE_URL=http://127.0.0.1:1234 uv run dba.py --provider local ...
```

Six things differ from a hosted gateway, all of them because the machine is
yours:

- **No key.** None is asked for and none of yours is sent. If your server does
  sit behind auth, put the credential in `DBA_SELFHOSTED_KEY` — a 401 says so.
- **No default model.** What the server serves is whatever was loaded onto it, so
  nothing can be pinned here: with one chat model loaded it is used, and with
  several `-m` says which (`DBA_SELFHOSTED_MODEL` to keep the choice). Embedding
  models are filtered out of the listing as everywhere else.
- **No bill.** The hardware was paid for before the run started, so replies are
  priced at zero and the report says *self-hosted — no per-token bill* rather
  than reporting an estimate or "cost n/a". Tokens are still counted, and
  `--max-cost` simply has nothing to trip on.
- **A longer first wait.** The first request usually loads the weights off disk,
  which on a large model is minutes, so a self-hosted run waits 900s for the
  first token instead of 180. `DO_INFERENCE_TIMEOUT` overrides it either way.
- **The model can be put away mid-run.** A box in an office unloads on an idle
  timer, and the next request comes back `400 Model unloaded.` — which ended a
  recorded run at step 4, with MySQL and PostgreSQL installed and neither database
  created. The weights are still on disk, so the same request is sent once more,
  which is what makes the server load them again, and the operator is told why the
  step took longer. A 400 that says anything else is still a refusal and is not
  retried.
- **A second listing.** `/v1/models` on such a server is three fields — id,
  object, owned_by — with no context length and no way to tell an embedding model
  from a chat one except by its name. LM Studio publishes the rest at
  `/api/v0/models`, which the harness reads alongside the OpenAI one, so
  `--list-models` and the run header can say `262K ctx` and mark which model is
  in memory:

  ```
  qwen/qwen3.8-27b        262K ctx  no per-token bill  loaded
  openai/gpt-oss-20b      131K ctx  no per-token bill
  ```

  A model named `loaded` is answering now; a cold one spends the first step being
  read off disk. It is an enrichment and never a requirement: vLLM, llama.cpp and
  Ollama have no such endpoint and answer 404, which costs the listing those two
  columns and nothing else. `/v1/models` is never contradicted, and a model only
  the detail endpoint knows about is not offered — the chat endpoint would refuse
  it.

`DBA_SELFHOSTED_BASE_URL` may be given as a bare host — `/v1` is filled in when
the URL carries no path of its own, so a server mounted behind a proxy at
`/openai/v1` is still taken exactly as written. The detail endpoint is derived from
the same address with the `/v1` taken off, so a proxy that moves one moves both.

A small local model is a real model on a real server: the guard, the plan gate
and the standing checks all apply unchanged, which is the point of trying one on
a task before spending on a hosted one.

If the pinned default has been retired — which on OpenRouter happens week to week
— the newest member of the first available preferred family is used instead, and
the substitution is printed. OpenRouter publishes its own rates in the same
`/v1/models` response the catalog comes from, so every model it serves is priced
exactly rather than falling back to "cost n/a"; a hand-set rate in `pricing.json`
still wins. Better than any published rate: it will report what it actually
charged for each reply, and the harness asks, so on OpenRouter the cost line is
the billed one (see [CLI flags](#cli-flags)).

## How a run goes

1. **Look first.** Read-only probes collect the OS, init system, package
   manager, privilege level, memory, disk, listening ports and whether the
   package lock is held. This goes to the model as facts so it does not guess,
   and into the report as *Server as found* (*Servers as found*, one block each,
   when there is more than one). `--probe` stops here and just prints it — no task
   needed, and nothing on the server changes.
2. **Plan.** In the default mode the model writes a plan and you approve it once.
3. **Step.** Every reply is exactly one step — `run`, `script`, `write_file`,
   `done` or `abort` — in a line-based format
   ([do_dba/protocol.py](do_dba/protocol.py)). A reply that cannot be read is
   explained back to the model rather than killing the run.
4. **Classify, then execute.** The guard sees the command before the server does.
   Output, exit code and timing go back as the next observation, so a failure is
   something the model reads and fixes.
5. **Verify.** The harness re-runs the checks itself and writes the report.

```
server
  os               Ubuntu 24.04.1 LTS
  init             systemd
  package manager  apt-get
  privilege        root
  listening        0.0.0.0:22
  package lock     free

run
  model    openai-gpt-oss-120b  ($0.05/M in · $0.40/M out · 131K ctx)
  mode     auto
  limits   40 steps · 300s per command · cost cap none
  context  131K window · 6,553 chars per result · 55K of results kept whole · replies capped at 16K

working
  ↳ apt-get install -y mysql-server postgresql postgresql-contrib
     ✓ exit 0 in 41.2s

  proposed run:
      sudo -u postgres psql -c "DROP DATABASE IF EXISTS app"
  drops a database, schema, table, user or role
  run it? [y/N]
```

and at the end, the checks the harness ran for itself:

```
result
  done
  MySQL 8.0 and PostgreSQL 16 are installed and enabled at boot. Both have an
  app database owned by an app login user; the passwords are in secrets.json.

  verified independently:
    systemctl is-active --quiet mysql && systemctl is-enabled --quiet mysql
      exit 0: (no output)
    mysql -e "SHOW DATABASES LIKE 'app'"
      exit 0: app
    sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='app'"
      exit 0: 1

  9 of 9 steps executed · 24,118 in / 3,402 out · $0.002914
  report   20260820-203015-203.0.113.10/report.md
  secrets  20260820-203015-203.0.113.10/secrets.json  (generated credentials, keep private)
```

## More than one server

Replication, a cluster, a load-balanced pair: repeat `--host`. Pass the machines and
nothing else, and working out which one takes which role is part of the task:

```bash
uv run dba.py --host 10.0.0.2 --host 10.0.0.3 \
  --task "set up MySQL replication between these servers over the private network"
```

The harness labels them `node1`, `node2`, … — two addresses a digit apart are the
worst thing to route steps by — and tells the model that the labels say nothing
about the roles: it reads the task, decides which server does what, states the
assignment in its first step, and holds to it. The servers are interchangeable, so
any assignment that satisfies the task is a good one; what would break the run is
changing its mind halfway, which is the one thing it is told not to do. The
addresses stay next to the labels everywhere they appear — the prompt, the report
header, the run directory — so there is never a question of which machine `node2`
is.

Name them yourself when the roles are not the model's to pick — an existing primary
you are attaching a replica to, a machine that must stay untouched:

```bash
uv run dba.py --host primary=10.0.0.2 --host replica=10.0.0.3 \
  --task "make the replica read from the primary over the private network"
```

Then the model is told the names were chosen by the operator and to follow them. A
name may also be given to only some of the servers; the rest are labelled around it
(`--host web=10.0.0.2 --host 10.0.0.3` gives `web` and `node1`).

The value is `[NAME=][USER@]HOST[:PORT]`; `-u` and `-p` supply the default user and
port for the ones that name none. One SSH credential covers the whole set — servers
built to work together are built the same way, and a run needing a different key per
server is a run better done one server at a time.

What changes with two or more:

- **Every step says where it runs.** `HOST: replica` is added to the step format,
  and a `run`, `script` or `write_file` step without one — or naming something that
  is not a server in the run — is refused and handed back unrun. Nothing is broadcast and
  nothing is guessed: on a pair the two servers are configured differently on
  purpose, and `DROP DATABASE app` is routine on the node being rebuilt and
  catastrophic on the one serving traffic. The name may be written as the model
  likes (`[replica]`, `` `replica` ``, `replica - the standby`, the address, a
  prefix); anything ambiguous counts as no answer.
- **The private network is found, not assumed.** What you pass to `--host` is
  normally a public address, because that is what you have to hand; the address the
  servers should say things to each other on is the private one, and nothing outside
  a server can discover it. So the survey reads it off each machine and then has
  every server try to open a connection to every other one — private addresses
  first, the public one only if none of them gets through, and port 22, since no
  database is listening yet. Each server is listed with `private: 10.116.0.3` next
  to its label, `peers: replica private 10.116.0.3:22 reachable` is a fact the model
  starts from, and the prompt says in words which address to bind to, point replicas
  at, and scope grants and `pg_hba.conf` lines to — the specific peer, never a range
  and never `%`.

  *Addresses*, plural, because a cloud host has several private ones and only one of
  them is the address its peers share. DigitalOcean puts an internal anchor address
  on the public interface, docker adds `172.17.0.1` to every machine that runs it,
  and both look exactly like a VPC address on `eth1`. So the survey records the
  interface with each address, tries up to three of a peer's — the ones on their own
  interface before anything sitting beside the public address, and never a bridge,
  whose address is this machine's too — and the one that answers is the one the run
  uses from then on. Probing the first and reporting "no private network" when it
  failed is how a working VPC gets missed.

  How it failed is part of the finding. A refused connection means the packets made
  the round trip and only sshd was not there; a port that times out on an address
  that answers ping is a firewall in front of a network that works; `ip route get`
  says whether this machine can send to it at all. The first two are private
  networks a cluster can be built on, so they count as reachable and are then said
  out loud — *the packets get through but port 22 did not answer* — with an
  instruction to open the database port to the peer's private address and prove the
  connection with a client rather than assume it.

  Three answers, because the failures are not the same failure. Every pair
  reachable privately: use those addresses. A private address that does not answer
  while the public one does — two machines in different VPCs, or private networking
  never enabled — falls back to the public address with the scoping unchanged, and
  the summary has to say the traffic crosses the public network. Nothing answering
  at all: no database configuration will fix that, so the model is told to report it
  and abort. You see the same finding at the top of the run, as one dim line or as a
  warning per pair, and every address that was tried is named in it.
- **One credential, both ends.** The same `{{DBA_SECRET:mysql_repl}}` placeholder
  resolves to the same value on every server, which is exactly how a shared
  credential is set up: write the placeholder on both sides and the harness makes
  them match.
- **Checks can be scoped.** `VERIFY: [replica] mysql -e "SHOW REPLICA STATUS\G"`
  runs on that server only; a `VERIFY:` with no `[name]` runs on **every** server,
  since half a cluster working is not the task. The model is also told to verify
  from both ends — a replica that reports itself healthy is not proof.
- **The record names the server.** Every step, verification and log line carries
  it, the report header lists the servers, and the run directory is named for the
  first plus a count (`20260821-014233-10.0.0.2-plus1`).

A run with one `--host` is unchanged in every respect: no `HOST:` line in the
format, no peer probe, no server name in the report.

## Modes

| `--mode` | Who is asked |
| --- | --- |
| `plan` (default) | The plan once, up front; then only steps the guard flags |
| `step` | Every step, before it runs |
| `auto` | Only steps the guard flags — no plan approval |
| `unattended` | Nobody. Every question is answered yes |

Four things stop a run to ask: an unknown host key, an account with neither root
nor passwordless sudo, the plan gate, and every step the guard flags. `--yes`
answers the last two — the ones about the work — and still asks about the host
key and the missing sudo. `--mode unattended` answers all four, for a run nobody
is sitting in front of: CONFIRM steps go ahead without a prompt, and an unknown
host key is trusted on first use as if `--accept-host-key` had been given.

Neither switch widens what the guard allows. A BLOCK is still refused with
nobody watching, and an automatically approved step says so in the output and in
the report, so you can see afterwards what ran unasked. The one thing yes cannot
answer is a credential, so `--mode unattended` with `--ask-password` or
`--ask-key-passphrase` is refused before the run starts — put those in
`$DBA_SSH_PASSWORD` and `$DBA_SSH_KEY_PASSPHRASE` instead. `--dry-run` runs the
whole loop, prints each step, and executes nothing but the read-only probes, so
you can see what a model intends to do to a server it will never touch.

## The guard

Every command, every script and every file write is classified before execution
([do_dba/guard.py](do_dba/guard.py)) as one of three outcomes:

- **ALLOW** — runs.
- **CONFIRM** — you are shown the command and the reason, and it runs only if you
  say yes. Restarting or powering off the host, `apt purge`, `DROP`/`TRUNCATE`/
  unqualified `DELETE`, `dropDatabase()`/`drop()`/`deleteMany({})`/`dropUser()`,
  `FLUSHALL`/`FLUSHDB`, deleting *or moving aside* `/var/lib/mysql`,
  `/var/lib/postgresql`, `/var/lib/mongodb` or `/var/lib/valkey` — to the service
  that was using it, `mv /var/lib/mysql /var/lib/mysql.bak` and
  `rm -rf /var/lib/mysql` are the same event, and the move is the recorded one
  (five steps across three runs, twice in the same step as
  `mysqld --initialize-insecure`) — binding a database to `0.0.0.0`
  (`bind-address`, `listen_addresses`, `bind` in valkey.conf, `bindIp` in
  mongod.conf), turning authentication off (`skip-grant-tables`,
  `authorization: disabled`, `CONFIG SET requirepass ''`), empty or `'%'` grants,
  a database server backgrounded with `&` instead of started through systemd (see
  below), touching `authorized_keys`/`sshd_config`/sudoers, `curl … | sh`, and
  anything that sends data off the box.
- **BLOCK** — never runs, and the reason is fed back so the model picks another
  approach. Two kinds: things that destroy the machine (`mkfs`, `dd of=/dev/sda`,
  `rm -rf /`, stopping sshd) and things that would hang forever on a closed
  stdin — `vim`, `less`, bare `mysql`/`mariadb`/`psql`/`mongosh`/`valkey-cli`,
  `apt install` without `-y` (but not `-s` or `--print-uris`, which print the plan
  or the `.deb` URLs and return before apt reaches a prompt — a recorded run was
  refused `--print-uris` and spent the next step downloading 134MB of package to
  learn the same thing), `mysql_secure_installation`, `crontab -e`, a
  database server started by hand instead of through systemd (`mysqld`,
  `mongod` without `--fork`, `valkey-server` without `--daemonize`), and cache
  commands that stream until interrupted (`MONITOR`, `SUBSCRIBE`, `--stat`). A
  server sent to the background with a trailing `&` is a CONFIRM instead of a
  BLOCK: the step returns, so the reason for refusing it is gone, and what is left
  is a server running outside systemd, where `systemctl status` will disagree with
  `ps` for the rest of the run. Three recorded blocks were that shape, two of them
  `mysqld --skip-grant-tables --skip-networking &` — the documented way back into a
  server whose root password is lost, and the one server systemd will not start for
  you. Put to an operator rather than refused: an unattended run with `--yes`
  proceeds, and the step is in the report either way. `&&` is a conditional and
  backgrounds nothing, so `mysqld && systemctl status mysql` still blocks. A
  daemon handed one job it ends after is not a start and runs:
  `mysqld --initialize-insecure` (the way out of a data directory too old to
  upgrade — a recorded run met `Cannot upgrade from 80046 to 90702` and was
  refused this step), `mysqld|mariadbd --validate-config`, which is how a bad
  setting gets found without restarting the service to discover it, and
  `--print-defaults`, which prints what the config files actually set and stops
  before the server starts — three recorded steps asked mysqld for it and were
  refused, two older ones asked mariadbd and were allowed only because this rule
  did not exist yet. A client asked for help is the same: `mysql --help` prints
  its options and the config files it read, then exits. Nor is a flag the only way
  one of them returns. A listing can be asked for by value —
  `mongod --setParameter help` prints the parameters the build accepts and stops,
  and a build that does not read `help` as a listing rejects it as a parameter name
  and stops too, so neither opens a port; two recorded steps piped it into `grep`
  looking for the knob behind a server that would not start, and both were refused
  a listing on the grounds that it was a start. `--outputConfig` is the same kind of
  question about the config rather than the parameters. What a set parameter is not
  is a listing: `mongod --setParameter tcmallocReleaseRate=5.0` is a server being
  started with a setting, and blocks. The program inside a command substitution is the one being
  run, so `missing=$(ldd "$PSM/bin/mysqld" | awk '/not found/{print $1}')` is an
  `ldd` and not a `mysqld` — read the other way round it was blocked twice in one
  run, refusing the ordinary way to find out which library a tarball build is
  missing, and the model's next step globbed the `bin` directory so that no command
  line would name the server binary at all.
  `LD_TRACE_LOADED_OBJECTS=1 /path/to/mysqld` is allowed for the same reason: it is
  what `/usr/bin/ldd` itself runs, and the loader prints the libraries the binary
  wants and exits before `main`, opening no port — while an empty value, or the
  same text anywhere but in front of the program, still reads as a start. What is
  final
  about a reinstall is the `rm`/`mv` of the data directory, which is classified on
  its own; `--initialize` refuses a directory that still has files in it. Also
  here: `su`/`runuser` with no command, which open an interactive shell as that
  user and sit there. The hang class matters more in practice: it is the
  difference between a model that gets a useful error and one that sits at a
  prompt until the timeout. Which is also why the reason a blocked server start
  gets back names two ways out, not one: the recorded steps that hit that rule were
  not trying to run a service by hand, they were trying to see why it would not
  start after systemctl had already failed, so the reason offers
  `timeout 20 mongod …` — bounded, so it returns on its own, and allowed.

Wrappers are stripped before classification and `bash -c` payloads are
classified recursively, so `sudo env … bash -c 'rm -rf /'` is not a way around
it. `runuser` and `setpriv` count as wrappers — a recorded run reached the
database account with `runuser -u mysql -- /usr/sbin/mariadbd --validate-config`,
and while they were unknown prefixes the program behind either of them was never
classified at all — and `su … -c '<script>'` is judged like `bash -c`, which is
what stops `su -c 'rm -rf /'` reading as one unrecognised program. It is a safety
net over known footguns, not a sandbox — the model is driving a root shell, so
read the plan.

Quoting is part of that. bash reads a `-c` string one command at a time, so a
quote left open on line 4 of a step runs lines 1 to 3 and only then fails, half
applying something you approved whole — and a text bash cannot parse is one the
guard cannot segment the way bash would either. So a command with an unclosed
quote is BLOCKed with an explanation instead of judged, and every step that gets
past the guard is parsed on the far end with `bash -n` before any of it runs. A
step with a syntax error costs a round trip and changes nothing.

That second pass earns its keep on more than quoting. A model that carries on past
the end of its own command leaves its reasoning inside it — one real step ended
`| tail -n 80It appears there is no PDMDB 8.0 package for Ubuntu 24.04 (noble).`,
with every quote closed and a bare `(` the shell will not have. The guard has no
shell parser and no business guessing at which words are English, so it passes the
step and `bash -n` refuses it.

## Scripts

Some work has no one-line form. A loop over three databases, a wait-for-it retry, a
check whose answer decides the next command — chained with `&&` across a screen's
width these are unreadable and unreviewable, and the recorded runs are full of them.
So a script is a step of its own:

```
THOUGHT: create both databases and the service user in one go
ACTION: script
INTERPRETER: bash
SCRIPT_BEGIN
#!/bin/bash
set -euo pipefail
for db in app logs; do
  mysql -e "CREATE DATABASE IF NOT EXISTS $db"
done
mysql -e "CREATE USER 'svc'@'localhost' IDENTIFIED BY '{{DBA_SECRET:mysql_svc}}'"
SCRIPT_END
```

The harness classifies the whole body, copies it to
`/tmp/dba-harness/step<NN>.sh` at mode 0700, parses it there without running it,
runs it by naming the interpreter, and hands back the exit code, stdout and stderr
exactly as for a command — plus the path, so a later step can read the file or fix
one line of it. `INTERPRETER:` is `bash` or `python3`, however the model spells it
(`sh`, `/bin/bash`, `python`, `python3.12`); with no such line the shebang decides,
and with neither it is bash. A third language is refused rather than guessed at,
because the guard has rules for those two and nothing to say about perl.

Five things follow from a script being one step rather than many:

- **It is judged whole, before any of it is copied.** One blocked line stops the
  whole script and the model is told which line — `line 3 of the script, 'mysql':
  bare mysql opens a client session`. Nothing above that line runs, which is the
  point: an interpreter reads a file as it goes, so a script judged line by line as
  it ran would half-apply a step that was approved whole. This is the same reason
  every command gets a `bash -n` pass, and scripts get one too (`py_compile` for
  python) before execution. A line means a logical line: a command split across
  physical lines with backslashes is joined back up first, because taken apart the
  halves are two commands the script never runs — seven recorded blocks were exactly
  that, `docker run -d \` … `mongod --config …` read as a bare mongod, and
  `gdb -batch … \` … `/usr/bin/mongod core` read as a server start when the program
  was gdb.
- **A heredoc body that lands in a file is judged as a file, not as shell.** The
  config rules read it — a `bind-address = 0.0.0.0` written by `cat > my.cnf <<EOF`
  is the same CONFIRM as one written by `write_file` — while the rules about what
  would hang this step step aside, because nothing here runs it, and because a
  unit's `ExecStart` and a wrapper script's `exec mongod` are *meant* to hold the
  foreground: that is how systemd supervises a service. Three recorded blocks were
  that, and one model wrote its wrapper with the comment "bypass the safety guard's
  detection of `mongod` in ExecStart" — a rule that teaches models to hide from it
  is worse than no rule. A heredoc with no file behind it (`mysql <<EOF`,
  `bash <<EOF`) is text a program will run, and stays a command. The cost of the
  trade is that commands written into a plain text file are no longer read as
  commands, which is already true of anything `write_file` puts there.
- **What you approve is the body.** A CONFIRM shows the script itself, not a summary
  of it, and the transcript and report keep it in full. Approving a description of a
  script is not approving the script.
- **Python is judged as python.** Read as shell, a python file is line after line of
  unknown program names — which is to say not judged at all. So it is parsed
  (`ast.parse`, which runs nothing) and its calls are mapped onto the rules that
  already exist: a string on its way to `os.system` or `subprocess.run` is
  classified as the command it is, `shutil.rmtree` and `os.remove` against the same
  path tables as `rm`, `shutil.move` and `os.rename` as `mv`, `open(path, 'w')` and
  `Path(path).write_text()` as a file write, and SQL handed to a driver's
  `execute()` as the SQL inside `mysql -e`. `eval`/`exec`/`compile`/`__import__` are
  BLOCKed — they run text the guard cannot read — and so is `input()`, which on a
  closed stdin raises `EOFError` on the spot rather than waiting. Python that does
  not parse is BLOCKed too: a script this cannot read is one whose danger it cannot
  assess. This also closed a hole that predated scripts, where a `.py` file written
  with `write_file` and run afterwards was never classified at all.
- **Placeholders still resolve on the way out only.** `{{DBA_SECRET:...}}` in a
  script body becomes a real password in the file on the server and nowhere else;
  the transcript, the report and the approval prompt all keep the placeholder. The
  file is chmod 0700 before the body is written to it, not after.

Two blind spots, both deliberate and both shared with shell:

- A value the parser cannot fold is not judged. `os.remove(target)`, where `target`
  was built ten lines up, passes — exactly as `rm -rf "$DATA"` does, for the same
  reason. Where a fragment *can* be read it is used: `f"rm -rf /var/lib/mysql/{db}"`
  still reads as a delete under the data directory.
- A whole command that cannot be read asks rather than passing quietly.
  `subprocess.run(cmd, shell=True)` is an instruction stream the guard cannot see,
  which is what `curl … | sh` is, and it gets the same CONFIRM. A partly-readable
  one does not: `subprocess.run(["mysql", "-e", sql])` goes straight through.

Nothing of the harness's own is injected into the body — no `pipefail`, no `set -e`.
What runs on the server is byte for byte what was judged, and what the model wants
of its shell it writes at the top itself.

## Passwords the model never sees

The model writes `{{DBA_SECRET:mysql_app}}` where a password belongs. The harness
generates a strong value on first use, substitutes it on the way to the server,
and substitutes it back out of everything on the way back — the model's own
context, the transcript, and the report
([do_dba/secrets.py](do_dba/secrets.py)). A weak password is therefore not
something the model can choose, and a real one is not something it can leak into
its own context. The real values are written once, to `secrets.json` in the run
directory, mode `0600`. Run directories are written to `output/` here, and
`.gitignore` excludes it — see
[Keep it private](#keep-it-private), which is the reason they are not in the
code project at all.

**And they stay on the servers, because the next run needs them.** Each server
keeps every credential of the run in `/etc/profile.d/dba-secrets.sh`, mode `0600`,
root only. A later run reads that file back before the model is asked anything,
lists the *names* it found on screen and in the prompt, and tells the model these
already work: write `{{DBA_SECRET:mysql_root}}` and it resolves to the value that
is really set, do not reset it. Nothing has to be passed on the command line, and
because the file sits in `/etc/profile.d` — every step already runs through
`bash -lc` — the values are in the environment of every command and of anyone who
logs in, so `mysql -p"$DBA_SECRET_MYSQL_ROOT"` works by hand too. New credentials
are pushed the moment they are generated, not at the end: a run that dies half way
through has still changed a password, and a password nobody can look up is the
whole problem.

That problem is on record. A run installed MySQL replication and set the root
password to a generated value; the next task against the same two servers opened
with a passwordless `mysql`, hunted through `/etc`, the error log, root's shell
history and cloud-init for a credential the harness had never written there, and
finally reset root through skip-grant-tables — ninety-one steps, and the
operator's note of the old password was silently wrong afterwards.

The trade is deliberate: the plaintext now sits on the machine it belongs to,
where a snapshot, a backup or a compromise of that machine exposes it, and one
file holds the whole fleet's credentials because a replication password belongs on
both ends. Root on that machine could already read the data these passwords
protect, which is what makes it a reasonable trade rather than a free one.
`--no-server-secrets` keeps them in the run directory only — and then a later run
is back to not knowing the password exists.

## "Done" is a claim, not a fact

When the model says the task is complete it must attach one `VERIFY:` command per
thing the task asked for — read-only, exiting non-zero if the work is not really
there. The harness runs them itself, along with its own standing checks (active
database services, listening sockets), and:

- if any check fails, the failures are handed back to the model and the run
  continues (twice, then the run ends as `unverified`) — and a check that failed
  without printing anything that could explain it is said to be that, rather than
  left as a bare exit code the model has to take apart itself;
- if the `done` step carries **no** check at all, it is handed back too. Nothing
  would have been re-run, so the claim would have been taken on trust — the one
  thing this harness will not do.

Statuses: `done`, `unverified` (finished but the checks disagree), `aborted` (the
model gave up and said why), `exhausted` (`--max-steps` or `--max-cost`),
`failed` (ten steps in a row failed, SSH dropped, or no readable step),
`api-error` (the gateway stopped serving the run — see below), `cancelled` (you
declined the plan or pressed Ctrl+C), `stuck` (three blocked steps in a row).
`done` exits 0; every other status exits non-zero, so a run is scriptable
without reading the report.

`api-error` is separate from `failed` because it is not the model's doing and not
a result to keep: a rate limit ended one benchmark run at step 2 of 120. A 429 is
waited out first — the gateway's `Retry-After` where it sends one, otherwise 5s,
15s, 30s, 60s — for up to two minutes per request in total, which
`--rate-limit-wait` sets, or `$DO_INFERENCE_RATE_LIMIT_WAIT` (`0` fails on the
first 429 instead). That budget is the *only* bound on the waiting: there is no
separate limit on the number of waits, so raising it always buys more patience —
`--rate-limit-wait 600` on a free tier that answers every other minute is a run
that finishes rather than one that ends at step 35. Only when the budget runs out
does the run end, and then it says what it waited and what the next wait would
have been.

A reply that is not a reply ends the same way. A gateway can answer 200 with a
body that is not JSON — OpenRouter pads a queued request with newlines, and one
run got 3.4 kB of padding with nothing after it — which is a `JSONDecodeError`
from inside the SDK rather than an API error, and used to end the run in a
traceback at step 3 of 100. It is asked again twice, 2s and 5s apart, and only
then reported, with what the gateway actually sent.

## What a run leaves behind

`output/<timestamp>-<host>/`, created on demand — `<host>` is the first server plus
a count when there is more than one:

| File | Contents |
| --- | --- |
| `report.md` | Task, the servers, model, the context budget it was run with, cost, each server as found, every step with the server it ran on, its command, exit code and output, and the independent verification |
| `transcript.jsonl` | One JSON line per event — steps, guard verdicts, approvals, verifications, and per reply its token counts, what it cost and where that figure came from — for after the fact |
| `secrets.json` | The generated credentials, mode `0600` (only if any were used) |

Both `report.md` and `transcript.jsonl` are redacted; the placeholders appear in
them, never the values. `--runs-dir PATH` moves the lot.

It also leaves something on the servers: `/etc/profile.d/dba-secrets.sh`, mode
`0600`, holding the same credentials as exports so the next run can log in with
them ([Passwords the model never sees](#passwords-the-model-never-sees)). Delete a
line from it and the next run generates a new value for that name;
`--no-server-secrets` writes nothing there at all.

## CLI flags

| Group | Flags |
| --- | --- |
| servers | `--host [NAME=][USER@]HOST[:PORT]` (required except with `--list-models`, repeatable; unnamed servers are labelled `node1`, `node2`, … and the model assigns the roles — see [More than one server](#more-than-one-server)), `-u/--user` (root), `-p/--port` (22), `-i/--key`, `--ask-key-passphrase`, `--ask-password`, `--no-server-secrets`, `--accept-host-key` |
| task | `--task ...`, `--task-file PATH`, `-m/--model`, `--provider {openrouter,digitalocean,selfhosted}` (openrouter), `--mode {plan,step,auto,unattended}`, `--dry-run`, `--yes`, `--probe` |
| limits | `--max-steps` (40), `--timeout` (300s per command), `--max-cost USD`, `--temperature` (0.2), `--effort {low,medium,high}` (off), `--rate-limit-wait SECONDS` (120, or `$DO_INFERENCE_RATE_LIMIT_WAIT`) |
| output | `--runs-dir` (`output/` in this project, or `$DBA_RUNS_DIR`), `--list-models`, `--no-color`, `--version` |

Cost is what the gateway says it charged, wherever it will say. Every OpenRouter
request asks for usage accounting, so the run's total is the billed figure —
cached prompt tokens, whichever upstream provider actually served the reply, and
any per-request fee already in it — and the report labels it *billed by the
gateway*. The transcript logs one `usage` event per reply carrying OpenRouter's
generation id, so a total can be taken apart and reconciled line by line with the
activity page. A gateway that reports nothing (DigitalOcean today) falls back to
tokens × the [pricing table](do_dba/inference/pricing.py), topped up with
whatever rates the gateway publishes about itself, and the report says *estimated
from published rates* — an estimate is never quietly added to a billed figure. A
self-hosted server sends no bill at all, so its replies are priced at zero and the
report says so instead of guessing.
Either way `--max-cost` stops the run when the spend reaches it. Command output
is truncated to 3000 characters per step before it goes into the prompt, and
older observations are trimmed, so a long run's prompt does not grow without
bound.

`--effort` asks the model to think before each step. It travels to the gateway as
one word and the gateway turns it into whatever the model upstream wants — a
token budget for Anthropic, a reasoning effort for OpenAI — which is why an
effort is portable where a per-provider knob is not. A model that cannot be asked
is not a failed run: the request is retried without the ask and the run says so.
Unset sends nothing at all, which is not the same as asking for none — a
reasoning model goes on reasoning at the gateway's own default for it, as every
run so far has. Thinking is billed as output, so a high effort can cost several
times a step that did none, and `--max-cost` is reached sooner. A long thinking
phase is also one silent request: the reply is not streamed, so raise
`DO_INFERENCE_TIMEOUT` (180s) if a model is cut off mid-thought.

## Notes on behaviour

- **Smaller models fail differently, not less.** A 20B model in testing kept
  inventing `CREATE ROLE IF NOT EXISTS` and `CREATE DATABASE` inside a `DO $$`
  block; PostgreSQL rejected each one, and once it had spent its run of
  consecutive failures the run stopped as `failed` rather than thrashing. That is
  the intended outcome — the harness's job is to make a model's mistakes visible
  and cheap, not to cover for them. Ten in a row is what it takes: fewer ends
  runs that were about to climb out of a bad patch, and an expired repository key
  or a package that has moved can cost several steps on its own. `--max-steps`
  and `--max-cost` are what bound a run; this only stops one going nowhere.
- **A cut-off reply is not a step.** When the service ends a reply at the output
  limit (`finish_reason: length`), a command chopped in half still parses as a
  command — `curl … | python3 -c "` is a real example from a real run. The
  harness checks for it, runs nothing, and asks for a smaller step.
- **The context window is a budget, and the harness spends it.** Where the gateway
  says how large a model's window is — hosted ones publish it in `/v1/models`, and a
  self-hosted LM Studio server on `/api/v0/models` — three limits are sized to it
  instead of being the same for every model:

  | window | chars per result | results kept whole | reply cap |
  | --- | --- | --- | --- |
  | 16K | 1,200 | 5K tokens | 2,048 |
  | 131K | 6,553 | 55K tokens | 16,384 |
  | 262K | 8,000 | 120K tokens | 16,384 |
  | 1M | 8,000 | 514K tokens | 16,384 |
  | not reported | 3,000 | the last 6 | the gateway's own |

  Three things follow. Results are shown at up to 8,000 characters rather than
  3,000, so a `SHOW REPLICA STATUS` arrives whole instead of costing another step to
  read the rest of. Old results are trimmed against a token budget rather than a
  count of six, which on any window above about 100K means a run never reaches it —
  the model still sees every result in full at step 40, where before it saw six and
  thirty-four 400-character stubs. And a reply is capped by the harness: measured
  over the 531 replies in [output/](output/) the median is 376 completion tokens and
  the largest with anything to say 13,391, but three ran to a gateway's own 65,536
  ceiling, cost 61% of their run and executed nothing — a reply cut off mid-command
  cannot be run, so it is thrown away and asked again. The model is told the cap in
  its own rules.

  Past 85% of the window the model's earlier replies are shortened too, oldest
  first, keeping the last two. They are the term that actually grows without bound:
  a reasoning model's scratchpad comes back inside the reply and stays in the
  conversation, and one recorded reply was 10,660 tokens of it. Trimming them is
  what a long run does instead of being refused by the gateway, which ends it.

  The budget a run was given is on the record: a `Context:` line in `report.md` and
  the same figures in the transcript's `run_started` event. Two runs of one task on
  one model are not comparable if one was shown 8,000 characters of each result and
  the other 3,000, and that is a difference the run directory would otherwise not
  mention.
- **Leaked end-of-turn markers are stripped.** Some models write the markers that
  were supposed to *end* the turn into the turn instead: `kimi-k3` finishes
  commands `… ; true<|close|>argument<|sep|>` and `deepseek-v4-flash` finishes
  them `… /etc/apt/sources.list.d/</antml>`. A shell reads either as a syntax
  error. The step in front of the leak is intact, so the harness cleans it, logs
  which markers it removed, and records them in the transcript rather than
  throwing the step away — a shell cannot parse a trailing marker under any
  reading, so removing one can only turn a step that could not run into the step
  the model meant. Markers inside a `CONTENT_BEGIN` block, or quoted inside a
  command, are left alone: there they are data, and a prompt template or an
  `</VirtualHost>` grep is allowed to contain them.
- **A reply that runs past the end of its own command is refused.** Past the end
  of its turn a model sometimes carries on writing, and what it writes lands in
  the command. Two shapes have been seen, both from `deepseek-v4-pro`: the
  harness's own result header, `| tail -n 5STEP 21 RESULT`, and the opening key of
  the next step, `| head -n 20THOUGHT: Clean up test/old PXC containers`. Unlike a
  trailing marker there is no telling where the command really ended — `-n 5` or
  `-n 50` — so the step is handed back to be sent again rather than trimmed into
  something plausible.

  The second shape is why this check sits in the parser and not only on the far
  end. `20THOUGHT:` is a perfectly good shell word, so `bash -n` has nothing to
  object to and the command runs: in the real step it removed three containers
  and only then failed on `head: invalid number of lines`, half-applying a step
  that had been approved as a whole. Only `THOUGHT:` and `ACTION:`, the two keys a
  step opens with, count as an overrun — the other keys would catch
  `echo "PATH: $PATH"` in an ordinary diagnostic, and no run has shown a step
  restarting at one of them.
- **A failure the model keeps misreading gets one line of explanation.** The
  observation is the shell's own output; the harness appends to it only where a
  recorded run shows the model unable to decode the diagnosis. There are three
  entries. `bad substitution` cost three steps in a row: given
  `docker run … bash -lc "… sed 's/[${}]//g' …"`, `deepseek-v4-pro` kept rewriting
  the backslashes and never spotted that its single quotes were *inside* double
  quotes, where they are literal characters and the outer shell tries to expand
  `${}` itself. `bash -n` cannot catch that class — parsing does not expand — so
  the hint is what shortens the loop.

  `Unknown command '\G'` is the second. `mysql -e "SHOW REPLICA STATUS\G"` appears
  in thirty-three steps across eight recorded runs, and which answer comes back
  depends on the client: every step where `\G` was honoured was an 8.x client, and
  all eight refusals came from 9.7.2. One run has both — the same command worked at
  step 8 against 8.4.11 and was refused at step 27 once that server had been
  upgraded — which is exactly why it reads as a quoting problem. It is not one: the
  wrap suite checks against a real shell that the backslash arrives exactly as
  written. One model re-sent the same statement in new quotes five times before
  trying `--vertical`, which worked first time and for the rest of the run; another
  simply gave up on `SHOW BINARY LOG STATUS`. The hint names `--vertical`, says a
  working `\G` elsewhere proves nothing about the quoting, and warns that `-N`
  drops the field names — `mysql -N -e "SHOW REPLICA STATUS\G"` prints the values
  bare, which cost one run two failed final checks that grepped for
  `Replica_IO_Running: Yes`, and five steps to work out why a healthy replica read
  as broken.

  `near 'MASTER STATUS'` is the third, and the same shape from the server side:
  8.4 renamed the statement to `SHOW BINARY LOG STATUS`, and error 1064 quotes only
  the fragment the parser choked on, which reads like a typo. Eight steps across
  four runs hit it, four of them re-sends of a statement already refused. One run
  sent it four times — semicolon added, then single quotes, then semicolon removed
  — thinking *"the semicolon might be causing shell parsing issues"* and then
  *"the semicolon is being stripped by the shell"*, and spent two more steps
  doubting the server was really MySQL before reaching the new name. That belief is
  what the hint is aimed at: a model that thinks the harness edits its SQL has no
  reason to trust any later step either. Nothing about the command changes.

  The needles are matched against **stdout as well as stderr**. 533 of the 935
  executed steps in the recorded runs — 57% — redirect with `2>&1`, which is the
  harness's own advice in rule 6, and that moves every diagnostic to stdout; only
  76 of the 935 produce any stderr at all. All eight `\G` refusals and all eight
  1064s arrived on stdout, so keyed on stderr neither entry would ever once have
  fired. The cost is a hint on a step that merely printed the phrase, a paragraph
  nobody needed; the alternative was a table blind on nine steps in ten.
- **A closed pipe is not a failed command.** `apt-cache search percona | head -30`
  does exactly what was asked and comes back exit 141: `head` stops reading, the
  shell kills the writer for it, and `set -o pipefail` makes that the pipeline's
  code. `curl` catches the write error itself and exits 23 —
  `curl -sSL …/InRelease | head -n 20`, complete output, `curl: (23) Failure
  writing output to destination`. Nine steps across six recorded runs ended this
  way and every one of them had worked. So when the exit code is 141 (or curl's
  23) and the command really does pipe into a truncating reader — `head`, or
  `grep` with `-q`/`-m`/`--max-count` — the harness says so in the observation,
  notes it on the step, and shows the operator `ok` rather than a red failure.
  It does not count toward the ten-failures-in-a-row stop, and it does not clear
  the count either: a run of `| head` steps cannot end a healthy run, and real
  failures either side of one still end a thrashing one. In a `VERIFY` the same
  code is a different problem — the writer's status is gone, so the check cannot
  say whether the work is there — so it is handed back as an *unusable check*
  with a request to send it again without the pipe, which leaves the run
  `unverified` rather than passing or failing it on nothing. `tail` never
  triggers this, since it has to read to the end to know what the end is, and a
  command that genuinely failed behind a `| head` comes back as that failure.
- **A filter that matched nothing is explained, not excused.** The other half of
  the same pipefail edge: `grep` exits 1 when no line matched, so
  `mysql -e "DROP TABLE …; DROP DATABASE …" 2>&1 | grep -v Warning` dropped both,
  printed only the password warning, had it filtered out, and came back exit 1
  with no output at all. Nine steps across seven runs look like that. It is the
  more dangerous shape, because the exit code is *also* what the command itself
  would report — so unlike a closed pipe it cannot be reclassified, and the
  harness says exactly that: exit 1 with no output from a step ending in a filter
  is either a failure or a filter that matched nothing, the filter removed
  whatever would say which, so check the state rather than assuming the work did
  not happen. The step stays a failure, still counts toward the ten-in-a-row
  stop, and the operator's line reads `nothing matched the filter` instead of a
  bare `exit 1`. Without that, models read the silence as the work not having
  happened: one decided "the defaults tool failed" when
  `my_print_defaults --mysqld | grep -E '^(bind-address|…)'` had simply missed the
  `--` that real output starts with, and another spent three steps undoing and
  redoing a replica it had already configured correctly. `grep -q` is left alone —
  there exit 1 is the answer to the question the model asked. Rule 6 now also
  tells the model not to write steps that end in a filter, and not to throw
  stderr away on a step it may have to debug.
- **An exit 0 is the last command's verdict, not the step's.** The third pipefail
  edge, and the only one that reads as good news: the step comes back 0 with a
  failure already behind it, the diagnosis on stderr and nothing else to say so.
  25 of the 1,293 executed steps now in the corpus — it has grown since the
  935-step figures above — came back that way, across nine runs. The worst is a
  repo file `dnf` would not parse: `Warning: failed loading
  '/etc/yum.repos.d/percona.repo', skipping.` on eight steps of two runs, six of
  them in one run that read every exit 0 and every `Nothing to do` as the
  repository being configured — it reinstalled `percona-release` twice, tried four
  package names, and ran out of steps having installed nothing. The next is an
  unterminated heredoc, which bash only *warns* about before ending the body at
  EOF: `cat > gr.cnf <<'EOF'` wrote a zero-length file, came back 0, and one run
  went on to configure a three-node cluster from empty config files — five steps
  across two runs, all of them now caught by the parser, which keeps the lines
  below a single-line `COMMAND:`, though a `COMMAND_BEGIN` block or a script can
  still do it. The singletons are the same shape: `awk: fatal: cannot open file
  /etc/mysql/debian.cnf`, `chown: … Operation not permitted` on a key file,
  `wget: invalid option -- 's'`, `Job for mysql.service failed`. So the harness
  says so: the observation quotes the line and says what the 0 does and does not
  cover, the operator's line carries it beside the exit code instead of a bare
  tick, and the step is recorded as `exited 0 with a failure on stderr`. An
  explanation, not a verdict — the step keeps the success it reported, because 0 is
  all the harness has to go on, so a run of them cannot end a run as `failed`
  either; and a step that failed outright gets nothing added, its exit code being
  the diagnosis already. The line has to be worth saying: 46 of the 71
  exit-0-with-stderr steps in the corpus say nothing of the sort — needrestart's
  `Running kernel seems to be up-to-date`, the client's password warning, `gpg`
  creating its keyring, `wget`'s progress line — and the failure-shaped lines that
  mean nothing are named as such, because `debconf: unable to initialize frontend`,
  `perl: warning: Setting locale failed` and curl's write error behind a `| head`
  sit on more steps than the real ones. Checks are left alone: none of the 324
  passing `VERIFY` checks in the corpus printed anything of the kind, so a check
  that passes is still a check that passes. Rule 7 now says it before the first
  step as well as after one has gone wrong.
- **A check that fails without saying why is told so.** Five of the seven failed
  `VERIFY` checks across the recorded runs printed nothing that could explain
  them, in three separate runs, and four discarded the answer themselves —
  `| grep -q '^1$'`, `test "$(…)" = "1"`. The worst shape collapses several facts
  into one boolean: `SELECT @@version LIKE '9.7.%' AND @@gtid_mode='ON' AND
  @@require_secure_transport='ON' AND @@server_id=1 | grep -q '^1$'` came back
  exit 1 on both servers of one run with the password warning as its only output,
  so the exit code could not say which of the four conjuncts was false. Everything
  it checked was in fact there; the fourth compared a *boolean* system variable to
  the string `'ON'`, and a boolean reads back as `1`. That run spent two steps
  splitting its own expression apart by hand — `SELECT
  @@require_secure_transport, @@require_secure_transport='ON'` — to find it. So
  when a check fails and nothing in its output could explain it, the harness says
  that, names the shape that kept the value where the command has one, and asks
  for a `run` step printing the values followed by one `VERIFY` line per fact. The
  password warning does not count as output, since it is on 59 of the recorded
  checks and explains none of them. It fires on neither a timeout (there the
  silence is the clock) nor the harness's own standing checks (not the model's to
  rewrite), and rule 10 now asks for checks that print what they looked at and
  names the boolean-variable trap up front.
- **Root or passwordless sudo is assumed.** The survey checks; an account with
  neither gets a warning and one chance to carry on before the run starts, since
  almost every step would otherwise fail on a `sudo` password prompt.
- **A question nobody can answer is answered no.** The model's commands run with
  stdin closed, so a prompt with no terminal on the other side would hang the run
  until the timeout. Where there is no terminal the harness says so and takes the
  refusing answer — a declined step, an untrusted host key — rather than waiting.
  `--mode unattended` is the way to say yes in advance instead; it changes who
  answers, not what the guard permits.
- **Idempotence is expected of the model, not enforced.** It is told the task may
  already be half done and to check before changing; a re-run on a finished
  server should end with checks and a `done`.
- **The task text is the specification.** Vague tasks produce vague `VERIFY`
  lines. "create a database called app with its own login user on each" gets
  checked per object; "set up the databases" gets whatever the model decides that
  meant.

## Layout

| File | Role |
| --- | --- |
| [pyproject.toml](pyproject.toml) | Project metadata, dependencies, the `do-dba` script |
| [dba.py](dba.py) | Entry point |
| [do_dba/agent.py](do_dba/agent.py) | The loop: system prompt, one step per reply, observations, independent verification, limits |
| [do_dba/protocol.py](do_dba/protocol.py) | The step format and its tolerant parser |
| [do_dba/fleet.py](do_dba/fleet.py) | The servers in a run: `--host` parsing, names, which one a step means, peer reachability |
| [do_dba/guard.py](do_dba/guard.py) | ALLOW / CONFIRM / BLOCK classification of commands, shell scripts, python scripts and file writes |
| [do_dba/ssh.py](do_dba/ssh.py) | SSH connection, host-key policy, command execution, script staging, file upload over SFTP |
| [do_dba/facts.py](do_dba/facts.py) | Read-only survey of the server before anything changes |
| [do_dba/secrets.py](do_dba/secrets.py) | `{{DBA_SECRET:name}}` generation, substitution, redaction, and the keeper file each server holds so the next run can log in |
| [do_dba/report.py](do_dba/report.py) | Run directory, `transcript.jsonl`, `report.md` |
| [do_dba/cli.py](do_dba/cli.py) | Flags, screens, approval prompts, plan gate |
| [do_dba/inference/](do_dba/inference/) | The gateways: client, model catalog, price table, credential discovery |
| [do_dba/term.py](do_dba/term.py) | Output encoding safety and ASCII glyph fallback |
| [run_tests.py](run_tests.py) | Runs every offline suite and prints one line each |
| [tests/](tests/) | `test_dba_*.py` — the suites, see [The tests](#the-tests) |
| [fake_droplet.py](fake_droplet.py) | A simulated Ubuntu droplet: apt, systemd, shell scripts with `if` and `for`, mysql, mariadb, postgres, mongodb, valkey, docker |
| [mock_do_server.py](mock_do_server.py) | A stand-in for `https://inference.do-ai.run/v1` |
| [probe_droplet.py](probe_droplet.py) | Checks the simulator against commands live models actually sent |
| [list_models.py](list_models.py) | What the current key can reach |
| `err429.py`, `stall.py` | Tiny servers for the rate-limit and stalled-stream paths |
| `output/<timestamp>-<host>/` | One recorded run each — see [The run corpus](#the-run-corpus) |
| `_scratch/` | Generated by the suites; delete it whenever |

## The tests

The code under test and the tests sit in one project, so there is nothing to point
`uv` at:

```bash
uv run python run_tests.py             # all nine
uv run python run_tests.py guard wrap  # by name
uv run python tests/test_dba_guard.py  # one directly
```

The suites live in [tests/](tests/) and find the harness by walking up one level,
so they can be run from anywhere; `run_tests.py` stays at the root.

`run_tests.py` starts `mock_do_server.py` for the one suite that needs it and
stops it afterwards. Everything it runs is offline: no key, no network, no server.

| Suite | What it covers |
| --- | --- |
| `guard` | Every guard verdict — commands, file bodies, shell scripts, python scripts — and every reply the parser has to survive, as a table of cases |
| `offline` | The whole loop against `fake_droplet.py` with a scripted model — statuses, verification, unreadable checks, a step that exits 0 with a failure on its stderr, secrets, protocol recovery, script steps, and the context budget: what each window size derives, what is trimmed and what is not, and the reply cap reaching the request |
| `prompts` | Who answers each y/n question — the operator, `--yes` or `--mode unattended` — and that answering them all still leaves the guard's blocks in force |
| `engines` | MariaDB, Valkey and MongoDB end to end — compatibility names, a runtime password that has to be rewritten to survive a restart, a vendor repository, authorization turned on after the fact |
| `fleet` | Several servers — `--host` forms, labelling the ones left unnamed, which server a `HOST:` line means, refusing a step that does not say, scoped checks, peer reachability, and two two-droplet MySQL replication runs judged from both ends |
| `secrets` | Credentials across runs — one name however it is spelled, the keeper file on the server and back off it, two servers disagreeing, and a second run that logs in with the first run's password without ever seeing it |
| `wrap` | `wrap_command` and the script pre-check handed to a real bash: syntax pre-flight, pipefail, quoting, heredocs, script paths |
| `providers` | Model id shapes, key discovery, the provider split, the price tiers, and the self-hosted detail endpoint — where it is asked, and that it fills fields in without overwriting any |
| `openrouter-wire` | The CLI end to end over real HTTP against a stub gateway |
| `selfhosted-wire` | The same against a stub LM Studio — a bare host completed to `/v1`, no credential sent, no prices reported, the context length and load state read from `/api/v0/models`, the listing surviving a 404 there, a model unloaded mid-run being asked for again, and a run whose cost line says so |
| `client` | The inference client against `mock_do_server.py` — replies, streaming, reasoning, usage, an effort that travels, a model that refuses two parameters one complaint at a time, a refused key |

`tests/test_dba_live.py` is not in the runner. It drives the harness with a real
model against the fake droplet — a real key and real money on a hosted gateway —
so it stays something you run on purpose:

```bash
uv run python tests/test_dba_live.py [model] [--task "..."] [--steps N]
uv run python tests/test_dba_live.py [model] --pair    # two droplets, replication
uv run python tests/test_dba_live.py qwen3.8-27b --provider local   # free
```

`--provider` takes any gateway from the table above and the model may be a
fragment of an id. On `local` the run costs nothing, which makes it the cheap way
to watch a model work a task through before paying a hosted one for the same run.

`--pair` is the multi-server case: two droplets on one private network, passed as a
bare list the way an operator passes them. The model has to pick which one becomes
the source, name the server on every step, configure the two differently, and end
with a replica the simulator agrees is actually receiving. The verdicts do not care
which of the two it picked — they read the roles back off the servers — only that it
picked one and stayed with it. It is the harder test by a distance, so it is opt-in
and given a larger step budget.

One caveat worth knowing before you trust a green line: a suite reports its own
verdict, and the runner only checks the exit code and greps the output for `FAIL`.
A suite that never asserts anything would pass.

## The run corpus

One directory per run of `dba.py`, named `<timestamp>-<host>`, under `output/`:

    output/20260821-012342-204.48.28.253/
      transcript.jsonl    every step, verdict, exit code and output, one JSON object per line
      report.md           the human-readable summary of the same run
      secrets.json        credentials the model generated, when the task created any

This is the regression corpus. Any change to the guard or the reply parser gets
replayed against every step recorded here before it counts as done — these are the
only real examples of what the models actually send, including the malformed ones.

```bash
# from this directory
uv run python - <<'PY'
import json
from pathlib import Path
from do_dba import guard
for t in sorted(Path("output").glob("*/transcript.jsonl")):
    steps = [json.loads(l) for l in t.read_text(encoding="utf-8").splitlines()]
    runs = [s for s in steps if s.get("kind") == "step" and s.get("action") == "run"]
    drift = [s for s in runs if guard.classify(s["detail"]).level != s["verdict"]]
    print(f"{t.parent.name}  {len(runs):>3} run steps  {len(drift)} verdicts changed")
PY
```

Read the files with `encoding="utf-8"` — real command output is full of box
drawing and accented characters, and the default codec on this machine fails on
them. That applies to writing, too: anything printing what a transcript contains
needs `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`.

As of 2026-08-22: 30 runs and 986 step records — 896 `run`, 65 `write_file`, 25
`done` — plus 43 protocol errors the harness recovered from. The expected replay
drift is 11 guard verdicts, all of them deliberate: see
[Notes on behaviour](#notes-on-behaviour) above. Several are steps that really did
execute at the time, which is the point of keeping the corpus.

The corpus predates `ACTION: script`, so it says nothing about that path. What it
does say is that adding it moved nothing else: replayed against the guard as it
stood before, all 896 `run` steps and 65 `write_file` steps come back with the same
verdict *and the same reason*, and all 986 step records re-parse identically. The
recorded runs are also why the script action exists — the corpus is full of steps
chaining five commands with `&&` because there was no other way to say it.

## Keep it private

`transcript.jsonl` is a complete record of what was done to a real server, and
three of these runs carry a `secrets.json` of generated database passwords.
Secrets substituted into commands are redacted in the transcript, but the
`secrets.json` files are not — they are the actual credentials.

The servers hold the same credentials, in `/etc/profile.d/dba-secrets.sh` at mode
`0600` — nothing to keep out of version control, but it does mean a snapshot or a
filesystem backup of a managed server carries its database passwords. Anyone with
root there could read the data those passwords protect anyway; if that is not the
threat model, `--no-server-secrets` turns it off and the passwords stay in the run
directory only.

`.gitignore` here excludes `output/`, timestamped run directories wherever
`--runs-dir` puts them, `.env` and `_scratch/`,
so the code in this project can go into version control without the records going
with it. Check that rule before you push, and before putting this directory in a
synced folder.
