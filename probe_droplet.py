"""Ad-hoc probe: check simulator fidelity on the commands live models actually sent."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Ahead of anything installed on purpose: the point is to test this tree.
sys.path.insert(0, str(HERE))

from fake_droplet import FakeDroplet

d = FakeDroplet()
d.run("apt-get update")
d.run("apt-get install -y mysql-server postgresql")

# Verbatim shapes taken from live gpt-oss-20b runs.
STEP4 = '''# MySQL
mysql -e "\\
CREATE DATABASE IF NOT EXISTS \\`app\\`; \\
CREATE USER IF NOT EXISTS 'app'@'localhost' IDENTIFIED WITH mysql_native_password BY 'pw1'; \\
GRANT ALL PRIVILEGES ON \\`app\\`.* TO 'app'@'localhost'; \\
FLUSH PRIVILEGES;"
# PostgreSQL
sudo -u postgres psql -c "\\
DO \\$\\$ BEGIN \\
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='app') THEN \\
    CREATE ROLE app WITH LOGIN PASSWORD 'pw2'; \\
  END IF; \\
END \\$\\$;"
sudo -u postgres createdb -O app app'''

PIPED_SQL = """echo "SHOW DATABASES LIKE 'app';" | mysql -u root | tail -n 20"""

MULTI_E = ('mysql -e "CREATE DATABASE IF NOT EXISTS two;" '
           '-e "CREATE USER IF NOT EXISTS \'two\'@\'localhost\' IDENTIFIED BY \'pw3\';"')

# From the run where the fixture, not the model, was wrong.
NESTED_SH = ('sudo -u postgres bash -c \'psql -c "CREATE ROLE appuser WITH LOGIN PASSWORD '
             "'\\''pw4'\\''\"; psql -c \"CREATE DATABASE appdb OWNER appuser;\"'")

IF_BLOCK = """if ! systemctl is-active --quiet postgresql; then
  systemctl start postgresql
fi
mysql -e "CREATE DATABASE IF NOT EXISTS guarded;\""""

IF_TAKEN = """if systemctl is-active --quiet postgresql; then
  mysql -e "CREATE DATABASE IF NOT EXISTS branch_taken;"
else
  mysql -e "CREATE DATABASE IF NOT EXISTS branch_skipped;"
fi"""

ONE_LINE_IF = 'if [ -f /etc/os-release ]; then mysql -e "CREATE DATABASE IF NOT EXISTS inline;"; fi'

for label, command in [
    ("backslash-continued SQL in one string", STEP4),
    ("sudo -u postgres bash -c with two psql calls", NESTED_SH),
    ("if guard whose condition is already satisfied", IF_BLOCK),
    ("if/else picks one branch", IF_TAKEN),
    ("one-line if", ONE_LINE_IF),
    ("echo piped into mysql", PIPED_SQL),
    ("repeated -e flags", MULTI_E),
    ("app user can connect", "mysql -u app -p'pw1' -e 'SELECT 1'"),
    ("pg role can connect", "PGPASSWORD='pw2' psql -U app -d app -c 'SELECT current_database()'"),
    ("mysql users", "mysql -e \"SELECT user FROM mysql.user\""),
]:
    r = d.run(command)
    print(f"--- {label}\n  exit {r.exit_code} | out={r.stdout.strip()!r} err={r.stderr.strip()[:120]!r}")
print("\nunhandled:", d.unhandled)
print(d.state())
