"""An SSH harness that lets a hosted model do DBA work on a server.

You give it a host, a credential, and a task in plain English ("install MySQL
and PostgreSQL"). The model gets no shell of its own: it proposes one step at a
time, a safety guard judges each one, the harness runs the survivors over SSH and
hands back the output. Every step is logged, generated credentials never reach
the model, and the result is checked by the harness rather than taken on trust.
"""

from pathlib import Path

__version__ = "1.0.0"

# This project's own root: do_dba/__init__.py -> do_dba -> the project directory.
# The .env we read, the pricing.json that overrides the built-in rates and the
# runs we write are all resolved from here rather than from the working
# directory, so dba.py behaves the same whatever directory it is invoked from -
# which matters for a tool whose output is the record of what it did to a live
# server.
PROJECT_DIR = Path(__file__).resolve().parent.parent
