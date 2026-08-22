#!/usr/bin/env python3
"""Entry point: run a DBA task on a server with a DigitalOcean-hosted model."""

import sys

from do_dba.cli import main

if __name__ == "__main__":
    sys.exit(main())
