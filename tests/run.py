#!/usr/bin/env python3
"""Run the AI Passport Hermes plugin test suite.

    python3 clients/hermes-passport/tests/run.py [-v]

Standard library only, so it runs under whatever interpreter the machine has and
inside a sealed Hermes venv. When a Hermes install is present (HERMES_AGENT_DIR
or ~/.hermes/hermes-agent) the suite runs against the REAL MemoryProvider
contract; otherwise it stubs the host and still passes.
"""

from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent


def main() -> int:
    sys.path.insert(0, str(TESTS_DIR))
    import harness

    where = harness.hermes_agent_dir()
    print(f"host contract: {'real Hermes at ' + str(where) if where else 'stubbed (no Hermes install found)'}")

    # The provider logs a warning per failure class by design, and the failure
    # paths are most of what this suite exercises. Keep the run readable; pass
    # --logs to see them.
    if "--logs" not in sys.argv:
        logging.disable(logging.CRITICAL)

    verbosity = 2 if "-v" in sys.argv else 1
    suite = unittest.defaultTestLoader.discover(str(TESTS_DIR), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
