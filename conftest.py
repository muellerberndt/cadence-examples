"""The repository root on the import path, so tests import agent, arm and world."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: a behavioural test that runs a learning life for minutes")
