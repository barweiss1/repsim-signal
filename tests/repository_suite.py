"""Automatic unittest suite for the repository's maintained test modules."""

from __future__ import annotations

import unittest
from pathlib import Path


EXCLUDED_MODULES = set()


def load_tests(
    loader: unittest.TestLoader,
    standard_tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del standard_tests, pattern
    suite = unittest.TestSuite()
    tests_dir = Path(__file__).resolve().parent
    for path in sorted(tests_dir.glob("test_*.py")):
        if path.stem in EXCLUDED_MODULES:
            continue
        suite.addTests(loader.loadTestsFromName(f"tests.{path.stem}"))
    return suite
