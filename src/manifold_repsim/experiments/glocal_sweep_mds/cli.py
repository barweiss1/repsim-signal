"""Command-line interface for the glocal sweep MDS experiment."""

from __future__ import annotations

import argparse

from .config import load_config
from .workflow import run


def main(argv: list[str] | None = None) -> None:
    """Run one versioned YAML experiment configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Versioned glocal sweep MDS YAML configuration")
    run(load_config(parser.parse_args(argv).config))
