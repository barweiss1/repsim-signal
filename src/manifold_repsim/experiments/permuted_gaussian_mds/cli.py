"""Command-line entry point for the synthetic permuted-Gaussian MDS workflow."""

from __future__ import annotations

import argparse

from .config import load_config
from .workflow import run


def main(argv: list[str] | None = None) -> None:
    """Load one YAML configuration and run its selected stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Versioned YAML experiment configuration")
    run(load_config(parser.parse_args(argv).config))
