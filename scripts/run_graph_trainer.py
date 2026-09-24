#!/usr/bin/env python
"""Thin wrapper around ReSi's graphs.graph_trainer CLI.

ReSi's graphs/gnn.py imports torcheval.metrics.functional.multiclass_accuracy.
torcheval is deliberately excluded from the pinned NGC container build (see
requirements-resi-container-ngc24.06.txt and
scripts/build_resi_enroot_container.sh, which explicitly uninstalls it)
because installing it can pull in a torch version that conflicts with the
NGC-pinned stack. manifold_repsim already ships a pure-torch stub for the one
function ReSi needs -- see
src/manifold_repsim/resi/runtime.py::_install_torcheval_fallback and the note
in docs/resi_cluster.md -- but it's normally only installed as a side effect of
loading ReSi's measure registry through the campaign task runner. This script
installs it directly, then hands off to graphs.graph_trainer's own CLI
unchanged, so ad hoc training runs (like PGNN-on-Cora) work the same way
outside the campaign workflow.

Usage: identical to `python -m graphs.graph_trainer`, e.g.
  python scripts/run_graph_trainer.py -a PGNN -d Cora -s 0 1 2 3 4
"""
import runpy
import sys

from manifold_repsim.resi.runtime import _install_torcheval_fallback

_install_torcheval_fallback()
sys.argv[0] = "graph_trainer.py"
runpy.run_module("graphs.graph_trainer", run_name="__main__")
