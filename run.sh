#!/bin/bash
# Reproduce the README results (all but the v1.0 "Original" rows): train, then evaluate and plot.
# Uses the active Python (activate the env from ./setup.sh first) unless PYTHON is set.
# Existing checkpoints are reused; pass --force to retrain them.
#
#   ./run.sh                     full method + ablations, 3 seeds (about 1 hour on 8 cores)
#   VARIANTS=full SEEDS=0 ./run.sh   the method alone, one seed
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
VARIANTS=${VARIANTS:-full no_hamiltonian no_hjb_grad no_hjb}
SEEDS=${SEEDS:-0 1 2}

echo "== Motion control =="
$PYTHON -m motion_control.train --dims 2 4 6 8 --variants $VARIANTS --seeds $SEEDS "$@"
$PYTHON -m motion_control.evaluate

echo "== Resource allocation =="
$PYTHON -m resource_allocation.train --variants $VARIANTS --seeds $SEEDS "$@"
$PYTHON -m resource_allocation.evaluate

echo "Results: motion_control/results/ and resource_allocation/results/"
