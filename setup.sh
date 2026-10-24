#!/bin/bash
# Create (or update) the neural_hjb conda environment and install this repo into it.
set -euo pipefail
cd "$(dirname "$0")"
ENV_NAME=neural_hjb

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda env '$ENV_NAME' already exists; updating it."
else
    conda create -n "$ENV_NAME" python=3.11 -y
fi
conda run -n "$ENV_NAME" pip install -e ".[dev]"

echo "
Setup complete. Next steps:
1. Activate env: conda activate $ENV_NAME
2. Run the tests: pytest
3. Reproduce the results: ./run.sh
"
