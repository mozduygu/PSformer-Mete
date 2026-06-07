#!/usr/bin/env bash
# Create a Python virtual environment named 'psformer-env' and install all
# dependencies needed to run the PSformer reproduction (training + inference).
#
# Usage:
#   bash scripts/setup_env.sh
#
# Then activate it (see the message printed at the end):
#   source psformer-env/bin/activate
#
# Requires Python 3.10+ (the code uses `X | None` type-hint syntax).

set -euo pipefail

ENV_NAME="psformer-env"

# Resolve repo root (parent of this script's directory) so the env is created
# at the top level regardless of where the script is called from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Pick a Python interpreter (prefer python3).
PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  PY="python"
fi

# Verify Python >= 3.10.
"$PY" - <<'PYEOF'
import sys
if sys.version_info < (3, 10):
    sys.exit(
        f"Python 3.10+ is required, found {sys.version.split()[0]}.\n"
        "Install a newer Python or set PYTHON=/path/to/python3.10 and re-run."
    )
print(f"Using Python {sys.version.split()[0]}")
PYEOF

# Create the venv (skip if it already exists).
if [ -d "$ENV_NAME" ]; then
  echo ">> '$ENV_NAME' already exists -- reusing it."
else
  echo ">> Creating virtual environment '$ENV_NAME'..."
  "$PY" -m venv "$ENV_NAME"
fi

# Install dependencies into the env without needing to 'activate' first.
echo ">> Upgrading pip and installing dependencies..."
"$ENV_NAME/bin/python" -m pip install --upgrade pip
"$ENV_NAME/bin/python" -m pip install -r requirements.txt

echo
echo "========================================================================"
echo "Done. Environment '$ENV_NAME' is ready."
echo
echo "Activate it with:"
echo "    source $ENV_NAME/bin/activate"
echo
echo "Then verify the install with the parameter-count check:"
echo "    python -m experiments.test_param_count"
echo
echo "Deactivate later with:  deactivate"
echo "========================================================================"
