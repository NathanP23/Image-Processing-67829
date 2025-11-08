 #!/usr/bin/env bash
 set -euo pipefail

 cd "/cs/usr/$(whoami)"

 ENV_NAME="IMPR_ex1_venv"
 VENV_DIR="${PWD}/${ENV_NAME}"
 VENV_PY="${VENV_DIR}/bin/python"

 echo "Creating venv at ${PWD}"
 umask 022
 python3 -m venv "${ENV_NAME}"

 echo "Upgrading pip inside the venv (no system pip)"
 "${VENV_PY}" -m pip install -q --upgrade pip --no-cache-dir


 echo "venv ready."
 echo
 echo "To activate in bash/zsh:"
 echo "  source ${VENV_DIR}/bin/activate"