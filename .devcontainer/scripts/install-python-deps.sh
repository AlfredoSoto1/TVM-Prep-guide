#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REQUIREMENTS="${1:-${REPO_ROOT}/requirements.txt}"
SITE_PACKAGES="$(python -m site --user-site)"
USER_BASE="$(python -m site --user-base)"

rm -rf "${SITE_PACKAGES}"
rm -rf "${USER_BASE}/share/jupyter/kernels/tvm-prep"

python -m pip install --upgrade pip setuptools wheel -r "${REQUIREMENTS}"

mkdir -p "${SITE_PACKAGES}"
printf '%s\n' "${REPO_ROOT}/examples/python" > "${SITE_PACKAGES}/tvm-prep-guide.pth"
python -m ipykernel install --user --name tvm-prep --display-name "Python (TVM Prep)"
