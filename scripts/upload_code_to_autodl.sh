#!/usr/bin/env bash
set -euo pipefail

# Upload the current B1 source tree directly to AutoDL with rsync.
# GitHub is not involved, and local data/checkpoints/outputs are not uploaded.
#
# Usage:
#   ./scripts/upload_code_to_autodl.sh
#   ./scripts/upload_code_to_autodl.sh /root/autodl-tmp/my-project-dir
#
# Environment overrides:
#   SERVER_USER=root
#   SERVER_HOST=connect.bjb1.seetacloud.com
#   SERVER_PORT=50490
#   REMOTE_CODE_DIR=/root/autodl-tmp/ocean-subsurface-recon-b1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SERVER_USER="${SERVER_USER:-root}"
SERVER_HOST="${SERVER_HOST:-connect.bjb1.seetacloud.com}"
SERVER_PORT="${SERVER_PORT:-50490}"
REMOTE_CODE_DIR="${1:-${REMOTE_CODE_DIR:-/root/autodl-tmp/ocean-subsurface-recon-b1}}"

if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [remote_code_dir]" >&2
  exit 2
fi

case "${REMOTE_CODE_DIR}" in
  /*) ;;
  *)
    echo "Remote code directory must be an absolute path: ${REMOTE_CODE_DIR}" >&2
    exit 2
    ;;
esac

case "${REMOTE_CODE_DIR}" in
  *"'"*)
    echo "Remote code directory must not contain a single quote" >&2
    exit 2
    ;;
esac

if [ ! -f "${PROJECT_ROOT}/train.py" ] || [ ! -f "${PROJECT_ROOT}/test.py" ]; then
  echo "Project root is invalid: ${PROJECT_ROOT}" >&2
  exit 1
fi

if ! command -v ssh >/dev/null 2>&1; then
  echo "ssh is required but was not found" >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required but was not found" >&2
  exit 1
fi

echo "Local code:  ${PROJECT_ROOT}/"
echo "Remote code: ${SERVER_USER}@${SERVER_HOST}:${REMOTE_CODE_DIR}/"
echo "Excluded: .git, data, checkpoints, outputs, caches"
echo

ssh -p "${SERVER_PORT}" "${SERVER_USER}@${SERVER_HOST}" \
  "mkdir -p '${REMOTE_CODE_DIR}'"

rsync -avhP \
  --exclude='.git' \
  --exclude='.DS_Store' \
  --exclude='.env' \
  --exclude='.env.*' \
  --exclude='.pytest_cache/' \
  --exclude='__pycache__/' \
  --exclude='*.py[cod]' \
  --exclude='*.pem' \
  --exclude='*.key' \
  --exclude='data/' \
  --exclude='checkpoints/' \
  --exclude='outputs/' \
  -e "ssh -p ${SERVER_PORT}" \
  "${PROJECT_ROOT%/}/" \
  "${SERVER_USER}@${SERVER_HOST}:${REMOTE_CODE_DIR}/"

echo
echo "Code upload finished."
echo "Server directory: ${REMOTE_CODE_DIR}"
