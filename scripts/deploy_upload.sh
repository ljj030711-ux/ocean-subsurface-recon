#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="ocean-subsurface-recon"
SERVER_USER="${SERVER_USER:-root}"
SERVER_HOST="${SERVER_HOST:-connect.bjb2.seetacloud.com}"
SERVER_PORT="${SERVER_PORT:-35533}"
REMOTE_PARENT_DIR="${REMOTE_PARENT_DIR:-/root}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARCHIVE_PATH="/tmp/${PROJECT_NAME}.tar.gz"
REMOTE_PROJECT_DIR="${REMOTE_PARENT_DIR}/${PROJECT_NAME}"

echo "Project: ${PROJECT_DIR}"
echo "Archive: ${ARCHIVE_PATH}"
echo "Remote archive: ${SERVER_USER}@${SERVER_HOST}:${REMOTE_PARENT_DIR}/"
echo

cd "${PROJECT_DIR}"

tar \
  --exclude='.git' \
  --exclude='.conda' \
  --exclude='.vscode' \
  --exclude='.idea' \
  --exclude='__pycache__' \
  --exclude='.DS_Store' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='*.tar.gz' \
  --exclude='data' \
  --exclude='checkpoints' \
  --exclude='outputs' \
  --exclude='runs' \
  --exclude='wandb' \
  --exclude='logs' \
  -czf "${ARCHIVE_PATH}" .

scp -P "${SERVER_PORT}" "${ARCHIVE_PATH}" "${SERVER_USER}@${SERVER_HOST}:${REMOTE_PARENT_DIR}/"

echo
echo "Upload finished."
echo
echo "Recommended unpack commands on the server:"
echo "ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST}"
echo "if [ -d '${REMOTE_PROJECT_DIR}' ]; then mv '${REMOTE_PROJECT_DIR}' '${REMOTE_PROJECT_DIR}_backup_'\"\$(date +%Y%m%d_%H%M%S)\"; fi"
echo "mkdir -p '${REMOTE_PROJECT_DIR}'"
echo "tar -xzf '${REMOTE_PARENT_DIR}/${PROJECT_NAME}.tar.gz' -C '${REMOTE_PROJECT_DIR}'"
echo
echo "One-command remote unpack:"
echo "ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} \"if [ -d '${REMOTE_PROJECT_DIR}' ]; then mv '${REMOTE_PROJECT_DIR}' '${REMOTE_PROJECT_DIR}_backup_\\\$(date +%Y%m%d_%H%M%S)'; fi; mkdir -p '${REMOTE_PROJECT_DIR}'; tar -xzf '${REMOTE_PARENT_DIR}/${PROJECT_NAME}.tar.gz' -C '${REMOTE_PROJECT_DIR}'\""
