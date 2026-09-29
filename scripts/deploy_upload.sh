#!/usr/bin/env bash
set -euo pipefail

# 禁止 macOS 在压缩包中写入 ._* AppleDouble 元数据文件。
export COPYFILE_DISABLE=1

# PROJECT_NAME="ocean-subsurface-recon-b4"
# SERVER_USER="${SERVER_USER:-root}"
# SERVER_HOST="${SERVER_HOST:-connect.bjb2.seetacloud.com}"
# SERVER_PORT="${SERVER_PORT:-23910}"
# REMOTE_PARENT_DIR="${REMOTE_PARENT_DIR:-/root}"

PROJECT_NAME="ocean-subsurface-recon-b4"
SERVER_USER="${SERVER_USER:-root}"
SERVER_HOST="${SERVER_HOST:-connect.bjb1.seetacloud.com}"
SERVER_PORT="${SERVER_PORT:-50490}"
REMOTE_PARENT_DIR="${REMOTE_PARENT_DIR:-/root}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_NAME="${PROJECT_NAME}_${DEPLOY_TIMESTAMP}.tar.gz"
ARCHIVE_PATH="/tmp/${ARCHIVE_NAME}"
REMOTE_ARCHIVE_PATH="${REMOTE_PARENT_DIR}/${ARCHIVE_NAME}"
REMOTE_PROJECT_DIR="${REMOTE_PARENT_DIR}/${PROJECT_NAME}"
REMOTE_STAGING_DIR="${REMOTE_PARENT_DIR}/.${PROJECT_NAME}_staging_${DEPLOY_TIMESTAMP}"
REMOTE_BACKUP_DIR="${REMOTE_PROJECT_DIR}_backup_${DEPLOY_TIMESTAMP}"

cleanup_local_archive() {
  rm -f "${ARCHIVE_PATH}"
}
trap cleanup_local_archive EXIT

for command_name in tar scp ssh; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Required command was not found: ${command_name}" >&2
    exit 1
  fi
done

if [ ! -f "${PROJECT_DIR}/train.py" ] || \
   [ ! -f "${PROJECT_DIR}/test.py" ] || \
   [ ! -f "${PROJECT_DIR}/models/b4_cplus/model.py" ] || \
   [ ! -f "${PROJECT_DIR}/models/b4_cplus/decoder.py" ] || \
   [ ! -f "${PROJECT_DIR}/models/b4_cplus/embeddings.py" ]; then
  echo "Project root is invalid: ${PROJECT_DIR}" >&2
  exit 1
fi

case "${REMOTE_PARENT_DIR}" in
  /*) ;;
  *)
    echo "REMOTE_PARENT_DIR must be an absolute path: ${REMOTE_PARENT_DIR}" >&2
    exit 2
    ;;
esac

case "${REMOTE_PARENT_DIR}" in
  *"'"*)
    echo "REMOTE_PARENT_DIR must not contain a single quote" >&2
    exit 2
    ;;
esac

echo "Local project:  ${PROJECT_DIR}"
echo "Local archive:  ${ARCHIVE_PATH}"
echo "Remote target:  ${SERVER_USER}@${SERVER_HOST}:${REMOTE_PROJECT_DIR}"
echo "Remote backup:  ${REMOTE_BACKUP_DIR}"
echo

cd "${PROJECT_DIR}"

tar \
  --exclude='.git' \
  --exclude='.conda' \
  --exclude='.vscode' \
  --exclude='.idea' \
  --exclude='.env' \
  --exclude='.env.*' \
  --exclude='*.pem' \
  --exclude='*.key' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='.DS_Store' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='*.tar.gz' \
  --exclude='data' \
  --exclude='checkpoints' \
  --exclude='outputs' \
  --exclude='caches' \
  --exclude='runs' \
  --exclude='wandb' \
  --exclude='logs' \
  --exclude='docs' \
  --exclude='reports' \
  --exclude='._*' \
  -czf "${ARCHIVE_PATH}" .

echo "Archive created. Uploading..."
scp -P "${SERVER_PORT}" \
  "${ARCHIVE_PATH}" \
  "${SERVER_USER}@${SERVER_HOST}:${REMOTE_ARCHIVE_PATH}"

echo "Upload finished. Deploying with backup..."
ssh -p "${SERVER_PORT}" "${SERVER_USER}@${SERVER_HOST}" \
  bash -s -- \
  "${REMOTE_ARCHIVE_PATH}" \
  "${REMOTE_PROJECT_DIR}" \
  "${REMOTE_STAGING_DIR}" \
  "${REMOTE_BACKUP_DIR}" <<'REMOTE_DEPLOY'
set -euo pipefail

archive_path="$1"
project_dir="$2"
staging_dir="$3"
backup_dir="$4"
backup_created=0

cleanup_staging() {
  if [ -d "${staging_dir}" ]; then
    rm -rf -- "${staging_dir}"
  fi
}

rollback_project() {
  if [ "${backup_created}" -eq 1 ] && [ ! -e "${project_dir}" ] && [ -d "${backup_dir}" ]; then
    mv -- "${backup_dir}" "${project_dir}"
  fi
}

trap 'status=$?; cleanup_staging; if [ "${status}" -ne 0 ]; then rollback_project; fi; exit "${status}"' EXIT

if [ ! -f "${archive_path}" ]; then
  echo "Remote archive was not found: ${archive_path}" >&2
  exit 1
fi
if [ -e "${staging_dir}" ] || [ -e "${backup_dir}" ]; then
  echo "Timestamp collision on the server; refusing to overwrite staging or backup" >&2
  exit 1
fi

mkdir -p -- "${staging_dir}"
tar -xzf "${archive_path}" -C "${staging_dir}"

if [ ! -f "${staging_dir}/train.py" ] || \
   [ ! -f "${staging_dir}/test.py" ] || \
   [ ! -f "${staging_dir}/models/b4_cplus/model.py" ] || \
   [ ! -f "${staging_dir}/models/b4_cplus/decoder.py" ] || \
   [ ! -f "${staging_dir}/models/b4_cplus/embeddings.py" ]; then
  echo "Extracted project validation failed; current deployment was not changed" >&2
  exit 1
fi

if [ -e "${project_dir}" ]; then
  mv -- "${project_dir}" "${backup_dir}"
  backup_created=1
fi

mv -- "${staging_dir}" "${project_dir}"
echo "Deployment complete: ${project_dir}"
if [ "${backup_created}" -eq 1 ]; then
  echo "Previous version backup: ${backup_dir}"
else
  echo "Previous version backup: none (first deployment)"
fi
echo "Uploaded archive retained at: ${archive_path}"
REMOTE_DEPLOY

echo
echo "B4 deployment completed successfully."
echo "Server project: ${REMOTE_PROJECT_DIR}"
echo "Server backup:  ${REMOTE_BACKUP_DIR} (when a previous version existed)"
echo "Server archive: ${REMOTE_ARCHIVE_PATH}"
