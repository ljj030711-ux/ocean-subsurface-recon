#!/usr/bin/env bash
set -euo pipefail

# Upload local NetCDF files to the AutoDL shared filesystem.
#
# Usage:
#   ./scripts/upload_nc_to_autodl_fs.sh SST /Users/lijunjie/Documents/上大硕士/ocean-subsurface-recon/data/inversion_data_depth_all_size/SSS
#   ./scripts/upload_nc_to_autodl_fs.sh SSH /Users/lijunjie/Documents/上大硕士/ocean-subsurface-recon/data/inversion_data_depth_all_size/SSH
#   ./scripts/upload_nc_to_autodl_fs.sh SSS /Users/lijunjie/Documents/上大硕士/ocean-subsurface-recon/data/inversion_data_depth_all_size/SST
#
# Environment overrides:
#   SERVER_USER=root
#   SERVER_HOST=connect.bjb1.seetacloud.com
#   SERVER_PORT=50490
#   REMOTE_NC_ROOT=/root/autodl-fs/raw_all_size/nc

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 {SSH|SST|SSS} /path/to/local_nc_dir" >&2
  exit 2
fi

VAR_NAME="$1"
LOCAL_DIR="$2"

case "${VAR_NAME}" in
  SSH|SST|SSS) ;;
  *)
    echo "VAR_NAME must be one of: SSH, SST, SSS" >&2
    exit 2
    ;;
esac

if [ ! -d "${LOCAL_DIR}" ]; then
  echo "Local directory does not exist: ${LOCAL_DIR}" >&2
  exit 1
fi

SERVER_USER="${SERVER_USER:-root}"
SERVER_HOST="${SERVER_HOST:-connect.bjb1.seetacloud.com}"
SERVER_PORT="${SERVER_PORT:-50490}"
REMOTE_NC_ROOT="${REMOTE_NC_ROOT:-/root/autodl-fs/raw_all_size/nc}"
REMOTE_DIR="${REMOTE_NC_ROOT}/${VAR_NAME}"

echo "Local:  ${LOCAL_DIR}/"
echo "Remote: ${SERVER_USER}@${SERVER_HOST}:${REMOTE_DIR}/"
echo

ssh -p "${SERVER_PORT}" "${SERVER_USER}@${SERVER_HOST}" \
  "mkdir -p '${REMOTE_DIR}'"

rsync -avhP \
  --include='*/' \
  --include='*.nc' \
  --exclude='*' \
  -e "ssh -p ${SERVER_PORT}" \
  "${LOCAL_DIR%/}/" \
  "${SERVER_USER}@${SERVER_HOST}:${REMOTE_DIR}/"

echo
echo "Upload finished."
