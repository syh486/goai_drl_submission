#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if (( $# < 1 )); then
  echo "Usage: $0 USER@AGX:DESTINATION [RSYNC_OPTION ...]" >&2
  exit 2
fi
DESTINATION="$1"
shift

echo "Syncing deployment workspace to ${DESTINATION}"
rsync -az --info=progress2 \
  --exclude='.git/' \
  --exclude='build/' \
  --exclude='install/' \
  --exclude='log/' \
  --exclude='logs/' \
  --exclude='**/__pycache__/' \
  "$@" "${ROOT}/" "${DESTINATION}"

echo "AGX_SYNC_OK"
