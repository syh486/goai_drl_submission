#!/usr/bin/env bash
set -eo pipefail

if (( $# != 2 )); then
  echo "Usage: $0 SESSION_DIR USER@AGX:DESTINATION_ROOT" >&2
  exit 2
fi

SESSION_DIR="$(realpath "$1")"
DESTINATION="$2"
for path in \
  "${SESSION_DIR}/route_rebound.yaml" \
  "${SESSION_DIR}/route.anchor.npz" \
  "${SESSION_DIR}/route.quality.json" \
  "${SESSION_DIR}/topometric_map/localization_map_manifest.json"; do
  [[ -e "${path}" ]] || { echo "Deployment bundle input is missing: ${path}" >&2; exit 2; }
done

ASSETS=(
  "${SESSION_DIR}/route_rebound.yaml"
  "${SESSION_DIR}/route.anchor.npz"
  "${SESSION_DIR}/route.quality.json"
  "${SESSION_DIR}/topometric_map"
)
for optional in \
  "${SESSION_DIR}/route.observations" \
  "${SESSION_DIR}/route_rebound_validation.json" \
  "${SESSION_DIR}/collection_validation.json"; do
  [[ ! -e "${optional}" ]] || ASSETS+=("${optional}")
done

REMOTE_SESSION="${DESTINATION%/}/$(basename "${SESSION_DIR}")/"
echo "Syncing deployment bundle to ${REMOTE_SESSION}"
rsync -az --mkpath --info=progress2 "${ASSETS[@]}" "${REMOTE_SESSION}"
echo "DEPLOYMENT_BUNDLE_SYNC_OK=$(basename "${SESSION_DIR}")"
