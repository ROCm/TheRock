#!/usr/bin/env bash
set -euo pipefail

AMDGPU_FAMILIES="${1:-gfx942}"
LOG_DIR="${2:-build/logs}"
INDEXER="${3:-build/indexer.py}"

echo "[index_logs.sh] Indexing logs in ${LOG_DIR}"

python "${INDEXER}" -f '*.log' "${LOG_DIR}"

INDEX_HTML="${LOG_DIR}/index.html"
if [[ -f "${INDEX_HTML}" ]]; then
  sed -i "s,a href=\"..\",a href=\"../../index-${AMDGPU_FAMILIES}.html\",g" "${INDEX_HTML}"
  echo "[index_logs.sh] Patched links in ${INDEX_HTML}"
fi