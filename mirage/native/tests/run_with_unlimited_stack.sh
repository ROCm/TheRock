#!/usr/bin/env bash
set -euo pipefail

ulimit -s unlimited
exec "$@"
