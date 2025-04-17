#!/bin/bash
set -e
set -o pipefail
trap 'kill -TERM 0' INT

set -o xtrace
pip install -r /therock/src/requirements.txt
time python /therock/src/build_tools/linux_python_package.py \
  --artifact-dir /therock/artifacts \
  --dest-dir /therock/output \
  "$@"
