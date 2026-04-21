#!/bin/bash
# Entrypoint: initialize Spack and activate the rocm environment before
# running any command passed to the container.

set -e

. "${SPACK_ROOT}/share/spack/setup-env.sh"
spack env activate "${SPACK_ENV_NAME}"

exec "$@"
