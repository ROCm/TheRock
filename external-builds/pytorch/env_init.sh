#!/usr/bin/bash

# Causes bash process to die immediately after child process returns error
# to make sure that script does not continue logic if error has happened.
set -e
set -o pipefail

SCRIPT_DIR="$(cd $(dirname $0) && pwd)"
if [ -n "${VIRTUAL_ENV}" ]; then
    echo "virtual env set"
    VENV_DIR_ENV_VARIABLE=$(realpath ${VIRTUAL_ENV})
    THEROCK_VENV_DEF_DIR=$(realpath $SCRIPT_DIR/../../.venv)

    echo "VENV_DIR_ENV_VARIABLE: ${VENV_DIR_ENV_VARIABLE}"
    echo "THEROCK_VENV_DEF_DIR: ${THEROCK_VENV_DEF_DIR}"
    if [ "${VENV_DIR_ENV_VARIABLE}" == "${THEROCK_VENV_DEF_DIR}" ];
    then
        echo "python virtual env: ${VIRTUAL_ENV}"
    else
        echo "Python VIRTUAL_ENV: ${VIRTUAL_ENV}"
        echo "THEROCK default Python virtual env not active: ${THEROCK_VENV_DEF_DIR}"
    fi
else
    if [ -f $SCRIPT_DIR/../../.venv/bin/activate ]; then
        source $SCRIPT_DIR/../../.venv/bin/activate
        echo "sourced python VIRTUAL_ENV: ${VIRTUAL_ENV}"
    else
        echo "ROCK python virtual env not available"
    fi
fi

if [ ! -n "${THEROCK_PATH_SET}" ]; then
    export THEROCK_PATH_SET=1
    if [ ! -n "${ROCM_HOME}" ]; then
        export ROCM_HOME=$(realpath $SCRIPT_DIR/../../build/dist/rocm)
    fi
    if [ -d ${ROCM_HOME} ]; then
        export PATH=${ROCM_HOME}/bin:$PATH
        export LD_LIBRARY_PATH=${ROCM_HOME}/lib
        echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
    else
        echo "Could not find ROCM_HOME: $ROCM_HOME"
        exit 1
    fi
fi
