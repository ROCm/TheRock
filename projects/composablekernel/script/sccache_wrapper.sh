#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set -e
COMPILERS_HASH_DIR=${COMPILERS_HASH_DIR:-"/tmp/.sccache"}
SCCACHE_EXTRAFILES=${SCCACHE_EXTRAFILES:-"${COMPILERS_HASH_DIR}/rocm_compilers_hash_file"}
SCCACHE_BIN=${SCCACHE_BIN:-"${SCCACHE_INSTALL_LOCATION}/sccache"}
ENFORCE_REDIS="false"
while [ "$1" != "" ];
do
    case $1 in
        --enforce_redis )
            shift; ENFORCE_REDIS="true" ;;
        --no-hipcc )
            shift ;;
        *)
            break ;;
    esac
done
setup_rocm_compilers_hash_file() {
    mkdir -p "$COMPILERS_HASH_DIR"
    HIPCC_MD5="$(md5sum "${ROCM_PATH}/bin/hipcc")"
    pushd "${ROCM_PATH}/amdgcn/bitcode"
        DEVICELIBS_BITCODES_MD5="$(find . -type f -exec md5sum {} \; | sort | md5sum)"
    popd
    HIPCC_HASH_VALUE="${HIPCC_MD5%% *}"
    DEVICELIBS_BITCODES_HASH_VALUE="${DEVICELIBS_BITCODES_MD5%% *}"
    # MD5 checksums of clang and clang-offload-bundler cannot be used since they will keep changing
    # if the ROCM_PATH changes, ie; for every mainline build.
    # This is because ROCM_PATH gets encoded into the clang/clang-offload-bundler binaries as part
    # of RPATH.
    # The versions themselves contain the commit hash of the compiler repo at the time of building.
    # Hence, this should be a viable alternative to using the binary checksum itself.
    CLANG_VERSION="$("${ROCM_PATH}/llvm/bin/clang" --version | head -n 1)"
    CLANG_OFFLOAD_BUNDLER_VERSION="$("${ROCM_PATH}/llvm/bin/clang-offload-bundler" --version | head -n 1)"
    printf '%s: %s\n' 'clang version' "${CLANG_VERSION}" | tee -a "$SCCACHE_EXTRAFILES"
    printf '%s: %s\n' 'clang-offload-bundler version' "${CLANG_OFFLOAD_BUNDLER_VERSION}" | tee -a "$SCCACHE_EXTRAFILES"
    printf '%s: %s\n' 'hipcc md5sum' "${HIPCC_HASH_VALUE}" | tee -a "$SCCACHE_EXTRAFILES"
    printf '%s: %s\n' 'devicelibs bitcode md5sum' "${DEVICELIBS_BITCODES_HASH_VALUE}" | tee -a "$SCCACHE_EXTRAFILES"
    echo "sccache-wrapper: compilers hash file set up at ${SCCACHE_EXTRAFILES}"
    cat "$SCCACHE_EXTRAFILES"
}
if [ "${ENFORCE_REDIS}" == "true" ]; then
    if [ -z "${SCCACHE_REDIS}" ]; then
        echo "SCCACHE_REDIS not set. Not wrapping compilers with sccache."
        exit 10
    else
        response=$(redis-cli -u ${SCCACHE_REDIS} ping) || true
        if [ "${response}" != "PONG" ]; then
            echo "Redis server unreachable. Not wrapping compilers with sccache."
            exit 20
        fi
    fi
fi
setup_rocm_compilers_hash_file
$SCCACHE_BIN --version
echo "=== Starting sccache server at $(date) ==="
$SCCACHE_BIN --start-server

# Log initial sccache statistics
echo "=== Initial sccache statistics ==="
$SCCACHE_BIN --show-stats || echo "Could not get initial stats"

# Test Redis connectivity and performance
echo "=== Testing Redis connectivity ==="
start_time=$(date +%s%N)
redis-cli -u ${SCCACHE_REDIS} ping || echo "Redis ping failed"
end_time=$(date +%s%N)
latency=$(( (end_time - start_time) / 1000000 ))
echo "Redis ping latency: ${latency}ms"

# Check Redis memory status
echo "=== Redis memory status ==="
redis-cli -u ${SCCACHE_REDIS} info memory | grep -E "(used_memory|maxmemory|evicted_keys)" || echo "Could not get Redis memory info"

