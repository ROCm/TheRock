#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -euo pipefail

cmake_args=(-DTHEROCK_ENABLE_ALL=OFF -DLLVM_HOST_TRIPLE=x86_64-unknown-linux-gnu)
if [[ -n "$AMDGPU_TARGETS" ]]; then
  cmake_args+=("-DTHEROCK_AMDGPU_TARGETS=$AMDGPU_TARGETS"
    "-DTHEROCK_DIST_AMDGPU_TARGETS=$AMDGPU_TARGETS"
    "-DTHEROCK_TEST_AMDGPU_TARGETS=$AMDGPU_TARGETS"
    "-DTHEROCK_AMDGPU_DIST_BUNDLE_NAME=$PROFILE")
else
  cmake_args+=('-DTHEROCK_DIST_AMDGPU_FAMILIES=dgpu-all;dcgpu-all;igpu-all;gpu-generic')
fi
compiler="$PWD/build/compiler/amd-llvm/dist/lib/llvm/bin"

case "$1" in
  compiler)
    python build_tools/artifact_manager.py fetch \
      --run-id "$BASELINE_RUN_ID" --run-github-repo ROCm/TheRock \
      --stage runtime-tests --generic-only --output-dir build --bootstrap \
      --exclude-components dbg,doc,test \
      --exclude-artifacts core-amdsmi,core-hip,core-hipinfo,core-hiptests,core-hrr,core-ocl,core-ocl-icd,core-runtime,sysdeps-hwloc,sysdeps-libmnl,sysdeps-libnl,sysdeps-libpciaccess
    test ! -e build/core/rocm-runtimes/stage.prebuilt
    test ! -e build/core/hip-tests/stage.prebuilt
    cmake -S . -B build -GNinja "${cmake_args[@]}" \
      -DTHEROCK_ENABLE_BASE=ON -DTHEROCK_ENABLE_COMPILER=ON
    cmake --build build --parallel 8 --target amd-llvm
    "$compiler/ld.lld" --version
    ;;
  runtime)
    if [[ "$RUN_HIP" == true ]]; then
      cmake_args+=(-DTHEROCK_ENABLE_CORE_RUNTIME=ON
        -DTHEROCK_ENABLE_HIP_RUNTIME=ON -DTHEROCK_ENABLE_CORE_HIPTESTS=ON
        -DTHEROCK_FLAG_USE_ROCDDI_FOR_HSA_RUNTIME=ON)
    fi
    cmake -S . -B build -GNinja "${cmake_args[@]}" \
      -DCMAKE_BUILD_TYPE=Release -DTHEROCK_ENABLE_ROCDDI_RUNTIMES=ON \
      -DTHEROCK_BUILD_TESTING=ON \
      "-DROCM_RUNTIMES_CARGO_HOME=$CARGO_HOME" \
      -DROCM_RUNTIMES_CARGO_JOBS=8 \
      "-DROCM_RUNTIMES_LLD=$compiler/ld.lld"
    cmake --build build --parallel 8 --target rocm-runtimes
    python - <<'PY'
import json, os, pathlib, subprocess

def revision(*args):
    return subprocess.check_output(["git", *args, "rev-parse", "HEAD"], text=True).strip()

actual = revision("-C", "rocm-systems")
expected = json.loads(os.environ["EXTERNAL_REPO_CONFIG"])["ref"]
if actual != expected:
    raise SystemExit(f"Runtime source changed: expected {expected}, got {actual}")
record = {
    "therock_commit": revision(),
    "rocm_systems_ref": os.environ["ROCM_SYSTEMS_REF"],
    "rocm_systems_commit": actual,
    "baseline_run_id": os.environ["BASELINE_RUN_ID"],
    "amdgpu_family": os.environ["AMDGPU_FAMILY"],
    "amdgpu_targets": os.environ["AMDGPU_TARGETS"],
    "test_runs_on": json.loads(os.environ["TEST_RUNS_ON"]),
    "llvm_version": subprocess.check_output(
        ["build/compiler/amd-llvm/dist/lib/llvm/bin/clang", "--version"], text=True
    ).strip(),
    "hip_test_label": os.environ["HIP_TEST_LABEL"],
    "hip_test_filter": os.environ["HIP_TEST_FILTER"],
    "hip_test_level": os.environ["HIP_TEST_LEVEL"],
}
pathlib.Path("SOURCE.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PY
    ;;
  cts)
    cmake -S rocm-systems/runtimes/api-headers -B build/cts-headers -GNinja
    cmake --install build/cts-headers --prefix "$PWD/build/cts-header-prefix"
    cmake -S rocm-systems/runtimes/testing -B build/cts -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      '-DROCM_RUNTIMES_TEST_FAMILIES=runtime-api;ddi;kfd' \
      "-DROCmRuntimeAPI_DIR=$PWD/build/cts-header-prefix/lib/cmake/ROCmRuntimeAPI" \
      "-DCMAKE_C_COMPILER=$compiler/clang" \
      "-DROCM_RUNTIMES_CARGO_HOME=$CARGO_HOME"
    cmake --build build/cts --parallel 8
    cmake --install build/cts --prefix "$PWD/build/cts-prefix"
    mkdir -p build/cts-runtime/lib
    cp -a build/core/rocm-runtimes/build/lib/*.so* build/cts-runtime/lib/
    tar -czf rocddi-cts.tar.gz SOURCE.json -C build cts-prefix cts-runtime
    ;;
  hip)
    cmake --build build --parallel 8 --target hip-tests
    test ! -e build/core/ROCR-Runtime
    test ! -e build/core/hsakmt
    cmp build/core/rocm-runtimes/build/lib/libhsa-runtime64.so.1 \
      build/core/hip-tests/dist/lib/libhsa-runtime64.so.1
    ctest --test-dir build -R therock-validate-shared-lib-libhsa-runtime64 \
      --output-on-failure --no-tests=error
    (cd build/core/hip-tests/dist && sha256sum lib/libhsa-runtime64.so.1) > HSA.sha256
    tar -czf rocddi-hip-tests.tar.gz -C build/core/hip-tests/dist . \
      -C "$PWD" SOURCE.json HSA.sha256
    ;;
  host-tests)
    cmake --build build --parallel 8 --target build-test-rocm-runtimes
    ;;
  *)
    echo "Unknown build step: $1" >&2
    exit 1
    ;;
esac
