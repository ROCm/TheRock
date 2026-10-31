#!/usr/bin/env bash
set -euo pipefail
export CCACHE_DISABLE=1
export THEROCK_PERF_COMMAND_DIR="$PWD/perf-commands"
mkdir -p "$THEROCK_PERF_COMMAND_DIR" perf-records
git rev-parse HEAD > perf-records/source-sha.txt
test "$(git rev-parse HEAD)" = 5b7cb88373362ab80dc0f19605915e534f4158aa
python build_tools/fetch_sources.py --stage compiler-runtime --jobs 12 --depth 1
python build_tools/fetch_sources.py --stage math-libs --jobs 12 --depth 1
git submodule status --recursive > perf-records/submodule-shas.txt
python build_tools/configure_stage.py --stage math-libs --artifacts solver --amdgpu-families gfx950-dcgpu --dist-amdgpu-families gfx950-dcgpu --manylinux --output-cmake-args perf-records/cmake-args.txt
mapfile -t stage_args < perf-records/cmake-args.txt
for i in "${!stage_args[@]}"; do stage_args[$i]="${stage_args[$i]//\"/}"; done
launcher="python3;$PWD/build_tools/perf_probe/command_trace.py"
cmake -B build -S . -GNinja --preset linux-release-asan-debug \
  -DTHEROCK_PACKAGE_VERSION=10.3.0.perf01a19119 \
  -DTHEROCK_BACKGROUND_BUILD_JOBS=9 -DLLVM_PARALLEL_LINK_JOBS=4 \
  -DCMAKE_C_COMPILER_LAUNCHER="$launcher" \
  -DCMAKE_CXX_COMPILER_LAUNCHER="$launcher" \
  -DCMAKE_HIP_COMPILER_LAUNCHER="$launcher" \
  "${stage_args[@]}"
python3 build_tools/perf_probe/collect_build_perf.py --output perf-bootstrap --build-dir build --interval 5 -- \
  cmake --build build --target rocSOLVER+configure -- -k 0 -d keeprsp
python3 build_tools/perf_probe/collect_build_perf.py --output perf-rocsolver --build-dir build --interval 1 -- \
  cmake --build build --target rocSOLVER+stage -- -k 0 -d keeprsp
git diff --submodule=short > perf-records/source-diff.txt
find build/math-libs/BLAS/rocSOLVER -name '*.so*' -type f -exec sha256sum '{}' \; > perf-records/solver-artifact-sha256.txt
