#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
test "${GITHUB_REPOSITORY:-}" = ROCm/TheRock
test "$(git rev-parse HEAD)" = 5b7cb88373362ab80dc0f19605915e534f4158aa
export CCACHE_DISABLE=1 TEATIME_FORCE_INTERACTIVE=0
tools="$PWD/build_tools/perf_probe"
records="$PWD/perf-records"
project="$PWD/build/math-libs/BLAS/rocSOLVER"
mkdir -p "$records"

configure_top() {
  local mode="$1" solver_args=""
  if test "$mode" = candidate; then
    solver_args='-DROCSOLVER_SHARD_GETF2_SMALL:BOOL=ON;-DROCSOLVER_SPLIT_GETRI_Z:BOOL=ON'
  fi
  mapfile -t stage_args < "$records/cmake-args.txt"
  for i in "${!stage_args[@]}"; do stage_args[$i]="${stage_args[$i]//\"/}"; done
  local launcher="python3;$tools/command_trace.py"
  cmake -B build -S . -GNinja --preset linux-release-asan-debug \
    -DTHEROCK_PACKAGE_VERSION=10.3.0.perf01a19119 \
    -DTHEROCK_BACKGROUND_BUILD_JOBS=9 -DLLVM_PARALLEL_LINK_JOBS=4 \
    "-Drocprofiler-sdk_CMAKE_ARGS=-DCODEOBJ_HIP_PATH=$PWD/build/core/clr/dist;-DCODEOBJ_HIP_DEVICE_LIB_PATH=$PWD/build/core/clr/dist/lib/llvm/amdgcn/bitcode" \
    "-DrocSOLVER_CMAKE_ARGS=$solver_args" \
    "-DCMAKE_C_COMPILER_LAUNCHER=$launcher" "-DCMAKE_CXX_COMPILER_LAUNCHER=$launcher" \
    "-DCMAKE_HIP_COMPILER_LAUNCHER=$launcher" "${stage_args[@]}"
}

configure_solver() {
  # Clear only this child's cache and configure stamp, outside measured trials.
  python3 - "$project" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1]).resolve()
assert p.as_posix().endswith('/build/math-libs/BLAS/rocSOLVER')
for rel in ('stamp/configure.stamp', 'build/CMakeCache.txt'):
    (p / rel).unlink(missing_ok=True)
PY
  cmake --build build --target rocSOLVER+configure -- -k 0 -d keeprsp
}

restore_base() {
  if test -f "$records/joint-active"; then
    git -C rocm-libraries apply --reverse --check "$tools/rocsolver-specialized-shards.patch"
    git -C rocm-libraries apply --reverse "$tools/rocsolver-specialized-shards.patch"
    rm "$records/joint-active"
  fi
  test "$(git -C rocm-libraries rev-parse HEAD)" = f4e24d58b3f313f1a31068def1e35376e43c2f49
  git -C rocm-libraries diff --exit-code
  test -z "$(git -C rocm-libraries ls-files --others --exclude-standard)"
  sha256sum --check "$records/original-specialized-source.sha256"
}

case "${1:?Specify prepare, baseline, candidate, compare or probes}" in
prepare)
  export THEROCK_PERF_COMMAND_DIR="$PWD/perf-commands-bootstrap"
  mkdir -p "$THEROCK_PERF_COMMAND_DIR"
  python3 "$tools/phase_probes.py" preflight --output "$records/allocation-prepare.json"
  git rev-parse HEAD > "$records/source-sha.txt"
  printf '%s\n' "${PERF_TRIAL_ORDER:?Missing pair order}" > "$records/trial-order.txt"
  python build_tools/fetch_sources.py --stage compiler-runtime --jobs 12 --depth 1
  python build_tools/fetch_sources.py --stage math-libs --jobs 12 --depth 1
  test "$(git -C rocm-libraries rev-parse HEAD)" = f4e24d58b3f313f1a31068def1e35376e43c2f49
  git -C rocm-libraries diff --exit-code
  test -z "$(git -C rocm-libraries ls-files --others --exclude-standard)"
  git -C rocm-systems apply --check "$tools/bootstrap-codeobj.patch"
  git -C rocm-systems apply "$tools/bootstrap-codeobj.patch"
  git -C rocm-systems diff > "$records/bootstrap-source.patch"
  git submodule status --recursive > "$records/submodule-shas.txt"
  sha256sum "$tools/rocsolver-specialized-shards.patch" > "$records/candidate-patch-sha256.txt"
  sha256sum rocm-libraries/projects/rocsolver/library/src/CMakeLists.txt \
    rocm-libraries/projects/rocsolver/library/src/specialized/roclapack_getf2_specialized_kernels.hpp \
    rocm-libraries/projects/rocsolver/library/src/specialized/roclapack_getf2_small_c.cpp \
    rocm-libraries/projects/rocsolver/library/src/specialized/roclapack_getri_specialized_kernels.hpp \
    rocm-libraries/projects/rocsolver/library/src/specialized/roclapack_getri_specialized_kernels_z.cpp \
    > "$records/original-specialized-source.sha256"
  python build_tools/configure_stage.py --stage math-libs --artifacts solver --amdgpu-families gfx950-dcgpu \
    --dist-amdgpu-families gfx950-dcgpu --manylinux --output-cmake-args "$records/cmake-args.txt"
  configure_top baseline
  python3 "$tools/collect_build_perf.py" --output perf-bootstrap-sdk-configure --build-dir build --interval 5 -- \
    cmake --build build --target rocprofiler-sdk+configure -- -k 0 -d keeprsp
  test -f "$PWD/build/core/clr/dist/include/hip/hip_runtime.h"
  test -d "$PWD/build/core/clr/dist/lib/llvm/amdgcn/bitcode"
  python3 "$tools/collect_build_perf.py" --output perf-bootstrap-fixture --build-dir build --interval 1 -- \
    cmake --build build/profiler/rocprofiler-sdk/build --target source/lib/tests/codeobj/syncthreads_kernel.bin -- -v -d keeprsp
  mkdir -p "$project/build"
  for name in getri_z getf2_c; do
    expected=getri-specialized
    test "$name" != getf2_c || expected=getf2-small
    cmake -E env --unset=ROCM_PATH --unset=ROCM_DIR --unset=HIP_PATH --unset=HIP_DIR -- \
      python3 "$tools/trace_getf2.py" --record "$tools/records/$name.json" --source-root "$PWD" \
      --expected-source "$expected" --output "$PWD/perf-capability/$name" --trace --device-link-trace --check-only
  done
  python3 "$tools/collect_build_perf.py" --output perf-bootstrap --build-dir build --interval 5 -- \
    cmake --build build --target rocSOLVER+configure -- -k 0 -d keeprsp
  # Establish the original input set before either trial can apply the patch.
  python3 "$tools/phase_probes.py" prime --output "$records" --label prepare
  ;;
baseline|candidate)
  mode="$1"
  export THEROCK_PERF_COMMAND_DIR="$PWD/perf-commands-$mode"
  mkdir -p "$THEROCK_PERF_COMMAND_DIR"
  # Clean the PREVIOUS graph too: C->B must remove candidate-only helper objects.
  python3 "$tools/phase_probes.py" reset --output "$records" --label "before-$mode-source-switch"
  restore_base
  if test "$mode" = candidate; then
    git -C rocm-libraries apply --check "$tools/rocsolver-specialized-shards.patch"
    git -C rocm-libraries apply "$tools/rocsolver-specialized-shards.patch"
    touch "$records/joint-active"
    cp "$tools/rocsolver-specialized-shards.patch" "$records/candidate-applied.patch"
    git -C rocm-libraries status --porcelain > "$records/candidate-status.txt"
  fi
  configure_top "$mode"
  configure_solver
  if test "$mode" = candidate; then
    grep -qx 'ROCSOLVER_SHARD_GETF2_SMALL:BOOL=ON' "$project/build/CMakeCache.txt"
    grep -qx 'ROCSOLVER_SPLIT_GETRI_Z:BOOL=ON' "$project/build/CMakeCache.txt"
  else
    ! grep -Eq '^ROCSOLVER_(SHARD_GETF2_SMALL|SPLIT_GETRI_Z):[^=]+=ON$' "$project/build/CMakeCache.txt"
  fi
  python3 "$tools/phase_probes.py" preflight --output "$records/allocation-$mode.json"
  python3 "$tools/phase_probes.py" reset --output "$records" --label "$mode"
  python3 "$tools/phase_probes.py" prime --output "$records" --label "$mode"
  python3 "$tools/collect_build_perf.py" --output "perf-$mode" --build-dir build --interval 1 -- \
    cmake --build build --target rocSOLVER+stage -- -k 0 -d keeprsp
  python3 "$tools/phase_probes.py" inventory --output "$records" --label "$mode"
  ;;
compare)
  python3 "$tools/phase_probes.py" compare --output "$records"
  ;;
probes)
  restore_base
  configure_top baseline
  configure_solver
  git -C rocm-libraries status --porcelain > "$records/libraries-status-before-probes.txt"
  python3 "$tools/phase_probes.py" prime --output "$records" --label phase-probes
  python3 "$tools/collect_build_perf.py" --output perf-phase-observer --build-dir build --interval 1 -- \
    python3 "$tools/phase_probes.py" probes --output "$PWD/perf-phase-probes"
  ;;
*) echo "Unknown phase: $1" >&2; exit 2 ;;
esac
