#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

rocm_root="${ROCM_ROOT:-${repo_root}/build-macos-egpu/dist/rocm}"
build_dir="${BUILD_DIR:-${repo_root}/build-macos-egpu/validation/math_smoke/build}"
blas_build_dir="${BLAS_BUILD_DIR:-${repo_root}/build-macos-egpu/validation/blas_smoke/build}"
queue_index="${ROCR_MACOS_DIRECT_QUEUE_INDEX:-}"

export ROCM_PATH="${ROCM_PATH:-${rocm_root}}"
export HIP_PATH="${HIP_PATH:-${rocm_root}}"
export DYLD_LIBRARY_PATH="${rocm_root}/lib:${rocm_root}/lib/llvm/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"

export AMD_GPU_MACOS_FORCE_DIRECT_COMPUTE="${AMD_GPU_MACOS_FORCE_DIRECT_COMPUTE:-1}"
if [[ -n "${queue_index}" ]]; then
  export ROCR_MACOS_DIRECT_QUEUE_INDEX="${queue_index}"
  direct_queue_desc="${ROCR_MACOS_DIRECT_QUEUE_INDEX}"
else
  unset ROCR_MACOS_DIRECT_QUEUE_INDEX
  direct_queue_desc="auto"
fi
export ROCR_MACOS_HOST_BLIT_ONLY="${ROCR_MACOS_HOST_BLIT_ONLY:-1}"
# Temporary validation discipline: avoid dequeue/reuse between short-lived
# smoke processes. This consumes one HQD per process but avoids recycling a
# queue before the direct PM4 completion path is proven to be a true EOP fence.
export ROCR_MACOS_DIRECT_QUEUE_SKIP_DESTROY="${ROCR_MACOS_DIRECT_QUEUE_SKIP_DESTROY:-1}"
# Active idle HQD attach is useful for targeted debugging, but cross-process
# reuse is still unsafe after larger library kernels and can poison HQD0.
export ROCR_MACOS_DIRECT_QUEUE_REUSE_ACTIVE="${ROCR_MACOS_DIRECT_QUEUE_REUSE_ACTIVE:-0}"
export ROCR_MACOS_DIRECT_QUEUE_ADVANCE_ON_INACTIVE="${ROCR_MACOS_DIRECT_QUEUE_ADVANCE_ON_INACTIVE:-0}"
# Limit automatic validation to HQD0-HQD5. HQD6/HQD7 are still unproven on
# the RX 9700 eGPU path and have poisoned subsequent submissions.
export ROCR_MACOS_DIRECT_QUEUE_MAX_QUEUES="${ROCR_MACOS_DIRECT_QUEUE_MAX_QUEUES:-6}"
export ROCR_MACOS_DIRECT_QUEUE_PQ_CONTROL="${ROCR_MACOS_DIRECT_QUEUE_PQ_CONTROL:-userspace}"
export ROCR_MACOS_DIRECT_QUEUE_KEEPALIVE="${ROCR_MACOS_DIRECT_QUEUE_KEEPALIVE:-0}"
# Proven full-stack recipe (validated 2026-05-12): keep hardware HQD0 fixed,
# explicitly firmware-dequeue after every submit, and rotate the backing
# MQD/ring/RPTR/WPTR slot after each dequeue. Rotating the hardware HQD instead
# of the backing memory poisons the command path; keep ROTATE_AFTER_DEQUEUE=0.
export ROCR_MACOS_DIRECT_QUEUE_DEQUEUE_AFTER_SUBMIT="${ROCR_MACOS_DIRECT_QUEUE_DEQUEUE_AFTER_SUBMIT:-1}"
export ROCR_MACOS_DIRECT_QUEUE_DEQUEUE_AFTER_SUBMIT_INTERVAL="${ROCR_MACOS_DIRECT_QUEUE_DEQUEUE_AFTER_SUBMIT_INTERVAL:-1}"
export ROCR_MACOS_DIRECT_QUEUE_ROTATE_AFTER_DEQUEUE="${ROCR_MACOS_DIRECT_QUEUE_ROTATE_AFTER_DEQUEUE:-0}"
export ROCR_MACOS_DIRECT_QUEUE_ROTATE_BACKING_AFTER_DEQUEUE="${ROCR_MACOS_DIRECT_QUEUE_ROTATE_BACKING_AFTER_DEQUEUE:-1}"
# Broad host-pointer copyback can overwrite adjacent malloc metadata when a
# kernarg contains pointers to small host objects. Keep staging read-only by
# default unless a targeted experiment opts into copyback.
export ROCR_MACOS_AQL_SKIP_HOST_COPYBACK="${ROCR_MACOS_AQL_SKIP_HOST_COPYBACK:-1}"

export ROCFFT_RTC_PROCESS="${ROCFFT_RTC_PROCESS:-1}"
export ROCFFT_RTC_PROCESS_HELPER="${ROCFFT_RTC_PROCESS_HELPER:-${rocm_root}/lib/rocfft/1.0.37/rocfft_rtc_helper}"
export ROCFFT_RTC_MAX_SUBPROCESSES="${ROCFFT_RTC_MAX_SUBPROCESSES:-1}"
export ROCR_MACOS_COMBINED_STACK_SMOKE="${ROCR_MACOS_COMBINED_STACK_SMOKE:-1}"

echo "Using ROCm root: ${rocm_root}"
echo "Using direct queue index: ${direct_queue_desc}"
echo "Skip direct queue destroy: ${ROCR_MACOS_DIRECT_QUEUE_SKIP_DESTROY}"
echo "Reuse active direct queue: ${ROCR_MACOS_DIRECT_QUEUE_REUSE_ACTIVE}"
echo "Advance on inactive direct queue: ${ROCR_MACOS_DIRECT_QUEUE_ADVANCE_ON_INACTIVE}"
echo "Max direct queues: ${ROCR_MACOS_DIRECT_QUEUE_MAX_QUEUES}"
echo "Direct queue PQ control mode: ${ROCR_MACOS_DIRECT_QUEUE_PQ_CONTROL}"
echo "Direct queue keepalive: ${ROCR_MACOS_DIRECT_QUEUE_KEEPALIVE}"
echo "Dequeue direct queue after submit: ${ROCR_MACOS_DIRECT_QUEUE_DEQUEUE_AFTER_SUBMIT}"
echo "Dequeue direct queue after submit interval: ${ROCR_MACOS_DIRECT_QUEUE_DEQUEUE_AFTER_SUBMIT_INTERVAL}"
echo "Rotate direct queue after dequeue: ${ROCR_MACOS_DIRECT_QUEUE_ROTATE_AFTER_DEQUEUE}"
echo "Rotate direct queue backing after dequeue: ${ROCR_MACOS_DIRECT_QUEUE_ROTATE_BACKING_AFTER_DEQUEUE}"
echo "Skip AQL host copyback: ${ROCR_MACOS_AQL_SKIP_HOST_COPYBACK}"
echo "Combined stack smoke: ${ROCR_MACOS_COMBINED_STACK_SMOKE}"
echo "Do not run these forced-HQD smokes in parallel."

if [[ "${ROCR_MACOS_COMBINED_STACK_SMOKE}" != "0" ]]; then
  "${build_dir}/rocm_combined_stack_smoke"
  exit $?
fi

"${build_dir}/hipsolver_potrf_smoke"
"${build_dir}/hipfft_c2c_smoke"
"${build_dir}/hipsparse_spmv_smoke"
"${build_dir}/hiprand_uniform_smoke"
"${blas_build_dir}/hipblas_saxpy_smoke"
"${blas_build_dir}/hipblas_gemm_smoke"
