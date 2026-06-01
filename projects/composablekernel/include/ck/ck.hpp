// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/config.h"
#include <stdint.h>

#if !defined(__HIPCC_RTC__) || !defined(CK_CODE_GEN_RTC)
#ifndef CK_DONT_USE_HIP_RUNTIME_HEADERS
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#endif
#endif
// to do: add various levels of logging with CK_LOG_LEVEL

#ifndef CK_TIME_KERNEL
#define CK_TIME_KERNEL 1
#endif

// constant address space for kernel parameter
// https://llvm.org/docs/AMDGPUUsage.html#address-spaces
#define CK_CONSTANT_ADDRESS_SPACE __attribute__((address_space(4)))

// launch bounds
#define CK_USE_LAUNCH_BOUNDS 1

#ifdef CK_USE_LAUNCH_BOUNDS
// for most kernels
#define CK_MAX_THREAD_PER_BLOCK 256
#define CK_MIN_BLOCK_PER_CU 2

// for wavelet GEMM kernel
#define CK_WAVELET_MAX_THREAD_PER_BLOCK 512
#define CK_WAVELET_MIN_BLOCK_PER_CU 2
#endif

// kernel attribute: amdgpu_waves_per_eu()
#ifdef CK_USE_WAVES_PER_EU
// for 1-wave kernels, control arguments of amdgpu_waves_per_eu() attribute
#ifndef CK_MIN_WAVES_PER_EU
#define CK_MIN_WAVES_PER_EU 0
#endif

#ifndef CK_MAX_WAVES_PER_EU
#define CK_MAX_WAVES_PER_EU 0
#endif

#else
#define CK_USE_WAVES_PER_EU 0
#endif

// define general macros for various architectures
#if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__) || \
    defined(__gfx9_4_generic__)
#define __gfx9__
#endif
#if defined(__gfx942__) || defined(__gfx950__) || defined(__gfx9_4_generic__)
#define __gfx94__
#endif
#if defined(__gfx1010__) || defined(__gfx1011__) || defined(__gfx1012__) || \
    defined(__gfx1013__) || defined(__gfx10_1_generic__)
#define __gfx101__
#endif
#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || \
    defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__) || \
    defined(__gfx10_3_generic__)
#define __gfx103__
#endif
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \
    defined(__gfx1152__) || defined(__gfx1153__) || defined(__gfx11_generic__)
#define __gfx11__
#endif
#if defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx12_generic__)
#define __gfx12__
#endif

// buffer resource
#ifndef __HIP_DEVICE_COMPILE__ // for host code
#define CK_BUFFER_RESOURCE_3RD_DWORD -1
#elif defined(__gfx803__) || defined(__gfx900__) || defined(__gfx906__) || defined(__gfx9__)
#define CK_BUFFER_RESOURCE_3RD_DWORD 0x00020000
#elif defined(__gfx101__) || defined(__gfx103__)
#define CK_BUFFER_RESOURCE_3RD_DWORD 0x31014000
#elif defined(__gfx11__) || defined(__gfx12__)
#define CK_BUFFER_RESOURCE_3RD_DWORD 0x31004000
#endif

// FMA instruction
#ifndef __HIP_DEVICE_COMPILE__ // for host code, define nothing
#elif defined(__gfx906__) || defined(__gfx9__) || defined(__gfx103__) || defined(__gfx1011__) || \
    defined(__gfx1012__) // for GPU code
#define CK_USE_AMD_V_FMAC_F32
#define CK_USE_AMD_V_DOT2_F32_F16
#define CK_USE_AMD_V_DOT4_I32_I8
#elif defined(__gfx803__) || defined(__gfx900__) || defined(__gfx101__)
#define CK_USE_AMD_V_MAC_F32
#elif defined(__gfx11__) || defined(__gfx12__)
#define CK_USE_AMD_V_FMAC_F32
#define CK_USE_AMD_V_DOT2_F32_F16
#define CK_USE_AMD_V_DOT4_I32_I8_GFX11
#endif

// MFMA instruction
#ifndef __HIP_DEVICE_COMPILE__ // for host code
#define CK_USE_AMD_MFMA
#elif defined(__gfx9__) // for GPU code
#define CK_USE_AMD_MFMA
#endif

#if(defined(__gfx90a__) || defined(__gfx94__))
#define CK_USE_AMD_MFMA_BF16_1K_OP
#endif

#if defined(__gfx94__)
#define CK_USE_AMD_MFMA_GFX940
#endif

// buffer load
#define CK_USE_AMD_BUFFER_LOAD 1

// buffer store
#define CK_USE_AMD_BUFFER_STORE 1

// buffer atomic add: integer
#define CK_USE_AMD_BUFFER_ATOMIC_ADD_INTEGER 1

// buffer atomic add: floating point
#ifndef __HIP_DEVICE_COMPILE__ // for host code
#define CK_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT 1
#elif defined(__gfx9__) || defined(__gfx12__) // for GPU code
#define CK_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT 1
#else // for GPU code
#define CK_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT 0
#endif

#if(defined(__gfx90a__) || defined(__gfx94__)) // for GPU code
#define CK_USE_AMD_BUFFER_ATOMIC_MAX_FLOAT64 1
#else
#define CK_USE_AMD_BUFFER_ATOMIC_MAX_FLOAT64 0
#endif

// inline asm
#define CK_USE_AMD_INLINE_ASM 1

// inner product (V_MAC/V_FMAC)
#define CK_USE_AMD_V_MAC_INLINE_ASM 1

// V_DOT inline instructions, less efficient since they require adding
// `s_nop`s to avoid hazard
#define CK_USE_AMD_V_DOT_INLINE_ASM 0

// inner product using V_DOT with DPP8 modifiers
#define CK_USE_AMD_V_DOT_DPP8_INLINE_ASM 1

// LDS direct loads using inline assembly
#define CK_USE_AMD_LDS_DIRECT_LOAD_INLINE_ASM 0

// set rounding to nearest even as default for bf16 conversions
#define CK_USE_RNE_BF16_CONVERSION 1

// set rounding to nearest even as default for f8 conversions
#define CK_USE_SR_F8_CONVERSION 0

// set rounding to nearest even as default for f6 conversions
#define CK_USE_SR_F6_CONVERSION 0

// set rounding to nearest even as default for f4 conversions
#define CK_USE_SR_F4_CONVERSION 0

// shuffle pk_i4 values during conversion to optimize number of binary
// operations
#define CK_USE_PK4_LAYOUT_SHUFFLE 1

// block synchronization only s_wait lgkmcnt(0), not vmcnt(0)
#define CK_EXPERIMENTAL_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM 1

// experimental feature: multi index implemented as array
#define CK_EXPERIMENTAL_USE_DYNAMICALLY_INDEXED_MULTI_INDEX 0

// experimental feature: static tensor descriptor
#define CK_EXPERIMENTAL_STATIC_TENSOR_DESCRIPTOR 0

// experimental feature: buffer load/store/atomic-add/ OOB trick
// This (ifndef) is a hack to use customized behavior for buffer load rather than using default
// setting. Don't use this hack unless absolutely necessary!
// FIXME: make the behavior of buffer load a configurable (template) parameter for each usage
#ifndef CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK
#define CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK 0
#endif
#define CK_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK 1
#define CK_EXPERIMENTAL_USE_BUFFER_ATOMIC_ADD_OOB_CHECK_OFFSET_TRICK 1
#define CK_EXPERIMENTAL_USE_BUFFER_ATOMIC_MAX_OOB_CHECK_OFFSET_TRICK 1

// experimental feature: in-regsiter sub-dword transpose
#define CK_EXPERIMENTAL_USE_IN_REGISTER_SUB_DWORD_TRANSPOSE 1

// experimental feature: merge transformation use magic number division
#define CK_EXPERIMENTAL_MERGE_USE_MAGIC_DIVISION 1

// experimental feature: use __builtin_memcpy instead of pointer cast to access a vector from
// pointer of scalar
#define CK_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS 0

// experimental feature: use __builtin_memcpy instead of union to do bit_cast
#define CK_EXPERIMENTAL_USE_MEMCPY_FOR_BIT_CAST 1

// experimental feature: optimize for inter-wave scheduling policy
#define CK_EXPERIMENTAL_INTER_WAVE_SCHEDULING 1
#define CK_EXPERIMENTAL_INTER_WAVE_SCHEDULING_MAC_CLUSTERS 1
// this will let make_default_loop_scheduler() return interwave scheduling flag by default
#define CK_EXPERIMENTAL_DEFAULT_TO_INTER_WAVE_SCHEDULING 0
// experimental feature: add instances using interwave scheduling
#define CK_EXPERIMENTAL_INTER_WAVE_INSTANCES 1
// experimental feature: add instances using pipeline v2
#define CK_EXPERIMENTAL_PIPELINE_V2_INSTANCES 1
// experimental feature: optimize pipeline v2 by IGLP strategy (value=ID of strategy)
#ifndef CK_EXPERIMENTAL_PIPELINE_V2_IGLP_OPT
#define CK_EXPERIMENTAL_PIPELINE_V2_IGLP_OPT 0
#endif

// hack: have underlying assumption that need to be satsified, otherwise it's a bug
// hack for forcing register to keep idx_diff_low_const in SGPR. idx_diff_low_const must be
// thread-invariant, otherwise it's a bug
// TODO: separate index calculation into "compile-time", "global", "block", "wave", "thread"
#define CK_HACK_MERGE_CALCULATE_IDX_DIFF_LOW_CONST_USE_AMD_GCN_READ_FIRST_LANE 0

// workaround: compiler crash when compiling recursive lambda
#define CK_WORKAROUND_SWDEV_275126 1

// workaround: compiler crash when using buffer load/store for i8
#define CK_WORKAROUND_SWDEV_XXXXXX_INT8_BUFFER_LOAD_STORE_ISSUE 1

// workaround: compiler gnerating inefficient ds_write instructions
#define CK_WORKAROUND_SWDEV_XXXXXX_INT8_DS_WRITE_ISSUE 1

// workaround: verifaction failure, due to compiler regression, for conv bwd-data fp16 using some
// tuning parameter
#define CK_WORKAROUND_SWDEV_325164 0

// workaround: compiler not emiting reciprocal instruction frm __frcp_rn()
#define CK_WORKAROUND_SWDEV_383542 1

// workaround: compiler issue on gfx908
#define CK_WORKAROUND_SWDEV_388832 1

// workaround: compiler issue on gfx950
#define CK_WORKAROUND_FP16_TO_FP8_CONVERSION 1

// workaround: compiler issue on gfx950
#define CK_WORKAROUND_BF16_TO_FP8_CONVERSION 1

// denorm test fix, necessary for gfx90a
#ifndef CK_GFX90A_DENORM_WORKAROUND
#define CK_GFX90A_DENORM_WORKAROUND 0
#endif // CK_GFX90A_DENORM_WORKAROUND
// Enable only for gfx90a
#if defined(__gfx90a__)
#if CK_GFX90A_DENORM_WORKAROUND
#define CK_GFX90A_DENORM_WORKAROUND 1
#endif // CK_GFX90A_DENORM_WORKAROUND is set to 1
#else
#define CK_GFX90A_DENORM_WORKAROUND 0
#endif // gfx90a

// set flag to 1 to build deprecated instances
#define CK_BUILD_DEPRECATED 1

namespace ck {

#if defined(__GFX9__) || !defined(__HIP_DEVICE_COMPILE__)
__device__ static constexpr int WarpSize = 64;
#else
__device__ static constexpr int WarpSize = 32;
#endif

enum struct InMemoryDataOperationEnum
{
    Set,
    AtomicAdd,
    AtomicMax,
    Add
};

// FIXME: use regular Sequence and remove this
template <InMemoryDataOperationEnum... Is>
struct InMemoryDataOperationEnumSequence
{
    static constexpr int mSize = sizeof...(Is);

    __host__ __device__ static constexpr InMemoryDataOperationEnum At(int I)
    {
        // the last dummy element is to prevent compiler complain about empty array, when mSize = 0
        const InMemoryDataOperationEnum mData[mSize + 1] = {Is..., InMemoryDataOperationEnum::Set};
        return mData[I];
    }
};

// index type
using index_t      = int32_t;
using long_index_t = int64_t;

} // namespace ck
