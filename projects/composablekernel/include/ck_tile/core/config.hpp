// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

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

#include "hip/hip_version.h"
#ifndef CK_TILE_DONT_USE_HIP_RUNTIME_HEADERS
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#endif

#ifdef __HIPCC__
#define CK_TILE_HOST inline __host__
#define CK_TILE_DEVICE inline __device__
#define CK_TILE_HOST_DEVICE inline __host__ __device__
#define CK_TILE_DEVICE_EXTERN __device__
#define CK_TILE_HOST_DEVICE_EXTERN __host__ __device__
#else
#define CK_TILE_HOST inline
#define CK_TILE_DEVICE inline
#define CK_TILE_HOST_DEVICE inline
#define CK_TILE_DEVICE_EXTERN
#define CK_TILE_HOST_DEVICE_EXTERN
#endif

// implementing the "memory address space" attribute
// https://llvm.org/docs/AMDGPUUsage.html#amdgpu-address-spaces-table
// WA for https://github.com/ROCm/composable_kernel/issues/1946
#if 0
#define CK_TILE_GENERIC_ADDR __attribute__((address_space(0)))
#define CK_TILE_GLOBAL_ADDR __attribute__((address_space(1)))
#define CK_TILE_LDS_ADDR __attribute__((address_space(3)))
#define CK_TILE_BUF_RES_ADDR __attribute__((address_space(8)))
#else
#define CK_TILE_GENERIC_ADDR
#define CK_TILE_GLOBAL_ADDR
#define CK_TILE_LDS_ADDR
#define CK_TILE_BUF_RES_ADDR
#endif
#ifndef CK_TILE_USE_CUSTOM_DATA_TYPE
#define CK_TILE_USE_CUSTOM_DATA_TYPE 0 // custom data type will generate extra move/bfi code
#endif

#define CK_TILE_FLOAT_TO_BFLOAT16_STANDARD 0
#define CK_TILE_FLOAT_TO_BFLOAT16_TRUNCATE_WITH_NAN 1
#define CK_TILE_FLOAT_TO_BFLOAT16_TRUNCATE 2
#define CK_TILE_FLOAT_TO_BFLOAT16_STANDARD_ASM 3
#define CK_TILE_FLOAT_TO_BFLOAT16_RTA_ASM 4

#ifndef CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT
#define CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT CK_TILE_FLOAT_TO_BFLOAT16_TRUNCATE
#endif

#define CK_TILE_FLOAT_TO_FP8_STANDARD 0
#define CK_TILE_FLOAT_TO_FP8_STOCHASTIC 1

#ifndef CK_TILE_FLOAT_TO_FP8_DEFAULT
#define CK_TILE_FLOAT_TO_FP8_DEFAULT CK_TILE_FLOAT_TO_FP8_STANDARD
#endif

// in the old rocm period, we have to use tuple array implementation to implement this
// so turn on the _USE_TUPLE if meet compiler error, otherwise _USE_ARRAY by default.
#define CK_TILE_STATICALLY_INDEXED_ARRAY_USE_ARRAY 0
#define CK_TILE_STATICALLY_INDEXED_ARRAY_USE_TUPLE 1
#ifndef CK_TILE_STATICALLY_INDEXED_ARRAY_DEFAULT
#define CK_TILE_STATICALLY_INDEXED_ARRAY_DEFAULT CK_TILE_STATICALLY_INDEXED_ARRAY_USE_TUPLE
#endif

#define CK_TILE_THREAD_BUFFER_USE_ARRAY 0
#define CK_TILE_THREAD_BUFFER_USE_TUPLE 1
#ifndef CK_TILE_THREAD_BUFFER_DEFAULT
#define CK_TILE_THREAD_BUFFER_DEFAULT CK_TILE_THREAD_BUFFER_USE_ARRAY
#endif

#ifndef CK_TILE_TUPLE_CTOR_WITH_INITIALIZER_LIST
#if CK_TILE_THREAD_BUFFER_DEFAULT == CK_TILE_THREAD_BUFFER_USE_TUPLE
// if using tuple-array as thread_buffer implementation, need to support {} brace init
// ... with similiar behavior as array
#define CK_TILE_TUPLE_CTOR_WITH_INITIALIZER_LIST 1
#else
#define CK_TILE_TUPLE_CTOR_WITH_INITIALIZER_LIST 0
#endif
#endif

#ifndef CK_TILE_USE_LAUNCH_BOUNDS
#define CK_TILE_USE_LAUNCH_BOUNDS 1
#endif

#ifndef CK_TILE_TIME_KERNEL
#define CK_TILE_TIME_KERNEL 1
#endif

#define CK_TILE_MAX_THREAD_PER_BLOCK 256
#define CK_TILE_MIN_BLOCK_PER_CU 2

#ifndef CK_TILE_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK
#define CK_TILE_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK 0
#endif

#ifndef CK_TILE_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK
#define CK_TILE_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK 1
#endif

#ifndef CK_TILE_EXPERIMENTAL_USE_BUFFER_ATOMIC_ADD_OOB_CHECK_OFFSET_TRICK
#define CK_TILE_EXPERIMENTAL_USE_BUFFER_ATOMIC_ADD_OOB_CHECK_OFFSET_TRICK 1
#endif

#ifndef CK_TILE_EXPERIMENTAL_USE_BUFFER_ATOMIC_MAX_OOB_CHECK_OFFSET_TRICK
#define CK_TILE_EXPERIMENTAL_USE_BUFFER_ATOMIC_MAX_OOB_CHECK_OFFSET_TRICK 1
#endif

#ifndef CK_TILE_USE_AMD_LDS_DIRECT_LOAD_INLINE_ASM
#define CK_TILE_USE_AMD_LDS_DIRECT_LOAD_INLINE_ASM 1
#endif

#ifndef CK_TILE_USE_AMD_BUFFER_LOAD
#define CK_TILE_USE_AMD_BUFFER_LOAD 1
#endif

#ifndef CK_TILE_USE_AMD_BUFFER_STORE
#define CK_TILE_USE_AMD_BUFFER_STORE 1
#endif

#ifndef CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_INTEGER
#define CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_INTEGER 1
#endif

#ifndef CK_TILE_USE_PK4_LAYOUT_SHUFFLE
#define CK_TILE_USE_PK4_LAYOUT_SHUFFLE 1
#endif

// buffer atomic add: floating point
#ifndef __HIP_DEVICE_COMPILE__ // for host code
#define CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT 1
#elif defined(__gfx9__) || defined(__gfx12__) // for GPU code
#define CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT 1
#else // for GPU code
#define CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT 0
#endif

#if(defined(__gfx90a__) || defined(__gfx94__)) // for GPU code
#define CK_TILE_USE_AMD_BUFFER_ATOMIC_MAX_FLOAT64 1
#else
#define CK_TILE_USE_AMD_BUFFER_ATOMIC_MAX_FLOAT64 0
#endif

#ifndef CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
#define CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS 0
#endif

#ifndef CK_TILE_WORKAROUND_SWDEV_XXXXXX_INT8_DS_WRITE_ISSUE
#define CK_TILE_WORKAROUND_SWDEV_XXXXXX_INT8_DS_WRITE_ISSUE 1
#endif

#ifndef CK_TILE_WORKAROUND_ROCM_6_1_SCRATCH_MEMORY_ISSUE
#if HIP_VERSION_MAJOR == 6 && HIP_VERSION_MINOR == 1 && HIP_VERSION_PATCH >= 40091
#define CK_TILE_WORKAROUND_ROCM_6_1_SCRATCH_MEMORY_ISSUE 1
#else
#define CK_TILE_WORKAROUND_ROCM_6_1_SCRATCH_MEMORY_ISSUE 0
#endif
#endif

// workaround for ROCm 6.2 and later
#ifndef CK_TILE_WORKAROUND_ROCM_6_2_SCRATCH_MEMORY_ISSUE
#if(HIP_VERSION_MAJOR == 6 && HIP_VERSION_MINOR == 2 && HIP_VERSION_PATCH >= 41133) ||  \
    (HIP_VERSION_MAJOR == 6 && HIP_VERSION_MINOR == 3 && HIP_VERSION_PATCH >= 42131) || \
    (HIP_VERSION_MAJOR == 6 && HIP_VERSION_MINOR > 3)
#define CK_TILE_WORKAROUND_ROCM_6_2_SCRATCH_MEMORY_ISSUE 1
#else
#define CK_TILE_WORKAROUND_ROCM_6_2_SCRATCH_MEMORY_ISSUE 0
#endif
#endif

// use llvm builtin bf16 data type after ROCm 6.5
#ifndef CK_TILE_USE_LLVM_BUILTIN_BF16
#if(HIP_VERSION_MAJOR == 6 && HIP_VERSION_MINOR == 5 && HIP_VERSION_PATCH >= 50421) || \
    (HIP_VERSION_MAJOR >= 7)
#define CK_TILE_USE_LLVM_BUILTIN_BF16 1
#else
#define CK_TILE_USE_LLVM_BUILTIN_BF16 0
#endif
#endif

#ifndef CK_TILE_DEBUG_LOG
#define CK_TILE_DEBUG_LOG 0
#endif

#ifndef __HIP_DEVICE_COMPILE__ // for host code
#define CK_TILE_BUFFER_RESOURCE_3RD_DWORD 0xffffffff
#elif defined(__gfx803__) || defined(__gfx900__) || defined(__gfx906__) || \
    defined(__gfx9__) // for GPU code
#define CK_TILE_BUFFER_RESOURCE_3RD_DWORD 0x00020000
#elif defined(__gfx101__) || defined(__gfx103__) // for GPU code
#define CK_TILE_BUFFER_RESOURCE_3RD_DWORD 0x31014000
#elif defined(__gfx11__) || defined(__gfx12__) // for GPU code
#define CK_TILE_BUFFER_RESOURCE_3RD_DWORD 0x31004000
#endif

#ifndef CK_TILE_EXPERIMENTAL_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM
#define CK_TILE_EXPERIMENTAL_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM 1
#endif

#ifndef CK_TILE_USE_SUBDWORD_TILE_CAST
#define CK_TILE_USE_SUBDWORD_TILE_CAST 0
#endif

#ifndef CK_TILE_USE_PK_FP16_TILE_CAST
#define CK_TILE_USE_PK_FP16_TILE_CAST 0
#endif

// TODO: better solve this inside compiler
#ifndef CK_TILE_FMHA_FWD_FAST_EXP2
#define CK_TILE_FMHA_FWD_FAST_EXP2 0
#endif

#ifndef CK_TILE_FMHA_FLOAT_TO_FLOAT16_RTN
#define CK_TILE_FMHA_FLOAT_TO_FLOAT16_RTN 0
#endif

#ifndef CK_TILE_BUFFER_LOAD_RAW_BF16_WA
#define CK_TILE_BUFFER_LOAD_RAW_BF16_WA 1
#endif

// workaround: compiler not emiting reciprocal instruction frm __frcp_rn()
#ifndef CK_TILE_WORKAROUND_SWDEV_383542
#define CK_TILE_WORKAROUND_SWDEV_383542 1
#endif

#ifndef CK_TILE_REFERENCE_MOE_SORTING_MOCK_ID
#define CK_TILE_REFERENCE_MOE_SORTING_MOCK_ID 1
#endif

#ifndef CK_TILE_USE_OCP_FP8
#if defined(__HIP_DEVICE_COMPILE__)
#if defined(__gfx950__) || defined(__gfx12__)
#define CK_TILE_USE_OCP_FP8 1
#else
#define CK_TILE_USE_OCP_FP8 0
#endif
#else
#define CK_TILE_USE_OCP_FP8 0
#endif
#endif

#ifndef CK_TILE_USE_BUFFER_ADDRESSING_BUILTIN
#if __clang_major__ >= 20 && !(defined(__gfx103__) || defined(__gfx11__) || defined(__gfx12__))
#define CK_TILE_USE_BUFFER_ADDRESSING_BUILTIN 1
#else
#define CK_TILE_USE_BUFFER_ADDRESSING_BUILTIN 0
#endif
#endif

#ifndef CK_TILE_WA_ISSUE_2028
#define CK_TILE_WA_ISSUE_2028 0
#endif

// Y pointed to R, we don't see a valuable use case.
// Will enforce encoding to check Y not pointed to R if set to zero
#ifndef CK_TILE_ENC_SUPPORT_Y_TO_R
#define CK_TILE_ENC_SUPPORT_Y_TO_R 0
#endif

// Mark unsupported features with a deprecation warning in debug builds
#if defined(NDEBUG)
#define CK_TILE_UNSUPPORTED_IMPL(MSG)
#else
#define CK_TILE_UNSUPPORTED_IMPL(MSG) __attribute__((deprecated(MSG)))
#endif

namespace ck_tile::core {
/**
 * @struct amdgcn_compiler_target_state
 * @brief Defines compiler states for supported AMDGCN devices.
 * @var CK_TILE_HOST_COMPILE Indicates if the compilation is for the host.
 * @var CK_TILE_DEVICE_COMPILE Indicates if the compilation is for AMDGCN device.
 * @var CK_TILE_ARCH_GFX908 Indicates if the compiler target architecture is GFX908.
 * @var CK_TILE_ARCH_GFX90A Indicates if the compiler target architecture is GFX90A.
 * @var CK_TILE_ARCH_GFX942 Indicates if the compiler target architecture is GFX942.
 * @var CK_TILE_ARCH_GFX950 Indicates if the compiler target architecture is GFX950.
 * @var CK_TILE_ARCH_GFX1030 Indicates if the compiler target architecture is GFX1030.
 * @var CK_TILE_ARCH_GFX1031 Indicates if the compiler target architecture is GFX1031.
 * @var CK_TILE_ARCH_GFX1032 Indicates if the compiler target architecture is GFX1032.
 * @var CK_TILE_ARCH_GFX1034 Indicates if the compiler target architecture is GFX1034.
 * @var CK_TILE_ARCH_GFX1035 Indicates if the compiler target architecture is GFX1035.
 * @var CK_TILE_ARCH_GFX1036 Indicates if the compiler target architecture is GFX1036.
 * @var CK_TILE_ARCH_GFX10_3_GENERIC Indicates if the compiler target architecture is GFX10.3
 * generic.
 * @var CK_TILE_ARCH_GFX1100 Indicates if the compiler target architecture is GFX1100.
 * @var CK_TILE_ARCH_GFX1101 Indicates if the compiler target architecture is GFX1101.
 * @var CK_TILE_ARCH_GFX1102 Indicates if the compiler target architecture is GFX1102.
 * @var CK_TILE_ARCH_GFX1151 Indicates if the compiler target architecture is GFX1151.
 * @var CK_TILE_ARCH_GFX1152 Indicates if the compiler target architecture is GFX1152.
 * @var CK_TILE_ARCH_GFX1153 Indicates if the compiler target architecture is GFX1153.
 * @var CK_TILE_ARCH_GFX11_GENERIC Indicates if the compiler target architecture is GFX11 generic.
 * @var CK_TILE_ARCH_GFX1200 Indicates if the compiler target architecture is GFX1200.
 * @var CK_TILE_ARCH_GFX1201 Indicates if the compiler target architecture is GFX1201.
 * @var CK_TILE_ARCH_GFX12_GENERIC Indicates if the compiler target architecture is GFX12 generic.
 */
struct amdgcn_compiler_target_state
{
    // Determine if we are compiling for device or host
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
    static constexpr bool CK_TILE_DEVICE_COMPILE = true;
    static constexpr bool CK_TILE_HOST_COMPILE   = false;
#else
    static constexpr bool CK_TILE_DEVICE_COMPILE = false;
    static constexpr bool CK_TILE_HOST_COMPILE   = true;
#endif // __HIP_DEVICE_COMPILE__ && __HIP_DEVICE_COMPILE__

    // GFX9
#if defined(__gfx908__)
    static constexpr bool CK_TILE_ARCH_GFX908 = true;
#else
    static constexpr bool CK_TILE_ARCH_GFX908 = false;
#endif // __gfx908__

#if defined(__gfx90a__)
    static constexpr bool CK_TILE_ARCH_GFX90A = true;
#else
    static constexpr bool CK_TILE_ARCH_GFX90A = false;
#endif // __gfx90a__

#if defined(__gfx942__)
    static constexpr bool CK_TILE_ARCH_GFX942 = true;
#else
    static constexpr bool CK_TILE_ARCH_GFX942 = false;
#endif // __gfx942__

#if defined(__gfx950__)
    static constexpr bool CK_TILE_ARCH_GFX950 = true;
#else
    static constexpr bool CK_TILE_ARCH_GFX950 = false;
#endif // __gfx950__

    // GFX10
#if defined(__gfx1010__)
    static constexpr bool CK_TILE_ARCH_GFX1010 = true;
#else
    static constexpr bool CK_TILE_ARCH_GFX1010 = false;
#endif
#if defined(__gfx1011__)
    static constexpr bool CK_TILE_ARCH_GFX1011 = true;
#else
    static constexpr bool CK_TILE_ARCH_GFX1011 = false;
#endif
#if defined(__gfx1012__)
    static constexpr bool CK_TILE_ARCH_GFX1012 = true;
#else
    static constexpr bool CK_TILE_ARCH_GFX1012 = false;
#endif
#if defined(__gfx1013__)
    static constexpr bool CK_TILE_ARCH_GFX1013 = true;
#else
    static constexpr bool CK_TILE_ARCH_GFX1013 = false;
#endif
#if defined(__gfx10_1_generic__)
    static constexpr bool CK_TILE_ARCH_GFX10_1_GENERIC = true;
#else
    static constexpr bool CK_TILE_ARCH_GFX10_1_GENERIC = false;
#endif // __gfx10_1_generic__

#if defined(__gfx1030__)
    static constexpr bool CK_TILE_ARCH_GFX1030 = true;
#else
    static constexpr bool CK_TILE_ARCH_GFX1030 = false;
#endif // __gfx1030__

#if defined(__gfx1031__)
    static constexpr bool CK_TILE_ARCH_GFX1031 = true;
#else
    static constexpr bool CK_TILE_ARCH_GFX1031 = false;
#endif // __gfx1031__

#if defined(__gfx1032__)
    static constexpr bool CK_TILE_ARCH_GFX1032 = true;
#else
    static constexpr bool CK_TILE_ARCH_GFX1032 = false;
#endif // __gfx1032__

#if defined(__gfx1034__)
    static constexpr bool CK_TILE_ARCH_GFX1034 = true;
#else
    static constexpr bool CK_TILE_ARCH_GFX1034 = false;
#endif // __gfx1034__

#if defined(__gfx1035__)
    static constexpr bool CK_TILE_ARCH_GFX1035 = true;
#else
    static constexpr bool CK_TILE_ARCH_GFX1035 = false;
#endif // __gfx1035__

#if defined(__gfx1036__)
    static constexpr bool CK_TILE_ARCH_GFX1036 = true;
#else
    static constexpr bool CK_TILE_ARCH_GFX1036 = false;
#endif // __gfx1036__

#if defined(__gfx10_3_generic__)
    static constexpr bool CK_TILE_ARCH_GFX10_3_GENERIC = true;
#else
    static constexpr bool CK_TILE_ARCH_GFX10_3_GENERIC = false;
#endif // __gfx10_3_generic__

    // GFX11
#if defined(__gfx1100__)
    static constexpr bool CK_TILE_ARCH_GFX1100 = true;
#else
    static constexpr bool CK_TILE_ARCH_GFX1100 = false;
#endif // __gfx1100__

#if defined(__gfx1101__)
    static constexpr bool CK_TILE_ARCH_GFX1101 = true;
#else
    static constexpr bool CK_TILE_ARCH_GFX1101 = false;
#endif // __gfx1101__

#if defined(__gfx1102__)
    static constexpr bool CK_TILE_ARCH_GFX1102 = true;
#else
    static constexpr bool CK_TILE_ARCH_GFX1102 = false;
#endif // __gfx1102__

#if defined(__gfx1103__)
    static constexpr bool CK_TILE_ARCH_GFX1103 = true;
#else
    static constexpr bool CK_TILE_ARCH_GFX1103 = false;
#endif // __gfx1103__

#if defined(__gfx1150__)
    static constexpr bool CK_TILE_ARCH_GFX1150 = true;
#else
    static constexpr bool CK_TILE_ARCH_GFX1150 = false;
#endif // __gfx1150__

#if defined(__gfx1151__)
    static constexpr bool CK_TILE_ARCH_GFX1151 = true;
#else
    static constexpr bool CK_TILE_ARCH_GFX1151 = false;
#endif // __gfx1151__

#if defined(__gfx1152__)
    static constexpr bool CK_TILE_ARCH_GFX1152 = true;
#else
    static constexpr bool CK_TILE_ARCH_GFX1152 = false;
#endif // __gfx1152__

#if defined(__gfx1153__)
    static constexpr bool CK_TILE_ARCH_GFX1153 = true;
#else
    static constexpr bool CK_TILE_ARCH_GFX1153 = false;
#endif // __gfx1153__

#if defined(__gfx11_generic__)
    static constexpr bool CK_TILE_ARCH_GFX11_GENERIC = true;
#else
    static constexpr bool CK_TILE_ARCH_GFX11_GENERIC = false;
#endif // __gfx11_generic__

    // GFX12
#if defined(__gfx1200__)
    static constexpr bool CK_TILE_ARCH_GFX1200 = true;
#else
    static constexpr bool CK_TILE_ARCH_GFX1200 = false;
#endif // __gfx1200__

#if defined(__gfx1201__)
    static constexpr bool CK_TILE_ARCH_GFX1201 = true;
#else
    static constexpr bool CK_TILE_ARCH_GFX1201 = false;
#endif // __gfx1201__

#if defined(__gfx12_generic__)
    static constexpr bool CK_TILE_ARCH_GFX12_GENERIC = true;
#else
    static constexpr bool CK_TILE_ARCH_GFX12_GENERIC = false;
#endif // __gfx12_generic__
};

/**
 * @brief Helper to count the number of times an item is contained within a list of values
 * @tparam T The type of the search value
 * @tparam Ts The types of the search list values
 * @param search The value to search for
 * @param searchList The list of values to search in
 * @return true if the search value is in the search list, false otherwise
 */
template <typename T, typename... Ts>
// TODO: c++20 concept    requires((std::is_convertible<Ts, T>::value && ...) && (sizeof...(Ts) >=
// 1))
CK_TILE_HOST_DEVICE static constexpr uint32_t count_values_of(T search, Ts... searchList)
{
    static_assert((std::is_convertible<Ts, T>::value && ...),
                  "All search list values must be convertible to the search value type");
    static_assert(sizeof...(Ts) >= 1, "At least one value must be provided to search in");

    return (static_cast<uint32_t>(search == static_cast<T>(searchList)) + ...);
}

#define CK_TILE_COMPILER_TARGETS_LIST                               \
    amdgcn_compiler_target_state::CK_TILE_ARCH_GFX908,              \
        amdgcn_compiler_target_state::CK_TILE_ARCH_GFX90A,          \
        amdgcn_compiler_target_state::CK_TILE_ARCH_GFX942,          \
        amdgcn_compiler_target_state::CK_TILE_ARCH_GFX950,          \
        amdgcn_compiler_target_state::CK_TILE_ARCH_GFX1010,         \
        amdgcn_compiler_target_state::CK_TILE_ARCH_GFX1011,         \
        amdgcn_compiler_target_state::CK_TILE_ARCH_GFX1012,         \
        amdgcn_compiler_target_state::CK_TILE_ARCH_GFX1013,         \
        amdgcn_compiler_target_state::CK_TILE_ARCH_GFX10_1_GENERIC, \
        amdgcn_compiler_target_state::CK_TILE_ARCH_GFX1030,         \
        amdgcn_compiler_target_state::CK_TILE_ARCH_GFX1031,         \
        amdgcn_compiler_target_state::CK_TILE_ARCH_GFX1032,         \
        amdgcn_compiler_target_state::CK_TILE_ARCH_GFX1034,         \
        amdgcn_compiler_target_state::CK_TILE_ARCH_GFX1035,         \
        amdgcn_compiler_target_state::CK_TILE_ARCH_GFX1036,         \
        amdgcn_compiler_target_state::CK_TILE_ARCH_GFX10_3_GENERIC, \
        amdgcn_compiler_target_state::CK_TILE_ARCH_GFX1100,         \
        amdgcn_compiler_target_state::CK_TILE_ARCH_GFX1101,         \
        amdgcn_compiler_target_state::CK_TILE_ARCH_GFX1102,         \
        amdgcn_compiler_target_state::CK_TILE_ARCH_GFX1103,         \
        amdgcn_compiler_target_state::CK_TILE_ARCH_GFX1150,         \
        amdgcn_compiler_target_state::CK_TILE_ARCH_GFX1151,         \
        amdgcn_compiler_target_state::CK_TILE_ARCH_GFX1152,         \
        amdgcn_compiler_target_state::CK_TILE_ARCH_GFX1153,         \
        amdgcn_compiler_target_state::CK_TILE_ARCH_GFX11_GENERIC,   \
        amdgcn_compiler_target_state::CK_TILE_ARCH_GFX1200,         \
        amdgcn_compiler_target_state::CK_TILE_ARCH_GFX1201,         \
        amdgcn_compiler_target_state::CK_TILE_ARCH_GFX12_GENERIC

// Sanity check: make sure only one target architecture is defined during device compile
static_assert(!amdgcn_compiler_target_state::CK_TILE_DEVICE_COMPILE ||
                  count_values_of(true, CK_TILE_COMPILER_TARGETS_LIST) == 1u,
              "Only one target architecture can be defined during device compile");

// Sanity check: make sure no device target architecture is defined during host compile
static_assert(!amdgcn_compiler_target_state::CK_TILE_HOST_COMPILE ||
                  count_values_of(true, CK_TILE_COMPILER_TARGETS_LIST) == 0u,
              "No device target architecture can be defined during host compile");

} // namespace ck_tile::core
