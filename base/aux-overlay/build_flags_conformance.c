// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "aux_overlay_build_flags.h"

#if ROCM_BUILD_FLAG(ROCM_BUILD_FLAGS_CANARY_BOOL_FALSE)
#error "False ROCm build flag canary evaluated as true"
#endif

#if !ROCM_BUILD_FLAG(ROCM_BUILD_FLAGS_CANARY_BOOL_TRUE)
#error "True ROCm build flag canary evaluated as false"
#endif

#if ROCM_BUILD_FLAG(ROCM_BUILD_FLAGS_CANARY_INTEGER_NEGATIVE) != -17
#error "Integer ROCm build flag canary has the wrong value"
#endif

_Static_assert(ROCM_BUILD_FLAG(ROCM_BUILD_FLAGS_CANARY_BOOL_FALSE) == 0,
               "False ROCm build flag canary must be zero");
_Static_assert(ROCM_BUILD_FLAG(ROCM_BUILD_FLAGS_CANARY_BOOL_TRUE) == 1,
               "True ROCm build flag canary must be one");
_Static_assert(ROCM_BUILD_FLAG(ROCM_BUILD_FLAGS_CANARY_INTEGER_NEGATIVE) == -17,
               "Integer ROCm build flag canary must preserve its value");
