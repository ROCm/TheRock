// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "aux_overlay_build_flags.h"

static_assert(ROCM_BUILD_FLAG(ROCM_BUILD_FLAGS_CANARY_BOOL_FALSE) == 0,
              "False ROCm build flag canary must be zero");
static_assert(ROCM_BUILD_FLAG(ROCM_BUILD_FLAGS_CANARY_BOOL_TRUE) == 1,
              "True ROCm build flag canary must be one");
static_assert(ROCM_BUILD_FLAG(ROCM_BUILD_FLAGS_CANARY_INTEGER_NEGATIVE) == -17,
              "Integer ROCm build flag canary must preserve its value");
