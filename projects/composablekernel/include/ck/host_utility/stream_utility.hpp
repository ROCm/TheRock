// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>

#include "ck/stream_config.hpp"
#include "ck/host_utility/hip_check_error.hpp"

namespace ck {

static inline int getAvailableComputeUnitCount(const StreamConfig& stream_config)
{
    constexpr int MAX_MASK_DWORDS = 64;

    // assume at most 64*32 = 2048 CUs
    uint32_t cuMask[MAX_MASK_DWORDS];

    for(int i = 0; i < MAX_MASK_DWORDS; i++)
        cuMask[i] = 0;

    auto countSetBits = [](uint32_t dword) {
        int count = 0;

        while(dword != 0)
        {
            if(dword & 0x1)
                count++;

            dword = dword >> 1;
        };

        return (count);
    };

    hip_check_error(hipExtStreamGetCUMask(stream_config.stream_id_, MAX_MASK_DWORDS, &cuMask[0]));

    int ret = 0;

    for(int i = 0; i < MAX_MASK_DWORDS; i++)
        ret += countSetBits(cuMask[i]);

    return (ret);
};

} // namespace ck
