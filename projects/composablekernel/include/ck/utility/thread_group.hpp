// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "get_id.hpp"

namespace ck {

template <index_t ThreadPerBlock>
struct ThisThreadBlock
{
    static constexpr index_t kNumThread_ = ThreadPerBlock;

    __device__ static constexpr index_t GetNumOfThread() { return kNumThread_; }

    __device__ static constexpr bool IsBelong() { return true; }

    __device__ static index_t GetThreadId() { return get_thread_local_1d_id(); }
};

} // namespace ck
