// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/scheduler_enum.hpp"

namespace ck {

/// @brief Helper function to get default loop scheduler
/// @details Returns the default loop scheduler based on compile-time configuration.
constexpr LoopScheduler make_default_loop_scheduler()
{
#if CK_EXPERIMENTAL_DEFAULT_TO_INTER_WAVE_SCHEDULING
    return LoopScheduler::Interwave;
#else
    return LoopScheduler::Default;
#endif
}

} // namespace ck
