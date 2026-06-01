// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>

namespace ck_tile {
/*
 * construct this structure with behavior as:
 *
 *   // create stream config with default stream(NULL), and not timing the kernel
 *   stream_config s = stream_config{};
 *
 *   // create stream config with _some_stream_id_, and not timing the kernel
 *   stream_config s = stream_config{_some_stream_id_};
 *
 *   // create stream config with _some_stream_id_, and benchmark with warmup/repeat as default
 *   stream_config s = stream_config{_some_stream_id_, true};
 *
 *   // create stream config with _some_stream_id_, and benchmark using cpu timer
 *   stream_config s = stream_config{_some_stream_id_, true, 0, 3, 10, false};
 *
 *   // create stream config with _some_stream_id_, and enable gpu timer for rotating buffer with
 *rotating buffer count stream_config s = stream_config{_some_stream_id_, true, 0, 3, 10, true,
 *true, 1};
 **/

struct stream_config
{
    hipStream_t stream_id_ = nullptr;
    bool time_kernel_      = false;
    int log_level_         = 0;
    int cold_niters_       = 3;
    int nrepeat_           = 10;
    bool is_gpu_timer_     = true; // keep compatible
    bool flush_cache_      = false;
    int rotating_count_    = 1;
};
} // namespace ck_tile
