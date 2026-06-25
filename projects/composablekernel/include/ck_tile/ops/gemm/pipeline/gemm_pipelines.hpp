// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile {

enum struct GemmPipeline
{
    COMPUTE_ASYNC,
    COMPUTE_V3,
    COMPUTE_V4,
    COMPUTE_V5,
    COMPUTE_V6,
    MEMORY,
    BASIC_V1,
    BASIC_V2,
    PRESHUFFLE_V2
};

} // namespace ck_tile
