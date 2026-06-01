// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/permute.hpp"
#include <string>

struct permute_traits
{
    std::string data_type;
};

using permute_args = ck_tile::GenericPermuteHostArgs;

// host API
float permute(permute_traits, permute_args, const ck_tile::stream_config&);
