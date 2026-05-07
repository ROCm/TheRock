// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>
#include "ck_tile/host.hpp"
#include "gtest/gtest.h"

using KernelTypesSmoothquant =
    ::testing::Types<std::tuple<ck_tile::fp16_t>, std::tuple<ck_tile::bf16_t>>;
