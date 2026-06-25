// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>
#include "ck_tile/host.hpp"
#include "gtest/gtest.h"

using F16Types = std::tuple<ck_tile::fp16_t>;
using KernelTypesPermute =
    ::testing::Types<F16Types, std::tuple<float>, std::tuple<ck_tile::fp8_t>>;
