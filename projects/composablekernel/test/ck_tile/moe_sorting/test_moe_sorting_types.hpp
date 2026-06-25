// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>
#include "ck_tile/host.hpp"
#include "gtest/gtest.h"

using KernelTypesMoeSorting = ::testing::Types<std::tuple<float, ck_tile::index_t>>;
