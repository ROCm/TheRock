// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_moe_sorting_types.hpp"
#include "test_moe_sorting_util.hpp"
#include "gtest/gtest.h"

#define TEST_SUITE_NAME TestCkTileMoeSorting

TYPED_TEST_SUITE(TestCkTileMoeSorting, KernelTypesMoeSorting);

#include "test_moe_sorting_cases.inc"

#undef TEST_SUITE_NAME
