// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_moe_smoothquant_types.hpp"
#include "test_moe_smoothquant_util.hpp"
#include "gtest/gtest.h"

#define TEST_SUITE_NAME TestCkTileMoeSmoothquant

TYPED_TEST_SUITE(TestCkTileMoeSmoothquant, KernelTypesMoeSmoothquant);

#include "test_moe_smoothquant_cases.inc"

#undef TEST_SUITE_NAME
