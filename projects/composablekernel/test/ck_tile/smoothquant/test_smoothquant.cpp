// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_smoothquant_types.hpp"
#include "test_smoothquant_util.hpp"
#include "gtest/gtest.h"

#define TEST_SUITE_NAME TestCkTileSmoothquant

TYPED_TEST_SUITE(TestCkTileSmoothquant, KernelTypesSmoothquant);

#include "test_smoothquant_cases.inc"

#undef TEST_SUITE_NAME
