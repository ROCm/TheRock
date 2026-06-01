// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_permute_types.hpp"
#include "test_permute_util.hpp"
#include "gtest/gtest.h"

#define TEST_SUITE_NAME TestCkTilePermute

TYPED_TEST_SUITE(TestCkTilePermute, KernelTypesPermute);

#include "test_permute_cases.inc"

#undef TEST_SUITE_NAME
