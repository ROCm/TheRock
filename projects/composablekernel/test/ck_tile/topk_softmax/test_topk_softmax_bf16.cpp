// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_topk_softmax.hpp"

int main() { return run_gemm_combinations<ck_tile::bf16_t>("bf16"); }
