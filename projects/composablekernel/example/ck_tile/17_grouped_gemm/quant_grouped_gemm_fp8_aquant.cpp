// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "quant_run_grouped_gemm_example.hpp"

template int run_gemm_example_persistency<ck_tile::fp8_t, ck_tile::QuantType::AQuantGrouped>(
    const ck_tile::ArgParser&, std::string, std::string, bool);
