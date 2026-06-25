// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/reduce.hpp"
#include "ck_tile/ops/batched_transpose.hpp"

#include <vector>
#include <string>

#pragma once

struct batched_transpose_trait
{
    std::string type;
    std::string layout;
    std::string pipeline;
};

struct batched_transpose_kargs : public ck_tile::BatchedTransposeHostArgs
{
};

float batched_transpose(batched_transpose_trait t,
                        batched_transpose_kargs a,
                        ck_tile::stream_config s);
