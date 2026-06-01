// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/topk_softmax.hpp"
#include <string>

struct topk_softmax_trait
{
    std::string input_type;
    std::string weight_type; // currently always float
    int experts;
    std::string activation; // "softmax" or "sigmoid"
};

struct topk_softmax_kargs : public ck_tile::TopkSoftmaxHostArgs
{
};

float topk_softmax(topk_softmax_trait t, topk_softmax_kargs a, ck_tile::stream_config s);
