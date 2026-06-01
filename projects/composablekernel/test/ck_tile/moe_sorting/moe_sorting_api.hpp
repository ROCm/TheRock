// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <string>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/fused_moe.hpp"

struct moe_sorting_trait
{
    std::string index_type;
    std::string weight_type;         // currently always float
    bool local_expert_masking;       // if mask experts as local expert
    bool clear_workspace_inside_api; // if true, no need clear workspace outsize (will take care of
                                     // it inside API)
    int dispatch_policy; // 0 - let the API choose kernel for you. 1 - always use single kerenl. 2 -
                         // always use mp kernel NOTE: moe_sorting_get_workspace_size() need use
                         // same dispatch_policy value. it will be undefined behavior if ppl using
                         // different value when get ws and call the kernel
};

struct moe_sorting_args : public ck_tile::MoeSortingHostArgs
{
};

// use below API before call moe_sorting() to indicate if need workspace or not
// if return non zero, means need workspace, you need to allocate a GPU buffer
// and set to moe_sorting_args.p_ws
// NOTE: workspace size are required to clear zero before use the API
int moe_sorting_get_workspace_size(int tokens, int num_experts, int topk, int dispatch_policy);
float moe_sorting(moe_sorting_trait t, moe_sorting_args a, ck_tile::stream_config s);
float moe_sorting_mp(moe_sorting_trait t, moe_sorting_args a, ck_tile::stream_config s);
