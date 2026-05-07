// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "topk_softmax_api.hpp"

#define TOPK_SOFTMAX_DISPATCH(experts_, use_softmax_)                                              \
    constexpr ck_tile::index_t ts_experts = experts_;                                              \
    constexpr bool ts_use_softmax         = use_softmax_;                                          \
    using ts_problem                      = ck_tile::TopkSoftmaxWarpPerRowProblem<ts_input_type,   \
                                                                                  ts_weight_type,  \
                                                                                  ts_index_type,   \
                                                                                  ts_experts,      \
                                                                                  ts_use_softmax>; \
    using ts_pipeline                     = ck_tile::TopkSoftmaxWarpPerRowPipeline<ts_problem>;    \
                                                                                                   \
    using kernel = ck_tile::TopkSoftmaxKernel<ts_pipeline>;                                        \
                                                                                                   \
    auto kargs = kernel::MakeKargs(a);                                                             \
                                                                                                   \
    const dim3 grids  = kernel::GridSize(a);                                                       \
    const dim3 blocks = kernel::BlockSize();                                                       \
                                                                                                   \
    float ave_time =                                                                               \
        ck_tile::launch_kernel(s, ck_tile::make_kernel<1>(kernel{}, grids, blocks, 0, kargs));     \
                                                                                                   \
    return ave_time;

float topk_softmax(topk_softmax_trait t, topk_softmax_kargs a, ck_tile::stream_config s)
{
    if(t.input_type == "fp16" && t.weight_type == "fp32" && t.activation == "softmax")
    {
        using ts_input_type  = ck_tile::fp16_t;
        using ts_weight_type = float;
        using ts_index_type  = ck_tile::index_t;
#if 1
        if(t.experts <= 8)
        {
            TOPK_SOFTMAX_DISPATCH(8, true)
        }
        else if(t.experts <= 16)
        {
            TOPK_SOFTMAX_DISPATCH(16, true)
        }
        else if(t.experts <= 32)
        {
            TOPK_SOFTMAX_DISPATCH(32, true)
        }
        else if(t.experts <= 64)
        {
            TOPK_SOFTMAX_DISPATCH(64, true)
        }
        else if(t.experts <= 128)
        {
            TOPK_SOFTMAX_DISPATCH(128, true)
        }
        else if(t.experts <= 192)
        {
            TOPK_SOFTMAX_DISPATCH(192, true)
        }
#else
        if(t.experts <= 128)
        {
            TOPK_SOFTMAX_DISPATCH(128, true)
        }
#endif
    }
    else if(t.input_type == "bf16" && t.weight_type == "fp32" && t.activation == "softmax")
    {
#if 1
        using ts_input_type  = ck_tile::bf16_t;
        using ts_weight_type = float;
        using ts_index_type  = ck_tile::index_t;
        if(t.experts <= 8)
        {
            TOPK_SOFTMAX_DISPATCH(8, true)
        }
        else if(t.experts <= 16)
        {
            TOPK_SOFTMAX_DISPATCH(16, true)
        }
        else if(t.experts <= 32)
        {
            TOPK_SOFTMAX_DISPATCH(32, true)
        }
        else if(t.experts <= 64)
        {
            TOPK_SOFTMAX_DISPATCH(64, true)
        }
        else if(t.experts <= 128)
        {
            TOPK_SOFTMAX_DISPATCH(128, true)
        }
        else if(t.experts <= 192)
        {
            TOPK_SOFTMAX_DISPATCH(192, true)
        }
#endif
    }
    else if(t.input_type == "fp16" && t.weight_type == "fp32" && t.activation == "sigmoid")
    {
        using ts_input_type  = ck_tile::fp16_t;
        using ts_weight_type = float;
        using ts_index_type  = ck_tile::index_t;
#if 1
        if(t.experts <= 8)
        {
            TOPK_SOFTMAX_DISPATCH(8, false)
        }
        else if(t.experts <= 16)
        {
            TOPK_SOFTMAX_DISPATCH(16, false)
        }
        else if(t.experts <= 32)
        {
            TOPK_SOFTMAX_DISPATCH(32, false)
        }
        else if(t.experts <= 64)
        {
            TOPK_SOFTMAX_DISPATCH(64, false)
        }
        else if(t.experts <= 128)
        {
            TOPK_SOFTMAX_DISPATCH(128, false)
        }
        else if(t.experts <= 192)
        {
            TOPK_SOFTMAX_DISPATCH(192, false)
        }
#else
        if(t.experts <= 128)
        {
            TOPK_SOFTMAX_DISPATCH(128, false)
        }
#endif
    }
    else if(t.input_type == "bf16" && t.weight_type == "fp32" && t.activation == "sigmoid")
    {
#if 1
        using ts_input_type  = ck_tile::bf16_t;
        using ts_weight_type = float;
        using ts_index_type  = ck_tile::index_t;
        if(t.experts <= 8)
        {
            TOPK_SOFTMAX_DISPATCH(8, false)
        }
        else if(t.experts <= 16)
        {
            TOPK_SOFTMAX_DISPATCH(16, false)
        }
        else if(t.experts <= 32)
        {
            TOPK_SOFTMAX_DISPATCH(32, false)
        }
        else if(t.experts <= 64)
        {
            TOPK_SOFTMAX_DISPATCH(64, false)
        }
        else if(t.experts <= 128)
        {
            TOPK_SOFTMAX_DISPATCH(128, false)
        }
        else if(t.experts <= 192)
        {
            TOPK_SOFTMAX_DISPATCH(192, false)
        }
#endif
    }
    return -1;
}
