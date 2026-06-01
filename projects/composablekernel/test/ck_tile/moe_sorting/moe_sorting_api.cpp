// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "moe_sorting_api.hpp"

#ifndef MOE_SORTING_USE_EX_KERNEL
#define MOE_SORTING_USE_EX_KERNEL 1
#endif

#ifndef MOE_SORTING_SUPPORT_LARGE_EXPERT
#define MOE_SORTING_SUPPORT_LARGE_EXPERT 0
#endif

#ifndef MOE_SORTING_SUPPORT_LARGE_TOPK
#define MOE_SORTING_SUPPORT_LARGE_TOPK 0
#endif

#if !MOE_SORTING_USE_EX_KERNEL

#define MOE_SORTING_DISPATCH_ETILE(unroll_num_, expert_tile_)                         \
    constexpr ck_tile::index_t unroll_num  = unroll_num_;                             \
    constexpr ck_tile::index_t expert_tile = expert_tile_;                            \
    using ms_problem =                                                                \
        ck_tile::MoeSortingProblem<index_t, ms_weight_type, unroll_num, expert_tile>; \
    using kernel         = ck_tile::MoeSortingKernel<ms_problem>;                     \
    auto kargs           = kernel::MakeKargs(a);                                      \
    const dim3 grids     = kernel::GridSize(a);                                       \
    const dim3 blocks    = kernel::BlockSize(a);                                      \
    const auto lds_bytes = kernel::GetSmemSize(a);                                    \
    float ave_time       = ck_tile::launch_kernel(                                    \
        s, ck_tile::make_kernel(kernel{}, grids, blocks, lds_bytes, kargs));    \
    return ave_time;

#else

#define MOE_SORTING_DISPATCH_(                                                                          \
    sub_token_tile_, sub_token_onshot_, local_expert_masking_, local_token_)                            \
    constexpr ck_tile::index_t sub_token_tile = sub_token_tile_;                                        \
    constexpr bool sub_token_onshot           = sub_token_onshot_;                                      \
    constexpr bool local_expert_masking       = local_expert_masking_;                                  \
    constexpr bool local_token                = local_token_;                                           \
    using ms_problem                          = ck_tile::MoeSortingProblemEx<index_t,                   \
                                                                             ms_weight_type,            \
                                                                             sub_token_tile,            \
                                                                             sub_token_onshot,          \
                                                                             local_expert_masking,      \
                                                                             local_token>;              \
    using kernel                              = ck_tile::MoeSortingKernel<ms_problem>;                  \
    auto kargs                                = kernel::MakeKargs(a);                                   \
    const dim3 grids                          = kernel::GridSize(a);                                    \
    const dim3 blocks                         = kernel::BlockSize(a);                                   \
    const auto lds_bytes                      = kernel::GetSmemSize(a);                                 \
    float ave_time                            = ck_tile::launch_kernel(                                 \
        s, ck_tile::make_kernel(kernel{}, grids, blocks, lds_bytes, kargs)); \
    return ave_time;

#define MOE_SORTING_DISPATCH_SUB_TOKEN_(                                                  \
    row_, sub_token_onshot_, local_expert_masking_, local_token_)                         \
    if(row_ % 8 == 0)                                                                     \
    {                                                                                     \
        MOE_SORTING_DISPATCH_(8, sub_token_onshot_, local_expert_masking_, local_token_); \
    }                                                                                     \
    else if(row_ % 4 == 0)                                                                \
    {                                                                                     \
        MOE_SORTING_DISPATCH_(4, sub_token_onshot_, local_expert_masking_, local_token_); \
    }                                                                                     \
    else if(row_ % 2 == 0)                                                                \
    {                                                                                     \
        MOE_SORTING_DISPATCH_(2, sub_token_onshot_, local_expert_masking_, local_token_); \
    }                                                                                     \
    else                                                                                  \
    {                                                                                     \
        MOE_SORTING_DISPATCH_(1, sub_token_onshot_, local_expert_masking_, local_token_); \
    }

#define MOE_SORTING_DISPATCH_DYNAMIC_TOKEN_(row_, sub_token_onshot_, local_expert_masking_)    \
    if(is_local_token)                                                                         \
    {                                                                                          \
        MOE_SORTING_DISPATCH_SUB_TOKEN_(row_, sub_token_onshot_, local_expert_masking_, true)  \
    }                                                                                          \
    else                                                                                       \
    {                                                                                          \
        MOE_SORTING_DISPATCH_SUB_TOKEN_(row_, sub_token_onshot_, local_expert_masking_, false) \
    }

#define MOE_SORTING_DISPATCH_SUBTO_(row_, local_expert_masking_)                \
    if(is_sub_token_onshot)                                                     \
    {                                                                           \
        MOE_SORTING_DISPATCH_DYNAMIC_TOKEN_(row_, true, local_expert_masking_)  \
    }                                                                           \
    else                                                                        \
    {                                                                           \
        MOE_SORTING_DISPATCH_DYNAMIC_TOKEN_(row_, false, local_expert_masking_) \
    }

#define MOE_SORTING_DISPATCH_EMASK_(row_)        \
    if(is_local_expert_masking)                  \
    {                                            \
        MOE_SORTING_DISPATCH_SUBTO_(row_, true)  \
    }                                            \
    else                                         \
    {                                            \
        MOE_SORTING_DISPATCH_SUBTO_(row_, false) \
    }

#endif

#if !MOE_SORTING_USE_EX_KERNEL
#define MOE_SORTING_DISPATCH(unroll_num_)           \
    if(a.num_experts <= 8)                          \
    {                                               \
        MOE_SORTING_DISPATCH_ETILE(unroll_num_, 8)  \
    }                                               \
    else if(a.num_experts <= 16)                    \
    {                                               \
        MOE_SORTING_DISPATCH_ETILE(unroll_num_, 16) \
    }                                               \
    else if(a.num_experts <= 32)                    \
    {                                               \
        MOE_SORTING_DISPATCH_ETILE(unroll_num_, 32) \
    }                                               \
    else if(a.num_experts <= 64)                    \
    {                                               \
        MOE_SORTING_DISPATCH_ETILE(unroll_num_, 64) \
    }                                               \
    else                                            \
    {                                               \
        MOE_SORTING_DISPATCH_ETILE(unroll_num_, 0)  \
    }
#endif

float moe_sorting(moe_sorting_trait t, moe_sorting_args a, ck_tile::stream_config s)
{
    if(t.weight_type == "fp32" && t.index_type == "int32")
    {
#if !MOE_SORTING_USE_EX_KERNEL
        if(a.num_experts > 127)
        {
            printf("lds size exceed, only support experts <127 \n");
            return -1;
        }
        if(a.moe_buf_bytes % 16)
        {
            printf("buf set size %d unaligned, must be multiple of 16\n", a.moe_buf_bytes);
            return -1;
        }
        using index_t              = ck_tile::index_t;
        using ms_weight_type       = float;
        index_t smem_io_unroll_num = ck_tile::integer_divide_ceil(a.tokens * a.topk, 64);
        switch(smem_io_unroll_num)
        {
        case(1): {
            MOE_SORTING_DISPATCH(1);
        }
        case(2): {
            MOE_SORTING_DISPATCH(2);
        }
        case(3): {
            MOE_SORTING_DISPATCH(3);
        }
        case(5): {
            MOE_SORTING_DISPATCH(5);
        }
        case(6): {
            MOE_SORTING_DISPATCH(6);
        }
        case(8): {
            MOE_SORTING_DISPATCH(8);
        }
        case(10): {
            MOE_SORTING_DISPATCH(10);
        }
        default: {
            MOE_SORTING_DISPATCH(4);
        }
        }
#else
        if(moe_sorting_get_workspace_size(a.tokens, a.num_experts, a.topk, t.dispatch_policy) != 0)
        {
            return moe_sorting_mp(t, a, s);
        }
        using index_t                = ck_tile::index_t;
        using ms_weight_type         = float;
        auto sub_token_              = ck_tile::moe_sorting_get_sub_token(a.tokens, a.num_experts);
        auto row_                    = sub_token_ / 8;
        bool is_sub_token_onshot     = a.tokens <= sub_token_;
        bool is_local_expert_masking = t.local_expert_masking;
        bool is_local_token          = a.p_local_tokens != nullptr;

        MOE_SORTING_DISPATCH_EMASK_(row_);
        // MOE_SORTING_DISPATCH_ETILE(0, 0);
#endif
    }
    return -1;
}

#define MOE_SORTING_MP_0_V1(mesh_type_, unroll_num_, expert_masking_, local_token_)          \
    [&]() {                                                                                  \
        constexpr ck_tile::index_t unroll_num = unroll_num_;                                 \
        constexpr bool expert_masking         = expert_masking_;                             \
        constexpr bool local_token            = local_token_;                                \
        using ms_problem                      = ck_tile::MoeSortingProblemMp<ms_index_t,     \
                                                                             ms_weight_type, \
                                                                             mesh_type_,     \
                                                                             unroll_num,     \
                                                                             expert_masking, \
                                                                             local_token>;   \
        using kernel      = ck_tile::MoeSortingMultiPhaseKernel_P0_v1<ms_problem>;           \
        auto kargs        = kernel::MakeKargs(a);                                            \
        const dim3 grids  = kernel::GridSize(a);                                             \
        const dim3 blocks = kernel::BlockSize(a);                                            \
        return ck_tile::make_kernel<kernel::kBlockSize>(kernel{}, grids, blocks, 0, kargs);  \
    }()

#define MOE_SORTING_MP_0_V2(mesh_type_, unroll_num_, expert_masking_, local_token_)          \
    [&]() {                                                                                  \
        constexpr ck_tile::index_t unroll_num = unroll_num_;                                 \
        constexpr bool expert_masking         = expert_masking_;                             \
        constexpr bool local_token            = local_token_;                                \
        using ms_problem                      = ck_tile::MoeSortingProblemMp<ms_index_t,     \
                                                                             ms_weight_type, \
                                                                             mesh_type_,     \
                                                                             unroll_num,     \
                                                                             expert_masking, \
                                                                             local_token>;   \
        using kernel      = ck_tile::MoeSortingMultiPhaseKernel_P0_v2<ms_problem>;           \
        auto kargs        = kernel::MakeKargs(a);                                            \
        const dim3 grids  = kernel::GridSize(a);                                             \
        const dim3 blocks = kernel::BlockSize(a);                                            \
        return ck_tile::make_kernel(kernel{}, grids, blocks, 0, kargs);                      \
    }()

#define MOE_SORTING_MP_1(mesh_type_, unroll_num_, expert_masking_, local_token_)                    \
    [&]() {                                                                                         \
        constexpr ck_tile::index_t unroll_num = unroll_num_;                                        \
        constexpr bool expert_masking         = expert_masking_;                                    \
        constexpr bool local_token            = local_token_;                                       \
        using ms_problem                      = ck_tile::MoeSortingProblemMp<ms_index_t,            \
                                                                             ms_weight_type,        \
                                                                             mesh_type_,            \
                                                                             unroll_num,            \
                                                                             expert_masking,        \
                                                                             local_token>;          \
        using kernel                          = ck_tile::MoeSortingMultiPhaseKernel_P1<ms_problem>; \
        auto kargs                            = kernel::MakeKargs(a);                               \
        const dim3 grids                      = kernel::GridSize(a);                                \
        const dim3 blocks                     = kernel::BlockSize(a);                               \
        return ck_tile::make_kernel(kernel{}, grids, blocks, 0, kargs);                             \
    }()
#if MOE_SORTING_SUPPORT_LARGE_EXPERT
#define MOE_SORTING_MP_2(mesh_type_, unroll_num_, expert_masking_, local_token_)                    \
    [&]() {                                                                                         \
        constexpr ck_tile::index_t unroll_num = unroll_num_;                                        \
        constexpr bool expert_masking         = expert_masking_;                                    \
        constexpr bool local_token            = local_token_;                                       \
        using ms_problem                      = ck_tile::MoeSortingProblemMp<ms_index_t,            \
                                                                             ms_weight_type,        \
                                                                             mesh_type_,            \
                                                                             unroll_num,            \
                                                                             expert_masking,        \
                                                                             local_token>;          \
        using kernel                          = ck_tile::MoeSortingMultiPhaseKernel_P2<ms_problem>; \
        auto kargs                            = kernel::MakeKargs(a);                               \
        const dim3 grids                      = kernel::GridSize(a);                                \
        const dim3 blocks                     = kernel::BlockSize(a);                               \
        return ck_tile::make_kernel(kernel{}, grids, blocks, 0, kargs);                             \
    }()

#define MOE_SORTING_MP_3(mesh_type_, unroll_num_, expert_masking_, local_token_)                    \
    [&]() {                                                                                         \
        constexpr ck_tile::index_t unroll_num = unroll_num_;                                        \
        constexpr bool expert_masking         = expert_masking_;                                    \
        constexpr bool local_token            = local_token_;                                       \
        using ms_problem                      = ck_tile::MoeSortingProblemMp<ms_index_t,            \
                                                                             ms_weight_type,        \
                                                                             mesh_type_,            \
                                                                             unroll_num,            \
                                                                             expert_masking,        \
                                                                             local_token>;          \
        using kernel                          = ck_tile::MoeSortingMultiPhaseKernel_P3<ms_problem>; \
        auto kargs                            = kernel::MakeKargs(a);                               \
        const dim3 grids                      = kernel::GridSize(a);                                \
        const dim3 blocks                     = kernel::BlockSize(a);                               \
        return ck_tile::make_kernel(kernel{}, grids, blocks, 0, kargs);                             \
    }()
#endif

#define MOE_SORTING_MP_23(mesh_type_, unroll_num_, expert_masking_, local_token_)                    \
    [&]() {                                                                                          \
        constexpr ck_tile::index_t unroll_num = unroll_num_;                                         \
        constexpr bool expert_masking         = expert_masking_;                                     \
        constexpr bool local_token            = local_token_;                                        \
        using ms_problem                      = ck_tile::MoeSortingProblemMp<ms_index_t,             \
                                                                             ms_weight_type,         \
                                                                             mesh_type_,             \
                                                                             unroll_num,             \
                                                                             expert_masking,         \
                                                                             local_token>;           \
        using kernel                          = ck_tile::MoeSortingMultiPhaseKernel_P23<ms_problem>; \
        auto kargs                            = kernel::MakeKargs(a);                                \
        const dim3 grids                      = kernel::GridSize(a);                                 \
        const dim3 blocks                     = kernel::BlockSize(a);                                \
        const auto lds_size                   = kernel::GetSmemSize(a);                              \
        return ck_tile::make_kernel(kernel{}, grids, blocks, lds_size, kargs);                       \
    }()

#define MOR_SORTING_MP_DISPATCH_SMALL_(mesh_type_, token_vec_0_, token_vec_1_, token_vec_23_)      \
    if(t.local_expert_masking)                                                                     \
    {                                                                                              \
        if(is_local_token)                                                                         \
        {                                                                                          \
            float ave_time =                                                                       \
                ck_tile::launch_kernel(s,                                                          \
                                       MOE_SORTING_MP_0_V2(mesh_type_, token_vec_0_, true, true),  \
                                       MOE_SORTING_MP_23(mesh_type_, token_vec_23_, true, true));  \
            return ave_time;                                                                       \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            float ave_time =                                                                       \
                ck_tile::launch_kernel(s,                                                          \
                                       MOE_SORTING_MP_0_V2(mesh_type_, token_vec_0_, true, false), \
                                       MOE_SORTING_MP_23(mesh_type_, token_vec_23_, true, false)); \
            return ave_time;                                                                       \
        }                                                                                          \
    }                                                                                              \
    else                                                                                           \
    {                                                                                              \
        if(is_local_token)                                                                         \
        {                                                                                          \
            float ave_time =                                                                       \
                ck_tile::launch_kernel(s,                                                          \
                                       MOE_SORTING_MP_0_V2(mesh_type_, token_vec_0_, false, true), \
                                       MOE_SORTING_MP_23(mesh_type_, token_vec_23_, false, true)); \
            return ave_time;                                                                       \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            float ave_time = ck_tile::launch_kernel(                                               \
                s,                                                                                 \
                MOE_SORTING_MP_0_V2(mesh_type_, token_vec_0_, false, false),                       \
                MOE_SORTING_MP_23(mesh_type_, token_vec_23_, false, false));                       \
            return ave_time;                                                                       \
        }                                                                                          \
    }

#define MOR_SORTING_MP_DISPATCH_(mesh_type_, token_vec_0_, token_vec_1_, token_vec_23_)            \
    if(t.local_expert_masking)                                                                     \
    {                                                                                              \
        if(is_local_token)                                                                         \
        {                                                                                          \
            float ave_time =                                                                       \
                ck_tile::launch_kernel(s,                                                          \
                                       maybe_clear_workspace,                                      \
                                       MOE_SORTING_MP_0_V1(mesh_type_, token_vec_0_, true, true),  \
                                       MOE_SORTING_MP_1(mesh_type_, token_vec_1_, true, true),     \
                                       MOE_SORTING_MP_23(mesh_type_, token_vec_23_, true, true));  \
            return ave_time;                                                                       \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            float ave_time =                                                                       \
                ck_tile::launch_kernel(s,                                                          \
                                       maybe_clear_workspace,                                      \
                                       MOE_SORTING_MP_0_V1(mesh_type_, token_vec_0_, true, false), \
                                       MOE_SORTING_MP_1(mesh_type_, token_vec_1_, true, false),    \
                                       MOE_SORTING_MP_23(mesh_type_, token_vec_23_, true, false)); \
            return ave_time;                                                                       \
        }                                                                                          \
    }                                                                                              \
    else                                                                                           \
    {                                                                                              \
        if(is_local_token)                                                                         \
        {                                                                                          \
            float ave_time =                                                                       \
                ck_tile::launch_kernel(s,                                                          \
                                       maybe_clear_workspace,                                      \
                                       MOE_SORTING_MP_0_V1(mesh_type_, token_vec_0_, false, true), \
                                       MOE_SORTING_MP_1(mesh_type_, token_vec_1_, false, true),    \
                                       MOE_SORTING_MP_23(mesh_type_, token_vec_23_, false, true)); \
            return ave_time;                                                                       \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            float ave_time = ck_tile::launch_kernel(                                               \
                s,                                                                                 \
                maybe_clear_workspace,                                                             \
                MOE_SORTING_MP_0_V1(mesh_type_, token_vec_0_, false, false),                       \
                MOE_SORTING_MP_1(mesh_type_, token_vec_1_, false, false),                          \
                MOE_SORTING_MP_23(mesh_type_, token_vec_23_, false, false));                       \
            return ave_time;                                                                       \
        }                                                                                          \
    }

#define MOR_SORTING_CLEAR_WS_DISPATCH_(is_local_token_, block_size_, occu_)                \
    [&]() {                                                                                \
        using problem_ =                                                                   \
            ck_tile::MoeSortingClearWorkspaceProblem<is_local_token_, block_size_, occu_>; \
        using kernel      = ck_tile::MoeSortingClearWorkspaceKernel<problem_>;             \
        auto kargs        = kernel::MakeKargs(a);                                          \
        const dim3 grids  = kernel::GridSize(a);                                           \
        const dim3 blocks = kernel::BlockSize(a);                                          \
        return ck_tile::make_kernel(kernel{}, grids, blocks, 0, kargs);                    \
    }()

float moe_sorting_mp(moe_sorting_trait t, moe_sorting_args a, ck_tile::stream_config s)
{
    bool is_local_token = a.p_local_tokens != nullptr;
    if(t.weight_type == "fp32" && t.index_type == "int32")
    {
        using ms_index_t     = ck_tile::index_t;
        using ms_weight_type = float;

        auto maybe_clear_workspace = [=](const ck_tile::stream_config& s_) {
            if(t.clear_workspace_inside_api)
            {
                if(is_local_token)
                {
                    auto k = MOR_SORTING_CLEAR_WS_DISPATCH_(true, 1024, 1);
                    k(s_);
                }
                else
                {
                    auto k = MOR_SORTING_CLEAR_WS_DISPATCH_(false, 1024, 1);
                    k(s_);
                }
            }
        };
        if(!ck_tile::is_gfx12_supported() && a.tokens < 2048)
        {
            if(ck_tile::impl::moe_sorting_get_smem_size_p23(a.num_experts) >
               ck_tile::get_smem_capacity())
            {
#if MOE_SORTING_SUPPORT_LARGE_EXPERT
                if(t.local_expert_masking)
                {
                    float ave_time =
                        ck_tile::launch_kernel(s,
                                               MOE_SORTING_MP_0_V2(ms_index_t, 1, true),
                                               MOE_SORTING_MP_2(ms_index_t, 1, true),
                                               MOE_SORTING_MP_3(ms_index_t, 1, true));
                    return ave_time;
                }
                else
                {
                    float ave_time =
                        ck_tile::launch_kernel(s,
                                               MOE_SORTING_MP_0_V2(ms_index_t, 1, false),
                                               MOE_SORTING_MP_2(ms_index_t, 1, false),
                                               MOE_SORTING_MP_3(ms_index_t, 1, false));
                    return ave_time;
                }
#else
                printf("do not support large expert %d\n", a.num_experts);
                return -1;
#endif
            }
            else
            {
                ck_tile::index_t mesh_byte_size =
                    ck_tile::impl::moe_sorting_mesh_byte_size(a.tokens, a.num_experts, a.topk);
                if(mesh_byte_size == 1)
                {
                    if(a.tokens * a.topk % 4 == 0)
                    {
                        MOR_SORTING_MP_DISPATCH_SMALL_(uint8_t, 4, 16, 16)
                    }
                    else
                    {
                        MOR_SORTING_MP_DISPATCH_SMALL_(uint8_t, 1, 16, 16)
                    }
                }
                else if(mesh_byte_size == 2)
                {
#if MOE_SORTING_SUPPORT_LARGE_TOPK
                    if(a.tokens * a.topk % 4 == 0)
                    {
                        MOR_SORTING_MP_DISPATCH_SMALL_(uint16_t, 4, 8, 8)
                    }
                    else
                    {
                        MOR_SORTING_MP_DISPATCH_SMALL_(uint16_t, 1, 8, 8)
                    }
#else
                    printf("do not support large topk %d\n", a.topk);
                    return -1;
#endif
                }
                else
                {
                    MOR_SORTING_MP_DISPATCH_SMALL_(ck_tile::index_t, 1, 1, 1)
                }
            }
        }
        else
        {
            if(ck_tile::impl::moe_sorting_get_smem_size_p23(a.num_experts) >
               ck_tile::get_smem_capacity())
            {
#if MOE_SORTING_SUPPORT_LARGE_EXPERT
                if(t.local_expert_masking)
                {
                    float ave_time =
                        ck_tile::launch_kernel(s,
                                               maybe_clear_workspace,
                                               MOE_SORTING_MP_0_V1(ms_index_t, 1, true),
                                               MOE_SORTING_MP_1(ms_index_t, 1, true),
                                               MOE_SORTING_MP_2(ms_index_t, 1, true),
                                               MOE_SORTING_MP_3(ms_index_t, 1, true));
                    return ave_time;
                }
                else
                {
                    float ave_time =
                        ck_tile::launch_kernel(s,
                                               maybe_clear_workspace,
                                               MOE_SORTING_MP_0_V1(ms_index_t, 1, false),
                                               MOE_SORTING_MP_1(ms_index_t, 1, false),
                                               MOE_SORTING_MP_2(ms_index_t, 1, false),
                                               MOE_SORTING_MP_3(ms_index_t, 1, false));
                    return ave_time;
                }
#else
                printf("do not support large expert %d\n", a.num_experts);
                return -1;
#endif
            }
            else
            {
                ck_tile::index_t mesh_byte_size =
                    ck_tile::impl::moe_sorting_mesh_byte_size(a.tokens, a.num_experts, a.topk);
                if(mesh_byte_size == 1)
                {
                    if(a.tokens * a.topk % 4 == 0)
                    {
                        MOR_SORTING_MP_DISPATCH_(uint8_t, 4, 16, 16)
                    }
                    else
                    {
                        MOR_SORTING_MP_DISPATCH_(uint8_t, 1, 16, 16)
                    }
                }
                else if(mesh_byte_size == 2)
                {
#if MOE_SORTING_SUPPORT_LARGE_TOPK
                    if(a.tokens * a.topk % 4 == 0)
                    {
                        MOR_SORTING_MP_DISPATCH_(uint16_t, 4, 8, 8)
                    }
                    else
                    {
                        MOR_SORTING_MP_DISPATCH_(uint16_t, 1, 8, 8)
                    }
#else
                    printf("do not support large topk %d\n", a.topk);
                    return -1;
#endif
                }
                else
                {
                    MOR_SORTING_MP_DISPATCH_(ck_tile::index_t, 1, 1, 1)
                }
            }
        }
    }
    return -1;
}

int moe_sorting_get_workspace_size(int tokens, int num_experts, int topk, int dispatch_policy)
{
    return ck_tile::moe_sorting_get_workspace_size(tokens, num_experts, topk, dispatch_policy);
}
