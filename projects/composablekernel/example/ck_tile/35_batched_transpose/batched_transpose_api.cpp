// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "batched_transpose_example.hpp"

namespace {

template <int32_t pipeline_id>
struct kernel_traits;

template <>
struct kernel_traits<0>
{
    template <typename ts_type, typename block_tile, typename warp_layout, bool kPadM, bool kPadN>
    using Problem =
        ck_tile::BatchedTransposeProblem<ts_type, block_tile, warp_layout, kPadM, kPadN>;
    using Policy = ck_tile::BatchedTransposePolicy;
    template <typename ts_type, typename block_tile, typename warp_layout, bool kPadM, bool kPadN>
    using Pipeline =
        ck_tile::BatchedTransposePipeline<Problem<ts_type, block_tile, warp_layout, kPadM, kPadN>,
                                          Policy>;
};

template <>
struct kernel_traits<1>
{
    template <typename ts_type, typename block_tile, typename warp_layout, bool kPadM, bool kPadN>
    using Problem =
        ck_tile::BatchedTransposeLdsProblem<ts_type, block_tile, warp_layout, kPadM, kPadN>;
    using Policy = ck_tile::BatchedTransposeLdsPolicy;
    template <typename ts_type, typename block_tile, typename warp_layout, bool kPadM, bool kPadN>
    using Pipeline = ck_tile::BatchedTransposeLdsPipeline<
        Problem<ts_type, block_tile, warp_layout, kPadM, kPadN>,
        Policy>;
};
} // namespace

template <typename InputType_,
          ck_tile::index_t BlockX_,
          ck_tile::index_t BlockY_,
          ck_tile::index_t NumWarpsX_,
          ck_tile::index_t NumWarpsY_,
          bool PadM_,
          bool PadN_,
          ck_tile::index_t PipelineId_>
struct BatchedTransposeConfig
{
    using InputType                               = InputType_;
    static constexpr ck_tile::index_t kBlockX     = BlockX_;
    static constexpr ck_tile::index_t kBlockY     = BlockY_;
    static constexpr ck_tile::index_t kNumWarpsX  = NumWarpsX_;
    static constexpr ck_tile::index_t kNumWarpsY  = NumWarpsY_;
    static constexpr bool kPadM                   = PadM_;
    static constexpr bool kPadN                   = PadN_;
    static constexpr ck_tile::index_t kPipelineId = PipelineId_;
};

template <typename Config>
float batched_transpose_dispatch(batched_transpose_kargs& a, ck_tile::stream_config& s)
{
    uint32_t dim_stride = a.height * a.width;

    a.dim_stride  = dim_stride;
    a.dim_block_h = Config::kBlockY;
    a.dim_block_w = Config::kBlockX;

    // TODO: this is fragile and slow to compile
    using kernel = ck_tile::BatchedTransposeKernel<
        typename kernel_traits<Config::kPipelineId>::template Pipeline<
            typename Config::InputType,
            ck_tile::sequence<Config::kBlockX, Config::kBlockY>,
            ck_tile::sequence<Config::kNumWarpsX, Config::kNumWarpsY>,
            Config::kPadM,
            Config::kPadN>>;

    auto kargs = kernel::MakeKargs(a);

    const dim3 grids  = kernel::GridSize(a);
    const dim3 blocks = kernel::BlockSize();

    printf("Pipeline: %d\n", Config::kPipelineId);
    printf("Grid: x=%u y=%u z=%u\n", grids.x, grids.y, grids.z);
    printf("Block: x=%u y=%u z=%u\n", blocks.x, blocks.y, blocks.z);
    printf(
        "Host args: batch=%d, height=%d, width=%d, dim_stride=%d, dim_block_h=%d, dim_block_w=%d\n",
        a.batch,
        a.height,
        a.width,
        a.dim_stride,
        a.dim_block_h,
        a.dim_block_w);
    printf("kargs: kargs.batch=%d kargs.height=%d kargs.width=%d kargs.dim_stride=%d\n",
           kargs.batch,
           kargs.height,
           kargs.width,
           kargs.dim_stride);

    printf("Launching Kernel...\n");

    float ave_time =
        ck_tile::launch_kernel(s, ck_tile::make_kernel<1>(kernel{}, grids, blocks, 0, kargs));

    printf("Kernel finished...\n");

    return ave_time;
}

// Param Comb: type_size, block_x & y, WarpNum_x & y
#define FOREACH_TRANSPOSE_PARAM(F)                          \
    F(fp8, ck_tile::fp8_t, 64, 64, 1, 1, true, true, 0)     \
    F(fp8, ck_tile::fp8_t, 64, 64, 1, 1, false, false, 0)   \
    F(fp16, ck_tile::fp16_t, 64, 64, 1, 1, true, true, 0)   \
    F(fp16, ck_tile::fp16_t, 64, 64, 1, 1, false, false, 0) \
    F(bf16, ck_tile::bf16_t, 64, 64, 1, 1, true, true, 0)   \
    F(bf16, ck_tile::bf16_t, 64, 64, 1, 1, false, false, 0) \
    F(fp8, ck_tile::fp8_t, 64, 64, 1, 1, true, true, 1)     \
    F(fp8, ck_tile::fp8_t, 64, 64, 1, 1, false, false, 1)   \
    F(fp16, ck_tile::fp16_t, 64, 64, 1, 1, true, true, 1)   \
    F(fp16, ck_tile::fp16_t, 64, 64, 1, 1, false, false, 1) \
    F(bf16, ck_tile::bf16_t, 64, 64, 1, 1, true, true, 1)   \
    F(bf16, ck_tile::bf16_t, 64, 64, 1, 1, false, false, 1)

// Macro that defines one static function per line
#define GEN_TRANSPOSE_FN(SHORT_NAME, REAL_TYPE, BX, BY, WX, WY, PADM, PADN, PIPE)          \
    static float                                                                           \
        transpose_fn_##SHORT_NAME##_##BX##_##BY##_##WX##_##WY##_##PADM##_##PADN##_v##PIPE( \
            batched_transpose_kargs& a, ck_tile::stream_config& s)                         \
    {                                                                                      \
        return batched_transpose_dispatch<                                                 \
            BatchedTransposeConfig<REAL_TYPE, BX, BY, WX, WY, PADM, PADN, PIPE>>(a, s);    \
    }

FOREACH_TRANSPOSE_PARAM(GEN_TRANSPOSE_FN)

float batched_transpose(batched_transpose_trait t,
                        batched_transpose_kargs a,
                        ck_tile::stream_config s)
{
    if(t.pipeline == "0")
    {
        if(t.type == "fp8")
        {
            if(a.height % 64 == 0 && a.width % 64 == 0)
            {
                return transpose_fn_fp8_64_64_1_1_false_false_v0(a, s);
            }
            else
            {
                return transpose_fn_fp8_64_64_1_1_true_true_v0(a, s);
            }
        }
        else if(t.type == "fp16")
        {
            if(a.height % 64 == 0 && a.width % 64 == 0)
            {
                return transpose_fn_fp16_64_64_1_1_false_false_v0(a, s);
            }
            else
            {
                return transpose_fn_fp16_64_64_1_1_true_true_v0(a, s);
            }
        }
        else if(t.type == "bf16")
        {
            if(a.height % 64 == 0 && a.width % 64 == 0)
            {
                return transpose_fn_bf16_64_64_1_1_false_false_v0(a, s);
            }
            else
            {
                return transpose_fn_bf16_64_64_1_1_true_true_v0(a, s);
            }
        }
    }
    else if(t.pipeline == "1")
    {
        if(t.type == "fp8")
        {
            if(a.height % 64 == 0 && a.width % 64 == 0)
            {
                return transpose_fn_fp8_64_64_1_1_false_false_v1(a, s);
            }
            else
            {
                return transpose_fn_fp8_64_64_1_1_true_true_v1(a, s);
            }
        }
        else if(t.type == "fp16")
        {
            if(a.height % 64 == 0 && a.width % 64 == 0)
            {
                return transpose_fn_fp16_64_64_1_1_false_false_v1(a, s);
            }
            else
            {
                return transpose_fn_fp16_64_64_1_1_true_true_v1(a, s);
            }
        }
        else if(t.type == "bf16")
        {
            if(a.height % 64 == 0 && a.width % 64 == 0)
            {
                return transpose_fn_bf16_64_64_1_1_false_false_v1(a, s);
            }
            else
            {
                return transpose_fn_bf16_64_64_1_1_true_true_v1(a, s);
            }
        }
    }

    return -1;
}
