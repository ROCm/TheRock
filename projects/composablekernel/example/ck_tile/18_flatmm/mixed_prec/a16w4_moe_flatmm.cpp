// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>
#include <memory>

#include "a16w4_moe_flatmm.hpp"

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/flatmm.hpp"
#include "ck_tile/ops/moe_flatmm.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/reference/reference_moe_gemm.hpp"

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

// gemm1
//   operand-A = [num_token, d_model]
//   operand-B = [num_expert, hidden, d_model]
//   operand-C = [num_token, topk, hidden]

// gemm2
//   operand-A = [num_token, topk, hidden]
//   operand-B = [num_expert, d_model, hidden]
//   operand-C = [num_token, d_model]

template <typename FlatmmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDatatype,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ck_tile::MoeFlatmmKind moe_kind = ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_only,
          typename CDEElementWise         = ck_tile::element_wise::PassThrough,
          typename MoeFlatmmHostArgs>
float a16w4_moe_gemm(const MoeFlatmmHostArgs& args, const ck_tile::stream_config& s)
{
    using CodegenFlatmmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<FlatmmConfig::M_Tile, FlatmmConfig::N_Tile, FlatmmConfig::K_Tile>,
        ck_tile::sequence<FlatmmConfig::M_Warp, FlatmmConfig::N_Warp, FlatmmConfig::K_Warp>,
        ck_tile::sequence<FlatmmConfig::M_Warp_Tile,
                          FlatmmConfig::N_Warp_Tile,
                          FlatmmConfig::K_Warp_Tile>>;

    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<CodegenFlatmmShape,
                                                   FlatmmConfig::TileParitionerGroupNum,
                                                   FlatmmConfig::TileParitionerM01>;

    using Traits = ck_tile::TileGemmTraits<FlatmmConfig::kPadM,
                                           FlatmmConfig::kPadN,
                                           FlatmmConfig::kPadK,
                                           ALayout,
                                           BLayout,
                                           ELayout,
                                           FlatmmConfig::NumWaveGroups>;

    using CodegenGemmTraits = ck_tile::TileGemmUniversalTraits<FlatmmConfig::kPadM,
                                                               FlatmmConfig::kPadN,
                                                               FlatmmConfig::kPadK,
                                                               FlatmmConfig::DoubleSmemBuffer,
                                                               ALayout,
                                                               BLayout,
                                                               ELayout,
                                                               FlatmmConfig::TransposeC,
                                                               FlatmmConfig::UseStructuredSparsity,
                                                               false, // UsePersistentKernel_
                                                               FlatmmConfig::NumWaveGroups,
                                                               true>; // Preshuffle_

    constexpr bool MXFP4_Pipeline = std::is_same_v<BDataType, ck_tile::pk_fp4_t>;

    if constexpr(!MXFP4_Pipeline && moe_kind == ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up)
    {
        static_assert(
            FlatmmConfig::N_Tile % (FlatmmConfig::N_Warp * FlatmmConfig::N_Warp_Tile * 2) == 0,
            "requires NRepeat is multiple of 2 for FFN_gemm1_gate_up");
    }

    using ComputeDataType = ADataType;
    static_assert(sizeof(ComputeDataType) >= sizeof(BDataType),
                  "mixed_prec_flatmm requires ADataType is a wider type than BDataType");

    using GemmPipelineProblem = ck_tile::GemmPipelineProblem<ComputeDataType,
                                                             ComputeDataType,
                                                             AccDataType,
                                                             CodegenFlatmmShape,
                                                             Traits>;

    using BaseGemmPipeline = ck_tile::BaseFlatmmPipelineAGmemBGmemCRegV1<GemmPipelineProblem>;

    const ck_tile::index_t k_grain     = args.k_batch * FlatmmConfig::K_Tile;
    const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain * FlatmmConfig::K_Tile;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);
    float ave_time{0};

    const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
        constexpr bool has_hot_loop_v = has_hot_loop_.value;
        constexpr auto tail_number_v  = tail_number_.value;
        constexpr auto scheduler      = FlatmmConfig::Scheduler;

        using CodegenPipelineProblem =
            std::conditional_t<MXFP4_Pipeline,
                               ck_tile::F16xMXF4FlatmmPipelineProblem<ADataType,
                                                                      BDataType,
                                                                      AccDataType,
                                                                      CodegenFlatmmShape,
                                                                      CodegenGemmTraits,
                                                                      scheduler,
                                                                      has_hot_loop_v,
                                                                      tail_number_v>,
                               ck_tile::FlatmmPipelineProblem<ADataType,
                                                              BDataType,
                                                              AccDataType,
                                                              CodegenFlatmmShape,
                                                              CodegenGemmTraits,
                                                              scheduler,
                                                              has_hot_loop_v,
                                                              tail_number_v>>;

        constexpr int BlockedXDLN_PerWarp = 2; // determined by scale shuffle pattern

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ComputeDataType,
                                             ComputeDataType,
                                             DsDatatype,
                                             AccDataType,
                                             CDataType,
                                             DsLayout,
                                             ELayout,
                                             CDEElementWise,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             FlatmmConfig::M_Warp,
                                             FlatmmConfig::N_Warp,
                                             FlatmmConfig::M_Warp_Tile,
                                             FlatmmConfig::N_Warp_Tile,
                                             FlatmmConfig::K_Warp_Tile,
                                             CodegenPipelineProblem::TransposeC,
                                             FlatmmConfig::NumWaveGroups,
                                             false,
                                             1,
                                             FlatmmConfig::TiledMMAPermuteN,
                                             BlockedXDLN_PerWarp>>;

        using CodegenFlatmmPipeline = std::conditional_t<
            MXFP4_Pipeline,
            ck_tile::F16xMXF4FlatmmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>,
            ck_tile::MoeFlatmmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>>;
        using FusedAct =
            std::conditional_t<MXFP4_Pipeline, ck_tile::moe::Swiglu, ck_tile::moe::MoeSilu>;

        using Kernel = ck_tile::MoeFlatmmKernel<TilePartitioner,
                                                CodegenFlatmmPipeline,
                                                GemmEpilogue,
                                                moe_kind,
                                                FusedAct>;

        auto kargs = Kernel::MakeKernelArgs(args);

        const dim3 grids      = Kernel::GridSize(kargs);
        constexpr dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel " << Kernel::GetName() << "\n"
                      << "with args:" << CodegenFlatmmShape::GetName() << "\n"
                      << "Shape: " << CodegenFlatmmShape::GetName() << "\n"
                      << "problem: " << CodegenPipelineProblem::GetName() << "\n"
                      << "pipeline: " << CodegenFlatmmPipeline::GetName() << "\n"
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << "\n"
                      << "k_batch: " << kargs.k_batch << std::endl;
        }

        if(s.flush_cache_)
        {
            std::cout << "Flushing cache..." << std::endl;
            static constexpr ck_tile::index_t APackedSize =
                std::is_same_v<BDataType, ck_tile::pk_int4_t> ||
                        std::is_same_v<BDataType, ck_tile::pk_fp4_t>
                    ? 2
                    : 1;
            static constexpr ck_tile::index_t BPackedSize =
                std::is_same_v<BDataType, ck_tile::pk_int4_t> ||
                        std::is_same_v<BDataType, ck_tile::pk_fp4_t>
                    ? 2
                    : 1;

            ck_tile::HostTensor<ADataType> a_m(ck_tile::host_tensor_descriptor(
                moe_kind == ck_tile::MoeFlatmmKind::kFFN_gemm2 ? args.NumTokens * args.TopK
                                                               : args.NumTokens,
                args.K,
                args.stride_A,
                is_row_major(ALayout{})));
            ck_tile::HostTensor<BDataType> b_n(ck_tile::host_tensor_descriptor(
                args.K, args.N * args.NumExperts, args.stride_B, is_row_major(BLayout{})));

            const int outputN =
                moe_kind == ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up ? args.N / 2 : args.N;

            auto size_a_buffer = a_m.get_element_space_size_in_bytes() / APackedSize;
            auto size_b_buffer = b_n.get_element_space_size_in_bytes() / BPackedSize;

            ck_tile::RotatingMemWrapper<ADataType, BDataType> rotating_mem(
                kargs.a_ptr, kargs.b_ptr, s.rotating_count_, size_a_buffer, size_b_buffer);
            rotating_mem.Print();

            auto run_flush_cache = [&]() {
                // flush icache
                ck_tile::flush_icache();
                // rotating mem
                rotating_mem.Next();
                // clear c mem
                if(moe_kind == ck_tile::MoeFlatmmKind::kFFN_gemm2)
                    hipGetErrorString(hipMemsetAsync(
                        args.e_ptr, 0, args.NumTokens * args.N * sizeof(CDataType), s.stream_id_));
                else if(args.k_batch > 1)
                    hipGetErrorString(
                        hipMemsetAsync(args.e_ptr,
                                       0,
                                       args.NumTokens * args.TopK * outputN * sizeof(CDataType),
                                       s.stream_id_));
            };
            ave_time = ck_tile::launch_kernel_time_mask(
                s,
                run_flush_cache,
                ck_tile::make_kernel<FlatmmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        }
        else
        {
            ave_time = ck_tile::launch_kernel(
                s,
                ck_tile::make_kernel<FlatmmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        }
        return ave_time;
    };

    BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
    return ave_time;
}

template <class FlatmmConfig, ck_tile::MoeFlatmmKind moe_kind, class IterSrc, class IterDst>
void shuffle_mxfp4_weight(const IterSrc src, IterDst dst, int experts_cnt, int N, int K)
{
    int KPack = 16;
    int NLane = FlatmmConfig::N_Warp_Tile;
    int KLane = 64 / NLane;
    int K_pk  = K / 2;
    int K0    = K_pk / (KLane * KPack);
    // K -> K0 KLane KPack
    // N -> N0 NLane
    // N, K -> N0 K0 KLane NLane KPack
    int tempk;

    if constexpr(moe_kind == ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up)
    {
        int up_stride = N / 2 / NLane;

        for(long eid = 0; eid < experts_cnt; ++eid)
        {
            for(int n = 0; n < N; ++n)
            {
                for(int k = 0; k < K_pk; ++k)
                {
                    int n0 = n / NLane;
                    int n1 = n % NLane;

                    // interleave gate and up part with granularity is 16.
                    int n0_interleave = n >= N / 2 ? (n0 - up_stride) * 2 + 1 : // up part
                                            n0 * 2;                             // gate part

                    int k0 = k / (KLane * KPack);
                    tempk  = k % (KLane * KPack);
                    int k1 = tempk / KPack;
                    int k2 = tempk % KPack;

                    long outputIndex = eid * N * K_pk + n0_interleave * KPack * NLane * KLane * K0 +
                                       k0 * KPack * NLane * KLane + k1 * KPack * NLane +
                                       n1 * KPack + k2;

                    dst[outputIndex] = src[eid * N * K_pk + n * K_pk + k];
                }
            }
        }
    }
    else
    {
        for(long eid = 0; eid < experts_cnt; ++eid)
        {
            for(int n = 0; n < N; ++n)
            {
                for(int k = 0; k < K_pk; ++k)
                {
                    int n0 = n / NLane;
                    int n1 = n % NLane;

                    int k0 = k / (KLane * KPack);
                    tempk  = k % (KLane * KPack);
                    int k1 = tempk / KPack;
                    int k2 = tempk % KPack;

                    long outputIndex = eid * N * K_pk + n0 * KPack * NLane * KLane * K0 +
                                       k0 * KPack * NLane * KLane + k1 * KPack * NLane +
                                       n1 * KPack + k2;

                    dst[outputIndex] = src[eid * N * K_pk + n * K_pk + k];
                }
            }
        }
    }
}

template <typename FlatmmConfig, ck_tile::MoeFlatmmKind moe_kind, typename T>
auto shuffle_mxfp4_scale(const ck_tile::HostTensor<T>& scale, int experts_cnt)
{
    assert(scale.get_lengths().size() == 2);
    int n_ = scale.get_lengths()[1];
    int k_ = scale.get_lengths()[0];

    int k_per_expert = k_ / experts_cnt;

    constexpr int K_Pack       = 2;  // fixed for mxfp4
    constexpr int N_Pack       = 2;  // fixed for mxfp4
    constexpr int GranularityK = 32; // fixed for mxfp4

    constexpr int K_Lane = 64 / FlatmmConfig::N_Warp_Tile; // 4

    static_assert(FlatmmConfig::N_Warp_Tile == 16, "only support XDL_N == 16");
    static_assert(FlatmmConfig::N_Repeat % N_Pack == 0);
    static_assert(FlatmmConfig::K_Tile % (K_Pack * K_Lane * GranularityK) == 0);

    if constexpr(moe_kind == ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up)
    {
        ck_tile::HostTensor<T> shfl_scale({
            experts_cnt,
            k_per_expert / K_Pack / K_Lane,
            K_Pack,
            K_Lane,
            N_Pack, // N_Pack = 2 is composed of Gate + Up.
            n_ / FlatmmConfig::N_Warp_Tile / N_Pack,
            FlatmmConfig::N_Warp_Tile,
        });
        std::copy(scale.begin(), scale.end(), shfl_scale.begin());
        return ck_tile::reference_permute(shfl_scale, {0, 5, 1, 3, 6, 2, 4});
    }
    else
    {
        ck_tile::HostTensor<T> shfl_scale({
            experts_cnt,
            k_per_expert / K_Pack / K_Lane,
            K_Pack,
            K_Lane,
            n_ / FlatmmConfig::N_Warp_Tile / N_Pack,
            N_Pack,
            FlatmmConfig::N_Warp_Tile,
        });
        std::copy(scale.begin(), scale.end(), shfl_scale.begin());
        return ck_tile::reference_permute(shfl_scale, {0, 4, 1, 3, 6, 2, 5});
    }
}

#include "run_a16w4_moe_flatmm_example.inc"

template <typename FlatmmConfig>
int run_a16w4_moe_flatmm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
    {
        return -1;
    }

    const std::string a_layout = arg_parser.get_str("a_layout");
    const std::string b_layout = arg_parser.get_str("b_layout");

    const std::string mixed_prec = arg_parser.get_str("mixed_prec");

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    if(a_layout == "R" && b_layout == "C")
    {
        const std::string gemm_kind = arg_parser.get_str("gemm_kind");
        if(gemm_kind == "gemm1_gate_up")
        {
            if(mixed_prec == "fp16xfp4")
            {
                return run_a16w4_moe_gemm_example_with_layouts<
                    ck_tile::half_t,
                    ck_tile::pk_fp4_t,
                    FlatmmConfig,
                    ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up>(argc, argv, Row{}, Col{}, Row{});
            }
            else if(mixed_prec == "bf16xfp4")
            {
                return run_a16w4_moe_gemm_example_with_layouts<
                    ck_tile::bfloat16_t,
                    ck_tile::pk_fp4_t,
                    FlatmmConfig,
                    ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up>(argc, argv, Row{}, Col{}, Row{});
            }
            else
            {
                throw std::runtime_error("Unsupported precision type for gemm1_gate_up!");
            }
        }
        else if(gemm_kind == "gemm2")
        {
            if(mixed_prec == "fp16xfp4")
            {
                return run_a16w4_moe_gemm_example_with_layouts<ck_tile::half_t,
                                                               ck_tile::pk_fp4_t,
                                                               FlatmmConfig,
                                                               ck_tile::MoeFlatmmKind::kFFN_gemm2>(
                    argc, argv, Row{}, Col{}, Row{});
            }
            else if(mixed_prec == "bf16xfp4")
            {
                return run_a16w4_moe_gemm_example_with_layouts<ck_tile::bfloat16_t,
                                                               ck_tile::pk_fp4_t,
                                                               FlatmmConfig,
                                                               ck_tile::MoeFlatmmKind::kFFN_gemm2>(
                    argc, argv, Row{}, Col{}, Row{});
            }
            else
            {
                throw std::runtime_error("Unsupported precision type for gemm2!");
            }
        }
        else if(gemm_kind == "gemm1_split_k")
        {
            if(mixed_prec == "fp16xfp4")
            {
                return run_a16w4_moe_gemm_example_with_layouts<
                    ck_tile::half_t,
                    ck_tile::pk_fp4_t,
                    FlatmmConfig,
                    ck_tile::MoeFlatmmKind::kFFN_gemm1_split_k>(argc, argv, Row{}, Col{}, Row{});
            }
            else if(mixed_prec == "bf16xfp4")
            {
                return run_a16w4_moe_gemm_example_with_layouts<
                    ck_tile::bfloat16_t,
                    ck_tile::pk_fp4_t,
                    FlatmmConfig,
                    ck_tile::MoeFlatmmKind::kFFN_gemm1_split_k>(argc, argv, Row{}, Col{}, Row{});
            }
            else
            {
                throw std::runtime_error("Unsupported precision type for gemm1_split_k!");
            }
        }
        else
        {
            throw std::runtime_error("Unrecoginized gemm_kind parameter, only accept value "
                                     "[gemm1_gate_up | gemm1_split_k | gemm2]");
        }
    }
    else
    {
        throw std::runtime_error("Unsupported data layout configuration for A,B and C tensors!");
    }
    return -1;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return EXIT_FAILURE;

    try
    {
        int warp_tile = arg_parser.get_int("warp_tile");
        if(warp_tile == 0)
        {
            return !run_a16w4_moe_flatmm_example<A16W4_FlatmmConfig16>(argc, argv);
        }
        // else if(warp_tile == 1)
        // {
        //     return !run_a16w4_moe_flatmm_example<A16W4_FlatmmConfig16_950>(argc, argv);
        // }
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
