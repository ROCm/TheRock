// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>
#include <memory>

#include "moe_flatmm.hpp"

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

template <typename FlatmmConfig, typename T>
auto flatmm_shuffle_b(const ck_tile::HostTensor<T>& t)
{
    assert(t.get_lengths().size() == 2);
    int n_ = t.get_lengths()[1];
    int k_ = t.get_lengths()[0];

    constexpr int MaxVecSize     = 16 / sizeof(T);
    constexpr int KLane          = ck_tile::get_warp_size() / FlatmmConfig::N_Warp_Tile;
    constexpr int ItemsPerAccess = std::min(MaxVecSize, FlatmmConfig::K_Warp_Tile / KLane);

    ck_tile::HostTensor<T> t_view({n_ / FlatmmConfig::N_Warp_Tile,
                                   FlatmmConfig::N_Warp_Tile,
                                   k_ / ItemsPerAccess,
                                   ItemsPerAccess});
    std::copy(t.begin(), t.end(), t_view.begin());
    return ck_tile::reference_permute(t_view, {0, 2, 1, 3});
}

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    // Calculate thresholds
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
    // Calculate error due to split_k accumulation
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
        max_accumulated_value, kbatch);
    // Use higher threshold
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
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
          typename ScaleM,
          typename ScaleN>
float moe_gemm(const ck_tile::MoeFlatmmHostArgs<ScaleM, ScaleN>& args,
               const ck_tile::stream_config& s)
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

    if constexpr(moe_kind == ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up)
    {
        static_assert(
            FlatmmConfig::N_Tile % (FlatmmConfig::N_Warp * FlatmmConfig::N_Warp_Tile * 2) == 0,
            "requires NRepeat is multiple of 2 for FFN_gemm1_gate_up");
    }

    using GemmPipelineProblem =
        ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, CodegenFlatmmShape, Traits>;

    using BaseGemmPipeline = ck_tile::BaseFlatmmPipelineAGmemBGmemCRegV1<GemmPipelineProblem>;

    const ck_tile::index_t k_grain     = args.k_batch * FlatmmConfig::K_Tile;
    const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain * FlatmmConfig::K_Tile;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

    const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
        constexpr bool has_hot_loop_v = has_hot_loop_.value;
        constexpr auto tail_number_v  = tail_number_.value;
        constexpr auto scheduler      = FlatmmConfig::Scheduler;

        using CodegenPipelineProblem = ck_tile::FlatmmPipelineProblem<ADataType,
                                                                      BDataType,
                                                                      AccDataType,
                                                                      CodegenFlatmmShape,
                                                                      CodegenGemmTraits,
                                                                      scheduler,
                                                                      has_hot_loop_v,
                                                                      tail_number_v>;

        constexpr int BlockedXDLN_PerWarp = moe_kind == ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up
                                                ? 2
                                                : 1; // determined by scale shuffle pattern

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
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

        using CodegenFlatmmPipeline =
            ck_tile::MoeFlatmmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;

        using Kernel = ck_tile::
            MoeFlatmmKernel<TilePartitioner, CodegenFlatmmPipeline, GemmEpilogue, moe_kind>;

        auto kargs = Kernel::MakeKernelArgs(args);

        const dim3 grids      = Kernel::GridSize(kargs);
        constexpr dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args:" << CodegenFlatmmShape::GetName() << "\n"
                      << "Shape: " << CodegenFlatmmShape::GetName() << "\n"
                      << "problem: " << CodegenPipelineProblem::GetName() << "\n"
                      << "pipeline: " << CodegenFlatmmPipeline::GetName() << "\n"
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }

        if(s.flush_cache_)
        {
            std::cout << "Flushing cache..." << std::endl;
            static constexpr ck_tile::index_t APackedSize =
                std::is_same_v<BDataType, ck_tile::pk_int4_t> ? 2 : 1;
            static constexpr ck_tile::index_t BPackedSize =
                std::is_same_v<BDataType, ck_tile::pk_int4_t> ? 2 : 1;

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
            return ck_tile::launch_kernel_time_mask(
                s,
                run_flush_cache,
                ck_tile::make_kernel<FlatmmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        }
        else
        {
            return ck_tile::launch_kernel(
                s,
                ck_tile::make_kernel<FlatmmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        }
    };

    float ave_time = BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
    return ave_time;
}

#include "run_moe_flatmm_example.inc"

template <template <typename PreType> typename FlatmmConfig>
int run_moe_flatmm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
    {
        return -1;
    }

    const std::string a_layout = arg_parser.get_str("a_layout");
    const std::string b_layout = arg_parser.get_str("b_layout");

    const std::string prec_type = arg_parser.get_str("prec");

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    if(a_layout == "R" && b_layout == "C")
    {
        const std::string gemm_kind = arg_parser.get_str("gemm_kind");
        if(gemm_kind == "gemm1_gate_up")
        {
            if(prec_type == "fp8")
            {
                return run_moe_gemm_example_with_layouts<
                    ck_tile::fp8_t,
                    FlatmmConfig<ck_tile::fp8_t>,
                    ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up>(argc, argv, Row{}, Col{}, Row{});
            }
            else if(prec_type == "bf8")
            {
                return run_moe_gemm_example_with_layouts<
                    ck_tile::bf8_t,
                    FlatmmConfig<ck_tile::bf8_t>,
                    ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up>(argc, argv, Row{}, Col{}, Row{});
            }
            else if(prec_type == "bf16")
            {
                return run_moe_gemm_example_with_layouts<
                    ck_tile::bfloat16_t,
                    FlatmmConfig<ck_tile::bfloat16_t>,
                    ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up>(argc, argv, Row{}, Col{}, Row{});
            }
            else if(prec_type == "fp16")
            {
                return run_moe_gemm_example_with_layouts<
                    ck_tile::half_t,
                    FlatmmConfig<ck_tile::half_t>,
                    ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up>(argc, argv, Row{}, Col{}, Row{});
            }
            else
            {
                throw std::runtime_error("Unsupported precision type for gemm1_gate_up!");
            }
        }
        else if(gemm_kind == "gemm1_gate_only")
        {
            if(prec_type == "fp8")
            {
                return run_moe_gemm_example_with_layouts<
                    ck_tile::fp8_t,
                    FlatmmConfig<ck_tile::fp8_t>,
                    ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_only>(argc, argv, Row{}, Col{}, Row{});
            }
            else if(prec_type == "bf8")
            {
                return run_moe_gemm_example_with_layouts<
                    ck_tile::bf8_t,
                    FlatmmConfig<ck_tile::bf8_t>,
                    ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_only>(argc, argv, Row{}, Col{}, Row{});
            }
            else if(prec_type == "bf16")
            {
                return run_moe_gemm_example_with_layouts<
                    ck_tile::bfloat16_t,
                    FlatmmConfig<ck_tile::bfloat16_t>,
                    ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_only>(argc, argv, Row{}, Col{}, Row{});
            }
            else if(prec_type == "fp16")
            {
                return run_moe_gemm_example_with_layouts<
                    ck_tile::half_t,
                    FlatmmConfig<ck_tile::half_t>,
                    ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_only>(argc, argv, Row{}, Col{}, Row{});
            }
            else
            {
                throw std::runtime_error("Unsupported precision type for gemm1_gate_up!");
            }
        }
        else if(gemm_kind == "gemm2")
        {
            if(prec_type == "fp8")
            {
                return run_moe_gemm_example_with_layouts<ck_tile::fp8_t,
                                                         FlatmmConfig<ck_tile::fp8_t>,
                                                         ck_tile::MoeFlatmmKind::kFFN_gemm2>(
                    argc, argv, Row{}, Col{}, Row{});
            }
            else if(prec_type == "bf8")
            {
                return run_moe_gemm_example_with_layouts<ck_tile::bf8_t,
                                                         FlatmmConfig<ck_tile::bf8_t>,
                                                         ck_tile::MoeFlatmmKind::kFFN_gemm2>(
                    argc, argv, Row{}, Col{}, Row{});
            }
            else if(prec_type == "bf16")
            {
                return run_moe_gemm_example_with_layouts<ck_tile::bfloat16_t,
                                                         FlatmmConfig<ck_tile::bfloat16_t>,
                                                         ck_tile::MoeFlatmmKind::kFFN_gemm2>(
                    argc, argv, Row{}, Col{}, Row{});
            }
            else if(prec_type == "fp16")
            {
                return run_moe_gemm_example_with_layouts<ck_tile::half_t,
                                                         FlatmmConfig<ck_tile::half_t>,
                                                         ck_tile::MoeFlatmmKind::kFFN_gemm2>(
                    argc, argv, Row{}, Col{}, Row{});
            }
            else
            {
                throw std::runtime_error("Unsupported precision type for gemm1_gate_up!");
            }
        }
        else
        {
            throw std::runtime_error("Unrecoginized gemm_kind parameter, only accept value "
                                     "[gemm1_gate_only | gemm1_gate_up | gemm2]");
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
            return !run_moe_flatmm_example<FlatmmConfig16>(argc, argv);
        }
        else if(warp_tile == 1)
        {
            return !run_moe_flatmm_example<FlatmmConfig32>(argc, argv);
        }
        else if(warp_tile == 2)
        {
            return !run_moe_flatmm_example<FlatmmConfig16_950>(argc, argv);
        }
        else
        {
            return !run_moe_flatmm_example<FlatmmConfig32_950>(argc, argv);
        }
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
