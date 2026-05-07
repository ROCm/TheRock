// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>

#include "flatmm_basic.hpp"

#include "ck_tile/host.hpp"

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("Ms", "1,1,1", "m dimension")
        .insert("Ns", "5120,5120,5120", "n dimension")
        .insert("Ks", "6144,6144,6144", "k dimension")
        .insert("group_count", "3", "group count")
        .insert("a_layout", "R", "A tensor data layout - Row by default")
        .insert("b_layout", "C", "B tensor data layout - Row by default")
        .insert("c_layout", "R", "C tensor data layout - Row by default")
        .insert("stride_a", "0", "Tensor A stride")
        .insert("stride_b", "0", "Tensor B stride")
        .insert("stride_c", "0", "Tensor C stride")
        .insert("v", "1", "0. No validation, 1. Validation on CPU, 2. Validation on GPU")
        .insert("prec", "fp8", "data type. fp16/bf16/fp8/bf8")
        .insert("mode",
                "masked",
                "grouped gemm mode: [general | contiguous | masked], general by default")
        .insert("wave_tile", "16", "only support 16(16x16) or 32(32x32)")
        .insert("warmup", "50", "number of iterations before benchmark the kernel")
        .insert("repeat", "100", "number of iterations to benchmark the kernel")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
        .insert("split_k", "1", "splitK value")
        .insert("init", "0", "0:random, 1:linear, 2:constant(1)")
        .insert("scale", "0", "0:without scale, 1:per-token/channel scale, only for fp8/bf8")
        .insert("warp_tile",
                "0",
                "0: 16x16, 1: 32x32, 2: 16x16x128 (950 only), 3: 32x32x64 (950 only)");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

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
          bool persistent,
          typename CDEElementWise,
          typename KernelArguments>
float grouped_flatmm(const KernelArguments& args, const ck_tile::stream_config& s)
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
                                                               persistent,
                                                               FlatmmConfig::NumWaveGroups,
                                                               true>;

    using GemmPipelineProblem =
        ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, CodegenFlatmmShape, Traits>;

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

        using CodegenPipelineProblem = ck_tile::FlatmmPipelineProblem<ADataType,
                                                                      BDataType,
                                                                      AccDataType,
                                                                      CodegenFlatmmShape,
                                                                      CodegenGemmTraits,
                                                                      scheduler,
                                                                      has_hot_loop_v,
                                                                      tail_number_v>;

        using CodegenFlatmmPipeline =
            ck_tile::FlatmmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;

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
                                             FlatmmConfig::NumWaveGroups>>;

        // ToDo: Will add the codegen part to test different pipeline policies in GEMM.
        // Now we only use the BlockGemmASmemBSmemCRegV1DefaultPolicy.
        using Kernel =
            ck_tile::GroupedFlatmmKernel<TilePartitioner, CodegenFlatmmPipeline, GemmEpilogue>;

        auto kargs = Kernel::MakeKernelArgs(args);

        const dim3 grids      = Kernel::GridSize(kargs);
        constexpr dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        if(s.flush_cache_)
        {
            std::cout << "Flushing cache..." << std::endl;
            static constexpr ck_tile::index_t APackedSize =
                std::is_same_v<BDataType, ck_tile::pk_int4_t> ? 2 : 1;
            static constexpr ck_tile::index_t BPackedSize =
                std::is_same_v<BDataType, ck_tile::pk_int4_t> ? 2 : 1;

            ck_tile::HostTensor<ADataType> a_m(ck_tile::host_tensor_descriptor(
                args.group_count * args.M, args.K, args.stride_A, is_row_major(ALayout{})));
            ck_tile::HostTensor<BDataType> b_n(ck_tile::host_tensor_descriptor(
                args.K, args.group_count * args.N, args.stride_B, is_row_major(BLayout{})));

            auto size_a_buffer = a_m.get_element_space_size_in_bytes() / APackedSize;
            auto size_b_buffer = b_n.get_element_space_size_in_bytes() / BPackedSize;

            ck_tile::RotatingMemWrapper<ADataType, BDataType> rotating_mem(
                kargs.a_ptr, kargs.b_shuffle_ptr, s.rotating_count_, size_a_buffer, size_b_buffer);
            rotating_mem.Print();

            auto run_flush_cache = [&]() {
                // flush icache
                ck_tile::flush_icache();
                // rotating mem
                rotating_mem.Next();
                // clear c mem
                if(args.k_batch > 1)
                    hipGetErrorString(
                        hipMemsetAsync(args.e_ptr,
                                       0,
                                       args.group_count * args.M * args.N * sizeof(CDataType),
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

#include "run_grouped_flatmm_example.inc"

template <template <typename PreType> typename FlatmmConfig>
int run_grouped_flatmm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    std::string data_type = arg_parser.get_str("prec");
    std::string mode      = arg_parser.get_str("mode");
    std::string a_layout  = arg_parser.get_str("a_layout");
    std::string b_layout  = arg_parser.get_str("b_layout");

    if(a_layout == "R" && b_layout == "C")
    {
        if(mode == "contiguous")
        {
            if(data_type == "fp16")
            {
                run_contiguous_grouped_flatmm_example_with_layouts<ck_tile::half_t,
                                                                   FlatmmConfig<ck_tile::half_t>>(
                    argc, argv, Row{}, Col{}, Row{});
            }
            else if(data_type == "bf16")
            {
                run_contiguous_grouped_flatmm_example_with_layouts<ck_tile::bf16_t,
                                                                   FlatmmConfig<ck_tile::bf16_t>>(
                    argc, argv, Row{}, Col{}, Row{});
            }
            else if(data_type == "fp8")
            {
                run_contiguous_grouped_flatmm_example_with_layouts<ck_tile::fp8_t,
                                                                   FlatmmConfig<ck_tile::fp8_t>>(
                    argc, argv, Row{}, Col{}, Row{});
            }
            else if(data_type == "bf8")
            {
                run_contiguous_grouped_flatmm_example_with_layouts<ck_tile::bf8_t,
                                                                   FlatmmConfig<ck_tile::bf8_t>>(
                    argc, argv, Row{}, Col{}, Row{});
            }
            else
            {
                throw std::runtime_error("Unsupported data_type!");
            }
        }
        else if(mode == "masked")
        {

            if(data_type == "fp16")
            {
                run_masked_grouped_flatmm_example_with_layouts<ck_tile::half_t,
                                                               FlatmmConfig<ck_tile::half_t>>(
                    argc, argv, Row{}, Col{}, Row{});
            }
            else if(data_type == "bf16")
            {
                run_masked_grouped_flatmm_example_with_layouts<ck_tile::bf16_t,
                                                               FlatmmConfig<ck_tile::bf16_t>>(
                    argc, argv, Row{}, Col{}, Row{});
            }
            else if(data_type == "fp8")
            {
                run_masked_grouped_flatmm_example_with_layouts<ck_tile::fp8_t,
                                                               FlatmmConfig<ck_tile::fp8_t>>(
                    argc, argv, Row{}, Col{}, Row{});
            }
            else if(data_type == "bf8")
            {
                run_masked_grouped_flatmm_example_with_layouts<ck_tile::bf8_t,
                                                               FlatmmConfig<ck_tile::bf8_t>>(
                    argc, argv, Row{}, Col{}, Row{});
            }
            else
            {
                throw std::runtime_error("Unsupported data_type!");
            }
        }
        else
        {
            throw std::runtime_error("Unsupported mode!");
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
            return !run_grouped_flatmm_example<FlatmmConfig16>(argc, argv);
        }
        // else if(warp_tile == 1)
        // {
        //     return !run_grouped_flatmm_example<FlatmmConfig32>(argc, argv);
        // }
        // else if(warp_tile == 2)
        // {
        //     return !run_grouped_flatmm_example<FlatmmConfig16_950>(argc, argv);
        // }
        // else
        // {
        //     return !run_grouped_flatmm_example<FlatmmConfig32_950>(argc, argv);
        // }
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
