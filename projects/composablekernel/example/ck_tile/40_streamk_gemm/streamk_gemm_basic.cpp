// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gemm_utils.hpp"
#include "ck_tile/ops/common.hpp"

template <typename GemmConfiguration,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccumulatorDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename CDEElementWise,
          ck_tile::StreamKReductionStrategy ReductionStrategy>
std::tuple<float, ck_tile::index_t> gemm(const ck_tile::StreamKHostArgs& args,
                                         const ck_tile::stream_config& stream_config)
{
    using GemmShape = ck_tile::TileGemmShape<ck_tile::sequence<GemmConfiguration::M_TILE,
                                                               GemmConfiguration::N_TILE,
                                                               GemmConfiguration::K_TILE>,
                                             ck_tile::sequence<GemmConfiguration::M_WARP,
                                                               GemmConfiguration::N_WARP,
                                                               GemmConfiguration::K_WARP>,
                                             ck_tile::sequence<GemmConfiguration::M_WARP_TILE,
                                                               GemmConfiguration::N_WARP_TILE,
                                                               GemmConfiguration::K_WARP_TILE>,
                                             GemmConfiguration::PERMUTE_A,
                                             GemmConfiguration::PERMUTE_B>;

    using TilePartitioner = ck_tile::
        StreamKTilePartitioner<GemmShape, ReductionStrategy, GemmConfiguration::PERSISTENT>;

    using GemmUniversalTraits =
        ck_tile::TileGemmUniversalTraits<GemmConfiguration::PAD_M,
                                         GemmConfiguration::PAD_N,
                                         GemmConfiguration::PAD_K,
                                         GemmConfiguration::DOUBLE_SMEM_BUFFER,
                                         ALayout,
                                         BLayout,
                                         ELayout,
                                         GemmConfiguration::TRANSPOSE_C,
                                         GemmConfiguration::USE_STRUCTURED_SPARSITY,
                                         GemmConfiguration::PERSISTENT,
                                         GemmConfiguration::NUM_WAVE_GROUPS,
                                         GemmConfiguration::PRESHUFFLE>;

    // We create the GEMM pipeline without specifying has_hot_loop or tail_num.
    // This is because num_loop can vary (a) per WG and (b) per iteration of the Stream-K
    // while loop. Instead, has_hot_loop and tail_num are determined in the Stream-K
    // Kernel's RunGemm function. This is a similar pattern used by grouped GEMM.
    using UniversalGemmProblem =
        ck_tile::UniversalGemmPipelineProblem<ADataType,
                                              BDataType,
                                              AccumulatorDataType,
                                              GemmShape,
                                              GemmUniversalTraits,
                                              GemmConfiguration::SCHEDULER>;

    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<UniversalGemmProblem>;

    using GemmEpilogue = ck_tile::CShuffleEpilogue<
        ck_tile::CShuffleEpilogueProblem<ADataType,
                                         BDataType,
                                         DsDataType,
                                         AccumulatorDataType,
                                         CDataType,
                                         DsLayout,
                                         ELayout,
                                         CDEElementWise,
                                         TilePartitioner::MPerBlock,
                                         TilePartitioner::NPerBlock,
                                         GemmConfiguration::M_WARP,
                                         GemmConfiguration::N_WARP,
                                         GemmConfiguration::M_WARP_TILE,
                                         GemmConfiguration::N_WARP_TILE,
                                         GemmConfiguration::K_WARP_TILE,
                                         UniversalGemmProblem::TransposeC,
                                         GemmConfiguration::NUM_WAVE_GROUPS>>;

    using Kernel = ck_tile::StreamKKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

    auto kernel_args          = Kernel::MakeKernelArgs(args);
    const auto workspace_size = Kernel::GetWorkSpaceSize(kernel_args);
    ck_tile::DeviceMem workspace_data(workspace_size);
    workspace_data.SetZero();
    kernel_args.workspace_ptr = workspace_data.GetDeviceBuffer();

    dim3 grids  = Kernel::GridSize(kernel_args.tile_partitioner);
    dim3 blocks = Kernel::BlockSize();

    if(!Kernel::IsSupportedArgument(kernel_args))
    {
        // Clear the output C tensor results after each repetition of the kernel
        hipGetErrorString(hipMemsetAsync(
            args.e_ptr, 0, args.M * args.N * sizeof(CDataType), stream_config.stream_id_));
    }

    if(stream_config.log_level_ > 0)
    {
        // Reset sk flags to zero before each repetition of the kernel
        workspace_data.SetZero();
    }

    auto reset_data_buffers = [&]() {
        if constexpr(ReductionStrategy == ck_tile::StreamKReductionStrategy::Atomic)
        {
            // Clear the output C tensor results after each repetition of the kernel
            hipGetErrorString(hipMemsetAsync(
                args.e_ptr, 0, args.M * args.N * sizeof(CDataType), stream_config.stream_id_));
        }
        else if constexpr(ReductionStrategy == ck_tile::StreamKReductionStrategy::Reduction)
        {
            // Reset sk flags to zero before each repetition of the kernel
            workspace_data.SetZero();
        }
    };

    std::function<void()> preprocess = reset_data_buffers;

    float average_time =
        ck_tile::launch_kernel_time_mask(stream_config,
                                         preprocess,
                                         ck_tile::make_kernel<GemmConfiguration::BLOCK_PER_CU>(
                                             Kernel{}, grids, blocks, 0, kernel_args));

    ck_tile::index_t num_wgs_per_tile = kernel_args.tile_partitioner.estimate_num_wgs_per_tile();
    return std::tuple{average_time, num_wgs_per_tile};
}

#include "run_gemm_example.inc"

template <typename GemmConfiguration, typename TypeConfiguration>
int runGemmExamplePrecisionType(std::string a_layout, std::string b_layout, int argc, char* argv[])
{
    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    if(a_layout == "R" && b_layout == "C")
    {
        return runGemmExampleWithLayouts<GemmConfiguration, TypeConfiguration>(
            argc, argv, Row{}, Col{}, Row{});
    }
    else
    {
        throw std::runtime_error("Unsupported layouts.");
    }

    return 0;
}

template <template <typename PrecisionType, bool IsPersistent> typename GemmConfiguration>
int runGemmExample(int argc, char* argv[])
{
    auto [result, arg_parser] = createArgs(argc, argv);
    if(!result)
        return -1;

    std::string data_type         = arg_parser.get_str("prec");
    std::string a_layout          = arg_parser.get_str("a_layout");
    std::string b_layout          = arg_parser.get_str("b_layout");
    auto persistent_data_parallel = arg_parser.get_bool("persistent_dp");

    if(data_type == "bf16")
    {
        using TypeConfiguration = StreamKGemmTypeConfiguration<ck_tile::bf16_t>;
        if(persistent_data_parallel)
        {
            return runGemmExamplePrecisionType<GemmConfiguration<ck_tile::bf16_t, true>,
                                               TypeConfiguration>(a_layout, b_layout, argc, argv);
        }
        else
        {
            return runGemmExamplePrecisionType<GemmConfiguration<ck_tile::bf16_t, false>,
                                               TypeConfiguration>(a_layout, b_layout, argc, argv);
        }
    }
    else if(data_type == "fp16")
    {
        using TypeConfiguration = StreamKGemmTypeConfiguration<ck_tile::half_t>;
        if(persistent_data_parallel)
        {
            return runGemmExamplePrecisionType<GemmConfiguration<ck_tile::half_t, true>,
                                               TypeConfiguration>(a_layout, b_layout, argc, argv);
        }
        else
        {
            return runGemmExamplePrecisionType<GemmConfiguration<ck_tile::half_t, false>,
                                               TypeConfiguration>(a_layout, b_layout, argc, argv);
        }
    }
    else if(data_type == "fp8")
    {
        using TypeConfiguration =
            StreamKGemmTypeConfiguration<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t>;
        if(persistent_data_parallel)
        {
            return runGemmExamplePrecisionType<GemmConfiguration<ck_tile::fp8_t, true>,
                                               TypeConfiguration>(a_layout, b_layout, argc, argv);
        }
        else
        {
            return runGemmExamplePrecisionType<GemmConfiguration<ck_tile::fp8_t, false>,
                                               TypeConfiguration>(a_layout, b_layout, argc, argv);
        }
    }
    else if(data_type == "bf8")
    {
        using TypeConfiguration =
            StreamKGemmTypeConfiguration<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t>;
        if(persistent_data_parallel)
        {
            return runGemmExamplePrecisionType<GemmConfiguration<ck_tile::bf8_t, true>,
                                               TypeConfiguration>(a_layout, b_layout, argc, argv);
        }
        else
        {
            return runGemmExamplePrecisionType<GemmConfiguration<ck_tile::bf8_t, false>,
                                               TypeConfiguration>(a_layout, b_layout, argc, argv);
        }
    }
    else
    {
        throw std::runtime_error("Unsupported data type for this operation !!!");
    }

    return false;
}

int main(int argc, char* argv[])
{
    return !runGemmExample<GemmConfigurationMemoryInterwave>(argc, argv);
}
