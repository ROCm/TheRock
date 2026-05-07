// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>

#include "ck_tile/host.hpp"
#include "flatmm_basic.hpp"
#include <type_traits>

template <typename T>
constexpr const char* DataTypeToString()
{
    if constexpr(std::is_same_v<T, ck_tile::half_t>)
    {
        return "fp16";
    }
    else if constexpr(std::is_same_v<T, ck_tile::fp8_t>)
    {
        return "fp8";
    }
    else if constexpr(std::is_same_v<T, ck_tile::bf8_t>)
    {
        return "bf8";
    }
    else if constexpr(std::is_same_v<T, ck_tile::bf16_t>)
    {
        return "bf16";
    }
    else
    {
        return "unknown";
    }
}

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

// mfma_type, 0:32x32, 1:16x16
template <typename FlatmmConfig, typename T>
auto shuffle_b_v0(const ck_tile::HostTensor<T>& t)
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

template <typename FlatmmConfig, typename T>
auto shuffle_b_v1(const ck_tile::HostTensor<T>& t)
{
    assert(t.get_lengths().size() == 2);
    int n_ = t.get_lengths()[1];
    int k_ = t.get_lengths()[0];

    constexpr int MaxVecSize     = 16 / sizeof(T);
    constexpr int KLane          = ck_tile::get_warp_size() / FlatmmConfig::N_Warp_Tile;
    constexpr int ItemsPerAccess = std::min(MaxVecSize, FlatmmConfig::K_Warp_Tile / KLane);
    constexpr int NRepeat = FlatmmConfig::N_Tile / FlatmmConfig::N_Warp_Tile / FlatmmConfig::N_Warp;

    ck_tile::HostTensor<T> t_view({n_ / FlatmmConfig::N_Tile,
                                   FlatmmConfig::N_Warp,
                                   FlatmmConfig::N_Warp_Tile,
                                   NRepeat,
                                   k_ / ItemsPerAccess,
                                   ItemsPerAccess});
    std::copy(t.begin(), t.end(), t_view.begin());
    return ck_tile::reference_permute(t_view, {0, 3, 1, 4, 2, 5});
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
          typename ScaleM,
          typename ScaleN,
          bool persistent,
          typename CDEElementWise>
float flatmm_calc(const ck_tile::ScaleFlatmmHostArgs<ScaleM, ScaleN>& args,
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
                                             FlatmmConfig::NumWaveGroups,
                                             false,
                                             1,
                                             FlatmmConfig::TiledMMAPermuteN>>;

        // ToDo: Will add the codegen part to test different pipeline policies in GEMM.
        // Now we only use the BlockGemmASmemBSmemCRegV1DefaultPolicy.
        using Kernel = ck_tile::FlatmmKernel<TilePartitioner, CodegenFlatmmPipeline, GemmEpilogue>;

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
                args.M, args.K, args.stride_A, is_row_major(ALayout{})));
            ck_tile::HostTensor<BDataType> b_n(ck_tile::host_tensor_descriptor(
                args.K, args.N, args.stride_B, is_row_major(BLayout{})));

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
                if(args.k_batch > 1)
                    hipGetErrorString(hipMemsetAsync(
                        args.e_ptr, 0, args.M * args.N * sizeof(CDataType), s.stream_id_));
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

template <typename FlatmmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDatatype,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename ScaleM,
          typename ScaleN,
          bool UsePersistentKernel = false,
          typename CDEElementWise  = ck_tile::element_wise::PassThrough>
float invoke_flatmm(ck_tile::DeviceMem& a_dev_buf,
                    ck_tile::DeviceMem& b_shuffle_dev_buf,
                    ck_tile::DeviceMem& c_dev_buf,
                    ck_tile::index_t M,
                    ck_tile::index_t N,
                    ck_tile::index_t K,
                    ck_tile::index_t stride_A,
                    ck_tile::index_t stride_B,
                    ck_tile::index_t stride_C,
                    ck_tile::index_t kbatch,
                    ScaleM scale_m,
                    ScaleN scale_n,
                    int n_warmup,
                    int n_repeat)
{
    ck_tile::ScaleFlatmmHostArgs<ScaleM, ScaleN> args = {a_dev_buf.GetDeviceBuffer(),
                                                         b_shuffle_dev_buf.GetDeviceBuffer(),
                                                         {},
                                                         c_dev_buf.GetDeviceBuffer(),
                                                         kbatch,
                                                         M,
                                                         N,
                                                         K,
                                                         stride_A,
                                                         stride_B,
                                                         {},
                                                         stride_C,
                                                         scale_m,
                                                         scale_n};

    float ave_time = flatmm_calc<FlatmmConfig,
                                 ADataType,
                                 BDataType,
                                 DsDatatype,
                                 AccDataType,
                                 CDataType,
                                 ALayout,
                                 BLayout,
                                 DsLayout,
                                 CLayout,
                                 ScaleM,
                                 ScaleN,
                                 UsePersistentKernel,
                                 CDEElementWise>(
        args, ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat, true, true, 50});

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_byte =
        sizeof(ADataType) * M * K + sizeof(BDataType) * N * K + sizeof(CDataType) * M * N;
    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Run Flatmm kernel with DataType = " << DataTypeToString<ADataType>()
              << " M =" << M << " N =" << N << " K =" << K << " StrideA =" << stride_A
              << " StrideB =" << stride_B << " StrideC =" << stride_C << " : " << ave_time
              << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, " << std::endl;

    return ave_time;
}

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "256", "m dimension")
        .insert("n", "256", "n dimension")
        .insert("k", "128", "k dimension")
        .insert("a_layout", "R", "A tensor data layout - Row by default")
        .insert("b_layout", "C", "B tensor data layout - Row by default")
        .insert("c_layout", "R", "C tensor data layout - Row by default")
        .insert("stride_a", "0", "Tensor A stride")
        .insert("stride_b", "0", "Tensor B stride")
        .insert("stride_c", "0", "Tensor C stride")
        .insert("v", "1", "0. No validation, 1. Validation on CPU, 2. Validation on GPU")
        .insert("prec", "fp8", "data type. fp16/bf16/fp8/bf8")
        .insert("wave_tile", "16", "only support 16(16x16) or 32(32x32)")
        .insert("warmup", "50", "number of iterations before benchmark the kernel")
        .insert("repeat", "100", "number of iterations to benchmark the kernel")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
        .insert("split_k", "1", "splitK value")
        .insert("init", "0", "0:random, 1:linear, 2:constant(1)")
        .insert("scale", "0", "0:without scale, 1:per-token/channel scale, only for fp8/bf8")
        .insert("persistent", "0", "0: no persistent, 1: persistent kernel")
        .insert("warp_tile",
                "0",
                "0: 16x16, 1: 32x32, 2: 16x16x128 (950 only), 3: 32x32x64 (950 only)");
    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

#include "run_flatmm_example.inc"

template <template <typename PreType> typename FlatmmConfig>
int run_flatmm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    std::string data_type = arg_parser.get_str("prec");
    std::string a_layout  = arg_parser.get_str("a_layout");
    std::string b_layout  = arg_parser.get_str("b_layout");
    int scale_opt         = arg_parser.get_int("scale");
    int persistent_opt    = arg_parser.get_int("persistent");
    if(a_layout == "R" && b_layout == "C")
    {
        if(data_type == "fp16")
        {
            run_flatmm_example_with_layouts<ck_tile::half_t, FlatmmConfig<ck_tile::half_t>>(
                argc, argv, Row{}, Col{}, Row{});
        }
        else if(data_type == "bf16")
        {
            run_flatmm_example_with_layouts<ck_tile::bf16_t, FlatmmConfig<ck_tile::bf16_t>>(
                argc, argv, Row{}, Col{}, Row{});
        }
        else if(data_type == "fp8")
        {
            if(scale_opt == 0)
            {
                if(persistent_opt == 0)
                {
                    run_flatmm_example_with_layouts<ck_tile::fp8_t, FlatmmConfig<ck_tile::fp8_t>>(
                        argc, argv, Row{}, Col{}, Row{});
                }
                else
                {
                    run_flatmm_example_with_layouts<ck_tile::fp8_t,
                                                    FlatmmConfig<ck_tile::fp8_t>,
                                                    -1,
                                                    -1,
                                                    true>(argc, argv, Row{}, Col{}, Row{});
                }
            }
            else
            {
                if(persistent_opt == 0)
                {
                    run_flatmm_example_with_layouts<ck_tile::fp8_t,
                                                    FlatmmConfig<ck_tile::fp8_t>,
                                                    1,
                                                    1>(argc, argv, Row{}, Col{}, Row{});
                }
                else
                {
                    run_flatmm_example_with_layouts<ck_tile::fp8_t,
                                                    FlatmmConfig<ck_tile::fp8_t>,
                                                    1,
                                                    1,
                                                    true>(argc, argv, Row{}, Col{}, Row{});
                }
            }
        }
        else if(data_type == "bf8")
        {
            if(scale_opt == 0)
            {
                run_flatmm_example_with_layouts<ck_tile::bf8_t, FlatmmConfig<ck_tile::bf8_t>>(
                    argc, argv, Row{}, Col{}, Row{});
            }
            else
            {
                run_flatmm_example_with_layouts<ck_tile::bf8_t, FlatmmConfig<ck_tile::bf8_t>, 1, 1>(
                    argc, argv, Row{}, Col{}, Row{});
            }
        }
        else
        {
            throw std::runtime_error("Unsupported data_type!");
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
            return !run_flatmm_example<FlatmmConfig16>(argc, argv);
        }
        else if(warp_tile == 1)
        {
            return !run_flatmm_example<FlatmmConfig32>(argc, argv);
        }
        else if(warp_tile == 2)
        {
            return !run_flatmm_example<FlatmmConfig16_950>(argc, argv);
        }
        else
        {
            return !run_flatmm_example<FlatmmConfig32_950>(argc, argv);
        }
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
