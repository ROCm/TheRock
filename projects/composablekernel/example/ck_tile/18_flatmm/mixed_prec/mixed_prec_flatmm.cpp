// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>
#include <type_traits>

#include "ck_tile/host.hpp"
#include "mixed_prec_flatmm.hpp"

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
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
float mixed_prec_flatmm_calc(const ck_tile::ScaleFlatmmHostArgs<ScaleM, ScaleN>& args,
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

        constexpr int BlockedXDLN_PerWarp = 2; // determined by scale shuffle pattern

        using CodegenPipelineProblem = ck_tile::F16xMXF4FlatmmPipelineProblem<ADataType,
                                                                              BDataType,
                                                                              AccDataType,
                                                                              CodegenFlatmmShape,
                                                                              CodegenGemmTraits,
                                                                              scheduler,
                                                                              has_hot_loop_v,
                                                                              tail_number_v>;

        using CodegenFlatmmPipeline =
            ck_tile::F16xMXF4FlatmmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;

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
                                             false, // FixedVectorSize
                                             1,     // VectorSizeC
                                             FlatmmConfig::TiledMMAPermuteN,
                                             BlockedXDLN_PerWarp>>;

        using Kernel =
            ck_tile::F16xMXF4FlatmmKernel<TilePartitioner, CodegenFlatmmPipeline, GemmEpilogue>;

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
            constexpr ck_tile::index_t APackedSize = ck_tile::numeric_traits<ADataType>::PackedSize;
            constexpr ck_tile::index_t BPackedSize = ck_tile::numeric_traits<BDataType>::PackedSize;

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
          typename ScaleN,
          bool UsePersistentKernel = false,
          typename CDEElementWise  = ck_tile::element_wise::PassThrough>
float invoke_mixed_prec_flatmm(ck_tile::DeviceMem& a_dev_buf,
                               ck_tile::DeviceMem& b_shuffle_dev_buf,
                               ck_tile::DeviceMem& c_dev_buf,
                               ck_tile::index_t M,
                               ck_tile::index_t N,
                               ck_tile::index_t K,
                               ck_tile::index_t stride_A,
                               ck_tile::index_t stride_B,
                               ck_tile::index_t stride_C,
                               ck_tile::index_t kbatch,
                               ScaleN dequant_scale_n,
                               int n_warmup,
                               int n_repeat)
{
    // Activation has no scale
    using ActScaleType = ck_tile::FlatmmScalePointer<-1>;

    ck_tile::ScaleFlatmmHostArgs<ActScaleType, ScaleN> args = {a_dev_buf.GetDeviceBuffer(),
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
                                                               {},
                                                               dequant_scale_n};

    float ave_time = mixed_prec_flatmm_calc<FlatmmConfig,
                                            ADataType,
                                            BDataType,
                                            DsDatatype,
                                            AccDataType,
                                            CDataType,
                                            ALayout,
                                            BLayout,
                                            DsLayout,
                                            CLayout,
                                            ActScaleType,
                                            ScaleN,
                                            UsePersistentKernel,
                                            CDEElementWise>(
        args, ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat, true, true, 50});

    constexpr int PackedSize = ck_tile::numeric_traits<BDataType>::PackedSize;

    std::size_t flop     = std::size_t(2) * M * N * K;
    std::size_t num_byte = sizeof(ADataType) * M * K + sizeof(BDataType) * N * K / PackedSize +
                           sizeof(CDataType) * M * N;
    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Run A16W4_Flatmm kernel " << " M =" << M << " N =" << N << " K =" << K
              << " StrideA =" << stride_A << " StrideB =" << stride_B << " StrideC =" << stride_C
              << " : " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << std::endl;

    return ave_time;
}

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "256", "m dimension")
        .insert("n", "256", "n dimension")
        .insert("k", "512", "k dimension")
        .insert("a_layout", "R", "A tensor data layout - Row by default")
        .insert("b_layout", "C", "B tensor data layout - Row by default")
        .insert("c_layout", "R", "C tensor data layout - Row by default")
        .insert("stride_a", "0", "Tensor A stride")
        .insert("stride_b", "0", "Tensor B stride")
        .insert("stride_c", "0", "Tensor C stride")
        .insert("v", "1", "0. No validation, 1. Validation on GPU")
        .insert("mixed_prec",
                "bf16xfp4",
                "data type for activation and weight, support: bf16xfp4, fp16xfp4")
        .insert("warmup", "50", "number of iterations before benchmark the kernel")
        .insert("repeat", "100", "number of iterations to benchmark the kernel")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
        .insert("split_k", "1", "splitK value")
        .insert("init", "0", "0:random, 1:constant(1)")
        .insert("persistent", "0", "0: no persistent, 1: persistent kernel")
        .insert("warp_tile",
                "0",
                "0: 16x16, 1: 32x32, 2: 16x16x128 (950 only), 3: 32x32x64 (950 only)");
    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <class FlatmmConfig, class IterSrc, class IterDst>
void preShuffleWeight(const IterSrc src, IterDst dst, int N, int K)
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

            int outputIndex = n0 * KPack * NLane * KLane * K0 + k0 * KPack * NLane * KLane +
                              k1 * KPack * NLane + n1 * KPack + k2;

            dst[outputIndex] = src[n * K_pk + k];
        }
    }
}

template <class FlatmmConfig, class T>
auto preShuffleScale(const ck_tile::HostTensor<T>& scale)
{
    assert(scale.get_lengths().size() == 2);
    int n_ = scale.get_lengths()[1];
    int k_ = scale.get_lengths()[0];

    constexpr int K_Pack       = 2;  // fixed for mxfp4
    constexpr int N_Pack       = 2;  // fixed for mxfp4
    constexpr int GranularityK = 32; // fixed for mxfp4

    constexpr int K_Lane = 64 / FlatmmConfig::N_Warp_Tile; // 4

    static_assert(FlatmmConfig::N_Warp_Tile == 16, "only support XDL_N == 16");
    static_assert(FlatmmConfig::N_Repeat % N_Pack == 0);
    static_assert(FlatmmConfig::K_Tile % (K_Pack * K_Lane * GranularityK) == 0);

    ck_tile::HostTensor<T> shfl_scale({
        k_ / K_Pack / K_Lane,
        K_Pack,
        K_Lane,
        n_ / FlatmmConfig::N_Warp_Tile / N_Pack,
        N_Pack,
        FlatmmConfig::N_Warp_Tile,
    });
    std::copy(scale.begin(), scale.end(), shfl_scale.begin());
    return ck_tile::reference_permute(shfl_scale, {3, 0, 2, 5, 1, 4});
}

#include "run_mixed_prec_flatmm.inc"

template <typename FlatmmConfig>
int run_mixed_prec_flatmm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    std::string mixed_prec = arg_parser.get_str("mixed_prec");
    std::string a_layout   = arg_parser.get_str("a_layout");
    std::string b_layout   = arg_parser.get_str("b_layout");
    int persistent_opt     = arg_parser.get_int("persistent");

    if(a_layout == "R" && b_layout == "C")
    {
        if(mixed_prec == "bf16xfp4")
        {
            if(persistent_opt == 0)
            {
                run_mixed_prec_flatmm_with_layouts<ck_tile::bf16_t,
                                                   ck_tile::pk_fp4_t,
                                                   FlatmmConfig,
                                                   false>(argc, argv, Row{}, Col{}, Row{});
            }
            else
            {
                run_mixed_prec_flatmm_with_layouts<ck_tile::bf16_t,
                                                   ck_tile::pk_fp4_t,
                                                   FlatmmConfig,
                                                   true>(argc, argv, Row{}, Col{}, Row{});
            }
        }
        else if(mixed_prec == "fp16xfp4")
        {
            if(persistent_opt == 0)
            {
                run_mixed_prec_flatmm_with_layouts<ck_tile::fp16_t,
                                                   ck_tile::pk_fp4_t,
                                                   FlatmmConfig,
                                                   false>(argc, argv, Row{}, Col{}, Row{});
            }
            else
            {
                run_mixed_prec_flatmm_with_layouts<ck_tile::fp16_t,
                                                   ck_tile::pk_fp4_t,
                                                   FlatmmConfig,
                                                   true>(argc, argv, Row{}, Col{}, Row{});
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
            return !run_mixed_prec_flatmm_example<A16W4_FlatmmConfig16>(argc, argv);
        }
        else if(warp_tile == 1)
        {
            return !run_mixed_prec_flatmm_example<A16W4_FlatmmConfig16_950>(argc, argv);
        }
        else
        {
            throw std::runtime_error("Unsupported warp_tile!");
        }
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
