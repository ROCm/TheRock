// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <cstring>
#include <iostream>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>

#include "ck_tile/core/config.hpp"
#include "ck_tile/ops/common/utils.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/permute_pk_int4.hpp"
#include "ck_tile/host/tensor_shuffle_utils.hpp"
#include "ck_tile/ops/gemm_quant.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "gemm_utils.hpp"

template <typename GemmConfig,
          typename TypeConfig,
          typename ALayout,
          typename AQLayout,
          typename BLayout,
          typename BQLayout,
          typename CLayout,
          typename AQuantGroupSize,
          typename BQuantGroupSize,
          ck_tile::QuantType QuantMode,
          typename CDEElementWise>
float gemm_calc_quant(const ck_tile::QuantGemmHostArgs& args, const ck_tile::stream_config& s)
{
    static_assert(std::is_same_v<CLayout, ck_tile::tensor_layout::gemm::RowMajor>);
    using ComputeDataType = std::conditional_t<QuantMode == ck_tile::QuantType::AQuantGrouped ||
                                                   QuantMode == ck_tile::QuantType::RowColQuant,
                                               typename TypeConfig::BDataType,
                                               typename TypeConfig::ADataType>;

    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
        ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
        ck_tile::
            sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>>;

    using TilePartitioner = ck_tile::GemmTile1DPartitioner<GemmShape>;

    using GemmTraits = ck_tile::TileGemmQuantTraits<GemmConfig::kPadM,
                                                    GemmConfig::kPadN,
                                                    GemmConfig::kPadK,
                                                    GemmConfig::PreshuffleQuant,
                                                    GemmConfig::PreshuffleB,
                                                    ALayout,
                                                    BLayout,
                                                    CLayout,
                                                    QuantMode,
                                                    AQLayout, // for AQLayout
                                                    BQLayout, // for BQLayout
                                                    false,
                                                    GemmConfig::DoubleSmemBuffer>;

    using GemmPipelineProblem = ck_tile::GemmPipelineProblemBase<typename TypeConfig::ADataType,
                                                                 typename TypeConfig::BDataType,
                                                                 typename TypeConfig::AccDataType,
                                                                 GemmShape,
                                                                 GemmTraits,
                                                                 ComputeDataType>;

    // Base pipeline selection based on quant mode and preshuffle settings
    using BaseGemmPipeline = std::conditional_t<
        GemmConfig::PreshuffleB == true,
        ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2<GemmPipelineProblem>,
        std::conditional_t<
            QuantMode == ck_tile::QuantType::AQuantGrouped && GemmConfig::PreshuffleQuant == true,
            ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmPipelineProblem>,
            std::conditional_t<
                QuantMode == ck_tile::QuantType::AQuantGrouped,
                ck_tile::BaseGemmPipelineAgBgCrMem<GemmPipelineProblem>,
                ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2<GemmPipelineProblem>>>>;

    const ck_tile::index_t K_split =
        (args.K + GemmConfig::K_Tile - 1) / GemmConfig::K_Tile * GemmConfig::K_Tile;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

    const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
        constexpr bool has_hot_loop_v = has_hot_loop_.value;
        constexpr auto tail_number_v  = tail_number_.value;
        constexpr bool transpose_c    = false;

        // row-col and tensor quants use the regular pipeline, A/B/AB quants use their own
        using PipelineProblem = std::conditional_t<
            QuantMode == ck_tile::QuantType::RowColQuant ||
                QuantMode == ck_tile::QuantType::TensorQuant,
            ck_tile::GemmRowColTensorQuantPipelineProblem<typename TypeConfig::ADataType,
                                                          typename TypeConfig::BDataType,
                                                          typename TypeConfig::AccDataType,
                                                          typename TypeConfig::AccDataType,
                                                          GemmShape,
                                                          GemmTraits,
                                                          transpose_c,
                                                          ComputeDataType,
                                                          GemmConfig::Scheduler,
                                                          has_hot_loop_v,
                                                          tail_number_v>,
            std::conditional_t<
                QuantMode == ck_tile::QuantType::AQuantGrouped,
                ck_tile::GemmAQuantPipelineProblem<typename TypeConfig::ADataType,
                                                   typename TypeConfig::QDataType,
                                                   typename TypeConfig::BDataType,
                                                   typename TypeConfig::AccDataType,
                                                   GemmShape,
                                                   GemmTraits,
                                                   AQuantGroupSize,
                                                   transpose_c,
                                                   ComputeDataType,
                                                   GemmConfig::Scheduler,
                                                   has_hot_loop_v,
                                                   tail_number_v>,
                std::conditional_t<
                    QuantMode == ck_tile::QuantType::BQuantGrouped,
                    ck_tile::GemmBQuantPipelineProblem<typename TypeConfig::ADataType,
                                                       typename TypeConfig::BDataType,
                                                       typename TypeConfig::QDataType,
                                                       typename TypeConfig::AccDataType,
                                                       GemmShape,
                                                       GemmTraits,
                                                       BQuantGroupSize,
                                                       ComputeDataType,
                                                       GemmConfig::Scheduler,
                                                       has_hot_loop_v,
                                                       tail_number_v>,
                    ck_tile::GemmABQuantPipelineProblem<typename TypeConfig::ADataType,
                                                        typename TypeConfig::QDataType, // For AQ
                                                        typename TypeConfig::BDataType,
                                                        typename TypeConfig::QDataType, // For BQ
                                                        typename TypeConfig::AccDataType,
                                                        GemmShape,
                                                        GemmTraits,
                                                        AQuantGroupSize,
                                                        BQuantGroupSize,
                                                        transpose_c,
                                                        ComputeDataType,
                                                        GemmConfig::Scheduler,
                                                        has_hot_loop_v,
                                                        tail_number_v>>>>;
        using AQuantPipeline =
            std::conditional_t<GemmConfig::PreshuffleQuant,
                               ck_tile::AQuantGemmPipelineAgBgCrCompV3<PipelineProblem>,
                               ck_tile::AQuantGemmPipelineAgBgCrMem<PipelineProblem>>;

        using BQuantPipeline = std::conditional_t<
            GemmConfig::PreshuffleB,
            ck_tile::WPQuantBPipelineAgBgCrV2<PipelineProblem>,
            std::conditional_t<
                std::is_same_v<typename TypeConfig::BDataType, ck_tile::pk_fp4_raw_t>,
                ck_tile::MxFp4GemmPipelineAgBgCrCompV3<PipelineProblem>,
                ck_tile::BQuantGemmPipelineAgBgCrCompV3<PipelineProblem>>>;

        using ABQuantPipeline =
            std::conditional_t<GemmConfig::DoubleSmemBuffer && GemmConfig::PreshuffleB,
                               ck_tile::WPABQuantBPipelineAgBgCrV2<PipelineProblem>,
                               ck_tile::ABQuantGemmPipelineAgBgCrCompV3<PipelineProblem>>;

        using GemmPipeline = std::conditional_t<
            QuantMode == ck_tile::QuantType::RowColQuant ||
                QuantMode == ck_tile::QuantType::TensorQuant,
            ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem>,
            std::conditional_t<QuantMode == ck_tile::QuantType::AQuantGrouped,
                               AQuantPipeline,
                               std::conditional_t<QuantMode == ck_tile::QuantType::ABQuantGrouped,
                                                  ABQuantPipeline,
                                                  BQuantPipeline>>>;

        constexpr bool TiledPermuteN =
            (BQuantGroupSize::kN > 1) ? false : GemmConfig::TiledMMAPermuteN;
        if(s.log_level_ > 0)
        {
            printf(
                "TiledPermuteN: %d (QuantGroupSize::kN=%d)\n", TiledPermuteN, BQuantGroupSize::kN);
        }
        using GemmEpilogue = ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
            typename TypeConfig::ADataType,
            std::conditional_t<
                std::is_same_v<typename TypeConfig::BDataType, ck_tile::pk_fp4_raw_t>,
                typename TypeConfig::ADataType,
                typename TypeConfig::BDataType>,
            ck_tile::tuple<>,
            typename TypeConfig::AccDataType,
            typename TypeConfig::CDataType,
            ck_tile::tuple<>,
            CLayout,
            CDEElementWise,
            TilePartitioner::MPerBlock,
            TilePartitioner::NPerBlock,
            GemmConfig::M_Warp,
            GemmConfig::N_Warp,
            GemmConfig::M_Warp_Tile,
            GemmConfig::N_Warp_Tile,
            GemmConfig::K_Warp_Tile,
            transpose_c,
            1,
            false,
            1,
            TiledPermuteN>>;
        using Kernel =
            ck_tile::QuantGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue, QuantMode>;

        auto kargs = Kernel::MakeKernelArgs(args);

        const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
        const dim3 blocks = Kernel::BlockSize();

        if(args.k_batch != 1)
        {
            throw std::runtime_error("split-k is not supported yet!");
        }

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args: " << Kernel::GetName() << '\n'
                      << "shape: " << GemmShape::GetName() << '\n'
                      << "problem: " << PipelineProblem::GetName() << '\n'
                      << "pipeline: " << GemmPipeline::GetName() << '\n'
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }
        float ave_time = 0;
        if(s.flush_cache_)
        {
            std::cout << "Flushing cache..." << std::endl;

            ck_tile::HostTensor<typename TypeConfig::ADataType> a_m(ck_tile::host_tensor_descriptor(
                args.M, args.K, args.stride_A, is_row_major(ALayout{})));
            ck_tile::HostTensor<typename TypeConfig::BDataType> b_n(ck_tile::host_tensor_descriptor(
                std::is_same_v<typename TypeConfig::BDataType, ck_tile::pk_fp4_raw_t> ? args.K / 2
                                                                                      : args.K,
                args.N,
                args.stride_B,
                is_row_major(BLayout{})));

            auto size_a_buffer = a_m.get_element_space_size_in_bytes();
            auto size_b_buffer = b_n.get_element_space_size_in_bytes();

            ck_tile::RotatingMemWrapper<typename TypeConfig::ADataType,
                                        typename TypeConfig::BDataType>
                rotating_mem(
                    kargs.a_ptr, kargs.b_ptr, s.rotating_count_, size_a_buffer, size_b_buffer);
            rotating_mem.Print();

            auto run_flush_cache = [&]() {
                // flush icache
                ck_tile::flush_icache();
                // rotating mem
                rotating_mem.Next();
                // clear c mem
                if(args.k_batch > 1)
                    hipGetErrorString(
                        hipMemsetAsync(args.c_ptr,
                                       0,
                                       args.M * args.N * sizeof(typename TypeConfig::CDataType),
                                       s.stream_id_));
            };
            ave_time = ck_tile::launch_kernel_time_mask(
                s,
                run_flush_cache,
                ck_tile::make_kernel<GemmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        }
        else
        {
            ave_time = ck_tile::launch_kernel(
                s,
                ck_tile::make_kernel<GemmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        }

        return ave_time;
    };
    return BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
}

template <typename GemmConfig,
          typename TypeConfig,
          typename ALayout,
          typename AQLayout,
          typename BLayout,
          typename BQLayout,
          typename CLayout,
          typename AQuantGroupSize,
          typename BQuantGroupSize,
          ck_tile::QuantType QuantMode,
          typename CDEElementWise = ck_tile::element_wise::PassThrough>
float invoke_gemm(ck_tile::DeviceMem& a_m_k_dev_buf,
                  ck_tile::DeviceMem* aq_dev_buf,
                  ck_tile::DeviceMem& b_k_n_dev_buf,
                  ck_tile::DeviceMem* bq_dev_buf,
                  ck_tile::DeviceMem& c_m_n_dev_buf,
                  ck_tile::index_t M,
                  ck_tile::index_t N,
                  ck_tile::index_t K,
                  ck_tile::index_t AQK,
                  ck_tile::index_t BQK,
                  ck_tile::index_t BQN,
                  ck_tile::index_t stride_A,
                  ck_tile::index_t stride_AQ,
                  ck_tile::index_t stride_B,
                  ck_tile::index_t stride_BQ,
                  ck_tile::index_t stride_C,
                  ck_tile::index_t kbatch,
                  int n_warmup,
                  int n_repeat,
                  bool flush_cache,
                  int rotating_count)
{
    ck_tile::QuantGemmHostArgs args;
    args.a_ptr     = a_m_k_dev_buf.GetDeviceBuffer();
    args.aq_ptr    = (aq_dev_buf != nullptr) ? aq_dev_buf->GetDeviceBuffer() : nullptr;
    args.b_ptr     = b_k_n_dev_buf.GetDeviceBuffer();
    args.bq_ptr    = (bq_dev_buf != nullptr) ? bq_dev_buf->GetDeviceBuffer() : nullptr;
    args.c_ptr     = c_m_n_dev_buf.GetDeviceBuffer();
    args.k_batch   = kbatch;
    args.M         = M;
    args.N         = N;
    args.K         = K;
    args.QK_A      = AQK;
    args.QK_B      = BQK;
    args.stride_A  = stride_A;
    args.stride_B  = stride_B;
    args.stride_C  = stride_C;
    args.stride_AQ = stride_AQ;
    args.stride_BQ = stride_BQ;

    float ave_time = gemm_calc_quant<GemmConfig,
                                     TypeConfig,
                                     ALayout,
                                     AQLayout,
                                     BLayout,
                                     BQLayout,
                                     CLayout,
                                     AQuantGroupSize,
                                     BQuantGroupSize,
                                     QuantMode,
                                     CDEElementWise>(
        args,
        ck_tile::stream_config{
            nullptr, true, 1, n_warmup, n_repeat, true, flush_cache, rotating_count});

    std::size_t flop     = std::size_t(2) * M * N * K;
    std::size_t num_byte = sizeof(typename TypeConfig::ADataType) * M * K +
                           sizeof(typename TypeConfig::BDataType) * N * K +
                           sizeof(typename TypeConfig::CDataType) * M * N;
    if(aq_dev_buf != nullptr)
    {
        num_byte += sizeof(typename TypeConfig::QDataType) * M * AQK;
    }
    if(bq_dev_buf != nullptr)
    {
        num_byte += sizeof(typename TypeConfig::QDataType) * BQN * BQK;
    }

    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Run Gemm kernel with M =" << M << " N =" << N << " K =" << K
              << " StrideA =" << stride_A << " StrideAQ =" << stride_AQ << " StrideB =" << stride_B
              << " StrideBQ =" << stride_BQ << " StrideC =" << stride_C
              << " A_Layout =" << ALayout::name << " B_Layout =" << BLayout::name
              << " C_Layout =" << CLayout::name << " AQ_Layout =" << AQLayout::name
              << " BQ_Layout =" << BQLayout::name;

    if constexpr(QuantMode == ck_tile::QuantType::ABQuantGrouped ||
                 QuantMode == ck_tile::QuantType::BQuantGrouped ||
                 QuantMode == ck_tile::QuantType::RowColQuant)
    {
        std::cout << " StrideBQ =" << stride_BQ;
    }
    std::cout << " A_Type = " << ck_tile::DataTypeTraits<typename TypeConfig::ADataType>::name
              << " AQ_Type = " << ck_tile::DataTypeTraits<typename TypeConfig::QDataType>::name
              << " B_Type = " << ck_tile::DataTypeTraits<typename TypeConfig::BDataType>::name;
    if constexpr(!std::is_same_v<typename TypeConfig::QDataType, void>)
    {
        std::cout << " BQ_Type = " << ck_tile::DataTypeTraits<typename TypeConfig::QDataType>::name;
    }
    std::cout << " Acc_Type = " << ck_tile::DataTypeTraits<typename TypeConfig::AccDataType>::name
              << " C_Type = " << ck_tile::DataTypeTraits<typename TypeConfig::CDataType>::name
              << " QuantMode = " << quant_type_to_string(QuantMode)
              << " PreshuffleQuant = " << (GemmConfig::PreshuffleQuant ? "true" : "false") << " : "
              << " PreshuffleB = " << (GemmConfig::PreshuffleB ? "true" : "false") << " : "
              << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << std::endl;

    return ave_time;
}

template <typename GemmConfig,
          typename TypeConfig,
          typename AQuantGroupSize,
          typename BQuantGroupSize,
          ck_tile::QuantType QuantMode,
          typename ALayout,
          typename AQLayout,
          typename BLayout,
          typename BQLayout,
          typename CLayout>
int run_gemm_example_with_layouts(const ck_tile::ArgParser& arg_parser,
                                  const ALayout a_layout                  = ALayout{},
                                  const AQLayout aq_layout                = AQLayout{},
                                  const BLayout b_layout                  = BLayout{},
                                  const BQLayout bq_layout                = BQLayout{},
                                  [[maybe_unused]] const CLayout c_layout = CLayout{})
{
    using ADataType   = typename TypeConfig::ADataType;
    using AQDataType  = typename TypeConfig::QDataType;
    using BDataType   = typename TypeConfig::BDataType;
    using BQDataType  = typename TypeConfig::QDataType;
    using AccDataType = typename TypeConfig::AccDataType;
    using CDataType   = typename TypeConfig::CDataType;

    ck_tile::index_t M = arg_parser.get_int("m");
    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t K = arg_parser.get_int("k");

    ck_tile::index_t AQK, BQK, BQN = 0;
    if constexpr(QuantMode == ck_tile::QuantType::AQuantGrouped)
    {
        AQK = ck_tile::integer_divide_ceil(
            K, AQuantGroupSize::kK); // Group quantization: AQK = K / GroupSize
        BQK = 0;                     // No B quantization
    }
    else if constexpr(QuantMode == ck_tile::QuantType::BQuantGrouped)
    {
        AQK = 0; // No A quantization
        BQK = ck_tile::integer_divide_ceil(
            K, BQuantGroupSize::kK); // Group quantization: BQK = K / GroupSize
        BQN = ck_tile::integer_divide_ceil(N, BQuantGroupSize::kN);
    }
    else if constexpr(QuantMode == ck_tile::QuantType::ABQuantGrouped)
    {
        AQK = ck_tile::integer_divide_ceil(
            K, AQuantGroupSize::kK); // Group quantization: AQK = K / GroupSize
        BQK = ck_tile::integer_divide_ceil(
            K, BQuantGroupSize::kK); // Group quantization: BQK = K / GroupSize
        BQN = ck_tile::integer_divide_ceil(N, BQuantGroupSize::kN);
    }
    else if constexpr(QuantMode == ck_tile::QuantType::RowColQuant ||
                      QuantMode == ck_tile::QuantType::TensorQuant)
    {
        AQK = 1; // Row quantization: tensor shape [M, 1] or [1]
        BQK = 1; // Column quantization: tensor shape [1, N] or [1]
        BQN = 1;
    }
    else
    {
        throw std::runtime_error("Unsupported QuantMode");
    }

    ck_tile::index_t stride_A  = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_AQ = arg_parser.get_int("stride_q");
    ck_tile::index_t stride_B  = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_BQ = arg_parser.get_int("stride_q");
    ck_tile::index_t stride_C  = arg_parser.get_int("stride_c");

    ck_tile::index_t kbatch      = arg_parser.get_int("split_k");
    int n_warmup                 = arg_parser.get_int("warmup");
    int n_repeat                 = arg_parser.get_int("repeat");
    ck_tile::index_t init_method = arg_parser.get_int("init");
    bool flush_cache             = arg_parser.get_bool("flush_cache");
    int rotating_count           = arg_parser.get_int("rotating_count");

    stride_A = ck_tile::get_default_stride(M, K, stride_A, is_row_major(a_layout));
    stride_B = ck_tile::get_default_stride(
        (std::is_same_v<BDataType, ck_tile::pk_fp4_raw_t>) ? (K / 2) : K,
        N,
        stride_B,
        is_row_major(b_layout));
    stride_C = ck_tile::get_default_stride(M, N, stride_C, is_row_major(CLayout{}));

    // Conditional stride calculation based on QuantMode
    if constexpr(QuantMode == ck_tile::QuantType::AQuantGrouped)
    {
        stride_AQ = ck_tile::get_default_stride(M, AQK, stride_AQ, is_row_major(aq_layout));
        stride_BQ = 0; // No B quantization
    }
    else if constexpr(QuantMode == ck_tile::QuantType::BQuantGrouped)
    {
        stride_AQ = 0; // No A quantization
        stride_BQ = ck_tile::get_default_stride(BQK, BQN, stride_BQ, is_row_major(bq_layout));
    }
    else if constexpr(QuantMode == ck_tile::QuantType::ABQuantGrouped)
    {
        stride_AQ = ck_tile::get_default_stride(M, AQK, stride_AQ, is_row_major(aq_layout));
        stride_BQ = ck_tile::get_default_stride(BQK, BQN, stride_BQ, is_row_major(bq_layout));
    }
    else if constexpr(QuantMode == ck_tile::QuantType::RowColQuant)
    {
        stride_AQ = ck_tile::get_default_stride(M, 1, stride_AQ, is_row_major(aq_layout));
        stride_BQ = ck_tile::get_default_stride(1, N, stride_BQ, is_row_major(bq_layout));
    }
    else if constexpr(QuantMode == ck_tile::QuantType::TensorQuant)
    {
        stride_AQ = 1; // Tensor quantization: tensor shape [1]
        stride_BQ = 1; // Tensor quantization: tensor shape [1]
    }

    ck_tile::HostTensor<ADataType> a_m_k(
        ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(a_layout)));
    ck_tile::HostTensor<BDataType> b_k_n(ck_tile::host_tensor_descriptor(
        (std::is_same_v<BDataType, ck_tile::pk_fp4_raw_t>) ? (K / 2) : K,
        N,
        stride_B,
        is_row_major(b_layout)));
    ck_tile::HostTensor<CDataType> c_m_n_dev_result(
        ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

    // Create AQ tensor with appropriate shape
    std::unique_ptr<ck_tile::HostTensor<AQDataType>> aq_tensor_ptr = nullptr;
    if constexpr(QuantMode == ck_tile::QuantType::AQuantGrouped ||
                 QuantMode == ck_tile::QuantType::ABQuantGrouped ||
                 QuantMode == ck_tile::QuantType::RowColQuant)
    {
        aq_tensor_ptr = std::make_unique<ck_tile::HostTensor<AQDataType>>(
            ck_tile::host_tensor_descriptor(M, AQK, stride_AQ, is_row_major(aq_layout)));
    }
    else if constexpr(QuantMode == ck_tile::QuantType::TensorQuant)
    {
        aq_tensor_ptr = std::make_unique<ck_tile::HostTensor<AQDataType>>(
            ck_tile::host_tensor_descriptor(1, 1, stride_AQ, is_row_major(aq_layout)));
    }

    // Create BQ tensor with appropriate shape
    std::unique_ptr<ck_tile::HostTensor<BQDataType>> bq_tensor_ptr = nullptr;
    if constexpr(QuantMode == ck_tile::QuantType::BQuantGrouped ||
                 QuantMode == ck_tile::QuantType::RowColQuant)
    {
        bq_tensor_ptr = std::make_unique<ck_tile::HostTensor<BQDataType>>(
            ck_tile::host_tensor_descriptor(BQK, BQN, stride_BQ, is_row_major(bq_layout)));
    }
    else if constexpr(QuantMode == ck_tile::QuantType::ABQuantGrouped)
    {
        bq_tensor_ptr = std::make_unique<ck_tile::HostTensor<BQDataType>>(
            ck_tile::host_tensor_descriptor(BQK, BQN, stride_BQ, is_row_major(bq_layout)));
    }
    else if constexpr(QuantMode == ck_tile::QuantType::TensorQuant)
    {
        bq_tensor_ptr = std::make_unique<ck_tile::HostTensor<BQDataType>>(
            ck_tile::host_tensor_descriptor(1, 1, stride_BQ, is_row_major(bq_layout)));
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<std::uint32_t> fill_seed(0, 500);

    if(init_method == 0)
    {
        if constexpr(QuantMode == ck_tile::QuantType::BQuantGrouped)
        {
            if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
            {
                ck_tile::FillUniformDistribution<ck_tile::pk_int4_t>{-5.0f, 5.0f, fill_seed(gen)}(
                    b_k_n);
                ck_tile::FillUniformDistribution<BQDataType>{-2.0f, 2.0f, fill_seed(gen)}(
                    *bq_tensor_ptr);
            }
            else if constexpr(std::is_same_v<BDataType, ck_tile::pk_fp4_raw_t>)
            {
                ck_tile::FillUniformDistribution<BDataType>{-5.0f, 5.0f, fill_seed(gen)}(b_k_n);
                ck_tile::FillUniformDistribution<BQDataType>{125.f, 130.f, fill_seed(gen)}(
                    *bq_tensor_ptr);
            }
            else
            {
                ck_tile::FillUniformDistribution<BDataType>{-2.0f, 3.0f, fill_seed(gen)}(b_k_n);
                ck_tile::FillUniformDistribution<BQDataType>{-2.0f, 2.0f, fill_seed(gen)}(
                    *bq_tensor_ptr);
            }

            ck_tile::FillUniformDistribution<ADataType>{-5.0f, 5.0f, fill_seed(gen)}(a_m_k);
        }
        else if constexpr(QuantMode == ck_tile::QuantType::AQuantGrouped)
        {
            if constexpr(std::is_same_v<ADataType, ck_tile::pk_int4_t>)
            {
                ck_tile::FillUniformDistribution<ck_tile::pk_int4_t>{-5.0f, 5.0f, fill_seed(gen)}(
                    a_m_k);
            }
            else
            {
                ck_tile::FillUniformDistribution<ADataType>{-2.0f, 3.0f, fill_seed(gen)}(a_m_k);
            }
            ck_tile::FillUniformDistribution<AQDataType>{-2.0f, 2.0f, fill_seed(gen)}(
                *aq_tensor_ptr);
            ck_tile::FillUniformDistribution<BDataType>{-5.0f, 5.0f, fill_seed(gen)}(b_k_n);
        }
        else if constexpr(QuantMode == ck_tile::QuantType::ABQuantGrouped)
        {
            if constexpr(std::is_same_v<ADataType, ck_tile::pk_int4_t>)
            {
                ck_tile::FillUniformDistribution<ck_tile::pk_int4_t>{-5.0f, 5.0f, fill_seed(gen)}(
                    a_m_k);
                ck_tile::FillUniformDistribution<ck_tile::pk_int4_t>{-5.0f, 5.0f, fill_seed(gen)}(
                    b_k_n);
            }
            else
            {
                ck_tile::FillUniformDistribution<ADataType>{-2.0f, 3.0f, fill_seed(gen)}(a_m_k);
                ck_tile::FillUniformDistribution<BDataType>{-2.0f, 3.0f, fill_seed(gen)}(b_k_n);
            }
            ck_tile::FillUniformDistribution<AQDataType>{-2.0f, 2.0f, fill_seed(gen)}(
                *aq_tensor_ptr);
            ck_tile::FillUniformDistribution<BQDataType>{-2.0f, 2.0f, fill_seed(gen)}(
                *bq_tensor_ptr);
        }
        else
        {
            ck_tile::FillUniformDistribution<ADataType>{-2.0f, 2.0f, fill_seed(gen)}(a_m_k);
            ck_tile::FillUniformDistribution<BDataType>{-2.0f, 2.0f, fill_seed(gen)}(b_k_n);
            ck_tile::FillUniformDistribution<AQDataType>{-2.0f, 2.0f, fill_seed(gen)}(
                *aq_tensor_ptr);
            ck_tile::FillUniformDistribution<BQDataType>{-2.0f, 2.0f, fill_seed(gen)}(
                *bq_tensor_ptr);
        }
    }
    else if(init_method == 1)
    {
        std::cout << "Monotonic initialization is not supported." << std::endl;
        return 0;
    }
    else if(init_method == 2)
    {
        if constexpr(QuantMode == ck_tile::QuantType::BQuantGrouped)
        {
            ck_tile::FillConstant<ADataType>{static_cast<ADataType>(0x38)}(a_m_k);
            ck_tile::FillConstant<BDataType>{static_cast<BDataType>(0x22)}(b_k_n);
            ck_tile::FillConstant<BQDataType>{static_cast<BQDataType>(0.5f)}(*bq_tensor_ptr);
        }
        else if constexpr(QuantMode == ck_tile::QuantType::ABQuantGrouped)
        {
            ck_tile::FillConstant<ADataType>{static_cast<ADataType>(0x38)}(a_m_k);
            ck_tile::FillConstant<BDataType>{static_cast<BDataType>(0x22)}(b_k_n);
            ck_tile::FillConstant<AQDataType>{static_cast<AQDataType>(0.5f)}(*aq_tensor_ptr);
            ck_tile::FillConstant<BQDataType>{static_cast<BQDataType>(0.5f)}(*bq_tensor_ptr);
        }
        else
        {
            ck_tile::FillConstant<ADataType>{static_cast<ADataType>(0x22)}(a_m_k);
            ck_tile::FillConstant<AQDataType>{static_cast<AQDataType>(0.5f)}(*aq_tensor_ptr);
            ck_tile::FillConstant<BDataType>{static_cast<BDataType>(0x38)}(b_k_n);

            if constexpr(QuantMode == ck_tile::QuantType::RowColQuant)
            {
                ck_tile::FillConstant<BQDataType>{static_cast<BQDataType>(0.5f)}(*bq_tensor_ptr);
            }
        }
    }
    else
    {
        a_m_k.SetZero();
        aq_tensor_ptr->SetZero();
        b_k_n.SetZero();
        bq_tensor_ptr->SetZero();
    }
    ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

    std::unique_ptr<ck_tile::DeviceMem> aq_dev_buf_ptr = nullptr;
    if constexpr(QuantMode == ck_tile::QuantType::AQuantGrouped ||
                 QuantMode == ck_tile::QuantType::ABQuantGrouped ||
                 QuantMode == ck_tile::QuantType::RowColQuant ||
                 QuantMode == ck_tile::QuantType::TensorQuant)
    {
        aq_dev_buf_ptr =
            std::make_unique<ck_tile::DeviceMem>(aq_tensor_ptr->get_element_space_size_in_bytes());
    }
    std::unique_ptr<ck_tile::DeviceMem> bq_dev_buf_ptr = nullptr;
    if constexpr(QuantMode == ck_tile::QuantType::BQuantGrouped ||
                 QuantMode == ck_tile::QuantType::ABQuantGrouped ||
                 QuantMode == ck_tile::QuantType::RowColQuant ||
                 QuantMode == ck_tile::QuantType::TensorQuant)
    {
        bq_dev_buf_ptr =
            std::make_unique<ck_tile::DeviceMem>(bq_tensor_ptr->get_element_space_size_in_bytes());
    }

    if constexpr(QuantMode == ck_tile::QuantType::AQuantGrouped ||
                 QuantMode == ck_tile::QuantType::ABQuantGrouped ||
                 QuantMode == ck_tile::QuantType::RowColQuant ||
                 QuantMode == ck_tile::QuantType::TensorQuant)
    {
        if constexpr(GemmConfig::PreshuffleQuant)
        {
            ck_tile::HostTensor<AQDataType> aq_shuffle_host =
                ck_tile::shuffle_aq(aq_tensor_ptr.get(), GemmConfig::K_Tile / AQuantGroupSize::kK);
            aq_dev_buf_ptr->ToDevice(aq_shuffle_host.data());
        }
        else
        {
            aq_dev_buf_ptr->ToDevice(aq_tensor_ptr->data());
        }
    }

    if constexpr(std::is_same_v<ADataType, ck_tile::pk_int4_t>)
    {
        // Permute vector pk_i4x4 data for device implementation
        ck_tile::HostTensor<ADataType> a_m_k_dev = a_m_k;
        ck_tile::permute_vectors_i4x4_b(a_m_k_dev);
        a_m_k_dev_buf.ToDevice(a_m_k_dev.data());
    }
    else
    {
        a_m_k_dev_buf.ToDevice(a_m_k.data());
    }

    ck_tile::HostTensor<BDataType> b_k_n_dev = b_k_n;
    if constexpr(GemmConfig::PreshuffleB)
    {
        if constexpr(GemmConfig::TiledMMAPermuteN && BQuantGroupSize::kN == 1)
        {
            printf("PreshuffleB with TiledMMAPermuteN\n");
            b_k_n_dev = ck_tile::shuffle_b_permuteN<GemmConfig>(b_k_n);
        }
        else
        {
            printf("PreshuffleB without TiledMMAPermuteN\n");
            b_k_n_dev = ck_tile::shuffle_b<GemmConfig>(b_k_n);
        }
    }
    if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
    {
        ck_tile::permute_vectors_i4x4_b(b_k_n_dev);
    }

    b_k_n_dev_buf.ToDevice(b_k_n_dev.data());

    c_m_n_dev_buf.SetZero();
    c_m_n_dev_result.SetZero();

    if constexpr(QuantMode == ck_tile::QuantType::BQuantGrouped ||
                 QuantMode == ck_tile::QuantType::ABQuantGrouped ||
                 QuantMode == ck_tile::QuantType::RowColQuant ||
                 QuantMode == ck_tile::QuantType::TensorQuant)
    {
        if constexpr(GemmConfig::PreshuffleB && GemmConfig::TiledMMAPermuteN &&
                     BQuantGroupSize::kN == 1)
        {
            ck_tile::HostTensor<BQDataType> bq_permuted_host =
                ck_tile::bq_permuteN<GemmConfig>(*bq_tensor_ptr, BQuantGroupSize::kN);

            if constexpr(GemmConfig::PreshuffleQuant)
            {
                ck_tile::HostTensor<BQDataType> bq_shuffle_host = ck_tile::shuffle_bq(
                    &bq_permuted_host, GemmConfig::K_Tile / BQuantGroupSize::kK);
                bq_dev_buf_ptr->ToDevice(bq_shuffle_host.data());
            }
            else
            {
                bq_dev_buf_ptr->ToDevice(bq_permuted_host.data());
            }
        }
        else if constexpr(GemmConfig::PreshuffleQuant)
        {
            ck_tile::HostTensor<BQDataType> bq_shuffle_host =
                ck_tile::shuffle_bq(bq_tensor_ptr.get(), GemmConfig::K_Tile / BQuantGroupSize::kK);
            bq_dev_buf_ptr->ToDevice(bq_shuffle_host.data());
        }
        else
        {
            bq_dev_buf_ptr->ToDevice(bq_tensor_ptr->data());
        }
    }

    invoke_gemm<GemmConfig,
                TypeConfig,
                ALayout,
                AQLayout,
                BLayout,
                BQLayout,
                CLayout,
                AQuantGroupSize,
                BQuantGroupSize,
                QuantMode>(a_m_k_dev_buf,
                           aq_dev_buf_ptr.get(),
                           b_k_n_dev_buf,
                           bq_dev_buf_ptr.get(),
                           c_m_n_dev_buf,
                           M,
                           N,
                           K,
                           AQK,
                           BQK,
                           BQN,
                           stride_A,
                           stride_AQ,
                           stride_B,
                           stride_BQ,
                           stride_C,
                           kbatch,
                           n_warmup,
                           n_repeat,
                           flush_cache,
                           rotating_count);

    c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());
    bool pass = true;

    if(arg_parser.get_int("v") == 1)
    {
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        if constexpr(QuantMode == ck_tile::QuantType::AQuantGrouped)
        {
            ck_tile::reference_gemm_quant<ADataType,
                                          AQDataType,
                                          BDataType,
                                          AccDataType,
                                          CDataType,
                                          AQuantGroupSize,
                                          true>(a_m_k, *aq_tensor_ptr, b_k_n, c_m_n_host_ref);
        }
        else if constexpr(QuantMode == ck_tile::QuantType::BQuantGrouped)
        {
            if constexpr(std::is_same_v<BDataType, ck_tile::pk_fp4_raw_t>)
                ck_tile::reference_mxfp4gemm_quant<ADataType,
                                                   BQDataType,
                                                   BDataType,
                                                   AccDataType,
                                                   CDataType,
                                                   BQuantGroupSize,
                                                   false>(
                    a_m_k, *bq_tensor_ptr, b_k_n, c_m_n_host_ref);
            else
                ck_tile::reference_gemm_quant<ADataType,
                                              AQDataType,
                                              BDataType,
                                              AccDataType,
                                              CDataType,
                                              BQuantGroupSize,
                                              false>(a_m_k, *bq_tensor_ptr, b_k_n, c_m_n_host_ref);
        }
        else if constexpr(QuantMode == ck_tile::QuantType::ABQuantGrouped)
        {
            ck_tile::reference_gemm_abquant<ADataType,
                                            AQDataType,
                                            BDataType,
                                            BQDataType,
                                            AccDataType,
                                            CDataType,
                                            AQuantGroupSize,
                                            BQuantGroupSize>(
                a_m_k, *aq_tensor_ptr, b_k_n, *bq_tensor_ptr, c_m_n_host_ref);
        }
        else if constexpr(QuantMode == ck_tile::QuantType::RowColQuant)
        {
            ck_tile::reference_gemm_rowcol_quant<ADataType,
                                                 AQDataType,
                                                 BDataType,
                                                 BQDataType,
                                                 AccDataType,
                                                 CDataType>(
                a_m_k, *aq_tensor_ptr, b_k_n, *bq_tensor_ptr, c_m_n_host_ref);
        }
        else if constexpr(QuantMode == ck_tile::QuantType::TensorQuant)
        {
            ck_tile::reference_gemm_tensor_quant<ADataType,
                                                 AQDataType,
                                                 BDataType,
                                                 BQDataType,
                                                 AccDataType,
                                                 CDataType>(
                a_m_k, *aq_tensor_ptr, b_k_n, *bq_tensor_ptr, c_m_n_host_ref);
        }

        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, kbatch, max_accumulated_value);
        pass = ck_tile::check_err(c_m_n_dev_result,
                                  c_m_n_host_ref,
                                  "Error: Incorrect results!",
                                  rtol_atol.at(ck_tile::number<0>{}),
                                  rtol_atol.at(ck_tile::number<1>{}));

        if(!pass)
        {
            std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                      << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                      << std::endl;
        }
        std::cout << "The CPU verification result is:" << (pass ? "correct" : "fail") << std::endl;
    }
    else if(arg_parser.get_int("v") == 2)
    {
        std::cout << "GPU verification is not implemented yet. Re-run with -v=1" << std::endl;
        return false;
    }

    return pass;
}
// Usage of Two-Matrix Quantization (AB-Quant)
template <typename GemmConfig,
          typename TypeConfig,
          typename AQuantGroupSize,
          typename BQuantGroupSize,
          ck_tile::QuantType QuantMode>
int run_gemm_example_prec_type(const ck_tile::ArgParser& arg_parser)
{
    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    if((QuantMode == ck_tile::QuantType::AQuantGrouped ||
        QuantMode == ck_tile::QuantType::RowColQuant ||
        std::is_same_v<typename TypeConfig::BDataType, ck_tile::pk_fp4_raw_t>) &&
       GemmConfig::PreshuffleB)
    {
        throw std::runtime_error(
            "Preshuffling weight matrix is not supported for AQuant, RowColQuant or bf16_fp4_gemm");
    }

    if constexpr(std::is_same_v<typename TypeConfig::ADataType, ck_tile::pk_int4_t> ||
                 std::is_same_v<typename TypeConfig::ADataType, ck_tile::fp8_t> ||
                 std::is_same_v<typename TypeConfig::ADataType, ck_tile::bf8_t> ||
                 std::is_same_v<typename TypeConfig::ADataType, ck_tile::bf16_t>)
    {
        std::string a_layout = arg_parser.get_str("a_layout");
        std::string b_layout = arg_parser.get_str("b_layout");

        if(a_layout == "R" && b_layout == "C")
        {
            return run_gemm_example_with_layouts<GemmConfig,
                                                 TypeConfig,
                                                 AQuantGroupSize,
                                                 BQuantGroupSize,
                                                 QuantMode>(
                arg_parser, Row{}, Row{}, Col{}, Col{}, Row{});
        }

        if constexpr((QuantMode == ck_tile::QuantType::AQuantGrouped ||
                      QuantMode == ck_tile::QuantType::ABQuantGrouped) &&
                     !GemmConfig::PreshuffleQuant && !GemmConfig::PreshuffleB)
        {
            if(a_layout == "R" && b_layout == "R")
            {
                return run_gemm_example_with_layouts<GemmConfig,
                                                     TypeConfig,
                                                     AQuantGroupSize,
                                                     BQuantGroupSize,
                                                     QuantMode>(
                    arg_parser, Row{}, Row{}, Row{}, Col{}, Row{});
            }
            else if(a_layout == "C" && b_layout == "R")
            {
                return run_gemm_example_with_layouts<GemmConfig,
                                                     TypeConfig,
                                                     AQuantGroupSize,
                                                     BQuantGroupSize,
                                                     QuantMode>(
                    arg_parser, Col{}, Row{}, Row{}, Col{}, Row{});
            }
        }
        if constexpr(QuantMode == ck_tile::QuantType::AQuantGrouped && !GemmConfig::PreshuffleQuant)
        {
            if(a_layout == "C" && b_layout == "C")
            {
                return run_gemm_example_with_layouts<GemmConfig,
                                                     TypeConfig,
                                                     AQuantGroupSize,
                                                     BQuantGroupSize,
                                                     QuantMode>(
                    arg_parser, Col{}, Col{}, Col{}, Col{}, Row{});
            }
        }
        else
        {
            throw std::runtime_error("Unsupported memory layout for the input matrices!");
        }
    }
    else
    {
        throw std::runtime_error("Unsupported data type for A.");
    }

    return 0;
}
// Support for Unilateral Quantization (A/B)
template <typename GemmConfig,
          typename TypeConfig,
          typename QuantGroupSize,
          ck_tile::QuantType QuantMode>
int run_gemm_example_prec_type(const ck_tile::ArgParser& arg_parser)
{
    return run_gemm_example_prec_type<GemmConfig,
                                      TypeConfig,
                                      QuantGroupSize,
                                      QuantGroupSize,
                                      QuantMode>(arg_parser);
}
