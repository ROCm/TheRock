// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <string>
#include <tuple>

#include "ck_tile/host.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

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

    // The logic below may need to become more advanced once bugs in Stream-K Tile Partitioner are
    // resolved. Because the number of WGs contributing to a macro tile in C may not be the same for
    // all macro tiles in C.

    // Calculate error due to more than 1 WG contributing to the same macro tile in C
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
        max_accumulated_value, kbatch);
    // Use higher threshold
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

ck_tile::index_t get_cu_count();

template <typename Tuple>
class TestCkTileStreamK : public ::testing::Test
{
    protected:
    using ALayout                            = std::tuple_element_t<0, Tuple>;
    using BLayout                            = std::tuple_element_t<1, Tuple>;
    using CLayout                            = std::tuple_element_t<2, Tuple>;
    using ADataType                          = std::tuple_element_t<3, Tuple>;
    using BDataType                          = std::tuple_element_t<4, Tuple>;
    using AccDataType                        = std::tuple_element_t<5, Tuple>;
    using CDataType                          = std::tuple_element_t<6, Tuple>;
    using DsLayout                           = ck_tile::tuple<>;
    using DsDataType                         = ck_tile::tuple<>;
    static constexpr ck_tile::index_t M_Tile = std::tuple_element_t<7, Tuple>::value;
    static constexpr ck_tile::index_t N_Tile = std::tuple_element_t<8, Tuple>::value;
    static constexpr ck_tile::index_t K_Tile = std::tuple_element_t<9, Tuple>::value;
    static constexpr bool Persistent         = std::tuple_element_t<10, Tuple>::value;

    template <ck_tile::StreamKReductionStrategy ReductionStrategy,
              bool PadM       = true,
              bool PadN       = true,
              bool PadK       = true,
              bool Preshuffle = false,
              bool TransposeC = false>
    ck_tile::index_t invoke_streamk(const ck_tile::StreamKHostArgs& args,
                                    const ck_tile::stream_config& s)
    {
        constexpr ck_tile::index_t M_Warp = 2;
        constexpr ck_tile::index_t N_Warp = 2;
        constexpr ck_tile::index_t K_Warp = 1;

        constexpr ck_tile::index_t M_Warp_Tile = 32;
        constexpr ck_tile::index_t N_Warp_Tile = 32;
        constexpr ck_tile::index_t K_Warp_Tile = 16;

        constexpr bool kPadM      = PadM;
        constexpr bool kPadN      = PadN;
        constexpr bool kPadK      = PadK;
        constexpr bool preshuffle = Preshuffle;

        constexpr bool DoubleSmemBuffer   = false;
        constexpr int kBlockPerCu         = 1;
        constexpr bool StructuredSparsity = false;
        constexpr bool NumWaveGroup       = 1;

        using GemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                   ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                   ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

        using TilePartitioner =
            ck_tile::StreamKTilePartitioner<GemmShape, ReductionStrategy, Persistent>;

        using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<kPadM,
                                                                     kPadN,
                                                                     kPadK,
                                                                     DoubleSmemBuffer,
                                                                     ALayout,
                                                                     BLayout,
                                                                     CLayout,
                                                                     TransposeC,
                                                                     StructuredSparsity,
                                                                     Persistent,
                                                                     NumWaveGroup,
                                                                     preshuffle>;

        constexpr auto scheduler = ck_tile::GemmPipelineScheduler::Intrawave;

        // We create the GEMM pipeline without specifying has_hot_loop or tail_num.
        // This is because num_loop can vary (a) per WG and (b) per iteration of the Stream-K
        // while loop. Instead, has_hot_loop and tail_num are determined in the Stream-K
        // Kernel's RunGemm function. This is a similar pattern used by grouped GEMM.
        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                           BDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           scheduler>;
        // For initial testing, we will just test with one pipeline.
        // More extensive testing is coming later and will test other pipelines.
        using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<UniversalGemmProblem>;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             ck_tile::tuple<>,
                                             AccDataType,
                                             CDataType,
                                             ck_tile::tuple<>,
                                             CLayout,
                                             ck_tile::element_wise::PassThrough,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             M_Warp,
                                             N_Warp,
                                             M_Warp_Tile,
                                             N_Warp_Tile,
                                             K_Warp_Tile,
                                             UniversalGemmProblem::TransposeC>>;

        using Kernel = ck_tile::StreamKKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

        auto kargs                = Kernel::MakeKernelArgs(args);
        const auto workspace_size = Kernel::GetWorkSpaceSize(kargs);
        ck_tile::DeviceMem workspace_data(workspace_size);
        workspace_data.SetZero();
        kargs.workspace_ptr = workspace_data.GetDeviceBuffer();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            EXPECT_TRUE(false);
        }

        dim3 grid_dims  = Kernel::GridSize(kargs.tile_partitioner);
        dim3 block_dims = Kernel::BlockSize();

        ck_tile::ignore = ck_tile::launch_kernel(
            s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grid_dims, block_dims, 0, kargs));

        return kargs.tile_partitioner.estimate_num_wgs_per_tile();
    }

    public:
    void Run(ck_tile::index_t M,
             ck_tile::index_t N,
             ck_tile::index_t K,
             ck_tile::StreamKReductionStrategy reduction_strategy =
                 ck_tile::StreamKReductionStrategy::Atomic,
             ck_tile::index_t stride_A = 0,
             ck_tile::index_t stride_B = 0,
             ck_tile::index_t stride_C = 0)
    {
        // Since M, N, and K will vary depending on the number of CUs, we print it here to
        // facilitate test output readability.
        std::cout << "M: " << M << ", N: " << N << ", K: " << K << std::endl;

        using namespace ck_tile::literals;

        auto f_host_tensor_descriptor = [](std::size_t row,
                                           std::size_t col,
                                           std::size_t stride,
                                           auto layout) {
            if constexpr(std::is_same_v<decltype(layout), ck_tile::tensor_layout::gemm::RowMajor>)
            {
                return ck_tile::HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return ck_tile::HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

        auto f_get_default_stride =
            [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
                if(stride == 0)
                {
                    if constexpr(std::is_same_v<decltype(layout),
                                                ck_tile::tensor_layout::gemm::RowMajor>)
                    {
                        return col;
                    }
                    else
                    {
                        return row;
                    }
                }
                else
                    return stride;
            };

        stride_A = f_get_default_stride(M, K, stride_A, ALayout{});
        stride_B = f_get_default_stride(K, N, stride_B, BLayout{});
        stride_C = f_get_default_stride(M, N, stride_C, CLayout{});

        ck_tile::HostTensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, stride_A, ALayout{}));
        ck_tile::HostTensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, stride_B, BLayout{}));
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(
            f_host_tensor_descriptor(M, N, stride_C, CLayout{}));

        ck_tile::FillUniformDistributionIntegerValue<ADataType>{-5, 5, /*seed*/ 11939}(a_m_k);
        ck_tile::FillUniformDistributionIntegerValue<BDataType>{-5, 5, /*seed*/ 11940}(b_k_n);

        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

        a_m_k_dev_buf.ToDevice(a_m_k.data());
        b_k_n_dev_buf.ToDevice(b_k_n.data());
        c_m_n_dev_buf.SetZero();
        c_m_n_dev_result.SetZero();

        ck_tile::StreamKHostArgs args{a_m_k_dev_buf.GetDeviceBuffer(),
                                      b_k_n_dev_buf.GetDeviceBuffer(),
                                      c_m_n_dev_buf.GetDeviceBuffer(),
                                      M,
                                      N,
                                      K,
                                      stride_A,
                                      stride_B,
                                      stride_C};

        ck_tile::index_t num_accumulations_per_tile;

        if(reduction_strategy == ck_tile::StreamKReductionStrategy::Atomic)
        {
            num_accumulations_per_tile = invoke_streamk<ck_tile::StreamKReductionStrategy::Atomic>(
                args, ck_tile::stream_config{nullptr, false, 0, 0, 1});
        }
        else if(reduction_strategy == ck_tile::StreamKReductionStrategy::Reduction)
        {
            num_accumulations_per_tile =
                invoke_streamk<ck_tile::StreamKReductionStrategy::Reduction>(
                    args, ck_tile::stream_config{nullptr, false, 0, 0, 1});
        }
        else
        {
            num_accumulations_per_tile =
                invoke_streamk<ck_tile::StreamKReductionStrategy::TreeReduction>(
                    args, ck_tile::stream_config{nullptr, false, 0, 0, 1});
        }

        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());

        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            f_host_tensor_descriptor(M, N, stride_C, CLayout{}));
        c_m_n_host_ref.SetZero();

        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_m_k, b_k_n, c_m_n_host_ref);

        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, num_accumulations_per_tile, max_accumulated_value);

        bool pass = ck_tile::check_err(c_m_n_dev_result,
                                       c_m_n_host_ref,
                                       "Error: Incorrect results!",
                                       rtol_atol.at(ck_tile::number<0>{}),
                                       rtol_atol.at(ck_tile::number<1>{}));

        EXPECT_TRUE(pass);
    };
};
