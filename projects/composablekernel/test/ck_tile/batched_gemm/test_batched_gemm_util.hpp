// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <sstream>
#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/batched_gemm_kernel.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

template <typename Tuple>
class TestCkTileBatchedGemm : public ::testing::Test
{
    protected:
    using ALayout     = std::tuple_element_t<0, Tuple>;
    using BLayout     = std::tuple_element_t<1, Tuple>;
    using CLayout     = std::tuple_element_t<2, Tuple>;
    using ADataType   = std::tuple_element_t<3, Tuple>;
    using BDataType   = std::tuple_element_t<4, Tuple>;
    using AccDataType = std::tuple_element_t<5, Tuple>;
    using CDataType   = std::tuple_element_t<6, Tuple>;
    using DsLayout    = ck_tile::tuple<>;
    using DsDataType  = ck_tile::tuple<>;

    struct GemmWarpConfig_Mfma
    {
        static constexpr ck_tile::index_t M_Tile      = 256;
        static constexpr ck_tile::index_t N_Tile      = 256;
        static constexpr ck_tile::index_t K_Tile      = 64;
        static constexpr ck_tile::index_t M_Warp_Tile = 32;
        static constexpr ck_tile::index_t N_Warp_Tile = 32;
        static constexpr ck_tile::index_t K_Warp_Tile = 16;
    };

    struct GemmWarpConfig_Wmma
    {
        static constexpr ck_tile::index_t M_Tile      = 128;
        static constexpr ck_tile::index_t N_Tile      = 128;
        static constexpr ck_tile::index_t K_Tile      = 64;
        static constexpr ck_tile::index_t M_Warp_Tile = 16;
        static constexpr ck_tile::index_t N_Warp_Tile = 16;
        static constexpr ck_tile::index_t K_Warp_Tile = 16;
    };

    template <typename GemmWarpConfig, typename ALayout, typename BLayout, typename CLayout>
    void invoke_batched_gemm(const ck_tile::BatchedGemmHostArgs& args,
                             const ck_tile::stream_config& s)
    {
        constexpr ck_tile::index_t M_Tile = GemmWarpConfig::M_Tile;
        constexpr ck_tile::index_t N_Tile = GemmWarpConfig::N_Tile;
        constexpr ck_tile::index_t K_Tile = GemmWarpConfig::K_Tile;

        constexpr ck_tile::index_t M_Warp = 2;
        constexpr ck_tile::index_t N_Warp = 2;
        constexpr ck_tile::index_t K_Warp = 1;

        constexpr ck_tile::index_t M_Warp_Tile = GemmWarpConfig::M_Warp_Tile;
        constexpr ck_tile::index_t N_Warp_Tile = GemmWarpConfig::N_Warp_Tile;
        constexpr ck_tile::index_t K_Warp_Tile = GemmWarpConfig::K_Warp_Tile;

        constexpr bool DoubleSmemBuffer = false;

        constexpr bool kPadM = false;
        constexpr bool kPadN = false;
        constexpr bool kPadK = false;

        constexpr bool TransposeC = false;

        constexpr int kBlockPerCu                         = 1;
        constexpr ck_tile::index_t TileParitionerGroupNum = 8;
        constexpr ck_tile::index_t TileParitionerM01      = 4;

        using GemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                   ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                   ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;
        using TilePartitioner = ck_tile::
            GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;

        using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<kPadM,
                                                                     kPadN,
                                                                     kPadK,
                                                                     DoubleSmemBuffer,
                                                                     ALayout,
                                                                     BLayout,
                                                                     CLayout,
                                                                     TransposeC>;
        constexpr auto scheduler  = ck_tile::GemmPipelineScheduler::Intrawave;

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                           BDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           scheduler>;

        using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<UniversalGemmProblem>;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             DsDataType,
                                             AccDataType,
                                             CDataType,
                                             DsLayout,
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
        using Kernel = ck_tile::BatchedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
        auto kargs   = Kernel::MakeKernelArgs(args);

        const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch, args.batch_count);
        const dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args: " << Kernel::GetName() << '\n'
                      << "shape: " << GemmShape::GetName() << '\n'
                      << "pipeline: " << GemmPipeline::GetName() << '\n'
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }

        ck_tile::ignore = ck_tile::launch_kernel(
            s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
    }

    public:
    void Run(const int M,
             const int N,
             const int K,
             int StrideA            = 512,
             int StrideB            = 512,
             int StrideC            = 256,
             const int BatchStrideA = 131072,
             const int BatchStrideB = 131072,
             const int BatchStrideC = 65536,
             const int BatchCount   = 8)
    {
        using namespace ck_tile::literals;

        auto f_host_tensor_descriptor = [](std::size_t batch_count_,
                                           std::size_t row,
                                           std::size_t col,
                                           std::size_t stride,
                                           std::size_t batch_stride,
                                           auto layout) {
            if constexpr(std::is_same_v<decltype(layout), ck_tile::tensor_layout::gemm::RowMajor>)
            {
                return ck_tile::HostTensorDescriptor({batch_count_, row, col},
                                                     {batch_stride, stride, 1_uz});
            }
            else
            {
                return ck_tile::HostTensorDescriptor({batch_count_, row, col},
                                                     {batch_stride, 1_uz, stride});
            }
        };

        auto f_get_default_stride =
            [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
                if(stride == 0)
                {
                    // give a chance if stride is zero, return a default packed stride
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

        StrideA = f_get_default_stride(M, K, StrideA, ALayout{});
        StrideB = f_get_default_stride(K, N, StrideB, BLayout{});
        StrideC = f_get_default_stride(M, N, StrideC, CLayout{});

        ck_tile::HostTensor<ADataType> a_m_k(
            f_host_tensor_descriptor(BatchCount, M, K, StrideA, BatchStrideA, ALayout{}));
        ck_tile::HostTensor<BDataType> b_k_n(
            f_host_tensor_descriptor(BatchCount, K, N, StrideB, BatchStrideB, BLayout{}));
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(
            f_host_tensor_descriptor(BatchCount, M, N, StrideC, BatchStrideC, CLayout{}));

        ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_m_k);
        ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_k_n);

        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

        a_m_k_dev_buf.ToDevice(a_m_k.data());
        b_k_n_dev_buf.ToDevice(b_k_n.data());
        c_m_n_dev_buf.SetZero();
        c_m_n_dev_result.SetZero();

        ck_tile::BatchedGemmHostArgs args{a_m_k_dev_buf.GetDeviceBuffer(),
                                          b_k_n_dev_buf.GetDeviceBuffer(),
                                          c_m_n_dev_buf.GetDeviceBuffer(),
                                          1,
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideC,
                                          BatchStrideA,
                                          BatchStrideB,
                                          BatchStrideC,
                                          BatchCount};
#if CK_TILE_USE_WMMA
        invoke_batched_gemm<GemmWarpConfig_Wmma, ALayout, BLayout, CLayout>(
            args, ck_tile::stream_config{nullptr, false});
#else
        invoke_batched_gemm<GemmWarpConfig_Mfma, ALayout, BLayout, CLayout>(
            args, ck_tile::stream_config{nullptr, false});
#endif

        std::cout << "Run kernel with M =" << M << " N =" << N << " K =" << K
                  << " StrideA =" << StrideA << " StrideB =" << StrideB << " StrideC =" << StrideC
                  << " BatchStrideA =" << BatchStrideA << " BatchStrideB =" << BatchStrideB
                  << " BatchStrideC =" << BatchStrideC << " BatchCount =" << BatchCount
                  << std::endl;

        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());
        bool pass = true;

        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            f_host_tensor_descriptor(BatchCount, M, N, StrideC, BatchStrideC, CLayout{}));
        c_m_n_host_ref.SetZero();

        const auto b_n_k = b_k_n.transpose({0, 2, 1});
        ck_tile::reference_batched_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_m_k, b_n_k, c_m_n_host_ref);

        pass = ck_tile::check_err(c_m_n_dev_result, c_m_n_host_ref);
        EXPECT_TRUE(pass);
    }
};
