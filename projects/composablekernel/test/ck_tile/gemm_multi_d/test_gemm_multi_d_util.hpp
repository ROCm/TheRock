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
#include "ck_tile/ops/gemm/kernel/gemm_multi_d_kernel.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

struct ElementWiseAddAdd
{
    template <typename E, typename C, typename D0, typename D1>
    CK_TILE_HOST_DEVICE auto operator()(E& e, const C& c, const D0& d0, const D1& d1) const -> void
    {
        const float x0_f = ck_tile::type_convert<float>(c) + ck_tile::type_convert<float>(d0) +
                           ck_tile::type_convert<float>(d1);

        e = ck_tile::type_convert<E>(x0_f);
    }
};

struct MultiplyMultiply
{
    template <typename E, typename C, typename D0, typename D1>
    CK_TILE_HOST_DEVICE auto operator()(E& e, const C& c, const D0& d0, const D1& d1) const -> void
    {
        const float x0_f = ck_tile::type_convert<float>(c) * ck_tile::type_convert<float>(d0) *
                           ck_tile::type_convert<float>(d1);

        e = ck_tile::type_convert<E>(x0_f);
    }
};

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename EDataType,
          typename DsDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeTypeAB =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;

    using ComputeType =
        std::conditional_t<sizeof(ComputeTypeAB) < sizeof(DsDataType), ComputeTypeAB, DsDataType>;

    // Calculate thresholds
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, EDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, EDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
    // Calculate error due to split_k accumulation
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<EDataType, EDataType, EDataType>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<EDataType, EDataType, EDataType>(
        max_accumulated_value, kbatch);
    // Use higher threshold
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

template <typename Tuple>
class TestCkTileGemmMultiD : public ::testing::Test
{
    protected:
    using ALayout           = std::tuple_element_t<0, Tuple>;
    using BLayout           = std::tuple_element_t<1, Tuple>;
    using D0Layout          = std::tuple_element_t<2, Tuple>;
    using D1Layout          = std::tuple_element_t<3, Tuple>;
    using ELayout           = std::tuple_element_t<4, Tuple>;
    using ADataType         = std::tuple_element_t<5, Tuple>;
    using BDataType         = std::tuple_element_t<6, Tuple>;
    using D0DataType        = std::tuple_element_t<7, Tuple>;
    using D1DataType        = std::tuple_element_t<8, Tuple>;
    using AccDataType       = std::tuple_element_t<9, Tuple>;
    using EDataType         = std::tuple_element_t<10, Tuple>;
    using CDElementWiseFn   = std::tuple_element_t<11, Tuple>;
    using UseCshuffleEpilog = std::tuple_element_t<12, Tuple>;
    using DsLayout          = ck_tile::tuple<D0Layout, D1Layout>;
    using DsDataType        = ck_tile::tuple<D0DataType, D1DataType>;

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

    template <typename GemmWarpConfig,
              typename ADataType,
              typename BDataType,
              typename DsDataType,
              typename AccDataType,
              typename EDataType,
              typename ALayout,
              typename BLayout,
              typename DsLayout,
              typename ELayout,
              typename CDEElementWise = ck_tile::element_wise::PassThrough>
    void invoke_gemm_multi_d(const ck_tile::GemmMultiDHostArgs<DsDataType::size()>& args,
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
                                                                     ELayout,
                                                                     TransposeC>;

        constexpr auto scheduler = ck_tile::GemmPipelineScheduler::Intrawave;

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                           BDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           scheduler>;

        using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<UniversalGemmProblem>;

        using DefaultGemmEpilogue = ck_tile::DefaultGemm2DEpilogue<
            ck_tile::DefaultGemm2DEpilogueProblem<ADataType,
                                                  BDataType,
                                                  DsDataType,
                                                  AccDataType,
                                                  EDataType,
                                                  DsLayout,
                                                  ELayout,
                                                  CDEElementWise,
                                                  TilePartitioner::MPerBlock,
                                                  TilePartitioner::NPerBlock,
                                                  kPadM,
                                                  kPadN,
                                                  M_Warp_Tile,
                                                  N_Warp_Tile,
                                                  K_Warp_Tile,
                                                  UniversalGemmProblem::TransposeC,
                                                  true>>;

        using CShuffleGemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             DsDataType,
                                             AccDataType,
                                             EDataType,
                                             DsLayout,
                                             ELayout,
                                             CDEElementWise,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             M_Warp,
                                             N_Warp,
                                             M_Warp_Tile,
                                             N_Warp_Tile,
                                             K_Warp_Tile,
                                             UniversalGemmProblem::TransposeC>>;

        using GemmEpilogue =
            std::conditional_t<UseCshuffleEpilog::value, CShuffleGemmEpilogue, DefaultGemmEpilogue>;

        using Kernel = ck_tile::GemmKernelMultiD<TilePartitioner, GemmPipeline, GemmEpilogue>;
        auto kargs   = Kernel::MakeKernelArgs(args);

        const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
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
    bool Run(const int M,
             const int N,
             const int K,
             const int k_batch,
             int StrideA  = 0,
             int StrideB  = 0,
             int StrideD0 = 0,
             int StrideD1 = 0,
             int StrideE  = 0)
    {
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

        StrideA  = f_get_default_stride(M, K, StrideA, ALayout{});
        StrideB  = f_get_default_stride(K, N, StrideB, BLayout{});
        StrideD0 = f_get_default_stride(M, N, StrideD0, D0Layout{});
        StrideD1 = f_get_default_stride(M, N, StrideD1, D1Layout{});
        StrideE  = f_get_default_stride(M, N, StrideE, ELayout{});

        ck_tile::HostTensor<ADataType> a_m_k_tesnor(
            f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
        ck_tile::HostTensor<BDataType> b_k_n_tensors(
            f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
        ck_tile::HostTensor<D0DataType> d0_m_n_tensors(
            f_host_tensor_descriptor(M, N, StrideD0, D0Layout{}));
        ck_tile::HostTensor<D1DataType> d1_m_n_tensors(
            f_host_tensor_descriptor(M, N, StrideD1, D1Layout{}));
        ck_tile::HostTensor<EDataType> e_m_n_device_result(
            f_host_tensor_descriptor(M, N, StrideE, ELayout{}));

        ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_m_k_tesnor);
        ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_k_n_tensors);
        ck_tile::FillUniformDistribution<D0DataType>{-1.f, 1.f}(d0_m_n_tensors);
        ck_tile::FillUniformDistribution<D1DataType>{-1.f, 1.f}(d1_m_n_tensors);

        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k_tesnor.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n_tensors.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d0_m_n_dev_buf(d0_m_n_tensors.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d1_m_n_dev_buf(d1_m_n_tensors.get_element_space_size_in_bytes());
        ck_tile::DeviceMem e_m_n_dev_buf(e_m_n_device_result.get_element_space_size_in_bytes());

        a_m_k_dev_buf.ToDevice(a_m_k_tesnor.mData.data());
        b_k_n_dev_buf.ToDevice(b_k_n_tensors.mData.data());
        d0_m_n_dev_buf.ToDevice(d0_m_n_tensors.mData.data());
        d1_m_n_dev_buf.ToDevice(d1_m_n_tensors.mData.data());

        e_m_n_dev_buf.SetZero();
        e_m_n_device_result.SetZero();

        std::array<const void*, DsDataType::size()> ds_ptr_buf = {d0_m_n_dev_buf.GetDeviceBuffer(),
                                                                  d1_m_n_dev_buf.GetDeviceBuffer()};
        std::array<ck_tile::index_t, DsDataType::size()> stridesDs = {StrideD0, StrideD1};

        ck_tile::GemmMultiDHostArgs<DsDataType::size()> args({a_m_k_dev_buf.GetDeviceBuffer(),
                                                              b_k_n_dev_buf.GetDeviceBuffer(),
                                                              ds_ptr_buf,
                                                              e_m_n_dev_buf.GetDeviceBuffer(),
                                                              k_batch,
                                                              M,
                                                              N,
                                                              K,
                                                              StrideA,
                                                              StrideB,
                                                              stridesDs,
                                                              StrideE});
#if CK_TILE_USE_WMMA
        invoke_gemm_multi_d<GemmWarpConfig_Wmma,
                            ADataType,
                            BDataType,
                            DsDataType,
                            AccDataType,
                            EDataType,
                            ALayout,
                            BLayout,
                            DsLayout,
                            ELayout,
                            CDElementWiseFn>(args, ck_tile::stream_config{nullptr, false});
#else
        invoke_gemm_multi_d<GemmWarpConfig_Mfma,
                            ADataType,
                            BDataType,
                            DsDataType,
                            AccDataType,
                            EDataType,
                            ALayout,
                            BLayout,
                            DsLayout,
                            ELayout,
                            CDElementWiseFn>(args, ck_tile::stream_config{nullptr, false});
#endif

        std::cout << "Run kernel with M =" << M << " N =" << N << " K =" << K
                  << " StrideA =" << StrideA << " StrideB =" << StrideB << " StrideE =" << StrideE
                  << " StrideD0 =" << StrideD0 << " StrideD1 =" << StrideD1 << std::endl;

        e_m_n_dev_buf.FromDevice(e_m_n_device_result.data());
        bool pass = true;

        ck_tile::HostTensor<EDataType> e_m_n_host_ref(
            f_host_tensor_descriptor(M, N, StrideE, ELayout{}));
        e_m_n_host_ref.SetZero();

        ck_tile::reference_gemm_multiple_d<ADataType,
                                           BDataType,
                                           DsDataType,
                                           AccDataType,
                                           EDataType,
                                           CDElementWiseFn>(
            a_m_k_tesnor, b_k_n_tensors, {d0_m_n_tensors, d1_m_n_tensors}, e_m_n_host_ref);

        const float max_accumulated_value =
            *std::max_element(e_m_n_host_ref.mData.begin(), e_m_n_host_ref.mData.end());
        const auto rtol_atol =
            calculate_rtol_atol<ADataType, BDataType, AccDataType, EDataType, DsDataType>(
                K, k_batch, max_accumulated_value);
        pass = ck_tile::check_err(e_m_n_device_result,
                                  e_m_n_host_ref,
                                  "Error: Incorrect results!",
                                  rtol_atol.at(ck_tile::number<0>{}),
                                  rtol_atol.at(ck_tile::number<1>{}));
        std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                  << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                  << std::endl;

        return pass;
    }
};
