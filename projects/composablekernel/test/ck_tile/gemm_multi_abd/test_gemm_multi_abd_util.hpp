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
#include "ck_tile/ops/gemm/kernel/gemm_multi_abd_kernel.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

using AddScale          = ck_tile::element_wise::AddScale;
using ElementWiseAddAdd = ck_tile::element_wise::MultiDAdd;
using MultiplyMultiply  = ck_tile::element_wise::MultiDMultiply;

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <typename PrecType, ck_tile::index_t M_Warp_Tile>
constexpr ck_tile::index_t get_k_warp_tile()
{
#if CK_TILE_USE_WMMA
    return 16;
#else
#if defined(CK_GFX950_SUPPORT)
    constexpr bool is_8bit_float =
        std::is_same_v<PrecType, ck_tile::fp8_t> || std::is_same_v<PrecType, ck_tile::bf8_t>;
    if constexpr(M_Warp_Tile == 32)
        return is_8bit_float ? 64 : 16;
    else
        return is_8bit_float ? 128 : 32;
#else
    if constexpr(M_Warp_Tile == 32)
        return 16;
    else
        return 32;
#endif
#endif
}

template <typename A0DataType,
          typename B0DataType,
          typename AccDataType,
          typename EDataType,
          typename D0DataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeTypeAB =
        std::conditional_t<sizeof(A0DataType) < sizeof(B0DataType), A0DataType, B0DataType>;

    using ComputeType =
        std::conditional_t<sizeof(ComputeTypeAB) < sizeof(D0DataType), ComputeTypeAB, D0DataType>;
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
class TestCkTileGemmMultiABD : public ::testing::Test
{
    protected:
    using A0Layout          = std::tuple_element_t<0, Tuple>;
    using A1Layout          = std::tuple_element_t<1, Tuple>;
    using B0Layout          = std::tuple_element_t<2, Tuple>;
    using B1Layout          = std::tuple_element_t<3, Tuple>;
    using D0Layout          = std::tuple_element_t<4, Tuple>;
    using D1Layout          = std::tuple_element_t<5, Tuple>;
    using ELayout           = std::tuple_element_t<6, Tuple>;
    using A0DataType        = std::tuple_element_t<7, Tuple>;
    using A1DataType        = std::tuple_element_t<8, Tuple>;
    using B0DataType        = std::tuple_element_t<9, Tuple>;
    using B1DataType        = std::tuple_element_t<10, Tuple>;
    using D0DataType        = std::tuple_element_t<11, Tuple>;
    using D1DataType        = std::tuple_element_t<12, Tuple>;
    using AccDataType       = std::tuple_element_t<13, Tuple>;
    using EDataType         = std::tuple_element_t<14, Tuple>;
    using AElementWiseFn    = std::tuple_element_t<15, Tuple>;
    using BElementWiseFn    = std::tuple_element_t<16, Tuple>;
    using CDElementWiseFn   = std::tuple_element_t<17, Tuple>;
    using UseCshuffleEpilog = std::tuple_element_t<18, Tuple>;

    using AsLayout   = ck_tile::tuple<A0Layout, A1Layout>;
    using AsDataType = ck_tile::tuple<A0DataType, A1DataType>;
    using BsLayout   = ck_tile::tuple<B0Layout, B1Layout>;
    using BsDataType = ck_tile::tuple<B0DataType, B1DataType>;
    using DsLayout   = ck_tile::tuple<D0Layout, D1Layout>;
    using DsDataType = ck_tile::tuple<D0DataType, D1DataType>;

    template <typename AsDataType,
              typename BsDataType,
              typename DsDataType,
              typename AccDataType,
              typename EDataType,
              typename AsLayout,
              typename BsLayout,
              typename DsLayout,
              typename ELayout,
              typename AElementWise    = ck_tile::element_wise::PassThrough,
              typename BElementWise    = ck_tile::element_wise::PassThrough,
              typename CDElementWiseFn = ck_tile::element_wise::PassThrough>
    void invoke_gemm_multi_abd(const ck_tile::GemmMultiABDHostArgs<AsDataType::size(),
                                                                   BsDataType::size(),
                                                                   DsDataType::size()>& args,
                               const ck_tile::stream_config& s)
    {
        constexpr ck_tile::index_t M_Tile = 128;
        constexpr ck_tile::index_t N_Tile = 128;
        constexpr ck_tile::index_t K_Tile = 64;

        constexpr ck_tile::index_t M_Warp = 2;
        constexpr ck_tile::index_t N_Warp = 2;
        constexpr ck_tile::index_t K_Warp = 1;

#if CK_TILE_USE_WMMA
        using ADataType =
            ck_tile::remove_cvref_t<std::tuple_element_t<ck_tile::number<0>{}, AsDataType>>;
        constexpr ck_tile::index_t M_Warp_Tile = 16;
        constexpr ck_tile::index_t N_Warp_Tile = 16;
        constexpr ck_tile::index_t K_Warp_Tile = get_k_warp_tile<ADataType, N_Warp_Tile>();
#else
        constexpr ck_tile::index_t M_Warp_Tile = 32;
        constexpr ck_tile::index_t N_Warp_Tile = 32;
        constexpr ck_tile::index_t K_Warp_Tile = 16;
#endif

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
                                                                     AsLayout,
                                                                     BsLayout,
                                                                     ELayout,
                                                                     TransposeC>;

        constexpr auto scheduler = ck_tile::GemmPipelineScheduler::Intrawave;

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<AsDataType,
                                                                           BsDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           scheduler,
                                                                           AElementWise,
                                                                           BElementWise>;

        using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<UniversalGemmProblem>;

        using DefaultGemmEpilogue = ck_tile::DefaultGemm2DEpilogue<
            ck_tile::DefaultGemm2DEpilogueProblem<AsDataType,
                                                  BsDataType,
                                                  DsDataType,
                                                  AccDataType,
                                                  EDataType,
                                                  DsLayout,
                                                  ELayout,
                                                  CDElementWiseFn,
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
            ck_tile::CShuffleEpilogueProblem<AsDataType,
                                             BsDataType,
                                             DsDataType,
                                             AccDataType,
                                             EDataType,
                                             DsLayout,
                                             ELayout,
                                             CDElementWiseFn,
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

        using Kernel = ck_tile::GemmKernelMultiABD<TilePartitioner, GemmPipeline, GemmEpilogue>;
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
             int StrideA0 = 0,
             int StrideA1 = 0,
             int StrideB0 = 0,
             int StrideB1 = 0,
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

        StrideA0 = f_get_default_stride(M, K, StrideA0, A0Layout{});
        StrideA1 = f_get_default_stride(M, K, StrideA1, A1Layout{});

        StrideB0 = f_get_default_stride(K, N, StrideB0, B0Layout{});
        StrideB1 = f_get_default_stride(K, N, StrideB1, B1Layout{});

        StrideD0 = f_get_default_stride(M, N, StrideD0, D0Layout{});
        StrideD1 = f_get_default_stride(M, N, StrideD1, D1Layout{});

        StrideE = f_get_default_stride(M, N, StrideE, ELayout{});

        ck_tile::HostTensor<A0DataType> a0_m_k_tesnor(
            f_host_tensor_descriptor(M, K, StrideA0, A0Layout{}));
        ck_tile::HostTensor<A1DataType> a1_m_k_tesnor(
            f_host_tensor_descriptor(M, K, StrideA1, A1Layout{}));

        ck_tile::HostTensor<B0DataType> b0_k_n_tensors(
            f_host_tensor_descriptor(K, N, StrideB0, B0Layout{}));
        ck_tile::HostTensor<B1DataType> b1_k_n_tensors(
            f_host_tensor_descriptor(K, N, StrideB1, B1Layout{}));

        ck_tile::HostTensor<D0DataType> d0_m_n_tensors(
            f_host_tensor_descriptor(M, N, StrideD0, D0Layout{}));
        ck_tile::HostTensor<D1DataType> d1_m_n_tensors(
            f_host_tensor_descriptor(M, N, StrideD1, D1Layout{}));

        ck_tile::HostTensor<EDataType> e_m_n_device_result(
            f_host_tensor_descriptor(M, N, StrideE, ELayout{}));

        ck_tile::FillUniformDistribution<A0DataType>{-1.f, 1.f}(a0_m_k_tesnor);
        ck_tile::FillUniformDistribution<A0DataType>{-1.f, 1.f}(a1_m_k_tesnor);

        ck_tile::FillUniformDistribution<B0DataType>{-1.f, 1.f}(b0_k_n_tensors);
        ck_tile::FillUniformDistribution<B1DataType>{-1.f, 1.f}(b1_k_n_tensors);

        ck_tile::FillUniformDistribution<D0DataType>{-1.f, 1.f}(d0_m_n_tensors);
        ck_tile::FillUniformDistribution<D1DataType>{-1.f, 1.f}(d1_m_n_tensors);

        ck_tile::DeviceMem a0_m_k_dev_buf(a0_m_k_tesnor.get_element_space_size_in_bytes());
        ck_tile::DeviceMem a1_m_k_dev_buf(a1_m_k_tesnor.get_element_space_size_in_bytes());

        ck_tile::DeviceMem b0_k_n_dev_buf(b0_k_n_tensors.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b1_k_n_dev_buf(b1_k_n_tensors.get_element_space_size_in_bytes());

        ck_tile::DeviceMem d0_m_n_dev_buf(d0_m_n_tensors.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d1_m_n_dev_buf(d1_m_n_tensors.get_element_space_size_in_bytes());

        ck_tile::DeviceMem e_m_n_dev_buf(e_m_n_device_result.get_element_space_size_in_bytes());

        a0_m_k_dev_buf.ToDevice(a0_m_k_tesnor.mData.data());
        a1_m_k_dev_buf.ToDevice(a1_m_k_tesnor.mData.data());

        b0_k_n_dev_buf.ToDevice(b0_k_n_tensors.mData.data());
        b1_k_n_dev_buf.ToDevice(b1_k_n_tensors.mData.data());

        d0_m_n_dev_buf.ToDevice(d0_m_n_tensors.mData.data());
        d1_m_n_dev_buf.ToDevice(d1_m_n_tensors.mData.data());

        e_m_n_dev_buf.SetZero();
        e_m_n_device_result.SetZero();

        std::array<const void*, DsDataType::size()> as_ptr_buf = {a0_m_k_dev_buf.GetDeviceBuffer(),
                                                                  a1_m_k_dev_buf.GetDeviceBuffer()};

        std::array<const void*, DsDataType::size()> bs_ptr_buf = {b0_k_n_dev_buf.GetDeviceBuffer(),
                                                                  b1_k_n_dev_buf.GetDeviceBuffer()};

        std::array<const void*, DsDataType::size()> ds_ptr_buf = {d0_m_n_dev_buf.GetDeviceBuffer(),
                                                                  d1_m_n_dev_buf.GetDeviceBuffer()};

        std::array<ck_tile::index_t, AsDataType::size()> strideAs = {StrideA0, StrideA1};
        std::array<ck_tile::index_t, BsDataType::size()> strideBs = {StrideB0, StrideB1};
        std::array<ck_tile::index_t, DsDataType::size()> strideDs = {StrideD0, StrideD1};

        ck_tile::GemmMultiABDHostArgs<AsDataType::size(), BsDataType::size(), DsDataType::size()>
            args({as_ptr_buf,
                  bs_ptr_buf,
                  ds_ptr_buf,
                  e_m_n_dev_buf.GetDeviceBuffer(),
                  k_batch,
                  M,
                  N,
                  K,
                  strideAs,
                  strideBs,
                  strideDs,
                  StrideE});

        invoke_gemm_multi_abd<AsDataType,
                              BsDataType,
                              DsDataType,
                              AccDataType,
                              EDataType,
                              AsLayout,
                              BsLayout,
                              DsLayout,
                              ELayout,
                              AElementWiseFn,
                              BElementWiseFn,
                              CDElementWiseFn>(args, ck_tile::stream_config{nullptr, false});

        std::cout << "Run kernel with M =" << M << " N =" << N << " K =" << K
                  << " StrideA0 =" << StrideA0 << " StrideA1 =" << StrideA1
                  << " StrideB0 =" << StrideB0 << " StrideB1 =" << StrideB1
                  << " StrideE =" << StrideE << " StrideD0 =" << StrideD0
                  << " StrideD1 =" << StrideD1 << std::endl;

        e_m_n_dev_buf.FromDevice(e_m_n_device_result.data());
        bool pass = true;

        ck_tile::HostTensor<A0DataType> a_m_k_host_ref_element_result(
            f_host_tensor_descriptor(M, K, StrideA0, A0Layout{}));
        ck_tile::HostTensor<B0DataType> b_k_n_host_ref_element_result(
            f_host_tensor_descriptor(K, N, StrideB0, B0Layout{}));
        ck_tile::HostTensor<EDataType> e_m_n_host_ref(
            f_host_tensor_descriptor(M, N, StrideE, ELayout{}));
        a_m_k_host_ref_element_result.SetZero();
        b_k_n_host_ref_element_result.SetZero();
        e_m_n_host_ref.SetZero();

        ck_tile::reference_gemm_multiple_abd<AsDataType,
                                             BsDataType,
                                             DsDataType,
                                             AccDataType,
                                             EDataType,
                                             AElementWiseFn,
                                             BElementWiseFn,
                                             CDElementWiseFn>({a0_m_k_tesnor, a1_m_k_tesnor},
                                                              {b0_k_n_tensors, b1_k_n_tensors},
                                                              {d0_m_n_tensors, d1_m_n_tensors},
                                                              a_m_k_host_ref_element_result,
                                                              b_k_n_host_ref_element_result,
                                                              e_m_n_host_ref);

        const float max_accumulated_value =
            *std::max_element(e_m_n_host_ref.mData.begin(), e_m_n_host_ref.mData.end());
        const auto rtol_atol =
            calculate_rtol_atol<A0DataType, B0DataType, AccDataType, EDataType, D0DataType>(
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
