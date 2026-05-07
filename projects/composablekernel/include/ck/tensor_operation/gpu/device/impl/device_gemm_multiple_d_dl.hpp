// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_dl_multiple_d.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {

template <typename GridwiseGemm,
          typename ABDataType,
          typename DsPointer,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename AGridDesc_K0_M0_M1_K1,
          typename BGridDesc_K0_N0_N1_K1,
          typename DsGridDesc_M0_M10_M11_N0_N10_N11,
          typename CGridDesc_M0_M10_M11_N0_N10_N11,
          typename Block2CTileMap,
          bool HasMainKBlockLoop,
          bool HasDoubleTailKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_gemm_dl_multiple_d(
        const ABDataType* __restrict__ p_a_grid,
        const ABDataType* __restrict__ p_b_grid,
        DsPointer p_ds_grid,
        EDataType* __restrict__ p_e_grid,
        const AElementwiseOperation a_element_op,
        const BElementwiseOperation b_element_op,
        const CDEElementwiseOperation cde_element_op,
        const AGridDesc_K0_M0_M1_K1 a_grid_desc_k0_m0_m1_k1,
        const BGridDesc_K0_N0_N1_K1 b_grid_desc_k0_n0_n1_k1,
        const DsGridDesc_M0_M10_M11_N0_N10_N11 ds_grid_desc_m0_m10_m11_n0_n10_n11,
        const CGridDesc_M0_M10_M11_N0_N10_N11 e_grid_desc_m0_m10_m11_n0_n10_n11,
        const Block2CTileMap block_2_ctile_map)
{
#if(defined(__gfx906__) || defined(__gfx9__) || defined(__gfx101__) || defined(__gfx103__) || \
    defined(__gfx11__) || defined(__gfx12__))

    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(ABDataType);

    __shared__ ABDataType p_shared[shared_block_size];

    GridwiseGemm::Run(p_a_grid,
                      p_b_grid,
                      p_ds_grid,
                      p_e_grid,
                      p_shared,
                      a_element_op,
                      b_element_op,
                      cde_element_op,
                      a_grid_desc_k0_m0_m1_k1,
                      b_grid_desc_k0_n0_n1_k1,
                      ds_grid_desc_m0_m10_m11_n0_n10_n11,
                      e_grid_desc_m0_m10_m11_n0_n10_n11,
                      block_2_ctile_map,
                      integral_constant<bool, HasMainKBlockLoop>{},
                      integral_constant<bool, HasDoubleTailKBlockLoop>{});
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_ds_grid;
    ignore = p_e_grid;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = cde_element_op;
    ignore = a_grid_desc_k0_m0_m1_k1;
    ignore = b_grid_desc_k0_n0_n1_k1;
    ignore = ds_grid_desc_m0_m10_m11_n0_n10_n11;
    ignore = e_grid_desc_m0_m10_m11_n0_n10_n11;
    ignore = block_2_ctile_map;
#endif
}
} // namespace ck

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          GemmSpecialization GemmSpec,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t K0PerBlock,
          index_t K1,
          index_t M1PerThread,
          index_t N1PerThread,
          index_t KPerThread,
          typename M1N1ThreadClusterM1Xs,
          typename M1N1ThreadClusterN1Xs,
          typename ABlockTransferThreadSliceLengths_K0_M0_M1_K1,
          typename ABlockTransferThreadClusterLengths_K0_M0_M1_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          typename ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1,
          typename ABlockTransferSrcVectorTensorContiguousDimOrder,
          typename ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1,
          typename BBlockTransferThreadSliceLengths_K0_N0_N1_K1,
          typename BBlockTransferThreadClusterLengths_K0_N0_N1_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          typename BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1,
          typename BBlockTransferSrcVectorTensorContiguousDimOrder,
          typename BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1,
          typename CThreadTransferSrcDstAccessOrder,
          index_t CThreadTransferSrcDstVectorDim,
          index_t CThreadTransferDstScalarPerVector,
          enable_if_t<
              is_same_v<AElementwiseOperation, ck::tensor_operation::element_wise::PassThrough> &&
                  is_same_v<BElementwiseOperation, ck::tensor_operation::element_wise::PassThrough>,
              bool> = false>
struct DeviceGemmMultipleD_Dl : public DeviceGemmMultipleD<ALayout,
                                                           BLayout,
                                                           DsLayout,
                                                           ELayout,
                                                           ADataType,
                                                           BDataType,
                                                           DsDataType,
                                                           EDataType,
                                                           AElementwiseOperation,
                                                           BElementwiseOperation,
                                                           CDEElementwiseOperation>

{
    using DeviceOp                      = DeviceGemmMultipleD_Dl;
    static constexpr index_t NumDTensor = DsDataType::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    static constexpr auto K1Number = Number<K1>{};

    static auto MakeAGridDescriptor_K0_M_K1(index_t M, index_t K, index_t StrideA)
    {
        assert(K % K1 == 0);

        const index_t K0 = K / K1;

        const auto a_grid_desc_m_k = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(StrideA, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(I1, StrideA));
            }
        }();

        if constexpr(GemmSpec == GemmSpecialization::MNPadding)
        {
            const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;

            return transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_right_pad_transform(M, PadM)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_pass_through_transform(M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
    }

    static auto MakeBGridDescriptor_K0_N_K1(index_t K, index_t N, index_t StrideB)
    {
        assert(K % K1 == 0);

        const index_t K0 = K / K1;

        const auto b_grid_desc_k_n = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(StrideB, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(I1, StrideB));
            }
        }();

        if constexpr(GemmSpec == GemmSpecialization::MNPadding)
        {
            const auto PadN = (NPerBlock - N % NPerBlock) % NPerBlock;

            return transform_tensor_descriptor(
                b_grid_desc_k_n,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_right_pad_transform(N, PadN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                b_grid_desc_k_n,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
    }

    template <typename ELay>
    static auto MakeEGridDescriptor_M_N(index_t M, index_t N, index_t StrideE)
    {
        const auto c_grid_desc_m_n = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ELay>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(StrideE, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ELay>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(I1, StrideE));
            }
        }();

        if constexpr(GemmSpec == GemmSpecialization::MNPadding)
        {
            const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;
            const auto PadN = (NPerBlock - N % NPerBlock) % NPerBlock;

            return transform_tensor_descriptor(
                c_grid_desc_m_n,
                make_tuple(make_right_pad_transform(M, PadM), make_right_pad_transform(N, PadN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else
        {

            return transform_tensor_descriptor(
                c_grid_desc_m_n,
                make_tuple(make_pass_through_transform(M), make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
    }

    static auto MakeDsGridDescriptor_M_N(const std::array<index_t, NumDTensor>& MRaws,
                                         const std::array<index_t, NumDTensor>& NRaws,
                                         const std::array<index_t, NumDTensor>& DsStride)
    {
        return generate_tuple(
            [&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

                return DeviceOp::MakeEGridDescriptor_M_N<DLayout>(MRaws[i], NRaws[i], DsStride[i]);
            },
            Number<NumDTensor>{});
    }

    using AGridDesc_K0_M_K1 = decltype(MakeAGridDescriptor_K0_M_K1(1, 1, 1));
    using BGridDesc_K0_N_K1 = decltype(MakeBGridDescriptor_K0_N_K1(1, 1, 1));
    using DsGridDesc_M_N    = decltype(MakeDsGridDescriptor_M_N({}, {}, {}));
    using EGridDesc_M_N     = decltype(MakeEGridDescriptor_M_N<ELayout>(1, 1, 1));

    // GridwiseGemm
    using GridwiseGemm =
        GridwiseGemmDlMultipleD_km_kn_mn<BlockSize,
                                         ADataType,
                                         AccDataType,
                                         DsDataType,
                                         EDataType,
                                         AElementwiseOperation,
                                         BElementwiseOperation,
                                         CDEElementwiseOperation,
                                         InMemoryDataOperationEnum::Set,
                                         AGridDesc_K0_M_K1,
                                         BGridDesc_K0_N_K1,
                                         EGridDesc_M_N,
                                         MPerBlock,
                                         NPerBlock,
                                         K0PerBlock,
                                         K1,
                                         M1PerThread,
                                         N1PerThread,
                                         KPerThread,
                                         M1N1ThreadClusterM1Xs,
                                         M1N1ThreadClusterN1Xs,
                                         ABlockTransferThreadSliceLengths_K0_M0_M1_K1,
                                         ABlockTransferThreadClusterLengths_K0_M0_M1_K1,
                                         ABlockTransferThreadClusterArrangeOrder,
                                         ABlockTransferSrcAccessOrder,
                                         ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1,
                                         ABlockTransferSrcVectorTensorContiguousDimOrder,
                                         ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1,
                                         BBlockTransferThreadSliceLengths_K0_N0_N1_K1,
                                         BBlockTransferThreadClusterLengths_K0_N0_N1_K1,
                                         BBlockTransferThreadClusterArrangeOrder,
                                         BBlockTransferSrcAccessOrder,
                                         BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1,
                                         BBlockTransferSrcVectorTensorContiguousDimOrder,
                                         BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1,
                                         CThreadTransferSrcDstAccessOrder,
                                         CThreadTransferSrcDstVectorDim,
                                         CThreadTransferDstScalarPerVector>;

    using AGridDesc_K0_M0_M1_K1 =
        decltype(GridwiseGemm::MakeAGridDescriptor_K0_M0_M1_K1(AGridDesc_K0_M_K1{}));
    using BGridDesc_K0_N0_N1_K1 =
        decltype(GridwiseGemm::MakeBGridDescriptor_K0_N0_N1_K1(BGridDesc_K0_N_K1{}));
    using DsGridDesc_M0_M10_M11_N0_N10_N11 =
        decltype(GridwiseGemm::MakeDsGridDescriptor_M0_M10_M11_N0_N10_N11(DsGridDesc_M_N{}));
    using EGridDesc_M0_M10_M11_N0_N10_N11 =
        decltype(GridwiseGemm::MakeCGridDescriptor_M0_M10_M11_N0_N10_N11(EGridDesc_M_N{}));
    using DefaultBlock2CTileMap =
        decltype(GridwiseGemm::MakeDefaultBlock2CTileMap(EGridDesc_M_N{}));

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const void* p_a_grid,
                 const void* p_b_grid,
                 std::array<const void*, NumDTensor> p_ds_grid,
                 void* p_e_grid,
                 index_t M,
                 index_t N,
                 index_t K,
                 index_t StrideA,
                 index_t StrideB,
                 std::array<index_t, NumDTensor> StrideDs,
                 index_t StrideE,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation cde_element_op)
            : p_a_grid_{static_cast<const ADataType*>(p_a_grid)},
              p_b_grid_{static_cast<const BDataType*>(p_b_grid)},
              p_ds_grid_{},
              p_e_grid_{static_cast<EDataType*>(p_e_grid)},
              a_grid_desc_k0_m0_m1_k1_{},
              b_grid_desc_k0_n0_n1_k1_{},
              e_grid_desc_m0_m10_m11_n0_n10_n11_{},
              block_2_ctile_map_{},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op}
        {
            a_grid_desc_k0_m_k1_ =
                DeviceGemmMultipleD_Dl::MakeAGridDescriptor_K0_M_K1(M, K, StrideA);
            b_grid_desc_k0_n_k1_ =
                DeviceGemmMultipleD_Dl::MakeBGridDescriptor_K0_N_K1(K, N, StrideB);
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DLayout   = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                // D pointer
                p_ds_grid_(i) = static_cast<const DDataType*>(p_ds_grid[i]);

                // D desc
                ds_grid_desc_m_n_(i) =
                    DeviceOp::MakeEGridDescriptor_M_N<DLayout>(M, N, StrideDs[i]);
            });
            e_grid_desc_m_n_ =
                DeviceGemmMultipleD_Dl::MakeEGridDescriptor_M_N<ELayout>(M, N, StrideE);

            if(GridwiseGemm::CheckValidity(
                   a_grid_desc_k0_m_k1_, b_grid_desc_k0_n_k1_, e_grid_desc_m_n_))
            {
                a_grid_desc_k0_m0_m1_k1_ =
                    GridwiseGemm::MakeAGridDescriptor_K0_M0_M1_K1(a_grid_desc_k0_m_k1_);
                b_grid_desc_k0_n0_n1_k1_ =
                    GridwiseGemm::MakeBGridDescriptor_K0_N0_N1_K1(b_grid_desc_k0_n_k1_);

                ds_grid_desc_m0_m10_m11_n0_n10_n11_ =
                    GridwiseGemm::MakeDsGridDescriptor_M0_M10_M11_N0_N10_N11(ds_grid_desc_m_n_);

                e_grid_desc_m0_m10_m11_n0_n10_n11_ =
                    GridwiseGemm::MakeCGridDescriptor_M0_M10_M11_N0_N10_N11(e_grid_desc_m_n_);

                block_2_ctile_map_ = GridwiseGemm::MakeDefaultBlock2CTileMap(e_grid_desc_m_n_);
            }
        }

        //  private:
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        typename GridwiseGemm::DsGridPointer p_ds_grid_;
        EDataType* p_e_grid_;

        AGridDesc_K0_M_K1 a_grid_desc_k0_m_k1_;
        BGridDesc_K0_N_K1 b_grid_desc_k0_n_k1_;
        DsGridDesc_M_N ds_grid_desc_m_n_;
        EGridDesc_M_N e_grid_desc_m_n_;

        AGridDesc_K0_M0_M1_K1 a_grid_desc_k0_m0_m1_k1_;
        BGridDesc_K0_N0_N1_K1 b_grid_desc_k0_n0_n1_k1_;
        DsGridDesc_M0_M10_M11_N0_N10_N11 ds_grid_desc_m0_m10_m11_n0_n10_n11_;
        EGridDesc_M0_M10_M11_N0_N10_N11 e_grid_desc_m0_m10_m11_n0_n10_n11_;

        DefaultBlock2CTileMap block_2_ctile_map_;

        // TODO: unused since gridwise_gemm_dl_v1r3 does NOT support prologue for the time being.
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceGemmMultipleD_Dl::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            {
                std::cout << "arg.a_grid_desc_k0_m0_m1_k1_{"
                          << arg.a_grid_desc_k0_m_k1_.GetLength(I0) << ", "
                          << arg.a_grid_desc_k0_m_k1_.GetLength(I1) << ", "
                          << arg.a_grid_desc_k0_m_k1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.b_grid_desc_k0_n0_n1_k1_{"
                          << arg.b_grid_desc_k0_n_k1_.GetLength(I0) << ", "
                          << arg.b_grid_desc_k0_n_k1_.GetLength(I1) << ", "
                          << arg.b_grid_desc_k0_n_k1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.e_grid_desc_m_n_{ " << arg.e_grid_desc_m_n_.GetLength(I0) << ", "
                          << arg.e_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
            }

            if(!GridwiseGemm::CheckValidity(
                   arg.a_grid_desc_k0_m_k1_, arg.b_grid_desc_k0_n_k1_, arg.e_grid_desc_m_n_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemmDlMultipleD_km_kn_mn has invalid setting");
            }

            const index_t grid_size = GridwiseGemm::CalculateGridSize(
                arg.e_grid_desc_m_n_.GetLength(I0), arg.e_grid_desc_m_n_.GetLength(I1));

            auto launch_kernel = [&](auto has_main_k_block_loop,
                                     auto has_double_tail_k_block_loop) {
                constexpr bool has_main_loop   = has_main_k_block_loop.value;
                constexpr bool has_double_loop = has_double_tail_k_block_loop.value;

                const auto kernel =
                    kernel_gemm_dl_multiple_d<GridwiseGemm,
                                              ADataType,
                                              typename GridwiseGemm::DsGridPointer,
                                              EDataType,
                                              AElementwiseOperation,
                                              BElementwiseOperation,
                                              CDEElementwiseOperation,
                                              DeviceOp::AGridDesc_K0_M0_M1_K1,
                                              DeviceOp::BGridDesc_K0_N0_N1_K1,
                                              DeviceOp::DsGridDesc_M0_M10_M11_N0_N10_N11,
                                              DeviceOp::EGridDesc_M0_M10_M11_N0_N10_N11,
                                              DefaultBlock2CTileMap,
                                              has_main_loop,
                                              has_double_loop>;

                return launch_and_time_kernel(stream_config,
                                              kernel,
                                              dim3(grid_size),
                                              dim3(BlockSize),
                                              0,
                                              arg.p_a_grid_,
                                              arg.p_b_grid_,
                                              arg.p_ds_grid_,
                                              arg.p_e_grid_,
                                              arg.a_element_op_,
                                              arg.b_element_op_,
                                              arg.cde_element_op_,
                                              arg.a_grid_desc_k0_m0_m1_k1_,
                                              arg.b_grid_desc_k0_n0_n1_k1_,
                                              arg.ds_grid_desc_m0_m10_m11_n0_n10_n11_,
                                              arg.e_grid_desc_m0_m10_m11_n0_n10_n11_,
                                              arg.block_2_ctile_map_);
            };

            const auto K0                    = arg.a_grid_desc_k0_m0_m1_k1_.GetLength(I0);
            const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K0);
            const bool has_double_tail_k_block_loop =
                GridwiseGemm::CalculateHasDoubleTailKBlockLoop(K0);

            if(has_main_k_block_loop && has_double_tail_k_block_loop)
            {
                return launch_kernel(integral_constant<bool, true>{},
                                     integral_constant<bool, true>{});
            }
            else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
            {
                return launch_kernel(integral_constant<bool, true>{},
                                     integral_constant<bool, false>{});
            }
            else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
            {
                return launch_kernel(integral_constant<bool, false>{},
                                     integral_constant<bool, true>{});
            }
            else
            {
                return launch_kernel(integral_constant<bool, false>{},
                                     integral_constant<bool, false>{});
            }
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(ck::get_device_name() == "gfx906" || ck::is_xdl_supported() ||
           ck::is_gfx103_supported() || ck::is_gfx11_supported() || ck::is_gfx12_supported())
        {
            return GridwiseGemm::CheckValidity(
                arg.a_grid_desc_k0_m_k1_, arg.b_grid_desc_k0_n_k1_, arg.e_grid_desc_m_n_);
        }
        else
        {
            return false;
        }
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const void* p_a,
                             const void* p_b,
                             std::array<const void*, NumDTensor> p_ds,
                             void* p_e,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideB,
                             std::array<ck::index_t, NumDTensor> StrideDs,
                             index_t StrideE,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CDEElementwiseOperation cde_element_op)
    {
        return Argument{p_a,
                        p_b,
                        p_ds,
                        p_e,
                        M,
                        N,
                        K,
                        StrideA,
                        StrideB,
                        StrideDs,
                        StrideE,
                        a_element_op,
                        b_element_op,
                        cde_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_e,
                        index_t M,
                        index_t N,
                        index_t K,
                        index_t StrideA,
                        index_t StrideB,
                        std::array<ck::index_t, NumDTensor> StrideDs,
                        index_t StrideE,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) override
    {
        return std::make_unique<Argument>(p_a,
                                          p_b,
                                          p_ds,
                                          p_e,
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideDs,
                                          StrideE,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceGemmMultipleD_Dl"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << K0PerBlock << ", "
            << K1 << ", "
            << M1PerThread << ", "
            << N1PerThread << ", "
            << KPerThread
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
