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
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_wmma_cshuffle.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"

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
          typename CShuffleDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          GemmSpecialization GemmSpec,
          ck::index_t NumPrefetch,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t K1,
          ck::index_t MPerWmma,
          ck::index_t NPerWmma,
          ck::index_t MRepeat,
          ck::index_t NRepeat,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_K1,
          bool ABlockLdsAddExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_K1,
          bool BBlockLdsAddExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEShuffleBlockTransferScalarPerVector_NPerBlock,
          ck::LoopScheduler LoopSched     = make_default_loop_scheduler(),
          ck::PipelineVersion PipelineVer = ck::PipelineVersion::v1>
struct DeviceGemmMultipleD_Wmma_CShuffle : public DeviceGemmMultipleD<ALayout,
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
    using DeviceOp                      = DeviceGemmMultipleD_Wmma_CShuffle;
    static constexpr index_t NumDTensor = DsDataType::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    // K1 = Max Vector Access Pixels
    static constexpr auto K1Number = Number<K1>{};

    static constexpr auto MWaves = MPerBlock / (MRepeat * MPerWmma);
    static constexpr auto NWaves = NPerBlock / (NRepeat * NPerWmma);
    static constexpr auto WmmaK  = K1 == 16 ? 32 : 16;

    static constexpr auto AEnableLds_auto =
        (NWaves == 1 && is_same<tensor_layout::gemm::RowMajor, ALayout>::value) ? false : true;
    static constexpr auto BEnableLds_auto =
        (MWaves == 1 && is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value) ? false : true;

    // If true, LDS is used unconditionally
    static constexpr auto AEnableLds_manu = false;
    static constexpr auto BEnableLds_manu = false;

    static constexpr auto AEnableLds = AEnableLds_auto || AEnableLds_manu || (NumPrefetch > 1);
    static constexpr auto BEnableLds = BEnableLds_auto || BEnableLds_manu || (NumPrefetch > 1);

    static constexpr auto matrix_padder =
        MatrixPadder<GemmSpec, index_t, index_t, index_t>{MPerBlock, NPerBlock, KPerBlock};

    // Describe how data read from Global memory
    static auto MakeAGridDescriptor(index_t MRaw, index_t KRaw, index_t StrideA)
    {
        const auto a_grid_desc_m_k = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
            {
                const auto a_grid_desc_mraw_kraw =
                    make_naive_tensor_descriptor(make_tuple(MRaw, KRaw), make_tuple(StrideA, I1));

                return matrix_padder.PadADescriptor_M_K(a_grid_desc_mraw_kraw);
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ALayout>::value)
            {
                const auto a_grid_desc_mraw_kraw =
                    make_naive_tensor_descriptor(make_tuple(MRaw, KRaw), make_tuple(I1, StrideA));

                return matrix_padder.PadADescriptor_M_K(a_grid_desc_mraw_kraw);
            }
        }();

        const auto M = a_grid_desc_m_k.GetLength(I0);
        const auto K = a_grid_desc_m_k.GetLength(I1);
        assert(K % K1 == 0);

        if constexpr(AEnableLds)
        {
            const index_t K0 = K / K1;

            return transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_pass_through_transform(M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
        else
        {
            constexpr auto A_KRow      = 2;
            constexpr auto A_K0PerWmma = WmmaK / A_KRow / K1Number;
            const auto A_KWmma         = K / WmmaK;

            const auto M0 = M / MPerBlock;
            // 0   1     0         1                2        3             4        5          6
            // M - K <-> A_KWmma - MBlock*MRepeat - MWaves - A_K0PerWmma - A_KRow - MPerWmma - A_K1
            return transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(
                               A_KWmma, Number<A_K0PerWmma>{}, Number<A_KRow>{}, K1Number)),
                           make_unmerge_transform(
                               make_tuple(M0 * MRepeat, Number<MWaves>{}, Number<MPerWmma>{}))),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 3, 4, 6>{}, Sequence<1, 2, 5>{}));
        }
    }

    static auto MakeBGridDescriptor(index_t KRaw, index_t NRaw, index_t StrideB)
    {
        const auto b_grid_desc_n_k = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {
                const auto b_grid_desc_nraw_kraw =
                    make_naive_tensor_descriptor(make_tuple(NRaw, KRaw), make_tuple(I1, StrideB));

                return matrix_padder.PadBDescriptor_N_K(b_grid_desc_nraw_kraw);
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, BLayout>)
            {
                const auto b_grid_desc_nraw_kraw =
                    make_naive_tensor_descriptor(make_tuple(NRaw, KRaw), make_tuple(StrideB, I1));

                return matrix_padder.PadBDescriptor_N_K(b_grid_desc_nraw_kraw);
            }
        }();

        const auto N = b_grid_desc_n_k.GetLength(I0);
        const auto K = b_grid_desc_n_k.GetLength(I1);
        assert(K % K1 == 0);

        if constexpr(BEnableLds)
        {
            const index_t K0 = K / K1;

            return transform_tensor_descriptor(
                b_grid_desc_n_k,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_pass_through_transform(N)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
        else
        {
            constexpr auto B_KRow      = 2;
            constexpr auto B_K0PerWmma = WmmaK / B_KRow / K1Number;
            const auto B_KWmma         = K / WmmaK;

            const auto N0 = N / NPerBlock;
            // 0   1     0         1                2        3             4        5          6
            // M - K <-> A_KWmma - MBlock*MRepeat - MWaves - A_K0PerWmma - A_KRow - MPerWmma - A_K1
            return transform_tensor_descriptor(
                b_grid_desc_n_k,
                make_tuple(make_unmerge_transform(make_tuple(
                               B_KWmma, Number<B_K0PerWmma>{}, Number<B_KRow>{}, K1Number)),
                           make_unmerge_transform(
                               make_tuple(N0 * NRepeat, Number<NWaves>{}, Number<NPerWmma>{}))),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 3, 4, 6>{}, Sequence<1, 2, 5>{}));
        }
    }

    template <typename ELayout_>
    static auto MakeEGridDescriptor_M_N(index_t MRaw, index_t NRaw, index_t StrideE)
    {
        const auto e_grid_desc_mraw_nraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ELayout_>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(StrideE, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ELayout_>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(I1, StrideE));
            }
        }();

        return matrix_padder.PadCDescriptor_M_N(e_grid_desc_mraw_nraw);
    }

    static auto MakeDsGridDescriptor_M_N(const std::array<index_t, NumDTensor>& Ms,
                                         const std::array<index_t, NumDTensor>& Ns,
                                         const std::array<index_t, NumDTensor>& DsStride)
    {
        return generate_tuple(
            [&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

                return DeviceOp::MakeEGridDescriptor_M_N<DLayout>(Ms[i], Ns[i], DsStride[i]);
            },
            Number<NumDTensor>{});
    }

    // Gridwise descriptor, mapping to whole given provblem.
    using AGridDesc      = decltype(MakeAGridDescriptor(1, 1, 1));
    using BGridDesc      = decltype(MakeBGridDescriptor(1, 1, 1));
    using DsGridDesc_M_N = remove_cvref_t<decltype(MakeDsGridDescriptor_M_N({}, {}, {}))>;
    using EGridDesc_M_N  = decltype(MakeEGridDescriptor_M_N<ELayout>(1, 1, 1));

    // GridwiseOp
    using GridwiseOp = GridwiseGemmMultipleD_Wmma<
        // DataType Family
        ADataType,
        BDataType,
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,
        // InMemory Data Descriptor
        AGridDesc,
        BGridDesc,
        DsGridDesc_M_N,
        EGridDesc_M_N,
        // ElementwiseOp Family
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        InMemoryDataOperationEnum::Set,
        // Tiling Family
        MPerBlock,
        NPerBlock,
        KPerBlock,
        MPerWmma,
        NPerWmma,
        K1,
        MRepeat,
        NRepeat,
        // ThreadCluster Family
        BlockSize,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        AEnableLds,
        ABlockLdsAddExtraM,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false, // BThreadTransferSrcResetCoordinateAfterRun,
        BEnableLds,
        BBlockLdsAddExtraN,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEShuffleBlockTransferScalarPerVector_NPerBlock,
        NumPrefetch,
        LoopSched,
        PipelineVer>;

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
                 index_t M01,
                 index_t N01,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation cde_element_op)
            : p_a_grid_{static_cast<const ADataType*>(p_a_grid)},
              p_b_grid_{static_cast<const BDataType*>(p_b_grid)},
              p_ds_grid_{},
              p_e_grid_{static_cast<EDataType*>(p_e_grid)},
              a_grid_desc{},
              b_grid_desc{},
              ds_grid_desc_m_n_{},
              e_grid_desc_m_n_{},
              ds_grid_desc_mblock_mperblock_nblock_nperblock{},
              e_grid_desc_mblock_mperblock_nblock_nperblock{},
              block_2_ctile_map_{},
              M01_{M01},
              N01_{N01},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op},
              MRaw_{M},
              NRaw_{N},
              KRaw_{K}
        {
            a_grid_desc = DeviceOp::MakeAGridDescriptor(M, K, StrideA);
            b_grid_desc = DeviceOp::MakeBGridDescriptor(K, N, StrideB);
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DLayout   = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                // D pointer
                p_ds_grid_(i) = static_cast<const DDataType*>(p_ds_grid[i]);

                // D desc
                ds_grid_desc_m_n_(i) =
                    DeviceOp::MakeEGridDescriptor_M_N<DLayout>(M, N, StrideDs[i]);
            });
            e_grid_desc_m_n_ = DeviceOp::MakeEGridDescriptor_M_N<ELayout>(M, N, StrideE);

            block_2_ctile_map_ = GridwiseOp::MakeDefaultBlock2CTileMap(e_grid_desc_m_n_, M01, N01);

            if(GridwiseOp::CheckValidity(a_grid_desc,
                                         b_grid_desc,
                                         ds_grid_desc_m_n_,
                                         e_grid_desc_m_n_,
                                         block_2_ctile_map_))
            {
                ds_grid_desc_mblock_mperblock_nblock_nperblock =
                    GridwiseOp::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        ds_grid_desc_m_n_);

                e_grid_desc_mblock_mperblock_nblock_nperblock =
                    GridwiseOp::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        e_grid_desc_m_n_);
            }
        }

        // Pointers
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        typename GridwiseOp::DsGridPointer p_ds_grid_;
        EDataType* p_e_grid_;

        // Tensor Descriptors
        AGridDesc a_grid_desc;
        BGridDesc b_grid_desc;
        DsGridDesc_M_N ds_grid_desc_m_n_;
        EGridDesc_M_N e_grid_desc_m_n_;
        typename GridwiseOp::DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            ds_grid_desc_mblock_mperblock_nblock_nperblock;
        typename GridwiseOp::EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            e_grid_desc_mblock_mperblock_nblock_nperblock;

        // Block to Tile mapping
        typename GridwiseOp::DefaultBlock2CTileMap block_2_ctile_map_;

        // Idle
        index_t M01_;
        index_t N01_;

        // ElementwiseOp
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;

        // for checking vector load/store
        index_t MRaw_;
        index_t NRaw_;
        index_t KRaw_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(!GridwiseOp::CheckValidity(arg.a_grid_desc,
                                          arg.b_grid_desc,
                                          arg.ds_grid_desc_m_n_,
                                          arg.e_grid_desc_m_n_,
                                          arg.block_2_ctile_map_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemm_k0mk1_k0nk1_m0nm1_wmma_v1r1 has invalid setting");
            }

            const index_t grid_size =
                arg.block_2_ctile_map_.CalculateGridSize(arg.e_grid_desc_m_n_);

            const auto K = [&]() {
                if constexpr(AEnableLds)
                {
                    return arg.a_grid_desc.GetLength(I0) * arg.a_grid_desc.GetLength(I2);
                }
                else
                {
                    return arg.a_grid_desc.GetLength(I0) * arg.a_grid_desc.GetLength(I3) *
                           arg.a_grid_desc.GetLength(I4) * arg.a_grid_desc.GetLength(I6);
                }
            }();

            auto launch_kernel = [&](auto has_main_k_block_loop) {
                const auto kernel = kernel_gemm_mupltipe_d_wmma_cshuffle<
                    GridwiseOp,
                    ADataType,
                    BDataType,
                    typename GridwiseOp::DsGridPointer,
                    EDataType,
                    remove_reference_t<typename DeviceOp::AGridDesc>,
                    remove_reference_t<typename DeviceOp::BGridDesc>,
                    remove_reference_t<
                        typename GridwiseOp::DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock>,
                    remove_reference_t<
                        typename GridwiseOp::EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock>,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CDEElementwiseOperation,
                    remove_reference_t<typename GridwiseOp::DefaultBlock2CTileMap>,
                    has_main_k_block_loop>; // Last Option is W/O

                return launch_and_time_kernel(stream_config,
                                              kernel,
                                              dim3(grid_size),
                                              dim3(BlockSize),
                                              0,
                                              arg.p_a_grid_,
                                              arg.p_b_grid_,
                                              arg.p_ds_grid_,
                                              arg.p_e_grid_,
                                              arg.a_grid_desc,
                                              arg.b_grid_desc,
                                              arg.ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                              arg.e_grid_desc_mblock_mperblock_nblock_nperblock,
                                              arg.a_element_op_,
                                              arg.b_element_op_,
                                              arg.cde_element_op_,
                                              arg.block_2_ctile_map_);
            };

            if(GridwiseOp::CalculateHasMainKBlockLoop(K))
            {
                return launch_kernel(integral_constant<bool, true>{});
            }
            else
            {
                return launch_kernel(integral_constant<bool, false>{});
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
        if(ck::is_gfx11_supported() || ck::is_gfx12_supported())
        {
            if constexpr(!(is_same_v<AccDataType, float> || is_same_v<AccDataType, int32_t>))
            {
                return false;
            }
        }
        else
        {
            return false;
        }
        // check vector load/store
        {
            using Row = ck::tensor_layout::gemm::RowMajor;
            using Col = ck::tensor_layout::gemm::ColumnMajor;

            // check vector load of A
            if constexpr(is_same_v<ALayout, Row> && ABlockTransferSrcVectorDim == 2)
            {
                if(arg.KRaw_ % ABlockTransferSrcScalarPerVector != 0)
                {
                    return false;
                }
            }
            else if constexpr(is_same_v<ALayout, Col> && ABlockTransferSrcVectorDim == 1)
            {
                // FIXME: not rigorous
                if(arg.MRaw_ % ABlockTransferSrcScalarPerVector != 0)
                {
                    return false;
                }
            }
            else
            {
                return false;
            }

            // check vector laod of B
            if constexpr(is_same_v<BLayout, Col> && BBlockTransferSrcVectorDim == 2)
            {
                if(arg.KRaw_ % BBlockTransferSrcScalarPerVector != 0)
                {
                    return false;
                }
            }
            else if constexpr(is_same_v<BLayout, Row> && BBlockTransferSrcVectorDim == 1)
            {
                // FIXME: not rigorous
                if(arg.NRaw_ % BBlockTransferSrcScalarPerVector != 0)
                {
                    return false;
                }
            }
            else
            {
                return false;
            }

            // check vector load of Ds
            // only support RowMajor for now
            bool all_valid = true;

            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

                if constexpr(!is_same_v<DLayout, Row>)
                {
                    all_valid = false;
                }
            });

            if(!all_valid)
            {
                return false;
            }

            // check vector store of E
            // only support RowMajor for now
            if constexpr(is_same_v<ELayout, Row>)
            {
                if(arg.NRaw_ % CDEShuffleBlockTransferScalarPerVector_NPerBlock != 0)
                {
                    return false;
                }
            }
            else
            {
                return false;
            }
        }

        return GridwiseOp::CheckValidity(arg.a_grid_desc,
                                         arg.b_grid_desc,
                                         arg.ds_grid_desc_m_n_,
                                         arg.e_grid_desc_m_n_,
                                         arg.block_2_ctile_map_);
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
                        1,
                        1,
                        a_element_op,
                        b_element_op,
                        cde_element_op};
    }

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
                                          1,
                                          1,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op);
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        std::map<LoopScheduler, std::string> LoopSchedToString{
            {LoopScheduler::Default, "Default"}, {LoopScheduler::Interwave, "Interwave"}};

        std::map<PipelineVersion, std::string> PipelineVersionToString{{PipelineVersion::v1, "v1"},
                                                                       {PipelineVersion::v2, "v2"}};

        // clang-format off
        str << "DeviceGemmMultipleD_Wmma_CShuffle"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << K1 << ", "
            << MPerWmma << ", "
            << NPerWmma << ", "
            << MRepeat << ", "
            << NRepeat
            << ">"
            << " AEnableLds: "
            << AEnableLds << ", "
            << "BEnableLds: "
            << BEnableLds << ", "
            << "NumPrefetch: "
            << NumPrefetch << ", "
            << "LoopScheduler: "
            << LoopSchedToString[LoopSched] << ", "
            << "PipelineVersion: "
            << PipelineVersionToString[PipelineVer];
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
