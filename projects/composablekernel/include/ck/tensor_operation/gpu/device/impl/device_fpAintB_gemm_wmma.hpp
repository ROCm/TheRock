// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_dequantB.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_fpAintB_gemm_wmma.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// 1. DequantB(K, N) = int2fp(B(K, N)) * scale(1, N)
// 2. C(M, N) = A(M, K) * DequantB(K, N)

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename ScaleDataType,
          typename CDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
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
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          ck::LoopScheduler LoopSched     = make_default_loop_scheduler(),
          ck::PipelineVersion PipelineVer = ck::PipelineVersion::weight_only>
struct DeviceFpAintBGemm_Wmma_CShuffle : public DeviceGemm_dequantB<ALayout,
                                                                    BLayout,
                                                                    CLayout,
                                                                    ADataType,
                                                                    BDataType,
                                                                    CDataType,
                                                                    AElementwiseOperation,
                                                                    BElementwiseOperation,
                                                                    CElementwiseOperation>
{
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
    // LDS bypass feature not implemented for dequantization pipeline.
    static constexpr auto AEnableLds_manu = true;
    static constexpr auto BEnableLds_manu = true;

    static constexpr auto AEnableLds = AEnableLds_auto || AEnableLds_manu || (NumPrefetch > 1);
    static constexpr auto BEnableLds = BEnableLds_auto || BEnableLds_manu || (NumPrefetch > 1);

    static constexpr auto matrix_padder =
        MatrixPadder<GemmSpec, index_t, index_t, index_t>{MPerBlock, NPerBlock, KPerBlock};

    using DeviceOp = DeviceFpAintBGemm_Wmma_CShuffle;

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

    static auto MakeScaleGridDescriptor(index_t KRaw, index_t NRaw, index_t StrideB = 0)
    {
        // assume Scale is [1, N]
        const auto scale_grid_desc_n_k = [&]() {
            const auto scale_grid_desc_nraw_kraw =
                make_naive_tensor_descriptor(make_tuple(NRaw, KRaw), make_tuple(I1, StrideB));

            return matrix_padder.PadBDescriptor_N_K(scale_grid_desc_nraw_kraw);
        }();

        const auto N = scale_grid_desc_n_k.GetLength(I0);
        const auto K = scale_grid_desc_n_k.GetLength(I1);
        // When K = 1, it might be scale tensor.
        assert(K % K1 == 0 && K != 1);

        if constexpr(BEnableLds)
        {
            const index_t K0 = K / K1;

            return transform_tensor_descriptor(
                scale_grid_desc_n_k,
                make_tuple(make_unmerge_transform(make_tuple(K0, 1)), // Reduce K1 = 1
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
                scale_grid_desc_n_k,
                make_tuple(make_unmerge_transform(make_tuple(
                               B_KWmma, Number<B_K0PerWmma>{}, Number<B_KRow>{}, K1Number)),
                           make_unmerge_transform(
                               make_tuple(N0 * NRepeat, Number<NWaves>{}, Number<NPerWmma>{}))),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 3, 4, 6>{}, Sequence<1, 2, 5>{}));
        }
    }

    static auto MakeCGridDescriptor_M_N(index_t MRaw, index_t NRaw, index_t StrideC)
    {
        const auto c_grid_desc_mraw_nraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(StrideC, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(I1, StrideC));
            }
        }();

        return matrix_padder.PadCDescriptor_M_N(c_grid_desc_mraw_nraw);
    }

    // Gridwise descriptor, mapping to whole given provblem.
    using AGridDesc     = decltype(MakeAGridDescriptor(1, 1, 1));
    using BGridDesc     = decltype(MakeBGridDescriptor(1, 1, 1));
    using ScaleGridDesc = decltype(MakeScaleGridDescriptor(1, 1, 0));
    using CGridDesc_M_N = decltype(MakeCGridDescriptor_M_N(1, 1, 1));

    // GridwiseGemm
    using GridwiseGemm = GridwiseFpAintBGemm_Wmma<
        BlockSize,
        ADataType,
        BDataType,
        ScaleDataType,
        AccDataType,
        CShuffleDataType,
        CDataType,
        InMemoryDataOperationEnum::Set,
        AGridDesc,
        BGridDesc,
        ScaleGridDesc,
        CGridDesc_M_N,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        MPerWmma,
        NPerWmma,
        K1,
        MRepeat,
        NRepeat,
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
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CShuffleBlockTransferScalarPerVector_NPerBlock,
        NumPrefetch,
        LoopSched,
        PipelineVer>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const ADataType* p_a_grid,
                 const BDataType* p_b_grid,
                 const ScaleDataType* p_scale_grid,
                 CDataType* p_c_grid,
                 index_t M,
                 index_t N,
                 index_t K,
                 index_t StrideA,
                 index_t StrideB,
                 index_t StrideC,
                 index_t M01,
                 index_t N01,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
            : p_a_grid_{p_a_grid},
              p_b_grid_{p_b_grid},
              p_scale_grid_{p_scale_grid},
              p_c_grid_{p_c_grid},
              a_grid_desc_{},
              b_grid_desc_{},
              scale_grid_desc_{},
              c_grid_desc_m_n_{},
              c_grid_desc_mblock_mperblock_nblock_nperblock{},
              block_2_ctile_map_{},
              M01_{M01},
              N01_{N01},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op},
              MRaw_{M},
              NRaw_{N},
              KRaw_{K}
        {
            a_grid_desc_     = DeviceOp::MakeAGridDescriptor(M, K, StrideA);
            b_grid_desc_     = DeviceOp::MakeBGridDescriptor(K, N, StrideB);
            scale_grid_desc_ = DeviceOp::MakeScaleGridDescriptor(K, N, 0);
            c_grid_desc_m_n_ = DeviceOp::MakeCGridDescriptor_M_N(M, N, StrideC);

            block_2_ctile_map_ =
                GridwiseGemm::MakeDefaultBlock2CTileMap(c_grid_desc_m_n_, M01, N01);

            if(GridwiseGemm::CheckValidity(
                   a_grid_desc_, b_grid_desc_, c_grid_desc_m_n_, block_2_ctile_map_))
            {
                c_grid_desc_mblock_mperblock_nblock_nperblock =
                    GridwiseGemm::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        c_grid_desc_m_n_);
            }
        }

        //  private:
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        const ScaleDataType* p_scale_grid_;
        CDataType* p_c_grid_;
        AGridDesc a_grid_desc_;
        BGridDesc b_grid_desc_;
        ScaleGridDesc scale_grid_desc_;
        CGridDesc_M_N c_grid_desc_m_n_;
        typename GridwiseGemm::CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            c_grid_desc_mblock_mperblock_nblock_nperblock;
        typename GridwiseGemm::DefaultBlock2CTileMap block_2_ctile_map_;
        index_t M01_;
        index_t N01_;
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
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
            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_,
                                            arg.b_grid_desc_,
                                            arg.c_grid_desc_m_n_,
                                            arg.block_2_ctile_map_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemm_k0mk1_k0nk1_m0nm1_wmma_v1r1 has invalid setting");
            }

            const index_t grid_size =
                arg.block_2_ctile_map_.CalculateGridSize(arg.c_grid_desc_m_n_);

            const auto K = [&]() {
                if constexpr(AEnableLds)
                {
                    return arg.a_grid_desc_.GetLength(I0) * arg.a_grid_desc_.GetLength(I2);
                }
                else
                {
                    return arg.a_grid_desc_.GetLength(I0) * arg.a_grid_desc_.GetLength(I3) *
                           arg.a_grid_desc_.GetLength(I4) * arg.a_grid_desc_.GetLength(I6);
                }
            }();
            auto launch_kernel = [&](auto has_main_k_block_loop) {
                const auto kernel = kernel_fpAintB_gemm_wmma<
                    GridwiseGemm,
                    ADataType,
                    BDataType,
                    ScaleDataType,
                    CDataType,
                    remove_reference_t<DeviceOp::AGridDesc>,
                    remove_reference_t<DeviceOp::BGridDesc>,
                    remove_reference_t<DeviceOp::ScaleGridDesc>,
                    remove_reference_t<
                        typename GridwiseGemm::CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock>,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CElementwiseOperation,
                    remove_reference_t<typename GridwiseGemm::DefaultBlock2CTileMap>,
                    has_main_k_block_loop>;

                return launch_and_time_kernel(stream_config,
                                              kernel,
                                              dim3(grid_size),
                                              dim3(BlockSize),
                                              0,
                                              arg.p_a_grid_,
                                              arg.p_b_grid_,
                                              arg.p_scale_grid_,
                                              arg.p_c_grid_,
                                              arg.a_grid_desc_,
                                              arg.b_grid_desc_,
                                              arg.scale_grid_desc_,
                                              arg.c_grid_desc_mblock_mperblock_nblock_nperblock,
                                              arg.a_element_op_,
                                              arg.b_element_op_,
                                              arg.c_element_op_,
                                              arg.block_2_ctile_map_);
            };

            if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
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
            if constexpr(!(is_same_v<AccDataType, float> || is_same_v<AccDataType, ck::half_t> ||
                           is_same_v<AccDataType, int32_t>))
            {
                printf("DeviceOp err: AccDataType");
                return false;
            }
        }
        else
        {
            printf("DeviceOp err: Arch");
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

            // check vector store of C
            // only support RowMajor for now
            if constexpr(is_same_v<CLayout, Row>)
            {
                if(arg.NRaw_ % CShuffleBlockTransferScalarPerVector_NPerBlock != 0)
                {
                    return false;
                }
            }
            else
            {
                return false;
            }
        }

        return GridwiseGemm::CheckValidity(
            arg.a_grid_desc_, arg.b_grid_desc_, arg.c_grid_desc_m_n_, arg.block_2_ctile_map_);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const ADataType* p_a,
                             const BDataType* p_b,
                             const ScaleDataType* p_scale,
                             CDataType* p_c,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideB,
                             index_t StrideC,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        return Argument{p_a,
                        p_b,
                        p_scale,
                        p_c,
                        M,
                        N,
                        K,
                        StrideA,
                        StrideB,
                        StrideC,
                        1,
                        1,
                        a_element_op,
                        b_element_op,
                        c_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      const void* p_scale,
                                                      void* p_c,
                                                      index_t M,
                                                      index_t N,
                                                      index_t K,
                                                      index_t StrideA,
                                                      index_t StrideB,
                                                      index_t StrideC,
                                                      AElementwiseOperation a_element_op,
                                                      BElementwiseOperation b_element_op,
                                                      CElementwiseOperation c_element_op) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          static_cast<const ScaleDataType*>(p_scale),
                                          static_cast<CDataType*>(p_c),
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideC,
                                          1,
                                          1,
                                          a_element_op,
                                          b_element_op,
                                          c_element_op);
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

        std::map<LoopScheduler, std::string> LoopSchedToString{
            {LoopScheduler::Default, "Default"}, {LoopScheduler::Interwave, "Interwave"}};

        std::map<PipelineVersion, std::string> PipelineVersionToString{
            {PipelineVersion::v1, "v1"},
            {PipelineVersion::v2, "v2"},
            {PipelineVersion::weight_only, "weight_only"}};

        // clang-format off
        str << "DeviceFpAintBGemm_Wmma_CShuffle"
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
