// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifndef __HIPCC_RTC__
#include <iostream>
#include <sstream>
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#endif

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle.hpp"

namespace ck {

template <typename GridwiseGemm,
          typename ADataType,
          typename BDataType,
          typename DsPointer,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename AGridDesc_AK0_M_AK1,
          typename BGridDesc_BK0_N_BK1,
          typename DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename Block2ETileMap,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_gemm_multiple_d_xdl_cshuffle(const ADataType* __restrict__ p_a_grid,
                                        const BDataType* __restrict__ p_b_grid,
                                        DsPointer p_ds_grid,
                                        EDataType* __restrict__ p_e_grid,
                                        const AElementwiseOperation a_element_op,
                                        const BElementwiseOperation b_element_op,
                                        const CDEElementwiseOperation cde_element_op,
                                        const AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1,
                                        const BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1,
                                        const DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                                            ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                        const EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                                            e_grid_desc_mblock_mperblock_nblock_nperblock,
                                        const Block2ETileMap block_2_etile_map)
{
#if defined(__gfx9__) || defined(__gfx11__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<>())
    {
        __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

        GridwiseGemm::template Run<HasMainKBlockLoop, InMemoryDataOperationEnum::Set>(
            p_a_grid,
            p_b_grid,
            p_ds_grid,
            p_e_grid,
            p_shared,
            a_element_op,
            b_element_op,
            cde_element_op,
            a_grid_desc_ak0_m_ak1,
            b_grid_desc_bk0_n_bk1,
            ds_grid_desc_mblock_mperblock_nblock_nperblock,
            e_grid_desc_mblock_mperblock_nblock_nperblock,
            block_2_etile_map);
    }
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_ds_grid;
    ignore = p_e_grid;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = cde_element_op;
    ignore = a_grid_desc_ak0_m_ak1;
    ignore = b_grid_desc_bk0_n_bk1;
    ignore = ds_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = e_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = block_2_etile_map;
#endif
}

} // namespace ck

namespace ck {
namespace tensor_operation {
namespace device {

// GEMM:
//   input : A[M, K]
//   input : B[N, K]
//   input : D0[M, N], D1[M, N], ...
//   output : E[M, N]
//   C = a_op(A) * b_op(B)
//   E = cde_op(C, D0, D1, ...)
// Assume:
//   D0, D1, ... and E have the same layout
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
          index_t NumGemmKPrefetchStage,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          index_t BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEBlockTransferScalarPerVector_NPerBlock,
          LoopScheduler LoopSched     = make_default_loop_scheduler(),
          PipelineVersion PipelineVer = PipelineVersion::v1,
          typename ComputeDataType    = EDataType>
struct DeviceGemmMultipleD_Xdl_CShuffle : public DeviceGemmMultipleD<ALayout,
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
    using DeviceOp = DeviceGemmMultipleD_Xdl_CShuffle;
    GET_NXDL_PER_WAVE_IMPL
    static constexpr auto NXdlPerWave64 = GetNXdlPerWave<true>();
    static constexpr auto NXdlPerWave32 = GetNXdlPerWave<false>();

    static constexpr index_t NumDTensor = DsDataType::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static constexpr auto matrix_padder =
        MatrixPadder<GemmSpec, index_t, index_t, index_t>{MPerBlock, NPerBlock, KPerBlock};

    static auto MakeAGridDescriptor_M_K(index_t MRaw, index_t KRaw, index_t StrideA)
    {
        const auto a_grid_desc_mraw_kraw = [&]() {
            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, KRaw),
                                                    make_tuple(StrideA, I1));
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, KRaw),
                                                    make_tuple(I1, StrideA));
            }
        }();

        return matrix_padder.PadADescriptor_M_K(a_grid_desc_mraw_kraw);
    }

    static auto MakeBGridDescriptor_N_K(index_t KRaw, index_t NRaw, index_t StrideB)
    {
        const auto b_grid_desc_nraw_kraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(I1, StrideB));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(StrideB, I1));
            }
        }();

        return matrix_padder.PadBDescriptor_N_K(b_grid_desc_nraw_kraw);
    }

    template <typename ELay>
    static auto MakeEGridDescriptor_M_N(index_t MRaw, index_t NRaw, index_t StrideE)
    {
        const auto e_grid_desc_mraw_nraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ELay>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(StrideE, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ELay>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(I1, StrideE));
            }
        }();

        return matrix_padder.PadCDescriptor_M_N(e_grid_desc_mraw_nraw);
    }

    static auto MakeDsGridDescriptor_M_N(const Array<index_t, NumDTensor>& MRaws,
                                         const Array<index_t, NumDTensor>& NRaws,
                                         const Array<index_t, NumDTensor>& DsStride)
    {
        return generate_tuple(
            [&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

                return DeviceOp::MakeEGridDescriptor_M_N<DLayout>(MRaws[i], NRaws[i], DsStride[i]);
            },
            Number<NumDTensor>{});
    }

    // desc for problem definition
    using AGridDesc_M_K  = decltype(MakeAGridDescriptor_M_K(1, 1, 1));
    using BGridDesc_N_K  = decltype(MakeBGridDescriptor_N_K(1, 1, 1));
    using DsGridDesc_M_N = remove_cvref_t<decltype(MakeDsGridDescriptor_M_N({}, {}, {}))>;
    using EGridDesc_M_N  = decltype(MakeEGridDescriptor_M_N<ELayout>(1, 1, 1));

    // GridwiseGemm
    template <index_t NXdlPerWave_>
    using GridwiseGemmBase = GridwiseGemmMultipleD_xdl_cshuffle<
        ADataType,
        BDataType,
        ComputeDataType,
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        NumGemmKPrefetchStage,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave_,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        false,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEBlockTransferScalarPerVector_NPerBlock,
        LoopSched,
        PipelineVer>;
    using GridwiseGemm64 = GridwiseGemmBase<math::max(NXdlPerWave64, 1)>;
    using GridwiseGemm32 = GridwiseGemmBase<NXdlPerWave32>;

    // desc for blockwise copy
    using AGridDesc_AK0_M_AK1 =
        remove_cvref_t<decltype(GridwiseGemm64::MakeDefaultAGridDescriptor_AK0_M_AK1(
            AGridDesc_M_K{}))>;
    using BGridDesc_BK0_N_BK1 =
        remove_cvref_t<decltype(GridwiseGemm64::MakeDefaultBGridDescriptor_BK0_N_BK1(
            BGridDesc_N_K{}))>;
    using DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock = remove_cvref_t<
        decltype(GridwiseGemm64::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            DsGridDesc_M_N{}))>;
    using EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock = remove_cvref_t<
        decltype(GridwiseGemm64::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            EGridDesc_M_N{}))>;

    // block-to-e-tile map
    using Block2ETileMap =
        remove_cvref_t<decltype(GridwiseGemm64::MakeDefaultBlock2ETileMap(EGridDesc_M_N{}))>;

#ifndef __HIPCC_RTC__
    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const void* p_a_grid,
                 const void* p_b_grid,
                 std::array<const void*, NumDTensor> p_ds_grid,
                 void* p_e_grid,
                 index_t MRaw,
                 index_t NRaw,
                 index_t KRaw,
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
              a_grid_desc_m_k_{DeviceOp::MakeAGridDescriptor_M_K(MRaw, KRaw, StrideA)},
              b_grid_desc_n_k_{DeviceOp::MakeBGridDescriptor_N_K(KRaw, NRaw, StrideB)},
              ds_grid_desc_m_n_{},
              e_grid_desc_m_n_{DeviceOp::MakeEGridDescriptor_M_N<ELayout>(MRaw, NRaw, StrideE)},
              a_grid_desc_ak0_m_ak1_{
                  GridwiseGemm64::MakeDefaultAGridDescriptor_AK0_M_AK1(a_grid_desc_m_k_)},
              b_grid_desc_bk0_n_bk1_{
                  GridwiseGemm64::MakeDefaultBGridDescriptor_BK0_N_BK1(b_grid_desc_n_k_)},
              block_2_etile_map_{GridwiseGemm64::MakeDefaultBlock2ETileMap(e_grid_desc_m_n_)},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op},
              MRaw_{MRaw},
              NRaw_{NRaw},
              KRaw_{KRaw}
        {
            // populate pointer, desc for Ds
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DLayout   = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                // D pointer
                p_ds_grid_(i) = static_cast<const DDataType*>(p_ds_grid[i]);

                // D desc
                ds_grid_desc_m_n_(i) =
                    DeviceOp::MakeEGridDescriptor_M_N<DLayout>(MRaw, NRaw, StrideDs[i]);
            });
        }

        void Print() const
        {
            std::cout << "A[M, K]: " << a_grid_desc_m_k_ << std::endl;
            std::cout << "B[N, K]: " << b_grid_desc_n_k_ << std::endl;
            static_for<0, NumDTensor, 1>{}(
                [&](auto i) { std::cout << "Ds[M, N]: " << ds_grid_desc_m_n_[i] << std::endl; });
            std::cout << "E[M, N]: " << e_grid_desc_m_n_ << std::endl;
        }

        //  private:
        // pointers
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        typename GridwiseGemm64::DsGridPointer p_ds_grid_;
        EDataType* p_e_grid_;

        // tensor descriptors for problem definiton
        AGridDesc_M_K a_grid_desc_m_k_;
        BGridDesc_N_K b_grid_desc_n_k_;
        DsGridDesc_M_N ds_grid_desc_m_n_;
        EGridDesc_M_N e_grid_desc_m_n_;

        // tensor descriptors for block/thread-wise copy
        AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1_;
        BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1_;

        // block-to-e-tile map
        Block2ETileMap block_2_etile_map_;

        // element-wise op
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

        template <typename GridwiseGemm>
        float RunImp(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_m_k_,
                                            arg.b_grid_desc_n_k_,
                                            arg.ds_grid_desc_m_n_,
                                            arg.e_grid_desc_m_n_,
                                            arg.block_2_etile_map_))
            {
                throw std::runtime_error("wrong! GridwiseGemm has invalid setting");
            }
            auto ds_grid_desc_mblock_mperblock_nblock_nperblock =
                GridwiseGemm::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                    arg.ds_grid_desc_m_n_);

            auto e_grid_desc_mblock_mperblock_nblock_nperblock =
                GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                    arg.e_grid_desc_m_n_);

            const index_t grid_size =
                arg.block_2_etile_map_.CalculateGridSize(arg.e_grid_desc_m_n_);

            auto launch_kernel = [&](auto has_main_k_block_loop) {
                constexpr bool has_main_loop = has_main_k_block_loop.value;

                const auto kernel = kernel_gemm_multiple_d_xdl_cshuffle<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    BDataType, // TODO: distiguish A/B datatype
                    typename GridwiseGemm::DsGridPointer,
                    EDataType,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CDEElementwiseOperation,
                    DeviceOp::AGridDesc_AK0_M_AK1,
                    DeviceOp::BGridDesc_BK0_N_BK1,
                    DeviceOp::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                    DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                    DeviceOp::Block2ETileMap,
                    has_main_loop>;

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
                                              arg.a_grid_desc_ak0_m_ak1_,
                                              arg.b_grid_desc_bk0_n_bk1_,
                                              ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                              e_grid_desc_mblock_mperblock_nblock_nperblock,
                                              arg.block_2_etile_map_);
            };

            const auto K = arg.a_grid_desc_m_k_.GetLength(I1);

            if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
            {
                return launch_kernel(integral_constant<bool, true>{});
            }
            else
            {
                return launch_kernel(integral_constant<bool, false>{});
            }
        }

        INVOKER_RUN_IMPL

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

#endif

    static constexpr bool IsSupported(index_t MRaw_, index_t NRaw_, index_t KRaw_)
    {
        // check vector load/store
        using Row = ck::tensor_layout::gemm::RowMajor;
        using Col = ck::tensor_layout::gemm::ColumnMajor;
        // check vector load of A
        if constexpr(is_same_v<ALayout, Row> && ABlockTransferSrcVectorDim == 2)
        {
            if(KRaw_ % ABlockTransferSrcScalarPerVector != 0)
            {
                return false;
            }
        }
        else if constexpr(is_same_v<ALayout, Col> && ABlockTransferSrcVectorDim == 1)
        {
            // FIXME: not rigorous
            if(MRaw_ % ABlockTransferSrcScalarPerVector != 0)
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
            if(KRaw_ % BBlockTransferSrcScalarPerVector != 0)
            {
                return false;
            }
        }
        else if constexpr(is_same_v<BLayout, Row> && BBlockTransferSrcVectorDim == 1)
        {
            // FIXME: not rigorous
            if(NRaw_ % BBlockTransferSrcScalarPerVector != 0)
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
            if(NRaw_ % CDEBlockTransferScalarPerVector_NPerBlock != 0)
            {
                return false;
            }
        }
        else
        {
            return false;
        }
        return true;
    }

#ifndef __HIPCC_RTC__
    static bool IsSupportedArgument(const Argument& arg)
    {
        if(!ck::is_xdl_wmma_supported<ADataType, BDataType, MPerXDL, NPerXDL>())
        {
            return false;
        }
        if(!IsSupported(arg.MRaw_, arg.NRaw_, arg.KRaw_))
        {
            return false;
        }

        if(get_warp_size() == 64)
        {
            if constexpr(NXdlPerWave64 > 0)
            {
                return GridwiseGemm64::CheckValidity(arg.a_grid_desc_m_k_,
                                                     arg.b_grid_desc_n_k_,
                                                     arg.ds_grid_desc_m_n_,
                                                     arg.e_grid_desc_m_n_,
                                                     arg.block_2_etile_map_);
            }
        }
        else
        {
            if constexpr(NXdlPerWave32 > 0)
            {
                return GridwiseGemm32::CheckValidity(arg.a_grid_desc_m_k_,
                                                     arg.b_grid_desc_n_k_,
                                                     arg.ds_grid_desc_m_n_,
                                                     arg.e_grid_desc_m_n_,
                                                     arg.block_2_etile_map_);
            }
        }
        return false;
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
                             index_t MRaw,
                             index_t NRaw,
                             index_t KRaw,
                             index_t StrideA,
                             index_t StrideB,
                             std::array<index_t, NumDTensor> StrideDs,
                             index_t StrideE,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CDEElementwiseOperation cde_element_op)
    {
        return Argument{p_a,
                        p_b,
                        p_ds,
                        p_e,
                        MRaw,
                        NRaw,
                        KRaw,
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
                        index_t MRaw,
                        index_t NRaw,
                        index_t KRaw,
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
                                          MRaw,
                                          NRaw,
                                          KRaw,
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

        std::map<LoopScheduler, std::string> LoopSchedToString{{LoopScheduler::Default, "Default"},
                                                               { LoopScheduler::Interwave,
                                                                 "Interwave" }};

        std::map<PipelineVersion, std::string> PipelineVersionToString{{PipelineVersion::v1, "v1"},
                                                                       { PipelineVersion::v2,
                                                                         "v2" }};

        // clang-format off
        str << "DeviceGemmMultipleD_Xdl_CShuffle"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1 << ", "
            << MPerXDL << ", "
            << NPerXDL << ", "
            << MXdlPerWave << ", "
            << NXdlPerWave << ", "
            << ABlockTransferSrcScalarPerVector << ", "
            << BBlockTransferSrcScalarPerVector << ", "
            << CShuffleMXdlPerWavePerShuffle << ", "
            << CShuffleNXdlPerWavePerShuffle << ", "
            << getGemmSpecializationString(GemmSpec)
            << ">"
            << " LoopScheduler: "
            << LoopSchedToString[LoopSched] << ", "
            << "PipelineVersion: "
            << PipelineVersionToString[PipelineVer];
        // clang-format on

        return str.str();
    }
#endif

    template <class ADesc, class BDesc, class DsDesc, class EDesc>
    struct Descriptor
    {
        static constexpr auto ds_tuple()
        {
            return transform_tuples(
                [&](auto d) constexpr { return DeviceOp::matrix_padder.PadCDescriptor_M_N(d); },
                DsDesc{});
        }
        using AGridDesc_M_K =
            remove_cvref_t<decltype(DeviceOp::matrix_padder.PadADescriptor_M_K(ADesc{}))>;
        using BGridDesc_N_K =
            remove_cvref_t<decltype(DeviceOp::matrix_padder.PadBDescriptor_N_K(BDesc{}))>;
        using DsGridDesc_M_N = remove_cvref_t<decltype(ds_tuple())>;
        using EGridDesc_M_N =
            remove_cvref_t<decltype(DeviceOp::matrix_padder.PadCDescriptor_M_N(EDesc{}))>;
        using AGridDesc_AK0_M_AK1 =
            remove_cvref_t<decltype(GridwiseGemm64::MakeDefaultAGridDescriptor_AK0_M_AK1(
                DeviceOp::matrix_padder.PadADescriptor_M_K(ADesc{})))>;
        using BGridDesc_BK0_N_BK1 =
            remove_cvref_t<decltype(GridwiseGemm64::MakeDefaultBGridDescriptor_BK0_N_BK1(
                DeviceOp::matrix_padder.PadBDescriptor_N_K(BDesc{})))>;
        using DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock = remove_cvref_t<
            decltype(GridwiseGemm64::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                ds_tuple()))>;
        using EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock = remove_cvref_t<
            decltype(GridwiseGemm64::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                DeviceOp::matrix_padder.PadCDescriptor_M_N(EDesc{})))>;
        using Block2ETileMap = remove_cvref_t<decltype(GridwiseGemm64::MakeDefaultBlock2ETileMap(
            DeviceOp::matrix_padder.PadCDescriptor_M_N(EDesc{})))>;

        // tensor descriptors for problem definiton
        AGridDesc_M_K a_grid_desc_m_k;
        BGridDesc_N_K b_grid_desc_n_k;
        DsGridDesc_M_N ds_grid_desc_m_n;
        EGridDesc_M_N e_grid_desc_m_n;

        // tensor descriptors for block/thread-wise copy
        AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1;
        BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1;
        DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock ds_grid_desc_mblock_mperblock_nblock_nperblock;
        EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock e_grid_desc_mblock_mperblock_nblock_nperblock;

        // block-to-e-tile map
        Block2ETileMap block_2_etile_map;

        // element-wise op
        AElementwiseOperation a_element_op;
        BElementwiseOperation b_element_op;
        CDEElementwiseOperation cde_element_op;

        // for checking vector load/store
        index_t MRaw;
        index_t NRaw;
        index_t KRaw;

        bool has_main_k_block_loop = true;

        constexpr Descriptor(ADesc a,
                             BDesc b,
                             DsDesc ds,
                             EDesc e,
                             AElementwiseOperation a_element_op_,
                             BElementwiseOperation b_element_op_,
                             CDEElementwiseOperation cde_element_op_)
            : a_grid_desc_m_k{DeviceOp::matrix_padder.PadADescriptor_M_K(a)},
              b_grid_desc_n_k{DeviceOp::matrix_padder.PadBDescriptor_N_K(b)},
              ds_grid_desc_m_n{transform_tuples(
                  [&](auto d) constexpr { return DeviceOp::matrix_padder.PadCDescriptor_M_N(d); },
                  ds)},
              e_grid_desc_m_n{DeviceOp::matrix_padder.PadCDescriptor_M_N(e)},
              a_grid_desc_ak0_m_ak1{
                  GridwiseGemm64::MakeDefaultAGridDescriptor_AK0_M_AK1(a_grid_desc_m_k)},
              b_grid_desc_bk0_n_bk1{
                  GridwiseGemm64::MakeDefaultBGridDescriptor_BK0_N_BK1(b_grid_desc_n_k)},
              ds_grid_desc_mblock_mperblock_nblock_nperblock{
                  GridwiseGemm64::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                      transform_tuples(
                          [&](auto d) constexpr {
                              return DeviceOp::matrix_padder.PadCDescriptor_M_N(d);
                          },
                          ds))},
              e_grid_desc_mblock_mperblock_nblock_nperblock{
                  GridwiseGemm64::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                      e_grid_desc_m_n)},
              block_2_etile_map{GridwiseGemm64::MakeDefaultBlock2ETileMap(e_grid_desc_m_n)},
              has_main_k_block_loop{GridwiseGemm64::CalculateHasMainKBlockLoop(
                  a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2))},
              a_element_op{a_element_op_},
              b_element_op{b_element_op_},
              cde_element_op{cde_element_op_},
              MRaw{e.GetLength(I0)},
              NRaw{e.GetLength(I1)},
              KRaw{a.GetLength(I1)}
        {
        }

        constexpr bool IsValid() const
        {
            if(get_warp_size() == 64)
            {
                if constexpr(NXdlPerWave64 > 0)
                {
                    return GridwiseGemm64::CheckValidity(a_grid_desc_m_k,
                                                         b_grid_desc_n_k,
                                                         ds_grid_desc_m_n,
                                                         e_grid_desc_m_n,
                                                         block_2_etile_map) and
                           IsSupported(MRaw, NRaw, KRaw) and
                           GridwiseGemm64::template IsValidCompilationParameter<>();
                }
            }
            else
            {
                if constexpr(NXdlPerWave32 > 0)
                {
                    return GridwiseGemm32::CheckValidity(a_grid_desc_m_k,
                                                         b_grid_desc_n_k,
                                                         ds_grid_desc_m_n,
                                                         e_grid_desc_m_n,
                                                         block_2_etile_map) and
                           IsSupported(MRaw, NRaw, KRaw) and
                           GridwiseGemm32::template IsValidCompilationParameter<>();
                }
            }
            return false;
        }

        constexpr index_t GetBlockSize() const { return BlockSize; }

        constexpr index_t GetGridSize() const
        {
            return block_2_etile_map.CalculateGridSize(e_grid_desc_m_n);
        }
    };

    template <class ADesc, class BDesc, class DsDesc, class EDesc>
    static constexpr auto
    make_descriptor(ADesc a,
                    BDesc b,
                    DsDesc ds,
                    EDesc e,
                    AElementwiseOperation a_element_op     = AElementwiseOperation{},
                    BElementwiseOperation b_element_op     = BElementwiseOperation{},
                    CDEElementwiseOperation cde_element_op = CDEElementwiseOperation{})
    {
        return Descriptor<ADesc, BDesc, DsDesc, EDesc>(
            a, b, ds, e, a_element_op, b_element_op, cde_element_op);
    }

    template <class Desc, class DsPointer>
    __device__ static void Run(const Desc& desc,
                               const ADataType* __restrict__ p_a_grid,
                               const BDataType* __restrict__ p_b_grid,
                               DsPointer p_ds_grid,
                               EDataType* __restrict__ p_e_grid)
    {

#ifndef __HIPCC_RTC__
        assert(desc.IsValid());
#endif
        using GridwiseGemm = conditional_t<get_warp_size() == 64, GridwiseGemm64, GridwiseGemm32>;
        __shared__ char p_shared_block[GridwiseGemm::GetSharedMemoryNumberOfByte()];
        if(desc.has_main_k_block_loop)
        {
            GridwiseGemm::template Run<true, InMemoryDataOperationEnum::Set>(
                p_a_grid,
                p_b_grid,
                p_ds_grid,
                p_e_grid,
                p_shared_block,
                desc.a_element_op,
                desc.b_element_op,
                desc.cde_element_op,
                desc.a_grid_desc_ak0_m_ak1,
                desc.b_grid_desc_bk0_n_bk1,
                desc.ds_grid_desc_mblock_mperblock_nblock_nperblock,
                desc.e_grid_desc_mblock_mperblock_nblock_nperblock,
                desc.block_2_etile_map);
        }
        else
        {
            GridwiseGemm::template Run<false, InMemoryDataOperationEnum::Set>(
                p_a_grid,
                p_b_grid,
                p_ds_grid,
                p_e_grid,
                p_shared_block,
                desc.a_element_op,
                desc.b_element_op,
                desc.cde_element_op,
                desc.a_grid_desc_ak0_m_ak1,
                desc.b_grid_desc_bk0_n_bk1,
                desc.ds_grid_desc_mblock_mperblock_nblock_nperblock,
                desc.e_grid_desc_mblock_mperblock_nblock_nperblock,
                desc.block_2_etile_map);
        }
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
