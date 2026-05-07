// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <queue>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_fwd_to_gemm.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_fwd_multiple_abd.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/io.hpp"
#ifdef CK_EXPERIMENTAL_BUILDER
#include "ck_tile/builder/reflect/conv_describe.hpp"
#include "ck_tile/builder/reflect/instance_traits_device_grouped_conv_fwd_multiple_d_xdl_large_tensor_cshuffle.hpp"
#endif

namespace ck {
namespace tensor_operation {
namespace device {

namespace {

template <typename GridwiseGemm,
          index_t MaxGemmsNum,
          typename GemmArgs,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename ComputePtrOffset,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_grouped_conv_fwd_multiple_d_grouped_gemm_xdl_cshuffle(
        Array<GemmArgs, MaxGemmsNum> gemm_desc_kernel_args,
        const index_t gemms_count,
        const AElementwiseOperation a_element_op,
        const BElementwiseOperation b_element_op,
        const CDEElementwiseOperation c_element_op,
        const ComputePtrOffset compute_ptr_offset_of_groups,
        const ComputePtrOffset compute_ptr_offset_of_n)
{
#if defined(__gfx9__) || defined(__gfx11__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<>())
    {
        __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

        const index_t block_id_x = __builtin_amdgcn_readfirstlane(blockIdx.x);
        const index_t g_idx      = __builtin_amdgcn_readfirstlane(blockIdx.y);
        const index_t n_idx      = __builtin_amdgcn_readfirstlane(blockIdx.z);

        const long_index_t a_group_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_groups.GetAPtrOffset(g_idx));
        const long_index_t b_group_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_groups.GetBPtrOffset(g_idx));
        const auto& ds_group_offset = compute_ptr_offset_of_groups.GetDsPtrOffset(g_idx);
        const long_index_t e_group_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_groups.GetEPtrOffset(g_idx));

        const long_index_t a_n_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_n.GetAPtrOffset(n_idx));
        const auto& ds_n_offset = compute_ptr_offset_of_n.GetDsPtrOffset(n_idx);
        const long_index_t e_n_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_n.GetEPtrOffset(n_idx));

        index_t left     = 0;
        index_t right    = gemms_count;
        index_t group_id = index_t((left + right) / 2);
        while((!(block_id_x >= gemm_desc_kernel_args[group_id].BlockStart_ &&
                 block_id_x < gemm_desc_kernel_args[group_id].BlockEnd_)) &&
              left <= right)
        {
            if(block_id_x < gemm_desc_kernel_args[group_id].BlockStart_)
            {
                right = group_id;
            }
            else
            {
                left = group_id;
            }
            group_id = index_t((left + right) / 2);
        }

        using DsPointer = decltype(gemm_desc_kernel_args[Number<0>{}].ds_ptr_);
        DsPointer p_ds_grid_grp;
        static constexpr index_t NumDTensor = DsPointer::Size();
        static_for<0, NumDTensor, 1>{}([&](auto i) {
            p_ds_grid_grp(i) =
                gemm_desc_kernel_args[group_id].ds_ptr_[i] + ds_group_offset[i] + ds_n_offset[i];
        });

        GridwiseGemm::template Run<HasMainKBlockLoop, InMemoryDataOperationEnum::Set>(
            gemm_desc_kernel_args[group_id].a_ptr_ + a_group_offset + a_n_offset,
            gemm_desc_kernel_args[group_id].b_ptr_ + b_group_offset,
            p_ds_grid_grp,
            gemm_desc_kernel_args[group_id].e_ptr_ + e_group_offset + e_n_offset,
            p_shared,
            a_element_op,
            b_element_op,
            c_element_op,
            gemm_desc_kernel_args[group_id].a_grid_desc_ak0_m_ak1_,
            gemm_desc_kernel_args[group_id].b_grid_desc_bk0_n_bk1_,
            gemm_desc_kernel_args[group_id].ds_grid_desc_mblock_mperblock_nblock_nperblock_,
            gemm_desc_kernel_args[group_id].e_grid_desc_mblock_mperblock_nblock_nperblock_,
            gemm_desc_kernel_args[group_id].block_2_etile_map_);
    }
#else
    ignore = gemm_desc_kernel_args;
    ignore = gemms_count;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
    ignore = compute_ptr_offset_of_groups;
    ignore = compute_ptr_offset_of_n;
#endif
}

} // namespace

template <typename T>
using is_tuple = decltype(std::declval<T&>().IsTuple());

template <index_t NDimSpatial,
          typename ALayout,
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
          ConvolutionForwardSpecialization ConvForwardSpecialization,
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
          typename AComputeDataType =
              decltype(UnpackDataType<is_detected<is_tuple, ADataType>::value,
                                      Number<0>,
                                      ADataType>()), // ComputeType is InputType by default (first
                                                     // in tuple for MultiAB), unpack if tuple was
                                                     // passed
          typename BComputeDataType = AComputeDataType,
          LoopScheduler LoopSched   = make_default_loop_scheduler()>
struct DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor
    : public DeviceGroupedConvFwdMultipleABD<NDimSpatial,
                                             ALayout,
                                             BLayout,
                                             DsLayout,
                                             ELayout,
                                             ADataType,
                                             BDataType,
                                             DsDataType,
                                             EDataType,
                                             AElementwiseOperation,
                                             BElementwiseOperation,
                                             CDEElementwiseOperation,
                                             AComputeDataType,
                                             BComputeDataType>
{
    using DeviceOp = DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor;
    GET_MXDL_PER_WAVE_IMPL
    // Force usage of 16x16 instruction for WMMA
    static constexpr bool Wave32Force16MNPerXDL =
        is_NSpatialGC_GKSpatial_NSpatialGK<ALayout, BLayout, ELayout>() &&
        sizeof(AComputeDataType) == 2 && sizeof(BComputeDataType) == 2 &&
        is_same_v<CDEElementwiseOperation, tensor_operation::element_wise::PassThrough> &&
        (ConvForwardSpecialization == ConvolutionForwardSpecialization::Filter1x1Stride1Pad0 ||
         ConvForwardSpecialization == ConvolutionForwardSpecialization::Default);
    static constexpr index_t Wave32MaxMNPerXDL =
        Wave32Force16MNPerXDL ? 16 : math::max(MPerXDL, NPerXDL);

    static constexpr auto MXdlPerWave64 = GetMXdlPerWave<true>();
    static constexpr auto MXdlPerWave32 =
        GetMXdlPerWave<false,
                       Wave32MaxMNPerXDL,
                       Wave32MaxMNPerXDL,
                       NXdlPerWave*(NPerXDL / Wave32MaxMNPerXDL)>();

    static constexpr index_t NumDTensor  = DsDataType::Size();
    static constexpr index_t MaxGemmsNum = 32;
    static constexpr bool DoElementwiseBeforeCShuffle =
        NumDTensor == 0 && is_same_v<EDataType, bhalf_t> &&
        !is_same_v<CDEElementwiseOperation, tensor_operation::element_wise::PassThrough>;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    using ConvToGemmFwdTransformerIndexT = TransformConvFwdToGemm<NDimSpatial,
                                                                  ConvForwardSpecialization,
                                                                  true /*SplitN*/,
                                                                  ADataType,
                                                                  EDataType,
                                                                  I1,
                                                                  index_t>;

    using ConvToGemmFwdTransformerLongIndexT = TransformConvFwdToGemm<NDimSpatial,
                                                                      ConvForwardSpecialization,
                                                                      true /*SplitN*/,
                                                                      ADataType,
                                                                      EDataType,
                                                                      I1,
                                                                      long_index_t>;

    static constexpr auto matrix_padder =
        MatrixPadder<GemmSpec, index_t, index_t, index_t>{MPerBlock, NPerBlock, KPerBlock};

    template <typename ALay>
    static auto
    MakeAGridDescriptor_M_K(const ConvToGemmFwdTransformerIndexT& conv_to_gemm_transformer)
    {
        const auto in_gemmmraw_gemmkraw_desc =
            conv_to_gemm_transformer.template MakeADescriptor_M_K<ALay>();

        const auto in_gemmm_gemmk_desc =
            matrix_padder.PadADescriptor_M_K(in_gemmmraw_gemmkraw_desc);

        return in_gemmm_gemmk_desc;
    }

    template <typename BLay>
    static auto
    MakeBGridDescriptor_N_K(const ConvToGemmFwdTransformerIndexT& conv_to_gemm_transformer)
    {
        const auto wei_gemmnraw_gemmkraw_desc =
            conv_to_gemm_transformer.template MakeBDescriptor_N_K<BLay>();

        const auto wei_gemmn_gemmk_desc =
            matrix_padder.PadBDescriptor_N_K(wei_gemmnraw_gemmkraw_desc);

        return wei_gemmn_gemmk_desc;
    }

    template <typename ELay>
    static auto
    MakeEGridDescriptor_M_N(const ConvToGemmFwdTransformerIndexT& conv_to_gemm_transformer)
    {
        const auto out_gemmmraw_gemmnraw_desc =
            conv_to_gemm_transformer.template MakeCDescriptor_M_N<ELay>();

        const auto out_gemmm_gemmn_desc =
            matrix_padder.PadCDescriptor_M_N(out_gemmmraw_gemmnraw_desc);

        return out_gemmm_gemmn_desc;
    }

    static auto
    MakeDsGridDescriptor_M_N(const ConvToGemmFwdTransformerIndexT& conv_to_gemm_transformer)
    {
        return generate_tuple(
            [&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

                return DeviceOp::MakeEGridDescriptor_M_N<DLayout>(conv_to_gemm_transformer);
            },
            Number<NumDTensor>{});
    }

    static auto CastDsPointers(const std::array<const void*, NumDTensor>& p_ds)
    {
        return generate_tuple(
            [&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
                return static_cast<const DDataType*>(p_ds[i]);
            },
            Number<NumDTensor>{});
    }

    using DsPointer = decltype(CastDsPointers(std::array<const void*, NumDTensor>{}));
    // desc for problem definition
    constexpr static ConvToGemmFwdTransformerIndexT dummy_conv_to_gemm_transformer;
    using AGridDesc_M_K =
        remove_cvref_t<decltype(MakeAGridDescriptor_M_K<ALayout>(dummy_conv_to_gemm_transformer))>;
    using BGridDesc_N_K =
        remove_cvref_t<decltype(MakeBGridDescriptor_N_K<BLayout>(dummy_conv_to_gemm_transformer))>;
    using DsGridDesc_M_N =
        remove_cvref_t<decltype(MakeDsGridDescriptor_M_N(dummy_conv_to_gemm_transformer))>;
    using EGridDesc_M_N =
        remove_cvref_t<decltype(MakeEGridDescriptor_M_N<ELayout>(dummy_conv_to_gemm_transformer))>;

    static auto
    GenerateConvToGemmTransforms(ConvToGemmFwdTransformerLongIndexT conv_to_gemm_transformer_base,
                                 const ADataType* a_grid_ptr_base,
                                 DsPointer ds_grid_ptr_base,
                                 EDataType* c_grid_ptr_base)
    {
        // Max number of splits
        // We need to use it to avoid infinity loop
        constexpr index_t max_split_numbers = MaxGemmsNum / 2;
        // Arrays to store transformers with smaller descs than 2GB
        Array<ConvToGemmFwdTransformerIndexT, MaxGemmsNum> conv_to_gemm_transformers_arr;
        Array<const ADataType*, MaxGemmsNum> a_grid_ptrs_arr;
        Array<DsPointer, MaxGemmsNum> ds_grid_ptrs_arr;
        Array<EDataType*, MaxGemmsNum> c_grid_ptrs_arr;
        // Queue for splitting
        std::queue<ConvToGemmFwdTransformerLongIndexT> conv_to_gemm_transformers_queue(
            {conv_to_gemm_transformer_base});
        std::queue<const ADataType*> a_grid_ptrs_queue({a_grid_ptr_base});
        std::queue<DsPointer> ds_grid_ptrs_queue({ds_grid_ptr_base});
        std::queue<EDataType*> c_grid_ptrs_queue({c_grid_ptr_base});

        index_t gemms_number  = 0;
        index_t split_numbers = 0;
        // Algorithm:
        // While queue is not empty:
        //  1. Get transformer from queue.
        //  2. If descs are smaller than 2GB push to result array.
        //  3. If descs are bigger than 2GB split into left and right transformer.
        //  and push the both into the queue.
        while(!conv_to_gemm_transformers_queue.empty() && split_numbers < max_split_numbers &&
              gemms_number < MaxGemmsNum)
        {
            // Get transformer from the queue
            const auto& conv_to_gemm_transformer = conv_to_gemm_transformers_queue.front();
            const ADataType* a_grid_ptr          = a_grid_ptrs_queue.front();
            DsPointer ds_grid_ptr                = ds_grid_ptrs_queue.front();
            EDataType* c_grid_ptr                = c_grid_ptrs_queue.front();

            // Check if convolution not exceed 2GB
            if(conv_to_gemm_transformer.AreDescriptorsSmallerThan2GB())
            {
                // If yes, push into result array
                conv_to_gemm_transformers_arr(gemms_number) =
                    ConvToGemmFwdTransformerIndexT{conv_to_gemm_transformer};
                a_grid_ptrs_arr(gemms_number)  = a_grid_ptr;
                ds_grid_ptrs_arr(gemms_number) = ds_grid_ptr;
                c_grid_ptrs_arr(gemms_number)  = c_grid_ptr;
                gemms_number++;
            }
            else
            {
                // If no, split into left and right convolutions
                ConvToGemmFwdTransformerLongIndexT conv_to_gemm_transformers_left_part,
                    conv_to_gemm_transformers_right_part;
                const ADataType* a_grid_right_ptr;
                DsPointer ds_grid_right_ptr;
                EDataType* c_grid_right_ptr;

                ck::tie(conv_to_gemm_transformers_left_part,
                        conv_to_gemm_transformers_right_part,
                        a_grid_right_ptr,
                        ds_grid_right_ptr,
                        c_grid_right_ptr) =
                    conv_to_gemm_transformer.SplitConvProblem(a_grid_ptr, ds_grid_ptr, c_grid_ptr);

                conv_to_gemm_transformers_queue.push(conv_to_gemm_transformers_left_part);
                conv_to_gemm_transformers_queue.push(conv_to_gemm_transformers_right_part);
                // Left offsets remain the same
                a_grid_ptrs_queue.push(a_grid_ptr);
                a_grid_ptrs_queue.push(a_grid_right_ptr);
                ds_grid_ptrs_queue.push(ds_grid_ptr);
                ds_grid_ptrs_queue.push(ds_grid_right_ptr);
                c_grid_ptrs_queue.push(c_grid_ptr);
                c_grid_ptrs_queue.push(c_grid_right_ptr);
                split_numbers++;
            }
            // Remove from the queue
            conv_to_gemm_transformers_queue.pop();
            a_grid_ptrs_queue.pop();
            ds_grid_ptrs_queue.pop();
            c_grid_ptrs_queue.pop();
        }

        const bool is_split_valid = conv_to_gemm_transformers_queue.empty();

        return ck::make_tuple(conv_to_gemm_transformers_arr,
                              a_grid_ptrs_arr,
                              ds_grid_ptrs_arr,
                              c_grid_ptrs_arr,
                              gemms_number,
                              is_split_valid);
    }

#define CK_GRIDWISE_GEMM_FWD_MULTIPLE_D_LARGE_TENSOR_TEMPLATE_PARAMETERS                          \
    ADataType, BDataType, AComputeDataType, AccDataType, CShuffleDataType, DsDataType, EDataType, \
        AElementwiseOperation, BElementwiseOperation, CDEElementwiseOperation,                    \
        NumGemmKPrefetchStage, BlockSize, MPerBlock, NPerBlock, KPerBlock, AK1, BK1, MPerXDL_,    \
        NPerXDL_, MXdlPerWave_, NXdlPerWave*(NPerXDL / NPerXDL_),                                 \
        ABlockTransferThreadClusterLengths_AK0_M_AK1, ABlockTransferThreadClusterArrangeOrder,    \
        ABlockTransferSrcAccessOrder, ABlockTransferSrcVectorDim,                                 \
        ABlockTransferSrcScalarPerVector, ABlockTransferDstScalarPerVector_AK1, false,            \
        ABlockLdsExtraM, BBlockTransferThreadClusterLengths_BK0_N_BK1,                            \
        BBlockTransferThreadClusterArrangeOrder, BBlockTransferSrcAccessOrder,                    \
        BBlockTransferSrcVectorDim, BBlockTransferSrcScalarPerVector,                             \
        BBlockTransferDstScalarPerVector_BK1, false, BBlockLdsExtraN,                             \
        CShuffleMXdlPerWavePerShuffle, CShuffleNXdlPerWavePerShuffle*(NPerXDL / NPerXDL_),        \
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,                         \
        CDEBlockTransferScalarPerVector_NPerBlock, LoopSched, PipelineVersion::v1,                \
        AComputeDataType, DoElementwiseBeforeCShuffle
    // Use appropriate gridwise gemm
    template <index_t MXdlPerWave_, index_t MPerXDL_, index_t NPerXDL_>
    using GridwiseGemmBase = GridwiseGemmMultipleD_xdl_cshuffle<
        CK_GRIDWISE_GEMM_FWD_MULTIPLE_D_LARGE_TENSOR_TEMPLATE_PARAMETERS>;
#undef CK_GRIDWISE_GEMM_FWD_MULTIPLE_D_LARGE_TENSOR_TEMPLATE_PARAMETERS
    using GridwiseGemm64 = GridwiseGemmBase<math::max(MXdlPerWave64, 1), MPerXDL, NPerXDL>;
    using GridwiseGemm32 = GridwiseGemmBase<MXdlPerWave32, Wave32MaxMNPerXDL, Wave32MaxMNPerXDL>;

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
    // Structure for each gemm(conv)
    struct GemmArgs
    {
        // pointers
        const ADataType* a_ptr_;
        const BDataType* b_ptr_;
        DsPointer ds_ptr_;
        EDataType* e_ptr_;

        // tensor descriptors for block/thread-wise copy
        AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1_;
        BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1_;
        DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
            ds_grid_desc_mblock_mperblock_nblock_nperblock_;
        EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock e_grid_desc_mblock_mperblock_nblock_nperblock_;

        // block-to-e-tile map
        Block2ETileMap block_2_etile_map_;
        ck::index_t BlockStart_, BlockEnd_;
    };

    // Argument
    struct Argument : public BaseArgument
    {
        template <typename GridwiseGemm, typename DsGridDesc_M_N_, typename EGridDescriptor_M_N_>
        void init_gemm_args(const ADataType* a_ptr,
                            const BDataType* b_ptr,
                            DsPointer ds_ptr,
                            EDataType* e_ptr,
                            const AGridDesc_M_K& a_grid_desc_m_k,
                            const BGridDesc_N_K& b_grid_desc_n_k,
                            const DsGridDesc_M_N_& ds_grid_desc_m_n,
                            const EGridDescriptor_M_N_& e_grid_desc_m_n,
                            const Block2ETileMap& block_2_etile_map,
                            index_t BlockStart,
                            index_t BlockEnd)
        {
            if(GridwiseGemm::CheckValidity(a_grid_desc_m_k,
                                           b_grid_desc_n_k,
                                           ds_grid_desc_m_n,
                                           e_grid_desc_m_n,
                                           block_2_etile_map))
            {
                gemm_desc_kernel_args_(valid_gemms_count_) =
                    GemmArgs{a_ptr,
                             b_ptr,
                             ds_ptr,
                             e_ptr,
                             GridwiseGemm::MakeDefaultAGridDescriptor_AK0_M_AK1(a_grid_desc_m_k),
                             GridwiseGemm::MakeDefaultBGridDescriptor_BK0_N_BK1(b_grid_desc_n_k),
                             GridwiseGemm::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                                 ds_grid_desc_m_n),
                             GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                                 e_grid_desc_m_n),
                             block_2_etile_map,
                             BlockStart,
                             BlockEnd};

                valid_gemms_count_++;
            }
        }
        Argument(const void* p_a,
                 const void* p_b,
                 const std::array<const void*, NumDTensor>& p_ds,
                 void* p_e,
                 const std::array<long_index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                 const std::array<long_index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                 const std::array<long_index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                 const std::array<long_index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                 const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_k_wos_lengths,
                 const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_k_wos_strides,
                 const std::array<long_index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                 const std::array<long_index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
                 const std::array<long_index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<long_index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<long_index_t, NDimSpatial>& input_left_pads,
                 const std::array<long_index_t, NDimSpatial>& input_right_pads,
                 const AElementwiseOperation& a_element_op,
                 const BElementwiseOperation& b_element_op,
                 const CDEElementwiseOperation& cde_element_op)
            : num_group_{static_cast<index_t>(a_g_n_c_wis_lengths[0])},
              compute_ptr_offset_of_groups_{},
              compute_ptr_offset_of_n_{},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op},
              a_g_n_c_wis_lengths_{a_g_n_c_wis_lengths},
              a_g_n_c_wis_strides_{a_g_n_c_wis_strides},
              b_g_k_c_xs_lengths_{b_g_k_c_xs_lengths},
              b_g_k_c_xs_strides_{b_g_k_c_xs_strides},
              ds_g_n_k_wos_lengths_{ds_g_n_k_wos_lengths},
              ds_g_n_k_wos_strides_{ds_g_n_k_wos_strides},
              e_g_n_k_wos_lengths_{e_g_n_k_wos_lengths},
              e_g_n_k_wos_strides_{e_g_n_k_wos_strides},
              conv_filter_strides_{conv_filter_strides},
              conv_filter_dilations_{conv_filter_dilations},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads}
        {
            // Perform grouped gemm, generate array of tranformer for convolution
            Array<ConvToGemmFwdTransformerIndexT, MaxGemmsNum> conv_to_gemm_transformer_arr;
            Array<const ADataType*, MaxGemmsNum> a_grid_ptrs;
            Array<DsPointer, MaxGemmsNum> ds_grid_ptrs;
            Array<EDataType*, MaxGemmsNum> c_grid_ptrs;

            DsPointer p_ds_casted = CastDsPointers(p_ds);

            ck::tie(conv_to_gemm_transformer_arr,
                    a_grid_ptrs,
                    ds_grid_ptrs,
                    c_grid_ptrs,
                    gemms_count_,
                    is_split_valid_) =
                GenerateConvToGemmTransforms(
                    ConvToGemmFwdTransformerLongIndexT{a_g_n_c_wis_lengths_,
                                                       a_g_n_c_wis_strides_,
                                                       b_g_k_c_xs_lengths_,
                                                       b_g_k_c_xs_strides_,
                                                       e_g_n_k_wos_lengths_,
                                                       e_g_n_k_wos_strides_,
                                                       conv_filter_strides_,
                                                       conv_filter_dilations_,
                                                       input_left_pads_,
                                                       input_right_pads_},
                    static_cast<const ADataType*>(p_a),
                    p_ds_casted,
                    static_cast<EDataType*>(p_e));

            grid_size_         = 0;
            valid_gemms_count_ = 0;

            if(is_split_valid_)
            {
                // Create GemmArg for each gemm(conv)
                for(index_t i = 0; i < gemms_count_; i++)
                {
                    const AGridDesc_M_K a_grid_desc_m_k{DeviceOp::MakeAGridDescriptor_M_K<ALayout>(
                        conv_to_gemm_transformer_arr[i])};
                    const BGridDesc_N_K b_grid_desc_n_k{DeviceOp::MakeBGridDescriptor_N_K<BLayout>(
                        conv_to_gemm_transformer_arr[i])};
                    const auto e_grid_desc_m_n =
                        DeviceOp::MakeEGridDescriptor_M_N<ELayout>(conv_to_gemm_transformer_arr[i]);

                    const auto ds_grid_desc_m_n =
                        generate_tuple([&](auto) { return e_grid_desc_m_n; }, Number<NumDTensor>{});

                    const auto block_2_etile_map =
                        GridwiseGemm64::MakeDefaultBlock2ETileMap(e_grid_desc_m_n);

                    const index_t grid_size_grp =
                        block_2_etile_map.CalculateGridSize(e_grid_desc_m_n);

                    const index_t BlockStart = grid_size_;
                    const index_t BlockEnd   = grid_size_ + grid_size_grp;

                    grid_size_ += grid_size_grp;

                    if(get_warp_size() == 64)
                    {
                        if constexpr(MXdlPerWave64 > 0)
                        {
                            init_gemm_args<GridwiseGemm64>(a_grid_ptrs[i],
                                                           static_cast<const BDataType*>(p_b),
                                                           ds_grid_ptrs[i],
                                                           c_grid_ptrs[i],
                                                           a_grid_desc_m_k,
                                                           b_grid_desc_n_k,
                                                           ds_grid_desc_m_n,
                                                           e_grid_desc_m_n,
                                                           block_2_etile_map,
                                                           BlockStart,
                                                           BlockEnd);
                        }
                    }
                    else
                    {
                        if constexpr(MXdlPerWave32 > 0)
                        {
                            init_gemm_args<GridwiseGemm32>(a_grid_ptrs[i],
                                                           static_cast<const BDataType*>(p_b),
                                                           ds_grid_ptrs[i],
                                                           c_grid_ptrs[i],
                                                           a_grid_desc_m_k,
                                                           b_grid_desc_n_k,
                                                           ds_grid_desc_m_n,
                                                           e_grid_desc_m_n,
                                                           block_2_etile_map,
                                                           BlockStart,
                                                           BlockEnd);
                        }
                    }
                }
                // N is the same for all convs
                conv_N_per_block_ = static_cast<index_t>(conv_to_gemm_transformer_arr[I0].N_);
            }

            // Strides for G and N remain the same
            compute_ptr_offset_of_groups_.BatchStrideA_ = a_g_n_c_wis_strides[0];
            compute_ptr_offset_of_groups_.BatchStrideB_ = b_g_k_c_xs_strides[0];
            compute_ptr_offset_of_groups_.BatchStrideE_ = e_g_n_k_wos_strides[0];

            compute_ptr_offset_of_n_.BatchStrideA_ = a_g_n_c_wis_strides[1] * conv_N_per_block_;
            compute_ptr_offset_of_n_.BatchStrideE_ = e_g_n_k_wos_strides[1] * conv_N_per_block_;

            static_for<0, NumDTensor, 1>{}([&](auto i) {
                compute_ptr_offset_of_groups_.BatchStrideDs_(i) = ds_g_n_k_wos_strides_[i][0];
                compute_ptr_offset_of_n_.BatchStrideDs_(i) =
                    ds_g_n_k_wos_strides_[i][1] * conv_N_per_block_;
            });
        }

        void Print() const
        {
            for(index_t i = 0; i < valid_gemms_count_; i++)
            {
                std::cout << "A[AK0, M, AK1]: " << gemm_desc_kernel_args_[i].a_grid_desc_ak0_m_ak1_
                          << std::endl;
                std::cout << "B[BK0, N, BK1]: " << gemm_desc_kernel_args_[i].b_grid_desc_bk0_n_bk1_
                          << std::endl;
                std::cout
                    << "E[MBlock, MPerBlock, NBlock, NPerBlock]: "
                    << gemm_desc_kernel_args_[i].e_grid_desc_mblock_mperblock_nblock_nperblock_
                    << std::endl;
            }
        }

        index_t num_group_;
        index_t conv_N_per_block_;

        Array<GemmArgs, MaxGemmsNum> gemm_desc_kernel_args_;

        index_t grid_size_;
        index_t gemms_count_;
        index_t valid_gemms_count_;

        bool is_split_valid_;

        // for computing batch offset
        ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor> compute_ptr_offset_of_groups_;
        ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor> compute_ptr_offset_of_n_;

        // element-wise op
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;

        // for checking IsSupportedArgument()
        std::array<long_index_t, NDimSpatial + 3> a_g_n_c_wis_lengths_;
        std::array<long_index_t, NDimSpatial + 3> a_g_n_c_wis_strides_;
        std::array<long_index_t, NDimSpatial + 3> b_g_k_c_xs_lengths_;
        std::array<long_index_t, NDimSpatial + 3> b_g_k_c_xs_strides_;
        std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_lengths_;
        std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_strides_;
        std::array<long_index_t, NDimSpatial + 3> e_g_n_k_wos_lengths_;
        std::array<long_index_t, NDimSpatial + 3> e_g_n_k_wos_strides_;
        std::array<long_index_t, NDimSpatial> conv_filter_strides_;
        std::array<long_index_t, NDimSpatial> conv_filter_dilations_;
        std::array<long_index_t, NDimSpatial> input_left_pads_;
        std::array<long_index_t, NDimSpatial> input_right_pads_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {

        using Argument = DeviceOp::Argument;
        template <typename GridwiseGemm>
        float RunImp(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                arg.Print();
            }

            const index_t num_workgroups_per_Conv_N =
                arg.a_g_n_c_wis_lengths_[I1] / arg.conv_N_per_block_;

            const index_t gdx = arg.grid_size_;
            const index_t gdy = arg.num_group_;
            const index_t gdz = num_workgroups_per_Conv_N;

            // K is constant for all gemms
            const auto K = arg.gemm_desc_kernel_args_[I0].a_grid_desc_ak0_m_ak1_.GetLength(I0) *
                           arg.gemm_desc_kernel_args_[I0].a_grid_desc_ak0_m_ak1_.GetLength(I2);

            auto launch_kernel = [&](auto has_main_k_block_loop) {
                constexpr bool has_main_loop = has_main_k_block_loop.value;
                const auto kernel = kernel_grouped_conv_fwd_multiple_d_grouped_gemm_xdl_cshuffle<
                    GridwiseGemm,
                    MaxGemmsNum,
                    GemmArgs,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CDEElementwiseOperation,
                    ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                    has_main_loop>;

                return launch_and_time_kernel(stream_config,
                                              kernel,
                                              dim3(gdx, gdy, gdz),
                                              dim3(BlockSize),
                                              0,
                                              arg.gemm_desc_kernel_args_,
                                              arg.gemms_count_,
                                              arg.a_element_op_,
                                              arg.b_element_op_,
                                              arg.cde_element_op_,
                                              arg.compute_ptr_offset_of_groups_,
                                              arg.compute_ptr_offset_of_n_);
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

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(get_warp_size() == 64)
            {
                if constexpr(MXdlPerWave64 > 0)
                {
                    return RunImp<GridwiseGemm64>(arg, stream_config);
                }
            }
            else
            {
                if constexpr(MXdlPerWave32 > 0)
                {
                    return RunImp<GridwiseGemm32>(arg, stream_config);
                }
            }
            return 0;
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static bool IsSupportedArgument(const Argument& arg)
    {
        namespace ctc = tensor_layout::convolution;

        const long_index_t K = arg.b_g_k_c_xs_lengths_[I1];
        const long_index_t C = arg.b_g_k_c_xs_lengths_[I2];

        bool ds_valid = true;
        static_for<0, NumDTensor, 1>{}([&](auto i) {
            for(int d = 0; d < NDimSpatial + I3; d++)
            {
                if(arg.ds_g_n_k_wos_strides_[i][d] != arg.e_g_n_k_wos_strides_[d])
                {
                    ds_valid = false;
                }
                if(arg.ds_g_n_k_wos_lengths_[i][d] != arg.e_g_n_k_wos_lengths_[d])
                {
                    ds_valid = false;
                }
            }

            using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
            static_assert(is_same_v<DDataType, EDataType>);
        });

        if(!ds_valid)
        {
            return false;
        }

        // Check if all descs are valid
        if(!(arg.is_split_valid_ && arg.gemms_count_ == arg.valid_gemms_count_))
        {
            return false;
        }
        // check device
        if(get_device_name() == "gfx908")
        {
            // FIXME: re-enable fp64 when SWDEV-335738 is fixed
            if constexpr(!(is_same_v<AccDataType, float> || is_same_v<AccDataType, int32_t>))
            {
                return false;
            }
        }
        if(!ck::is_xdl_wmma_supported<AComputeDataType,
                                      BComputeDataType,
                                      Wave32MaxMNPerXDL,
                                      Wave32MaxMNPerXDL>())
        {
            return false;
        }
        if constexpr(is_same_v<AComputeDataType, ck::tf32_t> ||
                     is_same_v<BComputeDataType, ck::tf32_t>)
        {
            if(!is_tf32_supported())
            {
                return false;
            }
            if constexpr(!is_same_v<AComputeDataType, BComputeDataType>)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "ComputeDataType for A and B should be same while using TF32"
                              << std::endl;
                }
                return false;
            }
        }
        // check ConvolutionForwardSpecialization
        if constexpr(ConvForwardSpecialization ==
                     ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            // check if it's 1x1, stride=1 conv
            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                const index_t X          = arg.b_g_k_c_xs_lengths_[i + 3];
                const index_t ConvStride = arg.conv_filter_strides_[i];
                const index_t LeftPad    = arg.input_left_pads_[i];
                const index_t RightPad   = arg.input_right_pads_[i];

                if(!(X == 1 && ConvStride == 1 && LeftPad == 0 && RightPad == 0))
                {
                    return false;
                }
            }
        }
        else if constexpr(ConvForwardSpecialization ==
                          ConvolutionForwardSpecialization::Filter1x1Pad0)
        {
            // check if it's 1x1 conv
            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                const index_t X        = arg.b_g_k_c_xs_lengths_[i + 3];
                const index_t LeftPad  = arg.input_left_pads_[i];
                const index_t RightPad = arg.input_right_pads_[i];

                if(!(X == 1 && LeftPad == 0 && RightPad == 0))
                {
                    return false;
                }
            }
        }
        else if constexpr(ConvForwardSpecialization == ConvolutionForwardSpecialization::Filter3x3)
        {
            if(C != 1)
            {
                return false;
            }
            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                const index_t filter_spatial_dim = arg.b_g_k_c_xs_lengths_[i + I3];

                if(filter_spatial_dim != I3)
                {
                    return false;
                }
            }
            if constexpr(!is_NSpatialGC_GKSpatial_NSpatialGK<ALayout, BLayout, ELayout>())
            {
                return false;
            }
        }

        // check vector access of A
        // FIXME: layout
        if constexpr(is_same_v<ALayout, ctc::G_NW_C> || is_same_v<ALayout, ctc::G_NHW_C> ||
                     is_same_v<ALayout, ctc::G_NDHW_C> || is_same_v<ALayout, ctc::GNWC> ||
                     is_same_v<ALayout, ctc::GNHWC> || is_same_v<ALayout, ctc::GNDHWC> ||
                     is_same_v<ALayout, ctc::NWGC> || is_same_v<ALayout, ctc::NHWGC> ||
                     is_same_v<ALayout, ctc::NDHWGC>)
        {
            // Check access per C
            if(!(ABlockTransferSrcVectorDim == 2 && C % ABlockTransferSrcScalarPerVector == 0))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        // check vector access of B
        // FIXME: layout
        if constexpr(is_same_v<BLayout, ctc::G_K_X_C> || is_same_v<BLayout, ctc::G_K_YX_C> ||
                     is_same_v<BLayout, ctc::G_K_ZYX_C> || is_same_v<BLayout, ctc::GKXC> ||
                     is_same_v<BLayout, ctc::GKYXC> || is_same_v<BLayout, ctc::GKZYXC> ||
                     is_same_v<BLayout, ctc::KXGC> || is_same_v<BLayout, ctc::KYXGC> ||
                     is_same_v<BLayout, ctc::KZYXGC>)

        {
            if(!(BBlockTransferSrcVectorDim == 2 && C % BBlockTransferSrcScalarPerVector == 0))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        // check vector access of E
        if constexpr(is_same_v<ELayout, ctc::G_NW_K> || is_same_v<ELayout, ctc::G_NHW_K> ||
                     is_same_v<ELayout, ctc::G_NDHW_K> || is_same_v<ELayout, ctc::GNWK> ||
                     is_same_v<ELayout, ctc::GNHWK> || is_same_v<ELayout, ctc::GNDHWK> ||
                     is_same_v<ELayout, ctc::NWGK> || is_same_v<ELayout, ctc::NHWGK> ||
                     is_same_v<ELayout, ctc::NDHWGK>)
        {
            if(!(K % CDEBlockTransferScalarPerVector_NPerBlock == 0))
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

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(
        const void* p_a,
        const void* p_b,
        const std::array<const void*, NumDTensor>& p_ds,
        void* p_e,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_lengths,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& input_right_pads,
        const AElementwiseOperation& a_element_op,
        const BElementwiseOperation& b_element_op,
        const CDEElementwiseOperation& cde_element_op)
    {
        std::array<long_index_t, NDimSpatial + 3> a_g_n_c_wis_lengths_i64;
        std::array<long_index_t, NDimSpatial + 3> a_g_n_c_wis_strides_i64;
        std::array<long_index_t, NDimSpatial + 3> b_g_k_c_xs_lengths_i64;
        std::array<long_index_t, NDimSpatial + 3> b_g_k_c_xs_strides_i64;
        std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_lengths_i64;
        std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_strides_i64;
        std::array<long_index_t, NDimSpatial + 3> e_g_n_k_wos_lengths_i64;
        std::array<long_index_t, NDimSpatial + 3> e_g_n_k_wos_strides_i64;
        std::array<long_index_t, NDimSpatial> conv_filter_strides_i64;
        std::array<long_index_t, NDimSpatial> conv_filter_dilations_i64;
        std::array<long_index_t, NDimSpatial> input_left_pads_i64;
        std::array<long_index_t, NDimSpatial> input_right_pads_i64;

        array_convert(a_g_n_c_wis_lengths_i64, a_g_n_c_wis_lengths);
        array_convert(a_g_n_c_wis_strides_i64, a_g_n_c_wis_strides);
        array_convert(b_g_k_c_xs_lengths_i64, b_g_k_c_xs_lengths);
        array_convert(b_g_k_c_xs_strides_i64, b_g_k_c_xs_strides);
        for(index_t d = 0; d < NumDTensor; d++)
        {
            array_convert(ds_g_n_k_wos_lengths_i64[d], ds_g_n_k_wos_lengths[d]);
            array_convert(ds_g_n_k_wos_strides_i64[d], ds_g_n_k_wos_strides[d]);
        }
        array_convert(e_g_n_k_wos_lengths_i64, e_g_n_k_wos_lengths);
        array_convert(e_g_n_k_wos_strides_i64, e_g_n_k_wos_strides);
        array_convert(conv_filter_strides_i64, conv_filter_strides);
        array_convert(conv_filter_dilations_i64, conv_filter_dilations);
        array_convert(input_left_pads_i64, input_left_pads);
        array_convert(input_right_pads_i64, input_right_pads);

        return Argument{p_a,
                        p_b,
                        p_ds,
                        p_e,
                        a_g_n_c_wis_lengths_i64,
                        a_g_n_c_wis_strides_i64,
                        b_g_k_c_xs_lengths_i64,
                        b_g_k_c_xs_strides_i64,
                        ds_g_n_k_wos_lengths_i64,
                        ds_g_n_k_wos_strides_i64,
                        e_g_n_k_wos_lengths_i64,
                        e_g_n_k_wos_strides_i64,
                        conv_filter_strides_i64,
                        conv_filter_dilations_i64,
                        input_left_pads_i64,
                        input_right_pads_i64,
                        a_element_op,
                        b_element_op,
                        cde_element_op};
    }

    static auto
    MakeArgument(const void* p_a,
                 const void* p_b,
                 const std::array<const void*, NumDTensor>& p_ds,
                 void* p_e,
                 const std::array<long_index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                 const std::array<long_index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                 const std::array<long_index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                 const std::array<long_index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                 const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_k_wos_lengths,
                 const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_k_wos_strides,
                 const std::array<long_index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                 const std::array<long_index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
                 const std::array<long_index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<long_index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<long_index_t, NDimSpatial>& input_left_pads,
                 const std::array<long_index_t, NDimSpatial>& input_right_pads,
                 const AElementwiseOperation& a_element_op,
                 const BElementwiseOperation& b_element_op,
                 const CDEElementwiseOperation& cde_element_op)
    {
        return Argument{p_a,
                        p_b,
                        p_ds,
                        p_e,
                        a_g_n_c_wis_lengths,
                        a_g_n_c_wis_strides,
                        b_g_k_c_xs_lengths,
                        b_g_k_c_xs_strides,
                        ds_g_n_k_wos_lengths,
                        ds_g_n_k_wos_strides,
                        e_g_n_k_wos_lengths,
                        e_g_n_k_wos_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        a_element_op,
                        b_element_op,
                        cde_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const void* p_a,
        const void* p_b,
        const std::array<const void*, NumDTensor>& p_ds,
        void* p_e,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_lengths,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& input_right_pads,
        const AElementwiseOperation& a_element_op,
        const BElementwiseOperation& b_element_op,
        const CDEElementwiseOperation& cde_element_op) override
    {

        std::array<long_index_t, NDimSpatial + 3> a_g_n_c_wis_lengths_i64;
        std::array<long_index_t, NDimSpatial + 3> a_g_n_c_wis_strides_i64;
        std::array<long_index_t, NDimSpatial + 3> b_g_k_c_xs_lengths_i64;
        std::array<long_index_t, NDimSpatial + 3> b_g_k_c_xs_strides_i64;
        std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_lengths_i64;
        std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_strides_i64;
        std::array<long_index_t, NDimSpatial + 3> e_g_n_k_wos_lengths_i64;
        std::array<long_index_t, NDimSpatial + 3> e_g_n_k_wos_strides_i64;
        std::array<long_index_t, NDimSpatial> conv_filter_strides_i64;
        std::array<long_index_t, NDimSpatial> conv_filter_dilations_i64;
        std::array<long_index_t, NDimSpatial> input_left_pads_i64;
        std::array<long_index_t, NDimSpatial> input_right_pads_i64;

        array_convert(a_g_n_c_wis_lengths_i64, a_g_n_c_wis_lengths);
        array_convert(a_g_n_c_wis_strides_i64, a_g_n_c_wis_strides);
        array_convert(b_g_k_c_xs_lengths_i64, b_g_k_c_xs_lengths);
        array_convert(b_g_k_c_xs_strides_i64, b_g_k_c_xs_strides);
        for(index_t d = 0; d < NumDTensor; d++)
        {
            array_convert(ds_g_n_k_wos_lengths_i64[d], ds_g_n_k_wos_lengths[d]);
            array_convert(ds_g_n_k_wos_strides_i64[d], ds_g_n_k_wos_strides[d]);
        }
        array_convert(e_g_n_k_wos_lengths_i64, e_g_n_k_wos_lengths);
        array_convert(e_g_n_k_wos_strides_i64, e_g_n_k_wos_strides);
        array_convert(conv_filter_strides_i64, conv_filter_strides);
        array_convert(conv_filter_dilations_i64, conv_filter_dilations);
        array_convert(input_left_pads_i64, input_left_pads);
        array_convert(input_right_pads_i64, input_right_pads);

        return std::make_unique<Argument>(p_a,
                                          p_b,
                                          p_ds,
                                          p_e,
                                          a_g_n_c_wis_lengths_i64,
                                          a_g_n_c_wis_strides_i64,
                                          b_g_k_c_xs_lengths_i64,
                                          b_g_k_c_xs_strides_i64,
                                          ds_g_n_k_wos_lengths_i64,
                                          ds_g_n_k_wos_strides_i64,
                                          e_g_n_k_wos_lengths_i64,
                                          e_g_n_k_wos_strides_i64,
                                          conv_filter_strides_i64,
                                          conv_filter_dilations_i64,
                                          input_left_pads_i64,
                                          input_right_pads_i64,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op);
    }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        const std::array<const void*, NumDTensor>& p_ds,
                        void* p_e,
                        const std::array<long_index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                        const std::array<long_index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                        const std::array<long_index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                        const std::array<long_index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                        const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>&
                            ds_g_n_k_wos_lengths,
                        const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>&
                            ds_g_n_k_wos_strides,
                        const std::array<long_index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                        const std::array<long_index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
                        const std::array<long_index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<long_index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<long_index_t, NDimSpatial>& input_left_pads,
                        const std::array<long_index_t, NDimSpatial>& input_right_pads,
                        const AElementwiseOperation& a_element_op,
                        const BElementwiseOperation& b_element_op,
                        const CDEElementwiseOperation& cde_element_op) override
    {

        return std::make_unique<Argument>(p_a,
                                          p_b,
                                          p_ds,
                                          p_e,
                                          a_g_n_c_wis_lengths,
                                          a_g_n_c_wis_strides,
                                          b_g_k_c_xs_lengths,
                                          b_g_k_c_xs_strides,
                                          ds_g_n_k_wos_lengths,
                                          ds_g_n_k_wos_strides,
                                          e_g_n_k_wos_lengths,
                                          e_g_n_k_wos_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor";
        if(get_warp_size() != 64) {
            str << "_WmmaPorted";
        }

        str    << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << getConvForwardSpecializationString(ConvForwardSpecialization) << ", "
            << MPerXDL << ", "
            << NPerXDL << ", "
            << MXdlPerWave << ", "
            << NXdlPerWave << ", "
            << ABlockTransferSrcScalarPerVector << ", "
            << BBlockTransferSrcScalarPerVector << ", "
            << CDEBlockTransferScalarPerVector_NPerBlock << ", "
            << CShuffleMXdlPerWavePerShuffle << ", "
            << CShuffleNXdlPerWavePerShuffle
            << ">";
        // clang-format on

        return str.str();
    }

#ifdef CK_EXPERIMENTAL_BUILDER
    std::string GetInstanceString() const override
    {
        static_assert(
            ck_tile::reflect::HasInstanceTraits<DeviceOp>,
            "Specialization of instance_traits not found. Please check that a "
            "specialization exists in file "
            "ck_tile/builder/reflect/"
            "instance_traits_device_grouped_conv_fwd_multiple_d_xdl_large_tensor_cshuffle.hpp "
            "for the given template parameters.");
        return ck_tile::reflect::instance_string<DeviceOp>();
    }

    std::unique_ptr<ck_tile::reflect::Description> describe() const override
    {
        static_assert(
            ck_tile::reflect::conv::HasConvTraits<DeviceOp>,
            "ConvTraits specialization not found for this device operation. "
            "If you modified the template parameters of this class, ensure that "
            "the corresponding ConvTraits specialization in "
            "ck_tile/builder/reflect/conv_traits.hpp is updated to match, or that "
            "InstanceTraits in "
            "ck_tile/builder/reflect/"
            "instance_traits_device_grouped_conv_fwd_multiple_d_xdl_large_tensor_cshuffle.hpp "
            "provides all required members for ConvTraits to work.");
        return std::make_unique<ck_tile::reflect::conv::ConvDescription>(
            ck_tile::reflect::describe<DeviceOp>());
    }
#endif
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
