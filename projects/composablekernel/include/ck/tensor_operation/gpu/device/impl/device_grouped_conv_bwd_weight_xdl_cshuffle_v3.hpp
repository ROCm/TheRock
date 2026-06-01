// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <numeric>
#include <sstream>

#include "ck/utility/common_header.hpp"

#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_weight.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_bwd_weight_to_gemm.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_bwd_weight_to_gemm_v2.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_weight_specialization.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_conv_v3.hpp"
#include <ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp>
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/tensor_operation/gpu/device/impl/split_k_utils.hpp"
#include "ck/tensor_operation/gpu/device/impl/split_k_arg.hpp"
#include "ck/tensor_operation/gpu/device/impl/split_k_offset_utils.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"

#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/flush_cache.hpp"

#ifdef CK_EXPERIMENTAL_BUILDER
#include "ck_tile/builder/reflect/description.hpp"
#include "ck_tile/builder/reflect/instance_traits_device_grouped_conv_bwd_weight_xdl_cshuffle_v3.hpp"
#endif

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemm,
          typename AGridDesc_AK0_M_K1,
          typename BGridDesc_BK0_N_K1,
          typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
          typename ComputePtrOffsetOfBatch,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3(
        typename GridwiseGemm::Argument karg,
        const AGridDesc_AK0_M_K1 a_grid_desc_ak0_m_ak1,
        const BGridDesc_BK0_N_K1 b_grid_desc_bk0_n_bk1,
        const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
            c_grid_desc_mblock_mperblock_nblock_nperblock,
        const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch,
        const index_t num_k_per_block,
        const long_index_t split_k_stride_a,
        const long_index_t split_k_stride_b,
        bool split_k_offset_hack)
{
#if defined(__gfx9__) || defined(__gfx11__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<CGlobalMemoryDataOperation>())
    {
        const index_t g_idx = __builtin_amdgcn_readfirstlane(blockIdx.z);
        const index_t k_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);

        const long_index_t split_k_offset_a = split_k_offset_hack ? k_idx * split_k_stride_a : 0;
        const long_index_t split_k_offset_b = split_k_offset_hack ? k_idx * split_k_stride_b : 0;

        const long_index_t a_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx));
        const long_index_t b_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx));
        const long_index_t e_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx));

        __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

        DispatchSplitKHack<GridwiseGemm,
                           AGridDesc_AK0_M_K1,
                           BGridDesc_BK0_N_K1,
                           CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                           HasMainKBlockLoop,
                           CGlobalMemoryDataOperation,
                           TailNum>(karg.p_a_grid + a_batch_offset + split_k_offset_a,
                                    karg.p_b_grid + b_batch_offset + split_k_offset_b,
                                    karg.p_c_grid + e_batch_offset,
                                    p_shared,
                                    karg,
                                    a_grid_desc_ak0_m_ak1,
                                    b_grid_desc_bk0_n_bk1,
                                    c_grid_desc_mblock_mperblock_nblock_nperblock,
                                    k_idx * num_k_per_block,
                                    gridDim.y,
                                    split_k_offset_hack);
    }
#else
    ignore = karg;
    ignore = a_grid_desc_ak0_m_ak1;
    ignore = b_grid_desc_bk0_n_bk1;
    ignore = c_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = compute_ptr_offset_of_batch;
    ignore = num_k_per_block;
    ignore = split_k_stride_a;
    ignore = split_k_stride_b;
    ignore = split_k_offset_hack;

#endif // end of if (defined(__gfx9__)
}

template <typename GridwiseGemm,
          typename AGridDesc_AK0_M_K1,
          typename BGridDesc_BK0_N_K1,
          typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
          typename ComputePtrOffsetOfBatch,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3_2lds(
        typename GridwiseGemm::Argument karg,
        const AGridDesc_AK0_M_K1 a_grid_desc_ak0_m_ak1,
        const BGridDesc_BK0_N_K1 b_grid_desc_bk0_n_bk1,
        const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
            c_grid_desc_mblock_mperblock_nblock_nperblock,
        const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch,
        const index_t num_k_per_block,
        const long_index_t split_k_stride_a,
        const long_index_t split_k_stride_b,
        bool split_k_offset_hack)
{
#if defined(__gfx9__) || defined(__gfx11__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<CGlobalMemoryDataOperation>())
    {
        // offset base pointer for each work-group
        const index_t g_idx = __builtin_amdgcn_readfirstlane(blockIdx.z);
        const index_t k_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);

        const long_index_t split_k_offset_a = split_k_offset_hack ? k_idx * split_k_stride_a : 0;
        const long_index_t split_k_offset_b = split_k_offset_hack ? k_idx * split_k_stride_b : 0;

        const long_index_t a_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx));
        const long_index_t b_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx));
        const long_index_t e_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx));

        // Pass two lds pointer is the key to tell compiler that ds_read/write
        // operate on different lds chunk at same time without order dependecy
        __shared__ char p_shared_0[GridwiseGemm::GetSharedMemoryNumberOfByte()];
        __shared__ char p_shared_1[GridwiseGemm::GetSharedMemoryNumberOfByte()];

        DispatchSplitKHack_2Lds<GridwiseGemm,
                                AGridDesc_AK0_M_K1,
                                BGridDesc_BK0_N_K1,
                                CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                                HasMainKBlockLoop,
                                CGlobalMemoryDataOperation,
                                TailNum>(karg.p_a_grid + a_batch_offset + split_k_offset_a,
                                         karg.p_b_grid + b_batch_offset + split_k_offset_b,
                                         karg.p_c_grid + e_batch_offset,
                                         p_shared_0,
                                         p_shared_1,
                                         karg,
                                         a_grid_desc_ak0_m_ak1,
                                         b_grid_desc_bk0_n_bk1,
                                         c_grid_desc_mblock_mperblock_nblock_nperblock,
                                         k_idx * num_k_per_block,
                                         gridDim.y,
                                         split_k_offset_hack);
    }
#else
    ignore = karg;
    ignore = a_grid_desc_ak0_m_ak1;
    ignore = b_grid_desc_bk0_n_bk1;
    ignore = c_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = compute_ptr_offset_of_batch;
    ignore = num_k_per_block;
    ignore = split_k_stride_a;
    ignore = split_k_stride_b;
    ignore = split_k_offset_hack;
#endif // end of if (defined(__gfx9__)
}

// out[N, Ho, Wo, K] = in[N, Hi, Wi, C] * wei[K, Y, X, C]
template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename AccDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          ConvolutionBackwardWeightSpecialization ConvBackwardWeightSpecialization,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t K0PerBlock,
          ck::index_t K1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
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
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CBlockTransferScalarPerVector_NWaveNPerXdl,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename ComputeTypeA                       = InDataType,
          typename ComputeTypeB                       = ComputeTypeA>
struct DeviceGroupedConvBwdWeight_Xdl_CShuffleV3
    : public DeviceGroupedConvBwdWeight<NDimSpatial,
                                        InLayout,
                                        WeiLayout,
                                        OutLayout,
                                        InDataType,
                                        WeiDataType,
                                        OutDataType,
                                        InElementwiseOperation,
                                        WeiElementwiseOperation,
                                        OutElementwiseOperation,
                                        ComputeTypeA,
                                        ComputeTypeB>
{
    static_assert(is_same_v<InElementwiseOperation, element_wise::PassThrough>);
    static_assert(is_same_v<WeiElementwiseOperation, element_wise::PassThrough>);
    static_assert(is_same_v<OutElementwiseOperation, element_wise::PassThrough>);

    using DeviceOp = DeviceGroupedConvBwdWeight_Xdl_CShuffleV3;
    GET_NXDL_PER_WAVE_IMPL
    static constexpr auto NXdlPerWave64 = GetNXdlPerWave<true>();
    static constexpr auto NXdlPerWave32 = GetNXdlPerWave<false>();

    using ADataType = OutDataType;
    using BDataType = InDataType;
    using CDataType = WeiDataType;

    using AElementwiseOperation = OutElementwiseOperation;
    using BElementwiseOperation = InElementwiseOperation;
    using CElementwiseOperation = WeiElementwiseOperation;

    // TODO make A/B datatype different
    using ABDataType = InDataType;

    static inline auto I0 = Number<0>{};
    static inline auto I1 = Number<1>{};
    static inline auto I2 = Number<2>{};
    static inline auto I3 = Number<3>{};
    static inline auto I4 = Number<4>{};
    static inline auto I5 = Number<5>{};

    static constexpr GemmSpecialization GemmSpec = GemmSpecialization::Default;
    static constexpr auto K1Number               = Number<K1>{};

    static constexpr auto conv_to_gemm_transformer =
        TransformConvBwdWeightToGemmV2<NDimSpatial,
                                       MPerBlock,
                                       NPerBlock,
                                       K1Number,
                                       K0PerBlock / K1Number,
                                       1 /*NumGroupsToMerge*/,
                                       ConvBackwardWeightSpecialization>{};

    template <ck::index_t NDim, typename ck::enable_if<NDim == 1, bool>::type = false>
    static auto GetABCGridDesc()
    {
        const ck::index_t dim   = 1;
        const ck::index_t batch = 1;
        const std::array<ck::index_t, NDimSpatial> lengths{1};
        const std::array<ck::index_t, NDimSpatial + 3> strides{1, 1, 1, 1};
        const std::array<ck::index_t, NDimSpatial> params{1};
        return conv_to_gemm_transformer.template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<1>(
            dim,
            dim,
            dim,
            lengths,
            lengths,
            lengths,
            strides,
            strides,
            strides,
            params,
            params,
            params,
            params,
            batch);
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 2, bool>::type = false>
    static auto GetABCGridDesc()
    {
        const ck::index_t dim   = 1;
        const ck::index_t batch = 1;
        const std::array<ck::index_t, NDimSpatial> lengths{1, 1};
        const std::array<ck::index_t, NDimSpatial + 3> strides{1, 1, 1, 1, 1};
        const std::array<ck::index_t, NDimSpatial> params{1, 1};
        return conv_to_gemm_transformer.template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<2>(
            dim,
            dim,
            dim,
            lengths,
            lengths,
            lengths,
            strides,
            strides,
            strides,
            params,
            params,
            params,
            params,
            batch);
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 3, bool>::type = false>
    static auto GetABCGridDesc()
    {
        const ck::index_t dim   = 1;
        const ck::index_t batch = 1;
        const std::array<ck::index_t, NDimSpatial> lengths{1, 1, 1};
        const std::array<ck::index_t, NDimSpatial + 3> strides{1, 1, 1, 1, 1, 1};
        const std::array<ck::index_t, NDimSpatial> params{1, 1, 1};
        return conv_to_gemm_transformer.template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<3>(
            dim,
            dim,
            dim,
            lengths,
            lengths,
            lengths,
            strides,
            strides,
            strides,
            params,
            params,
            params,
            params,
            batch);
    }

    using ABCGridDescs = decltype(GetABCGridDesc<NDimSpatial>());

    using AGridDesc_K0_M_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I0])>;
    using BGridDesc_K0_N_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I1])>;
    using CGridDesc_M_N     = remove_cvref_t<decltype(ABCGridDescs{}[I2])>;

    template <index_t NXdlPerWave_>
    using GridwiseGemmBase = GridwiseGemm_xdl_cshuffle_conv_v3<
        tensor_layout::gemm::RowMajor,
        tensor_layout::gemm::ColumnMajor,
        tensor_layout::gemm::RowMajor,
        ADataType,
        BDataType,
        AccDataType,
        CDataType,
        CDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        GemmSpec,
        BlockSize,
        MPerBlock,
        NPerBlock,
        K0PerBlock,
        K1,
        K1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave_,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false,
        ABlockLdsAddExtraM,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false,
        BBlockLdsAddExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CBlockTransferScalarPerVector_NWaveNPerXdl,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB>;
    using GridwiseGemm64 = GridwiseGemmBase<math::max(NXdlPerWave64, 1)>;
    using GridwiseGemm32 = GridwiseGemmBase<NXdlPerWave32>;

    // Argument
    using CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock =
        decltype(GridwiseGemm64::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            CGridDesc_M_N{}, 1, 1));

    struct ActiveWorkgroupsPerCU
    {
        template <typename GridwiseGemm>
        static int GetMaxOccupancy()
        {
            constexpr int dynamic_smem_size = 0;
            constexpr index_t minimum_occupancy =
                BlkGemmPipeSched == BlockGemmPipelineScheduler::Intrawave ? 1 : 2;
            int max_occupancy = 0;

            if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v4)
            {
                hip_check_error(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                    &max_occupancy,
                    kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3_2lds<
                        GridwiseGemm,
                        remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                        remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                        remove_reference_t<DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                        ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                        true,
                        InMemoryDataOperationEnum::AtomicAdd,
                        minimum_occupancy>,
                    BlockSize,
                    dynamic_smem_size));
            }
            else
            {
                hip_check_error(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                    &max_occupancy,
                    kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                        GridwiseGemm,
                        remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                        remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                        remove_reference_t<DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                        ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                        true,
                        InMemoryDataOperationEnum::AtomicAdd,
                        minimum_occupancy>,
                    BlockSize,
                    dynamic_smem_size));
            }
            return std::max(1, max_occupancy);
        }

        ActiveWorkgroupsPerCU()
        {
            max_occupancy_ = 1;
            if(get_warp_size() == 64)
            {
                if constexpr(NXdlPerWave64 > 0)
                {
                    max_occupancy_ = GetMaxOccupancy<GridwiseGemm64>();
                }
            }
            else
            {
                if constexpr(NXdlPerWave32 > 0)
                {
                    max_occupancy_ = GetMaxOccupancy<GridwiseGemm32>();
                }
            }
        }
        int max_occupancy_;
    };

    struct Argument : public BaseArgument, public ArgumentSplitK
    {
        Argument(const InDataType* p_in_grid,
                 WeiDataType* p_wei_grid,
                 const OutDataType* p_out_grid,
                 const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_lengths, // input
                 const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths, // weight
                 const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_strides,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<ck::index_t, NDimSpatial>& input_left_pads,
                 const std::array<ck::index_t, NDimSpatial>& input_right_pads,
                 const ck::index_t M01,
                 const ck::index_t N01,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op,
                 ck::index_t split_k)
            : p_a_grid_{p_out_grid},
              p_b_grid_{p_in_grid},
              p_c_grid_{p_wei_grid},
              a_grid_desc_k0_m_k1_{},
              b_grid_desc_k0_n_k1_{},
              c_grid_desc_m_n_{},
              c_grid_desc_mblock_mperblock_nblock_nperblock_{},
              compute_ptr_offset_of_batch_{},
              M01_{M01},
              N01_{N01},
              a_element_op_{out_element_op},
              b_element_op_{in_element_op},
              c_element_op_{wei_element_op},
              Conv_G_{b_g_n_c_wis_lengths[0]},
              Conv_N_{b_g_n_c_wis_lengths[1]},
              Conv_K_{e_g_k_c_xs_lengths[1]},
              Conv_C_{b_g_n_c_wis_lengths[2]},
              input_spatial_lengths_{},
              filter_spatial_lengths_{},
              output_spatial_lengths_{},
              conv_filter_strides_{conv_filter_strides},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads}
        {
            static ActiveWorkgroupsPerCU active_workgroups_per_cu;

            c_space_size_bytes =
                ck::accumulate_n<long_index_t>(
                    e_g_k_c_xs_lengths.begin(), NDimSpatial + I3, 1, std::multiplies<>()) *
                sizeof(WeiDataType);

            constexpr index_t spatial_offset = 3;
            std::copy(begin(b_g_n_c_wis_lengths) + spatial_offset,
                      end(b_g_n_c_wis_lengths),
                      begin(input_spatial_lengths_));
            std::copy(begin(e_g_k_c_xs_lengths) + spatial_offset,
                      end(e_g_k_c_xs_lengths),
                      begin(filter_spatial_lengths_));
            std::copy(begin(a_g_n_k_wos_lengths) + spatial_offset,
                      end(a_g_n_k_wos_lengths),
                      begin(output_spatial_lengths_));

#if !DISABLE_SPLIT_K_AUTODEDUCE_FOR_ONE_STAGE_KERNELS
            if(split_k < 0)
            {
                ck::index_t gemmM, gemmN, gemmK;
                std::tie(gemmM, gemmN, gemmK) =
                    get_bwd_weight_gemm_sizes<NDimSpatial>(a_g_n_k_wos_lengths, e_g_k_c_xs_lengths);

                const auto grid_size =
                    calculate_mn_grid_size<MPerBlock, NPerBlock>(gemmM, gemmN) * Conv_G_;
                k_batch_ = get_best_occupancy_k_batch_value(active_workgroups_per_cu.max_occupancy_,
                                                            grid_size);

                // Ensure that k_batch_ does not exceed the maximum value
                // for the GEMM pipeline.
                const auto k_batch_max = static_cast<index_t>((gemmK - 1) / K0PerBlock);
                k_batch_               = std::max(std::min(k_batch_, k_batch_max), 1);

                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "[SPLIT-K AUTODEDUCE] k_batch max value: " << k_batch_max
                              << std::endl;
                    std::cout << "[SPLIT-K AUTODEDUCE] Final k_batch value: " << k_batch_
                              << std::endl;
                }
            }
            else
#endif
            {
                k_batch_ = split_k;
            }

            // Create descriptors first (with hack flags temporarily set to false)
            // so we can check if element space sizes match product of dimensions
            const auto descs_initial =
                conv_to_gemm_transformer
                    .template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<NDimSpatial>(
                        Conv_N_,
                        Conv_K_,
                        Conv_C_,
                        input_spatial_lengths_,
                        filter_spatial_lengths_,
                        output_spatial_lengths_,
                        b_g_n_c_wis_strides,
                        e_g_k_c_xs_strides,
                        a_g_n_k_wos_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        k_batch_,
                        false, // split_k_offset_b_hack (temporary)
                        true); // use_full_batch_kindex=true for V1-compatible descriptors

            split_k_offset_hack_ =
                SplitKHackEligibility<NDimSpatial, InLayout, WeiLayout, OutLayout>::Check(
                    descs_initial[I0],
                    descs_initial[I1],
                    k_batch_,
                    Conv_N_,
                    output_spatial_lengths_,
                    K0PerBlock);

            // Now create descriptors with the correct hack flag
            const auto descs =
                conv_to_gemm_transformer
                    .template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<NDimSpatial>(
                        Conv_N_,
                        Conv_K_,
                        Conv_C_,
                        input_spatial_lengths_,
                        filter_spatial_lengths_,
                        output_spatial_lengths_,
                        b_g_n_c_wis_strides,
                        e_g_k_c_xs_strides,
                        a_g_n_k_wos_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        k_batch_,
                        split_k_offset_hack_,
                        true); // use_full_batch_kindex=true for V1-compatible descriptors

            a_grid_desc_k0_m_k1_ = descs[I0];
            b_grid_desc_k0_n_k1_ = descs[I1];
            c_grid_desc_m_n_     = descs[I2];

            // Calculate stride using CalculateOffset method for accurate stride
            // This works correctly for any descriptor transform pipeline
            split_k_stride_a_ = a_grid_desc_k0_m_k1_.GetElementSpaceSize();
            if(split_k_offset_hack_)
                split_k_stride_a_ /= k_batch_;

            split_k_stride_b_ = b_grid_desc_k0_n_k1_.GetElementSpaceSize();
            if(split_k_offset_hack_)
                split_k_stride_b_ /= k_batch_;

            // A/B/C Batch Stride
            compute_ptr_offset_of_batch_.BatchStrideA_ = a_g_n_k_wos_strides[0];
            compute_ptr_offset_of_batch_.BatchStrideB_ = b_g_n_c_wis_strides[0];
            compute_ptr_offset_of_batch_.BatchStrideC_ =
                Conv_K_ * Conv_C_ *
                std::accumulate(begin(filter_spatial_lengths_),
                                end(filter_spatial_lengths_),
                                index_t{1},
                                std::multiplies<>{});
            const index_t GemmM = a_grid_desc_k0_m_k1_.GetLength(I1);
            const index_t GemmN = b_grid_desc_k0_n_k1_.GetLength(I1);

            c_grid_desc_mblock_mperblock_nblock_nperblock_ =
                GridwiseGemm64::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                    c_grid_desc_m_n_,
                    GridwiseGemm64::CalculateMBlock(GemmM),
                    GridwiseGemm64::CalculateNBlock(GemmN));
        }

        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;
        AGridDesc_K0_M_K1 a_grid_desc_k0_m_k1_;
        BGridDesc_K0_N_K1 b_grid_desc_k0_n_k1_;
        CGridDesc_M_N c_grid_desc_m_n_;
        CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock c_grid_desc_mblock_mperblock_nblock_nperblock_;

        // for computing batch offset
        ComputePtrOffsetOfStridedBatch<I1, I1, I0> compute_ptr_offset_of_batch_;

        index_t M01_;
        index_t N01_;

        OutElementwiseOperation a_element_op_;
        InElementwiseOperation b_element_op_;
        WeiElementwiseOperation c_element_op_;

        // for checking IsSupportedArgument()
        const index_t Conv_G_;
        const index_t Conv_N_;
        const index_t Conv_K_;
        const index_t Conv_C_;
        std::array<ck::index_t, NDimSpatial> input_spatial_lengths_;
        std::array<ck::index_t, NDimSpatial> filter_spatial_lengths_;
        std::array<ck::index_t, NDimSpatial> output_spatial_lengths_;
        const std::array<ck::index_t, NDimSpatial>& conv_filter_strides_;
        const std::array<ck::index_t, NDimSpatial>& input_left_pads_;
        const std::array<ck::index_t, NDimSpatial>& input_right_pads_;
        long_index_t c_space_size_bytes;

        bool split_k_offset_hack_;
        long_index_t split_k_stride_a_, split_k_stride_b_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        void ShowInfo(const Argument& arg)
        {
            std::cout << "arg.a_grid_desc_k0_m_k1_{" << arg.a_grid_desc_k0_m_k1_.GetLength(I0)
                      << ", " << arg.a_grid_desc_k0_m_k1_.GetLength(I1) << ", "
                      << arg.a_grid_desc_k0_m_k1_.GetLength(I2) << ", "
                      << arg.a_grid_desc_k0_m_k1_.GetLength(I3) << "}" << std::endl;

            std::cout << "arg.b_grid_desc_k0_n_k1_{" << arg.b_grid_desc_k0_n_k1_.GetLength(I0)
                      << ", " << arg.b_grid_desc_k0_n_k1_.GetLength(I1) << ", "
                      << arg.b_grid_desc_k0_n_k1_.GetLength(I2) << ", "
                      << arg.b_grid_desc_k0_n_k1_.GetLength(I3) << "}" << std::endl;

            std::cout << "arg.c_grid_desc_m_n_{" << arg.c_grid_desc_m_n_.GetLength(I0) << ", "
                      << arg.c_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
        }

        template <typename GridwiseGemm>
        float RunImp(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const index_t GemmM = arg.a_grid_desc_k0_m_k1_.GetLength(I1);
            const index_t GemmN = arg.b_grid_desc_k0_n_k1_.GetLength(I1);
            const index_t GemmK =
                arg.a_grid_desc_k0_m_k1_.GetLength(I0) * arg.a_grid_desc_k0_m_k1_.GetLength(I2);

            const ADataType* p_a_grid = arg.p_a_grid_;
            const BDataType* p_b_grid = arg.p_b_grid_;
            typename GridwiseGemm::Argument gemm_arg{
                p_a_grid, p_b_grid, arg.p_c_grid_, GemmM, GemmN, GemmK, I0, I0, I0, arg.k_batch_};

            index_t gdx, gdy, gdz;
            std::tie(gdx, gdy, gdz) = GridwiseGemm::CalculateGridSize(
                gemm_arg.M, gemm_arg.N, gemm_arg.KBatch, arg.Conv_G_);

            float ave_time = 0;

            index_t k_grain                  = gemm_arg.KBatch * K0PerBlock;
            index_t K_split                  = (gemm_arg.K + k_grain - 1) / k_grain * K0PerBlock;
            const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K_split);

            const auto num_k_per_block =
                arg.a_grid_desc_k0_m_k1_.GetLength(Number<0>{}) / gemm_arg.KBatch;

            const auto clear_workspace = [&]() {
                if(arg.k_batch_ > 1)
                {
                    hip_check_error(hipMemsetAsync(
                        gemm_arg.p_c_grid, 0, arg.c_space_size_bytes, stream_config.stream_id_));
                }
            };

            const auto Run = [&](const auto& kernel) {
                if(stream_config.flush_cache)
                {
                    typename GridwiseGemm::Argument gemm_arg_ = gemm_arg;
                    ck::utility::RotatingMemWrapper<typename GridwiseGemm::Argument> rotating_mem(
                        gemm_arg_,
                        stream_config.rotating_count,
                        gemm_arg_.M * gemm_arg_.K * sizeof(ADataType),
                        gemm_arg_.K * gemm_arg_.N * sizeof(BDataType));
                    rotating_mem.Print();

                    auto run_flush_cache = [&]() {
                        // flush icache
                        ck::utility::flush_icache();
                        // rotating mem
                        rotating_mem.Next();
                        clear_workspace();
                    };
                    ave_time += ck::utility::launch_and_time_kernel_with_preprocess<false>(
                        stream_config,
                        run_flush_cache,
                        kernel,
                        dim3(gdx, gdy, gdz),
                        dim3(BlockSize),
                        0,
                        gemm_arg_,
                        arg.a_grid_desc_k0_m_k1_,
                        arg.b_grid_desc_k0_n_k1_,
                        arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                        arg.compute_ptr_offset_of_batch_,
                        num_k_per_block,
                        arg.split_k_stride_a_,
                        arg.split_k_stride_b_,
                        arg.split_k_offset_hack_);
                }
                else
                {
                    ave_time += launch_and_time_kernel_with_preprocess(
                        stream_config,
                        clear_workspace,
                        kernel,
                        dim3(gdx, gdy, gdz),
                        dim3(BlockSize),
                        0,
                        gemm_arg,
                        arg.a_grid_desc_k0_m_k1_,
                        arg.b_grid_desc_k0_n_k1_,
                        arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                        arg.compute_ptr_offset_of_batch_,
                        num_k_per_block,
                        arg.split_k_stride_a_,
                        arg.split_k_stride_b_,
                        arg.split_k_offset_hack_);
                }
            };

            constexpr index_t minimum_occupancy =
                BlkGemmPipeSched == BlockGemmPipelineScheduler::Intrawave ? 1 : 2;

            if(has_main_k_block_loop)
            {
                // Tail number always full
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 ||
                             BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                {
                    if(arg.k_batch_ > 1)
                    {
                        const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                            GridwiseGemm,
                            remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                            remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                            remove_reference_t<
                                DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                            ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                            true,
                            InMemoryDataOperationEnum::AtomicAdd,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                            GridwiseGemm,
                            remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                            remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                            remove_reference_t<
                                DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                            ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                            true,
                            InMemoryDataOperationEnum::Set,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                }
                // Tail number could be One to Seven
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v2)
                {
                    if(arg.k_batch_ > 1)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::One)
                        {
                            const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                GridwiseGemm,
                                remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                remove_reference_t<
                                    DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                true,
                                InMemoryDataOperationEnum::AtomicAdd,
                                minimum_occupancy,
                                TailNumber::One>;
                            Run(kernel);
                        }
                        else if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                                TailNumber::Full)
                        {
                            const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                GridwiseGemm,
                                remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                remove_reference_t<
                                    DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                true,
                                InMemoryDataOperationEnum::AtomicAdd,
                                minimum_occupancy,
                                TailNumber::Full>;
                            Run(kernel);
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 2)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Two)
                            {
                                const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                    GridwiseGemm,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Two>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 3)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Three)
                            {
                                const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                    GridwiseGemm,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Three>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 4)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Four)
                            {
                                const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                    GridwiseGemm,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Four>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 5)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Five)
                            {
                                const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                    GridwiseGemm,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Five>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 6)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Six)
                            {
                                const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                    GridwiseGemm,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Six>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 7)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Seven)
                            {
                                const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                    GridwiseGemm,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Seven>;
                                Run(kernel);
                            }
                        }
                    }
                    else
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::One)
                        {
                            const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                GridwiseGemm,
                                remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                remove_reference_t<
                                    DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::One>;
                            Run(kernel);
                        }
                        else if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                                TailNumber::Full)
                        {
                            const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                GridwiseGemm,
                                remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                remove_reference_t<
                                    DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Full>;
                            Run(kernel);
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 2)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Two)
                            {
                                const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                    GridwiseGemm,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Two>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 3)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Three)
                            {
                                const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                    GridwiseGemm,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Three>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 4)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Four)
                            {
                                const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                    GridwiseGemm,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Four>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 5)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Five)
                            {
                                const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                    GridwiseGemm,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Five>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 6)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Six)
                            {
                                const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                    GridwiseGemm,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Six>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 7)
                        {
                            if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Seven)
                            {
                                const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                    GridwiseGemm,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Seven>;
                                Run(kernel);
                            }
                        }
                    }
                }

                // Tail number could be Odd or Even
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v4)
                {
                    if(arg.k_batch_ > 1)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                        {
                            const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3_2lds<
                                GridwiseGemm,
                                remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                remove_reference_t<
                                    DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                true,
                                InMemoryDataOperationEnum::AtomicAdd,
                                minimum_occupancy,
                                TailNumber::Odd>;
                            Run(kernel);
                        }
                        else
                        {
                            const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3_2lds<
                                GridwiseGemm,
                                remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                remove_reference_t<
                                    DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                true,
                                InMemoryDataOperationEnum::AtomicAdd,
                                minimum_occupancy,
                                TailNumber::Even>;
                            Run(kernel);
                        }
                    }
                    else
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                        {
                            const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3_2lds<
                                GridwiseGemm,
                                remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                remove_reference_t<
                                    DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Odd>;
                            Run(kernel);
                        }
                        else
                        {
                            const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3_2lds<
                                GridwiseGemm,
                                remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                remove_reference_t<
                                    DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Even>;
                            Run(kernel);
                        }
                    }
                }
                else
                {
                    if(arg.k_batch_ > 1)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                        {
                            const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                GridwiseGemm,
                                remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                remove_reference_t<
                                    DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                true,
                                InMemoryDataOperationEnum::AtomicAdd,
                                minimum_occupancy,
                                TailNumber::Odd>;
                            Run(kernel);
                        }
                        else
                        {
                            const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                GridwiseGemm,
                                remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                remove_reference_t<
                                    DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                true,
                                InMemoryDataOperationEnum::AtomicAdd,
                                minimum_occupancy,
                                TailNumber::Even>;
                            Run(kernel);
                        }
                    }
                    else
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                        {
                            const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                GridwiseGemm,
                                remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                remove_reference_t<
                                    DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Odd>;
                            Run(kernel);
                        }
                        else
                        {
                            const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                GridwiseGemm,
                                remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                remove_reference_t<
                                    DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Even>;
                            Run(kernel);
                        }
                    }
                }
            }
            else
            {
                // Tail number always 1
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
                {
                    if(arg.k_batch_ > 1)
                    {
                        const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                            GridwiseGemm,
                            remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                            remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                            remove_reference_t<
                                DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                            ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                            false,
                            InMemoryDataOperationEnum::AtomicAdd,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                            GridwiseGemm,
                            remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                            remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                            remove_reference_t<
                                DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                            ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                            false,
                            InMemoryDataOperationEnum::Set,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                }
            }
            return ave_time;
        }

        INVOKER_RUN_IMPL

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
#if DISABLE_SPLIT_K_AUTODEDUCE_FOR_ONE_STAGE_KERNELS
        if(arg.k_batch_ < 0)
        {
            return false;
        }
#endif

        const index_t GemmM = arg.a_grid_desc_k0_m_k1_.GetLength(I1);
        const index_t GemmN = arg.b_grid_desc_k0_n_k1_.GetLength(I1);
        const index_t GemmK =
            arg.a_grid_desc_k0_m_k1_.GetLength(I0) * arg.a_grid_desc_k0_m_k1_.GetLength(I2);

        if constexpr(is_same_v<ComputeTypeA, ck::tf32_t> || is_same_v<ComputeTypeB, ck::tf32_t>)
        {
            if(!is_tf32_supported())
            {
                return false;
            }
            if constexpr(!is_same_v<ComputeTypeA, ComputeTypeB>)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "ComputeDataType for A and B should be same while using TF32"
                              << std::endl;
                }
                return false;
            }
        }

        if(get_warp_size() == 64)
        {
            if constexpr(NXdlPerWave64 > 0)
            {
                typename GridwiseGemm64::Argument gemm_arg{
                    nullptr, nullptr, nullptr, GemmM, GemmN, GemmK, I0, I0, I0, arg.k_batch_};

                const auto num_k_loop = gemm_arg.AK0 / (K0PerBlock / K1);
                if constexpr(BlkGemmPipelineVer != BlockGemmPipelineVersion::v1)
                {
                    if(num_k_loop <= GridwiseGemm64::BlockwiseGemmPipe::PrefetchStages)
                    {
                        return false;
                    }
                }
            }
            else
            {
                return false;
            }
        }
        else
        {
            if constexpr(NXdlPerWave32 > 0)
            {
                typename GridwiseGemm32::Argument gemm_arg{
                    nullptr, nullptr, nullptr, GemmM, GemmN, GemmK, I0, I0, I0, arg.k_batch_};

                const auto num_k_loop = gemm_arg.AK0 / (K0PerBlock / K1);
                if constexpr(BlkGemmPipelineVer != BlockGemmPipelineVersion::v1)
                {
                    if(num_k_loop <= GridwiseGemm64::BlockwiseGemmPipe::PrefetchStages)
                    {
                        return false;
                    }
                }
            }
            else
            {
                return false;
            }
        }

        if(!ck::is_xdl_wmma_supported<ComputeTypeA, ComputeTypeB, MPerXDL, NPerXDL>())
        {
            return false;
        }
        if(is_gfx11_supported() && arg.k_batch_ > 1)
        {
            return false;
        }
        if(!is_bf16_atomic_supported() && std::is_same_v<CDataType, ck::bhalf_t> &&
           arg.k_batch_ > 1)
        {
            return false;
        }

        if constexpr(NDimSpatial == 1)
        {
            if constexpr(!is_GNWC_GKXC_GNWK<InLayout, WeiLayout, OutLayout>())
            {
                return false;
            }
        }
        else if constexpr(NDimSpatial == 2)
        {
            if constexpr(!(is_NHWGC_GKYXC_NHWGK<InLayout, WeiLayout, OutLayout>() ||
                           is_GNHWC_GKYXC_GNHWK<InLayout, WeiLayout, OutLayout>()))
            {
                return false;
            }
        }
        else if constexpr(NDimSpatial == 3)
        {
            if constexpr(!(is_NDHWGC_GKZYXC_NDHWGK<InLayout, WeiLayout, OutLayout>() ||
                           is_GNDHWC_GKZYXC_GNDHWK<InLayout, WeiLayout, OutLayout>()))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        if constexpr(ConvBackwardWeightSpecialization ==
                     ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
        {
            // check if it's 1x1, stride=1 pad = 0 conv
            for(int i = 0; i < NDimSpatial; i++)
            {
                if(!(arg.filter_spatial_lengths_[i] == 1 && arg.conv_filter_strides_[i] == 1 &&
                     arg.input_left_pads_[i] == 0 && arg.input_right_pads_[i] == 0))
                {
                    return false;
                }
            }
        }
        if(!(ABlockTransferSrcVectorDim == 1 && BBlockTransferSrcVectorDim == 1 &&
             arg.Conv_K_ % ABlockTransferSrcScalarPerVector == 0 &&
             arg.Conv_C_ % BBlockTransferSrcScalarPerVector == 0))
        {
            return false;
        }

        // vector store C matrix into global memory
        if(!(arg.Conv_C_ % CBlockTransferScalarPerVector_NWaveNPerXdl == 0))
        {
            return false;
        }

        constexpr long_index_t TwoGB = (long_index_t{1} << 31);
        const bool a_small_enough    = arg.a_grid_desc_k0_m_k1_.GetElementSpaceSize() /
                                        (arg.split_k_offset_hack_ ? arg.k_batch_ : 1) *
                                        sizeof(ADataType) <=
                                    TwoGB;
        const bool b_small_enough = arg.b_grid_desc_k0_n_k1_.GetElementSpaceSize() /
                                        (arg.split_k_offset_hack_ ? arg.k_batch_ : 1) *
                                        sizeof(BDataType) <=
                                    TwoGB;
        const bool c_small_enough =
            arg.c_grid_desc_m_n_.GetElementSpaceSize() * sizeof(CDataType) <= TwoGB;
        if(!(a_small_enough && b_small_enough && c_small_enough))
        {
            return false;
        }

        // Gridwise GEMM size
        return true;
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto
    MakeArgument(const InDataType* p_in_grid,
                 WeiDataType* p_wei_grid,
                 const OutDataType* p_out_grid,
                 const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_lengths, // input
                 const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths, // weight
                 const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_strides,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<ck::index_t, NDimSpatial>& input_left_pads,
                 const std::array<ck::index_t, NDimSpatial>& input_right_pads,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op,
                 const ck::index_t split_k)
    {
        return Argument{p_in_grid,
                        p_wei_grid,
                        p_out_grid,
                        b_g_n_c_wis_lengths, // input
                        b_g_n_c_wis_strides,
                        e_g_k_c_xs_lengths, // weight
                        e_g_k_c_xs_strides,
                        a_g_n_k_wos_lengths, // output
                        a_g_n_k_wos_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        1,
                        1,
                        in_element_op,
                        wei_element_op,
                        out_element_op,
                        split_k};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in_grid,
                        void* p_wei_grid,
                        const void* p_out_grid,
                        const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_lengths, // input
                        const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_strides,
                        const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths, // weight
                        const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_strides,
                        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output
                        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
                        const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<ck::index_t, NDimSpatial>& input_left_pads,
                        const std::array<ck::index_t, NDimSpatial>& input_right_pads,
                        InElementwiseOperation in_element_op,
                        WeiElementwiseOperation wei_element_op,
                        OutElementwiseOperation out_element_op,
                        const ck::index_t split_k) override
    {
        return std::make_unique<Argument>(static_cast<const InDataType*>(p_in_grid),
                                          static_cast<WeiDataType*>(p_wei_grid),
                                          static_cast<const OutDataType*>(p_out_grid),
                                          b_g_n_c_wis_lengths, // input
                                          b_g_n_c_wis_strides,
                                          e_g_k_c_xs_lengths, // weight
                                          e_g_k_c_xs_strides,
                                          a_g_n_k_wos_lengths, // output
                                          a_g_n_k_wos_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          1,
                                          1,
                                          in_element_op,
                                          wei_element_op,
                                          out_element_op,
                                          split_k);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceGroupedConvBwdWeight_Xdl_CShuffleV3"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << K0PerBlock << ", "
            << getConvBackwardWeightSpecializationString(ConvBackwardWeightSpecialization) << ", "
            << K1 << ", "
            << MXdlPerWave << ", "
            << NXdlPerWave << ", "
            << ABlockTransferSrcScalarPerVector << ", "
            << ABlockTransferDstScalarPerVector_K1 << ", "
            << BBlockTransferSrcScalarPerVector << ", "
            << BBlockTransferDstScalarPerVector_K1 << ", "
            << CShuffleMXdlPerWavePerShuffle << ", "
            << CShuffleNXdlPerWavePerShuffle << ", "
            << CBlockTransferScalarPerVector_NWaveNPerXdl
            << ">";
        // clang-format on

        return str.str();
    }

#ifdef CK_EXPERIMENTAL_BUILDER
    std::string GetInstanceString() const override
    {
        static_assert(ck_tile::reflect::HasInstanceTraits<DeviceOp>,
                      "Specialization of instance_traits not found. Please check that a "
                      "specialization exists in file "
                      "ck_tile/builder/reflect/"
                      "instance_traits_device_grouped_conv_bwd_weight_xdl_cshuffle_v3.hpp "
                      "for the given template parameters.");
        return ck_tile::reflect::instance_string<DeviceOp>();
    }

    std::unique_ptr<ck_tile::reflect::Description> describe() const override
    {
        return std::make_unique<ck_tile::reflect::InstanceStringDescription>(GetInstanceString());
    }
#endif
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
