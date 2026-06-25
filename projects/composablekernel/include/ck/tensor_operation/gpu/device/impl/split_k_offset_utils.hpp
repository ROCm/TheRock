// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <numeric>
#include "ck/utility/common_header.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_selector.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// Check if a tensor descriptor has compact layout
// Compact means: GetElementSpaceSize() == product of all dimension lengths
// Non-compact descriptors have complex transform pipelines that may not support split-k hack
template <typename Descriptor>
bool IsDescriptorCompact(const Descriptor& desc)
{
    // Calculate product of all dimensions
    long_index_t dims_product  = 1;
    constexpr index_t num_dims = Descriptor::GetNumOfDimension();

    // Use template recursion to multiply all dimension lengths
    static_for<0, num_dims, 1>{}(
        [&](auto i) { dims_product *= static_cast<long_index_t>(desc.GetLength(i)); });

    return desc.GetElementSpaceSize() == dims_product;
}

// Determine split-k hack eligibility for descriptor pair
// This checks all the conditions required for safely using the split-k offset hack
template <index_t NDimSpatial, typename InLayout, typename WeiLayout, typename OutLayout>
struct SplitKHackEligibility
{
    template <typename ADescriptor, typename BDescriptor>
    static bool
    Check(const ADescriptor& a_desc,
          const BDescriptor& b_desc,
          index_t k_batch,
          index_t Conv_N,
          const std::array<index_t, NDimSpatial>& output_spatial_lengths,
          index_t k_block_size) // K0PerBlock*K1 for v1, K0PerBlock for v3, KPerBlock for two-stage
    {
        // Only enable hack if k_batch > 1
        if(k_batch <= 1)
        {
            return false;
        }

        // Calculate output spatial product
        const index_t output_spatial_acum = std::accumulate(output_spatial_lengths.begin(),
                                                            output_spatial_lengths.end(),
                                                            index_t{1},
                                                            std::multiplies<index_t>());

        // Check various divisibility and layout requirements
        const bool is_k_not_paded = (Conv_N * output_spatial_acum) % (k_block_size * k_batch) == 0;

        const bool can_divide_n_spatial_by_k_batch = (Conv_N * output_spatial_acum) % k_batch == 0;

        const bool can_divide_n_by_k_batch = Conv_N % k_batch == 0;

        const bool is_correct_layout =
            is_NSpatialGC_GKSpatial_NSpatialGK<InLayout, WeiLayout, OutLayout>();

        const bool is_a_stride_divisible = a_desc.GetElementSpaceSize() % k_batch == 0;

        const bool is_b_stride_divisible = b_desc.GetElementSpaceSize() % k_batch == 0;

        // Check descriptor compactness
        const bool is_a_compact = IsDescriptorCompact(a_desc);
        const bool is_b_compact = IsDescriptorCompact(b_desc);

        // Require BOTH A and B to be eligible for the hack to avoid KBatch dimension mismatch
        // The gridwise kernel's CheckValidity requires A.KBatch == B.KBatch, so we must
        // apply the hack uniformly to both tensors to maintain kernel applicability
        const bool eligible = can_divide_n_spatial_by_k_batch && can_divide_n_by_k_batch &&
                              is_k_not_paded && is_correct_layout && is_a_stride_divisible &&
                              is_b_stride_divisible && is_a_compact && is_b_compact;

        return eligible;
    }
};

// Helper function to dispatch split-K hack for standard kernel (single LDS)
// Reduces code duplication in device layer implementations
template <typename GridwiseGemm,
          typename AGridDesc_AK0_M_K1,
          typename BGridDesc_BK0_N_K1,
          typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          TailNumber TailNum,
          typename ADataType,
          typename BDataType,
          typename CDataType>
__device__ void DispatchSplitKHack(const ADataType* p_a_grid,
                                   const BDataType* p_b_grid,
                                   CDataType* p_c_grid,
                                   void* p_shared,
                                   const typename GridwiseGemm::Argument& karg,
                                   const AGridDesc_AK0_M_K1& a_grid_desc_ak0_m_ak1,
                                   const BGridDesc_BK0_N_K1& b_grid_desc_bk0_n_bk1,
                                   const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock&
                                       c_grid_desc_mblock_mperblock_nblock_nperblock,
                                   index_t k_id,
                                   index_t k_batch,
                                   bool split_k_offset_hack)
{
    if(split_k_offset_hack)
    {
        GridwiseGemm::template Run<AGridDesc_AK0_M_K1,
                                   BGridDesc_BK0_N_K1,
                                   CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                                   HasMainKBlockLoop,
                                   CGlobalMemoryDataOperation,
                                   TailNum,
                                   true>(p_a_grid,
                                         p_b_grid,
                                         p_c_grid,
                                         p_shared,
                                         karg,
                                         a_grid_desc_ak0_m_ak1,
                                         b_grid_desc_bk0_n_bk1,
                                         c_grid_desc_mblock_mperblock_nblock_nperblock,
                                         k_id,
                                         k_batch);
    }
    else
    {
        GridwiseGemm::template Run<AGridDesc_AK0_M_K1,
                                   BGridDesc_BK0_N_K1,
                                   CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                                   HasMainKBlockLoop,
                                   CGlobalMemoryDataOperation,
                                   TailNum,
                                   false>(p_a_grid,
                                          p_b_grid,
                                          p_c_grid,
                                          p_shared,
                                          karg,
                                          a_grid_desc_ak0_m_ak1,
                                          b_grid_desc_bk0_n_bk1,
                                          c_grid_desc_mblock_mperblock_nblock_nperblock,
                                          k_id,
                                          k_batch);
    }
}

// Helper function to dispatch split-K hack for 2lds kernel
// Reduces code duplication in device layer implementations
template <typename GridwiseGemm,
          typename AGridDesc_AK0_M_K1,
          typename BGridDesc_BK0_N_K1,
          typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          TailNumber TailNum,
          typename ADataType,
          typename BDataType,
          typename CDataType>
__device__ void DispatchSplitKHack_2Lds(const ADataType* p_a_grid,
                                        const BDataType* p_b_grid,
                                        CDataType* p_c_grid,
                                        void* p_shared_0,
                                        void* p_shared_1,
                                        const typename GridwiseGemm::Argument& karg,
                                        const AGridDesc_AK0_M_K1& a_grid_desc_ak0_m_ak1,
                                        const BGridDesc_BK0_N_K1& b_grid_desc_bk0_n_bk1,
                                        const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock&
                                            c_grid_desc_mblock_mperblock_nblock_nperblock,
                                        index_t k_id,
                                        index_t k_batch,
                                        bool split_k_offset_hack)
{
    if(split_k_offset_hack)
    {
        GridwiseGemm::template Run_2Lds<AGridDesc_AK0_M_K1,
                                        BGridDesc_BK0_N_K1,
                                        CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                                        HasMainKBlockLoop,
                                        CGlobalMemoryDataOperation,
                                        TailNum,
                                        true>(p_a_grid,
                                              p_b_grid,
                                              p_c_grid,
                                              p_shared_0,
                                              p_shared_1,
                                              karg,
                                              a_grid_desc_ak0_m_ak1,
                                              b_grid_desc_bk0_n_bk1,
                                              c_grid_desc_mblock_mperblock_nblock_nperblock,
                                              k_id,
                                              k_batch);
    }
    else
    {
        GridwiseGemm::template Run_2Lds<AGridDesc_AK0_M_K1,
                                        BGridDesc_BK0_N_K1,
                                        CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                                        HasMainKBlockLoop,
                                        CGlobalMemoryDataOperation,
                                        TailNum,
                                        false>(p_a_grid,
                                               p_b_grid,
                                               p_c_grid,
                                               p_shared_0,
                                               p_shared_1,
                                               karg,
                                               a_grid_desc_ak0_m_ak1,
                                               b_grid_desc_bk0_n_bk1,
                                               c_grid_desc_mblock_mperblock_nblock_nperblock,
                                               k_id,
                                               k_batch);
    }
}

} // namespace device
} // namespace tensor_operation
} // namespace ck
