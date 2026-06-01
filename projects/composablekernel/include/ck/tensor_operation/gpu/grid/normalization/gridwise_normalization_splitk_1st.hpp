// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/math.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_welford.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_welford.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

template <typename XDataType,
          typename ComputeDataType,
          typename MeanVarDataType,
          typename XGridDesc_M_K,
          typename MeanVarGridDesc_M_KBlock,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t XSrcVectorDim,
          index_t XSrcVectorSize>
struct GridwiseNormalizationSplitK1st
{
    static_assert((XSrcVectorDim == 0 && MThreadSliceSize % XSrcVectorSize == 0) ||
                      (XSrcVectorDim == 1 && KThreadSliceSize % XSrcVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    static constexpr bool reorder_thread_cluster = (XSrcVectorDim == 0);

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    using ThreadClusterLengths_M_K = Sequence<MThreadClusterSize, KThreadClusterSize>;

    using ThreadBufferDimAccessOrder =
        typename conditional<reorder_thread_cluster, Sequence<1, 0>, Sequence<0, 1>>::type;

    using ThreadClusterArrangeOrder =
        typename conditional<reorder_thread_cluster, Sequence<1, 0>, Sequence<0, 1>>::type;

    static constexpr auto thread_cluster_desc =
        make_cluster_descriptor(ThreadClusterLengths_M_K{}, ThreadClusterArrangeOrder{});

    using ThreadBufferLengths_M_K                = Sequence<MThreadSliceSize, XSrcVectorSize>;
    static constexpr auto thread_buffer_desc_m_k = make_naive_tensor_descriptor_packed(
        make_tuple(Number<MThreadSliceSize>{}, Number<XSrcVectorSize>{}));

    using ThreadBufferLengths_M_1 = Sequence<MThreadSliceSize, 1>;
    static constexpr auto thread_buffer_desc_m_1 =
        make_naive_tensor_descriptor_packed(make_tuple(Number<MThreadSliceSize>{}, I1));

    using ThreadReduceSrcDesc_M_K = decltype(make_naive_tensor_descriptor_packed(
        make_tuple(Number<MThreadSliceSize>{}, Number<XSrcVectorSize>{})));
    using ThreadReduceDstDesc_M =
        decltype(make_naive_tensor_descriptor_packed(make_tuple(Number<MThreadSliceSize>{})));

    using ThreadwiseWelford =
        ThreadwiseWelford<ComputeDataType, ThreadReduceSrcDesc_M_K, ThreadReduceDstDesc_M>;

    using BlockwiseWelford = BlockwiseWelford<ComputeDataType,
                                              BlockSize,
                                              ThreadClusterLengths_M_K,
                                              ThreadClusterArrangeOrder,
                                              false>;

    using PassThroughOp = tensor_operation::element_wise::PassThrough;

    static constexpr index_t M_BlockTileSize     = MThreadClusterSize * MThreadSliceSize;
    static constexpr index_t K_BlockTileSize     = KThreadClusterSize * KThreadSliceSize;
    static constexpr index_t K_BlockTileStepSize = KThreadClusterSize * XSrcVectorSize;

    static constexpr auto ThreadBufferNumber = Number<KThreadSliceSize / XSrcVectorSize>{};

    __device__ static int
    GetKPerThread(int k, int kRaw, int kGridSize, int block_k_cluster_id, int thread_k_cluster_id)
    {
        bool is_rightmost_block = block_k_cluster_id == kGridSize - 1;

        if(is_rightmost_block)
        {
            int left_kPerBlock  = math::integer_divide_ceil(k, kGridSize);
            int kRightmostBlock = kRaw - left_kPerBlock * (kGridSize - 1);
            int kPerThread      = kRightmostBlock < K_BlockTileSize
                                      ? 0
                                      : KThreadSliceSize * (kRightmostBlock / K_BlockTileSize);
            int kPerBlockTail   = kRightmostBlock - kPerThread * KThreadClusterSize;

            if(kPerBlockTail > 0)
            {
                static_for<0, ThreadBufferNumber, 1>{}([&](auto i) {
                    int thread_max_len =
                        (thread_k_cluster_id + 1) * XSrcVectorSize + K_BlockTileStepSize * i;
                    int delta = thread_max_len - kPerBlockTail;
                    delta     = math::clamp(thread_max_len - kPerBlockTail, 0, XSrcVectorSize);
                    kPerThread += XSrcVectorSize - delta;
                });
            }

            return kPerThread;
        }
        else
        {
            int kPerBlock = math::integer_divide_ceil(k, kGridSize);
            return KThreadSliceSize * (kPerBlock / K_BlockTileSize);
        }
    }

    // Calculate mean and variance by welford along k dimension
    __device__ static void Run(const XGridDesc_M_K& x_grid_desc_m_k,
                               const MeanVarGridDesc_M_KBlock& mean_var_grid_desc_m_kblock,
                               index_t num_k_block_tile_iteration,
                               const XDataType* const __restrict__ p_x_global,
                               MeanVarDataType* const p_mean_global,
                               MeanVarDataType* const p_variance_global,
                               int32_t* const p_welford_count_global)
    {
        auto x_thread_buf = generate_tuple(
            [&](auto) {
                return StaticBuffer<AddressSpaceEnum::Vgpr,
                                    ComputeDataType,
                                    MThreadSliceSize * XSrcVectorSize,
                                    true>{};
            },
            Number<ThreadBufferNumber>{});

        StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, MThreadSliceSize, true>
            mean_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, MThreadSliceSize, true>
            var_thread_buf;

        const index_t thread_local_id = get_thread_local_1d_id();
        const index_t block_global_id = get_block_1d_id();

        const index_t k_grid_size        = mean_var_grid_desc_m_kblock.GetLength(I1);
        const index_t block_m_cluster_id = block_global_id / k_grid_size;
        const index_t block_k_cluster_id = block_global_id % k_grid_size;

        const auto thread_cluster_idx =
            thread_cluster_desc.CalculateBottomIndex(make_multi_index(thread_local_id));

        const auto thread_m_cluster_id = thread_cluster_idx[I0];
        const auto thread_k_cluster_id = thread_cluster_idx[I1];

        const index_t reduceSizePerBlock = K_BlockTileSize * num_k_block_tile_iteration;

        auto threadwise_x_load = ThreadwiseTensorSliceTransfer_v2<XDataType,
                                                                  ComputeDataType,
                                                                  XGridDesc_M_K,
                                                                  decltype(thread_buffer_desc_m_k),
                                                                  ThreadBufferLengths_M_K,
                                                                  ThreadBufferDimAccessOrder,
                                                                  XSrcVectorDim,
                                                                  XSrcVectorSize,
                                                                  1,
                                                                  true>(
            x_grid_desc_m_k,
            make_multi_index(
                block_m_cluster_id * M_BlockTileSize + thread_m_cluster_id * MThreadSliceSize,
                block_k_cluster_id * reduceSizePerBlock + thread_k_cluster_id * XSrcVectorSize));

        auto mean_var_count_store_index = make_multi_index(
            block_m_cluster_id * M_BlockTileSize + thread_m_cluster_id * MThreadSliceSize,
            block_k_cluster_id);

        auto threadwise_welford_mean_var_store =
            ThreadwiseTensorSliceTransfer_v1r3<ComputeDataType,
                                               MeanVarDataType,
                                               decltype(thread_buffer_desc_m_1),
                                               MeanVarGridDesc_M_KBlock,
                                               PassThroughOp,
                                               ThreadBufferLengths_M_1,
                                               Sequence<0, 1>,
                                               1,
                                               1,
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               true>(
                mean_var_grid_desc_m_kblock, mean_var_count_store_index, PassThroughOp{});

        constexpr auto thread_copy_fwd_step_m_k = make_multi_index(0, K_BlockTileStepSize);

        const auto x_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_x_global, x_grid_desc_m_k.GetElementSpaceSize());

        auto mean_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_mean_global, mean_var_grid_desc_m_kblock.GetElementSpaceSize());

        auto var_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_variance_global, mean_var_grid_desc_m_kblock.GetElementSpaceSize());

        auto threadwise_welford       = ThreadwiseWelford();
        int kRaw                      = x_grid_desc_m_k.GetTransforms()[I2].GetUpperLengths()[I0];
        threadwise_welford.max_count_ = GetKPerThread(x_grid_desc_m_k.GetLength(I1),
                                                      kRaw,
                                                      k_grid_size,
                                                      block_k_cluster_id,
                                                      thread_k_cluster_id);

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            mean_thread_buf(I) = type_convert<ComputeDataType>(0.0f);
            var_thread_buf(I)  = type_convert<ComputeDataType>(0.0f);
        });

        for(index_t k = 0; k < num_k_block_tile_iteration; ++k)
        {
            static_for<0, ThreadBufferNumber, 1>{}([&](auto i) {
                threadwise_x_load.Run(x_grid_desc_m_k,
                                      x_global_val_buf,
                                      thread_buffer_desc_m_k,
                                      make_tuple(I0, I0),
                                      x_thread_buf(i));
                threadwise_x_load.MoveSrcSliceWindow(x_grid_desc_m_k, thread_copy_fwd_step_m_k);
                threadwise_welford.Run(x_thread_buf[i], mean_thread_buf, var_thread_buf);
            });
        }

        int welford_count = 0;
        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            if constexpr(I > 0)
                block_sync_lds();

            int count = threadwise_welford.cur_count_;
            BlockwiseWelford::Run(mean_thread_buf(I), var_thread_buf(I), count);

            // The value of count is same for all I
            if constexpr(I == MThreadSliceSize - 1)
                welford_count = count;
        });

        if(thread_k_cluster_id == 0)
        {
            threadwise_welford_mean_var_store.Run(thread_buffer_desc_m_1,
                                                  make_tuple(I0, I0),
                                                  mean_thread_buf,
                                                  mean_var_grid_desc_m_kblock,
                                                  mean_global_val_buf);

            threadwise_welford_mean_var_store.Run(thread_buffer_desc_m_1,
                                                  make_tuple(I0, I0),
                                                  var_thread_buf,
                                                  mean_var_grid_desc_m_kblock,
                                                  var_global_val_buf);

            if(block_m_cluster_id == 0 && thread_m_cluster_id == 0)
                p_welford_count_global[block_k_cluster_id] = welford_count;
        }
    }
};

} // namespace ck
