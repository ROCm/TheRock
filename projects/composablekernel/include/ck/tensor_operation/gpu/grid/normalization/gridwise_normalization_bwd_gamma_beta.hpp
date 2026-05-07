// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/block/reduction_functions_blockwise.hpp"

namespace ck {

// dgamma = reduce_sum(dy * (x - mean) * inv_std)
// dbeta = reduce_sum(dy)
template <typename DYDataType,
          typename XDataType,
          typename MeanInvStdDataType,
          typename ComputeDataType,
          typename DGammaDataType,
          typename DBetaDataType,
          typename GridDesc_M_K,
          typename GridDesc_M,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t DYSrcVectorDim,
          index_t DYSrcVectorSize,
          index_t XSrcVectorDim,
          index_t XSrcVectorSize,
          index_t MeanInvStdSrcVectorDim,
          index_t MeanInvStdSrcVectorSize,
          index_t DGammaDstVectorSize,
          index_t DBetaDstVectorSize>
struct GridwiseNormalizationBwdGammaBeta_mk_to_k
{
    // if we just check ThreadSliceSize % VectorSize == 0, the performance may be poor (coalesce)
    static_assert(((DYSrcVectorDim == 0 && MThreadSliceSize == DYSrcVectorSize) ||
                   (DYSrcVectorDim == 1 && KThreadSliceSize == DYSrcVectorSize)),
                  "Invalid thread slice sizes and/or dy vector sizes configuration, please check!");

    static_assert(((XSrcVectorDim == 0 && MThreadSliceSize == XSrcVectorSize) ||
                   (XSrcVectorDim == 1 && KThreadSliceSize == XSrcVectorSize)),
                  "Invalid thread slice sizes and/or x vector sizes configuration, please check!");

    // do not force SliceSize == MeanInvStdSrcVectorSize for groupnorm
    static_assert(
        ((MeanInvStdSrcVectorDim == 0 && MThreadSliceSize % MeanInvStdSrcVectorSize == 0) ||
         (MeanInvStdSrcVectorDim == 1 && KThreadSliceSize % MeanInvStdSrcVectorSize == 0)),
        "Invalid thread slice sizes and/or mean/inv_std vector sizes configuration, please check!");

    static_assert(MThreadSliceSize == DGammaDstVectorSize && MThreadSliceSize == DBetaDstVectorSize,
                  "Invalid thread slice sizes and/or dx vector sizes configuration, please check!");

    using ThreadClusterLengths_M_K = Sequence<MThreadClusterSize, KThreadClusterSize>;

    using DYThreadBufferDimAccessOrder =
        typename conditional<DYSrcVectorDim == 0, Sequence<1, 0>, Sequence<0, 1>>::type;

    using XThreadBufferDimAccessOrder =
        typename conditional<XSrcVectorDim == 0, Sequence<1, 0>, Sequence<0, 1>>::type;

    using MeanInvStdThreadBufferDimAccessOrder =
        typename conditional<MeanInvStdSrcVectorDim == 0, Sequence<1, 0>, Sequence<0, 1>>::type;

    using ThreadClusterArrangeOrder = DYThreadBufferDimAccessOrder;

    static constexpr auto thread_cluster_desc =
        make_cluster_descriptor(ThreadClusterLengths_M_K{}, ThreadClusterArrangeOrder{});

    using ThreadBufferLengths_M_K = Sequence<MThreadSliceSize, KThreadSliceSize>;
    using ThreadBufferLengths_M   = Sequence<MThreadSliceSize>;

    static constexpr auto thread_buffer_desc_m_k = make_naive_tensor_descriptor_packed(
        make_tuple(Number<MThreadSliceSize>{}, Number<KThreadSliceSize>{}));

    static constexpr auto thread_buffer_desc_m =
        make_naive_tensor_descriptor_packed(make_tuple(Number<MThreadSliceSize>{}));

    using PassThroughOp = tensor_operation::element_wise::PassThrough;

    using BlockwiseSumReduce = PartitionedBlockwiseReduction<ComputeDataType,
                                                             BlockSize,
                                                             ThreadClusterLengths_M_K,
                                                             ThreadClusterArrangeOrder,
                                                             reduce::Add,
                                                             true>;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    static constexpr index_t M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    static constexpr index_t K_BlockTileSize = KThreadClusterSize * KThreadSliceSize;

    __device__ static void Run(const GridDesc_M_K& dy_grid_desc_m_k,
                               const GridDesc_M_K& x_grid_desc_m_k,
                               const GridDesc_M_K& mean_grid_desc_m_k,
                               const GridDesc_M_K& inv_std_grid_desc_m_k,
                               const GridDesc_M& dgamma_grid_desc_m,
                               const GridDesc_M& dbeta_grid_desc_m,
                               index_t num_k_block_tile_iteration,
                               const DYDataType* const __restrict__ p_dy_global,
                               const XDataType* const __restrict__ p_x_global,
                               const MeanInvStdDataType* const __restrict__ p_mean_global,
                               const MeanInvStdDataType* const __restrict__ p_inv_std_global,
                               DGammaDataType* const __restrict__ p_dgamma_global,
                               DBetaDataType* const __restrict__ p_dbeta_global)
    {
        // LDS
        __shared__ ComputeDataType p_reduce_work_buffer[BlockSize];

        auto reduce_work_buf =
            make_dynamic_buffer<AddressSpaceEnum::Lds>(p_reduce_work_buffer, BlockSize);

        // Global
        const auto dy_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_dy_global, dy_grid_desc_m_k.GetElementSpaceSize());

        const auto x_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_x_global, x_grid_desc_m_k.GetElementSpaceSize());

        const auto mean_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_mean_global, mean_grid_desc_m_k.GetElementSpaceSize());

        const auto inv_std_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_inv_std_global, inv_std_grid_desc_m_k.GetElementSpaceSize());

        auto dgamma_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_dgamma_global, dgamma_grid_desc_m.GetElementSpaceSize());

        auto dbeta_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_dbeta_global, dbeta_grid_desc_m.GetElementSpaceSize());

        // VGPR
        auto dy_thread_buf = StaticBuffer<AddressSpaceEnum::Vgpr,
                                          ComputeDataType,
                                          MThreadSliceSize * KThreadSliceSize,
                                          true>{};

        auto x_thread_buf = StaticBuffer<AddressSpaceEnum::Vgpr,
                                         ComputeDataType,
                                         MThreadSliceSize * KThreadSliceSize,
                                         true>{};

        auto mean_thread_buf = StaticBuffer<AddressSpaceEnum::Vgpr,
                                            ComputeDataType,
                                            MThreadSliceSize * KThreadSliceSize,
                                            true>{};

        auto inv_std_thread_buf = StaticBuffer<AddressSpaceEnum::Vgpr,
                                               ComputeDataType,
                                               MThreadSliceSize * KThreadSliceSize,
                                               true>{};

        auto dgamma_thread_buf =
            StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, MThreadSliceSize, true>{};

        auto dbeta_thread_buf =
            StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, MThreadSliceSize, true>{};

        const index_t thread_local_id = get_thread_local_1d_id();
        const index_t block_global_id = get_block_1d_id();

        const auto thread_cluster_idx =
            thread_cluster_desc.CalculateBottomIndex(make_multi_index(thread_local_id));

        const auto thread_m_cluster_id = thread_cluster_idx[I0];
        const auto thread_k_cluster_id = thread_cluster_idx[I1];

        // IO
        auto threadwise_dy_load = ThreadwiseTensorSliceTransfer_v2<DYDataType,
                                                                   ComputeDataType,
                                                                   GridDesc_M_K,
                                                                   decltype(thread_buffer_desc_m_k),
                                                                   ThreadBufferLengths_M_K,
                                                                   DYThreadBufferDimAccessOrder,
                                                                   DYSrcVectorDim,
                                                                   DYSrcVectorSize,
                                                                   1,
                                                                   true>(
            dy_grid_desc_m_k,
            make_multi_index(block_global_id * M_BlockTileSize +
                                 thread_m_cluster_id * MThreadSliceSize,
                             thread_k_cluster_id * KThreadSliceSize));

        auto threadwise_x_load = ThreadwiseTensorSliceTransfer_v2<XDataType,
                                                                  ComputeDataType,
                                                                  GridDesc_M_K,
                                                                  decltype(thread_buffer_desc_m_k),
                                                                  ThreadBufferLengths_M_K,
                                                                  XThreadBufferDimAccessOrder,
                                                                  XSrcVectorDim,
                                                                  XSrcVectorSize,
                                                                  1,
                                                                  true>(
            x_grid_desc_m_k,
            make_multi_index(block_global_id * M_BlockTileSize +
                                 thread_m_cluster_id * MThreadSliceSize,
                             thread_k_cluster_id * KThreadSliceSize));

        auto threadwise_mean_load =
            ThreadwiseTensorSliceTransfer_v2<MeanInvStdDataType,
                                             ComputeDataType,
                                             GridDesc_M_K,
                                             decltype(thread_buffer_desc_m_k),
                                             ThreadBufferLengths_M_K,
                                             MeanInvStdThreadBufferDimAccessOrder,
                                             MeanInvStdSrcVectorDim,
                                             MeanInvStdSrcVectorSize,
                                             1,
                                             true>(
                mean_grid_desc_m_k,
                make_multi_index(block_global_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 thread_k_cluster_id * KThreadSliceSize));

        auto threadwise_inv_std_load =
            ThreadwiseTensorSliceTransfer_v2<MeanInvStdDataType,
                                             ComputeDataType,
                                             GridDesc_M_K,
                                             decltype(thread_buffer_desc_m_k),
                                             ThreadBufferLengths_M_K,
                                             MeanInvStdThreadBufferDimAccessOrder,
                                             MeanInvStdSrcVectorDim,
                                             MeanInvStdSrcVectorSize,
                                             1,
                                             true>(
                inv_std_grid_desc_m_k,
                make_multi_index(block_global_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 thread_k_cluster_id * KThreadSliceSize));

        auto threadwise_dgamma_store =
            ThreadwiseTensorSliceTransfer_v1r3<ComputeDataType,
                                               DGammaDataType,
                                               decltype(thread_buffer_desc_m),
                                               GridDesc_M,
                                               PassThroughOp,
                                               ThreadBufferLengths_M,
                                               Sequence<0>,
                                               0,
                                               DGammaDstVectorSize,
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               true>(
                dgamma_grid_desc_m,
                make_multi_index(block_global_id * M_BlockTileSize +
                                 thread_m_cluster_id * MThreadSliceSize),
                PassThroughOp{});

        auto threadwise_dbeta_store =
            ThreadwiseTensorSliceTransfer_v1r3<ComputeDataType,
                                               DBetaDataType,
                                               decltype(thread_buffer_desc_m),
                                               GridDesc_M,
                                               PassThroughOp,
                                               ThreadBufferLengths_M,
                                               Sequence<0>,
                                               0,
                                               DBetaDstVectorSize,
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               true>(
                dbeta_grid_desc_m,
                make_multi_index(block_global_id * M_BlockTileSize +
                                 thread_m_cluster_id * MThreadSliceSize),
                PassThroughOp{});

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            dgamma_thread_buf(I) = type_convert<ComputeDataType>(0.0f);
            dbeta_thread_buf(I)  = type_convert<ComputeDataType>(0.0f);
        });

        constexpr auto thread_copy_fwd_step_m_k = make_multi_index(0, K_BlockTileSize);

        for(index_t reducedTiles = 0; reducedTiles < num_k_block_tile_iteration; ++reducedTiles)
        {
            threadwise_dy_load.Run(dy_grid_desc_m_k,
                                   dy_global_val_buf,
                                   thread_buffer_desc_m_k,
                                   make_tuple(I0, I0),
                                   dy_thread_buf);

            threadwise_x_load.Run(x_grid_desc_m_k,
                                  x_global_val_buf,
                                  thread_buffer_desc_m_k,
                                  make_tuple(I0, I0),
                                  x_thread_buf);

            threadwise_mean_load.Run(mean_grid_desc_m_k,
                                     mean_global_val_buf,
                                     thread_buffer_desc_m_k,
                                     make_tuple(I0, I0),
                                     mean_thread_buf);

            threadwise_inv_std_load.Run(inv_std_grid_desc_m_k,
                                        inv_std_global_val_buf,
                                        thread_buffer_desc_m_k,
                                        make_tuple(I0, I0),
                                        inv_std_thread_buf);

            threadwise_dy_load.MoveSrcSliceWindow(dy_grid_desc_m_k, thread_copy_fwd_step_m_k);
            threadwise_x_load.MoveSrcSliceWindow(x_grid_desc_m_k, thread_copy_fwd_step_m_k);
            threadwise_mean_load.MoveSrcSliceWindow(mean_grid_desc_m_k, thread_copy_fwd_step_m_k);
            threadwise_inv_std_load.MoveSrcSliceWindow(inv_std_grid_desc_m_k,
                                                       thread_copy_fwd_step_m_k);

            static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                constexpr auto offset_m =
                    Number<thread_buffer_desc_m.CalculateOffset(make_tuple(iM))>{};

                static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {
                    constexpr auto offset_m_k =
                        Number<thread_buffer_desc_m_k.CalculateOffset(make_tuple(iM, iK))>{};

                    dgamma_thread_buf(offset_m) +=
                        dy_thread_buf[offset_m_k] * inv_std_thread_buf[offset_m_k] *
                        (x_thread_buf[offset_m_k] - mean_thread_buf[offset_m_k]);

                    dbeta_thread_buf(offset_m) += dy_thread_buf[offset_m_k];
                });
            });
        }

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            if constexpr(I > 0)
                block_sync_lds();

            BlockwiseSumReduce::Reduce(reduce_work_buf, dbeta_thread_buf(I));
            block_sync_lds();
            BlockwiseSumReduce::Reduce(reduce_work_buf, dgamma_thread_buf(I));
        });

        if(thread_k_cluster_id == 0)
        {
            threadwise_dgamma_store.Run(thread_buffer_desc_m,
                                        make_tuple(I0),
                                        dgamma_thread_buf,
                                        dgamma_grid_desc_m,
                                        dgamma_global_val_buf);

            threadwise_dbeta_store.Run(thread_buffer_desc_m,
                                       make_tuple(I0),
                                       dbeta_thread_buf,
                                       dbeta_grid_desc_m,
                                       dbeta_global_val_buf);
        }
    }
};

} // namespace ck
