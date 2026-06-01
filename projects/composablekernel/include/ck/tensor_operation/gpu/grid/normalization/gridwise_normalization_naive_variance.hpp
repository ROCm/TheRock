// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/data_type.hpp"

#include "ck/utility/reduction_operator.hpp"
#include "ck/tensor_operation/gpu/block/reduction_functions_blockwise.hpp"
#include "ck/tensor_operation/gpu/thread/reduction_functions_threadwise.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

// Y = Normalization(X, Beta, Gamma)
template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename YDataType,
          typename SaveMeanInvStdDataType,
          typename ComputeDataType,
          typename YElementwiseOperation,
          typename GridDesc_M_K,
          typename GridDesc_M,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t XSrcVectorDim,
          index_t XSrcVectorSize,
          index_t GammaSrcVectorDim,
          index_t GammaSrcVectorSize,
          index_t BetaSrcVectorDim,
          index_t BetaSrcVectorSize,
          index_t YDstVectorDim,
          index_t YDstVectorSize,
          index_t SaveMeanInvStdDstVectorSize,
          bool SweepOnce>
struct GridwiseNormalizationNaiveVariance_mk_to_mk
{
    static_assert((XSrcVectorDim == 0 && MThreadSliceSize % XSrcVectorSize == 0) ||
                      (XSrcVectorDim == 1 && KThreadSliceSize % XSrcVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    static_assert((YDstVectorDim == 0 && MThreadSliceSize % YDstVectorSize == 0) ||
                      (YDstVectorDim == 1 && KThreadSliceSize % YDstVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    static_assert(MThreadSliceSize % SaveMeanInvStdDstVectorSize == 0,
                  "Invalid thread slice sizes and/or save mean and inverse std vector sizes "
                  "configuration, please check!");

    static_assert(XSrcVectorSize == YDstVectorSize);
    static_assert(XSrcVectorSize == GammaSrcVectorSize);
    static_assert(XSrcVectorSize == BetaSrcVectorSize);

    static constexpr bool reorder_thread_cluster = (XSrcVectorDim == 0);

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

    using ThreadBufferLengths_M = Sequence<MThreadSliceSize>;
    static constexpr auto thread_buffer_desc_m =
        make_naive_tensor_descriptor_packed(make_tuple(Number<MThreadSliceSize>{}));

    using ThreadReduceSrcDesc_M_K = decltype(make_naive_tensor_descriptor_packed(
        make_tuple(Number<MThreadSliceSize>{}, Number<XSrcVectorSize>{})));
    using ThreadReduceDstDesc_M =
        decltype(make_naive_tensor_descriptor_packed(make_tuple(Number<MThreadSliceSize>{})));

    using BlockwiseSumReduce = PartitionedBlockwiseReduction<ComputeDataType,
                                                             BlockSize,
                                                             ThreadClusterLengths_M_K,
                                                             ThreadClusterArrangeOrder,
                                                             reduce::Add,
                                                             true>;

    using ThreadwiseSumReduce = ThreadwiseReduction<ComputeDataType,
                                                    ThreadReduceSrcDesc_M_K,
                                                    ThreadReduceDstDesc_M,
                                                    reduce::Add,
                                                    true>;

    using PassThroughOp = tensor_operation::element_wise::PassThrough;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static constexpr index_t M_BlockTileSize     = MThreadClusterSize * MThreadSliceSize;
    static constexpr index_t K_BlockTileSize     = KThreadClusterSize * KThreadSliceSize;
    static constexpr index_t K_BlockTileStepSize = KThreadClusterSize * XSrcVectorSize;

    static constexpr auto ThreadBufferNumber = Number<KThreadSliceSize / XSrcVectorSize>{};

    __device__ static void Run(const GridDesc_M_K& x_grid_desc_m_k,
                               const GridDesc_M_K& gamma_grid_desc_m_k,
                               const GridDesc_M_K& beta_grid_desc_m_k,
                               const GridDesc_M_K& y_grid_desc_m_k,
                               const GridDesc_M& save_mean_grid_desc_m,
                               const GridDesc_M& save_inv_std_grid_desc_m,
                               index_t num_k_block_tile_iteration,
                               ComputeDataType epsilon,
                               const XDataType* const __restrict__ p_x_global,
                               const GammaDataType* const __restrict__ p_gamma_global,
                               const BetaDataType* const __restrict__ p_beta_global,
                               YDataType* const __restrict__ p_y_global,
                               SaveMeanInvStdDataType* const __restrict__ p_save_mean_global,
                               SaveMeanInvStdDataType* const __restrict__ p_save_inv_std_global,
                               const YElementwiseOperation y_elementwise_op)
    {
        // LDS
        __shared__ ComputeDataType p_reduce_work_buffer[BlockSize];

        auto reduce_work_buf =
            make_dynamic_buffer<AddressSpaceEnum::Lds>(p_reduce_work_buffer, BlockSize);

        auto y_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_y_global, y_grid_desc_m_k.GetElementSpaceSize());

        auto save_mean_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_save_mean_global, save_mean_grid_desc_m.GetElementSpaceSize());

        auto save_inv_std_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_save_inv_std_global, save_inv_std_grid_desc_m.GetElementSpaceSize());

        auto x_thread_buf = generate_tuple(
            [&](auto) {
                return StaticBuffer<AddressSpaceEnum::Vgpr,
                                    ComputeDataType,
                                    MThreadSliceSize * XSrcVectorSize,
                                    true>{};
            },
            Number<ThreadBufferNumber>{});

        auto gamma_thread_buf = generate_tuple(
            [&](auto) {
                return StaticBuffer<AddressSpaceEnum::Vgpr,
                                    ComputeDataType,
                                    MThreadSliceSize * GammaSrcVectorSize,
                                    true>{};
            },
            Number<ThreadBufferNumber>{});

        auto& beta_thread_buf = gamma_thread_buf;

        auto y_thread_buf = generate_tuple(
            [&](auto) {
                return StaticBuffer<AddressSpaceEnum::Vgpr,
                                    ComputeDataType,
                                    MThreadSliceSize * YDstVectorSize,
                                    true>{};
            },
            Number<ThreadBufferNumber>{});

        auto& x_square_thread_buf = y_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, MThreadSliceSize, true>
            mean_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, MThreadSliceSize, true>
            mean_square_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, MThreadSliceSize, true>&
            var_thread_buf = mean_square_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, MThreadSliceSize, true>&
            inv_std_thread_buf = mean_square_thread_buf;

        const index_t thread_local_id = get_thread_local_1d_id();
        const index_t block_global_id = get_block_1d_id();

        const auto thread_cluster_idx =
            thread_cluster_desc.CalculateBottomIndex(make_multi_index(thread_local_id));

        const auto thread_m_cluster_id = thread_cluster_idx[I0];
        const auto thread_k_cluster_id = thread_cluster_idx[I1];

        auto threadwise_x_load = ThreadwiseTensorSliceTransfer_v2<XDataType,
                                                                  ComputeDataType,
                                                                  GridDesc_M_K,
                                                                  decltype(thread_buffer_desc_m_k),
                                                                  ThreadBufferLengths_M_K,
                                                                  ThreadBufferDimAccessOrder,
                                                                  XSrcVectorDim,
                                                                  XSrcVectorSize,
                                                                  1,
                                                                  true>(
            x_grid_desc_m_k,
            make_multi_index(block_global_id * M_BlockTileSize +
                                 thread_m_cluster_id * MThreadSliceSize,
                             thread_k_cluster_id * XSrcVectorSize));

        auto threadwise_gamma_load =
            ThreadwiseTensorSliceTransfer_v2<GammaDataType,
                                             ComputeDataType,
                                             GridDesc_M_K,
                                             decltype(thread_buffer_desc_m_k),
                                             ThreadBufferLengths_M_K,
                                             ThreadBufferDimAccessOrder,
                                             GammaSrcVectorDim,
                                             GammaSrcVectorSize,
                                             1,
                                             true>(
                gamma_grid_desc_m_k,
                make_multi_index(block_global_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 thread_k_cluster_id * GammaSrcVectorSize));

        auto threadwise_beta_load =
            ThreadwiseTensorSliceTransfer_v2<BetaDataType,
                                             ComputeDataType,
                                             GridDesc_M_K,
                                             decltype(thread_buffer_desc_m_k),
                                             ThreadBufferLengths_M_K,
                                             ThreadBufferDimAccessOrder,
                                             BetaSrcVectorDim,
                                             BetaSrcVectorSize,
                                             1,
                                             true>(
                beta_grid_desc_m_k,
                make_multi_index(block_global_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 thread_k_cluster_id * BetaSrcVectorSize));

        auto threadwise_y_store =
            ThreadwiseTensorSliceTransfer_v1r3<ComputeDataType,
                                               YDataType,
                                               decltype(thread_buffer_desc_m_k),
                                               GridDesc_M_K,
                                               YElementwiseOperation,
                                               ThreadBufferLengths_M_K,
                                               ThreadBufferDimAccessOrder,
                                               YDstVectorDim,
                                               YDstVectorSize,
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               true>(
                y_grid_desc_m_k,
                make_multi_index(block_global_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 thread_k_cluster_id * YDstVectorSize),
                y_elementwise_op);

        auto threadwise_mean_store =
            ThreadwiseTensorSliceTransfer_v1r3<ComputeDataType,
                                               SaveMeanInvStdDataType,
                                               decltype(thread_buffer_desc_m),
                                               GridDesc_M,
                                               PassThroughOp,
                                               ThreadBufferLengths_M,
                                               Sequence<0>,                 // DimAccessOrder
                                               0,                           // SrcVectorDim
                                               SaveMeanInvStdDstVectorSize, // ScalarPerVector
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               true>(
                save_mean_grid_desc_m,
                make_multi_index(block_global_id * M_BlockTileSize +
                                 thread_m_cluster_id * MThreadSliceSize),
                PassThroughOp{});

        auto threadwise_inv_std_store =
            ThreadwiseTensorSliceTransfer_v1r3<ComputeDataType,
                                               SaveMeanInvStdDataType,
                                               decltype(thread_buffer_desc_m),
                                               GridDesc_M,
                                               PassThroughOp,
                                               ThreadBufferLengths_M,
                                               Sequence<0>,                 // DimAccessOrder
                                               0,                           // SrcVectorDim
                                               SaveMeanInvStdDstVectorSize, // ScalarPerVector
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               true>(
                save_inv_std_grid_desc_m,
                make_multi_index(block_global_id * M_BlockTileSize +
                                 thread_m_cluster_id * MThreadSliceSize),
                PassThroughOp{});

        constexpr auto thread_copy_fwd_step_m_k = make_multi_index(0, K_BlockTileStepSize);
        constexpr auto thread_copy_bwd_step_m_k =
            make_multi_index(0, SweepOnce ? 0 : -K_BlockTileSize);

        const auto x_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_x_global, x_grid_desc_m_k.GetElementSpaceSize());

        const auto gamma_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_gamma_global, gamma_grid_desc_m_k.GetElementSpaceSize());

        const auto beta_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_beta_global, beta_grid_desc_m_k.GetElementSpaceSize());

        // E(x), E[x^2], var(x)
        // FIXME: Should not hack the transform from deviceOP
        ComputeDataType reduce_length = type_convert<ComputeDataType>(
            x_grid_desc_m_k.GetTransforms()[I2].GetUpperLengths()[I0]);

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            mean_thread_buf(I)        = reduce::Add::template GetIdentityValue<ComputeDataType>();
            mean_square_thread_buf(I) = reduce::Add::template GetIdentityValue<ComputeDataType>();
        });

        // Separate sweep once and sweep twice pipeline
        if constexpr(SweepOnce)
        {
            static_for<0, ThreadBufferNumber, 1>{}([&](auto i) {
                threadwise_x_load.Run(x_grid_desc_m_k,
                                      x_global_val_buf,
                                      thread_buffer_desc_m_k,
                                      make_tuple(I0, I0),
                                      x_thread_buf(i));

                threadwise_gamma_load.Run(gamma_grid_desc_m_k,
                                          gamma_global_val_buf,
                                          thread_buffer_desc_m_k,
                                          make_tuple(I0, I0),
                                          gamma_thread_buf(i));

                static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                    static_for<0, XSrcVectorSize, 1>{}([&](auto iK) {
                        constexpr auto offset_m_k =
                            thread_buffer_desc_m_k.CalculateOffset(make_tuple(iM, iK));
                        x_square_thread_buf(i)(Number<offset_m_k>{}) =
                            x_thread_buf(i)(Number<offset_m_k>{}) *
                            x_thread_buf(i)(Number<offset_m_k>{});
                    });
                });

                ThreadwiseSumReduce::Reduce(x_thread_buf[i], mean_thread_buf);
                ThreadwiseSumReduce::Reduce(x_square_thread_buf[i], mean_square_thread_buf);

                if constexpr(i != ThreadBufferNumber - 1)
                {
                    threadwise_x_load.MoveSrcSliceWindow(x_grid_desc_m_k, thread_copy_fwd_step_m_k);
                    threadwise_gamma_load.MoveSrcSliceWindow(gamma_grid_desc_m_k,
                                                             thread_copy_fwd_step_m_k);
                }
            });

            static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
                if constexpr(I > 0)
                    block_sync_lds();

                BlockwiseSumReduce::Reduce(reduce_work_buf, mean_thread_buf(I));
                mean_thread_buf(I) = mean_thread_buf(I) / reduce_length;

                block_sync_lds();

                BlockwiseSumReduce::Reduce(reduce_work_buf, mean_square_thread_buf(I));
                mean_square_thread_buf(I) = mean_square_thread_buf(I) / reduce_length;

                // var(x) = E[x^2] - E[x]^2
                var_thread_buf(I) =
                    mean_square_thread_buf(I) - (mean_thread_buf(I) * mean_thread_buf(I));

                inv_std_thread_buf(I) = type_convert<ComputeDataType>(1.0f) /
                                        ck::math::sqrt(var_thread_buf(I) + epsilon);
            });

            // save mean and inverse std for backward (optional)
            if(thread_k_cluster_id == 0)
            {
                if(p_save_mean_global != nullptr)
                {
                    threadwise_mean_store.Run(thread_buffer_desc_m,
                                              make_tuple(I0),
                                              mean_thread_buf,
                                              save_mean_grid_desc_m,
                                              save_mean_global_val_buf);
                }
                if(p_save_inv_std_global != nullptr)
                {
                    threadwise_inv_std_store.Run(thread_buffer_desc_m,
                                                 make_tuple(I0),
                                                 inv_std_thread_buf,
                                                 save_inv_std_grid_desc_m,
                                                 save_inv_std_global_val_buf);
                }
            }

            // normalization
            static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                static_for<0, ThreadBufferNumber, 1>{}([&](auto iK0) {
                    static_for<0, XSrcVectorSize, 1>{}([&](auto iK1) {
                        constexpr auto offset_m_k =
                            thread_buffer_desc_m_k.CalculateOffset(make_tuple(iM, iK1));

                        // normalize
                        y_thread_buf(iK0)(Number<offset_m_k>{}) =
                            (x_thread_buf(iK0)(Number<offset_m_k>{}) - mean_thread_buf(iM)) *
                            inv_std_thread_buf(iM);

                        // gamma & beta
                        y_thread_buf(iK0)(Number<offset_m_k>{}) =
                            y_thread_buf(iK0)(Number<offset_m_k>{}) *
                            gamma_thread_buf(iK0)(Number<offset_m_k>{});
                    });
                });
            });

            static_for<0, ThreadBufferNumber, 1>{}([&](auto i) {
                threadwise_beta_load.Run(beta_grid_desc_m_k,
                                         beta_global_val_buf,
                                         thread_buffer_desc_m_k,
                                         make_tuple(I0, I0),
                                         beta_thread_buf(i));

                if constexpr(i != ThreadBufferNumber - 1)
                    threadwise_beta_load.MoveSrcSliceWindow(beta_grid_desc_m_k,
                                                            thread_copy_fwd_step_m_k);
            });

            static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                static_for<0, ThreadBufferNumber, 1>{}([&](auto iK0) {
                    static_for<0, XSrcVectorSize, 1>{}([&](auto iK1) {
                        constexpr auto offset_m_k =
                            thread_buffer_desc_m_k.CalculateOffset(make_tuple(iM, iK1));

                        // beta
                        y_thread_buf(iK0)(Number<offset_m_k>{}) =
                            y_thread_buf(iK0)(Number<offset_m_k>{}) +
                            beta_thread_buf(iK0)(Number<offset_m_k>{});
                    });
                });
            });

            static_for<0, ThreadBufferNumber, 1>{}([&](auto i) {
                threadwise_y_store.Run(thread_buffer_desc_m_k,
                                       make_tuple(I0, I0),
                                       y_thread_buf(i),
                                       y_grid_desc_m_k,
                                       y_global_val_buf);

                if constexpr(i != ThreadBufferNumber - 1)
                    threadwise_y_store.MoveDstSliceWindow(y_grid_desc_m_k,
                                                          thread_copy_fwd_step_m_k);
            });
        } // end of sweep once
        else
        {
            for(index_t reducedTiles = 0; reducedTiles < num_k_block_tile_iteration; ++reducedTiles)
            {
                static_for<0, ThreadBufferNumber, 1>{}([&](auto i) {
                    threadwise_x_load.Run(x_grid_desc_m_k,
                                          x_global_val_buf,
                                          thread_buffer_desc_m_k,
                                          make_tuple(I0, I0),
                                          x_thread_buf(i));
                    threadwise_x_load.MoveSrcSliceWindow(x_grid_desc_m_k, thread_copy_fwd_step_m_k);

                    static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                        static_for<0, XSrcVectorSize, 1>{}([&](auto iK) {
                            constexpr auto offset_m_k =
                                thread_buffer_desc_m_k.CalculateOffset(make_tuple(iM, iK));
                            x_square_thread_buf(i)(Number<offset_m_k>{}) =
                                x_thread_buf(i)(Number<offset_m_k>{}) *
                                x_thread_buf(i)(Number<offset_m_k>{});
                        });
                    });

                    ThreadwiseSumReduce::Reduce(x_thread_buf[i], mean_thread_buf);
                    ThreadwiseSumReduce::Reduce(x_square_thread_buf[i], mean_square_thread_buf);
                });
            }

            static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
                if constexpr(I > 0)
                    block_sync_lds();

                BlockwiseSumReduce::Reduce(reduce_work_buf, mean_thread_buf(I));
                mean_thread_buf(I) = mean_thread_buf(I) / reduce_length;

                block_sync_lds();

                BlockwiseSumReduce::Reduce(reduce_work_buf, mean_square_thread_buf(I));
                mean_square_thread_buf(I) = mean_square_thread_buf(I) / reduce_length;

                // var(x) = E[x^2] - E[x]^2
                var_thread_buf(I) =
                    mean_square_thread_buf(I) - (mean_thread_buf(I) * mean_thread_buf(I));

                inv_std_thread_buf(I) = 1 / ck::math::sqrt(var_thread_buf(I) + epsilon);
            });

            if(thread_k_cluster_id == 0)
            {
                if(p_save_mean_global != nullptr)
                {
                    threadwise_mean_store.Run(thread_buffer_desc_m,
                                              make_tuple(I0),
                                              mean_thread_buf,
                                              save_mean_grid_desc_m,
                                              save_mean_global_val_buf);
                }
                if(p_save_inv_std_global != nullptr)
                {
                    threadwise_inv_std_store.Run(thread_buffer_desc_m,
                                                 make_tuple(I0),
                                                 inv_std_thread_buf,
                                                 save_inv_std_grid_desc_m,
                                                 save_inv_std_global_val_buf);
                }
            }

            auto thread_copy_tail_m_k =
                (num_k_block_tile_iteration - 1) * ThreadBufferNumber * thread_copy_fwd_step_m_k;

            threadwise_x_load.MoveSrcSliceWindow(x_grid_desc_m_k, thread_copy_bwd_step_m_k);
            threadwise_gamma_load.MoveSrcSliceWindow(gamma_grid_desc_m_k, thread_copy_tail_m_k);
            threadwise_beta_load.MoveSrcSliceWindow(beta_grid_desc_m_k, thread_copy_tail_m_k);
            threadwise_y_store.MoveDstSliceWindow(y_grid_desc_m_k, thread_copy_tail_m_k);

            for(index_t reducedTiles = 0; reducedTiles < num_k_block_tile_iteration; ++reducedTiles)
            {
                static_for<0, ThreadBufferNumber, 1>{}([&](auto i) {
                    threadwise_x_load.Run(x_grid_desc_m_k,
                                          x_global_val_buf,
                                          thread_buffer_desc_m_k,
                                          make_tuple(I0, I0),
                                          x_thread_buf(i));
                    threadwise_x_load.MoveSrcSliceWindow(x_grid_desc_m_k, thread_copy_fwd_step_m_k);
                });

                static_for<0, ThreadBufferNumber, 1>{}([&](auto i) {
                    threadwise_gamma_load.Run(gamma_grid_desc_m_k,
                                              gamma_global_val_buf,
                                              thread_buffer_desc_m_k,
                                              make_tuple(I0, I0),
                                              gamma_thread_buf(i));

                    threadwise_gamma_load.MoveSrcSliceWindow(gamma_grid_desc_m_k,
                                                             thread_copy_fwd_step_m_k);
                });

                static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                    static_for<0, ThreadBufferNumber, 1>{}([&](auto iK0) {
                        static_for<0, XSrcVectorSize, 1>{}([&](auto iK1) {
                            constexpr auto offset_m_k =
                                thread_buffer_desc_m_k.CalculateOffset(make_tuple(iM, iK1));

                            // normalize
                            y_thread_buf(iK0)(Number<offset_m_k>{}) =
                                (x_thread_buf(iK0)(Number<offset_m_k>{}) - mean_thread_buf(iM)) *
                                inv_std_thread_buf(iM);

                            // gamma
                            y_thread_buf(iK0)(Number<offset_m_k>{}) =
                                y_thread_buf(iK0)(Number<offset_m_k>{}) *
                                gamma_thread_buf(iK0)(Number<offset_m_k>{});
                        });
                    });
                });

                static_for<0, ThreadBufferNumber, 1>{}([&](auto i) {
                    threadwise_beta_load.Run(beta_grid_desc_m_k,
                                             beta_global_val_buf,
                                             thread_buffer_desc_m_k,
                                             make_tuple(I0, I0),
                                             beta_thread_buf(i));
                    threadwise_beta_load.MoveSrcSliceWindow(beta_grid_desc_m_k,
                                                            thread_copy_fwd_step_m_k);
                });

                static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                    static_for<0, ThreadBufferNumber, 1>{}([&](auto iK0) {
                        static_for<0, XSrcVectorSize, 1>{}([&](auto iK1) {
                            constexpr auto offset_m_k =
                                thread_buffer_desc_m_k.CalculateOffset(make_tuple(iM, iK1));

                            // beta
                            y_thread_buf(iK0)(Number<offset_m_k>{}) =
                                y_thread_buf(iK0)(Number<offset_m_k>{}) +
                                beta_thread_buf(iK0)(Number<offset_m_k>{});
                        });
                    });
                });

                static_for<0, ThreadBufferNumber, 1>{}([&](auto i) {
                    threadwise_y_store.Run(thread_buffer_desc_m_k,
                                           make_tuple(I0, I0),
                                           y_thread_buf(i),
                                           y_grid_desc_m_k,
                                           y_global_val_buf);
                    threadwise_y_store.MoveDstSliceWindow(y_grid_desc_m_k,
                                                          thread_copy_fwd_step_m_k);
                });

                threadwise_x_load.MoveSrcSliceWindow(x_grid_desc_m_k, 2 * thread_copy_bwd_step_m_k);
                threadwise_gamma_load.MoveSrcSliceWindow(gamma_grid_desc_m_k,
                                                         2 * thread_copy_bwd_step_m_k);
                threadwise_beta_load.MoveSrcSliceWindow(beta_grid_desc_m_k,
                                                        2 * thread_copy_bwd_step_m_k);
                threadwise_y_store.MoveDstSliceWindow(y_grid_desc_m_k,
                                                      2 * thread_copy_bwd_step_m_k);
            }
        } // end of sweep twice
    }
};

} // namespace ck
