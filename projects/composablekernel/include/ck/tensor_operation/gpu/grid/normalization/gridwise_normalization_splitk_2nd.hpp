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

template <typename MeanVarDataType,
          typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename YDataType,
          typename SaveMeanInvStdDataType,
          typename ComputeDataType,
          typename YElementwiseOperation,
          typename MeanVarGridDesc_M_KBlock,
          typename CountGridDesc_M_KBlock,
          typename XYGammaBetaGridDesc_M_K,
          typename SaveMeanInvStdGridDesc_M,
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
          index_t SaveMeanInvStdDstVectorSize>
struct GridwiseNormalizationSplitK2nd
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

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

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

    using ThreadBufferLengths_M_1 = Sequence<MThreadSliceSize, 1>;
    static constexpr auto thread_buffer_desc_m_1 =
        make_naive_tensor_descriptor_packed(make_tuple(Number<MThreadSliceSize>{}, I1));

    using ThreadWelfordSrcDesc_M_1 = decltype(thread_buffer_desc_m_1);
    using ThreadWelfordDstDesc_M =
        decltype(make_naive_tensor_descriptor_packed(make_tuple(Number<MThreadSliceSize>{})));

    using ThreadwiseWelford =
        ThreadwiseWelfordMerge<ComputeDataType, ThreadWelfordSrcDesc_M_1, ThreadWelfordDstDesc_M>;

    using BlockwiseWelford = BlockwiseWelford<ComputeDataType,
                                              BlockSize,
                                              ThreadClusterLengths_M_K,
                                              ThreadClusterArrangeOrder>;

    using PassThroughOp = tensor_operation::element_wise::PassThrough;

    static constexpr index_t M_BlockTileSize     = MThreadClusterSize * MThreadSliceSize;
    static constexpr index_t K_BlockTileSize     = KThreadClusterSize * KThreadSliceSize;
    static constexpr index_t K_BlockTileStepSize = KThreadClusterSize * XSrcVectorSize;

    static constexpr auto ThreadBufferNumber = Number<KThreadSliceSize / XSrcVectorSize>{};

    __device__ static void Run(const MeanVarGridDesc_M_KBlock& mean_var_grid_desc_m_kblock,
                               const CountGridDesc_M_KBlock& count_grid_desc_m_kblock,
                               const XYGammaBetaGridDesc_M_K& x_grid_desc_m_k,
                               const XYGammaBetaGridDesc_M_K& gamma_grid_desc_m_k,
                               const XYGammaBetaGridDesc_M_K& beta_grid_desc_m_k,
                               const XYGammaBetaGridDesc_M_K& y_grid_desc_m_k,
                               const SaveMeanInvStdGridDesc_M& save_mean_grid_desc_m,
                               const SaveMeanInvStdGridDesc_M& save_inv_std_grid_desc_m,
                               index_t num_k_mean_var_count_iteration,
                               index_t num_k_block_tile_iteration,
                               index_t k_grid_size,
                               ComputeDataType epsilon,
                               const MeanVarDataType* const p_mean_global,
                               const MeanVarDataType* const p_variance_global,
                               const int32_t* const p_welford_count_global,
                               const XDataType* const __restrict__ p_x_global,
                               const GammaDataType* const __restrict__ p_gamma_global,
                               const BetaDataType* const __restrict__ p_beta_global,
                               YDataType* const __restrict__ p_y_global,
                               SaveMeanInvStdDataType* const __restrict__ p_save_mean_global,
                               SaveMeanInvStdDataType* const __restrict__ p_save_inv_std_global,
                               const YElementwiseOperation y_elementwise_op)
    {
        // Thread/Block id
        const index_t thread_local_id    = get_thread_local_1d_id();
        const index_t block_global_id    = get_block_1d_id();
        const index_t block_m_cluster_id = block_global_id / k_grid_size;
        const index_t block_k_cluster_id = block_global_id % k_grid_size;
        const auto thread_cluster_idx =
            thread_cluster_desc.CalculateBottomIndex(make_multi_index(thread_local_id));

        const auto thread_m_cluster_id = thread_cluster_idx[I0];
        const auto thread_k_cluster_id = thread_cluster_idx[I1];

        // Global Memory
        const auto mean_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_mean_global, mean_var_grid_desc_m_kblock.GetElementSpaceSize());

        const auto var_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_variance_global, mean_var_grid_desc_m_kblock.GetElementSpaceSize());

        const auto welford_count_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_welford_count_global, count_grid_desc_m_kblock.GetElementSpaceSize());

        const auto x_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_x_global, x_grid_desc_m_k.GetElementSpaceSize());

        const auto gamma_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_gamma_global, gamma_grid_desc_m_k.GetElementSpaceSize());

        const auto beta_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_beta_global, beta_grid_desc_m_k.GetElementSpaceSize());

        auto y_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_y_global, y_grid_desc_m_k.GetElementSpaceSize());

        auto save_mean_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_save_mean_global, save_mean_grid_desc_m.GetElementSpaceSize());

        auto save_inv_std_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_save_inv_std_global, save_inv_std_grid_desc_m.GetElementSpaceSize());

        // VGPR
        StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, MThreadSliceSize, true>
            in_mean_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, MThreadSliceSize, true>
            in_var_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, int32_t, MThreadSliceSize, true>
            in_welford_count_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, MThreadSliceSize, true>
            mean_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, MThreadSliceSize, true>
            var_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, int32_t, MThreadSliceSize, true>
            welford_count_thread_buf;
        auto& inv_std_thread_buf = var_thread_buf;

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
        auto& y_thread_buf    = x_thread_buf;

        // IO
        auto threadwise_mean_var_load_m_kblock =
            ThreadwiseTensorSliceTransfer_v2<MeanVarDataType,
                                             ComputeDataType,
                                             MeanVarGridDesc_M_KBlock,
                                             decltype(thread_buffer_desc_m_1),
                                             ThreadBufferLengths_M_1,
                                             Sequence<0, 1>,
                                             1,
                                             1,
                                             1,
                                             true>(
                mean_var_grid_desc_m_kblock,
                make_multi_index(block_m_cluster_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 thread_k_cluster_id));

        auto threadwise_count_load_m_kblock =
            ThreadwiseTensorSliceTransfer_v2<int32_t,
                                             int32_t,
                                             CountGridDesc_M_KBlock,
                                             decltype(thread_buffer_desc_m_1),
                                             ThreadBufferLengths_M_1,
                                             Sequence<0, 1>,
                                             1,
                                             1,
                                             1,
                                             true>(
                count_grid_desc_m_kblock,
                make_multi_index(block_m_cluster_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 thread_k_cluster_id));

        auto threadwise_x_load = ThreadwiseTensorSliceTransfer_v2<XDataType,
                                                                  ComputeDataType,
                                                                  XYGammaBetaGridDesc_M_K,
                                                                  decltype(thread_buffer_desc_m_k),
                                                                  ThreadBufferLengths_M_K,
                                                                  ThreadBufferDimAccessOrder,
                                                                  XSrcVectorDim,
                                                                  XSrcVectorSize,
                                                                  1,
                                                                  true>(
            x_grid_desc_m_k,
            make_multi_index(block_m_cluster_id * M_BlockTileSize +
                                 thread_m_cluster_id * MThreadSliceSize,
                             block_k_cluster_id * K_BlockTileSize * num_k_block_tile_iteration +
                                 thread_k_cluster_id * XSrcVectorSize));

        auto threadwise_gamma_load =
            ThreadwiseTensorSliceTransfer_v2<GammaDataType,
                                             ComputeDataType,
                                             XYGammaBetaGridDesc_M_K,
                                             decltype(thread_buffer_desc_m_k),
                                             ThreadBufferLengths_M_K,
                                             ThreadBufferDimAccessOrder,
                                             GammaSrcVectorDim,
                                             GammaSrcVectorSize,
                                             1,
                                             true>(
                gamma_grid_desc_m_k,
                make_multi_index(block_m_cluster_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 block_k_cluster_id * K_BlockTileSize * num_k_block_tile_iteration +
                                     thread_k_cluster_id * GammaSrcVectorSize));

        auto threadwise_beta_load =
            ThreadwiseTensorSliceTransfer_v2<BetaDataType,
                                             ComputeDataType,
                                             XYGammaBetaGridDesc_M_K,
                                             decltype(thread_buffer_desc_m_k),
                                             ThreadBufferLengths_M_K,
                                             ThreadBufferDimAccessOrder,
                                             BetaSrcVectorDim,
                                             BetaSrcVectorSize,
                                             1,
                                             true>(
                beta_grid_desc_m_k,
                make_multi_index(block_m_cluster_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 block_k_cluster_id * K_BlockTileSize * num_k_block_tile_iteration +
                                     thread_k_cluster_id * BetaSrcVectorSize));

        auto threadwise_y_store =
            ThreadwiseTensorSliceTransfer_v1r3<ComputeDataType,
                                               YDataType,
                                               decltype(thread_buffer_desc_m_k),
                                               XYGammaBetaGridDesc_M_K,
                                               YElementwiseOperation,
                                               ThreadBufferLengths_M_K,
                                               ThreadBufferDimAccessOrder,
                                               YDstVectorDim,
                                               YDstVectorSize,
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               true>(
                y_grid_desc_m_k,
                make_multi_index(block_m_cluster_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 block_k_cluster_id * K_BlockTileSize * num_k_block_tile_iteration +
                                     thread_k_cluster_id * YDstVectorSize),
                y_elementwise_op);

        auto threadwise_mean_store =
            ThreadwiseTensorSliceTransfer_v1r3<ComputeDataType,
                                               SaveMeanInvStdDataType,
                                               decltype(thread_buffer_desc_m),
                                               SaveMeanInvStdGridDesc_M,
                                               PassThroughOp,
                                               ThreadBufferLengths_M,
                                               Sequence<0>,                 // DimAccessOrder
                                               0,                           // SrcVectorDim
                                               SaveMeanInvStdDstVectorSize, // ScalarPerVector
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               true>(
                save_mean_grid_desc_m,
                make_multi_index(block_m_cluster_id * M_BlockTileSize +
                                 thread_m_cluster_id * MThreadSliceSize),
                PassThroughOp{});

        auto threadwise_inv_std_store =
            ThreadwiseTensorSliceTransfer_v1r3<ComputeDataType,
                                               SaveMeanInvStdDataType,
                                               decltype(thread_buffer_desc_m),
                                               SaveMeanInvStdGridDesc_M,
                                               PassThroughOp,
                                               ThreadBufferLengths_M,
                                               Sequence<0>,                 // DimAccessOrder
                                               0,                           // SrcVectorDim
                                               SaveMeanInvStdDstVectorSize, // ScalarPerVector
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               true>(
                save_inv_std_grid_desc_m,
                make_multi_index(block_m_cluster_id * M_BlockTileSize +
                                 thread_m_cluster_id * MThreadSliceSize),
                PassThroughOp{});

        // step1: Merge mean and variance
        constexpr auto mean_var_count_thread_copy_step_I0_k =
            make_multi_index(I0, KThreadClusterSize);

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            mean_thread_buf(I)          = type_convert<ComputeDataType>(0.0f);
            var_thread_buf(I)           = type_convert<ComputeDataType>(0.0f);
            welford_count_thread_buf(I) = 0;
        });

        for(index_t k = 0; k < num_k_mean_var_count_iteration; ++k)
        {
            threadwise_mean_var_load_m_kblock.Run(mean_var_grid_desc_m_kblock,
                                                  mean_global_val_buf,
                                                  thread_buffer_desc_m_1,
                                                  make_tuple(I0, I0),
                                                  in_mean_thread_buf);

            threadwise_mean_var_load_m_kblock.Run(mean_var_grid_desc_m_kblock,
                                                  var_global_val_buf,
                                                  thread_buffer_desc_m_1,
                                                  make_tuple(I0, I0),
                                                  in_var_thread_buf);

            threadwise_count_load_m_kblock.Run(count_grid_desc_m_kblock,
                                               welford_count_global_val_buf,
                                               thread_buffer_desc_m_1,
                                               make_tuple(I0, I0),
                                               in_welford_count_thread_buf);

            ThreadwiseWelford::Run(in_mean_thread_buf,
                                   in_var_thread_buf,
                                   in_welford_count_thread_buf,
                                   mean_thread_buf,
                                   var_thread_buf,
                                   welford_count_thread_buf);

            threadwise_mean_var_load_m_kblock.MoveSrcSliceWindow(
                mean_var_grid_desc_m_kblock, mean_var_count_thread_copy_step_I0_k);
            threadwise_count_load_m_kblock.MoveSrcSliceWindow(count_grid_desc_m_kblock,
                                                              mean_var_count_thread_copy_step_I0_k);
        }

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
            if constexpr(I > 0)
                block_sync_lds();

            BlockwiseWelford::Run(
                mean_thread_buf(I), var_thread_buf(I), welford_count_thread_buf(I));

            inv_std_thread_buf(I) =
                type_convert<ComputeDataType>(1.0f) / ck::math::sqrt(var_thread_buf(I) + epsilon);
        });

        // step2: save mean and inverse std for backward (optional)
        if(block_k_cluster_id == 0 && thread_k_cluster_id == 0)
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

        // step3: normalization
        constexpr auto thread_copy_fwd_step_m_k = make_multi_index(0, K_BlockTileStepSize);

        for(index_t k = 0; k < num_k_block_tile_iteration; ++k)
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
                threadwise_y_store.MoveDstSliceWindow(y_grid_desc_m_k, thread_copy_fwd_step_m_k);
            });
        } // end for (normalization)
    }
};

} // namespace ck
