// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7r2.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r2.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor/static_tensor.hpp"
#include "ck/utility/common_header.hpp"

namespace ck {

template <typename GridwiseElementwiseFunctor,
          typename InGridDescTuple,
          typename OutGridDescTuple,
          typename InDataTypePointerTuple,
          typename OutDataTypePointerTuple,
          typename Block2TileMap,
          typename ElementwiseOperation>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_elementwise(const InGridDescTuple in_grid_desc_tuple,
                       const OutGridDescTuple out_grid_desc_tuple,
                       const InDataTypePointerTuple p_in_global_tuple,
                       const OutDataTypePointerTuple p_out_global_tuple,
                       const Block2TileMap block_2_tile_map,
                       const ElementwiseOperation elementwise_op)
{
    GridwiseElementwiseFunctor::Run(in_grid_desc_tuple,
                                    out_grid_desc_tuple,
                                    p_in_global_tuple,
                                    p_out_global_tuple,
                                    block_2_tile_map,
                                    elementwise_op);
}

template <typename GridwiseElementwiseFunctorA,
          typename GridwiseElementwiseFunctorB,
          typename InAGridDescTuple,
          typename InBGridDescTuple,
          typename OutAGridDescTuple,
          typename OutBGridDescTuple,
          typename InADataTypePointerTuple,
          typename InBDataTypePointerTuple,
          typename OutADataTypePointerTuple,
          typename OutBDataTypePointerTuple,
          typename Block2TileMapA,
          typename Block2TileMapB,
          typename ElementwiseOperation>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_elementwise_dual(const InAGridDescTuple in_grid_desc_tuple_a,
                            const InBGridDescTuple in_grid_desc_tuple_b,
                            const OutAGridDescTuple out_grid_desc_tuple_a,
                            const OutBGridDescTuple out_grid_desc_tuple_b,
                            const InADataTypePointerTuple p_in_global_tuple_a,
                            const InBDataTypePointerTuple p_in_global_tuple_b,
                            const OutADataTypePointerTuple p_out_global_tuple_a,
                            const OutBDataTypePointerTuple p_out_global_tuple_b,
                            const Block2TileMapA block_2_tile_map_a,
                            const Block2TileMapB block_2_tile_map_b,
                            const ElementwiseOperation elementwise_op,
                            const index_t a_grid_size)
{
    if(get_block_1d_id() < a_grid_size)
    {
        GridwiseElementwiseFunctorA::Run(in_grid_desc_tuple_a,
                                         out_grid_desc_tuple_a,
                                         p_in_global_tuple_a,
                                         p_out_global_tuple_a,
                                         block_2_tile_map_a,
                                         elementwise_op,
                                         get_block_1d_id());
    }
    else
    {
        GridwiseElementwiseFunctorB::Run(in_grid_desc_tuple_b,
                                         out_grid_desc_tuple_b,
                                         p_in_global_tuple_b,
                                         p_out_global_tuple_b,
                                         block_2_tile_map_b,
                                         elementwise_op,
                                         get_block_1d_id() - a_grid_size);
    }
}

template <typename GridwiseElementwiseFunctorA,
          typename GridwiseElementwiseFunctorB,
          typename InAGridDescTuple,
          typename InBGridDescTuple,
          typename OutAGridDescTuple,
          typename OutBGridDescTuple,
          typename InADataTypePointerTuple,
          typename InBDataTypePointerTuple,
          typename OutADataTypePointerTuple,
          typename OutBDataTypePointerTuple,
          typename Block2TileMapA,
          typename Block2TileMapB,
          typename ElementwiseOperation,
          index_t NumInputsA,
          index_t NumInputsB,
          index_t NumOutputsA,
          index_t NumOutputsB>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_elementwise_batched_dual(const InAGridDescTuple in_grid_desc_tuple_a,
                                    const InBGridDescTuple in_grid_desc_tuple_b,
                                    const OutAGridDescTuple out_grid_desc_tuple_a,
                                    const OutBGridDescTuple out_grid_desc_tuple_b,
                                    const InADataTypePointerTuple p_in_global_tuple_a,
                                    const InBDataTypePointerTuple p_in_global_tuple_b,
                                    const OutADataTypePointerTuple p_out_global_tuple_a,
                                    const OutBDataTypePointerTuple p_out_global_tuple_b,
                                    const Block2TileMapA block_2_tile_map_a,
                                    const Block2TileMapB block_2_tile_map_b,
                                    const ElementwiseOperation elementwise_op,
                                    const index_t a_grid_size,
                                    const index_t batch_count_a,
                                    const index_t batch_count_b,
                                    const std::array<index_t, NumInputsA> input_batch_strides_a,
                                    const std::array<index_t, NumInputsB> input_batch_strides_b,
                                    const std::array<index_t, NumOutputsA> output_batch_strides_a,
                                    const std::array<index_t, NumOutputsB> output_batch_strides_b)
{
    static_assert(InAGridDescTuple::Size() == NumInputsA &&
                  InADataTypePointerTuple::Size() == NumInputsA);
    static_assert(OutAGridDescTuple::Size() == NumOutputsA &&
                  OutADataTypePointerTuple::Size() == NumOutputsA);
    static_assert(InBGridDescTuple::Size() == NumInputsB &&
                  InBDataTypePointerTuple::Size() == NumInputsB);
    static_assert(OutBGridDescTuple::Size() == NumOutputsB &&
                  OutBDataTypePointerTuple::Size() == NumOutputsB);

    const index_t block_id = __builtin_amdgcn_readfirstlane(get_block_1d_id());

    if(block_id < a_grid_size)
    {
        const index_t num_blocks_per_batch =
            __builtin_amdgcn_readfirstlane(a_grid_size / batch_count_a);
        const index_t g_idx = __builtin_amdgcn_readfirstlane(block_id / num_blocks_per_batch);

        InADataTypePointerTuple p_in_global_with_offset_tuple;
        OutADataTypePointerTuple p_out_global_with_offset_tuple;

        static_for<0, InADataTypePointerTuple::Size(), 1>{}([&](auto i) {
            p_in_global_with_offset_tuple(i) =
                p_in_global_tuple_a.At(i) +
                type_convert<long_index_t>(input_batch_strides_a[i]) * g_idx;
        });

        static_for<0, OutADataTypePointerTuple::Size(), 1>{}([&](auto i) {
            p_out_global_with_offset_tuple(i) =
                p_out_global_tuple_a.At(i) +
                type_convert<long_index_t>(output_batch_strides_a[i]) * g_idx;
        });

        GridwiseElementwiseFunctorA::Run(in_grid_desc_tuple_a,
                                         out_grid_desc_tuple_a,
                                         p_in_global_with_offset_tuple,
                                         p_out_global_with_offset_tuple,
                                         block_2_tile_map_a,
                                         elementwise_op,
                                         block_id);
    }
    else
    {
        const index_t num_blocks_per_batch =
            __builtin_amdgcn_readfirstlane((get_grid_size() - a_grid_size) / batch_count_b);
        const index_t g_idx =
            __builtin_amdgcn_readfirstlane((block_id - a_grid_size) / num_blocks_per_batch);

        InBDataTypePointerTuple p_in_global_with_offset_tuple;
        OutBDataTypePointerTuple p_out_global_with_offset_tuple;

        static_for<0, InBDataTypePointerTuple::Size(), 1>{}([&](auto i) {
            p_in_global_with_offset_tuple(i) =
                p_in_global_tuple_b.At(i) +
                type_convert<long_index_t>(input_batch_strides_b[i]) * g_idx;
        });

        static_for<0, OutBDataTypePointerTuple::Size(), 1>{}([&](auto i) {
            p_out_global_with_offset_tuple(i) =
                p_out_global_tuple_b.At(i) +
                type_convert<long_index_t>(output_batch_strides_b[i]) * g_idx;
        });

        GridwiseElementwiseFunctorB::Run(in_grid_desc_tuple_b,
                                         out_grid_desc_tuple_b,
                                         p_in_global_with_offset_tuple,
                                         p_out_global_with_offset_tuple,
                                         block_2_tile_map_b,
                                         elementwise_op,
                                         block_id - a_grid_size);
    }
}

template <typename GridwiseElementwiseFunctor,
          typename InGridDescTuple,
          typename OutGridDescTuple,
          typename InDataTypePointerTuple,
          typename OutDataTypePointerTuple,
          typename Block2TileMap,
          typename ElementwiseOperation,
          index_t NumInputs,
          index_t NumOutputs>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_batched_elementwise(const InGridDescTuple in_grid_desc_tuple,
                               const OutGridDescTuple out_grid_desc_tuple,
                               const InDataTypePointerTuple p_in_global_tuple,
                               const OutDataTypePointerTuple p_out_global_tuple,
                               const Block2TileMap block_2_tile_map,
                               const ElementwiseOperation elementwise_op,
                               const index_t batch_count,
                               const std::array<index_t, NumInputs> input_batch_strides,
                               const std::array<index_t, NumOutputs> output_batch_strides)
{
    static_assert(InGridDescTuple::Size() == NumInputs &&
                  InDataTypePointerTuple::Size() == NumInputs);
    static_assert(OutGridDescTuple::Size() == NumOutputs &&
                  OutDataTypePointerTuple::Size() == NumOutputs);

    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    InDataTypePointerTuple p_in_global_with_offset_tuple;
    OutDataTypePointerTuple p_out_global_with_offset_tuple;

    static_for<0, InDataTypePointerTuple::Size(), 1>{}([&](auto i) {
        p_in_global_with_offset_tuple(i) =
            p_in_global_tuple.At(i) + type_convert<long_index_t>(input_batch_strides[i]) * g_idx;
    });

    static_for<0, OutDataTypePointerTuple::Size(), 1>{}([&](auto i) {
        p_out_global_with_offset_tuple(i) =
            p_out_global_tuple.At(i) + type_convert<long_index_t>(output_batch_strides[i]) * g_idx;
    });

    GridwiseElementwiseFunctor::Run(in_grid_desc_tuple,
                                    out_grid_desc_tuple,
                                    p_in_global_with_offset_tuple,
                                    p_out_global_with_offset_tuple,
                                    block_2_tile_map,
                                    elementwise_op);
}

template <typename InGridDescTuple,
          typename OutGridDescTuple,
          typename InDataTypePointerTuple,
          typename OutDataTypePointerTuple,
          typename Block2TileMap,
          typename ElementwiseOperation,
          index_t BlockSize,
          index_t M0PerBlock,
          index_t M1PerBlock,
          index_t M0PerThread,
          index_t M1PerThread,
          typename ThreadClusterArrangeOrder,
          typename InScalarPerVectorSeq,
          typename OutScalarPerVectorSeq,
          index_t SrcVectorDim,
          index_t DstVectorDim>
struct GridwiseElementwise
{
    static constexpr index_t NumInput  = InDataTypePointerTuple::Size();
    static constexpr index_t NumOutput = OutDataTypePointerTuple::Size();

    static_assert(NumInput == InScalarPerVectorSeq::Size() &&
                      NumOutput == OutScalarPerVectorSeq::Size() &&
                      NumInput == InGridDescTuple::Size() && NumOutput == OutGridDescTuple::Size(),
                  "Tuple size is inconsistent with the number of in/out!");

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    static_assert((SrcVectorDim == I0 || SrcVectorDim == I1) &&
                      (DstVectorDim == I0 || DstVectorDim == I1),
                  "Vector dim must be equal to 0 or 1.");

    using PassThroughOp = tensor_operation::element_wise::PassThrough;

    __device__ static void Run(const InGridDescTuple& in_grid_desc_tuple,
                               const OutGridDescTuple& out_grid_desc_tuple,
                               const InDataTypePointerTuple& p_in_global_tuple,
                               const OutDataTypePointerTuple& p_out_global_tuple,
                               const Block2TileMap& block_2_tile_map,
                               const ElementwiseOperation& elementwise_op,
                               const index_t block_id = get_block_1d_id())
    {

        constexpr auto src_datas = generate_tuple(
            [&](auto I) {
                using DataTypePointer = remove_cvref_t<decltype(InDataTypePointerTuple{}[I])>;
                using DataType        = remove_cv_t<remove_pointer_t<DataTypePointer>>;

                return DataType{};
            },
            Number<NumInput>{});

        constexpr auto dst_datas = generate_tuple(
            [&](auto I) {
                using DataTypePointer = remove_cvref_t<decltype(OutDataTypePointerTuple{}[I])>;
                using DataType        = remove_pointer_t<DataTypePointer>;

                return DataType{};
            },
            Number<NumOutput>{});

        const auto in_global_buf_tuple = generate_tuple(
            [&](auto I) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_in_global_tuple[I], in_grid_desc_tuple[I].GetElementSpaceSize());
            },
            Number<NumInput>{});

        auto out_global_buf_tuple = generate_tuple(
            [&](auto I) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_out_global_tuple[I], out_grid_desc_tuple[I].GetElementSpaceSize());
            },
            Number<NumOutput>{});

        const auto block_work_idx =
            block_2_tile_map.CalculateBottomIndex(make_multi_index(block_id));

        const index_t m0_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * M0PerBlock);
        const index_t m1_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * M1PerBlock);
        const auto input_thread_grid_offset = generate_tuple(
            [&](auto) {
                return make_multi_index(m0_block_data_idx_on_grid, m1_block_data_idx_on_grid);
            },
            Number<NumInput>{});
        const auto output_thread_grid_offset = generate_tuple(
            [&](auto) {
                return make_multi_index(m0_block_data_idx_on_grid, m1_block_data_idx_on_grid);
            },
            Number<NumOutput>{});

        using ThisThreadBlock = ThisThreadBlock<BlockSize>;
        // If src and dst have same vector dim, then:
        //     M0 dim - for src and dst vector load/store
        // else:
        //     M0 dim - for dst vector load
        //     M1 dim - for src vector store
        using SrcDimAccessOrder =
            std::conditional_t<SrcVectorDim == I1, Sequence<0, 1>, Sequence<1, 0>>;
        using DstDimAccessOrder =
            std::conditional_t<DstVectorDim == I1, Sequence<0, 1>, Sequence<1, 0>>;

        using ThreadClusterLengths =
            Sequence<Number<M0PerBlock / M0PerThread>{}, Number<M1PerBlock / M1PerThread>{}>;

        auto global_to_global_transfer = ThreadGroupTensorSliceTransfer_v4r2<
            ThisThreadBlock,
            ElementwiseOperation,
            uniform_sequence_gen_t<NumOutput, static_cast<index_t>(InMemoryDataOperationEnum::Set)>,
            Sequence<M0PerBlock, M1PerBlock>,
            ThreadClusterLengths,
            ThreadClusterArrangeOrder,
            decltype(src_datas),
            decltype(dst_datas),
            InGridDescTuple,
            OutGridDescTuple,
            SrcDimAccessOrder,
            DstDimAccessOrder,
            SrcVectorDim,
            DstVectorDim,
            InScalarPerVectorSeq,
            OutScalarPerVectorSeq,
            uniform_sequence_gen_t<NumInput, 1>,
            uniform_sequence_gen_t<NumOutput, 1>,
            uniform_sequence_gen_t<NumInput, false>,
            uniform_sequence_gen_t<NumOutput, false>>{in_grid_desc_tuple,
                                                      input_thread_grid_offset,
                                                      out_grid_desc_tuple,
                                                      output_thread_grid_offset,
                                                      elementwise_op};
        global_to_global_transfer.Run(
            in_grid_desc_tuple, in_global_buf_tuple, out_grid_desc_tuple, out_global_buf_tuple, I0);
    }
};

} // namespace ck
