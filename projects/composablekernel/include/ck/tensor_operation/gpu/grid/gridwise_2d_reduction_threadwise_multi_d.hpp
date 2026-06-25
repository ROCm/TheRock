// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/reduction_common.hpp"
#include "ck/utility/reduction_operator.hpp"
#include "ck/utility/reduction_functions_accumulate.hpp"
#include "ck/tensor_operation/gpu/thread/reduction_functions_threadwise.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/tuple_helper.hpp"

namespace ck {

template <typename GridwiseReduction,
          typename InDataType,
          typename OutDataType,
          typename AccDataType,
          typename InGridDesc_M_K,
          typename DsGridDesc_M,
          typename OutGridDesc_M,
          typename InElementwiseOperation,
          typename OutElementwiseOperation,
          typename DsGridPointer>
__global__ void
kernel_reduce_threadwise_multi_d(const InGridDesc_M_K in_grid_desc_m_k,
                                 const DsGridDesc_M ds_grid_desc_m,
                                 const OutGridDesc_M out_grid_desc_m,
                                 const InElementwiseOperation in_elementwise_op,
                                 const OutElementwiseOperation out_elementwise_op,
                                 const InDataType* const __restrict__ p_in_value_global,
                                 const DsGridPointer p_ds_value_global,
                                 OutDataType* const __restrict__ p_out_value_global)
{
    GridwiseReduction::Run(in_grid_desc_m_k,
                           ds_grid_desc_m,
                           out_grid_desc_m,
                           in_elementwise_op,
                           out_elementwise_op,
                           p_in_value_global,
                           p_ds_value_global,
                           p_out_value_global);
}

template <typename InDataType,
          typename DsDataType,
          typename OutDataType,
          typename AccDataType,
          typename InGridDesc_M_K,
          typename DsGridDesc_M,
          typename OutGridDesc_M,
          typename ReduceOperation,
          typename InElementwiseOperation,
          typename OutElementwiseOperation,
          InMemoryDataOperationEnum OutMemoryDataOperation,
          index_t BlockSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t InSrcVectorDim,
          index_t InSrcVectorSize,
          index_t OutDstVectorSize,
          typename DsVectorSize>
struct GridwiseReduction_mk_to_m_threadwise_multi_d
{
    static_assert(((InSrcVectorDim == 0 && MThreadSliceSize % InSrcVectorSize == 0) ||
                   (InSrcVectorDim == 1 && KThreadSliceSize % InSrcVectorSize == 0)) &&
                      (MThreadSliceSize % OutDstVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    using ThreadBufferDimAccessOrder =
        typename conditional<InSrcVectorDim == 0, Sequence<1, 0>, Sequence<0, 1>>::type;

    using ThreadReduceSrcDesc_M_K = decltype(make_naive_tensor_descriptor_packed(
        make_tuple(Number<MThreadSliceSize>{}, Number<KThreadSliceSize>{})));
    using ThreadReduceDstDesc_M =
        decltype(make_naive_tensor_descriptor_packed(make_tuple(Number<MThreadSliceSize>{})));

    using PassThrough = tensor_operation::element_wise::PassThrough;

    static constexpr auto I0 = Number<0>{};

    static constexpr index_t NumDTensor = DsDataType::Size();

    // ck::Tuple<const D0DataType*, const D1DataType*, ...>
    static constexpr auto MakeDsGridPointer()
    {
        return generate_tuple(
            [&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                return static_cast<const DDataType*>(nullptr);
            },
            Number<NumDTensor>{});
    }

    using DsGridPointer = decltype(MakeDsGridPointer());

    __device__ static void Run(const InGridDesc_M_K& in_grid_desc_m_k,
                               const DsGridDesc_M& ds_grid_desc_m,
                               const OutGridDesc_M& out_grid_desc_m,
                               const InElementwiseOperation& in_elementwise_op,
                               const OutElementwiseOperation& out_elementwise_op,
                               const InDataType* const __restrict__ p_in_value_global,
                               const DsGridPointer p_ds_grid,
                               OutDataType* const __restrict__ p_out_value_global)
    {
        using ThreadwiseReduce = ThreadwiseReduction<AccDataType,
                                                     ThreadReduceSrcDesc_M_K,
                                                     ThreadReduceDstDesc_M,
                                                     ReduceOperation,
                                                     false>;

        const auto identityVal = ReduceOperation::template GetIdentityValue<AccDataType>();

        const auto in_global_val_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_in_value_global,
            in_grid_desc_m_k.GetElementSpaceSize(),
            ReduceOperation::template GetIdentityValue<InDataType>());
        auto dst_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_out_value_global, out_grid_desc_m.GetElementSpaceSize());

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize * KThreadSliceSize, true>
            in_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MThreadSliceSize, true> accu_value_buf;

        static_for<0, MThreadSliceSize, 1>{}([&](auto I) { accu_value_buf(I) = identityVal; });

        const auto toReduceLength = in_grid_desc_m_k.GetLength(Number<1>{});

        using ThreadBufferLengths         = Sequence<MThreadSliceSize, KThreadSliceSize>;
        constexpr auto thread_buffer_desc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<MThreadSliceSize>{}, Number<KThreadSliceSize>{}));

        index_t thread_global_1d_id = get_block_1d_id() * BlockSize + get_thread_local_1d_id();

        auto threadwise_src_val_load =
            ThreadwiseTensorSliceTransfer_v2<InDataType,
                                             AccDataType,
                                             InGridDesc_M_K,
                                             decltype(thread_buffer_desc),
                                             ThreadBufferLengths,
                                             ThreadBufferDimAccessOrder,
                                             InSrcVectorDim,
                                             InSrcVectorSize,
                                             1,
                                             false>(
                in_grid_desc_m_k, make_multi_index(thread_global_1d_id * MThreadSliceSize, 0));

        constexpr auto in_thread_copy_step = make_multi_index(0, KThreadSliceSize);

        index_t reducedLength = 0;
        do
        {
            threadwise_src_val_load.Run(in_grid_desc_m_k,
                                        in_global_val_buf,
                                        thread_buffer_desc,
                                        make_tuple(I0, I0),
                                        in_thread_buf);

            static_for<0, MThreadSliceSize, 1>{}([&](auto iM) {
                // do element-wise pre-reduction operation
                static_for<0, KThreadSliceSize, 1>{}([&](auto iK) {
                    constexpr auto offset = thread_buffer_desc.CalculateOffset(make_tuple(iM, iK));
                    in_elementwise_op(in_thread_buf(Number<offset>{}),
                                      in_thread_buf(Number<offset>{}));
                });
            });

            ThreadwiseReduce::Reduce(in_thread_buf, accu_value_buf);

            threadwise_src_val_load.MoveSrcSliceWindow(in_grid_desc_m_k, in_thread_copy_step);

            reducedLength += KThreadSliceSize;
        } while(reducedLength < toReduceLength);

        constexpr auto reduced_data_desc = ThreadReduceDstDesc_M{};

        auto ds_thread_buf = generate_tuple(
            [&](auto I) {
                using DataTypePointer = remove_cvref_t<decltype(DsGridPointer{}[I])>;
                using DataType        = remove_cv_t<remove_pointer_t<DataTypePointer>>;

                return StaticBuffer<AddressSpaceEnum::Vgpr, DataType, MThreadSliceSize, true>{};
            },
            Number<NumDTensor>{});

        auto ds_global_buf = generate_tuple(
            [&](auto I) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_ds_grid[I], ds_grid_desc_m[I].GetElementSpaceSize());
            },
            Number<NumDTensor>{});

        auto ds_global_load = generate_tuple(
            [&](auto I) {
                using DataTypePointer = remove_cvref_t<decltype(DsGridPointer{}[I])>;
                using DataType        = remove_cv_t<remove_pointer_t<DataTypePointer>>;

                return ThreadwiseTensorSliceTransfer_v2<DataType,
                                                        DataType,
                                                        decltype(ds_grid_desc_m[I]),
                                                        decltype(reduced_data_desc),
                                                        Sequence<MThreadSliceSize>, // SliceLengths
                                                        Sequence<0>,    // DimAccessOrder
                                                        InSrcVectorDim, // SrcVectorDim
                                                        DsVectorSize{}[I],
                                                        1, // SrcScalarStrideInVector
                                                        true>{
                    ds_grid_desc_m[I], make_multi_index(thread_global_1d_id * MThreadSliceSize)};
            },
            Number<NumDTensor>{});

        static_for<0, NumDTensor, 1>{}([&](auto I) {
            ds_global_load(I).Run(ds_grid_desc_m[I],
                                  ds_global_buf[I],
                                  reduced_data_desc,
                                  make_tuple(I0),
                                  ds_thread_buf(I));
        });

        StaticBuffer<AddressSpaceEnum::Vgpr, OutDataType, MThreadSliceSize, true> out_value_buf;

        // if constexpr(NumDTensor > 0)
        {
            static_for<0, MThreadSliceSize, 1>{}([&](auto I) {
                const auto c_ds_buf_refs = concat_tuple_of_reference(
                    tie(accu_value_buf[I]),
                    generate_tie([&](auto Id) -> const auto& { return ds_thread_buf[Id][I]; },
                                 Number<NumDTensor>{}));

                unpack2(out_elementwise_op, tie(out_value_buf(I)), c_ds_buf_refs);
            });
        }

        auto threadwise_dst_store = ThreadwiseTensorSliceTransfer_v1r3<OutDataType,
                                                                       OutDataType,
                                                                       decltype(reduced_data_desc),
                                                                       OutGridDesc_M,
                                                                       PassThrough,
                                                                       Sequence<MThreadSliceSize>,
                                                                       Sequence<0>,
                                                                       0,
                                                                       OutDstVectorSize,
                                                                       OutMemoryDataOperation,
                                                                       1,
                                                                       false>(
            out_grid_desc_m,
            make_multi_index(thread_global_1d_id * MThreadSliceSize),
            PassThrough{});

        threadwise_dst_store.Run(
            reduced_data_desc, make_tuple(I0), out_value_buf, out_grid_desc_m, dst_global_buf);
    }
};

} // namespace ck
