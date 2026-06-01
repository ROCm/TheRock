// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"

namespace ck {

template <typename GridwisePutElementwise1dFunctor,
          typename InGrid1dDesc,
          typename InDataType,
          typename IndexDataType,
          typename OutDataType,
          typename ElementwiseOperation>
__global__ void kernel_put_element_1d(const InGrid1dDesc in_grid_1d_desc,
                                      const InDataType* __restrict__ p_in_global,
                                      const IndexDataType* __restrict__ p_indices_global,
                                      OutDataType* __restrict__ p_out_global,
                                      const ElementwiseOperation elementwise_op)
{
    GridwisePutElementwise1dFunctor::Run(
        in_grid_1d_desc, p_in_global, p_indices_global, p_out_global, elementwise_op);
}

// output[indices] = input
template <typename InGrid1dDesc,
          typename InDataType,
          typename IndexDataType,
          typename OutDataType,
          typename ElementwiseOperation,
          InMemoryDataOperationEnum MemOp,
          index_t InVectorSize>
struct GridwisePutElement_1D
{
    static constexpr auto I0 = Number<0>{};

    static constexpr auto thread_buffer_desc_m =
        make_naive_tensor_descriptor_packed(make_tuple(Number<InVectorSize>{}));

    __device__ static void Run(const InGrid1dDesc& in_grid_1d_desc,
                               const InDataType* __restrict__ p_in_global,
                               const IndexDataType* __restrict__ p_indices_global,
                               OutDataType* __restrict__ p_out_global,
                               const ElementwiseOperation& elementwise_op)
    {
        // Global Memory
        const auto in_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_in_global, in_grid_1d_desc.GetElementSpaceSize());

        const auto indices_global_buf =
            make_dynamic_buffer<AddressSpaceEnum::Global>(p_indices_global,
                                                          in_grid_1d_desc.GetElementSpaceSize(),
                                                          NumericLimits<IndexDataType>::Lowest());

        // VGPR
        StaticBuffer<AddressSpaceEnum::Vgpr, InDataType, InVectorSize, true> in_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, IndexDataType, InVectorSize, true> indices_thread_buf;

        // Thread id, Block id and index
        const index_t thread_global_id  = get_thread_global_1d_id();
        const auto thread_global_offset = make_multi_index(thread_global_id * InVectorSize);
        const index_t blockSize         = get_block_size();
        const index_t blockPerGrid      = get_grid_size();
        const auto M                    = in_grid_1d_desc.GetLength(I0);
        const index_t loop_step         = blockPerGrid * blockSize * InVectorSize;
        const auto loop_step_index      = make_multi_index(loop_step);

        auto in_global_load =
            ThreadwiseTensorSliceTransfer_v2<InDataType,
                                             InDataType,
                                             decltype(in_grid_1d_desc),
                                             decltype(thread_buffer_desc_m),
                                             Sequence<InVectorSize>, // SliceLengths
                                             Sequence<0>,            // DimAccessOrder
                                             0,                      // SrcVectorDim
                                             InVectorSize,           // ScalarPerVector
                                             1,                      // SrcScalarStrideInVector
                                             false>{in_grid_1d_desc, thread_global_offset};

        auto indices_global_load =
            ThreadwiseTensorSliceTransfer_v2<IndexDataType,
                                             IndexDataType,
                                             decltype(in_grid_1d_desc),
                                             decltype(thread_buffer_desc_m),
                                             Sequence<InVectorSize>, // SliceLengths
                                             Sequence<0>,            // DimAccessOrder
                                             0,                      // SrcVectorDim
                                             InVectorSize,           // ScalarPerVector
                                             1,                      // SrcScalarStrideInVector
                                             false>{in_grid_1d_desc, thread_global_offset};

        index_t num_iter = M / loop_step;
        do
        {
            in_global_load.Run(in_grid_1d_desc,
                               in_global_buf,
                               thread_buffer_desc_m,
                               make_tuple(I0),
                               in_thread_buf);

            in_global_load.MoveSrcSliceWindow(in_grid_1d_desc, loop_step_index);

            static_for<0, InVectorSize, 1>{}(
                [&](auto iM) { elementwise_op(in_thread_buf(iM), in_thread_buf[iM]); });

            indices_global_load.Run(in_grid_1d_desc,
                                    indices_global_buf,
                                    thread_buffer_desc_m,
                                    make_tuple(I0),
                                    indices_thread_buf);

            indices_global_load.MoveSrcSliceWindow(in_grid_1d_desc, loop_step_index);

            static_for<0, InVectorSize, 1>{}([&](auto iM) {
                if(indices_thread_buf[iM] >= 0)
                {
                    if constexpr(MemOp == InMemoryDataOperationEnum::Set)
                    {
                        // User should guarantee each index in p_indices_global is different
                        *(p_out_global + indices_thread_buf[iM]) =
                            ck::type_convert<OutDataType>(in_thread_buf[iM]);
                    }
                    else if constexpr(MemOp == InMemoryDataOperationEnum::AtomicAdd)
                    {
                        atomic_add<OutDataType>(p_out_global + indices_thread_buf[iM],
                                                ck::type_convert<OutDataType>(in_thread_buf[iM]));
                    }
                    else if constexpr(MemOp == InMemoryDataOperationEnum::AtomicMax)
                    {
                        atomic_max<OutDataType>(p_out_global + indices_thread_buf[iM],
                                                ck::type_convert<OutDataType>(in_thread_buf[iM]));
                    }
                    else if constexpr(MemOp == InMemoryDataOperationEnum::Add)
                    {
                        // User should guarantee each index in p_indices_global is different
                        *(p_out_global + indices_thread_buf[iM]) +=
                            ck::type_convert<OutDataType>(in_thread_buf[iM]);
                    }
                    else
                    {
                        static_assert(MemOp == InMemoryDataOperationEnum::Set ||
                                      MemOp == InMemoryDataOperationEnum::AtomicAdd ||
                                      MemOp == InMemoryDataOperationEnum::AtomicMax ||
                                      MemOp == InMemoryDataOperationEnum::Add);
                    }
                }
            });

        } while(--num_iter);
    }
};

} // namespace ck
