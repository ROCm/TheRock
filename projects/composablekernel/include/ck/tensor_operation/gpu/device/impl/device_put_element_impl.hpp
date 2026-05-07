// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/device_put_element.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_put_element_1d.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/stream_utility.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// output[indices] = input
template <typename InDataType,
          typename IndexDataType,
          typename OutDataType,
          typename ElementwiseOperation,
          InMemoryDataOperationEnum MemOp,
          ck::index_t InVectorSize>
struct DevicePutElementImpl
    : public DevicePutElement<InDataType, IndexDataType, OutDataType, ElementwiseOperation, MemOp>
{
    template <typename Desc_M>
    static auto PadDescriptor_M_1d(Desc_M desc_m, index_t gridSize, index_t blockSize)
    {
        constexpr auto I0 = Number<0>{};

        const auto m            = desc_m.GetLength(I0);
        const index_t loop_step = gridSize * blockSize * InVectorSize;
        const auto pad          = math::integer_least_multiple(m, loop_step) - m;
        const auto desc_m_pad =
            transform_tensor_descriptor(desc_m,
                                        make_tuple(make_right_pad_transform(m, pad)),
                                        make_tuple(Sequence<0>{}),
                                        make_tuple(Sequence<0>{}));
        return desc_m_pad;
    }

    static auto MakeDescriptor_M(index_t length, index_t gridSize, index_t blockSize)
    {
        const auto desc_m = make_naive_tensor_descriptor_packed(make_tuple(length));
        return PadDescriptor_M_1d(desc_m, gridSize, blockSize);
    }

    using InGrid1dDesc = decltype(MakeDescriptor_M(1, 1, 1));

    using GridwisePutElement = GridwisePutElement_1D<InGrid1dDesc,
                                                     InDataType,
                                                     IndexDataType,
                                                     OutDataType,
                                                     ElementwiseOperation,
                                                     MemOp,
                                                     InVectorSize>;

    struct Argument : public BaseArgument
    {
        Argument(const InDataType* p_input,
                 const IndexDataType* p_indices,
                 OutDataType* p_output,
                 index_t input_length,
                 ElementwiseOperation elementwise_op)
            : p_input_{p_input},
              p_indices_{p_indices},
              p_output_{p_output},
              input_length_raw_{input_length},
              elementwise_op_{elementwise_op},
              blockSize_{256}
        {
        }

        const InDataType* p_input_;
        const IndexDataType* p_indices_;
        OutDataType* p_output_;
        index_t input_length_raw_;
        ElementwiseOperation elementwise_op_;
        index_t blockSize_;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            index_t gridSize = getAvailableComputeUnitCount(stream_config);
            InGrid1dDesc in_grid_desc =
                MakeDescriptor_M(arg.input_length_raw_, gridSize, arg.blockSize_);

            const auto kernel = kernel_put_element_1d<GridwisePutElement,
                                                      InGrid1dDesc,
                                                      InDataType,
                                                      IndexDataType,
                                                      OutDataType,
                                                      ElementwiseOperation>;

            float elapsed_time = launch_and_time_kernel(stream_config,
                                                        kernel,
                                                        dim3(gridSize),
                                                        dim3(arg.blockSize_),
                                                        0,
                                                        in_grid_desc,
                                                        arg.p_input_,
                                                        arg.p_indices_,
                                                        arg.p_output_,
                                                        arg.elementwise_op_);
            return elapsed_time;
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        const Argument* pArg = dynamic_cast<const Argument*>(p_arg);

        if(pArg->input_length_raw_ % InVectorSize != 0)
        {
            return false;
        }
        return true;
    }

    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_input,
                                                      const void* p_indices,
                                                      void* p_output,
                                                      index_t input_length,
                                                      index_t,
                                                      ElementwiseOperation elementwise_op) override
    {
        return std::make_unique<Argument>(static_cast<const InDataType*>(p_input),
                                          static_cast<const IndexDataType*>(p_indices),
                                          static_cast<OutDataType*>(p_output),
                                          input_length,
                                          elementwise_op);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
