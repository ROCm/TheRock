// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/math.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/tensor_operation/gpu/device/device_elementwise_scale.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_elementwise_1d_scale.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"

#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/stream_utility.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

/**
 * \note This structure is deprecated (left for backwards compatibility). Please use
 *       DeviceElementwiseImpl from device_elementwise_dynamic_vector_dims_impl.hpp.
 */
template <typename InDataTypeTuple,
          typename OutDataTypeTuple,
          typename ElementwiseOperation,
          typename UnaryOperation,
          typename Scale,
          index_t NumDim,
          index_t MPerThread,
          typename InScalarPerVectorSeq,
          typename OutScalarPerVectorSeq>
struct DeviceElementwiseImpl : public DeviceElementwise<InDataTypeTuple,
                                                        OutDataTypeTuple,
                                                        ElementwiseOperation,
                                                        UnaryOperation,
                                                        Scale,
                                                        NumDim>
{
    static constexpr int NumInput  = InDataTypeTuple::Size();
    static constexpr int NumOutput = OutDataTypeTuple::Size();

    static_assert(NumInput == InScalarPerVectorSeq::Size() &&
                      NumOutput == OutScalarPerVectorSeq::Size(),
                  "Tuple size is inconsistent with the number of in/out!");

    static auto GenerateInDataTypePointerTuple()
    {
        return generate_tuple(
            [&](auto I) {
                using DataType = remove_cvref_t<decltype(InDataTypeTuple{}[I])>;

                return static_cast<const DataType*>(nullptr);
            },
            Number<NumInput>{});
    };

    static auto GenerateOutDataTypePointerTuple()
    {
        return generate_tuple(
            [&](auto I) {
                using DataType = remove_cvref_t<decltype(OutDataTypeTuple{}[I])>;

                return static_cast<DataType*>(nullptr);
            },
            Number<NumOutput>{});
    };

    using InDataTypePointerTuple  = decltype(GenerateInDataTypePointerTuple());
    using OutDataTypePointerTuple = decltype(GenerateOutDataTypePointerTuple());

    template <typename Desc_M>
    static auto PadDescriptor_M_1d(Desc_M desc_m, index_t gridSize, index_t blockSize)
    {
        constexpr auto I0 = Number<0>{};

        const auto m            = desc_m.GetLength(I0);
        const index_t loop_step = gridSize * blockSize * MPerThread;
        const auto pad          = math::integer_least_multiple(m, loop_step) - m;
        const auto desc_m_pad =
            transform_tensor_descriptor(desc_m,
                                        make_tuple(make_right_pad_transform(m, pad)),
                                        make_tuple(Sequence<0>{}),
                                        make_tuple(Sequence<0>{}));
        return desc_m_pad;
    }

    static auto MakeDescriptor_M(const std::array<index_t, NumDim>& lengths,
                                 const std::array<index_t, NumDim>& stride,
                                 index_t gridSize,
                                 index_t blockSize)
    {
        auto tupleOfShape  = generate_tuple([&](auto I) { return lengths[I]; }, Number<NumDim>{});
        auto tupleOfStride = generate_tuple([&](auto I) { return stride[I]; }, Number<NumDim>{});

        // nd desc - [s0, s1, s2, ...]
        const auto desc = make_naive_tensor_descriptor(tupleOfShape, tupleOfStride);

        // merge nd to 1d desc - [s0 * s1 * ...]
        if constexpr(NumDim > 1)
        {
            const auto desc_m = transform_tensor_descriptor(
                desc,
                make_tuple(make_merge_transform(tupleOfShape)),
                make_tuple(generate_sequence_v2([&](auto I) { return I; }, Number<NumDim>{})),
                make_tuple(Sequence<0>{}));

            return PadDescriptor_M_1d(desc_m, gridSize, blockSize);
        }
        else
            return PadDescriptor_M_1d(desc, gridSize, blockSize);
    }

    template <index_t TupleSize>
    static auto GenerateInOutGrid1dDescTuple(Number<TupleSize>)
    {
        return generate_tuple(
            [&](auto) {
                if constexpr(NumDim > 1)
                {
                    return MakeDescriptor_M({1, 1}, {1, 1}, 1, 1);
                }
                else
                {
                    return MakeDescriptor_M({1}, {1}, 1, 1);
                };
            },
            Number<TupleSize>{});
    };

    using InGrid1dDescTuple  = decltype(GenerateInOutGrid1dDescTuple(Number<NumInput>{}));
    using OutGrid1dDescTuple = decltype(GenerateInOutGrid1dDescTuple(Number<NumOutput>{}));

    using GridwiseElementwise = GridwiseElementwise_1D<InGrid1dDescTuple,
                                                       OutGrid1dDescTuple,
                                                       InDataTypePointerTuple,
                                                       OutDataTypePointerTuple,
                                                       ElementwiseOperation,
                                                       UnaryOperation,
                                                       Scale,
                                                       MPerThread,
                                                       InScalarPerVectorSeq,
                                                       OutScalarPerVectorSeq>;

    struct Argument : public BaseArgument
    {
        Argument(const std::array<index_t, NumDim> lengths,
                 const std::array<std::array<index_t, NumDim>, NumInput> inStridesArray,
                 const std::array<std::array<index_t, NumDim>, NumOutput> outStridesArray,
                 const std::array<const void*, NumInput> in_dev_buffers,
                 const std::array<void*, NumOutput> out_dev_buffers,
                 ElementwiseOperation elementwise_op,
                 UnaryOperation unary_op,
                 Scale scale_op)

            : lengths_(lengths),
              inStridesArray_(inStridesArray),
              outStridesArray_(outStridesArray),
              elementwise_op_(elementwise_op),
              unary_op_(unary_op),
              scale_op_(scale_op),
              blockSize_(256)
        {
            in_dev_buffers_ = generate_tuple(
                [&](auto I) {
                    using DataType = remove_cvref_t<decltype(InDataTypeTuple{}[I])>;
                    return static_cast<const DataType*>(in_dev_buffers[I.value]);
                },
                Number<NumInput>{});

            out_dev_buffers_ = generate_tuple(
                [&](auto I) {
                    using DataType = remove_cvref_t<decltype(OutDataTypeTuple{}[I])>;
                    return static_cast<DataType*>(out_dev_buffers[I.value]);
                },
                Number<NumOutput>{});
        }

        InDataTypePointerTuple in_dev_buffers_;
        OutDataTypePointerTuple out_dev_buffers_;

        std::array<index_t, NumDim> lengths_;
        std::array<std::array<index_t, NumDim>, NumInput> inStridesArray_;
        std::array<std::array<index_t, NumDim>, NumOutput> outStridesArray_;

        ElementwiseOperation elementwise_op_;
        UnaryOperation unary_op_;
        Scale scale_op_;
        index_t blockSize_;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            index_t gridSize = getAvailableComputeUnitCount(stream_config);

            auto in_grid_1d_desc_tuple = generate_tuple(
                [&](auto I) {
                    return MakeDescriptor_M(
                        arg.lengths_, arg.inStridesArray_[I.value], gridSize, arg.blockSize_);
                },
                Number<NumInput>{});

            auto out_grid_1d_desc_tuple = generate_tuple(
                [&](auto I) {
                    return MakeDescriptor_M(
                        arg.lengths_, arg.outStridesArray_[I.value], gridSize, arg.blockSize_);
                },
                Number<NumOutput>{});

            const auto kernel = kernel_elementwise_1d<GridwiseElementwise,
                                                      InGrid1dDescTuple,
                                                      OutGrid1dDescTuple,
                                                      InDataTypePointerTuple,
                                                      OutDataTypePointerTuple,
                                                      ElementwiseOperation,
                                                      UnaryOperation,
                                                      Scale>;

            float elapsed_time = launch_and_time_kernel(stream_config,
                                                        kernel,
                                                        dim3(gridSize),
                                                        dim3(arg.blockSize_),
                                                        0,
                                                        in_grid_1d_desc_tuple,
                                                        out_grid_1d_desc_tuple,
                                                        arg.in_dev_buffers_,
                                                        arg.out_dev_buffers_,
                                                        arg.elementwise_op_,
                                                        arg.unary_op_,
                                                        arg.scale_op_);
            return elapsed_time;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(arg.lengths_.back() % MPerThread != 0)
            return false;

        auto IsScalarPerVectorValid = [&](const std::array<index_t, NumDim>& lengths,
                                          const std::array<index_t, NumDim>& strides,
                                          index_t scalarPerVector) {
            if(strides.back() == 1 && lengths.back() % scalarPerVector == 0)
                return true;

            if(strides.back() != 1 && scalarPerVector == 1)
                return true;

            return false;
        };

        bool valid = true;
        static_for<0, NumInput, 1>{}([&](auto I) {
            if(!IsScalarPerVectorValid(
                   arg.lengths_, arg.inStridesArray_[I.value], InScalarPerVectorSeq::At(I)))
                valid = false;
        });

        static_for<0, NumOutput, 1>{}([&](auto I) {
            if(!IsScalarPerVectorValid(
                   arg.lengths_, arg.outStridesArray_[I.value], OutScalarPerVectorSeq::At(I)))
                valid = false;
        });

        return valid;
    };

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto
    MakeArgument(const std::array<index_t, NumDim> lengths,
                 const std::array<std::array<index_t, NumDim>, NumInput> inStridesArray,
                 const std::array<std::array<index_t, NumDim>, NumOutput> outStridesArray,
                 const std::array<const void*, NumInput> in_dev_buffers,
                 const std::array<void*, NumOutput> out_dev_buffers,
                 ElementwiseOperation elementwise_op,
                 UnaryOperation unary_op,
                 Scale scale_op)
    {
        return Argument{lengths,
                        inStridesArray,
                        outStridesArray,
                        in_dev_buffers,
                        out_dev_buffers,
                        elementwise_op,
                        unary_op,
                        scale_op};
    }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::array<index_t, NumDim> lengths,
                        const std::array<std::array<index_t, NumDim>, NumInput> inStridesArray,
                        const std::array<std::array<index_t, NumDim>, NumOutput> outStridesArray,
                        const std::array<const void*, NumInput> in_dev_buffers,
                        const std::array<void*, NumOutput> out_dev_buffers,
                        ElementwiseOperation elementwise_op,
                        UnaryOperation unary_op,
                        Scale scale_op) override
    {
        return std::make_unique<Argument>(lengths,
                                          inStridesArray,
                                          outStridesArray,
                                          in_dev_buffers,
                                          out_dev_buffers,
                                          elementwise_op,
                                          unary_op,
                                          scale_op);
    }

    static auto MakeInvoker() { return Invoker{}; }
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    };

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceElementwiseNormalizationImpl<";
        str << NumDim << ", ";
        str << MPerThread << ">";
        // clang-format on

        return str.str();
    }
}; // namespace device

} // namespace device
} // namespace tensor_operation
} // namespace ck
