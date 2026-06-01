// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/math.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/tensor_operation/gpu/device/device_elementwise.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_elementwise_2d.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"

#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/stream_utility.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename InDataTypeTuple,
          typename OutDataTypeTuple,
          typename ElementwiseOperation,
          index_t NumDim,
          index_t BlockSize,
          index_t M0PerBlock,
          index_t M1PerBlock,
          index_t M0PerThread,
          index_t M1PerThread,
          typename ThreadClusterArrangeOrder,
          typename InScalarPerVectorSeq,
          typename OutScalarPerVectorSeq>
struct DeviceElementwiseImpl
    : public DeviceElementwise<InDataTypeTuple, OutDataTypeTuple, ElementwiseOperation, NumDim>
{
    static constexpr int NumInput  = InDataTypeTuple::Size();
    static constexpr int NumOutput = OutDataTypeTuple::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

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

    static index_t GetLowestStrideDim(const std::array<index_t, NumDim>& strides)
    {
        index_t most_continous_dim        = NumDim - 1;
        index_t most_continous_dim_stride = strides[most_continous_dim];
        for(index_t dim = 0; dim < NumDim; dim++)
        {
            if(strides[dim] < most_continous_dim_stride)
            {
                most_continous_dim_stride = strides[dim];
                most_continous_dim        = dim;
            }
        }
        return most_continous_dim;
    }

    template <typename InOutDescriptor>
    static auto PadInputOutputDescriptor(const InOutDescriptor& desc)
    {
        const auto M0     = desc.GetLength(I0);
        const auto M1     = desc.GetLength(I1);
        const auto pad_M0 = math::integer_divide_ceil(M0, M0PerThread) * M0PerThread - M0;
        const auto pad_M1 = math::integer_divide_ceil(M1, M1PerThread) * M1PerThread - M1;

        const auto padded_desc = transform_tensor_descriptor(
            desc,
            make_tuple(make_right_pad_transform(M0, pad_M0), make_right_pad_transform(M1, pad_M1)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return padded_desc;
    }

    static auto GenerateBatchDimsLenghtsTuple(const std::array<index_t, NumDim>& lengths,
                                              const index_t M0_dim,
                                              const index_t M1_dim)
    {
        // Generate batch dims, they will be merged to M0
        // Add one more dim than needed in case that M0 is equal to M1
        // If M0 is equal to M1, then will be one more batch dim
        std::array<index_t, NumDim - 1> batch_dims;
        index_t batch_dim = 0;
        for(index_t i = 0; i < NumDim; i++)
        {
            if(i != M0_dim && i != M1_dim)
            {
                batch_dims[batch_dim] = lengths[i];
                batch_dim++;
            }
        }
        // Add dummy dim if M0_dim is not equal to M1_dim
        if(M0_dim != M1_dim && NumDim >= 2)
            batch_dims[NumDim - 2] = 1;
        return generate_tuple([&](auto I) { return batch_dims[I]; }, Number<NumDim - 1>{});
    }

    static auto MakeDescriptor(const std::array<index_t, NumDim>& lengths,
                               const std::array<index_t, NumDim>& in_strides,
                               const std::array<index_t, NumDim>& out_strides,
                               const std::array<index_t, NumDim>& desc_strides)
    {
        const auto M0_dim = GetLowestStrideDim(out_strides);
        const auto M1_dim = GetLowestStrideDim(in_strides);

        // If M0_dim is equal to M1_dim, then make M0_dim dummy
        const auto M0        = M0_dim == M1_dim ? I1 : lengths[M0_dim];
        const auto M1        = lengths[M1_dim];
        const auto M0_stride = M0_dim == M1_dim ? I1 : desc_strides[M0_dim];
        const auto M1_stride = desc_strides[M1_dim];

        const auto batch_dims_lenghts = GenerateBatchDimsLenghtsTuple(lengths, M0_dim, M1_dim);
        const auto batch_dims_strides = GenerateBatchDimsLenghtsTuple(desc_strides, M0_dim, M1_dim);

        const auto desc = make_naive_tensor_descriptor(
            concat_tuple(batch_dims_lenghts, make_tuple(M0), make_tuple(M1)),
            concat_tuple(batch_dims_strides, make_tuple(M0_stride), make_tuple(M1_stride)));
        // Merged batch dims with M0
        const auto transforms =
            make_tuple(make_merge_transform(concat_tuple(batch_dims_lenghts, make_tuple(M0))),
                       make_pass_through_transform(M1));
        using BatchElemsSequence =
            typename arithmetic_sequence_gen<0, decltype(batch_dims_lenghts)::Size() + 1, 1>::type;
        const auto lower_dims = make_tuple(BatchElemsSequence{}, Sequence<NumDim>{});
        const auto upper_dims = make_tuple(Sequence<0>{}, Sequence<1>{});
        // desc: (merged_dims + M0, M1)
        auto merged_desc = transform_tensor_descriptor(desc, transforms, lower_dims, upper_dims);
        return PadInputOutputDescriptor(merged_desc);
    }

    template <index_t NumTensors>
    static auto GenerateInOutGridDescTuple()
    {
        std::array<index_t, NumDim> ones;
        for(index_t d = 0; d < NumDim; d++)
        {
            ones[d] = 1;
        }

        return generate_tuple([&](auto) { return MakeDescriptor(ones, ones, ones, ones); },
                              Number<NumTensors>{});
    };

    using InGridDescTuple  = decltype(GenerateInOutGridDescTuple<NumInput>());
    using OutGridDescTuple = decltype(GenerateInOutGridDescTuple<NumOutput>());

    using Block2TileMap = BlockToCTileMap_M00_N0_M01Adapt<M0PerBlock, M1PerBlock>;

    using GridwiseElementwiseOp = GridwiseElementwise<InGridDescTuple,
                                                      OutGridDescTuple,
                                                      InDataTypePointerTuple,
                                                      OutDataTypePointerTuple,
                                                      Block2TileMap,
                                                      ElementwiseOperation,
                                                      BlockSize,
                                                      M0PerBlock,
                                                      M1PerBlock,
                                                      M0PerThread,
                                                      M1PerThread,
                                                      ThreadClusterArrangeOrder,
                                                      InScalarPerVectorSeq,
                                                      OutScalarPerVectorSeq,
                                                      I1,
                                                      I0>;

    using GridwiseElementwiseOpSameInOutVectorDim = GridwiseElementwise<InGridDescTuple,
                                                                        OutGridDescTuple,
                                                                        InDataTypePointerTuple,
                                                                        OutDataTypePointerTuple,
                                                                        Block2TileMap,
                                                                        ElementwiseOperation,
                                                                        BlockSize,
                                                                        M0PerBlock,
                                                                        M1PerBlock,
                                                                        M0PerThread,
                                                                        M1PerThread,
                                                                        ThreadClusterArrangeOrder,
                                                                        InScalarPerVectorSeq,
                                                                        OutScalarPerVectorSeq,
                                                                        I1,
                                                                        I1>;

    struct Argument : public BaseArgument
    {
        Argument(const std::array<index_t, NumDim> lengths,
                 const std::array<std::array<index_t, NumDim>, NumInput> inStridesArray,
                 const std::array<std::array<index_t, NumDim>, NumOutput> outStridesArray,
                 const std::array<const void*, NumInput> in_dev_buffers,
                 const std::array<void*, NumOutput> out_dev_buffers,
                 ElementwiseOperation elementwise_op)

            : lengths_(lengths),
              inStridesArray_(inStridesArray),
              outStridesArray_(outStridesArray),
              elementwise_op_(elementwise_op)
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
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            auto in_grid_desc_tuple = generate_tuple(
                [&](auto src_i) {
                    // Use Strides from first tensor to assert that M0 dim and
                    // M1 dim are the same for each tensor.
                    return MakeDescriptor(arg.lengths_,
                                          arg.inStridesArray_[I0],
                                          arg.outStridesArray_[I0],
                                          arg.inStridesArray_[src_i]);
                },
                Number<NumInput>{});

            auto out_grid_desc_tuple = generate_tuple(
                [&](auto dst_i) {
                    return MakeDescriptor(arg.lengths_,
                                          arg.inStridesArray_[I0],
                                          arg.outStridesArray_[I0],
                                          arg.outStridesArray_[dst_i]);
                },
                Number<NumOutput>{});

            const index_t M0 = in_grid_desc_tuple.At(I0).GetLength(Number<I0>{});
            const index_t M1 = in_grid_desc_tuple.At(I0).GetLength(Number<I1>{});

            const auto block_2_tile_map = Block2TileMap(M0, M1);
            const index_t grid_size     = block_2_tile_map.CalculateGridSize(M0, M1);

            const bool in_out_same_vector_dim = GetLowestStrideDim(arg.inStridesArray_[I0]) ==
                                                GetLowestStrideDim(arg.outStridesArray_[I0]);

            const auto kernel = in_out_same_vector_dim
                                    ? kernel_elementwise<GridwiseElementwiseOpSameInOutVectorDim,
                                                         InGridDescTuple,
                                                         OutGridDescTuple,
                                                         InDataTypePointerTuple,
                                                         OutDataTypePointerTuple,
                                                         Block2TileMap,
                                                         ElementwiseOperation>
                                    : kernel_elementwise<GridwiseElementwiseOp,
                                                         InGridDescTuple,
                                                         OutGridDescTuple,
                                                         InDataTypePointerTuple,
                                                         OutDataTypePointerTuple,
                                                         Block2TileMap,
                                                         ElementwiseOperation>;

            float elapsed_time = launch_and_time_kernel(stream_config,
                                                        kernel,
                                                        dim3(grid_size),
                                                        dim3(BlockSize),
                                                        0,
                                                        in_grid_desc_tuple,
                                                        out_grid_desc_tuple,
                                                        arg.in_dev_buffers_,
                                                        arg.out_dev_buffers_,
                                                        block_2_tile_map,
                                                        arg.elementwise_op_);
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
        const index_t M0_dim = GetLowestStrideDim(arg.inStridesArray_[I0]);
        const index_t M1_dim = GetLowestStrideDim(arg.outStridesArray_[I0]);

        auto IsScalarPerVectorValid = [&](const std::array<index_t, NumDim>& lengths,
                                          const std::array<index_t, NumDim>& strides,
                                          index_t scalarPerVector,
                                          index_t M_dim) {
            if(scalarPerVector == 1)
            {
                return true;
            }
            if(strides[M_dim] == 1 && lengths[M_dim] % scalarPerVector == 0)
            {
                return true;
            }
            return false;
        };

        bool is_valid = true;
        static_for<0, NumInput, 1>{}([&](auto I) {
            static_assert(M0PerThread % InScalarPerVectorSeq::At(I) == 0 &&
                          M1PerThread % InScalarPerVectorSeq::At(I) == 0);
            is_valid &= IsScalarPerVectorValid(
                arg.lengths_, arg.inStridesArray_[I.value], InScalarPerVectorSeq::At(I), M0_dim);
        });

        static_for<0, NumOutput, 1>{}([&](auto I) {
            static_assert(M0PerThread % OutScalarPerVectorSeq::At(I) == 0 &&
                          M1PerThread % OutScalarPerVectorSeq::At(I) == 0);
            is_valid &= IsScalarPerVectorValid(
                arg.lengths_, arg.outStridesArray_[I.value], OutScalarPerVectorSeq::At(I), M1_dim);
        });

        return is_valid;
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
                 ElementwiseOperation elementwise_op)
    {
        return Argument{lengths,
                        inStridesArray,
                        outStridesArray,
                        in_dev_buffers,
                        out_dev_buffers,
                        elementwise_op};
    }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::array<index_t, NumDim> lengths,
                        const std::array<std::array<index_t, NumDim>, NumInput> inStridesArray,
                        const std::array<std::array<index_t, NumDim>, NumOutput> outStridesArray,
                        const std::array<const void*, NumInput> in_dev_buffers,
                        const std::array<void*, NumOutput> out_dev_buffers,
                        ElementwiseOperation elementwise_op) override
    {
        return std::make_unique<Argument>(lengths,
                                          inStridesArray,
                                          outStridesArray,
                                          in_dev_buffers,
                                          out_dev_buffers,
                                          elementwise_op);
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
        str << "DeviceElementwiseImpl<";
        str << NumDim << ", ";
        str << BlockSize << ", ";
        str << M0PerBlock << ", ";
        str << M1PerBlock << ", ";
        str << M0PerThread << ", ";
        str << M1PerThread << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
