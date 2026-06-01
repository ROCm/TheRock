// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>
#include <array>

#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce_multi_d.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_reduce_common.hpp"

#include "ck/tensor_operation/gpu/grid/gridwise_2d_reduction_threadwise_multi_d.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename InDataType,
          typename DsDataType,
          typename AccDataType,
          typename OutDataType,
          index_t Rank,
          index_t NumReduceDim,
          typename ReduceOperation,
          typename InElementwiseOperation,
          typename OutElementwiseOperation,
          index_t BlockSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t InSrcVectorDim,
          index_t InSrcVectorSize,
          index_t OutDstVectorSize,
          typename DsVectorSizeSequence>
struct DeviceReduceThreadWiseMultiD : public DeviceReduceMultiD<InDataType,
                                                                DsDataType,
                                                                AccDataType,
                                                                OutDataType,
                                                                Rank,
                                                                NumReduceDim,
                                                                ReduceOperation,
                                                                InElementwiseOperation,
                                                                OutElementwiseOperation>

{
    static_assert(Rank <= 12, "Bigger Rank size is not supported!");

    static_assert(((InSrcVectorDim == 0 && MThreadSliceSize % InSrcVectorSize == 0) ||
                   (InSrcVectorDim == 1 && KThreadSliceSize % InSrcVectorSize == 0)) &&
                      (MThreadSliceSize % OutDstVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    using IndexDataType = int32_t;

    static constexpr index_t NumInvariantDim = Rank - NumReduceDim;

    static constexpr index_t NumDTensor = DsDataType::Size();

    static constexpr index_t NumSrcDim = Rank;
    static constexpr index_t NumDstDim = (NumInvariantDim == 0) ? 1 : NumInvariantDim;
    static constexpr bool reduceAllDim = (NumInvariantDim == 0);

    static constexpr index_t M_BlockTileSize = BlockSize * MThreadSliceSize;
    static constexpr index_t K_BlockTileSize = 1 * KThreadSliceSize;

    static auto MakeSrc2dDescriptor(const std::array<index_t, Rank>& inLengths,
                                    const std::array<index_t, Rank>& inStrides)
    {
        const auto tupleSrcLengths =
            generate_tuple([&](auto I) { return inLengths[I]; }, Number<Rank>{});
        const auto tupleSrcStrides =
            generate_tuple([&](auto I) { return inStrides[I]; }, Number<Rank>{});

        const auto inDesc = make_naive_tensor_descriptor(tupleSrcLengths, tupleSrcStrides);

        const auto in_grid_desc_m_k = [&]() {
            if constexpr(reduceAllDim)
            {
                const auto one_dim_inDesc = transform_tensor_descriptor(
                    inDesc,
                    make_tuple(make_merge_transform(tupleSrcLengths)),
                    make_tuple(typename arithmetic_sequence_gen<0, NumSrcDim, 1>::type{}),
                    make_tuple(Sequence<0>{}));

                return transform_tensor_descriptor(one_dim_inDesc,
                                                   make_tuple(make_unmerge_transform(make_tuple(
                                                       1, one_dim_inDesc.GetLength(Number<0>{})))),
                                                   make_tuple(Sequence<0>{}),
                                                   make_tuple(Sequence<0, 1>{}));
            }
            else
            {
                using InvariantDims = typename arithmetic_sequence_gen<0, NumInvariantDim, 1>::type;
                using ReduceDims = typename arithmetic_sequence_gen<NumInvariantDim, Rank, 1>::type;

                const auto reduceDimLengths = generate_tuple(
                    [&](auto I) { return inLengths[NumInvariantDim + I]; }, Number<NumReduceDim>{});
                const auto invariantDimLengths =
                    generate_tuple([&](auto I) { return inLengths[I]; }, Number<NumInvariantDim>{});

                return transform_tensor_descriptor(
                    inDesc,
                    make_tuple(make_merge_transform(invariantDimLengths),
                               make_merge_transform(reduceDimLengths)),
                    make_tuple(InvariantDims{}, ReduceDims{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }();

        const auto invariantLength = in_grid_desc_m_k.GetLength(Number<0>{});
        const auto reduceLength    = in_grid_desc_m_k.GetLength(Number<1>{});

        const auto inPad_M =
            math::integer_least_multiple(invariantLength, M_BlockTileSize) - invariantLength;
        const auto inPad_K =
            math::integer_least_multiple(reduceLength, K_BlockTileSize) - reduceLength;

        auto in_grid_desc_m_k_padded = transform_tensor_descriptor(
            in_grid_desc_m_k,
            make_tuple(make_right_pad_transform(invariantLength, inPad_M),
                       make_right_pad_transform(reduceLength, inPad_K)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return (in_grid_desc_m_k_padded);
    };

    static auto MakeDst1dDescriptor(const std::array<index_t, NumDstDim>& outLengths,
                                    const std::array<index_t, NumDstDim>& outStrides)
    {
        const auto tupleDstLengths =
            generate_tuple([&](auto I) { return outLengths[I]; }, Number<NumDstDim>{});
        const auto tupleDstStrides =
            generate_tuple([&](auto I) { return outStrides[I]; }, Number<NumDstDim>{});

        auto outDesc = make_naive_tensor_descriptor(tupleDstLengths, tupleDstStrides);

        auto out_grid_desc_m = transform_tensor_descriptor(
            outDesc,
            make_tuple(make_merge_transform(tupleDstLengths)),
            make_tuple(typename arithmetic_sequence_gen<0, NumDstDim, 1>::type{}),
            make_tuple(Sequence<0>{}));

        const auto invariantLength = out_grid_desc_m.GetLength(Number<0>{});

        const auto outPad =
            math::integer_least_multiple(invariantLength, M_BlockTileSize) - invariantLength;

        auto out_grid_desc_m_padded = transform_tensor_descriptor(
            out_grid_desc_m,
            make_tuple(make_right_pad_transform(invariantLength, outPad)),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0>{}));
        return (out_grid_desc_m_padded);
    };

    static auto
    MakeDsDescriptor(const std::array<std::array<index_t, NumDstDim>, NumDTensor> DsLengths,
                     std::array<std::array<index_t, NumDstDim>, NumDTensor> DsStrides)
    {
        return generate_tuple(
            [&](auto i) {
                return DeviceReduceThreadWiseMultiD::MakeDst1dDescriptor(DsLengths[i],
                                                                         DsStrides[i]);
            },
            Number<NumDTensor>{});
    }

    using InGridDesc_M_K = decltype(MakeSrc2dDescriptor({}, {}));
    using OutGridDesc_M  = decltype(MakeDst1dDescriptor({}, {}));
    using DsGridDesc_M   = decltype(MakeDsDescriptor({}, {}));

    using GridwiseReduce =
        GridwiseReduction_mk_to_m_threadwise_multi_d<InDataType,
                                                     DsDataType,
                                                     OutDataType,
                                                     AccDataType,
                                                     InGridDesc_M_K,
                                                     DsGridDesc_M,
                                                     OutGridDesc_M,
                                                     ReduceOperation,
                                                     InElementwiseOperation,
                                                     OutElementwiseOperation,
                                                     InMemoryDataOperationEnum::Set,
                                                     BlockSize,
                                                     MThreadSliceSize,
                                                     KThreadSliceSize,
                                                     InSrcVectorDim,
                                                     InSrcVectorSize,
                                                     OutDstVectorSize,
                                                     DsVectorSizeSequence>;

    using DsGridPointer = typename GridwiseReduce::DsGridPointer;

    struct Argument : public BaseArgument
    {
        Argument(const std::array<index_t, Rank> inLengths,
                 const std::array<index_t, Rank> inStrides,
                 const std::array<std::array<index_t, NumDstDim>, NumDTensor> DsLengths,
                 const std::array<std::array<index_t, NumDstDim>, NumDTensor> DsStrides,
                 const std::array<index_t, NumDstDim> outLengths,
                 const std::array<index_t, NumDstDim> outStrides,
                 const std::array<int, NumReduceDim> reduceDims,
                 const InDataType* in_dev,
                 const std::array<const void*, NumDTensor> ds_dev,
                 OutDataType* out_dev,
                 const InElementwiseOperation in_elementwise_op,
                 const OutElementwiseOperation out_elementwise_op)
            : DsLengths_{DsLengths},
              DsStrides_{DsStrides},
              outLengths_{outLengths},
              outStrides_{outStrides},
              in_dev_{in_dev},
              out_dev_{out_dev},
              in_elementwise_op_{in_elementwise_op},
              out_elementwise_op_{out_elementwise_op}
        {
            inLengths_ = shuffle_tensor_dimensions<Rank, NumReduceDim>(inLengths, reduceDims);
            inStrides_ = shuffle_tensor_dimensions<Rank, NumReduceDim>(inStrides, reduceDims);

            std::tie(invariant_total_length, reduce_total_length) =
                get_2d_lengths<Rank, NumReduceDim>(inLengths_);

            if constexpr(NumInvariantDim == 0)
                invariant_lowest_length = 1;
            else
                invariant_lowest_length = inLengths_[NumInvariantDim - 1];

            reduce_lowest_length = inLengths_[Rank - 1];

            numBlockTileIteration = (reduce_total_length + K_BlockTileSize - 1) / K_BlockTileSize;

            gridSize = math::integer_least_multiple(invariant_total_length, M_BlockTileSize) /
                       M_BlockTileSize;

            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
                p_ds_grid_(i)   = static_cast<const DDataType*>(ds_dev[i]);
            });

            ds_grid_desc_m_ = MakeDsDescriptor(DsLengths, DsStrides);
        }

        std::array<index_t, Rank> inLengths_;
        std::array<index_t, Rank> inStrides_;

        std::array<std::array<index_t, NumDstDim>, NumDTensor> DsLengths_;
        std::array<std::array<index_t, NumDstDim>, NumDTensor> DsStrides_;

        std::array<index_t, NumDstDim> outLengths_;
        std::array<index_t, NumDstDim> outStrides_;

        const InDataType* in_dev_;
        OutDataType* out_dev_;

        DsGridPointer p_ds_grid_;

        InElementwiseOperation in_elementwise_op_;
        OutElementwiseOperation out_elementwise_op_;

        DsGridDesc_M ds_grid_desc_m_;

        index_t invariant_lowest_length;
        index_t reduce_lowest_length;
        long_index_t invariant_total_length;
        long_index_t reduce_total_length;

        int numBlockTileIteration;
        size_t gridSize;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const auto in_grid_desc_m_k =
                DeviceReduceThreadWiseMultiD::MakeSrc2dDescriptor(arg.inLengths_, arg.inStrides_);
            const auto out_grid_desc_m =
                DeviceReduceThreadWiseMultiD::MakeDst1dDescriptor(arg.outLengths_, arg.outStrides_);

            float avg_time = 0;

            const auto kernel = kernel_reduce_threadwise_multi_d<GridwiseReduce,
                                                                 InDataType,
                                                                 OutDataType,
                                                                 AccDataType,
                                                                 InGridDesc_M_K,
                                                                 DsGridDesc_M,
                                                                 OutGridDesc_M,
                                                                 InElementwiseOperation,
                                                                 OutElementwiseOperation,
                                                                 DsGridPointer>;

            avg_time = launch_and_time_kernel(stream_config,
                                              kernel,
                                              dim3(arg.gridSize),
                                              dim3(BlockSize),
                                              0,
                                              in_grid_desc_m_k,
                                              arg.ds_grid_desc_m_,
                                              out_grid_desc_m,
                                              arg.in_elementwise_op_,
                                              arg.out_elementwise_op_,
                                              arg.in_dev_,
                                              arg.p_ds_grid_,
                                              arg.out_dev_);

            return (avg_time);
        };

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        };
    };

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        const Argument* pArg = dynamic_cast<const Argument*>(p_arg);

        if constexpr(InSrcVectorDim == 0)
        {
            if constexpr(NumInvariantDim == 0)
            {
                return (false);
            }
            else
            {
                if(pArg->inStrides_[NumInvariantDim - 1] != 1)
                    return (false);

                if(pArg->invariant_lowest_length % InSrcVectorSize != 0)
                    return (false);
            };
        }
        else
        {
            if(pArg->inStrides_[Rank - 1] != 1)
                return (false);

            if(pArg->reduce_lowest_length % InSrcVectorSize != 0)
                return (false);
        };

        // To improve
        if(pArg->invariant_lowest_length % OutDstVectorSize != 0)
            return (false);

        std::cerr << "reduce_total_length = " << pArg->reduce_total_length
                  << " KThreadSliceSize = " << KThreadSliceSize << std::endl;

        // cases with big reduce_total_length should be handled by Blockwise kernel
        if(pArg->reduce_total_length / KThreadSliceSize >= 32)
            return (false);

        return (true);
    };

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::array<index_t, Rank> inLengths,
                        const std::array<index_t, Rank> inStrides,
                        const std::array<std::array<index_t, NumDstDim>, NumDTensor> DsLengths,
                        const std::array<std::array<index_t, NumDstDim>, NumDTensor> DsStrides,
                        const std::array<index_t, NumDstDim> outLengths,
                        const std::array<index_t, NumDstDim> outStrides,
                        const std::array<int, NumReduceDim> reduceDims,
                        const void* in_dev,
                        const std::array<const void*, NumDTensor> ds_dev,
                        void* out_dev,
                        const InElementwiseOperation in_elementwise_op,
                        const OutElementwiseOperation out_elementwise_op) override
    {
        return std::make_unique<Argument>(inLengths,
                                          inStrides,
                                          DsLengths,
                                          DsStrides,
                                          outLengths,
                                          outStrides,
                                          reduceDims,
                                          static_cast<const InDataType*>(in_dev),
                                          ds_dev,
                                          static_cast<OutDataType*>(out_dev),
                                          in_elementwise_op,
                                          out_elementwise_op);
    };

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    };

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceReduceThreadWiseMultiD<" << BlockSize << ",";
        str << "M_C" << BlockSize << "_S" << MThreadSliceSize << ",";
        str << "K_C" << 1 << "_S" << KThreadSliceSize << ",";
        str << "InSrcVectorDim_" << InSrcVectorDim << "_InSrcVectorSize_" << InSrcVectorSize << "_OutDstVectorSize_" << OutDstVectorSize << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
