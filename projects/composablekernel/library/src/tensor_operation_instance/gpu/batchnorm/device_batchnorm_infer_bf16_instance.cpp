// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/ck.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_elementwise_dynamic_vector_dims_impl.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using BF16 = ck::bhalf_t;
using F32  = float;

using Normalize = ck::tensor_operation::element_wise::NormalizeInInfer;

// clang-format off
template <index_t Rank>
using device_batchnorm_infer_bf16_instances =
     std::tuple <
        // Tuple<XDataType, MeanDataType, VarDataType, ScaleDataType, BiasDataType>, Tuple<YDataType>, NormalizeOp, Rank, BlockSize, MPerBlock, NPerBlock, MPerThread, NPerThread, ThreadClusterArrangerOrder, Sequence<XVectorSize, MeanDataType, VarDataType, ScaleVectorSize, BiasVectorSize>, Sequence<YVectorSize> 
        DeviceElementwiseImpl<Tuple<BF16, F32, F32, BF16, BF16>, Tuple<BF16>, Normalize, Rank, 64,  8,  8, 1, 1, ck::Sequence<1, 0>, Sequence<1, 1, 1, 1, 1>, Sequence<1> >,
        DeviceElementwiseImpl<Tuple<BF16, F32, F32, BF16, BF16>, Tuple<BF16>, Normalize, Rank, 64, 16, 16, 2, 2, ck::Sequence<1, 0>, Sequence<1, 1, 1, 1, 1>, Sequence<1> >,
        DeviceElementwiseImpl<Tuple<BF16, F32, F32, BF16, BF16>, Tuple<BF16>, Normalize, Rank, 64, 16, 16, 2, 2, ck::Sequence<1, 0>, Sequence<2, 1, 1, 1, 1>, Sequence<2> >,
        DeviceElementwiseImpl<Tuple<BF16, F32, F32, BF16, BF16>, Tuple<BF16>, Normalize, Rank, 64, 16, 16, 2, 2, ck::Sequence<1, 0>, Sequence<1, 2, 2, 2, 2>, Sequence<1> >,
        DeviceElementwiseImpl<Tuple<BF16, F32, F32, BF16, BF16>, Tuple<BF16>, Normalize, Rank, 64, 16, 16, 2, 2, ck::Sequence<1, 0>, Sequence<2, 2, 2, 2, 2>, Sequence<2> >,
        DeviceElementwiseImpl<Tuple<BF16, F32, F32, BF16, BF16>, Tuple<BF16>, Normalize, Rank, 64, 32, 32, 4, 4, ck::Sequence<1, 0>, Sequence<1, 1, 1, 1, 1>, Sequence<1> >,
        DeviceElementwiseImpl<Tuple<BF16, F32, F32, BF16, BF16>, Tuple<BF16>, Normalize, Rank, 64, 32, 32, 4, 4, ck::Sequence<1, 0>, Sequence<2, 1, 1, 1, 1>, Sequence<2> >,
        DeviceElementwiseImpl<Tuple<BF16, F32, F32, BF16, BF16>, Tuple<BF16>, Normalize, Rank, 64, 32, 32, 4, 4, ck::Sequence<1, 0>, Sequence<1, 2, 2, 2, 2>, Sequence<1> >,
        DeviceElementwiseImpl<Tuple<BF16, F32, F32, BF16, BF16>, Tuple<BF16>, Normalize, Rank, 64, 32, 32, 4, 4, ck::Sequence<1, 0>, Sequence<2, 2, 2, 2, 2>, Sequence<2> >,
        DeviceElementwiseImpl<Tuple<BF16, F32, F32, BF16, BF16>, Tuple<BF16>, Normalize, Rank, 64, 32, 32, 4, 4, ck::Sequence<1, 0>, Sequence<4, 1, 1, 1, 1>, Sequence<4> >,
        DeviceElementwiseImpl<Tuple<BF16, F32, F32, BF16, BF16>, Tuple<BF16>, Normalize, Rank, 64, 32, 32, 4, 4, ck::Sequence<1, 0>, Sequence<1, 4, 4, 4, 4>, Sequence<1> >,
        DeviceElementwiseImpl<Tuple<BF16, F32, F32, BF16, BF16>, Tuple<BF16>, Normalize, Rank, 64, 32, 32, 4, 4, ck::Sequence<1, 0>, Sequence<4, 2, 2, 2, 2>, Sequence<4> >,
        DeviceElementwiseImpl<Tuple<BF16, F32, F32, BF16, BF16>, Tuple<BF16>, Normalize, Rank, 64, 32, 32, 4, 4, ck::Sequence<1, 0>, Sequence<2, 4, 4, 4, 4>, Sequence<2> >,
        DeviceElementwiseImpl<Tuple<BF16, F32, F32, BF16, BF16>, Tuple<BF16>, Normalize, Rank, 64, 32, 32, 4, 4, ck::Sequence<1, 0>, Sequence<4, 4, 4, 4, 4>, Sequence<4> >
     >;
// clang-format on

void add_device_batchnorm_infer_rank_4_bf16_instances(
    std::vector<std::unique_ptr<
        DeviceElementwise<Tuple<BF16, F32, F32, BF16, BF16>, Tuple<BF16>, Normalize, 4>>>&
        instances)
{
    add_device_operation_instances(instances, device_batchnorm_infer_bf16_instances<4>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
