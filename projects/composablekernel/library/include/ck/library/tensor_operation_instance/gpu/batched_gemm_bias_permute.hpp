// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <vector>
#include <memory>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_contraction_multiple_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
#ifdef CK_ENABLE_FP16
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_batched_contraction_bias_permute_m2_n3_k1_xdl_c_shuffle_f16_f16_f16_f16_mnnm_instance(
    std::vector<std::unique_ptr<
        DeviceBatchedContractionMultipleD<1,
                                          2,
                                          3,
                                          1,
                                          F16,
                                          F16,
                                          F16_Tuple,
                                          F16,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::Add>>>& instances);

// Contraction + add
template <index_t NumDimG,
          index_t NumDimM,
          index_t NumDimN,
          index_t NumDimK,
          typename ADataType,
          typename BDataType,
          typename DDataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceBatchedContractionMultipleD<
        NumDimG,
        NumDimM,
        NumDimN,
        NumDimK,
        ADataType,
        BDataType,
        ck::Tuple<DDataType>,
        EDataType,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::Add>>
{
    using DeviceOp =
        DeviceBatchedContractionMultipleD<NumDimG,
                                          NumDimM,
                                          NumDimN,
                                          NumDimK,
                                          ADataType,
                                          BDataType,
                                          ck::Tuple<DDataType>,
                                          EDataType,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::Add>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<ADataType, ck::half_t> && is_same_v<BDataType, ck::half_t> &&
                     is_same_v<DDataType, ck::half_t> && is_same_v<EDataType, ck::half_t>)
        {
            if constexpr(NumDimG == 1 && NumDimM == 2 && NumDimN == 3 && NumDimK == 1)
            {
                add_device_batched_contraction_bias_permute_m2_n3_k1_xdl_c_shuffle_f16_f16_f16_f16_mnnm_instance(
                    op_ptrs);
            }
        }

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
