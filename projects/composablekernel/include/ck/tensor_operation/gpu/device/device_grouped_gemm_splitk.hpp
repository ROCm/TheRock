// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "device_grouped_gemm.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct DeviceGroupedGemmSplitK : public DeviceGroupedGemm<ALayout,
                                                          BLayout,
                                                          DsLayout,
                                                          ELayout,
                                                          ADataType,
                                                          BDataType,
                                                          DsDataType,
                                                          EDataType,
                                                          AElementwiseOperation,
                                                          BElementwiseOperation,
                                                          CElementwiseOperation>
{
    //----------------------------------------------------------------------------------------------
    /// @brief      Sets the k batch size.
    ///
    /// @param      p_arg   Pointer to the Argument we're going to change.
    /// @param[in]  kbatch  The kbatch value.
    ///
    virtual void SetKBatchSize(BaseArgument* p_arg, index_t kbatch) const = 0;
    //----------------------------------------------------------------------------------------------
    /// @brief      Sets the k batch size.
    ///
    /// @param      p_arg   Pointer to the Argument we're going to change.
    /// @param[in]  kbatch  The kbatch value.
    ///
    virtual void SetKBatch(BaseArgument* p_arg, index_t kbatch) const
    {
        this->SetKBatchSize(p_arg, kbatch);
    };
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
