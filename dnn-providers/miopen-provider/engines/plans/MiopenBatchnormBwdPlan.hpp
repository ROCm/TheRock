// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <optional>

#include <hipdnn_data_sdk/data_objects/batchnorm_backward_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>

#include "MiopenActivationDescriptor.hpp"
#include "MiopenTensor.hpp"
#include "PlanInterface.hpp"

namespace miopen_legacy_plugin
{

class BatchnormBwdParams
{
public:
    BatchnormBwdParams(
        const hipdnn_data_sdk::data_objects::BatchnormBackwardAttributes& attributes,
        const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    BatchnormBwdParams(
        const hipdnn_data_sdk::data_objects::BatchnormBackwardAttributes&
            batchnormBackwardAttributes,
        const hipdnn_data_sdk::data_objects::PointwiseAttributes& pointwiseAttributes,
        const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes&
            batchnormInferenceAttributes,
        const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    BatchnormBwdParams(const BatchnormBwdParams&) = delete;
    BatchnormBwdParams& operator=(const BatchnormBwdParams&) = delete;

    BatchnormBwdParams(BatchnormBwdParams&&) = default;
    BatchnormBwdParams& operator=(BatchnormBwdParams&&) = default;

    const MiopenTensor& x() const;
    const MiopenTensor& dy() const;
    const MiopenTensor& dx() const;
    const MiopenTensor& scale() const;
    const MiopenTensor& dscale() const;
    const MiopenTensor& dbias() const;

    const std::optional<MiopenTensor>& optMean() const;
    const std::optional<MiopenTensor>& optInvVariance() const;
    const std::optional<MiopenActivationDescriptor>& optActivation() const;
    const std::optional<MiopenTensor>& optBias() const;

private:
    MiopenTensor _x;
    MiopenTensor _dy;
    MiopenTensor _dx;
    MiopenTensor _scale;
    MiopenTensor _dscale;
    MiopenTensor _dbias;

    std::optional<MiopenTensor> _optMean;
    std::optional<MiopenTensor> _optInvVariance;
    std::optional<MiopenActivationDescriptor> _optActivation;
    std::optional<MiopenTensor> _optBias;
};

class BatchnormBwdPlan : public IPlan
{
public:
    BatchnormBwdPlan(BatchnormBwdParams&& params);

    BatchnormBwdPlan(const BatchnormBwdPlan&) = delete;
    BatchnormBwdPlan& operator=(const BatchnormBwdPlan&) = delete;

    BatchnormBwdPlan(BatchnormBwdPlan&&) = default;
    BatchnormBwdPlan& operator=(BatchnormBwdPlan&&) = default;

    size_t getWorkspaceSize(const HipdnnEnginePluginHandle& handle) const override;

    void execute(const HipdnnEnginePluginHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    BatchnormBwdParams _params;
};

} // namespace miopen_legacy_plugin
