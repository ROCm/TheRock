// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include "MiopenActivationDescriptor.hpp"
#include "MiopenTensor.hpp"
#include "MiopenUtils.hpp"
#include "PlanBuilderInterface.hpp" //temporary
#include "PlanInterface.hpp"

namespace miopen_legacy_plugin
{

class BatchnormFwdInferenceParams
{
public:
    BatchnormFwdInferenceParams(
        const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes& attributes,
        const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    BatchnormFwdInferenceParams(
        const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes& inferenceAttributes,
        const hipdnn_data_sdk::data_objects::PointwiseAttributes& pointwiseAttributes,
        const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    BatchnormFwdInferenceParams(const BatchnormFwdInferenceParams&) = delete;
    BatchnormFwdInferenceParams& operator=(const BatchnormFwdInferenceParams&) = delete;

    BatchnormFwdInferenceParams(BatchnormFwdInferenceParams&&) = default;
    BatchnormFwdInferenceParams& operator=(BatchnormFwdInferenceParams&&) = default;

    const MiopenTensor& x() const;
    const MiopenTensor& y() const;
    const MiopenTensor& scale() const;
    const MiopenTensor& bias() const;
    const MiopenTensor& estMean() const;
    const MiopenTensor& invVariance() const;

    const std::optional<MiopenActivationDescriptor>& optActivation() const;
    const std::optional<MiopenTensor>& activationOut() const;

private:
    MiopenTensor _x;
    MiopenTensor _y;
    MiopenTensor _scale;
    MiopenTensor _bias;
    MiopenTensor _estMean;
    MiopenTensor _invVariance;

    std::optional<MiopenActivationDescriptor> _optActivation;
    std::optional<MiopenTensor> _activationOut;
};

class BatchnormFwdInferencePlan : public IPlan
{
public:
    BatchnormFwdInferencePlan(BatchnormFwdInferenceParams&& inferenceParams);

    BatchnormFwdInferencePlan(const BatchnormFwdInferencePlan&) = delete;
    BatchnormFwdInferencePlan& operator=(const BatchnormFwdInferencePlan&) = delete;

    BatchnormFwdInferencePlan(BatchnormFwdInferencePlan&&) = default;
    BatchnormFwdInferencePlan& operator=(BatchnormFwdInferencePlan&&) = default;

    size_t getWorkspaceSize(const HipdnnEnginePluginHandle& handle) const override;

    void execute(const HipdnnEnginePluginHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    BatchnormFwdInferenceParams _inferenceParams;
};

}
