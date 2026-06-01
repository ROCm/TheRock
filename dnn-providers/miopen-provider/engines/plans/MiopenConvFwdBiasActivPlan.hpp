// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <optional>

#include <hipdnn_data_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_data_sdk/utilities/ScopedResource.hpp>
#include <miopen/miopen.h>

#include "MiopenConvDescriptor.hpp"
#include "MiopenTensor.hpp"
#include "MiopenUtils.hpp"
#include "PlanInterface.hpp"

namespace miopen_legacy_plugin
{

class ConvFwdBiasActivParams
{
public:
    ConvFwdBiasActivParams(
        const hipdnn_data_sdk::data_objects::ConvolutionFwdAttributes& convAttr,
        const hipdnn_data_sdk::data_objects::PointwiseAttributes* biasAttr,
        const hipdnn_data_sdk::data_objects::PointwiseAttributes& activAttr,
        const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
            tensorMap);
    ConvFwdBiasActivParams(const ConvFwdBiasActivParams&) = delete;
    ConvFwdBiasActivParams& operator=(const ConvFwdBiasActivParams&) = delete;

    ConvFwdBiasActivParams(ConvFwdBiasActivParams&&) = default;
    ConvFwdBiasActivParams& operator=(ConvFwdBiasActivParams&&) = default;

    const MiopenTensor& x() const;
    const MiopenTensor& w() const;
    const MiopenConvDescriptor& conv() const;
    const std::optional<MiopenTensor>& bias() const;
    const miopen_utils::ActivationParams& activParams() const;
    const MiopenTensor& y() const;

private:
    size_t _spatialDimCount;
    MiopenTensor _x;
    MiopenTensor _w;
    MiopenConvDescriptor _conv;
    std::optional<MiopenTensor> _bias;
    miopen_utils::ActivationParams _activParams;
    MiopenTensor _y;
};

class ConvFwdBiasActivPlan : public IPlan
{
public:
    ConvFwdBiasActivPlan(const HipdnnEnginePluginHandle& handle,
                         ConvFwdBiasActivParams&& params,
                         bool compile = true,
                         bool getWsSize = true);
    ~ConvFwdBiasActivPlan() override = default;

    ConvFwdBiasActivPlan(const ConvFwdBiasActivPlan&) = delete;
    ConvFwdBiasActivPlan& operator=(const ConvFwdBiasActivPlan&) = delete;

    ConvFwdBiasActivPlan(ConvFwdBiasActivPlan&& other) = default;
    ConvFwdBiasActivPlan& operator=(ConvFwdBiasActivPlan&& other) = default;

    size_t getWorkspaceSize(const HipdnnEnginePluginHandle& handle) const override;

    void execute(const HipdnnEnginePluginHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    ConvFwdBiasActivParams _params;
    hipdnn_data_sdk::utilities::ScopedResource<miopenFusionPlanDescriptor_t> _fusePlanDesc;
    size_t _workspaceSize = 0;
};

} // namespace miopen_legacy_plugin
