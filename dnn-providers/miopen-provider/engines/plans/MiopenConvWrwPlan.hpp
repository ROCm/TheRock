// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>

#include <hipdnn_data_sdk/data_objects/convolution_wrw_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_data_sdk/utilities/ScopedResource.hpp>
#include <miopen/miopen.h>

#include "MiopenConvDescriptor.hpp"
#include "MiopenTensor.hpp"
#include "PlanInterface.hpp"

namespace miopen_legacy_plugin
{

class ConvWrwParams
{
public:
    ConvWrwParams(
        const hipdnn_data_sdk::data_objects::ConvolutionWrwAttributes& attributes,
        const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    ConvWrwParams(const ConvWrwParams&) = delete;
    ConvWrwParams& operator=(const ConvWrwParams&) = delete;

    ConvWrwParams(ConvWrwParams&&) = default;
    ConvWrwParams& operator=(ConvWrwParams&&) = default;

    const MiopenTensor& x() const;
    const MiopenTensor& dw() const;
    const MiopenTensor& dy() const;
    const MiopenConvDescriptor& conv() const;

    bool validTensors() const;

private:
    size_t _spatialDimCount;
    MiopenTensor _x;
    MiopenTensor _dw;
    MiopenTensor _dy;
    MiopenConvDescriptor _conv;
    bool _tensorsValid;
};

class ConvWrwPlan : public IPlan
{
public:
    ConvWrwPlan(const HipdnnEnginePluginHandle& handle, ConvWrwParams&& params);
    ~ConvWrwPlan() override = default;

    ConvWrwPlan(const ConvWrwPlan&) = delete;
    ConvWrwPlan& operator=(const ConvWrwPlan&) = delete;

    ConvWrwPlan(ConvWrwPlan&& other) = default;
    ConvWrwPlan& operator=(ConvWrwPlan&& other) = default;

    size_t getWorkspaceSize(const HipdnnEnginePluginHandle& handle) const override;

    void execute(const HipdnnEnginePluginHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    ConvWrwParams _params;
    hipdnn_data_sdk::utilities::ScopedResource<miopenSolution_t> _solution;
    size_t _workspaceSize = 0;
};

} // namespace miopen_legacy_plugin
