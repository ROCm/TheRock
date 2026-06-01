// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>

#include <hipdnn_data_sdk/data_objects/convolution_bwd_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_data_sdk/utilities/ScopedResource.hpp>
#include <miopen/miopen.h>

#include "MiopenConvDescriptor.hpp"
#include "MiopenTensor.hpp"
#include "PlanInterface.hpp"

namespace miopen_legacy_plugin
{

class ConvBwdParams
{
public:
    ConvBwdParams(
        const hipdnn_data_sdk::data_objects::ConvolutionBwdAttributes& attributes,
        const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    ConvBwdParams(const ConvBwdParams&) = delete;
    ConvBwdParams& operator=(const ConvBwdParams&) = delete;

    ConvBwdParams(ConvBwdParams&&) = default;
    ConvBwdParams& operator=(ConvBwdParams&&) = default;

    const MiopenTensor& dx() const;
    const MiopenTensor& w() const;
    const MiopenTensor& dy() const;
    const MiopenConvDescriptor& conv() const;

    bool validTensors() const;

private:
    size_t _spatialDimCount;
    MiopenTensor _dx;
    MiopenTensor _w;
    MiopenTensor _dy;
    MiopenConvDescriptor _conv;
    bool _tensorsValid;
};

class ConvBwdPlan : public IPlan
{
public:
    ConvBwdPlan(const HipdnnEnginePluginHandle& handle, ConvBwdParams&& params);
    ~ConvBwdPlan() override = default;

    ConvBwdPlan(const ConvBwdPlan&) = delete;
    ConvBwdPlan& operator=(const ConvBwdPlan&) = delete;

    ConvBwdPlan(ConvBwdPlan&& other) = default;
    ConvBwdPlan& operator=(ConvBwdPlan&& other) = default;

    size_t getWorkspaceSize(const HipdnnEnginePluginHandle& handle) const override;

    void execute(const HipdnnEnginePluginHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    ConvBwdParams _params;
    hipdnn_data_sdk::utilities::ScopedResource<miopenSolution_t> _solution;
    size_t _workspaceSize = 0;
};

} // namespace miopen_legacy_plugin
