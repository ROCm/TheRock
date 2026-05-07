// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>

#include <hipdnn_data_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_data_sdk/utilities/ScopedResource.hpp>
#include <miopen/miopen.h>

#include "MiopenConvDescriptor.hpp"
#include "MiopenTensor.hpp"
#include "PlanInterface.hpp"

namespace miopen_legacy_plugin
{

class ConvFwdParams
{
public:
    ConvFwdParams(
        const hipdnn_data_sdk::data_objects::ConvolutionFwdAttributes& attributes,
        const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    ConvFwdParams(const ConvFwdParams&) = delete;
    ConvFwdParams& operator=(const ConvFwdParams&) = delete;

    ConvFwdParams(ConvFwdParams&&) = default;
    ConvFwdParams& operator=(ConvFwdParams&&) = default;

    const MiopenTensor& x() const;
    const MiopenTensor& w() const;
    const MiopenTensor& y() const;
    const MiopenConvDescriptor& conv() const;

    bool validTensors() const;

private:
    size_t _spatialDimCount;
    MiopenTensor _x;
    MiopenTensor _w;
    MiopenTensor _y;
    MiopenConvDescriptor _conv;
    bool _tensorsValid;
};

class ConvFwdPlan : public IPlan
{
public:
    ConvFwdPlan(const HipdnnEnginePluginHandle& handle, ConvFwdParams&& params);
    ~ConvFwdPlan() override = default;

    ConvFwdPlan(const ConvFwdPlan&) = delete;
    ConvFwdPlan& operator=(const ConvFwdPlan&) = delete;

    ConvFwdPlan(ConvFwdPlan&& other) = default;
    ConvFwdPlan& operator=(ConvFwdPlan&& other) = default;

    size_t getWorkspaceSize(const HipdnnEnginePluginHandle& handle) const override;

    void execute(const HipdnnEnginePluginHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    ConvFwdParams _params;
    hipdnn_data_sdk::utilities::ScopedResource<miopenSolution_t> _solution;
    size_t _workspaceSize = 0;
};

} // namespace miopen_legacy_plugin
