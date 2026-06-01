// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/data_objects/convolution_bwd_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/convolution_wrw_attributes_generated.h>
#include <miopen/miopen.h>

namespace miopen_legacy_plugin
{

class MiopenConvDescriptor
{
public:
    MiopenConvDescriptor() = default;
    MiopenConvDescriptor(size_t spatialDimCount,
                         const hipdnn_data_sdk::data_objects::ConvolutionFwdAttributes& attributes,
                         int groupCount);
    MiopenConvDescriptor(size_t spatialDimCount,
                         const hipdnn_data_sdk::data_objects::ConvolutionBwdAttributes& attributes,
                         int groupCount);
    MiopenConvDescriptor(size_t spatialDimCount,
                         const hipdnn_data_sdk::data_objects::ConvolutionWrwAttributes& attributes,
                         int groupCount);

    MiopenConvDescriptor(const MiopenConvDescriptor&) = delete;
    MiopenConvDescriptor& operator=(const MiopenConvDescriptor&) = delete;

    MiopenConvDescriptor(MiopenConvDescriptor&& other) noexcept;
    MiopenConvDescriptor& operator=(MiopenConvDescriptor&& other) noexcept;

    ~MiopenConvDescriptor();

    miopenConvolutionDescriptor_t convDescriptor() const;

private:
    miopenConvolutionDescriptor_t _descriptor = nullptr;

    void createDescriptorInternal(size_t spatialDimCount,
                                  const flatbuffers::Vector<int64_t>* attrPrePadding,
                                  const flatbuffers::Vector<int64_t>* attrPostPadding,
                                  const flatbuffers::Vector<int64_t>* attrStride,
                                  const flatbuffers::Vector<int64_t>* attrDilation,
                                  hipdnn_data_sdk::data_objects::ConvMode convMode,
                                  int groupCount);
};

}
