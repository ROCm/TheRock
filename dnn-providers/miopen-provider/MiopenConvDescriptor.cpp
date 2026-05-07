// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <limits>
#include <vector>

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "MiopenConvDescriptor.hpp"
#include "MiopenUtils.hpp"

namespace miopen_legacy_plugin
{

namespace
{

void copyWithCheck(const flatbuffers::Vector<int64_t>* src,
                   std::vector<int>& dst,
                   size_t expectedSize,
                   const char* name,
                   const char* expectedSizeName)
{
    if(src->size() != expectedSize)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "MiopenConvDescriptor: " + std::string(name)
                                                           + " size must be equal to "
                                                           + std::string(expectedSizeName));
    }
    if(!std::all_of(src->begin(), src->end(), [](int64_t v) {
           return v <= std::numeric_limits<int>::max();
       }))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "MiopenConvDescriptor: " + std::string(name)
                                                           + " values must be less than INT_MAX");
    }
    std::copy(src->begin(), src->end(), dst.begin());
}

}

MiopenConvDescriptor::MiopenConvDescriptor(
    size_t spatialDimCount,
    const hipdnn_data_sdk::data_objects::ConvolutionFwdAttributes& attributes,
    int groupCount)
{
    createDescriptorInternal(spatialDimCount,
                             attributes.pre_padding(),
                             attributes.post_padding(),
                             attributes.stride(),
                             attributes.dilation(),
                             attributes.conv_mode(),
                             groupCount);
}

MiopenConvDescriptor::MiopenConvDescriptor(
    size_t spatialDimCount,
    const hipdnn_data_sdk::data_objects::ConvolutionBwdAttributes& attributes,
    int groupCount)
{
    createDescriptorInternal(spatialDimCount,
                             attributes.pre_padding(),
                             attributes.post_padding(),
                             attributes.stride(),
                             attributes.dilation(),
                             attributes.conv_mode(),
                             groupCount);
}

MiopenConvDescriptor::MiopenConvDescriptor(
    size_t spatialDimCount,
    const hipdnn_data_sdk::data_objects::ConvolutionWrwAttributes& attributes,
    int groupCount)
{
    createDescriptorInternal(spatialDimCount,
                             attributes.pre_padding(),
                             attributes.post_padding(),
                             attributes.stride(),
                             attributes.dilation(),
                             attributes.conv_mode(),
                             groupCount);
}

MiopenConvDescriptor::MiopenConvDescriptor(MiopenConvDescriptor&& other) noexcept
    : _descriptor(other._descriptor)
{
    other._descriptor = nullptr;
}

MiopenConvDescriptor& MiopenConvDescriptor::operator=(MiopenConvDescriptor&& other) noexcept
{
    if(this != &other)
    {
        if(_descriptor != nullptr)
        {
            LOG_ON_MIOPEN_FAILURE(miopenDestroyConvolutionDescriptor(_descriptor));
        }

        _descriptor = other._descriptor;
        other._descriptor = nullptr;
    }
    return *this;
}

MiopenConvDescriptor::~MiopenConvDescriptor()
{
    if(_descriptor != nullptr)
    {
        LOG_ON_MIOPEN_FAILURE(miopenDestroyConvolutionDescriptor(_descriptor));
    }
}

miopenConvolutionDescriptor_t MiopenConvDescriptor::convDescriptor() const
{
    return _descriptor;
}

void MiopenConvDescriptor::createDescriptorInternal(
    size_t spatialDimCount,
    const flatbuffers::Vector<int64_t>* attrPrePadding,
    const flatbuffers::Vector<int64_t>* attrPostPadding,
    const flatbuffers::Vector<int64_t>* attrStride,
    const flatbuffers::Vector<int64_t>* attrDilation,
    hipdnn_data_sdk::data_objects::ConvMode convMode,
    int groupCount)
{
    if(spatialDimCount > std::numeric_limits<int>::max())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "MiopenConvDescriptor: spatialDimCount must be not greater than INT_MAX");
    }

    if(convMode != hipdnn_data_sdk::data_objects::ConvMode::CROSS_CORRELATION)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "MiopenConvDescriptor: only ConvMode::CROSS_CORRELATION is supported");
    }

    if(attrPrePadding == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "MiopenConvDescriptor: pre_padding must be set");
    }

    if(attrPostPadding == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "MiopenConvDescriptor: post_padding must be set");
    }

    if(attrStride == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "MiopenConvDescriptor: stride must be set");
    }

    if(attrDilation == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "MiopenConvDescriptor: dilation must be set");
    }

    if(attrPrePadding->size() != attrPostPadding->size())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "MiopenConvDescriptor: pre_padding and post_padding sizes must be equal");
    }

    if(!std::equal(attrPrePadding->begin(), attrPrePadding->end(), attrPostPadding->begin()))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "MiopenConvDescriptor: asymmetric padding is not supported");
    }

    if(groupCount < 1)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "MiopenConvDescriptor: groupCount < 1 is not supported");
    }

    std::vector<int> padding(spatialDimCount);
    std::vector<int> stride(spatialDimCount);
    std::vector<int> dilation(spatialDimCount);

    copyWithCheck(attrPrePadding, padding, spatialDimCount, "attrPadding", "spatialDimCount");
    copyWithCheck(attrStride, stride, spatialDimCount, "attrStride", "spatialDimCount");
    copyWithCheck(attrDilation, dilation, spatialDimCount, "attrDilation", "spatialDimCount");

    THROW_ON_MIOPEN_FAILURE(miopenCreateConvolutionDescriptor(&_descriptor));
    THROW_ON_MIOPEN_FAILURE(miopenInitConvolutionNdDescriptor(_descriptor,
                                                              static_cast<int>(spatialDimCount),
                                                              padding.data(),
                                                              stride.data(),
                                                              dilation.data(),
                                                              miopenConvolution));
    THROW_ON_MIOPEN_FAILURE(miopenSetConvolutionGroupCount(_descriptor, groupCount));
}
}
