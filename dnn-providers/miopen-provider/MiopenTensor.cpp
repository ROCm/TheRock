// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenTensor.hpp"
#include "MiopenUtils.hpp"
#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace miopen_legacy_plugin
{

MiopenTensor::MiopenTensor(const hipdnn_data_sdk::data_objects::TensorAttributes& tensor)
    : _uid(tensor.uid())
{
    THROW_ON_MIOPEN_FAILURE(miopenCreateTensorDescriptor(&_descriptor));

    PLUGIN_THROW_IF_NULL(tensor.dims(),
                         HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                         "Tensor dims pointer is null for tensor UID: " + std::to_string(_uid));
    PLUGIN_THROW_IF_NULL(tensor.strides(),
                         HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                         "Tensor strides pointer is null for tensor UID: " + std::to_string(_uid));
    std::vector<int> dims(tensor.dims()->begin(), tensor.dims()->end());
    std::vector<int> strides(tensor.strides()->begin(), tensor.strides()->end());
    THROW_ON_MIOPEN_FAILURE(
        miopenSetTensorDescriptor(_descriptor,
                                  miopen_utils::tensorDataTypeToMiopenDataType(tensor.data_type()),
                                  static_cast<int>(dims.size()),
                                  reinterpret_cast<int*>(dims.data()),
                                  reinterpret_cast<int*>(strides.data())));
}

MiopenTensor::MiopenTensor(MiopenTensor&& other) noexcept
    : _uid(other._uid)
    , _descriptor(other._descriptor)
{
    other._descriptor = nullptr;
}

MiopenTensor& MiopenTensor::operator=(MiopenTensor&& other) noexcept
{
    if(this != &other)
    {
        if(_descriptor != nullptr)
        {
            LOG_ON_MIOPEN_FAILURE(miopenDestroyTensorDescriptor(_descriptor));
        }

        _uid = other._uid;
        _descriptor = other._descriptor;

        other._descriptor = nullptr;
    }
    return *this;
}

MiopenTensor::~MiopenTensor()
{
    if(_descriptor != nullptr)
    {
        LOG_ON_MIOPEN_FAILURE(miopenDestroyTensorDescriptor(_descriptor));
    }
}

int64_t MiopenTensor::uid() const
{
    return _uid;
}

miopenTensorDescriptor_t MiopenTensor::tensorDescriptor() const
{
    return _descriptor;
}

}
