// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <optional>
#include <unordered_map>

#include <hipdnn_data_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_data_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <miopen/miopen.h>

#include "MiopenTensor.hpp"

#define LOG_ON_MIOPEN_FAILURE(status)                                                    \
    do                                                                                   \
    {                                                                                    \
        if(status != miopenStatusSuccess)                                                \
        {                                                                                \
            HIPDNN_LOG_ERROR("MIOpen error occurred: {}", miopenGetErrorString(status)); \
        }                                                                                \
    } while(0)

#define THROW_ON_MIOPEN_FAILURE(status)                                                 \
    do                                                                                  \
    {                                                                                   \
        if(status != miopenStatusSuccess)                                               \
        {                                                                               \
            throw hipdnn_plugin_sdk::HipdnnPluginException(                             \
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,                                    \
                "MIOpen error occurred: " + std::string(miopenGetErrorString(status))); \
        }                                                                               \
    } while(0)

#define HIPDNN_PREPEND_MESSAGE_ON_THROW(statement, message)                               \
    do                                                                                    \
    {                                                                                     \
        try                                                                               \
        {                                                                                 \
            statement;                                                                    \
        }                                                                                 \
        catch(hipdnn_plugin_sdk::HipdnnPluginException error)                             \
        {                                                                                 \
            throw hipdnn_plugin_sdk::HipdnnPluginException(error.getStatus(),             \
                                                           message + error.getMessage()); \
        }                                                                                 \
    } while(0)

namespace miopen_legacy_plugin::miopen_utils
{

struct ActivationParams
{
    miopenActivationMode_t mode;
    double alpha;
    double beta;
    double gamma;
};

ActivationParams mapPointwiseModeToMiopenActivation(
    const hipdnn_data_sdk::data_objects::PointwiseAttributes& attrs);

hipdnnPluginDeviceBuffer_t findDeviceBuffer(int64_t uid,
                                            const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                            uint32_t numDeviceBuffers);

miopenDataType_t
    tensorDataTypeToMiopenDataType(const hipdnn_data_sdk::data_objects::DataType& dataType);

const hipdnn_data_sdk::data_objects::TensorAttributes& findTensorAttributes(
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    int64_t uid);

MiopenTensor createTensor(
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    int64_t uid);

size_t getSpatialDimCount(const hipdnn_data_sdk::data_objects::TensorAttributes& attr);

using hipdnn_data_sdk::utilities::extractDoubleFromTensorValue;
using hipdnn_data_sdk::utilities::extractValueFromTensorValue;

} // namespace miopen_legacy_plugin::miopen_utils
