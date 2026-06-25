// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <numeric>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "MiopenBatchnormApplicabilityChecks.hpp"
#include "MiopenUtils.hpp"

namespace miopen_legacy_plugin
{

// --- Type Configuration Helpers ---

std::unordered_set<hipdnn_data_sdk::data_objects::DataType> bn_type_configs::getAllowedIoTypes()
{
    std::unordered_set<hipdnn_data_sdk::data_objects::DataType> types;
    for(const auto& config : VALID)
    {
        types.insert(config.io);
    }
    return types;
}

std::unordered_set<hipdnn_data_sdk::data_objects::DataType> bn_type_configs::getAllowedAffineTypes()
{
    std::unordered_set<hipdnn_data_sdk::data_objects::DataType> types;
    for(const auto& config : VALID)
    {
        types.insert(config.affine);
    }
    return types;
}

std::unordered_set<hipdnn_data_sdk::data_objects::DataType> bn_type_configs::getAllowedStatTypes()
{
    std::unordered_set<hipdnn_data_sdk::data_objects::DataType> types;
    for(const auto& config : VALID)
    {
        types.insert(config.stat);
    }
    return types;
}

// --- Tensor Descriptor Implementation ---

BatchnormTensorDescriptor::BatchnormTensorDescriptor(
    const hipdnn_data_sdk::data_objects::TensorAttributes* attr)
    : dims(attr->dims()->begin(), attr->dims()->end())
    , strides(attr->strides()->begin(), attr->strides()->end())
    , strideOrder(hipdnn_data_sdk::utilities::extractStrideOrder(strides))
{
}

bool BatchnormTensorDescriptor::isPacked() const
{
    return hipdnn_data_sdk::utilities::isTensorPacked(dims, strides);
}

// --- Validation Utilities ---

namespace validators
{

void validateDimensionCount(size_t numDims)
{
    constexpr size_t MIN_SUPPORTED_DIMS = 4;
    constexpr size_t MAX_SUPPORTED_DIMS = 5;

    if(numDims < MIN_SUPPORTED_DIMS || numDims > MAX_SUPPORTED_DIMS)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm implementation supports only 4D or 5D tensors.");
    }
}

void validateConsistentDimensions(const std::vector<BatchnormTensorDescriptor>& tensors)
{
    if(tensors.empty())
    {
        return;
    }

    const size_t expectedDims = tensors[0].numDims();
    validateDimensionCount(expectedDims);

    for(size_t i = 1; i < tensors.size(); ++i)
    {
        if(tensors[i].numDims() != expectedDims)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "All tensors for batchnorm must have the same number of dimensions.");
        }
    }
}

void validatePackedTensors(const std::vector<BatchnormTensorDescriptor>& tensors)
{
    for(const auto& tensor : tensors)
    {
        if(!tensor.isPacked())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Batchnorm implementation supports only packed tensors.");
        }
    }
}

void validateSupportedLayout(const std::vector<int64_t>& strideOrder, size_t numDims)
{
    if(numDims == 4)
    {
        const auto layoutNchw = hipdnn_data_sdk::utilities::TensorLayout::NCHW;
        const auto layoutNhwc = hipdnn_data_sdk::utilities::TensorLayout::NHWC;

        if(strideOrder != layoutNchw.strideOrder && strideOrder != layoutNhwc.strideOrder)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Batchnorm implementation supports only NCHW and NHWC layouts for 4D tensors.");
        }
    }
    else
    {
        const auto layoutNcdhw = hipdnn_data_sdk::utilities::TensorLayout::NCDHW;
        const auto layoutNdhwc = hipdnn_data_sdk::utilities::TensorLayout::NDHWC;

        if(strideOrder != layoutNcdhw.strideOrder && strideOrder != layoutNdhwc.strideOrder)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Batchnorm implementation supports only NCDHW and NDHWC layouts for 5D tensors.");
        }
    }
}

void validateConsistentLayouts(const std::vector<BatchnormTensorDescriptor>& tensors)
{
    if(tensors.empty())
    {
        return;
    }

    // Use first tensor with meaningful layout as reference
    int64_t referenceIndex = -1;
    for(size_t i = 0; i < tensors.size(); ++i)
    {
        if(hipdnn_data_sdk::utilities::isLayoutAgnostic(tensors[i].dims))
        {
            continue;
        }

        if(referenceIndex == -1)
        {
            referenceIndex = static_cast<int64_t>(i);
            validateSupportedLayout(tensors[static_cast<size_t>(referenceIndex)].strideOrder,
                                    tensors[static_cast<size_t>(referenceIndex)].numDims());
        }
        else
        {
            if(tensors[i].strideOrder != tensors[static_cast<size_t>(referenceIndex)].strideOrder)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "All tensors for batchnorm must have the same layout.");
            }
        }
    }
}

void validateDataTypeIsSupported(
    hipdnn_data_sdk::data_objects::DataType dataType,
    const std::unordered_set<hipdnn_data_sdk::data_objects::DataType>& allowedTypes,
    const std::string& errorMessage)
{
    if(allowedTypes.count(dataType) > 0)
    {
        return;
    }
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM, errorMessage);
}

void validateConsistentDataTypes(
    const std::vector<int64_t>& tensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    const std::unordered_set<hipdnn_data_sdk::data_objects::DataType>& allowedTypes,
    const std::string& typeErrorMessage,
    const std::string& consistencyErrorMessage)
{
    if(tensorIds.empty())
    {
        return;
    }

    const auto& firstTensor = miopen_utils::findTensorAttributes(tensorMap, tensorIds[0]);
    const auto referenceType = firstTensor.data_type();

    validateDataTypeIsSupported(referenceType, allowedTypes, typeErrorMessage);

    for(size_t i = 1; i < tensorIds.size(); ++i)
    {
        const auto& tensor = miopen_utils::findTensorAttributes(tensorMap, tensorIds[i]);
        if(tensor.data_type() != referenceType)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                           consistencyErrorMessage);
        }
    }
}

void validateFixedDataType(
    const std::vector<int64_t>& tensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    hipdnn_data_sdk::data_objects::DataType expectedType,
    const std::string& errorMessage)
{
    for(const auto tensorId : tensorIds)
    {
        const auto& tensor = miopen_utils::findTensorAttributes(tensorMap, tensorId);
        if(tensor.data_type() != expectedType)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                           errorMessage);
        }
    }
}

void validateConsistentShapes(
    const std::vector<int64_t>& tensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    const std::vector<int64_t>& referenceShape,
    const std::string& errorMessage)
{
    for(const auto tensorId : tensorIds)
    {
        const auto& tensorAttr = miopen_utils::findTensorAttributes(tensorMap, tensorId);
        const std::vector<int64_t> dims(tensorAttr.dims()->begin(), tensorAttr.dims()->end());
        if(dims != referenceShape)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                           errorMessage);
        }
    }
}

void validateSpatialDimensions(const std::vector<int64_t>& ioDims)
{
    if(ioDims.size() < 3)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "IO tensor must have at least 3 dimensions for batchnorm.");
    }

    const auto spatialSize
        = std::accumulate(ioDims.begin() + 2, ioDims.end(), int64_t{1}, std::multiplies<>());

    if(ioDims[0] * spatialSize <= 1)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "The product of the batch size and spatial dimensions must be greater than 1 for "
            "batchnorm.");
    }
}

void validatePeerStatsNotPopulated(const flatbuffers::Vector<int64_t>* peerStatsTensorUid,
                                   const std::string& errorMessage)
{
    if(peerStatsTensorUid != nullptr && !peerStatsTensorUid->empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       errorMessage);
    }
}

} // namespace validators

// --- Component Validators ---

void checkTensorLayoutsAndDimsSupported(
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    // Skip scalars (epsilon, momentum) - their type is validated by MIOpen
    std::vector<BatchnormTensorDescriptor> tensors;
    tensors.reserve(tensorMap.size());

    for(const auto& [id, attr] : tensorMap)
    {
        if(attr->value_type() != hipdnn_data_sdk::data_objects::TensorValue::NONE)
        {
            continue;
        }
        tensors.emplace_back(attr);
    }

    validators::validateConsistentDimensions(tensors);
    validators::validatePackedTensors(tensors);
    validators::validateConsistentLayouts(tensors);
}

void checkTensorDataTypesSupported(
    const std::vector<int64_t>& ioTensorIds,
    const std::vector<int64_t>& affineTensorIds,
    const std::vector<int64_t>& statTensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    validators::validateConsistentDataTypes(
        ioTensorIds,
        tensorMap,
        bn_type_configs::getAllowedIoTypes(),
        "Batchnorm implementation supports only FLOAT, HALF, and BFLOAT16 data types for x, y, "
        "dy, and dx tensors.",
        "All IO tensors for batchnorm must have the same data type.");

    const auto allowedAffineTypes = bn_type_configs::getAllowedAffineTypes();
    if(allowedAffineTypes.size() == 1)
    {
        validators::validateFixedDataType(affineTensorIds,
                                          tensorMap,
                                          *allowedAffineTypes.begin(),
                                          "Batchnorm implementation supports only FLOAT data type "
                                          "for scale and bias tensors.");
    }
    else
    {
        validators::validateConsistentDataTypes(
            affineTensorIds,
            tensorMap,
            allowedAffineTypes,
            "Batchnorm affine tensors use unsupported data type.",
            "All affine tensors for batchnorm must have the same data type.");
    }

    const auto allowedStatTypes = bn_type_configs::getAllowedStatTypes();
    if(allowedStatTypes.size() == 1)
    {
        validators::validateFixedDataType(statTensorIds,
                                          tensorMap,
                                          *allowedStatTypes.begin(),
                                          "Batchnorm implementation supports only FLOAT data type "
                                          "for mean and variance tensors.");
    }
    else
    {
        validators::validateConsistentDataTypes(
            statTensorIds,
            tensorMap,
            allowedStatTypes,
            "Batchnorm stat tensors use unsupported data type.",
            "All stat tensors for batchnorm must have the same data type.");
    }
}

void checkTensorShapesSupported(
    const std::vector<int64_t>& ioTensorIds,
    const std::vector<int64_t>& affineTensorIds,
    const std::vector<int64_t>& statTensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    bool isTraining)
{
    if(ioTensorIds.empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "At least one IO tensor must be provided for batchnorm.");
    }

    const auto& ioTensorAttr = miopen_utils::findTensorAttributes(tensorMap, ioTensorIds[0]);
    const std::vector<int64_t> ioDims(ioTensorAttr.dims()->begin(), ioTensorAttr.dims()->end());

    validators::validateConsistentShapes(
        ioTensorIds, tensorMap, ioDims, "All IO tensors for batchnorm must have the same shape.");

    const auto derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(ioDims);
    validators::validateConsistentShapes(affineTensorIds,
                                         tensorMap,
                                         derivedDims,
                                         "Scale and bias tensors for batchnorm must have shape "
                                         "derived from IO tensor shape.");
    validators::validateConsistentShapes(statTensorIds,
                                         tensorMap,
                                         derivedDims,
                                         "Mean and variance tensors for batchnorm must have shape "
                                         "derived from IO tensor shape.");

    if(isTraining)
    {
        validators::validateSpatialDimensions(ioDims);
    }
}

// --- High-Level Configuration Validators ---

namespace
{

void checkBatchnormTensorConfigSupported(
    const std::vector<int64_t>& ioTensorIds,
    const std::vector<int64_t>& affineTensorIds,
    const std::vector<int64_t>& statTensorIds,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    bool isTraining)
{
    checkTensorLayoutsAndDimsSupported(tensorMap);
    checkTensorDataTypesSupported(ioTensorIds, affineTensorIds, statTensorIds, tensorMap);
    checkTensorShapesSupported(ioTensorIds, affineTensorIds, statTensorIds, tensorMap, isTraining);
}

} // namespace

void checkBatchnormTensorConfigSupported(
    const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    std::vector<int64_t> ioTensorIds = {bnInfAttr.x_tensor_uid(), bnInfAttr.y_tensor_uid()};
    std::vector<int64_t> affineTensorIds
        = {bnInfAttr.scale_tensor_uid(), bnInfAttr.bias_tensor_uid()};
    std::vector<int64_t> statTensorIds
        = {bnInfAttr.mean_tensor_uid(), bnInfAttr.inv_variance_tensor_uid()};

    checkBatchnormTensorConfigSupported(
        ioTensorIds, affineTensorIds, statTensorIds, tensorMap, false);
}

void checkBatchnormTensorConfigSupported(
    const hipdnn_data_sdk::data_objects::BatchnormAttributes& bnAttr,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    validators::validatePeerStatsNotPopulated(
        bnAttr.peer_stats_tensor_uid(),
        "Batchnorm forward training does not support peer statistics");

    std::vector<int64_t> ioTensorIds = {bnAttr.x_tensor_uid(), bnAttr.y_tensor_uid()};
    std::vector<int64_t> affineTensorIds = {bnAttr.scale_tensor_uid(), bnAttr.bias_tensor_uid()};
    std::vector<int64_t> statTensorIds;
    if(bnAttr.mean_tensor_uid().has_value())
    {
        statTensorIds.push_back(bnAttr.mean_tensor_uid().value());
    }
    if(bnAttr.inv_variance_tensor_uid().has_value())
    {
        statTensorIds.push_back(bnAttr.inv_variance_tensor_uid().value());
    }

    checkBatchnormTensorConfigSupported(
        ioTensorIds, affineTensorIds, statTensorIds, tensorMap, true);
}

void checkBatchnormTensorConfigSupported(
    const hipdnn_data_sdk::data_objects::BatchnormBackwardAttributes& bnBwdAttr,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    validators::validatePeerStatsNotPopulated(
        bnBwdAttr.peer_stats_tensor_uid(), "Batchnorm backward does not support peer statistics");

    std::vector<int64_t> ioTensorIds
        = {bnBwdAttr.x_tensor_uid(), bnBwdAttr.dy_tensor_uid(), bnBwdAttr.dx_tensor_uid()};
    std::vector<int64_t> affineTensorIds = {
        bnBwdAttr.scale_tensor_uid(), bnBwdAttr.dscale_tensor_uid(), bnBwdAttr.dbias_tensor_uid()};
    std::vector<int64_t> statTensorIds;
    if(bnBwdAttr.mean_tensor_uid().has_value())
    {
        statTensorIds.push_back(bnBwdAttr.mean_tensor_uid().value());
    }
    if(bnBwdAttr.inv_variance_tensor_uid().has_value())
    {
        statTensorIds.push_back(bnBwdAttr.inv_variance_tensor_uid().value());
    }

    checkBatchnormTensorConfigSupported(
        ioTensorIds, affineTensorIds, statTensorIds, tensorMap, true);
}

void checkBatchnormTensorConfigSupported(
    const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const hipdnn_data_sdk::data_objects::PointwiseAttributes& actAttr,
    const hipdnn_data_sdk::data_objects::BatchnormBackwardAttributes& bnBwdAttr,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    checkBatchnormBwdActivationModeSupported(actAttr);

    validators::validatePeerStatsNotPopulated(
        bnBwdAttr.peer_stats_tensor_uid(),
        "Batchnorm backward fusion does not support peer statistics");

    std::vector<int64_t> ioTensorIds
        = {bnBwdAttr.x_tensor_uid(), actAttr.in_1_tensor_uid().value(), bnBwdAttr.dx_tensor_uid()};
    std::vector<int64_t> affineTensorIds = {bnBwdAttr.scale_tensor_uid(),
                                            bnBwdAttr.dscale_tensor_uid(),
                                            bnBwdAttr.dbias_tensor_uid(),
                                            bnInfAttr.bias_tensor_uid()};
    std::vector<int64_t> statTensorIds;
    if(bnBwdAttr.mean_tensor_uid().has_value())
    {
        statTensorIds.push_back(bnBwdAttr.mean_tensor_uid().value());
    }
    if(bnBwdAttr.inv_variance_tensor_uid().has_value())
    {
        statTensorIds.push_back(bnBwdAttr.inv_variance_tensor_uid().value());
    }

    checkBatchnormTensorConfigSupported(
        ioTensorIds, affineTensorIds, statTensorIds, tensorMap, true);
}

// --- Activation Mode Validators ---

namespace
{

void checkBatchnormActivationModeSupported(
    const hipdnn_data_sdk::data_objects::PointwiseAttributes& activAttr, bool isBwd)
{
    // MIOpen supports: PASSTHRU, RELU, CLIPPEDREU, CLAMP (no Leaky ReLU)

    if(activAttr.operation() == hipdnn_data_sdk::data_objects::PointwiseMode::IDENTITY)
    {
        return;
    }

    if(activAttr.operation()
       == (isBwd ? hipdnn_data_sdk::data_objects::PointwiseMode::RELU_BWD
                 : hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD))
    {
        if(!activAttr.relu_lower_clip_slope())
        {
            return;
        }
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm fused activation does not support Leaky ReLU.");
    }

    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Unsupported activation mode for batchnorm fusion.");
}

} // namespace

void checkBatchnormFwdActivationModeSupported(
    const hipdnn_data_sdk::data_objects::PointwiseAttributes& activAttr)
{
    checkBatchnormActivationModeSupported(activAttr, false);
}

void checkBatchnormBwdActivationModeSupported(
    const hipdnn_data_sdk::data_objects::PointwiseAttributes& activAttr)
{
    checkBatchnormActivationModeSupported(activAttr, true);
}

} // namespace miopen_legacy_plugin
