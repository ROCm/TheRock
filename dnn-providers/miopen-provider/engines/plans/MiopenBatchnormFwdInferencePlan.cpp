// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenBatchnormFwdInferencePlan.hpp"
#include "MiopenUtils.hpp"

#include <hipdnn_data_sdk/utilities/Constants.hpp>

namespace miopen_legacy_plugin
{

// We have made the intentional decision to hardcode the batchnorm mode to miopenBNSpatial
// rather than making it configurable and adding extra complexity.

// NOTE: BN inference temporarily disabled due to https://github.com/ROCm/rocm-libraries/issues/2459
//const miopenBatchNormMode_t MIOPEN_BATCHNORM_MODE = miopenBNSpatial;

BatchnormFwdInferenceParams::BatchnormFwdInferenceParams(
    const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes& attributes,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _x(miopen_utils::createTensor(tensorMap, attributes.x_tensor_uid()))
    , _y(miopen_utils::createTensor(tensorMap, attributes.y_tensor_uid()))
    , _scale(miopen_utils::createTensor(tensorMap, attributes.scale_tensor_uid()))
    , _bias(miopen_utils::createTensor(tensorMap, attributes.bias_tensor_uid()))
    , _estMean(miopen_utils::createTensor(tensorMap, attributes.mean_tensor_uid()))
    , _invVariance(miopen_utils::createTensor(tensorMap, attributes.inv_variance_tensor_uid()))
{
}

BatchnormFwdInferenceParams::BatchnormFwdInferenceParams(
    const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes& inferenceAttributes,
    const hipdnn_data_sdk::data_objects::PointwiseAttributes& pointwiseAttributes,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _x(miopen_utils::createTensor(tensorMap, inferenceAttributes.x_tensor_uid()))
    , _y(miopen_utils::createTensor(tensorMap, inferenceAttributes.y_tensor_uid()))
    , _scale(miopen_utils::createTensor(tensorMap, inferenceAttributes.scale_tensor_uid()))
    , _bias(miopen_utils::createTensor(tensorMap, inferenceAttributes.bias_tensor_uid()))
    , _estMean(miopen_utils::createTensor(tensorMap, inferenceAttributes.mean_tensor_uid()))
    , _invVariance(
          miopen_utils::createTensor(tensorMap, inferenceAttributes.inv_variance_tensor_uid()))
    , _optActivation(pointwiseAttributes)
    , _activationOut(miopen_utils::createTensor(tensorMap, pointwiseAttributes.out_0_tensor_uid()))
{
}

const MiopenTensor& BatchnormFwdInferenceParams::x() const
{
    return _x;
}

const MiopenTensor& BatchnormFwdInferenceParams::y() const
{
    return _y;
}

const MiopenTensor& BatchnormFwdInferenceParams::scale() const
{
    return _scale;
}

const MiopenTensor& BatchnormFwdInferenceParams::bias() const
{
    return _bias;
}

const MiopenTensor& BatchnormFwdInferenceParams::estMean() const
{
    return _estMean;
}

const MiopenTensor& BatchnormFwdInferenceParams::invVariance() const
{
    return _invVariance;
}

const std::optional<MiopenActivationDescriptor>& BatchnormFwdInferenceParams::optActivation() const
{
    return _optActivation;
}

const std::optional<MiopenTensor>& BatchnormFwdInferenceParams::activationOut() const
{
    return _activationOut;
}

BatchnormFwdInferencePlan::BatchnormFwdInferencePlan(BatchnormFwdInferenceParams&& inferenceParams)
    : _inferenceParams(std::move(inferenceParams))
{
}

size_t BatchnormFwdInferencePlan::getWorkspaceSize(
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle) const
{
    // No workspace needed for batchnorm inference
    return 0;
}

void BatchnormFwdInferencePlan::execute(
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle,
    [[maybe_unused]] const hipdnnPluginDeviceBuffer_t* deviceBuffers,
    [[maybe_unused]] uint32_t numDeviceBuffers,
    [[maybe_unused]] void* workspace) const
{
    // Hardcoded values from bn_driver in miopen
    // auto alpha = static_cast<float>(1);
    // auto beta = static_cast<float>(0);
    // double epsilon = hipdnn_data_sdk::utilities::BATCHNORM_DEFAULT_EPSILON;

    // auto xBuffer = miopen_utils::findDeviceBuffer(
    //     _inferenceParams.x().uid(), deviceBuffers, numDeviceBuffers);
    // auto scaleBuffer = miopen_utils::findDeviceBuffer(
    //     _inferenceParams.scale().uid(), deviceBuffers, numDeviceBuffers);
    // auto biasBuffer = miopen_utils::findDeviceBuffer(
    //     _inferenceParams.bias().uid(), deviceBuffers, numDeviceBuffers);
    // auto estMeanBuffer = miopen_utils::findDeviceBuffer(
    //     _inferenceParams.estMean().uid(), deviceBuffers, numDeviceBuffers);
    // auto invVarianceBuffer = miopen_utils::findDeviceBuffer(
    //     _inferenceParams.invVariance().uid(), deviceBuffers, numDeviceBuffers);

    // if(_inferenceParams.optActivation().has_value() && _inferenceParams.activationOut().has_value())
    // {
    //     auto activationOutBuffer = miopen_utils::findDeviceBuffer(
    //         _inferenceParams.activationOut()->uid(), deviceBuffers, numDeviceBuffers);

    //     THROW_ON_MIOPEN_FAILURE(miopenBatchNormForwardInferenceActivation(
    //         handle.miopenHandle,
    //         MIOPEN_BATCHNORM_MODE,
    //         &alpha,
    //         &beta,
    //         _inferenceParams.x().tensorDescriptor(),
    //         xBuffer.ptr,
    //         _inferenceParams.activationOut().value().tensorDescriptor(),
    //         activationOutBuffer.ptr,
    //         _inferenceParams.scale().tensorDescriptor(),
    //         _inferenceParams.bias().tensorDescriptor(),
    //         _inferenceParams.estMean().tensorDescriptor(),
    //         _inferenceParams.invVariance().tensorDescriptor(),
    //         scaleBuffer.ptr,
    //         biasBuffer.ptr,
    //         estMeanBuffer.ptr,
    //         invVarianceBuffer.ptr,
    //         epsilon,
    //         _inferenceParams.optActivation().value().activationDescriptor()));
    // }
    // else
    // {
    //     auto yBuffer = miopen_utils::findDeviceBuffer(
    //         _inferenceParams.y().uid(), deviceBuffers, numDeviceBuffers);
    //     THROW_ON_MIOPEN_FAILURE(miopenBatchNormalizationForwardInference_V2(
    //         handle.miopenHandle,
    //         MIOPEN_BATCHNORM_MODE,
    //         &alpha,
    //         &beta,
    //         _inferenceParams.x().tensorDescriptor(),
    //         xBuffer.ptr,
    //         _inferenceParams.y().tensorDescriptor(),
    //         yBuffer.ptr,
    //         _inferenceParams.scale().tensorDescriptor(),
    //         _inferenceParams.bias().tensorDescriptor(),
    //         _inferenceParams.estMean().tensorDescriptor(),
    //         _inferenceParams.invVariance().tensorDescriptor(),
    //         scaleBuffer.ptr,
    //         biasBuffer.ptr,
    //         estMeanBuffer.ptr,
    //         invVarianceBuffer.ptr,
    //         epsilon));
    // }
    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
        "BatchnormFwdInferencePlan execution is currently disabled due to implementation issues "
        "with the BatchnormInference op.");
}

}
