// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenBatchnormFwdTrainingPlan.hpp"
#include "MiopenUtils.hpp"
#include <hipdnn_data_sdk/utilities/ScopedResource.hpp>

namespace miopen_legacy_plugin
{

// We have made the intentional decision to hardcode the batchnorm mode to miopenBNSpatial
// rather than making it configurable and adding extra complexity.
const miopenBatchNormMode_t MIOPEN_BATCHNORM_MODE_TRAINING = miopenBNSpatial;

BatchnormFwdTrainingParams::BatchnormFwdTrainingParams(
    const hipdnn_data_sdk::data_objects::BatchnormAttributes& attributes,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _x(miopen_utils::createTensor(tensorMap, attributes.x_tensor_uid()))
    , _y(miopen_utils::createTensor(tensorMap, attributes.y_tensor_uid()))
    , _scale(miopen_utils::createTensor(tensorMap, attributes.scale_tensor_uid()))
    , _bias(miopen_utils::createTensor(tensorMap, attributes.bias_tensor_uid()))
{
    // Extract epsilon value from pass-by-value tensor (cast to double for MIOpen compatibility)
    auto epsilonTensorAttr = tensorMap.at(attributes.epsilon_tensor_uid());
    _epsilonValue = miopen_utils::extractDoubleFromTensorValue(epsilonTensorAttr, "Epsilon");

    // Save mean and inv_variance are optional (controlled by MIO_SAVE_MEAN_VARIANCE)
    if(attributes.mean_tensor_uid().has_value())
    {
        _mean = miopen_utils::createTensor(tensorMap, attributes.mean_tensor_uid().value());
    }

    if(attributes.inv_variance_tensor_uid().has_value())
    {
        _invVariance
            = miopen_utils::createTensor(tensorMap, attributes.inv_variance_tensor_uid().value());
    }

#if 0 // Running statistics not supported - API mismatch between hipDNN and MIOpen \
    // hipDNN uses separate prev/next buffers, MIOpen requires IN/OUT buffers
    if(attributes.prev_running_mean_tensor_uid().has_value()
       && attributes.prev_running_variance_tensor_uid().has_value()
       && attributes.momentum_tensor_uid().has_value()
       && attributes.next_running_mean_tensor_uid().has_value()
       && attributes.next_running_variance_tensor_uid().has_value())
    {
        // Extract momentum value from pass-by-value tensor (cast to double for MIOpen compatibility)
        auto momentumTensorAttr = tensorMap.at(attributes.momentum_tensor_uid().value());
        _momentumValue = miopen_utils::extractDoubleFromTensorValue(momentumTensorAttr, "Momentum");

        _prevRunningMean = miopen_utils::createTensor(
            tensorMap, attributes.prev_running_mean_tensor_uid().value());
        _prevRunningVariance = miopen_utils::createTensor(
            tensorMap, attributes.prev_running_variance_tensor_uid().value());
        _nextRunningMean = miopen_utils::createTensor(
            tensorMap, attributes.next_running_mean_tensor_uid().value());
        _nextRunningVariance = miopen_utils::createTensor(
            tensorMap, attributes.next_running_variance_tensor_uid().value());
        _hasRunningStats = true;
    }
#else
    // Defensive check: should have been rejected by plan builder
    if(attributes.prev_running_mean_tensor_uid().has_value()
       || attributes.prev_running_variance_tensor_uid().has_value()
       || attributes.momentum_tensor_uid().has_value()
       || attributes.next_running_mean_tensor_uid().has_value()
       || attributes.next_running_variance_tensor_uid().has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "Running statistics should have been rejected by plan builder");
    }
#endif
}

BatchnormFwdTrainingParams::BatchnormFwdTrainingParams(
    const hipdnn_data_sdk::data_objects::BatchnormAttributes& attributes,
    const hipdnn_data_sdk::data_objects::PointwiseAttributes& pointwiseAttributes,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _x(miopen_utils::createTensor(tensorMap, attributes.x_tensor_uid()))
    , _y(miopen_utils::createTensor(tensorMap, attributes.y_tensor_uid()))
    , _scale(miopen_utils::createTensor(tensorMap, attributes.scale_tensor_uid()))
    , _bias(miopen_utils::createTensor(tensorMap, attributes.bias_tensor_uid()))
    , _activationOut(miopen_utils::createTensor(tensorMap, pointwiseAttributes.out_0_tensor_uid()))
{
    using namespace miopen_utils;

    // Extract epsilon value from pass-by-value tensor (cast to double for MIOpen compatibility)
    auto epsilonTensorAttr = tensorMap.at(attributes.epsilon_tensor_uid());
    _epsilonValue = extractDoubleFromTensorValue(epsilonTensorAttr, "Epsilon");

    // Validate that activation input matches batchnorm output
    if(pointwiseAttributes.in_0_tensor_uid() != attributes.y_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "BatchnormFwdTrainingParams: Activation input must match batchnorm output");
    }

    // Get activation parameters
    HIPDNN_PREPEND_MESSAGE_ON_THROW(_optActivation
                                    = mapPointwiseModeToMiopenActivation(pointwiseAttributes),
                                    "BatchnormFwdTrainingParams: ");

    // Save mean and inv_variance are optional (controlled by MIO_SAVE_MEAN_VARIANCE)
    if(attributes.mean_tensor_uid().has_value())
    {
        _mean = createTensor(tensorMap, attributes.mean_tensor_uid().value());
    }

    if(attributes.inv_variance_tensor_uid().has_value())
    {
        _invVariance = createTensor(tensorMap, attributes.inv_variance_tensor_uid().value());
    }

    // Running statistics not supported - API mismatch between hipDNN and MIOpen
    // Defensive check: should have been rejected by plan builder
    if(attributes.prev_running_mean_tensor_uid().has_value()
       || attributes.prev_running_variance_tensor_uid().has_value()
       || attributes.momentum_tensor_uid().has_value()
       || attributes.next_running_mean_tensor_uid().has_value()
       || attributes.next_running_variance_tensor_uid().has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "Running statistics should have been rejected by plan builder");
    }
}

const MiopenTensor& BatchnormFwdTrainingParams::x() const
{
    return _x;
}

const MiopenTensor& BatchnormFwdTrainingParams::y() const
{
    return _y;
}

const MiopenTensor& BatchnormFwdTrainingParams::scale() const
{
    return _scale;
}

const MiopenTensor& BatchnormFwdTrainingParams::bias() const
{
    return _bias;
}

double BatchnormFwdTrainingParams::epsilonValue() const
{
    return _epsilonValue;
}

bool BatchnormFwdTrainingParams::hasSaveMeanVariance() const
{
    return _mean.has_value() && _invVariance.has_value();
}

const MiopenTensor& BatchnormFwdTrainingParams::mean() const
{
    return _mean.value();
}

const MiopenTensor& BatchnormFwdTrainingParams::invVariance() const
{
    return _invVariance.value();
}

bool BatchnormFwdTrainingParams::hasRunningStats() const
{
    return _hasRunningStats;
}

const MiopenTensor& BatchnormFwdTrainingParams::prevRunningMean() const
{
    return _prevRunningMean.value();
}

const MiopenTensor& BatchnormFwdTrainingParams::prevRunningVariance() const
{
    return _prevRunningVariance.value();
}

double BatchnormFwdTrainingParams::momentumValue() const
{
    return _momentumValue.value();
}

const MiopenTensor& BatchnormFwdTrainingParams::nextRunningMean() const
{
    return _nextRunningMean.value();
}

const MiopenTensor& BatchnormFwdTrainingParams::nextRunningVariance() const
{
    return _nextRunningVariance.value();
}

const std::optional<miopen_utils::ActivationParams>&
    BatchnormFwdTrainingParams::optActivation() const
{
    return _optActivation;
}

const std::optional<MiopenTensor>& BatchnormFwdTrainingParams::activationOut() const
{
    return _activationOut;
}

BatchnormFwdTrainingPlan::BatchnormFwdTrainingPlan(BatchnormFwdTrainingParams&& trainingParams)
    : _trainingParams(std::move(trainingParams))
{
}

size_t BatchnormFwdTrainingPlan::getWorkspaceSize(
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle) const
{
    // No workspace needed for batchnorm training
    return 0;
}

void BatchnormFwdTrainingPlan::execute(const HipdnnEnginePluginHandle& handle,
                                       const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                       uint32_t numDeviceBuffers,
                                       [[maybe_unused]] void* workspace) const
{
    float alpha = 1.0f;
    float beta = 0.0f;

    // Extract epsilon from pass-by-value tensor attribute (type-safe, no buffer lookup needed)
    // Note: Type validation already done in constructor
    double epsilon = _trainingParams.epsilonValue();

    // Extract momentum from pass-by-value tensor attribute if running stats exist
    double expAvgFactor = 0.0;
    if(_trainingParams.hasRunningStats())
    {
        expAvgFactor = _trainingParams.momentumValue();
    }

    // Get all required device buffers
    auto xBuffer = miopen_utils::findDeviceBuffer(
        _trainingParams.x().uid(), deviceBuffers, numDeviceBuffers);
    auto scaleBuffer = miopen_utils::findDeviceBuffer(
        _trainingParams.scale().uid(), deviceBuffers, numDeviceBuffers);
    auto biasBuffer = miopen_utils::findDeviceBuffer(
        _trainingParams.bias().uid(), deviceBuffers, numDeviceBuffers);

    // Handle save mean/variance if provided (optional)
    void* resultSaveMeanPtr = nullptr;
    void* resultSaveInvVariancePtr = nullptr;
    miopenTensorDescriptor_t savedMeanDesc = nullptr;
    miopenTensorDescriptor_t savedVarDesc = nullptr;

    if(_trainingParams.hasSaveMeanVariance())
    {
        auto meanBuffer = miopen_utils::findDeviceBuffer(
            _trainingParams.mean().uid(), deviceBuffers, numDeviceBuffers);
        auto invVarianceBuffer = miopen_utils::findDeviceBuffer(
            _trainingParams.invVariance().uid(), deviceBuffers, numDeviceBuffers);

        resultSaveMeanPtr = meanBuffer.ptr;
        resultSaveInvVariancePtr = invVarianceBuffer.ptr;
        savedMeanDesc = _trainingParams.mean().tensorDescriptor();
        savedVarDesc = _trainingParams.invVariance().tensorDescriptor();
    }

    // Running statistics should have been rejected by plan builder
    if(_trainingParams.hasRunningStats())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "Running statistics should have been rejected by plan builder");
    }

    void* resultRunningMeanPtr = nullptr;
    void* resultRunningVariancePtr = nullptr;

    // Check if activation fusion is enabled
    if(_trainingParams.optActivation().has_value())
    {
        // Use activation fusion API
        auto yBuffer = miopen_utils::findDeviceBuffer(
            _trainingParams.activationOut().value().uid(), deviceBuffers, numDeviceBuffers);

        // Create activation descriptor
        miopenActivationDescriptor_t activationDesc;
        THROW_ON_MIOPEN_FAILURE(miopenCreateActivationDescriptor(&activationDesc));
        auto activationDescRes
            = hipdnn_data_sdk::utilities::ScopedResource<miopenActivationDescriptor_t>(
                activationDesc, [](miopenActivationDescriptor_t desc) {
                    auto status = miopenDestroyActivationDescriptor(desc);
                    if(status != miopenStatusSuccess)
                    {
                        HIPDNN_LOG_ERROR("miopenDestroyActivationDescriptor failed in "
                                         "BatchnormFwdTrainingPlan::execute");
                    }
                });

        const auto& activParams = _trainingParams.optActivation().value();
        THROW_ON_MIOPEN_FAILURE(miopenSetActivationDescriptor(activationDesc,
                                                              activParams.mode,
                                                              activParams.alpha,
                                                              activParams.beta,
                                                              activParams.gamma));

        THROW_ON_MIOPEN_FAILURE(miopenBatchNormForwardTrainingActivation(
            handle.miopenHandle,
            MIOPEN_BATCHNORM_MODE_TRAINING,
            &alpha,
            &beta,
            _trainingParams.x().tensorDescriptor(),
            xBuffer.ptr,
            _trainingParams.activationOut().value().tensorDescriptor(),
            yBuffer.ptr,
            _trainingParams.scale().tensorDescriptor(),
            _trainingParams.bias().tensorDescriptor(),
            savedMeanDesc,
            savedVarDesc,
            scaleBuffer.ptr,
            biasBuffer.ptr,
            expAvgFactor,
            resultRunningMeanPtr,
            resultRunningVariancePtr,
            epsilon,
            resultSaveMeanPtr,
            resultSaveInvVariancePtr,
            activationDesc));
    }
    else
    {
        // Use standard batchnorm training API (no activation)
        auto yBuffer = miopen_utils::findDeviceBuffer(
            _trainingParams.y().uid(), deviceBuffers, numDeviceBuffers);

        THROW_ON_MIOPEN_FAILURE(
            miopenBatchNormalizationForwardTraining(handle.miopenHandle,
                                                    MIOPEN_BATCHNORM_MODE_TRAINING,
                                                    &alpha,
                                                    &beta,
                                                    _trainingParams.x().tensorDescriptor(),
                                                    xBuffer.ptr,
                                                    _trainingParams.y().tensorDescriptor(),
                                                    yBuffer.ptr,
                                                    _trainingParams.scale().tensorDescriptor(),
                                                    scaleBuffer.ptr,
                                                    biasBuffer.ptr,
                                                    expAvgFactor,
                                                    resultRunningMeanPtr,
                                                    resultRunningVariancePtr,
                                                    epsilon,
                                                    resultSaveMeanPtr,
                                                    resultSaveInvVariancePtr));
    }
}

}
