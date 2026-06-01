// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_data_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>

#include "HipdnnEnginePluginHandle.hpp"
#include "MiopenConvFwdBiasActivPlan.hpp"

// MIOpen's fusion API does not calculate the workspace size correctly
#define WORKAROUND_LWPMIOPEN_1815 1

#if WORKAROUND_LWPMIOPEN_1815
#include <algorithm>
#include <array>
#include <numeric>
#endif

namespace miopen_legacy_plugin
{

ConvFwdBiasActivParams::ConvFwdBiasActivParams(
    const hipdnn_data_sdk::data_objects::ConvolutionFwdAttributes& convAttr,
    const hipdnn_data_sdk::data_objects::PointwiseAttributes* biasAttr,
    const hipdnn_data_sdk::data_objects::PointwiseAttributes& activAttr,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _spatialDimCount(miopen_utils::getSpatialDimCount(
          miopen_utils::findTensorAttributes(tensorMap, convAttr.x_tensor_uid())))
    , _x(miopen_utils::createTensor(tensorMap, convAttr.x_tensor_uid()))
    , _w(miopen_utils::createTensor(tensorMap, convAttr.w_tensor_uid()))
    , _y(miopen_utils::createTensor(tensorMap, activAttr.out_0_tensor_uid()))
{
    using namespace miopen_utils;

    const auto& attrX = findTensorAttributes(tensorMap, _x.uid());
    const auto& attrW = findTensorAttributes(tensorMap, _w.uid());

    const auto xDims = hipdnn_data_sdk::utilities::convertFlatBufferVectorToStdVector(attrX.dims());
    const auto wDims = hipdnn_data_sdk::utilities::convertFlatBufferVectorToStdVector(attrW.dims());
    const auto groupCount = hipdnn_data_sdk::utilities::calculateGroupCount(xDims, wDims);

    _conv = MiopenConvDescriptor(_spatialDimCount, convAttr, static_cast<int>(groupCount));

    if(biasAttr != nullptr)
    {
        if(!biasAttr->in_1_tensor_uid().has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "ConvFwdBiasActivParams: biasAttr missing in_1_tensor_uid");
        }

        if(biasAttr->in_0_tensor_uid() == convAttr.y_tensor_uid())
        {
            _bias = createTensor(tensorMap, biasAttr->in_1_tensor_uid().value());
        }
        else if(biasAttr->in_1_tensor_uid().value() == convAttr.y_tensor_uid())
        {
            _bias = createTensor(tensorMap, biasAttr->in_0_tensor_uid());
        }
        else
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "ConvFwdBiasActivParams: biasAttr tensor UIDs do not match convAttr y_tensor_uid");
        }
    }

    if(activAttr.elu_alpha().has_value() || activAttr.softplus_beta().has_value()
       || activAttr.swish_beta().has_value() || activAttr.relu_lower_clip_slope().has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "ConvFwdBiasActivParams: Fusion supports only relu, clipped relu "
            "(relu_upper_clip set), "
            "or CLAMP "
            "(relu_upper_clip and relu_lower_clip set)");
    }

    HIPDNN_PREPEND_MESSAGE_ON_THROW(_activParams = mapPointwiseModeToMiopenActivation(activAttr),
                                    "ConvFwdBiasActivParams: ");
}

const MiopenTensor& ConvFwdBiasActivParams::x() const
{
    return _x;
}

const MiopenTensor& ConvFwdBiasActivParams::w() const
{
    return _w;
}

const MiopenConvDescriptor& ConvFwdBiasActivParams::conv() const
{
    return _conv;
}

const std::optional<MiopenTensor>& ConvFwdBiasActivParams::bias() const
{
    return _bias;
}

const miopen_utils::ActivationParams& ConvFwdBiasActivParams::activParams() const
{
    return _activParams;
}

const MiopenTensor& ConvFwdBiasActivParams::y() const
{
    return _y;
}

ConvFwdBiasActivPlan::ConvFwdBiasActivPlan(const HipdnnEnginePluginHandle& handle,
                                           ConvFwdBiasActivParams&& params,
                                           bool compile,
                                           bool getWsSize)
    : _params(std::move(params))
{
    miopenFusionPlanDescriptor_t fusePlanDesc;
    THROW_ON_MIOPEN_FAILURE(miopenCreateFusionPlan(
        &fusePlanDesc, miopenVerticalFusion, _params.x().tensorDescriptor()));
    _fusePlanDesc = hipdnn_data_sdk::utilities::ScopedResource<miopenFusionPlanDescriptor_t>(
        fusePlanDesc, [](miopenFusionPlanDescriptor_t desc) {
            auto status = miopenDestroyFusionPlan(desc);
            if(status != miopenStatusSuccess)
            {
                HIPDNN_LOG_ERROR(
                    "miopenDestroyFusionPlan failed in ConvFwdBiasActivPlan destructor");
            }
        });

    miopenFusionOpDescriptor_t convOp;
    THROW_ON_MIOPEN_FAILURE(miopenCreateOpConvForward(
        fusePlanDesc, &convOp, _params.conv().convDescriptor(), _params.w().tensorDescriptor()));

    if(_params.bias().has_value())
    {
        miopenFusionOpDescriptor_t biasOp;
        THROW_ON_MIOPEN_FAILURE(miopenCreateOpBiasForward(
            fusePlanDesc, &biasOp, _params.bias().value().tensorDescriptor()));
    }

    miopenFusionOpDescriptor_t activOp;
    THROW_ON_MIOPEN_FAILURE(
        miopenCreateOpActivationForward(fusePlanDesc, &activOp, _params.activParams().mode));

    // Usage scenarios:
    // 1. Applicability check: requires compilation only
    // 2. Workspace size: requires workspace size retrieval only
    // 3. Execution plan: requires both compilation and workspace size retrieval

    if(compile)
    {
        THROW_ON_MIOPEN_FAILURE(miopenCompileFusionPlan(handle.miopenHandle, fusePlanDesc));
    }

    if(getWsSize)
    {
#if WORKAROUND_LWPMIOPEN_1815
        // MIOpen's fusion API does not calculate the workspace size correctly. To work around
        // this issue, the workspace size is calculated as the sum of the sizes of the input,
        // weight, and output tensors, each aligned to 256 bytes.
        // Refer to ConvCKIgemmGrpFwdBiasActivFused::GetWorkspaceSize() in MIOpen's source code
        // for more details.

        size_t xSize;
        THROW_ON_MIOPEN_FAILURE(miopenGetTensorNumBytes(_params.x().tensorDescriptor(), &xSize));
        size_t wSize;
        THROW_ON_MIOPEN_FAILURE(miopenGetTensorNumBytes(_params.w().tensorDescriptor(), &wSize));
        size_t ySize;
        THROW_ON_MIOPEN_FAILURE(miopenGetTensorNumBytes(_params.y().tensorDescriptor(), &ySize));

        std::array<size_t, 3> sizes = {xSize, wSize, ySize};

        // Align each size to 256 bytes
        constexpr size_t ALIGNMENT_BOUNDARY = 256;
        constexpr size_t ALIGNMENT = ALIGNMENT_BOUNDARY - 1;
        auto alignToBoundary = [](size_t size) { return (size + ALIGNMENT) & ~ALIGNMENT; };
        std::transform(sizes.begin(), sizes.end(), sizes.begin(), alignToBoundary);

        // Calculate the total workspace size as the sum of aligned sizes
        _workspaceSize = std::accumulate(sizes.begin(), sizes.end(), static_cast<size_t>(0));
#else
        THROW_ON_MIOPEN_FAILURE(miopenFusionPlanGetWorkSpaceSize(
            handle.miopenHandle,
            fusePlanDesc,
            &_workspaceSize,
            static_cast<miopenConvFwdAlgorithm_t>(-1))); // Algo is not used in MIOpen
#endif
    }
}

size_t ConvFwdBiasActivPlan::getWorkspaceSize(
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle) const
{
    return _workspaceSize;
}

void ConvFwdBiasActivPlan::execute(const HipdnnEnginePluginHandle& handle,
                                   const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                   uint32_t numDeviceBuffers,
                                   void* workspace) const
{
    miopenOperatorArgs_t fusionArgs;
    THROW_ON_MIOPEN_FAILURE(miopenCreateOperatorArgs(&fusionArgs));
    auto fusionArgsRes = hipdnn_data_sdk::utilities::ScopedResource<miopenOperatorArgs_t>(
        fusionArgs, [](miopenOperatorArgs_t args) {
            auto status = miopenDestroyOperatorArgs(args);
            if(status != miopenStatusSuccess)
            {
                HIPDNN_LOG_ERROR(
                    "miopenDestroyOperatorArgs failed in ConvFwdBiasActivPlan destructor");
            }
        });

    auto wBuffer
        = miopen_utils::findDeviceBuffer(_params.w().uid(), deviceBuffers, numDeviceBuffers);

    int opIdx = 0;
    miopenFusionOpDescriptor_t convoOp;
    THROW_ON_MIOPEN_FAILURE(miopenFusionPlanGetOp(_fusePlanDesc.get(), opIdx++, &convoOp));
    THROW_ON_MIOPEN_FAILURE(miopenSetOpArgsConvForward(fusionArgs,
                                                       convoOp,
                                                       nullptr, // Default value for alpha is 1.0f
                                                       nullptr, // Default value for beta is 0.0f
                                                       wBuffer.ptr));

    if(_params.bias().has_value())
    {
        auto biasBuffer = miopen_utils::findDeviceBuffer(
            _params.bias().value().uid(), deviceBuffers, numDeviceBuffers);

        miopenFusionOpDescriptor_t biasOp;
        THROW_ON_MIOPEN_FAILURE(miopenFusionPlanGetOp(_fusePlanDesc.get(), opIdx++, &biasOp));
        THROW_ON_MIOPEN_FAILURE(miopenSetOpArgsBiasForward(fusionArgs,
                                                           biasOp,
                                                           nullptr, // alpha is not used in MIOpen
                                                           nullptr, // beta is not used in MIOpen
                                                           biasBuffer.ptr));
    }

    miopenFusionOpDescriptor_t activOp;
    THROW_ON_MIOPEN_FAILURE(miopenFusionPlanGetOp(_fusePlanDesc.get(), opIdx, &activOp));
    THROW_ON_MIOPEN_FAILURE(miopenSetOpArgsActivForward(fusionArgs,
                                                        activOp,
                                                        nullptr, // alpha is not used in MIOpen
                                                        nullptr, // beta is not used in MIOpen
                                                        _params.activParams().alpha,
                                                        _params.activParams().beta,
                                                        _params.activParams().gamma));

    size_t workspaceSize = 0;
    if(workspace != nullptr)
    {
        // Assume the provided workspace is large enough
        workspaceSize = _workspaceSize;
    }

    auto xBuffer
        = miopen_utils::findDeviceBuffer(_params.x().uid(), deviceBuffers, numDeviceBuffers);
    auto yBuffer
        = miopen_utils::findDeviceBuffer(_params.y().uid(), deviceBuffers, numDeviceBuffers);

    THROW_ON_MIOPEN_FAILURE(miopenExecuteFusionPlan_v2(handle.miopenHandle,
                                                       _fusePlanDesc.get(),
                                                       _params.x().tensorDescriptor(),
                                                       xBuffer.ptr,
                                                       _params.y().tensorDescriptor(),
                                                       yBuffer.ptr,
                                                       fusionArgs,
                                                       workspace,
                                                       workspaceSize));
}

} // namespace miopen_legacy_plugin
