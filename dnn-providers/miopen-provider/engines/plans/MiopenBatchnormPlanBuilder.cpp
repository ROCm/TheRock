// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_data_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <miopen/miopen.h>
#include <string>
#include <unordered_set>

#include "MiopenBatchnormPlanBuilder.hpp"
#include "MiopenUtils.hpp"
#include "engines/plans/MiopenBatchnormApplicabilityChecks.hpp"
#include "engines/plans/MiopenBatchnormBwdPlan.hpp"
#include "engines/plans/MiopenBatchnormFwdInferencePlan.hpp"
#include "engines/plans/MiopenBatchnormFwdTrainingPlan.hpp"

namespace miopen_legacy_plugin
{

namespace
{

std::tuple<const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes&,
           const hipdnn_data_sdk::data_objects::PointwiseAttributes&,
           const hipdnn_data_sdk::data_objects::BatchnormBackwardAttributes&>
    getBatchnormBackwardFusionNodeAttrs(const hipdnn_plugin_sdk::IGraph& opGraph)
{
    if(opGraph.nodeCount() != 3)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm fusion requires exactly 3 nodes. Graph has "
                + std::to_string(opGraph.nodeCount()) + " nodes");
    }

    const auto& bnInfAttr
        = opGraph.getNodeWrapper(0)
              .attributesAs<hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes>();

    const auto& actAttr = opGraph.getNodeWrapper(1)
                              .attributesAs<hipdnn_data_sdk::data_objects::PointwiseAttributes>();

    const auto& bnBwdAttr
        = opGraph.getNodeWrapper(2)
              .attributesAs<hipdnn_data_sdk::data_objects::BatchnormBackwardAttributes>();

    return {bnInfAttr, actAttr, bnBwdAttr};
}

auto getBatchnormBackwardFusionNodeAttrsLogErrors(const hipdnn_plugin_sdk::IGraph& opGraph)
    -> std::optional<decltype(getBatchnormBackwardFusionNodeAttrs(opGraph))>
{
    try
    {
        return getBatchnormBackwardFusionNodeAttrs(opGraph);
    }
    catch(const std::exception& e)
    {
        HIPDNN_LOG_INFO(e.what());
        return {};
    }
}

void batchnormBwdFusionCheckTensors(
    const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const hipdnn_data_sdk::data_objects::PointwiseAttributes& actAttr,
    const hipdnn_data_sdk::data_objects::BatchnormBackwardAttributes& bnBwdAttr,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    using PM = hipdnn_data_sdk::data_objects::PointwiseMode;
    static const std::unordered_set<PM> s_supportedActivations = {PM::RELU_BWD};

    if(s_supportedActivations.count(actAttr.operation()) == 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm fusion currently only supports RELU_BWD activation");
    }

    // in_0 must be the batchnorm inference output (forward path)
    if(actAttr.in_0_tensor_uid() != bnInfAttr.y_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation in_0 must be the batchnorm inference output tensor (y)");
    }

    // in_1 must be the dy (gradient from downstream)
    if(!actAttr.in_1_tensor_uid().has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation backward requires in_1 tensor (dy gradient)");
    }

    // Verify activation backwards output is BN backward dy input
    if(actAttr.out_0_tensor_uid() != bnBwdAttr.dy_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward dy input must be the activation output tensor");
    }

    // Verify that different BN operations use shared inputs where applicable
    if(bnBwdAttr.x_tensor_uid() != bnInfAttr.x_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward must use the same X tensor as batchnorm inference");
    }

    if(bnBwdAttr.mean_tensor_uid().has_value()
       && bnBwdAttr.mean_tensor_uid().value() != bnInfAttr.mean_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward must use the same mean tensor as batchnorm inference");
    }

    if(bnBwdAttr.inv_variance_tensor_uid().has_value()
       && bnBwdAttr.inv_variance_tensor_uid().value() != bnInfAttr.inv_variance_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward must use the same inv_variance tensor as batchnorm inference");
    }

    if(bnBwdAttr.scale_tensor_uid() != bnInfAttr.scale_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward must use the same scale tensor as batchnorm inference");
    }

    // Check for virtual tensors
    const auto& bnInfTensorX
        = miopen_utils::findTensorAttributes(tensorMap, bnInfAttr.x_tensor_uid());
    const auto& bnInfTensorMean
        = miopen_utils::findTensorAttributes(tensorMap, bnInfAttr.mean_tensor_uid());
    const auto& bnInfTensorInvVar
        = miopen_utils::findTensorAttributes(tensorMap, bnInfAttr.inv_variance_tensor_uid());
    const auto& bnInfTensorScale
        = miopen_utils::findTensorAttributes(tensorMap, bnInfAttr.scale_tensor_uid());
    const auto& bnInfTensorBias
        = miopen_utils::findTensorAttributes(tensorMap, bnInfAttr.bias_tensor_uid());
    const auto& bnInfTensorY
        = miopen_utils::findTensorAttributes(tensorMap, bnInfAttr.y_tensor_uid());

    if(bnInfTensorX.virtual_() || bnInfTensorMean.virtual_() || bnInfTensorInvVar.virtual_()
       || bnInfTensorScale.virtual_() || bnInfTensorBias.virtual_() || !bnInfTensorY.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm inference input tensors must be non-virtual, output tensor must be virtual");
    }

    const auto& actTensorIn1
        = miopen_utils::findTensorAttributes(tensorMap, actAttr.in_1_tensor_uid().value());

    if(actTensorIn1.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "Activation in_1 (dy gradient) must be non-virtual");
    }

    const auto& actTensorIn0
        = miopen_utils::findTensorAttributes(tensorMap, actAttr.in_0_tensor_uid());
    const auto& actTensorOut
        = miopen_utils::findTensorAttributes(tensorMap, actAttr.out_0_tensor_uid());

    if(!actTensorIn0.virtual_() || !actTensorOut.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation input from batchnorm must be virtual, output must be virtual");
    }

    const auto& bnBwdTensorDy
        = miopen_utils::findTensorAttributes(tensorMap, bnBwdAttr.dy_tensor_uid());
    const auto& bnBwdTensorDx
        = miopen_utils::findTensorAttributes(tensorMap, bnBwdAttr.dx_tensor_uid());
    const auto& bnBwdTensorDscale
        = miopen_utils::findTensorAttributes(tensorMap, bnBwdAttr.dscale_tensor_uid());
    const auto& bnBwdTensorDbias
        = miopen_utils::findTensorAttributes(tensorMap, bnBwdAttr.dbias_tensor_uid());

    if(!bnBwdTensorDy.virtual_() || bnBwdTensorDx.virtual_() || bnBwdTensorDscale.virtual_()
       || bnBwdTensorDbias.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm backward dy input must be virtual, output tensors must be non-virtual");
    }
}

void batchnormFwdFusionCheckTensors(
    const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const hipdnn_data_sdk::data_objects::PointwiseAttributes& actAttr,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    using PM = hipdnn_data_sdk::data_objects::PointwiseMode;
    static const std::unordered_set<PM> s_supportedActivations = {PM::RELU_FWD};

    if(s_supportedActivations.count(actAttr.operation()) == 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm fusion currently only supports RELU_FWD activation");
    }

    // in_0 must be the batchnorm inference output (forward path)
    if(actAttr.in_0_tensor_uid() != bnInfAttr.y_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation in_0 must be the batchnorm inference output tensor (y)");
    }

    // Check for virtual tensors
    const auto& bnInfTensorX
        = miopen_utils::findTensorAttributes(tensorMap, bnInfAttr.x_tensor_uid());
    const auto& bnInfTensorMean
        = miopen_utils::findTensorAttributes(tensorMap, bnInfAttr.mean_tensor_uid());
    const auto& bnInfTensorInvVar
        = miopen_utils::findTensorAttributes(tensorMap, bnInfAttr.inv_variance_tensor_uid());
    const auto& bnInfTensorScale
        = miopen_utils::findTensorAttributes(tensorMap, bnInfAttr.scale_tensor_uid());
    const auto& bnInfTensorBias
        = miopen_utils::findTensorAttributes(tensorMap, bnInfAttr.bias_tensor_uid());
    const auto& bnInfTensorY
        = miopen_utils::findTensorAttributes(tensorMap, bnInfAttr.y_tensor_uid());

    if(bnInfTensorX.virtual_() || bnInfTensorMean.virtual_() || bnInfTensorInvVar.virtual_()
       || bnInfTensorScale.virtual_() || bnInfTensorBias.virtual_() || !bnInfTensorY.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Batchnorm inference input tensors must be non-virtual, output tensor must be virtual");
    }

    const auto& actTensorIn0
        = miopen_utils::findTensorAttributes(tensorMap, actAttr.in_0_tensor_uid());
    const auto& actTensorOut
        = miopen_utils::findTensorAttributes(tensorMap, actAttr.out_0_tensor_uid());

    if(!actTensorIn0.virtual_() || actTensorOut.virtual_())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Activation input from batchnorm must be virtual, output must be non virtual");
    }
}

bool batchnormBwdFusionCheckTensorsLogErrors(
    const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const hipdnn_data_sdk::data_objects::PointwiseAttributes& actAttr,
    const hipdnn_data_sdk::data_objects::BatchnormBackwardAttributes& bnBwdAttr,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    try
    {
        batchnormBwdFusionCheckTensors(bnInfAttr, actAttr, bnBwdAttr, tensorMap);
        return true;
    }
    catch(const std::exception& e)
    {
        HIPDNN_LOG_INFO(e.what());
        return false;
    }
}

bool batchnormFwdFusionCheckTensorsLogErrors(
    const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
    const hipdnn_data_sdk::data_objects::PointwiseAttributes& actAttr,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    try
    {
        batchnormFwdFusionCheckTensors(bnInfAttr, actAttr, tensorMap);
        return true;
    }
    catch(const std::exception& e)
    {
        HIPDNN_LOG_INFO(e.what());
        return false;
    }
}

} // namespace

bool MiopenBatchnormPlanBuilder::isApplicable(
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle,
    const hipdnn_plugin_sdk::IGraph& opGraph) const
{
    auto areAllNodesF32Compute = [&]() {
        return !std::all_of(
            opGraph.nodeWrappers().begin(), opGraph.nodeWrappers().end(), [](const auto& node) {
                return node->computeDataType() == hipdnn_data_sdk::data_objects::DataType::FLOAT;
            });
    };

    switch(opGraph.nodeCount())
    {
    case 1:
    {
        if(areAllNodesF32Compute())
        {
            HIPDNN_LOG_ERROR("Batchnorm plan builder only supports nodes with an fp32 "
                             "compute_data_type");
            return false;
        }

        if(!opGraph.hasOnlySupportedAttributes(
               std::set<hipdnn_data_sdk::data_objects::NodeAttributes>{
                   hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormAttributes,
                   hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
                   hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes}))
        {
            HIPDNN_LOG_INFO("Batchnorm plan builder is not applicable for this graph");
            return false;
        }

        const auto& node = opGraph.getNode(0);

        // Only batchnorm training (BatchnormAttributes) has running statistics
        if(node.attributes_type()
           == hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormAttributes)
        {
            const auto* attr = node.attributes_as_BatchnormAttributes();
            if(attr != nullptr && attr->prev_running_mean_tensor_uid().has_value()
               && attr->prev_running_variance_tensor_uid().has_value()
               && attr->momentum_tensor_uid().has_value()
               && attr->next_running_mean_tensor_uid().has_value()
               && attr->next_running_variance_tensor_uid().has_value())
            {
                HIPDNN_LOG_INFO("Batchnorm plan builder does not support running statistics");
                return false;
            }
        }

        if(node.attributes_type()
           == hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes)
        {
            HIPDNN_LOG_WARN("Batchnorm inference support is temporarily disabled.");
            return false;
        }

        try
        {
            switch(node.attributes_type())
            {
            case hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormAttributes:
                checkBatchnormTensorConfigSupported(*node.attributes_as_BatchnormAttributes(),
                                                    opGraph.getTensorMap());
                break;
            case hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes:
                checkBatchnormTensorConfigSupported(
                    *node.attributes_as_BatchnormInferenceAttributes(), opGraph.getTensorMap());
                break;
            case hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes:
                checkBatchnormTensorConfigSupported(
                    *node.attributes_as_BatchnormBackwardAttributes(), opGraph.getTensorMap());
                break;
            default:
                throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                               "Unexpected node attribute type");
            }
        }
        catch(const std::exception& e)
        {
            HIPDNN_LOG_INFO(e.what());
            return false;
        }

        return true;
    }
    case 2:
    {
        if(areAllNodesF32Compute())
        {
            HIPDNN_LOG_ERROR("Batchnorm plan builder only supports nodes with an fp32 "
                             "compute_data_type");
            return false;
        }

        const auto& node0 = opGraph.getNodeWrapper(0);
        const auto& node1 = opGraph.getNodeWrapper(1);

        bool isFwdInferenceFirst
            = node0.attributesType()
              == hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes;
        bool isPointwiseSecond
            = node1.attributesType()
              == hipdnn_data_sdk::data_objects::NodeAttributes::PointwiseAttributes;

        if(!(isFwdInferenceFirst && isPointwiseSecond))
        {
            HIPDNN_LOG_INFO(
                "Batchnorm plan builder is not applicable for this graph node order and types");
            return false;
        }

        const auto& bnInfAttr
            = node0.attributesAs<hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes>();
        const auto& actAttr
            = node1.attributesAs<hipdnn_data_sdk::data_objects::PointwiseAttributes>();
        if(!batchnormFwdFusionCheckTensorsLogErrors(bnInfAttr, actAttr, opGraph.getTensorMap()))
        {
            return false;
        }

        // Since MIOpen does not provide an API to validate batchnorm applicability, we perform the
        // checks manually.
        try
        {
            checkBatchnormTensorConfigSupported(bnInfAttr, opGraph.getTensorMap());
            checkBatchnormFwdActivationModeSupported(actAttr);
        }
        catch(const std::exception& e)
        {
            HIPDNN_LOG_INFO(e.what());
            return false;
        }

        HIPDNN_LOG_INFO("Batchnorm plan builder applicable for batchnorm inference + "
                        "activation fusion");
        return true;
    }
    case 3:
    {
        // batchnorm inference -> activation -> batchnorm backward
        if(areAllNodesF32Compute())
        {
            HIPDNN_LOG_ERROR("Batchnorm plan builder only supports nodes with an fp32 "
                             "compute_data_type");
            return false;
        }
        const auto nodeAttrs = getBatchnormBackwardFusionNodeAttrsLogErrors(opGraph);
        if(!nodeAttrs.has_value())
        {
            return false;
        }

        if(!batchnormBwdFusionCheckTensorsLogErrors(std::get<0>(nodeAttrs.value()),
                                                    std::get<1>(nodeAttrs.value()),
                                                    std::get<2>(nodeAttrs.value()),
                                                    opGraph.getTensorMap()))
        {
            return false;
        }

        // Since MIOpen does not provide an API to validate batchnorm applicability, we perform the
        // checks manually.
        try
        {
            checkBatchnormTensorConfigSupported(std::get<0>(nodeAttrs.value()),
                                                std::get<1>(nodeAttrs.value()),
                                                std::get<2>(nodeAttrs.value()),
                                                opGraph.getTensorMap());
            checkBatchnormBwdActivationModeSupported(std::get<1>(nodeAttrs.value()));
        }
        catch(const std::exception& e)
        {
            HIPDNN_LOG_INFO(e.what());
            return false;
        }

        HIPDNN_LOG_INFO("Batchnorm plan builder applicable for batchnorm inference + "
                        "activation + batchnorm backward fusion");
        return true;
    }
    default:
    {
        HIPDNN_LOG_INFO("Batchnorm plan builder is applicable only for 1, 2 or 3 node graphs. "
                        "Graph has {} nodes",
                        opGraph.nodeCount());
        return false;
    }
    }
}

size_t MiopenBatchnormPlanBuilder::getWorkspaceSize(
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle,
    [[maybe_unused]] const hipdnn_plugin_sdk::IGraph& opGraph) const
{
    //batchnorm plan builder does not require workspace size
    return 0u;
}

namespace
{

void buildPlanInferenceSingleNode([[maybe_unused]] const HipdnnEnginePluginHandle& handle,
                                  const hipdnn_plugin_sdk::IGraph& opGraph,
                                  const hipdnn_plugin_sdk::INodeWrapper& nodeWrapper,
                                  HipdnnEnginePluginExecutionContext& executionContext)
{
    const auto& attr
        = nodeWrapper.attributesAs<hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes>();

    BatchnormFwdInferenceParams params(attr, opGraph.getTensorMap());
    auto plan = std::make_unique<BatchnormFwdInferencePlan>(std::move(params));
    executionContext.setPlan(std::move(plan));
}

void buildPlanFwdTrainingSingleNode([[maybe_unused]] const HipdnnEnginePluginHandle& handle,
                                    const hipdnn_plugin_sdk::IGraph& opGraph,
                                    const hipdnn_plugin_sdk::INodeWrapper& nodeWrapper,
                                    HipdnnEnginePluginExecutionContext& executionContext)
{
    const auto& attr
        = nodeWrapper.attributesAs<hipdnn_data_sdk::data_objects::BatchnormAttributes>();

    BatchnormFwdTrainingParams params(attr, opGraph.getTensorMap());
    auto plan = std::make_unique<BatchnormFwdTrainingPlan>(std::move(params));
    executionContext.setPlan(std::move(plan));
}

void buildPlanBwdSingleNode([[maybe_unused]] const HipdnnEnginePluginHandle& handle,
                            const hipdnn_plugin_sdk::IGraph& opGraph,
                            const hipdnn_plugin_sdk::INodeWrapper& nodeWrapper,
                            HipdnnEnginePluginExecutionContext& executionContext)
{
    const auto& attr
        = nodeWrapper.attributesAs<hipdnn_data_sdk::data_objects::BatchnormBackwardAttributes>();

    BatchnormBwdParams params(attr, opGraph.getTensorMap());
    auto plan = std::make_unique<BatchnormBwdPlan>(std::move(params));
    executionContext.setPlan(std::move(plan));
}

void buildPlanFusedBackwardsActivation([[maybe_unused]] const HipdnnEnginePluginHandle& handle,
                                       const hipdnn_plugin_sdk::IGraph& opGraph,
                                       HipdnnEnginePluginExecutionContext& executionContext)
{
    const auto [bnInfAttr, actAttr, bnBwdAttr] = getBatchnormBackwardFusionNodeAttrs(opGraph);
    batchnormBwdFusionCheckTensors(bnInfAttr, actAttr, bnBwdAttr, opGraph.getTensorMap());

    BatchnormBwdParams params(bnBwdAttr, actAttr, bnInfAttr, opGraph.getTensorMap());
    auto plan = std::make_unique<BatchnormBwdPlan>(std::move(params));
    executionContext.setPlan(std::move(plan));
}

void buildPlanFusedFwdInferenceActivation([[maybe_unused]] const HipdnnEnginePluginHandle& handle,
                                          const hipdnn_plugin_sdk::IGraph& opGraph,
                                          HipdnnEnginePluginExecutionContext& executionContext)
{
    const auto& node0 = opGraph.getNodeWrapper(0);
    const auto& node1 = opGraph.getNodeWrapper(1);

    const auto& fwdInference
        = node0.attributesAs<hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes>();
    const auto& activation
        = node1.attributesAs<hipdnn_data_sdk::data_objects::PointwiseAttributes>();

    BatchnormFwdInferenceParams params(fwdInference, activation, opGraph.getTensorMap());
    auto plan = std::make_unique<BatchnormFwdInferencePlan>(std::move(params));
    executionContext.setPlan(std::move(plan));
}

} // namespace

void MiopenBatchnormPlanBuilder::buildPlan(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_plugin_sdk::IGraph& opGraph,
    HipdnnEnginePluginExecutionContext& executionContext) const
{
    if(opGraph.nodeCount() == 2)
    {
        HIPDNN_LOG_INFO("Building batchnorm inference + activation fusion plan");
        buildPlanFusedFwdInferenceActivation(handle, opGraph, executionContext);
        return;
    }
    if(opGraph.nodeCount() == 3)
    {
        HIPDNN_LOG_INFO(
            "Building batchnorm inference + activation + batchnorm backward fusion plan");
        buildPlanFusedBackwardsActivation(handle, opGraph, executionContext);
        return;
    }

    const auto& nodeWrapper = opGraph.getNodeWrapper(0);
    const auto nodeName = nodeWrapper.name();

    switch(nodeWrapper.attributesType())
    {
    case hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes:
        HIPDNN_LOG_INFO("Building batchnorm fwd inference plan for node: {}", nodeName);
        buildPlanInferenceSingleNode(handle, opGraph, nodeWrapper, executionContext);
        break;
    case hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormAttributes:
        HIPDNN_LOG_INFO("Building batchnorm fwd training plan for node: {}", nodeName);
        buildPlanFwdTrainingSingleNode(handle, opGraph, nodeWrapper, executionContext);
        break;
    case hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes:
        HIPDNN_LOG_INFO("Building batchnorm backward plan for node: {}", nodeName);
        buildPlanBwdSingleNode(handle, opGraph, nodeWrapper, executionContext);
        break;
    default:
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported node type for batchnorm plan builder: "
                + std::string(
                    hipdnn_data_sdk::data_objects::toString(nodeWrapper.attributesType())));
    }
}

} // namespace miopen_legacy_plugin
