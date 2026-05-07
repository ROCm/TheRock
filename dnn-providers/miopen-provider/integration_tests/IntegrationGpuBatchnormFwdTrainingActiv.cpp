// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
#include <random>

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMiopenRmsValidation.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../tests/common/ActivationCommon.hpp"
#include "../tests/common/BatchnormCommon.hpp"
#include "IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace miopen_legacy_plugin::test_utilities;

namespace
{

using test_activation_common::ActivTestCase;
using test_bn_common::BatchnormTestCase;

struct BatchnormFwdTrainingActivTensorIds
{
    static constexpr int64_t X_UID = 1;
    static constexpr int64_t SCALE_UID = 2;
    static constexpr int64_t BIAS_UID = 3;
    static constexpr int64_t PREV_RUNNING_MEAN_UID = 4;
    static constexpr int64_t PREV_RUNNING_VARIANCE_UID = 5;
    static constexpr int64_t NEXT_RUNNING_MEAN_UID = 6;
    static constexpr int64_t NEXT_RUNNING_VARIANCE_UID = 7;
};

// Note: hipDNN BatchNorm implements Spatial normalization only (miopenBNSpatial).
// The mode is hardcoded in the MIOpen plugin (see MiopenBatchnormFwdTrainingActivPlan.cpp).
// Per-activation normalization would require LayerNorm or InstanceNorm operations.
//
// These scenarios test different output combinations in forward training:
// - WITH_BATCH_STATS: Computes batch statistics (mean/invVariance) without updating running stats
// - FULL_TRAINING: Computes batch statistics AND updates running mean/variance via EMA
enum class BatchnormTrainingScenario
{
    WITH_BATCH_STATS, // Batch stats only (no running stats update)
    FULL_TRAINING // Batch stats + running stats update (canonical training)
};

template <typename InputType, typename IntermediateType>
class BatchnormFwdTrainingActivation
    : public IntegrationGraphVerificationHarness<
          InputType,
          std::tuple<test_bn_common::BatchnormTestCase, test_activation_common::ActivTestCase>>
{
protected:
    void runGraphTest(InputType tolerance, const TensorLayout& layout = TensorLayout::NCHW) override
    {
        runGraphTestWithScenario(tolerance, BatchnormTrainingScenario::WITH_BATCH_STATS, layout);
    }

    void runGraphTestWithScenario(InputType tolerance,
                                  BatchnormTrainingScenario scenario,
                                  const TensorLayout& layout = TensorLayout::NCHW)
    {
        const auto& [bnTestCase, activTestCase] = this->GetParam();

        HIPDNN_LOG_INFO("Test is using {} for its random seed", bnTestCase.seed);

        hipdnn_frontend::graph::Graph graphObj;
        graphObj.set_name("BatchnormFwdTrainingActivTest");

        auto inputDataType = getDataTypeEnumFromType<InputType>();
        auto intermediateDataType = getDataTypeEnumFromType<IntermediateType>();
        graphObj.set_intermediate_data_type(intermediateDataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(inputDataType);

        auto dims = bnTestCase.dims;
        auto derivedDims = getDerivedShape(dims);

        // Create input tensor attributes
        auto xAttr
            = graph::makeTensorAttributes("X", dims, generateStrides(dims, layout.strideOrder));
        xAttr.set_uid(BatchnormFwdTrainingActivTensorIds::X_UID);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        // Channel-only tensors are layout-agnostic, specifying stride order is unnecessary
        auto scaleAttr = graph::makeTensorAttributes(
            "scale", intermediateDataType, derivedDims, generateStrides(derivedDims));
        scaleAttr.set_uid(BatchnormFwdTrainingActivTensorIds::SCALE_UID);
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr = graph::makeTensorAttributes(
            "bias", intermediateDataType, derivedDims, generateStrides(derivedDims));
        biasAttr.set_uid(BatchnormFwdTrainingActivTensorIds::BIAS_UID);
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        // Epsilon: use pass-by-value with double (matches MIOpen API)
        auto epsilonTensorAttr = std::make_shared<graph::TensorAttributes>();
        std::mt19937 gen(bnTestCase.seed);
        std::uniform_real_distribution<double> epsilonDist(1e-6, 1e-4);
        epsilonTensorAttr->set_value(epsilonDist(gen)).set_name("epsilon");

        // Conditionally setup running statistics based on scenario
        std::shared_ptr<graph::TensorAttributes> prevRunningMeanTensorAttr;
        std::shared_ptr<graph::TensorAttributes> prevRunningVarianceTensorAttr;
        std::shared_ptr<graph::TensorAttributes> momentumTensorAttr;

        if(scenario == BatchnormTrainingScenario::FULL_TRAINING)
        {
            auto prevRunningMeanAttr = graph::makeTensorAttributes("prev_running_mean",
                                                                   intermediateDataType,
                                                                   derivedDims,
                                                                   generateStrides(derivedDims));
            prevRunningMeanAttr.set_uid(BatchnormFwdTrainingActivTensorIds::PREV_RUNNING_MEAN_UID);
            prevRunningMeanTensorAttr
                = std::make_shared<graph::TensorAttributes>(std::move(prevRunningMeanAttr));

            auto prevRunningVarianceAttr
                = graph::makeTensorAttributes("prev_running_variance",
                                              intermediateDataType,
                                              derivedDims,
                                              generateStrides(derivedDims));
            prevRunningVarianceAttr.set_uid(
                BatchnormFwdTrainingActivTensorIds::PREV_RUNNING_VARIANCE_UID);
            prevRunningVarianceTensorAttr
                = std::make_shared<graph::TensorAttributes>(std::move(prevRunningVarianceAttr));

            // Momentum: use pass-by-value with double (matches MIOpen API)
            momentumTensorAttr = std::make_shared<graph::TensorAttributes>();
            std::uniform_real_distribution<double> momentumDist(0.05, 0.15);
            momentumTensorAttr->set_value(momentumDist(gen)).set_name("momentum");
        }

        // Create batchnorm attributes
        graph::BatchnormAttributes bnAttrs;

        if(prevRunningMeanTensorAttr && prevRunningVarianceTensorAttr && momentumTensorAttr)
        {
            bnAttrs.set_previous_running_stats(
                prevRunningMeanTensorAttr, prevRunningVarianceTensorAttr, momentumTensorAttr);
        }

        bnAttrs.set_epsilon(epsilonTensorAttr);

        auto [yBnTensorAttr,
              meanTensorAttr,
              invVarianceTensorAttr,
              nextRunningMeanTensorAttr,
              nextRunningVarianceTensorAttr]
            = graphObj.batchnorm(xTensorAttr, scaleTensorAttr, biasTensorAttr, bnAttrs);

        yBnTensorAttr->set_data_type(inputDataType);

        // Add activation node with parameters from test case
        graph::PointwiseAttributes activAttrs;
        activAttrs.set_mode(static_cast<hipdnn_frontend::PointwiseMode>(activTestCase.mode));

        // Set activation-specific parameters
        if(activTestCase.reluLowerClip.has_value())
        {
            activAttrs.set_relu_lower_clip(activTestCase.reluLowerClip.value());
        }
        if(activTestCase.reluUpperClip.has_value())
        {
            activAttrs.set_relu_upper_clip(activTestCase.reluUpperClip.value());
        }
        if(activTestCase.reluLowerClipSlope.has_value())
        {
            activAttrs.set_relu_lower_clip_slope(activTestCase.reluLowerClipSlope.value());
        }
        if(activTestCase.swishBeta.has_value())
        {
            activAttrs.set_swish_beta(activTestCase.swishBeta.value());
        }
        if(activTestCase.eluAlpha.has_value())
        {
            activAttrs.set_elu_alpha(activTestCase.eluAlpha.value());
        }
        if(activTestCase.softplusBeta.has_value())
        {
            activAttrs.set_softplus_beta(activTestCase.softplusBeta.value());
        }

        auto yActivTensorAttr = graphObj.pointwise(yBnTensorAttr, activAttrs);

        // Set final activation output tensor
        yActivTensorAttr->set_output(true);

        // Configure batch statistics outputs
        if(meanTensorAttr)
        {
            meanTensorAttr->set_output(true);
            meanTensorAttr->set_data_type(intermediateDataType);
        }

        if(invVarianceTensorAttr)
        {
            invVarianceTensorAttr->set_output(true);
            invVarianceTensorAttr->set_data_type(intermediateDataType);
        }

        // Configure running statistics outputs if they exist
        if(nextRunningMeanTensorAttr)
        {
            nextRunningMeanTensorAttr->set_uid(
                BatchnormFwdTrainingActivTensorIds::NEXT_RUNNING_MEAN_UID);
            nextRunningMeanTensorAttr->set_name("next_running_mean");
            nextRunningMeanTensorAttr->set_output(true);
            nextRunningMeanTensorAttr->set_data_type(intermediateDataType);
        }

        if(nextRunningVarianceTensorAttr)
        {
            nextRunningVarianceTensorAttr->set_uid(
                BatchnormFwdTrainingActivTensorIds::NEXT_RUNNING_VARIANCE_UID);
            nextRunningVarianceTensorAttr->set_name("next_running_variance");
            nextRunningVarianceTensorAttr->set_output(true);
            nextRunningVarianceTensorAttr->set_data_type(intermediateDataType);
        }

        // Register validators for all output tensors
        this->registerValidator(yActivTensorAttr, tolerance);
        this->registerValidator(meanTensorAttr, tolerance);
        this->registerValidator(invVarianceTensorAttr, tolerance);
        if(nextRunningMeanTensorAttr)
        {
            this->registerValidator(nextRunningMeanTensorAttr, tolerance);
        }
        if(nextRunningVarianceTensorAttr)
        {
            this->registerValidator(nextRunningVarianceTensorAttr, tolerance);
        }

        this->verifyGraph(graphObj, bnTestCase.seed);
    }

    void initializeBundle([[maybe_unused]] const hipdnn_frontend::graph::Graph& graph,
                          GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        // X input: default range
        bundle.tensors.at(BatchnormFwdTrainingActivTensorIds::X_UID)
            ->fillTensorWithRandomValues(-1.0f, 1.0f, seed);

        // Scale and bias: -2.0 to 2.0 to match MIOpen
        bundle.tensors.at(BatchnormFwdTrainingActivTensorIds::SCALE_UID)
            ->fillTensorWithRandomValues(-2.0f, 2.0f, seed + 1);
        bundle.tensors.at(BatchnormFwdTrainingActivTensorIds::BIAS_UID)
            ->fillTensorWithRandomValues(-2.0f, 2.0f, seed + 2);

        // Running mean: prev and next must start with SAME values
        // because MIOpen's API uses IN/OUT parameter semantics
        if(bundle.tensors.find(BatchnormFwdTrainingActivTensorIds::PREV_RUNNING_MEAN_UID)
               != bundle.tensors.end()
           && bundle.tensors.find(BatchnormFwdTrainingActivTensorIds::NEXT_RUNNING_MEAN_UID)
                  != bundle.tensors.end())
        {
            unsigned runningMeanSeed = seed + 1000;
            bundle.tensors.at(BatchnormFwdTrainingActivTensorIds::PREV_RUNNING_MEAN_UID)
                ->fillTensorWithRandomValues(-2.0f, 2.0f, runningMeanSeed);
            bundle.tensors.at(BatchnormFwdTrainingActivTensorIds::NEXT_RUNNING_MEAN_UID)
                ->fillTensorWithRandomValues(-2.0f, 2.0f, runningMeanSeed);
        }

        // Running variance: prev and next must start with SAME values
        if(bundle.tensors.find(BatchnormFwdTrainingActivTensorIds::PREV_RUNNING_VARIANCE_UID)
               != bundle.tensors.end()
           && bundle.tensors.find(BatchnormFwdTrainingActivTensorIds::NEXT_RUNNING_VARIANCE_UID)
                  != bundle.tensors.end())
        {
            unsigned runningVarianceSeed = seed + 2000;
            bundle.tensors.at(BatchnormFwdTrainingActivTensorIds::PREV_RUNNING_VARIANCE_UID)
                ->fillTensorWithRandomValues(-2.0f, 2.0f, runningVarianceSeed);
            bundle.tensors.at(BatchnormFwdTrainingActivTensorIds::NEXT_RUNNING_VARIANCE_UID)
                ->fillTensorWithRandomValues(-2.0f, 2.0f, runningVarianceSeed);
        }
    }
};

// NCHW 2D
using IntegrationGpuBatchnormFwdTrainingActivNchwFp32
    = BatchnormFwdTrainingActivation<float, float>;
using IntegrationGpuBatchnormFwdTrainingActivNchwFp16 = BatchnormFwdTrainingActivation<half, float>;
using IntegrationGpuBatchnormFwdTrainingActivNchwBfp16
    = BatchnormFwdTrainingActivation<hip_bfloat16, float>;

// NHWC 2D
using IntegrationGpuBatchnormFwdTrainingActivNhwcFp32
    = BatchnormFwdTrainingActivation<float, float>;
using IntegrationGpuBatchnormFwdTrainingActivNhwcFp16 = BatchnormFwdTrainingActivation<half, float>;
using IntegrationGpuBatchnormFwdTrainingActivNhwcBfp16
    = BatchnormFwdTrainingActivation<hip_bfloat16, float>;

// NCDHW 3D
using IntegrationGpuBatchnormFwdTrainingActivNcdhwFp32
    = BatchnormFwdTrainingActivation<float, float>;
using IntegrationGpuBatchnormFwdTrainingActivNcdhwFp16
    = BatchnormFwdTrainingActivation<half, float>;
using IntegrationGpuBatchnormFwdTrainingActivNcdhwBfp16
    = BatchnormFwdTrainingActivation<hip_bfloat16, float>;

// NDHWC 3D
using IntegrationGpuBatchnormFwdTrainingActivNdhwcFp32
    = BatchnormFwdTrainingActivation<float, float>;
using IntegrationGpuBatchnormFwdTrainingActivNdhwcFp16
    = BatchnormFwdTrainingActivation<half, float>;
using IntegrationGpuBatchnormFwdTrainingActivNdhwcBfp16
    = BatchnormFwdTrainingActivation<hip_bfloat16, float>;

} // namespace

// ============================================================================
// NCHW 2D Tests
// ============================================================================

// NOTE: FullTraining tests are disabled due to API mismatch between hipDNN and MIOpen.
// hipDNN's graph API uses separate prev_running_mean/variance (input) and next_running_mean/variance (output)
// buffers, but MIOpen's API requires single IN/OUT buffers for running statistics.
// This cannot be correctly bridged without either:
// 1. Updating MIOpen API to support separate input/output buffers, or
// 2. Implementing buffer copy operations (with performance overhead)
// Until MIOpen is updated, batchnorm training with running statistics is not supported.
// BatchStatsOnly tests (without running statistics) continue to work correctly.

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNchwFp32, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<float>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NCHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNchwFp32, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<float>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NCHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNchwFp16, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<half>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NCHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNchwFp16, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<half>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NCHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNchwBfp16, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<hip_bfloat16>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NCHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNchwBfp16, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<hip_bfloat16>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NCHW);
}

// ============================================================================
// NHWC 2D Tests
// ============================================================================

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNhwcFp32, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<float>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNhwcFp32, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<float>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNhwcFp16, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<half>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNhwcFp16, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<half>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNhwcBfp16, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<hip_bfloat16>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNhwcBfp16, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<hip_bfloat16>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NHWC);
}

// ============================================================================
// NCDHW 3D Tests
// ============================================================================

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNcdhwFp32, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<float>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNcdhwFp32, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<float>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNcdhwFp16, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<half>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNcdhwFp16, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<half>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNcdhwBfp16, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<hip_bfloat16>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNcdhwBfp16, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<hip_bfloat16>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NCDHW);
}

// ============================================================================
// NDHWC 3D Tests
// ============================================================================

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNdhwcFp32, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<float>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNdhwcFp32, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<float>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNdhwcFp16, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<half>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNdhwcFp16, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<half>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNdhwcBfp16, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<hip_bfloat16>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingActivNdhwcBfp16, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<hip_bfloat16>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NDHWC);
}

// ============================================================================
// Test Instantiation
// ============================================================================

// 2D NCHW Tests
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivNchwFp32,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActivNchwFp32,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingFull2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivNchwFp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActivNchwFp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingFull2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivNchwBfp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActivNchwBfp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingFull2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

// 2D NHWC Tests
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivNhwcFp32,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActivNhwcFp32,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingFull2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivNhwcFp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActivNhwcFp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingFull2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivNhwcBfp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActivNhwcBfp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingFull2dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

// 3D NCDHW Tests
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivNcdhwFp32,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActivNcdhwFp32,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingFull3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivNcdhwFp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActivNcdhwFp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingFull3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivNcdhwBfp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActivNcdhwBfp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingFull3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

// 3D NDHWC Tests
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivNdhwcFp32,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActivNdhwcFp32,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingFull3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivNdhwcFp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActivNdhwcFp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingFull3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdTrainingActivNdhwcBfp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdTrainingActivNdhwcBfp16,
    testing::Combine(testing::ValuesIn(test_bn_common::getBnFwdTrainingFull3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));
