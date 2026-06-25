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

#include "../tests/common/BatchnormCommon.hpp"
#include "IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace miopen_legacy_plugin::test_utilities;

namespace
{

using test_bn_common::BatchnormTestCase;

struct BatchnormFwdTrainingTensorIds
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
// The mode is hardcoded in the MIOpen plugin (see MiopenBatchnormFwdTrainingPlan.cpp).
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

template <typename InputType, typename IntermediateType, typename TestCaseType>
class BatchnormForwardTraining : public IntegrationGraphVerificationHarness<InputType, TestCaseType>
{
protected:
    void runGraphTest(InputType tolerance, const TensorLayout& layout = TensorLayout::NCHW) override
    {
        runGraphTestWithScenario(tolerance, BatchnormTrainingScenario::FULL_TRAINING, layout);
    }

    void runGraphTestWithScenario(InputType tolerance,
                                  BatchnormTrainingScenario scenario,
                                  const TensorLayout& layout = TensorLayout::NCHW)
    {
        const TestCaseType& testCase = this->GetParam();

        HIPDNN_LOG_INFO("Test is using {} for its random seed", testCase.seed);

        hipdnn_frontend::graph::Graph graphObj;
        graphObj.set_name("BatchnormForwardTrainingTest");

        auto inputDataType = getDataTypeEnumFromType<InputType>();
        auto intermediateDataType = getDataTypeEnumFromType<IntermediateType>();
        graphObj.set_intermediate_data_type(intermediateDataType)
            .set_compute_data_type(DataType::FLOAT)
            .set_io_data_type(inputDataType);

        auto dims = testCase.dims;
        auto derivedDims = getDerivedShape(dims);

        // Create input tensor attributes
        auto xAttr
            = graph::makeTensorAttributes("X", dims, generateStrides(dims, layout.strideOrder));
        xAttr.set_uid(BatchnormFwdTrainingTensorIds::X_UID);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        // Channel-only tensors are layout-agnostic, specifying stride order is unnecessary
        auto scaleAttr = graph::makeTensorAttributes(
            "scale", intermediateDataType, derivedDims, generateStrides(derivedDims));
        scaleAttr.set_uid(BatchnormFwdTrainingTensorIds::SCALE_UID);
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr = graph::makeTensorAttributes(
            "bias", intermediateDataType, derivedDims, generateStrides(derivedDims));
        biasAttr.set_uid(BatchnormFwdTrainingTensorIds::BIAS_UID);
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        // Epsilon: use pass-by-value with double (matches MIOpen API)
        auto epsilonTensorAttr = std::make_shared<graph::TensorAttributes>();
        std::mt19937 gen(testCase.seed);
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
            prevRunningMeanAttr.set_uid(BatchnormFwdTrainingTensorIds::PREV_RUNNING_MEAN_UID);
            prevRunningMeanTensorAttr
                = std::make_shared<graph::TensorAttributes>(std::move(prevRunningMeanAttr));

            auto prevRunningVarianceAttr
                = graph::makeTensorAttributes("prev_running_variance",
                                              intermediateDataType,
                                              derivedDims,
                                              generateStrides(derivedDims));
            prevRunningVarianceAttr.set_uid(
                BatchnormFwdTrainingTensorIds::PREV_RUNNING_VARIANCE_UID);
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

        auto [yTensorAttr,
              meanTensorAttr,
              invVarianceTensorAttr,
              nextRunningMeanTensorAttr,
              nextRunningVarianceTensorAttr]
            = graphObj.batchnorm(xTensorAttr, scaleTensorAttr, biasTensorAttr, bnAttrs);

        // Set output tensor attributes
        yTensorAttr->set_output(true);

        // Configure batch statistics outputs if they exist
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
                BatchnormFwdTrainingTensorIds::NEXT_RUNNING_MEAN_UID);
            nextRunningMeanTensorAttr->set_output(true);
            nextRunningMeanTensorAttr->set_data_type(intermediateDataType);
        }

        if(nextRunningVarianceTensorAttr)
        {
            nextRunningVarianceTensorAttr->set_uid(
                BatchnormFwdTrainingTensorIds::NEXT_RUNNING_VARIANCE_UID);
            nextRunningVarianceTensorAttr->set_output(true);
            nextRunningVarianceTensorAttr->set_data_type(intermediateDataType);
        }

        // Register validators for all output tensors
        this->registerValidator(yTensorAttr, tolerance);
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

        this->verifyGraph(graphObj, testCase.seed);
    }

    void initializeBundle([[maybe_unused]] const hipdnn_frontend::graph::Graph& graph,
                          GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        // Note: Epsilon and momentum are pass-by-value (set via set_value()), not buffers

        // X input: default range
        bundle.tensors.at(BatchnormFwdTrainingTensorIds::X_UID)
            ->fillTensorWithRandomValues(-1.0f, 1.0f, seed);

        // Scale and bias: -2.0 to 2.0 to match MIOpen
        bundle.tensors.at(BatchnormFwdTrainingTensorIds::SCALE_UID)
            ->fillTensorWithRandomValues(-2.0f, 2.0f, seed + 1);
        bundle.tensors.at(BatchnormFwdTrainingTensorIds::BIAS_UID)
            ->fillTensorWithRandomValues(-2.0f, 2.0f, seed + 2);

        // Running mean: prev and next must start with SAME values
        // because MIOpen's API uses IN/OUT parameter semantics
        if(bundle.tensors.find(BatchnormFwdTrainingTensorIds::PREV_RUNNING_MEAN_UID)
               != bundle.tensors.end()
           && bundle.tensors.find(BatchnormFwdTrainingTensorIds::NEXT_RUNNING_MEAN_UID)
                  != bundle.tensors.end())
        {
            unsigned runningMeanSeed = seed + 1000;
            bundle.tensors.at(BatchnormFwdTrainingTensorIds::PREV_RUNNING_MEAN_UID)
                ->fillTensorWithRandomValues(-2.0f, 2.0f, runningMeanSeed);
            bundle.tensors.at(BatchnormFwdTrainingTensorIds::NEXT_RUNNING_MEAN_UID)
                ->fillTensorWithRandomValues(-2.0f, 2.0f, runningMeanSeed);
        }

        // Running variance: prev and next must start with SAME values
        if(bundle.tensors.find(BatchnormFwdTrainingTensorIds::PREV_RUNNING_VARIANCE_UID)
               != bundle.tensors.end()
           && bundle.tensors.find(BatchnormFwdTrainingTensorIds::NEXT_RUNNING_VARIANCE_UID)
                  != bundle.tensors.end())
        {
            unsigned runningVarianceSeed = seed + 2000;
            bundle.tensors.at(BatchnormFwdTrainingTensorIds::PREV_RUNNING_VARIANCE_UID)
                ->fillTensorWithRandomValues(-2.0f, 2.0f, runningVarianceSeed);
            bundle.tensors.at(BatchnormFwdTrainingTensorIds::NEXT_RUNNING_VARIANCE_UID)
                ->fillTensorWithRandomValues(-2.0f, 2.0f, runningVarianceSeed);
        }
    }
};

// NCHW 2D
using IntegrationGpuBatchnormFwdTrainingNchwFp32
    = BatchnormForwardTraining<float, float, BatchnormTestCase>;
using IntegrationGpuBatchnormFwdTrainingNchwFp16
    = BatchnormForwardTraining<half, float, BatchnormTestCase>;
using IntegrationGpuBatchnormFwdTrainingNchwBfp16
    = BatchnormForwardTraining<hip_bfloat16, float, BatchnormTestCase>;

// NHWC 2D
using IntegrationGpuBatchnormFwdTrainingNhwcFp32
    = BatchnormForwardTraining<float, float, BatchnormTestCase>;
using IntegrationGpuBatchnormFwdTrainingNhwcFp16
    = BatchnormForwardTraining<half, float, BatchnormTestCase>;
using IntegrationGpuBatchnormFwdTrainingNhwcBfp16
    = BatchnormForwardTraining<hip_bfloat16, float, BatchnormTestCase>;

// NCDHW 3D
using IntegrationGpuBatchnormFwdTrainingNcdhwFp32
    = BatchnormForwardTraining<float, float, BatchnormTestCase>;
using IntegrationGpuBatchnormFwdTrainingNcdhwFp16
    = BatchnormForwardTraining<half, float, BatchnormTestCase>;
using IntegrationGpuBatchnormFwdTrainingNcdhwBfp16
    = BatchnormForwardTraining<hip_bfloat16, float, BatchnormTestCase>;

// NDHWC 3D
using IntegrationGpuBatchnormFwdTrainingNdhwcFp32
    = BatchnormForwardTraining<float, float, BatchnormTestCase>;
using IntegrationGpuBatchnormFwdTrainingNdhwcFp16
    = BatchnormForwardTraining<half, float, BatchnormTestCase>;
using IntegrationGpuBatchnormFwdTrainingNdhwcBfp16
    = BatchnormForwardTraining<hip_bfloat16, float, BatchnormTestCase>;

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

TEST_P(IntegrationGpuBatchnormFwdTrainingNchwFp32, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<float>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NCHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingNchwFp32, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<float>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NCHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingNchwFp16, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<half>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NCHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingNchwFp16, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<half>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NCHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingNchwBfp16, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<hip_bfloat16>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NCHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingNchwBfp16, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<hip_bfloat16>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NCHW);
}

// ============================================================================
// NHWC 2D Tests
// ============================================================================

TEST_P(IntegrationGpuBatchnormFwdTrainingNhwcFp32, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<float>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingNhwcFp32, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<float>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingNhwcFp16, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<half>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingNhwcFp16, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<half>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingNhwcBfp16, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<hip_bfloat16>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingNhwcBfp16, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<hip_bfloat16>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NHWC);
}

// ============================================================================
// NCDHW 3D Tests
// ============================================================================

TEST_P(IntegrationGpuBatchnormFwdTrainingNcdhwFp32, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<float>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingNcdhwFp32, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<float>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingNcdhwFp16, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<half>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingNcdhwFp16, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<half>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingNcdhwBfp16, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<hip_bfloat16>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingNcdhwBfp16, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<hip_bfloat16>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NCDHW);
}

// ============================================================================
// NDHWC 3D Tests
// ============================================================================

TEST_P(IntegrationGpuBatchnormFwdTrainingNdhwcFp32, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<float>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingNdhwcFp32, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<float>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingNdhwcFp16, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<half>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingNdhwcFp16, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<half>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingNdhwcBfp16, DISABLED_FullTraining)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<hip_bfloat16>(),
                             BatchnormTrainingScenario::FULL_TRAINING,
                             TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuBatchnormFwdTrainingNdhwcBfp16, BatchStatsOnly)
{
    runGraphTestWithScenario(batchnorm::getToleranceTraining<hip_bfloat16>(),
                             BatchnormTrainingScenario::WITH_BATCH_STATS,
                             TensorLayout::NDHWC);
}

// ============================================================================
// Test Instantiation
// ============================================================================

// 2D NCHW Tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormFwdTrainingNchwFp32,
                         testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke2dTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormFwdTrainingNchwFp32,
                         testing::ValuesIn(test_bn_common::getBnFwdTrainingFull2dTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormFwdTrainingNchwFp16,
                         testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke2dTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormFwdTrainingNchwFp16,
                         testing::ValuesIn(test_bn_common::getBnFwdTrainingFull2dTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormFwdTrainingNchwBfp16,
                         testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke2dTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormFwdTrainingNchwBfp16,
                         testing::ValuesIn(test_bn_common::getBnFwdTrainingFull2dTestCases()));

// 2D NHWC Tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormFwdTrainingNhwcFp32,
                         testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke2dTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormFwdTrainingNhwcFp32,
                         testing::ValuesIn(test_bn_common::getBnFwdTrainingFull2dTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormFwdTrainingNhwcFp16,
                         testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke2dTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormFwdTrainingNhwcFp16,
                         testing::ValuesIn(test_bn_common::getBnFwdTrainingFull2dTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormFwdTrainingNhwcBfp16,
                         testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke2dTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormFwdTrainingNhwcBfp16,
                         testing::ValuesIn(test_bn_common::getBnFwdTrainingFull2dTestCases()));

// 3D NCDHW Tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormFwdTrainingNcdhwFp32,
                         testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke3dTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormFwdTrainingNcdhwFp32,
                         testing::ValuesIn(test_bn_common::getBnFwdTrainingFull3dTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormFwdTrainingNcdhwFp16,
                         testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke3dTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormFwdTrainingNcdhwFp16,
                         testing::ValuesIn(test_bn_common::getBnFwdTrainingFull3dTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormFwdTrainingNcdhwBfp16,
                         testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke3dTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormFwdTrainingNcdhwBfp16,
                         testing::ValuesIn(test_bn_common::getBnFwdTrainingFull3dTestCases()));

// 3D NDHWC Tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormFwdTrainingNdhwcFp32,
                         testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke3dTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormFwdTrainingNdhwcFp32,
                         testing::ValuesIn(test_bn_common::getBnFwdTrainingFull3dTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormFwdTrainingNdhwcFp16,
                         testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke3dTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormFwdTrainingNdhwcFp16,
                         testing::ValuesIn(test_bn_common::getBnFwdTrainingFull3dTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormFwdTrainingNdhwcBfp16,
                         testing::ValuesIn(test_bn_common::getBnFwdTrainingSmoke3dTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormFwdTrainingNdhwcBfp16,
                         testing::ValuesIn(test_bn_common::getBnFwdTrainingFull3dTestCases()));
