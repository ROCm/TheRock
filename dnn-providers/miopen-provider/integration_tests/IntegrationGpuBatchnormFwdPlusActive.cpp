// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
#include <random>

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../tests/common/ActivationCommon.hpp"
#include "../tests/common/BatchnormCommon.hpp"

#include "IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace miopen_legacy_plugin::test_utilities;
using namespace test_bn_common;

namespace
{

template <typename DataType, typename IntermediateType, typename TestCaseType>
class BatchnormFwdPlusActiv : public IntegrationGraphVerificationHarness<DataType, TestCaseType>
{
protected:
    void runGraphTest(DataType tolerance, const TensorLayout& layout = TensorLayout::NCHW) override
    {
        const auto& [testCase, activeCase] = this->GetParam();

        auto derivedDims = getDerivedShape(testCase.dims);

        hipdnn_frontend::graph::Graph graphObj;
        graphObj.set_name("BatchnormFwd+ActivTest");

        auto dataType = getDataTypeEnumFromType<DataType>();
        auto intermediateDataType = getDataTypeEnumFromType<IntermediateType>();
        graphObj.set_intermediate_data_type(intermediateDataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto xAttr = graph::makeTensorAttributes(
            "x", testCase.dims, generateStrides(testCase.dims, layout.strideOrder));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        // Channel-only tensors are layout-agnostic, specifying stride order is unnecessary
        auto meanAttr = graph::makeTensorAttributes(
            "mean", intermediateDataType, derivedDims, generateStrides(derivedDims));
        auto meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));

        auto invVarianceAttr = graph::makeTensorAttributes(
            "inv_variance", intermediateDataType, derivedDims, generateStrides(derivedDims));
        auto invVarianceTensorAttr
            = std::make_shared<graph::TensorAttributes>(std::move(invVarianceAttr));

        auto scaleAttr = graph::makeTensorAttributes(
            "scale", intermediateDataType, derivedDims, generateStrides(derivedDims));
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr = graph::makeTensorAttributes(
            "bias", intermediateDataType, derivedDims, generateStrides(derivedDims));
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        graph::BatchnormInferenceAttributes bnAttrs;

        auto yTensorAttr = graphObj.batchnorm_inference(xTensorAttr,
                                                        meanTensorAttr,
                                                        invVarianceTensorAttr,
                                                        scaleTensorAttr,
                                                        biasTensorAttr,
                                                        bnAttrs);
        yTensorAttr->set_data_type(dataType);

        graph::PointwiseAttributes pointwiseAttrs;
        pointwiseAttrs.set_mode(static_cast<hipdnn_frontend::PointwiseMode>(activeCase.mode));
        if(activeCase.reluLowerClip.has_value())
        {
            pointwiseAttrs.set_relu_lower_clip(activeCase.reluLowerClip.value());
        }
        if(activeCase.reluUpperClip.has_value())
        {
            pointwiseAttrs.set_relu_upper_clip(activeCase.reluUpperClip.value());
        }
        if(activeCase.reluLowerClipSlope.has_value())
        {
            pointwiseAttrs.set_relu_lower_clip_slope(activeCase.reluLowerClipSlope.value());
        }
        if(activeCase.swishBeta.has_value())
        {
            pointwiseAttrs.set_swish_beta(activeCase.swishBeta.value());
        }
        if(activeCase.eluAlpha.has_value())
        {
            pointwiseAttrs.set_elu_alpha(activeCase.eluAlpha.value());
        }
        if(activeCase.softplusBeta.has_value())
        {
            pointwiseAttrs.set_softplus_beta(activeCase.softplusBeta.value());
        }

        auto outTensorAttr = graphObj.pointwise(yTensorAttr, pointwiseAttrs);
        outTensorAttr->set_output(true);

        this->registerValidator(outTensorAttr, tolerance);
        this->verifyGraph(graphObj, testCase.seed);
    }
};

//NCHW
using IntegrationGpuBatchnormFwdPlusActivNchwFp32
    = BatchnormFwdPlusActiv<float,
                            float,
                            std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormFwdPlusActivNchwBfp16
    = BatchnormFwdPlusActiv<hip_bfloat16,
                            float,
                            std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormFwdPlusActivNchwFp16
    = BatchnormFwdPlusActiv<half,
                            float,
                            std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

//NHWC
using IntegrationGpuBatchnormFwdPlusActivNhwcFp32
    = BatchnormFwdPlusActiv<float,
                            float,
                            std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormFwdPlusActivNhwcBfp16
    = BatchnormFwdPlusActiv<hip_bfloat16,
                            float,
                            std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormFwdPlusActivNhwcFp16
    = BatchnormFwdPlusActiv<half,
                            float,
                            std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

//NCDHW
using IntegrationGpuBatchnormFwdPlusActivNcdhwFp32
    = BatchnormFwdPlusActiv<float,
                            float,
                            std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormFwdPlusActivNcdhwBfp16
    = BatchnormFwdPlusActiv<hip_bfloat16,
                            float,
                            std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormFwdPlusActivNcdhwFp16
    = BatchnormFwdPlusActiv<half,
                            float,
                            std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

//NDHWC
using IntegrationGpuBatchnormFwdPlusActivNdhwcFp32
    = BatchnormFwdPlusActiv<float,
                            float,
                            std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormFwdPlusActivNdhwcBfp16
    = BatchnormFwdPlusActiv<hip_bfloat16,
                            float,
                            std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormFwdPlusActivNdhwcFp16
    = BatchnormFwdPlusActiv<half,
                            float,
                            std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

} // namespace

TEST_P(IntegrationGpuBatchnormFwdPlusActivNchwFp32, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<float>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdPlusActivNchwFp32,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdPlusActivNchwFp32,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormFwdPlusActivNchwBfp16, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<hip_bfloat16>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdPlusActivNchwBfp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdPlusActivNchwBfp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormFwdPlusActivNchwFp16, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<half>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdPlusActivNchwFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdPlusActivNchwFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormFwdPlusActivNhwcFp32, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<float>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdPlusActivNhwcFp32,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdPlusActivNhwcFp32,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormFwdPlusActivNhwcBfp16, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<hip_bfloat16>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdPlusActivNhwcBfp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdPlusActivNhwcBfp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormFwdPlusActivNhwcFp16, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<half>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdPlusActivNhwcFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormFwdPlusActivNhwcFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

//Ncdhw
TEST_P(IntegrationGpuBatchnormFwdPlusActivNcdhwFp32, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<float>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdPlusActivNcdhwFp32,
    testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormFwdPlusActivNcdhwBfp16, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<hip_bfloat16>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdPlusActivNcdhwBfp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormFwdPlusActivNcdhwFp16, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<half>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdPlusActivNcdhwFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

//NDHWC
TEST_P(IntegrationGpuBatchnormFwdPlusActivNdhwcFp32, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<float>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdPlusActivNdhwcFp32,
    testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormFwdPlusActivNdhwcBfp16, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<hip_bfloat16>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdPlusActivNdhwcBfp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormFwdPlusActivNdhwcFp16, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<half>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdPlusActivNdhwcFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
