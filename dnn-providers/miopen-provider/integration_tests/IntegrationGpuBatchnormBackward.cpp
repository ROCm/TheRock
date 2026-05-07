// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
#include <random>

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../tests/common/BatchnormCommon.hpp"
#include "IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace miopen_legacy_plugin::test_utilities;
using namespace test_bn_common;

namespace
{

struct BatchnormBwdTensorIds
{
    static constexpr int64_t X_UID = 1;
    static constexpr int64_t DY_UID = 2;
    static constexpr int64_t SCALE_UID = 3;
    static constexpr int64_t MEAN_UID = 4;
    static constexpr int64_t INV_VARIANCE_UID = 5;
};

template <typename DataType, typename IntermediateType, bool CalcStats = false>
class BatchnormBackward : public IntegrationGraphVerificationHarness<DataType, BatchnormTestCase>
{
protected:
    void initializeBundle([[maybe_unused]] const graph::Graph& graph,
                          GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        bundle.tensors.at(BatchnormBwdTensorIds::X_UID)
            ->fillTensorWithRandomValues(-1.0f, 1.0f, seed);
        bundle.tensors.at(BatchnormBwdTensorIds::DY_UID)
            ->fillTensorWithRandomValues(-0.1f, 0.1f, seed);
        bundle.tensors.at(BatchnormBwdTensorIds::SCALE_UID)
            ->fillTensorWithRandomValues(-0.1f, 0.1f, seed);

        if(!CalcStats)
        {
            bundle.tensors.at(BatchnormBwdTensorIds::MEAN_UID)
                ->fillTensorWithRandomValues(-0.1f, 0.1f, seed);

            bundle.tensors.at(BatchnormBwdTensorIds::INV_VARIANCE_UID)
                ->fillTensorWithRandomValues(1.9f, 2.0f, seed);
        }
    }

    void runGraphTest(DataType tolerance, const TensorLayout& layout = TensorLayout::NCHW) override
    {
        const BatchnormTestCase& testCase = this->GetParam();

        auto derivedDims = getDerivedShape(testCase.dims);

        auto dataType = getDataTypeEnumFromType<DataType>();
        auto intermediateDataType = getDataTypeEnumFromType<IntermediateType>();

        hipdnn_frontend::graph::Graph graphObj;

        graphObj.set_name("BatchnormBackwardTest");
        graphObj.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
        graphObj.set_intermediate_data_type(intermediateDataType);
        graphObj.set_io_data_type(dataType);

        auto xAttr = graph::makeTensorAttributes(
            "x", testCase.dims, generateStrides(testCase.dims, layout.strideOrder));
        xAttr.set_uid(BatchnormBwdTensorIds::X_UID);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto dyAttr = graph::makeTensorAttributes(
            "dy", testCase.dims, generateStrides(testCase.dims, layout.strideOrder));
        dyAttr.set_uid(BatchnormBwdTensorIds::DY_UID);
        auto dyTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dyAttr));

        // Channel-only tensors are layout-agnostic, specifying stride order is unnecessary
        auto scaleAttr = graph::makeTensorAttributes(
            "scale", intermediateDataType, derivedDims, generateStrides(derivedDims));
        scaleAttr.set_uid(BatchnormBwdTensorIds::SCALE_UID);
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        graph::BatchnormBackwardAttributes bnAttrs;

        if(!CalcStats)
        {
            auto meanAttr = graph::makeTensorAttributes(
                "mean", intermediateDataType, derivedDims, generateStrides(derivedDims));
            meanAttr.set_uid(BatchnormBwdTensorIds::MEAN_UID);
            auto meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));

            auto invVarianceAttr = graph::makeTensorAttributes(
                "inv_variance", intermediateDataType, derivedDims, generateStrides(derivedDims));
            invVarianceAttr.set_uid(BatchnormBwdTensorIds::INV_VARIANCE_UID);
            auto invVarianceTensorAttr
                = std::make_shared<graph::TensorAttributes>(std::move(invVarianceAttr));

            bnAttrs.set_saved_mean_and_inv_variance(meanTensorAttr, invVarianceTensorAttr);
        }

        auto outputTensorsAttr
            = graphObj.batchnorm_backward(dyTensorAttr, xTensorAttr, scaleTensorAttr, bnAttrs);

        auto& dxTensorAttr = outputTensorsAttr[0];
        dxTensorAttr->set_output(true);

        auto& dscaleTensorAttr = outputTensorsAttr[1];
        dscaleTensorAttr->set_data_type(intermediateDataType);
        dscaleTensorAttr->set_output(true);

        auto& dbiasTensorAttr = outputTensorsAttr[2];
        dbiasTensorAttr->set_data_type(intermediateDataType);
        dbiasTensorAttr->set_output(true);

        auto intermediateTolerance = batchnorm::getToleranceBackward<IntermediateType>();

        this->registerValidator(dxTensorAttr, tolerance);
        this->registerValidator(dscaleTensorAttr, intermediateTolerance);
        this->registerValidator(dbiasTensorAttr, intermediateTolerance);

        this->verifyGraph(graphObj, testCase.seed);
    }
};

using IntegrationGpuBatchnormBackwardNchwFp32 = BatchnormBackward<float, float>;

using IntegrationGpuBatchnormBackwardNchwBfp16 = BatchnormBackward<hip_bfloat16, float>;

using IntegrationGpuBatchnormBackwardNchwFp16 = BatchnormBackward<half, float>;

using IntegrationGpuBatchnormBackwardNhwcFp32 = BatchnormBackward<float, float>;

using IntegrationGpuBatchnormBackwardNhwcBfp16 = BatchnormBackward<hip_bfloat16, float>;

using IntegrationGpuBatchnormBackwardNhwcFp16 = BatchnormBackward<half, float>;

using IntegrationGpuBatchnormBackwardNcdhwFp32 = BatchnormBackward<float, float>;

using IntegrationGpuBatchnormBackwardNcdhwBfp16 = BatchnormBackward<hip_bfloat16, float>;

using IntegrationGpuBatchnormBackwardNcdhwFp16 = BatchnormBackward<half, float>;

using IntegrationGpuBatchnormBackwardNdhwcFp32 = BatchnormBackward<float, float>;

using IntegrationGpuBatchnormBackwardNdhwcBfp16 = BatchnormBackward<hip_bfloat16, float>;

using IntegrationGpuBatchnormBackwardNdhwcFp16 = BatchnormBackward<half, float>;

using IntegrationGpuBatchnormBackwardCalcStatsNchwFp32 = BatchnormBackward<float, float, true>;

using IntegrationGpuBatchnormBackwardCalcStatsNchwBfp16
    = BatchnormBackward<hip_bfloat16, float, true>;

using IntegrationGpuBatchnormBackwardCalcStatsNchwFp16 = BatchnormBackward<half, float, true>;

using IntegrationGpuBatchnormBackwardCalcStatsNhwcFp32 = BatchnormBackward<float, float, true>;

using IntegrationGpuBatchnormBackwardCalcStatsNhwcBfp16
    = BatchnormBackward<hip_bfloat16, float, true>;

using IntegrationGpuBatchnormBackwardCalcStatsNhwcFp16 = BatchnormBackward<half, float, true>;

using IntegrationGpuBatchnormBackwardCalcStatsNcdhwFp32 = BatchnormBackward<float, float, true>;

using IntegrationGpuBatchnormBackwardCalcStatsNcdhwBfp16
    = BatchnormBackward<hip_bfloat16, float, true>;

using IntegrationGpuBatchnormBackwardCalcStatsNcdhwFp16 = BatchnormBackward<half, float, true>;

using IntegrationGpuBatchnormBackwardCalcStatsNdhwcFp32 = BatchnormBackward<float, float, true>;

using IntegrationGpuBatchnormBackwardCalcStatsNdhwcBfp16
    = BatchnormBackward<hip_bfloat16, float, true>;

using IntegrationGpuBatchnormBackwardCalcStatsNdhwcFp16 = BatchnormBackward<half, float, true>;

} // namespace

TEST_P(IntegrationGpuBatchnormBackwardNchwFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNchwFp32,
                         testing::ValuesIn(getBnBwdTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormBackwardNchwFp32,
                         testing::ValuesIn(getBnBwdFullTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNchwBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNchwBfp16,
                         testing::ValuesIn(getBnBwdTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormBackwardNchwBfp16,
                         testing::ValuesIn(getBnBwdFullTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNchwFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNchwFp16,
                         testing::ValuesIn(getBnBwdTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormBackwardNchwFp16,
                         testing::ValuesIn(getBnBwdFullTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNhwcFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNhwcFp32,
                         testing::ValuesIn(getBnBwdTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormBackwardNhwcFp32,
                         testing::ValuesIn(getBnBwdFullTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNhwcBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNhwcBfp16,
                         testing::ValuesIn(getBnBwdTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormBackwardNhwcBfp16,
                         testing::ValuesIn(getBnBwdFullTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNhwcFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNhwcFp16,
                         testing::ValuesIn(getBnBwdTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormBackwardNhwcFp16,
                         testing::ValuesIn(getBnBwdFullTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNcdhwFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNcdhwFp32,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNcdhwBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNcdhwBfp16,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNcdhwFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNcdhwFp16,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNdhwcFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNdhwcFp32,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNdhwcBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNdhwcBfp16,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNdhwcFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNdhwcFp16,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardCalcStatsNchwFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardCalcStatsNchwFp32,
                         testing::ValuesIn(getBnBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardCalcStatsNchwBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardCalcStatsNchwBfp16,
                         testing::ValuesIn(getBnBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardCalcStatsNchwFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardCalcStatsNchwFp16,
                         testing::ValuesIn(getBnBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardCalcStatsNhwcFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardCalcStatsNhwcFp32,
                         testing::ValuesIn(getBnBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardCalcStatsNhwcBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardCalcStatsNhwcBfp16,
                         testing::ValuesIn(getBnBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardCalcStatsNhwcFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardCalcStatsNhwcFp16,
                         testing::ValuesIn(getBnBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardCalcStatsNcdhwFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardCalcStatsNcdhwFp32,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardCalcStatsNcdhwBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardCalcStatsNcdhwBfp16,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardCalcStatsNcdhwFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardCalcStatsNcdhwFp16,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardCalcStatsNdhwcFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardCalcStatsNdhwcFp32,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardCalcStatsNdhwcBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardCalcStatsNdhwcBfp16,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardCalcStatsNdhwcFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardCalcStatsNdhwcFp16,
                         testing::ValuesIn(getBnBwd3dTestCases()));
