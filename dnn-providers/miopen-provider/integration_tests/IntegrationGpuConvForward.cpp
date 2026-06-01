// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
#include <random>

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../tests/common/ConvolutionCommon.hpp"
#include "IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace miopen_legacy_plugin::test_utilities;
using namespace test_conv_common;

namespace
{

template <typename DataType>
class ConvForward : public IntegrationGraphVerificationHarness<DataType, ConvTestCase>
{
protected:
    void runGraphTest(DataType tolerance, const TensorLayout& layout = TensorLayout::NCHW) override
    {
        // Skipping until CK is working on Windows
        SKIP_IF_WINDOWS();

        const ConvTestCase& testCase = this->GetParam();

        hipdnn_frontend::graph::Graph graphObj;
        graphObj.set_name("ConvolutionForwardTest");

        auto dataType = getDataTypeEnumFromType<DataType>();
        graphObj.set_intermediate_data_type(dataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto xAttr = graph::makeTensorAttributes(
            "x", testCase.xDims, generateStrides(testCase.xDims, layout.strideOrder));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto wAttr = graph::makeTensorAttributes(
            "w", testCase.wDims, generateStrides(testCase.wDims, layout.strideOrder));
        auto wTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(wAttr));

        graph::ConvFpropAttributes convAttrs;
        convAttrs.set_pre_padding(testCase.convPrePadding);
        convAttrs.set_post_padding(testCase.convPostPadding);
        convAttrs.set_stride(testCase.convStride);
        convAttrs.set_dilation(testCase.convDilation);

        auto yAttr = graphObj.conv_fprop(xTensorAttr, wTensorAttr, convAttrs);

        yAttr->set_output(true);

        this->registerValidator(yAttr, tolerance);

        this->verifyGraph(graphObj, testCase.seed);
    }
};

using IntegrationGpuConvFwdNchwFp32 = ConvForward<float>;
using IntegrationGpuConvFwdNcdhwFp32 = ConvForward<float>;

using IntegrationGpuConvFwdNchwBfp16 = ConvForward<hip_bfloat16>;
using IntegrationGpuConvFwdNcdhwBfp16 = ConvForward<hip_bfloat16>;

using IntegrationGpuConvFwdNchwFp16 = ConvForward<half>;
using IntegrationGpuConvFwdNcdhwFp16 = ConvForward<half>;

using IntegrationGpuConvFwdNhwcFp32 = ConvForward<float>;
using IntegrationGpuConvFwdNdhwcFp32 = ConvForward<float>;

using IntegrationGpuConvFwdNhwcBfp16 = ConvForward<hip_bfloat16>;
using IntegrationGpuConvFwdNdhwcBfp16 = ConvForward<hip_bfloat16>;

using IntegrationGpuConvFwdNhwcFp16 = ConvForward<half>;
using IntegrationGpuConvFwdNdhwcFp16 = ConvForward<half>;

} // namespace

TEST_P(IntegrationGpuConvFwdNchwFp32, Correctness)
{
    runGraphTest(4e-6f, TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvFwdNcdhwFp32, Correctness)
{
    runGraphTest(conv::getToleranceFwd<float>(), TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvFwdNchwBfp16, Correctness)
{
    runGraphTest(conv::getToleranceFwd<hip_bfloat16>(), TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvFwdNcdhwBfp16, Correctness)
{
    runGraphTest(conv::getToleranceFwd<hip_bfloat16>(), TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvFwdNchwFp16, Correctness)
{
    runGraphTest(conv::getToleranceFwd<half>(), TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvFwdNcdhwFp16, Correctness)
{
    runGraphTest(conv::getToleranceFwd<half>(), TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvFwdNhwcFp32, Correctness)
{
    runGraphTest(conv::getToleranceFwd<float>(), TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvFwdNdhwcFp32, Correctness)
{
    runGraphTest(conv::getToleranceFwd<float>(), TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuConvFwdNhwcBfp16, Correctness)
{
    runGraphTest(conv::getToleranceFwd<hip_bfloat16>(), TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvFwdNdhwcBfp16, Correctness)
{
    runGraphTest(conv::getToleranceFwd<hip_bfloat16>(), TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuConvFwdNhwcFp16, Correctness)
{
    runGraphTest(conv::getToleranceFwd<half>(), TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvFwdNdhwcFp16, Correctness)
{
    runGraphTest(conv::getToleranceFwd<half>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvFwdNchwFp32,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvFwdNchwBfp16,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvFwdNchwFp16,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvFwdNhwcFp32,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvFwdNhwcBfp16,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvFwdNhwcFp16,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvFwdNcdhwFp32,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvFwdNcdhwBfp16,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvFwdNcdhwFp16,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvFwdNdhwcFp32,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvFwdNdhwcBfp16,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvFwdNdhwcFp16,
                         testing::ValuesIn(getConvTestCases5D()));
