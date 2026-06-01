/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/

#include <algorithm>
#include <cstdlib>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt-ext-op.h>
#include <hipblaslt_init.hpp>
#include <limits>
#include <numeric>
#include <vector>

#include "../include/hipblaslt_random.hpp"
#include "../include/unit.hpp"
#include "hipblaslt_arguments.hpp"

namespace
{
    template <typename DType>
    void cpuSoftmax(DType* m, DType* a, std::uint32_t numRows, std::uint32_t numCols)
    {
        for(std::uint32_t i = 0; i < numRows; ++i)
        {
            const auto rowMax = *std::max_element(a + i * numCols, a + i * numCols + numCols);
            auto       rowSum = 0.f;
            std::transform(a + i * numCols,
                           a + i * numCols + numCols,
                           m + i * numCols,
                           [&rowSum, rowMax](auto v) {
                               const auto u = std::exp(v - rowMax);
                               rowSum += u;
                               return u;
                           });

            std::transform(m + i * numCols,
                           m + i * numCols + numCols,
                           m + i * numCols,
                           [rowSum](auto v) { return v / rowSum; });
        }
    }

    template <typename DType>
    void cpuLayerNorm(DType*        out,
                      DType*        mean,
                      DType*        invvar,
                      DType*        in,
                      std::uint32_t batch,
                      std::uint32_t length,
                      DType         eps   = 1e-05,
                      DType*        gamma = nullptr,
                      DType*        beta  = nullptr)
    {
        // calculate mean
        for(int i = 0; i < batch; i++)
        {
            int    count = 0;
            DType* inC   = in + i * length;
            DType* outC  = out + i * length;

            for(int j = 0; j < length; j++)
            {
                count        = count + 1;
                float delta  = inC[j] - mean[i];
                mean[i]      = mean[i] + delta / count;
                float delta2 = inC[j] - mean[i];
                invvar[i]    = invvar[i] + delta * delta2;
            }
            invvar[i] = 1 / std::sqrt((invvar[i] / length) + eps);

            // calculate invvar
            for(int j = 0; j < length; j++)
            {
                outC[j] = (inC[j] - mean[i]) * invvar[i];

                if(gamma != nullptr)
                    outC[j] = outC[j] * gamma[j];

                if(beta != nullptr)
                    outC[j] = outC[j] + beta[j];
            }
        }
    }

    template <typename T>
    T abs(T a)
    {
        return (a > 0) ? a : -a;
    }

    template <typename T>
    T max(T a, T b)
    {
        return (a > b) ? a : b;
    }

    template <typename Ti, typename To>
    void cpuAMax(To* out, Ti* in, std::uint32_t length)
    {
        // calculate amax
        Ti m = 0;
        for(int j = 0; j < length; j++)
        {
            m = max(m, abs(in[j]));
        }
        out[0] = To(m);
    }

}

enum class amaxInitMethod
{
    hpl = 111,
    nan = 222,
    max = 333,
    min = 444
};

struct AMaxTestData
{
    hipDataType type;
    hipDataType dtype;
    uint32_t    m;
    uint32_t    n;
};

class ExtOpSoftmaxTest : public testing::TestWithParam<uint32_t>
{
};
class ExtOpSoftmaxUnsupportedDatatypeTest : public testing::TestWithParam<hipDataType>
{
};

class ExtOpLayerNormTest : public testing::TestWithParam<uint32_t>
{
};
class ExtOpLayerNormUnsupportedDatatypeTest : public testing::TestWithParam<hipDataType>
{
};

class ExtOpAMaxTest : public testing::TestWithParam<AMaxTestData>
{
};
class ExtOpAMaxUnsupportedDatatypeTest : public testing::TestWithParam<hipDataType>
{
};

TEST_P(ExtOpSoftmaxTest, softmaxSuccess)
{
    uint32_t           m = GetParam();
    uint32_t           n = 16;
    std::vector<float> input(m * n, 0.f);
    std::vector<float> output(m * n, 0.f);
    hipblaslt_uniform_int_1_10_run_float(input.data(), input.size());
    float* gpuInput{};
    float* gpuOutput{};

    auto err          = hipMalloc(&gpuInput, m * n * sizeof(float));
    err               = hipMalloc(&gpuOutput, m * n * sizeof(float));
    err               = hipMemcpyHtoD(gpuInput, input.data(), m * n * sizeof(float));
    auto hipblasltErr = hipblasltExtSoftmax(HIP_R_32F, m, n, 1, gpuOutput, gpuInput, nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_SUCCESS);
    err = hipDeviceSynchronize();
    EXPECT_EQ(err, hipSuccess);
    std::vector<float> cpuRef(m * n, 0.f);
    cpuSoftmax(cpuRef.data(), input.data(), m, n);
    err = hipMemcpyDtoH(output.data(), gpuOutput, m * n * sizeof(float));

    for(std::size_t i = 0; i < m * n; ++i)
    {
        EXPECT_NEAR(output[i], cpuRef[i], 1e-5);
    }

    err = hipFree(gpuInput);
    err = hipFree(gpuOutput);
}

TEST_P(ExtOpLayerNormTest, layernormSuccess)
{
    uint32_t m = GetParam();
    uint32_t n = 16;

    std::vector<float> output(m * n, 0.f);
    std::vector<float> mean(m, 0.f);
    std::vector<float> invvar(m, 0.f);
    std::vector<float> input(m * n, 0.f);
    std::vector<float> gamma(n, 1.f);
    std::vector<float> beta(n, 0.f);

    hipblaslt_init_hpl(input, n, m, n);
    hipblaslt_init_hpl(gamma, n, 1, n);
    hipblaslt_init_hpl(beta, n, 1, n);

    float* gpuOutput{};
    float* gpuMean{};
    float* gpuInvvar{};
    float* gpuInput{};
    float* gpuGamma{};
    float* gpuBeta{};

    auto err = hipMalloc(&gpuOutput, m * n * sizeof(float));
    err      = hipMalloc(&gpuMean, m * sizeof(float));
    err      = hipMalloc(&gpuInvvar, m * sizeof(float));
    err      = hipMalloc(&gpuInput, m * n * sizeof(float));
    err      = hipMalloc(&gpuGamma, n * sizeof(float));
    err      = hipMalloc(&gpuBeta, n * sizeof(float));

    err = hipMemcpyHtoD(gpuInput, input.data(), m * n * sizeof(float));
    err = hipMemcpyHtoD(gpuGamma, gamma.data(), n * sizeof(float));
    err = hipMemcpyHtoD(gpuBeta, beta.data(), n * sizeof(float));

    auto hipblasltErr = hipblasltExtLayerNorm(HIP_R_32F,
                                              gpuOutput,
                                              gpuMean,
                                              gpuInvvar,
                                              gpuInput,
                                              m,
                                              n,
                                              1e-05,
                                              gpuGamma,
                                              gpuBeta,
                                              nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_SUCCESS);
    err = hipDeviceSynchronize();
    EXPECT_EQ(err, hipSuccess);

    std::vector<float> cpuRef(m * n, 0.0f);
    std::vector<float> cpuMean(m, 0.0f);
    std::vector<float> cpuInvvar(m, 0.0f);
    cpuLayerNorm<float>(cpuRef.data(),
                        cpuMean.data(),
                        cpuInvvar.data(),
                        input.data(),
                        m,
                        n,
                        1e-05,
                        gamma.data(),
                        beta.data());

    err = hipMemcpyDtoH(output.data(), gpuOutput, m * n * sizeof(float));
    err = hipMemcpyDtoH(mean.data(), gpuMean, m * sizeof(float));
    err = hipMemcpyDtoH(invvar.data(), gpuInvvar, m * sizeof(float));

    for(std::size_t i = 0; i < m * n; ++i)
    {
        EXPECT_NEAR(output[i], cpuRef[i], 1e-5);
    }
    for(std::size_t i = 0; i < m; ++i)
    {
        EXPECT_NEAR(mean[i], cpuMean[i], 1e-5);
    }
    for(std::size_t i = 0; i < m; ++i)
    {
        EXPECT_NEAR(invvar[i], cpuInvvar[i], 1e-5);
    }

    err = hipFree(gpuOutput);
    err = hipFree(gpuMean);
    err = hipFree(gpuInvvar);
    err = hipFree(gpuInput);
    err = hipFree(gpuGamma);
    err = hipFree(gpuBeta);
}

template <typename Ti, typename To>
void AMaxTest(hipDataType type, hipDataType dtype, std::size_t m, std::size_t n)
{
    std::size_t numElements = m * n;
    std::size_t inNumBytes  = sizeof(Ti);
    std::size_t outNumBytes = sizeof(To);

    To* gpuOutput{nullptr};
    Ti* gpuInput{nullptr};

    auto hipErr = hipMalloc(&gpuOutput, outNumBytes);
    hipErr      = hipMalloc(&gpuInput, m * n * inNumBytes);

    std::vector<To> cpuOutput(1, 0.f);
    std::vector<Ti> cpuInput(m * n, 0.f);
    std::vector<To> refOutput(1, 0.f);

    hipblaslt_init_hpl(cpuInput, m * n, 1, m * n);

    hipErr = hipMemcpyHtoD(gpuInput, cpuInput.data(), m * n * inNumBytes);

    hipStream_t stream{};
    hipErr            = hipStreamCreate(&stream);
    auto hipblasltErr = hipblasltExtAMax(type, dtype, gpuOutput, gpuInput, m, n, stream);

    hipErr = hipMemcpyDtoH(cpuOutput.data(), gpuOutput, outNumBytes);

    cpuAMax(refOutput.data(), cpuInput.data(), m * n);

    EXPECT_NEAR(float(refOutput[0]), float(cpuOutput[0]), 1e-5);

    hipErr = hipStreamDestroy(stream);
    hipErr = hipFree(gpuOutput);
    hipErr = hipFree(gpuInput);
}

TEST_P(ExtOpAMaxTest, amaxSuccess)
{
    AMaxTestData    testdata = GetParam();

    if(testdata.type == HIP_R_32F && testdata.dtype == HIP_R_32F)
    {
        AMaxTest<float, float>(testdata.type, testdata.dtype, testdata.m, testdata.n);
    }
    else if(testdata.type == HIP_R_32F && testdata.dtype == HIP_R_16F)
    {
        AMaxTest<float, hipblasLtHalf>(testdata.type, testdata.dtype, testdata.m, testdata.n);
    }
    else if(testdata.type == HIP_R_16F && testdata.dtype == HIP_R_32F)
    {
        AMaxTest<hipblasLtHalf, float>(testdata.type, testdata.dtype, testdata.m, testdata.n);
    }
    else if(testdata.type == HIP_R_16F && testdata.dtype == HIP_R_16F)
    {
        AMaxTest<hipblasLtHalf, hipblasLtHalf>(
            testdata.type, testdata.dtype, testdata.m, testdata.n);
    }
}

TEST_P(ExtOpSoftmaxUnsupportedDatatypeTest, softmaxFailureUnsupportedDatatype)
{
    auto hipblasltErr = hipblasltExtSoftmax(GetParam(), 16, 16, 1, nullptr, nullptr, nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_NOT_SUPPORTED);
}

TEST(ExtOpTest, softmaxFailureUnsupportedShapeOrReductionDim)
{
    auto hipblasltErr = hipblasltExtSoftmax(HIP_R_32F, 16, 512, 1, nullptr, nullptr, nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_INVALID_VALUE);
    hipblasltErr = hipblasltExtSoftmax(HIP_R_32F, 16, 16, 0, nullptr, nullptr, nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_NOT_SUPPORTED);
}

TEST_P(ExtOpLayerNormUnsupportedDatatypeTest, layernormFailureUnsupportedDatatype)
{
    auto hipblasltErr = hipblasltExtLayerNorm(
        GetParam(), nullptr, nullptr, nullptr, nullptr, 16, 1024, 1e-05, nullptr, nullptr, nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_NOT_SUPPORTED);
}

TEST(ExtOpTest, layernormFailureInvalidValue)
{
    auto hipblasltErr = hipblasltExtLayerNorm(
        HIP_R_32F, nullptr, nullptr, nullptr, nullptr, 16, 1024, 1e-05, nullptr, nullptr, nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_P(ExtOpAMaxUnsupportedDatatypeTest, amaxFailureUnsupportedDatatype)
{
    auto hipblasltErr = hipblasltExtAMax(GetParam(), GetParam(), nullptr, nullptr, 0, 0, nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_NOT_SUPPORTED);
}

TEST(ExtOpTest, amaxFailureInvalidValue)
{
    auto hipblasltErr = hipblasltExtAMax(HIP_R_32F, HIP_R_32F, nullptr, nullptr, 0, 0, nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_INVALID_VALUE);
}

INSTANTIATE_TEST_SUITE_P(ExtOpTest, ExtOpSoftmaxTest, testing::Values<uint32_t>(1, 16, 1335));
INSTANTIATE_TEST_SUITE_P(ExtOpTest,
                         ExtOpSoftmaxUnsupportedDatatypeTest,
                         testing::Values<hipDataType>(HIP_R_16F, HIP_R_16BF));

INSTANTIATE_TEST_SUITE_P(ExtOpTest,
                         ExtOpLayerNormTest,
                         testing::Values<uint32_t>(1, 16, 1335, 6666));
INSTANTIATE_TEST_SUITE_P(ExtOpTest,
                         ExtOpLayerNormUnsupportedDatatypeTest,
                         testing::Values<hipDataType>(HIP_R_16F, HIP_R_16BF));

INSTANTIATE_TEST_SUITE_P(
    ExtOpTest,
    ExtOpAMaxTest,
    testing::Values<AMaxTestData>(AMaxTestData{HIP_R_32F, HIP_R_32F, 1, 1},
                                  AMaxTestData{HIP_R_32F, HIP_R_32F, 16, 16},
                                  AMaxTestData{HIP_R_32F, HIP_R_32F, 1335, 666},
                                  AMaxTestData{HIP_R_32F, HIP_R_16F, 1, 1},
                                  AMaxTestData{HIP_R_32F, HIP_R_16F, 16, 16},
                                  AMaxTestData{HIP_R_32F, HIP_R_16F, 1335, 666},
                                  AMaxTestData{HIP_R_16F, HIP_R_32F, 1, 1},
                                  AMaxTestData{HIP_R_16F, HIP_R_32F, 16, 16},
                                  AMaxTestData{HIP_R_16F, HIP_R_32F, 1335, 666},
                                  AMaxTestData{HIP_R_16F, HIP_R_16F, 1, 1},
                                  AMaxTestData{HIP_R_16F, HIP_R_16F, 16, 16},
                                  AMaxTestData{HIP_R_16F, HIP_R_16F, 1335, 666}));
INSTANTIATE_TEST_SUITE_P(ExtOpTest,
                         ExtOpAMaxUnsupportedDatatypeTest,
                         testing::Values<hipDataType>(HIP_R_16BF));
