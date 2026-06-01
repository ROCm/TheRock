/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <algorithm>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <iostream>

#include "TensorDataManipulation.hpp"
#include "datatype_interface.hpp"
#include "helper.h"

void calculateKforSwizzling(hipDataType datatype, size_t& MiK, size_t& MiKv, size_t& PackK)
{
    switch(datatype)
    {
    case HIP_R_32F:
        MiK  = 4;
        MiKv = 1;
        break;
    case HIP_R_16F:
    case HIP_R_16BF:
        MiK  = 16;
        MiKv = 4;
        break;
    case HIP_R_8F_E4M3_FNUZ:
    case HIP_R_8F_E5M2_FNUZ:
        MiK  = 32;
        MiKv = 8;
        break;
    default:
        std::cerr << "unsupported datatype in calculateKforSwizzling" << '\n';
    }

    PackK = 16 / MiKv / realDataTypeSize(datatype);
}

template <typename T>
void swizzleTensor(T* dst, const T* src, size_t n, size_t k, bool colMaj)
{
    using Tensor = Tensor::Manipulation::Tensor;
    size_t MiN   = 16;
    size_t MiK = 0, MiKv = 0, PackK = 0;
    calculateKforSwizzling(hipblaslt_type2datatype<T>(), MiK, MiKv, PackK);
    auto tmpTensor = Tensor::create<T>({n, k});
    std::copy(src, src + (n * k), tmpTensor.template as<T>());

    if(colMaj)
    {
        auto orgTensor = Tensor::create<T>({k, n});
        std::copy(src, src + (n * k), orgTensor.template as<T>());
        tmpTensor = permute(orgTensor, {1, 0});
    }

    tmpTensor.reshape({n / MiN, MiN, k / (MiK * PackK), MiK / MiKv, MiKv * PackK});
    Tensor permuted = permute(tmpTensor, {0, 2, 3, 1, 4});
    std::copy(permuted.template as<T>(), permuted.template as<T>() + (n * k), dst);
}

void simpleGemm(hipblasLtHandle_t  handle,
                hipblasOperation_t trans_a,
                hipblasOperation_t trans_b,
                int64_t            m,
                int64_t            n,
                int64_t            k,
                int64_t            batch_count,
                float&             alpha,
                float&             beta,
                void*              d_a,
                void*              d_b,
                void*              d_c,
                void*              d_d,
                void*              d_workspace,
                int64_t            max_workspace_size,
                hipDataType        TiAB,
                bool               swizzleB,
                hipStream_t        stream);

int main()
{
    constexpr int64_t m{1280};
    constexpr int64_t n{1024};
    constexpr int64_t k{512};

    // Non-swizzle runner: TN, batch count = 1, alpha, beta = 1.0f
    Runner<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float, float> runner(
        m, n, k, 1, 1.f, 1.f, 32 * 128 * 128);

    runner.run([&runner] {
        simpleGemm(runner.handle,
                   HIPBLAS_OP_T,
                   HIPBLAS_OP_N,
                   runner.m,
                   runner.n,
                   runner.k,
                   runner.batch_count,
                   runner.alpha,
                   runner.beta,
                   runner.d_a,
                   runner.d_b,
                   runner.d_c,
                   runner.d_d,
                   runner.d_workspace,
                   runner.max_workspace_size,
                   HIP_R_16F,
                   false,
                   runner.stream);
    });

    // swizzleB runner: TN, batch count = 1, alpha, beta = 1.0f
    Runner<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float, float> swizzleRunner(
        m, n, k, 1, 1.f, 1.f, 32 * 128 * 128);

    swizzleRunner.run([&swizzleRunner, &runner, m, n, k] {
        // copy inputs from first runner for comparison and validation
        CHECK_HIP_ERROR(hipMemcpy(
            swizzleRunner.d_a, runner.d_a, m * k * sizeof(hipblasLtHalf), hipMemcpyDeviceToDevice));
        CHECK_HIP_ERROR(hipMemcpy(
            swizzleRunner.d_b, runner.d_b, n * k * sizeof(hipblasLtHalf), hipMemcpyDeviceToDevice));
        CHECK_HIP_ERROR(hipMemcpy(
            swizzleRunner.d_c, runner.d_c, m * n * sizeof(hipblasLtHalf), hipMemcpyDeviceToDevice));

        simpleGemm(swizzleRunner.handle,
                   HIPBLAS_OP_T,
                   HIPBLAS_OP_N,
                   swizzleRunner.m,
                   swizzleRunner.n,
                   swizzleRunner.k,
                   swizzleRunner.batch_count,
                   swizzleRunner.alpha,
                   swizzleRunner.beta,
                   swizzleRunner.d_a,
                   swizzleRunner.d_b,
                   swizzleRunner.d_c,
                   swizzleRunner.d_d,
                   swizzleRunner.d_workspace,
                   swizzleRunner.max_workspace_size,
                   HIP_R_16F,
                   true,
                   swizzleRunner.stream);
    });

    // Compare results from non-swizzling with swizzling
    const hipblasLtHalf* regularCpuD  = static_cast<hipblasLtHalf*>(runner.d);
    const hipblasLtHalf* swizzledCpuD = static_cast<hipblasLtHalf*>(swizzleRunner.d);

    for(size_t i = 0; i < m * n; ++i)
    {
        const auto diff = std::abs(float(regularCpuD[i] - float(swizzledCpuD[i])));
        if(diff > 1e-5)
        {
            std::cerr << "Swizzle Validation Error at index: " << i << ", diff: " << diff << '\n';
            break;
        }
    }

    std::cout << "Matrix multiplication and validation completed successfully." << std::endl;

    return 0;
}

void simpleGemm(hipblasLtHandle_t  handle,
                hipblasOperation_t trans_a,
                hipblasOperation_t trans_b,
                int64_t            m,
                int64_t            n,
                int64_t            k,
                int64_t            batch_count,
                float&             alpha,
                float&             beta,
                void*              d_a,
                void*              d_b,
                void*              d_c,
                void*              d_d,
                void*              d_workspace,
                int64_t            max_workspace_size,
                hipDataType        TiAB,
                bool               swizzleB,
                hipStream_t        stream)
{
    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, TiAB, k, m, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, TiAB, k, n, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, m, n, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, HIP_R_16F, m, n, m));

    // swizzle case and input = FP16
    if(swizzleB && TiAB == HIP_R_16F)
    {
        hipblasLtOrder_t orderB = HIPBLASLT_ORDER_COL16_4R8;
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matB, HIPBLASLT_MATRIX_LAYOUT_ORDER, &orderB, sizeof(orderB)));
        std::vector<hipblasLtHalf> src(n * k, 0);
        std::vector<hipblasLtHalf> dst(n * k, 0);

        // pre-shuffle input data in host memory
        CHECK_HIP_ERROR(
            hipMemcpy(src.data(), d_b, n * k * sizeof(hipblasLtHalf), hipMemcpyDeviceToHost));
        swizzleTensor(dst.data(), src.data(), n, k, false);
        CHECK_HIP_ERROR(
            hipMemcpy(d_b, dst.data(), n * k * sizeof(hipblasLtHalf), hipMemcpyHostToDevice));
    }

    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));

    // Set User Preference attributes
    hipblasLtMatmulPreference_t pref;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatmulPreferenceSetAttribute(pref,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &max_workspace_size,
                                              sizeof(max_workspace_size)));

    const int                        request_solutions = 1;
    hipblasLtMatmulHeuristicResult_t heuristicResult[request_solutions];
    int                              returnedAlgoCount = 0;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                          matmul,
                                                          matA,
                                                          matB,
                                                          matC,
                                                          matD,
                                                          pref,
                                                          request_solutions,
                                                          heuristicResult,
                                                          &returnedAlgoCount));

    if(returnedAlgoCount == 0)
    {
        std::cerr << "No valid solution found!" << std::endl;
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
        return;
    }

    uint64_t workspace_size = 0;
    for(int i = 0; i < returnedAlgoCount; i++)
        workspace_size = max(workspace_size, heuristicResult[i].workspaceSize);

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle,
                                          matmul,
                                          &alpha,
                                          d_a,
                                          matA,
                                          d_b,
                                          matB,
                                          &beta,
                                          d_c,
                                          matC,
                                          d_d,
                                          matD,
                                          &heuristicResult[0].algo,
                                          d_workspace,
                                          workspace_size,
                                          stream));

    // Clean up resources
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
    return;
}
