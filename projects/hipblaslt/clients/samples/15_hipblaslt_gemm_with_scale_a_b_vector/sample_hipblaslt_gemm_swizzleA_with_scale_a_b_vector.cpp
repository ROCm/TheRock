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
void swizzleTensor(T* dst, const T* src, size_t m, size_t k, bool colMaj)
{
    using Tensor = Tensor::Manipulation::Tensor;
    size_t MiM   = 16;
    size_t MiK = 0, MiKv = 0, PackK = 0;
    calculateKforSwizzling(hipblaslt_type2datatype<T>(), MiK, MiKv, PackK);
    auto tmpTensor = Tensor::create<T>({m, k});
    std::copy(src, src + (m * k), tmpTensor.template as<T>());

    if(colMaj)
    {
        auto orgTensor = Tensor::create<T>({k, m});
        std::copy(src, src + (m * k), orgTensor.template as<T>());
        tmpTensor = permute(orgTensor, {1, 0});
    }

    tmpTensor.reshape({m / MiM, MiM, k / (MiK * PackK), MiK / MiKv, MiKv * PackK});
    Tensor permuted = permute(tmpTensor, {0, 2, 3, 1, 4});
    std::copy(permuted.template as<T>(), permuted.template as<T>() + (m * k), dst);
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
                bool               swizzleA,
                hipStream_t        stream,
                const float*       h_scale_a_vec,
                const float*       h_scale_b_vec);

int main()
{
    constexpr int64_t m{5280};
    constexpr int64_t n{2048};
    constexpr int64_t k{1024};

    // Non-swizzle runner: TN, ScaleABVec, batch count = 1, alpha, beta = 1.0f
    Runner<hipblaslt_f8_fnuz, hipblaslt_f8_fnuz, hip_bfloat16, float, float> runner(
        m, n, k, 1, 1.f, 1.f, 32 * 128 * 128);

    std::vector<float> scale_a_vec = std::vector<float>(m, 0.5f); // scale A vector = vector len M
    std::vector<float> scale_b_vec = std::vector<float>(n, 2.0f); // scale B vector = vector len N
    std::cout << "Running with Scale A Vector with all values = " << scale_a_vec[0]
              << " and Scale B Vector with all values = " << scale_b_vec[0] << std::endl;

    runner.run([&runner, scale_a_vec, scale_b_vec] {
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
                   HIP_R_8F_E4M3_FNUZ,
                   false,
                   runner.stream,
                   scale_a_vec.data(),
                   scale_b_vec.data());
    });

    // swizzleA runner: TN, ScaleABVec, batch count = 1, alpha, beta = 1.0f
    Runner<hipblaslt_f8_fnuz, hipblaslt_f8_fnuz, hip_bfloat16, float, float> swizzleRunner(
        m, n, k, 1, 1.f, 1.f, 32 * 128 * 128);

    swizzleRunner.run([&swizzleRunner, &runner, m, n, k, scale_a_vec, scale_b_vec] {
        // copy inputs from first runner for comparison and validation
        CHECK_HIP_ERROR(hipMemcpy(swizzleRunner.d_a,
                  runner.d_a,
                  m * k * sizeof(hipblaslt_f8_fnuz),
                  hipMemcpyDeviceToDevice));
        CHECK_HIP_ERROR(hipMemcpy(swizzleRunner.d_b,
                  runner.d_b,
                  n * k * sizeof(hipblaslt_f8_fnuz),
                  hipMemcpyDeviceToDevice));
        CHECK_HIP_ERROR(hipMemcpy(
            swizzleRunner.d_c, runner.d_c, m * n * sizeof(hip_bfloat16), hipMemcpyDeviceToDevice));

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
                   HIP_R_8F_E4M3_FNUZ,
                   true,
                   swizzleRunner.stream,
                   scale_a_vec.data(),
                   scale_b_vec.data());
    });

    // Compare results from non-swizzling with swizzling
    const hip_bfloat16* regularCpuD  = static_cast<hip_bfloat16*>(runner.d);
    const hip_bfloat16* swizzledCpuD = static_cast<hip_bfloat16*>(swizzleRunner.d);

    for(size_t i = 0; i < m * n; ++i)
    {
        const auto diff = std::abs(float(regularCpuD[i] - float(swizzledCpuD[i])));
        if(diff > 1e-5)
        {
            std::cerr << "F8 Swizzle Validation Error at index: " << i << ", diff: " << diff
                      << '\n';
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
                bool               swizzleA,
                hipStream_t        stream,
                const float*       h_scale_a_vec,
                const float*       h_scale_b_vec)
{
    // Scale A, B Vector
    float* d_scale_a_vec;
    float* d_scale_b_vec;
    CHECK_HIP_ERROR(hipMalloc(&d_scale_a_vec, m * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_scale_b_vec, n * sizeof(float)));
    CHECK_HIP_ERROR(hipMemcpyAsync(
        d_scale_a_vec, h_scale_a_vec, m * sizeof(float), hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(hipMemcpyAsync(
        d_scale_b_vec, h_scale_b_vec, n * sizeof(float), hipMemcpyHostToDevice, stream));

    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, TiAB, k, m, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, TiAB, k, n, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, HIP_R_16BF, m, n, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, HIP_R_16BF, m, n, m));

    // swizzle case and input = FP8
    if(swizzleA && TiAB == HIP_R_8F_E4M3_FNUZ)
    {
        hipblasLtOrder_t orderA = HIPBLASLT_ORDER_COL16_4R16;
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matA, HIPBLASLT_MATRIX_LAYOUT_ORDER, &orderA, sizeof(orderA)));
        std::vector<hipblaslt_f8_fnuz> src(m * k, 0);
        std::vector<hipblaslt_f8_fnuz> dst(m * k, 0);

        // pre-shuffle input data in host memory
        CHECK_HIP_ERROR(hipMemcpy(src.data(), d_a, m * k * sizeof(hipblaslt_f8_fnuz), hipMemcpyDeviceToHost));
        swizzleTensor(dst.data(), src.data(), m, k, false);
        CHECK_HIP_ERROR(hipMemcpy(d_a, dst.data(), m * k * sizeof(hipblaslt_f8_fnuz), hipMemcpyHostToDevice));
    }

    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));

    // Set ScaleA, B mode (Vector)
    hipblasLtMatmulMatrixScale_t mode = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(uint32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(uint32_t)));

    // Set A and B matrix scale factors
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scale_a_vec, sizeof(float*)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scale_b_vec, sizeof(float*)));

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
        CHECK_HIP_ERROR(hipFree(d_scale_a_vec));
        CHECK_HIP_ERROR(hipFree(d_scale_b_vec));
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
    CHECK_HIP_ERROR(hipFree(d_scale_a_vec));
    CHECK_HIP_ERROR(hipFree(d_scale_b_vec));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
    return;
}
