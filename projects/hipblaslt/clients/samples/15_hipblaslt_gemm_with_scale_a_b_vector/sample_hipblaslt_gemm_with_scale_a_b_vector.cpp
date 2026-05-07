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

#include "helper.h"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/library_types.h>
#include <hipblaslt/hipblaslt.h>
#include <iostream>
#include <vector>

void simpleGemmScaleAB(hipblasLtHandle_t  handle,
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
                       hipStream_t        stream,
                       const float*       h_scale_a_vec,
                       const float*       h_scale_b_vec);

int main()
{
    Runner<hipblaslt_f8_fnuz, hipblaslt_f8_fnuz, hip_bfloat16, float, float> runner(
        128, 128, 128, 1, 1.f, 0.f, 32 * 1024 * 1024);

    std::vector<float> scale_a_vec
        = std::vector<float>(128, 0.5f); // scale A vector = vector len M(128)
    std::vector<float> scale_b_vec
        = std::vector<float>(128, 2.0f); // scale B vector = vector len N(128)
    std::cout << "Running with Scale A Vector with all values = " << scale_a_vec[0]
              << " and Scale B Vector with all values = " << scale_b_vec[0] << std::endl;
    runner.run([&runner, scale_a_vec, scale_b_vec] {
        simpleGemmScaleAB(runner.handle,
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
                          runner.stream,
                          scale_a_vec.data(),
                          scale_b_vec.data());
    });

    return 0;
}

void simpleGemmScaleAB(hipblasLtHandle_t  handle,
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
                       hipStream_t        stream,
                       const float*       h_scale_a_vec,
                       const float*       h_scale_b_vec)
{
    float* d_scale_a_vec;
    float* d_scale_b_vec;
    CHECK_HIP_ERROR(hipMalloc(&d_scale_a_vec, m * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_scale_b_vec, n * sizeof(float)));
    CHECK_HIP_ERROR(hipMemcpyAsync(
        d_scale_a_vec, h_scale_a_vec, m * sizeof(float), hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(hipMemcpyAsync(
        d_scale_b_vec, h_scale_b_vec, n * sizeof(float), hipMemcpyHostToDevice, stream));

    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, HIP_R_8F_E4M3_FNUZ, k, m, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, HIP_R_8F_E4M3_FNUZ, k, n, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, HIP_R_16BF, m, n, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, HIP_R_16BF, m, n, m));

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

    uint64_t workspace_size = max_workspace_size;
    for(int i = 0; i < returnedAlgoCount; i++)
        workspace_size = std::max(workspace_size, heuristicResult[i].workspaceSize);

    // Perform matrix multiplication
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
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));

    std::cout << "Matrix multiplication completed successfully." << std::endl;
}
