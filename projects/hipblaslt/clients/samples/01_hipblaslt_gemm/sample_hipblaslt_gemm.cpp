/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
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

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <iostream>

#include "helper.h"

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
                hipStream_t        stream);

int main()
{
    /** This is a NN example with
     *  a = (m, k). lda = m
     *  b = (k, n). ldb = k
     *  c = d = (m, n). ldc = ldd = m
     */
    Runner<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float, float> runner(
        1024, 512, 1024, 1, 1.f, 1.f, 32 * 1024 * 1024);

    runner.run([&runner] {
        simpleGemm(runner.handle,
                   HIPBLAS_OP_N,
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
                   runner.stream);
    });

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
                hipStream_t        stream)
{
    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, m, k, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, k, n, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, m, n, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, HIP_R_16F, m, n, m));

    if(batch_count > 1)
    {
        int64_t stride_a = m * k;
        int64_t stride_b = k * n;
        int64_t stride_c = m * n;
        int64_t stride_d = m * n;
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matA, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_a, sizeof(stride_a)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matB, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_b, sizeof(stride_b)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matC, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matC, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_c, sizeof(stride_c)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matD, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matD, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_d, sizeof(stride_d)));
    }

    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));

    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

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
        return;
    }

    uint64_t workspace_size = 0;
    for(int i = 0; i < returnedAlgoCount; i++)
        workspace_size = max(workspace_size, heuristicResult[i].workspaceSize);
    // In this sample, the workspace is already allocated with max_workspace_size
    // If not, allocate d_workspace here
    // CHECK_HIP_ERRORhipMalloc(&d_workspace, workspace_size));

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
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
    return;
}
