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
#include <hipblaslt/hipblaslt-ext-op.h>
#include <hipblaslt/hipblaslt.h>
#include <iostream>

#include "helper.h"

void simpleGemmAmaxWithScale(hipblasLtHandle_t  handle,
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
    Runner<hipblaslt_f8_fnuz, hipblaslt_f8_fnuz, hipblaslt_f8_fnuz, float, float> runner(
        1024, 512, 1024, 1, 1.f, 0.f, 32 * 1024 * 1024);

    runner.run([&runner] {
        simpleGemmAmaxWithScale(runner.handle,
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

void simpleGemmAmaxWithScale(hipblasLtHandle_t  handle,
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
    // allocate data for amax
    void *out_tmp, *in_scale, *out_amax; // host
    void *d_out_tmp, *d_in_scale, *d_out_amax; // device

    CHECK_HIP_ERROR(hipMalloc(&d_in_scale, 1 * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_out_amax, 1 * sizeof(float)));

    CHECK_HIP_ERROR(hipHostMalloc(&in_scale, 1 * sizeof(float)));
    CHECK_HIP_ERROR(hipHostMalloc(&out_amax, 1 * sizeof(float)));

    // copy amax data to device
    *(float*)in_scale = (float)1.0;
    CHECK_HIP_ERROR(
        hipMemcpyAsync(d_in_scale, in_scale, 1 * sizeof(float), hipMemcpyHostToDevice, stream));

    // set matrix layout for gemm
    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, HIP_R_8F_E4M3_FNUZ, m, k, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, HIP_R_8F_E4M3_FNUZ, k, n, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, HIP_R_8F_E4M3_FNUZ, m, n, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, HIP_R_8F_E4M3_FNUZ, m, n, m));

    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));

    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER, &d_out_amax, sizeof(void*)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_in_scale, sizeof(void*)));

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

    uint64_t workspace_size = max_workspace_size;
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

    // deallocate memory space of amax
    CHECK_HIP_ERROR(hipFree(d_in_scale));
    CHECK_HIP_ERROR(hipFree(d_out_amax));

    CHECK_HIP_ERROR(hipFree(in_scale));
    CHECK_HIP_ERROR(hipFree(out_amax));

    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));

    return;
}
