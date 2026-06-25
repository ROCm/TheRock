/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc.
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
#include <algorithm>

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
    std::copy(src, src + m * k, static_cast<T*>(tmpTensor.template as<void>()));

    if(colMaj)
    {
        auto orgTensor = Tensor::create<T>({k, m});
        std::copy(src, src + m * k, static_cast<T*>(orgTensor.template as<void>()));
        tmpTensor = permute(orgTensor, {1, 0});
    }

    tmpTensor.reshape({m / MiM, MiM, k / (MiK * PackK), MiK / MiKv, MiKv * PackK});
    Tensor permuted = permute(tmpTensor, {0, 2, 3, 1, 4});
    std::copy(static_cast<const T*>(permuted.template as<void>()), static_cast<const T*>(permuted.template as<void>()) + m * k, dst);
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
                hipStream_t        stream);

int main()
{
    constexpr int64_t                                                 m{5280};
    constexpr int64_t                                                 n{2048};
    constexpr int64_t                                                 k{1024};
    Runner<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float, float> runner(
        m, n, k, 1, 1.f, 1.f, 32 * 128 * 128);

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
                   HIP_R_16F,
                   false,
                   runner.stream);
    });

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
        /** This is an example with swizzle-A
         *  a = (k, m). lda = k
         *  b = (k, n). ldb = k
         *  c = d = (m, n). ldc = ldd = m
         */
        simpleGemm(swizzleRunner.handle,
                   /*For swizzle-A, it forces to use TN*/
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

    Runner<hipblaslt_f8_fnuz, hipblaslt_f8_fnuz, hipblasLtHalf, float, float> swizzleRunner_F8(
        m, n, k, 1, 1.f, 1.f, 32 * 128 * 128);

    swizzleRunner_F8.run([&swizzleRunner_F8, &runner, m, n, k] {
        // convert inputs from reference runner to fp8
        std::vector<hipblasLtHalf>     cpuAF16(m * k, hipblasLtHalf(0.f));
        std::vector<hipblasLtHalf>     cpuBF16(k * n, hipblasLtHalf(0.f));
        std::vector<hipblaslt_f8_fnuz> cpuAF8(m * k, hipblaslt_f8_fnuz(0.f));
        std::vector<hipblaslt_f8_fnuz> cpuBF8(k * n, hipblaslt_f8_fnuz(0.f));

        CHECK_HIP_ERROR(hipMemcpy(cpuAF16.data(),
                  runner.d_a,
                  cpuAF16.size() * sizeof(hipblasLtHalf),
                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(cpuBF16.data(),
                  runner.d_b,
                  cpuBF16.size() * sizeof(hipblasLtHalf),
                  hipMemcpyDeviceToHost));

        for(size_t i = 0; i < cpuAF16.size(); ++i)
        {
            cpuAF8[i] = hipblaslt_f8_fnuz(float(cpuAF16[i]));
        }

        for(size_t i = 0; i < cpuBF16.size(); ++i)
        {
            cpuBF8[i] = hipblaslt_f8_fnuz(float(cpuBF16[i]));
        }

        // copy inputs from first runner for comparison and validation
        CHECK_HIP_ERROR(hipMemcpy(swizzleRunner_F8.d_a,
                  cpuAF8.data(),
                  m * k * sizeof(hipblaslt_f8_fnuz),
                  hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(swizzleRunner_F8.d_b,
                  cpuBF8.data(),
                  n * k * sizeof(hipblaslt_f8_fnuz),
                  hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(swizzleRunner_F8.d_c,
                  runner.d_c,
                  m * n * sizeof(hipblasLtHalf),
                  hipMemcpyDeviceToDevice));
        /** This is an example with swizzle-A
         *  a = (k, m). lda = k
         *  b = (k, n). ldb = k
         *  c = d = (m, n). ldc = ldd = m
         */
        simpleGemm(swizzleRunner_F8.handle,
                   /*For swizzle-A, it forces to use TN*/
                   HIPBLAS_OP_T,
                   HIPBLAS_OP_N,
                   swizzleRunner_F8.m,
                   swizzleRunner_F8.n,
                   swizzleRunner_F8.k,
                   swizzleRunner_F8.batch_count,
                   swizzleRunner_F8.alpha,
                   swizzleRunner_F8.beta,
                   swizzleRunner_F8.d_a,
                   swizzleRunner_F8.d_b,
                   swizzleRunner_F8.d_c,
                   swizzleRunner_F8.d_d,
                   swizzleRunner_F8.d_workspace,
                   swizzleRunner_F8.max_workspace_size,
                   HIP_R_8F_E4M3_FNUZ,
                   true,
                   swizzleRunner_F8.stream);
    });

    const hipblasLtHalf* regularCpuD     = static_cast<hipblasLtHalf*>(runner.d);
    const hipblasLtHalf* swizzledCpuD    = static_cast<hipblasLtHalf*>(swizzleRunner.d);
    const hipblasLtHalf* swizzledCpuD_F8 = static_cast<hipblasLtHalf*>(swizzleRunner_F8.d);

    for(size_t i = 0; i < m * n; ++i)
    {
        const auto diff = std::abs(float(regularCpuD[i] - float(swizzledCpuD[i])));
        if(diff > 1e-5)
        {
            std::cerr << "F16 Swizzle Validation Error at index: " << i << ", diff: " << diff
                      << '\n';
            break;
        }
    }

    for(size_t i = 0; i < m * n; ++i)
    {
        const auto diff = std::abs(float(regularCpuD[i] - float(swizzledCpuD_F8[i])));
        if(diff > 1e-5)
        {
            std::cerr << "F8 Swizzle Validation Error at index: " << i << ", diff: " << diff
                      << '\n';
            break;
        }
    }

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
                hipStream_t        stream)
{
    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, TiAB, k, n, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, m, n, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, HIP_R_16F, m, n, m));

    if(trans_a == HIPBLAS_OP_T)
    {
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, TiAB, k, m, k));

        if(swizzleA && TiAB == HIP_R_16F)
        {
            hipblasLtOrder_t orderA = HIPBLASLT_ORDER_COL16_4R8;
            CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
                matA, HIPBLASLT_MATRIX_LAYOUT_ORDER, &orderA, sizeof(orderA)));
            std::vector<hipblasLtHalf> src(m * k, 0);
            std::vector<hipblasLtHalf> dst(m * k, 0);
            CHECK_HIP_ERROR(hipMemcpy(src.data(), d_a, m * k * sizeof(hipblasLtHalf), hipMemcpyDeviceToHost));
            swizzleTensor(dst.data(), src.data(), m, k, true);
            CHECK_HIP_ERROR(hipMemcpy(d_a, dst.data(), m * k * sizeof(hipblasLtHalf), hipMemcpyHostToDevice));
        }
        else if(swizzleA && TiAB == HIP_R_8F_E4M3_FNUZ)
        {
            hipblasLtOrder_t orderA = HIPBLASLT_ORDER_COL16_4R16;
            CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
                matA, HIPBLASLT_MATRIX_LAYOUT_ORDER, &orderA, sizeof(orderA)));
            hipblaslt_f8_fnuz* src;
            hipblaslt_f8_fnuz* dst;
            CHECK_HIP_ERROR(
                hipMalloc(&src, m * k * sizeof(hipblaslt_f8_fnuz))); // Allocate memory on device
            CHECK_HIP_ERROR(
                hipMalloc(&dst, m * k * sizeof(hipblaslt_f8_fnuz))); // Allocate memory on device
            CHECK_HIP_ERROR(hipMemcpy(src, d_a, m * k * sizeof(hipblaslt_f8_fnuz), hipMemcpyDeviceToHost));
            swizzleTensor(dst, src, m, k, true);
            CHECK_HIP_ERROR(hipMemcpy(d_a, dst, m * k * sizeof(hipblaslt_f8_fnuz), hipMemcpyHostToDevice));
        }
    }
    else
    {
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, m, k, m));
    }

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

    const int                        request_solutions = 100;
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
    // In this sample, the workspace is already allocated with max_workspace_size
    // If not, allocate d_workspace here
    // CHECK_HIP_ERRORhipMalloc(&d_workspace, workspace_size));
    float         bestTimeMs = std::numeric_limits<float>::max();
    constexpr int numWarmupRuns{100};
    constexpr int numRuns{1000};

    for(int j = 0; j < returnedAlgoCount; ++j)
    {
        for(int i = 0; i < numWarmupRuns; ++i)
        {
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
                                                  &heuristicResult[j].algo,
                                                  d_workspace,
                                                  workspace_size,
                                                  stream));
        }

        hipEvent_t start, stop;
        CHECK_HIP_ERROR(hipEventCreate(&start));
        CHECK_HIP_ERROR(hipEventCreate(&stop));
        CHECK_HIP_ERROR(hipEventRecord(start, stream));

        for(int i = 0; i < numRuns; ++i)
        {
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
                                                  &heuristicResult[j].algo,
                                                  d_workspace,
                                                  workspace_size,
                                                  stream));
        }

        CHECK_HIP_ERROR(hipEventRecord(stop, stream));
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        CHECK_HIP_ERROR(hipDeviceSynchronize());
        float timeMs{};
        CHECK_HIP_ERROR(hipEventElapsedTime(&timeMs, start, stop));
        CHECK_HIP_ERROR(hipEventDestroy(start));
        CHECK_HIP_ERROR(hipEventDestroy(stop));
        bestTimeMs = std::min(timeMs, bestTimeMs);
    }

    std::cout << "Best solution time: " << bestTimeMs / numRuns * 1000
              << " us (swizzleA == " << int(swizzleA) << ")\n";
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
    return;
}
