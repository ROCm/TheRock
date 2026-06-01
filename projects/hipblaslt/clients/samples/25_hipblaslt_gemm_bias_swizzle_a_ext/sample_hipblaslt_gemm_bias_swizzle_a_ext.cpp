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
#include <hipblaslt/hipblaslt-ext.hpp>
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
    case HIP_R_8F_E5M2_FNUZ:
    case HIP_R_8F_E4M3_FNUZ:
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
    memcpy(tmpTensor.template as<void>(), src, m * k * sizeof(T));

    if(colMaj)
    {
        auto orgTensor = Tensor::create<T>({k, m});
        memcpy(orgTensor.template as<void>(), src, m * k * sizeof(T));
        tmpTensor = permute(orgTensor, {1, 0});
    }
    auto                          MultipleM = MiM;
    auto                          MultipleK = MiK * PackK;
    const auto                    paddedM   = (m / MultipleM + !!(m % MultipleM)) * MultipleM;
    const auto                    paddedK   = (k / MultipleK + !!(k % MultipleK)) * MultipleK;
    ::Tensor::Manipulation::Shape paddedShape{paddedM, paddedK};
    auto paddedTensor = ::Tensor::Manipulation::pad(tmpTensor, paddedShape, T(0));
    paddedTensor.reshape({paddedM / MiM, MiM, paddedK / (MiK * PackK), MiK / MiKv, MiKv * PackK});
    Tensor permuted = permute(paddedTensor, {0, 2, 3, 1, 4});
    memcpy(dst, permuted.template as<void>(), paddedM * paddedK * sizeof(T));
}

void swizzleGemmEpilogueBiasVecExt(hipblasLtHandle_t  handle,
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
                                   bool               swizzleA,
                                   hipStream_t        stream);

int main()
{
    constexpr int64_t m{5280};
    constexpr int64_t n{2048};
    constexpr int64_t k{1024};

    Runner<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float, float> swizzleRunner(
        m, n, k, 1, 1.f, 1.f, 32 * 128 * 128);

    swizzleRunner.run([&swizzleRunner, m, n, k] {
        swizzleGemmEpilogueBiasVecExt(swizzleRunner.handle,
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
                                      true,
                                      swizzleRunner.stream);
    });

    return 0;
}

void swizzleGemmEpilogueBiasVecExt(hipblasLtHandle_t  handle,
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
                                   bool               swizzleA,
                                   hipStream_t        stream)
{
    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, k, n, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, m, n, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, HIP_R_16F, m, n, m));

    if(trans_a == HIPBLAS_OP_T)
    {
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, k, m, k));

        if(swizzleA)
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

    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_BIAS;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    // Allocate and set the bias tensor
    std::vector<hipblasLtHalf> h_bias(m, 1.0f); // Example bias values, adjust as needed
    void*                      d_bias;
    CHECK_HIP_ERROR(hipMalloc(&d_bias, m * sizeof(hipblasLtHalf))); // Allocate memory for bias
    CHECK_HIP_ERROR(hipMemcpy(d_bias,
                              h_bias.data(),
                              m * sizeof(hipblasLtHalf),
                              hipMemcpyHostToDevice)); // Copy bias to device

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias, sizeof(void*)));

    hipblaslt_ext::Gemm gemm(
        handle, matmul, &alpha, d_a, matA, d_b, matB, &beta, d_c, matC, d_d, matD);

    hipblaslt_ext::GemmPreference gemmPref;
    gemmPref.setMaxWorkspaceBytes(max_workspace_size);

    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResults;
    const int                                     requestedSolutions = 1;
    CHECK_HIPBLASLT_ERROR(gemm.algoGetHeuristic(requestedSolutions, gemmPref, heuristicResults));

    if(heuristicResults.size() == 0)
    {
        std::cerr << "No valid solution found!" << std::endl;
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
        CHECK_HIP_ERROR(hipFree(d_bias));
        return;
    }

    gemm.setMaxWorkspaceBytes(max_workspace_size);
    CHECK_HIPBLASLT_ERROR(gemm.initialize(heuristicResults[0].algo, d_workspace));
    CHECK_HIPBLASLT_ERROR(gemm.run(stream));

    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIP_ERROR(hipFree(d_bias));
    return;
}
