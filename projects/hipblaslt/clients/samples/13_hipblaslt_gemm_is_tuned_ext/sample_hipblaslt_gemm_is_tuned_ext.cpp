/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024 Advanced Micro Devices, Inc.
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
#include <iostream>

void printResult(int tuned, uint64_t m, uint64_t n, uint64_t k) {
    if (tuned == 1) {
        std::cout << "[" << m << ", " << n << ", " << k << "] is tuned\n";
    } else {
        std::cout << "[" << m << ", " << n << ", " << k << "] is un-tuned\n";
    }
}

int main(int argc, char **argv)
{
    hipblasLtHandle_t handle{};
    hipblasLtCreate(&handle);
    hipblasLtMatmulDesc_t matmulDesc{};
    hipblasLtMatrixLayout_t matA{};
    hipblasLtMatrixLayout_t matB{};
    hipblasLtMatrixLayout_t matC{};
    hipblasLtMatrixLayout_t matD{};
    hipblasLtMatmulDescCreate(&matmulDesc, hipblasComputeType_t::HIPBLAS_COMPUTE_32F, HIP_R_32F);
    hipblasOperation_t opA = HIPBLAS_OP_T;
    hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
    hipblasLtPointerMode_t pMode = HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST;
    hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_POINTER_MODE, &pMode, sizeof(pMode));
    const uint64_t m = argc > 3 ? std::atoll(argv[1]): 128;
    const uint64_t n = argc > 3 ? std::atoll(argv[2]): 128;
    const uint64_t k = argc > 3 ? std::atoll(argv[3]): 128;
    hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, k, m, k);
    hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, k, n, k);
    hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, m, n, m);
    hipblasLtMatrixLayoutCreate(&matD, HIP_R_16F, m, n, m);
    auto tuned = hipblaslt_ext::matmulIsTuned(handle, matmulDesc, matA, matB, matC, matD);
    printResult(tuned, m, n, k);
    hipblasLtMatmulDescDestroy(matmulDesc);
    hipblasLtMatrixLayoutDestroy(matA);
    hipblasLtMatrixLayoutDestroy(matB);
    hipblasLtMatrixLayoutDestroy(matC);
    hipblasLtMatrixLayoutDestroy(matD);
    hipblasLtDestroy(handle);
    return 0;
}
