/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
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
#include <hipblaslt_arguments.hpp>
#include <iostream>

#include "helper.h"

void simpleLayerNorm(hipDataType type,
                     void*       d_out,
                     void*       d_mean,
                     void*       d_invvar,
                     void*       d_in,
                     int64_t     m,
                     int64_t     n,
                     float       eps,
                     void*       d_gamma,
                     void*       d_beta,
                     hipStream_t stream);

int main()
{
    LayerNormRunner<float> runnerF32(135, 345);

    runnerF32.run([&runnerF32] {
        simpleLayerNorm(HIP_R_32F,
                        runnerF32.d_out,
                        runnerF32.d_mean,
                        runnerF32.d_invvar,
                        runnerF32.d_in,
                        runnerF32.m,
                        runnerF32.n,
                        1e-5,
                        runnerF32.d_gamma,
                        runnerF32.d_beta,
                        runnerF32.stream);
    });

    return 0;
}

void simpleLayerNorm(hipDataType type,
                     void*       d_out,
                     void*       d_mean,
                     void*       d_invvar,
                     void*       d_in,
                     int64_t     m,
                     int64_t     n,
                     float       eps,
                     void*       d_gamma,
                     void*       d_beta,
                     hipStream_t stream)
{
    CHECK_HIPBLASLT_ERROR(hipblasltExtLayerNorm(
        type, d_out, d_mean, d_invvar, d_in, m, n, eps, d_gamma, d_beta, stream));
}
