/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */
#include "testing_common.hpp"

/* ============================================================================================ */

inline void testname_set_get_math_mode(const Arguments& arg, std::string& name)
{
    ArgumentModel<>{}.test_name(arg, name);
}

void testing_set_get_math_mode(const Arguments& arg)
{
    hipblasMath_t mode = HIPBLAS_DEFAULT_MATH;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    hipblasStatus_t status = hipblasSetMathMode(handle, HIPBLAS_DEFAULT_MATH);
    EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
    status = hipblasGetMathMode(handle, &mode);
    EXPECT_EQ(mode, HIPBLAS_DEFAULT_MATH);

#ifdef __HIP_PLATFORM_NVCC__
    // Both cuBLAS and hipBLAS have these math modes, but there isn't really much overlap.
    status = hipblasSetMathMode(handle, HIPBLAS_XF32_XDL_MATH);
    EXPECT_EQ(status, HIPBLAS_STATUS_NOT_SUPPORTED);

    status = hipblasSetMathMode(handle, HIPBLAS_PEDANTIC_MATH);
    EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
    status = hipblasGetMathMode(handle, &mode);
    EXPECT_EQ(mode, HIPBLAS_PEDANTIC_MATH);

    status = hipblasSetMathMode(handle, HIPBLAS_TF32_TENSOR_OP_MATH);
    EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
    status = hipblasGetMathMode(handle, &mode);
    EXPECT_EQ(mode, HIPBLAS_TF32_TENSOR_OP_MATH);

    status = hipblasSetMathMode(handle, HIPBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
    EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
    status = hipblasGetMathMode(handle, &mode);
    EXPECT_EQ(mode, HIPBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);

    status = hipblasSetMathMode(handle, HIPBLAS_TENSOR_OP_MATH);
    EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
    status = hipblasGetMathMode(handle, &mode);
    EXPECT_EQ(mode, HIPBLAS_TENSOR_OP_MATH);
#else
    status = hipblasSetMathMode(handle, HIPBLAS_XF32_XDL_MATH);
    EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
    status = hipblasGetMathMode(handle, &mode);

    // XF32 only for gfx94x
    if(arg.bad_arg_all)
        EXPECT_EQ(mode, HIPBLAS_XF32_XDL_MATH);

    status = hipblasSetMathMode(handle, HIPBLAS_PEDANTIC_MATH);
    EXPECT_EQ(status, HIPBLAS_STATUS_NOT_SUPPORTED);

    status = hipblasSetMathMode(handle, HIPBLAS_TF32_TENSOR_OP_MATH);
    EXPECT_EQ(status, HIPBLAS_STATUS_NOT_SUPPORTED);

    status = hipblasSetMathMode(handle, HIPBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
    EXPECT_EQ(status, HIPBLAS_STATUS_NOT_SUPPORTED);

    status = hipblasSetMathMode(handle, HIPBLAS_TENSOR_OP_MATH);
    EXPECT_EQ(status, HIPBLAS_STATUS_NOT_SUPPORTED);
#endif

    hipblasDestroy(handle);
}
