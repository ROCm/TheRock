/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

inline void testname_set_get_pointer_mode(const Arguments& arg, std::string& name)
{
    ArgumentModel<>{}.test_name(arg, name);
}

void testing_set_get_pointer_mode(const Arguments& arg)
{
    hipblasPointerMode_t mode = HIPBLAS_POINTER_MODE_DEVICE;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    hipblasStatus_t status = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);
    EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

    status = hipblasGetPointerMode(handle, &mode);
    EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

    EXPECT_EQ(HIPBLAS_POINTER_MODE_DEVICE, mode);

    status = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
    EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

    status = hipblasGetPointerMode(handle, &mode);
    EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

    EXPECT_EQ(HIPBLAS_POINTER_MODE_HOST, mode);

    hipblasDestroy(handle);
}
