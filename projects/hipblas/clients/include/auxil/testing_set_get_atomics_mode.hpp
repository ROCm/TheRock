/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

inline void testname_set_get_atomics_mode(const Arguments& arg, std::string& name)
{
    ArgumentModel<>{}.test_name(arg, name);
}

void testing_set_get_atomics_mode(const Arguments& arg)
{
    bool FORTRAN                 = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasSetAtomicsModeFn = FORTRAN ? hipblasSetAtomicsModeFortran : hipblasSetAtomicsMode;
    auto hipblasGetAtomicsModeFn = FORTRAN ? hipblasGetAtomicsModeFortran : hipblasGetAtomicsMode;

    hipblasAtomicsMode_t mode;
    hipblasLocalHandle   handle(arg);

    // Not checking default as rocBLAS defaults to allowed
    // and cuBLAS defaults to not allowed.
    // CHECK_HIPBLAS_ERROR(hipblasGetAtomicsModeFn(handle, &mode));

    // EXPECT_EQ(HIPBLAS_ATOMICS_ALLOWED, mode);

    // Make sure set()/get() functions work
    CHECK_HIPBLAS_ERROR(hipblasSetAtomicsModeFn(handle, HIPBLAS_ATOMICS_NOT_ALLOWED));
    CHECK_HIPBLAS_ERROR(hipblasGetAtomicsModeFn(handle, &mode));

    EXPECT_EQ(HIPBLAS_ATOMICS_NOT_ALLOWED, mode);

    CHECK_HIPBLAS_ERROR(hipblasSetAtomicsModeFn(handle, HIPBLAS_ATOMICS_ALLOWED));
    CHECK_HIPBLAS_ERROR(hipblasGetAtomicsModeFn(handle, &mode));

    EXPECT_EQ(HIPBLAS_ATOMICS_ALLOWED, mode);
}
