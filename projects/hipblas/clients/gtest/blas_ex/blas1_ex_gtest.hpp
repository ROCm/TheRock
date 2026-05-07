/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */
#include "hipblas_data.hpp"
#include "hipblas_datatype2string.hpp"
#include "type_dispatch.hpp"
#include "utility.h"

enum class blas1_ex
{
    axpy_ex,
    axpy_batched_ex,
    axpy_strided_batched_ex,
    dot_ex,
    dotc_ex,
    dot_batched_ex,
    dotc_batched_ex,
    dot_strided_batched_ex,
    dotc_strided_batched_ex,
    nrm2_ex,
    nrm2_batched_ex,
    nrm2_strided_batched_ex,
    rot_ex,
    rot_batched_ex,
    rot_strided_batched_ex,
    scal_ex,
    scal_batched_ex,
    scal_strided_batched_ex,
};
