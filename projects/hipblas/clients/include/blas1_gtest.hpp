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
#pragma once

enum class blas1
{
    nrm2,
    nrm2_batched,
    nrm2_strided_batched,
    asum,
    asum_batched,
    asum_strided_batched,
    iamax,
    iamax_batched,
    iamax_strided_batched,
    iamin,
    iamin_batched,
    iamin_strided_batched,
    axpy,
    axpy_batched,
    axpy_strided_batched,
    copy,
    copy_batched,
    copy_strided_batched,
    dot,
    dotc,
    dot_batched,
    dotc_batched,
    dot_strided_batched,
    dotc_strided_batched,
    scal,
    scal_batched,
    scal_strided_batched,
    swap,
    swap_batched,
    swap_strided_batched,
    rot,
    rot_batched,
    rot_strided_batched,
    rotg,
    rotg_batched,
    rotg_strided_batched,
    rotm,
    rotm_batched,
    rotm_strided_batched,
    rotmg,
    rotmg_batched,
    rotmg_strided_batched,
};
