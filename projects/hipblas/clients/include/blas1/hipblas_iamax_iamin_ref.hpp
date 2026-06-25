/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "type_utils.h"

namespace hipblas_iamax_iamin_ref
{
    template <typename T>
    inline real_t<T> iamax_iamin_abs(const T& x)
    {
        return hipblas_abs(x);
    }

    template <>
    inline float iamax_iamin_abs(const std::complex<float>& c)
    {
        return hipblas_abs(c.real()) + hipblas_abs(c.imag());
    }

    template <>
    inline double iamax_iamin_abs(const std::complex<double>& c)
    {
        return hipblas_abs(c.real()) + hipblas_abs(c.imag());
    }

    template <typename T>
    inline void local_iamin_ensure_min_index(int64_t N, const T* X, int64_t incx, int64_t* result)
    {
        int64_t minpos = -1;
        if(N > 0 && incx > 0)
        {
            auto min = iamax_iamin_abs(X[0]);
            minpos   = 0;
            for(size_t i = 1; i < N; ++i)
            {
                auto a = iamax_iamin_abs(X[i * incx]);
                if(a < min)
                {
                    min    = a;
                    minpos = i;
                }
            }
        }
        *result = minpos + 1; // change to Fortran 1 based indexing as in BLAS standard
    }

    template <typename T>
    inline void local_iamax_ensure_min_index(int64_t N, const T* X, int64_t incx, int64_t* result)
    {
        int64_t maxpos = -1;
        if(N > 0 && incx > 0)
        {
            auto max = iamax_iamin_abs(X[0]);
            maxpos   = 0;
            for(size_t i = 1; i < N; ++i)
            {
                auto a = iamax_iamin_abs(X[i * incx]);
                if(a > max)
                {
                    max    = a;
                    maxpos = i;
                }
            }
        }
        *result = maxpos + 1; // change to Fortran 1 based indexing as in BLAS standard
    }

    template <typename T>
    void iamin(int64_t N, const T* X, int64_t incx, int64_t* result)
    {
        local_iamin_ensure_min_index(N, X, incx, result);
    }

    template <typename T>
    void iamax(int64_t N, const T* X, int64_t incx, int64_t* result)
    {
        local_iamax_ensure_min_index(N, X, incx, result);
    }

}
