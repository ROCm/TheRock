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
 *
 * ************************************************************************/

#pragma once

// DAPI refers to dual API (original and ILP64 version ending in _64)

#define UNWRAP_ARGS(...) __VA_ARGS__

#define DAPI_DISPATCH(name_, args_)    \
    if(arg.api & c_API_64)             \
    {                                  \
        name_##_64(UNWRAP_ARGS args_); \
    }                                  \
    else                               \
    {                                  \
        name_(UNWRAP_ARGS args_);      \
    }

#define DAPI_EXPECT(val_, name_, args_)                               \
    if(arg.api & c_API_64)                                            \
    {                                                                 \
        EXPECT_HIPBLAS_STATUS((name_##_64(UNWRAP_ARGS args_)), val_); \
    }                                                                 \
    else                                                              \
    {                                                                 \
        EXPECT_HIPBLAS_STATUS((name_(UNWRAP_ARGS args_)), val_);      \
    }

#define DAPI_CHECK(name_, args_)                              \
    if(arg.api & c_API_64)                                    \
    {                                                         \
        CHECK_HIPBLAS_ERROR((name_##_64(UNWRAP_ARGS args_))); \
    }                                                         \
    else                                                      \
    {                                                         \
        CHECK_HIPBLAS_ERROR((name_(UNWRAP_ARGS args_)));      \
    }
