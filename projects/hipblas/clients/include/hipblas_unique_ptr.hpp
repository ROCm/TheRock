/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef GUARD_HIPBLAS_MANAGE_PTR_HPP
#define GUARD_HIPBLAS_MANAGE_PTR_HPP

#include <memory>

#include "hipblas.h"

#define PRINT_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                \
    {                                                             \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                    \
        {                                                         \
            fprintf(stderr,                                       \
                    "hip error code: %d at %s:%d\n",              \
                    TMP_STATUS_FOR_CHECK,                         \
                    __FILE__,                                     \
                    __LINE__);                                    \
        }                                                         \
    }

namespace hipblas
{
    // device_malloc wraps hipMalloc and provides same API as malloc
    static void* device_malloc(size_t byte_size)
    {
        void* pointer;
        PRINT_IF_HIP_ERROR(hipMalloc(&pointer, byte_size));
        return pointer;
    }

    // device_free wraps hipFree and provides same API as free
    static void device_free(void* ptr)
    {
        PRINT_IF_HIP_ERROR(hipFree(ptr));
    }

} // namespace hipblas

using hipblas_unique_ptr = std::unique_ptr<void, void (*)(void*)>;

#endif
