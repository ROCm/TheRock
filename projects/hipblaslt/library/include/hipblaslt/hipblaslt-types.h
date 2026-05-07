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

/*! \file
 * \brief hipblaslt-types.h defines data types used by hipblaslt
 */

#pragma once
#ifndef _HIPBLASLT_TYPES_H_
#define _HIPBLASLT_TYPES_H_


#if defined(__HIPCC__)
#include <hip/hip_fp8.h>
#endif

#include "hipblaslt_float8.h"
#include <float.h>

// Generic API

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Single precision floating point type */
typedef float hipblasLtFloat;

#ifdef ROCM_USE_FLOAT16
typedef _Float16 hipblasLtHalf;
#else
/*! \brief Structure definition for hipblasLtHalf */
typedef struct _hipblasLtHalf
{
    uint16_t data;
} hipblasLtHalf;
#endif

typedef hip_bfloat16 hipblasLtBfloat16;

typedef int8_t  hipblasLtInt8;
typedef int32_t hipblasLtInt32;

#ifdef __cplusplus
}
#endif

int const HIP_R_6F_E2M3_EXT = 31;
int const HIP_R_6F_E3M2_EXT = 32;
int const HIP_R_4F_E2M1_EXT = 33;

#endif /* _HIPBLASLT_TYPES_H_ */
