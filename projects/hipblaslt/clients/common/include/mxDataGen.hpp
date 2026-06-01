/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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

#pragma once

#include <hipblaslt/hipblaslt.h>
#include <stdint.h>

#ifdef HIPBLASLT_USE_ROCROLLER
std::vector<float> generateMXInput(hipDataType                dataType,
                                   void*                      data,
                                   void*                      scale,
                                   uint64_t                   row,
                                   uint64_t                   col,
                                   uint64_t                   stride,
                                   bool                       isTranspose,
                                   const std::vector<size_t>& preSwizzleTile,
                                   int const                  scaleBlockRowSize,
                                   int const                  scaleBlockColSize,
                                   bool                       isMatrixA,
                                   std::string_view const     initMethod = "Bounded",
                                   float                      min_val    = -1.0f,
                                   float                      max_val    = 1.0f);
#endif
