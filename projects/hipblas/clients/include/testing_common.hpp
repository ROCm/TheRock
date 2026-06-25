/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once
#ifndef _TESTING_COMMON_HPP_
#define _TESTING_COMMON_HPP_

// do not add special case includes here, keep those in the testing_ file
#include "argument_model.hpp"
#include "bytes.hpp"
#include "cblas_interface.h"
#include "flops.hpp"
#include "hipblas.hpp"
#ifndef WIN32
#include "hipblas_fortran.hpp"
#else
#include "hipblas_no_fortran.hpp"
#endif
#include "device_batch_matrix.hpp"
#include "device_batch_vector.hpp"
#include "device_matrix.hpp"
#include "device_strided_batch_matrix.hpp"
#include "device_strided_batch_vector.hpp"
#include "device_vector.hpp"
#include "hipblas_init.hpp"
#include "hipblas_matrix.hpp"
#include "hipblas_test.hpp"
#include "hipblas_vector.hpp"
#include "host_batch_matrix.hpp"
#include "host_batch_vector.hpp"
#include "host_matrix.hpp"
#include "host_strided_batch_matrix.hpp"
#include "host_strided_batch_vector.hpp"
#include "host_vector.hpp"
#include "near.h"
#include "norm.h"
#include "testing_macros.hpp"
#include "type_utils.h"
#include "unit.h"
#include "utility.h"

#endif
