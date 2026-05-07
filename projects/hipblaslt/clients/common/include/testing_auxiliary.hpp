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

#pragma once

#include "flops.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_init.hpp"
#include "hipblaslt_math.hpp"
#include "hipblaslt_random.hpp"
#include "hipblaslt_test.hpp"
#include "hipblaslt_vector.hpp"
#ifdef CODE_COVERAGE
#include "hipblaslt_internal.hpp"
#include "rocblaslt/rocblaslt_mat_utils.hpp"
#include "rocblaslt/rocroller_host.hpp"
#include "rocblaslt/status.h"
#include "rocblaslt/tensile_host.hpp"
#include "rocblaslt/utility.hpp"
#endif
#include "unit.hpp"
#include "utility.hpp"
#include <hipblaslt/hipblaslt-ext-op.h>
#include <hipblaslt/hipblaslt-ext.hpp> // Add check for hipblaslt-ext
#include <hipblaslt/hipblaslt.h>

void testing_aux_handle_init_bad_arg(const Arguments& arg)
{
    EXPECT_HIPBLAS_STATUS(hipblasLtCreate(nullptr), HIPBLAS_STATUS_INVALID_VALUE);
}

void testing_aux_handle_destroy_bad_arg(const Arguments& arg)
{
    EXPECT_HIPBLAS_STATUS(hipblasLtDestroy(nullptr), HIPBLAS_STATUS_INVALID_VALUE);
}

void testing_aux_handle(const Arguments& arg)
{
    hipblasLtHandle_t handle;
    EXPECT_HIPBLAS_STATUS(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    EXPECT_HIPBLAS_STATUS(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

void testing_aux_mat_init_bad_arg(const Arguments& arg)
{
    const int64_t row = 128;
    const int64_t col = 128;
    const int64_t ld  = 128;

    hipblasLtMatrixLayout_t m_descr;

    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutCreate(nullptr, arg.a_type, row, col, ld),
                          HIPBLAS_STATUS_INVALID_VALUE);
}

void testing_aux_mat_init(const Arguments& arg)
{
    const int64_t row = 128;
    const int64_t col = 128;
    const int64_t ld  = 128;

    hipblaslt_local_handle        handle{arg};
    hipblaslt_local_matrix_layout mat(row, col, ld, arg.a_type);
    EXPECT_HIPBLAS_STATUS(mat.status(), HIPBLAS_STATUS_SUCCESS);
}

void testing_aux_mat_destroy_bad_arg(const Arguments& arg)
{
    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutDestroy(nullptr), HIPBLAS_STATUS_INVALID_VALUE);
}

void testing_aux_mat_set_attr_bad_arg(const Arguments& arg)
{
    const int64_t row = 128;
    const int64_t col = 128;
    const int64_t ld  = 128;

    int32_t  data32;
    int64_t  data64;
    uint32_t udata32;
    uint64_t udata64;

    hipblaslt_local_matrix_layout mat(row, col, ld, arg.a_type);
    EXPECT_HIPBLAS_STATUS(mat.status(), HIPBLAS_STATUS_SUCCESS);
    // Test with null matrix layout
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutSetAttribute(
            nullptr, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &data32, sizeof(int32_t)),
        HIPBLAS_STATUS_NOT_INITIALIZED);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                              mat, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, nullptr, sizeof(int32_t)),
                          HIPBLAS_STATUS_INVALID_VALUE);

    // Test with zero size
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutSetAttribute(mat, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &data32, 0),
        HIPBLAS_STATUS_INVALID_VALUE);

    // Test HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT with insufficient buffer size
    data32 = 1;
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutSetAttribute(
            mat, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &data32, sizeof(int32_t) / 2),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutSetAttribute(mat, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &data32, 1),
        HIPBLAS_STATUS_INVALID_VALUE);

    // Test HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET with insufficient buffer size
    data64 = ld * col;
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutSetAttribute(
            mat, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &data64, sizeof(int64_t) / 2),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                              mat, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &data64, 1),
                          HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutSetAttribute(
            mat, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, nullptr, sizeof(int64_t)),
        HIPBLAS_STATUS_INVALID_VALUE);

    // Test HIPBLASLT_MATRIX_LAYOUT_TYPE with insufficient buffer size
    udata32 = static_cast<uint32_t>(arg.a_type);
    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                              mat, HIPBLASLT_MATRIX_LAYOUT_TYPE, &udata32, sizeof(uint32_t) / 2),
                          HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutSetAttribute(mat, HIPBLASLT_MATRIX_LAYOUT_TYPE, &udata32, 1),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                              mat, HIPBLASLT_MATRIX_LAYOUT_TYPE, nullptr, sizeof(uint32_t)),
                          HIPBLAS_STATUS_INVALID_VALUE);

    // Test HIPBLASLT_MATRIX_LAYOUT_ORDER with insufficient buffer size
    data32 = HIPBLASLT_ORDER_COL;
    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                              mat, HIPBLASLT_MATRIX_LAYOUT_ORDER, &data32, sizeof(int32_t) / 2),
                          HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutSetAttribute(mat, HIPBLASLT_MATRIX_LAYOUT_ORDER, &data32, 1),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                              mat, HIPBLASLT_MATRIX_LAYOUT_ORDER, nullptr, sizeof(int32_t)),
                          HIPBLAS_STATUS_INVALID_VALUE);

    // Test HIPBLASLT_MATRIX_LAYOUT_ROWS with insufficient buffer size
    udata64 = static_cast<uint64_t>(row);
    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                              mat, HIPBLASLT_MATRIX_LAYOUT_ROWS, &udata64, sizeof(uint64_t) / 2),
                          HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutSetAttribute(mat, HIPBLASLT_MATRIX_LAYOUT_ROWS, &udata64, 1),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                              mat, HIPBLASLT_MATRIX_LAYOUT_ROWS, nullptr, sizeof(uint64_t)),
                          HIPBLAS_STATUS_INVALID_VALUE);

    // Test HIPBLASLT_MATRIX_LAYOUT_COLS with insufficient buffer size
    udata64 = static_cast<uint64_t>(col);
    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                              mat, HIPBLASLT_MATRIX_LAYOUT_COLS, &udata64, sizeof(uint64_t) / 2),
                          HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutSetAttribute(mat, HIPBLASLT_MATRIX_LAYOUT_COLS, &udata64, 1),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                              mat, HIPBLASLT_MATRIX_LAYOUT_COLS, nullptr, sizeof(uint64_t)),
                          HIPBLAS_STATUS_INVALID_VALUE);

    // Test HIPBLASLT_MATRIX_LAYOUT_LD with insufficient buffer size
    data64 = ld;
    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                              mat, HIPBLASLT_MATRIX_LAYOUT_LD, &data64, sizeof(int64_t) / 2),
                          HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutSetAttribute(mat, HIPBLASLT_MATRIX_LAYOUT_LD, &data64, 1),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                              mat, HIPBLASLT_MATRIX_LAYOUT_LD, nullptr, sizeof(int64_t)),
                          HIPBLAS_STATUS_INVALID_VALUE);

    // Test with invalid attribute (assuming there's a MAX value for bounds checking)
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutSetAttribute(
            mat, static_cast<hipblasLtMatrixLayoutAttribute_t>(999), &data32, sizeof(int32_t)),
        HIPBLAS_STATUS_INVALID_VALUE);
}

void testing_aux_mat_get_attr_bad_arg(const Arguments& arg)
{
    const int64_t row = 128;
    const int64_t col = 128;
    const int64_t ld  = 128;

    hipblaslt_local_matrix_layout mat(row, col, ld, arg.a_type);
    EXPECT_HIPBLAS_STATUS(mat.status(), HIPBLAS_STATUS_SUCCESS);

    int     data;
    int64_t data64;
    size_t  sizeWritten;

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutGetAttribute(
            nullptr, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &data, sizeof(int), &sizeWritten),
        HIPBLAS_STATUS_NOT_INITIALIZED);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutGetAttribute(
            mat, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, nullptr, sizeof(int), &sizeWritten),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutGetAttribute(
                              mat, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, nullptr, 0, nullptr),
                          HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutGetAttribute(
                              mat, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &data, 1, &sizeWritten),
                          HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutGetAttribute(mat,
                                          HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                          nullptr,
                                          sizeof(int64_t),
                                          &sizeWritten),
        HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutGetAttribute(
            mat, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &data64, sizeof(int), &sizeWritten),
        HIPBLAS_STATUS_INVALID_VALUE);
    // test sizeWritten is nullptr, and the return state should be success
    data = 0;
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutGetAttribute(
            mat, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &data, sizeof(int), nullptr),
        HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(data, 1);
}

void testing_aux_mat_set_get_attr(const Arguments& arg)
{
    const int64_t row = 128;
    const int64_t col = 128;
    const int64_t ld  = 128;

    hipblaslt_local_matrix_layout mat(row, col, ld, arg.a_type);
    EXPECT_HIPBLAS_STATUS(mat.status(), HIPBLAS_STATUS_SUCCESS);

    int32_t data, data_r;
    size_t  sizeWritten;

    data = 2;
    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                              mat, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &data, sizeof(data)),
                          HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutGetAttribute(
            mat, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &data_r, sizeof(data), &sizeWritten),
        HIPBLAS_STATUS_SUCCESS);
    ASSERT_TRUE(data_r == data);

    std::vector<int64_t> data64_v = {0, ld * col};
    int64_t              data64_r = 0;
    for(int64_t data64 : data64_v)
    {
        EXPECT_HIPBLAS_STATUS(
            hipblasLtMatrixLayoutSetAttribute(
                mat, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &data64, sizeof(int64_t)),
            HIPBLAS_STATUS_SUCCESS);

        EXPECT_HIPBLAS_STATUS(
            hipblasLtMatrixLayoutGetAttribute(mat,
                                              HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                              &data64_r,
                                              sizeof(int64_t),
                                              &sizeWritten),
            HIPBLAS_STATUS_SUCCESS);
        ASSERT_TRUE(data64_r == data64);
    }
}

void testing_aux_matmul_init_bad_arg(const Arguments& arg)
{
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescCreate(nullptr, arg.compute_type, arg.scale_type),
                          HIPBLAS_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_init(const Arguments& arg)
{
    const hipblasOperation_t opA = HIPBLAS_OP_T;
    const hipblasOperation_t opB = HIPBLAS_OP_N;

    hipblaslt_local_matmul_descr matmul(opA, opB, arg.compute_type, arg.scale_type);
    EXPECT_HIPBLAS_STATUS(matmul.status(), HIPBLAS_STATUS_SUCCESS);
}

void testing_aux_matmul_set_attr_bad_arg(const Arguments& arg)
{
    const hipblasOperation_t opA = HIPBLAS_OP_T;
    const hipblasOperation_t opB = HIPBLAS_OP_N;

    hipblaslt_local_matmul_descr matmul(opA, opB, arg.compute_type, arg.scale_type);
    EXPECT_HIPBLAS_STATUS(matmul.status(), HIPBLAS_STATUS_SUCCESS);

    hipblasLtEpilogue_t data = HIPBLASLT_EPILOGUE_RELU;
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                              nullptr, HIPBLASLT_MATMUL_DESC_EPILOGUE, &data, sizeof(data)),
                          HIPBLAS_STATUS_NOT_INITIALIZED);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                              matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, nullptr, sizeof(data)),
                          HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &data, 1),
        HIPBLAS_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_get_attr_bad_arg(const Arguments& arg)
{
    const hipblasOperation_t opA = HIPBLAS_OP_T;
    const hipblasOperation_t opB = HIPBLAS_OP_N;

    hipblaslt_local_matmul_descr matmul(opA, opB, arg.compute_type, arg.scale_type);
    EXPECT_HIPBLAS_STATUS(matmul.status(), HIPBLAS_STATUS_SUCCESS);

    hipblasLtEpilogue_t data = HIPBLASLT_EPILOGUE_RELU;
    size_t              sizeWritten;
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescGetAttribute(
            nullptr, HIPBLASLT_MATMUL_DESC_EPILOGUE, &data, sizeof(data), &sizeWritten),
        HIPBLAS_STATUS_NOT_INITIALIZED);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescGetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, nullptr, sizeof(data), &sizeWritten),
        HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                              matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &data, 1, &sizeWritten),
                          HIPBLAS_STATUS_INVALID_VALUE);

    void* dBias = nullptr;
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                              matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &dBias, 4, &sizeWritten),
                          HIPBLAS_STATUS_INVALID_VALUE);
    // test sizeWritten is nullptr, and the return state should be success
    data = HIPBLASLT_EPILOGUE_RELU;
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                              matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &data, sizeof(data), nullptr),
                          HIPBLAS_STATUS_SUCCESS);
    // test return buffer value equals HIPBLASLT_EPILOGUE_DEFAULT
    EXPECT_EQ(data, HIPBLASLT_EPILOGUE_DEFAULT);
}

void testing_aux_matmul_set_get_attr(const Arguments& arg)
{
    size_t                sizeWritten;
    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, arg.compute_type, arg.scale_type));
    int64_t m = 1;
    int64_t n = 1;
    int64_t k = 1;

    // For HIPBLASLT_MATMUL_DESC_TRANSA
    hipblasOperation_t transA   = HIPBLAS_OP_T;
    hipblasOperation_t transA_r = HIPBLAS_OP_N;

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, 0),
        HIPBLAS_STATUS_INVALID_VALUE); // edge case

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                              matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(int32_t) / 2),
                          HIPBLAS_STATUS_INVALID_VALUE); // edge case

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                              matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(int32_t)),
                          HIPBLAS_STATUS_SUCCESS); // normal case

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                              matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &transA_r, 0, nullptr),
                          HIPBLASLT_MATMUL_DESC_BIAS_POINTER); // edge case

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescGetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &transA_r, sizeof(transA_r) / 2, &sizeWritten),
        HIPBLAS_STATUS_INVALID_VALUE); // edge case

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescGetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &transA_r, sizeof(transA_r), &sizeWritten),
        HIPBLAS_STATUS_SUCCESS); // normal case

    ASSERT_TRUE(transA_r == transA); // validate

    // For HIPBLASLT_MATMUL_DESC_TRANSB

    hipblasOperation_t transB   = HIPBLAS_OP_N;
    hipblasOperation_t transB_r = HIPBLAS_OP_T;

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                              matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(int32_t) / 2),
                          HIPBLAS_STATUS_INVALID_VALUE); // edge case

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                              matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(int32_t)),
                          HIPBLAS_STATUS_SUCCESS); // normal case

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescGetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &transB_r, sizeof(transB_r) / 2, &sizeWritten),
        HIPBLAS_STATUS_INVALID_VALUE); // edge case

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescGetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &transB_r, sizeof(transB_r), &sizeWritten),
        HIPBLAS_STATUS_SUCCESS); // normal case

    ASSERT_TRUE(transB_r == transB); // validate

    // for HIPBLASLT_MATMUL_DESC_EPILOGUE
    hipblasLtEpilogue_t data   = HIPBLASLT_EPILOGUE_DEFAULT;
    hipblasLtEpilogue_t data_r = HIPBLASLT_EPILOGUE_RELU;
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                              matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &data, sizeof(data)),
                          HIPBLAS_STATUS_SUCCESS);
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescGetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &data_r, sizeof(data_r), &sizeWritten),
        HIPBLAS_STATUS_SUCCESS);
    ASSERT_TRUE(data_r == data);

    // for HIPBLASLT_MATMUL_DESC_BIAS_POINTER set and get
    void* d_bias;
    void* d_bias_r;
    CHECK_HIP_ERROR(hipMalloc(&d_bias, k * sizeof(hipblasLtHalf)));

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias, sizeof(void*) / 2),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                              matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias, sizeof(void*)),
                          HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescGetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias_r, sizeof(d_bias_r), &sizeWritten),
        HIPBLAS_STATUS_SUCCESS);

    ASSERT_TRUE(d_bias_r == d_bias);

    // for HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER
    // We create a new matmul_descr becasue we don't want to set matmulDesc->scaleAType as Scalar in advance.
    // If we set it as Scalar, it will skip some cases, which is not what we desire for.
    const hipblasOperation_t opA = HIPBLAS_OP_T;
    const hipblasOperation_t opB = HIPBLAS_OP_N;

    hipblaslt_local_matmul_descr matmul_descr(opA, opB, arg.compute_type, arg.scale_type);
    EXPECT_HIPBLAS_STATUS(matmul_descr.status(), HIPBLAS_STATUS_SUCCESS);

    float  h_scale_a_for_desc_a_scale_pointer = 2.f; // Example scale value, adjust as needed
    float* d_scale_a;
    float* d_scale_a_r;
    CHECK_HIP_ERROR(hipMalloc(&d_scale_a, sizeof(float)));
    CHECK_HIP_ERROR(hipMemcpy(
        d_scale_a, &h_scale_a_for_desc_a_scale_pointer, sizeof(float), hipMemcpyHostToDevice));

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(
            matmul_descr, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scale_a, sizeof(float*) / 2),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(
            matmul_descr, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scale_a, sizeof(float*)),
        HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(matmul_descr,
                                                          HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                          &d_scale_a_r,
                                                          sizeof(float*) / 2,
                                                          &sizeWritten),
                          HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(matmul_descr,
                                                          HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                          &d_scale_a_r,
                                                          sizeof(float*),
                                                          &sizeWritten),
                          HIPBLAS_STATUS_SUCCESS);
    ASSERT_TRUE(d_scale_a_r == d_scale_a); // validate

    // HIPBLASLT_MATMUL_DESC_A_SCALE_MODE

    hipblasLtMatmulMatrixScale_t scale_mode_a   = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    hipblasLtMatmulMatrixScale_t scale_mode_a_r = HIPBLASLT_MATMUL_MATRIX_SCALE_END;

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(matmul,
                                                          HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                                          &scale_mode_a_r,
                                                          sizeof(uint32_t),
                                                          &sizeWritten),
                          HIPBLAS_STATUS_INVALID_VALUE); // we didn't set it yet, so we get error

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode_a, sizeof(uint32_t) / 2),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode_a, sizeof(uint32_t)),
        HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(matmul,
                                                          HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                                          &scale_mode_a_r,
                                                          sizeof(uint32_t) / 2,
                                                          &sizeWritten),
                          HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(matmul,
                                                          HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                                          &scale_mode_a_r,
                                                          sizeof(uint32_t),
                                                          &sizeWritten),
                          HIPBLAS_STATUS_SUCCESS);

    ASSERT_TRUE(scale_mode_a_r == scale_mode_a); // validate

    // HIPBLASLT_MATMUL_DESC_B_SCALE_MODE

    hipblasLtMatmulMatrixScale_t scale_mode_b   = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    hipblasLtMatmulMatrixScale_t scale_mode_b_r = HIPBLASLT_MATMUL_MATRIX_SCALE_END;

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(matmul,
                                                          HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                                          &scale_mode_b_r,
                                                          sizeof(uint32_t),
                                                          &sizeWritten),
                          HIPBLAS_STATUS_INVALID_VALUE); // we didn't set it yet, so we get error

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode_b, sizeof(uint32_t) / 2),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode_b, sizeof(uint32_t)),
        HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(matmul,
                                                          HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                                          &scale_mode_b_r,
                                                          sizeof(uint32_t) / 2,
                                                          &sizeWritten),
                          HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(matmul,
                                                          HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                                          &scale_mode_b_r,
                                                          sizeof(uint32_t),
                                                          &sizeWritten),
                          HIPBLAS_STATUS_SUCCESS);

    ASSERT_TRUE(scale_mode_b_r == scale_mode_b); // validate

    scale_mode_a = HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    scale_mode_b = HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode_a, sizeof(uint32_t)),
        HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(matmul,
                                                          HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                                          &scale_mode_a_r,
                                                          sizeof(uint32_t),
                                                          &sizeWritten),
                          HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode_b, sizeof(uint32_t)),
        HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(matmul,
                                                          HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                                          &scale_mode_b_r,
                                                          sizeof(uint32_t),
                                                          &sizeWritten),
                          HIPBLAS_STATUS_SUCCESS);

    ASSERT_TRUE(scale_mode_a_r == scale_mode_a); // validate
    ASSERT_TRUE(scale_mode_b_r == scale_mode_b); // ditto

    scale_mode_a = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
    scale_mode_b = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode_a, sizeof(uint32_t)),
        HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(matmul,
                                                          HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                                          &scale_mode_a_r,
                                                          sizeof(uint32_t),
                                                          &sizeWritten),
                          HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode_b, sizeof(uint32_t)),
        HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(matmul,
                                                          HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                                          &scale_mode_b_r,
                                                          sizeof(uint32_t),
                                                          &sizeWritten),
                          HIPBLAS_STATUS_SUCCESS);

    ASSERT_TRUE(scale_mode_a_r == scale_mode_a); // validate

    scale_mode_a = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3; // will not set anything
    scale_mode_b = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3; // ditto

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode_a, sizeof(uint32_t)),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(matmul,
                                                          HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                                          &scale_mode_a_r,
                                                          sizeof(uint32_t),
                                                          &sizeWritten),
                          HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode_b, sizeof(uint32_t)),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(matmul,
                                                          HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                                          &scale_mode_b_r,
                                                          sizeof(uint32_t),
                                                          &sizeWritten),
                          HIPBLAS_STATUS_SUCCESS);

    ASSERT_TRUE(
        scale_mode_a_r
        == HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F); // validate, it's still the previous value as expected
    ASSERT_TRUE(scale_mode_b_r == HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F); // ditto

    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    // for HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER
    float  h_scale_b = 3.f;
    float* d_scale_b;
    float* d_scale_b_r;
    CHECK_HIP_ERROR(hipMalloc(&d_scale_b, sizeof(float)));
    CHECK_HIP_ERROR(
        hipMemcpyAsync(d_scale_b, &h_scale_b, sizeof(float), hipMemcpyHostToDevice, stream));

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scale_b, sizeof(float*) / 2),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scale_b, sizeof(float*)),
        HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(matmul,
                                                          HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                          &d_scale_b_r,
                                                          sizeof(float*) / 2,
                                                          &sizeWritten),
                          HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(matmul,
                                                          HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                          &d_scale_b_r,
                                                          sizeof(float*),
                                                          &sizeWritten),
                          HIPBLAS_STATUS_SUCCESS);

    ASSERT_TRUE(d_scale_b == d_scale_b_r); // validate

    // for HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER & HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER
    float  h_scale_c = 3.f;
    float  h_scale_d = 3.f;
    float* d_scale_c;
    float* d_scale_d;
    CHECK_HIP_ERROR(hipMalloc(&d_scale_c, sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_scale_d, sizeof(float)));
    CHECK_HIP_ERROR(
        hipMemcpyAsync(d_scale_c, &h_scale_c, sizeof(float), hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(
        hipMemcpyAsync(d_scale_d, &h_scale_d, sizeof(float), hipMemcpyHostToDevice, stream));

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER, &d_scale_c, sizeof(float*) / 2),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER, &d_scale_c, sizeof(float*)),
        HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_scale_d, sizeof(float*) / 2),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_scale_d, sizeof(float*)),
        HIPBLAS_STATUS_SUCCESS);

    // for HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER
    float  h_scale_e = 3.f;
    float* d_scale_e;
    CHECK_HIP_ERROR(hipMalloc(&d_scale_e, sizeof(float)));
    CHECK_HIP_ERROR(
        hipMemcpyAsync(d_scale_e, &h_scale_e, sizeof(float), hipMemcpyHostToDevice, stream));

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(matmul,
                                        HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER,
                                        &d_scale_e,
                                        sizeof(float*) / 2),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER, &d_scale_e, sizeof(float*)),
        HIPBLAS_STATUS_SUCCESS);

    // For HIPBLASLT_MATMUL_DESC_POINTER_MODE
    hipblasLtPointerMode_t pMode   = HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST;
    hipblasLtPointerMode_t pMode_r = HIPBLASLT_POINTER_MODE_HOST;
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_POINTER_MODE, &pMode, sizeof(pMode) / 2),
        HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                              matmul, HIPBLASLT_MATMUL_DESC_POINTER_MODE, &pMode, sizeof(pMode)),
                          HIPBLAS_STATUS_SUCCESS);
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(matmul,
                                                          HIPBLASLT_MATMUL_DESC_POINTER_MODE,
                                                          &pMode_r,
                                                          sizeof(pMode_r) / 2,
                                                          &sizeWritten),
                          HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescGetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_POINTER_MODE, &pMode_r, sizeof(pMode_r), &sizeWritten),
        HIPBLAS_STATUS_SUCCESS);

    ASSERT_TRUE(pMode_r == pMode); // validate

    // For HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE Set Desc Bias Data Type
    int32_t bias_data_type   = HIP_R_16F;
    int32_t bias_data_type_r = HIP_R_32F;

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(matmul,
                                                          HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                          &bias_data_type,
                                                          sizeof(bias_data_type) / 2),
                          HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type, sizeof(bias_data_type)),
        HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(matmul,
                                                          HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                          &bias_data_type_r,
                                                          sizeof(bias_data_type_r) / 2,
                                                          &sizeWritten),
                          HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(matmul,
                                                          HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                          &bias_data_type_r,
                                                          sizeof(bias_data_type_r),
                                                          &sizeWritten),
                          HIPBLAS_STATUS_SUCCESS);

    ASSERT_TRUE(bias_data_type_r == bias_data_type); // validate

    // For HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER Set auxiliary buffer
    void* d_aux_buffer;
    CHECK_HIP_ERROR(hipMalloc(&d_aux_buffer, m * n * sizeof(hipblasLtHalf)));
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &d_aux_buffer, sizeof(void*) / 2),
        HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &d_aux_buffer, sizeof(void*)),
        HIPBLAS_STATUS_SUCCESS);

    // For HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD Set auxiliary leading dimension (ld)
    const int64_t aux_ld = m;
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &aux_ld, sizeof(aux_ld) / 2),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &aux_ld, sizeof(aux_ld)),
        HIPBLAS_STATUS_SUCCESS);

    // for HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE & Set Epilogue Aux Batch Stride
    const int64_t aux_batch_stride = m * n;
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(matmul,
                                        HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE,
                                        &aux_batch_stride,
                                        sizeof(aux_batch_stride) / 2),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(matmul,
                                        HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE,
                                        &aux_batch_stride,
                                        sizeof(aux_batch_stride)),
        HIPBLAS_STATUS_SUCCESS);

    // for HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER
    void* d_out_amax;
    void* d_out_amax_r;
    CHECK_HIP_ERROR(hipMalloc(&d_out_amax, 1 * sizeof(float)));
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER, &d_out_amax, sizeof(void*) / 2),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER, &d_out_amax, sizeof(void*)),
        HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(matmul,
                                                          HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER,
                                                          &d_out_amax_r,
                                                          sizeof(d_out_amax_r) / 2,
                                                          &sizeWritten),
                          HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(matmul,
                                                          HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER,
                                                          &d_out_amax_r,
                                                          sizeof(d_out_amax_r),
                                                          &sizeWritten),
                          HIPBLAS_STATUS_SUCCESS);

    ASSERT_TRUE(d_out_amax == d_out_amax_r); // validate

    // for HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE
    hipDataType aux_type_r;
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(matmul,
                                        HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE,
                                        &arg.aux_type,
                                        sizeof(hipDataType) / 2),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(matmul,
                                        HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE,
                                        &arg.aux_type,
                                        sizeof(hipDataType)),
        HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescGetAttribute(matmul,
                                        HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE,
                                        &aux_type_r,
                                        sizeof(aux_type_r) / 2,
                                        &sizeWritten),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescGetAttribute(matmul,
                                        HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE,
                                        &aux_type_r,
                                        sizeof(aux_type_r),
                                        &sizeWritten),
        HIPBLAS_STATUS_SUCCESS);

    ASSERT_TRUE(arg.aux_type == aux_type_r); // validate

    // for HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT & HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT
    hipDataType computeTypeA   = HIP_R_16F;
    hipDataType computeTypeA_r = HIP_R_32F;
    hipDataType computeTypeB   = HIP_R_16F;
    hipDataType computeTypeB_r = HIP_R_32F;

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(matmul,
                                        HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT,
                                        &computeTypeA,
                                        sizeof(computeTypeA) / 2),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(matmul,
                                        HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT,
                                        &computeTypeA,
                                        sizeof(computeTypeA)),
        HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(matmul,
                                        HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT,
                                        &computeTypeB,
                                        sizeof(computeTypeB) / 2),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(matmul,
                                        HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT,
                                        &computeTypeB,
                                        sizeof(computeTypeB)),
        HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescGetAttribute(matmul,
                                        HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT,
                                        &computeTypeA_r,
                                        sizeof(computeTypeA_r) / 2,
                                        &sizeWritten),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescGetAttribute(matmul,
                                        HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT,
                                        &computeTypeA_r,
                                        sizeof(computeTypeA_r),
                                        &sizeWritten),
        HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescGetAttribute(matmul,
                                        HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT,
                                        &computeTypeB_r,
                                        sizeof(computeTypeB_r) / 2,
                                        &sizeWritten),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescGetAttribute(matmul,
                                        HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT,
                                        &computeTypeB_r,
                                        sizeof(computeTypeB_r),
                                        &sizeWritten),
        HIPBLAS_STATUS_SUCCESS);

    ASSERT_TRUE(computeTypeA_r == computeTypeA);
    ASSERT_TRUE(computeTypeB_r == computeTypeB);

    // for default

    void* default_ptr;
    void* default_ptr_r;
    CHECK_HIP_ERROR(hipMalloc(&default_ptr, sizeof(hipblasLtHalf)));
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                              matmul, HIPBLASLT_MATMUL_DESC_MAX, &default_ptr, sizeof(void*)),
                          HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescGetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_MAX, &default_ptr_r, sizeof(default_ptr_r), &sizeWritten),
        HIPBLAS_STATUS_INVALID_VALUE);

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
    CHECK_HIP_ERROR(hipFree(d_scale_a));
    CHECK_HIP_ERROR(hipFree(d_bias));
    CHECK_HIP_ERROR(hipFree(d_scale_b));
    CHECK_HIP_ERROR(hipFree(d_scale_c));
    CHECK_HIP_ERROR(hipFree(d_scale_d));
    CHECK_HIP_ERROR(hipFree(d_aux_buffer));
    CHECK_HIP_ERROR(hipFree(d_out_amax));
    CHECK_HIP_ERROR(hipFree(default_ptr));
}

void testing_aux_matmul_pref_get_attr_bad_arg(const Arguments& arg)
{
    hipblaslt_local_preference pref;
    EXPECT_HIPBLAS_STATUS(pref.status(), HIPBLAS_STATUS_SUCCESS);

    uint64_t data;
    size_t   sizeWritten;

    // Test with null preference (should be INVALID_VALUE)
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulPreferenceGetAttribute(
            nullptr, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &data, sizeof(data), &sizeWritten),
        HIPBLAS_STATUS_INVALID_VALUE);

    // Test with sizeInBytes = 0 and sizeWritten = NULL (should be INVALID_VALUE)
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulPreferenceGetAttribute(
                              pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &data, 0, nullptr),
                          HIPBLAS_STATUS_INVALID_VALUE);

    // Test with non-zero sizeInBytes and buf = NULL (should be INVALID_VALUE)
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulPreferenceGetAttribute(pref,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              nullptr,
                                              sizeof(uint64_t),
                                              &sizeWritten),
        HIPBLAS_STATUS_INVALID_VALUE);

    // Test with correct size (should be SUCCESS)
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulPreferenceGetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &data, sizeof(uint64_t), &sizeWritten),
        HIPBLAS_STATUS_SUCCESS);
}

void testing_aux_matmul_pref_get_attr(const Arguments& arg)
{
    hipblaslt_local_preference pref;
    EXPECT_HIPBLAS_STATUS(pref.status(), HIPBLAS_STATUS_SUCCESS);

    uint64_t data_set = 1024;
    uint64_t data_get = 0;
    size_t   sizeWritten;

    // Test edge cases
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulPreferenceSetAttribute(
            nullptr, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &data_set, sizeof(data_set)),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulPreferenceSetAttribute(
                              pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &data_set, 0),
                          HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulPreferenceGetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, nullptr, 0, &sizeWritten),
        HIPBLAS_STATUS_INVALID_VALUE);
    // Set the attribute
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulPreferenceSetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &data_set, sizeof(data_set)),
        HIPBLAS_STATUS_SUCCESS);

    // Get the attribute
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulPreferenceGetAttribute(pref,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &data_get,
                                              sizeof(data_get),
                                              &sizeWritten),
        HIPBLAS_STATUS_SUCCESS);

    // Verify that the get value matches the set value
    ASSERT_TRUE(data_get == data_set);
    ASSERT_TRUE(sizeWritten == sizeof(data_get));

    // Test getting other attributes (assuming they have default values)
    int32_t search_mode   = 0;
    int32_t search_mode_r = 1;
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulPreferenceSetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_SEARCH_MODE, &search_mode, sizeof(search_mode)),
        HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulPreferenceGetAttribute(pref,
                                                                HIPBLASLT_MATMUL_PREF_SEARCH_MODE,
                                                                &search_mode_r,
                                                                sizeof(search_mode_r),
                                                                &sizeWritten),
                          HIPBLAS_STATUS_SUCCESS);
    ASSERT_TRUE(data_get == data_set);

    // Test default value
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulPreferenceSetAttribute(
                              pref, HIPBLASLT_MATMUL_PREF_MAX, &search_mode, sizeof(search_mode)),
                          HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulPreferenceGetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX, &search_mode_r, sizeof(search_mode_r), &sizeWritten),
        HIPBLAS_STATUS_INVALID_VALUE);

    // You might want to add more attribute tests here
}

void testing_aux_matmul_alg_init_bad_arg(const Arguments& arg) {}

void testing_aux_matmul_alg_init(const Arguments& arg) {}

void testing_aux_get_sol_with_null_biasaddr(const Arguments& arg)
{
    using InTypeA   = hipblasLtHalf;
    using InTypeB   = hipblasLtHalf;
    using OutType   = hipblasLtHalf;
    using AlphaType = hipblasLtFloat;
    using BetaType  = hipblasLtFloat;

    hipStream_t        stream;
    hipblasLtHandle_t  handle;
    hipblasOperation_t trans_a     = HIPBLAS_OP_N;
    hipblasOperation_t trans_b     = HIPBLAS_OP_T;
    int64_t            m           = arg.M[0];
    int64_t            n           = arg.N[0];
    int64_t            k           = arg.K[0];
    int64_t            batch_count = 1;
    float              alpha       = arg.alpha;
    float              beta        = arg.beta;
    void*              d_a;
    void*              d_b;
    void*              d_c;
    void*              d_d;
    void*              a;
    void*              b;
    void*              c;
    void*              d;

    CHECK_HIP_ERROR(hipStreamCreate(&stream));
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));
    CHECK_HIP_ERROR(hipMalloc(&d_a, m * k * batch_count * sizeof(InTypeA)));
    CHECK_HIP_ERROR(hipMalloc(&d_b, n * k * batch_count * sizeof(InTypeB)));
    CHECK_HIP_ERROR(hipMalloc(&d_c, m * n * batch_count * sizeof(OutType)));
    CHECK_HIP_ERROR(hipMalloc(&d_d, m * n * batch_count * sizeof(OutType)));
    CHECK_HIP_ERROR(hipHostMalloc(&a, m * k * batch_count * sizeof(InTypeA)));
    CHECK_HIP_ERROR(hipHostMalloc(&b, n * k * batch_count * sizeof(InTypeB)));
    CHECK_HIP_ERROR(hipHostMalloc(&c, m * n * batch_count * sizeof(OutType)));
    CHECK_HIP_ERROR(hipHostMalloc(&d, m * n * batch_count * sizeof(OutType)));

    CHECK_HIP_ERROR(hipMemcpyAsync(
        d_a, a, m * k * batch_count * sizeof(InTypeA), hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(hipMemcpyAsync(
        d_b, b, n * k * batch_count * sizeof(InTypeB), hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(hipMemcpyAsync(
        d_c, c, m * n * batch_count * sizeof(OutType), hipMemcpyHostToDevice, stream));

    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, arg.a_type, m, k, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, arg.a_type, k, n, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, arg.a_type, m, n, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, arg.a_type, m, n, m));

    hipblasLtMatmulDesc_t matmul;
    hipblasLtEpilogue_t   epilogue = HIPBLASLT_EPILOGUE_BIAS;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, arg.compute_type, arg.scale_type));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    hipblasLtMatmulPreference_t pref;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
    const int                        request_solutions = 1;
    hipblasLtMatmulHeuristicResult_t heuristicResult[request_solutions];
    int                              returnedAlgoCount = 0;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                          matmul,
                                                          matA,
                                                          matB,
                                                          matC,
                                                          matD,
                                                          pref,
                                                          request_solutions,
                                                          heuristicResult,
                                                          &returnedAlgoCount));

    CHECK_SOLUTION_FOUND(returnedAlgoCount);

    CHECK_HIP_ERROR(hipFree(a));
    CHECK_HIP_ERROR(hipFree(b));
    CHECK_HIP_ERROR(hipFree(c));
    CHECK_HIP_ERROR(hipFree(d));
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_c));
    CHECK_HIP_ERROR(hipFree(d_d));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
    CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
}

// hipBLASLt API: For testing case of (alpha=0 && (A=NULL || B=NULL))
void testing_aux_get_sol_with_zero_alpha_null_a_b(const Arguments& arg)
{
    using InTypeA   = hipblasLtHalf;
    using InTypeB   = hipblasLtHalf;
    using OutType   = hipblasLtHalf;
    using AlphaType = hipblasLtFloat;
    using BetaType  = hipblasLtFloat;

    hipStream_t        stream;
    hipblasLtHandle_t  handle;
    hipblasOperation_t trans_a     = HIPBLAS_OP_N;
    hipblasOperation_t trans_b     = HIPBLAS_OP_T;
    int64_t            m           = arg.M[0];
    int64_t            n           = arg.N[0];
    int64_t            k           = arg.K[0];
    int64_t            batch_count = 1;
    // Setting alpha = 0.
    float alpha = 0;
    float beta  = arg.beta;
    // Setting d_a, d_b as nullptr.
    void* d_a = NULL;
    void* d_b = NULL;
    void* d_c;
    void* d_d;

    CHECK_HIP_ERROR(hipStreamCreate(&stream));
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));
    CHECK_HIP_ERROR(hipMalloc(&d_c, m * n * batch_count * sizeof(OutType)));
    CHECK_HIP_ERROR(hipMalloc(&d_d, m * n * batch_count * sizeof(OutType)));

    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, arg.a_type, m, k, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, arg.a_type, k, n, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, arg.a_type, m, n, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, arg.a_type, m, n, m));

    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, arg.compute_type, arg.scale_type));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));

    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    // Set User Preference attributes
    hipblasLtMatmulPreference_t pref;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));

    const int                        request_solutions = 1;
    hipblasLtMatmulHeuristicResult_t heuristicResult[request_solutions];
    int                              returnedAlgoCount = 0;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                          matmul,
                                                          matA,
                                                          matB,
                                                          matC,
                                                          matD,
                                                          pref,
                                                          request_solutions,
                                                          heuristicResult,
                                                          &returnedAlgoCount));

    CHECK_SOLUTION_FOUND(returnedAlgoCount);

    // Validation for solution running.
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle,
                                          matmul,
                                          &alpha,
                                          d_a,
                                          matA,
                                          d_b,
                                          matB,
                                          &beta,
                                          d_c,
                                          matC,
                                          d_d,
                                          matD,
                                          &heuristicResult[0].algo,
                                          nullptr,
                                          0,
                                          stream));

    CHECK_HIP_ERROR(hipFree(d_c));
    CHECK_HIP_ERROR(hipFree(d_d));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
    CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
}

// hipBLASLtExt API: For testing case of (alpha=0 && (A=NULL || B=NULL))
void testing_aux_get_sol_with_zero_alpha_null_a_b_ext(const Arguments& arg)
{
    using InTypeA   = hipblasLtHalf;
    using InTypeB   = hipblasLtHalf;
    using OutType   = hipblasLtHalf;
    using AlphaType = hipblasLtFloat;
    using BetaType  = hipblasLtFloat;

    hipStream_t        stream;
    hipblasLtHandle_t  handle;
    hipblasOperation_t trans_a     = HIPBLAS_OP_N;
    hipblasOperation_t trans_b     = HIPBLAS_OP_T;
    int64_t            m           = arg.M[0];
    int64_t            n           = arg.N[0];
    int64_t            k           = arg.K[0];
    int64_t            batch_count = 1;
    // Setting alpha = 0.
    float alpha = 0;
    float beta  = arg.beta;
    // Setting d_a, d_b as nullptr.
    void* d_a = NULL;
    void* d_b = NULL;
    void* d_c;
    void* d_d;

    CHECK_HIP_ERROR(hipStreamCreate(&stream));
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));
    CHECK_HIP_ERROR(hipMalloc(&d_c, m * n * batch_count * sizeof(OutType)));
    CHECK_HIP_ERROR(hipMalloc(&d_d, m * n * batch_count * sizeof(OutType)));

    hipblaslt_ext::GemmPreference gemmPref;
    hipblaslt_ext::Gemm           gemm(
        handle, trans_a, trans_b, arg.a_type, arg.a_type, arg.a_type, arg.a_type, arg.compute_type);

    hipblaslt_ext::GemmEpilogue
        epilogue; // No action needed, default is HIPBLASLT_EPILOGUE_DEFAULT. (Gemm only)
    hipblaslt_ext::GemmInputs inputs;
    inputs.setA(d_a);
    inputs.setB(d_b);
    inputs.setC(d_c);
    inputs.setD(d_d);
    inputs.setAlpha(&alpha);
    inputs.setBeta(&beta);
    gemm.setProblem(m, n, k, batch_count, epilogue, inputs);

    const int                                     request_solutions = 1;
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
    CHECK_HIPBLASLT_ERROR(gemm.algoGetHeuristic(request_solutions, gemmPref, heuristicResult));
    CHECK_SOLUTION_FOUND(heuristicResult.size());

    // Make sure to initialize every time when algo changes
    CHECK_HIPBLASLT_ERROR(gemm.initialize(heuristicResult[0].algo, nullptr));
    // Validation for solution running.
    CHECK_HIPBLASLT_ERROR(gemm.run(stream));

    CHECK_HIP_ERROR(hipFree(d_c));
    CHECK_HIP_ERROR(hipFree(d_d));
    CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
}

void testing_aux_matmul_alg_get_attr_bad_arg(const Arguments& arg) {}

void testing_aux_matmul_alg_null_matmul(const Arguments& arg)
{
    using InTypeA   = hipblasLtHalf;
    using InTypeB   = hipblasLtHalf;
    using OutType   = hipblasLtHalf;
    using AlphaType = hipblasLtFloat;
    using BetaType  = hipblasLtFloat;

    hipStream_t        stream;
    hipblasLtHandle_t  handle;
    hipblasOperation_t trans_a     = HIPBLAS_OP_N;
    hipblasOperation_t trans_b     = HIPBLAS_OP_T;
    int64_t            m           = arg.M[0];
    int64_t            n           = arg.N[0];
    int64_t            k           = arg.K[0];
    int64_t            batch_count = 1;
    float              alpha       = arg.alpha;
    float              beta        = arg.beta;
    void*              d_a;
    void*              d_b;
    void*              d_c;
    void*              d_d;
    void*              a;
    void*              b;
    void*              c;
    void*              d;

    CHECK_HIP_ERROR(hipStreamCreate(&stream));
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));
    CHECK_HIP_ERROR(hipMalloc(&d_a, m * k * batch_count * sizeof(InTypeA)));
    CHECK_HIP_ERROR(hipMalloc(&d_b, n * k * batch_count * sizeof(InTypeB)));
    CHECK_HIP_ERROR(hipMalloc(&d_c, m * n * batch_count * sizeof(OutType)));
    CHECK_HIP_ERROR(hipMalloc(&d_d, m * n * batch_count * sizeof(OutType)));
    CHECK_HIP_ERROR(hipHostMalloc(&a, m * k * batch_count * sizeof(InTypeA)));
    CHECK_HIP_ERROR(hipHostMalloc(&b, n * k * batch_count * sizeof(InTypeB)));
    CHECK_HIP_ERROR(hipHostMalloc(&c, m * n * batch_count * sizeof(OutType)));
    CHECK_HIP_ERROR(hipHostMalloc(&d, m * n * batch_count * sizeof(OutType)));

    CHECK_HIP_ERROR(hipMemcpyAsync(
        d_a, a, m * k * batch_count * sizeof(InTypeA), hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(hipMemcpyAsync(
        d_b, b, n * k * batch_count * sizeof(InTypeB), hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(hipMemcpyAsync(
        d_c, c, m * n * batch_count * sizeof(OutType), hipMemcpyHostToDevice, stream));

    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, arg.a_type, m, k, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, arg.a_type, k, n, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, arg.a_type, m, n, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, arg.a_type, m, n, m));

    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, arg.compute_type, arg.scale_type));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle,
                                          matmul,
                                          &alpha,
                                          d_a,
                                          matA,
                                          d_b,
                                          matB,
                                          &beta,
                                          d_c,
                                          matC,
                                          d_d,
                                          matD,
                                          nullptr,
                                          nullptr,
                                          0,
                                          0));

    CHECK_HIP_ERROR(hipFree(a));
    CHECK_HIP_ERROR(hipFree(b));
    CHECK_HIP_ERROR(hipFree(c));
    CHECK_HIP_ERROR(hipFree(d));
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_c));
    CHECK_HIP_ERROR(hipFree(d_d));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
    CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
}

void testing_aux_matmul_bad_ws_size(const Arguments& arg)
{
    using InTypeA   = hipblasLtHalf;
    using InTypeB   = hipblasLtHalf;
    using OutType   = hipblasLtHalf;
    using AlphaType = hipblasLtFloat;
    using BetaType  = hipblasLtFloat;

    hipStream_t        stream;
    hipblasLtHandle_t  handle;
    hipblasOperation_t trans_a     = HIPBLAS_OP_N;
    hipblasOperation_t trans_b     = HIPBLAS_OP_N;
    int64_t            m           = 2048;
    int64_t            n           = 2048;
    int64_t            k           = 2048;
    int64_t            batch_count = 1;
    float              alpha       = arg.alpha;
    float              beta        = arg.beta;
    void*              d_a;
    void*              d_b;
    void*              d_c;
    void*              d_d;
    void*              a;
    void*              b;
    void*              c;
    void*              d;

    CHECK_HIP_ERROR(hipStreamCreate(&stream));
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));
    CHECK_HIP_ERROR(hipMalloc(&d_a, m * k * batch_count * sizeof(InTypeA)));
    CHECK_HIP_ERROR(hipMalloc(&d_b, n * k * batch_count * sizeof(InTypeB)));
    CHECK_HIP_ERROR(hipMalloc(&d_c, m * n * batch_count * sizeof(OutType)));
    CHECK_HIP_ERROR(hipMalloc(&d_d, m * n * batch_count * sizeof(OutType)));
    CHECK_HIP_ERROR(hipHostMalloc(&a, m * k * batch_count * sizeof(InTypeA)));
    CHECK_HIP_ERROR(hipHostMalloc(&b, n * k * batch_count * sizeof(InTypeB)));
    CHECK_HIP_ERROR(hipHostMalloc(&c, m * n * batch_count * sizeof(OutType)));
    CHECK_HIP_ERROR(hipHostMalloc(&d, m * n * batch_count * sizeof(OutType)));

    CHECK_HIP_ERROR(hipMemcpyAsync(
        d_a, a, m * k * batch_count * sizeof(InTypeA), hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(hipMemcpyAsync(
        d_b, b, n * k * batch_count * sizeof(InTypeB), hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(hipMemcpyAsync(
        d_c, c, m * n * batch_count * sizeof(OutType), hipMemcpyHostToDevice, stream));

    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, arg.a_type, m, k, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, arg.a_type, k, n, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, arg.a_type, m, n, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, arg.a_type, m, n, m));

    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, arg.compute_type, arg.scale_type));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));

    hipblasLtMatmulPreference_t pref;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
    size_t max_workspace_size = 128 * 1024 * 1024;
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulPreferenceSetAttribute(pref,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &max_workspace_size,
                                              sizeof(max_workspace_size)),
        HIPBLAS_STATUS_SUCCESS);
    const int                        request_solutions = 1000;
    hipblasLtMatmulHeuristicResult_t heuristicResult[request_solutions];
    int                              returnedAlgoCount = 0;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                          matmul,
                                                          matA,
                                                          matB,
                                                          matC,
                                                          matD,
                                                          pref,
                                                          request_solutions,
                                                          heuristicResult,
                                                          &returnedAlgoCount));

    void* d_ws;
    CHECK_HIP_ERROR(hipMalloc(&d_ws, 1));
    for(int i = 0; i < returnedAlgoCount; i++)
    {
        if(heuristicResult[i].workspaceSize > 0)
        {
            //The normal behavior with insufficient ws size would be:
            //1. run normally without crash and return HIPBLAS_STATUS_SUCCESS
            //2. return HIPBLAS_STATUS_INVALID_VALUE
            auto status = hipblasLtMatmul(handle,
                                          matmul,
                                          &alpha,
                                          d_a,
                                          matA,
                                          d_b,
                                          matB,
                                          &beta,
                                          d_c,
                                          matC,
                                          d_d,
                                          matD,
                                          &heuristicResult[i].algo,
                                          d_ws,
                                          1,
                                          stream);
            CHECK_SUCCESS((status == HIPBLAS_STATUS_SUCCESS)
                          || (status == HIPBLAS_STATUS_INVALID_VALUE));
        }
    }
    CHECK_HIP_ERROR(hipFree(d_ws));

    CHECK_HIP_ERROR(hipFree(a));
    CHECK_HIP_ERROR(hipFree(b));
    CHECK_HIP_ERROR(hipFree(c));
    CHECK_HIP_ERROR(hipFree(d));
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_c));
    CHECK_HIP_ERROR(hipFree(d_d));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
    CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
}
void testing_aux_matmul_pref_init_bad_arg(const Arguments& arg)
{
    hipblasLtMatmulPreference_t pref;
    size_t                      workspace_size = 0;

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulPreferenceCreate(nullptr), HIPBLAS_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_pref_init(const Arguments& arg)
{
    hipblaslt_local_preference pref;
    EXPECT_HIPBLAS_STATUS(pref.status(), HIPBLAS_STATUS_SUCCESS);
}

void testing_aux_mat_copy(const Arguments& arg)
{
    hipblasLtMatmulDesc_t matmul_src;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul_src, arg.compute_type, arg.scale_type));
    hipblasLtMatmulDesc_t matmul_dest;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul_dest, HIPBLAS_COMPUTE_32F, HIP_R_32F));

    EXPECT_HIPBLAS_STATUS(hipblaslt_ext::copyMatmul(nullptr, matmul_dest),
                          HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_HIPBLAS_STATUS(hipblaslt_ext::copyMatmul(matmul_src, nullptr),
                          HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_HIPBLAS_STATUS(hipblaslt_ext::copyMatmul(matmul_src, matmul_dest),
                          HIPBLAS_STATUS_SUCCESS);

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul_src));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul_dest));
}

#ifdef CODE_COVERAGE
void testing_aux_auxiliary_func(const Arguments& arg)
{
    // Test gpu_arch_match
    int             deviceId;
    hipDeviceProp_t deviceProperties;
    static_cast<void>(hipGetDevice(&deviceId));
    static_cast<void>(hipGetDeviceProperties(&deviceProperties, deviceId));
    ASSERT_TRUE(gpu_arch_match(deviceProperties.gcnArchName, ""));
    ASSERT_TRUE(gpu_arch_match(deviceProperties.gcnArchName, "\\d"));

    // Test hipblas_status_to_string
    ASSERT_TRUE(hipblas_status_to_string(HIPBLAS_STATUS_SUCCESS) == "HIPBLAS_STATUS_SUCCESS");
    ASSERT_TRUE(hipblas_status_to_string(HIPBLAS_STATUS_NOT_INITIALIZED)
                == "HIPBLAS_STATUS_NOT_INITIALIZED");
    ASSERT_TRUE(hipblas_status_to_string(HIPBLAS_STATUS_ALLOC_FAILED)
                == "HIPBLAS_STATUS_ALLOC_FAILED");
    ASSERT_TRUE(hipblas_status_to_string(HIPBLAS_STATUS_INVALID_VALUE)
                == "HIPBLAS_STATUS_INVALID_VALUE");
    ASSERT_TRUE(hipblas_status_to_string(HIPBLAS_STATUS_MAPPING_ERROR)
                == "HIPBLAS_STATUS_MAPPING_ERROR");
    ASSERT_TRUE(hipblas_status_to_string(HIPBLAS_STATUS_EXECUTION_FAILED)
                == "HIPBLAS_STATUS_EXECUTION_FAILED");
    ASSERT_TRUE(hipblas_status_to_string(HIPBLAS_STATUS_INTERNAL_ERROR)
                == "HIPBLAS_STATUS_INTERNAL_ERROR");
    ASSERT_TRUE(hipblas_status_to_string(HIPBLAS_STATUS_NOT_SUPPORTED)
                == "HIPBLAS_STATUS_NOT_SUPPORTED");
    ASSERT_TRUE(hipblas_status_to_string(HIPBLAS_STATUS_ARCH_MISMATCH)
                == "HIPBLAS_STATUS_ARCH_MISMATCH");
    ASSERT_TRUE(hipblas_status_to_string(HIPBLAS_STATUS_INVALID_ENUM)
                == "HIPBLAS_STATUS_INVALID_ENUM");
    ASSERT_TRUE(hipblas_status_to_string(HIPBLAS_STATUS_UNKNOWN) == "HIPBLAS_STATUS_UNKNOWN");
    ASSERT_TRUE(hipblas_status_to_string(static_cast<hipblasStatus_t>(12))
                == "<undefined hipblasStatus_t value>");

    // Test hipblas_operation_to_string

    ASSERT_TRUE(hipblas_operation_to_string(HIPBLAS_OP_N) == "N");
    ASSERT_TRUE(hipblas_operation_to_string(HIPBLAS_OP_T) == "T");
    ASSERT_TRUE(hipblas_operation_to_string(HIPBLAS_OP_C) == "C");
    ASSERT_TRUE(hipblas_operation_to_string(static_cast<hipblasOperation_t>(114)) == "invalid");

    // Test char_to_hipblas_operation
    ASSERT_TRUE(char_to_hipblas_operation('N') == HIPBLAS_OP_N);
    ASSERT_TRUE(char_to_hipblas_operation('n') == HIPBLAS_OP_N);
    ASSERT_TRUE(char_to_hipblas_operation('T') == HIPBLAS_OP_T);
    ASSERT_TRUE(char_to_hipblas_operation('t') == HIPBLAS_OP_T);
    ASSERT_TRUE(char_to_hipblas_operation('C') == HIPBLAS_OP_C);
    ASSERT_TRUE(char_to_hipblas_operation('c') == HIPBLAS_OP_C);
    ASSERT_TRUE(char_to_hipblas_operation('X') == HIPBLASLT_OPERATION_INVALID);

    // Test char_to_hipblas_operation
    ASSERT_TRUE(char_to_hipblas_operation('N') == HIPBLAS_OP_N);
    ASSERT_TRUE(char_to_hipblas_operation('n') == HIPBLAS_OP_N);
    ASSERT_TRUE(char_to_hipblas_operation('T') == HIPBLAS_OP_T);
    ASSERT_TRUE(char_to_hipblas_operation('t') == HIPBLAS_OP_T);
    ASSERT_TRUE(char_to_hipblas_operation('C') == HIPBLAS_OP_C);
    ASSERT_TRUE(char_to_hipblas_operation('c') == HIPBLAS_OP_C);

    // Test hip_datatype_to_string
    ASSERT_TRUE(hip_datatype_to_string(HIP_R_32F) == "f32_r");
    ASSERT_TRUE(hip_datatype_to_string(HIP_R_64F) == "f64_r");
    ASSERT_TRUE(hip_datatype_to_string(HIP_R_16F) == "f16_r");
    ASSERT_TRUE(hip_datatype_to_string(HIP_R_16BF) == "bf16_r");
    ASSERT_TRUE(hip_datatype_to_string(HIP_R_8I) == "i8_r");
    ASSERT_TRUE(hip_datatype_to_string(HIP_R_32I) == "i32_r");
    ASSERT_TRUE(hip_datatype_to_string(HIP_R_8F_E4M3_FNUZ) == "f8_fnuz_r");
    ASSERT_TRUE(hip_datatype_to_string(HIP_R_8F_E5M2_FNUZ) == "bf8_fnuz_r");
    ASSERT_TRUE(hip_datatype_to_string(HIP_R_8F_E4M3) == "f8_r");
    ASSERT_TRUE(hip_datatype_to_string(HIP_R_8F_E5M2) == "bf8_r");
    ASSERT_TRUE(hip_datatype_to_string(static_cast<hipDataType>(HIP_R_6F_E2M3_EXT)) == "f6_r");
    ASSERT_TRUE(hip_datatype_to_string(static_cast<hipDataType>(HIP_R_6F_E3M2_EXT)) == "bf6_r");
    ASSERT_TRUE(hip_datatype_to_string(static_cast<hipDataType>(HIP_R_4F_E2M1_EXT)) == "f4_r");

    // Test hipblas_computetype_to_string
    hipblas_computetype_to_string(HIPBLAS_COMPUTE_16F);
    ASSERT_TRUE(hipblas_computetype_to_string(HIPBLAS_COMPUTE_16F) == "f16_r");
    ASSERT_TRUE(hipblas_computetype_to_string(HIPBLAS_COMPUTE_32F) == "f32_r");
    ASSERT_TRUE(hipblas_computetype_to_string(HIPBLAS_COMPUTE_32F_FAST_TF32) == "xf32_r");
    ASSERT_TRUE(hipblas_computetype_to_string(HIPBLAS_COMPUTE_64F) == "f64_r");
    ASSERT_TRUE(hipblas_computetype_to_string(HIPBLAS_COMPUTE_32I) == "i32_r");
    ASSERT_TRUE(hipblas_computetype_to_string(HIPBLAS_COMPUTE_32F_FAST_16F) == "f32_f16_r");
    ASSERT_TRUE(hipblas_computetype_to_string(HIPBLAS_COMPUTE_32F_FAST_16BF) == "f32_bf16_r");
    ASSERT_TRUE(hipblas_computetype_to_string(HIPBLAS_COMPUTE_64F_PEDANTIC)
                == "non-supported compute type");

    // Test string_to_hip_datatype
    ASSERT_TRUE(string_to_hip_datatype("f8_fnuz_r") == HIP_R_8F_E4M3_FNUZ);
    ASSERT_TRUE(string_to_hip_datatype("bf8_fnuz_r") == HIP_R_8F_E5M2_FNUZ);
    ASSERT_TRUE(string_to_hip_datatype("f8_r") == HIP_R_8F_E4M3);
    ASSERT_TRUE(string_to_hip_datatype("bf8_r") == HIP_R_8F_E5M2);
    ASSERT_TRUE(string_to_hip_datatype("f32_r") == HIP_R_32F);
    ASSERT_TRUE(string_to_hip_datatype("f64_r") == HIP_R_64F);
    ASSERT_TRUE(string_to_hip_datatype("f16_r") == HIP_R_16F);
    ASSERT_TRUE(string_to_hip_datatype("bf16_r") == HIP_R_16BF);
    ASSERT_TRUE(string_to_hip_datatype("i8_r") == HIP_R_8I);
    ASSERT_TRUE(string_to_hip_datatype("f6_r") == static_cast<hipDataType>(HIP_R_6F_E2M3_EXT));
    ASSERT_TRUE(string_to_hip_datatype("bf6_r") == static_cast<hipDataType>(HIP_R_6F_E3M2_EXT));
    ASSERT_TRUE(string_to_hip_datatype("f4_r") == static_cast<hipDataType>(HIP_R_4F_E2M1_EXT));
    ASSERT_TRUE(string_to_hip_datatype("i32_r") == HIP_R_32I);
    ASSERT_TRUE(string_to_hip_datatype("") == HIPBLASLT_DATATYPE_INVALID);

    // Test string_to_hip_datatype_assert
    string_to_hip_datatype_assert("f8_fnuz_r");

    // Test string_to_hipblas_computetype
    ASSERT_TRUE(string_to_hipblas_computetype("f32_r") == HIPBLAS_COMPUTE_32F);
    ASSERT_TRUE(string_to_hipblas_computetype("xf32_r") == HIPBLAS_COMPUTE_32F_FAST_TF32);
    ASSERT_TRUE(string_to_hipblas_computetype("f64_r") == HIPBLAS_COMPUTE_64F);
    ASSERT_TRUE(string_to_hipblas_computetype("i32_r") == HIPBLAS_COMPUTE_32I);
    ASSERT_TRUE(string_to_hipblas_computetype("f32_f16_r") == HIPBLAS_COMPUTE_32F_FAST_16F);
    ASSERT_TRUE(string_to_hipblas_computetype("f32_bf16_r") == HIPBLAS_COMPUTE_32F_FAST_16BF);
    ASSERT_TRUE(string_to_hipblas_computetype("") == HIPBLASLT_COMPUTE_TYPE_INVALID);

    // Test string_to_hipblas_computetype_assert
    string_to_hipblas_computetype_assert("f32_r");

    // Test string_to_epilogue_type
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_RELU") == HIPBLASLT_EPILOGUE_RELU);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_BIAS") == HIPBLASLT_EPILOGUE_BIAS);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_RELU_BIAS")
                == HIPBLASLT_EPILOGUE_RELU_BIAS);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_GELU") == HIPBLASLT_EPILOGUE_GELU);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_GELU_BIAS")
                == HIPBLASLT_EPILOGUE_GELU_BIAS);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_RELU_AUX")
                == HIPBLASLT_EPILOGUE_RELU_AUX);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_RELU_AUX_BIAS")
                == HIPBLASLT_EPILOGUE_RELU_AUX_BIAS);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_GELU_AUX")
                == HIPBLASLT_EPILOGUE_GELU_AUX);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_GELU_AUX_BIAS")
                == HIPBLASLT_EPILOGUE_GELU_AUX_BIAS);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_SIGMOID")
                == HIPBLASLT_EPILOGUE_SIGMOID);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_DGELU") == HIPBLASLT_EPILOGUE_DGELU);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_DGELU_BGRAD")
                == HIPBLASLT_EPILOGUE_DGELU_BGRAD);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_BGRADA") == HIPBLASLT_EPILOGUE_BGRADA);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_BGRADB") == HIPBLASLT_EPILOGUE_BGRADB);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_SWISH_EXT")
                == HIPBLASLT_EPILOGUE_SWISH_EXT);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT")
                == HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_CLAMP_EXT")
                == HIPBLASLT_EPILOGUE_CLAMP_EXT);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT")
                == HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_CLAMP_AUX_EXT")
                == HIPBLASLT_EPILOGUE_CLAMP_AUX_EXT);
    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT")
                == HIPBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT);

    ASSERT_TRUE(string_to_epilogue_type("HIPBLASLT_EPILOGUE_DEFAULT")
                == HIPBLASLT_EPILOGUE_DEFAULT);
    ASSERT_TRUE(string_to_epilogue_type("test") == static_cast<hipblasLtEpilogue_t>(0));

    // Test string_to_epilogue_type_assert
    ASSERT_TRUE(string_to_epilogue_type_assert("HIPBLASLT_EPILOGUE_RELU")
                == HIPBLASLT_EPILOGUE_RELU);

    // Test hipblaslt_isnan
    uint32_t other_type_value = 5;
    ASSERT_FALSE(hipblaslt_isnan(other_type_value));

    float non_integral_nan = std::numeric_limits<float>::quiet_NaN();
    ASSERT_TRUE(hipblaslt_isnan(non_integral_nan));

    hipblaslt_f8_fnuz arg_hipblaslt_f8_fnuz;
    arg_hipblaslt_f8_fnuz.__x = 0x80;
    ASSERT_TRUE(hipblaslt_isnan(arg_hipblaslt_f8_fnuz));

    hipblaslt_bf8_fnuz arg_hipblaslt_bf8_fnuz;
    arg_hipblaslt_bf8_fnuz.__x = 0x80;
    ASSERT_TRUE(hipblaslt_isnan(arg_hipblaslt_bf8_fnuz));

    hipblaslt_f8 arg_hipblaslt_f8;
    arg_hipblaslt_f8.__x = 0x80;
    ASSERT_TRUE(hipblaslt_isnan(arg_hipblaslt_f8));

    hipblaslt_bf8 arg_hipblaslt_bf8;
    arg_hipblaslt_bf8.__x = 0x7d;
    ASSERT_TRUE(hipblaslt_isnan(arg_hipblaslt_bf8));

    hipblasLtHalf pos_inf = static_cast<hipblasLtHalf>(INFINITY);
    ASSERT_TRUE(hipblaslt_isinf(pos_inf));

    hipblasLtHalf nan = static_cast<hipblasLtHalf>(NAN);
    ASSERT_TRUE(hipblaslt_isnan(nan));
}

void testing_aux_float8_func(const Arguments& arg)
{
    // Test hipblaslt_float8
    _Float16          f16 = 2.0;
    hipblaslt_f8_fnuz f8_fnuz_data(f16);
    ASSERT_TRUE(f16 == static_cast<_Float16>(f8_fnuz_data));
    f8_fnuz_data.__x = 0x00;
    ASSERT_TRUE(f8_fnuz_data.is_zero());
    f8_fnuz_data.__x = 0x80;
    ASSERT_TRUE(f8_fnuz_data.is_inf());
    hipblaslt_f8_fnuz f8_fnuz_data_copy;
    f8_fnuz_data_copy = f8_fnuz_data;
    ASSERT_TRUE(f8_fnuz_data_copy.__x == f8_fnuz_data.__x);

    // Test hipblaslt_f8
    hipblaslt_f8 f8_data(f16);
    ASSERT_TRUE(f16 == static_cast<_Float16>(f8_data));
    f8_data.__x = 0x00;
    ASSERT_TRUE(f8_data.is_zero());
    f8_data.__x = 0x80;
    ASSERT_TRUE(f8_data.is_inf());
    hipblaslt_f8 f8_data_copy;
    f8_data_copy = f8_data;
    ASSERT_TRUE(f8_data_copy.__x == f8_data.__x);

    // Test hipblaslt_bf8_fnuz
    hipblaslt_bf8_fnuz bf8_fnuz_data(f16);
    ASSERT_TRUE(f16 == static_cast<_Float16>(bf8_fnuz_data));
    bf8_fnuz_data.__x = 0x00;
    ASSERT_TRUE(bf8_fnuz_data.is_zero());
    bf8_fnuz_data.__x = 0x80;
    ASSERT_TRUE(bf8_fnuz_data.is_inf());
    hipblaslt_bf8_fnuz bf8_fnuz_data_copy;
    bf8_fnuz_data_copy = bf8_fnuz_data;
    ASSERT_TRUE(bf8_fnuz_data_copy.__x == bf8_fnuz_data.__x);

    // Test hipblaslt_bf8
    hipblaslt_bf8 bf8_data(f16);
    ASSERT_TRUE(f16 == static_cast<_Float16>(bf8_data));
    bf8_data.__x = 0x00;
    ASSERT_TRUE(bf8_data.is_zero());
    bf8_data.__x = 0xff;
    ASSERT_TRUE(bf8_data.is_nan());
    bf8_data.__x = 0xfc;
    ASSERT_TRUE(bf8_data.is_inf());
    hipblaslt_bf8 bf8_data_copy;
    bf8_data_copy = bf8_data;
    ASSERT_TRUE(bf8_data_copy.__x == bf8_data.__x);

    // namespace std functions
    // Test hipblaslt_f8_fnuz sin function
    hipblaslt_f8_fnuz f8_fnuz_sin_val(f16);
    hipblaslt_f8_fnuz f8_fnuz_sin_val_r = std::sin(f8_fnuz_sin_val);
    ASSERT_TRUE(f8_fnuz_sin_val_r == hipblaslt_f8_fnuz(sinf(float(f8_fnuz_sin_val))));

    // Test hipblaslt_f8_fnuz cos function
    hipblaslt_f8_fnuz f8_fnuz_cos_val(f16);
    hipblaslt_f8_fnuz f8_fnuz_cos_val_r = std::cos(f8_fnuz_cos_val);
    ASSERT_TRUE(f8_fnuz_cos_val_r == hipblaslt_f8_fnuz(cosf(float(f8_fnuz_cos_val))));

    // Test hipblaslt_f8 sin function
    hipblaslt_f8 f8_sin_val(f16);
    hipblaslt_f8 f8_sin_val_r = std::sin(f8_sin_val);
    ASSERT_TRUE(f8_sin_val_r == hipblaslt_f8(sinf(float(f8_sin_val))));

    // Test hipblaslt_f8 cos function
    hipblaslt_f8 f8_cos_val(f16);
    hipblaslt_f8 f8_cos_val_r = std::cos(f8_cos_val);
    ASSERT_TRUE(f8_cos_val_r == hipblaslt_f8(cosf(float(f8_cos_val))));

    // Test hipblaslt_bf8_fnuz sin function
    hipblaslt_bf8_fnuz bf8_fnuz_sin_val(f16);
    hipblaslt_bf8_fnuz bf8_fnuz_sin_val_r = std::sin(bf8_fnuz_sin_val);
    ASSERT_TRUE(bf8_fnuz_sin_val_r == hipblaslt_bf8_fnuz(sinf(float(bf8_fnuz_sin_val))));

    // Test hipblaslt_bf8_fnuz cos function
    hipblaslt_bf8_fnuz bf8_fnuz_cos_val(f16);
    hipblaslt_bf8_fnuz bf8_fnuz_cos_val_r = std::cos(bf8_fnuz_cos_val);
    ASSERT_TRUE(bf8_fnuz_cos_val_r == hipblaslt_bf8_fnuz(cosf(float(bf8_fnuz_cos_val))));

    // Test hipblaslt_bf8 sin function
    hipblaslt_bf8 bf8_sin_val(f16);
    hipblaslt_bf8 bf8_sin_val_r = std::sin(bf8_sin_val);
    ASSERT_TRUE(bf8_sin_val_r == hipblaslt_bf8(sinf(float(bf8_sin_val))));

    // Test hipblaslt_bf8 cos function
    hipblaslt_bf8 bf8_cos_val(f16);
    hipblaslt_bf8 bf8_cos_val_r = std::cos(bf8_cos_val);
    ASSERT_TRUE(bf8_cos_val_r == hipblaslt_bf8(cosf(float(bf8_cos_val))));

    // Test hipblaslt_f8_fnuz real function
    hipblaslt_f8_fnuz f8_fnuz_real_val(f16);
    hipblaslt_f8_fnuz f8_fnuz_real_val_r = std::real(f8_fnuz_real_val);
    ASSERT_TRUE(f8_fnuz_real_val_r == hipblaslt_f8_fnuz(std::real(float(f8_fnuz_real_val))));

    // Test hipblaslt_f8 real function
    hipblaslt_f8 f8_real_val(f16);
    hipblaslt_f8 f8_real_val_r = std::real(f8_real_val);
    ASSERT_TRUE(f8_real_val_r == hipblaslt_f8(std::real(float(f8_real_val))));

    // Test hipblaslt_bf8_fnuz real function
    hipblaslt_bf8_fnuz bf8_fnuz_real_val(f16);
    hipblaslt_bf8_fnuz bf8_fnuz_real_val_r = std::real(bf8_fnuz_real_val);
    ASSERT_TRUE(bf8_fnuz_real_val_r == hipblaslt_bf8_fnuz(std::real(float(bf8_fnuz_real_val))));

    // Test hipblaslt_bf8 real function
    hipblaslt_bf8 bf8_real_val(f16);
    hipblaslt_bf8 bf8_real_val_r = std::real(bf8_real_val);
    ASSERT_TRUE(bf8_real_val_r == hipblaslt_bf8(std::real(float(bf8_real_val))));

    // Test hipblaslt_f8_fnuz ostream operator
    hipblaslt_f8_fnuz  f8_fnuz_stream_val(f16);
    std::ostringstream f8_fnuz_stream;
    f8_fnuz_stream << f8_fnuz_stream_val;
    std::ostringstream f8_fnuz_expected;
    f8_fnuz_expected << float(f8_fnuz_stream_val);
    ASSERT_TRUE(f8_fnuz_stream.str() == f8_fnuz_expected.str());

    // Test hipblaslt_f8 ostream operator
    hipblaslt_f8       f8_stream_val(f16);
    std::ostringstream f8_stream;
    f8_stream << f8_stream_val;
    std::ostringstream f8_expected;
    f8_expected << float(f8_stream_val);
    ASSERT_TRUE(f8_stream.str() == f8_expected.str());

    // Test hipblaslt_bf8_fnuz ostream operator
    hipblaslt_bf8_fnuz bf8_fnuz_stream_val(f16);
    std::ostringstream bf8_fnuz_stream;
    bf8_fnuz_stream << bf8_fnuz_stream_val;
    std::ostringstream bf8_fnuz_expected;
    bf8_fnuz_expected << float(bf8_fnuz_stream_val);
    ASSERT_TRUE(bf8_fnuz_stream.str() == bf8_fnuz_expected.str());

    // Test hipblaslt_bf8 ostream operator
    hipblaslt_bf8      bf8_stream_val(f16);
    std::ostringstream bf8_stream;
    bf8_stream << bf8_stream_val;
    std::ostringstream bf8_expected;
    bf8_expected << float(bf8_stream_val);

    // Test code for hipblaslt_float8.h operator overloads
    _Float16 f16_a      = 2.5;
    _Float16 f16_b      = 1.5;
    float    float_val  = 3.0f;
    int32_t  int_val    = 4;
    double   double_val = 5.0;

    // Test addition operators with mixed types (float + f8 types)
    hipblaslt_f8_fnuz  f8_fnuz_val(f16_a);
    hipblaslt_f8       f8_val(f16_a);
    hipblaslt_bf8_fnuz bf8_fnuz_val(f16_a);
    hipblaslt_bf8      bf8_val(f16_a);

    // Test float + f8 types
    float result_f = float_val + f8_fnuz_val;
    ASSERT_TRUE(result_f == (float_val + float(f8_fnuz_val)));
    result_f = float_val + f8_val;
    ASSERT_TRUE(result_f == (float_val + float(f8_val)));
    result_f = float_val + bf8_fnuz_val;
    ASSERT_TRUE(result_f == (float_val + float(bf8_fnuz_val)));
    result_f = float_val + bf8_val;
    ASSERT_TRUE(result_f == (float_val + float(bf8_val)));

    // Test f8 types + float
    result_f = f8_fnuz_val + float_val;
    ASSERT_TRUE(result_f == (float(f8_fnuz_val) + float_val));
    result_f = f8_val + float_val;
    ASSERT_TRUE(result_f == (float(f8_val) + float_val));
    result_f = bf8_fnuz_val + float_val;
    ASSERT_TRUE(result_f == (float(bf8_fnuz_val) + float_val));
    result_f = bf8_val + float_val;
    ASSERT_TRUE(result_f == (float(bf8_val) + float_val));

    // Test mixed f8 types addition (returns float)
    result_f = f8_fnuz_val + bf8_fnuz_val;
    ASSERT_TRUE(result_f == (float(f8_fnuz_val) + float(bf8_fnuz_val)));
    result_f = f8_val + bf8_val;
    ASSERT_TRUE(result_f == (float(f8_val) + float(bf8_val)));
    result_f = bf8_fnuz_val + f8_fnuz_val;
    ASSERT_TRUE(result_f == (float(bf8_fnuz_val) + float(f8_fnuz_val)));
    result_f = bf8_val + f8_val;
    ASSERT_TRUE(result_f == (float(bf8_val) + float(f8_val)));

    // Test same type addition (returns same type)
    hipblaslt_f8_fnuz f8_fnuz_b(f16_b);
    hipblaslt_f8_fnuz f8_fnuz_result = f8_fnuz_val + f8_fnuz_b;
    ASSERT_TRUE(f8_fnuz_result == hipblaslt_f8_fnuz(float(f8_fnuz_val) + float(f8_fnuz_b)));

    hipblaslt_f8 f8_b(f16_b);
    hipblaslt_f8 f8_result = f8_val + f8_b;
    ASSERT_TRUE(f8_result == hipblaslt_f8(float(f8_val) + float(f8_b)));

    hipblaslt_bf8_fnuz bf8_fnuz_b(f16_b);
    hipblaslt_bf8_fnuz bf8_fnuz_result = bf8_fnuz_val + bf8_fnuz_b;
    ASSERT_TRUE(bf8_fnuz_result == hipblaslt_bf8_fnuz(float(bf8_fnuz_val) + float(bf8_fnuz_b)));

    hipblaslt_bf8 bf8_b(f16_b);
    hipblaslt_bf8 bf8_result = bf8_val + bf8_b;
    ASSERT_TRUE(bf8_result == hipblaslt_bf8(float(bf8_val) + float(bf8_b)));

    // Test += operators
    hipblaslt_f8_fnuz f8_fnuz_copy = f8_fnuz_val;
    f8_fnuz_copy += f8_fnuz_b;
    ASSERT_TRUE(f8_fnuz_copy == hipblaslt_f8_fnuz(float(f8_fnuz_val) + float(f8_fnuz_b)));

    hipblaslt_f8 f8_copy = f8_val;
    f8_copy += f8_b;
    ASSERT_TRUE(f8_copy == hipblaslt_f8(float(f8_val) + float(f8_b)));

    hipblaslt_bf8_fnuz bf8_fnuz_copy = bf8_fnuz_val;
    bf8_fnuz_copy += bf8_fnuz_b;
    ASSERT_TRUE(bf8_fnuz_copy == hipblaslt_bf8_fnuz(float(bf8_fnuz_val) + float(bf8_fnuz_b)));

    hipblaslt_bf8 bf8_copy = bf8_val;
    bf8_copy += bf8_b;
    ASSERT_TRUE(bf8_copy == hipblaslt_bf8(float(bf8_val) + float(bf8_b)));

    // Test multiplication operators (all return float)
    // Same type multiplication
    result_f = f8_fnuz_val * f8_fnuz_b;
    ASSERT_TRUE(result_f == (float(f8_fnuz_val) * float(f8_fnuz_b)));
    result_f = f8_val * f8_b;
    ASSERT_TRUE(result_f == (float(f8_val) * float(f8_b)));

    // Test float * f8 types
    result_f = float_val * f8_fnuz_val;
    ASSERT_TRUE(result_f == (float_val * float(f8_fnuz_val)));
    result_f = float_val * f8_val;
    ASSERT_TRUE(result_f == (float_val * float(f8_val)));

    // Test f8 types * float
    result_f = f8_fnuz_val * float_val;
    ASSERT_TRUE(result_f == (float(f8_fnuz_val) * float_val));
    result_f = f8_val * float_val;
    ASSERT_TRUE(result_f == (float(f8_val) * float_val));

    // Test int32_t * f8 types
    result_f = int_val * f8_fnuz_val;
    ASSERT_TRUE(result_f == ((float)int_val * float(f8_fnuz_val)));
    result_f = int_val * f8_val;
    ASSERT_TRUE(result_f == ((float)int_val * float(f8_val)));

    // Test double * f8 types
    result_f = double_val * f8_fnuz_val;
    ASSERT_TRUE(result_f == ((float)double_val * float(f8_fnuz_val)));
    result_f = double_val * f8_val;
    ASSERT_TRUE(result_f == ((float)double_val * float(f8_val)));

    // Test bf8 type multiplication tests
    result_f = bf8_fnuz_val * bf8_fnuz_b;
    ASSERT_TRUE(result_f == (float(bf8_fnuz_val) * float(bf8_fnuz_b)));
    result_f = bf8_val * bf8_b;
    ASSERT_TRUE(result_f == (float(bf8_val) * float(bf8_b)));

    result_f = float_val * bf8_fnuz_val;
    ASSERT_TRUE(result_f == (float_val * float(bf8_fnuz_val)));
    result_f = float_val * bf8_val;
    ASSERT_TRUE(result_f == (float_val * float(bf8_val)));

    result_f = bf8_fnuz_val * float_val;
    ASSERT_TRUE(result_f == (float(bf8_fnuz_val) * float_val));
    result_f = bf8_val * float_val;
    ASSERT_TRUE(result_f == (float(bf8_val) * float_val));

    result_f = int_val * bf8_fnuz_val;
    ASSERT_TRUE(result_f == ((float)int_val * float(bf8_fnuz_val)));
    result_f = int_val * bf8_val;
    ASSERT_TRUE(result_f == ((float)int_val * float(bf8_val)));

    result_f = double_val * bf8_fnuz_val;
    ASSERT_TRUE(result_f == ((float)double_val * float(bf8_fnuz_val)));
    result_f = double_val * bf8_val;
    ASSERT_TRUE(result_f == ((float)double_val * float(bf8_val)));

    // Test Mixed f8 and bf8 multiplication
    result_f = f8_fnuz_val * bf8_fnuz_val;
    ASSERT_TRUE(result_f == (float(f8_fnuz_val) * float(bf8_fnuz_val)));
    result_f = f8_val * bf8_val;
    ASSERT_TRUE(result_f == (float(f8_val) * float(bf8_val)));
    result_f = bf8_fnuz_val * f8_fnuz_val;
    ASSERT_TRUE(result_f == (float(bf8_fnuz_val) * float(f8_fnuz_val)));
    result_f = bf8_val * f8_val;
    ASSERT_TRUE(result_f == (float(bf8_val) * float(f8_val)));

    // Test comparison operators
    // Equality operators
    bool result_b = (f8_fnuz_val == f8_fnuz_b);
    ASSERT_TRUE(result_b == (f8_fnuz_val.__x == f8_fnuz_b.__x));
    result_b = (f8_val == f8_b);
    ASSERT_TRUE(result_b == (f8_val.__x == f8_b.__x));
    result_b = (bf8_fnuz_val == bf8_fnuz_b);
    ASSERT_TRUE(result_b == (bf8_fnuz_val.__x == bf8_fnuz_b.__x));
    result_b = (bf8_val == bf8_b);
    ASSERT_TRUE(result_b == (bf8_val.__x == bf8_b.__x));

    // Test Inequality operators
    result_b = (f8_fnuz_val != f8_fnuz_b);
    ASSERT_TRUE(result_b == (f8_fnuz_val.__x != f8_fnuz_b.__x));
    result_b = (f8_val != f8_b);
    ASSERT_TRUE(result_b == (f8_val.__x != f8_b.__x));
    result_b = (bf8_fnuz_val != bf8_fnuz_b);
    ASSERT_TRUE(result_b == (bf8_fnuz_val.__x != bf8_fnuz_b.__x));
    result_b = (bf8_val != bf8_b);
    ASSERT_TRUE(result_b == (bf8_val.__x != bf8_b.__x));

    // Test Greater than or equal operators
    result_b = (f8_fnuz_val >= f8_fnuz_b);
    ASSERT_TRUE(result_b == (static_cast<float>(f8_fnuz_val) >= static_cast<float>(f8_fnuz_b)));
    result_b = (f8_val >= f8_b);
    ASSERT_TRUE(result_b == (static_cast<float>(f8_val) >= static_cast<float>(f8_b)));

    // Test Greater than operators
    result_b = (f8_fnuz_val > f8_fnuz_b);
    ASSERT_TRUE(result_b == (static_cast<float>(f8_fnuz_val) > static_cast<float>(f8_fnuz_b)));
    result_b = (f8_val > f8_b);
    ASSERT_TRUE(result_b == (static_cast<float>(f8_val) > static_cast<float>(f8_b)));
}

void testing_aux_hipblaslt_ext_op_func(const Arguments& arg)
{
    ASSERT_TRUE(hipblasltGetTotalGranularityValue()
                == hipblasltClientPerformanceArgs::totalGranularity);
    ASSERT_TRUE(hipblasltGetTilesPerCuValue() == hipblasltClientPerformanceArgs::tilesPerCu);
    ASSERT_TRUE(hipblasltGetTile0Granularity() == hipblasltClientPerformanceArgs::tile0Granularity);
    ASSERT_TRUE(hipblasltGetTile1Granularity() == hipblasltClientPerformanceArgs::tile1Granularity);
    ASSERT_TRUE(hipblasltGetCuGranularity() == hipblasltClientPerformanceArgs::cuGranularity);
    ASSERT_TRUE(hipblasltGetWaveGranularity() == hipblasltClientPerformanceArgs::waveGranularity);
    ASSERT_TRUE(hipblasltGetCUs() == hipblasltClientPerformanceArgs::CUs);
    ASSERT_TRUE(hipblasltGetMemWriteBytesD() == hipblasltClientPerformanceArgs::memWriteBytesD);
    ASSERT_TRUE(hipblasltGetMemReadBytes() == hipblasltClientPerformanceArgs::memReadBytes);
}

void testing_aux_rocblaslt_utility_func(const Arguments& arg)
{
    // Test basic prefix functionality
    std::string result = prefix("TEST_LAYER", "test_caller");

    // Check that result is not empty
    ASSERT_TRUE(!result.empty());

    // Check that result contains expected components
    ASSERT_TRUE(result.find("[HIPBLASLT]") != std::string::npos);
    ASSERT_TRUE(result.find("[TEST_LAYER]") != std::string::npos);
    ASSERT_TRUE(result.find("[test_caller]") != std::string::npos);

    // Check date format pattern (YYYY-MM-DD)
    std::regex date_pattern(R"(\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\])");
    ASSERT_TRUE(std::regex_search(result, date_pattern));

    // Check process ID is included (numeric value in brackets)
    std::regex pid_pattern(R"(\[\d+\])");
    ASSERT_TRUE(std::regex_search(result, pid_pattern));

    // Test with different layer and caller names
    std::string result2 = prefix("ERROR", "function_name");
    ASSERT_TRUE(result2.find("[ERROR]") != std::string::npos);
    ASSERT_TRUE(result2.find("[function_name]") != std::string::npos);

    // Test with empty strings (edge case)
    std::string result3 = prefix("", "");
    ASSERT_TRUE(result3.find("[]") != std::string::npos); // Should contain empty brackets
    ASSERT_TRUE(result3.find("[HIPBLASLT]")
                != std::string::npos); // HIPBLASLT should always be present

    // Test format consistency - two calls should have similar structure
    std::string result4 = prefix("LAYER1", "caller1");
    std::string result5 = prefix("LAYER2", "caller2");

    // Both should contain HIPBLASLT
    ASSERT_TRUE(result4.find("[HIPBLASLT]") != std::string::npos);
    ASSERT_TRUE(result5.find("[HIPBLASLT]") != std::string::npos);

    // Both should match the expected format pattern
    std::regex full_pattern(
        R"(\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\[HIPBLASLT\]\[\d+\]\[.*\]\[.*\])");
    ASSERT_TRUE(std::regex_match(result4, full_pattern));
    ASSERT_TRUE(std::regex_match(result5, full_pattern));

    // Test hipDataType_to_string function
    ASSERT_TRUE(std::string_view{hipDataType_to_string(HIP_R_16F)} == "R_16F");
    ASSERT_TRUE(std::string_view{hipDataType_to_string(HIP_R_16BF)} == "R_16BF");
    ASSERT_TRUE(std::string_view{hipDataType_to_string(HIP_R_32F)} == "R_32F");
    ASSERT_TRUE(std::string_view{hipDataType_to_string(HIP_R_64F)} == "R_64F");
    ASSERT_TRUE(std::string_view{hipDataType_to_string(HIP_R_8F_E4M3_FNUZ)} == "R_8F_E4M3_FNUZ");
    ASSERT_TRUE(std::string_view{hipDataType_to_string(HIP_R_8F_E5M2_FNUZ)} == "R_8F_E5M2_FNUZ");
    ASSERT_TRUE(std::string_view{hipDataType_to_string(HIP_R_8F_E4M3)} == "R_8F_E4M3");
    ASSERT_TRUE(std::string_view{hipDataType_to_string(HIP_R_8F_E5M2)} == "R_8F_E5M2");
    ASSERT_TRUE(std::string_view{hipDataType_to_string(HIP_R_8I)} == "R_8I");
    ASSERT_TRUE(std::string_view{hipDataType_to_string(static_cast<hipDataType>(HIP_R_6F_E2M3_EXT))}
                == "R_6F_E2M3");
    ASSERT_TRUE(std::string_view{hipDataType_to_string(static_cast<hipDataType>(HIP_R_6F_E3M2_EXT))}
                == "R_6F_E3M2");
    ASSERT_TRUE(std::string_view{hipDataType_to_string(static_cast<hipDataType>(HIP_R_4F_E2M1_EXT))}
                == "R_4F_E2M1");
    ASSERT_TRUE(std::string_view{hipDataType_to_string(static_cast<hipDataType>(999))}
                == "Invalid");

    ASSERT_TRUE(std::string_view{hipDataType_to_bench_string(HIP_R_32F)} == "f32_r");
    ASSERT_TRUE(std::string_view{hipDataType_to_bench_string(HIP_R_64F)} == "f64_r");
    ASSERT_TRUE(std::string_view{hipDataType_to_bench_string(HIP_R_16F)} == "f16_r");
    ASSERT_TRUE(std::string_view{hipDataType_to_bench_string(HIP_R_16BF)} == "bf16_r");
    ASSERT_TRUE(std::string_view{hipDataType_to_bench_string(HIP_R_8I)} == "i8_r");
    ASSERT_TRUE(std::string_view{hipDataType_to_bench_string(HIP_R_32I)} == "i32_r");
    ASSERT_TRUE(std::string_view{hipDataType_to_bench_string(HIP_R_8F_E4M3_FNUZ)} == "f8_r");
    ASSERT_TRUE(std::string_view{hipDataType_to_bench_string(HIP_R_8F_E5M2_FNUZ)} == "bf8_r");
    ASSERT_TRUE(std::string_view{hipDataType_to_bench_string(HIP_R_8F_E4M3)} == "f8_r");
    ASSERT_TRUE(std::string_view{hipDataType_to_bench_string(HIP_R_8F_E5M2)} == "bf8_r");
    ASSERT_TRUE(std::string_view{hipDataType_to_bench_string(HIP_R_8F_E5M2_FNUZ)} == "bf8_r");
    ASSERT_TRUE(
        std::string_view{hipDataType_to_bench_string(static_cast<hipDataType>(HIP_R_6F_E2M3_EXT))}
        == "f6_r");
    ASSERT_TRUE(
        std::string_view{hipDataType_to_bench_string(static_cast<hipDataType>(HIP_R_6F_E3M2_EXT))}
        == "bf6_r");
    ASSERT_TRUE(
        std::string_view{hipDataType_to_bench_string(static_cast<hipDataType>(HIP_R_4F_E2M1_EXT))}
        == "f4_r");
    ASSERT_TRUE(std::string_view{
                    hipDataType_to_bench_string(static_cast<hipDataType>(HIP_R_4F_E2M1_EXT + 1))}
                == "invalid");

    // Test rocblaslt_compute_type_to_string
    ASSERT_TRUE(std::string_view{rocblaslt_compute_type_to_string(rocblaslt_compute_f16)}
                == "COMPUTE_16F");
    ASSERT_TRUE(std::string_view{rocblaslt_compute_type_to_string(rocblaslt_compute_f32)}
                == "COMPUTE_32F");
    ASSERT_TRUE(std::string_view{rocblaslt_compute_type_to_string(rocblaslt_compute_f32_fast_xf32)}
                == "COMPUTE_32XF");
    ASSERT_TRUE(std::string_view{rocblaslt_compute_type_to_string(rocblaslt_compute_f64)}
                == "COMPUTE_64F");
    ASSERT_TRUE(std::string_view{rocblaslt_compute_type_to_string(rocblaslt_compute_i32)}
                == "COMPUTE_32I");
    ASSERT_TRUE(std::string_view{rocblaslt_compute_type_to_string(rocblaslt_compute_f32_fast_f16)}
                == "COMPUTE_32F_16F");
    ASSERT_TRUE(std::string_view{rocblaslt_compute_type_to_string(rocblaslt_compute_f32_fast_bf16)}
                == "COMPUTE_32F_16BF");
    ASSERT_TRUE(
        std::string_view{rocblaslt_compute_type_to_string(static_cast<rocblaslt_compute_type>(999))}
        == "Invalid");

    // Test rocblaslt_matrix_layout_attributes_to_string
    ASSERT_TRUE(std::string_view{rocblaslt_matrix_layout_attributes_to_string(
                    ROCBLASLT_MATRIX_LAYOUT_BATCH_COUNT)}
                == "MATRIX_LAYOUT_BATCH_COUNT");
    ASSERT_TRUE(std::string_view{rocblaslt_matrix_layout_attributes_to_string(
                    ROCBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET)}
                == "MATRIX_LAYOUT_STRIDED_BATCH_OFFSET");
    ASSERT_TRUE(
        std::string_view{rocblaslt_matrix_layout_attributes_to_string(ROCBLASLT_MATRIX_LAYOUT_TYPE)}
        == "ROCBLASLT_MATRIX_LAYOUT_TYPE");
    ASSERT_TRUE(std::string_view{
                    rocblaslt_matrix_layout_attributes_to_string(ROCBLASLT_MATRIX_LAYOUT_ORDER)}
                == "ROCBLASLT_MATRIX_LAYOUT_ORDER");
    ASSERT_TRUE(
        std::string_view{rocblaslt_matrix_layout_attributes_to_string(ROCBLASLT_MATRIX_LAYOUT_ROWS)}
        == "ROCBLASLT_MATRIX_LAYOUT_ROWS");
    ASSERT_TRUE(
        std::string_view{rocblaslt_matrix_layout_attributes_to_string(ROCBLASLT_MATRIX_LAYOUT_COLS)}
        == "ROCBLASLT_MATRIX_LAYOUT_COLS");
    ASSERT_TRUE(
        std::string_view{rocblaslt_matrix_layout_attributes_to_string(ROCBLASLT_MATRIX_LAYOUT_LD)}
        == "ROCBLASLT_MATRIX_LAYOUT_LD");
    ASSERT_TRUE(
        std::string_view{rocblaslt_matrix_layout_attributes_to_string(ROCBLASLT_MATRIX_LAYOUT_MAX)}
        == "ROCBLASLT_MATRIX_LAYOUT_MAX");
    ASSERT_TRUE(std::string_view{rocblaslt_matrix_layout_attributes_to_string(
                    static_cast<rocblaslt_matrix_layout_attribute_>(999))}
                == "Invalid");

    // Test rocblaslt_matmul_desc_attributes_to_string
    ASSERT_TRUE(
        std::string_view{rocblaslt_matmul_desc_attributes_to_string(ROCBLASLT_MATMUL_DESC_TRANSA)}
        == "MATMUL_DESC_TRANSA");
    ASSERT_TRUE(
        std::string_view{rocblaslt_matmul_desc_attributes_to_string(ROCBLASLT_MATMUL_DESC_TRANSB)}
        == "MATMUL_DESC_TRANSB");
    ASSERT_TRUE(
        std::string_view{rocblaslt_matmul_desc_attributes_to_string(ROCBLASLT_MATMUL_DESC_EPILOGUE)}
        == "MATMUL_DESC_EPILOGUE");
    ASSERT_TRUE(std::string_view{
                    rocblaslt_matmul_desc_attributes_to_string(ROCBLASLT_MATMUL_DESC_BIAS_POINTER)}
                == "MATMUL_DESC_BIAS_POINTER");
    ASSERT_TRUE(std::string_view{rocblaslt_matmul_desc_attributes_to_string(
                    ROCBLASLT_MATMUL_DESC_BIAS_DATA_TYPE)}
                == "MATMUL_DESC_BIAS_DATA_TYPE");
    ASSERT_TRUE(std::string_view{rocblaslt_matmul_desc_attributes_to_string(
                    ROCBLASLT_MATMUL_DESC_A_SCALE_POINTER)}
                == "MATMUL_DESC_A_SCALE_POINTER");
    ASSERT_TRUE(std::string_view{rocblaslt_matmul_desc_attributes_to_string(
                    ROCBLASLT_MATMUL_DESC_B_SCALE_POINTER)}
                == "MATMUL_DESC_B_SCALE_POINTER");
    ASSERT_TRUE(std::string_view{rocblaslt_matmul_desc_attributes_to_string(
                    ROCBLASLT_MATMUL_DESC_C_SCALE_POINTER)}
                == "MATMUL_DESC_C_SCALE_POINTER");
    ASSERT_TRUE(std::string_view{rocblaslt_matmul_desc_attributes_to_string(
                    ROCBLASLT_MATMUL_DESC_D_SCALE_POINTER)}
                == "MATMUL_DESC_D_SCALE_POINTER");
    ASSERT_TRUE(std::string_view{rocblaslt_matmul_desc_attributes_to_string(
                    ROCBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER)}
                == "MATMUL_DESC_EPILOGUE_AUX_POINTER");
    ASSERT_TRUE(std::string_view{rocblaslt_matmul_desc_attributes_to_string(
                    ROCBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD)}
                == "MATMUL_DESC_EPILOGUE_AUX_LD");
    ASSERT_TRUE(std::string_view{rocblaslt_matmul_desc_attributes_to_string(
                    ROCBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE)}
                == "MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE");
    ASSERT_TRUE(std::string_view{
                    rocblaslt_matmul_desc_attributes_to_string(ROCBLASLT_MATMUL_DESC_POINTER_MODE)}
                == "MATMUL_DESC_POINTER_MODE");
    ASSERT_TRUE(std::string_view{rocblaslt_matmul_desc_attributes_to_string(
                    ROCBLASLT_MATMUL_DESC_AMAX_D_POINTER)}
                == "MATMUL_DESC_AMAX_D_POINTER");
    ASSERT_TRUE(std::string_view{rocblaslt_matmul_desc_attributes_to_string(
                    ROCBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE)}
                == "MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE");
    ASSERT_TRUE(std::string_view{rocblaslt_matmul_desc_attributes_to_string(
                    ROCBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT)}
                == "MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT");
    ASSERT_TRUE(std::string_view{rocblaslt_matmul_desc_attributes_to_string(
                    ROCBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT)}
                == "MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT");
    ASSERT_TRUE(std::string_view{rocblaslt_matmul_desc_attributes_to_string(
                    static_cast<rocblaslt_matmul_desc_attributes>(999))}
                == "Invalid");

    // Test hipblasOperation_to_string
    ASSERT_TRUE(std::string_view{hipblasOperation_to_string(HIPBLAS_OP_N)} == "OP_N");
    ASSERT_TRUE(std::string_view{hipblasOperation_to_string(HIPBLAS_OP_T)} == "OP_T");
    ASSERT_TRUE(std::string_view{hipblasOperation_to_string(HIPBLAS_OP_C)} == "OP_C");
    ASSERT_TRUE(std::string_view{hipblasOperation_to_string(static_cast<hipblasOperation_t>(999))}
                == "Invalid");

    // Test rocblaslt_layer_mode2string
    ASSERT_TRUE(std::string_view{rocblaslt_layer_mode2string(rocblaslt_layer_mode_none)} == "None");
    ASSERT_TRUE(std::string_view{rocblaslt_layer_mode2string(rocblaslt_layer_mode_log_error)}
                == "Error");
    ASSERT_TRUE(std::string_view{rocblaslt_layer_mode2string(rocblaslt_layer_mode_log_trace)}
                == "Trace");
    ASSERT_TRUE(std::string_view{rocblaslt_layer_mode2string(rocblaslt_layer_mode_log_hints)}
                == "Hints");
    ASSERT_TRUE(std::string_view{rocblaslt_layer_mode2string(rocblaslt_layer_mode_log_info)}
                == "Info");
    ASSERT_TRUE(std::string_view{rocblaslt_layer_mode2string(rocblaslt_layer_mode_log_api)}
                == "Api");
    ASSERT_TRUE(std::string_view{rocblaslt_layer_mode2string(rocblaslt_layer_mode_log_bench)}
                == "Bench");
    ASSERT_TRUE(std::string_view{rocblaslt_layer_mode2string(rocblaslt_layer_mode_log_profile)}
                == "Profile");
    ASSERT_TRUE(
        std::string_view{rocblaslt_layer_mode2string(rocblaslt_layer_mode_log_extended_profile)}
        == "ExtendedProfile");
    ASSERT_TRUE(
        std::string_view{rocblaslt_layer_mode2string(static_cast<rocblaslt_layer_mode>(999))}
        == "Invalid");

    // Test rocblaslt_epilogue_to_string
    ASSERT_TRUE(std::string_view{rocblaslt_epilogue_to_string(ROCBLASLT_EPILOGUE_DEFAULT)}
                == "EPILOGUE_DEFAULT");
    ASSERT_TRUE(std::string_view{rocblaslt_epilogue_to_string(ROCBLASLT_EPILOGUE_RELU)}
                == "EPILOGUE_RELU");
    ASSERT_TRUE(std::string_view{rocblaslt_epilogue_to_string(ROCBLASLT_EPILOGUE_BIAS)}
                == "EPILOGUE_BIAS");
    ASSERT_TRUE(std::string_view{rocblaslt_epilogue_to_string(ROCBLASLT_EPILOGUE_RELU_BIAS)}
                == "EPILOGUE_RELU_BIAS");
    ASSERT_TRUE(std::string_view{rocblaslt_epilogue_to_string(ROCBLASLT_EPILOGUE_GELU)}
                == "EPILOGUE_GELU");
    ASSERT_TRUE(std::string_view{rocblaslt_epilogue_to_string(ROCBLASLT_EPILOGUE_DGELU)}
                == "EPILOGUE_DGELU");
    ASSERT_TRUE(std::string_view{rocblaslt_epilogue_to_string(ROCBLASLT_EPILOGUE_GELU_BIAS)}
                == "EPILOGUE_GELU_BIAS");
    ASSERT_TRUE(std::string_view{rocblaslt_epilogue_to_string(ROCBLASLT_EPILOGUE_GELU_AUX)}
                == "EPILOGUE_GELU_AUX");
    ASSERT_TRUE(std::string_view{rocblaslt_epilogue_to_string(ROCBLASLT_EPILOGUE_GELU_AUX_BIAS)}
                == "EPILOGUE_GELU_AUX_BIAS");
    ASSERT_TRUE(std::string_view{rocblaslt_epilogue_to_string(ROCBLASLT_EPILOGUE_SIGMOID)}
                == "EPILOGUE_SIGMOID");
    ASSERT_TRUE(std::string_view{rocblaslt_epilogue_to_string(ROCBLASLT_EPILOGUE_DGELU_BGRAD)}
                == "EPILOGUE_DGELU_BGRAD");
    ASSERT_TRUE(std::string_view{rocblaslt_epilogue_to_string(ROCBLASLT_EPILOGUE_BGRADA)}
                == "EPILOGUE_DGELU_BGRADA");
    ASSERT_TRUE(std::string_view{rocblaslt_epilogue_to_string(ROCBLASLT_EPILOGUE_BGRADB)}
                == "EPILOGUE_DGELU_BGRADB");
    ASSERT_TRUE(std::string_view{rocblaslt_epilogue_to_string(ROCBLASLT_EPILOGUE_SWISH_EXT)}
                == "EPILOGUE_SWISH_EXT");
    ASSERT_TRUE(std::string_view{rocblaslt_epilogue_to_string(ROCBLASLT_EPILOGUE_SWISH_BIAS_EXT)}
                == "EPILOGUE_SWISH_BIAS_EXT");
    ASSERT_TRUE(std::string_view{rocblaslt_epilogue_to_string(static_cast<rocblaslt_epilogue>(999))}
                == "Invalid epilogue");

    // Test all valid rocblaslt_compute_type values
    ASSERT_TRUE(std::string_view{rocblaslt_compute_type_string(rocblaslt_compute_f16)} == "f16_r");
    ASSERT_TRUE(std::string_view{rocblaslt_compute_type_string(rocblaslt_compute_f32)} == "f32_r");
    ASSERT_TRUE(std::string_view{rocblaslt_compute_type_string(rocblaslt_compute_f32_fast_xf32)}
                == "xf32_r");
    ASSERT_TRUE(std::string_view{rocblaslt_compute_type_string(rocblaslt_compute_i32)} == "i32_r");
    ASSERT_TRUE(std::string_view{rocblaslt_compute_type_string(rocblaslt_compute_f64)} == "f64_r");
    ASSERT_TRUE(std::string_view{rocblaslt_compute_type_string(rocblaslt_compute_f32_fast_f16)}
                == "f32_f16_r");
    ASSERT_TRUE(std::string_view{rocblaslt_compute_type_string(rocblaslt_compute_f32_fast_bf16)}
                == "f32_bf16_r");
    ASSERT_TRUE(std::string_view{rocblaslt_compute_type_string(rocblaslt_compute_f32_fast_f8)}
                == "f32_f8_r");
    ASSERT_TRUE(std::string_view{rocblaslt_compute_type_string(rocblaslt_compute_f32_fast_f8_fnuz)}
                == "f32_f8_fnuz_r");
    ASSERT_TRUE(std::string_view{rocblaslt_compute_type_string(rocblaslt_compute_f32_fast_bf8)}
                == "f32_bf8_fnuz_r");
    ASSERT_TRUE(std::string_view{rocblaslt_compute_type_string(rocblaslt_compute_f32_fast_bf8_fnuz)}
                == "f32_bf8_r");
    ASSERT_TRUE(std::string_view{rocblaslt_compute_type_string(rocblaslt_compute_f32_fast_f8bf8)}
                == "f32_f8bf8_r");
    ASSERT_TRUE(
        std::string_view{rocblaslt_compute_type_string(rocblaslt_compute_f32_fast_f8bf8_fnuz)}
        == "f32_f8bf8_fnuz_r");
    ASSERT_TRUE(std::string_view{rocblaslt_compute_type_string(rocblaslt_compute_f32_fast_bf8f8)}
                == "f32_bf8f8_r");
    ASSERT_TRUE(
        std::string_view{rocblaslt_compute_type_string(rocblaslt_compute_f32_fast_bf8f8_fnuz)}
        == "f32_bf8f8_fnuz_r");
    ASSERT_TRUE(
        std::string_view{rocblaslt_compute_type_string(static_cast<rocblaslt_compute_type>(999))}
        == "invalidType");

    // Test rocblaslt_matrix_layout_to_string function
    // Create test matrix layout structures
    _rocblaslt_matrix_layout mat1, mat2, mat3, mat4;

    // Test case 1: Single batch matrix (batch_count <= 1)
    mat1.type         = HIP_R_32F;
    mat1.m            = 128;
    mat1.n            = 256;
    mat1.ld           = 128;
    mat1.batch_count  = 1;
    mat1.batch_stride = 0;

    std::string mat1_result = rocblaslt_matrix_layout_to_string(&mat1);
    ASSERT_TRUE(!mat1_result.empty());
    ASSERT_TRUE(mat1_result.find("[type=R_32F") != std::string::npos);
    ASSERT_TRUE(mat1_result.find("rows=128") != std::string::npos);
    ASSERT_TRUE(mat1_result.find("cols=256") != std::string::npos);
    ASSERT_TRUE(mat1_result.find("ld=128") != std::string::npos);
    // Should NOT contain batch_count and batch_stride for single batch
    ASSERT_TRUE(mat1_result.find("batch_count") == std::string::npos);
    ASSERT_TRUE(mat1_result.find("batch_stride") == std::string::npos);

    // Test case 2: Batch matrix (batch_count > 1)
    mat2.type         = HIP_R_16F;
    mat2.m            = 64;
    mat2.n            = 128;
    mat2.ld           = 64;
    mat2.batch_count  = 4;
    mat2.batch_stride = 8192;

    std::string mat2_result = rocblaslt_matrix_layout_to_string(&mat2);
    ASSERT_TRUE(!mat2_result.empty());
    ASSERT_TRUE(mat2_result.find("[type=R_16F") != std::string::npos);
    ASSERT_TRUE(mat2_result.find("rows=64") != std::string::npos);
    ASSERT_TRUE(mat2_result.find("cols=128") != std::string::npos);
    ASSERT_TRUE(mat2_result.find("ld=64") != std::string::npos);
    // Should contain batch_count and batch_stride for batch matrix
    ASSERT_TRUE(mat2_result.find("batch_count=4") != std::string::npos);
    ASSERT_TRUE(mat2_result.find("batch_stride=8192") != std::string::npos);

    // Test case 3: Zero batch count
    mat3.type         = HIP_R_64F;
    mat3.m            = 32;
    mat3.n            = 64;
    mat3.ld           = 32;
    mat3.batch_count  = 0;
    mat3.batch_stride = 0;

    std::string mat3_result = rocblaslt_matrix_layout_to_string(&mat3);
    ASSERT_TRUE(!mat3_result.empty());
    ASSERT_TRUE(mat3_result.find("[type=R_64F") != std::string::npos);
    ASSERT_TRUE(mat3_result.find("rows=32") != std::string::npos);
    ASSERT_TRUE(mat3_result.find("cols=64") != std::string::npos);
    ASSERT_TRUE(mat3_result.find("ld=32") != std::string::npos);
    // batch_count <= 1, so should NOT contain batch info
    ASSERT_TRUE(mat3_result.find("batch_count") == std::string::npos);
    ASSERT_TRUE(mat3_result.find("batch_stride") == std::string::npos);

    // Test case 4: Different data types
    mat4.type         = HIP_R_8F_E4M3_FNUZ;
    mat4.m            = 16;
    mat4.n            = 32;
    mat4.ld           = 16;
    mat4.batch_count  = 8;
    mat4.batch_stride = 512;

    std::string mat4_result = rocblaslt_matrix_layout_to_string(&mat4);
    ASSERT_TRUE(!mat4_result.empty());
    ASSERT_TRUE(mat4_result.find("[type=R_8F_E4M3_FNUZ") != std::string::npos);
    ASSERT_TRUE(mat4_result.find("rows=16") != std::string::npos);
    ASSERT_TRUE(mat4_result.find("cols=32") != std::string::npos);
    ASSERT_TRUE(mat4_result.find("ld=16") != std::string::npos);
    ASSERT_TRUE(mat4_result.find("batch_count=8") != std::string::npos);
    ASSERT_TRUE(mat4_result.find("batch_stride=512") != std::string::npos);

    // Test format consistency - all results should start with '[' and end with ']'
    ASSERT_TRUE(mat1_result.front() == '[' && mat1_result.back() == ']');
    ASSERT_TRUE(mat2_result.front() == '[' && mat2_result.back() == ']');
    ASSERT_TRUE(mat3_result.front() == '[' && mat3_result.back() == ']');
    ASSERT_TRUE(mat4_result.front() == '[' && mat4_result.back() == ']');

    // Test edge case: Large dimensions
    _rocblaslt_matrix_layout mat_large;
    mat_large.type         = HIP_R_16BF;
    mat_large.m            = 4096;
    mat_large.n            = 8192;
    mat_large.ld           = 4096;
    mat_large.batch_count  = 16;
    mat_large.batch_stride = 33554432; // 4096 * 8192

    std::string result_large = rocblaslt_matrix_layout_to_string(&mat_large);
    ASSERT_TRUE(!result_large.empty());
    ASSERT_TRUE(result_large.find("rows=4096") != std::string::npos);
    ASSERT_TRUE(result_large.find("cols=8192") != std::string::npos);
    ASSERT_TRUE(result_large.find("batch_count=16") != std::string::npos);
    ASSERT_TRUE(result_large.find("batch_stride=33554432") != std::string::npos);

    // Test rocblaslt_matmul_desc_to_string function
    // Create test matmul descriptor structures
    _rocblaslt_matmul_desc desc1, desc2, desc3, desc4;

    // Test case 1: No bias, no epilogue extension (simplest case)
    desc1.compute_type = rocblaslt_compute_f32;
    desc1.scale_type   = HIP_R_32F;
    desc1.op_A         = HIPBLAS_OP_N;
    desc1.op_B         = HIPBLAS_OP_T;
    desc1.epilogue     = ROCBLASLT_EPILOGUE_DEFAULT;
    desc1.bias         = nullptr;
    desc1.bias_type    = HIPBLASLT_DATATYPE_INVALID;
    desc1.aux_type     = HIPBLASLT_DATATYPE_INVALID;
    desc1.e            = nullptr;
    desc1.lde          = 0;

    std::string desc_result1 = rocblaslt_matmul_desc_to_string(&desc1);
    ASSERT_TRUE(!desc_result1.empty());
    ASSERT_TRUE(desc_result1.find("[computeType=COMPUTE_32F") != std::string::npos);
    ASSERT_TRUE(desc_result1.find("scaleType=R_32F") != std::string::npos);
    ASSERT_TRUE(desc_result1.find("transA=OP_N") != std::string::npos);
    ASSERT_TRUE(desc_result1.find("transB=OP_T") != std::string::npos);
    ASSERT_TRUE(desc_result1.find("epilogue=EPILOGUE_DEFAULT") != std::string::npos);
    ASSERT_TRUE(desc_result1.find("biasPointer=0x") != std::string::npos);
    // Should NOT contain epilogue aux or bias type info
    ASSERT_TRUE(desc_result1.find("epilogueAuxPointer") == std::string::npos);
    ASSERT_TRUE(desc_result1.find("biasType") == std::string::npos);
    ASSERT_TRUE(desc_result1.front() == '[' && desc_result1.back() == ']');

    // Test case 2: With bias, no epilogue extension
    desc2.compute_type = rocblaslt_compute_f16;
    desc2.scale_type   = HIP_R_16F;
    desc2.op_A         = HIPBLAS_OP_T;
    desc2.op_B         = HIPBLAS_OP_N;
    desc2.epilogue     = ROCBLASLT_EPILOGUE_BIAS;
    desc2.bias         = reinterpret_cast<void*>(0x12345678);
    desc2.bias_type    = HIP_R_16F;
    desc2.aux_type     = HIPBLASLT_DATATYPE_INVALID;
    desc2.e            = nullptr;
    desc2.lde          = 0;

    std::string desc_result2 = rocblaslt_matmul_desc_to_string(&desc2);
    ASSERT_TRUE(!desc_result2.empty());
    ASSERT_TRUE(desc_result2.find("computeType=COMPUTE_16F") != std::string::npos);
    ASSERT_TRUE(desc_result2.find("scaleType=R_16F") != std::string::npos);
    ASSERT_TRUE(desc_result2.find("transA=OP_T") != std::string::npos);
    ASSERT_TRUE(desc_result2.find("transB=OP_N") != std::string::npos);
    ASSERT_TRUE(desc_result2.find("epilogue=EPILOGUE_BIAS") != std::string::npos);
    ASSERT_TRUE(desc_result2.find("biasType=R_16F") != std::string::npos);
    // Should NOT contain epilogue aux info
    ASSERT_TRUE(desc_result2.find("epilogueAuxPointer") == std::string::npos);
    ASSERT_TRUE(desc_result2.front() == '[' && desc_result2.back() == ']');

    // Test case 3: No bias, with epilogue extension
    desc3.compute_type = rocblaslt_compute_f32;
    desc3.scale_type   = HIP_R_32F;
    desc3.op_A         = HIPBLAS_OP_N;
    desc3.op_B         = HIPBLAS_OP_N;
    desc3.epilogue     = ROCBLASLT_EPILOGUE_GELU_AUX;
    desc3.bias         = nullptr;
    desc3.bias_type    = HIPBLASLT_DATATYPE_INVALID;
    desc3.aux_type     = HIP_R_32F;
    desc3.e            = reinterpret_cast<void*>(0x87654321);
    desc3.lde          = 128;

    std::string desc_result3 = rocblaslt_matmul_desc_to_string(&desc3);
    ASSERT_TRUE(!desc_result3.empty());
    ASSERT_TRUE(desc_result3.find("computeType=COMPUTE_32F") != std::string::npos);
    ASSERT_TRUE(desc_result3.find("epilogue=EPILOGUE_GELU_AUX") != std::string::npos);
    ASSERT_TRUE(desc_result3.find("epilogueAuxPointer=0x") != std::string::npos);
    ASSERT_TRUE(desc_result3.find("epilogueAuxLd=128") != std::string::npos);
    ASSERT_TRUE(desc_result3.find("epilogueAuxDataType=R_32F") != std::string::npos);
    // Should NOT contain bias type info
    ASSERT_TRUE(desc_result3.find("biasType") == std::string::npos);
    ASSERT_TRUE(desc_result3.front() == '[' && desc_result3.back() == ']');

    // Test case 4: With bias and epilogue extension (most complex case)
    desc4.compute_type = rocblaslt_compute_f64;
    desc4.scale_type   = HIP_R_64F;
    desc4.op_A         = HIPBLAS_OP_C;
    desc4.op_B         = HIPBLAS_OP_C;
    desc4.epilogue     = ROCBLASLT_EPILOGUE_GELU_AUX_BIAS;
    desc4.bias         = reinterpret_cast<void*>(0xABCDEF00);
    desc4.bias_type    = HIP_R_64F;
    desc4.aux_type     = HIP_R_64F;
    desc4.e            = reinterpret_cast<void*>(0x11223344);
    desc4.lde          = 256;

    std::string desc_result4 = rocblaslt_matmul_desc_to_string(&desc4);
    ASSERT_TRUE(!desc_result4.empty());
    ASSERT_TRUE(desc_result4.find("computeType=COMPUTE_64F") != std::string::npos);
    ASSERT_TRUE(desc_result4.find("scaleType=R_64F") != std::string::npos);
    ASSERT_TRUE(desc_result4.find("transA=OP_C") != std::string::npos);
    ASSERT_TRUE(desc_result4.find("transB=OP_C") != std::string::npos);
    ASSERT_TRUE(desc_result4.find("epilogue=EPILOGUE_GELU_AUX_BIAS") != std::string::npos);
    ASSERT_TRUE(desc_result4.find("biasType=R_64F") != std::string::npos);
    ASSERT_TRUE(desc_result4.find("epilogueAuxPointer=0x") != std::string::npos);
    ASSERT_TRUE(desc_result4.find("epilogueAuxLd=256") != std::string::npos);
    ASSERT_TRUE(desc_result4.find("epilogueAuxDataType=R_64F") != std::string::npos);
    ASSERT_TRUE(desc_result4.front() == '[' && desc_result4.back() == ']');

    // Test case 5: Epilogue extension without aux_type
    _rocblaslt_matmul_desc desc5;
    desc5.compute_type = rocblaslt_compute_i32;
    desc5.scale_type   = HIP_R_32I;
    desc5.op_A         = HIPBLAS_OP_N;
    desc5.op_B         = HIPBLAS_OP_N;
    desc5.epilogue     = ROCBLASLT_EPILOGUE_DGELU_BGRAD;
    desc5.bias         = nullptr;
    desc5.bias_type    = HIPBLASLT_DATATYPE_INVALID;
    desc5.aux_type     = HIPBLASLT_DATATYPE_INVALID; // Invalid aux type
    desc5.e            = reinterpret_cast<void*>(0x55555555);
    desc5.lde          = 64;

    std::string desc_result5 = rocblaslt_matmul_desc_to_string(&desc5);
    ASSERT_TRUE(!desc_result5.empty());
    ASSERT_TRUE(desc_result5.find("computeType=COMPUTE_32I") != std::string::npos);
    ASSERT_TRUE(desc_result5.find("epilogue=EPILOGUE_DGELU_BGRAD") != std::string::npos);
    ASSERT_TRUE(desc_result5.find("epilogueAuxPointer=0x") != std::string::npos);
    ASSERT_TRUE(desc_result5.find("epilogueAuxLd=64") != std::string::npos);
    // Should NOT contain epilogueAuxDataType since aux_type is invalid
    ASSERT_TRUE(desc_result5.find("epilogueAuxDataType") == std::string::npos);
    ASSERT_TRUE(desc_result5.front() == '[' && desc_result5.back() == ']');

    // Test different data types
    _rocblaslt_matmul_desc desc6;
    desc6.compute_type = rocblaslt_compute_f32_fast_bf16;
    desc6.scale_type   = HIP_R_16BF;
    desc6.op_A         = HIPBLAS_OP_N;
    desc6.op_B         = HIPBLAS_OP_T;
    desc6.epilogue     = ROCBLASLT_EPILOGUE_RELU_BIAS;
    desc6.bias         = reinterpret_cast<void*>(0x99999999);
    desc6.bias_type    = HIP_R_8F_E4M3_FNUZ;
    desc6.aux_type     = HIPBLASLT_DATATYPE_INVALID;
    desc6.e            = nullptr;
    desc6.lde          = 0;

    std::string desc_result6 = rocblaslt_matmul_desc_to_string(&desc6);
    ASSERT_TRUE(!desc_result6.empty());
    ASSERT_TRUE(desc_result6.find("computeType=COMPUTE_32F_16BF") != std::string::npos);
    ASSERT_TRUE(desc_result6.find("scaleType=R_16BF") != std::string::npos);
    ASSERT_TRUE(desc_result6.find("epilogue=EPILOGUE_RELU_BIAS") != std::string::npos);
    ASSERT_TRUE(desc_result6.find("biasType=R_8F_E4M3_FNUZ") != std::string::npos);
    ASSERT_TRUE(desc_result6.front() == '[' && desc_result6.back() == ']');

    // Test case 1: No exception (nullptr) - should return success
    rocblaslt_status excep_result1 = exception_to_rocblaslt_status(nullptr);
    ASSERT_TRUE(excep_result1 == rocblaslt_status_success);

    // Test case 2: Catch rocblaslt_status exception - should return the status value
    std::exception_ptr status_exception;
    try
    {
        throw rocblaslt_status_invalid_handle;
    }
    catch(...)
    {
        status_exception = std::current_exception();
    }

    rocblaslt_status excep_result2 = exception_to_rocblaslt_status(status_exception);
    ASSERT_TRUE(excep_result2 == rocblaslt_status_invalid_handle);

    // Test case 3: Different rocblaslt_status values
    std::exception_ptr status_exception2;
    try
    {
        throw rocblaslt_status_not_initialized;
    }
    catch(...)
    {
        status_exception2 = std::current_exception();
    }

    rocblaslt_status excep_result3 = exception_to_rocblaslt_status(status_exception2);
    ASSERT_TRUE(excep_result3 == rocblaslt_status_not_initialized);

    // Test case 4: Catch std::bad_alloc - should return memory error
    std::exception_ptr bad_alloc_exception;
    try
    {
        throw std::bad_alloc();
    }
    catch(...)
    {
        bad_alloc_exception = std::current_exception();
    }

    rocblaslt_status excep_result4 = exception_to_rocblaslt_status(bad_alloc_exception);
    ASSERT_TRUE(excep_result4 == rocblaslt_status_memory_error);

    // Test all epilogue values that should return true (activation enabled)
    ASSERT_TRUE(is_act_enabled(ROCBLASLT_EPILOGUE_RELU) == true);
    ASSERT_TRUE(is_act_enabled(ROCBLASLT_EPILOGUE_RELU_BIAS) == true);
    ASSERT_TRUE(is_act_enabled(ROCBLASLT_EPILOGUE_GELU) == true);
    ASSERT_TRUE(is_act_enabled(ROCBLASLT_EPILOGUE_GELU_BIAS) == true);
    ASSERT_TRUE(is_act_enabled(ROCBLASLT_EPILOGUE_GELU_AUX) == true);
    ASSERT_TRUE(is_act_enabled(ROCBLASLT_EPILOGUE_GELU_AUX_BIAS) == true);
    ASSERT_TRUE(is_act_enabled(ROCBLASLT_EPILOGUE_DGELU) == true);
    ASSERT_TRUE(is_act_enabled(ROCBLASLT_EPILOGUE_DGELU_BGRAD) == true);
    ASSERT_TRUE(is_act_enabled(ROCBLASLT_EPILOGUE_SWISH_EXT) == true);
    ASSERT_TRUE(is_act_enabled(ROCBLASLT_EPILOGUE_SWISH_BIAS_EXT) == true);
    ASSERT_TRUE(is_act_enabled(ROCBLASLT_EPILOGUE_CLAMP_EXT) == true);
    ASSERT_TRUE(is_act_enabled(ROCBLASLT_EPILOGUE_CLAMP_BIAS_EXT) == true);
    ASSERT_TRUE(is_act_enabled(ROCBLASLT_EPILOGUE_SIGMOID) == true);
    // Test all epilogue values that should return false (activation disabled)
    ASSERT_TRUE(is_act_enabled(ROCBLASLT_EPILOGUE_DEFAULT) == false);
    ASSERT_TRUE(is_act_enabled(ROCBLASLT_EPILOGUE_BIAS) == false);
}

void testing_aux_status_func(const Arguments& arg)
{
    // Test get_rocblaslt_status_for_hip_status function
    ASSERT_TRUE(get_rocblaslt_status_for_hip_status(hipSuccess) == rocblaslt_status_success);
    ASSERT_TRUE(get_rocblaslt_status_for_hip_status(hipErrorMemoryAllocation)
                == rocblaslt_status_memory_error);
    ASSERT_TRUE(get_rocblaslt_status_for_hip_status(hipErrorLaunchOutOfResources)
                == rocblaslt_status_memory_error);
    ASSERT_TRUE(get_rocblaslt_status_for_hip_status(hipErrorInvalidDevicePointer)
                == rocblaslt_status_invalid_pointer);
    ASSERT_TRUE(get_rocblaslt_status_for_hip_status(hipErrorInvalidDevice)
                == rocblaslt_status_invalid_handle);
    ASSERT_TRUE(get_rocblaslt_status_for_hip_status(hipErrorInvalidResourceHandle)
                == rocblaslt_status_invalid_handle);
    ASSERT_TRUE(get_rocblaslt_status_for_hip_status(hipErrorInvalidValue)
                == rocblaslt_status_internal_error);
    ASSERT_TRUE(get_rocblaslt_status_for_hip_status(hipErrorNoDevice)
                == rocblaslt_status_internal_error);
    ASSERT_TRUE(get_rocblaslt_status_for_hip_status(hipErrorUnknown)
                == rocblaslt_status_internal_error);

    // Test default case with unknown/unmapped hip errors -> rocblaslt_status_internal_error
    ASSERT_TRUE(get_rocblaslt_status_for_hip_status(static_cast<hipError_t>(999))
                == rocblaslt_status_internal_error);
}

void testing_aux_hipblaslt_func(const Arguments& arg)
{
    // Test RocBlasLtStatusToHIPStatus
    ASSERT_TRUE(RocBlasLtStatusToHIPStatus(rocblaslt_status_success) == HIPBLAS_STATUS_SUCCESS);
    ASSERT_TRUE(RocBlasLtStatusToHIPStatus(rocblaslt_status_invalid_handle)
                == HIPBLAS_STATUS_NOT_INITIALIZED);
    ASSERT_TRUE(RocBlasLtStatusToHIPStatus(rocblaslt_status_not_implemented)
                == HIPBLAS_STATUS_INTERNAL_ERROR);
    ASSERT_TRUE(RocBlasLtStatusToHIPStatus(rocblaslt_status_invalid_pointer)
                == HIPBLAS_STATUS_INVALID_VALUE);
    ASSERT_TRUE(RocBlasLtStatusToHIPStatus(rocblaslt_status_invalid_size)
                == HIPBLAS_STATUS_INVALID_VALUE);
    ASSERT_TRUE(RocBlasLtStatusToHIPStatus(rocblaslt_status_memory_error)
                == HIPBLAS_STATUS_ALLOC_FAILED);
    ASSERT_TRUE(RocBlasLtStatusToHIPStatus(rocblaslt_status_internal_error)
                == HIPBLAS_STATUS_INTERNAL_ERROR);
    ASSERT_TRUE(RocBlasLtStatusToHIPStatus(rocblaslt_status_invalid_value)
                == HIPBLAS_STATUS_INVALID_VALUE);
    ASSERT_TRUE(RocBlasLtStatusToHIPStatus(rocblaslt_status_arch_mismatch)
                == HIPBLAS_STATUS_ARCH_MISMATCH);

    // Test default case - should throw HIPBLAS_STATUS_INVALID_ENUM
    bool            exception_thrown = false;
    hipblasStatus_t thrown_status    = HIPBLAS_STATUS_SUCCESS;

    try
    {
        RocBlasLtStatusToHIPStatus(static_cast<rocblaslt_status_>(999));
    }
    catch(hipblasStatus_t status)
    {
        exception_thrown = true;
        thrown_status    = status;
    }
    catch(...)
    {
        exception_thrown = true;
        thrown_status    = HIPBLAS_STATUS_INTERNAL_ERROR; // Unexpected exception type
    }

    ASSERT_TRUE(exception_thrown);
    ASSERT_TRUE(thrown_status == HIPBLAS_STATUS_INVALID_ENUM);

    // Test hipblasLtMatrixTransformDescSetAttribute function
    hipblasLtMatrixTransformDesc_t transformDesc;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixTransformDescCreate(&transformDesc, HIP_R_32F));

    // Test SCALE_TYPE attribute
    hipDataType scaleDatatype{HIP_R_16F};
    ASSERT_TRUE(hipblasLtMatrixTransformDescSetAttribute(transformDesc,
                                                         HIPBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE,
                                                         &scaleDatatype,
                                                         sizeof(scaleDatatype))
                == HIPBLAS_STATUS_SUCCESS);

    // Test POINTER_MODE attribute
    hipblasLtPointerMode_t pointerMode{HIPBLASLT_POINTER_MODE_DEVICE};
    ASSERT_TRUE(
        hipblasLtMatrixTransformDescSetAttribute(transformDesc,
                                                 HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
                                                 &pointerMode,
                                                 sizeof(pointerMode))
        == HIPBLAS_STATUS_SUCCESS);

    // Test TRANSA attribute
    hipblasOperation_t transA{HIPBLAS_OP_T};
    ASSERT_TRUE(hipblasLtMatrixTransformDescSetAttribute(
                    transformDesc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &transA, sizeof(transA))
                == HIPBLAS_STATUS_SUCCESS);

    // Test TRANSB attribute
    hipblasOperation_t transB{HIPBLAS_OP_C};
    ASSERT_TRUE(hipblasLtMatrixTransformDescSetAttribute(
                    transformDesc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &transB, sizeof(transB))
                == HIPBLAS_STATUS_SUCCESS);

    // Test invalid buffer (nullptr)
    ASSERT_TRUE(
        hipblasLtMatrixTransformDescSetAttribute(
            transformDesc, HIPBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE, nullptr, sizeof(hipDataType))
        == HIPBLAS_STATUS_INVALID_VALUE);

    // Test invalid size
    ASSERT_TRUE(hipblasLtMatrixTransformDescSetAttribute(transformDesc,
                                                         HIPBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE,
                                                         &scaleDatatype,
                                                         sizeof(int16_t))
                == HIPBLAS_STATUS_INVALID_VALUE);

    // Test SCALE_TYPE attribute get
    int32_t retrievedScaleType;
    size_t  sizeWritten;
    ASSERT_TRUE(hipblasLtMatrixTransformDescGetAttribute(transformDesc,
                                                         HIPBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE,
                                                         &retrievedScaleType,
                                                         sizeof(retrievedScaleType),
                                                         &sizeWritten)
                == HIPBLAS_STATUS_SUCCESS);
    ASSERT_TRUE(static_cast<hipDataType>(retrievedScaleType) == HIP_R_16F);
    ASSERT_TRUE(sizeWritten == sizeof(int32_t));

    // Test POINTER_MODE attribute get
    int32_t retrievedPointerMode;
    ASSERT_TRUE(
        hipblasLtMatrixTransformDescGetAttribute(transformDesc,
                                                 HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
                                                 &retrievedPointerMode,
                                                 sizeof(retrievedPointerMode),
                                                 &sizeWritten)
        == HIPBLAS_STATUS_SUCCESS);
    ASSERT_TRUE(static_cast<hipblasLtPointerMode_t>(retrievedPointerMode)
                == HIPBLASLT_POINTER_MODE_DEVICE);
    ASSERT_TRUE(sizeWritten == sizeof(int32_t));

    // Test TRANSA attribute get
    int32_t retrievedTransA;
    ASSERT_TRUE(hipblasLtMatrixTransformDescGetAttribute(transformDesc,
                                                         HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA,
                                                         &retrievedTransA,
                                                         sizeof(retrievedTransA),
                                                         &sizeWritten)
                == HIPBLAS_STATUS_SUCCESS);
    ASSERT_TRUE(static_cast<hipblasOperation_t>(retrievedTransA) == HIPBLAS_OP_T);
    ASSERT_TRUE(sizeWritten == sizeof(int32_t));

    // Test TRANSB attribute get
    int32_t retrievedTransB;
    ASSERT_TRUE(hipblasLtMatrixTransformDescGetAttribute(transformDesc,
                                                         HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB,
                                                         &retrievedTransB,
                                                         sizeof(retrievedTransB),
                                                         &sizeWritten)
                == HIPBLAS_STATUS_SUCCESS);
    ASSERT_TRUE(static_cast<hipblasOperation_t>(retrievedTransB) == HIPBLAS_OP_C);
    ASSERT_TRUE(sizeWritten == sizeof(int32_t));

    // Test error cases: both sizeInBytes and sizeWritten are 0/nullptr
    ASSERT_TRUE(hipblasLtMatrixTransformDescGetAttribute(transformDesc,
                                                         HIPBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE,
                                                         &retrievedScaleType,
                                                         0,
                                                         nullptr)
                == HIPBLAS_STATUS_INVALID_VALUE);

    // Test error case: sizeInBytes provided but sizeWritten is nullptr
    ASSERT_TRUE(hipblasLtMatrixTransformDescGetAttribute(transformDesc,
                                                         HIPBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE,
                                                         &retrievedScaleType,
                                                         sizeof(retrievedScaleType),
                                                         nullptr)
                == HIPBLAS_STATUS_INVALID_VALUE);

    // Test error case: invalid size (not sizeof(int32_t))
    ASSERT_TRUE(hipblasLtMatrixTransformDescGetAttribute(transformDesc,
                                                         HIPBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE,
                                                         &retrievedScaleType,
                                                         sizeof(int16_t),
                                                         &sizeWritten)
                == HIPBLAS_STATUS_INVALID_VALUE);

    // Test error case: invalid attribute (using default case)
    ASSERT_TRUE(hipblasLtMatrixTransformDescGetAttribute(
                    transformDesc,
                    static_cast<hipblasLtMatrixTransformDescAttributes_t>(999),
                    &retrievedScaleType,
                    sizeof(retrievedScaleType),
                    &sizeWritten)
                == HIPBLAS_STATUS_INVALID_VALUE);

    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixTransformDescDestroy(transformDesc));
}

void testing_aux_tensile_host_func(const Arguments& arg)
{
    // Test hipDataType_to_tensile_type function
    ASSERT_TRUE(hipDataType_to_tensile_type(HIP_R_16F) == rocisa::DataType::Half);
    ASSERT_TRUE(hipDataType_to_tensile_type(HIP_R_32F) == rocisa::DataType::Float);
    ASSERT_TRUE(hipDataType_to_tensile_type(HIP_R_64F) == rocisa::DataType::Double);
    ASSERT_TRUE(hipDataType_to_tensile_type(HIP_R_16BF) == rocisa::DataType::BFloat16);
    ASSERT_TRUE(hipDataType_to_tensile_type(HIP_R_8F_E4M3_FNUZ) == rocisa::DataType::Float8_fnuz);
    ASSERT_TRUE(hipDataType_to_tensile_type(HIP_R_8F_E5M2_FNUZ) == rocisa::DataType::BFloat8_fnuz);
    ASSERT_TRUE(hipDataType_to_tensile_type(HIP_R_8F_E4M3) == rocisa::DataType::Float8);
    ASSERT_TRUE(hipDataType_to_tensile_type(HIP_R_8F_E5M2) == rocisa::DataType::BFloat8);
    ASSERT_TRUE(hipDataType_to_tensile_type(HIP_R_8I) == rocisa::DataType::Int8);
    ASSERT_TRUE(hipDataType_to_tensile_type(HIP_R_32I) == rocisa::DataType::Int32);

    // Test rocComputeType_to_tensile_type function
    ASSERT_TRUE(rocComputeType_to_tensile_type(rocblaslt_compute_f32_fast_xf32)
                == rocisa::DataType::XFloat32);
    ASSERT_TRUE(rocComputeType_to_tensile_type(rocblaslt_compute_f32_fast_f16)
                == rocisa::DataType::Half);
    ASSERT_TRUE(rocComputeType_to_tensile_type(rocblaslt_compute_f32_fast_bf16)
                == rocisa::DataType::BFloat16);
    ASSERT_TRUE(rocComputeType_to_tensile_type(rocblaslt_compute_f16) == rocisa::DataType::Float);
    ASSERT_TRUE(rocComputeType_to_tensile_type(rocblaslt_compute_f32) == rocisa::DataType::Float);
    ASSERT_TRUE(rocComputeType_to_tensile_type(rocblaslt_compute_f32_fast_f8_fnuz)
                == rocisa::DataType::Float);
    ASSERT_TRUE(rocComputeType_to_tensile_type(rocblaslt_compute_f32_fast_bf8_fnuz)
                == rocisa::DataType::Float);
    ASSERT_TRUE(rocComputeType_to_tensile_type(rocblaslt_compute_f32_fast_f8bf8_fnuz)
                == rocisa::DataType::Float);
    ASSERT_TRUE(rocComputeType_to_tensile_type(rocblaslt_compute_f32_fast_bf8f8_fnuz)
                == rocisa::DataType::Float);
    ASSERT_TRUE(rocComputeType_to_tensile_type(rocblaslt_compute_f32_fast_f8)
                == rocisa::DataType::Float);
    ASSERT_TRUE(rocComputeType_to_tensile_type(rocblaslt_compute_f32_fast_bf8)
                == rocisa::DataType::Float);
    ASSERT_TRUE(rocComputeType_to_tensile_type(rocblaslt_compute_f32_fast_f8bf8)
                == rocisa::DataType::Float);
    ASSERT_TRUE(rocComputeType_to_tensile_type(rocblaslt_compute_f32_fast_bf8f8)
                == rocisa::DataType::Float);
    ASSERT_TRUE(rocComputeType_to_tensile_type(rocblaslt_compute_f64) == rocisa::DataType::Double);
    ASSERT_TRUE(rocComputeType_to_tensile_type(rocblaslt_compute_i32) == rocisa::DataType::Int32);
}

void testing_aux_tuple_helper_equal_func(const Arguments& arg)
{
    // Test tuple_helper equal functionality

    // Test basic integer equality
    auto tuple1 = std::make_tuple("key1", 42, "key2", 100);
    auto tuple2 = std::make_tuple("key1", 42, "key2", 100);
    auto tuple3 = std::make_tuple("key1", 42, "key2", 99);

    tuple_helper::equal_t<decltype(tuple1)> equal_checker;

    // Test equal tuples
    ASSERT_TRUE(equal_checker(tuple1, tuple2));

    // Test unequal tuples (different values)
    ASSERT_FALSE(equal_checker(tuple1, tuple3));

    // Test C-string equality
    auto str_tuple1 = std::make_tuple("name", "test", "value", "data");
    auto str_tuple2 = std::make_tuple("name", "test", "value", "data");
    auto str_tuple3 = std::make_tuple("name", "test", "value", "different");

    tuple_helper::equal_t<decltype(str_tuple1)> str_equal_checker;

    // Test equal C-string tuples
    ASSERT_TRUE(str_equal_checker(str_tuple1, str_tuple2));

    // Test unequal C-string tuples
    ASSERT_FALSE(str_equal_checker(str_tuple1, str_tuple3));
}

void testing_aux_rocblaslt_rocroller_host_func(const Arguments& arg)
{
    hipblasLtHandle_t handle;
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));

    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, arg.compute_type, arg.scale_type));

    hipblasOperation_t opA     = HIPBLAS_OP_T;
    hipblasOperation_t opB     = HIPBLAS_OP_N;
    int64_t            m       = 1;
    int64_t            n       = 1;
    int64_t            k       = 1;
    float              alpha_v = arg.alpha;
    void*              alpha   = &alpha_v;
    float              beta_v  = arg.beta;
    void*              beta    = &beta_v;
    int64_t            lda     = 1;
    int64_t            ldb     = 1;
    int64_t            ldc     = 1;
    int64_t            ldd     = 1;
    int64_t            lde     = 1;

    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));
    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, arg.a_type, m, k, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, arg.a_type, k, n, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, arg.a_type, m, n, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, arg.a_type, m, n, m));
    hipDataType a_type, b_type, c_type, d_type;

    const int                        request_solutions = 1;
    hipblasLtMatmulHeuristicResult_t heuristicResult[request_solutions];
    int                              returnedAlgoCount  = 0;
    size_t                           max_workspace_size = 128 * 1024 * 1024;
    hipblasLtMatmulPreference_t      pref;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatmulPreferenceSetAttribute(pref,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &max_workspace_size,
                                              sizeof(max_workspace_size)));

    void* d_workspace;
    if(max_workspace_size > 0)
        CHECK_HIP_ERROR(hipMalloc(&d_workspace, max_workspace_size));

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                          matmul,
                                                          matA,
                                                          matB,
                                                          matC,
                                                          matD,
                                                          pref,
                                                          request_solutions,
                                                          heuristicResult,
                                                          &returnedAlgoCount));

    CHECK_SOLUTION_FOUND(returnedAlgoCount);
    rocblaslt_matmul_desc matmul_descr = (rocblaslt_matmul_desc)matmul;
    int64_t               batch_count  = 1;
    void *                A, *B, *C, *D, *E, *d_alphaVec; // device
    CHECK_HIP_ERROR(hipMalloc(&A, m * k * batch_count * sizeof(int64_t)));
    CHECK_HIP_ERROR(hipMalloc(&B, n * k * batch_count * sizeof(int64_t)));
    CHECK_HIP_ERROR(hipMalloc(&C, m * n * batch_count * sizeof(int64_t)));
    CHECK_HIP_ERROR(hipMalloc(&D, m * n * batch_count * sizeof(int64_t)));
    CHECK_HIP_ERROR(hipMalloc(&E, m * n * batch_count * sizeof(int64_t)));

    int64_t                batch_stride_a = 0;
    int64_t                batch_stride_b = 0;
    int64_t                batch_stride_c = 0;
    int64_t                batch_stride_d = 0;
    int64_t                batch_stride_e = 0;
    bool                   strided_batch  = true;
    bool                   grouped_gemm   = true;
    void*                  bias           = nullptr;
    hipDataType            bias_type;
    void*                  scaleAlphaVec = nullptr;
    hipDataType            aux_type;
    bool                   gradient = false;
    rocblaslt_compute_type compute_type;
    rocblaslt_handle       roc_handle = (rocblaslt_handle)handle;

    rocblaslt_status isValid = rocblaslt_matmul_valid_args(matmul_descr,
                                                           A,
                                                           B,
                                                           C,
                                                           D,
                                                           (rocblaslt_matrix_layout)matA,
                                                           (rocblaslt_matrix_layout)matB,
                                                           (rocblaslt_matrix_layout)matC,
                                                           (rocblaslt_matrix_layout)matD,
                                                           alpha,
                                                           beta,
                                                           m,
                                                           n,
                                                           k,
                                                           a_type,
                                                           lda,
                                                           batch_stride_a,
                                                           b_type,
                                                           ldb,
                                                           batch_stride_b,
                                                           c_type,
                                                           ldc,
                                                           batch_stride_c,
                                                           d_type,
                                                           ldd,
                                                           batch_stride_d,
                                                           lde,
                                                           batch_stride_e,
                                                           bias,
                                                           bias_type,
                                                           scaleAlphaVec,
                                                           E,
                                                           aux_type,
                                                           gradient,
                                                           compute_type,
                                                           false,
                                                           false);

    ASSERT_TRUE(isValid == rocblaslt_status_continue);

    RocblasltContractionProblem problem{opA,
                                        opB,
                                        m,
                                        n,
                                        k,
                                        alpha,
                                        a_type,
                                        A, // A
                                        nullptr,
                                        lda, // arg.lda
                                        batch_stride_a,
                                        b_type,
                                        B, // B
                                        nullptr,
                                        ldb, // arg.ldb
                                        batch_stride_b,
                                        beta,
                                        c_type,
                                        C, // C
                                        nullptr,
                                        ldc, // arg.ldc
                                        batch_stride_c,
                                        d_type,
                                        D, // D
                                        nullptr,
                                        ldd, // arg.ldc
                                        batch_stride_d,
                                        E, // E
                                        nullptr,
                                        lde, //arg.lde
                                        batch_stride_e,
                                        batch_count,
                                        strided_batch,
                                        grouped_gemm,
                                        gradient,
                                        matmul_descr->compute_type,
                                        HIPBLASLT_DATATYPE_INVALID,
                                        bias,
                                        matmul_descr->scaleA,
                                        matmul_descr->scaleB,
                                        matmul_descr->scaleC,
                                        matmul_descr->scaleD,
                                        matmul_descr->scaleE,
                                        scaleAlphaVec,
                                        matmul_descr->scaleAType,
                                        matmul_descr->scaleBType,
                                        arg.bias_type,
                                        arg.aux_type,
                                        matmul_descr->epilogue,
                                        matmul_descr->amaxD,
                                        nullptr,
                                        max_workspace_size, // workspaceSize
                                        HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT, // act0
                                        HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT, // act1
                                        stream, // stream
                                        roc_handle->Synchronizer,
                                        arg.swizzle_a, // swizzleA
                                        arg.swizzle_b}; // swizzleB

    const hipblasLtMatmulAlgo_t* hip_algo = &heuristicResult[0].algo;
    const rocblaslt_matmul_algo* roc_algo = (const rocblaslt_matmul_algo*)hip_algo;

    ASSERT_TRUE(runRocRollerContractionProblem(roc_handle, nullptr, problem)
                != rocblaslt_status_success);
    ASSERT_TRUE(runRocRollerContractionProblem(roc_handle, roc_algo, problem)
                != rocblaslt_status_success);

    hipblasLtMatmulAlgo_t* hip_algo2 = &heuristicResult[0].algo;
    rocblaslt_matmul_algo* roc_algo2 = (rocblaslt_matmul_algo*)hip_algo;
    ASSERT_TRUE(isRocRollerSolutionSupported(roc_handle, problem, roc_algo2, &max_workspace_size)
                != rocblaslt_status_success);

    // Free GPU memory allocations
    CHECK_HIP_ERROR(hipFree(A));
    CHECK_HIP_ERROR(hipFree(B));
    CHECK_HIP_ERROR(hipFree(C));
    CHECK_HIP_ERROR(hipFree(D));
    CHECK_HIP_ERROR(hipFree(E));

    // Free workspace
    CHECK_HIP_ERROR(hipFree(d_workspace));

    // Destroy preference object
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));

    // Destroy handles
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
    CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
}
#endif // CODE_COVERAGE
