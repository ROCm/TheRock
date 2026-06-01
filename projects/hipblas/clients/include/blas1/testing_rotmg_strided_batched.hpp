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

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

using hipblasRotmgStridedBatchedModel = ArgumentModel<e_a_type, e_stride_scale, e_batch_count>;

inline void testname_rotmg_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasRotmgStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_rotmg_strided_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasRotmgStridedBatchedFn
        = FORTRAN ? hipblasRotmgStridedBatched<T, true> : hipblasRotmgStridedBatched<T, false>;
    auto hipblasRotmgStridedBatchedFn_64 = arg.api == FORTRAN_64
                                               ? hipblasRotmgStridedBatched_64<T, true>
                                               : hipblasRotmgStridedBatched_64<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t       batch_count  = 2;
    hipblasStride stride_d1    = 5;
    hipblasStride stride_d2    = 5;
    hipblasStride stride_x1    = 5;
    hipblasStride stride_y1    = 5;
    hipblasStride stride_param = 10;

    device_strided_batch_vector<T> d1(1, 1, stride_d1, batch_count);
    device_strided_batch_vector<T> d2(1, 1, stride_d2, batch_count);
    device_strided_batch_vector<T> x1(1, 1, stride_x1, batch_count);
    device_strided_batch_vector<T> y1(1, 1, stride_y1, batch_count);
    device_strided_batch_vector<T> param(1, 1, stride_param, batch_count);

    DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                hipblasRotmgStridedBatchedFn,
                (nullptr,
                 d1,
                 stride_d1,
                 d2,
                 stride_d2,
                 x1,
                 stride_x1,
                 y1,
                 stride_y1,
                 param,
                 stride_param,
                 batch_count));
    DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                hipblasRotmgStridedBatchedFn,
                (handle,
                 nullptr,
                 stride_d1,
                 d2,
                 stride_d2,
                 x1,
                 stride_x1,
                 y1,
                 stride_y1,
                 param,
                 stride_param,
                 batch_count));
    DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                hipblasRotmgStridedBatchedFn,
                (handle,
                 d1,
                 stride_d1,
                 nullptr,
                 stride_d2,
                 x1,
                 stride_x1,
                 y1,
                 stride_y1,
                 param,
                 stride_param,
                 batch_count));
    DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                hipblasRotmgStridedBatchedFn,
                (handle,
                 d1,
                 stride_d1,
                 d2,
                 stride_d2,
                 nullptr,
                 stride_x1,
                 y1,
                 stride_y1,
                 param,
                 stride_param,
                 batch_count));
    DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                hipblasRotmgStridedBatchedFn,
                (handle,
                 d1,
                 stride_d1,
                 d2,
                 stride_d2,
                 x1,
                 stride_x1,
                 nullptr,
                 stride_y1,
                 param,
                 stride_param,
                 batch_count));
    DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                hipblasRotmgStridedBatchedFn,
                (handle,
                 d1,
                 stride_d1,
                 d2,
                 stride_d2,
                 x1,
                 stride_x1,
                 y1,
                 stride_y1,
                 nullptr,
                 stride_param,
                 batch_count));
}

template <typename T>
void testing_rotmg_strided_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasRotmgStridedBatchedFn
        = FORTRAN ? hipblasRotmgStridedBatched<T, true> : hipblasRotmgStridedBatched<T, false>;
    auto hipblasRotmgStridedBatchedFn_64 = arg.api == FORTRAN_64
                                               ? hipblasRotmgStridedBatched_64<T, true>
                                               : hipblasRotmgStridedBatched_64<T, false>;

    int64_t       batch_count  = arg.batch_count;
    double        stride_scale = arg.stride_scale;
    hipblasStride stride_d1    = stride_scale;
    hipblasStride stride_d2    = stride_scale;
    hipblasStride stride_x1    = stride_scale;
    hipblasStride stride_y1    = stride_scale;
    hipblasStride stride_param = 5 * stride_scale;

    const T rel_error = std::numeric_limits<T>::epsilon() * 1000;

    // check to prevent undefined memory allocation error
    if(batch_count <= 0)
        return;

    double gpu_time_used{0}, hipblas_error_host{0}, hipblas_error_device{0};

    hipblasLocalHandle handle(arg);

    // Initial Data on CPU
    // host data for hipBLAS host test
    host_strided_batch_vector<T> hd1(1, 1, stride_d1, batch_count);
    host_strided_batch_vector<T> hd2(1, 1, stride_d2, batch_count);
    host_strided_batch_vector<T> hx1(1, 1, stride_x1, batch_count);
    host_strided_batch_vector<T> hy1(1, 1, stride_y1, batch_count);
    host_strided_batch_vector<T> hparams(5, 1, stride_param, batch_count);

    // host data for hipBLAS device test
    host_strided_batch_vector<T> hd1_d(1, 1, stride_d1, batch_count);
    host_strided_batch_vector<T> hd2_d(1, 1, stride_d2, batch_count);
    host_strided_batch_vector<T> hx1_d(1, 1, stride_x1, batch_count);
    host_strided_batch_vector<T> hy1_d(1, 1, stride_y1, batch_count);
    host_strided_batch_vector<T> hparams_d(5, 1, stride_param, batch_count);

    // host data for CBLAS test
    host_strided_batch_vector<T> cd1(1, 1, stride_d1, batch_count);
    host_strided_batch_vector<T> cd2(1, 1, stride_d2, batch_count);
    host_strided_batch_vector<T> cx1(1, 1, stride_x1, batch_count);
    host_strided_batch_vector<T> cy1(1, 1, stride_y1, batch_count);
    host_strided_batch_vector<T> cparams(5, 1, stride_param, batch_count);

    hipblas_init_vector(hparams, arg, hipblas_client_alpha_sets_nan, true);
    hipblas_init_vector(hd1, arg, hipblas_client_alpha_sets_nan, false);
    hipblas_init_vector(hd2, arg, hipblas_client_alpha_sets_nan, false);
    hipblas_init_vector(hx1, arg, hipblas_client_alpha_sets_nan, false);
    hipblas_init_vector(hy1, arg, hipblas_client_alpha_sets_nan, false);

    cd1.copy_from(hd1);
    cd2.copy_from(hd2);
    cx1.copy_from(hx1);
    cy1.copy_from(hy1);
    cparams.copy_from(hparams);

    // device data for hipBLAS device test
    device_strided_batch_vector<T> dd1(1, 1, stride_d1, batch_count);
    device_strided_batch_vector<T> dd2(1, 1, stride_d2, batch_count);
    device_strided_batch_vector<T> dx1(1, 1, stride_x1, batch_count);
    device_strided_batch_vector<T> dy1(1, 1, stride_y1, batch_count);
    device_strided_batch_vector<T> dparams(5, 1, stride_param, batch_count);

    CHECK_DEVICE_ALLOCATION(dd1.memcheck());
    CHECK_DEVICE_ALLOCATION(dd2.memcheck());
    CHECK_DEVICE_ALLOCATION(dx1.memcheck());
    CHECK_DEVICE_ALLOCATION(dy1.memcheck());
    CHECK_DEVICE_ALLOCATION(dparams.memcheck());

    CHECK_HIP_ERROR(dd1.transfer_from(hd1));
    CHECK_HIP_ERROR(dd2.transfer_from(hd2));
    CHECK_HIP_ERROR(dx1.transfer_from(hx1));
    CHECK_HIP_ERROR(dy1.transfer_from(hy1));
    CHECK_HIP_ERROR(dparams.transfer_from(hparams));

    if(arg.unit_check || arg.norm_check)
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasRotmgStridedBatchedFn,
                   (handle,
                    hd1,
                    stride_d1,
                    hd2,
                    stride_d2,
                    hx1,
                    stride_x1,
                    hy1,
                    stride_y1,
                    hparams,
                    stride_param,
                    batch_count));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasRotmgStridedBatchedFn,
                   (handle,
                    dd1,
                    stride_d1,
                    dd2,
                    stride_d2,
                    dx1,
                    stride_x1,
                    dy1,
                    stride_y1,
                    dparams,
                    stride_param,
                    batch_count));

        CHECK_HIP_ERROR(hd1_d.transfer_from(dd1));
        CHECK_HIP_ERROR(hd2_d.transfer_from(dd2));
        CHECK_HIP_ERROR(hx1_d.transfer_from(dx1));
        CHECK_HIP_ERROR(hy1_d.transfer_from(dy1));
        CHECK_HIP_ERROR(hparams_d.transfer_from(dparams));

        for(int64_t b = 0; b < batch_count; b++)
        {
            ref_rotmg<T>(cd1[b], cd2[b], cx1[b], cy1[b], cparams[b]);
        }

        if(arg.unit_check)
        {
            near_check_general<T>(1, 1, batch_count, 1, stride_d1, cd1, hd1, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, stride_d2, cd2, hd2, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, stride_x1, cx1, hx1, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, stride_y1, cy1, hy1, rel_error);
            near_check_general<T>(1, 5, batch_count, 1, stride_param, cparams, hparams, rel_error);

            near_check_general<T>(1, 1, batch_count, 1, stride_d1, cd1, hd1_d, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, stride_d2, cd2, hd2_d, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, stride_x1, cx1, hx1_d, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, stride_y1, cy1, hy1_d, rel_error);
            near_check_general<T>(
                1, 5, batch_count, 1, stride_param, cparams, hparams_d, rel_error);
        }

        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', 1, 1, 1, stride_d1, cd1, hd1, batch_count);
            hipblas_error_host
                += norm_check_general<T>('F', 1, 1, 1, stride_d2, cd2, hd2, batch_count);
            hipblas_error_host
                += norm_check_general<T>('F', 1, 1, 1, stride_x1, cx1, hx1, batch_count);
            hipblas_error_host
                += norm_check_general<T>('F', 1, 1, 1, stride_y1, cy1, hy1, batch_count);
            hipblas_error_host
                += norm_check_general<T>('F', 1, 5, 1, stride_param, cparams, hparams, batch_count);

            hipblas_error_device
                = norm_check_general<T>('F', 1, 1, 1, stride_d1, cd1, hd1_d, batch_count);
            hipblas_error_device
                += norm_check_general<T>('F', 1, 1, 1, stride_d2, cd2, hd2_d, batch_count);
            hipblas_error_device
                += norm_check_general<T>('F', 1, 1, 1, stride_x1, cx1, hx1_d, batch_count);
            hipblas_error_device
                += norm_check_general<T>('F', 1, 1, 1, stride_y1, cy1, hy1_d, batch_count);
            hipblas_error_device += norm_check_general<T>(
                'F', 1, 5, 1, stride_param, cparams, hparams_d, batch_count);
        }
    }

    if(arg.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_CHECK(hipblasRotmgStridedBatchedFn,
                       (handle,
                        dd1,
                        stride_d1,
                        dd2,
                        stride_d2,
                        dx1,
                        stride_x1,
                        dy1,
                        stride_y1,
                        dparams,
                        stride_param,
                        batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasRotmgStridedBatchedModel{}.log_args<T>(std::cout,
                                                      arg,
                                                      gpu_time_used,
                                                      ArgumentLogging::NA_value,
                                                      ArgumentLogging::NA_value,
                                                      hipblas_error_host,
                                                      hipblas_error_device);
    }
}
