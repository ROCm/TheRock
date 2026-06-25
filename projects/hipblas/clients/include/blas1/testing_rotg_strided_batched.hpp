/* ************************************************************************
 * Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
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

using hipblasRotgStridedBatchedModel = ArgumentModel<e_a_type, e_stride_scale, e_batch_count>;

inline void testname_rotg_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasRotgStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_rotg_strided_batched_bad_arg(const Arguments& arg)
{
    using U      = real_t<T>;
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasRotgStridedBatchedFn
        = FORTRAN ? hipblasRotgStridedBatched<T, U, true> : hipblasRotgStridedBatched<T, U, false>;
    auto hipblasRotgStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasRotgStridedBatched_64<T, U, true>
                                              : hipblasRotgStridedBatched_64<T, U, false>;

    hipblasLocalHandle handle(arg);

    hipblasStride stride_a    = 10;
    hipblasStride stride_b    = 10;
    hipblasStride stride_c    = 10;
    hipblasStride stride_s    = 10;
    int64_t       batch_count = 5;

    device_strided_batch_vector<T> da(1, 1, stride_a, batch_count);
    device_strided_batch_vector<T> db(1, 1, stride_b, batch_count);
    device_strided_batch_vector<U> dc(1, 1, stride_c, batch_count);
    device_strided_batch_vector<T> ds(1, 1, stride_s, batch_count);

    DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                hipblasRotgStridedBatchedFn,
                (nullptr, da, stride_a, db, stride_b, dc, stride_c, ds, stride_s, batch_count));
    DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                hipblasRotgStridedBatchedFn,
                (handle, nullptr, stride_a, db, stride_b, dc, stride_c, ds, stride_s, batch_count));
    DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                hipblasRotgStridedBatchedFn,
                (handle, da, stride_a, nullptr, stride_b, dc, stride_c, ds, stride_s, batch_count));
    DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                hipblasRotgStridedBatchedFn,
                (handle, da, stride_a, db, stride_b, nullptr, stride_c, ds, stride_s, batch_count));
    DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                hipblasRotgStridedBatchedFn,
                (handle, da, stride_a, db, stride_b, dc, stride_c, nullptr, stride_s, batch_count));
}

template <typename T>
void testing_rotg_strided_batched(const Arguments& arg)
{
    using U      = real_t<T>;
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasRotgStridedBatchedFn
        = FORTRAN ? hipblasRotgStridedBatched<T, U, true> : hipblasRotgStridedBatched<T, U, false>;
    auto hipblasRotgStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasRotgStridedBatched_64<T, U, true>
                                              : hipblasRotgStridedBatched_64<T, U, false>;

    double        stride_scale = arg.stride_scale;
    hipblasStride stride_a     = stride_scale;
    hipblasStride stride_b     = stride_scale;
    hipblasStride stride_c     = stride_scale;
    hipblasStride stride_s     = stride_scale;
    int64_t       batch_count  = arg.batch_count;

    const U rel_error = std::numeric_limits<U>::epsilon() * 1000;

    // check to prevent undefined memory allocation error
    if(batch_count <= 0)
        return;

    double gpu_time_used{0}, hipblas_error_host{0}, hipblas_error_device{0};

    hipblasLocalHandle handle(arg);

    host_strided_batch_vector<T> ha(1, 1, stride_a, batch_count);
    host_strided_batch_vector<T> hb(1, 1, stride_b, batch_count);
    host_strided_batch_vector<U> hc(1, 1, stride_c, batch_count);
    host_strided_batch_vector<T> hs(1, 1, stride_s, batch_count);

    // CPU_BLAS
    host_strided_batch_vector<T> ca(1, 1, stride_a, batch_count);
    host_strided_batch_vector<T> cb(1, 1, stride_b, batch_count);
    host_strided_batch_vector<U> cc(1, 1, stride_c, batch_count);
    host_strided_batch_vector<T> cs(1, 1, stride_s, batch_count);

    // result vector for hipBLAS device
    host_strided_batch_vector<T> ra(1, 1, stride_a, batch_count);
    host_strided_batch_vector<T> rb(1, 1, stride_b, batch_count);
    host_strided_batch_vector<U> rc(1, 1, stride_c, batch_count);
    host_strided_batch_vector<T> rs(1, 1, stride_s, batch_count);

    // Initial data on CPU
    hipblas_init_vector(ha, arg, hipblas_client_alpha_sets_nan, true);
    hipblas_init_vector(hb, arg, hipblas_client_alpha_sets_nan, false);
    hipblas_init_vector(hc, arg, hipblas_client_alpha_sets_nan, false);
    hipblas_init_vector(hs, arg, hipblas_client_alpha_sets_nan, false);

    ca.copy_from(ha);
    cb.copy_from(hb);
    cc.copy_from(hc);
    cs.copy_from(hs);
    ra.copy_from(ha);
    rb.copy_from(hb);
    rc.copy_from(hc);
    rs.copy_from(hs);

    device_strided_batch_vector<T> da(1, 1, stride_a, batch_count);
    device_strided_batch_vector<T> db(1, 1, stride_b, batch_count);
    device_strided_batch_vector<U> dc(1, 1, stride_c, batch_count);
    device_strided_batch_vector<T> ds(1, 1, stride_s, batch_count);

    CHECK_DEVICE_ALLOCATION(da.memcheck());
    CHECK_DEVICE_ALLOCATION(db.memcheck());
    CHECK_DEVICE_ALLOCATION(dc.memcheck());
    CHECK_DEVICE_ALLOCATION(ds.memcheck());

    CHECK_HIP_ERROR(da.transfer_from(ha));
    CHECK_HIP_ERROR(db.transfer_from(hb));
    CHECK_HIP_ERROR(dc.transfer_from(hc));
    CHECK_HIP_ERROR(ds.transfer_from(hs));

    if(arg.unit_check || arg.norm_check)
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasRotgStridedBatchedFn,
                   (handle,
                    ha.internal_type(),
                    stride_a,
                    hb.internal_type(),
                    stride_b,
                    hc.internal_type(),
                    stride_c,
                    hs.internal_type(),
                    stride_s,
                    batch_count));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasRotgStridedBatchedFn,
                   (handle, da, stride_a, db, stride_b, dc, stride_c, ds, stride_s, batch_count));

        CHECK_HIP_ERROR(ra.transfer_from(da));
        CHECK_HIP_ERROR(rb.transfer_from(db));
        CHECK_HIP_ERROR(rc.transfer_from(dc));
        CHECK_HIP_ERROR(rs.transfer_from(ds));

        for(int64_t b = 0; b < batch_count; b++)
        {
            ref_rotg<T, U>(ca.data() + b * stride_a,
                           cb.data() + b * stride_b,
                           cc.data() + b * stride_c,
                           cs.data() + b * stride_s);
        }

        if(arg.unit_check)
        {
            near_check_general<T>(1, 1, batch_count, 1, stride_a, ca, ha, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, stride_b, cb, hb, rel_error);
            near_check_general<U>(1, 1, batch_count, 1, stride_c, cc, hc, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, stride_s, cs, hs, rel_error);

            near_check_general<T>(1, 1, batch_count, 1, stride_a, ca, ra, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, stride_b, cb, rb, rel_error);
            near_check_general<U>(1, 1, batch_count, 1, stride_c, cc, rc, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, stride_s, cs, rs, rel_error);
        }

        if(arg.norm_check)
        {
            hipblas_error_host = norm_check_general<T>('F', 1, 1, 1, stride_a, ca, ha, batch_count);
            hipblas_error_host
                += norm_check_general<T>('F', 1, 1, 1, stride_b, cb, hb, batch_count);
            hipblas_error_host
                += norm_check_general<U>('F', 1, 1, 1, stride_c, cc, hc, batch_count);
            hipblas_error_host
                += norm_check_general<T>('F', 1, 1, 1, stride_s, cs, hs, batch_count);

            hipblas_error_device
                = norm_check_general<T>('F', 1, 1, 1, stride_a, ca, ra, batch_count);
            hipblas_error_device
                += norm_check_general<T>('F', 1, 1, 1, stride_b, cb, rb, batch_count);
            hipblas_error_device
                += norm_check_general<U>('F', 1, 1, 1, stride_c, cc, rc, batch_count);
            hipblas_error_device
                += norm_check_general<T>('F', 1, 1, 1, stride_s, cs, rs, batch_count);
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

            DAPI_CHECK(
                hipblasRotgStridedBatchedFn,
                (handle, da, stride_a, db, stride_b, dc, stride_c, ds, stride_s, batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasRotgStridedBatchedModel{}.log_args<T>(std::cout,
                                                     arg,
                                                     gpu_time_used,
                                                     ArgumentLogging::NA_value,
                                                     ArgumentLogging::NA_value,
                                                     hipblas_error_host,
                                                     hipblas_error_device);
    }
}
