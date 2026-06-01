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

using hipblasRotgBatchedModel = ArgumentModel<e_a_type, e_batch_count>;

inline void testname_rotg_batched(const Arguments& arg, std::string& name)
{
    hipblasRotgBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_rotg_batched_bad_arg(const Arguments& arg)
{
    using U      = real_t<T>;
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasRotgBatchedFn
        = FORTRAN ? hipblasRotgBatched<T, U, true> : hipblasRotgBatched<T, U, false>;
    auto hipblasRotgBatchedFn_64 = arg.api == FORTRAN_64 ? hipblasRotgBatched_64<T, U, true>
                                                         : hipblasRotgBatched_64<T, U, false>;

    hipblasLocalHandle handle(arg);

    int64_t batch_count = 2;

    device_batch_vector<T> da(1, 1, batch_count);
    device_batch_vector<T> db(1, 1, batch_count);
    device_batch_vector<U> dc(1, 1, batch_count);
    device_batch_vector<T> ds(1, 1, batch_count);

    DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                hipblasRotgBatchedFn,
                (nullptr, da, db, dc, ds, batch_count));
    DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                hipblasRotgBatchedFn,
                (handle, nullptr, db, dc, ds, batch_count));
    DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                hipblasRotgBatchedFn,
                (handle, da, nullptr, dc, ds, batch_count));
    DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                hipblasRotgBatchedFn,
                (handle, da, db, nullptr, ds, batch_count));
    DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                hipblasRotgBatchedFn,
                (handle, da, db, dc, nullptr, batch_count));
}

template <typename T>
void testing_rotg_batched(const Arguments& arg)
{
    using U      = real_t<T>;
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasRotgBatchedFn
        = FORTRAN ? hipblasRotgBatched<T, U, true> : hipblasRotgBatched<T, U, false>;
    auto hipblasRotgBatchedFn_64 = arg.api == FORTRAN_64 ? hipblasRotgBatched_64<T, U, true>
                                                         : hipblasRotgBatched_64<T, U, false>;

    int64_t batch_count = arg.batch_count;

    const U rel_error = std::numeric_limits<U>::epsilon() * 1000;

    // check to prevent undefined memory allocation error
    if(batch_count <= 0)
        return;

    double gpu_time_used{0}, hipblas_error_host{0}, hipblas_error_device{0};

    hipblasLocalHandle handle(arg);

    // Initial Data on CPU
    // host vectors for hipblas host result
    host_batch_vector<T> ha(1, 1, batch_count);
    host_batch_vector<T> hb(1, 1, batch_count);
    host_batch_vector<U> hc(1, 1, batch_count);
    host_batch_vector<T> hs(1, 1, batch_count);

    // host vectors for cblas
    host_batch_vector<T> ca(1, 1, batch_count);
    host_batch_vector<T> cb(1, 1, batch_count);
    host_batch_vector<U> cc(1, 1, batch_count);
    host_batch_vector<T> cs(1, 1, batch_count);

    // host vectors for hipblas device result
    host_batch_vector<T> ra(1, 1, batch_count);
    host_batch_vector<T> rb(1, 1, batch_count);
    host_batch_vector<U> rc(1, 1, batch_count);
    host_batch_vector<T> rs(1, 1, batch_count);

    // device vectors for hipblas device
    device_batch_vector<T> da(1, 1, batch_count);
    device_batch_vector<T> db(1, 1, batch_count);
    device_batch_vector<U> dc(1, 1, batch_count);
    device_batch_vector<T> ds(1, 1, batch_count);

    CHECK_DEVICE_ALLOCATION(da.memcheck());
    CHECK_DEVICE_ALLOCATION(db.memcheck());
    CHECK_DEVICE_ALLOCATION(dc.memcheck());
    CHECK_DEVICE_ALLOCATION(ds.memcheck());

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

    CHECK_HIP_ERROR(da.transfer_from(ha));
    CHECK_HIP_ERROR(db.transfer_from(hb));
    CHECK_HIP_ERROR(dc.transfer_from(hc));
    CHECK_HIP_ERROR(ds.transfer_from(hs));

    if(arg.unit_check || arg.norm_check)
    {
        // hipBLAS
        // test host
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasRotgBatchedFn,
                   (handle,
                    ha.internal_type(),
                    hb.internal_type(),
                    hc.internal_type(),
                    hs.internal_type(),
                    batch_count));

        // test device
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasRotgBatchedFn,
                   (handle,
                    da.ptr_on_device(),
                    db.ptr_on_device(),
                    dc.ptr_on_device(),
                    ds.ptr_on_device(),
                    batch_count));

        CHECK_HIP_ERROR(ra.transfer_from(da));
        CHECK_HIP_ERROR(rb.transfer_from(db));
        CHECK_HIP_ERROR(rc.transfer_from(dc));
        CHECK_HIP_ERROR(rs.transfer_from(ds));

        // CBLAS
        for(int64_t b = 0; b < batch_count; b++)
        {
            ref_rotg<T, U>(ca[b], cb[b], cc[b], cs[b]);
        }

        if(arg.unit_check)
        {
            for(int64_t b = 0; b < batch_count; b++)
            {
                near_check_general<T>(1, 1, 1, ca[b], ha[b], rel_error);
                near_check_general<T>(1, 1, 1, cb[b], hb[b], rel_error);
                near_check_general<U>(1, 1, 1, cc[b], hc[b], rel_error);
                near_check_general<T>(1, 1, 1, cs[b], hs[b], rel_error);

                near_check_general<T>(1, 1, 1, ca[b], ra[b], rel_error);
                near_check_general<T>(1, 1, 1, cb[b], rb[b], rel_error);
                near_check_general<U>(1, 1, 1, cc[b], rc[b], rel_error);
                near_check_general<T>(1, 1, 1, cs[b], rs[b], rel_error);
            }
        }

        if(arg.norm_check)
        {
            hipblas_error_host = norm_check_general<T>('F', 1, 1, 1, ca, ha, batch_count);
            hipblas_error_host += norm_check_general<T>('F', 1, 1, 1, cb, hb, batch_count);
            hipblas_error_host += norm_check_general<U>('F', 1, 1, 1, cc, hc, batch_count);
            hipblas_error_host += norm_check_general<T>('F', 1, 1, 1, cs, hs, batch_count);

            hipblas_error_device = norm_check_general<T>('F', 1, 1, 1, ca, ra, batch_count);
            hipblas_error_device += norm_check_general<T>('F', 1, 1, 1, cb, rb, batch_count);
            hipblas_error_device += norm_check_general<U>('F', 1, 1, 1, cc, rc, batch_count);
            hipblas_error_device += norm_check_general<T>('F', 1, 1, 1, cs, rs, batch_count);
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

            DAPI_CHECK(hipblasRotgBatchedFn,
                       (handle,
                        da.ptr_on_device(),
                        db.ptr_on_device(),
                        dc.ptr_on_device(),
                        ds.ptr_on_device(),
                        batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasRotgBatchedModel{}.log_args<T>(std::cout,
                                              arg,
                                              gpu_time_used,
                                              ArgumentLogging::NA_value,
                                              ArgumentLogging::NA_value,
                                              hipblas_error_host,
                                              hipblas_error_device);
    }
}
