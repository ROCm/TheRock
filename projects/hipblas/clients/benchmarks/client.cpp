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

#include "program_options.hpp"

#include "hipblas.hpp"

#include "argument_model.hpp"
#include "clients_common.hpp"
#include "hipblas_data.hpp"
#include "hipblas_datatype2string.hpp"
#include "hipblas_parse_data.hpp"
#include "hipblas_test.hpp"
#include "test_cleanup.hpp"
#include "type_dispatch.hpp"
#include "utility.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>

using namespace roc; // For emulated program_options

int hipblas_bench_datafile()
{
    int ret = 0;
    for(Arguments arg : HipBLAS_TestData())
        ret |= run_bench_test(arg, 0, 1);
    test_cleanup::cleanup();
    return ret;
}

void thread_init_device(int id, const Arguments& arg)
{
    int count;
    CHECK_HIP_ERROR(hipGetDeviceCount(&count));

    if(id < count)
        CHECK_HIP_ERROR(hipSetDevice(id));

    Arguments a(arg);
    a.cold_iters = 1;
    a.iters      = 0;
    run_bench_test(a, 0, 1);
}

void thread_run_bench(int id, const Arguments& arg)
{
    int count;
    CHECK_HIP_ERROR(hipGetDeviceCount(&count));

    if(id < count)
        CHECK_HIP_ERROR(hipSetDevice(id));

    Arguments a(arg);
    run_bench_test(a, 0, 1);
}

int run_bench_multi_gpu_test(int parallel_devices, Arguments& arg)
{
    int count;
    CHECK_HIP_ERROR(hipGetDeviceCount(&count));

    if(parallel_devices > count || parallel_devices < 1)
        return 1;

    // initialization
    auto thread_init = std::make_unique<std::thread[]>(parallel_devices);

    for(int id = 0; id < parallel_devices; ++id)
        thread_init[id] = std::thread(::thread_init_device, id, arg);

    for(int id = 0; id < parallel_devices; ++id)
        thread_init[id].join();

    // synchronzied launch of cold & hot calls
    auto thread = std::make_unique<std::thread[]>(parallel_devices);

    for(int id = 0; id < parallel_devices; ++id)
        thread[id] = std::thread(::thread_run_bench, id, arg);

    for(int id = 0; id < parallel_devices; ++id)
        thread[id].join();

    return 0;
}

// Replace --batch with --batch_count for backward compatibility
void fix_batch(int argc, char* argv[])
{
    static char b_c[] = "--batch_count";
    for(int i = 1; i < argc; ++i)
        if(!strcmp(argv[i], "--batch"))
        {
            static int once = (std::cerr << argv[0]
                                         << " warning: --batch is deprecated, and --batch_count "
                                            "should be used instead."
                                         << std::endl,
                               0);
            argv[i]         = b_c;
        }
}

int main(int argc, char* argv[])
try
{
    fix_batch(argc, argv);
    Arguments   arg;
    std::string function;
    std::string precision;
    std::string a_type;
    std::string b_type;
    std::string c_type;
    std::string d_type;
    std::string compute_type;
    std::string compute_type_gemm;
    std::string initialization;
    int         device_id;
    int         parallel_devices;
    int32_t     api     = 0;
    bool        fortran = false;

    bool datafile            = hipblas_parse_data(argc, argv);
    bool atomics_not_allowed = false;
    bool log_function_name   = false;
    bool log_datatype        = false;

    options_description desc("hipblas-bench command line options");

    // clang-format off
    desc.add_options()

        ("sizem,m",
         value<int64_t>(&arg.M)->default_value(128),
         "Specific matrix size: sizem is only applicable to BLAS-2 & BLAS-3: the number of "
         "rows or columns in matrix.")

        ("sizen,n",
         value<int64_t>(&arg.N)->default_value(128),
         "Specific matrix/vector size: BLAS-1: the length of the vector. BLAS-2 & "
         "BLAS-3: the number of rows or columns in matrix")

        ("sizek,k",
         value<int64_t>(&arg.K)->default_value(128),
         "Specific matrix size: BLAS-2: the number of sub or super-diagonals of A. BLAS-3: "
         "the number of columns in A and rows in B.")

        ("kl",
         value<int64_t>(&arg.KL)->default_value(128),
         "Specific matrix size: kl is only applicable to BLAS-2: The number of sub-diagonals "
         "of the banded matrix A.")

        ("ku",
         value<int64_t>(&arg.KU)->default_value(128),
         "Specific matrix size: ku is only applicable to BLAS-2: The number of super-diagonals "
         "of the banded matrix A.")

        ("lda",
         value<int64_t>(&arg.lda)->default_value(128),
         "Leading dimension of matrix A, is only applicable to BLAS-2 & BLAS-3.")

        ("ldb",
         value<int64_t>(&arg.ldb)->default_value(128),
         "Leading dimension of matrix B, is only applicable to BLAS-2 & BLAS-3.")

        ("ldc",
         value<int64_t>(&arg.ldc)->default_value(128),
         "Leading dimension of matrix C, is only applicable to BLAS-2 & BLAS-3.")

        ("ldd",
         value<int64_t>(&arg.ldd)->default_value(128),
         "Leading dimension of matrix D, is only applicable to BLAS-EX ")

        ("stride_a",
         value<hipblasStride>(&arg.stride_a)->default_value(128*128),
         "Specific stride of strided_batched matrix A, is only applicable to strided batched"
         "BLAS-2 and BLAS-3: second dimension * leading dimension.")

        ("stride_b",
         value<hipblasStride>(&arg.stride_b)->default_value(128*128),
         "Specific stride of strided_batched matrix B, is only applicable to strided batched"
         "BLAS-2 and BLAS-3: second dimension * leading dimension.")

        ("stride_c",
         value<hipblasStride>(&arg.stride_c)->default_value(128*128),
         "Specific stride of strided_batched matrix C, is only applicable to strided batched"
         "BLAS-2 and BLAS-3: second dimension * leading dimension.")

        ("stride_d",
         value<hipblasStride>(&arg.stride_d)->default_value(128*128),
         "Specific stride of strided_batched matrix D, is only applicable to strided batched"
         "BLAS_EX: second dimension * leading dimension.")

        ("stride_x",
         value<hipblasStride>(&arg.stride_x)->default_value(128),
         "Specific stride of strided_batched vector x, is only applicable to strided batched"
         "BLAS_2: second dimension.")

        ("stride_y",
         value<hipblasStride>(&arg.stride_y)->default_value(128),
         "Specific stride of strided_batched vector y, is only applicable to strided batched"
         "BLAS_2: leading dimension.")

        ("incx",
         value<int64_t>(&arg.incx)->default_value(1),
         "increment between values in x vector")

        ("incy",
         value<int64_t>(&arg.incy)->default_value(1),
         "increment between values in y vector")

        ("alpha",
          value<double>(&arg.alpha)->default_value(1.0), "specifies the scalar alpha")

        ("alphai",
         value<double>(&arg.alphai)->default_value(0.0), "specifies the imaginary part of the scalar alpha")

        ("beta",
         value<double>(&arg.beta)->default_value(0.0), "specifies the scalar beta")

        ("betai",
         value<double>(&arg.betai)->default_value(0.0), "specifies the imaginary part of the scalar beta")

        ("function,f",
         value<std::string>(&function),
         "BLAS function to test.")

        ("precision,r",
         value<std::string>(&precision)->default_value("f32_r"), "Precision. "
         "Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("a_type",
         value<std::string>(&a_type), "Precision of matrix A. "
         "Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("b_type",
         value<std::string>(&b_type), "Precision of matrix B. "
         "Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("c_type",
         value<std::string>(&c_type), "Precision of matrix C. "
         "Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("d_type",
         value<std::string>(&d_type), "Precision of matrix D. "
         "Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("compute_type",
         value<std::string>(&compute_type), "Precision of computation. See compute_type_gemm for gemm_ex"
         "Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("compute_type_gemm",
         value<std::string>(&compute_type_gemm), "Precision of computation for gemm_ex"
         "Options: c16f,c16f_pedantic,c32f,c32f_pedantic,c32f_fast_16f,c32f_fast_16bf,c32f_fast_tf32,c64f,c64f_pedantic,c32i,c32i_pedantic")

        ("initialization",
         value<std::string>(&initialization)->default_value("hpl"),
         "Intialize with random integers, trig functions sin and cos, or hpl-like input. "
         "Options: rand_int, trig_float, hpl")

        ("transposeA",
         value<char>(&arg.transA)->default_value('N'),
         "N = no transpose, T = transpose, C = conjugate transpose")

        ("transposeB",
         value<char>(&arg.transB)->default_value('N'),
         "N = no transpose, T = transpose, C = conjugate transpose")

        ("side",
         value<char>(&arg.side)->default_value('L'),
         "L = left, R = right. Only applicable to certain routines")

        ("uplo",
         value<char>(&arg.uplo)->default_value('U'),
         "U = upper, L = lower. Only applicable to certain routines") // xsymv xsyrk xsyr2k xtrsm xtrsm_ex
                                                                     // xtrmm xtrsv
        ("diag",
         value<char>(&arg.diag)->default_value('N'),
         "U = unit diagonal, N = non unit diagonal. Only applicable to certain routines") // xtrsm xtrsm_ex xtrsv xtrmm

        ("batch_count",
         value<int64_t>(&arg.batch_count)->default_value(1),
         "Number of matrices. Only applicable to batched and strided_batched routines")

        ("inplace",
         value<bool>(&arg.inplace)->default_value(false),
         "Whether or not to use the in place version of the algorithm. Only applicable to trmm routines")

        ("verify,v",
         value<int>(&arg.norm_check)->default_value(0),
         "Validate GPU results with CPU? 0 = No, 1 = Yes (default: No)")

        ("iters,i",
         value<int>(&arg.iters)->default_value(10),
         "Iterations to run inside timing loop")

        ("cold_iters,j",
         value<int>(&arg.cold_iters)->default_value(2),
         "Cold Iterations to run before entering the timing loop")

        ("algo",
         value<uint32_t>(&arg.algo)->default_value(0),
         "extended precision gemm algorithm")

        ("solution_index",
         value<int32_t>(&arg.solution_index)->default_value(0),
         "extended precision gemm solution index")

        ("flags",
         value<uint32_t>(&arg.flags)->default_value(0),
         "gemm_ex flags")

        ("atomics_not_allowed",
         bool_switch(&atomics_not_allowed)->default_value(false),
         "Atomic operations with non-determinism in results are not allowed")

        ("device",
         value<int>(&device_id)->default_value(0),
         "Set default device to be used for subsequent program runs")

        ("parallel_devices",
         value<int>(&parallel_devices)->default_value(0),
         "Set number of devices used for parallel runs (device 0 to parallel_devices-1)")

        // ("c_noalias_d",
        //  bool_switch(&arg.c_noalias_d)->default_value(false),
        //  "C and D are stored in separate memory")

        ("log_function_name",
         bool_switch(&log_function_name)->default_value(false),
         "Function name precedes other itmes.")

        ("log_datatype",
         bool_switch(&log_datatype)->default_value(false),
         "Include datatypes used in output.")

        ("fortran",
         bool_switch(&fortran)->default_value(false),
         "Run using Fortran interface")

        ("api",
         value<int32_t>(&api)->default_value(0),
         "Use API, supercedes fortran flag (0==C, 1==C_64, ...)")

        ("workspace",
         value<size_t>(&arg.user_allocated_workspace)->default_value(0),
         "Set workspace available in handle using xxblasSetWorkspace() API after handle creation")

        ("help,h", "produces this help message");

        //("version", "Prints the version number");

    // clang-format on

    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    if((argc <= 1 && !datafile) || vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 0;
    }

    // if(vm.find("version") != vm.end())
    // {
    //     char blas_version[100];
    //     hipblas_get_version_string(blas_version, sizeof(blas_version));
    //     std::cout << "hipBLAS version: " << blas_version << std::endl;
    //     return 0;
    // }

    // transfer local variable state

    arg.atomics_mode = atomics_not_allowed ? HIPBLAS_ATOMICS_NOT_ALLOWED : HIPBLAS_ATOMICS_ALLOWED;

    if(api)
        arg.api = hipblas_client_api(api);
    else if(fortran)
        arg.api = FORTRAN;

    ArgumentModel_set_log_function_name(log_function_name);

    ArgumentModel_set_log_datatype(log_datatype);

    // Device Query
    int device_count = query_device_property();

    std::cout << std::endl;
    if(device_count <= device_id)
        throw std::invalid_argument("Invalid Device ID");
    set_device(device_id);

    if(datafile)
        return hipblas_bench_datafile();

    std::transform(precision.begin(), precision.end(), precision.begin(), ::tolower);
    auto prec = string2hip_datatype(precision);

    arg.a_type            = a_type == "" ? prec : string2hip_datatype(a_type);
    arg.b_type            = b_type == "" ? prec : string2hip_datatype(b_type);
    arg.c_type            = c_type == "" ? prec : string2hip_datatype(c_type);
    arg.d_type            = d_type == "" ? prec : string2hip_datatype(d_type);
    arg.compute_type      = compute_type == "" ? prec : string2hip_datatype(compute_type);
    arg.compute_type_gemm = string2hipblas_computetype(compute_type_gemm);

    arg.initialization = string2hipblas_initialization(initialization);
    if(arg.initialization == static_cast<hipblas_initialization>(0)) // invalid enum
        throw std::invalid_argument("Invalid value for --initialization " + initialization);

    if(arg.M < 0)
        throw std::invalid_argument("Invalid value for -m " + std::to_string(arg.M));
    if(arg.N < 0)
        throw std::invalid_argument("Invalid value for -n " + std::to_string(arg.N));
    if(arg.K < 0)
        throw std::invalid_argument("Invalid value for -k " + std::to_string(arg.K));

    int copied = snprintf(arg.function, sizeof(arg.function), "%s", function.c_str());
    if(copied <= 0 || copied >= sizeof(arg.function))
        throw std::invalid_argument("Invalid value for --function");

    if(!parallel_devices)
        return run_bench_test(arg, 0, 1);
    else
        return run_bench_multi_gpu_test(parallel_devices, arg);
}
catch(const std::invalid_argument& exp)
{
    std::cerr << exp.what() << std::endl;
    return -1;
}
