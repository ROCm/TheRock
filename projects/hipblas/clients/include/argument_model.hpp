/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef _ARGUMENT_MODEL_HPP_
#define _ARGUMENT_MODEL_HPP_

#include "hipblas_arguments.hpp"
#include <algorithm>
#include <iostream>
#include <sstream>

namespace ArgumentLogging
{
    const double NA_value = -1.0; // invalid for time, GFlop, GB
}

// these aren't static as ArgumentModel is instantiated for many Arg lists
void ArgumentModel_set_log_function_name(bool f);
bool ArgumentModel_get_log_function_name();

void ArgumentModel_set_log_datatype(bool d);
bool ArgumentModel_get_log_datatype();

// ArgumentModel template has a variadic list of argument enums
template <hipblas_argument... Args>
class ArgumentModel
{
    // Whether model has a particular parameter
    // TODO: Replace with C++17 fold expression ((Args == param) || ...)
    static bool has(hipblas_argument param)
    {
        return false;
    }
    template <class T, class... Ts>
    static bool has(hipblas_argument param, T const& first, Ts const&... rest)
    {
        if(param == first)
            return true;
        return has(param, rest...);
    }

public:
    void log_perf(std::stringstream& name_line,
                  std::stringstream& val_line,
                  const Arguments&   arg,
                  double             gpu_us,
                  double             gflops,
                  double             gbytes,
                  double             norm1,
                  double             norm2)
    {
        bool has_batch_count = has(e_batch_count, Args...);
        int  batch_count     = has_batch_count ? arg.batch_count : 1;
        int  hot_calls       = arg.iters < 1 ? 1 : arg.iters;

        // per/us to per/sec *10^6
        double hipblas_gflops = gflops * batch_count * hot_calls / gpu_us * 1e6;
        double hipblas_GBps   = gbytes * batch_count * hot_calls / gpu_us * 1e6;

        // append performance fields
        if(name_line.rdbuf()->in_avail())
            name_line << ",";
        name_line << "hipblas-Gflops,hipblas-GB/s,hipblas-us,";
        if(val_line.rdbuf()->in_avail())
            val_line << ",";
        val_line << hipblas_gflops << ", " << hipblas_GBps << ", " << gpu_us / hot_calls << ", ";

        if(arg.unit_check || arg.norm_check)
        {
            if(arg.norm_check)
            {
                name_line << "norm_error_host_ptr,norm_error_device_ptr,";
                val_line << norm1 << ", " << norm2 << ", ";
            }
        }
    }

    template <typename T>
    void log_args(std::ostream&    str,
                  const Arguments& arg,
                  double           gpu_us,
                  double           gflops,
                  double           gpu_bytes = 0,
                  double           norm1     = 0,
                  double           norm2     = 0)
    {
        if(arg.iters < 1)
            return; // warmup test only

        std::stringstream name_list;
        std::stringstream value_list;

        if(ArgumentModel_get_log_function_name())
        {
            auto delim = ",";
            name_list << "function" << delim;
            value_list << arg.function << delim;
        }

        if(ArgumentModel_get_log_datatype())
        {
            auto delim = ",";
            name_list << "a_type" << delim;
            value_list << hip_datatype2string(arg.a_type) << delim;
            name_list << "b_type" << delim;
            value_list << hip_datatype2string(arg.b_type) << delim;
            name_list << "c_type" << delim;
            value_list << hip_datatype2string(arg.c_type) << delim;
            name_list << "d_type" << delim;
            value_list << hip_datatype2string(arg.d_type) << delim;
            name_list << "compute_type" << delim;
            value_list << hip_datatype2string(arg.compute_type) << delim;
            name_list << "compute_type_gemm" << delim;
            value_list << hipblas_computetype2string(arg.compute_type_gemm) << delim;
        }

        // Output (name, value) pairs to name_list and value_list
        auto print = [&, delim = ""](const char* name, auto&& value) mutable {
            name_list << delim << name;
            value_list << delim << value;
            delim = ",";
        };

        // Args is a parameter pack of type:   hipblas_argument...
        // The hipblas_argument enum values in Args correspond to the function arguments that
        // will be printed by hipblas_test or hipblas_bench. For example, the function:
        //
        //  hipblas_ddot(hipblas_handle handle,
        //                                 int            n,
        //                                 const double*  x,
        //                                 int            incx,
        //                                 const double*  y,
        //                                 int            incy,
        //                                 double*        result);
        // will have <Args> = <e_N, e_incx, e_incy>
        //
        // print is a lambda defined above this comment block
        //
        // arg is an instance of the Arguments struct
        //
        // apply is a templated lambda for C++17 and a templated fuctor for C++14
        //
        // For hipblas_ddot, the following template specialization of apply will be called:
        // apply<e_N>(print, arg, T{}), apply<e_incx>(print, arg, T{}),, apply<e_incy>(print, arg, T{})
        //
        // apply in turn calls print with a string corresponding to the enum, for example "N" and the value of N
        //

#if __cplusplus >= 201703L
        // C++17
        (ArgumentsHelper::apply<Args>(print, arg, T{}), ...);
#else
        // C++14. TODO: Remove when C++17 is used
        (void)(int[]){(ArgumentsHelper::apply<Args>{}()(print, arg, T{}), 0)...};
#endif

        if(arg.timing)
            log_perf(name_list, value_list, arg, gpu_us, gflops, gpu_bytes, norm1, norm2);

        str << name_list.str() << "\n" << value_list.str() << std::endl;
    }

    void test_name(const Arguments& arg, std::string& name)
    {
        std::stringstream name_list;

        auto sep = "_";
        name_list << sep << arg.function;

        // Output (name, value) pairs to name_list and value_list
        auto print = [&](const char* name, auto&& value) mutable {
            if(std::string(arg.function).find("bad_arg") == std::string::npos
               || std::string(name).find("_type") != std::string::npos)
                name_list << sep << name << sep << value;
        };

#if __cplusplus >= 201703L
        // C++17
        (ArgumentsHelper::apply<Args>(print, arg, float{}), ...);
#else
        // C++14. TODO: Remove when C++17 is used
        (void)(int[]){(ArgumentsHelper::apply<Args>{}()(print, arg, float{}), 0)...};
#endif

        std::string params = name_list.str();
        std::replace(params.begin(), params.end(), '-', 'n');
        std::replace(params.begin(), params.end(), '.', 'p');
        name += params;
        name += (arg.api == hipblas_client_api::FORTRAN      ? "_F"
                 : arg.api == hipblas_client_api::C          ? "_C"
                 : arg.api == hipblas_client_api::FORTRAN_64 ? "_F64"
                                                             : "_C64");
    }
};

#endif
