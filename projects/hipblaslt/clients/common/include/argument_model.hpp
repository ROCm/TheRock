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

#include "efficiency_monitor.hpp"
#include "hipblaslt_arguments.hpp"
#include <fstream>
#include <string>

namespace ArgumentLogging
{
    const double NA_value = -1.0; // invalid for time, GFlop, GB
}

void ArgumentModel_set_log_function_name(bool f);
bool ArgumentModel_get_log_function_name();

void ArgumentModel_log_performance(hipblaslt_internal_ostream& name_line,
                                   hipblaslt_internal_ostream& val_line);

void ArgumentModel_log_efficiency(hipblaslt_internal_ostream& name_line,
                                  hipblaslt_internal_ostream& val_line,
                                  const Arguments&            arg,
                                  double                      hipblaslt_gflops);

// ArgumentModel template has a variadic list of argument enums
template <hipblaslt_argument... Args>
class ArgumentModel
{
    // Whether model has a particular parameter
    // TODO: Replace with C++17 fold expression ((Args == param) || ...)
    static constexpr bool has(hipblaslt_argument param)
    {
        for(auto x : {Args...})
            if(x == param)
                return true;
        return false;
    }

public:
    void log_perf(hipblaslt_internal_ostream& name_line,
                  hipblaslt_internal_ostream& val_line,
                  const Arguments&            arg,
                  double                      gpu_us,
                  double                      flush_us,
                  double                      gflops,
                  double                      gbytes,
                  double                      cpu_us,
                  double                      norm,
                  double                      atol,
                  double                      rtol)
    {
        // requires enablement for frequency logging
        ArgumentModel_log_performance(name_line, val_line);

        constexpr bool has_batch_count = has(e_batch_count);
        int64_t        batch_count     = has_batch_count ? arg.batch_count : 1;
        int64_t        hot_calls       = arg.iters < 1 ? 1 : arg.iters;

        // gpu time is total cumulative over hot calls, cpu is not
        if(hot_calls > 1)
            gpu_us /= hot_calls;

        if(flush_us > 0)
        {
            gpu_us -= flush_us;
        }

        // per/us to per/sec *10^6
        double hipblaslt_gflops = gflops * batch_count / gpu_us * 1e6;
        double hipblaslt_GBps   = gbytes / gpu_us * 1e6;

        // append performance fields
        if(gflops != ArgumentLogging::NA_value)
        {
            name_line << ",hipblaslt-Gflops";
            val_line << "," << hipblaslt_gflops;
            ArgumentModel_log_efficiency(name_line, val_line, arg, hipblaslt_gflops);
        }

        if(gbytes != ArgumentLogging::NA_value)
        {
            // GB/s not usually reported for non-memory bound functions
            name_line << ",hipblaslt-GB/s";
            val_line << "," << hipblaslt_GBps;
        }

        name_line << ",us";
        val_line << "," << gpu_us;

        if(arg.unit_check || arg.norm_check || arg.allclose_check)
        {
            if(cpu_us != ArgumentLogging::NA_value)
            {
                if(gflops != ArgumentLogging::NA_value)
                {
                    double cblas_gflops = gflops * batch_count / cpu_us * 1e6;
                    name_line << ",CPU-Gflops";
                    val_line << "," << cblas_gflops;
                }

                name_line << ",CPU-us";
                val_line << "," << cpu_us;
            }
            if(arg.norm_check)
            {
                if(norm != ArgumentLogging::NA_value)
                {
                    name_line << ",norm_error";
                    val_line << "," << norm;
                }
            }
            if(arg.allclose_check)
            {
                if(atol != ArgumentLogging::NA_value)
                {
                    name_line << ",atol";
                    if(atol == 1) // atol == init value
                        val_line << ","
                                 << "failed";
                    else
                        val_line << "," << atol;
                }
                if(rtol != ArgumentLogging::NA_value)
                {
                    name_line << ",rtol";
                    if(rtol == 1) // rtol == init value
                        val_line << ","
                                 << "failed";
                    else
                        val_line << "," << rtol;
                }
            }
        }
    }

    void log_args(hipDataType                 Tc,
                  hipblaslt_internal_ostream& str,
                  size_t                      index,
                  int32_t                     solution_index,
                  std::string&                solution_name,
                  std::string&                kernel_name,
                  std::string&                archName,
                  std::string&                cuNum,
                  const Arguments&            arg,
                  uint32_t                    splitK,
                  uint32_t                    wgm,
                  double                      gpu_us,
                  double                      flush_us,
                  double                      gflops,
                  double                      gbytes = ArgumentLogging::NA_value,
                  double                      cpu_us = ArgumentLogging::NA_value,
                  double                      norm   = ArgumentLogging::NA_value,
                  double                      atol   = ArgumentLogging::NA_value,
                  double                      rtol   = ArgumentLogging::NA_value)
    {
        hipblaslt_internal_ostream name_list;
        hipblaslt_internal_ostream value_list;

        name_list << "[" << index << "]:";
        value_list << "    ";

        if(ArgumentModel_get_log_function_name())
        {
            auto delim = ",";
            name_list << "function" << delim;
            value_list << arg.function << delim;
        }

        // Output (name, value) pairs to name_list and value_list
        auto print = [&, delim = ""](const char* name, auto&& value) mutable {
            name_list << delim << name;
            value_list << delim << value;
            delim = ",";
        };

#if __cplusplus >= 201703L
        // C++17
        //(ArgumentsHelper::apply<Args>(print, arg, T{}), ...);
        switch(Tc)
        {
        case HIP_R_32F:
            (ArgumentsHelper::apply<Args>(print, arg, float{}), ...);
            break;
        case HIP_R_64F:
            (ArgumentsHelper::apply<Args>(print, arg, double{}), ...);
            break;
        case HIP_R_16F:
            (ArgumentsHelper::apply<Args>(print, arg, hipblasLtHalf{}), ...);
            break;
        case HIP_R_32I:
            (ArgumentsHelper::apply<Args>(print, arg, int32_t{}), ...);
            break;
        default:
            hipblaslt_cerr << "Error type in log_args" << std::endl;
            (ArgumentsHelper::apply<Args>(print, arg, float{}), ...);
            break;
        }
#else
        // C++14. TODO: Remove when C++17 is used
        //(void)(int[]){(ArgumentsHelper::apply<Args>{}()(print, arg, T{}), 0)...};
        switch(Tc)
        {
        case HIP_R_32F:
            (void)(int[]){(ArgumentsHelper::apply<Args>{}()(print, arg, float{}), 0)...};
            break;
        case HIP_R_64F:
            (void)(int[]){(ArgumentsHelper::apply<Args>{}()(print, arg, double{}), 0)...};
            break;
        case HIP_R_16F:
            (void)(int[]){(ArgumentsHelper::apply<Args>{}()(print, arg, hipblasLtHalf{}), 0)...};
            break;
        case HIP_R_32I:
            (void)(int[]){(ArgumentsHelper::apply<Args>{}()(print, arg, int32_t{}), 0)...};
            break;
        default:
            hipblaslt_cerr << "Error type in log_args" << std::endl;
            (void)(int[]){(ArgumentsHelper::apply<Args>{}()(print, arg, float{}), 0)...};
            break;
        }
#endif

        // Additional name and value list
        if(splitK > 0)
            print("splitK", splitK);
        if(wgm > 0)
            print("wgm", wgm);

        // Printing rotating buffer even if it is set to default value
        print("rotating_buffer", arg.rotating);

        // Printing flush even if it is set to default value
        print("flush", arg.flush);

        // Printing use_gpu_timer even if it is set to default value
        print("use_gpu_timer", arg.use_gpu_timer);

        if(arg.timing)
            log_perf(name_list,
                     value_list,
                     arg,
                     gpu_us,
                     flush_us,
                     gflops,
                     gbytes,
                     cpu_us,
                     norm,
                     atol,
                     rtol);

        if(archName != "")
        {
            auto delim = ",";
            name_list << delim << "solution_index";
            value_list << delim << solution_index;

            const char*   tuningEnv  = getenv("HIPBLASLT_TUNING_FILE");
            std::string   tuningPath = tuningEnv;
            std::ofstream file(tuningPath, std::ios::app);

            std::string name_list_str  = name_list.str();
            std::string value_list_str = value_list.str();

            name_list_str.erase(0, std::string("[" + std::to_string(index) + "]:").length());

            if(ArgumentModel_get_log_function_name())
            {
                name_list_str.erase(0, std::string("function,").length());
                value_list_str.erase(0, std::string(arg.function).length() + 1);
            }

            file << "    " << name_list_str << delim << "gcnArchName" << delim << "CUs"
                 << std::endl;
            file << value_list_str << delim << archName << delim << cuNum << std::endl;
        }

        str << name_list << "\n" << value_list << std::endl;

        if(solution_name != "")
        {
            str << "    --Solution index: " << solution_index << "\n"
                << "    --Solution name:  " << solution_name << "\n"
                << "    --kernel name:    " << kernel_name << std::endl;
        }
    }
};
