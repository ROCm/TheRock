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

#include "program_options.hpp"

#include "hipblaslt_data.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_parse_data.hpp"
#include "utility.hpp"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "efficiency_monitor.hpp"

#include "testing_matmul.hpp"

using namespace roc; // For emulated program_options
using namespace std::literals; // For std::string literals of form "str"s

struct perf_matmul : hipblaslt_test_valid
{
    void operator()(const Arguments& arg)
    {
        if(strcmp(arg.function, "matmul"))
            throw std::invalid_argument("Invalid combination --function "s + arg.function
                                        + " --a_type "s + hip_datatype_to_string(arg.a_type));

        testing_matmul(arg);
    }
};

int run_bench_test(Arguments&         arg,
                   const std::string& filter,
                   bool               any_stride,
                   hipDeviceProp_t&   props,
                   bool               yaml = false)
{
    hipblaslt_cout << std::setiosflags(std::ios::fixed)
                   << std::setprecision(7); // Set precision to 7 digits

    // disable unit_check in client benchmark, it is only used in gtest unit test
    arg.unit_check = 0;

    // enable timing check,otherwise no performance data collected
    arg.timing = 1;

    // One stream and one thread (0 indicates to use default behavior)
    arg.streams = 0;
    arg.threads = 0;

    // Enable information cout
    arg.print_solution_found = true;

    // Skip past any testing_ prefix in function
    static constexpr char prefix[] = "testing_";
    const char*           function = arg.function;
    if(!strncmp(function, prefix, sizeof(prefix) - 1))
        function += sizeof(prefix) - 1;

    if(yaml && strstr(function, "_bad_arg"))
        return 0;
    if(!filter.empty())
    {
        if(!strstr(function, filter.c_str()))
            return 0;
    }

    // adjust dimension for GEMM routines
    size_t gemmNum = arg.grouped_gemm == 0 ? 1 : arg.grouped_gemm;
    for(size_t i = 0; i < gemmNum; i++)
    {
        int64_t min_lda = arg.transA == 'N' ? arg.M[i] : arg.K[i];
        int64_t min_ldb = arg.transB == 'N' ? arg.K[i] : arg.N[i];
        int64_t min_ldc = arg.M[i];
        int64_t min_ldd = arg.M[i];
        int64_t min_lde = arg.M[i];
        if(arg.lda[i] < min_lda)
        {
            //hipblaslt_cout << "hipblaslt-bench INFO: lda < min_lda, set lda = " << min_lda << std::endl;
            arg.lda[i] = min_lda;
        }
        if(arg.ldb[i] < min_ldb)
        {
            //hipblaslt_cout << "hipblaslt-bench INFO: ldb < min_ldb, set ldb = " << min_ldb << std::endl;
            arg.ldb[i] = min_ldb;
        }
        if(arg.ldc[i] < min_ldc)
        {
            //hipblaslt_cout << "hipblaslt-bench INFO: ldc < min_ldc, set ldc = " << min_ldc << std::endl;
            arg.ldc[i] = min_ldc;
        }
        if(arg.ldd[i] < min_ldd)
        {
            //hipblaslt_cout << "hipblaslt-bench INFO: ldd < min_ldd, set ldd = " << min_ldc << std::endl;
            arg.ldd[i] = min_ldd;
        }
        if(arg.lde[i] < min_lde)
        {
            //hipblaslt_cout << "hipblaslt-bench INFO: lde < min_lde, set lde = " << min_lde << std::endl;
            arg.lde[i] = min_lde;
        }
        int64_t min_stride_a = arg.lda[i] * (arg.transA == 'N' ? arg.K[i] : arg.M[i]);
        int64_t min_stride_b = arg.ldb[i] * (arg.transB == 'N' ? arg.N[i] : arg.K[i]);
        int64_t min_stride_c = arg.ldc[i] * arg.N[i];
        int64_t min_stride_d = arg.ldd[i] * arg.N[i];
        int64_t min_stride_e = arg.lde[i] * arg.N[i];
        if(!any_stride && arg.stride_a[i] < min_stride_a)
        {
            //hipblaslt_cout << "hipblaslt-bench INFO: stride_a < min_stride_a, set stride_a = "
            //               << min_stride_a << std::endl;
            arg.stride_a[i] = min_stride_a;
        }
        if(!any_stride && arg.stride_b[i] < min_stride_b)
        {
            //hipblaslt_cout << "hipblaslt-bench INFO: stride_b < min_stride_b, set stride_b = "
            //               << min_stride_b << std::endl;
            arg.stride_b[i] = min_stride_b;
        }
        if(!any_stride && arg.stride_c[i] < min_stride_c)
        {
            //hipblaslt_cout << "hipblaslt-bench INFO: stride_c < min_stride_c, set stride_c = "
            //               << min_stride_c << std::endl;
            arg.stride_c[i] = min_stride_c;
        }
        if(!any_stride && arg.stride_d[i] < min_stride_d)
        {
            //hipblaslt_cout << "hipblaslt-bench INFO: stride_d < min_stride_d, set stride_d = "
            //               << min_stride_d << std::endl;
            arg.stride_d[i] = min_stride_d;
        }
        if(!any_stride && arg.stride_e[i] < min_stride_e)
        {
            //hipblaslt_cout << "hipblaslt-bench INFO: stride_e < min_stride_e, set stride_e = "
            //               << min_stride_e << std::endl;
            arg.stride_e[i] = min_stride_e;
        }
    }

    // Check for F8 OCP data types and convert to NANOO
    {
        std::string deviceFullString(props.gcnArchName);
        std::string deviceString = deviceFullString.substr(0, deviceFullString.find(":"));

        bool isGFX942 = deviceString.find("gfx942") != std::string::npos;

        if(isGFX942)
        {
            auto convertF8Type = [](hipDataType type) {
                if(type == HIP_R_8F_E4M3)
                {
                    hipblaslt_cerr << "hipblaslt-bench INFO: Using f8_fnuz_r instead of f8_r"
                                   << std::endl;
                    return HIP_R_8F_E4M3_FNUZ;
                }
                if(type == HIP_R_8F_E5M2)
                {
                    hipblaslt_cerr << "hipblaslt-bench INFO: Using b8_fnuz_r instead of b8_r"
                                   << std::endl;
                    return HIP_R_8F_E5M2_FNUZ;
                }
                else
                    return type;
            };

            arg.a_type   = convertF8Type(arg.a_type);
            arg.b_type   = convertF8Type(arg.b_type);
            arg.c_type   = convertF8Type(arg.c_type);
            arg.d_type   = convertF8Type(arg.d_type);
            arg.aux_type = convertF8Type(arg.aux_type);
        }
    }

    perf_matmul{}(arg);
    return 0;
}

int hipblaslt_bench_datafile(const std::string& filter, bool any_stride, hipDeviceProp_t& props)
{
    int ret = 0;
    for(Arguments arg : HipBlasLt_TestData())
        ret |= run_bench_test(arg, filter, any_stride, props, true);
    test_cleanup::cleanup();
    return ret;
}

// Replace --batch with --batch_count for backward compatibility
void fix_batch(int argc, char* argv[])
{
    static char b_c[] = "--batch_count";
    for(int i = 1; i < argc; ++i)
        if(!strcmp(argv[i], "--batch"))
        {
            static int once
                = (hipblaslt_cerr << argv[0]
                                  << " warning: --batch is deprecated, and --batch_count "
                                     "should be used instead."
                                  << std::endl,
                   0);
            argv[i] = b_c;
        }
}

bool tuning_path_compare_git_version(const char* tuningEnv)
{
    char                   git_version[128];
    hipblaslt_local_handle handle;
    hipblasLtGetGitRevision(handle, &git_version[0]);
    std::string   tuningPath = tuningEnv;
    std::ifstream file_read(tuningPath);

    if(file_read.peek() == std::ifstream::traits_type::eof())
    {
        std::ofstream file_write(tuningPath, std::ios::app);
        file_write << "Git Version: " << (std::string)git_version << std::endl;
        hipblaslt_cout << "Initialize tuning file." << std::endl;
        return true;
    }
    else
    {
        std::string firstline;
        std::string prefix = "Git Version: ";
        std::getline(file_read, firstline);
        size_t pos = firstline.find(prefix);
        if(pos != std::string::npos)
        {
            std::string file_version = firstline.substr(pos + prefix.length());
            hipblaslt_cout << "tuning file git version: " << file_version << std::endl;
            if(file_version == git_version)
            {
                return true;
            }
        }
    }

    hipblaslt_cout << "The hipBLASLt git version and the tuning file git version are not the same."
                   << std::endl;
    return false;
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
    std::string compute_input_typeA;
    std::string compute_input_typeB;
    std::string scale_type;
    std::string bias_type;
    std::string bias_source;
    std::string initialization;
    std::string filter;
    std::string activation_type;
    std::string aux_type;
    int         scaleAFormat;
    int         scaleBFormat;
    int         scaleCFormat;
    int         scaleDFormat;
    int         device_id;
    int         flags             = 0;
    bool        datafile          = hipblaslt_parse_data(argc, argv);
    bool        log_function_name = false;
    bool        any_stride        = false;
    bool        dump_matrix       = false;

    int         api_method      = 0;
    std::string api_method_str  = "";
    std::string algo_method_str = "";

    bool verify = 0;

    bool                  grouped_gemm;
    std::vector<int64_t>  m, n, k;
    std::vector<int64_t>  lda, ldb, ldc, ldd, lde;
    std::vector<int64_t>  stride_a, stride_b, stride_c, stride_d, stride_e;
    std::vector<uint32_t> gsu_vector, wgm_vector;
    arg.init(); // set all defaults
    const char* tuningEnv          = getenv("HIPBLASLT_TUNING_FILE");
    const char* tuningMaxWorkSpace = getenv("HIPBLASLT_TUNING_USER_MAX_WORKSPACE");
    if(tuningEnv)
    {
        bool tuning_success = tuning_path_compare_git_version(tuningEnv);
        if(tuning_success)
        {
            hipblaslt_cout << "HIPBLASLT_TUNING_FILE is the correct setting." << std::endl;
        }
        else
            return 1;
    }

    options_description desc("hipblaslt-bench command line options");
    desc.add_options()
        // clang-format off
        ("sizem,m",
         valueVec<int64_t>(&m)->default_value(128),
         "Specific matrix size: the number of rows or columns in matrix.")

        ("sizen,n",
         valueVec<int64_t>(&n)->default_value(128),
         "Specific matrix the number of rows or columns in matrix")

        ("sizek,k",
         valueVec<int64_t>(&k)->default_value(128),
         "Specific matrix size: the number of columns in A and rows in B.")

        ("lda",
         valueVec<int64_t>(&lda),
         "Leading dimension of matrix A.")

        ("ldb",
         valueVec<int64_t>(&ldb),
         "Leading dimension of matrix B.")

        ("ldc",
         valueVec<int64_t>(&ldc),
         "Leading dimension of matrix C.")

        ("ldd",
         valueVec<int64_t>(&ldd),
         "Leading dimension of matrix D.")

        ("lde",
         valueVec<int64_t>(&lde),
         "Leading dimension of matrix E.")

        ("any_stride",
         value<bool>(&any_stride)->default_value(false),
         "Do not modify input strides based on leading dimensions")

        ("stride_a",
         valueVec<int64_t>(&stride_a),
         "Specific stride of strided_batched matrix A, second dimension * leading dimension.")

        ("stride_b",
         valueVec<int64_t>(&stride_b),
         "Specific stride of strided_batched matrix B, second dimension * leading dimension.")

        ("stride_c",
         valueVec<int64_t>(&stride_c),
         "Specific stride of strided_batched matrix C, second dimension * leading dimension.")

        ("stride_d",
         valueVec<int64_t>(&stride_d),
         "Specific stride of strided_batched matrix D, second dimension * leading dimension.")

        ("stride_e",
         valueVec<int64_t>(&stride_e),
         "Specific stride of strided_batched matrix E, second dimension * leading dimension.")

        ("alpha",
          value<float>(&arg.alpha)->default_value(1.0), "specifies the scalar alpha")

        ("beta",
         value<float>(&arg.beta)->default_value(0.0), "specifies the scalar beta")

        ("function,f",
         value<std::string>(&function)->default_value("matmul"), "BLASLt function to test. "
         "Options: matmul")

        ("precision,r",
         value<std::string>(&precision)->default_value("f16_r"), "Precision of matrix A,B,C,D  "
         "Options: f32_r,f16_r,bf16_r,f64_r,i32_r,i8_r")

        ("a_type",
         value<std::string>(&a_type), "Precision of matrix A. "
        "Options: f32_r,f16_r,bf16_r,i8_r")

        ("b_type",
         value<std::string>(&b_type), "Precision of matrix B. "
        "Options: f32_r,f16_r,bf16_r,i8_r")

        ("c_type",
         value<std::string>(&c_type), "Precision of matrix C. "
         "Options: f32_r,f16_r,bf16_r,i8_r")

        ("d_type",
         value<std::string>(&d_type), "Precision of matrix D. "
        "Options: f32_r,f16_r,bf16_r,i8_r")

        ("compute_type",
         value<std::string>(&compute_type)->default_value("f32_r"), "Precision of computation. "
         "Options: s,f32_r,x,xf32_r,f64_r,i32_r,f32_bf16_r")

        ("compute_input_typeA",
         value<std::string>(&compute_input_typeA), "Precision of computation input A. "
         "Options: f32_r, f16_r, bf16_r, f8_r, bf8_r, f8_fnuz_r, bf8_fnuz_r, The default value indicates that the compute_input_typeA has no effect.")

        ("compute_input_typeB",
         value<std::string>(&compute_input_typeB), "Precision of computation input B. "
         "Options: f32_r, f16_r, bf16_r, f8_r, bf8_r, f8_fnuz_r, bf8_fnuz_r, The default value indicates that the compute_input_typeA has no effect.")

        ("scale_type",
         value<std::string>(&scale_type), "Precision of scalar. "
        "Options: f16_r,bf16_r")

        ("initialization",
         value<std::string>(&initialization)->default_value("hpl"),
         "Initialize matrix data."
         "Options: rand_int, trig_float, hpl(floating), special, zero, norm_dist, uniform_01")

        ("transA",
         value<char>(&arg.transA)->default_value('N'),
         "N = no transpose, T = transpose")

        ("transB",
         value<char>(&arg.transB)->default_value('N'),
         "N = no transpose, T = transpose")

        ("swizzleA",
         value<bool>(&arg.swizzle_a)->default_value(false),
         "Enable tensor swizzling for A")

        ("swizzleB",
         value<bool>(&arg.swizzle_b)->default_value(false),
         "Enable tensor swizzling for B")

        ("batch_count",
         value<int32_t>(&arg.batch_count)->default_value(1),
         "Number of matrices. Only applicable to batched and strided_batched routines")

        ("HMM",
         value<bool>(&arg.HMM)->default_value(false),
         "Parameter requesting the use of HipManagedMemory")

        ("verify,v",
         value<bool>(&verify)->default_value(false),
         "Validate GPU results with CPU?")

        ("iters,i",
         value<int32_t>(&arg.iters)->default_value(tuningEnv? 1000 : 10),
         "Iterations to run inside timing loop")

        ("cold_iters,j",
         value<int32_t>(&arg.cold_iters)->default_value(tuningEnv? 1000 : 2),
         "Cold Iterations to run before entering the timing loop")

        ("algo_method",
         value<std::string>(&algo_method_str)->default_value("heuristic"),
         "Use different algorithm search API. Options: heuristic, all, index.")

        ("solution_index",
         value<int32_t>(&arg.solution_index)->default_value(-1),
         "Used with --algo_method 2.  Specify solution index to use in benchmark.")

        ("requested_solution",
         value<int32_t>(&arg.requested_solution_num)->default_value(tuningEnv? -1 : 1),
         "Requested solution num. Set to -1 to get all solutions. Only valid when algo_method is set to heuristic.")

        ("activation_type",
         value<std::string>(&activation_type)->default_value("none"),
         "Options: none, gelu, relu, swish, clamp")

        ("activation_arg1",
         value<float>(&arg.activation_arg1)->default_value(0),
         "First extra argument for activation function if needed.")

        ("activation_arg2",
         value<float>(&arg.activation_arg2)->default_value(std::numeric_limits<float>::infinity()),
         "Second extra argument for activation function if needed.")

        ("bias_type",
         value<std::string>(&bias_type), "Precision of bias vector."
        "Options: f16_r,bf16_r,f32_r,default(same with D type)")

        ("bias_source",
         value<std::string>(&bias_source)->default_value("d"),
         "Choose bias source: a, b, d")

        ("bias_vector",
         bool_switch(&arg.bias_vector)->default_value(false),
         "Apply bias vector")

        ("scaleA",
         value<int>(&scaleAFormat)->default_value(0),
         "Apply scale for A buffer. 0 = None, 1 = scalar, 2 = vector, 3 = block, 1001 = block_preswizzled_32x8.")

        ("scaleB",
         value<int>(&scaleBFormat)->default_value(0),
         "Apply scale for B buffer. 0 = None, 1 = scalar, 2 = vector, 3 = block, 1001 = block_preswizzled_32x8.")

        ("scaleC",
         value<int>(&scaleCFormat)->default_value(0),
         "Apply scale for C buffer. 0 = None, 1 = scalar")

        ("scaleD",
         value<int>(&scaleDFormat)->default_value(0),
         "Apply scale for D buffer. 0 = None, 1 = scalar")

        ("scaleAlpha_vector",
         bool_switch(&arg.scaleAlpha_vector)->default_value(false),
         "Apply scaleAlpha vector")

        ("amaxScaleA",
         bool_switch(&arg.amaxScaleA)->default_value(false),
         "Apply scale for A buffer by abs max of A buffer")

        ("amaxScaleB",
         bool_switch(&arg.amaxScaleB)->default_value(false),
         "Apply scale for B buffer by abs max of B buffer")

        ("amaxD",
         bool_switch(&arg.amaxD)->default_value(false),
         "Output Amax of intermediate D matrix")

        ("use_e",
         bool_switch(&arg.use_e)->default_value(false),
         "Apply AUX output/ gradient input")

        ("aux_type",
         value<std::string>(&aux_type), "Used with --use_e. Precision of AUX output (matrix E)."
	 "Options: f16_r, default (same with D type)")

        ("gradient",
         bool_switch(&arg.gradient)->default_value(false),
         "Enable gradient")

        ("grouped_gemm",
         value<bool>(&grouped_gemm)->default_value(false),
         "Use grouped_gemm.")

        ("use_user_args",
        value<bool>(&arg.use_user_args)->default_value(false),
        "Use UserArguments located in device memory for grouped gemm.")

        ("device",
         value<int>(&device_id)->default_value(0),
         "Set default device to be used for subsequent program runs")

        ("c_equal_d",
         bool_switch(&arg.c_equal_d)->default_value(false),
         "C and D are stored in same memory")

        ("workspace",
         value<size_t>(&arg.user_allocated_workspace)->default_value(tuningEnv && tuningMaxWorkSpace ? atoi(tuningMaxWorkSpace) : 128 * 1024 * 1024),
         "Set fixed workspace memory size (bytes) instead of using hipblaslt managed memory")

        ("log_function_name",
         bool_switch(&log_function_name)->default_value(false),
         "Function name precedes other items.")

        ("function_filter",
         value<std::string>(&filter),
         "Simple strstr filter on function name only without wildcards")

        ("api_method",
         value<std::string>(&api_method_str)->default_value("c"),
         "Use extension API. c: C style API. mix: declaration with C hipblasLtMatmul Layout/Desc but set, initialize, and run the problem with C++ extension API. cpp: Using C++ extension API only. "
         "Options: c, mix, cpp.")

        ("print_kernel_info",
         value<bool>(&arg.print_kernel_info)->default_value(false),
         "Print solution, kernel name and solution index.")

        ("rotating",
         value<int32_t>(&arg.rotating)->default_value(tuningEnv ? 512 : 0),
         "Use rotating memory blocks for each iteration, size in MB.")

        ("use_gpu_timer",
         value<bool>(&arg.use_gpu_timer)->default_value(false),
         "Use hipEventElapsedTime to profile elapsed time.")

        ("skip_slow_solution_ratio",
          value<float>(&arg.skip_slow_solution_ratio)->default_value(0.0),
          "Specifies a ratio to skip slow solution when warm up stage. "
          "Skip condition: (current solution's warm up time * ratio) > best solution's warm up time. "
          "Ratio range: 0 ~ 1. 0 means no skip.")

        ("splitk",
         valueVec<uint32_t>(&gsu_vector),
         "[Tuning parameter] Set split K for a solution, 0 is use solution's default value. (Only support GEMM + api_method mix or cpp)")

        ("wgm",
         valueVec<uint32_t>(&wgm_vector),
         "[Tuning parameter] Set workgroup mapping for a solution, 0 is use solution's default value. (Only support GEMM + api_method mix or cpp)")

        ("flush",
        value<bool>(&arg.flush)->default_value(tuningEnv ? true : false),
        "Flush icache, only works for gemm.")

        ("dump_matrix",
        value<bool>(&arg.dump_matrix)->default_value(false),
        "Dump input and output matrices to a file.")

        ("help,h", "produces this help message")

        ("version", "Prints the version number");
    // clang-format on

    // parse command line into arg structure and stack variables using desc
    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    if((argc <= 1 && !datafile) || vm.count("help"))
    {
        hipblaslt_cout << desc << std::endl;
        return 0;
    }

    hipblaslt_print_version();

    if(vm.find("version") != vm.end())
    {
        return 0;
    }

    if(api_method_str.compare("c") == 0)
    {
        api_method = 0;
    }
    else if(api_method_str.compare("mix") == 0)
    {
        api_method = 1;
    }
    else if(api_method_str.compare("cpp") == 0)
    {
        api_method = 2;
    }
    else
    {
        hipblaslt_cerr << "Invalid api method: " << api_method_str << std::endl;
        return 1;
    }

    if(algo_method_str.compare("heuristic") == 0)
    {
        arg.algo_method = 0;
    }
    else if(algo_method_str.compare("all") == 0)
    {
        arg.algo_method = tuningEnv ? 0 : 1;
    }
    else if(algo_method_str.compare("index") == 0)
    {
        arg.algo_method = tuningEnv ? 0 : 2;
    }
    else
    {
        hipblaslt_cerr << "Invalid algo method: " << algo_method_str << std::endl;
        return 1;
    }

    int max_gsu = 0;
    if(gsu_vector.size() > MAX_SUPPORTED_NUM_PROBLEMS)
    {
        hipblaslt_cerr << "Too many gsu parameters, maximum is: " << MAX_SUPPORTED_NUM_PROBLEMS
                       << std::endl;
        return 1;
    }
    for(size_t i = 0; i < gsu_vector.size(); i++)
    {
        if(gsu_vector[i] < 0 || gsu_vector[i] > 255)
        {
            hipblaslt_cerr << "SplitK range is 0~255." << std::endl;
            return 1;
        }
        arg.gsu_vector[i] = gsu_vector[i];
        max_gsu           = max(max_gsu, arg.gsu_vector[i]);
    }
    if((max_gsu > 0) && ((api_method == 0) || arg.grouped_gemm))
    {
        hipblaslt_cerr << "Currently split K only supports GEMM + api_method mix or cpp."
                       << std::endl;
        return 1;
    }
    int max_wgm = 0;
    if(wgm_vector.size() > MAX_SUPPORTED_NUM_PROBLEMS)
    {
        hipblaslt_cerr << "Too many wgm parameters, maximum is: " << MAX_SUPPORTED_NUM_PROBLEMS
                       << std::endl;
        return 1;
    }
    for(size_t i = 0; i < wgm_vector.size(); i++)
    {
        if(wgm_vector[i] < 0 || wgm_vector[i] > 255)
        {
            hipblaslt_cerr << "Workgroup mapping range is 0~255." << std::endl;
            return 1;
        }
        arg.wgm_vector[i] = wgm_vector[i];
        max_wgm           = max(max_wgm, arg.wgm_vector[i]);
    }
    if((max_wgm > 0) && (api_method == 0))
    {
        hipblaslt_cerr << "Currently workgroup mapping only supports api_method mix or cpp."
                       << std::endl;
        return 1;
    }

    // transfer local variable state
    ArgumentModel_set_log_function_name(log_function_name);

    // Fill in the sizes to arguments
    size_t length = 1;
    if(grouped_gemm)
    {
        length           = std::max(m.size(), n.size());
        length           = std::max(length, k.size());
        length           = std::max(length, lda.size());
        length           = std::max(length, ldb.size());
        length           = std::max(length, ldc.size());
        length           = std::max(length, ldd.size());
        length           = std::max(length, lde.size());
        length           = std::max(length, stride_a.size());
        length           = std::max(length, stride_b.size());
        length           = std::max(length, stride_c.size());
        length           = std::max(length, stride_d.size());
        length           = std::max(length, stride_e.size());
        length           = std::min(length, MAX_SUPPORTED_NUM_PROBLEMS);
        arg.grouped_gemm = length;
    }
    else
    {
        arg.grouped_gemm = 0;
    }
    for(size_t i = 0; i < length; i++)
    {
        if(m.size() > 0)
            arg.M[i] = m.size() >= length ? m[i] : m[m.size() - 1];
        if(n.size() > 0)
            arg.N[i] = n.size() >= length ? n[i] : n[n.size() - 1];
        if(k.size() > 0)
            arg.K[i] = k.size() >= length ? k[i] : k[k.size() - 1];

        if(lda.size() > 0)
            arg.lda[i] = lda.size() >= length ? lda[i] : lda[lda.size() - 1];
        if(ldb.size() > 0)
            arg.ldb[i] = ldb.size() >= length ? ldb[i] : ldb[ldb.size() - 1];
        if(ldc.size() > 0)
            arg.ldc[i] = ldc.size() >= length ? ldc[i] : ldc[ldc.size() - 1];
        if(ldd.size() > 0)
            arg.ldd[i] = ldd.size() >= length ? ldd[i] : ldd[ldd.size() - 1];
        if(lde.size() > 0)
            arg.lde[i] = lde.size() >= length ? lde[i] : lde[lde.size() - 1];

        if(stride_a.size() > 0)
            arg.stride_a[i]
                = stride_a.size() >= length ? stride_a[i] : stride_a[stride_a.size() - 1];
        if(stride_b.size() > 0)
            arg.stride_b[i]
                = stride_b.size() >= length ? stride_b[i] : stride_b[stride_b.size() - 1];
        if(stride_c.size() > 0)
            arg.stride_c[i]
                = stride_c.size() >= length ? stride_c[i] : stride_c[stride_c.size() - 1];
        if(stride_d.size() > 0)
            arg.stride_d[i]
                = stride_d.size() >= length ? stride_d[i] : stride_d[stride_d.size() - 1];
        if(stride_e.size() > 0)
            arg.stride_e[i]
                = stride_e.size() >= length ? stride_e[i] : stride_e[stride_e.size() - 1];
    }

    // Device Query
    hipDeviceProp_t props;
    int64_t         device_count = query_device_property(device_id, props);

    hipblaslt_cout << std::endl;
    if(device_count <= device_id)
        throw std::invalid_argument("Invalid Device ID");
    set_device(device_id);

    EfficiencyMonitor& perf_monitor = getEfficiencyMonitor();
    perf_monitor.set_device_id(device_id);

    if(datafile)
        return hipblaslt_bench_datafile(filter, any_stride, props);

    // single bench run

    // validate arguments
    if(arg.algo_method == 1)
    {
        arg.requested_solution_num = HIPBLASLT_MAX_REQUESTED_SOLUTION_NUM;
    }
    else if(arg.algo_method == 2)
    {
        arg.requested_solution_num = 1;
    }

    std::transform(precision.begin(), precision.end(), precision.begin(), ::tolower);
    auto prec = string_to_hip_datatype(precision);
    if(prec == HIPBLASLT_DATATYPE_INVALID)
        throw std::invalid_argument("Invalid value for --precision " + precision);

    arg.a_type = a_type == "" ? prec : string_to_hip_datatype(a_type);
    if(arg.a_type == HIPBLASLT_DATATYPE_INVALID)
        throw std::invalid_argument("Invalid value for --a_type " + a_type);

    arg.b_type = b_type == "" ? prec : string_to_hip_datatype(b_type);
    if(arg.b_type == HIPBLASLT_DATATYPE_INVALID)
        throw std::invalid_argument("Invalid value for --b_type " + b_type);

    arg.c_type = c_type == "" ? prec : string_to_hip_datatype(c_type);
    if(arg.c_type == HIPBLASLT_DATATYPE_INVALID)
        throw std::invalid_argument("Invalid value for --c_type " + c_type);

    arg.d_type = d_type == "" ? prec : string_to_hip_datatype(d_type);
    if(arg.d_type == HIPBLASLT_DATATYPE_INVALID)
        throw std::invalid_argument("Invalid value for --d_type " + d_type);

    if(arg.c_type != arg.d_type)
        throw std::invalid_argument(
            "Invalid: --c_type " + std::string(hip_datatype_to_string(arg.c_type))
            + " is not equal to --d_type " + std::string(hip_datatype_to_string(arg.d_type)));

    bool is_f16 = arg.a_type == HIP_R_16F || arg.a_type == HIP_R_16BF;
    bool is_f32 = arg.a_type == HIP_R_32F;
    arg.compute_type
        = compute_type == "" ? (HIPBLAS_COMPUTE_32F) : string_to_hipblas_computetype(compute_type);
    if(arg.compute_type == HIPBLASLT_COMPUTE_TYPE_INVALID)
        throw std::invalid_argument("Invalid value for --compute_type " + compute_type);

    // The value HIPBLASLT_DATATYPE_INVALID indicates that the compute_input_typeA has no effect.
    arg.compute_input_typeA = (compute_input_typeA != "")
                                  ? string_to_hip_datatype(compute_input_typeA)
                                  : HIPBLASLT_DATATYPE_INVALID;
    if(arg.compute_input_typeA == HIPBLASLT_DATATYPE_INVALID && compute_input_typeA != "")
        throw std::invalid_argument("Invalid value for --compute_input_typeA "
                                    + compute_input_typeA);

    arg.compute_input_typeB = (compute_input_typeB != "")
                                  ? string_to_hip_datatype(compute_input_typeB)
                                  : HIPBLASLT_DATATYPE_INVALID;
    if(arg.compute_input_typeB == HIPBLASLT_DATATYPE_INVALID && compute_input_typeB != "")
        throw std::invalid_argument("Invalid value for --compute_input_typeB "
                                    + compute_input_typeB);

    if(string_to_hip_datatype(bias_type) == HIPBLASLT_DATATYPE_INVALID && bias_type != ""
       && bias_type != "default")
        throw std::invalid_argument("Invalid value for --bias_type " + bias_type);
    else
        arg.bias_type = string_to_hip_datatype(bias_type);

    arg.initialization = string2hipblaslt_initialization(initialization);
    if(arg.initialization == static_cast<hipblaslt_initialization>(0))
        throw std::invalid_argument("Invalid value for --initialization " + initialization);

    arg.activation_type = string_to_hipblaslt_activation_type(activation_type);
    if(arg.activation_type == static_cast<hipblaslt_activation_type>(-1))
        throw std::invalid_argument("Invalid value for --activation_type " + activation_type);

    arg.bias_source = string_to_hipblaslt_bias_source(bias_source);

    if(!(aux_type == "" || aux_type == "default" || arg.use_e))
        hipblaslt_cerr << "warning: --use_e not set but --aux_type is provided" << std::endl;

    arg.aux_type
        = (aux_type == "" || aux_type == "default") ? arg.d_type : string_to_hip_datatype(aux_type);
    if(arg.aux_type == HIPBLASLT_DATATYPE_INVALID)
        throw std::invalid_argument("Invalid value for --aux_type " + aux_type);

    if(arg.swizzle_a
       && (arg.transA != 'T' || arg.transB != 'N'
           || (arg.a_type != string_to_hip_datatype("f16_r")
               && arg.a_type != string_to_hip_datatype("f8_fnuz_r")
               && arg.a_type != string_to_hip_datatype("f8_r")
               && arg.a_type != string_to_hip_datatype("bf16_r"))))
    {
        hipblaslt_cerr << "For swizzle-A, problem type must be FP16 or BF16 or FP8 TN" << std::endl;
        return 1;
    }

    if(arg.swizzle_b
       && (arg.transA != 'T' || arg.transB != 'N'
           || (arg.b_type != string_to_hip_datatype("f16_r")
               && arg.b_type != string_to_hip_datatype("f8_fnuz_r")
               && arg.b_type != string_to_hip_datatype("f8_r")
               && arg.b_type != string_to_hip_datatype("bf16_r"))))
    {
        hipblaslt_cerr << "For swizzle-B, problem type must be FP16 or BF16 or FP8 TN" << std::endl;
        return 1;
    }

    auto scaleInt2Enum = [](int s) {
        if(s == 0)
            return hipblaslt_scaling_format::none;
        if(s == 1)
            return hipblaslt_scaling_format::Scalar;
        if(s == 2)
            return hipblaslt_scaling_format::Vector;
        if(s == 3)
            return hipblaslt_scaling_format::Block_32_UE8M0;
        if(s == 1001)
            return hipblaslt_scaling_format::Block_32_UE8M0_32_8_EXT;
        return hipblaslt_scaling_format::none;
    };
    arg.scaleA = scaleInt2Enum(scaleAFormat);
    arg.scaleB = scaleInt2Enum(scaleBFormat);
    arg.scaleC = scaleCFormat;
    arg.scaleD = scaleDFormat;

    // Validation for F4 and F6
    if(arg.a_type == HIP_R_4F_E2M1_EXT || arg.a_type == HIP_R_6F_E2M3_EXT
       || arg.a_type == HIP_R_6F_E3M2_EXT)
    {
        if(!isBlockScaling(arg.scaleA))
            throw std::invalid_argument("scaleA must be block format for F4 and F6 types");
    }
    if(arg.b_type == HIP_R_4F_E2M1_EXT || arg.b_type == HIP_R_6F_E2M3_EXT
       || arg.b_type == HIP_R_6F_E3M2_EXT)
    {
        if(!isBlockScaling(arg.scaleB))
            throw std::invalid_argument("scaleB must be block format for F4 and F6 types");
    }

    // Block scaling only allows F8/F6/F4
    if(isBlockScaling(arg.scaleA))
    {
        if(arg.a_type != HIP_R_8F_E4M3 && arg.a_type != HIP_R_8F_E5M2
           && arg.a_type != HIP_R_4F_E2M1_EXT && arg.a_type != HIP_R_6F_E2M3_EXT
           && arg.a_type != HIP_R_6F_E3M2_EXT)
            throw std::invalid_argument("Invalid a_type for block scaling format: "s
                                        + hip_datatype_to_string(arg.a_type));
    }
    if(isBlockScaling(arg.scaleB))
    {
        if(arg.b_type != HIP_R_8F_E4M3 && arg.b_type != HIP_R_8F_E5M2
           && arg.b_type != HIP_R_4F_E2M1_EXT && arg.b_type != HIP_R_6F_E2M3_EXT
           && arg.b_type != HIP_R_6F_E3M2_EXT)
            throw std::invalid_argument("Invalid b_type for block scaling format: "s
                                        + hip_datatype_to_string(arg.b_type));
    }
    if(isBlockScaling(arg.scaleA) || isBlockScaling(arg.scaleB))
    {
        if(arg.compute_type != HIPBLAS_COMPUTE_32F)
            throw std::invalid_argument("Block scaling only supports f32 as compute type not "s
                                        + compute_type);

        // For C and D, only F32/BF16/F16 allowed when A or B is block scaling format
        if(arg.c_type != HIP_R_32F && arg.c_type != HIP_R_16F && arg.c_type != HIP_R_16BF)
            throw std::invalid_argument("Invalid c_type for block scaling format: "s
                                        + hip_datatype_to_string(arg.c_type));
        if(arg.d_type != HIP_R_32F && arg.d_type != HIP_R_16F && arg.d_type != HIP_R_16BF)
            throw std::invalid_argument("Invalid d_type for block scaling format: "s
                                        + hip_datatype_to_string(arg.d_type));
    }

    if(arg.M[0] < 0)
        throw std::invalid_argument("Invalid value for -m " + std::to_string(arg.M[0]));
    if(arg.N[0] < 0)
        throw std::invalid_argument("Invalid value for -n " + std::to_string(arg.N[0]));
    if(arg.K[0] < 0)
        throw std::invalid_argument("Invalid value for -k " + std::to_string(arg.K[0]));

    int copied = snprintf(arg.function, sizeof(arg.function), "%s", function.c_str());
    if(copied <= 0 || copied >= sizeof(arg.function))
        throw std::invalid_argument("Invalid value for --function");

    if(arg.skip_slow_solution_ratio < 0 || arg.skip_slow_solution_ratio > 1)
        throw std::invalid_argument(
            "Valid value for --skip_slow_solution_ratio is in range (0.0 ~ 1.0).");

    if(verify)
    {
        arg.norm_check     = 1;
        arg.allclose_check = 1;
    }

    switch(api_method)
    {
    case 0:
        arg.use_ext            = false;
        arg.use_ext_setproblem = false;
        break;
    case 1:
        arg.use_ext            = true;
        arg.use_ext_setproblem = false;
        break;
    case 2:
        arg.use_ext            = true;
        arg.use_ext_setproblem = true;
        break;
    default:
        throw std::invalid_argument("Invalid value for api_method: " + std::to_string(api_method));
        break;
    }

    arg.norm_check_assert = false;
    int status            = run_bench_test(arg, filter, any_stride, props);
    freeEfficiencyMonitor();
    return status;
}
catch(const std::invalid_argument& exp)
{
    hipblaslt_cerr << exp.what() << std::endl;
    return -1;
}
