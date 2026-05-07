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

#include "hipblaslt_arguments.hpp"
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <istream>
#include <ostream>
#include <regex>
#include <utility>

void Arguments::init()
{
    // match python in hipblaslt_common.py

    strcpy(function, "matmul");
    strcpy(name, "hipblaslt-bench");
    category[0]            = 0;
    known_bug_platforms[0] = 0;

    alpha = 1.0;
    beta  = 0.0;

    memset(stride_a, 0, sizeof(int64_t) * MAX_SUPPORTED_NUM_PROBLEMS);
    memset(stride_b, 0, sizeof(int64_t) * MAX_SUPPORTED_NUM_PROBLEMS);
    memset(stride_c, 0, sizeof(int64_t) * MAX_SUPPORTED_NUM_PROBLEMS);
    memset(stride_d, 0, sizeof(int64_t) * MAX_SUPPORTED_NUM_PROBLEMS);
    memset(stride_e, 0, sizeof(int64_t) * MAX_SUPPORTED_NUM_PROBLEMS);

    user_allocated_workspace = 128 * 1024 * 1024;

    M[0] = 128;
    N[0] = 128;
    K[0] = 128;

    memset(lda, 0, sizeof(int64_t) * MAX_SUPPORTED_NUM_PROBLEMS);
    memset(ldb, 0, sizeof(int64_t) * MAX_SUPPORTED_NUM_PROBLEMS);
    memset(ldc, 0, sizeof(int64_t) * MAX_SUPPORTED_NUM_PROBLEMS);
    memset(ldd, 0, sizeof(int64_t) * MAX_SUPPORTED_NUM_PROBLEMS);
    memset(lde, 0, sizeof(int64_t) * MAX_SUPPORTED_NUM_PROBLEMS);

    batch_count = 1;

    iters      = 10;
    cold_iters = 2;

    algo                   = 0;
    solution_index         = -1;
    requested_solution_num = 1;

    a_type              = HIP_R_16F;
    b_type              = HIP_R_16F;
    c_type              = HIP_R_16F;
    d_type              = HIP_R_16F;
    compute_type        = HIPBLAS_COMPUTE_32F;
    compute_input_typeA = HIPBLASLT_DATATYPE_INVALID;
    compute_input_typeB = HIPBLASLT_DATATYPE_INVALID;
    scale_type          = HIP_R_32F;

    initialization = hipblaslt_initialization::hpl;

    // memory padding for testing write out of bounds
    pad = 4096;

    // 16 bit
    threads = 0;
    streams = 0;

    // bytes
    devices = 0;

    norm_check     = 0;
    allclose_check = 0;
    unit_check     = 1;
    timing         = 0;

    transA = '*';
    transB = '*';

    activation_type   = hipblaslt_activation_type::none;
    activation_arg1   = 0.0f;
    activation_arg2   = std::numeric_limits<float>::infinity();
    bias_type         = HIPBLASLT_DATATYPE_INVALID;
    bias_source       = hipblaslt_bias_source::d;
    bias_vector       = false;
    scaleA            = hipblaslt_scaling_format::none;
    scaleB            = hipblaslt_scaling_format::none;
    scaleC            = false;
    scaleD            = false;
    scaleE            = false;
    scaleAlpha_vector = false;
    grouped_gemm      = 0;
    c_equal_d         = false;
    HMM               = false;
    use_e             = false;
    aux_type          = HIPBLASLT_DATATYPE_INVALID;
    gradient          = false;
    norm_check_assert = true;

    use_ext                  = false;
    use_ext_setproblem       = false;
    algo_method              = 0;
    use_user_args            = false;
    rotating                 = 0;
    use_gpu_timer            = false;
    skip_slow_solution_ratio = 0.0;

    // tuning
    gsu_vector[0] = 0;
    for(int32_t i = 1; i < MAX_SUPPORTED_NUM_PROBLEMS; i++)
    {
        gsu_vector[i] = -1;
    }
    wgm_vector[0] = 0;
    for(int32_t i = 1; i < MAX_SUPPORTED_NUM_PROBLEMS; i++)
    {
        wgm_vector[i] = -1;
    }

    print_solution_found              = false;
    flush                             = false;
    tensile_solution_selection_method = -1;
}

// Function to print Arguments out to stream in YAML format
hipblaslt_internal_ostream& operator<<(hipblaslt_internal_ostream& os, const Arguments& arg)
{
    // delim starts as "{ " and becomes ", " afterwards
    auto print_pair = [&, delim = "{ "](const char* name, const auto& value) mutable {
        os << delim << std::make_pair(name, value);
        delim = ", ";
    };

    // Print each (name, value) tuple pair
#define NAME_VALUE_PAIR(NAME) print_pair(#NAME, arg.NAME)
    // cppcheck-suppress unknownMacro
    FOR_EACH_ARGUMENT(NAME_VALUE_PAIR, ;);

    // Closing brace
    return os << " }\n";
}

// Google Tests uses this automatically with std::ostream to dump parameters
std::ostream& operator<<(std::ostream& os, const Arguments& arg)
{
    hipblaslt_internal_ostream oss;
    // Print to hipblaslt_internal_ostream, then transfer to std::ostream
    return os << (oss << arg);
}

// Function to read Structures data from stream
std::istream& operator>>(std::istream& is, Arguments& arg)
{
    is.read(reinterpret_cast<char*>(&arg), sizeof(arg));
    return is;
}

// Error message about incompatible binary file format
static void validation_error [[noreturn]] (const char* name)
{
    hipblaslt_cerr << "Arguments field \"" << name
                   << "\" does not match format.\n\n"
                      "Fatal error: Binary test data does match input format.\n"
                      "Ensure that hipblaslt_arguments.hpp and hipblaslt_common.yaml\n"
                      "define exactly the same Arguments, that hipblaslt_gentest.py\n"
                      "generates the data correctly, and that endianness is the same."
                   << std::endl;
    hipblaslt_abort();
}

// hipblaslt_gentest.py is expected to conform to this format.
// hipblaslt_gentest.py uses hipblaslt_common.yaml to generate this format.
void Arguments::validate(std::istream& ifs)
{
    char      header[10]{}, trailer[10]{};
    Arguments arg{};

    ifs.read(header, sizeof(header));
    ifs >> arg;
    ifs.read(trailer, sizeof(trailer));
    if(strcmp(header, "hipBLASLt"))
        validation_error("header");
    if(strcmp(trailer, "HIPblaslT"))
        validation_error("trailer");

    auto check_func = [sig = 0u](const char* name, const auto& value) mutable {
        static_assert(sizeof(value) <= 256,
                      "Fatal error: Arguments field is too large (greater than 256 bytes).");
        for(size_t i = 0; i < sizeof(value); ++i)
        {
            if(reinterpret_cast<const unsigned char*>(&value)[i] ^ sig ^ i)
                validation_error(name);
        }
        sig = (sig + 89) % 256;
    };

    // Apply check_func to each pair (name, value) of Arguments as a tuple
#define CHECK_FUNC(NAME) check_func(#NAME, arg.NAME)
    FOR_EACH_ARGUMENT(CHECK_FUNC, ;);
}
