/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "hipblas_arguments.hpp"
#include "tuple_helper.hpp"
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <utility>

bool gpu_arch_match(const std::string& gpu_arch, const char pattern[4])
{
    int         gpu_len = gpu_arch.length();
    const char* gpu     = gpu_arch.c_str();

    // gpu is currently "gfx" followed by 3 or 4 characters, followed by optional ":" sections
    int prefix_len = 3;
    for(int i = 0; i < 4; i++)
    {
        if(!pattern[i])
            break;
        else if(pattern[i] == '?')
            continue;
        else if(prefix_len + i >= gpu_len || pattern[i] != gpu[prefix_len + i])
            return false;
    }
    return true;
};

// Pairs for YAML output
template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& os, std::pair<T1, T2> p)
{
    os << p.first << ": ";
    os << p.second;
    return os;
}

// Function to print Arguments out to stream in YAML format
std::ostream& operator<<(std::ostream& os, const Arguments& arg)
{
    // delim starts as "{ " and becomes ", " afterwards
    auto print_pair = [&, delim = "{ "](const char* name, const auto& value) mutable {
        os << delim << std::make_pair(name, value);
        delim = ", ";
    };

    // Print each (name, value) tuple pair
#define NAME_VALUE_PAIR(NAME) print_pair(#NAME, arg.NAME)
    FOR_EACH_ARGUMENT(NAME_VALUE_PAIR, ;);

    // Closing brace
    return os << " }\n";
}

// Google Tests uses this automatically with std::ostream to dump parameters
/*
std::ostream& operator<<(std::ostream& os, const Arguments& arg)
{
    std::ostream oss;
    // Print to std::ostream, then transfer to std::ostream
    return os << arg;
}*/

// Function to read Structures data from stream
std::istream& operator>>(std::istream& is, Arguments& arg)
{
    is.read(reinterpret_cast<char*>(&arg), sizeof(arg));
    return is;
}

// Error message about incompatible binary file format
static void validation_error [[noreturn]] (const char* name)
{
    std::cerr << "Arguments field \"" << name
              << "\" does not match format.\n\n"
                 "Fatal error: Binary test data does match input format.\n"
                 "Ensure that hipblas_arguments.hpp and hipblas_common.yaml\n"
                 "define exactly the same Arguments, that hipblas_gentest.py\n"
                 "generates the data correctly, and that endianness is the same."
              << std::endl;
    abort();
}

// hipblas_gentest.py is expected to conform to this format.
// hipblas_gentest.py uses hipblas_common.yaml to generate this format.
void Arguments::validate(std::istream& ifs)
{
    char      header[8]{}, trailer[8]{};
    Arguments arg{};

    ifs.read(header, sizeof(header));
    ifs >> arg;
    ifs.read(trailer, sizeof(trailer));

    if(strcmp(header, "hipBLAS"))
        validation_error("header");

    if(strcmp(trailer, "HIPblas"))
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
