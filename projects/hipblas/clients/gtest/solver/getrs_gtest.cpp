/* ************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include "hipblas_data.hpp"
#include "hipblas_test.hpp"
#include "solver/testing_getrs.hpp"
#include "solver/testing_getrs_batched.hpp"
#include "solver/testing_getrs_strided_batched.hpp"
#include "type_dispatch.hpp"

namespace
{
    // possible getrs test cases
    enum getrs_test_type
    {
        GETRS,
        GETRS_BATCHED,
        GETRS_STRIDED_BATCHED,
    };

    //getrs test template
    template <template <typename...> class FILTER, getrs_test_type GETRS_TYPE>
    struct getrs_template : HipBLAS_Test<getrs_template<FILTER, GETRS_TYPE>, FILTER>
    {
        template <typename... T>
        struct type_filter_functor
        {
            bool operator()(const Arguments& args)
            {
                // additional global filters applied first
                if(!hipblas_client_global_filters(args))
                    return false;

                // type filters
                return static_cast<bool>(FILTER<T...>{});
            }
        };

        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return hipblas_simple_dispatch<getrs_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(GETRS_TYPE)
            {
            case GETRS:
                return !strcmp(arg.function, "getrs") || !strcmp(arg.function, "getrs_bad_arg");
            case GETRS_BATCHED:
                return !strcmp(arg.function, "getrs_batched")
                       || !strcmp(arg.function, "getrs_batched_bad_arg");
            case GETRS_STRIDED_BATCHED:
                return !strcmp(arg.function, "getrs_strided_batched")
                       || !strcmp(arg.function, "getrs_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            std::string name;
            if constexpr(GETRS_TYPE == GETRS)
                testname_getrs(arg, name);
            else if constexpr(GETRS_TYPE == GETRS_BATCHED)
                testname_getrs_batched(arg, name);
            else if constexpr(GETRS_TYPE == GETRS_STRIDED_BATCHED)
                testname_getrs_strided_batched(arg, name);
            return std::move(name);
        }
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct getrs_testing : hipblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct getrs_testing<
        T,
        std::enable_if_t<
            std::is_same_v<
                T,
                float> || std::is_same_v<T, double> || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>>>
        : hipblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "getrs"))
                testing_getrs<T>(arg);
            else if(!strcmp(arg.function, "getrs_bad_arg"))
                testing_getrs_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "getrs_batched"))
                testing_getrs_batched<T>(arg);
            else if(!strcmp(arg.function, "getrs_batched_bad_arg"))
                testing_getrs_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "getrs_strided_batched"))
                testing_getrs_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "getrs_strided_batched_bad_arg"))
                testing_getrs_strided_batched_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using getrs = getrs_template<getrs_testing, GETRS>;
    TEST_P(getrs, solver)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            hipblas_simple_dispatch<getrs_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(getrs);

    using getrs_batched = getrs_template<getrs_testing, GETRS_BATCHED>;
    TEST_P(getrs_batched, solver)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            hipblas_simple_dispatch<getrs_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(getrs_batched);

    using getrs_strided_batched = getrs_template<getrs_testing, GETRS_STRIDED_BATCHED>;
    TEST_P(getrs_strided_batched, solver)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            hipblas_simple_dispatch<getrs_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(getrs_strided_batched);

} // namespace
