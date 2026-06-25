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

#include "blas2/testing_syr2.hpp"
#include "blas2/testing_syr2_batched.hpp"
#include "blas2/testing_syr2_strided_batched.hpp"
#include "hipblas_data.hpp"
#include "hipblas_test.hpp"
#include "type_dispatch.hpp"

namespace
{
    // possible syr2 test cases
    enum syr2_test_type
    {
        SYR2,
        SYR2_BATCHED,
        SYR2_STRIDED_BATCHED,
    };

    //syr2 test template
    template <template <typename...> class FILTER, syr2_test_type SYR2_TYPE>
    struct syr2_template : HipBLAS_Test<syr2_template<FILTER, SYR2_TYPE>, FILTER>
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
            return hipblas_simple_dispatch<syr2_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(SYR2_TYPE)
            {
            case SYR2:
                return !strcmp(arg.function, "syr2") || !strcmp(arg.function, "syr2_bad_arg");
            case SYR2_BATCHED:
                return !strcmp(arg.function, "syr2_batched")
                       || !strcmp(arg.function, "syr2_batched_bad_arg");
            case SYR2_STRIDED_BATCHED:
                return !strcmp(arg.function, "syr2_strided_batched")
                       || !strcmp(arg.function, "syr2_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            std::string name;
            if constexpr(SYR2_TYPE == SYR2)
                testname_syr2(arg, name);
            else if constexpr(SYR2_TYPE == SYR2_BATCHED)
                testname_syr2_batched(arg, name);
            else if constexpr(SYR2_TYPE == SYR2_STRIDED_BATCHED)
                testname_syr2_strided_batched(arg, name);
            return std::move(name);
        }
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct syr2_testing : hipblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct syr2_testing<
        T,
        std::enable_if_t<
            std::is_same_v<
                T,
                float> || std::is_same_v<T, double> || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>>>
        : hipblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "syr2"))
                testing_syr2<T>(arg);
            else if(!strcmp(arg.function, "syr2_bad_arg"))
                testing_syr2_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "syr2_batched"))
                testing_syr2_batched<T>(arg);
            else if(!strcmp(arg.function, "syr2_batched_bad_arg"))
                testing_syr2_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "syr2_strided_batched"))
                testing_syr2_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "syr2_strided_batched_bad_arg"))
                testing_syr2_strided_batched_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using syr2 = syr2_template<syr2_testing, SYR2>;
    TEST_P(syr2, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<syr2_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(syr2);

    using syr2_batched = syr2_template<syr2_testing, SYR2_BATCHED>;
    TEST_P(syr2_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<syr2_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(syr2_batched);

    using syr2_strided_batched = syr2_template<syr2_testing, SYR2_STRIDED_BATCHED>;
    TEST_P(syr2_strided_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<syr2_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(syr2_strided_batched);

} // namespace
