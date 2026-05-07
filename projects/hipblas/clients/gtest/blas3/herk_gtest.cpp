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

#include "blas3/testing_herk.hpp"
#include "blas3/testing_herk_batched.hpp"
#include "blas3/testing_herk_strided_batched.hpp"
#include "hipblas_data.hpp"
#include "hipblas_test.hpp"
#include "type_dispatch.hpp"

namespace
{
    // possible herk test cases
    enum herk_test_type
    {
        HERK,
        HERK_BATCHED,
        HERK_STRIDED_BATCHED,
    };

    // herk test template
    template <template <typename...> class FILTER, herk_test_type HERK_TYPE>
    struct herk_template : HipBLAS_Test<herk_template<FILTER, HERK_TYPE>, FILTER>
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
            return hipblas_simple_dispatch<herk_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(HERK_TYPE)
            {
            case HERK:
                return !strcmp(arg.function, "herk") || !strcmp(arg.function, "herk_bad_arg");
            case HERK_BATCHED:
                return !strcmp(arg.function, "herk_batched")
                       || !strcmp(arg.function, "herk_batched_bad_arg");
            case HERK_STRIDED_BATCHED:
                return !strcmp(arg.function, "herk_strided_batched")
                       || !strcmp(arg.function, "herk_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            std::string name;
            if constexpr(HERK_TYPE == HERK)
                testname_herk(arg, name);
            else if constexpr(HERK_TYPE == HERK_BATCHED)
                testname_herk_batched(arg, name);
            else if constexpr(HERK_TYPE == HERK_STRIDED_BATCHED)
                testname_herk_strided_batched(arg, name);
            return std::move(name);
        }
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct herk_testing : hipblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct herk_testing<
        T,
        std::enable_if_t<
            std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>>>
        : hipblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "herk"))
                testing_herk<T>(arg);
            else if(!strcmp(arg.function, "herk_bad_arg"))
                testing_herk_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "herk_batched"))
                testing_herk_batched<T>(arg);
            else if(!strcmp(arg.function, "herk_batched_bad_arg"))
                testing_herk_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "herk_strided_batched"))
                testing_herk_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "herk_strided_batched_bad_arg"))
                testing_herk_strided_batched_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using herk = herk_template<herk_testing, HERK>;
    TEST_P(herk, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<herk_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(herk);

    using herk_batched = herk_template<herk_testing, HERK_BATCHED>;
    TEST_P(herk_batched, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<herk_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(herk_batched);

    using herk_strided_batched = herk_template<herk_testing, HERK_STRIDED_BATCHED>;
    TEST_P(herk_strided_batched, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<herk_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(herk_strided_batched);

} // namespace
