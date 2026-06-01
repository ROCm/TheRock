/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "blas2/testing_sbmv.hpp"
#include "blas2/testing_sbmv_batched.hpp"
#include "blas2/testing_sbmv_strided_batched.hpp"
#include "hipblas_data.hpp"
#include "hipblas_test.hpp"
#include "type_dispatch.hpp"

namespace
{
    // possible sbmv test cases
    enum sbmv_test_type
    {
        SBMV,
        SBMV_BATCHED,
        SBMV_STRIDED_BATCHED,
    };

    //sbmv test template
    template <template <typename...> class FILTER, sbmv_test_type SBMV_TYPE>
    struct sbmv_template : HipBLAS_Test<sbmv_template<FILTER, SBMV_TYPE>, FILTER>
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
            return hipblas_simple_dispatch<sbmv_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(SBMV_TYPE)
            {
            case SBMV:
                return !strcmp(arg.function, "sbmv") || !strcmp(arg.function, "sbmv_bad_arg");
            case SBMV_BATCHED:
                return !strcmp(arg.function, "sbmv_batched")
                       || !strcmp(arg.function, "sbmv_batched_bad_arg");
            case SBMV_STRIDED_BATCHED:
                return !strcmp(arg.function, "sbmv_strided_batched")
                       || !strcmp(arg.function, "sbmv_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            std::string name;
            if constexpr(SBMV_TYPE == SBMV)
                testname_sbmv(arg, name);
            else if constexpr(SBMV_TYPE == SBMV_BATCHED)
                testname_sbmv_batched(arg, name);
            else if constexpr(SBMV_TYPE == SBMV_STRIDED_BATCHED)
                testname_sbmv_strided_batched(arg, name);
            return std::move(name);
        }
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct sbmv_testing : hipblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct sbmv_testing<T, std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double>>>
        : hipblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "sbmv"))
                testing_sbmv<T>(arg);
            else if(!strcmp(arg.function, "sbmv_bad_arg"))
                testing_sbmv_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "sbmv_batched"))
                testing_sbmv_batched<T>(arg);
            else if(!strcmp(arg.function, "sbmv_batched_bad_arg"))
                testing_sbmv_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "sbmv_strided_batched"))
                testing_sbmv_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "sbmv_strided_batched_bad_arg"))
                testing_sbmv_strided_batched_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using sbmv = sbmv_template<sbmv_testing, SBMV>;
    TEST_P(sbmv, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<sbmv_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(sbmv);

    using sbmv_batched = sbmv_template<sbmv_testing, SBMV_BATCHED>;
    TEST_P(sbmv_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<sbmv_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(sbmv_batched);

    using sbmv_strided_batched = sbmv_template<sbmv_testing, SBMV_STRIDED_BATCHED>;
    TEST_P(sbmv_strided_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<sbmv_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(sbmv_strided_batched);

} // namespace
