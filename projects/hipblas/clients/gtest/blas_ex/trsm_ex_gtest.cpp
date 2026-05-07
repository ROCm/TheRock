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

#include "blas_ex/testing_trsm_batched_ex.hpp"
#include "blas_ex/testing_trsm_ex.hpp"
#include "blas_ex/testing_trsm_strided_batched_ex.hpp"
#include "hipblas_data.hpp"
#include "hipblas_test.hpp"
#include "type_dispatch.hpp"

namespace
{
    // possible trsm test cases
    enum trsm_ex_test_type
    {
        TRSM_EX,
        TRSM_BATCHED_EX,
        TRSM_STRIDED_BATCHED_EX,
    };

    // trsm_ex test template
    template <template <typename...> class FILTER, trsm_ex_test_type TRSM_EX_TYPE>
    struct trsm_ex_template : HipBLAS_Test<trsm_ex_template<FILTER, TRSM_EX_TYPE>, FILTER>
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
            return hipblas_simple_dispatch<trsm_ex_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(TRSM_EX_TYPE)
            {
            case TRSM_EX:
                return !strcmp(arg.function, "trsm_ex") || !strcmp(arg.function, "trsm_ex_bad_arg");
            case TRSM_BATCHED_EX:
                return !strcmp(arg.function, "trsm_batched_ex")
                       || !strcmp(arg.function, "trsm_batched_ex_bad_arg");
            case TRSM_STRIDED_BATCHED_EX:
                return !strcmp(arg.function, "trsm_strided_batched_ex")
                       || !strcmp(arg.function, "trsm_strided_batched_ex_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            std::string name;
            if constexpr(TRSM_EX_TYPE == TRSM_EX)
                testname_trsm_ex(arg, name);
            else if constexpr(TRSM_EX_TYPE == TRSM_BATCHED_EX)
                testname_trsm_batched_ex(arg, name);
            else if constexpr(TRSM_EX_TYPE == TRSM_STRIDED_BATCHED_EX)
                testname_trsm_strided_batched_ex(arg, name);
            return std::move(name);
        }
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct trsm_ex_testing : hipblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct trsm_ex_testing<
        T,
        std::enable_if_t<
            std::is_same_v<
                T,
                float> || std::is_same_v<T, double> || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>>>
        : hipblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "trsm_ex"))
                testing_trsm_ex<T>(arg);
            else if(!strcmp(arg.function, "trsm_ex_bad_arg"))
                testing_trsm_ex_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "trsm_batched_ex"))
                testing_trsm_batched_ex<T>(arg);
            else if(!strcmp(arg.function, "trsm_batched_ex_bad_arg"))
                testing_trsm_batched_ex_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "trsm_strided_batched_ex"))
                testing_trsm_strided_batched_ex<T>(arg);
            else if(!strcmp(arg.function, "trsm_strided_batched_ex_bad_arg"))
                testing_trsm_strided_batched_ex_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using trsm_ex = trsm_ex_template<trsm_ex_testing, TRSM_EX>;
    TEST_P(trsm_ex, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            hipblas_simple_dispatch<trsm_ex_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(trsm_ex);

    using trsm_batched_ex = trsm_ex_template<trsm_ex_testing, TRSM_BATCHED_EX>;
    TEST_P(trsm_batched_ex, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            hipblas_simple_dispatch<trsm_ex_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(trsm_batched_ex);

    using trsm_strided_batched_ex = trsm_ex_template<trsm_ex_testing, TRSM_STRIDED_BATCHED_EX>;
    TEST_P(trsm_strided_batched_ex, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            hipblas_simple_dispatch<trsm_ex_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(trsm_strided_batched_ex);

} // namespace
