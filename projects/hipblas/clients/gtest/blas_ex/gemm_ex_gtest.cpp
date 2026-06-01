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

#include "blas_ex/testing_gemm_batched_ex.hpp"
#include "blas_ex/testing_gemm_ex.hpp"
#include "blas_ex/testing_gemm_strided_batched_ex.hpp"
#include "hipblas_data.hpp"
#include "hipblas_test.hpp"
#include "type_dispatch.hpp"

namespace
{
    // possible gemm test cases
    enum gemm_ex_test_type
    {
        GEMM_EX,
        GEMM_BATCHED_EX,
        GEMM_STRIDED_BATCHED_EX,
    };

    // gemm test template
    template <template <typename...> class FILTER, gemm_ex_test_type GEMM_EX_TYPE>
    struct gemm_ex_template : HipBLAS_Test<gemm_ex_template<FILTER, GEMM_EX_TYPE>, FILTER>
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
            return hipblas_simple_dispatch<gemm_ex_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(GEMM_EX_TYPE)
            {
            case GEMM_EX:
                return !strcmp(arg.function, "gemm_ex") || !strcmp(arg.function, "gemm_ex_bad_arg");
            case GEMM_BATCHED_EX:
                return !strcmp(arg.function, "gemm_batched_ex")
                       || !strcmp(arg.function, "gemm_batched_ex_bad_arg");
            case GEMM_STRIDED_BATCHED_EX:
                return !strcmp(arg.function, "gemm_strided_batched_ex")
                       || !strcmp(arg.function, "gemm_strided_batched_ex_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            std::string name;
            if constexpr(GEMM_EX_TYPE == GEMM_EX)
                testname_gemm_ex(arg, name);
            else if constexpr(GEMM_EX_TYPE == GEMM_BATCHED_EX)
                testname_gemm_batched_ex(arg, name);
            else if constexpr(GEMM_EX_TYPE == GEMM_STRIDED_BATCHED_EX)
                testname_gemm_strided_batched_ex(arg, name);
            return std::move(name);
        }
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename Ti, typename To = Ti, typename Tc = To, typename = void>
    struct gemm_ex_testing : hipblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename Ti, typename To, typename Tc>
    struct gemm_ex_testing<
        Ti,
        To,
        Tc,
        std::enable_if_t<
            !std::is_same_v<
                Ti,
                void> && !(std::is_same_v<Ti, Tc> && std::is_same_v<Ti, hipblasBfloat16>)>>
        : hipblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "gemm_ex"))
                testing_gemm_ex<Ti, To, Tc>(arg);
            else if(!strcmp(arg.function, "gemm_ex_bad_arg"))
                testing_gemm_ex_bad_arg<Ti, To, Tc>(arg);
            else if(!strcmp(arg.function, "gemm_batched_ex"))
                testing_gemm_batched_ex<Ti, To, Tc>(arg);
            else if(!strcmp(arg.function, "gemm_batched_ex_bad_arg"))
                testing_gemm_batched_ex_bad_arg<Ti, To, Tc>(arg);
            else if(!strcmp(arg.function, "gemm_strided_batched_ex"))
                testing_gemm_strided_batched_ex<Ti, To, Tc>(arg);
            else if(!strcmp(arg.function, "gemm_strided_batched_ex_bad_arg"))
                testing_gemm_strided_batched_ex_bad_arg<Ti, To, Tc>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using gemm_ex = gemm_ex_template<gemm_ex_testing, GEMM_EX>;
    TEST_P(gemm_ex, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            hipblas_gemm_dispatch<gemm_ex_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemm_ex);

    using gemm_batched_ex = gemm_ex_template<gemm_ex_testing, GEMM_BATCHED_EX>;
    TEST_P(gemm_batched_ex, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            hipblas_gemm_dispatch<gemm_ex_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemm_batched_ex);

    using gemm_strided_batched_ex = gemm_ex_template<gemm_ex_testing, GEMM_STRIDED_BATCHED_EX>;
    TEST_P(gemm_strided_batched_ex, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            hipblas_gemm_dispatch<gemm_ex_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemm_strided_batched_ex);

} // namespace
