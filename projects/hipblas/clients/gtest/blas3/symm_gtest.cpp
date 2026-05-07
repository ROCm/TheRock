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

#include "blas3/testing_symm.hpp"
#include "blas3/testing_symm_batched.hpp"
#include "blas3/testing_symm_strided_batched.hpp"
#include "hipblas_data.hpp"
#include "hipblas_test.hpp"
#include "type_dispatch.hpp"

namespace
{
    // possible symm test cases
    enum symm_test_type
    {
        SYMM,
        SYMM_BATCHED,
        SYMM_STRIDED_BATCHED,
    };

    // symm test template
    template <template <typename...> class FILTER, symm_test_type SYMM_TYPE>
    struct symm_template : HipBLAS_Test<symm_template<FILTER, SYMM_TYPE>, FILTER>
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
            return hipblas_simple_dispatch<symm_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(SYMM_TYPE)
            {
            case SYMM:
                return !strcmp(arg.function, "symm") || !strcmp(arg.function, "symm_bad_arg");
            case SYMM_BATCHED:
                return !strcmp(arg.function, "symm_batched")
                       || !strcmp(arg.function, "symm_batched_bad_arg");
            case SYMM_STRIDED_BATCHED:
                return !strcmp(arg.function, "symm_strided_batched")
                       || !strcmp(arg.function, "symm_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            std::string name;
            if constexpr(SYMM_TYPE == SYMM)
                testname_symm(arg, name);
            else if constexpr(SYMM_TYPE == SYMM_BATCHED)
                testname_symm_batched(arg, name);
            else if constexpr(SYMM_TYPE == SYMM_STRIDED_BATCHED)
                testname_symm_strided_batched(arg, name);
            return std::move(name);
        }
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct symm_testing : hipblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct symm_testing<
        T,
        std::enable_if_t<
            std::is_same_v<
                T,
                float> || std::is_same_v<T, double> || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>>>
        : hipblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "symm"))
                testing_symm<T>(arg);
            else if(!strcmp(arg.function, "symm_bad_arg"))
                testing_symm_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "symm_batched"))
                testing_symm_batched<T>(arg);
            else if(!strcmp(arg.function, "symm_batched_bad_arg"))
                testing_symm_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "symm_strided_batched"))
                testing_symm_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "symm_strided_batched_bad_arg"))
                testing_symm_strided_batched_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using symm = symm_template<symm_testing, SYMM>;
    TEST_P(symm, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<symm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(symm);

    using symm_batched = symm_template<symm_testing, SYMM_BATCHED>;
    TEST_P(symm_batched, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<symm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(symm_batched);

    using symm_strided_batched = symm_template<symm_testing, SYMM_STRIDED_BATCHED>;
    TEST_P(symm_strided_batched, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<symm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(symm_strided_batched);

} // namespace
