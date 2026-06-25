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

#include "blas2/testing_gemv.hpp"
#include "blas2/testing_gemv_batched.hpp"
#include "blas2/testing_gemv_strided_batched.hpp"
#include "hipblas_data.hpp"
#include "hipblas_test.hpp"
#include "type_dispatch.hpp"

namespace
{
    // possible gemv test cases
    enum gemv_test_type
    {
        GEMV,
        GEMV_BATCHED,
        GEMV_STRIDED_BATCHED,
    };

    //gemv test template
    template <template <typename...> class FILTER, gemv_test_type GEMV_TYPE>
    struct gemv_template : HipBLAS_Test<gemv_template<FILTER, GEMV_TYPE>, FILTER>
    {
        template <typename... T>
        struct type_filter_functor
        {
            bool operator()(const Arguments& args)
            {
                // additional global filters applied first
                if(!hipblas_client_global_filters(args))
                    return false;

#if defined(__HIP_PLATFORM_NVCC__) && CUBLAS_VERSION < 110700
                // avoid gemvBatched/gemvStridedBatched tests with cuBLAS older than 11.7.0
                if(!strcmp(args.function, "gemv_batched")
                   || !strcmp(args.function, "gemv_batched_bad_arg")
                   || !strcmp(args.function, "gemv_strided_batched")
                   || !strcmp(args.function, "gemv_strided_batched_bad_arg"))
                    return false;
#endif

                // type filters
                return static_cast<bool>(FILTER<T...>{});
            }
        };

        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return hipblas_simple_dispatch<gemv_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(GEMV_TYPE)
            {
            case GEMV:
                return !strcmp(arg.function, "gemv") || !strcmp(arg.function, "gemv_bad_arg");
            case GEMV_BATCHED:
                return !strcmp(arg.function, "gemv_batched")
                       || !strcmp(arg.function, "gemv_batched_bad_arg");
            case GEMV_STRIDED_BATCHED:
                return !strcmp(arg.function, "gemv_strided_batched")
                       || !strcmp(arg.function, "gemv_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            std::string name;
            if constexpr(GEMV_TYPE == GEMV)
                testname_gemv(arg, name);
            else if constexpr(GEMV_TYPE == GEMV_BATCHED)
                testname_gemv_batched(arg, name);
            else if constexpr(GEMV_TYPE == GEMV_STRIDED_BATCHED)
                testname_gemv_strided_batched(arg, name);
            return std::move(name);
        }
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct gemv_testing : hipblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct gemv_testing<
        T,
        std::enable_if_t<
            std::is_same_v<
                T,
                float> || std::is_same_v<T, double> || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>>>
        : hipblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "gemv"))
                testing_gemv<T>(arg);
            else if(!strcmp(arg.function, "gemv_bad_arg"))
                testing_gemv_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "gemv_batched"))
                testing_gemv_batched<T>(arg);
            else if(!strcmp(arg.function, "gemv_batched_bad_arg"))
                testing_gemv_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "gemv_strided_batched"))
                testing_gemv_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "gemv_strided_batched_bad_arg"))
                testing_gemv_strided_batched_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using gemv = gemv_template<gemv_testing, GEMV>;
    TEST_P(gemv, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<gemv_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemv);

    using gemv_batched = gemv_template<gemv_testing, GEMV_BATCHED>;
    TEST_P(gemv_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<gemv_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemv_batched);

    using gemv_strided_batched = gemv_template<gemv_testing, GEMV_STRIDED_BATCHED>;
    TEST_P(gemv_strided_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<gemv_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gemv_strided_batched);

} // namespace
