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
#include "solver/testing_getrf.hpp"
#include "solver/testing_getrf_batched.hpp"
#include "solver/testing_getrf_npvt.hpp"
#include "solver/testing_getrf_npvt_batched.hpp"
#include "solver/testing_getrf_npvt_strided_batched.hpp"
#include "solver/testing_getrf_strided_batched.hpp"
#include "type_dispatch.hpp"

namespace
{
    // possible getrf test cases
    enum getrf_test_type
    {
        GETRF,
        GETRF_BATCHED,
        GETRF_STRIDED_BATCHED,
        GETRF_NPVT,
        GETRF_NPVT_BATCHED,
        GETRF_NPVT_STRIDED_BATCHED
    };

    //getrf test template
    template <template <typename...> class FILTER, getrf_test_type GETRF_TYPE>
    struct getrf_template : HipBLAS_Test<getrf_template<FILTER, GETRF_TYPE>, FILTER>
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
            return hipblas_simple_dispatch<getrf_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(GETRF_TYPE)
            {
            case GETRF:
                return !strcmp(arg.function, "getrf") || !strcmp(arg.function, "getrf_bad_arg");
            case GETRF_BATCHED:
                return !strcmp(arg.function, "getrf_batched")
                       || !strcmp(arg.function, "getrf_batched_bad_arg");
            case GETRF_STRIDED_BATCHED:
                return !strcmp(arg.function, "getrf_strided_batched")
                       || !strcmp(arg.function, "getrf_strided_batched_bad_arg");
            case GETRF_NPVT:
                return !strcmp(arg.function, "getrf_npvt")
                       || !strcmp(arg.function, "getrf_npvt_bad_arg");
            case GETRF_NPVT_BATCHED:
                return !strcmp(arg.function, "getrf_npvt_batched")
                       || !strcmp(arg.function, "getrf_npvt_batched_bad_arg");
            case GETRF_NPVT_STRIDED_BATCHED:
                return !strcmp(arg.function, "getrf_npvt_strided_batched")
                       || !strcmp(arg.function, "getrf_npvt_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            std::string name;
            if constexpr(GETRF_TYPE == GETRF)
                testname_getrf(arg, name);
            else if constexpr(GETRF_TYPE == GETRF_BATCHED)
                testname_getrf_batched(arg, name);
            else if constexpr(GETRF_TYPE == GETRF_STRIDED_BATCHED)
                testname_getrf_strided_batched(arg, name);
            else if constexpr(GETRF_TYPE == GETRF_NPVT)
                testname_getrf_npvt(arg, name);
            else if constexpr(GETRF_TYPE == GETRF_NPVT_BATCHED)
                testname_getrf_npvt_batched(arg, name);
            else if constexpr(GETRF_TYPE == GETRF_NPVT_STRIDED_BATCHED)
                testname_getrf_npvt_strided_batched(arg, name);
            return std::move(name);
        }
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct getrf_testing : hipblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct getrf_testing<
        T,
        std::enable_if_t<
            std::is_same_v<
                T,
                float> || std::is_same_v<T, double> || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>>>
        : hipblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "getrf"))
                testing_getrf<T>(arg);
            else if(!strcmp(arg.function, "getrf_bad_arg"))
                testing_getrf_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "getrf_batched"))
                testing_getrf_batched<T>(arg);
            else if(!strcmp(arg.function, "getrf_batched_bad_arg"))
                testing_getrf_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "getrf_strided_batched"))
                testing_getrf_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "getrf_strided_batched_bad_arg"))
                testing_getrf_strided_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "getrf_npvt"))
                testing_getrf_npvt<T>(arg);
            else if(!strcmp(arg.function, "getrf_npvt_bad_arg"))
                testing_getrf_npvt_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "getrf_npvt_batched"))
                testing_getrf_npvt_batched<T>(arg);
            else if(!strcmp(arg.function, "getrf_npvt_batched_bad_arg"))
                testing_getrf_npvt_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "getrf_npvt_strided_batched"))
                testing_getrf_npvt_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "getrf_npvt_strided_batched_bad_arg"))
                testing_getrf_npvt_strided_batched_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using getrf = getrf_template<getrf_testing, GETRF>;
    TEST_P(getrf, solver)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            hipblas_simple_dispatch<getrf_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(getrf);

    using getrf_batched = getrf_template<getrf_testing, GETRF_BATCHED>;
    TEST_P(getrf_batched, solver)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            hipblas_simple_dispatch<getrf_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(getrf_batched);

    using getrf_strided_batched = getrf_template<getrf_testing, GETRF_STRIDED_BATCHED>;
    TEST_P(getrf_strided_batched, solver)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            hipblas_simple_dispatch<getrf_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(getrf_strided_batched);

    using getrf_npvt = getrf_template<getrf_testing, GETRF_NPVT>;
    TEST_P(getrf_npvt, solver)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            hipblas_simple_dispatch<getrf_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(getrf_npvt);

    using getrf_npvt_batched = getrf_template<getrf_testing, GETRF_NPVT_BATCHED>;
    TEST_P(getrf_npvt_batched, solver)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            hipblas_simple_dispatch<getrf_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(getrf_npvt_batched);

    using getrf_npvt_strided_batched = getrf_template<getrf_testing, GETRF_NPVT_STRIDED_BATCHED>;
    TEST_P(getrf_npvt_strided_batched, solver)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            hipblas_simple_dispatch<getrf_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(getrf_npvt_strided_batched);

} // namespace
