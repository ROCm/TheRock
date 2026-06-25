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

#include "blas2/testing_ger.hpp"
#include "blas2/testing_ger_batched.hpp"
#include "blas2/testing_ger_strided_batched.hpp"
#include "hipblas_data.hpp"
#include "hipblas_test.hpp"
#include "type_dispatch.hpp"

namespace
{
    // possible ger test cases
    enum ger_test_type
    {
        GER,
        GERU,
        GERC,
        GER_BATCHED,
        GERU_BATCHED,
        GERC_BATCHED,
        GER_STRIDED_BATCHED,
        GERU_STRIDED_BATCHED,
        GERC_STRIDED_BATCHED
    };

    //ger test template
    template <template <typename...> class FILTER, ger_test_type GER_TYPE>
    struct ger_template : HipBLAS_Test<ger_template<FILTER, GER_TYPE>, FILTER>
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
            return hipblas_simple_dispatch<ger_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(GER_TYPE)
            {
            case GER:
                return !strcmp(arg.function, "ger") || !strcmp(arg.function, "ger_bad_arg");
            case GER_BATCHED:
                return !strcmp(arg.function, "ger_batched")
                       || !strcmp(arg.function, "ger_batched_bad_arg");
            case GER_STRIDED_BATCHED:
                return !strcmp(arg.function, "ger_strided_batched")
                       || !strcmp(arg.function, "ger_strided_batched_bad_arg");
            case GERU:
                return !strcmp(arg.function, "geru") || !strcmp(arg.function, "geru_bad_arg");
            case GERU_BATCHED:
                return !strcmp(arg.function, "geru_batched")
                       || !strcmp(arg.function, "geru_batched_bad_arg");
            case GERU_STRIDED_BATCHED:
                return !strcmp(arg.function, "geru_strided_batched")
                       || !strcmp(arg.function, "geru_strided_batched_bad_arg");
            case GERC:
                return !strcmp(arg.function, "gerc") || !strcmp(arg.function, "gerc_bad_arg");
            case GERC_BATCHED:
                return !strcmp(arg.function, "gerc_batched")
                       || !strcmp(arg.function, "gerc_batched_bad_arg");
            case GERC_STRIDED_BATCHED:
                return !strcmp(arg.function, "gerc_strided_batched")
                       || !strcmp(arg.function, "gerc_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            std::string name;
            if constexpr(GER_TYPE == GER || GER_TYPE == GERU || GER_TYPE == GERC)
                testname_ger(arg, name);
            else if constexpr(GER_TYPE == GER_BATCHED || GER_TYPE == GERU_BATCHED
                              || GER_TYPE == GERC_BATCHED)
                testname_ger_batched(arg, name);
            else if constexpr(GER_TYPE == GER_STRIDED_BATCHED || GER_TYPE == GERU_STRIDED_BATCHED
                              || GER_TYPE == GERC_STRIDED_BATCHED)
                testname_ger_strided_batched(arg, name);
            return std::move(name);
        }
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct ger_testing : hipblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct ger_testing<T, std::enable_if_t<(std::is_same_v<T, float> || std::is_same_v<T, double>)>>
        : hipblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "ger"))
                testing_ger<T, false>(arg);
            else if(!strcmp(arg.function, "ger_bad_arg"))
                testing_ger_bad_arg<T, false>(arg);
            else if(!strcmp(arg.function, "ger_batched"))
                testing_ger_batched<T, false>(arg);
            else if(!strcmp(arg.function, "ger_batched_bad_arg"))
                testing_ger_batched_bad_arg<T, false>(arg);
            else if(!strcmp(arg.function, "ger_strided_batched"))
                testing_ger_strided_batched<T, false>(arg);
            else if(!strcmp(arg.function, "ger_strided_batched_bad_arg"))
                testing_ger_strided_batched_bad_arg<T, false>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    template <typename, typename = void>
    struct geru_testing : hipblas_test_invalid
    {
    };

    template <typename T>
    struct geru_testing<
        T,
        std::enable_if_t<(
            std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>)>>
        : hipblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "geru"))
                testing_ger<T, false>(arg);
            else if(!strcmp(arg.function, "geru_bad_arg"))
                testing_ger_bad_arg<T, false>(arg);
            else if(!strcmp(arg.function, "geru_batched"))
                testing_ger_batched<T, false>(arg);
            else if(!strcmp(arg.function, "geru_batched_bad_arg"))
                testing_ger_batched_bad_arg<T, false>(arg);
            else if(!strcmp(arg.function, "geru_strided_batched"))
                testing_ger_strided_batched<T, false>(arg);
            else if(!strcmp(arg.function, "geru_strided_batched_bad_arg"))
                testing_ger_strided_batched_bad_arg<T, false>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    template <typename, typename = void>
    struct gerc_testing : hipblas_test_invalid
    {
    };

    template <typename T>
    struct gerc_testing<
        T,
        std::enable_if_t<(
            std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>)>>
        : hipblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "gerc"))
                testing_ger<T, true>(arg);
            else if(!strcmp(arg.function, "gerc_bad_arg"))
                testing_ger_bad_arg<T, true>(arg);
            else if(!strcmp(arg.function, "gerc_batched"))
                testing_ger_batched<T, true>(arg);
            else if(!strcmp(arg.function, "gerc_batched_bad_arg"))
                testing_ger_batched_bad_arg<T, true>(arg);
            else if(!strcmp(arg.function, "gerc_strided_batched"))
                testing_ger_strided_batched<T, true>(arg);
            else if(!strcmp(arg.function, "gerc_strided_batched_bad_arg"))
                testing_ger_strided_batched_bad_arg<T, true>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using ger = ger_template<ger_testing, GER>;
    TEST_P(ger, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<ger_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(ger);

    using ger_batched = ger_template<ger_testing, GER_BATCHED>;
    TEST_P(ger_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<ger_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(ger_batched);

    using ger_strided_batched = ger_template<ger_testing, GER_STRIDED_BATCHED>;
    TEST_P(ger_strided_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<ger_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(ger_strided_batched);

    using geru = ger_template<geru_testing, GERU>;
    TEST_P(geru, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<geru_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(geru);

    using geru_batched = ger_template<geru_testing, GERU_BATCHED>;
    TEST_P(geru_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<geru_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(geru_batched);

    using geru_strided_batched = ger_template<geru_testing, GERU_STRIDED_BATCHED>;
    TEST_P(geru_strided_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<geru_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(geru_strided_batched);

    using gerc = ger_template<gerc_testing, GERC>;
    TEST_P(gerc, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<gerc_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gerc);

    using gerc_batched = ger_template<gerc_testing, GERC_BATCHED>;
    TEST_P(gerc_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<gerc_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gerc_batched);

    using gerc_strided_batched = ger_template<gerc_testing, GERC_STRIDED_BATCHED>;
    TEST_P(gerc_strided_batched, blas2)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<gerc_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(gerc_strided_batched);

} // namespace
