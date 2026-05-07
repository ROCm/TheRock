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
#include "blas1_gtest.hpp"

#include "blas1/testing_copy.hpp"
#include "blas1/testing_copy_batched.hpp"
#include "blas1/testing_copy_strided_batched.hpp"
#include "hipblas_data.hpp"
#include "hipblas_test.hpp"
#include "type_dispatch.hpp"

namespace
{
    // ----------------------------------------------------------------------------
    // BLAS1 testing template
    // ----------------------------------------------------------------------------
    template <template <typename...> class FILTER, blas1 BLAS1>
    struct copy_test_template : public HipBLAS_Test<copy_test_template<FILTER, BLAS1>, FILTER>
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
            return hipblas_blas1_dispatch<copy_test_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg);

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            std::string name;
            if constexpr(BLAS1 == blas1::copy)
                testname_copy(arg, name);
            else if constexpr(BLAS1 == blas1::copy_batched)
                testname_copy_batched(arg, name);
            else if constexpr(BLAS1 == blas1::copy_strided_batched)
                testname_copy_strided_batched(arg, name);
            return std::move(name);
        }
    };

    // This tells whether the BLAS1 tests are enabled
    template <blas1 BLAS1, typename Ti, typename To, typename Tc>
    using copy_enabled = std::integral_constant<
        bool,
        ((BLAS1 == blas1::copy || BLAS1 == blas1::copy_batched
          || BLAS1 == blas1::copy_strided_batched)
         && std::is_same_v<
             Ti,
             To> && std::is_same_v<To, Tc> && (std::is_same_v<Ti, std::complex<float>> || std::is_same_v<Ti, std::complex<double>> || std::is_same_v<Ti, float> || std::is_same_v<Ti, double>))>;

// Creates tests for one of the BLAS 1 functions
// ARG passes 1-3 template arguments to the testing_* function
#define BLAS1_TESTING(NAME, ARG)                                                              \
    struct blas1_##NAME                                                                       \
    {                                                                                         \
        template <typename Ti, typename To = Ti, typename Tc = To, typename = void>           \
        struct testing : hipblas_test_invalid                                                 \
        {                                                                                     \
        };                                                                                    \
                                                                                              \
        template <typename Ti, typename To, typename Tc>                                      \
        struct testing<Ti, To, Tc, std::enable_if_t<copy_enabled<blas1::NAME, Ti, To, Tc>{}>> \
            : hipblas_test_valid                                                              \
        {                                                                                     \
            void operator()(const Arguments& arg)                                             \
            {                                                                                 \
                if(!strcmp(arg.function, #NAME))                                              \
                    testing_##NAME<ARG(Ti, To, Tc)>(arg);                                     \
                else if(!strcmp(arg.function, #NAME "_bad_arg"))                              \
                    testing_##NAME##_bad_arg<ARG(Ti, To, Tc)>(arg);                           \
                else                                                                          \
                    FAIL() << "Internal error: Test called with unknown function: "           \
                           << arg.function;                                                   \
            }                                                                                 \
        };                                                                                    \
    };                                                                                        \
                                                                                              \
    using NAME = copy_test_template<blas1_##NAME::template testing, blas1::NAME>;             \
                                                                                              \
    template <>                                                                               \
    inline bool NAME::function_filter(const Arguments& arg)                                   \
    {                                                                                         \
        return !strcmp(arg.function, #NAME) || !strcmp(arg.function, #NAME "_bad_arg");       \
    }                                                                                         \
                                                                                              \
    TEST_P(NAME, blas1)                                                                       \
    {                                                                                         \
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(                                             \
            hipblas_blas1_dispatch<blas1_##NAME::template testing>(GetParam()));              \
    }                                                                                         \
                                                                                              \
    INSTANTIATE_TEST_CATEGORIES(NAME)

#define ARG1(Ti, To, Tc) Ti
#define ARG2(Ti, To, Tc) Ti, To
#define ARG3(Ti, To, Tc) Ti, To, Tc

    BLAS1_TESTING(copy, ARG1)
    BLAS1_TESTING(copy_batched, ARG1)
    BLAS1_TESTING(copy_strided_batched, ARG1)

} // namespace
