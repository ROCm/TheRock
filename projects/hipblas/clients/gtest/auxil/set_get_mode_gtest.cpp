/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "auxil/testing_set_get_atomics_mode.hpp"
#include "auxil/testing_set_get_math_mode.hpp"
#include "auxil/testing_set_get_pointer_mode.hpp"
#include "hipblas_data.hpp"
#include "hipblas_test.hpp"
#include "type_dispatch.hpp"

namespace
{
    // possible aux test cases
    enum aux_mode_test_type
    {
        SG_POINTER,
        SG_ATOMICS,
        SG_MATH,
    };

    // aux test template
    template <template <typename...> class FILTER, aux_mode_test_type AUX_TYPE>
    struct aux_mode_template : HipBLAS_Test<aux_mode_template<FILTER, AUX_TYPE>, FILTER>
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
            return hipblas_simple_dispatch<aux_mode_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(AUX_TYPE)
            {
            case SG_POINTER:
                return !strcmp(arg.function, "set_get_pointer_mode");
            case SG_ATOMICS:
                return !strcmp(arg.function, "set_get_atomics_mode");
            case SG_MATH:
                return !strcmp(arg.function, "set_get_math_mode");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            std::string name;
            if constexpr(AUX_TYPE == SG_POINTER)
                testname_set_get_pointer_mode(arg, name);
            else if constexpr(AUX_TYPE == SG_ATOMICS)
                testname_set_get_atomics_mode(arg, name);
            else if constexpr(AUX_TYPE == SG_MATH)
                testname_set_get_math_mode(arg, name);

            return std::move(name);
        }
    };

    template <typename...>
    struct aux_mode_testing : hipblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "set_get_pointer_mode"))
                testing_set_get_pointer_mode(arg);
            else if(!strcmp(arg.function, "set_get_atomics_mode"))
                testing_set_get_atomics_mode(arg);
            else if(!strcmp(arg.function, "set_get_math_mode"))
                testing_set_get_math_mode(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using set_get_pointer = aux_mode_template<aux_mode_testing, SG_POINTER>;
    TEST_P(set_get_pointer, aux)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(aux_mode_testing<>{}(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(set_get_pointer);

    using set_get_atomics = aux_mode_template<aux_mode_testing, SG_ATOMICS>;
    TEST_P(set_get_atomics, aux)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(aux_mode_testing<>{}(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(set_get_atomics);

    using set_get_math = aux_mode_template<aux_mode_testing, SG_MATH>;
    TEST_P(set_get_math, aux)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(aux_mode_testing<>{}(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(set_get_math);

} // namespace
