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

#include "auxil/testing_set_get_matrix.hpp"
#include "auxil/testing_set_get_matrix_async.hpp"
#include "auxil/testing_set_get_vector.hpp"
#include "auxil/testing_set_get_vector_async.hpp"
#include "hipblas_data.hpp"
#include "hipblas_test.hpp"
#include "type_dispatch.hpp"

namespace
{
    // possible aux test cases
    enum aux_test_type
    {
        SG_MATRIX,
        SG_MATRIX_ASYNC,
        SG_VECTOR,
        SG_VECTOR_ASYNC
    };

    // aux test template
    template <template <typename...> class FILTER, aux_test_type AUX_TYPE>
    struct aux_template : HipBLAS_Test<aux_template<FILTER, AUX_TYPE>, FILTER>
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
            return hipblas_simple_dispatch<aux_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(AUX_TYPE)
            {
            case SG_MATRIX:
                return !strcmp(arg.function, "set_get_matrix");
            case SG_MATRIX_ASYNC:
                return !strcmp(arg.function, "set_get_matrix_async");
            case SG_VECTOR:
                return !strcmp(arg.function, "set_get_vector");
            case SG_VECTOR_ASYNC:
                return !strcmp(arg.function, "set_get_vector_async");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            std::string name;
            if constexpr(AUX_TYPE == SG_MATRIX)
                testname_set_get_matrix(arg, name);
            else if constexpr(AUX_TYPE == SG_MATRIX_ASYNC)
                testname_set_get_matrix_async(arg, name);
            else if constexpr(AUX_TYPE == SG_VECTOR)
                testname_set_get_vector(arg, name);
            else if constexpr(AUX_TYPE == SG_VECTOR_ASYNC)
                testname_set_get_vector_async(arg, name);
            return std::move(name);
        }
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct aux_testing : hipblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct aux_testing<
        T,
        std::enable_if_t<
            std::is_same_v<
                T,
                float> || std::is_same_v<T, double> || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>>>
        : hipblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "set_get_matrix"))
                testing_set_get_matrix<T>(arg);
            else if(!strcmp(arg.function, "set_get_matrix_async"))
                testing_set_get_matrix_async<T>(arg);
            else if(!strcmp(arg.function, "set_get_vector"))
                testing_set_get_vector<T>(arg);
            else if(!strcmp(arg.function, "set_get_vector_async"))
                testing_set_get_vector_async<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using set_get_matrix = aux_template<aux_testing, SG_MATRIX>;
    TEST_P(set_get_matrix, aux)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<aux_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(set_get_matrix);

    using set_get_matrix_async = aux_template<aux_testing, SG_MATRIX_ASYNC>;
    TEST_P(set_get_matrix_async, aux)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<aux_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(set_get_matrix_async);

    using set_get_vector = aux_template<aux_testing, SG_VECTOR>;
    TEST_P(set_get_vector, aux)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<aux_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(set_get_vector);

    using set_get_vector_async = aux_template<aux_testing, SG_VECTOR_ASYNC>;
    TEST_P(set_get_vector_async, aux)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<aux_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(set_get_vector_async);

} // namespace
