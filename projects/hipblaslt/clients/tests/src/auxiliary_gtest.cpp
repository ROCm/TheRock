/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include "hipblaslt_data.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_test.hpp"
#include "testing_auxiliary.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{

    // ----------------------------------------------------------------------------
    // aux
    // ----------------------------------------------------------------------------

    struct aux_testing : hipblaslt_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "aux_handle_init_bad_arg"))
                testing_aux_handle_init_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_handle_destroy_bad_arg"))
                testing_aux_handle_destroy_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_handle"))
                testing_aux_handle(arg);
            else if(!strcmp(arg.function, "aux_mat_init_bad_arg"))
                testing_aux_mat_init_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_mat_init_arg"))
                testing_aux_mat_init(arg);
            else if(!strcmp(arg.function, "aux_mat_destroy_bad_arg"))
                testing_aux_mat_destroy_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_mat_set_attr_bad_arg"))
                testing_aux_mat_set_attr_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_mat_get_attr_bad_arg"))
                testing_aux_mat_get_attr_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_mat_set_get_attr"))
                testing_aux_mat_set_get_attr(arg);
            else if(!strcmp(arg.function, "aux_matmul_init_bad_arg"))
                testing_aux_matmul_init_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_matmul_init"))
                testing_aux_matmul_init(arg);
            else if(!strcmp(arg.function, "aux_matmul_set_attr_bad_arg"))
                testing_aux_matmul_set_attr_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_matmul_get_attr_bad_arg"))
                testing_aux_matmul_get_attr_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_matmul_set_get_attr"))
                testing_aux_matmul_set_get_attr(arg);
            else if(!strcmp(arg.function, "aux_matmul_pref_get_attr_bad_arg"))
                testing_aux_matmul_pref_get_attr_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_matmul_pref_get_attr"))
                testing_aux_matmul_pref_get_attr(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_init_bad_arg"))
                testing_aux_matmul_alg_init_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_init"))
                testing_aux_matmul_alg_init(arg);
            else if(!strcmp(arg.function, "aux_get_sol_with_null_biasaddr"))
                testing_aux_get_sol_with_null_biasaddr(arg);
            else if(!strcmp(arg.function, "aux_get_sol_with_zero_alpha_null_a_b"))
                testing_aux_get_sol_with_zero_alpha_null_a_b(arg);
            else if(!strcmp(arg.function, "aux_get_sol_with_zero_alpha_null_a_b_ext"))
                testing_aux_get_sol_with_zero_alpha_null_a_b_ext(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_get_attr_bad_arg"))
                testing_aux_matmul_alg_get_attr_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_matmul_plan_init_bad_arg"))
                testing_aux_matmul_pref_init_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_matmul_plan_init"))
                testing_aux_matmul_pref_init(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_null_matmul"))
                testing_aux_matmul_alg_null_matmul(arg);
            else if(!strcmp(arg.function, "aux_matmul_bad_ws_size"))
                testing_aux_matmul_bad_ws_size(arg);
#ifdef CODE_COVERAGE
            else if(!strcmp(arg.function, "aux_auxiliary_func"))
                testing_aux_auxiliary_func(arg);
            else if(!strcmp(arg.function, "aux_float8_func"))
                testing_aux_float8_func(arg);
            else if(!strcmp(arg.function, "aux_hipblaslt_ext_op_func"))
                testing_aux_hipblaslt_ext_op_func(arg);
            else if(!strcmp(arg.function, "aux_rocblaslt_utility_func"))
                testing_aux_rocblaslt_utility_func(arg);
            else if(!strcmp(arg.function, "aux_status_func"))
                testing_aux_status_func(arg);
            else if(!strcmp(arg.function, "aux_hipblaslt_func"))
                testing_aux_hipblaslt_func(arg);
            else if(!strcmp(arg.function, "aux_tensile_host_func"))
                testing_aux_tensile_host_func(arg);
            else if(!strcmp(arg.function, "aux_tuple_helper_equal_func"))
                testing_aux_tuple_helper_equal_func(arg);
            else if(!strcmp(arg.function, "aux_rocblaslt_rocroller_host_func"))
                testing_aux_rocblaslt_rocroller_host_func(arg);
#endif
            else if(!strcmp(arg.function, "aux_mat_copy"))
                testing_aux_mat_copy(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    struct aux_test : RocBlasLt_Test<aux_test, aux_testing>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return type_filter_functor{}(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "aux_handle_init_bad_arg")
                   || !strcmp(arg.function, "aux_handle_destroy_bad_arg")
                   || !strcmp(arg.function, "aux_handle")
                   || !strcmp(arg.function, "aux_mat_init_bad_arg")
                   || !strcmp(arg.function, "aux_mat_init_arg")
                   || !strcmp(arg.function, "aux_mat_destroy_bad_arg")
                   || !strcmp(arg.function, "aux_mat_set_attr_bad_arg")
                   || !strcmp(arg.function, "aux_mat_get_attr_bad_arg")
                   || !strcmp(arg.function, "aux_mat_set_get_attr")
                   || !strcmp(arg.function, "aux_matmul_init_bad_arg")
                   || !strcmp(arg.function, "aux_matmul_init")
                   || !strcmp(arg.function, "aux_matmul_set_attr_bad_arg")
                   || !strcmp(arg.function, "aux_matmul_get_attr_bad_arg")
                   || !strcmp(arg.function, "aux_matmul_set_get_attr")
                   || !strcmp(arg.function, "aux_matmul_alg_init_bad_arg")
                   || !strcmp(arg.function, "aux_matmul_alg_init")
                   || !strcmp(arg.function, "aux_get_sol_with_null_biasaddr")
                   || !strcmp(arg.function, "aux_get_sol_with_zero_alpha_null_a_b")
                   || !strcmp(arg.function, "aux_get_sol_with_zero_alpha_null_a_b_ext")
                   || !strcmp(arg.function, "aux_matmul_alg_get_attr_bad_arg")
                   || !strcmp(arg.function, "aux_matmul_plan_init_bad_arg")
                   || !strcmp(arg.function, "aux_matmul_plan_init")
                   || !strcmp(arg.function, "aux_matmul_alg_null_matmul")
                   || !strcmp(arg.function, "aux_matmul_bad_ws_size")
                   || !strcmp(arg.function, "aux_matmul_pref_get_attr_bad_arg")
                   || !strcmp(arg.function, "aux_matmul_pref_get_attr")
#ifdef CODE_COVERAGE
                   || !strcmp(arg.function, "aux_auxiliary_func")
                   || !strcmp(arg.function, "aux_float8_func")
                   || !strcmp(arg.function, "aux_hipblaslt_ext_op_func")
                   || !strcmp(arg.function, "aux_rocblaslt_utility_func")
                   || !strcmp(arg.function, "aux_status_func")
                   || !strcmp(arg.function, "aux_hipblaslt_func")
                   || !strcmp(arg.function, "aux_tensile_host_func")
                   || !strcmp(arg.function, "aux_tuple_helper_equal_func")
                   || !strcmp(arg.function, "aux_rocblaslt_rocroller_host_func")
#endif
                   || !strcmp(arg.function, "aux_mat_copy");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBlasLt_TestName<aux_test> name(arg.name);

            name << hip_datatype_to_string(arg.a_type);

            return std::move(name);
        }
    };

    TEST_P(aux_test, conversion)
    {
        RUN_TEST_ON_THREADS_STREAMS(aux_testing{}(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(aux_test);

} // namespace
