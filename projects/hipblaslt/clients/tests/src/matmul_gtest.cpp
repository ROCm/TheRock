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
#include "testing_matmul.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

#include <gtest/gtest-spi.h>

namespace
{

    // ----------------------------------------------------------------------------
    // matmul
    // ----------------------------------------------------------------------------

    struct matmul_testing : hipblaslt_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "matmul"))
                testing_matmul(arg);
            else if(!strcmp(arg.function, "matmul_bad_arg"))
                testing_matmul_bad_arg(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    struct matmul_test : RocBlasLt_Test<matmul_test, matmul_testing>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return type_filter_functor{}(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "matmul") || !strcmp(arg.function, "matmul_bad_arg");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBlasLt_TestName<matmul_test> name(arg.name);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "bad_arg";
            }
            else
            {
                name << hip_datatype_to_string(arg.a_type) << hip_datatype_to_string(arg.b_type)
                     << hip_datatype_to_string(arg.c_type) << hip_datatype_to_string(arg.d_type)
                     << hipblas_computetype_to_string(arg.compute_type);

                if(arg.activation_type != hipblaslt_activation_type::none)
                {
                    name << '_' << hipblaslt_activation_type_to_string(arg.activation_type);
                }

                if(arg.bias_vector)
                {
                    name << "_BIAS" << hipblaslt_bias_source_to_string(arg.bias_source);
                    name << "_" << hip_datatype_to_string(arg.bias_type);
                }

                if(arg.gradient)
                {
                    if(arg.use_e)
                    {
                        name << "_GRAD";
                    }
                }
                else
                {
                    if(arg.use_e)
                    {
                        name << "_AUX";
                        if(arg.aux_type != HIPBLASLT_DATATYPE_INVALID)
                            name << "_" << hip_datatype_to_string(arg.aux_type);
                    }
                }

                name << '_' << (char)std::toupper(arg.transA) << (char)std::toupper(arg.transB);

                name << '_' << arg.M[0] << '_' << arg.N[0] << '_' << arg.K[0] << '_' << arg.alpha
                     << '_' << arg.lda[0] << '_' << arg.ldb[0] << '_' << arg.beta << '_'
                     << arg.ldc[0] << '_' << arg.ldd[0];

                if(arg.use_e)
                {
                    name << '_' << arg.lde[0];
                }

                name << '_' << arg.batch_count;

                if(arg.scaleA == hipblaslt_scaling_format::Scalar)
                    name << "_SA";
                else if(arg.scaleA == hipblaslt_scaling_format::Vector)
                    name << "_SAV";
                else if(arg.scaleA == hipblaslt_scaling_format::Block_32_UE8M0)
                    name << "_SAMX_32_UE8M0";
                else if(arg.scaleA == hipblaslt_scaling_format::Block_32_UE8M0_32_8_EXT)
                    name << "_SAMX_32_UE8M0_32_8";

                if(arg.scaleB == hipblaslt_scaling_format::Scalar)
                    name << "_SB";
                else if(arg.scaleB == hipblaslt_scaling_format::Vector)
                    name << "_SBV";
                else if(arg.scaleB == hipblaslt_scaling_format::Block_32_UE8M0)
                    name << "_SBMX_32_UE8M0";
                else if(arg.scaleB == hipblaslt_scaling_format::Block_32_UE8M0_32_8_EXT)
                    name << "_SBMX_32_UE8M0_32_8";

                if(arg.scaleC)
                    name << "_SC";

                if(arg.scaleD)
                    name << "_SD";

                if(arg.scaleE)
                    name << "_SAux";

                if(arg.scaleAlpha_vector)
                    name << "_SAV";

                if(arg.amaxScaleA)
                    name << "_ASA";

                if(arg.amaxScaleB)
                    name << "_ASB";

                if(arg.amaxD)
                    name << "_AMaxD";

                if(arg.grouped_gemm > 0)
                    name << "_GG" << arg.grouped_gemm;

                if(arg.c_equal_d)
                    name << "_C_EQUAL_D";
                // grouped gemm only supports ext
                if(arg.use_ext || arg.grouped_gemm > 0)
                    name << "_APIExt";
                if(arg.use_ext_setproblem)
                    name << "_APIExtSet";
                if(arg.algo_method == 2)
                    name << "_APIAlgoIndex";
                else if(arg.algo_method == 1)
                    name << "_APIFindAllAlgo";
                if(arg.use_user_args)
                    name << "_UserArgs";
                if(arg.gsu_vector[0])
                    name << "_GSU" << (int)arg.gsu_vector[0];
                if(arg.wgm_vector[0])
                    name << "_WGM" << (int)arg.wgm_vector[0];
            }

            return std::move(name);
        }
    };

    TEST_P(matmul_test, matmul)
    {
        RUN_TEST_ON_THREADS_STREAMS(matmul_testing{}(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(matmul_test);

#ifdef HIPBLASLT_USE_ROCROLLER
    // ----------------------------------------------------------------------------
    // rocRoller
    // ----------------------------------------------------------------------------

    struct rocroller_predicate_testing : hipblaslt_test_valid
    {
        void operator()(const Arguments& arg)
        {
            testing_matmul(arg);
        }
    };

    struct rocroller_predicate_test
        : RocBlasLt_Test<rocroller_predicate_test, rocroller_predicate_testing>
    {
        static bool type_filter(const Arguments& arg)
        {
            return type_filter_functor{}(arg);
        }

        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "rocroller_predicate");
        }

        static std::string name_suffix(const Arguments& arg)
        {
            return matmul_test::name_suffix(arg);
        }
    };

    TEST_P(rocroller_predicate_test, unrollXYK)
    {
        // rocRoller has predicates that check the dimensions (M/N/K) must be
        // multiples of the work group sizes. This test set the K dimension
        // to not be a multiple, and thus we shall see failure.
        EXPECT_FATAL_FAILURE(rocroller_predicate_testing{}(GetParam()), "NO solution found!");
    }
    INSTANTIATE_TEST_CATEGORIES(rocroller_predicate_test);
#endif

} // namespace
