/* ************************************************************************
 * Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */

#include "hipblas.hpp"

#include "argument_model.hpp"
#include "hipblas_data.hpp"
#include "hipblas_datatype2string.hpp"
#include "hipblas_parse_data.hpp"
#include "hipblas_test.hpp"
#include "test_cleanup.hpp"
#include "type_dispatch.hpp"
#include "utility.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
// aux
#include "auxil/testing_set_get_matrix.hpp"
#include "auxil/testing_set_get_matrix_async.hpp"
#include "auxil/testing_set_get_vector.hpp"
#include "auxil/testing_set_get_vector_async.hpp"
// blas1
#include "blas1/testing_asum.hpp"
#include "blas1/testing_asum_batched.hpp"
#include "blas1/testing_asum_strided_batched.hpp"
#include "blas1/testing_axpy.hpp"
#include "blas1/testing_axpy_batched.hpp"
#include "blas1/testing_axpy_strided_batched.hpp"
#include "blas1/testing_copy.hpp"
#include "blas1/testing_copy_batched.hpp"
#include "blas1/testing_copy_strided_batched.hpp"
#include "blas1/testing_dot.hpp"
#include "blas1/testing_dot_batched.hpp"
#include "blas1/testing_dot_strided_batched.hpp"
#include "blas1/testing_iamax_iamin.hpp"
#include "blas1/testing_iamax_iamin_batched.hpp"
#include "blas1/testing_iamax_iamin_strided_batched.hpp"
#include "blas1/testing_nrm2.hpp"
#include "blas1/testing_nrm2_batched.hpp"
#include "blas1/testing_nrm2_strided_batched.hpp"
#include "blas1/testing_rot.hpp"
#include "blas1/testing_rot_batched.hpp"
#include "blas1/testing_rot_strided_batched.hpp"
#include "blas1/testing_rotg.hpp"
#include "blas1/testing_rotg_batched.hpp"
#include "blas1/testing_rotg_strided_batched.hpp"
#include "blas1/testing_rotm.hpp"
#include "blas1/testing_rotm_batched.hpp"
#include "blas1/testing_rotm_strided_batched.hpp"
#include "blas1/testing_rotmg.hpp"
#include "blas1/testing_rotmg_batched.hpp"
#include "blas1/testing_rotmg_strided_batched.hpp"
#include "blas1/testing_scal.hpp"
#include "blas1/testing_scal_batched.hpp"
#include "blas1/testing_scal_strided_batched.hpp"
#include "blas1/testing_swap.hpp"
#include "blas1/testing_swap_batched.hpp"
#include "blas1/testing_swap_strided_batched.hpp"
// blas2
#include "blas2/testing_gbmv.hpp"
#include "blas2/testing_gbmv_batched.hpp"
#include "blas2/testing_gbmv_strided_batched.hpp"
#include "blas2/testing_gemv.hpp"
#include "blas2/testing_gemv_batched.hpp"
#include "blas2/testing_gemv_strided_batched.hpp"
#include "blas2/testing_ger.hpp"
#include "blas2/testing_ger_batched.hpp"
#include "blas2/testing_ger_strided_batched.hpp"
#include "blas2/testing_hbmv.hpp"
#include "blas2/testing_hbmv_batched.hpp"
#include "blas2/testing_hbmv_strided_batched.hpp"
#include "blas2/testing_hemv.hpp"
#include "blas2/testing_hemv_batched.hpp"
#include "blas2/testing_hemv_strided_batched.hpp"
#include "blas2/testing_her.hpp"
#include "blas2/testing_her2.hpp"
#include "blas2/testing_her2_batched.hpp"
#include "blas2/testing_her2_strided_batched.hpp"
#include "blas2/testing_her_batched.hpp"
#include "blas2/testing_her_strided_batched.hpp"
#include "blas2/testing_hpmv.hpp"
#include "blas2/testing_hpmv_batched.hpp"
#include "blas2/testing_hpmv_strided_batched.hpp"
#include "blas2/testing_hpr.hpp"
#include "blas2/testing_hpr2.hpp"
#include "blas2/testing_hpr2_batched.hpp"
#include "blas2/testing_hpr2_strided_batched.hpp"
#include "blas2/testing_hpr_batched.hpp"
#include "blas2/testing_hpr_strided_batched.hpp"
#include "blas2/testing_sbmv.hpp"
#include "blas2/testing_sbmv_batched.hpp"
#include "blas2/testing_sbmv_strided_batched.hpp"
#include "blas2/testing_spmv.hpp"
#include "blas2/testing_spmv_batched.hpp"
#include "blas2/testing_spmv_strided_batched.hpp"
#include "blas2/testing_spr.hpp"
#include "blas2/testing_spr2.hpp"
#include "blas2/testing_spr2_batched.hpp"
#include "blas2/testing_spr2_strided_batched.hpp"
#include "blas2/testing_spr_batched.hpp"
#include "blas2/testing_spr_strided_batched.hpp"
#include "blas2/testing_symv.hpp"
#include "blas2/testing_symv_batched.hpp"
#include "blas2/testing_symv_strided_batched.hpp"
#include "blas2/testing_syr.hpp"
#include "blas2/testing_syr2.hpp"
#include "blas2/testing_syr2_batched.hpp"
#include "blas2/testing_syr2_strided_batched.hpp"
#include "blas2/testing_syr_batched.hpp"
#include "blas2/testing_syr_strided_batched.hpp"
#include "blas2/testing_tbmv.hpp"
#include "blas2/testing_tbmv_batched.hpp"
#include "blas2/testing_tbmv_strided_batched.hpp"
#include "blas2/testing_tbsv.hpp"
#include "blas2/testing_tbsv_batched.hpp"
#include "blas2/testing_tbsv_strided_batched.hpp"
#include "blas2/testing_tpmv.hpp"
#include "blas2/testing_tpmv_batched.hpp"
#include "blas2/testing_tpmv_strided_batched.hpp"
#include "blas2/testing_tpsv.hpp"
#include "blas2/testing_tpsv_batched.hpp"
#include "blas2/testing_tpsv_strided_batched.hpp"
#include "blas2/testing_trmv.hpp"
#include "blas2/testing_trmv_batched.hpp"
#include "blas2/testing_trmv_strided_batched.hpp"
#include "blas2/testing_trsv.hpp"
#include "blas2/testing_trsv_batched.hpp"
#include "blas2/testing_trsv_strided_batched.hpp"
// blas3
#include "blas3/testing_dgmm.hpp"
#include "blas3/testing_dgmm_batched.hpp"
#include "blas3/testing_dgmm_strided_batched.hpp"
#include "blas3/testing_geam.hpp"
#include "blas3/testing_geam_batched.hpp"
#include "blas3/testing_geam_strided_batched.hpp"
#include "blas3/testing_gemm.hpp"
#include "blas3/testing_gemm_batched.hpp"
#include "blas3/testing_gemm_strided_batched.hpp"
#include "blas3/testing_hemm.hpp"
#include "blas3/testing_hemm_batched.hpp"
#include "blas3/testing_hemm_strided_batched.hpp"
#include "blas3/testing_her2k.hpp"
#include "blas3/testing_her2k_batched.hpp"
#include "blas3/testing_her2k_strided_batched.hpp"
#include "blas3/testing_herk.hpp"
#include "blas3/testing_herk_batched.hpp"
#include "blas3/testing_herk_strided_batched.hpp"
#include "blas3/testing_herkx.hpp"
#include "blas3/testing_herkx_batched.hpp"
#include "blas3/testing_herkx_strided_batched.hpp"
#include "blas3/testing_symm.hpp"
#include "blas3/testing_symm_batched.hpp"
#include "blas3/testing_symm_strided_batched.hpp"
#include "blas3/testing_syr2k.hpp"
#include "blas3/testing_syr2k_batched.hpp"
#include "blas3/testing_syr2k_strided_batched.hpp"
#include "blas3/testing_syrk.hpp"
#include "blas3/testing_syrk_batched.hpp"
#include "blas3/testing_syrk_strided_batched.hpp"
#include "blas3/testing_syrkx.hpp"
#include "blas3/testing_syrkx_batched.hpp"
#include "blas3/testing_syrkx_strided_batched.hpp"
#include "blas3/testing_trmm.hpp"
#include "blas3/testing_trmm_batched.hpp"
#include "blas3/testing_trmm_strided_batched.hpp"
#include "blas3/testing_trsm.hpp"
#include "blas3/testing_trsm_batched.hpp"
#include "blas3/testing_trsm_strided_batched.hpp"
#include "blas3/testing_trtri.hpp"
#include "blas3/testing_trtri_batched.hpp"
#include "blas3/testing_trtri_strided_batched.hpp"
#include "syrkx_reference.hpp"
// blas_ex
#include "blas_ex/testing_axpy_batched_ex.hpp"
#include "blas_ex/testing_axpy_ex.hpp"
#include "blas_ex/testing_axpy_strided_batched_ex.hpp"
#include "blas_ex/testing_dot_batched_ex.hpp"
#include "blas_ex/testing_dot_ex.hpp"
#include "blas_ex/testing_dot_strided_batched_ex.hpp"
#include "blas_ex/testing_gemm_batched_ex.hpp"
#include "blas_ex/testing_gemm_ex.hpp"
#include "blas_ex/testing_gemm_strided_batched_ex.hpp"
#include "blas_ex/testing_herk_ex.hpp"
#include "blas_ex/testing_nrm2_batched_ex.hpp"
#include "blas_ex/testing_nrm2_ex.hpp"
#include "blas_ex/testing_nrm2_strided_batched_ex.hpp"
#include "blas_ex/testing_rot_batched_ex.hpp"
#include "blas_ex/testing_rot_ex.hpp"
#include "blas_ex/testing_rot_strided_batched_ex.hpp"
#include "blas_ex/testing_scal_batched_ex.hpp"
#include "blas_ex/testing_scal_ex.hpp"
#include "blas_ex/testing_scal_strided_batched_ex.hpp"
#include "blas_ex/testing_syrk_ex.hpp"
#include "blas_ex/testing_trsm_batched_ex.hpp"
#include "blas_ex/testing_trsm_ex.hpp"
#include "blas_ex/testing_trsm_strided_batched_ex.hpp"
// solver functions
#ifdef __HIP_PLATFORM_SOLVER__
#include "solver/testing_gels.hpp"
#include "solver/testing_gels_batched.hpp"
#include "solver/testing_gels_strided_batched.hpp"
#include "solver/testing_geqrf.hpp"
#include "solver/testing_geqrf_batched.hpp"
#include "solver/testing_geqrf_strided_batched.hpp"
#include "solver/testing_getrf.hpp"
#include "solver/testing_getrf_batched.hpp"
#include "solver/testing_getrf_npvt.hpp"
#include "solver/testing_getrf_npvt_batched.hpp"
#include "solver/testing_getrf_npvt_strided_batched.hpp"
#include "solver/testing_getrf_strided_batched.hpp"
#include "solver/testing_getri_batched.hpp"
#include "solver/testing_getri_npvt_batched.hpp"
#include "solver/testing_getrs.hpp"
#include "solver/testing_getrs_batched.hpp"
#include "solver/testing_getrs_strided_batched.hpp"
#endif

#include "utility.h"
#include <algorithm>
#undef I

//using namespace roc; // For emulated program_options
using namespace std::literals; // For std::string literals of form "str"s

struct str_less
{
    bool operator()(const char* a, const char* b) const
    {
        return strcmp(a, b) < 0;
    }
};

// Map from const char* to function taking const Arguments& using comparison above
using func_map = std::map<const char*, void (*)(const Arguments&), str_less>;

// Run a function by using map to map arg.function to function
void run_function(const func_map& map, const Arguments& arg, const std::string& msg = "")
{
    auto match = map.find(arg.function);
    if(match == map.end())
        throw std::invalid_argument("Invalid combination --function "s + arg.function
                                    + " --a_type "s + hip_datatype2string(arg.a_type) + msg);
    match->second(arg);
}

void get_test_name(const Arguments& arg, std::string& name)
{
    // Map from const char* to function taking const Arguments& using comparison above
    using name_to_f_testname_map
        = std::map<const char*, void (*)(const Arguments&, std::string&), str_less>;

    static const name_to_f_testname_map fmap = {
        // L1
        {"asum", testname_asum},
        {"asum_batched", testname_asum_batched},
        {"asum_strided_batched", testname_asum_strided_batched},
        {"axpy", testname_axpy},
        {"axpy_batched", testname_axpy_batched},
        {"axpy_strided_batched", testname_axpy_strided_batched},
        {"axpy_ex", testname_axpy_ex},
        {"axpy_batched_ex", testname_axpy_batched_ex},
        {"axpy_strided_batched_ex", testname_axpy_strided_batched_ex},
        {"copy", testname_copy},
        {"copy_batched", testname_copy_batched},
        {"copy_strided_batched", testname_copy_strided_batched},
        {"dot", testname_dot},
        {"dot_batched", testname_dot_batched},
        {"dot_strided_batched", testname_dot_strided_batched},
        {"dotc", testname_dotc},
        {"dotc_batched", testname_dotc_batched},
        {"dotc_strided_batched", testname_dotc_strided_batched},
        {"iamax", testname_iamax},
        {"iamax_batched", testname_iamax_batched},
        {"iamax_strided_batched", testname_iamax_strided_batched},
        {"iamin", testname_iamin},
        {"iamin_batched", testname_iamin_batched},
        {"iamin_strided_batched", testname_iamin_strided_batched},
        {"nrm2", testname_nrm2},
        {"nrm2_batched", testname_nrm2_batched},
        {"nrm2_strided_batched", testname_nrm2_strided_batched},
        {"nrm2_ex", testname_nrm2_ex},
        {"nrm2_batched_ex", testname_nrm2_batched_ex},
        {"nrm2_strided_batched_ex", testname_nrm2_strided_batched_ex},
        {"rot", testname_rot},
        {"rot_batched", testname_rot_batched},
        {"rot_strided_batched", testname_rot_strided_batched},
        {"rot_ex", testname_rot_ex},
        {"rot_batched_ex", testname_rot_batched_ex},
        {"rot_strided_batched_ex", testname_rot_strided_batched_ex},
        {"rotg", testname_rotg},
        {"rotg_batched", testname_rotg_batched},
        {"rotg_strided_batched", testname_rotg_strided_batched},
        {"rotm", testname_rotm},
        {"rotm_batched", testname_rotm_batched},
        {"rotm_strided_batched", testname_rotm_strided_batched},
        {"rotmg", testname_rotmg},
        {"rotmg_batched", testname_rotmg_batched},
        {"rotmg_strided_batched", testname_rotmg_strided_batched},
        {"swap", testname_swap},
        {"swap_batched", testname_swap_batched},
        {"swap_strided_batched", testname_swap_strided_batched},
        {"scal", testname_scal},
        {"scal_batched", testname_scal_batched},
        {"scal_strided_batched", testname_scal_strided_batched},
        {"scal_ex", testname_scal_ex},
        {"scal_batched_ex", testname_scal_batched_ex},
        {"scal_strided_batched_ex", testname_scal_strided_batched_ex},

        // L2
        {"gbmv", testname_gbmv},
        {"gbmv_batched", testname_gbmv_batched},
        {"gbmv_strided_batched", testname_gbmv_strided_batched},
        {"gemv", testname_gemv},
        {"gemv_batched", testname_gemv_batched},
        {"gemv_strided_batched", testname_gemv_strided_batched},
        {"ger", testname_ger},
        {"ger_batched", testname_ger_batched},
        {"ger_strided_batched", testname_ger_strided_batched},
        {"geru", testname_ger},
        {"geru_batched", testname_ger_batched},
        {"geru_strided_batched", testname_ger_strided_batched},
        {"gerc", testname_ger},
        {"gerc_batched", testname_ger_batched},
        {"gerc_strided_batched", testname_ger_strided_batched},
        {"hbmv", testname_hbmv},
        {"hbmv_batched", testname_hbmv_batched},
        {"hbmv_strided_batched", testname_hbmv_strided_batched},
        {"hemv", testname_hemv},
        {"hemv_batched", testname_hemv_batched},
        {"hemv_strided_batched", testname_hemv_strided_batched},
        {"her", testname_her},
        {"her_batched", testname_her_batched},
        {"her_strided_batched", testname_her_strided_batched},
        {"her2", testname_her2},
        {"her2_batched", testname_her2_batched},
        {"her2_strided_batched", testname_her2_strided_batched},
        {"hpmv", testname_hpmv},
        {"hpmv_batched", testname_hpmv_batched},
        {"hpmv_strided_batched", testname_hpmv_strided_batched},
        {"hpr", testname_hpr},
        {"hpr_batched", testname_hpr_batched},
        {"hpr_strided_batched", testname_hpr_strided_batched},
        {"hpr2", testname_hpr2},
        {"hpr2_batched", testname_hpr2_batched},
        {"hpr2_strided_batched", testname_hpr2_strided_batched},
        {"sbmv", testname_sbmv},
        {"sbmv_batched", testname_sbmv_batched},
        {"sbmv_strided_batched", testname_sbmv_strided_batched},
        {"spmv", testname_spmv},
        {"spmv_batched", testname_spmv_batched},
        {"spmv_strided_batched", testname_spmv_strided_batched},
        {"spr", testname_spr},
        {"spr_batched", testname_spr_batched},
        {"spr_strided_batched", testname_spr_strided_batched},
        {"spr2", testname_spr2},
        {"spr2_batched", testname_spr2_batched},
        {"spr2_strided_batched", testname_spr2_strided_batched},
        {"symv", testname_symv},
        {"symv_batched", testname_symv_batched},
        {"symv_strided_batched", testname_symv_strided_batched},
        {"syr", testname_syr},
        {"syr_batched", testname_syr_batched},
        {"syr_strided_batched", testname_syr_strided_batched},
        {"syr2", testname_syr2},
        {"syr2_batched", testname_syr2_batched},
        {"syr2_strided_batched", testname_syr2_strided_batched},
        {"tbmv", testname_tbmv},
        {"tbmv_batched", testname_tbmv_batched},
        {"tbmv_strided_batched", testname_tbmv_strided_batched},
        {"tbsv", testname_tbsv},
        {"tbsv_batched", testname_tbsv_batched},
        {"tbsv_strided_batched", testname_tbsv_strided_batched},
        {"tpmv", testname_tpmv},
        {"tpmv_batched", testname_tpmv_batched},
        {"tpmv_strided_batched", testname_tpmv_strided_batched},
        {"tpsv", testname_tpsv},
        {"tpsv_batched", testname_tpsv_batched},
        {"tpsv_strided_batched", testname_tpsv_strided_batched},
        {"trmv", testname_trmv},
        {"trmv_batched", testname_trmv_batched},
        {"trmv_strided_batched", testname_trmv_strided_batched},
        {"trsv", testname_trsv},
        {"trsv_batched", testname_trsv_batched},
        {"trsv_strided_batched", testname_trsv_strided_batched},

        // L3
        {"dgmm", testname_dgmm},
        {"dgmm_batched", testname_dgmm_batched},
        {"dgmm_strided_batched", testname_dgmm_strided_batched},
        {"geam", testname_geam},
        {"geam_batched", testname_geam_batched},
        {"geam_strided_batched", testname_geam_strided_batched},
        {"gemm", testname_gemm},
        {"gemm_batched", testname_gemm_batched},
        {"gemm_strided_batched", testname_gemm_strided_batched},
        {"gemm_ex", testname_gemm_ex},
        {"gemm_batched_ex", testname_gemm_batched_ex},
        {"gemm_strided_batched_ex", testname_gemm_strided_batched_ex},
        {"hemm", testname_hemm},
        {"hemm_batched", testname_hemm_batched},
        {"hemm_strided_batched", testname_hemm_strided_batched},
        {"herk", testname_herk},
        {"herk_batched", testname_herk_batched},
        {"herk_strided_batched", testname_herk_strided_batched},
        {"her2k", testname_her2k},
        {"her2k_batched", testname_her2k_batched},
        {"her2k_strided_batched", testname_her2k_strided_batched},
        {"herkx", testname_herkx},
        {"herkx_batched", testname_herkx_batched},
        {"herkx_strided_batched", testname_herkx_strided_batched},
        {"herk_ex", testname_herk_ex},
        {"symm", testname_symm},
        {"symm_batched", testname_symm_batched},
        {"symm_strided_batched", testname_symm_strided_batched},
        {"syrk", testname_syrk},
        {"syrk_batched", testname_syrk_batched},
        {"syrk_strided_batched", testname_syrk_strided_batched},
        {"syr2k", testname_syr2k},
        {"syr2k_batched", testname_syr2k_batched},
        {"syr2k_strided_batched", testname_syr2k_strided_batched},
        {"syrkx", testname_syrkx},
        {"syrkx_batched", testname_syrkx_batched},
        {"syrkx_strided_batched", testname_syrkx_strided_batched},
        {"syrk_ex", testname_syrk_ex},
        {"trmm", testname_trmm},
        {"trmm_batched", testname_trmm_batched},
        {"trmm_strided_batched", testname_trmm_strided_batched},
        {"trsm", testname_trsm},
        {"trsm_batched", testname_trsm_batched},
        {"trsm_strided_batched", testname_trsm_strided_batched},
        {"trsm_ex", testname_trsm_ex},
        {"trsm_batched_ex", testname_trsm_batched_ex},
        {"trsm_strided_batched_ex", testname_trsm_strided_batched_ex},
        {"trtri", testname_trtri},
        {"trtri_batched", testname_trtri_batched},
        {"trtri_strided_batched", testname_trtri_strided_batched},

#ifdef __HIP_PLATFORM_SOLVER__
        {"geqrf", testname_geqrf},
        {"geqrf_batched", testname_geqrf_batched},
        {"geqrf_strided_batched", testname_geqrf_strided_batched},
        {"getrf", testname_getrf},
        {"getrf_batched", testname_getrf_batched},
        {"getrf_strided_batched", testname_getrf_strided_batched},
        {"getrf_npvt", testname_getrf_npvt},
        {"getrf_npvt_batched", testname_getrf_npvt_batched},
        {"getrf_npvt_strided_batched", testname_getrf_npvt_strided_batched},
        {"getri_batched", testname_getri_batched},
        {"getri_npvt_batched", testname_getri_npvt_batched},
        {"getrs", testname_getrs},
        {"getrs_batched", testname_getrs_batched},
        {"getrs_strided_batched", testname_getrs_strided_batched},
        {"gels", testname_gels},
        {"gels_batched", testname_gels_batched},
        {"gels_strided_batched", testname_gels_strided_batched},
#endif

        // Aux
        {"set_get_vector", testname_set_get_vector},
        {"set_get_vector_async", testname_set_get_vector_async},
        {"set_get_matrix", testname_set_get_matrix},
        {"set_get_matrix_async", testname_set_get_matrix_async},
    };

    auto match = fmap.find(arg.function);
    if(match != fmap.end())
        match->second(arg, name);
}

// Template to dispatch testing_gemm_ex for performance tests
// When Ti == void or Ti == To == Tc == bfloat16, the test is marked invalid
template <typename Ti, typename To = Ti, typename Tc = To, typename = void>
struct perf_gemm_ex : hipblas_test_invalid
{
};

template <typename Ti, typename To, typename Tc>
struct perf_gemm_ex<Ti,
                    To,
                    Tc,
                    std::enable_if_t<!std::is_same<Ti, void>{}
                                     && !(std::is_same<Ti, To>{} && std::is_same<Ti, Tc>{}
                                          && std::is_same<Ti, hipblasBfloat16>{})>>
    : hipblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"gemm_ex", testing_gemm_ex<Ti, To, Tc>},
            {"gemm_batched_ex", testing_gemm_batched_ex<Ti, To, Tc>},
        };
        run_function(map, arg);
    }
};

// Template to dispatch testing_gemm_strided_batched_ex for performance tests
// When Ti == void or Ti == To == Tc == bfloat16, the test is marked invalid
template <typename Ti, typename To = Ti, typename Tc = To, typename = void>
struct perf_gemm_strided_batched_ex : hipblas_test_invalid
{
};

template <typename Ti, typename To, typename Tc>
struct perf_gemm_strided_batched_ex<
    Ti,
    To,
    Tc,
    std::enable_if_t<!std::is_same<Ti, void>{}
                     && !(std::is_same<Ti, To>{} && std::is_same<Ti, Tc>{}
                          && std::is_same<Ti, hipblasBfloat16>{})>> : hipblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"gemm_strided_batched_ex", testing_gemm_strided_batched_ex<Ti, To, Tc>},
        };
        run_function(map, arg);
    }
};

// Template to dispatch testing_syrk_ex for performance tests
// When Ti == void or Ti == To == Tc == bfloat16, the test is marked invalid
template <typename Ti, typename To = Ti, typename Tc = To, typename = void>
struct perf_syrk_ex : hipblas_test_invalid
{
};

template <typename Ti, typename To, typename Tc>
struct perf_syrk_ex<Ti,
                    To,
                    Tc,
                    std::enable_if_t<!std::is_same<Ti, void>{}
                                     && !(std::is_same<Ti, To>{} && std::is_same<Ti, Tc>{}
                                          && std::is_same<Ti, hipblasBfloat16>{})>>
    : hipblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"syrk_ex", testing_syrk_ex<Ti, To, Tc>},
        };
        run_function(map, arg);
    }
};

// Template to dispatch testing_herk_ex for performance tests
// When Ti == void or not complex in and out, the test is marked invalid
template <typename Ti, typename To = Ti, typename Tc = To, typename = void>
struct perf_herk_ex : hipblas_test_invalid
{
};
template <typename Ti, typename To, typename Tc>
struct perf_herk_ex<
    Ti,
    To,
    Tc,
    std::enable_if_t<!std::is_same<Ti, void>{} && (is_complex<To> && is_complex<Ti>)>>
    : hipblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"herk_ex", testing_herk_ex<Ti, To, Tc>},
        };
        run_function(map, arg);
    }
};

template <typename T, typename U = T, typename = void>
struct perf_blas : hipblas_test_invalid
{
};

template <typename T, typename U>
struct perf_blas<T, U, std::enable_if_t<std::is_same<T, float>{} || std::is_same<T, double>{}>>
    : hipblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map fmap = {
            // L1
            {"asum", testing_asum<T>},
            {"asum_batched", testing_asum_batched<T>},
            {"asum_strided_batched", testing_asum_strided_batched<T>},
            {"axpy", testing_axpy<T>},
            {"axpy_batched", testing_axpy_batched<T>},
            {"axpy_strided_batched", testing_axpy_strided_batched<T>},
            {"copy", testing_copy<T>},
            {"copy_batched", testing_copy_batched<T>},
            {"copy_strided_batched", testing_copy_strided_batched<T>},
            {"dot", testing_dot<T>},
            {"dot_batched", testing_dot_batched<T>},
            {"dot_strided_batched", testing_dot_strided_batched<T>},
            {"iamax", testing_iamax<T>},
            {"iamax_batched", testing_iamax_batched<T>},
            {"iamax_strided_batched", testing_iamax_strided_batched<T>},
            {"iamin", testing_iamin<T>},
            {"iamin_batched", testing_iamin_batched<T>},
            {"iamin_strided_batched", testing_iamin_strided_batched<T>},
            {"nrm2", testing_nrm2<T>},
            {"nrm2_batched", testing_nrm2_batched<T>},
            {"nrm2_strided_batched", testing_nrm2_strided_batched<T>},
            {"rotg", testing_rotg<T>},
            {"rotg_batched", testing_rotg_batched<T>},
            {"rotg_strided_batched", testing_rotg_strided_batched<T>},
            {"rotm", testing_rotm<T>},
            {"rotm_batched", testing_rotm_batched<T>},
            {"rotm_strided_batched", testing_rotm_strided_batched<T>},
            {"rotmg", testing_rotmg<T>},
            {"rotmg_batched", testing_rotmg_batched<T>},
            {"rotmg_strided_batched", testing_rotmg_strided_batched<T>},
            {"swap", testing_swap<T>},
            {"swap_batched", testing_swap_batched<T>},
            {"swap_strided_batched", testing_swap_strided_batched<T>},
            {"scal", testing_scal<T>},
            {"scal_batched", testing_scal_batched<T>},
            {"scal_strided_batched", testing_scal_strided_batched<T>},

            // L2
            {"gbmv", testing_gbmv<T>},
            {"gbmv_batched", testing_gbmv_batched<T>},
            {"gbmv_strided_batched", testing_gbmv_strided_batched<T>},
            {"gemv", testing_gemv<T>},
            {"gemv_batched", testing_gemv_batched<T>},
            {"gemv_strided_batched", testing_gemv_strided_batched<T>},
            {"ger", testing_ger<T, false>},
            {"ger_batched", testing_ger_batched<T, false>},
            {"ger_strided_batched", testing_ger_strided_batched<T, false>},
            {"sbmv", testing_sbmv<T>},
            {"sbmv_batched", testing_sbmv_batched<T>},
            {"sbmv_strided_batched", testing_sbmv_strided_batched<T>},
            {"spmv", testing_spmv<T>},
            {"spmv_batched", testing_spmv_batched<T>},
            {"spmv_strided_batched", testing_spmv_strided_batched<T>},
            {"spr", testing_spr<T>},
            {"spr_batched", testing_spr_batched<T>},
            {"spr_strided_batched", testing_spr_strided_batched<T>},
            {"spr2", testing_spr2<T>},
            {"spr2_batched", testing_spr2_batched<T>},
            {"spr2_strided_batched", testing_spr2_strided_batched<T>},
            {"symv", testing_symv<T>},
            {"symv_batched", testing_symv_batched<T>},
            {"symv_strided_batched", testing_symv_strided_batched<T>},
            {"syr", testing_syr<T>},
            {"syr_batched", testing_syr_batched<T>},
            {"syr_strided_batched", testing_syr_strided_batched<T>},
            {"syr2", testing_syr2<T>},
            {"syr2_batched", testing_syr2_batched<T>},
            {"syr2_strided_batched", testing_syr2_strided_batched<T>},
            {"tbmv", testing_tbmv<T>},
            {"tbmv_batched", testing_tbmv_batched<T>},
            {"tbmv_strided_batched", testing_tbmv_strided_batched<T>},
            {"tbsv", testing_tbsv<T>},
            {"tbsv_batched", testing_tbsv_batched<T>},
            {"tbsv_strided_batched", testing_tbsv_strided_batched<T>},
            {"tpmv", testing_tpmv<T>},
            {"tpmv_batched", testing_tpmv_batched<T>},
            {"tpmv_strided_batched", testing_tpmv_strided_batched<T>},
            {"tpsv", testing_tpsv<T>},
            {"tpsv_batched", testing_tpsv_batched<T>},
            {"tpsv_strided_batched", testing_tpsv_strided_batched<T>},
            {"trmv", testing_trmv<T>},
            {"trmv_batched", testing_trmv_batched<T>},
            {"trmv_strided_batched", testing_trmv_strided_batched<T>},
            {"trsv", testing_trsv<T>},
            {"trsv_batched", testing_trsv_batched<T>},
            {"trsv_strided_batched", testing_trsv_strided_batched<T>},

            // L3
            {"geam", testing_geam<T>},
            {"geam_batched", testing_geam_batched<T>},
            {"geam_strided_batched", testing_geam_strided_batched<T>},
            {"dgmm", testing_dgmm<T>},
            {"dgmm_batched", testing_dgmm_batched<T>},
            {"dgmm_strided_batched", testing_dgmm_strided_batched<T>},
            {"trmm", testing_trmm<T>},
            {"trmm_batched", testing_trmm_batched<T>},
            {"trmm_strided_batched", testing_trmm_strided_batched<T>},
            {"gemm", testing_gemm<T>},
            {"gemm_batched", testing_gemm_batched<T>},
            {"gemm_strided_batched", testing_gemm_strided_batched<T>},
            {"symm", testing_symm<T>},
            {"symm_batched", testing_symm_batched<T>},
            {"symm_strided_batched", testing_symm_strided_batched<T>},
            {"syrk", testing_syrk<T>},
            {"syrk_batched", testing_syrk_batched<T>},
            {"syrk_strided_batched", testing_syrk_strided_batched<T>},
            {"syr2k", testing_syr2k<T>},
            {"syr2k_batched", testing_syr2k_batched<T>},
            {"syr2k_strided_batched", testing_syr2k_strided_batched<T>},
            {"trtri", testing_trtri<T>},
            {"trtri_batched", testing_trtri_batched<T>},
            {"trtri_strided_batched", testing_trtri_strided_batched<T>},
            {"syrkx", testing_syrkx<T>},
            {"syrkx_batched", testing_syrkx_batched<T>},
            {"syrkx_strided_batched", testing_syrkx_strided_batched<T>},
            {"trsm", testing_trsm<T>},
            {"trsm_ex", testing_trsm_ex<T>},
            {"trsm_batched", testing_trsm_batched<T>},
            {"trsm_batched_ex", testing_trsm_batched_ex<T>},
            {"trsm_strided_batched", testing_trsm_strided_batched<T>},
            {"trsm_strided_batched_ex", testing_trsm_strided_batched_ex<T>},

#ifdef __HIP_PLATFORM_SOLVER__
            {"geqrf", testing_geqrf<T>},
            {"geqrf_batched", testing_geqrf_batched<T>},
            {"geqrf_strided_batched", testing_geqrf_strided_batched<T>},
            {"getrf", testing_getrf<T>},
            {"getrf_batched", testing_getrf_batched<T>},
            {"getrf_strided_batched", testing_getrf_strided_batched<T>},
            {"getrf_npvt", testing_getrf_npvt<T>},
            {"getrf_npvt_batched", testing_getrf_npvt_batched<T>},
            {"getrf_npvt_strided_batched", testing_getrf_npvt_strided_batched<T>},
            {"getri_batched", testing_getri_batched<T>},
            {"getri_npvt_batched", testing_getri_npvt_batched<T>},
            {"getrs", testing_getrs<T>},
            {"getrs_batched", testing_getrs_batched<T>},
            {"getrs_strided_batched", testing_getrs_strided_batched<T>},
            {"gels", testing_gels<T>},
            {"gels_batched", testing_gels_batched<T>},
            {"gels_strided_batched", testing_gels_strided_batched<T>},
#endif

            // Aux
            {"set_get_vector", testing_set_get_vector<T>},
            {"set_get_vector_async", testing_set_get_vector_async<T>},
            {"set_get_matrix", testing_set_get_matrix<T>},
            {"set_get_matrix_async", testing_set_get_matrix_async<T>},
        };
        run_function(fmap, arg);
    }
};

template <typename T, typename U>
struct perf_blas<T, U, std::enable_if_t<std::is_same<T, hipblasBfloat16>{}>> : hipblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"dot", testing_dot<T>},
            {"dot_batched", testing_dot_batched<T>},
            {"dot_strided_batched", testing_dot_strided_batched<T>},
        };
        run_function(map, arg);
    }
};

template <typename T, typename U>
struct perf_blas<T, U, std::enable_if_t<std::is_same<T, hipblasHalf>{}>> : hipblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"axpy", testing_axpy<T>},
            {"axpy_batched", testing_axpy_batched<T>},
            {"axpy_strided_batched", testing_axpy_strided_batched<T>},
            {"dot", testing_dot<T>},
            {"dot_batched", testing_dot_batched<T>},
            {"dot_strided_batched", testing_dot_strided_batched<T>},
            {"gemm", testing_gemm<T>},
            {"gemm_batched", testing_gemm_batched<T>},
            {"gemm_strided_batched", testing_gemm_strided_batched<T>},

        };
        run_function(map, arg);
    }
};

template <typename T, typename U>
struct perf_blas<T,
                 U,
                 std::enable_if_t<std::is_same<T, std::complex<double>>{}
                                  || std::is_same<T, std::complex<float>>{}>> : hipblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            // L1
            {"asum", testing_asum<T>},
            {"asum_batched", testing_asum_batched<T>},
            {"asum_strided_batched", testing_asum_strided_batched<T>},
            {"axpy", testing_axpy<T>},
            {"axpy_batched", testing_axpy_batched<T>},
            {"axpy_strided_batched", testing_axpy_strided_batched<T>},
            {"copy", testing_copy<T>},
            {"copy_batched", testing_copy_batched<T>},
            {"copy_strided_batched", testing_copy_strided_batched<T>},
            {"dot", testing_dot<T>},
            {"dot_batched", testing_dot_batched<T>},
            {"dot_strided_batched", testing_dot_strided_batched<T>},
            {"dotc", testing_dotc<T>},
            {"dotc_batched", testing_dotc_batched<T>},
            {"dotc_strided_batched", testing_dotc_strided_batched<T>},
            {"iamax", testing_iamax<T>},
            {"iamax_batched", testing_iamax_batched<T>},
            {"iamax_strided_batched", testing_iamax_strided_batched<T>},
            {"iamin", testing_iamin<T>},
            {"iamin_batched", testing_iamin_batched<T>},
            {"iamin_strided_batched", testing_iamin_strided_batched<T>},
            {"nrm2", testing_nrm2<T>},
            {"nrm2_batched", testing_nrm2_batched<T>},
            {"nrm2_strided_batched", testing_nrm2_strided_batched<T>},
            {"rotg", testing_rotg<T>},
            {"rotg_batched", testing_rotg_batched<T>},
            {"rotg_strided_batched", testing_rotg_strided_batched<T>},
            {"swap", testing_swap<T>},
            {"swap_batched", testing_swap_batched<T>},
            {"swap_strided_batched", testing_swap_strided_batched<T>},
            {"scal", testing_scal<T>},
            {"scal_batched", testing_scal_batched<T>},
            {"scal_strided_batched", testing_scal_strided_batched<T>},

            // L2
            {"gemv", testing_gemv<T>},
            {"gemv_batched", testing_gemv_batched<T>},
            {"gemv_strided_batched", testing_gemv_strided_batched<T>},
            {"gbmv", testing_gbmv<T>},
            {"gbmv_batched", testing_gbmv_batched<T>},
            {"gbmv_strided_batched", testing_gbmv_strided_batched<T>},
            {"geru", testing_ger<T, false>},
            {"geru_batched", testing_ger_batched<T, false>},
            {"geru_strided_batched", testing_ger_strided_batched<T, false>},
            {"gerc", testing_ger<T, true>},
            {"gerc_batched", testing_ger_batched<T, true>},
            {"gerc_strided_batched", testing_ger_strided_batched<T, true>},
            {"hbmv", testing_hbmv<T>},
            {"hbmv_batched", testing_hbmv_batched<T>},
            {"hbmv_strided_batched", testing_hbmv_strided_batched<T>},
            {"hemv", testing_hemv<T>},
            {"hemv_batched", testing_hemv_batched<T>},
            {"hemv_strided_batched", testing_hemv_strided_batched<T>},
            {"her", testing_her<T>},
            {"her_batched", testing_her_batched<T>},
            {"her_strided_batched", testing_her_strided_batched<T>},
            {"her2", testing_her2<T>},
            {"her2_batched", testing_her2_batched<T>},
            {"her2_strided_batched", testing_her2_strided_batched<T>},
            {"hpmv", testing_hpmv<T>},
            {"hpmv_batched", testing_hpmv_batched<T>},
            {"hpmv_strided_batched", testing_hpmv_strided_batched<T>},
            {"hpr", testing_hpr<T>},
            {"hpr_batched", testing_hpr_batched<T>},
            {"hpr_strided_batched", testing_hpr_strided_batched<T>},
            {"hpr2", testing_hpr2<T>},
            {"hpr2_batched", testing_hpr2_batched<T>},
            {"hpr2_strided_batched", testing_hpr2_strided_batched<T>},
            {"spr", testing_spr<T>},
            {"spr_batched", testing_spr_batched<T>},
            {"spr_strided_batched", testing_spr_strided_batched<T>},
            {"symv", testing_symv<T>},
            {"symv_batched", testing_symv_batched<T>},
            {"symv_strided_batched", testing_symv_strided_batched<T>},
            {"syr", testing_syr<T>},
            {"syr_batched", testing_syr_batched<T>},
            {"syr_strided_batched", testing_syr_strided_batched<T>},
            {"syr2", testing_syr2<T>},
            {"syr2_batched", testing_syr2_batched<T>},
            {"syr2_strided_batched", testing_syr2_strided_batched<T>},
            {"tbmv", testing_tbmv<T>},
            {"tbmv_batched", testing_tbmv_batched<T>},
            {"tbmv_strided_batched", testing_tbmv_strided_batched<T>},
            {"tbsv", testing_tbsv<T>},
            {"tbsv_batched", testing_tbsv_batched<T>},
            {"tbsv_strided_batched", testing_tbsv_strided_batched<T>},
            {"tpmv", testing_tpmv<T>},
            {"tpmv_batched", testing_tpmv_batched<T>},
            {"tpmv_strided_batched", testing_tpmv_strided_batched<T>},
            {"tpsv", testing_tpsv<T>},
            {"tpsv_batched", testing_tpsv_batched<T>},
            {"tpsv_strided_batched", testing_tpsv_strided_batched<T>},
            {"trmv", testing_trmv<T>},
            {"trmv_batched", testing_trmv_batched<T>},
            {"trmv_strided_batched", testing_trmv_strided_batched<T>},
            {"trsv", testing_trsv<T>},
            {"trsv_batched", testing_trsv_batched<T>},
            {"trsv_strided_batched", testing_trsv_strided_batched<T>},

            // L3
            {"dgmm", testing_dgmm<T>},
            {"dgmm_batched", testing_dgmm_batched<T>},
            {"dgmm_strided_batched", testing_dgmm_strided_batched<T>},
            {"geam", testing_geam<T>},
            {"geam_batched", testing_geam_batched<T>},
            {"geam_strided_batched", testing_geam_strided_batched<T>},
            {"gemm", testing_gemm<T>},
            {"gemm_batched", testing_gemm_batched<T>},
            {"gemm_strided_batched", testing_gemm_strided_batched<T>},
            {"hemm", testing_hemm<T>},
            {"hemm_batched", testing_hemm_batched<T>},
            {"hemm_strided_batched", testing_hemm_strided_batched<T>},
            {"herk", testing_herk<T>},
            {"herk_batched", testing_herk_batched<T>},
            {"herk_strided_batched", testing_herk_strided_batched<T>},
            {"her2k", testing_her2k<T>},
            {"her2k_batched", testing_her2k_batched<T>},
            {"her2k_strided_batched", testing_her2k_strided_batched<T>},
            {"herkx", testing_herkx<T>},
            {"herkx_batched", testing_herkx_batched<T>},
            {"herkx_strided_batched", testing_herkx_strided_batched<T>},
            {"symm", testing_symm<T>},
            {"symm_batched", testing_symm_batched<T>},
            {"symm_strided_batched", testing_symm_strided_batched<T>},
            {"syrk", testing_syrk<T>},
            {"syrk_batched", testing_syrk_batched<T>},
            {"syrk_strided_batched", testing_syrk_strided_batched<T>},
            {"syr2k", testing_syr2k<T>},
            {"syr2k_batched", testing_syr2k_batched<T>},
            {"syr2k_strided_batched", testing_syr2k_strided_batched<T>},
            {"trtri", testing_trtri<T>},
            {"trtri_batched", testing_trtri_batched<T>},
            {"trtri_strided_batched", testing_trtri_strided_batched<T>},
            {"syrkx", testing_syrkx<T>},
            {"syrkx_batched", testing_syrkx_batched<T>},
            {"syrkx_strided_batched", testing_syrkx_strided_batched<T>},
            {"trsm", testing_trsm<T>},
            {"trsm_batched", testing_trsm_batched<T>},
            {"trsm_strided_batched", testing_trsm_strided_batched<T>},
            {"trsm_ex", testing_trsm_ex<T>},
            {"trsm_batched_ex", testing_trsm_batched_ex<T>},
            {"trsm_strided_batched_ex", testing_trsm_strided_batched_ex<T>},
            {"trmm", testing_trmm<T>},
            {"trmm_batched", testing_trmm_batched<T>},
            {"trmm_strided_batched", testing_trmm_strided_batched<T>},

#ifdef __HIP_PLATFORM_SOLVER__
            {"geqrf", testing_geqrf<T>},
            {"geqrf_batched", testing_geqrf_batched<T>},
            {"geqrf_strided_batched", testing_geqrf_strided_batched<T>},
            {"getrf", testing_getrf<T>},
            {"getrf_batched", testing_getrf_batched<T>},
            {"getrf_strided_batched", testing_getrf_strided_batched<T>},
            {"getrf_npvt", testing_getrf_npvt<T>},
            {"getrf_npvt_batched", testing_getrf_npvt_batched<T>},
            {"getrf_npvt_strided_batched", testing_getrf_npvt_strided_batched<T>},
            {"getri_batched", testing_getri_batched<T>},
            {"getri_npvt_batched", testing_getri_npvt_batched<T>},
            {"getrs", testing_getrs<T>},
            {"getrs_batched", testing_getrs_batched<T>},
            {"getrs_strided_batched", testing_getrs_strided_batched<T>},
            {"gels", testing_gels<T>},
            {"gels_batched", testing_gels_batched<T>},
            {"gels_strided_batched", testing_gels_strided_batched<T>},
#endif
        };
        run_function(map, arg);
    }
};

template <typename Ta, typename Tx = Ta, typename Ty = Tx, typename Tex = Ty, typename = void>
struct perf_blas_axpy_ex : hipblas_test_invalid
{
};

template <typename Ta, typename Tx, typename Ty, typename Tex>
struct perf_blas_axpy_ex<
    Ta,
    Tx,
    Ty,
    Tex,
    std::enable_if_t<
        (std::is_same_v<
             Ta,
             float> && std::is_same_v<Ta, Tx> && std::is_same_v<Tx, Ty> && std::is_same_v<Ty, Tex>)
        || (std::is_same_v<
                Ta,
                double> && std::is_same_v<Ta, Tx> && std::is_same_v<Tx, Ty> && std::is_same_v<Ty, Tex>)
        || (std::is_same_v<
                Ta,
                hipblasHalf> && std::is_same_v<Ta, Tx> && std::is_same_v<Tx, Ty> && std::is_same_v<Ty, Tex>)
        || (std::is_same_v<
                Ta,
                std::complex<
                    float>> && std::is_same_v<Ta, Tx> && std::is_same_v<Tx, Ty> && std::is_same_v<Ty, Tex>)
        || (std::is_same_v<
                Ta,
                std::complex<
                    double>> && std::is_same_v<Ta, Tx> && std::is_same_v<Tx, Ty> && std::is_same_v<Ty, Tex>)
        || (std::is_same_v<
                Ta,
                hipblasHalf> && std::is_same_v<Ta, Tx> && std::is_same_v<Tx, Ty> && std::is_same_v<Tex, float>)
        || (std::is_same_v<
                Ta,
                float> && std::is_same_v<Tx, hipblasHalf> && std::is_same_v<Ta, Tex> && std::is_same_v<Tx, Ty>)
        || (std::is_same_v<
                Ta,
                hipblasBfloat16> && std::is_same_v<Ta, Tx> && std::is_same_v<Tx, Ty> && std::is_same_v<Tex, float>)
        || (std::is_same_v<
                Ta,
                float> && std::is_same_v<Tx, hipblasBfloat16> && std::is_same_v<Tx, Ty> && std::is_same_v<Ta, Tex>)>>
    : hipblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"axpy_ex", testing_axpy_ex<Ta, Tx, Ty, Tex>},
            {"axpy_batched_ex", testing_axpy_batched_ex<Ta, Tx, Ty, Tex>},
            {"axpy_strided_batched_ex", testing_axpy_strided_batched_ex<Ta, Tx, Ty, Tex>},
        };
        run_function(map, arg);
    }
};

template <typename Tx, typename Ty = Tx, typename Tr = Ty, typename Tex = Tr, typename = void>
struct perf_blas_dot_ex : hipblas_test_invalid
{
};

template <typename Tx, typename Ty, typename Tr, typename Tex>
struct perf_blas_dot_ex<
    Tx,
    Ty,
    Tr,
    Tex,
    std::enable_if_t<(std::is_same<Tx, float>{} && std::is_same<Tx, Ty>{} && std::is_same<Ty, Tr>{}
                      && std::is_same<Tr, Tex>{})
                     || (std::is_same<Tx, double>{} && std::is_same<Tx, Ty>{}
                         && std::is_same<Ty, Tr>{} && std::is_same<Tr, Tex>{})
                     || (std::is_same<Tx, hipblasHalf>{} && std::is_same<Tx, Ty>{}
                         && std::is_same<Ty, Tr>{} && std::is_same<Tr, Tex>{})
                     || (std::is_same<Tx, std::complex<float>>{} && std::is_same<Tx, Ty>{}
                         && std::is_same<Ty, Tr>{} && std::is_same<Tr, Tex>{})
                     || (std::is_same<Tx, std::complex<double>>{} && std::is_same<Tx, Ty>{}
                         && std::is_same<Ty, Tr>{} && std::is_same<Tr, Tex>{})
                     || (std::is_same<Tx, hipblasHalf>{} && std::is_same<Tx, Ty>{}
                         && std::is_same<Ty, Tr>{} && std::is_same<Tex, float>{})
                     || (std::is_same<Tx, hipblasBfloat16>{} && std::is_same<Tx, Ty>{}
                         && std::is_same<Ty, Tr>{} && std::is_same<Tex, float>{})>>
    : hipblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"dot_ex", testing_dot_ex<Tx, Ty, Tr, Tex, false>},
            {"dot_batched_ex", testing_dot_batched_ex<Tx, Ty, Tr, Tex, false>},
            {"dot_strided_batched_ex", testing_dot_strided_batched_ex<Tx, Ty, Tr, Tex, false>},
            {"dotc_ex", testing_dot_ex<Tx, Ty, Tr, Tex, true>},
            {"dotc_batched_ex", testing_dot_batched_ex<Tx, Ty, Tr, Tex, true>},
            {"dotc_strided_batched_ex", testing_dot_strided_batched_ex<Tx, Ty, Tr, Tex, true>},
        };
        run_function(map, arg);
    }
};

template <typename Tx, typename Tr = Tx, typename Tex = Tr, typename = void>
struct perf_blas_nrm2_ex : hipblas_test_invalid
{
};

template <typename Tx, typename Tr, typename Tex>
struct perf_blas_nrm2_ex<
    Tx,
    Tr,
    Tex,
    std::enable_if_t<
        (std::is_same<Tx, float>{} && std::is_same<Tx, Tr>{} && std::is_same<Tr, Tex>{})
        || (std::is_same<Tx, double>{} && std::is_same<Tx, Tr>{} && std::is_same<Tr, Tex>{})
        || (std::is_same<Tx, std::complex<float>>{} && std::is_same<Tr, float>{}
            && std::is_same<Tr, Tex>{})
        || (std::is_same<Tx, std::complex<double>>{} && std::is_same<Tr, double>{}
            && std::is_same<Tr, Tex>{})
        || (std::is_same<Tx, hipblasHalf>{} && std::is_same<Tr, Tx>{} && std::is_same<Tex, float>{})
        || (std::is_same<Tx, hipblasBfloat16>{} && std::is_same<Tr, Tx>{}
            && std::is_same<Tex, float>{})>> : hipblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"nrm2_ex", testing_nrm2_ex<Tx, Tr, Tex>},
            {"nrm2_batched_ex", testing_nrm2_batched_ex<Tx, Tr, Tex>},
            {"nrm2_strided_batched_ex", testing_nrm2_strided_batched_ex<Tx, Tr, Tex>},
        };
        run_function(map, arg);
    }
};

template <typename Tx, typename Ty = Tx, typename Tcs = Ty, typename Tex = Tcs, typename = void>
struct perf_blas_rot_ex : hipblas_test_invalid
{
};

template <typename Tx, typename Ty, typename Tcs, typename Tex>
struct perf_blas_rot_ex<
    Tx,
    Ty,
    Tcs,
    Tex,
    std::enable_if_t<(std::is_same<Tx, float>{} && std::is_same<Tx, Ty>{} && std::is_same<Ty, Tcs>{}
                      && std::is_same<Tcs, Tex>{})
                     || (std::is_same<Tx, double>{} && std::is_same<Ty, Tx>{}
                         && std::is_same<Ty, Tcs>{} && std::is_same<Tex, Tcs>{})
                     || (std::is_same<Tx, std::complex<float>>{} && std::is_same<Ty, Tx>{}
                         && std::is_same<Tcs, Ty>{} && std::is_same<Tcs, Tex>{})
                     || (std::is_same<Tx, std::complex<double>>{} && std::is_same<Tx, Ty>{}
                         && std::is_same<Tcs, Ty>{} && std::is_same<Tex, Tcs>{})
                     || (std::is_same<Tx, std::complex<float>>{} && std::is_same<Ty, Tx>{}
                         && std::is_same<Tcs, float>{} && std::is_same<Tex, std::complex<float>>{})
                     || (std::is_same<Tx, std::complex<double>>{} && std::is_same<Tx, Ty>{}
                         && std::is_same<Tcs, double>{}
                         && std::is_same<Tex, std::complex<double>>{})
                     || (std::is_same<Tx, hipblasHalf>{} && std::is_same<Ty, Tx>{}
                         && std::is_same<Tcs, Ty>{} && std::is_same<Tex, float>{})
                     || (std::is_same<Tx, hipblasBfloat16>{} && std::is_same<Ty, Tx>{}
                         && std::is_same<Tcs, Ty>{} && std::is_same<Tex, float>{})>>
    : hipblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"rot_ex", testing_rot_ex<Tx, Ty, Tcs, Tex>},
            {"rot_batched_ex", testing_rot_batched_ex<Tx, Ty, Tcs, Tex>},
            {"rot_strided_batched_ex", testing_rot_strided_batched_ex<Tx, Ty, Tcs, Tex>},
        };
        run_function(map, arg);
    }
};

template <typename Ti, typename To = Ti, typename Tc = To, typename = void>
struct perf_blas_rot : hipblas_test_invalid
{
};

template <typename Ti, typename To, typename Tc>
struct perf_blas_rot<
    Ti,
    To,
    Tc,
    std::enable_if_t<(std::is_same<Ti, float>{} && std::is_same<Ti, To>{} && std::is_same<To, Tc>{})
                     || (std::is_same<Ti, double>{} && std::is_same<Ti, To>{}
                         && std::is_same<To, Tc>{})
                     || (std::is_same<Ti, std::complex<float>>{} && std::is_same<To, float>{}
                         && std::is_same<Tc, std::complex<float>>{})
                     || (std::is_same<Ti, std::complex<float>>{} && std::is_same<To, float>{}
                         && std::is_same<Tc, float>{})
                     || (std::is_same<Ti, std::complex<double>>{} && std::is_same<To, double>{}
                         && std::is_same<Tc, std::complex<double>>{})
                     || (std::is_same<Ti, std::complex<double>>{} && std::is_same<To, double>{}
                         && std::is_same<Tc, double>{})>> : hipblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"rot", testing_rot<Ti, To, Tc>},
            {"rot_batched", testing_rot_batched<Ti, To, Tc>},
            {"rot_strided_batched", testing_rot_strided_batched<Ti, To, Tc>},
        };
        run_function(map, arg);
    }
};

template <typename Ta, typename Tb = Ta, typename = void>
struct perf_blas_scal : hipblas_test_invalid
{
};

template <typename Ta, typename Tb>
struct perf_blas_scal<
    Ta,
    Tb,
    std::enable_if_t<(std::is_same<Ta, double>{} && std::is_same<Tb, std::complex<double>>{})
                     || (std::is_same<Ta, float>{} && std::is_same<Tb, std::complex<float>>{})
                     || (std::is_same<Ta, Tb>{} && std::is_same<Ta, float>{})
                     || (std::is_same<Ta, Tb>{} && std::is_same<Ta, double>{})
                     || (std::is_same<Ta, Tb>{} && std::is_same<Ta, std::complex<float>>{})
                     || (std::is_same<Ta, Tb>{} && std::is_same<Ta, std::complex<double>>{})>>
    : hipblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"scal", testing_scal<Ta, Tb>},
            {"scal_batched", testing_scal_batched<Ta, Tb>},
            {"scal_strided_batched", testing_scal_strided_batched<Ta, Tb>},
        };
        run_function(map, arg);
    }
};

template <typename Ta, typename Tx = Ta, typename Tex = Tx, typename = void>
struct perf_blas_scal_ex : hipblas_test_invalid
{
};

template <typename Ta, typename Tx, typename Tex>
struct perf_blas_scal_ex<
    Ta,
    Tx,
    Tex,
    std::enable_if_t<
        (std::is_same<Ta, float>{} && std::is_same<Ta, Tx>{} && std::is_same<Tx, Tex>{})
        || (std::is_same<Ta, double>{} && std::is_same<Ta, Tx>{} && std::is_same<Tx, Tex>{})
        || (std::is_same<Ta, hipblasHalf>{} && std::is_same<Ta, Tx>{} && std::is_same<Tx, Tex>{})
        || (std::is_same<Ta, std::complex<float>>{} && std::is_same<Ta, Tx>{}
            && std::is_same<Tx, Tex>{})
        || (std::is_same<Ta, std::complex<double>>{} && std::is_same<Ta, Tx>{}
            && std::is_same<Tx, Tex>{})
        || (std::is_same<Ta, hipblasHalf>{} && std::is_same<Ta, Tx>{} && std::is_same<Tex, float>{})
        || (std::is_same<Ta, float>{} && std::is_same<Tx, hipblasHalf>{} && std::is_same<Ta, Tex>{})
        || (std::is_same<Ta, float>{} && std::is_same<Tx, std::complex<float>>{}
            && std::is_same<Tx, Tex>{})
        || (std::is_same<Ta, double>{} && std::is_same<Tx, std::complex<double>>{}
            && std::is_same<Tx, Tex>{})
        || (std::is_same<Ta, hipblasBfloat16>{} && std::is_same<Ta, Tx>{}
            && std::is_same<Tex, float>{})
        || (std::is_same<Ta, float>{} && std::is_same<Tx, hipblasBfloat16>{}
            && std::is_same<Tex, float>{})>> : hipblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"scal_ex", testing_scal_ex<Ta, Tx, Tex>},
            {"scal_batched_ex", testing_scal_batched_ex<Ta, Tx, Tex>},
            {"scal_strided_batched_ex", testing_scal_strided_batched_ex<Ta, Tx, Tex>},
        };
        run_function(map, arg);
    }
};

int run_bench_test(Arguments& arg, int unit_check, int timing)
{
    //hipblas_initialize(); // Initialize rocBLAS

    std::cout << std::setiosflags(std::ios::fixed)
              << std::setprecision(7); // Set precision to 7 digits

    // disable unit_check in client benchmark, it is only used in gtest unit test
    arg.unit_check = unit_check;

    // enable timing check,otherwise no performance data collected
    arg.timing = timing;

    // Skip past any testing_ prefix in function
    static constexpr char prefix[] = "testing_";
    const char*           function = arg.function;
    if(!strncmp(function, prefix, sizeof(prefix) - 1))
        function += sizeof(prefix) - 1;

    if(!strcmp(function, "gemm") || !strcmp(function, "gemm_batched"))
    {
        // adjust dimension for GEMM routines
        int64_t min_lda = arg.transA == 'N' ? arg.M : arg.K;
        int64_t min_ldb = arg.transB == 'N' ? arg.K : arg.N;
        int64_t min_ldc = arg.M;

        if(arg.lda < min_lda)
        {
            std::cout << "hipblas-bench INFO: lda < min_lda, set lda = " << min_lda << std::endl;
            arg.lda = min_lda;
        }
        if(arg.ldb < min_ldb)
        {
            std::cout << "hipblas-bench INFO: ldb < min_ldb, set ldb = " << min_ldb << std::endl;
            arg.ldb = min_ldb;
        }
        if(arg.ldc < min_ldc)
        {
            std::cout << "hipblas-bench INFO: ldc < min_ldc, set ldc = " << min_ldc << std::endl;
            arg.ldc = min_ldc;
        }
    }
    else if(!strcmp(function, "gemm_strided_batched"))
    {
        // adjust dimension for GEMM routines
        int64_t min_lda = arg.transA == 'N' ? arg.M : arg.K;
        int64_t min_ldb = arg.transB == 'N' ? arg.K : arg.N;
        int64_t min_ldc = arg.M;
        if(arg.lda < min_lda)
        {
            std::cout << "hipblas-bench INFO: lda < min_lda, set lda = " << min_lda << std::endl;
            arg.lda = min_lda;
        }
        if(arg.ldb < min_ldb)
        {
            std::cout << "hipblas-bench INFO: ldb < min_ldb, set ldb = " << min_ldb << std::endl;
            arg.ldb = min_ldb;
        }
        if(arg.ldc < min_ldc)
        {
            std::cout << "hipblas-bench INFO: ldc < min_ldc, set ldc = " << min_ldc << std::endl;
            arg.ldc = min_ldc;
        }

        //      int64_t min_stride_a =
        //          arg.transA == 'N' ? arg.K * arg.lda : arg.M * arg.lda;
        //      int64_t min_stride_b =
        //          arg.transB == 'N' ? arg.N * arg.ldb : arg.K * arg.ldb;
        //      int64_t min_stride_a =
        //          arg.transA == 'N' ? arg.K * arg.lda : arg.M * arg.lda;
        //      int64_t min_stride_b =
        //          arg.transB == 'N' ? arg.N * arg.ldb : arg.K * arg.ldb;
        int64_t min_stride_c = arg.ldc * arg.N;
        //      if (arg.stride_a < min_stride_a)
        //      {
        //          std::cout << "hipblas-bench INFO: stride_a < min_stride_a, set stride_a = " <<
        //          min_stride_a << std::endl;
        //          arg.stride_a = min_stride_a;
        //      }
        //      if (arg.stride_b < min_stride_b)
        //      {
        //          std::cout << "hipblas-bench INFO: stride_b < min_stride_b, set stride_b = " <<
        //          min_stride_b << std::endl;
        //          arg.stride_b = min_stride_b;
        //      }
        if(arg.stride_c < min_stride_c)
        {
            std::cout << "hipblas-bench INFO: stride_c < min_stride_c, set stride_c = "
                      << min_stride_c << std::endl;
            arg.stride_c = min_stride_c;
        }
    }

    if(!strcmp(function, "gemm_ex") || !strcmp(function, "gemm_batched_ex"))
    {
        // adjust dimension for GEMM routines
        int64_t min_lda = arg.transA == 'N' ? arg.M : arg.K;
        int64_t min_ldb = arg.transB == 'N' ? arg.K : arg.N;
        int64_t min_ldc = arg.M;
        int64_t min_ldd = arg.M;

        if(arg.lda < min_lda)
        {
            std::cout << "hipblas-bench INFO: lda < min_lda, set lda = " << min_lda << std::endl;
            arg.lda = min_lda;
        }
        if(arg.ldb < min_ldb)
        {
            std::cout << "hipblas-bench INFO: ldb < min_ldb, set ldb = " << min_ldb << std::endl;
            arg.ldb = min_ldb;
        }
        if(arg.ldc < min_ldc)
        {
            std::cout << "hipblas-bench INFO: ldc < min_ldc, set ldc = " << min_ldc << std::endl;
            arg.ldc = min_ldc;
        }
        if(arg.ldd < min_ldd)
        {
            std::cout << "hipblas-bench INFO: ldd < min_ldd, set ldd = " << min_ldc << std::endl;
            arg.ldd = min_ldd;
        }
        hipblas_gemm_dispatch<perf_gemm_ex>(arg);
    }
    else if(!strcmp(function, "gemm_strided_batched_ex"))
    {
        // adjust dimension for GEMM routines
        int64_t min_lda = arg.transA == 'N' ? arg.M : arg.K;
        int64_t min_ldb = arg.transB == 'N' ? arg.K : arg.N;
        int64_t min_ldc = arg.M;
        int64_t min_ldd = arg.M;
        if(arg.lda < min_lda)
        {
            std::cout << "hipblas-bench INFO: lda < min_lda, set lda = " << min_lda << std::endl;
            arg.lda = min_lda;
        }
        if(arg.ldb < min_ldb)
        {
            std::cout << "hipblas-bench INFO: ldb < min_ldb, set ldb = " << min_ldb << std::endl;
            arg.ldb = min_ldb;
        }
        if(arg.ldc < min_ldc)
        {
            std::cout << "hipblas-bench INFO: ldc < min_ldc, set ldc = " << min_ldc << std::endl;
            arg.ldc = min_ldc;
        }
        if(arg.ldd < min_ldd)
        {
            std::cout << "hipblas-bench INFO: ldd < min_ldd, set ldd = " << min_ldc << std::endl;
            arg.ldd = min_ldd;
        }
        int64_t min_stride_c = arg.ldc * arg.N;
        if(arg.stride_c < min_stride_c)
        {
            std::cout << "hipblas-bench INFO: stride_c < min_stride_c, set stride_c = "
                      << min_stride_c << std::endl;
            arg.stride_c = min_stride_c;
        }

        hipblas_gemm_dispatch<perf_gemm_strided_batched_ex>(arg);
    }
    else if(!strcmp(function, "syrk_ex"))
    {
        hipblas_syrk_ex_dispatch<perf_syrk_ex>(arg);
    }
    else if(!strcmp(function, "herk_ex"))
    {
        hipblas_herk_ex_dispatch<perf_herk_ex>(arg);
    }
    else
    {
        if(!strcmp(function, "scal_ex") || !strcmp(function, "scal_batched_ex")
           || !strcmp(function, "scal_strided_batched_ex"))
            hipblas_blas1_ex_dispatch<perf_blas_scal_ex>(arg);
        /*
        if(!strcmp(function, "scal") || !strcmp(function, "scal_batched")
           || !strcmp(function, "scal_strided_batched"))
            hipblas_blas1_dispatch<perf_blas_scal>(arg);
        */
        else if(!strcmp(function, "rot") || !strcmp(function, "rot_batched")
                || !strcmp(function, "rot_strided_batched"))
            hipblas_rot_dispatch<perf_blas_rot>(arg);
        else if(!strcmp(function, "axpy_ex") || !strcmp(function, "axpy_batched_ex")
                || !strcmp(function, "axpy_strided_batched_ex"))
            hipblas_blas1_ex_dispatch<perf_blas_axpy_ex>(arg);
        else if(!strcmp(function, "dot_ex") || !strcmp(function, "dot_batched_ex")
                || !strcmp(function, "dot_strided_batched_ex") || !strcmp(function, "dotc_ex")
                || !strcmp(function, "dotc_batched_ex")
                || !strcmp(function, "dotc_strided_batched_ex"))
            hipblas_blas1_ex_dispatch<perf_blas_dot_ex>(arg);
        else if(!strcmp(function, "nrm2_ex") || !strcmp(function, "nrm2_batched_ex")
                || !strcmp(function, "nrm2_strided_batched_ex"))
            hipblas_blas1_ex_dispatch<perf_blas_nrm2_ex>(arg);
        else if(!strcmp(function, "rot_ex") || !strcmp(function, "rot_batched_ex")
                || !strcmp(function, "rot_strided_batched_ex"))
            hipblas_blas1_ex_dispatch<perf_blas_rot_ex>(arg);
        else
            hipblas_simple_dispatch<perf_blas>(arg);
    }
    return 0;
}
