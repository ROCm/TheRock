/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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

#include <chrono>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt_arguments.hpp>
#include <iostream>

#include "helper.h"

using clock_point  = std::chrono::time_point<std::chrono::steady_clock>;
using double_micro = std::chrono::duration<double, std::micro>;
using HHSRunner    = Runner<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float, float>;

size_t warmups     = 10;
size_t hotIters    = 1000;
size_t getAllIters = 100; // used for when requested_solution != 1
int    algoIdx;

double_micro total_getHeur         = double_micro::zero();
double_micro total_getHeur100      = double_micro::zero();
double_micro total_getHeurNeg1     = double_micro::zero();
double_micro total_matmul          = double_micro::zero();
double_micro total_ext_getHeur     = double_micro::zero(); // start of ext
double_micro total_ext_getHeur100  = double_micro::zero();
double_micro total_ext_getHeurNeg1 = double_micro::zero();
double_micro total_ext_getAll      = double_micro::zero();
double_micro total_ext_getByIdx    = double_micro::zero();
double_micro total_run             = double_micro::zero();

double_micro best_getHeur         = double_micro::max();
double_micro best_getHeur100      = double_micro::max();
double_micro best_getHeurNeg1     = double_micro::max();
double_micro best_matmul          = double_micro::max();
double_micro best_ext_getHeur     = double_micro::max(); // start of ext
double_micro best_ext_getHeur100  = double_micro::max();
double_micro best_ext_getHeurNeg1 = double_micro::max();
double_micro best_ext_getAll      = double_micro::max();
double_micro best_ext_getByIdx    = double_micro::max();
double_micro best_run             = double_micro::max();

void printUsage(char* programName)
{
    std::cout << "Usage: " << programName << " <options>\n"
              << "options:\n"
              << "\t-h, --help\t\t\tShow this help message\n"
              << "\t-i, --iters\t\t\tHot iterations, default is 1000.\n"
              << "\t-j, --cold_iters\t\t\tWarmup iterations, default is 10.\n"
              << "\t-k, --get_all_iters\t\t\tHot iterations for getAll, default is 100.\n";
}

int parseArgs(int argc, char** argv, size_t& cold, size_t& hot, size_t& getAllIters)
{
    if(argc <= 1)
    {
        return EXIT_SUCCESS;
    }

    for(int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if((arg.at(0) == '-') || ((arg.at(0) == '-') && (arg.at(1) == '-')))
        {
            if((arg == "-h") || (arg == "--help"))
            {
                return EXIT_FAILURE;
            }
            else if(arg == "-i" || arg == "--iters")
            {
                hot = std::stoul(argv[++i]);
            }
            else if(arg == "-j" || arg == "--cold_iters")
            {
                cold = std::stoul(argv[++i]);
            }
            else if(arg == "-k" || arg == "--get_all_iters")
            {
                getAllIters = std::stoul(argv[++i]);
            }
        }
        else
        {
            std::cerr << "error with " << arg << std::endl;
            std::cerr << "option must start with - or --" << std::endl << std::endl;
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}

void simpleGemm(hipblasLtHandle_t  handle,
                hipblasOperation_t trans_a,
                hipblasOperation_t trans_b,
                int64_t            m,
                int64_t            n,
                int64_t            k,
                int64_t            batch_count,
                float&             alpha,
                float&             beta,
                void*              d_a,
                void*              d_b,
                void*              d_c,
                void*              d_d,
                void*              d_workspace,
                int64_t            max_workspace_size,
                hipStream_t        stream,
                int                requested_solutions,
                int&               returned_solutions);

void simpleGemmExt(hipblasLtHandle_t  handle,
                   hipblasOperation_t trans_a,
                   hipblasOperation_t trans_b,
                   int64_t            m,
                   int64_t            n,
                   int64_t            k,
                   int64_t            batch_count,
                   float&             alpha,
                   float&             beta,
                   void*              d_a,
                   void*              d_b,
                   void*              d_c,
                   void*              d_d,
                   void*              d_workspace,
                   int64_t            max_workspace_size,
                   hipStream_t        stream,
                   int                requested_solutions,
                   int&               returned_solutions);

void simpleGemmGetAllAlgosExt(hipblasLtHandle_t  handle,
                              hipblasOperation_t trans_a,
                              hipblasOperation_t trans_b,
                              int64_t            m,
                              int64_t            n,
                              int64_t            k,
                              int64_t            batch_count,
                              float&             alpha,
                              float&             beta,
                              void*              d_a,
                              void*              d_b,
                              void*              d_c,
                              void*              d_d,
                              void*              d_workspace,
                              int64_t            max_workspace_size,
                              hipStream_t        stream,
                              int&               returned_solutions);

void simpleGemmGetAlgoByIndexExt(hipblasLtHandle_t  handle,
                                 hipblasOperation_t trans_a,
                                 hipblasOperation_t trans_b,
                                 int64_t            m,
                                 int64_t            n,
                                 int64_t            k,
                                 int64_t            batch_count,
                                 float&             alpha,
                                 float&             beta,
                                 void*              d_a,
                                 void*              d_b,
                                 void*              d_c,
                                 void*              d_d,
                                 void*              d_workspace,
                                 int64_t            max_workspace_size,
                                 hipStream_t        stream,
                                 int&               returned_solutions);

int calcOverheadGetHeuristic(int requested_solutions = 1);
int calcOverheadExtGetHeuristic(int requested_solutions = 1);
int calcOverheadExtGetAllAlgos();
int calcOverheadExtGetAlgoByIdx();

int main(int argc, char** argv)
{
    if(auto err = parseArgs(argc, argv, warmups, hotIters, getAllIters))
    {
        printUsage(argv[0]);
        return err;
    }

    int returned_sol;
    // new: test get heuristic with requested-solution = 100, -1
    int requested_solutions = 1;

    // ----------- Heuristic ----------- //
    std::cout << "[overhead]:function,api_name,num_sols,us/iter,best_us\n";
    // calls hipblasLtMatmul hotIters times
    returned_sol = calcOverheadGetHeuristic();
    std::cout << "api_overhead,hipblasLtMatmulAlgoGetHeuristic," << std::to_string(returned_sol)
              << "," << std::to_string(total_getHeur.count() / hotIters) << ","
              << std::to_string(best_getHeur.count()) << std::endl;

    // calls ext::run hotIters times
    returned_sol = calcOverheadExtGetHeuristic();
    std::cout << "api_overhead,hipblaslt_ext::algoGetHeuristic," << std::to_string(returned_sol)
              << "," << std::to_string(total_ext_getHeur.count() / hotIters) << ","
              << std::to_string(best_ext_getHeur.count()) << std::endl;

    // ----------- GetAll ----------- //
    // won't calls ext::run, and note that the hot iter is getAllIters
    returned_sol = calcOverheadExtGetAllAlgos();
    std::cout << "api_overhead,hipblaslt_ext::getAllAlgos," << std::to_string(returned_sol) << ","
              << std::to_string(total_ext_getAll.count() / getAllIters) << ","
              << std::to_string(best_ext_getAll.count()) << std::endl;

    // ----------- GetByIdx ----------- //
    // calls ext::run hotIters times
    returned_sol = calcOverheadExtGetAlgoByIdx();
    std::cout << "api_overhead,hipblaslt_ext::getAlgosFromIndex," << std::to_string(returned_sol)
              << "," << std::to_string(total_ext_getByIdx.count() / hotIters) << ","
              << std::to_string(best_ext_getByIdx.count()) << std::endl;

    // ----------- GetHeuristic(100) ----------- //
    requested_solutions = 100;
    // won't calls hipblasLtMatmul if requested_solutions != 1, and note that the hot iter is getAllIters
    returned_sol = calcOverheadGetHeuristic(requested_solutions);
    std::cout << "api_overhead,hipblasLtMatmulAlgoGetHeuristic[100],"
              << std::to_string(returned_sol) << ","
              << std::to_string(total_getHeur100.count() / getAllIters) << ","
              << std::to_string(best_getHeur100.count()) << std::endl;
    // won't calls ext::run if requested_solutions != 1, and note that the hot iter is getAllIters
    returned_sol = calcOverheadExtGetHeuristic(requested_solutions);
    std::cout << "api_overhead,hipblaslt_ext::algoGetHeuristic[100],"
              << std::to_string(returned_sol) << ","
              << std::to_string(total_ext_getHeur100.count() / getAllIters) << ","
              << std::to_string(best_ext_getHeur100.count()) << std::endl;

    // ----------- GetHeuristic(-1) ----------- //
    requested_solutions = -1;
    // won't calls hipblasLtMatmul if requested_solutions != 1, and note that the hot iter is getAllIters
    returned_sol = calcOverheadGetHeuristic(requested_solutions);
    std::cout << "api_overhead,hipblasLtMatmulAlgoGetHeuristic[Neg1],"
              << std::to_string(returned_sol) << ","
              << std::to_string(total_getHeurNeg1.count() / getAllIters) << ","
              << std::to_string(best_getHeurNeg1.count()) << std::endl;
    // won't calls ext::run if requested_solutions != 1, and note that the hot iter is getAllIters
    returned_sol = calcOverheadExtGetHeuristic(requested_solutions);
    std::cout << "api_overhead,hipblaslt_ext::algoGetHeuristic[Neg1],"
              << std::to_string(returned_sol) << ","
              << std::to_string(total_ext_getHeurNeg1.count() / getAllIters) << ","
              << std::to_string(best_ext_getHeurNeg1.count()) << std::endl;

    // ----------- matmul ----------- //
    // put matmul exec time in the end
    returned_sol = 0;
    std::cout << "api_overhead,hipblasLtMatmul," << std::to_string(returned_sol) << ","
              << std::to_string(total_matmul.count() / hotIters) << ","
              << std::to_string(best_matmul.count()) << std::endl;
    std::cout << "api_overhead,hipblaslt_ext::run," << std::to_string(returned_sol) << ","
              << std::to_string(total_run.count() / (hotIters * 2)) << ","
              << std::to_string(best_run.count()) << std::endl;

    return 0;
}

/////////////////////////////////////////
// hipBLASLt GetHeuristic & MatMul
/////////////////////////////////////////
int calcOverheadGetHeuristic(int requested_solutions)
{
    requested_solutions
        = requested_solutions < 0 ? HIPBLASLT_MAX_REQUESTED_SOLUTION_NUM : requested_solutions;

    HHSRunner runner(1024, 512, 1024, 1, 1.f, 1.f, 32 * 1024 * 1024);
    int       returned_sols = 0;
    runner.run([&] {
        simpleGemm(runner.handle,
                   HIPBLAS_OP_N,
                   HIPBLAS_OP_N,
                   runner.m,
                   runner.n,
                   runner.k,
                   runner.batch_count,
                   runner.alpha,
                   runner.beta,
                   runner.d_a,
                   runner.d_b,
                   runner.d_c,
                   runner.d_d,
                   runner.d_workspace,
                   runner.max_workspace_size,
                   runner.stream,
                   requested_solutions,
                   returned_sols);
    });

    return returned_sols;
}

void simpleGemm(hipblasLtHandle_t  handle,
                hipblasOperation_t trans_a,
                hipblasOperation_t trans_b,
                int64_t            m,
                int64_t            n,
                int64_t            k,
                int64_t            batch_count,
                float&             alpha,
                float&             beta,
                void*              d_a,
                void*              d_b,
                void*              d_c,
                void*              d_d,
                void*              d_workspace,
                int64_t            max_workspace_size,
                hipStream_t        stream,
                int                requested_solutions,
                int&               returned_solutions)
{
    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, m, k, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, k, n, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, m, n, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, HIP_R_16F, m, n, m));

    if(batch_count > 1)
    {
        int64_t stride_a = m * k;
        int64_t stride_b = k * n;
        int64_t stride_c = m * n;
        int64_t stride_d = m * n;
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matA, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_a, sizeof(stride_a)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matB, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_b, sizeof(stride_b)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matC, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matC, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_c, sizeof(stride_c)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matD, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matD, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_d, sizeof(stride_d)));
    }

    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));

    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    // Set User Preference attributes
    hipblasLtMatmulPreference_t pref;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatmulPreferenceSetAttribute(pref,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &max_workspace_size,
                                              sizeof(max_workspace_size)));

    auto freeResource = [&]() {
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
    };

    //
    // warm-ups
    //
    for(size_t iter = 0; iter < warmups; ++iter)
    {
        std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult(requested_solutions);
        int                                           returnedAlgoCount = 0;
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                              matmul,
                                                              matA,
                                                              matB,
                                                              matC,
                                                              matD,
                                                              pref,
                                                              requested_solutions,
                                                              heuristicResult.data(),
                                                              &returnedAlgoCount));

        returned_solutions = returnedAlgoCount;
        if(returnedAlgoCount == 0)
        {
            std::cerr << "No valid solution found!" << std::endl;
            freeResource();
            return;
        }

        uint64_t workspace_size = 0;
        if(requested_solutions == 1)
        {
            for(int i = 0; i < returnedAlgoCount; i++)
                workspace_size = max(workspace_size, heuristicResult[i].workspaceSize);

            CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle,
                                                  matmul,
                                                  &alpha,
                                                  d_a,
                                                  matA,
                                                  d_b,
                                                  matB,
                                                  &beta,
                                                  d_c,
                                                  matC,
                                                  d_d,
                                                  matD,
                                                  &heuristicResult[0].algo,
                                                  d_workspace,
                                                  workspace_size,
                                                  stream));
        }
    }

    clock_point start_getHeur, end_getHeur;
    clock_point start_matmul, end_matmul;
    size_t      iters = (requested_solutions == 1) ? hotIters : getAllIters;

    //
    // hot-iters
    //
    for(size_t iter = 0; iter < iters; ++iter)
    {
        std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult(requested_solutions);
        int                                           returnedAlgoCount = 0;

        start_getHeur = std::chrono::steady_clock::now();
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                              matmul,
                                                              matA,
                                                              matB,
                                                              matC,
                                                              matD,
                                                              pref,
                                                              requested_solutions,
                                                              heuristicResult.data(),
                                                              &returnedAlgoCount));
        end_getHeur           = std::chrono::steady_clock::now();
        double_micro duration = end_getHeur - start_getHeur;
        if(requested_solutions == 1)
        {
            best_getHeur = std::min(best_getHeur, duration);
            total_getHeur += duration;
        }
        else if(requested_solutions == HIPBLASLT_MAX_REQUESTED_SOLUTION_NUM)
        {
            best_getHeurNeg1 = std::min(best_getHeurNeg1, duration);
            total_getHeurNeg1 += duration;
        }
        else
        {
            best_getHeur100 = std::min(best_getHeur100, duration);
            total_getHeur100 += duration;
        }

        returned_solutions = returnedAlgoCount;
        if(returnedAlgoCount == 0)
        {
            std::cerr << "No valid solution found!" << std::endl;
            freeResource();
            return;
        }

        uint64_t workspace_size = 0;
        if(requested_solutions == 1)
        {
            for(int i = 0; i < returnedAlgoCount; i++)
                workspace_size = max(workspace_size, heuristicResult[i].workspaceSize);

            start_matmul = std::chrono::steady_clock::now();
            CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle,
                                                  matmul,
                                                  &alpha,
                                                  d_a,
                                                  matA,
                                                  d_b,
                                                  matB,
                                                  &beta,
                                                  d_c,
                                                  matC,
                                                  d_d,
                                                  matD,
                                                  &heuristicResult[0].algo,
                                                  d_workspace,
                                                  workspace_size,
                                                  stream));
            end_matmul  = std::chrono::steady_clock::now();
            duration    = end_matmul - start_matmul;
            best_matmul = std::min(best_matmul, duration);
            total_matmul += duration;
        }
    }

    freeResource();

    return;
}

/////////////////////////////////////////
// hipBLASLt_ext - algoGetHeuristic & run
/////////////////////////////////////////
int calcOverheadExtGetHeuristic(int requested_solutions)
{
    requested_solutions
        = requested_solutions < 0 ? HIPBLASLT_MAX_REQUESTED_SOLUTION_NUM : requested_solutions;

    HHSRunner runner(1024, 512, 1024, 1, 1.f, 1.f, 32 * 1024 * 1024);
    int       returned_sols = 0;
    runner.run([&] {
        simpleGemmExt(runner.handle,
                      HIPBLAS_OP_N,
                      HIPBLAS_OP_N,
                      runner.m,
                      runner.n,
                      runner.k,
                      runner.batch_count,
                      runner.alpha,
                      runner.beta,
                      runner.d_a,
                      runner.d_b,
                      runner.d_c,
                      runner.d_d,
                      runner.d_workspace,
                      runner.max_workspace_size,
                      runner.stream,
                      requested_solutions,
                      returned_sols);
    });

    return returned_sols;
}

void simpleGemmExt(hipblasLtHandle_t  handle,
                   hipblasOperation_t trans_a,
                   hipblasOperation_t trans_b,
                   int64_t            m,
                   int64_t            n,
                   int64_t            k,
                   int64_t            batch_count,
                   float&             alpha,
                   float&             beta,
                   void*              d_a,
                   void*              d_b,
                   void*              d_c,
                   void*              d_d,
                   void*              d_workspace,
                   int64_t            max_workspace_size,
                   hipStream_t        stream,
                   int                requested_solutions,
                   int&               returned_solutions)
{
    hipblaslt_ext::GemmPreference gemmPref;
    gemmPref.setMaxWorkspaceBytes(max_workspace_size);
    hipblaslt_ext::Gemm gemm(
        handle, trans_a, trans_b, HIP_R_16F, HIP_R_16F, HIP_R_16F, HIP_R_16F, HIPBLAS_COMPUTE_32F);

    hipblaslt_ext::GemmEpilogue
        epilogue; // No action needed, default is HIPBLASLT_EPILOGUE_DEFAULT. (Gemm only)
    hipblaslt_ext::GemmInputs inputs;
    inputs.setA(d_a);
    inputs.setB(d_b);
    inputs.setC(d_c);
    inputs.setD(d_d);
    inputs.setAlpha(&alpha);
    inputs.setBeta(&beta);
    gemm.setProblem(m, n, k, batch_count, epilogue, inputs);

    //
    // warm-ups
    //
    for(size_t iter = 0; iter < warmups; ++iter)
    {
        std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
        CHECK_HIPBLASLT_ERROR(
            gemm.algoGetHeuristic(requested_solutions, gemmPref, heuristicResult));

        returned_solutions = heuristicResult.size();
        if(heuristicResult.empty())
        {
            std::cerr << "[GemmExt]:No valid solution found!" << std::endl;
            return;
        }

        if(requested_solutions == 1)
        {
            // Make sure to initialize every time when algo changes
            gemm.setMaxWorkspaceBytes(max_workspace_size);
            CHECK_HIPBLASLT_ERROR(gemm.initialize(heuristicResult[0].algo, d_workspace));
            CHECK_HIPBLASLT_ERROR(gemm.run(stream));
        }
    }

    clock_point start_ext_getHeur, end_ext_getHeur;
    clock_point start_ext_run, end_ext_run;
    size_t      iters = (requested_solutions == 1) ? hotIters : getAllIters;

    //
    // hot-iters
    //
    for(size_t iter = 0; iter < iters; ++iter)
    {
        std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;

        start_ext_getHeur = std::chrono::steady_clock::now();
        CHECK_HIPBLASLT_ERROR(
            gemm.algoGetHeuristic(requested_solutions, gemmPref, heuristicResult));
        end_ext_getHeur       = std::chrono::steady_clock::now();
        double_micro duration = end_ext_getHeur - start_ext_getHeur;

        if(requested_solutions == 1)
        {
            best_ext_getHeur = std::min(best_ext_getHeur, duration);
            total_ext_getHeur += duration;
        }
        else if(requested_solutions == HIPBLASLT_MAX_REQUESTED_SOLUTION_NUM)
        {
            best_ext_getHeurNeg1 = std::min(best_ext_getHeurNeg1, duration);
            total_ext_getHeurNeg1 += duration;
        }
        else
        {
            best_ext_getHeur100 = std::min(best_ext_getHeur100, duration);
            total_ext_getHeur100 += duration;
        }

        returned_solutions = heuristicResult.size();
        if(heuristicResult.empty())
        {
            std::cerr << "[GemmExt]:No valid solution found!" << std::endl;
            return;
        }

        if(requested_solutions == 1)
        {
            // save the algoIdx for testing getByIdx
            algoIdx = hipblaslt_ext::getIndexFromAlgo(heuristicResult[0].algo);

            // Make sure to initialize every time when algo changes
            gemm.setMaxWorkspaceBytes(max_workspace_size);
            CHECK_HIPBLASLT_ERROR(gemm.initialize(heuristicResult[0].algo, d_workspace));

            start_ext_run = std::chrono::steady_clock::now();
            CHECK_HIPBLASLT_ERROR(gemm.run(stream));
            end_ext_run = std::chrono::steady_clock::now();
            duration    = end_ext_run - start_ext_run;
            best_run    = std::min(best_run, duration);
            total_run += duration;
        }
    }

    return;
}

/////////////////////////////////////////
// hipBLASLt_ext - getAllAlgos & run
/////////////////////////////////////////
int calcOverheadExtGetAllAlgos()
{
    HHSRunner runner(1024, 512, 1024, 1, 1.f, 1.f, 32 * 1024 * 1024);
    int       returned_sols = 0;
    runner.run([&runner, &returned_sols] {
        simpleGemmGetAllAlgosExt(runner.handle,
                                 HIPBLAS_OP_N,
                                 HIPBLAS_OP_N,
                                 runner.m,
                                 runner.n,
                                 runner.k,
                                 runner.batch_count,
                                 runner.alpha,
                                 runner.beta,
                                 runner.d_a,
                                 runner.d_b,
                                 runner.d_c,
                                 runner.d_d,
                                 runner.d_workspace,
                                 runner.max_workspace_size,
                                 runner.stream,
                                 returned_sols);
    });

    return returned_sols;
}

void simpleGemmGetAllAlgosExt(hipblasLtHandle_t  handle,
                              hipblasOperation_t trans_a,
                              hipblasOperation_t trans_b,
                              int64_t            m,
                              int64_t            n,
                              int64_t            k,
                              int64_t            batch_count,
                              float&             alpha,
                              float&             beta,
                              void*              d_a,
                              void*              d_b,
                              void*              d_c,
                              void*              d_d,
                              void*              d_workspace,
                              int64_t            max_workspace_size,
                              hipStream_t        stream,
                              int&               returned_solutions)
{
    hipblaslt_ext::GemmPreference gemmPref;
    gemmPref.setMaxWorkspaceBytes(max_workspace_size);
    hipblaslt_ext::Gemm gemm(
        handle, trans_a, trans_b, HIP_R_16F, HIP_R_16F, HIP_R_16F, HIP_R_16F, HIPBLAS_COMPUTE_32F);

    hipblaslt_ext::GemmEpilogue epilogue;
    hipblaslt_ext::GemmInputs   inputs;
    inputs.setA(d_a);
    inputs.setB(d_b);
    inputs.setC(d_c);
    inputs.setD(d_d);
    inputs.setAlpha(&alpha);
    inputs.setBeta(&beta);
    gemm.setProblem(m, n, k, batch_count, epilogue, inputs);

    //
    // warm-ups
    //
    for(size_t iter = 0; iter < warmups; ++iter)
    {
        std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
        CHECK_HIPBLASLT_ERROR(hipblaslt_ext::getAllAlgos(handle,
                                                         hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                                         trans_a,
                                                         trans_a,
                                                         HIP_R_16F,
                                                         HIP_R_16F,
                                                         HIP_R_16F,
                                                         HIP_R_16F,
                                                         HIPBLAS_COMPUTE_32F,
                                                         heuristicResult));

        uint64_t            workspace_size = 0;
        std::vector<size_t> validIdx;
        for(size_t i = 0; i < heuristicResult.size(); i++)
        {
            size_t workspaceSizeInBytes = 0;
            if(gemm.isAlgoSupported(heuristicResult[i].algo, workspaceSizeInBytes)
               == HIPBLAS_STATUS_SUCCESS)
            {
                if(workspaceSizeInBytes <= max_workspace_size)
                {
                    workspace_size = max(workspace_size, workspaceSizeInBytes);
                    validIdx.push_back(i);
                }
            }
        }

        returned_solutions = validIdx.size();
        if(validIdx.empty())
        {
            std::cerr << "[GetAllAlgosExt]:No valid solution found!" << std::endl;
            return;
        }
    }

    clock_point start_ext_getAll, end_ext_getAll;
    clock_point start_ext_run, end_ext_run;

    //
    // hot-iters
    //
    for(size_t iter = 0; iter < getAllIters; ++iter)
    {
        std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
        start_ext_getAll = std::chrono::steady_clock::now();
        CHECK_HIPBLASLT_ERROR(hipblaslt_ext::getAllAlgos(handle,
                                                         hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                                         trans_a,
                                                         trans_a,
                                                         HIP_R_16F,
                                                         HIP_R_16F,
                                                         HIP_R_16F,
                                                         HIP_R_16F,
                                                         HIPBLAS_COMPUTE_32F,
                                                         heuristicResult));
        end_ext_getAll        = std::chrono::steady_clock::now();
        double_micro duration = end_ext_getAll - start_ext_getAll;
        best_ext_getAll       = std::min(best_ext_getAll, duration);
        total_ext_getAll += duration;

        uint64_t            workspace_size = 0;
        std::vector<size_t> validIdx;
        for(size_t i = 0; i < heuristicResult.size(); i++)
        {
            size_t workspaceSizeInBytes = 0;
            if(gemm.isAlgoSupported(heuristicResult[i].algo, workspaceSizeInBytes)
               == HIPBLAS_STATUS_SUCCESS)
            {
                if(workspaceSizeInBytes <= max_workspace_size)
                {
                    workspace_size = max(workspace_size, workspaceSizeInBytes);
                    validIdx.push_back(i);
                }
            }
        }

        returned_solutions = validIdx.size();
        if(validIdx.empty())
        {
            std::cerr << "[GetAllAlgosExt]:No valid solution found!" << std::endl;
            return;
        }
    }

    return;
}

/////////////////////////////////////////
// hipBLASLt_ext - getAlgosFromIndex & run
/////////////////////////////////////////
int calcOverheadExtGetAlgoByIdx()
{
    HHSRunner runner(1024, 512, 1024, 1, 1.f, 1.f, 32 * 1024 * 1024);
    int       returned_sols = 0;
    runner.run([&runner, &returned_sols] {
        simpleGemmGetAlgoByIndexExt(runner.handle,
                                    HIPBLAS_OP_N,
                                    HIPBLAS_OP_N,
                                    runner.m,
                                    runner.n,
                                    runner.k,
                                    runner.batch_count,
                                    runner.alpha,
                                    runner.beta,
                                    runner.d_a,
                                    runner.d_b,
                                    runner.d_c,
                                    runner.d_d,
                                    runner.d_workspace,
                                    runner.max_workspace_size,
                                    runner.stream,
                                    returned_sols);
    });

    return returned_sols;
}

void simpleGemmGetAlgoByIndexExt(hipblasLtHandle_t  handle,
                                 hipblasOperation_t trans_a,
                                 hipblasOperation_t trans_b,
                                 int64_t            m,
                                 int64_t            n,
                                 int64_t            k,
                                 int64_t            batch_count,
                                 float&             alpha,
                                 float&             beta,
                                 void*              d_a,
                                 void*              d_b,
                                 void*              d_c,
                                 void*              d_d,
                                 void*              d_workspace,
                                 int64_t            max_workspace_size,
                                 hipStream_t        stream,
                                 int&               returned_solutions)
{
    hipblaslt_ext::GemmPreference gemmPref;
    gemmPref.setMaxWorkspaceBytes(max_workspace_size);
    hipblaslt_ext::Gemm gemm(
        handle, trans_a, trans_b, HIP_R_16F, HIP_R_16F, HIP_R_16F, HIP_R_16F, HIPBLAS_COMPUTE_32F);

    hipblaslt_ext::GemmEpilogue
        epilogue; // No action needed, default is HIPBLASLT_EPILOGUE_DEFAULT. (Gemm only)
    hipblaslt_ext::GemmInputs inputs;
    inputs.setA(d_a);
    inputs.setB(d_b);
    inputs.setC(d_c);
    inputs.setD(d_d);
    inputs.setAlpha(&alpha);
    inputs.setBeta(&beta);
    gemm.setProblem(m, n, k, batch_count, epilogue, inputs);

    //
    // warm-ups
    //
    for(size_t iter = 0; iter < warmups; ++iter)
    {
        // only find one solution with previously saved algoIdx
        std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
        std::vector<int>                              algoIndexVec{algoIdx};
        std::vector<hipblasLtMatmulHeuristicResult_t> testResults;
        if(HIPBLAS_STATUS_INVALID_VALUE
           == hipblaslt_ext::getAlgosFromIndex(handle, algoIndexVec, testResults))
        {
            std::cout << "Indexes are all out of bound." << std::endl;
            break;
        }

        for(size_t i = 0; i < testResults.size(); i++)
        {
            size_t workspaceSizeInBytes = 0;
            size_t workspace_size       = 0;
            if(gemm.isAlgoSupported(testResults[i].algo, workspaceSizeInBytes)
               == HIPBLAS_STATUS_SUCCESS)
            {
                if(workspaceSizeInBytes <= max_workspace_size)
                {
                    workspace_size = max(workspace_size, workspaceSizeInBytes);
                    // std::cout << "Algo index found: "
                    //           << hipblaslt_ext::getIndexFromAlgo(testResults[i].algo) << std::endl;
                    heuristicResult.push_back(testResults[i]);
                    break;
                }
            }
        }

        returned_solutions = heuristicResult.size();
        if(heuristicResult.empty())
        {
            std::cerr << "[GetAlgoByIndexExt]:No valid solution found!" << std::endl;
            return;
        }

        // Make sure to initialize every time when algo changes
        gemm.setMaxWorkspaceBytes(max_workspace_size);
        CHECK_HIPBLASLT_ERROR(gemm.initialize(heuristicResult[0].algo, d_workspace));
        CHECK_HIPBLASLT_ERROR(gemm.run(stream));
    }

    clock_point start_ext_getByIdx, end_ext_getByIdx;
    clock_point start_ext_run, end_ext_run;

    //
    // hot-iters
    //
    for(size_t iter = 0; iter < hotIters; ++iter)
    {
        // only find one solution with previously saved algoIdx
        std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
        std::vector<int>                              algoIndexVec{algoIdx};
        std::vector<hipblasLtMatmulHeuristicResult_t> testResults;

        start_ext_getByIdx = std::chrono::steady_clock::now();
        if(HIPBLAS_STATUS_INVALID_VALUE
           == hipblaslt_ext::getAlgosFromIndex(handle, algoIndexVec, testResults))
        {
            std::cout << "Indexes are all out of bound." << std::endl;
            break;
        }
        end_ext_getByIdx      = std::chrono::steady_clock::now();
        double_micro duration = end_ext_getByIdx - start_ext_getByIdx;
        best_ext_getByIdx     = std::min(best_ext_getByIdx, duration);
        total_ext_getByIdx += duration;

        for(size_t i = 0; i < testResults.size(); i++)
        {
            size_t workspaceSizeInBytes = 0;
            size_t workspace_size       = 0;
            if(gemm.isAlgoSupported(testResults[i].algo, workspaceSizeInBytes)
               == HIPBLAS_STATUS_SUCCESS)
            {
                if(workspaceSizeInBytes <= max_workspace_size)
                {
                    workspace_size = max(workspace_size, workspaceSizeInBytes);
                    // std::cout << "Algo index found: "
                    //           << hipblaslt_ext::getIndexFromAlgo(testResults[i].algo) << std::endl;
                    heuristicResult.push_back(testResults[i]);
                    break;
                }
            }
        }

        returned_solutions = heuristicResult.size();
        if(heuristicResult.empty())
        {
            std::cerr << "[GetAlgoByIndexExt]:No valid solution found!" << std::endl;
            return;
        }

        // Make sure to initialize every time when algo changes
        gemm.setMaxWorkspaceBytes(max_workspace_size);
        CHECK_HIPBLASLT_ERROR(gemm.initialize(heuristicResult[0].algo, d_workspace));

        start_ext_run = std::chrono::steady_clock::now();
        CHECK_HIPBLASLT_ERROR(gemm.run(stream));
        end_ext_run = std::chrono::steady_clock::now();
        duration    = end_ext_run - start_ext_run;
        best_run    = std::min(best_run, duration);
        total_run += duration;
    }

    return;
}
