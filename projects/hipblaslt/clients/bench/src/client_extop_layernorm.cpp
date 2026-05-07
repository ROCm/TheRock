/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc.
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
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt-ext-op.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt_datatype2string.hpp>
#include <hipblaslt_init.hpp>
#include <iostream>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

void printUsage(char* programName)
{
    std::cout << "Usage: " << programName << " <options>\n"
              << "options:\n"
              << "\t-h, --help\t\t\tShow this help message\n"
              << "\t-m, --m\t\t\t\tSize of dim 0, default is 1335\n"
              << "\t-n, --n\t\t\t\tSize of dim 1, default is 16\n"
              << "\t-a, --affine\t\t\t\tEnable Gamma and Beta, default is false\n"
              << "\t--initialization \t\tInitialize matrix data. Options: rand_int, trig_float, "
                 "hpl(floating), special, zero. (default is hpl)\n";
}

void cpuLayerNorm(float*        out,
                  float*        gpuMean,
                  float*        gpuInvvar,
                  float*        in,
                  std::uint32_t batch,
                  std::uint32_t length,
                  float         eps   = 1e-05,
                  float*        gamma = nullptr,
                  float*        beta  = nullptr)
{
    // calculate gpuMean
    for(int i = 0; i < batch; i++)
    {
        int    count = 0;
        float* inC   = in + i * length;
        float* outC  = out + i * length;

        for(int j = 0; j < length; j++)
        {
            count        = count + 1;
            float delta  = inC[j] - gpuMean[i];
            gpuMean[i]   = gpuMean[i] + delta / count;
            float delta2 = inC[j] - gpuMean[i];
            gpuInvvar[i] = gpuInvvar[i] + delta * delta2;
        }
        gpuInvvar[i] = 1 / std::sqrt((gpuInvvar[i] / length) + eps);

        // calculate gpuInvvar
        for(int j = 0; j < length; j++)
        {
            outC[j] = (inC[j] - gpuMean[i]) * gpuInvvar[i];

            if(gamma != nullptr)
                outC[j] = outC[j] * gamma[j];

            if(beta != nullptr)
                outC[j] = outC[j] + beta[j];
        }
    }
}

int parseArgs(
    int argc, char** argv, size_t* m, size_t* n, bool* affine, hipblaslt_initialization* init)
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

            if(arg == "-m" || arg == "--m")
            {
                *m = std::stoul(argv[++i]);
            }
            else if(arg == "-n" || arg == "--n")
            {
                *n = std::stoul(argv[++i]);
            }
            else if(arg == "-a" || arg == "--affine")
            {
                *affine = std::stoul(argv[++i]);
            }
            else if(arg == "--initialization" || arg == "--init")
            {
                const std::string initStr{argv[++i]};

                if(initStr != "rand_int" && initStr != "trig_float" && initStr != "hpl" && initStr != "special" && initStr != "zero")
                {
                    std::cerr << "Invalid initialization type: " << initStr << '\n';
                    return EXIT_FAILURE;
                }

                *init = string2hipblaslt_initialization(initStr);
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

void dumpBuffer(const char* title, float* data, int M, int N)
{
    std::cout << "----- " << title << "----- " << std::endl;
    for(int m = 0; m < M; m++)
    {
        for(int n = 0; n < N; n++)
        {
            std::cout << data[m * N + n] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
void compare(const char* title, const std::vector<T>& cpuOutput, const std::vector<T>& refOutput)
{
    float maxErr = 0.0;
    int   gpunan = 0;
    int   cpunan = 0;
    int   gpuinf = 0;
    int   cpuinf = 0;
    for(int i = 0; i < cpuOutput.size(); i++)
    {
        float err = std::abs(refOutput[i] - cpuOutput[i]);
        maxErr    = (maxErr > err) ? maxErr : err;
        if(std::isnan(cpuOutput[i]))
        {
            gpunan += 1;
        }
        if(std::isnan(refOutput[i]))
        {
            cpunan += 1;
        }
        if(std::isinf(cpuOutput[i]))
        {
            gpuinf += 1;
        }
        if(std::isinf(refOutput[i]))
        {
            cpuinf += 1;
        }
    }

    std::cout << "----- " << title << " result"
              << " -----" << std::endl;
    if(gpunan)
        std::cout << "gpunan: " << gpunan << std::endl;
    if(cpunan)
        std::cout << "cpunan: " << cpunan << std::endl;
    if(gpuinf)
        std::cout << "gpuinf: " << gpuinf << std::endl;
    if(cpuinf)
        std::cout << "cpuinf: " << cpuinf << std::endl;
    std::cout << "max error : " << maxErr << std::endl;
}

template <typename DType>
void initData(DType* data, std::size_t numElements, hipblaslt_initialization initMethod)
{
    switch(initMethod)
    {
    case hipblaslt_initialization::rand_int:
        hipblaslt_init<DType>(data, numElements, 1, 1);
        break;
    case hipblaslt_initialization::trig_float:
        hipblaslt_init_cos<DType>(data, numElements, 1, 1);
        break;
    case hipblaslt_initialization::hpl:
        hipblaslt_init_hpl<DType>(data, numElements, 1, 1);
        break;
    case hipblaslt_initialization::special:
        hipblaslt_init_alt_impl_big<DType>(data, numElements, 1, 1);
        break;
    case hipblaslt_initialization::zero:
        hipblaslt_init_zero<DType>(data, numElements, 1, 1);
        break;
    default:
        break;
    }
}

int main(int argc, char** argv)
{
    std::size_t              m{1};
    std::size_t              n{64};
    bool                     affine{false};
    hipblaslt_initialization init{hipblaslt_initialization::hpl};

    if(auto err = parseArgs(argc, argv, &m, &n, &affine, &init))
    {
        printUsage(argv[0]);
        return err;
    }

    std::size_t numElements     = m * n;
    std::size_t elementNumBytes = sizeof(float);

    float* gpuOutput{nullptr};
    float* gpuMean{nullptr};
    float* gpuInvvar{nullptr};
    float* gpuInput{nullptr};
    float* gpuGamma{nullptr};
    float* gpuBeta{nullptr};

    auto hipErr = hipMalloc(&gpuOutput, numElements * elementNumBytes);
    hipErr      = hipMalloc(&gpuMean, m * elementNumBytes);
    hipErr      = hipMalloc(&gpuInvvar, m * elementNumBytes);
    hipErr      = hipMalloc(&gpuInput, numElements * elementNumBytes);
    if(affine)
    {
        hipErr = hipMalloc(&gpuGamma, n * elementNumBytes);
        hipErr = hipMalloc(&gpuBeta, n * elementNumBytes);
    }

    std::vector<float> cpuOutput(numElements, 0.f);
    std::vector<float> cpuMean(m, 0.f);
    std::vector<float> cpuInvvar(m, 0.f);
    std::vector<float> cpuInput(numElements, 0.f);
    std::vector<float> cpuGamma(n, 1.f);
    std::vector<float> cpuBeta(n, 0.f);

    std::vector<float> refOutput(numElements, 0.f);
    std::vector<float> refMean(m, 0.f);
    std::vector<float> refInvvar(m, 0.f);

    initData(cpuInput.data(), cpuInput.size(), init);

    if(affine)
    {
        initData(cpuGamma.data(), cpuGamma.size(), init);
        initData(cpuBeta.data(), cpuBeta.size(), init);
    }

    hipErr = hipMemcpyHtoD(gpuInput, cpuInput.data(), numElements * elementNumBytes);
    if(affine)
    {
        hipErr = hipMemcpyHtoD(gpuGamma, cpuGamma.data(), n * elementNumBytes);
        hipErr = hipMemcpyHtoD(gpuBeta, cpuBeta.data(), n * elementNumBytes);
    }

    hipStream_t stream{};
    hipErr = hipStreamCreate(&stream);
    //warmup
    auto hipblasltErr = hipblasltExtLayerNorm(
        HIP_R_32F, gpuOutput, gpuMean, gpuInvvar, gpuInput, m, n, 1e-05, gpuGamma, gpuBeta, stream);

    hipErr = hipMemcpyDtoH(cpuOutput.data(), gpuOutput, numElements * elementNumBytes);
    hipErr = hipMemcpyDtoH(cpuMean.data(), gpuMean, m * elementNumBytes);
    hipErr = hipMemcpyDtoH(cpuInvvar.data(), gpuInvvar, m * elementNumBytes);

    cpuLayerNorm(refOutput.data(),
                 refMean.data(),
                 refInvvar.data(),
                 cpuInput.data(),
                 m,
                 n,
                 1e-05,
                 cpuGamma.data(),
                 cpuBeta.data());

    //    dumpBuffer("GPU", cpuOutput.data(), m, n);
    //    dumpBuffer("CPU", refOutput.data(), m, n);

    compare("Output", cpuOutput, refOutput);
    compare("Mean", cpuMean, refMean);
    compare("Invvar", cpuInvvar, refInvvar);

    hipEvent_t beg, end;
    hipErr      = hipEventCreate(&beg);
    hipErr      = hipEventCreate(&end);
    int numRuns = 200;
    hipErr      = hipEventRecord(beg, stream);

    for(int i = 0; i < numRuns; ++i)
    {
        hipblasltErr = hipblasltExtLayerNorm(HIP_R_32F,
                                             gpuOutput,
                                             gpuMean,
                                             gpuInvvar,
                                             gpuInput,
                                             m,
                                             n,
                                             1e-05,
                                             gpuGamma,
                                             gpuBeta,
                                             stream);
    }
    hipErr = hipEventRecord(end, stream);
    hipErr = hipEventSynchronize(end);
    hipErr = hipStreamSynchronize(stream);
    float dur{};
    hipErr = hipEventElapsedTime(&dur, beg, end);
    std::cout << "Time elapsed: " << std::to_string(dur / numRuns) << " ms\n";

    hipErr = hipEventDestroy(beg);
    hipErr = hipEventDestroy(end);
    hipErr = hipStreamDestroy(stream);
    hipErr = hipFree(gpuOutput);
    hipErr = hipFree(gpuMean);
    hipErr = hipFree(gpuInvvar);
    hipErr = hipFree(gpuInput);
    return 0;
}
