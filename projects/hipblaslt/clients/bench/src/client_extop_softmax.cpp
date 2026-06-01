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
#include <vector>

void printUsage(char* programName)
{
    std::cout << "Usage: " << programName << " <options>\n"
              << "options:\n"
              << "\t-h, --help\t\t\tShow this help message\n"
              << "\t-m, --m\t\t\t\tSize of dim 0, default is 1335\n"
              << "\t-n, --n\t\t\t\tSize of dim 1, default is 16\n"
              << "\t--initialization \t\tInitialize matrix data. Options: rand_int, trig_float, "
                 "hpl(floating), special, zero. (default is hpl)\n";
}

int parseArgs(int argc, char** argv, size_t* m, size_t* n, hipblaslt_initialization* init)
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
    std::size_t              m{1335};
    std::size_t              n{16};
    hipblaslt_initialization init{hipblaslt_initialization::hpl};

    if(auto err = parseArgs(argc, argv, &m, &n, &init))
    {
        printUsage(argv[0]);
        return err;
    }

    std::size_t numElements     = m * n;
    std::size_t elementNumBytes = sizeof(float);
    float*      input{};
    float*      output{};
    auto        hipErr = hipMalloc(&input, numElements * elementNumBytes);
    hipErr             = hipMalloc(&output, numElements * elementNumBytes);
    std::vector<float> data(numElements, 0.f);
    // std::iota(begin(data), end(data), 0.f);
    initData(input, numElements, init);
    hipErr = hipMemcpyHtoD(input, data.data(), numElements * elementNumBytes);
    hipStream_t stream{};
    hipErr = hipStreamCreate(&stream);
    //warmup
    auto hipblasltErr = hipblasltExtSoftmax(HIP_R_32F, m, n, 1, output, input, stream);

    if(hipblasltErr)
    {
        std::cerr << "Invalid shape (" << m << ", " << n << "), currently support n <= 256\n";
        hipErr = hipFree(input);
        hipErr = hipFree(output);
        hipErr = hipStreamDestroy(stream);
        return EXIT_FAILURE;
    }

    hipEvent_t beg, end;
    hipErr      = hipEventCreate(&beg);
    hipErr      = hipEventCreate(&end);
    int numRuns = 50;
    hipErr      = hipEventRecord(beg, stream);

    for(int i = 0; i < numRuns; ++i)
    {
        hipblasltErr = hipblasltExtSoftmax(HIP_R_32F, m, n, 1, output, input, stream);
    }

    hipErr = hipEventRecord(end, stream);
    hipErr = hipEventSynchronize(end);
    hipErr = hipStreamSynchronize(stream);
    float dur{};
    hipErr = hipEventElapsedTime(&dur, beg, end);
    std::cout << "Time elapsed: " << std::to_string(dur / numRuns) << " ms\n";
    std::vector<float> gpuOutput(numElements, 0.f);
    hipErr = hipEventDestroy(beg);
    hipErr = hipEventDestroy(end);
    hipErr = hipMemcpyDtoH(gpuOutput.data(), output, numElements * elementNumBytes);
    hipErr = hipStreamDestroy(stream);
    hipErr = hipFree(input);
    hipErr = hipFree(output);
    return 0;
}
