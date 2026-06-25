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
#pragma once
#include <functional>

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "Hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

#ifndef CHECK_HIPBLASLT_ERROR
#define CHECK_HIPBLASLT_ERROR(error)                                                      \
    if(error != HIPBLAS_STATUS_SUCCESS)                                                   \
    {                                                                                     \
        fprintf(stderr, "hipBLASLt error(Err=%d) at %s:%d\n", error, __FILE__, __LINE__); \
        fprintf(stderr, "\n");                                                            \
        exit(EXIT_FAILURE);                                                               \
    }
#endif

template <typename InTypeA,
          typename InTypeB,
          typename OutType,
          typename AlphaType,
          typename BetaType,
          typename BiasType = OutType>
struct Runner
{
    Runner(int64_t   m,
           int64_t   n,
           int64_t   k,
           int64_t   batch_count,
           AlphaType alpha,
           BetaType  beta,
           int64_t   max_workspace_size_in_bytes)
        : m(m)
        , n(n)
        , k(k)
        , batch_count(batch_count)
        , alpha(alpha)
        , beta(beta)
        , max_workspace_size(max_workspace_size_in_bytes)
    {
        CHECK_HIP_ERROR(hipStreamCreate(&stream));
        CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));
        CHECK_HIP_ERROR(hipMalloc(&d_a, m * k * batch_count * sizeof(InTypeA)));
        CHECK_HIP_ERROR(hipMalloc(&d_b, n * k * batch_count * sizeof(InTypeB)));
        CHECK_HIP_ERROR(hipMalloc(&d_c, m * n * batch_count * sizeof(OutType)));
        CHECK_HIP_ERROR(hipMalloc(&d_d, m * n * batch_count * sizeof(OutType)));
        CHECK_HIP_ERROR(hipMalloc(&d_alphaVec, m * batch_count * sizeof(float)));

        CHECK_HIP_ERROR(hipHostMalloc(&a, m * k * batch_count * sizeof(InTypeA)));
        CHECK_HIP_ERROR(hipHostMalloc(&b, n * k * batch_count * sizeof(InTypeB)));
        CHECK_HIP_ERROR(hipHostMalloc(&c, m * n * batch_count * sizeof(OutType)));
        CHECK_HIP_ERROR(hipHostMalloc(&d, m * n * batch_count * sizeof(OutType)));
        CHECK_HIP_ERROR(hipHostMalloc(&alphaVec, m * batch_count * sizeof(float)));

        if(max_workspace_size > 0)
            CHECK_HIP_ERROR(hipMalloc(&d_workspace, max_workspace_size));

        for(int i = 0; i < m * k * batch_count; i++)
            ((InTypeA*)a)[i] = static_cast<InTypeA>((rand() % 7) - 3);
        for(int i = 0; i < n * k * batch_count; i++)
            ((InTypeB*)b)[i] = static_cast<InTypeB>((rand() % 7) - 3);
        for(int i = 0; i < m * n * batch_count; i++)
            ((OutType*)c)[i] = static_cast<OutType>((rand() % 7) - 3);
        for(int i = 0; i < m * batch_count; ++i)
            ((float*)alphaVec)[i] = static_cast<float>((rand() % 7) - 3);
    }

    ~Runner()
    {
        CHECK_HIP_ERROR(hipFree(d_workspace));
        CHECK_HIP_ERROR(hipFree(a));
        CHECK_HIP_ERROR(hipFree(b));
        CHECK_HIP_ERROR(hipFree(c));
        CHECK_HIP_ERROR(hipFree(d));
        CHECK_HIP_ERROR(hipFree(alphaVec));
        CHECK_HIP_ERROR(hipFree(d_a));
        CHECK_HIP_ERROR(hipFree(d_b));
        CHECK_HIP_ERROR(hipFree(d_c));
        CHECK_HIP_ERROR(hipFree(d_d));
        CHECK_HIP_ERROR(hipFree(d_alphaVec));
        CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
        CHECK_HIP_ERROR(hipStreamDestroy(stream));

        if(biasVec)
        {
            CHECK_HIP_ERROR(hipFree(biasVec));
            CHECK_HIP_ERROR(hipFree(d_biasVec));
        }
    }

    void setBiasInfo(bool useBias, char biasSrc)
    {
        biasElems = 0;
        if(useBias)
        {
            if(biasSrc == 'B' || biasSrc == 'b')
                biasElems = n;
            else if(biasSrc == 'A' || biasSrc == 'a' || biasSrc == 'D' || biasSrc == 'd')
                biasElems = m;
            // else, biasElems = 0
        }

        // alloc bias if use bias
        if(biasElems > 0)
        {
            if(biasVec)
            {
                CHECK_HIP_ERROR(hipFree(biasVec));
                CHECK_HIP_ERROR(hipFree(d_biasVec));
            }

            CHECK_HIP_ERROR(hipMalloc(&d_biasVec, biasElems * sizeof(BiasType)));
            CHECK_HIP_ERROR(hipHostMalloc(&biasVec, biasElems * sizeof(BiasType)));
            for(int i = 0; i < biasElems; ++i)
                ((BiasType*)biasVec)[i] = static_cast<BiasType>((rand() % 7) - 3);
        }
    }

    void hostToDevice()
    {
        CHECK_HIP_ERROR(hipMemcpyAsync(
            d_a, a, m * k * batch_count * sizeof(InTypeA), hipMemcpyHostToDevice, stream));
        CHECK_HIP_ERROR(hipMemcpyAsync(
            d_b, b, n * k * batch_count * sizeof(InTypeB), hipMemcpyHostToDevice, stream));
        CHECK_HIP_ERROR(hipMemcpyAsync(
            d_c, c, m * n * batch_count * sizeof(OutType), hipMemcpyHostToDevice, stream));
        CHECK_HIP_ERROR(hipMemcpyAsync(
            d_alphaVec, alphaVec, m * batch_count * sizeof(float), hipMemcpyHostToDevice, stream));

        // cpy bias if needed
        if(biasVec)
            CHECK_HIP_ERROR(hipMemcpyAsync(
                d_biasVec, biasVec, biasElems * sizeof(BiasType), hipMemcpyHostToDevice, stream));
    }

    void deviceToHost()
    {
        CHECK_HIP_ERROR(hipMemcpyAsync(
            d, d_d, m * n * batch_count * sizeof(OutType), hipMemcpyDeviceToHost, stream));
    }

    void run(const std::function<void()>& func)
    {
        hostToDevice();

        static_cast<void>(func());

        deviceToHost();
        static_cast<void>(hipStreamSynchronize(stream));
    }

    int64_t   m;
    int64_t   n;
    int64_t   k;
    int64_t   batch_count;
    AlphaType alpha;
    BetaType  beta;

    void *a, *b, *c, *d, *alphaVec; // host
    void *d_a, *d_b, *d_c, *d_d, *d_alphaVec; // device

    void*   d_workspace;
    int64_t max_workspace_size;

    int64_t biasElems = 0;
    void*   biasVec   = nullptr; // host
    void*   d_biasVec = nullptr; // device

    hipStream_t       stream;
    hipblasLtHandle_t handle;
};

template <typename InTypeA,
          typename InTypeB,
          typename OutType,
          typename AlphaType,
          typename BetaType>
struct RunnerVec
{
    RunnerVec(const std::vector<int64_t>   m,
              const std::vector<int64_t>   n,
              const std::vector<int64_t>   k,
              const std::vector<int64_t>   batch_count,
              const std::vector<AlphaType> alpha,
              const std::vector<BetaType>  beta,
              const int64_t                max_workspace_size_in_bytes)
        : m(m)
        , n(n)
        , k(k)
        , batch_count(batch_count)
        , alpha(alpha)
        , beta(beta)
        , max_workspace_size(max_workspace_size_in_bytes)
    {
        CHECK_HIP_ERROR(hipStreamCreate(&stream));
        CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));
        d_a.resize(m.size(), nullptr);
        d_b.resize(m.size(), nullptr);
        d_c.resize(m.size(), nullptr);
        d_d.resize(m.size(), nullptr);
        d_alphaVec.resize(m.size(), nullptr);
        a.resize(m.size(), nullptr);
        b.resize(m.size(), nullptr);
        c.resize(m.size(), nullptr);
        d.resize(m.size(), nullptr);
        alphaVec.resize(m.size(), nullptr);
        for(int j = 0; j < m.size(); j++)
        {
            CHECK_HIP_ERROR(hipMalloc(&d_a[j], m[j] * k[j] * batch_count[j] * sizeof(InTypeA)));
            CHECK_HIP_ERROR(hipMalloc(&d_b[j], n[j] * k[j] * batch_count[j] * sizeof(InTypeB)));
            CHECK_HIP_ERROR(hipMalloc(&d_c[j], m[j] * n[j] * batch_count[j] * sizeof(OutType)));
            CHECK_HIP_ERROR(hipMalloc(&d_d[j], m[j] * n[j] * batch_count[j] * sizeof(OutType)));
            CHECK_HIP_ERROR(hipMalloc(&d_alphaVec[j], m[j] * batch_count[j] * sizeof(float)));

            CHECK_HIP_ERROR(hipHostMalloc(&a[j], m[j] * k[j] * batch_count[j] * sizeof(InTypeA)));
            CHECK_HIP_ERROR(hipHostMalloc(&b[j], n[j] * k[j] * batch_count[j] * sizeof(InTypeB)));
            CHECK_HIP_ERROR(hipHostMalloc(&c[j], m[j] * n[j] * batch_count[j] * sizeof(OutType)));
            CHECK_HIP_ERROR(hipHostMalloc(&d[j], m[j] * n[j] * batch_count[j] * sizeof(OutType)));
            CHECK_HIP_ERROR(hipHostMalloc(&alphaVec[j], m[j] * batch_count[j] * sizeof(float)));

            for(int i = 0; i < m[j] * k[j] * batch_count[j]; i++)
                ((InTypeA*)a[j])[i] = static_cast<InTypeA>((rand() % 7) - 3);
            for(int i = 0; i < n[j] * k[j] * batch_count[j]; i++)
                ((InTypeB*)b[j])[i] = static_cast<InTypeB>((rand() % 7) - 3);
            for(int i = 0; i < m[j] * n[j] * batch_count[j]; i++)
                ((OutType*)c[j])[i] = static_cast<OutType>((rand() % 7) - 3);
            for(int i = 0; i < m[j] * batch_count[j]; i++)
                ((float*)alphaVec[j])[i] = static_cast<float>((rand() % 7) - 3);
        }
        if(max_workspace_size > 0)
            CHECK_HIP_ERROR(hipMalloc(&d_workspace, max_workspace_size));
    }

    ~RunnerVec()
    {
        for(int j = 0; j < m.size(); j++)
        {
            CHECK_HIP_ERROR(hipFree(a[j]));
            CHECK_HIP_ERROR(hipFree(b[j]));
            CHECK_HIP_ERROR(hipFree(c[j]));
            CHECK_HIP_ERROR(hipFree(d[j]));
            CHECK_HIP_ERROR(hipFree(alphaVec[j]));
            CHECK_HIP_ERROR(hipFree(d_a[j]));
            CHECK_HIP_ERROR(hipFree(d_b[j]));
            CHECK_HIP_ERROR(hipFree(d_c[j]));
            CHECK_HIP_ERROR(hipFree(d_d[j]));
            CHECK_HIP_ERROR(hipFree(d_alphaVec[j]));
        }
        CHECK_HIP_ERROR(hipFree(d_workspace));
        CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
        CHECK_HIP_ERROR(hipStreamDestroy(stream));
    }

    void hostToDevice()
    {
        for(int j = 0; j < m.size(); j++)
        {
            CHECK_HIP_ERROR(hipMemcpyAsync(d_a[j],
                                           a[j],
                                           m[j] * k[j] * batch_count[j] * sizeof(InTypeA),
                                           hipMemcpyHostToDevice,
                                           stream));
            CHECK_HIP_ERROR(hipMemcpyAsync(d_b[j],
                                           b[j],
                                           n[j] * k[j] * batch_count[j] * sizeof(InTypeB),
                                           hipMemcpyHostToDevice,
                                           stream));
            CHECK_HIP_ERROR(hipMemcpyAsync(d_c[j],
                                           c[j],
                                           m[j] * n[j] * batch_count[j] * sizeof(OutType),
                                           hipMemcpyHostToDevice,
                                           stream));
            CHECK_HIP_ERROR(hipMemcpyAsync(d_alphaVec[j],
                                           alphaVec[j],
                                           m[j] * batch_count[j] * sizeof(float),
                                           hipMemcpyHostToDevice,
                                           stream));
        }
    }

    void deviceToHost()
    {
        for(int j = 0; j < m.size(); j++)
        {
            CHECK_HIP_ERROR(hipMemcpyAsync(d[j],
                                           d_d[j],
                                           m[j] * n[j] * batch_count[j] * sizeof(OutType),
                                           hipMemcpyDeviceToHost,
                                           stream));
        }
    }

    void run(const std::function<void()>& func)
    {
        hostToDevice();

        static_cast<void>(func());

        deviceToHost();
        static_cast<void>(hipStreamSynchronize(stream));
    }

    std::vector<int64_t>   m;
    std::vector<int64_t>   n;
    std::vector<int64_t>   k;
    std::vector<int64_t>   batch_count;
    std::vector<AlphaType> alpha;
    std::vector<BetaType>  beta;

    std::vector<void*> a, b, c, d, alphaVec; // host
    std::vector<void*> d_a, d_b, d_c, d_d, d_alphaVec; // device

    void*   d_workspace;
    int64_t max_workspace_size;

    hipStream_t       stream;
    hipblasLtHandle_t handle;
};

template <typename Type>
struct LayerNormRunner
{
    LayerNormRunner(int64_t m, int64_t n)
        : m(m)
        , n(n)
    {
        CHECK_HIP_ERROR(hipStreamCreate(&stream));
        CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));

        CHECK_HIP_ERROR(hipMalloc(&d_out, m * n * sizeof(Type)));
        CHECK_HIP_ERROR(hipMalloc(&d_mean, m * sizeof(Type)));
        CHECK_HIP_ERROR(hipMalloc(&d_invvar, m * sizeof(Type)));
        CHECK_HIP_ERROR(hipMalloc(&d_in, m * n * sizeof(Type)));
        CHECK_HIP_ERROR(hipMalloc(&d_gamma, n * sizeof(Type)));
        CHECK_HIP_ERROR(hipMalloc(&d_beta, n * sizeof(Type)));

        CHECK_HIP_ERROR(hipHostMalloc(&out, m * n * sizeof(Type)));
        CHECK_HIP_ERROR(hipHostMalloc(&mean, m * sizeof(Type)));
        CHECK_HIP_ERROR(hipHostMalloc(&invvar, m * sizeof(Type)));
        CHECK_HIP_ERROR(hipHostMalloc(&in, m * n * sizeof(Type)));
        CHECK_HIP_ERROR(hipHostMalloc(&gamma, n * sizeof(Type)));
        CHECK_HIP_ERROR(hipHostMalloc(&beta, n * sizeof(Type)));

        for(int i = 0; i < m * n; i++)
            ((Type*)in)[i] = static_cast<Type>((rand() % 7) - 3);
        for(int i = 0; i < n; i++)
            ((Type*)gamma)[i] = static_cast<Type>((rand() % 7) - 3);
        for(int i = 0; i < n; i++)
            ((Type*)beta)[i] = static_cast<Type>((rand() % 7) - 3);
    }

    ~LayerNormRunner()
    {
        CHECK_HIP_ERROR(hipFree(d_out));
        CHECK_HIP_ERROR(hipFree(d_mean));
        CHECK_HIP_ERROR(hipFree(d_invvar));
        CHECK_HIP_ERROR(hipFree(d_in));
        CHECK_HIP_ERROR(hipFree(d_gamma));
        CHECK_HIP_ERROR(hipFree(d_beta));

        CHECK_HIP_ERROR(hipFree(out));
        CHECK_HIP_ERROR(hipFree(mean));
        CHECK_HIP_ERROR(hipFree(invvar));
        CHECK_HIP_ERROR(hipFree(in));
        CHECK_HIP_ERROR(hipFree(gamma));
        CHECK_HIP_ERROR(hipFree(beta));

        CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
        CHECK_HIP_ERROR(hipStreamDestroy(stream));
    }

    void hostToDevice()
    {
        CHECK_HIP_ERROR(
            hipMemcpyAsync(d_in, in, m * n * sizeof(Type), hipMemcpyHostToDevice, stream));
        CHECK_HIP_ERROR(
            hipMemcpyAsync(d_gamma, gamma, n * sizeof(Type), hipMemcpyHostToDevice, stream));
        CHECK_HIP_ERROR(
            hipMemcpyAsync(d_beta, beta, n * sizeof(Type), hipMemcpyHostToDevice, stream));
    }

    void deviceToHost()
    {
        CHECK_HIP_ERROR(
            hipMemcpyAsync(out, d_out, m * n * sizeof(Type), hipMemcpyDeviceToHost, stream));
        CHECK_HIP_ERROR(
            hipMemcpyAsync(mean, d_mean, m * sizeof(Type), hipMemcpyDeviceToHost, stream));
        CHECK_HIP_ERROR(
            hipMemcpyAsync(invvar, d_invvar, m * sizeof(Type), hipMemcpyDeviceToHost, stream));
    }

    void run(const std::function<void()>& func)
    {
        hostToDevice();

        static_cast<void>(func());

        deviceToHost();
        static_cast<void>(hipStreamSynchronize(stream));
    }

    int64_t m;
    int64_t n;

    void *out, *mean, *invvar, *in, *gamma, *beta; // host
    void *d_out, *d_mean, *d_invvar, *d_in, *d_gamma, *d_beta; // host

    hipStream_t       stream;
    hipblasLtHandle_t handle;
};

template <typename Type>
struct OptAMaxRunner
{
    OptAMaxRunner(int64_t m, int64_t n)
        : m(m)
        , n(n)
    {
        CHECK_HIP_ERROR(hipStreamCreate(&stream));
        CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));

        CHECK_HIP_ERROR(hipMalloc(&d_out, sizeof(Type)));
        CHECK_HIP_ERROR(hipMalloc(&d_in, m * n * sizeof(Type)));

        CHECK_HIP_ERROR(hipHostMalloc(&out, sizeof(Type)));
        CHECK_HIP_ERROR(hipHostMalloc(&in, m * n * sizeof(Type)));

        for(int i = 0; i < m * n; i++)
            ((Type*)in)[i] = static_cast<Type>((rand() % 7) - 3);
    }

    ~OptAMaxRunner()
    {
        CHECK_HIP_ERROR(hipFree(d_out));
        CHECK_HIP_ERROR(hipFree(d_in));

        CHECK_HIP_ERROR(hipFree(out));
        CHECK_HIP_ERROR(hipFree(in));

        CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
        CHECK_HIP_ERROR(hipStreamDestroy(stream));
    }

    void hostToDevice()
    {
        CHECK_HIP_ERROR(
            hipMemcpyAsync(d_in, in, m * n * sizeof(Type), hipMemcpyHostToDevice, stream));
    }

    void deviceToHost()
    {
        CHECK_HIP_ERROR(hipMemcpyAsync(out, d_out, sizeof(Type), hipMemcpyDeviceToHost, stream));
    }

    void run(const std::function<void()>& func)
    {
        hostToDevice();

        static_cast<void>(func());

        deviceToHost();
        static_cast<void>(hipStreamSynchronize(stream));
    }

    int64_t m;
    int64_t n;

    void *in, *out; // host
    void *d_in, *d_out; // device

    hipStream_t       stream;
    hipblasLtHandle_t handle;
};
