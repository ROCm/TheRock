// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <random>
#pragma once
struct ProblemSize final
{
    ck::index_t M = 128;
    ck::index_t N = 128;
    ck::index_t K = 384;

    ck::index_t stride_A = K;
    ck::index_t stride_B = K;
    ck::index_t stride_C = N;

    ck::index_t batch_stride_A = M * K;
    ck::index_t batch_stride_B = K * N;
    ck::index_t batch_stride_C = M * N;

    // Batched Gemm count
    ck::index_t batch_count = 2;

    // Split K count
    ck::index_t KBatch = 1;
};

struct ExecutionConfig final
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
};

template <typename DataType>
inline __host__ __device__ constexpr double get_rtol()
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, double>)
    {
        return 1e-6;
    }
    else if constexpr(std::is_same_v<DataType, ck::half_t>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, ck::bhalf_t>)
    {
        return 5e-2;
    }
    else if constexpr(std::is_same_v<DataType, int32_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, int8_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, ck::f8_t>)
    {
        return 1e-1; // 240 and 224 are acceptable
    }
    else if constexpr(std::is_same_v<DataType, ck::bf8_t>)
    {
        return 1.5e-1; // 57344 and 49152 are acceptable
    }
    else
    {
        return 1e-3;
    }
}

template <typename DataType>
inline __host__ __device__ constexpr double get_atol()
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, double>)
    {
        return 1e-6;
    }
    else if constexpr(std::is_same_v<DataType, ck::half_t>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, ck::bhalf_t>)
    {
        return 5e-2;
    }
    else if constexpr(std::is_same_v<DataType, int32_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, int8_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, ck::f8_t>)
    {
        return 16.1; // 240 and 224 are acceptable
    }
    else if constexpr(std::is_same_v<DataType, ck::bf8_t>)
    {
        return 8192.1; // 57344 and 49152 are acceptable
    }
    else
    {
        return 1e-3;
    }
}

bool run_batched_gemm(const ProblemSize& problem_size, const ExecutionConfig& config)
{
    using namespace ck::literals;
    using Bypass = ck::tensor_layout::BypassLayoutVerification;

    auto& [M,
           N,
           K,
           stride_A,
           stride_B,
           stride_C,
           batch_stride_A,
           batch_stride_B,
           batch_stride_C,
           batch_count,
           KBatch] = problem_size;

    auto f_host_tensor_descriptor = [](std::size_t batch_count_,
                                       std::size_t row,
                                       std::size_t col,
                                       std::size_t stride,
                                       std::size_t batch_stride,
                                       auto layout) {
        if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
        {
            return HostTensorDescriptor(
                {batch_count_, row, col}, {batch_stride, stride, 1_uz}, Bypass{});
        }
        else
        {
            return HostTensorDescriptor(
                {batch_count_, row, col}, {batch_stride, 1_uz, stride}, Bypass{});
        }
    };

    ck::index_t Scale_Stride_BN = (K + Scale_Block_K - 1) / Scale_Block_K;
    ck::index_t batch_BScale_Stride =
        ((K + Scale_Block_K - 1) / Scale_Block_K) * ((N + Scale_Block_N - 1) / Scale_Block_N);

    Tensor<ADataType> a_g_m_k(
        f_host_tensor_descriptor(batch_count, M, K, stride_A, batch_stride_A, ALayout{}));
    Tensor<BDataType> b_g_k_n(
        f_host_tensor_descriptor(batch_count, K, N, stride_B, batch_stride_B, BLayout{}));
    Tensor<BDataType> b_g_k_n_permute(
        f_host_tensor_descriptor(batch_count, K, N, stride_B, batch_stride_B, BLayout{}));
    Tensor<BScaleDataType> b1_g_k_n(
        f_host_tensor_descriptor(batch_count,
                                 (K + Scale_Block_K - 1) / Scale_Block_K,
                                 (N + Scale_Block_N - 1) / Scale_Block_N,
                                 Scale_Stride_BN,
                                 batch_BScale_Stride,
                                 BLayout{}));

    switch(config.init_method)
    {
    case 0:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        b_g_k_n.GenerateTensorValue(GeneratorTensor_1<BDataType>{1});
        b1_g_k_n.GenerateTensorValue(GeneratorTensor_1<BScaleDataType>{1});
        break;
    case 1:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_g_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        b1_g_k_n.GenerateTensorValue(GeneratorTensor_3<BScaleDataType>{0, 1.0});
        break;
    case 2:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        b_g_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        b1_g_k_n.GenerateTensorValue(GeneratorTensor_1<BScaleDataType>{1});
        break;
    case 3:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_g_k_n.GenerateTensorValue(GeneratorTensor_1<BDataType>{1});
        b1_g_k_n.GenerateTensorValue(GeneratorTensor_1<BScaleDataType>{1});
        break;
    case 4:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        b_g_k_n.GenerateTensorValue(GeneratorTensor_1<BDataType>{1});
        b1_g_k_n.GenerateTensorValue(GeneratorTensor_3<BScaleDataType>{0, 1.0});
        break;
    case 5:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_g_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        b1_g_k_n.GenerateTensorValue(GeneratorTensor_1<BScaleDataType>{1});
        break;
    default:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.5, 0.5});
        b_g_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        b1_g_k_n.GenerateTensorValue(GeneratorTensor_3<BScaleDataType>{0, 1.0});
    }

    Tensor<CDataType> c_g_m_n_host_result(
        f_host_tensor_descriptor(batch_count, M, N, stride_C, batch_stride_C, CLayout{}));
    Tensor<CDataType> c_g_m_n_device_result(
        f_host_tensor_descriptor(batch_count, M, N, stride_C, batch_stride_C, CLayout{}));

    std::cout << "a_g_m_k: " << a_g_m_k.mDesc << std::endl;
    std::cout << "b_g_k_n: " << b_g_k_n.mDesc << std::endl;
    std::cout << "b1_g_k_n: " << b1_g_k_n.mDesc << std::endl;
    std::cout << "c_g_m_n: " << c_g_m_n_host_result.mDesc << std::endl;

    DeviceMem a_g_m_k_device_buf(sizeof(ADataType) * a_g_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_g_k_n_device_buf(sizeof(BDataType) * b_g_k_n_permute.mDesc.GetElementSpaceSize() /
                                 2);
    DeviceMem b1_g_scale_device_buf(sizeof(BScaleDataType) * b1_g_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_g_m_n_device_buf(sizeof(CDataType) *
                                 c_g_m_n_device_result.mDesc.GetElementSpaceSize());

    printf("a_g_m_k size: %zu, b_g_k_n size: %zu, b1_g_k_n size: %zu, c_g_m_n size: %zu\n",
           a_g_m_k.mDesc.GetElementSpaceSize(),
           b_g_k_n_permute.mDesc.GetElementSpaceSize(),
           b1_g_k_n.mDesc.GetElementSpaceSize(),
           c_g_m_n_device_result.mDesc.GetElementSpaceSize());

    // weight permute
    if constexpr(PermuteB)
    {
        printf("Permute B\n");
        int K1 = KPerBlock;
        int K0 = K / KPerBlock;

        // int K0, N, K1
        for(int bs = 0; bs < batch_count; bs++)
        {
            for(int j = 0; j < K0; j++)
            {
                for(int i = 0; i < N; i++)
                {
                    for(int jj = 0; jj < K1; jj++)
                    {
                        b_g_k_n_permute(bs * batch_stride_B + j * N * K1 + i * K1 + jj) =
                            b_g_k_n(bs * batch_stride_B + i * K + (j * K1 + jj));
                    }
                }
            }
        }
    }
    else
    {
        b_g_k_n_permute = b_g_k_n;
    }

    // vector pk_i4x4 permute
    for(int bs = 0; bs < batch_count; bs++)
    {
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < K; j += 8)
            {
                int input[8];

                for(int k = 0; k < 4; k++)
                {
                    int i4x2         = b_g_k_n_permute(bs, j + k * 2, i).data;
                    input[k * 2 + 0] = (i4x2 >> 4) & 0xf;
                    input[k * 2 + 1] = (i4x2 >> 0) & 0xf;
                }

                // permute 01234567->20643175
                {
                    int hi   = input[2];
                    int lo   = input[0];
                    int i4x2 = (hi << 4) | lo;

                    b_g_k_n_permute(bs, j + 0, i) = i4x2;
                }

                {
                    int hi   = input[6];
                    int lo   = input[4];
                    int i4x2 = (hi << 4) | lo;

                    b_g_k_n_permute(bs, j + 2, i) = i4x2;
                }

                {
                    int hi   = input[3];
                    int lo   = input[1];
                    int i4x2 = (hi << 4) | lo;

                    b_g_k_n_permute(bs, j + 4, i) = i4x2;
                }

                {
                    int hi   = input[7];
                    int lo   = input[5];
                    int i4x2 = (hi << 4) | lo;

                    b_g_k_n_permute(bs, j + 6, i) = i4x2;
                }
            }
        }
    }

    a_g_m_k_device_buf.ToDevice(a_g_m_k.mData.data());
    b_g_k_n_device_buf.ToDevice(b_g_k_n_permute.mData.data());
    b1_g_scale_device_buf.ToDevice(b1_g_k_n.mData.data());
    DeviceMem workspace;

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    // do GEMM
    auto gemm      = DeviceBatchedGemmV2Instance{};
    auto invoker   = gemm.MakeInvoker();
    float ave_time = 0;

    auto argument =
        gemm.MakeArgument(static_cast<ADataType*>(a_g_m_k_device_buf.GetDeviceBuffer()),
                          static_cast<BDataType*>(b_g_k_n_device_buf.GetDeviceBuffer()),
                          static_cast<CDataType*>(c_g_m_n_device_buf.GetDeviceBuffer()),
                          M,
                          N,
                          K,
                          stride_A,
                          stride_B,
                          stride_C,
                          Scale_Stride_BN,
                          batch_stride_A,
                          batch_stride_B,
                          batch_stride_C,
                          batch_BScale_Stride,
                          static_cast<BScaleDataType*>(b1_g_scale_device_buf.GetDeviceBuffer()),
                          batch_count, // batch count
                          KBatch,      // split K count
                          a_element_op,
                          b_element_op,
                          c_element_op);

    if(!gemm.IsSupportedArgument(argument))
    {
        std::cerr << gemm.GetTypeString() << " does not support this problem" << std::endl;

        return false;
    }

    bool pass = true;
    Tensor<float> b_g_k_n_dequant({batch_count, K, N});
    if(config.do_verification)
    {
        float v_b = 0;
        for(int bs = 0; bs < batch_count; bs++)
        {
            for(int n = 0; n < N; n++)
            {
                for(int k = 0; k < K; k++)
                {
                    ck::pk_i4_t i4x2 = b_g_k_n(bs, k, n).data;
                    int8_t i4        = 0;
                    if(k % 2 == 1)
                        i4 = (i4x2.data >> 0) & 0xf;
                    else
                        i4 = (i4x2.data >> 4) & 0xf;
                    i4  = i4 - 8;
                    v_b = ck::type_convert<float>(i4);

                    b_g_k_n_dequant(bs, k, n) =
                        ck::type_convert<float>(v_b) *
                        ck::type_convert<float>(b1_g_k_n(bs, k / Scale_Block_K, n / Scale_Block_N));
                }
            }
        }

        auto ref_gemm    = ReferenceBatchedGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(a_g_m_k,
                                                  b_g_k_n_dequant,
                                                  c_g_m_n_host_result,
                                                  PassThrough{},
                                                  PassThrough{},
                                                  PassThrough{});

        ref_invoker.Run(ref_argument);

        ave_time = invoker.Run(argument, StreamConfig{nullptr, false, 0});
        hip_check_error(hipDeviceSynchronize());

        c_g_m_n_device_buf.FromDevice(c_g_m_n_device_result.mData.data());

        pass &= ck::utils::check_err(c_g_m_n_device_result,
                                     c_g_m_n_host_result,
                                     "Error: Incorrect results!",
                                     get_rtol<CDataType>(),
                                     get_atol<CDataType>());
    }

    if(config.time_kernel)
    {
        ave_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});

        std::size_t flop = 2_uz * M * N * K;
        std::size_t num_btype =
            sizeof(ADataType) * M * K +
            sizeof(BDataType) * K * N /
                (ck::is_same_v<ck::remove_cvref_t<BDataType>, ck::pk_i4_t> ? 2 : 1) +
            sizeof(CDataType) * M * N;

        float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

        float gb_per_sec = num_btype / 1.E6 / ave_time;

        std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s, " << gemm.GetTypeString() << std::endl;
    }

#if 0
    // print A matrix
    printf("A matrix:\n");
    for(int bs = 0; bs < batch_count; bs++)
    {
        printf("batch %d -> Address: %p\n", bs, static_cast<void*>(&a_g_m_k(bs, 0, 0)));
        for(int i = 0; i < M; i++)
        {
            for(int j = 0; j < K; j++)
            {
                printf("%.2f,", static_cast<float>(a_g_m_k(bs, i, j)));
            }
            printf("\n");
        }
    }

    // print B matrix original
    printf("B matrix original:\n");
    for(int bs = 0; bs < batch_count; bs++)
    {
        printf("batch %d -> Address: %p\n", bs, static_cast<void*>(&b_g_k_n(bs, 0, 0)));
        for(int n = 0; n < N; n++)
        {
            for(int k = 0; k < K; k++)
            {
                ck::pk_i4_t i4x2 = b_g_k_n(bs, k, n).data;
                int8_t i4        = 0;
                if(k % 2 == 1)
                    i4 = (i4x2.data >> 0) & 0xf;
                else
                    i4 = (i4x2.data >> 4) & 0xf;
                i4 = i4 - 8;
                printf("%d,", static_cast<int>(i4));
            }
            printf("\n");
        }
    }

    // print B matrix
    printf("B matrix:\n");
    for(int bs = 0; bs < batch_count; bs++)
    {
        printf("batch %d -> Address: %p\n", bs, static_cast<void*>(&b_g_k_n_dequant(bs, 0, 0)));
        for(int i = 0; i < K; i++)
        {
            for(int j = 0; j < N; j++)
            {
                printf("%.2f, ", static_cast<float>(b_g_k_n_dequant(bs, i, j)));
            }
            printf("\n");
        }
    }

    // print B scale matrix
    printf("B Scale matrix:\n");
    for(int bs = 0; bs < batch_count; bs++)
    {
        printf("batch %d -> Address: %p\n", bs, static_cast<void*>(&b1_g_k_n(bs, 0, 0)));
        for(int i = 0; i < (K + Scale_Block_K - 1) / Scale_Block_K; i++)
        {
            for(int j = 0; j < (N + Scale_Block_N - 1) / Scale_Block_N; j++)
            {
                printf("%.2f, ", static_cast<float>(b1_g_k_n(bs, i, j)));
            }
            printf("\n");
        }
    }

    // print C matrix
    printf("C matrix:\n");
    for(int bs = 0; bs < batch_count; bs++)
    {
        printf(
            "batch %d -> Address: %p\n", bs, static_cast<void*>(&c_g_m_n_device_result(bs, 0, 0)));
        for(int i = 0; i < M; i++)
        {
            for(int j = 0; j < N; j++)
            {
                printf("%.2f, ", static_cast<float>(c_g_m_n_device_result(bs, i, j)));
            }
            printf("\n");
        }
    }

    printf("C reference matrix:\n");
    for(int bs = 0; bs < batch_count; bs++)
    {
        printf("batch %d -> Address: %p\n", bs, static_cast<void*>(&c_g_m_n_host_result(bs, 0, 0)));
        for(int i = 0; i < M; i++)
        {
            for(int j = 0; j < N; j++)
            {
                printf("%.2f, ", static_cast<float>(c_g_m_n_host_result(bs, i, j)));
            }
            printf("\n");
        }
    }
#endif

    return pass;
}

bool run_batched_gemm_fp16_int4_b_scale_example(int argc, char* argv[])
{
    if(ck::is_gfx11_supported() || ck::is_gfx12_supported())
    {
        return 1;
    }

    ProblemSize problem_size;
    ExecutionConfig config;

    std::mt19937 gen(11939);
    std::uniform_int_distribution<int> dis(0, 15);

    problem_size.M = 128 * (dis(gen) + 1);
    problem_size.N = 128 * (dis(gen) + 1);
    problem_size.K = 256 * (dis(gen) + 2);

    problem_size.batch_count = 2;

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4 || argc >= 7)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
        if(argc >= 7)
        {
            problem_size.M = std::stoi(argv[4]);
            problem_size.N = std::stoi(argv[5]);
            problem_size.K = std::stoi(argv[6]);

            if(argc >= 8)
            {
                problem_size.batch_count = std::stoi(argv[7]);
            }

            if(argc >= 9)
            {
                problem_size.KBatch = std::stoi(argv[8]);
            }
        }
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=n0, 1=yes)\n");
        printf("arg4-6: problem size (M, N, K)\n");
        printf("arg7: batch count\n");
        printf("arg8: KBatch\n");
        exit(1);
    }

    problem_size.stride_A = problem_size.K;
    problem_size.stride_B = problem_size.K;
    problem_size.stride_C = problem_size.N;

    problem_size.batch_stride_A = problem_size.M * problem_size.K;
    problem_size.batch_stride_B = problem_size.K * problem_size.N;
    problem_size.batch_stride_C = problem_size.M * problem_size.N;

    return run_batched_gemm(problem_size, config);
}
