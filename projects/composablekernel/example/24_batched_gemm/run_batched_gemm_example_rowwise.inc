// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <random>
#pragma once

struct ProblemSize final
{
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t stride_A = K;
    ck::index_t stride_B = K;
    ck::index_t stride_C = N;

    ck::index_t stride_D0 = 0;
    ck::index_t stride_D1 = 0;

    ck::index_t batch_stride_A = M * K;
    ck::index_t batch_stride_B = K * N;
    ck::index_t batch_stride_C = M * N;

    ck::index_t batch_stride_D0 = N;
    ck::index_t batch_stride_D1 = M;

    ck::index_t batch_count = 16;
};

struct ExecutionConfig final
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
};

bool run_batched_gemm_rowwise(const ProblemSize& problem_size, const ExecutionConfig& config)
{
    using namespace ck::literals;
    using Bypass = ck::tensor_layout::BypassLayoutVerification;

    auto& [M,
           N,
           K,
           stride_A,
           stride_B,
           stride_C,
           stride_D0,
           stride_D1,
           batch_stride_A,
           batch_stride_B,
           batch_stride_C,
           batch_stride_D0,
           batch_stride_D1,
           batch_count] = problem_size;

    // GEMM shape
    auto f_host_tensor_descriptor = [](std::size_t batch_count_,
                                       std::size_t row,
                                       std::size_t col,
                                       std::size_t stride,
                                       std::size_t batch_stride,
                                       auto layout) {
        using namespace ck::literals;

        if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
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

    Tensor<ADataType> a_g_m_k(
        f_host_tensor_descriptor(batch_count, M, K, stride_A, batch_stride_A, ALayout{}));
    Tensor<BDataType> b_g_k_n(
        f_host_tensor_descriptor(batch_count, K, N, stride_B, batch_stride_B, BLayout{}));
    Tensor<D0DataType> d0_g_m_n(
        f_host_tensor_descriptor(batch_count, M, N, stride_D0, batch_stride_D0, D0Layout{}));
    Tensor<D1DataType> d1_g_m_n(
        f_host_tensor_descriptor(batch_count, M, N, stride_D1, batch_stride_D1, D1Layout{}));
    Tensor<EDataType> e_g_m_n_device_result(
        f_host_tensor_descriptor(batch_count, M, N, stride_C, batch_stride_C, ELayout{}));

    std::cout << "a_g_m_k: " << a_g_m_k.mDesc << std::endl;
    std::cout << "b_g_k_n: " << b_g_k_n.mDesc << std::endl;
    std::cout << "d0_g_m_n: " << d0_g_m_n.mDesc << std::endl;
    std::cout << "d1_g_m_n: " << d1_g_m_n.mDesc << std::endl;
    std::cout << "e_g_m_n: " << e_g_m_n_device_result.mDesc << std::endl;

    switch(config.init_method)
    {
    case 0: break;
    case 1:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b_g_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        break;
    default:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_g_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        break;
    }

    d0_g_m_n.GenerateTensorValue(GeneratorTensor_3<D0DataType>{0.0, 1.0});
    d1_g_m_n.GenerateTensorValue(GeneratorTensor_3<D1DataType>{0.0, 1.0});

    DeviceMem a_device_buf(sizeof(ADataType) * a_g_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_g_k_n.mDesc.GetElementSpaceSize());
    DeviceMem d0_device_buf(sizeof(D0DataType) * d0_g_m_n.mDesc.GetElementSpaceSize());
    DeviceMem d1_device_buf(sizeof(D1DataType) * d1_g_m_n.mDesc.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(EDataType) * e_g_m_n_device_result.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_g_m_k.mData.data());
    b_device_buf.ToDevice(b_g_k_n.mData.data());

    d0_device_buf.ToDevice(d0_g_m_n.mData.data());
    d1_device_buf.ToDevice(d1_g_m_n.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    auto gemm    = DeviceGemmInstance{};
    auto invoker = gemm.MakeInvoker();

    // do GEMM
    auto argument =
        gemm.MakeArgument(a_device_buf.GetDeviceBuffer(),
                          b_device_buf.GetDeviceBuffer(),
                          {d0_device_buf.GetDeviceBuffer(), d1_device_buf.GetDeviceBuffer()},
                          c_device_buf.GetDeviceBuffer(),
                          M,
                          N,
                          K,
                          batch_count,
                          stride_A,
                          stride_B,
                          {stride_D0, stride_D1},
                          stride_C,
                          batch_stride_A,
                          batch_stride_B,
                          {batch_stride_D0, batch_stride_D1},
                          batch_stride_C,
                          a_element_op,
                          b_element_op,
                          cde_element_op);

    if(!gemm.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    invoker.Run(argument, StreamConfig{nullptr, false});
    bool pass = true;

    if(config.do_verification)
    {
        c_device_buf.FromDevice(e_g_m_n_device_result.mData.data());

        Tensor<CShuffleDataType> c_g_m_n({batch_count, M, N});

        using ReferenceBatchedGemmInstance =
            ck::tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                             BDataType,
                                                             CShuffleDataType,
                                                             AccDataType,
                                                             AElementOp,
                                                             BElementOp,
                                                             PassThrough>;

        auto ref_batched_gemm = ReferenceBatchedGemmInstance{};
        auto ref_invoker      = ref_batched_gemm.MakeInvoker();

        Tensor<EDataType> e_g_m_n_host_result(
            f_host_tensor_descriptor(batch_count, M, N, stride_C, batch_stride_C, ELayout{}));

        auto ref_argument = ref_batched_gemm.MakeArgument(
            a_g_m_k, b_g_k_n, c_g_m_n, a_element_op, b_element_op, PassThrough{});

        ref_invoker.Run(ref_argument);

        for(int b = 0; b < batch_count; ++b)
        {
            for(int m = 0; m < M; ++m)
            {
                for(int n = 0; n < N; ++n)
                {
                    cde_element_op(e_g_m_n_host_result(b, m, n),
                                   c_g_m_n(b, m, n),
                                   d0_g_m_n(b, m, n),
                                   d1_g_m_n(b, m, n));
                }
            }
        }

        pass = ck::utils::check_err(
            e_g_m_n_device_result, e_g_m_n_host_result, "Error: Incorrect results c");
    }

    if(config.time_kernel)
    {
        float ave_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});

        std::size_t flop      = std::size_t(2) * batch_count * M * N * K;
        std::size_t num_btype = sizeof(ADataType) * batch_count * M * K +
                                sizeof(BDataType) * batch_count * K * N +
                                sizeof(EDataType) * batch_count * M * N;

        float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
        float gb_per_sec = num_btype / 1.E6 / ave_time;
        std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s, " << gemm.GetTypeString() << std::endl;
    }

    return pass ? 0 : 1;
}

bool run_batched_gemm_rowwise_example(int argc, char* argv[])
{
    ProblemSize problem_size;
    ExecutionConfig config;

    std::mt19937 gen(11939);
    std::uniform_int_distribution<int> dis(0, 15);

    problem_size.M = 256 * (dis(gen) + 1);
    problem_size.N = 128 * (dis(gen) + 1);
    problem_size.K = 128 * (dis(gen) + 2);

    problem_size.batch_count = 2;

    if(argc == 4)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 8)
    {
        config.do_verification   = std::stoi(argv[1]);
        config.init_method       = std::stoi(argv[2]);
        config.time_kernel       = std::stoi(argv[3]);
        problem_size.M           = std::stoi(argv[4]);
        problem_size.N           = std::stoi(argv[5]);
        problem_size.K           = std::stoi(argv[6]);
        problem_size.batch_count = std::stoi(argv[7]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=n0, 1=yes)\n");
        printf("optinal\n");
        printf("arg4-7: M = %d N = %d K = %d Batch = %d\n",
               problem_size.M,
               problem_size.N,
               problem_size.K,
               problem_size.batch_count);
        exit(0);
    }

    problem_size.stride_A = problem_size.K;
    problem_size.stride_B = problem_size.K;
    problem_size.stride_C = problem_size.N;

    problem_size.stride_D0 = 0;
    problem_size.stride_D1 = 0;

    problem_size.batch_stride_A = problem_size.M * problem_size.K;
    problem_size.batch_stride_B = problem_size.K * problem_size.N;
    problem_size.batch_stride_C = problem_size.M * problem_size.N;

    problem_size.batch_stride_D0 = problem_size.N;
    problem_size.batch_stride_D1 = problem_size.M;

    return run_batched_gemm_rowwise(problem_size, config);
}
