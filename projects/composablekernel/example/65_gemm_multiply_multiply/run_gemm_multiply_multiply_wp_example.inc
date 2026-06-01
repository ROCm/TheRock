// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

int run_gemm_example(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // GEMM shape
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA = K;
    ck::index_t StrideB = K;
    ck::index_t StrideD = 0;
    ck::index_t StrideE = N;

    ck::index_t KBatch = 1;

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 12)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M = std::stoi(argv[4]);
        N = std::stoi(argv[5]);
        K = std::stoi(argv[6]);

        StrideA = std::stoi(argv[7]);
        StrideB = std::stoi(argv[8]);
        StrideD = std::stoi(argv[9]);
        StrideE = std::stoi(argv[10]);

        KBatch = std::stoi(argv[11]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf(
            "arg4 to 11: M (256x), N(128x), K(32x), StrideA, StrideB, StrideD, StrideE, KBatch\n");
        exit(0);
    }

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            using namespace ck::literals;

            if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
            {
                return ck::HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return ck::HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    ck::Tensor<A0DataType> a0_m_k(f_host_tensor_descriptor(M, K, StrideA, A0Layout{}));
    ck::Tensor<B0DataType> b0_k_n(f_host_tensor_descriptor(K, N, StrideB, B0Layout{}));
    ck::Tensor<B0DataType> b0_preshuffled(
        f_host_tensor_descriptor(K, N, StrideB, B0Layout{})); // use laout only for size
    ck::Tensor<D0DataType> d0_m_n(f_host_tensor_descriptor(M, N, StrideD, D0Layout{}));
    ck::Tensor<D1DataType> d1_m_n(f_host_tensor_descriptor(M, N, StrideD, D1Layout{}));
    ck::Tensor<EDataType> e_m_n_host_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));
    ck::Tensor<EDataType> e_m_n_device_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));

    // Update strides based on tensor properties if they are <= 0
    auto get_stride = [](auto& tensor, auto layout, ck::index_t current_stride) -> ck::index_t {
        if(current_stride <= 0)
        {
            if constexpr(std::is_same_v<decltype(layout), Row>)
            {
                return tensor.GetStrides()[0];
            }
            else
            {
                return tensor.GetStrides()[1];
            }
        }
        return current_stride;
    };

    StrideA              = get_stride(a0_m_k, A0Layout{}, StrideA);
    StrideB              = get_stride(b0_k_n, B0Layout{}, StrideB);
    ck::index_t StrideD0 = get_stride(d0_m_n, D0Layout{}, StrideD);
    ck::index_t StrideD1 = get_stride(d1_m_n, D1Layout{}, StrideD);
    StrideE              = get_stride(e_m_n_host_result, ELayout{}, StrideE);

    std::cout << "a0_m_k: " << a0_m_k.mDesc << std::endl;
    std::cout << "b0_k_n: " << b0_k_n.mDesc << std::endl;
    std::cout << "d1_m_n: " << d1_m_n.mDesc << std::endl;
    std::cout << "d0_m_n: " << d0_m_n.mDesc << std::endl;
    std::cout << "e_m_n : " << e_m_n_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a0_m_k.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-2, 2});
        b0_k_n.GenerateTensorValue(GeneratorTensor_2<B0DataType>{0, 2});
        d0_m_n.GenerateTensorValue(GeneratorTensor_2<D0DataType>{-2, 2});
        d1_m_n.GenerateTensorValue(GeneratorTensor_2<D1DataType>{-2, 2});
        break;
    case 2:
        a0_m_k.GenerateTensorValue(GeneratorTensor_1<A0DataType>{});
        b0_k_n.GenerateTensorValue(GeneratorTensor_1<B0DataType>{});
        d0_m_n.GenerateTensorValue(GeneratorTensor_1<D0DataType>{});
        d1_m_n.GenerateTensorValue(GeneratorTensor_1<D1DataType>{});
        break;
    default:
        a0_m_k.GenerateTensorValue(GeneratorTensor_3<A0DataType>{0.0, 1.0});
        b0_k_n.GenerateTensorValue(GeneratorTensor_3<B0DataType>{-0.5, 0.5});
        d0_m_n.GenerateTensorValue(GeneratorTensor_3<D0DataType>{0.0, 1.0});
        d1_m_n.GenerateTensorValue(GeneratorTensor_3<D1DataType>{0.0, 1.0});
    }
    ck::DeviceMem a0_device_buf(sizeof(A0DataType) * a0_m_k.mDesc.GetElementSpaceSize());
    ck::DeviceMem b0_device_buf(sizeof(B0DataType) * b0_k_n.mDesc.GetElementSpaceSize());
    ck::DeviceMem d0_device_buf(sizeof(D0DataType) * d0_m_n.mDesc.GetElementSpaceSize());
    ck::DeviceMem d1_device_buf(sizeof(D1DataType) * d1_m_n.mDesc.GetElementSpaceSize());
    ck::DeviceMem e_device_buf(sizeof(EDataType) * e_m_n_device_result.mDesc.GetElementSpaceSize());

    a0_device_buf.ToDevice(a0_m_k.mData.data());
    d0_device_buf.ToDevice(d0_m_n.mData.data());
    d1_device_buf.ToDevice(d1_m_n.mData.data());
    e_device_buf.ToDevice(e_m_n_device_result.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    constexpr ck::index_t NumDTensor = DsDataType::Size();

    // do GEMM
    auto device_op = DeviceOpInstance{};

    int NPerWmma = device_op.GetPreShuffleParameters();

    preShuffleBuffer<KPack>(b0_k_n.mData.data(), b0_preshuffled.mData.data(), N, K, NPerWmma);

    b0_device_buf.ToDevice(b0_preshuffled.mData.data());

    auto invoker = device_op.MakeInvoker();
    auto argument =
        device_op.MakeArgument(a0_device_buf.GetDeviceBuffer(),
                               b0_device_buf.GetDeviceBuffer(),
                               std::array<const void*, NumDTensor>{d0_device_buf.GetDeviceBuffer(),
                                                                   d1_device_buf.GetDeviceBuffer()},
                               e_device_buf.GetDeviceBuffer(),
                               M,
                               N,
                               K,
                               StrideA,
                               StrideB,
                               std::array<ck::index_t, NumDTensor>{StrideD0, StrideD1},
                               StrideE,
                               KBatch,
                               a_element_op,
                               b_element_op,
                               cde_element_op);

    if(!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel, 0, 50, 50, false, 1});

    std::size_t flop      = std::size_t(2) * M * N * K;
    std::size_t num_btype = sizeof(A0DataType) * M * K + sizeof(B0DataType) * K * N +
                            sizeof(D0DataType) * M * N + sizeof(D1DataType) * M * N +
                            sizeof(EDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    if(do_verification)
    {
        invoker.Run(argument, StreamConfig{nullptr, false});

        e_device_buf.FromDevice(e_m_n_device_result.mData.data());

        ck::Tensor<CShuffleDataType> c_m_n({M, N});

        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<A0DataType,
                                                                                B0DataType,
                                                                                CShuffleDataType,
                                                                                AccDataType,
                                                                                PassThrough,
                                                                                PassThrough,
                                                                                PassThrough>;
        auto ref_gemm               = ReferenceGemmInstance{};
        auto ref_invoker            = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a0_m_k, b0_k_n, c_m_n, PassThrough{}, PassThrough{}, PassThrough{});

        ref_invoker.Run(ref_argument);

        for(int m = 0; m < M; ++m)
        {
            for(int n = 0; n < N; ++n)
            {
                cde_element_op(e_m_n_host_result(m, n), c_m_n(m, n), d0_m_n(m, n), d1_m_n(m, n));
            }
        }

        e_device_buf.FromDevice(e_m_n_device_result.mData.data());

        if(ck::utils::check_err(
               e_m_n_device_result, e_m_n_host_result, "Error: Incorrect results!", 1e-3, 5e-2))
        {
            std::cout << "Example PASS\n";
            return 0;
        }
        else
        {
            std::cout << "Example FAIL\n";
            return 1;
        }
    }

    return 0;
}
