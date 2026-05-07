// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

template <typename ProblemType>
bool run_gemm(const ProblemType& problem_size, const ExecutionConfig& config)
{
    using namespace ck::literals;

    auto M        = problem_size.M;
    auto N        = problem_size.N;
    auto K        = problem_size.K;
    auto StrideA  = problem_size.StrideA;
    auto StrideB  = problem_size.StrideB;
    auto StrideC  = problem_size.StrideC;
    auto StrideD0 = problem_size.StrideC;
    auto KBatch   = problem_size.KBatch;

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    auto f_get_default_stride =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if(stride == 0)
            {
                // give a chance if stride is zero, return a default packed stride
                if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
                {
                    return col;
                }
                else
                {
                    return row;
                }
            }
            else
                return stride;
        };

    StrideA  = f_get_default_stride(M, K, StrideA, ALayout{});
    StrideB  = f_get_default_stride(K, N, StrideB, BLayout{});
    StrideC  = f_get_default_stride(M, N, StrideC, CLayout{});
    StrideD0 = f_get_default_stride(M, N, StrideD0, D0Layout{});

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
    Tensor<D0DataType> d0_m_n(f_host_tensor_descriptor(M, N, StrideD0, D0Layout{}));

    switch(config.init_method)
    {
    case 0:
        a_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        b_k_n.GenerateTensorValue(GeneratorTensor_1<BDataType>{1});
        d0_m_n.GenerateTensorValue(GeneratorTensor_1<D0DataType>{1});
        break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{-0.5, 0.5});
        b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        d0_m_n.GenerateTensorValue(GeneratorTensor_3<D0DataType>{-0.5, 0.5});
        break;
    case 2:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        d0_m_n.GenerateTensorValue(GeneratorTensor_2<D0DataType>{-2, 2});
        break;
    case 3:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_k_n.GenerateTensorValue(GeneratorTensor_1<BDataType>{1});
        d0_m_n.GenerateTensorValue(GeneratorTensor_1<D0DataType>{1});
        break;
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        d0_m_n.GenerateTensorValue(GeneratorTensor_3<D0DataType>{-0.5, 0.5});
    }
#if 0
    printf("B matrix:\n");
    for (int in = 0; in < N; in++)
    {
        for (int ik = 0; ik < K; ik++)
        {
            printf("%02x ", *(reinterpret_cast<uint8_t*>(&b_k_n(ik,in))));
            if(ik%8==7) printf("|");
        }
        printf("\n");
    }
#endif

    Tensor<CDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;
    std::cout << "init method: " << config.init_method << std::endl;
    std::cout << "KBatch: " << KBatch << std::endl;

    DeviceMem a_m_k_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpaceSize());
    DeviceMem d0_m_n_device_buf(sizeof(D0DataType) * d0_m_n.mDesc.GetElementSpaceSize());

    a_m_k_device_buf.ToDevice(a_m_k.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n.mData.data());
    d0_m_n_device_buf.ToDevice(d0_m_n.mData.data());

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CDEElementOp{};

    // do GEMM
    auto gemm        = DeviceGemmV2Instance{};
    auto invoker     = gemm.MakeInvoker();
    float ave_time   = 0;
    auto get_argment = [&]() {
        if constexpr(DsDataType::Size() > 0)
        {
            return gemm.MakeArgument(static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                                     static_cast<BDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
                                     {d0_m_n_device_buf.GetDeviceBuffer()},
                                     static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
                                     M,
                                     N,
                                     K,
                                     StrideA,
                                     StrideB,
                                     {StrideD0},
                                     StrideC,
                                     KBatch,
                                     a_element_op,
                                     b_element_op,
                                     c_element_op);
        }
        else
        {
            return gemm.MakeArgument(static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                                     static_cast<BDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
                                     {},
                                     static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
                                     M,
                                     N,
                                     K,
                                     StrideA,
                                     StrideB,
                                     {},
                                     StrideC,
                                     KBatch,
                                     a_element_op,
                                     b_element_op,
                                     c_element_op);
        }
    };
    auto argument = get_argment();

    if(!gemm.IsSupportedArgument(argument))
    {
        std::cerr << gemm.GetTypeString() << " does not support this problem" << std::endl;

        return true;
    }

    DeviceMem gemm_workspace_dev(gemm.GetWorkSpaceSize(&argument));
    gemm.SetWorkSpacePointer(&argument, gemm_workspace_dev.GetDeviceBuffer(), StreamConfig{});

    bool pass = true;
    if(config.do_verification)
    {
        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, c_m_n_host_result, PassThrough{}, PassThrough{}, PassThrough{});

        ref_invoker.Run(ref_argument);

        ave_time = invoker.Run(argument, StreamConfig{nullptr, false, 1});

        c_m_n_device_buf.FromDevice(c_m_n_device_result.mData.data());

        if constexpr(DsDataType::Size() > 0)
        {
            c_m_n_host_result.ForEach(
                [&](auto& self, auto idx) { c_element_op(self(idx), self(idx), d0_m_n(idx)); });
        }

        pass &= ck::utils::check_err(c_m_n_device_result,
                                     c_m_n_host_result,
                                     "Error: Incorrect results!",
                                     get_rtol<CDataType>(),
                                     get_atol<CDataType>());
    }

    if(config.time_kernel)
    {
        ave_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});

        std::size_t flop = 2_uz * M * N * K;
        std::size_t num_btype =
            sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

        float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

        float gb_per_sec = num_btype / 1.E6 / ave_time;

        std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s, " << gemm.GetTypeString() << std::endl;
    }
    return pass;
}

bool run_gemm_splitk_example(int argc, char* argv[])
{
    ProblemSizeSplitK problem_size;
    ExecutionConfig config;

    return !parse_cmd_args(argc, argv, problem_size, config) || run_gemm(problem_size, config);
}
