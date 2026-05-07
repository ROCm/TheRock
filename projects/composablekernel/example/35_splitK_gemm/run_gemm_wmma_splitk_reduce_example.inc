// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

template <typename ProblemType>
bool run_wmma_gemm(const ProblemType& problem_size, const ExecutionConfig& config)
{
    using namespace ck::literals;

    auto M       = problem_size.M;
    auto N       = problem_size.N;
    auto K       = problem_size.K;
    auto StrideA = problem_size.StrideA;
    auto StrideB = problem_size.StrideB;
    auto StrideC = problem_size.StrideC;
    auto KBatch  = problem_size.KBatch;

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

    StrideA = f_get_default_stride(M, K, StrideA, ALayout{});
    StrideB = f_get_default_stride(K, N, StrideB, BLayout{});
    StrideC = f_get_default_stride(M, N, StrideC, CLayout{});

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));

    switch(config.init_method)
    {
    case 0:
        a_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        b_k_n.GenerateTensorValue(GeneratorTensor_1<BDataType>{1});
        break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{-0.5, 0.5});
        b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        break;
    case 2:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        break;
    case 3:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_k_n.GenerateTensorValue(GeneratorTensor_1<BDataType>{1});
        break;
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
    }

    Tensor<CDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;
    std::cout << "init method: " << config.init_method << std::endl;
    std::cout << "KBatch: " << KBatch << std::endl;

    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    // device GEMM
    auto device_op = DeviceWmmaGemmInstance{};
    auto invoker   = device_op.MakeInvoker();

    auto argument =
        device_op.MakeArgumentPointer(static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
                                      static_cast<BDataType*>(b_device_buf.GetDeviceBuffer()),
                                      std::array<const void*, 0>{}, // empty D tensors
                                      static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
                                      M,
                                      N,
                                      K,
                                      StrideA,
                                      StrideB,
                                      std::array<ck::index_t, 0>{}, // empty D strides
                                      StrideC,
                                      KBatch,
                                      a_element_op,
                                      b_element_op,
                                      cde_element_op);

    // Allocate workspace for split-K reduction if needed
    size_t workspace_size = device_op.GetWorkSpaceSize(argument.get());
    DeviceMem workspace_buf(workspace_size);
    std::cout << "Workspace size: " << workspace_size << " bytes" << std::endl;
    if(workspace_size > 0)
    {
        argument->p_workspace_ = workspace_buf.GetDeviceBuffer();
        std::cout << "Allocated workspace of size: " << workspace_size << " bytes" << std::endl;
    }

    if(!device_op.IsSupportedArgument(argument.get()))
    {
        std::cout << "The runtime argument is not supported!" << std::endl;
        std::cout << "Debug info:" << std::endl;
        std::cout << "  M=" << M << ", N=" << N << ", K=" << K << ", KBatch=" << KBatch
                  << std::endl;
        std::cout << "  StrideA=" << StrideA << ", StrideB=" << StrideB << ", StrideC=" << StrideC
                  << std::endl;
        return false;
    }

    bool pass      = true;
    float ave_time = 0;

    if(config.do_verification)
    {
        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, c_m_n_host_result, a_element_op, b_element_op, cde_element_op);

        ref_invoker.Run(ref_argument);

        ave_time = invoker.Run(argument.get(), StreamConfig{nullptr, false});

        c_device_buf.FromDevice(c_m_n_device_result.mData.data());

        pass = ck::utils::check_err(c_m_n_device_result.mData,
                                    c_m_n_host_result.mData,
                                    "Error: Incorrect results!",
                                    get_rtol<CDataType>(),
                                    get_atol<CDataType>());
    }

    if(config.time_kernel)
    {
        ave_time = invoker.Run(argument.get(), StreamConfig{nullptr, config.time_kernel});

        std::size_t flop = std::size_t(2) * M * N * K;

        std::size_t num_btype =
            sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

        float tflops = static_cast<float>(flop) / 1.E12 / ave_time;

        float gb_per_sec = num_btype / 1.E9 / ave_time;

        std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s, " << device_op.GetTypeString() << std::endl;
    }

    return pass;
}

bool run_wmma_gemm_splitk_example(int argc, char* argv[])
{
    ProblemSizeSplitK problem_size;
    ExecutionConfig config;

    return !parse_cmd_args(argc, argv, problem_size, config) || run_wmma_gemm(problem_size, config);
}
