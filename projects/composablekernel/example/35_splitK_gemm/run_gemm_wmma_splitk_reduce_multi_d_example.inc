// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

template <typename ProblemSize>
bool run_wmma_gemm(const ProblemSize& problem_size, const ExecutionConfig& config)
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

    Tensor<ADataType> a_m_k(
        f_host_tensor_descriptor(problem_size.M, problem_size.K, problem_size.StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(
        f_host_tensor_descriptor(problem_size.K, problem_size.N, problem_size.StrideB, BLayout{}));
    Tensor<D0DataType> d0_m_n(
        f_host_tensor_descriptor(problem_size.M, problem_size.N, problem_size.StrideC, D0Layout{}));

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

    Tensor<CDataType> c_m_n_host_result(
        f_host_tensor_descriptor(problem_size.M, problem_size.N, problem_size.StrideC, CLayout{}));
    Tensor<CDataType> c_m_n_device_result(
        f_host_tensor_descriptor(problem_size.M, problem_size.N, problem_size.StrideC, CLayout{}));

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
    auto gemm                                              = DeviceGemmV2Instance{};
    auto invoker                                           = gemm.MakeInvoker();
    constexpr auto kNum_DTensors                           = DsDataType::Size();
    const std::array<const void*, kNum_DTensors> p_ds      = {d0_m_n_device_buf.GetDeviceBuffer()};
    const std::array<ck::index_t, kNum_DTensors> d_strides = {problem_size.StrideC};

    auto argument =
        gemm.MakeArgumentPointer(static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                                 static_cast<BDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
                                 p_ds,
                                 static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
                                 problem_size.M,
                                 problem_size.N,
                                 problem_size.K,
                                 problem_size.StrideA,
                                 problem_size.StrideB,
                                 d_strides,
                                 problem_size.StrideC,
                                 problem_size.KBatch,
                                 a_element_op,
                                 b_element_op,
                                 c_element_op);

    if(!gemm.IsSupportedArgument(argument.get()))
    {
        std::cerr << gemm.GetTypeString() << " does not support this problem" << std::endl;
        return false;
    }

    auto workspace_size = gemm.GetWorkSpaceSize(argument.get());
    DeviceMem workspace_device_buf(workspace_size);

    std::cout << "Workspace size: " << workspace_size << " bytes" << std::endl;
    std::cout << "Allocated workspace of size: " << workspace_size << " bytes" << std::endl;

    if(workspace_size > 0)
    {
        argument->p_workspace_ = workspace_device_buf.GetDeviceBuffer();
    }

    if(config.do_verification)
    {
        using ReferenceGemmInstanceMultiD = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                                      BDataType,
                                                                                      CDataType,
                                                                                      AccDataType,
                                                                                      AElementOp,
                                                                                      BElementOp,
                                                                                      PassThrough>;

        auto ref_gemm    = ReferenceGemmInstanceMultiD{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, c_m_n_host_result, a_element_op, b_element_op, PassThrough{});

        ref_invoker.Run(ref_argument);

        c_m_n_host_result.ForEach(
            [&](auto& self, auto idx) { c_element_op(self(idx), self(idx), d0_m_n(idx)); });
    }

    std::cout << "init method: " << config.init_method << std::endl;
    std::cout << "KBatch: " << problem_size.KBatch << std::endl;

    float ave_time = invoker.Run(argument.get(), StreamConfig{nullptr, config.time_kernel});

    std::size_t flop      = std::size_t(2) * problem_size.M * problem_size.N * problem_size.K;
    std::size_t num_btype = sizeof(ADataType) * problem_size.M * problem_size.K +
                            sizeof(BDataType) * problem_size.K * problem_size.N +
                            sizeof(CDataType) * problem_size.M * problem_size.N +
                            sizeof(D0DataType) * problem_size.M * problem_size.N;

    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;

    if(config.do_verification)
    {
        c_m_n_device_buf.FromDevice(c_m_n_device_result.mData.data());

        double rtol = get_rtol<CDataType>();
        double atol = get_atol<CDataType>();

        return ck::utils::check_err(
            c_m_n_device_result, c_m_n_host_result, "Error: Incorrect results!", rtol, atol);
    }

    return true;
}

int run_gemm_splitk_multi_d_example(int argc, char* argv[])
{
    ProblemSizeSplitK problem_size;
    ExecutionConfig config;

    return !parse_cmd_args(argc, argv, problem_size, config) || run_wmma_gemm(problem_size, config);
}
