// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/host_utility/device_prop.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_streamk.hpp"

template <typename ProblemType>
bool run_gemm(const ProblemType& problem_size, const ExecutionConfig& config)
{
#if defined(BUILD_INT4_EXAMPLE) && defined(CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4)
    static_assert(sizeof(ck::int4_t) == sizeof(int8_t));
#endif

    using namespace ck::literals;

    auto M       = problem_size.M;
    auto N       = problem_size.N;
    auto K       = problem_size.K;
    auto StrideA = problem_size.StrideA;
    auto StrideB = problem_size.StrideB;
    auto StrideC = problem_size.StrideC;

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
        [](std::size_t row, std::size_t col, ck::index_t stride, auto layout) {
            if(stride == -1 || stride == 0)
            {
                // give a chance if stride is -1, return a default packed stride
                if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
                {
                    return static_cast<std::size_t>(col);
                }
                else
                {
                    return static_cast<std::size_t>(row);
                }
            }
            else
                return static_cast<std::size_t>(stride);
        };

    StrideA = f_get_default_stride(M, K, StrideA, ALayout{});
    StrideB = f_get_default_stride(K, N, StrideB, BLayout{});
    StrideC = f_get_default_stride(M, N, StrideC, CLayout{});

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));

    switch(config.init_method)
    {
    case 0:
        ck::utils::FillConstant<ADataType>{ck::type_convert<ADataType>(1.f)}(a_m_k);
        ck::utils::FillConstant<BDataType>{ck::type_convert<BDataType>(1.f)}(b_k_n);
        break;
    case 1:
        ck::utils::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_m_k);
        ck::utils::FillUniformDistributionIntegerValue<BDataType>{-5.f, 5.f}(b_k_n);
        break;
    case 2:
        ck::utils::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k);
        ck::utils::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n);
        break;
    case 3:
        ck::utils::FillUniformDistributionIntegerValue<ADataType>{1.f, 1.f}(a_m_k);
        ck::utils::FillUniformDistributionIntegerValue<BDataType>{-5.f, 5.f}(b_k_n);
        break;
    case 4:
        ck::utils::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_m_k);
        ck::utils::FillUniformDistributionIntegerValue<BDataType>{1.f, 1.f}(b_k_n);
        break;
    case 5:
        ck::utils::FillUniformDistributionIntegerValue<ADataType>{-2.f, 2.f}(a_m_k);
        ck::utils::FillUniformDistributionIntegerValue<BDataType>{-2.f, 2.f}(b_k_n);
        break;
    default:
        ck::utils::FillUniformDistribution<ADataType>{-0.1f, 0.1f}(a_m_k);
        ck::utils::FillUniformDistribution<BDataType>{-0.1f, 0.1f}(b_k_n);
    }

    Tensor<CDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<CDataType> c_m_n_device_ref_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;

#ifdef BUILD_INT4_EXAMPLE
    DeviceMem a_m_k_device_buf(sizeof(KernelADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_device_buf(sizeof(KernelBDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(KernelCDataType) *
                               c_m_n_device_result.mDesc.GetElementSpaceSize());

    const Tensor<KernelADataType> a_m_k_converted(a_m_k);
    const Tensor<KernelBDataType> b_k_n_converted(b_k_n);

    a_m_k_device_buf.ToDevice(a_m_k_converted.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n_converted.mData.data());
#else
    DeviceMem a_m_k_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_ref_buf(sizeof(CDataType) *
                                   c_m_n_device_ref_result.mDesc.GetElementSpaceSize());

    a_m_k_device_buf.ToDevice(a_m_k.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n.mData.data());
#endif
    DeviceMem workspace;

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    using BaseStreamK = ck::tensor_operation::device::DeviceGemmStreamK<ALayout,
                                                                        BLayout,
                                                                        CLayout,
                                                                        ADataType,
                                                                        BDataType,
                                                                        CDataType,
                                                                        AElementOp,
                                                                        BElementOp,
                                                                        CElementOp>;

    // do GEMM
    static_assert(std::is_base_of<BaseStreamK, DeviceGemmInstance>::value &&
                  std::is_base_of<BaseStreamK, DeviceGemmInstance2>::value);
    auto gemm           = DeviceGemmInstance{};
    auto gemm2          = DeviceGemmInstance2{}; // instance for double rate mfma instruction
    BaseStreamK* op_ptr = (ck::get_device_name() == "gfx950") ? static_cast<BaseStreamK*>(&gemm2)
                                                              : static_cast<BaseStreamK*>(&gemm);

    float ave_time   = 0;
    auto invoker_ptr = op_ptr->MakeInvokerPointer();

    auto argument_ptr = op_ptr->MakeArgumentPointer(
#ifdef BUILD_INT4_EXAMPLE
        static_cast<KernelADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
        static_cast<KernelBDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
        static_cast<KernelCDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
#else
        static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
        static_cast<BDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
        static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
#endif
        M,
        N,
        K,
        StrideA,
        StrideB,
        StrideC,
        a_element_op,
        b_element_op,
        c_element_op,
        problem_size.NumSKBlocks);

    if(!op_ptr->IsSupportedArgument(argument_ptr.get()))
    {
        std::cerr << op_ptr->GetTypeString() << " does not support this problem" << std::endl;

        return true;
    }

    auto argument              = argument_ptr.get();
    std::size_t workspace_size = op_ptr->GetWorkSpaceSize(argument);
    if(workspace_size != 0)
    {
        workspace.Realloc(workspace_size);
        op_ptr->SetWorkSpacePointer(argument, workspace.GetDeviceBuffer());
    }

    ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, config.time_kernel});

    std::size_t flop = 2_uz * M * N * K;
    std::size_t num_btype =
        sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << op_ptr->GetTypeString() << std::endl;

    bool pass = true;

    if((config.do_verification == 1) || (config.do_verification == 3))
    {
        // CPU verification
        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, c_m_n_host_result, a_element_op, b_element_op, c_element_op);

        std::cout << "Running verification on CPU." << std::endl;
        ref_invoker.Run(ref_argument);

#ifdef BUILD_INT4_EXAMPLE
        Tensor<CDataType> c_m_n_device_result_converted(c_m_n_host_result.mDesc);

        c_m_n_device_buf.FromDevice(c_m_n_device_result_converted.mData.data());

        c_m_n_device_result = c_m_n_device_result_converted.CopyAsType<CDataType>();

        return ck::utils::check_err(c_m_n_device_result_converted, c_m_n_host_result);
#else
        c_m_n_device_buf.FromDevice(c_m_n_device_result.mData.data());

        pass &= ck::utils::check_err(c_m_n_device_result,
                                     c_m_n_host_result,
                                     "Error: Incorrect results!",
                                     get_rtol<CDataType>(),
                                     get_atol<CDataType>());
#endif
    }

    if((config.do_verification == 2) || (config.do_verification == 3))
    {
        // GPU verification
        auto ref_gemm_gpu    = ReferenceGemmInstanceGPU{};
        auto ref_invoker_gpu = ref_gemm_gpu.MakeInvoker();

        auto ref_argument_gpu = ref_gemm_gpu.MakeArgument(
            static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
            static_cast<BDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
            static_cast<CDataType*>(c_m_n_device_ref_buf.GetDeviceBuffer()),
            M,
            N,
            K,
            a_element_op,
            b_element_op,
            c_element_op);

        std::cout << "Running verification on GPU." << std::endl;
        ref_invoker_gpu.Run(ref_argument_gpu, StreamConfig{});

        c_m_n_device_ref_buf.FromDevice(c_m_n_device_ref_result.mData.data());
        c_m_n_device_buf.FromDevice(c_m_n_device_result.mData.data());

        pass &= ck::utils::check_err(c_m_n_device_result,
                                     c_m_n_device_ref_result,
                                     "Error: Incorrect results!",
                                     get_rtol<CDataType>(),
                                     get_atol<CDataType>());
    }

    return pass == true;
}

bool run_gemm_streamk_example(int argc, char* argv[])
{
    ProblemSizeStreamK problem_size;
    ExecutionConfig config;

    return !parse_cmd_args(argc, argv, problem_size, config) || run_gemm(problem_size, config);
}
