// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "ck/ck.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/numeric.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_contraction.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

int run_complex_contraction_bilinear_example(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // A[M0, M1, K0, K1]
    std::vector<ck::index_t> a_ms_ks_lengths{30, 128, 32, 64};
    std::vector<ck::index_t> a_ms_ks_strides{524288, 4096, 128, 1};
    // B[N0, N1, K0, K1]
    std::vector<ck::index_t> b_ns_ks_lengths{32, 64, 32, 64};
    std::vector<ck::index_t> b_ns_ks_strides{524288, 4096, 128, 1};
    // D[M0, M1, N0, N1]
    std::vector<ck::index_t> d_ms_ns_lengths{30, 128, 32, 64};
    std::vector<ck::index_t> d_ms_ns_strides{524288, 4096, 128, 1};
    // E[M0, M1, N0, N1]
    std::vector<ck::index_t> e_ms_ns_lengths{30, 128, 32, 64};
    std::vector<ck::index_t> e_ms_ns_strides{524288, 4096, 128, 1};

    float alpha = 1.f;
    float beta  = 1.f;

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
    else if(argc == 28)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        const ck::index_t M0 = std::stoi(argv[4]);
        const ck::index_t M1 = std::stoi(argv[5]);

        const ck::index_t N0 = std::stoi(argv[6]);
        const ck::index_t N1 = std::stoi(argv[7]);

        const ck::index_t K0 = std::stoi(argv[8]);
        const ck::index_t K1 = std::stoi(argv[9]);

        a_ms_ks_lengths = {M0, M1, K0, K1};
        a_ms_ks_strides = {
            std::stoi(argv[10]), std::stoi(argv[11]), std::stoi(argv[12]), std::stoi(argv[13])};

        b_ns_ks_lengths = {N0, N1, K0, K1};
        b_ns_ks_strides = {
            std::stoi(argv[14]), std::stoi(argv[15]), std::stoi(argv[16]), std::stoi(argv[17])};

        d_ms_ns_lengths = {M0, M1, N0, N1};
        d_ms_ns_strides = {
            std::stoi(argv[18]), std::stoi(argv[19]), std::stoi(argv[20]), std::stoi(argv[21])};

        e_ms_ns_lengths = {M0, M1, N0, N1};
        e_ms_ns_strides = {
            std::stoi(argv[22]), std::stoi(argv[23]), std::stoi(argv[24]), std::stoi(argv[25])};

        alpha = std::stof(argv[26]);
        beta  = std::stof(argv[27]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 9: M0, M1, N0, N1, K0, K1\n");
        printf("arg10 to 13: Stride_A_M0, Stride_A_M1, Stride_A_K0, Stride_A_K1\n");
        printf("arg14 to 17: Stride_B_N0, Stride_B_N1, Stride_B_K0, Stride_B_K1\n");
        printf("arg18 to 21: Stride_D_M0, Stride_D_M1, Stride_D_N0, Stride_D_N1\n");
        printf("arg22 to 25: Stride_E_M0, Stride_E_M1, Stride_E_N0, Stride_E_N1\n");
        printf("arg26 to 27: alpha, beta\n");
        exit(0);
    }

    using DefaultLayout = ck::tensor_layout::gemm::RowMajor;
    // For Real Part of Complex Tensor
    Tensor<ADataType> a_ms_ks_re(a_ms_ks_lengths, a_ms_ks_strides, DefaultLayout{});
    Tensor<BDataType> b_ns_ks_re(b_ns_ks_lengths, b_ns_ks_strides, DefaultLayout{});
    Tensor<EDataType> d_ms_ns_re(d_ms_ns_lengths, d_ms_ns_strides, DefaultLayout{});

    Tensor<EDataType> e_ms_ns_host_result_re(e_ms_ns_lengths, e_ms_ns_strides, DefaultLayout{});
    Tensor<EDataType> e_ms_ns_device_result_re(e_ms_ns_lengths, e_ms_ns_strides, DefaultLayout{});

    // For Imaginary Part of Complex Tensor
    Tensor<ADataType> a_ms_ks_img(a_ms_ks_lengths, a_ms_ks_strides, DefaultLayout{});
    Tensor<BDataType> b_ns_ks_img(b_ns_ks_lengths, b_ns_ks_strides, DefaultLayout{});
    Tensor<EDataType> d_ms_ns_img(d_ms_ns_lengths, d_ms_ns_strides, DefaultLayout{});

    Tensor<EDataType> e_ms_ns_host_result_img(e_ms_ns_lengths, e_ms_ns_strides, DefaultLayout{});
    Tensor<EDataType> e_ms_ns_device_result_img(e_ms_ns_lengths, e_ms_ns_strides, DefaultLayout{});

    // Intermediate E tensor Definition
    Tensor<EDataType> e_ms_ns_device_result_re1(e_ms_ns_lengths, e_ms_ns_strides, DefaultLayout{});
    Tensor<EDataType> e_ms_ns_device_result_img1(e_ms_ns_lengths, e_ms_ns_strides, DefaultLayout{});

    std::cout << "a_ms_ks_re: " << a_ms_ks_re.mDesc << std::endl;
    std::cout << "b_ns_ks_re: " << b_ns_ks_re.mDesc << std::endl;
    std::cout << "d_ms_ns_re: " << d_ms_ns_re.mDesc << std::endl;
    std::cout << "e_ms_ns_re: " << e_ms_ns_host_result_re.mDesc << std::endl;

    std::cout << "a_ms_ks_img: " << a_ms_ks_img.mDesc << std::endl;
    std::cout << "b_ns_ks_img: " << b_ns_ks_img.mDesc << std::endl;
    std::cout << "d_ms_ns_img: " << d_ms_ns_img.mDesc << std::endl;
    std::cout << "e_ms_ns_img: " << e_ms_ns_host_result_img.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:

        a_ms_ks_re.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b_ns_ks_re.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        d_ms_ns_re.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});

        a_ms_ks_img.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b_ns_ks_img.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        d_ms_ns_img.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        break;

    default:
        a_ms_ks_re.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_ns_ks_re.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        d_ms_ns_re.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});

        a_ms_ks_img.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_ns_ks_img.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        d_ms_ns_img.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});

        break;
    }

    DeviceMem a_device_buf_re(sizeof(ADataType) * a_ms_ks_re.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf_re(sizeof(BDataType) * b_ns_ks_re.mDesc.GetElementSpaceSize());
    DeviceMem d_device_buf_re(sizeof(DDataType) * d_ms_ns_re.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf_re(sizeof(EDataType) *
                              e_ms_ns_device_result_re.mDesc.GetElementSpaceSize());

    DeviceMem a_device_buf_img(sizeof(ADataType) * a_ms_ks_img.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf_img(sizeof(BDataType) * b_ns_ks_img.mDesc.GetElementSpaceSize());
    DeviceMem d_device_buf_img(sizeof(DDataType) * d_ms_ns_img.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf_img(sizeof(EDataType) *
                               e_ms_ns_device_result_img.mDesc.GetElementSpaceSize());

    // Intermediate Value For E Real and Img
    DeviceMem e_device_buf_re1(sizeof(EDataType) *
                               e_ms_ns_device_result_re.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf_img1(sizeof(EDataType) *
                                e_ms_ns_device_result_img.mDesc.GetElementSpaceSize());

    a_device_buf_re.ToDevice(a_ms_ks_re.mData.data());
    b_device_buf_re.ToDevice(b_ns_ks_re.mData.data());
    d_device_buf_re.ToDevice(d_ms_ns_re.mData.data());

    a_device_buf_img.ToDevice(a_ms_ks_img.mData.data());
    b_device_buf_img.ToDevice(b_ns_ks_img.mData.data());
    d_device_buf_img.ToDevice(d_ms_ns_img.mData.data());

    // set zero
    e_device_buf_re.SetZero();
    e_device_buf_img.SetZero();

    // set zero for intermediate values
    e_device_buf_re1.SetZero();
    e_device_buf_img1.SetZero();

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{alpha, beta};

    // device operation
    // For real Intermediate Value re_1

    auto op      = DeviceOpInstance{};
    auto invoker = op.MakeInvoker();
    auto argument_re1 =
        op.MakeArgument(a_device_buf_re.GetDeviceBuffer(),
                        b_device_buf_re.GetDeviceBuffer(),
                        std::array<const void*, 1>{d_device_buf_re.GetDeviceBuffer()},
                        e_device_buf_re1.GetDeviceBuffer(),
                        a_ms_ks_lengths,
                        a_ms_ks_strides,
                        b_ns_ks_lengths,
                        b_ns_ks_strides,
                        std::array<std::vector<ck::index_t>, 1>{d_ms_ns_lengths},
                        std::array<std::vector<ck::index_t>, 1>{d_ms_ns_strides},
                        e_ms_ns_lengths,
                        e_ms_ns_strides,
                        a_element_op,
                        b_element_op,
                        cde_element_op);

    if(!op.IsSupportedArgument(argument_re1))
    {
        std::cout << op.GetTypeString() << " does not support this problem" << std::endl;

        return 0;
    }

    float ave_time_re1 = invoker.Run(argument_re1, StreamConfig{nullptr, time_kernel});

    alpha = -1.f;
    beta  = 1.f;

    a_element_op   = AElementOp{};
    b_element_op   = BElementOp{};
    cde_element_op = CDEElementOp{alpha, beta};

    // device operation
    // For real Intermediate Value re_2
    // auto op       = DeviceOpInstance{};
    // auto invoker  = op.MakeInvoker();
    auto argument_re2 =
        op.MakeArgument(a_device_buf_img.GetDeviceBuffer(),
                        b_device_buf_img.GetDeviceBuffer(),
                        std::array<const void*, 1>{e_device_buf_re1.GetDeviceBuffer()},
                        e_device_buf_re.GetDeviceBuffer(),
                        a_ms_ks_lengths,
                        a_ms_ks_strides,
                        b_ns_ks_lengths,
                        b_ns_ks_strides,
                        std::array<std::vector<ck::index_t>, 1>{d_ms_ns_lengths},
                        std::array<std::vector<ck::index_t>, 1>{d_ms_ns_strides},
                        e_ms_ns_lengths,
                        e_ms_ns_strides,
                        a_element_op,
                        b_element_op,
                        cde_element_op);

    if(!op.IsSupportedArgument(argument_re2))
    {
        std::cout << op.GetTypeString() << " does not support this problem" << std::endl;

        return 0;
    }

    float ave_time_re2 = invoker.Run(argument_re2, StreamConfig{nullptr, time_kernel});

    alpha = 1.f;
    beta  = 1.f;

    a_element_op   = AElementOp{};
    b_element_op   = BElementOp{};
    cde_element_op = CDEElementOp{alpha, beta};

    auto argument_img1 =
        op.MakeArgument(a_device_buf_re.GetDeviceBuffer(),
                        b_device_buf_img.GetDeviceBuffer(),
                        std::array<const void*, 1>{d_device_buf_img.GetDeviceBuffer()},
                        e_device_buf_img1.GetDeviceBuffer(),
                        a_ms_ks_lengths,
                        a_ms_ks_strides,
                        b_ns_ks_lengths,
                        b_ns_ks_strides,
                        std::array<std::vector<ck::index_t>, 1>{d_ms_ns_lengths},
                        std::array<std::vector<ck::index_t>, 1>{d_ms_ns_strides},
                        e_ms_ns_lengths,
                        e_ms_ns_strides,
                        a_element_op,
                        b_element_op,
                        cde_element_op);

    if(!op.IsSupportedArgument(argument_img1))
    {
        std::cout << op.GetTypeString() << " does not support this problem" << std::endl;

        return 0;
    }

    float ave_time_img1 = invoker.Run(argument_img1, StreamConfig{nullptr, time_kernel});

    alpha = 1.f;
    beta  = 1.f;

    auto argument_img2 =
        op.MakeArgument(a_device_buf_img.GetDeviceBuffer(),
                        b_device_buf_re.GetDeviceBuffer(),
                        std::array<const void*, 1>{e_device_buf_img1.GetDeviceBuffer()},
                        e_device_buf_img.GetDeviceBuffer(),
                        a_ms_ks_lengths,
                        a_ms_ks_strides,
                        b_ns_ks_lengths,
                        b_ns_ks_strides,
                        std::array<std::vector<ck::index_t>, 1>{d_ms_ns_lengths},
                        std::array<std::vector<ck::index_t>, 1>{d_ms_ns_strides},
                        e_ms_ns_lengths,
                        e_ms_ns_strides,
                        a_element_op,
                        b_element_op,
                        cde_element_op);

    if(!op.IsSupportedArgument(argument_img2))
    {
        std::cout << op.GetTypeString() << " does not support this problem" << std::endl;

        return 0;
    }

    float ave_time_img2 = invoker.Run(argument_img2, StreamConfig{nullptr, time_kernel});

    ck::index_t M =
        ck::accumulate_n<ck::index_t>(e_ms_ns_lengths.begin(), NumDimM, 1, std::multiplies<>{});

    ck::index_t N = ck::accumulate_n<ck::index_t>(
        e_ms_ns_lengths.begin() + NumDimM, NumDimN, 1, std::multiplies<>{});

    ck::index_t K = ck::accumulate_n<ck::index_t>(
        a_ms_ks_lengths.begin() + NumDimM, NumDimK, 1, std::multiplies<>{});

    std::size_t flop      = std::size_t(2) * M * N * K * 2;
    std::size_t num_btype = sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
                            sizeof(DDataType) * M * N + sizeof(EDataType) * M * N * 2;

    float ave_time = ave_time_img2 + ave_time_img1 + ave_time_re2 + ave_time_re1;

    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << op.GetTypeString() << std::endl;

    e_device_buf_re.FromDevice(e_ms_ns_device_result_re.mData.data());
    e_device_buf_img.FromDevice(e_ms_ns_device_result_img.mData.data());

    auto isRealOk = 0;
    auto isImgOk  = 0;

    if(do_verification)
    {
        // Real Part Verification
        Tensor<CShuffleDataType> c_ms_ns_host_result_re(
            e_ms_ns_lengths, e_ms_ns_strides, DefaultLayout{});
        Tensor<CShuffleDataType> c_ms_ns_host_result_re1(
            e_ms_ns_lengths, e_ms_ns_strides, DefaultLayout{});

        using ReferenceOpInstance =
            ck::tensor_operation::host::ReferenceContraction_M2_N2_K2<NumDimM,
                                                                      NumDimN,
                                                                      NumDimK,
                                                                      ADataType,
                                                                      BDataType,
                                                                      CShuffleDataType,
                                                                      AccDataType,
                                                                      F32,
                                                                      AElementOp,
                                                                      BElementOp>;

        auto ref_op      = ReferenceOpInstance{};
        auto ref_invoker = ref_op.MakeInvoker();

        auto ref_argument_re = ref_op.MakeArgument(
            a_ms_ks_re, b_ns_ks_re, c_ms_ns_host_result_re, a_element_op, b_element_op);

        ref_invoker.Run(ref_argument_re);

        alpha = 1.f;
        beta  = 1.f;

        cde_element_op = CDEElementOp{alpha, beta};

        for(size_t m0 = 0; m0 < e_ms_ns_host_result_re.mDesc.GetLengths()[0]; ++m0)
        {
            for(size_t m1 = 0; m1 < e_ms_ns_host_result_re.mDesc.GetLengths()[1]; ++m1)
            {
                for(size_t n0 = 0; n0 < e_ms_ns_host_result_re.mDesc.GetLengths()[2]; ++n0)
                {
                    for(size_t n1 = 0; n1 < e_ms_ns_host_result_re.mDesc.GetLengths()[3]; ++n1)
                    {
                        cde_element_op(e_ms_ns_host_result_re(m0, m1, n0, n1),
                                       c_ms_ns_host_result_re(m0, m1, n0, n1),
                                       d_ms_ns_re(m0, m1, n0, n1));
                    }
                }
            }
        }

        alpha = 1.f;
        beta  = -1.f;

        cde_element_op = CDEElementOp{alpha, beta};

        auto ref_argument_re1 = ref_op.MakeArgument(
            a_ms_ks_img, b_ns_ks_img, c_ms_ns_host_result_re1, a_element_op, b_element_op);

        ref_invoker.Run(ref_argument_re1);

        for(size_t m0 = 0; m0 < e_ms_ns_host_result_re.mDesc.GetLengths()[0]; ++m0)
        {
            for(size_t m1 = 0; m1 < e_ms_ns_host_result_re.mDesc.GetLengths()[1]; ++m1)
            {
                for(size_t n0 = 0; n0 < e_ms_ns_host_result_re.mDesc.GetLengths()[2]; ++n0)
                {
                    for(size_t n1 = 0; n1 < e_ms_ns_host_result_re.mDesc.GetLengths()[3]; ++n1)
                    {
                        cde_element_op(e_ms_ns_host_result_re(m0, m1, n0, n1),
                                       e_ms_ns_host_result_re(m0, m1, n0, n1),
                                       c_ms_ns_host_result_re1(m0, m1, n0, n1));
                    }
                }
            }
        }

        isRealOk = ck::utils::check_err(e_ms_ns_device_result_re, e_ms_ns_host_result_re) ? 0 : 1;

        // Img Part Verification
        Tensor<CShuffleDataType> c_ms_ns_host_result_img(
            e_ms_ns_lengths, e_ms_ns_strides, DefaultLayout{});
        Tensor<CShuffleDataType> c_ms_ns_host_result_img1(
            e_ms_ns_lengths, e_ms_ns_strides, DefaultLayout{});

        auto ref_argument_img = ref_op.MakeArgument(
            a_ms_ks_re, b_ns_ks_img, c_ms_ns_host_result_img, a_element_op, b_element_op);

        ref_invoker.Run(ref_argument_img);

        alpha = 1.f;
        beta  = 1.f;

        cde_element_op = CDEElementOp{alpha, beta};

        for(size_t m0 = 0; m0 < e_ms_ns_host_result_img.mDesc.GetLengths()[0]; ++m0)
        {
            for(size_t m1 = 0; m1 < e_ms_ns_host_result_img.mDesc.GetLengths()[1]; ++m1)
            {
                for(size_t n0 = 0; n0 < e_ms_ns_host_result_img.mDesc.GetLengths()[2]; ++n0)
                {
                    for(size_t n1 = 0; n1 < e_ms_ns_host_result_img.mDesc.GetLengths()[3]; ++n1)
                    {
                        cde_element_op(e_ms_ns_host_result_img(m0, m1, n0, n1),
                                       c_ms_ns_host_result_img(m0, m1, n0, n1),
                                       d_ms_ns_img(m0, m1, n0, n1));
                    }
                }
            }
        }

        auto ref_argument_img1 = ref_op.MakeArgument(
            a_ms_ks_img, b_ns_ks_re, c_ms_ns_host_result_img1, a_element_op, b_element_op);

        ref_invoker.Run(ref_argument_img1);

        for(size_t m0 = 0; m0 < e_ms_ns_host_result_img.mDesc.GetLengths()[0]; ++m0)
        {
            for(size_t m1 = 0; m1 < e_ms_ns_host_result_img.mDesc.GetLengths()[1]; ++m1)
            {
                for(size_t n0 = 0; n0 < e_ms_ns_host_result_img.mDesc.GetLengths()[2]; ++n0)
                {
                    for(size_t n1 = 0; n1 < e_ms_ns_host_result_img.mDesc.GetLengths()[3]; ++n1)
                    {
                        cde_element_op(e_ms_ns_host_result_img(m0, m1, n0, n1),
                                       e_ms_ns_host_result_img(m0, m1, n0, n1),
                                       c_ms_ns_host_result_img1(m0, m1, n0, n1));
                    }
                }
            }
        }

        isImgOk = ck::utils::check_err(e_ms_ns_device_result_re, e_ms_ns_host_result_re) ? 0 : 1;

        return (isRealOk && isImgOk);
    }

    return 0;
}
