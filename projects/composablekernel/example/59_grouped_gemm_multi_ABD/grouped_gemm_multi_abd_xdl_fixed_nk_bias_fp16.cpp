// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multi_abd_xdl_fixed_nk.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_multi_abd.hpp"
#include "ck/tensor_operation/gpu/element/combined_element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

using ::ck::DeviceMem;
using ::ck::hip_check_error;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row    = ck::tensor_layout::gemm::RowMajor;
using Col    = ck::tensor_layout::gemm::ColumnMajor;
using Bypass = ck::tensor_layout::BypassLayoutVerification;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Add         = ck::tensor_operation::element_wise::Add;
using Scale       = ck::tensor_operation::element_wise::Scale;
using AddScale = ck::tensor_operation::element_wise::BinaryWithUnaryCombinedOp<Add, Scale, Scale>;

using A0DataType       = F16;
using A1DataType       = F32;
using AsDataType       = ck::Tuple<A0DataType, A1DataType>;
using B0DataType       = F16;
using BsDataType       = ck::Tuple<B0DataType>;
using AccDataType      = F32;
using CShuffleDataType = F32;
using D0DataType       = F32;
using DsDataType       = ck::Tuple<D0DataType>;
using EDataType        = F32;

using A0Layout = Row;
using A1Layout = Row;
using AsLayout = ck::Tuple<A0Layout, A1Layout>;
using B0Layout = Col;
using BsLayout = ck::Tuple<B0Layout>;
using D0Layout = Row;
using DsLayout = ck::Tuple<D0Layout>;
using ELayout  = Row;

using AElementOp = AddScale;
using BElementOp = PassThrough;

using CDEElementOp = Add;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

using DeviceGemmInstance = ck::tensor_operation::device::DeviceGroupedGemm_Xdl_Multi_ABD_Fixed_NK
    // clang-format off
///######|  ALayout|  BLayout| DsLayout| ELayout|      AData|      BData|     AccData|         CShuffle|     DsData|     EData|           A|           B|          CDE|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
///######|         |         |         |        |       Type|       Type|        Type|         DataType|       Type|      Type| Elementwise| Elementwise|  Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
///######|         |         |         |        |           |           |            |                 |           |          |   Operation|   Operation|    Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
///######|         |         |         |        |           |           |            |                 |           |          |            |            |             |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
         < AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,    GemmDefault,        1,   128,    16,   128,    32,   8,   8,   16,   16,    1,    4,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              1,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              1,         1,           1,           1,               S<1, 16, 1, 8>,               1,  ck::half_t>;

// clang-format on

struct ProblemSize final
{
    std::vector<ck::index_t> Ms;
    std::vector<ck::index_t> Ns;
    std::vector<ck::index_t> Ks;

    std::vector<ck::index_t> stride_As;
    std::vector<ck::index_t> stride_Bs;
    std::vector<ck::index_t> stride_Cs;

    ck::index_t group_count;
};

struct ExecutionConfig final
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
    int k_batch          = 1;
};

bool run_grouped_gemm(const ProblemSize& problem_size, const ExecutionConfig& config)
{
    auto group_count = problem_size.group_count;

    // GEMM shape
    std::vector<ck::tensor_operation::device::GemmMultiABDDesc> gemm_descs;

    gemm_descs.reserve(group_count);

    int sum_of_m = 0;

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            using namespace ck::literals;

            if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz}, Bypass{});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride}, Bypass{});
            }
        };

    std::vector<Tensor<A0DataType>> a0_tensors;
    std::vector<Tensor<A1DataType>> a1_tensors;
    std::vector<Tensor<B0DataType>> b_tensors;
    std::vector<Tensor<D0DataType>> d0_tensors;
    std::vector<Tensor<EDataType>> e_host_tensors;
    std::vector<Tensor<EDataType>> e_device_tensors;

    a0_tensors.reserve(group_count);
    a1_tensors.reserve(group_count);
    b_tensors.reserve(group_count);
    d0_tensors.reserve(group_count);
    e_host_tensors.reserve(group_count);
    e_device_tensors.reserve(group_count);

    using DeviceMemPtr = std::unique_ptr<DeviceMem>;

    std::vector<DeviceMemPtr> a0_tensors_device, a1_tensors_device, b_tensors_device,
        d0_tensors_device, c_tensors_device;

    a0_tensors_device.reserve(group_count);
    a1_tensors_device.reserve(group_count);
    b_tensors_device.reserve(group_count);
    d0_tensors_device.reserve(group_count);
    c_tensors_device.reserve(group_count);

    std::size_t flop = 0, num_btype = 0;

    for(int i = 0; i < group_count; i++)
    {
        sum_of_m += problem_size.Ms[i];
        a0_tensors.push_back(Tensor<A0DataType>(f_host_tensor_descriptor(
            problem_size.Ms[i], problem_size.Ks[i], problem_size.stride_As[i], A0Layout{})));
        a1_tensors.push_back(Tensor<A1DataType>(f_host_tensor_descriptor(
            problem_size.Ms[i], problem_size.Ks[i], problem_size.stride_As[i], A1Layout{})));
        b_tensors.push_back(Tensor<B0DataType>(f_host_tensor_descriptor(
            problem_size.Ks[i], problem_size.Ns[i], problem_size.stride_Bs[i], B0Layout{})));
        d0_tensors.push_back(Tensor<D0DataType>(
            f_host_tensor_descriptor(problem_size.Ms[i], problem_size.Ns[i], 0, ELayout{})));
        e_host_tensors.push_back(Tensor<EDataType>(f_host_tensor_descriptor(
            problem_size.Ms[i], problem_size.Ns[i], problem_size.stride_Cs[i], ELayout{})));
        e_device_tensors.push_back(Tensor<EDataType>(f_host_tensor_descriptor(
            problem_size.Ms[i], problem_size.Ns[i], problem_size.stride_Cs[i], ELayout{})));
        std::cout << "gemm[" << i << "] a_m_k: " << a0_tensors[i].mDesc
                  << " b_k_n: " << b_tensors[i].mDesc << " d_m_n: " << d0_tensors[i].mDesc
                  << " c_m_n: " << e_device_tensors[i].mDesc << std::endl;

        flop += std::size_t(2) * problem_size.Ms[i] * problem_size.Ks[i] * problem_size.Ns[i];
        num_btype += sizeof(A0DataType) * a0_tensors[i].mDesc.GetElementSize() +
                     sizeof(A1DataType) * a1_tensors[i].mDesc.GetElementSize() +
                     sizeof(B0DataType) * b_tensors[i].mDesc.GetElementSize() +
                     sizeof(D0DataType) * d0_tensors[i].mDesc.GetElementSize() +
                     sizeof(EDataType) * e_device_tensors[i].mDesc.GetElementSize();

        switch(config.init_method)
        {
        case 0: break;
        case 1:
            a0_tensors[i].GenerateTensorValue(GeneratorTensor_2<A0DataType>{-5, 5});
            a1_tensors[i].GenerateTensorValue(GeneratorTensor_2<A1DataType>{-5, 5});
            b_tensors[i].GenerateTensorValue(GeneratorTensor_2<B0DataType>{-5, 5});
            break;
        case 2:
            a0_tensors[i].GenerateTensorValue(GeneratorTensor_3<A0DataType>{0.0, 1.0});
            a1_tensors[i].GenerateTensorValue(GeneratorTensor_3<A1DataType>{0.0, 1.0});
            b_tensors[i].GenerateTensorValue(GeneratorTensor_3<B0DataType>{-0.5, 0.5});
            break;
        default:
            a0_tensors[i].GenerateTensorValue(GeneratorTensor_Sequential<A0DataType, 0>{});
            a1_tensors[i].GenerateTensorValue(GeneratorTensor_Sequential<A1DataType, 0>{});
            b_tensors[i].GenerateTensorValue(GeneratorTensor_Sequential<B0DataType, 1>{});
        }

        d0_tensors[i].GenerateTensorValue(GeneratorTensor_3<D0DataType>{-0.5, 0.5});
    }

    constexpr ck::index_t NumATensor = 2;
    constexpr ck::index_t NumBTensor = 1;
    constexpr ck::index_t NumDTensor = 1;

    using GroupedGemmKernelArgument = ck::tensor_operation::device::
        GroupedGemmMultiABDKernelArgument<NumATensor, NumBTensor, NumDTensor>;

    std::vector<GroupedGemmKernelArgument> grouped_gemm_kernel_args_;
    grouped_gemm_kernel_args_.reserve(group_count);

    for(int i = 0; i < group_count; i++)
    {
        a0_tensors_device.emplace_back(
            std::make_unique<DeviceMem>(sizeof(A0DataType) * sum_of_m * problem_size.Ks[i]));

        a1_tensors_device.emplace_back(
            std::make_unique<DeviceMem>(sizeof(A1DataType) * sum_of_m * problem_size.Ks[i]));

        b_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(B0DataType) * problem_size.Ns[i] * problem_size.Ks[i]));

        d0_tensors_device.emplace_back(
            std::make_unique<DeviceMem>(sizeof(D0DataType) * problem_size.Ns[i]));

        c_tensors_device.emplace_back(
            std::make_unique<DeviceMem>(sizeof(EDataType) * sum_of_m * problem_size.Ns[i]));

        a0_tensors_device[i]->ToDevice(a0_tensors[i].mData.data(),
                                       a0_tensors[i].mDesc.GetElementSpaceSize() *
                                           sizeof(A0DataType));

        a1_tensors_device[i]->ToDevice(a1_tensors[i].mData.data(),
                                       a1_tensors[i].mDesc.GetElementSpaceSize() *
                                           sizeof(A1DataType));
        b_tensors_device[i]->ToDevice(b_tensors[i].mData.data(),
                                      b_tensors[i].mDesc.GetElementSpaceSize() *
                                          sizeof(B0DataType));
        d0_tensors_device[i]->ToDevice(d0_tensors[i].mData.data());
        c_tensors_device[i]->SetZero();

        gemm_descs.push_back({sum_of_m,
                              problem_size.Ns[i],
                              problem_size.Ks[i],
                              {1, 1},
                              {problem_size.stride_Bs[i]},
                              {0},
                              1});

        grouped_gemm_kernel_args_.push_back(
            {std::array<const void*, NumATensor>{a0_tensors_device[i]->GetDeviceBuffer(),
                                                 a1_tensors_device[i]->GetDeviceBuffer()},
             std::array<const void*, NumBTensor>{b_tensors_device[i]->GetDeviceBuffer()},
             std::array<const void*, NumDTensor>{d0_tensors_device[i]->GetDeviceBuffer()},
             c_tensors_device[i]->GetDeviceBuffer(),
             problem_size.Ms[i],
             problem_size.Ns[i],
             problem_size.Ks[i],
             std::array<ck::index_t, NumATensor>{problem_size.stride_As[i],
                                                 problem_size.stride_As[i]},
             std::array<ck::index_t, NumBTensor>{problem_size.stride_Bs[i]},
             std::array<ck::index_t, NumDTensor>{0},
             problem_size.stride_Cs[i]});
    }

    constexpr float scale = 1.f;
    auto a_element_op     = AElementOp{Add{}, Scale{scale}, Scale{scale}};
    auto b_element_op     = BElementOp{};
    auto cde_element_op   = CDEElementOp{};

    auto gemm    = DeviceGemmInstance{};
    auto invoker = gemm.MakeInvoker();

    std::vector<std::array<const void*, NumATensor>> p_As = {};
    std::vector<std::array<const void*, NumBTensor>> p_Bs = {};
    std::vector<std::array<const void*, NumDTensor>> p_Ds = {};
    std::vector<void*> p_Cs                               = {};

    // do GEMM
    auto argument = gemm.MakeArgument(p_As, p_Bs, p_Ds, p_Cs, gemm_descs);

    if(!gemm.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    DeviceMem gemm_workspace_dev(gemm.GetWorkSpaceSize(&argument));
    gemm.SetWorkSpacePointer(&argument, gemm_workspace_dev.GetDeviceBuffer());

    DeviceMem gemm_kernel_args_dev(gemm.GetDeviceKernelArgSize(&argument));
    hip_check_error(hipMemcpy(gemm_kernel_args_dev.GetDeviceBuffer(),
                              grouped_gemm_kernel_args_.data(),
                              gemm.GetDeviceKernelArgSize(&argument),
                              hipMemcpyHostToDevice));

    gemm.SetDeviceKernelArgs(argument, gemm_kernel_args_dev.GetDeviceBuffer());
    gemm.SetKBatch(argument, config.k_batch);

    gemm.SetElementwiseOps(argument, a_element_op, b_element_op, cde_element_op);

    invoker.Run(argument, StreamConfig{nullptr, false});

    if(config.time_kernel)
    {
        float ave_time   = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});
        float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
        float gb_per_sec = num_btype / 1.E6 / ave_time;

        std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s, " << gemm.GetTypeString() << std::endl;
    }

    bool pass = true;
    if(config.do_verification)
    {

        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<A0DataType,
                                                                                B0DataType,
                                                                                EDataType,
                                                                                AccDataType,
                                                                                PassThrough,
                                                                                BElementOp,
                                                                                PassThrough>;

        for(std::size_t i = 0; i < gemm_descs.size(); i++)
        {
            for(int m = 0; m < problem_size.Ms[i]; ++m)
            {
                for(int k = 0; k < problem_size.Ks[i]; ++k)
                {
                    a_element_op(a0_tensors[i](m, k), a0_tensors[i](m, k), a1_tensors[i](m, k));
                }
            }

            c_tensors_device[i]->FromDevice(e_device_tensors[i].mData.data(),
                                            e_device_tensors[i].mDesc.GetElementSize() *
                                                sizeof(EDataType));

            auto ref_gemm    = ReferenceGemmInstance{};
            auto ref_invoker = ref_gemm.MakeInvoker();

            auto ref_argument = ref_gemm.MakeArgument(a0_tensors[i],
                                                      b_tensors[i],
                                                      e_host_tensors[i],
                                                      PassThrough{},
                                                      b_element_op,
                                                      PassThrough{});

            ref_invoker.Run(ref_argument);

            for(int m = 0; m < problem_size.Ms[i]; ++m)
            {
                for(int n = 0; n < problem_size.Ns[i]; ++n)
                {
                    cde_element_op(
                        e_host_tensors[i](m, n), e_host_tensors[i](m, n), d0_tensors[i](m, n));
                }
            }

            pass &= ck::utils::check_err(e_device_tensors[i], e_host_tensors[i]);
        }
    }

    return pass;
}

int main(int argc, char* argv[])
{
    ProblemSize problem_size;
    ExecutionConfig config;

    problem_size.group_count = 16;

    for(int i = 0; i < problem_size.group_count; i++)
    {
        problem_size.Ms.push_back(32 + rand() % 32);
        problem_size.Ns.push_back(1024);
        problem_size.Ks.push_back(512);

        problem_size.stride_As.push_back(problem_size.Ks[i]);
        problem_size.stride_Bs.push_back(problem_size.Ks[i]);
        problem_size.stride_Cs.push_back(problem_size.Ns[i]);
    }

    if(argc == 5)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
        config.k_batch         = std::stoi(argv[4]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=n0, 1=yes)\n");
        printf("arg4: k_batch (>0)\n");
        exit(0);
    }

    return !run_grouped_gemm(problem_size, config);
}
