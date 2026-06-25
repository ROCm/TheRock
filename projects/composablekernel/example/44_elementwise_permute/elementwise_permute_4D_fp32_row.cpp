// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/combined_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_elementwise_dynamic_vector_dims_impl.hpp"

#include "ck/library/reference_tensor_operation/cpu/reference_elementwise.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

using F16 = ck::half_t;
using F32 = float;

using ADataType = F32;
using BDataType = F32;

using NchwLayout = ck::tensor_layout::convolution::NCHW;
using NhwcLayout = ck::tensor_layout::convolution::NHWC;

using UnaryScale  = ck::tensor_operation::element_wise::Scale;
using UnarySquare = ck::tensor_operation::element_wise::UnarySquare;
using UnaryScaleSquare =
    ck::tensor_operation::element_wise::UnaryCombinedOp<UnarySquare, UnaryScale>;
using DeviceElementwisePermuteInstance = ck::tensor_operation::device::DeviceElementwiseImpl<
    ck::Tuple<ADataType>, // InDataTypeTuple
    ck::Tuple<BDataType>, // OutDataTypeTuple
    UnaryScaleSquare,     // UnaryScaleSquare
    4,                    // NumDim
    256,                  // BlockSize
    128,                  // M0PerBlock
    128,                  // M1PerBlock
    8,                    // M0PerThread
    8,                    // M1PerThread
    ck::Sequence<1, 0>,   // ThreadClusterArrangeOrder
    ck::Sequence<8>,      // InScalarPerVectorSeq
    ck::Sequence<8>>;     // OutScalarPerVectorSeq

int main(int argc, char* argv[])
{
    bool do_verification = true;
    bool time_kernel     = false;

    if(argc == 1)
    {
        // use default
    }
    else if(argc == 3)
    {
        do_verification = std::stoi(argv[1]);
        time_kernel     = std::stoi(argv[2]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: time kernel (0=no, 1=yes)\n");
        exit(0);
    }

    std::vector<std::size_t> nchw = {16, 128, 32, 64};
    std::vector<std::size_t> nhwc = {16, 32, 64, 128};

    std::array<ck::index_t, 4> ab_lengths;
    std::array<ck::index_t, 4> a_strides = {static_cast<int>(nchw[1] * nchw[2] * nchw[3]),
                                            static_cast<int>(nchw[2] * nchw[3]),
                                            static_cast<int>(nchw[3]),
                                            1};
    std::array<ck::index_t, 4> b_strides = {static_cast<int>(nhwc[1] * nhwc[2] * nhwc[3]),
                                            1,
                                            static_cast<int>(nhwc[2] * nhwc[3]),
                                            static_cast<int>(nhwc[3])};

    ck::ranges::copy(nchw, ab_lengths.begin());

    std::array<Tensor<ADataType>, 1> as = {Tensor<ADataType>(ab_lengths, a_strides, NchwLayout{})};
    Tensor<ADataType>& a                = as[0];
    Tensor<BDataType> b(ab_lengths, b_strides, NhwcLayout{});
    float scale = 2.f;
    a.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});

    DeviceMem a_device_buf(sizeof(ADataType) * a.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a.mData.data());

    std::array<const void*, 1> input = {a_device_buf.GetDeviceBuffer()};
    std::array<void*, 1> output      = {b_device_buf.GetDeviceBuffer()};

    auto broadcastPermute = DeviceElementwisePermuteInstance{};
    auto argument =
        broadcastPermute.MakeArgumentPointer(ab_lengths,
                                             {a_strides},
                                             {b_strides},
                                             input,
                                             output,
                                             UnaryScaleSquare{UnarySquare{}, UnaryScale{scale}});

    if(!broadcastPermute.IsSupportedArgument(argument.get()))
    {
        throw std::runtime_error(
            "The runtime parameters seems not supported by the device instance, exiting!");
    };

    std::cout << "A (nchw): " << a.mDesc << std::endl;
    std::cout << "B (nhwc): " << b.mDesc << std::endl;

    auto broadcastPermute_invoker_ptr = broadcastPermute.MakeInvokerPointer();
    float ave_time =
        broadcastPermute_invoker_ptr->Run(argument.get(), StreamConfig{nullptr, time_kernel});
    std::size_t flop = std::size_t(2) * nchw[0] * nchw[1] * nchw[2] * nchw[3];

    std::size_t num_btype = sizeof(ADataType) * (nchw[0] * nchw[1] * nchw[2] * nchw[3]) +
                            sizeof(BDataType) * (nchw[0] * nchw[1] * nchw[2] * nchw[3]);

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    bool pass = true;

    if(do_verification)
    {
        Tensor<BDataType> host_b(ab_lengths, b_strides, NhwcLayout{});
        using ReferenceElementwiseInstance = ck::tensor_operation::host::
            ReferenceElementwise<1, ADataType, BDataType, UnaryScaleSquare>;
        auto ref_elementwise = ReferenceElementwiseInstance{};
        auto ref_invoker     = ref_elementwise.MakeInvoker();

        auto ref_argument = ref_elementwise.MakeArgument(
            as, host_b, UnaryScaleSquare{UnarySquare{}, UnaryScale{scale}});
        ref_invoker.Run(ref_argument);

        b_device_buf.FromDevice(b.mData.data());
        pass &=
            ck::utils::check_err(b.mData, host_b.mData, "Error: Incorrect results b", 1e-3, 1e-3);
    }

    return pass ? 0 : 1;
}
