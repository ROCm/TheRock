// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>

#include "ck/ck.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_reduce.hpp"
#include "ck/tensor_operation/gpu/element/combined_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_elementwise_dynamic_vector_dims_impl.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_reduce_multiblock.hpp"
#include "ck/utility/reduction_operator.hpp"
#include "ck/utility/reduction_enums.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"
#include "ck/utility/type.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

namespace ew = ck::tensor_operation::element_wise;

using PassThrough   = ew::PassThrough;
using ConvScaleRelu = ew::UnaryCombinedOp<ew::Scale, ew::Scale, ew::Relu>;
using ConvScale     = ew::UnaryCombinedOp<ew::Scale, ew::Scale, PassThrough>;

using UnaryScaleConvert = ew::Scale;

void print_helper_msg()
{
    std::cout << "arg1: verification (0=no, 1=yes)\n"
              << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg3: time kernel (0=no, 1=yes)\n"
              << ck::utils::conv::get_conv_param_parser_helper_msg() << std::endl;
}

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

template <ck::index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename ConvOutDataType,
          typename OutDataType,
          typename InElementOp,
          typename WeiElementOp,
          typename ConvElementOp,
          typename DeviceConvNDFwdInstance>
bool run_grouped_conv_fwd(bool do_verification,
                          int init_method,
                          bool time_kernel,
                          const ck::utils::conv::ConvParam& conv_param,
                          const HostTensorDescriptor& in_g_n_c_wis_desc,
                          const HostTensorDescriptor& wei_g_k_c_xs_desc,
                          const HostTensorDescriptor& out_g_n_k_wos_desc,
                          const InElementOp& in_element_op,
                          const WeiElementOp& wei_element_op)
{
    Tensor<InDataType> in(in_g_n_c_wis_desc);
    Tensor<WeiDataType> wei(wei_g_k_c_xs_desc);
    Tensor<ConvOutDataType> host_conv(out_g_n_k_wos_desc);
    Tensor<ConvOutDataType> device_conv(out_g_n_k_wos_desc);
    Tensor<OutDataType> out_host(out_g_n_k_wos_desc);
    Tensor<OutDataType> out_device(out_g_n_k_wos_desc);

    std::cout << "in: " << in.mDesc << std::endl;
    std::cout << "wei: " << wei.mDesc << std::endl;
    std::cout << "out: " << out_host.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        wei.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        break;
    case 11: // used for debugging
        in.GenerateTensorValue(GeneratorTensor_1<InDataType>{1});
        wei.GenerateTensorValue(GeneratorTensor_1<WeiDataType>{1});
        break;
    default:
        in.GenerateTensorValue(GeneratorTensor_3<InDataType>{-1.0, 1.0});
        wei.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei.mDesc.GetElementSpaceSize());
    DeviceMem conv_device_buf(conv_param.GetOutputByte<ConvOutDataType>());
    DeviceMem out_device_buf(conv_param.GetOutputByte<OutDataType>());

    in_device_buf.ToDevice(in.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());

    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_strides{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_strides{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input_left_pads{};
    std::array<ck::index_t, NDimSpatial> input_right_pads{};

    auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(in_g_n_c_wis_desc.GetLengths(), a_g_n_c_wis_lengths);
    copy(in_g_n_c_wis_desc.GetStrides(), a_g_n_c_wis_strides);
    copy(wei_g_k_c_xs_desc.GetLengths(), b_g_k_c_xs_lengths);
    copy(wei_g_k_c_xs_desc.GetStrides(), b_g_k_c_xs_strides);
    copy(out_g_n_k_wos_desc.GetLengths(), e_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), e_g_n_k_wos_strides);
    copy(conv_param.conv_filter_strides_, conv_filter_strides);
    copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_param.input_left_pads_, input_left_pads);
    copy(conv_param.input_right_pads_, input_right_pads);

    // random scale values
    float scale_in  = float(std::rand()) / float(RAND_MAX);
    float scale_wei = float(std::rand()) / float(RAND_MAX);
    float scale_out = float(std::rand()) / float(RAND_MAX);

    std::cout << std::endl;
    std::cout << "scale_in: " << scale_in << std::endl;
    std::cout << "scale_wei: " << scale_wei << std::endl;
    std::cout << "scale_out: " << scale_out << std::endl;

    // convolution elementwise operation
    auto conv_element_op = ConvElementOp{ew::Scale{scale_in}, ew::Scale{scale_wei}, {}};
    auto scale_convert   = UnaryScaleConvert{scale_out}; // elementwise scale and type cast

    // do Conv
    auto conv         = DeviceConvNDFwdInstance{};
    auto conv_invoker = conv.MakeInvoker();
    auto conv_argument =
        conv.MakeArgument(in_device_buf.GetDeviceBuffer(),
                          wei_device_buf.GetDeviceBuffer(),
                          std::array<const void*, 0>{},
                          conv_device_buf.GetDeviceBuffer(),
                          a_g_n_c_wis_lengths,
                          a_g_n_c_wis_strides,
                          b_g_k_c_xs_lengths,
                          b_g_k_c_xs_strides,
                          std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                          std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                          e_g_n_k_wos_lengths,
                          e_g_n_k_wos_strides,
                          conv_filter_strides,
                          conv_filter_dilations,
                          input_left_pads,
                          input_right_pads,
                          in_element_op,
                          wei_element_op,
                          conv_element_op);

    if(!conv.IsSupportedArgument(conv_argument))
    {
        throw std::runtime_error(
            "wrong! device_conv with the specified compilation parameters does "
            "not support this Conv problem");
    }

    std::string kernels = conv.GetTypeString();

    float avg_time = conv_invoker.Run(conv_argument, StreamConfig{nullptr, time_kernel});

    using DeviceElementwiseScale = ck::tensor_operation::device::DeviceElementwiseImpl<
        ck::Tuple<ConvOutDataType>, // InDataTypeTuple
        ck::Tuple<OutDataType>,     // OutDataTypeTuple
        UnaryScaleConvert,          // UnaryScaleConvert
        NDimSpatial + 3,            // NumDim
        256,                        // BlockSize
        128,                        // M0PerBlock
        128,                        // M1PerBlock
        8,                          // M0PerThread
        8,                          // M1PerThread
        ck::Sequence<1, 0>,         // ThreadClusterArrangeOrder
        ck::Sequence<8>,            // InScalarPerVectorSeq
        ck::Sequence<8>>;           // OutScalarPerVectorSeq

    auto device_ew_scale = DeviceElementwiseScale{};
    auto scale_invoker   = device_ew_scale.MakeInvoker();
    auto scale_argument  = device_ew_scale.MakeArgument(e_g_n_k_wos_lengths,
                                                        {e_g_n_k_wos_strides},
                                                        {e_g_n_k_wos_strides},
                                                        {conv_device_buf.GetDeviceBuffer()},
                                                        {out_device_buf.GetDeviceBuffer()},
                                                       scale_convert);

    if(!device_ew_scale.IsSupportedArgument(scale_argument))
    {
        throw std::runtime_error(
            "wrong! DeviceElementwiseScale with the specified compilation parameters does "
            "not support this problem");
    }

    kernels += std::string("\n\t\t ") + device_ew_scale.GetTypeString();

    avg_time += scale_invoker.Run(scale_argument, StreamConfig{nullptr, time_kernel});

    constexpr auto ReduceOpId = ck::ReduceTensorOp::AMAX;
    using ReduceOperation     = typename ck::reduce_binary_operator<ReduceOpId>::opType;
    using InElementwiseOperation =
        typename ck::reduce_unary_operator<ReduceOpId, true, true>::InElementwiseOperation;
    using AccElementwiseOperation =
        typename ck::reduce_unary_operator<ReduceOpId, true, true>::AccElementwiseOperation;
    using DeviceReduceInstance =
        ck::tensor_operation::device::DeviceReduceMultiBlock<ConvOutDataType,
                                                             ConvOutDataType,
                                                             ConvOutDataType,
                                                             NDimSpatial + 3,
                                                             NDimSpatial + 3,
                                                             ReduceOperation,
                                                             InElementwiseOperation,
                                                             AccElementwiseOperation,
                                                             ck::InMemoryDataOperationEnum::Set,
                                                             true,  // PropagateNan
                                                             false, // OutputIndex
                                                             false, // HaveIndexInputIfOutputIndex
                                                             256,   // BlockSize
                                                             4,     // MThreadClusterSize
                                                             64,    // KThreadClusterSize
                                                             1,     // MThreadSliceSize
                                                             1,     // KThreadSliceSize
                                                             1,     // InSrcVectorDim
                                                             1,     // InSrceVectorSize
                                                             1>;    // OutDstVectorSize

    std::vector<size_t> outLengths = {1};
    Tensor<ConvOutDataType> amax_host(outLengths);
    Tensor<ConvOutDataType> amax_from_device(outLengths);
    auto amax_host_strides = amax_host.mDesc.GetStrides();

    std::array<int, NDimSpatial + 3> reduce_dims;
    std::iota(reduce_dims.begin(), reduce_dims.end(), 0); // 0,..., NDimSpatial+3-1

    std::array<ck::index_t, 1> reduce_out_lengths{1};
    std::array<ck::index_t, 1> reduce_out_strides{static_cast<ck::index_t>(amax_host_strides[0])};

    DeviceMem amax_device(sizeof(ConvOutDataType) * amax_host.mDesc.GetElementSpaceSize());
    DeviceMem index_device;

    InElementwiseOperation in_elementwise_op;
    AccElementwiseOperation acc_elementwise_op;
    std::tie(in_elementwise_op, acc_elementwise_op) =
        ck::reduce_unary_operator<ReduceOpId, true, true>::GetElementwiseOperator(
            static_cast<int32_t>(host_conv.mDesc.GetElementSize()));

    // Hack convolution output strides for reduction as kernel expects stride 1 for the last
    // dimension. It only works because the reduction is done on the whole tensor and result is
    // independent of the order of elements.
    std::array<ck::index_t, NDimSpatial + 3> reduction_strides{};
    copy(HostTensorDescriptor(e_g_n_k_wos_lengths).GetStrides(), reduction_strides);

    auto device_reduce   = DeviceReduceInstance{};
    auto reduce_invoker  = device_reduce.MakeInvokerPointer();
    auto reduce_argument = device_reduce.MakeArgumentPointer(e_g_n_k_wos_lengths,
                                                             reduction_strides,
                                                             reduce_out_lengths,
                                                             reduce_out_strides,
                                                             reduce_dims,
                                                             1.0,
                                                             0.0,
                                                             conv_device_buf.GetDeviceBuffer(),
                                                             nullptr,
                                                             amax_device.GetDeviceBuffer(),
                                                             nullptr,
                                                             in_elementwise_op,
                                                             acc_elementwise_op);

    if(!device_reduce.IsSupportedArgument(reduce_argument.get()))
    {
        throw std::runtime_error(
            "wrong! DeviceReduceInstance with the specified compilation parameters does "
            "not support this runtime parameters!");
    };

    kernels += std::string("\n\t\t ") + device_reduce.GetTypeString();

    float reduce_time =
        reduce_invoker->Run(reduce_argument.get(), StreamConfig{nullptr, time_kernel});

    if(time_kernel)
        std::cout << "\nReduce time: " << reduce_time << " ms" << std::endl;

    avg_time += reduce_time;

    std::size_t flop    = conv_param.GetFlops();      // convolution FLOPs
    auto conv_out_elems = host_conv.GetElementSize(); // number of elements in conv result tensor

    // 3 element-wise scale multipliers + 1 AMAX
    std::size_t elementwise_ops = 3 + 1;
    if constexpr(ck::is_same_v<ConvElementOp, ConvScaleRelu>)
    {
        elementwise_ops += 1; // +1 element-wise relu
    }

    flop += elementwise_ops * conv_out_elems;

    // convolution + elementwise scaling (in + wei + output byte count)
    std::size_t num_btype = conv_param.GetByte<InDataType, WeiDataType, ConvOutDataType>();
    num_btype += sizeof(float) + sizeof(float); //  + 2 scales

    // elementwise scaling + F8 conversion
    num_btype += conv_param.GetOutputByte<ConvOutDataType>() + sizeof(float) +
                 conv_param.GetOutputByte<OutDataType>();

    // AMAX
    num_btype += conv_param.GetOutputByte<ConvOutDataType>() + sizeof(float);

    if(time_kernel)
    {
        float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
        float gb_per_sec = num_btype / 1.E6 / avg_time;
        std::cout << "Perf: " << avg_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s, " << std::endl;
    }

    std::cout << "\nKernels: " << kernels << std::endl;

    if(do_verification)
    {
        auto ref_conv = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                     InDataType,
                                                                     WeiDataType,
                                                                     ConvOutDataType,
                                                                     InElementOp,
                                                                     WeiElementOp,
                                                                     ConvElementOp>();

        auto ref_invoker  = ref_conv.MakeInvoker();
        auto ref_argument = ref_conv.MakeArgument(in,
                                                  wei,
                                                  host_conv,
                                                  conv_param.conv_filter_strides_,
                                                  conv_param.conv_filter_dilations_,
                                                  conv_param.input_left_pads_,
                                                  conv_param.input_right_pads_,
                                                  in_element_op,
                                                  wei_element_op,
                                                  conv_element_op);

        ref_invoker.Run(ref_argument);

        conv_device_buf.FromDevice(device_conv.mData.data());

        out_device_buf.FromDevice(out_device.mData.data());

        out_host.ForEach([&](auto&, auto idx) { scale_convert(out_host(idx), host_conv(idx)); });

        std::cout << "\nComparing output to reference: " << std::endl;
        auto tight_tol_check = ck::utils::check_err(out_device, out_host, "Error: ");
        if(!tight_tol_check)
        {
            std::cout << "\n\tRecompare applying tolerances...\n";
            std::cout << "\t\trtol = " << get_rtol<OutDataType>() << std::endl;
            std::cout << "\t\tatol = " << get_atol<OutDataType>() << std::endl;
            auto loose_tol_check = ck::utils::check_err(out_device,
                                                        out_host,
                                                        "Error: incorrect convolution results!",
                                                        get_rtol<OutDataType>(),
                                                        get_atol<OutDataType>());
            if(!loose_tol_check)
            {
                return false;
            }
        }
        std::cout << "Success!" << std::endl;

        /// Verify AMAX

        using RefReduceInstance =
            ck::tensor_operation::host::ReferenceReduce<ConvOutDataType,
                                                        ConvOutDataType,
                                                        ConvOutDataType,
                                                        NDimSpatial + 3,
                                                        NDimSpatial + 3,
                                                        ReduceOperation,
                                                        InElementwiseOperation,
                                                        AccElementwiseOperation,
                                                        true,
                                                        false>;

        auto ref_reduce          = RefReduceInstance{};
        auto ref_reduce_invoker  = ref_reduce.MakeInvokerPointer();
        auto ref_reduce_argument = ref_reduce.MakeArgumentPointer(e_g_n_k_wos_lengths,
                                                                  e_g_n_k_wos_strides,
                                                                  reduce_out_lengths,
                                                                  reduce_out_strides,
                                                                  reduce_dims,
                                                                  1.0,
                                                                  0.0,
                                                                  host_conv.mData.data(),
                                                                  nullptr,
                                                                  amax_host.mData.data(),
                                                                  nullptr,
                                                                  in_elementwise_op,
                                                                  acc_elementwise_op);

        if(!ref_reduce.IsSupportedArgument(ref_reduce_argument.get()))
        {
            throw std::runtime_error(
                "wrong! RefReduceInstance with the specified compilation parameters does "
                "not support this runtime parameters!");
        };

        ref_reduce_invoker->Run(ref_reduce_argument.get());

        amax_device.FromDevice(amax_from_device.mData.data());

        std::cout << "\namax: " << amax_from_device.mData[0] << std::endl;
        std::cout << "amax_ref: " << amax_host.mData[0] << std::endl;

        return ck::utils::check_err(amax_from_device, amax_host, "Error: incorrect AMAX results!");
    }

    return true;
}
