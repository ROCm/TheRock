// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_elementwise_dynamic_vector_dims_impl.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_reduce_multiblock.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"

#include "ck/library/reference_tensor_operation/cpu/reference_elementwise.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/utility/reduction_enums.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

using F16 = ck::half_t;
using F32 = float;
using F8  = ck::f8_t;

using InputDataType  = F16;
using ScaleDataType  = F32;
using OutputDataType = F8;

static constexpr ck::index_t NumDim = 2;

constexpr ck::ReduceTensorOp ReduceOpId = ck::ReduceTensorOp::MAX;
constexpr bool PropagateNan             = true;
constexpr bool OutputIndex              = false;

using ReduceOperation = typename ck::reduce_binary_operator<ReduceOpId>::opType;

struct ScalePassThrough
{
    ScalePassThrough(const float alpha = 1.f) : alpha_(alpha) {}

    __host__ __device__ constexpr void
    operator()(OutputDataType& y0, OutputDataType& y1, const InputDataType& x0) const
    {
        y0 = ck::type_convert<OutputDataType>(ck::type_convert<ScaleDataType>(x0) * alpha_);
        y1 = y0;
    }

    const ScaleDataType alpha_;
};

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using UnaryAbs    = ck::tensor_operation::element_wise::UnaryAbs;

using DeviceElementwisePermuteInstance = ck::tensor_operation::device::DeviceElementwiseImpl<
    ck::Tuple<InputDataType>,                  // InDataTypeTuple
    ck::Tuple<OutputDataType, OutputDataType>, // OutDataTypeTuple
    ScalePassThrough,                          // Elementwise
    NumDim,                                    // NumDim
    256,                                       // BlockSize
    128,                                       // M0PerBlock
    128,                                       // M1PerBlock
    8,                                         // M0PerThread
    8,                                         // M1PerThread
    ck::Sequence<1, 0>,                        // ThreadClusterArrangeOrder
    ck::Sequence<8>,                           // InScalarPerVectorSeq
    ck::Sequence<8, 1>>;                       // OutScalarPerVectorSeq

using DeviceReduceInstance =
    ck::tensor_operation::device::DeviceReduceMultiBlock<OutputDataType,
                                                         ScaleDataType,
                                                         OutputDataType,
                                                         NumDim,
                                                         NumDim,
                                                         ReduceOperation,
                                                         UnaryAbs,
                                                         PassThrough,
                                                         ck::InMemoryDataOperationEnum::Set,
                                                         PropagateNan,
                                                         OutputIndex,
                                                         false, // HaveIndexInputIfOutputIndex
                                                         1024,  // BlockSize
                                                         1,     // MThreadClusterSize
                                                         1024,  // KThreadClusterSize
                                                         1,     // MThreadSliceSize
                                                         16,    // KThreadSliceSize
                                                         1,     // InSrcVectorDim
                                                         16,    // InSrceVectorSize
                                                         1>;    // OutDstVectorSize

void reference_scale_permute_amax(Tensor<InputDataType>& input,
                                  Tensor<OutputDataType>& host_output_scaled_casted_transposed,
                                  Tensor<OutputDataType>& host_output_scaled_casted,
                                  Tensor<OutputDataType>& host_output_amax,
                                  const float scale)
{
    ScalePassThrough out_element_op(scale);
    const ck::index_t M = input.GetLengths()[0];
    const ck::index_t K = input.GetLengths()[1];
    for(ck::index_t m = 0; m < M; m++)
    {
        for(ck::index_t k = 0; k < K; k++)
        {
            OutputDataType y0, y1;
            out_element_op(y0, y1, input(m, k));

            host_output_scaled_casted(m, k)            = y0;
            host_output_scaled_casted_transposed(m, k) = y1;
            const OutputDataType y_fabs =
                ck::type_convert<OutputDataType>(ck::math::abs(ck::type_convert<float>(y0)));
            host_output_amax(0) = ck::type_convert<OutputDataType>(ck::math::max(
                ck::type_convert<float>(y_fabs), ck::type_convert<float>(host_output_amax(0))));
        }
    }
}

int main(int argc, char* argv[])
{
    bool do_verification = true;
    bool time_kernel     = false;

    const float scale = 2.f;

    ck::index_t M = 1024;
    ck::index_t K = 1024;

    if(argc == 1)
    {
        // use default
    }
    else if(argc == 3)
    {
        do_verification = std::stoi(argv[1]);
        time_kernel     = std::stoi(argv[2]);
    }
    else if(argc == 5)
    {
        do_verification = std::stoi(argv[1]);
        time_kernel     = std::stoi(argv[2]);
        M               = std::stoi(argv[3]);
        K               = std::stoi(argv[4]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: time kernel (0=no, 1=yes)\n");
        printf("arg3-4: M(default=1024), K(default=1024)\n");
        exit(1);
    }

    std::array<ck::index_t, 2> dims        = {M, K};
    std::array<ck::index_t, 2> in_strides  = {K, 1};
    std::array<ck::index_t, 2> out_strides = {1, M};

    Tensor<InputDataType> input(dims, in_strides);
    Tensor<OutputDataType> output_scaled_casted_transposed(dims, out_strides);
    Tensor<OutputDataType> output_scaled_casted(dims, in_strides);
    Tensor<OutputDataType> output_amax({1});

    input.GenerateTensorValue(GeneratorTensor_3<InputDataType>{0.0, 1.0});

    DeviceMem input_dev_buf(sizeof(InputDataType) * input.mDesc.GetElementSpaceSize());
    DeviceMem output_scaled_casted_transposed_dev_buf(
        sizeof(OutputDataType) * output_scaled_casted_transposed.mDesc.GetElementSpaceSize());
    DeviceMem output_scaled_casted_dev_buf(sizeof(OutputDataType) *
                                           output_scaled_casted.mDesc.GetElementSpaceSize());
    DeviceMem output_amax_dev_buf(sizeof(OutputDataType) * output_amax.mDesc.GetElementSpaceSize());

    input_dev_buf.ToDevice(input.mData.data());

    std::array<const void*, 1> inputs = {input_dev_buf.GetDeviceBuffer()};
    std::array<void*, 2> outputs      = {output_scaled_casted_transposed_dev_buf.GetDeviceBuffer(),
                                         output_scaled_casted_dev_buf.GetDeviceBuffer()};

    std::cout << "Input: " << input.mDesc << std::endl;
    std::cout << "Scale: " << scale << std::endl;
    std::cout << "Output scaled casted transposed: " << output_scaled_casted_transposed.mDesc
              << std::endl;
    std::cout << "Output scaled casted: " << output_scaled_casted.mDesc << std::endl;
    std::cout << "Output amax: " << output_amax.mDesc << std::endl;

    auto launch_transpose_scale = [&]() {
        auto transposeScale = DeviceElementwisePermuteInstance{};
        auto argument       = transposeScale.MakeArgumentPointer(dims,
                                                                 {in_strides},
                                                                 {out_strides, in_strides},
                                                           inputs,
                                                           outputs,
                                                           ScalePassThrough{scale});

        if(!transposeScale.IsSupportedArgument(argument.get()))
        {
            throw std::runtime_error(
                "The runtime parameters seems not supported by the device instance, exiting!");
        };

        auto transposeScale_invoker_ptr = transposeScale.MakeInvokerPointer();
        return transposeScale_invoker_ptr->Run(argument.get(), StreamConfig{nullptr, time_kernel});
    };

    auto launch_reduce = [&]() {
        auto reduce = DeviceReduceInstance{};
        auto reduce_argument_ptr =
            reduce.MakeArgumentPointer(dims,
                                       in_strides,
                                       {1},    // Output Lengths
                                       {1},    // Output Strides
                                       {0, 1}, // Reduce Dims
                                       static_cast<double>(1.f),
                                       static_cast<double>(0.f),
                                       output_scaled_casted_dev_buf.GetDeviceBuffer(),
                                       nullptr,
                                       output_amax_dev_buf.GetDeviceBuffer(),
                                       nullptr,
                                       UnaryAbs{},
                                       PassThrough{});

        if(!reduce.IsSupportedArgument(reduce_argument_ptr.get()))
        {
            throw std::runtime_error(
                "The runtime parameters seems not supported by the device instance, exiting!");
        };

        auto invoker_ptr = reduce.MakeInvokerPointer();

        return invoker_ptr->Run(reduce_argument_ptr.get(), StreamConfig{nullptr, time_kernel});
    };

    float ave_time = launch_transpose_scale();
    ave_time += launch_reduce();
    std::cout << "Perf: " << ave_time << " ms" << std::endl;
    bool pass = true;

    if(do_verification)
    {
        Tensor<OutputDataType> host_output_scaled_casted_transposed(dims, out_strides);
        Tensor<OutputDataType> host_output_scaled_casted(dims, in_strides);
        Tensor<OutputDataType> host_output_amax({1});

        reference_scale_permute_amax(input,
                                     host_output_scaled_casted_transposed,
                                     host_output_scaled_casted,
                                     host_output_amax,
                                     scale);

        output_scaled_casted_transposed_dev_buf.FromDevice(
            output_scaled_casted_transposed.mData.data());
        output_scaled_casted_dev_buf.FromDevice(output_scaled_casted.mData.data());
        output_amax_dev_buf.FromDevice(output_amax.mData.data());

        pass &= ck::utils::check_err(output_scaled_casted_transposed.mData,
                                     host_output_scaled_casted_transposed.mData,
                                     "Error: Incorrect results scaled transposed",
                                     1e-3,
                                     1e-3);
        pass &= ck::utils::check_err(output_scaled_casted.mData,
                                     host_output_scaled_casted.mData,
                                     "Error: Incorrect results scaled",
                                     1e-3,
                                     1e-3);
        pass &= ck::utils::check_err(
            output_amax.mData, host_output_amax.mData, "Error: Incorrect results amax", 1e-3, 1e-3);
    }

    return pass ? 0 : 1;
}
