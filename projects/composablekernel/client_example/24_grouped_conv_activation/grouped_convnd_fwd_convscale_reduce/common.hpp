// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "ck/ck.hpp"
#include "ck/library/utility/algorithm.hpp"
#include "ck/tensor_operation/gpu/device/device_elementwise.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/utility/type.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_convscale_relu.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_convscale.hpp"
#include "ck/utility/reduction_enums.hpp"
#include "ck/library/tensor_operation_instance/gpu/permute_scale.hpp"
#include "ck/library/tensor_operation_instance/gpu/reduce/reduce.hpp"
#include "ck/library/utility/host_tensor.hpp"

using ::ck::HostTensorDescriptor;

using PassThrough   = ck::tensor_operation::element_wise::PassThrough;
using ConvScaleRelu = ck::tensor_operation::element_wise::ScaleScaleRelu;
using ConvScale     = ck::tensor_operation::element_wise::ScaleScalePass;

struct SimpleDeviceMem
{
    SimpleDeviceMem() = delete;

    SimpleDeviceMem(std::size_t mem_size) : p_mem_{}
    {
        (void)hipMalloc(static_cast<void**>(&p_mem_), mem_size);
    }

    void* GetDeviceBuffer() { return p_mem_; }

    ~SimpleDeviceMem() { (void)hipFree(p_mem_); }

    void* p_mem_;
};

template <ck::index_t NumDimSpatial, ck::index_t NumNonSpatialDim = 3>
std::size_t
GetFlops(const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& output_lengths,
         const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& weights_lengths,
         const std::size_t& ds_size)
{
    // 2 * G * N * K * C * <output spatial lengths product> * <filter spatial lengths product> +
    // + ds_size * <output tensor size> =>
    // => <output tensor size> * ( 2 * C * <filter spatial lengths product> + ds_size) =>
    // => G * N * K * <output spatial lengths product> * (2 * C * <filter spatial lengths product> +
    // ds_size)
    ck::index_t G = weights_lengths[0];
    ck::index_t N = output_lengths[1];
    ck::index_t K = weights_lengths[1];
    ck::index_t C = weights_lengths[2];

    return G * N * K *
           std::accumulate(std::next(std::begin(output_lengths), NumNonSpatialDim),
                           std::end(output_lengths),
                           static_cast<std::size_t>(1),
                           std::multiplies<>()) *
           (ds_size + static_cast<std::size_t>(2) * C *
                          std::accumulate(std::next(std::begin(weights_lengths), NumNonSpatialDim),
                                          std::end(weights_lengths),
                                          static_cast<std::size_t>(1),
                                          std::multiplies<>()));
}

template <ck::index_t NumDimSpatial, ck::index_t NumNonSpatialDim = 3>
std::size_t GetTensorSize(const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& lengths)
{

    return std::accumulate(std::begin(lengths),
                           std::end(lengths),
                           static_cast<std::size_t>(1),
                           std::multiplies<std::size_t>());
}

template <typename InDataType, ck::index_t NumDimSpatial, ck::index_t NumNonSpatialDim = 3>
std::size_t
GetInputByte(const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& input_lengths)
{
    // sizeof(InDataType) * (G * N * C * <input spatial lengths product>) +
    return sizeof(InDataType) * GetTensorSize<NumDimSpatial>(input_lengths);
}

template <typename WeiDataType, ck::index_t NumDimSpatial, ck::index_t NumNonSpatialDim = 3>
std::size_t
GetWeightByte(const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& weights_lengths)
{
    // sizeof(WeiDataType) * (G * K * C * <filter spatial lengths product>) +
    return sizeof(WeiDataType) * GetTensorSize<NumDimSpatial>(weights_lengths);
}

template <typename OutDataType, ck::index_t NumDimSpatial, ck::index_t NumNonSpatialDim = 3>
std::size_t
GetOutputByte(const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& output_lengths)
{
    // sizeof(OutDataType) * (G * N * K * <output spatial lengths product>);
    return sizeof(OutDataType) * GetTensorSize<NumDimSpatial>(output_lengths);
}

template <typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename ConvElementOp,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          ck::index_t NumDimSpatial,
          ck::index_t NumNonSpatialDim = 3,
          typename AComputeType        = InDataType,
          typename BComputeType        = AComputeType>
bool ConvolutionScale(SimpleDeviceMem& in,
                      SimpleDeviceMem& wei,
                      SimpleDeviceMem& out,
                      ConvElementOp elementwise_op,
                      const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& in_lengths,
                      const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& in_strides,
                      const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& wei_lengths,
                      const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& wei_strides,
                      const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& out_lengths,
                      const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& out_strides);

template <typename InDataType,
          typename OutDataType,
          ck::index_t NumDimSpatial,
          ck::index_t NumNonSpatialDim = 3>
bool TensorScaleConvert(SimpleDeviceMem& in,
                        SimpleDeviceMem& out,
                        float scale_out,
                        const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& lengths,
                        const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& strides);

template <typename InDataType,
          typename OutDataType,
          ck::ReduceTensorOp ReduceOpId,
          ck::index_t NumDimSpatial,
          ck::index_t NumNonSpatialDim = 3>
bool TensorFullReduction(SimpleDeviceMem& tensor,
                         SimpleDeviceMem& out_amax,
                         const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& lengths,
                         const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& strides);

template <ck::index_t NumDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename ConvOutDataType,
          typename OutDataType,
          typename ConvElementOp,
          ck::ReduceTensorOp ReduceOp,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          ck::index_t NumNonSpatialDim = 3,
          typename AComputeType        = InDataType,
          typename BComputeType        = AComputeType>
bool run_grouped_conv_fwd_convscale_reduce(
    std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim> in_lengths,
    std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim> wei_lengths,
    std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim> out_lengths)
{

    namespace ctc = ck::tensor_layout::convolution;
    static_assert(NumDimSpatial == 3 && ck::is_same_v<InLayout, ctc::NDHWGC> &&
                      ck::is_same_v<WeiLayout, ctc::GKZYXC> &&
                      ck::is_same_v<OutLayout, ctc::NDHWGK>,
                  "Unsupported configuration");

    const ck::index_t G  = in_lengths[4];
    const ck::index_t N  = in_lengths[0];
    const ck::index_t K  = wei_lengths[1];
    const ck::index_t C  = in_lengths[5];
    const ck::index_t Z  = wei_lengths[2];
    const ck::index_t Y  = wei_lengths[3];
    const ck::index_t X  = wei_lengths[4];
    const ck::index_t Di = in_lengths[1];
    const ck::index_t Hi = in_lengths[2];
    const ck::index_t Wi = in_lengths[3];
    const ck::index_t Do = out_lengths[1];
    const ck::index_t Ho = out_lengths[2];
    const ck::index_t Wo = out_lengths[3];

    const std::size_t in_mem_size       = sizeof(InDataType) * N * Di * Hi * Wi * G * C;
    const std::size_t wei_mem_size      = sizeof(WeiDataType) * G * K * Z * Y * X * C;
    const std::size_t conv_out_mem_size = sizeof(ConvOutDataType) * N * Do * Ho * Wo * G * K;
    const std::size_t out_mem_size      = sizeof(OutDataType) * N * Do * Ho * Wo * G * K;

    SimpleDeviceMem in(in_mem_size);
    SimpleDeviceMem wei(wei_mem_size);
    SimpleDeviceMem conv_out(conv_out_mem_size);
    SimpleDeviceMem out(out_mem_size);

    float scale_in  = float(std::rand()) / float(RAND_MAX);
    float scale_wei = float(std::rand()) / float(RAND_MAX);
    float scale_out = float(std::rand()) / float(RAND_MAX);

    // We have NDHWGC/GKZYXC/NDHWGK (x, weight, y) in memory space.
    // However, CK's API only accepts lengths and strides with order of GNCDHW/GKCZYX/GNKDHW.
    // Hence, we need to adjust the order of strides.
    const std::array<ck::index_t, NumDimSpatial + 3> input_lengths{G, N, C, Di, Hi, Wi};
    const std::array<ck::index_t, NumDimSpatial + 3> input_strides{
        C, Di * Hi * Wi * G * C, 1, Hi * Wi * G * C, Wi * G * C, G * C};
    const std::array<ck::index_t, NumDimSpatial + 3> weights_lengths{G, K, C, Z, Y, X};
    const std::array<ck::index_t, NumDimSpatial + 3> weights_strides{
        K * Z * Y * X * C, Z * Y * X * C, 1, Y * X * C, X * C, C};
    const std::array<ck::index_t, NumDimSpatial + 3> output_lengths{G, N, K, Do, Ho, Wo};
    const std::array<ck::index_t, NumDimSpatial + 3> output_strides{
        K, Do * Ho * Wo * G * K, 1, Ho * Wo * G * K, Wo * G * K, G * K};

    /*
     * FP8 Convolution with Scaling
     */
    std::cout << "\n\nConvolution with scale Benchmarking:" << std::endl;
    auto elementwise_op = ConvElementOp{ck::tensor_operation::element_wise::Scale{scale_in},
                                        ck::tensor_operation::element_wise::Scale{scale_wei},
                                        {}};
    auto conv_ok        = ConvolutionScale<InDataType,
                                           WeiDataType,
                                           ConvOutDataType,
                                           ConvElementOp,
                                           InLayout,
                                           WeiLayout,
                                           OutLayout,
                                           NumDimSpatial>(in,
                                                   wei,
                                                   conv_out,
                                                   elementwise_op,
                                                   input_lengths,
                                                   input_strides,
                                                   weights_lengths,
                                                   weights_strides,
                                                   output_lengths,
                                                   output_strides);

    if(!conv_ok)
        return false;

    /*
     *  Scale with output weight and convert to FP8
     */
    std::cout << "\n\nElement-wise scale + convert Benchmarking:" << std::endl;
    auto elem_wise_ok = TensorScaleConvert<ConvOutDataType, OutDataType, NumDimSpatial>(
        conv_out, out, scale_out, output_lengths, output_strides);

    if(!elem_wise_ok)
        return false;

    /*
     *  Compute AMAX
     */
    std::cout << "\n\nAMAX Benchmarking:" << std::endl;
    SimpleDeviceMem amax_device(sizeof(ConvOutDataType));
    auto reduction_ok =
        TensorFullReduction<ConvOutDataType,
                            ConvOutDataType,
                            ck::ReduceTensorOp::AMAX,
                            NumDimSpatial>(conv_out, amax_device, output_lengths, output_strides);

    if(!reduction_ok)
        return false;

    return true;
}

template <typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename ConvElementOp,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          ck::index_t NumDimSpatial,
          ck::index_t NumNonSpatialDim,
          typename AComputeType,
          typename BComputeType>
bool ConvolutionScale(SimpleDeviceMem& in,
                      SimpleDeviceMem& wei,
                      SimpleDeviceMem& out,
                      ConvElementOp elementwise_op,
                      const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& in_lengths,
                      const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& in_strides,
                      const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& wei_lengths,
                      const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& wei_strides,
                      const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& out_lengths,
                      const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& out_strides)
{

    const std::array<ck::index_t, NumDimSpatial> conv_filter_strides{1, 1, 1};
    const std::array<ck::index_t, NumDimSpatial> conv_filter_dilations{1, 1, 1};
    const std::array<ck::index_t, NumDimSpatial> input_left_pads{1, 1, 1};
    const std::array<ck::index_t, NumDimSpatial> input_right_pads{1, 1, 1};

    const auto in_mem_size  = GetInputByte<InDataType, NumDimSpatial>(in_lengths);
    const auto wei_mem_size = GetWeightByte<WeiDataType, NumDimSpatial>(wei_lengths);
    const auto out_mem_size = GetOutputByte<OutDataType, NumDimSpatial>(out_lengths);

    std::size_t ds_size = 2; // 2 element-wise scale multipliers
    if constexpr(ck::is_same_v<ConvElementOp, ConvScaleRelu>)
    {
        ds_size += 1; // +1 element-wise relu
    }
    std::size_t flop = GetFlops<NumDimSpatial>(out_lengths, wei_lengths, ds_size);
    std::size_t num_bytes =
        in_mem_size + wei_mem_size + sizeof(float) + sizeof(float) + out_mem_size;

    using ConvDeviceOp =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<NumDimSpatial,
                                                                      InLayout,
                                                                      WeiLayout,
                                                                      ck::Tuple<>,
                                                                      OutLayout,
                                                                      InDataType,
                                                                      WeiDataType,
                                                                      ck::Tuple<>,
                                                                      OutDataType,
                                                                      PassThrough,
                                                                      PassThrough,
                                                                      ConvElementOp,
                                                                      AComputeType,
                                                                      BComputeType>;
    // get device op instances
    const auto conv_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        ConvDeviceOp>::GetInstances();

    std::cout << "found " << conv_ptrs.size() << " instances" << std::endl;

    std::string conv_best_op_name;
    int conv_best_op_id        = -1;
    float conv_best_avg_time   = std::numeric_limits<float>::max();
    float conv_best_gb_per_sec = 0;
    float conv_best_tflops     = 0;

    // profile device operation instances
    std::cout << "Run all convolution instances and do timing" << std::endl;

    for(int i = 0; i < conv_ptrs.size(); ++i)
    {
        auto& op_ptr      = conv_ptrs[i];
        auto argument_ptr = op_ptr->MakeArgumentPointer(
            in.GetDeviceBuffer(),
            wei.GetDeviceBuffer(),
            std::array<const void*, 0>{},
            out.GetDeviceBuffer(),
            in_lengths,
            in_strides,
            wei_lengths,
            wei_strides,
            std::array<std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>, 0>{},
            std::array<std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>, 0>{},
            out_lengths,
            out_strides,
            conv_filter_strides,
            conv_filter_dilations,
            input_left_pads,
            input_right_pads,
            PassThrough{},
            PassThrough{},
            elementwise_op);

        auto invoker_ptr    = op_ptr->MakeInvokerPointer();
        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
            float gb_per_sec = num_bytes / 1.E6 / avg_time;

            std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << op_name << std::endl;

            if(tflops > conv_best_tflops)
            {
                conv_best_op_id      = i;
                conv_best_op_name    = op_name;
                conv_best_avg_time   = avg_time;
                conv_best_gb_per_sec = gb_per_sec;
                conv_best_tflops     = tflops;
            }
        }
        else
        {
            std::cerr << op_name << " does not support this problem" << std::endl;
        }
    }

    if(conv_best_op_id < 0)
    {
        std::cerr << "no suitable instance" << std::endl;
        return false;
    }

    std::cout << "Best Perf: " << std::setw(10) << conv_best_avg_time << " ms, " << conv_best_tflops
              << " TFlops, " << conv_best_gb_per_sec << " GB/s, " << conv_best_op_name << std::endl;

    // run the best instance
    {
        auto& op_ptr = conv_ptrs[conv_best_op_id];
        std::cout << "Run the best instance without timing: " << op_ptr->GetTypeString()
                  << std::endl;
        auto argument_ptr = op_ptr->MakeArgumentPointer(
            in.GetDeviceBuffer(),
            wei.GetDeviceBuffer(),
            std::array<const void*, 0>{},
            out.GetDeviceBuffer(),
            in_lengths,
            in_strides,
            wei_lengths,
            wei_strides,
            std::array<std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>, 0>{},
            std::array<std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>, 0>{},
            out_lengths,
            out_strides,
            conv_filter_strides,
            conv_filter_dilations,
            input_left_pads,
            input_right_pads,
            PassThrough{},
            PassThrough{},
            elementwise_op);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }

        std::cout << "Done" << std::endl;
    }

    return true;
}

template <typename InDataType,
          typename OutDataType,
          ck::index_t NumDimSpatial,
          ck::index_t NumNonSpatialDim>
bool TensorScaleConvert(SimpleDeviceMem& in,
                        SimpleDeviceMem& out,
                        float scale_out,
                        const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& lengths,
                        const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& strides)
{

    const auto tensor_size = GetTensorSize<NumDimSpatial>(lengths);

    const std::size_t in_mem_size  = sizeof(InDataType) * tensor_size;
    const std::size_t out_mem_size = sizeof(OutDataType) * tensor_size;

    std::size_t flop = 2 * tensor_size; // element-wise scale + convert

    std::size_t bytes =
        in_mem_size + sizeof(float) + out_mem_size; // read from in, scale, write to out

    using DeviceScaleConvert =
        ck::tensor_operation::device::DeviceElementwise<ck::Tuple<InDataType>,
                                                        ck::Tuple<OutDataType>,
                                                        ck::tensor_operation::element_wise::Scale,
                                                        NumDimSpatial + NumNonSpatialDim>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceScaleConvert>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    std::string best_op_name;
    int best_op_id        = -1;
    float best_avg_time   = std::numeric_limits<float>::max();
    float best_gb_per_sec = 0;
    float best_tflops     = 0;

    // profile device operation instances
    std::cout << "Run all DeviceScaleConvert instances and do timing" << std::endl;

    auto scale_convert = ck::tensor_operation::element_wise::Scale{scale_out};

    for(int i = 0; i < op_ptrs.size(); ++i)
    {
        auto& op_ptr      = op_ptrs[i];
        auto argument_ptr = op_ptr->MakeArgumentPointer(lengths,
                                                        {strides},
                                                        {strides},
                                                        {in.GetDeviceBuffer()},
                                                        {out.GetDeviceBuffer()},
                                                        scale_convert);

        auto invoker_ptr    = op_ptr->MakeInvokerPointer();
        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
            float gb_per_sec = bytes / 1.E6 / avg_time;

            std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << op_name << std::endl;

            if(tflops > best_tflops)
            {
                best_op_id      = i;
                best_op_name    = op_name;
                best_avg_time   = avg_time;
                best_gb_per_sec = gb_per_sec;
                best_tflops     = tflops;
            }
        }
        else
        {
            std::cerr << op_name << " does not support this problem" << std::endl;
        }
    }

    if(best_op_id < 0)
    {
        std::cerr << "no suitable instance found." << std::endl;
        return false;
    }
    else
    {
        std::cout << "Best Perf: " << std::setw(10) << best_avg_time << " ms, " << best_tflops
                  << " TFlops, " << best_gb_per_sec << " GB/s, " << best_op_name << std::endl;

        // run the best intance
        auto& op_ptr = op_ptrs[best_op_id];
        std::cout << "Run the best instance without timing: " << op_ptr->GetTypeString()
                  << std::endl;
        auto argument_ptr = op_ptr->MakeArgumentPointer(lengths,
                                                        {strides},
                                                        {strides},
                                                        {in.GetDeviceBuffer()},
                                                        {out.GetDeviceBuffer()},
                                                        scale_convert);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }

        std::cout << "Done" << std::endl;
    }

    return true;
}

template <typename InDataType,
          typename OutDataType,
          ck::ReduceTensorOp ReduceOpId,
          ck::index_t NumDimSpatial,
          ck::index_t NumNonSpatialDim>
bool TensorFullReduction(SimpleDeviceMem& tensor,
                         SimpleDeviceMem& out_amax,
                         const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& lengths,
                         const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& strides)
{
    const auto spatial_dim_size = std::accumulate(std::next(std::begin(lengths), NumNonSpatialDim),
                                                  std::end(lengths),
                                                  static_cast<std::size_t>(1),
                                                  std::multiplies<>());
    const auto tensor_size      = GetTensorSize<NumDimSpatial>(lengths);

    auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    // Get the reduction operation
    using ReduceOperation = typename ck::reduce_binary_operator<ReduceOpId>::opType;
    using InElementwiseOperation =
        typename ck::reduce_unary_operator<ReduceOpId, true, true>::InElementwiseOperation;
    using AccElementwiseOperation =
        typename ck::reduce_unary_operator<ReduceOpId, true, true>::AccElementwiseOperation;

    InElementwiseOperation in_elementwise_op;
    AccElementwiseOperation acc_elementwise_op;
    std::tie(in_elementwise_op, acc_elementwise_op) =
        ck::reduce_unary_operator<ReduceOpId, true, true>::GetElementwiseOperator(
            static_cast<int32_t>(tensor_size));

    std::array<ck::index_t, 1> reduce_out_lengths{1};
    std::array<ck::index_t, 1> reduce_out_strides{1};

    SimpleDeviceMem partial_reduce_tensor(sizeof(OutDataType) * spatial_dim_size);
    std::array<ck::index_t, NumDimSpatial> reduce_part_lengths;
    std::copy(std::next(std::begin(lengths), NumNonSpatialDim),
              std::end(lengths),
              std::begin(reduce_part_lengths));
    std::array<ck::index_t, NumDimSpatial> reduce_part_strides;
    copy(HostTensorDescriptor(reduce_part_lengths).GetStrides(), reduce_part_strides);

    {
        std::cout << "\nReduction of nonspatial dimensions:" << std::endl;
        using DeviceOp =
            ck::tensor_operation::device::DeviceReduce<InDataType,
                                                       OutDataType,
                                                       OutDataType,
                                                       NumDimSpatial + NumNonSpatialDim,
                                                       NumNonSpatialDim,
                                                       ReduceOperation,
                                                       InElementwiseOperation,
                                                       PassThrough,
                                                       true,   // PropagateNan
                                                       false>; // OutputIndex
        const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
            DeviceOp>::GetInstances();

        std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

        std::string best_op_name;
        int best_op_id        = -1;
        float best_ave_time   = std::numeric_limits<float>::max();
        float best_gb_per_sec = 0;

        std::array<int, NumNonSpatialDim> reduce_dims;
        std::iota(reduce_dims.begin(), reduce_dims.end(), 0); // 0,..., NumNonSpatialDim-1

        ck::index_t num_in_elements  = tensor_size;
        ck::index_t num_out_elements = spatial_dim_size;

        // profile device operation instances
        std::cout << "Run partial reduction and do timing" << std::endl;

        for(int i = 0; i < op_ptrs.size(); ++i)
        {
            auto& op_ptr = op_ptrs[i];

            auto argument_ptr   = op_ptr->MakeArgumentPointer(lengths,
                                                            strides,
                                                            reduce_part_lengths,
                                                            reduce_part_strides,
                                                            reduce_dims,
                                                            1.0,
                                                            0.0,
                                                            tensor.GetDeviceBuffer(),
                                                            nullptr,
                                                            partial_reduce_tensor.GetDeviceBuffer(),
                                                            nullptr,
                                                            in_elementwise_op,
                                                            PassThrough{});
            auto invoker_ptr    = op_ptr->MakeInvokerPointer();
            std::string op_name = op_ptr->GetTypeString();

            if(op_ptr->IsSupportedArgument(argument_ptr.get()))
            {
                float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});
                std::size_t num_bytes =
                    num_in_elements * sizeof(InDataType) + num_out_elements * sizeof(OutDataType);

                float gb_per_sec = num_bytes / 1.E6 / ave_time;

                std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << gb_per_sec
                          << " GB/s, " << op_name << std::endl;

                if(ave_time < best_ave_time)
                {
                    best_op_id      = i;
                    best_op_name    = op_name;
                    best_ave_time   = ave_time;
                    best_gb_per_sec = gb_per_sec;
                }
            }
            else
            {
                std::cout << op_name << " does not support this problem" << std::endl;
            }
        }

        if(best_op_id < 0)
        {
            std::cerr << "no suitable instance found." << std::endl;
            return false;
        }
        else
        {
            std::cout << "Best Perf: " << best_ave_time << " ms, " << best_gb_per_sec << " GB/s, "
                      << best_op_name << std::endl;

            // run the best instance
            auto& op_ptr = op_ptrs[best_op_id];
            std::cout << "Run the best instance without timing: " << op_ptr->GetTypeString()
                      << std::endl;
            auto argument_ptr = op_ptr->MakeArgumentPointer(lengths,
                                                            strides,
                                                            reduce_part_lengths,
                                                            reduce_part_strides,
                                                            reduce_dims,
                                                            1.0,
                                                            0.0,
                                                            tensor.GetDeviceBuffer(),
                                                            nullptr,
                                                            partial_reduce_tensor.GetDeviceBuffer(),
                                                            nullptr,
                                                            in_elementwise_op,
                                                            PassThrough{});

            auto invoker_ptr = op_ptr->MakeInvokerPointer();

            if(op_ptr->IsSupportedArgument(argument_ptr.get()))
            {
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
            }

            std::cout << "Done" << std::endl;
        }
    }

    {
        std::cout << "\nReduction of spatial dimensions:" << std::endl;
        using DeviceOp     = ck::tensor_operation::device::DeviceReduce<OutDataType,
                                                                        OutDataType,
                                                                        OutDataType,
                                                                        NumDimSpatial,
                                                                        NumDimSpatial,
                                                                        ReduceOperation,
                                                                        PassThrough,
                                                                        AccElementwiseOperation,
                                                                        true,   // PropagateNan
                                                                        false>; // OutputIndex
        const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
            DeviceOp>::GetInstances();

        std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

        std::string best_op_name;
        int best_op_id        = -1;
        float best_ave_time   = std::numeric_limits<float>::max();
        float best_gb_per_sec = 0;

        std::array<int, NumDimSpatial> reduce_dims;
        std::iota(reduce_dims.begin(), reduce_dims.end(), 0); // 0,..., NumDimSpatial-1

        ck::index_t num_in_elements  = spatial_dim_size;
        ck::index_t num_out_elements = 1;

        // profile device operation instances
        std::cout << "Run final reduction and do timing" << std::endl;

        for(int i = 0; i < op_ptrs.size(); ++i)
        {
            auto& op_ptr = op_ptrs[i];

            auto argument_ptr   = op_ptr->MakeArgumentPointer(reduce_part_lengths,
                                                            reduce_part_strides,
                                                            reduce_out_lengths,
                                                            reduce_out_strides,
                                                            reduce_dims,
                                                            1.0,
                                                            0.0,
                                                            partial_reduce_tensor.GetDeviceBuffer(),
                                                            nullptr,
                                                            out_amax.GetDeviceBuffer(),
                                                            nullptr,
                                                            PassThrough{},
                                                            acc_elementwise_op);
            auto invoker_ptr    = op_ptr->MakeInvokerPointer();
            std::string op_name = op_ptr->GetTypeString();

            if(op_ptr->IsSupportedArgument(argument_ptr.get()))
            {
                float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

                std::size_t num_bytes =
                    num_in_elements * sizeof(OutDataType) + num_out_elements * sizeof(OutDataType);

                float gb_per_sec = num_bytes / 1.E6 / ave_time;

                std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << gb_per_sec
                          << " GB/s, " << op_name << std::endl;

                if(ave_time < best_ave_time)
                {
                    best_op_id      = i;
                    best_op_name    = op_name;
                    best_ave_time   = ave_time;
                    best_gb_per_sec = gb_per_sec;
                }
            }
            else
            {
                std::cout << op_name << " does not support this problem" << std::endl;
            }
        }

        if(best_op_id < 0)
        {
            std::cerr << "no suitable instance found." << std::endl;
            return false;
        }
        else
        {
            std::cout << "Best Perf: " << best_ave_time << " ms, " << best_gb_per_sec << " GB/s, "
                      << best_op_name << std::endl;

            // run the best instance
            auto& op_ptr = op_ptrs[best_op_id];
            std::cout << "Run the best instance without timing: " << op_ptr->GetTypeString()
                      << std::endl;
            auto argument_ptr = op_ptr->MakeArgumentPointer(reduce_part_lengths,
                                                            reduce_part_strides,
                                                            reduce_out_lengths,
                                                            reduce_out_strides,
                                                            reduce_dims,
                                                            1.0,
                                                            0.0,
                                                            partial_reduce_tensor.GetDeviceBuffer(),
                                                            nullptr,
                                                            out_amax.GetDeviceBuffer(),
                                                            nullptr,
                                                            PassThrough{},
                                                            acc_elementwise_op);

            auto invoker_ptr = op_ptr->MakeInvokerPointer();

            if(op_ptr->IsSupportedArgument(argument_ptr.get()))
            {
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
            }

            std::cout << "Done" << std::endl;
        }
    }

    return true;
}
