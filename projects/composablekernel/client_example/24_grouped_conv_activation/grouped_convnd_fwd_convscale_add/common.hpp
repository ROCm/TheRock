// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "ck/ck.hpp"
#include "ck/utility/type.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_convscale_add.hpp"

using PassThrough  = ck::tensor_operation::element_wise::PassThrough;
using ConvScaleAdd = ck::tensor_operation::element_wise::ConvScaleAdd;
using F32          = float;
using BiasDataType = F32;

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
    // G * N * C * <output spatial lengths product> * (2 * K * <filter spatial lengths product> +
    // <number of scale factors>)
    ck::index_t G = weights_lengths[0];
    ck::index_t N = output_lengths[1];
    ck::index_t K = weights_lengths[1];
    ck::index_t C = weights_lengths[2];

    return G * N * C *
           std::accumulate(std::next(std::begin(output_lengths), NumNonSpatialDim),
                           std::end(output_lengths),
                           static_cast<std::size_t>(1),
                           std::multiplies<>()) *
           (static_cast<std::size_t>(2) * K *
                std::accumulate(std::next(std::begin(weights_lengths), NumNonSpatialDim),
                                std::end(weights_lengths),
                                static_cast<std::size_t>(1),
                                std::multiplies<>()) +
            ds_size);
}

template <typename InDataType, ck::index_t NumDimSpatial, ck::index_t NumNonSpatialDim = 3>
std::size_t
GetInputByte(const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& input_lengths)
{
    // sizeof(InDataType) * (G * N * C * <input spatial lengths product>) +
    return sizeof(InDataType) * std::accumulate(std::begin(input_lengths),
                                                std::end(input_lengths),
                                                static_cast<std::size_t>(1),
                                                std::multiplies<>());
}

template <typename WeiDataType, ck::index_t NumDimSpatial, ck::index_t NumNonSpatialDim = 3>
std::size_t
GetWeightByte(const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& weights_lengths)
{
    // sizeof(WeiDataType) * (G * K * C * <filter spatial lengths product>) +
    return sizeof(WeiDataType) * std::accumulate(std::begin(weights_lengths),
                                                 std::end(weights_lengths),
                                                 static_cast<std::size_t>(1),
                                                 std::multiplies<>());
}

template <typename OutDataType, ck::index_t NumDimSpatial, ck::index_t NumNonSpatialDim = 3>
std::size_t
GetOutputByte(const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& output_lengths)
{
    // sizeof(OutDataType) * (G * N * K * <output spatial lengths product>);
    return sizeof(OutDataType) * std::accumulate(std::begin(output_lengths),
                                                 std::end(output_lengths),
                                                 static_cast<std::size_t>(1),
                                                 std::multiplies<std::size_t>());
}

template <ck::index_t NumDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          ck::index_t NumNonSpatialDim = 3,
          typename AComputeType        = InDataType,
          typename BComputeType        = AComputeType>
bool run_grouped_conv_fwd_convscale_add(
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

    const std::size_t in_mem_size   = sizeof(InDataType) * N * Di * Hi * Wi * G * C;
    const std::size_t wei_mem_size  = sizeof(WeiDataType) * G * K * Z * Y * X * C;
    const std::size_t out_mem_size  = sizeof(OutDataType) * N * Do * Ho * Wo * G * K;
    const std::size_t bias_mem_size = sizeof(BiasDataType) * N * Do * Ho * Wo * G * K;

    SimpleDeviceMem in(in_mem_size);
    SimpleDeviceMem wei(wei_mem_size);
    SimpleDeviceMem out(out_mem_size);
    SimpleDeviceMem bias(bias_mem_size);

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
    const std::array<ck::index_t, NumDimSpatial + 3> bias_lengths{G, N, K, Do, Ho, Wo};
    const std::array<ck::index_t, NumDimSpatial + 3> bias_strides{
        K, Do * Ho * Wo * G * K, 1, Ho * Wo * G * K, Wo * G * K, G * K};
    const std::array<ck::index_t, NumDimSpatial> conv_filter_strides{1, 1, 1};
    const std::array<ck::index_t, NumDimSpatial> conv_filter_dilations{1, 1, 1};
    const std::array<ck::index_t, NumDimSpatial> input_left_pads{1, 1, 1};
    const std::array<ck::index_t, NumDimSpatial> input_right_pads{1, 1, 1};

    std::size_t ds_size   = 3 + 1; // 3 element-wise scale multipliers + 1 elementwise Bias
    std::size_t flop      = GetFlops<NumDimSpatial>(output_lengths, weights_lengths, ds_size);
    std::size_t num_bytes = in_mem_size + wei_mem_size + sizeof(float) + sizeof(float) +
                            sizeof(float) + out_mem_size + bias_mem_size;

    using DeviceOp =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<NumDimSpatial,
                                                                      InLayout,
                                                                      WeiLayout,
                                                                      ck::Tuple<OutLayout>,
                                                                      OutLayout,
                                                                      InDataType,
                                                                      WeiDataType,
                                                                      ck::Tuple<BiasDataType>,
                                                                      OutDataType,
                                                                      PassThrough,
                                                                      PassThrough,
                                                                      ConvScaleAdd,
                                                                      AComputeType,
                                                                      BComputeType>;
    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    std::string best_op_name;
    int best_op_id        = -1;
    float best_avg_time   = std::numeric_limits<float>::max();
    float best_gb_per_sec = 0;
    float best_tflops     = 0;

    // profile device operation instances
    std::cout << "Run all instances and do timing" << std::endl;

    for(int i = 0; i < op_ptrs.size(); ++i)
    {
        auto& op_ptr      = op_ptrs[i];
        auto argument_ptr = op_ptr->MakeArgumentPointer(
            in.GetDeviceBuffer(),
            wei.GetDeviceBuffer(),
            std::array<const void*, 1>{bias.GetDeviceBuffer()},
            out.GetDeviceBuffer(),
            input_lengths,
            input_strides,
            weights_lengths,
            weights_strides,
            std::array<std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>, 1>{
                {bias_lengths}},
            std::array<std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>, 1>{
                {bias_strides}},
            output_lengths,
            output_strides,
            conv_filter_strides,
            conv_filter_dilations,
            input_left_pads,
            input_right_pads,
            PassThrough{},
            PassThrough{},
            ConvScaleAdd{scale_in, scale_wei, scale_out});

        auto invoker_ptr    = op_ptr->MakeInvokerPointer();
        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
            float gb_per_sec = num_bytes / 1.E6 / avg_time;

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
        std::cerr << "no suitable instance" << std::endl;
        return false;
    }

    std::cout << "Best Perf: " << std::setw(10) << best_avg_time << " ms, " << best_tflops
              << " TFlops, " << best_gb_per_sec << " GB/s, " << best_op_name << std::endl;

    // run the best intance
    {
        auto& op_ptr = op_ptrs[best_op_id];
        std::cout << "Run the best instance without timing: " << op_ptr->GetTypeString()
                  << std::endl;
        auto argument_ptr = op_ptr->MakeArgumentPointer(
            in.GetDeviceBuffer(),
            wei.GetDeviceBuffer(),
            std::array<const void*, 1>{bias.GetDeviceBuffer()},
            out.GetDeviceBuffer(),
            input_lengths,
            input_strides,
            weights_lengths,
            weights_strides,
            std::array<std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>, 1>{
                {bias_lengths}},
            std::array<std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>, 1>{
                {bias_strides}},
            output_lengths,
            output_strides,
            conv_filter_strides,
            conv_filter_dilations,
            input_left_pads,
            input_right_pads,
            PassThrough{},
            PassThrough{},
            ConvScaleAdd{scale_in, scale_wei, scale_out});

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }

        std::cout << "Done" << std::endl;
    }
    return true;
}
