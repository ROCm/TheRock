// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iomanip>
#include <iostream>
#include <vector>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/quantization/grouped_convolution_bias_forward_perchannel_quantization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_conv_fwd.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

using InDataType           = int8_t;
using WeiDataType          = int8_t;
using BiasDataType         = int32_t;
using RequantScaleDataType = float;
using OutDataType          = int8_t;

using InLayout           = ck::tensor_layout::convolution::NHWGC;
using WeiLayout          = ck::tensor_layout::convolution::GKYXC;
using BiasLayout         = ck::tensor_layout::convolution::G_K;
using RequantScaleLayout = ck::tensor_layout::convolution::G_K;
using OutLayout          = ck::tensor_layout::convolution::NHWGK;
using PassThrough        = ck::tensor_operation::element_wise::PassThrough;
using ActivationOp       = ck::tensor_operation::element_wise::TanH;
using OutElementOp =
    ck::tensor_operation::element_wise::Add_Mul2_Activation_Mul_Clamp<ActivationOp>;

static constexpr ck::index_t NumDimSpatial = 2;
static constexpr ck::index_t G             = 4;
static constexpr ck::index_t N             = 4;    // batch size
static constexpr ck::index_t K             = 32;   // output channel
static constexpr ck::index_t C             = 64;   // input channel (per group)
static constexpr ck::index_t Y             = 3;    // filter H
static constexpr ck::index_t X             = 3;    // filter W
static constexpr ck::index_t Hi            = 71;   // input H
static constexpr ck::index_t Wi            = 71;   // input W
static constexpr ck::index_t Ho            = 36;   // output H
static constexpr ck::index_t Wo            = 36;   // output W
static constexpr float sz_inv              = 0.5f; // inverse of scale_z

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

int main(int argc, char* argv[])
{
    // We have NHWGC/GKYXC/NHWGK (x, weight, y) in memory space
    // However, CK's API only accept length and stride with order of GNCHW/GKCYX/GNCHW
    // Hence, we need to adjust the order of stride
    std::array<ck::index_t, 5> in_lengths{G, N, C, Hi, Wi};
    std::array<ck::index_t, 5> in_strides{C, Hi * Wi * G * C, 1, Wi * G * C, G * C};
    std::array<ck::index_t, 5> weight_lengths{G, K, C, Y, X};
    std::array<ck::index_t, 5> weight_strides{K * Y * X * C, Y * X * C, 1, X * C, C};
    std::array<ck::index_t, 5> bias_lengths{G, N, K, Ho, Wo};
    std::array<ck::index_t, 5> bias_strides{K, 0, 1, 0, 0};
    std::array<ck::index_t, 5> requant_scale_lengths{G, N, K, Ho, Wo};
    std::array<ck::index_t, 5> requant_scale_strides{K, 0, 1, 0, 0};
    std::array<ck::index_t, 5> out_lengths{G, N, K, Ho, Wo};
    std::array<ck::index_t, 5> out_strides{C, Ho * Wo * G * C, 1, Wo * G * C, G * C};

    std::array<ck::index_t, 2> in_left_pad{1, 1};
    std::array<ck::index_t, 2> in_right_pad{1, 1};
    std::array<ck::index_t, 2> conv_strides{2, 2};
    std::array<ck::index_t, 2> conv_dilations{1, 1};

    SimpleDeviceMem in(sizeof(InDataType) * N * Hi * Wi * G * C);
    SimpleDeviceMem wei(sizeof(WeiDataType) * G * K * Y * X * C);
    SimpleDeviceMem bias(sizeof(BiasDataType) * G * K);
    SimpleDeviceMem requant_scale(sizeof(RequantScaleDataType) * G * K);
    SimpleDeviceMem out(sizeof(OutDataType) * N * Ho * Wo * G * K);

    using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
        NumDimSpatial,
        InLayout,
        WeiLayout,
        ck::Tuple<BiasLayout, RequantScaleLayout>,
        OutLayout,
        InDataType,
        WeiDataType,
        ck::Tuple<BiasDataType, RequantScaleDataType>,
        OutDataType,
        PassThrough,
        PassThrough,
        OutElementOp>;
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
        auto& op_ptr = op_ptrs[i];
        auto argument_ptr =
            op_ptr->MakeArgumentPointer(in.GetDeviceBuffer(),
                                        wei.GetDeviceBuffer(),
                                        {bias.GetDeviceBuffer(), requant_scale.GetDeviceBuffer()},
                                        out.GetDeviceBuffer(),
                                        in_lengths,
                                        in_strides,
                                        weight_lengths,
                                        weight_strides,
                                        {bias_lengths, requant_scale_lengths},
                                        {bias_strides, requant_scale_strides},
                                        out_lengths,
                                        out_strides,
                                        conv_strides,
                                        conv_dilations,
                                        in_left_pad,
                                        in_right_pad,
                                        PassThrough{},
                                        PassThrough{},
                                        OutElementOp{sz_inv, ActivationOp{}});

        auto invoker_ptr    = op_ptr->MakeInvokerPointer();
        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            std::size_t flop = G * 2 * N * K * C * Ho * Wo * Y * X;
            std::size_t num_bytes =
                G * sizeof(InDataType) * N * Hi * Wi * C + G * sizeof(WeiDataType) * K * Y * X * C +
                G * sizeof(BiasDataType) * K + G * sizeof(RequantScaleDataType) * K +
                G * sizeof(OutDataType) * N * Ho * Wo * K;

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
            std::cout << op_name << " does not support this problem" << std::endl;
        }
    }

    // run the best intance
    if(best_op_id != -1)
    {
        std::cout << "Best Perf: " << std::setw(10) << best_avg_time << " ms, " << best_tflops
                  << " TFlops, " << best_gb_per_sec << " GB/s, " << best_op_name << std::endl;

        auto& op_ptr = op_ptrs[best_op_id];
        std::cout << "Run the best instance without timing: " << op_ptr->GetTypeString()
                  << std::endl;
        auto argument_ptr =
            op_ptr->MakeArgumentPointer(in.GetDeviceBuffer(),
                                        wei.GetDeviceBuffer(),
                                        {bias.GetDeviceBuffer(), requant_scale.GetDeviceBuffer()},
                                        out.GetDeviceBuffer(),
                                        in_lengths,
                                        in_strides,
                                        weight_lengths,
                                        weight_strides,
                                        {bias_lengths, requant_scale_lengths},
                                        {bias_strides, requant_scale_strides},
                                        out_lengths,
                                        out_strides,
                                        conv_strides,
                                        conv_dilations,
                                        in_left_pad,
                                        in_right_pad,
                                        PassThrough{},
                                        PassThrough{},
                                        OutElementOp{sz_inv, ActivationOp{}});

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }

        std::cout << "Done" << std::endl;
    }

    return 0;
}
