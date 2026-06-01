// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>

#include "ck/utility/data_type.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_bilinear.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

using InDataType  = ck::half_t;
using WeiDataType = ck::half_t;
using OutDataType = ck::half_t;
// Use std tuple instead of ck tuple to avoid clang
// implicit instantiation of undefined template error.
using DDataTypes = std::tuple<ck::half_t>;

using InLayout    = ck::tensor_layout::convolution::NDHWGC;
using WeiLayout   = ck::tensor_layout::convolution::GKZYXC;
using OutLayout   = ck::tensor_layout::convolution::NDHWGK;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Bilinear    = ck::tensor_operation::element_wise::Bilinear;

static constexpr ck::index_t NumDimSpatial = 3;
static constexpr ck::index_t G             = 32;
static constexpr ck::index_t N             = 64; // batch size
static constexpr ck::index_t K             = 64; // output channel
static constexpr ck::index_t C             = 32; // input channel (per group)
static constexpr ck::index_t Z             = 3;  // filter D
static constexpr ck::index_t Y             = 3;  // filter H
static constexpr ck::index_t X             = 3;  // filter W
static constexpr ck::index_t Di            = 14; // input D
static constexpr ck::index_t Hi            = 14; // input H
static constexpr ck::index_t Wi            = 14; // input W
static constexpr ck::index_t Do            = 14; // output D
static constexpr ck::index_t Ho            = 14; // output H
static constexpr ck::index_t Wo            = 14; // output W

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

int execute_conv_fwd_bilinear()
{
    // We have NHWGC/GKYXC/NHWGK (x, weight, y) in memory space.
    // However, CK's API only accepts lengths and strides with order of GNCDHW/GKCZYX/GNKDHW.
    // Hence, we need to adjust the order of strides.
    std::array<ck::index_t, 6> in_lengths{G, N, C, Di, Hi, Wi};
    std::array<ck::index_t, 6> in_strides{
        C, Di * Hi * Wi * G * C, 1, Hi * Wi * G * C, Wi * G * C, G * C};
    std::array<ck::index_t, 6> wei_lengths{G, K, C, Z, Y, X};
    std::array<ck::index_t, 6> wei_strides{
        K * Z * Y * X * C, Z * Y * X * C, 1, Y * X * C, X * C, C};
    std::array<ck::index_t, 6> out_lengths{G, N, K, Do, Ho, Wo};
    std::array<ck::index_t, 6> out_strides{
        K, Do * Ho * Wo * G * K, 1, Ho * Wo * G * K, Wo * G * K, G * K};
    // Logical broadcast bias (we have to pass bias lengths in the same format as output - GNKDHW)
    std::array<ck::index_t, 6> bias_lengths{G, 1, K, 1, 1, 1};
    std::array<ck::index_t, 6> bias_strides{K, 0, 1, 0, 0, 0};

    std::array<ck::index_t, NumDimSpatial> filter_strides{1, 1, 1};
    std::array<ck::index_t, NumDimSpatial> filter_dilations{1, 1, 1};
    std::array<ck::index_t, NumDimSpatial> input_left_pads{1, 1, 1};
    std::array<ck::index_t, NumDimSpatial> input_right_pads{1, 1, 1};

    SimpleDeviceMem in(sizeof(InDataType) * N * Di * Hi * Wi * G * C);
    SimpleDeviceMem wei(sizeof(WeiDataType) * G * K * Z * Y * X * C);
    SimpleDeviceMem out(sizeof(OutDataType) * N * Do * Ho * Wo * G * K);

    using DeviceOp =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<NumDimSpatial,
                                                                      InLayout,
                                                                      WeiLayout,
                                                                      ck::Tuple<OutLayout>,
                                                                      OutLayout,
                                                                      InDataType,
                                                                      WeiDataType,
                                                                      ck::Tuple<OutDataType>,
                                                                      OutDataType,
                                                                      PassThrough,
                                                                      PassThrough,
                                                                      Bilinear>;

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
        auto& op_ptr        = op_ptrs[i];
        auto argument_ptr   = op_ptr->MakeArgumentPointer(in.GetDeviceBuffer(),
                                                        wei.GetDeviceBuffer(),
                                                          {out.GetDeviceBuffer()},
                                                        out.GetDeviceBuffer(),
                                                        in_lengths,
                                                        in_strides,
                                                        wei_lengths,
                                                        wei_strides,
                                                          {out_lengths},
                                                          {out_strides},
                                                        out_lengths,
                                                        out_strides,
                                                        filter_strides,
                                                        filter_dilations,
                                                        input_left_pads,
                                                        input_right_pads,
                                                        PassThrough{},
                                                        PassThrough{},
                                                        Bilinear{2.f, 2.f});
        auto invoker_ptr    = op_ptr->MakeInvokerPointer();
        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            std::size_t flop =
                std::size_t(2) * G * N * K * C * Ho * Wo * Y * X + 3 * N * Ho * Wo * G * K;
            std::size_t num_bytes = sizeof(InDataType) * N * Hi * Wi * G * C +
                                    sizeof(WeiDataType) * G * K * Y * X * C +
                                    sizeof(OutDataType) * 2 * N * Ho * Wo * G * K;

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
        return EXIT_FAILURE;
    }

    std::cout << "Best Perf: " << std::setw(10) << best_avg_time << " ms, " << best_tflops
              << " TFlops, " << best_gb_per_sec << " GB/s, " << best_op_name << std::endl;

    // run the best intance
    {
        auto& op_ptr = op_ptrs[best_op_id];
        std::cout << "Run the best instance without timing: " << op_ptr->GetTypeString()
                  << std::endl;
        auto argument_ptr = op_ptr->MakeArgumentPointer(in.GetDeviceBuffer(),
                                                        wei.GetDeviceBuffer(),
                                                        {out.GetDeviceBuffer()},
                                                        out.GetDeviceBuffer(),
                                                        in_lengths,
                                                        in_strides,
                                                        wei_lengths,
                                                        wei_strides,
                                                        {out_lengths},
                                                        {out_strides},
                                                        out_lengths,
                                                        out_strides,
                                                        filter_strides,
                                                        filter_dilations,
                                                        input_left_pads,
                                                        input_right_pads,
                                                        PassThrough{},
                                                        PassThrough{},
                                                        Bilinear{2.f, 2.f});

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }

        std::cout << "Done" << std::endl;
    }
    return 0;
}

int main() { return execute_conv_fwd_bilinear(); }
