// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Investigation-only contract check for the exact F32 registration entry point.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_data_multiple_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

using namespace ck::tensor_layout::convolution;
using Pass = ck::tensor_operation::element_wise::PassThrough;
using Op = ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD<
    2, NHWGK, GKYXC, ck::Tuple<>, NHWGC, float, float, ck::Tuple<>, float,
    Pass, Pass, Pass>;

namespace ck::tensor_operation::device::instance {
void add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f32_instances(
    std::vector<std::unique_ptr<Op>>& instances);
}

void hip_check(hipError_t status)
{
    if(status != hipSuccess)
        throw std::runtime_error(hipGetErrorString(status));
}

struct Buffer
{
    void* ptr = nullptr;
    explicit Buffer(std::size_t size) { hip_check(hipMalloc(&ptr, size)); }
    ~Buffer() { if(ptr) hipFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

// Logical axes: output G,N,K,H,W; weights G,K,C,Y,X; input G,N,C,H,W.
using Dims = std::array<ck::long_index_t, 5>;
using Spatial = std::array<ck::long_index_t, 2>;
std::size_t offset(const Dims& strides, const Dims& indices)
{
    std::size_t result = 0;
    for(std::size_t i = 0; i < indices.size(); ++i)
        result += strides[i] * indices[i];
    return result;
}

struct Case
{
    const char* name;
    int groups, batches, channels, outputs, height, width, filter, stride, pad;
};

std::pair<const char*, const char*> family(std::size_t index)
{
    if(index < 17) return {"generic", "Default"};
    if(index < 34) return {"generic", "Filter1x1Stride1Pad0"};
    if(index < 49) return {"noshuffle", "Default"};
    if(index < 64) return {"noshuffle", "Filter1x1Stride1Pad0"};
    if(index < 77) return {"nongrouped_match", "Default"};
    return {"nongrouped_match", "Filter1x1Stride1Pad0"};
}

bool check_case(const Case& shape, std::vector<std::unique_ptr<Op>>& instances,
                std::vector<unsigned>& executed)
{
    const auto [name, G, N, C, K, H, W, F, S, P] = shape;
    const int OH = (H + 2 * P - F) / S + 1;
    const int OW = (W + 2 * P - F) / S + 1;
    const Dims out_lengths{G, N, K, OH, OW};
    const Dims out_strides{K, OH * OW * G * K, 1, OW * G * K, G * K};
    const Dims wei_lengths{G, K, C, F, F};
    const Dims wei_strides{K * F * F * C, F * F * C, 1, F * C, C};
    const Dims in_lengths{G, N, C, H, W};
    const Dims in_strides{C, H * W * G * C, 1, W * G * C, G * C};
    const Spatial strides{S, S}, dilations{1, 1}, pads{P, P};
    std::vector<float> output(G * N * K * OH * OW), weight(G * K * C * F * F);
    std::vector<float> reference(G * N * C * H * W), actual(reference.size());
    for(std::size_t i = 0; i < output.size(); ++i)
        output[i] = static_cast<int>((i * 7 + 3) % 5) - 2;
    for(std::size_t i = 0; i < weight.size(); ++i)
        weight[i] = static_cast<int>((i * 3 + 1) % 5) - 2;

    // Independent scalar convolution-gradient reference, using double accumulation.
    // Integer-valued operands keep these bounded cases exactly representable in F32.
    for(int g = 0; g < G; ++g)
    for(int n = 0; n < N; ++n)
    for(int c = 0; c < C; ++c)
    for(int h = 0; h < H; ++h)
    for(int w = 0; w < W; ++w)
    {
        double total = 0;
        for(int y = 0; y < F; ++y)
        for(int x = 0; x < F; ++x)
        {
            const int hy = h + P - y, wx = w + P - x;
            if(hy < 0 || wx < 0 || hy % S || wx % S) continue;
            const int oh = hy / S, ow = wx / S;
            if(oh >= OH || ow >= OW) continue;
            for(int k = 0; k < K; ++k)
                total += static_cast<double>(output[offset(out_strides, {g,n,k,oh,ow})]) *
                         weight[offset(wei_strides, {g,k,c,y,x})];
        }
        reference[offset(in_strides, {g,n,c,h,w})] = static_cast<float>(total);
    }

    Buffer out_gpu(output.size() * sizeof(float));
    Buffer wei_gpu(weight.size() * sizeof(float));
    Buffer in_gpu(actual.size() * sizeof(float));
    hip_check(hipMemcpy(out_gpu.ptr, output.data(), output.size() * sizeof(float), hipMemcpyHostToDevice));
    hip_check(hipMemcpy(wei_gpu.ptr, weight.data(), weight.size() * sizeof(float), hipMemcpyHostToDevice));
    bool pass = true;
    for(std::size_t index = 0; index < instances.size(); ++index)
    {
        auto& op = instances[index];
        auto arg = op->MakeArgumentPointer(out_gpu.ptr, wei_gpu.ptr, {}, in_gpu.ptr,
            out_lengths, out_strides, wei_lengths, wei_strides, {}, {},
            in_lengths, in_strides, strides, dilations, pads, pads, Pass{}, Pass{}, Pass{}, 1);
        if(!op->IsSupportedArgument(arg.get()))
        {
            std::cout << "CASE\t" << name << '\t' << index << "\tUNSUPPORTED\n";
            continue;
        }
        const auto workspace_size = op->GetWorkSpaceSize(arg.get());
        std::unique_ptr<Buffer> workspace;
        if(workspace_size)
        {
            workspace = std::make_unique<Buffer>(workspace_size);
            op->SetWorkSpacePointer(arg.get(), workspace->ptr);
        }
        hip_check(hipMemset(in_gpu.ptr, 0, actual.size() * sizeof(float)));
        op->MakeInvokerPointer()->Run(arg.get(), StreamConfig{nullptr, false});
        hip_check(hipDeviceSynchronize());
        hip_check(hipMemcpy(actual.data(), in_gpu.ptr, actual.size() * sizeof(float), hipMemcpyDeviceToHost));
        ++executed[index];
        bool correct = true;
        double max_error = 0;
        for(std::size_t i = 0; i < actual.size(); ++i)
        {
            const double error = std::abs(static_cast<double>(actual[i]) - reference[i]);
            max_error = std::max(max_error, error);
            if(!std::isfinite(actual[i]) || error > 1e-4 * (1 + std::abs(reference[i])))
                correct = false;
        }
        pass &= correct;
        std::cout << "CASE\t" << name << '\t' << index << '\t'
                  << (correct ? "PASS" : "FAIL") << '\t' << max_error << '\n';
    }
    return pass;
}

int main(int argc, char** argv)
{
    try
    {
        std::vector<std::unique_ptr<Op>> instances;
        ck::tensor_operation::device::instance::
            add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f32_instances(instances);
        std::cout << "COUNT\t" << instances.size() << '\n';
        if(instances.size() != 90) return 2;
        for(std::size_t i = 0; i < instances.size(); ++i)
        {
            const auto [group, specialization] = family(i);
            // RTTI is the full template identity; GetTypeString can omit parameters.
            std::cout << "INSTANCE\t" << i << '\t' << group << '\t' << specialization
                      << '\t' << std::quoted(typeid(*instances[i]).name())
                      << '\t' << std::quoted(instances[i]->GetTypeString()) << '\n';
        }
        if(argc == 2 && std::string(argv[1]) == "--inventory") return 0;
        if(argc != 1) throw std::runtime_error("usage: ck_f32_contract [--inventory]");
        hipDeviceProp_t properties{};
        hip_check(hipGetDeviceProperties(&properties, 0));
        std::cout << "GPU\t" << properties.gcnArchName << '\n';
        const std::array<Case, 4> cases{{
            {"grouped_3x3_padded", 2, 2, 128, 64, 7, 7, 3, 1, 1},
            {"grouped_1x1", 2, 2, 128, 64, 7, 7, 1, 1, 0},
            {"nongrouped_1x1", 1, 2, 128, 64, 7, 7, 1, 1, 0},
            {"grouped_3x3_stride2", 2, 2, 128, 64, 7, 7, 3, 2, 1},
        }};
        std::vector<unsigned> executed(instances.size());
        bool pass = true;
        for(const auto& shape : cases) pass = check_case(shape, instances, executed) && pass;
        for(std::size_t i = 0; i < executed.size(); ++i)
        {
            std::cout << "EXECUTED\t" << i << '\t' << executed[i] << '\n';
            if(executed[i] == 0) pass = false; // No silent coverage holes.
        }
        std::cout << "RESULT\t" << (pass ? "PASS" : "FAIL") << '\n';
        return pass ? 0 : 1;
    }
    catch(const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 3;
    }
}
