// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iomanip>
#include <vector>
#include <iostream>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_normalization_fwd.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/normalization_fwd.hpp"

using XDataType              = ck::half_t;
using GammaDataType          = ck::half_t;
using BetaDataType           = ck::half_t;
using YDataType              = ck::half_t;
using SaveMeanInvStdDataType = ck::half_t;
using PassThrough            = ck::tensor_operation::element_wise::PassThrough;

#define SAVE_MEAN_INV_STD

constexpr int Rank         = 4;
constexpr int NumReduceDim = 3;

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
    ck::index_t N = 256;
    ck::index_t H = 16;
    ck::index_t W = 16;
    ck::index_t C = 8;

    std::vector<ck::index_t> strideXY             = {H * W * C, W * C, C, 1};
    std::vector<ck::index_t> strideGammaBeta      = {0, W * C, C, 1};
    std::vector<ck::index_t> strideSaveMeanInvStd = {1};

    SimpleDeviceMem x_device_buf(sizeof(XDataType) * N * H * W * C);
    SimpleDeviceMem gamma_device_buf(sizeof(GammaDataType) * H * W * C);
    SimpleDeviceMem beta_device_buf(sizeof(BetaDataType) * H * W * C);
    SimpleDeviceMem y_device_buf(sizeof(YDataType) * N * H * W * C);
#ifdef SAVE_MEAN_INV_STD
    SimpleDeviceMem save_mean_device_buf(sizeof(SaveMeanInvStdDataType) * N);
    SimpleDeviceMem save_inv_std_device_buf(sizeof(SaveMeanInvStdDataType) * N);
#endif

    using DeviceOp = ck::tensor_operation::device::DeviceNormalizationFwd<XDataType,
                                                                          GammaDataType,
                                                                          BetaDataType,
                                                                          YDataType,
                                                                          SaveMeanInvStdDataType,
                                                                          PassThrough,
                                                                          Rank,
                                                                          NumReduceDim>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    std::string best_op_name;
    bool found            = false;
    int best_op_id        = -1;
    float best_ave_time   = std::numeric_limits<float>::max();
    float best_gb_per_sec = 0;

    // profile device operation instances
    std::cout << "Run all instances and do timing" << std::endl;

    for(int i = 0; i < op_ptrs.size(); ++i)
    {
        auto& op_ptr = op_ptrs[i];

        auto argument_ptr =
            op_ptr->MakeArgumentPointer({N, H, W, C},         // lengths
                                        strideXY,             // xStrides
                                        strideGammaBeta,      // gammaStrides
                                        strideGammaBeta,      // betaStrides
                                        strideXY,             // yStrides
                                        strideSaveMeanInvStd, // save_mean Strides
                                        strideSaveMeanInvStd, // save_inv_std Strides
                                        {1, 2, 3},            // reduceDims
                                        1e-4,
                                        x_device_buf.GetDeviceBuffer(),
                                        gamma_device_buf.GetDeviceBuffer(),
                                        beta_device_buf.GetDeviceBuffer(),
                                        y_device_buf.GetDeviceBuffer(),
#ifdef SAVE_MEAN_INV_STD
                                        save_mean_device_buf.GetDeviceBuffer(),
                                        save_inv_std_device_buf.GetDeviceBuffer(),
#else
                                        nullptr,
                                        nullptr,
#endif
                                        PassThrough{});

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            size_t workspace_sz = op_ptr->GetWorkSpaceSize(argument_ptr.get());
            SimpleDeviceMem workspace(workspace_sz);
            op_ptr->SetWorkSpacePointer(argument_ptr.get(), workspace.GetDeviceBuffer());

            float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            std::size_t num_byte =
                sizeof(XDataType) * N * H * W * C + sizeof(GammaDataType) * H * W * C +
                sizeof(BetaDataType) * H * W * C + sizeof(YDataType) * N * H * W * C;

#ifdef SAVE_MEAN_INV_STD
            num_byte += sizeof(SaveMeanInvStdDataType) * N * 2;
#endif

            float gb_per_sec = num_byte / 1.E6 / ave_time;

            std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << gb_per_sec << " GB/s, "
                      << op_name << std::endl;

            if(ave_time < best_ave_time)
            {
                found           = true;
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

    std::cout << "Best Perf: " << best_ave_time << " ms, " << best_gb_per_sec << " GB/s, "
              << best_op_name << std::endl;

    // run the best intance
    if(found)
    {
        auto& op_ptr = op_ptrs[best_op_id];
        std::cout << "Run the best instance without timing: " << op_ptr->GetTypeString()
                  << std::endl;

        auto argument_ptr =
            op_ptr->MakeArgumentPointer({N, H, W, C},         // lengths
                                        strideXY,             // xStrides
                                        strideGammaBeta,      // gammaStrides
                                        strideGammaBeta,      // betaStrides
                                        strideXY,             // yStrides
                                        strideSaveMeanInvStd, // save_mean Strides
                                        strideSaveMeanInvStd, // save_inv_std Strides
                                        {1, 2, 3},            // reduceDims
                                        1e-4,
                                        x_device_buf.GetDeviceBuffer(),
                                        gamma_device_buf.GetDeviceBuffer(),
                                        beta_device_buf.GetDeviceBuffer(),
                                        y_device_buf.GetDeviceBuffer(),
#ifdef SAVE_MEAN_INV_STD
                                        save_mean_device_buf.GetDeviceBuffer(),
                                        save_inv_std_device_buf.GetDeviceBuffer(),
#else
                                        nullptr,
                                        nullptr,
#endif
                                        PassThrough{});

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            size_t workspace_sz = op_ptr->GetWorkSpaceSize(argument_ptr.get());
            SimpleDeviceMem workspace(workspace_sz);
            op_ptr->SetWorkSpacePointer(argument_ptr.get(), workspace.GetDeviceBuffer());

            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }

        std::cout << "Done" << std::endl;
    }

    return 0;
}
