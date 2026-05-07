// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <getopt.h>

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_common_util.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"

#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_normalization_bwd_data_impl.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_normalization_bwd_gamma_beta_impl.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_groupnorm_bwd.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

using DYDataType         = float;
using XDataType          = float;
using GammaDataType      = float;
using MeanInvStdDataType = float;
using DGammaDataType     = float;
using DBetaDataType      = float;
using DXDataType         = float;
using ComputeDataType    = float;

constexpr int Rank         = 5;
constexpr int NumReduceDim = 3;

// Grouprnorm
// kernel 1:                  M    , K
// dy:     N, H, W, G, C -> N * G, H * W * C
// x:      N, H, W, G, C -> N * G, H * W * C
// gamma:  1, 1, 1, G, C -> 1 * G, 1 * 1 * C
// mean:   N, 1, 1, G, 1 -> N * G, 1 * 1 * 1
// rstd:   N, 1, 1, G, 1 -> N * G, 1 * 1 * 1

// dx:     N, H, W, G, C -> N * G, H * W * C

using XDeviceInstance = ck::tensor_operation::device::DeviceNormalizationBwdDataImpl<
    DYDataType,
    XDataType,
    GammaDataType,
    MeanInvStdDataType,
    ComputeDataType,
    DXDataType,
    Rank,
    NumReduceDim,
    256,   // BlockSize
    8,     // MThreadClusterSize
    32,    // KThreadClusterSize
    1,     // MThreadSliceSize
    4,     // KThreadSliceSize
    true,  // IsDYFastestDimReduced
    4,     // DYSrcVectorSize
    true,  // IsXFastestDimReduced
    4,     // XSrcVectorSize
    true,  // IsGammaFastestDimReduced
    4,     // GammaSrcVectorSize
    false, // IsMeanInvStdFastestDimReduced
    1,     // MeanInvStdSrcVectorSize
    true,  // IsDXFastestDimReduced
    4>;    // DXDstVectorSize

// kernel 2:                  M    , K
// dy:     N, H, W, G, C -> G * C, N * H * W
// x:      N, H, W, G, C -> G * C, N * H * W
// mean:   N, 1, 1, G, 1 -> G * 1, N * 1 * 1
// rstd:   N, 1, 1, G, 1 -> G * 1, N * 1 * 1

// dgamma: 1, 1, 1, G, C -> G * C
// dbeta:  1, 1, 1, G, C -> G * C

// reduced axis: 0, 1, 2

using GammaBetaDeviceInstance = ck::tensor_operation::device::DeviceNormalizationBwdGammaBetaImpl<
    DYDataType,
    XDataType,
    MeanInvStdDataType,
    ComputeDataType,
    DGammaDataType,
    DBetaDataType,
    Rank,
    NumReduceDim,
    256,   // BlockSize
    8,     // ClusterInvariant
    32,    // ClusterReduce
    4,     // SliceInvariant
    1,     // SliceReduce
    false, // IsDYFastestDimReduced
    4,     // DYSrcVectorSize
    false, // IsXFastestDimReduced
    4,     // XSrcVectorSize
    false, // IsMeanInvStdFastestDimReduced
    1,     // MeanInvStdSrcVectorSize
    4,     // DGammaDstVectorSize
    4>;    // DBetaDstVectorSize

int main(int argc, char* argv[])
{
    bool time_kernel = false;

    ck::index_t N = 16;
    ck::index_t H = 16;
    ck::index_t W = 16;
    ck::index_t G = 32;
    ck::index_t C = 64;

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 6)
    {
        N = std::stoi(argv[1]);
        H = std::stoi(argv[2]);
        W = std::stoi(argv[3]);
        G = std::stoi(argv[4]);
        C = std::stoi(argv[5]);
    }
    else
    {
        std::cerr << "arg1 to 5: N, H, W, G, C" << std::endl;

        return 1;
    }

    Tensor<DYDataType> dy({N, H, W, G, C});
    Tensor<XDataType> x({N, H, W, G, C});
    Tensor<GammaDataType> gamma({G, C});
    Tensor<MeanInvStdDataType> mean({N, G});
    Tensor<MeanInvStdDataType> inv_std({N, G});

    Tensor<DGammaDataType> dgamma({G, C});
    Tensor<DBetaDataType> dbeta({G, C});
    Tensor<DXDataType> dx({N, H, W, G, C});

    dy.GenerateTensorValue(GeneratorTensor_3<DYDataType>{0.0, 1.0});
    x.GenerateTensorValue(GeneratorTensor_3<XDataType>{0.0, 1.0});
    gamma.GenerateTensorValue(GeneratorTensor_3<GammaDataType>{0.0, 1.0});
    mean.GenerateTensorValue(GeneratorTensor_3<MeanInvStdDataType>{0.0, 1.0});
    inv_std.GenerateTensorValue(GeneratorTensor_3<MeanInvStdDataType>{0.0, 1.0});

    DeviceMem dy_dev(sizeof(DYDataType) * dy.mDesc.GetElementSpaceSize());
    DeviceMem x_dev(sizeof(XDataType) * x.mDesc.GetElementSpaceSize());
    DeviceMem gamma_dev(sizeof(GammaDataType) * gamma.mDesc.GetElementSpaceSize());
    DeviceMem mean_dev(sizeof(MeanInvStdDataType) * mean.mDesc.GetElementSpaceSize());
    DeviceMem inv_std_dev(sizeof(MeanInvStdDataType) * inv_std.mDesc.GetElementSpaceSize());
    DeviceMem dx_dev(sizeof(DXDataType) * dx.mDesc.GetElementSpaceSize());
    DeviceMem dgamma_dev(sizeof(DGammaDataType) * dgamma.mDesc.GetElementSpaceSize());
    DeviceMem dbeta_dev(sizeof(DBetaDataType) * dbeta.mDesc.GetElementSpaceSize());

    dy_dev.ToDevice(dy.mData.data());
    x_dev.ToDevice(x.mData.data());
    gamma_dev.ToDevice(gamma.mData.data());
    mean_dev.ToDevice(mean.mData.data());
    inv_std_dev.ToDevice(inv_std.mData.data());

    std::vector<ck::index_t> dyStrides{dy.mDesc.GetStrides().begin(), dy.mDesc.GetStrides().end()};
    std::vector<ck::index_t> xStrides{x.mDesc.GetStrides().begin(), x.mDesc.GetStrides().end()};
    std::vector<ck::index_t> gammaStrides  = {0, 0, 0, C, 1};
    std::vector<ck::index_t> meanStrides   = {G, 0, 0, 1, 0};
    std::vector<ck::index_t> invStdStrides = {G, 0, 0, 1, 0};
    std::vector<ck::index_t> dxStrides{dx.mDesc.GetStrides().begin(), dx.mDesc.GetStrides().end()};

    // backward x
    auto x_device_instance = XDeviceInstance{};

    auto x_argument_ptr = x_device_instance.MakeArgumentPointer({N, H, W, G, C}, // lengths
                                                                dyStrides,       // dyStrides
                                                                xStrides,        // xStrides
                                                                gammaStrides,    // gammaStrides
                                                                meanStrides,     // meanStrides
                                                                invStdStrides,   // invStdStrides
                                                                dxStrides,       // dxStrides
                                                                {1, 2, 4},       // reduceDims
                                                                dy_dev.GetDeviceBuffer(),
                                                                x_dev.GetDeviceBuffer(),
                                                                gamma_dev.GetDeviceBuffer(),
                                                                mean_dev.GetDeviceBuffer(),
                                                                inv_std_dev.GetDeviceBuffer(),
                                                                dx_dev.GetDeviceBuffer());

    if(!x_device_instance.IsSupportedArgument(x_argument_ptr.get()))
    {
        std::cout << "The runtime parameters are not supported." << __FILE__ << ":" << __LINE__
                  << std::endl;
        return 1;
    };

    auto x_invoker_ptr = x_device_instance.MakeInvokerPointer();
    x_invoker_ptr->Run(x_argument_ptr.get(), StreamConfig{nullptr, time_kernel});

    // backward gamma & beta

    auto gamma_beta_device_instance = GammaBetaDeviceInstance{};
    auto gamma_beta_argument_ptr =
        gamma_beta_device_instance.MakeArgumentPointer({N, H, W, G, C}, // inLengths
                                                       dyStrides,       // dyStrides
                                                       xStrides,        // xStrides
                                                       meanStrides,     // meanStrides
                                                       invStdStrides,   // invStdStrides
                                                       {G, C},          // outLengths
                                                       {C, 1},          // dgammaStrides
                                                       {C, 1},          // dbetaStrides
                                                       {0, 1, 2},       // reduceDims
                                                       dy_dev.GetDeviceBuffer(),
                                                       x_dev.GetDeviceBuffer(),
                                                       mean_dev.GetDeviceBuffer(),
                                                       inv_std_dev.GetDeviceBuffer(),
                                                       dgamma_dev.GetDeviceBuffer(),
                                                       dbeta_dev.GetDeviceBuffer());

    if(!gamma_beta_device_instance.IsSupportedArgument(gamma_beta_argument_ptr.get()))
    {
        std::cout << "The runtime parameters are not supported." << __FILE__ << ":" << __LINE__
                  << std::endl;
        return 1;
    };

    auto gamma_beta_invoker_ptr = gamma_beta_device_instance.MakeInvokerPointer();
    gamma_beta_invoker_ptr->Run(gamma_beta_argument_ptr.get(), StreamConfig{nullptr, time_kernel});

    bool pass = true;
    {
        Tensor<DGammaDataType> host_dgamma({G, C});
        Tensor<DBetaDataType> host_dbeta({G, C});
        Tensor<DXDataType> host_dx({N, H, W, G, C});
        using ReferenceInstance =
            ck::tensor_operation::host::ReferenceGroupnormBwd<DYDataType,
                                                              XDataType,
                                                              GammaDataType,
                                                              MeanInvStdDataType,
                                                              DGammaDataType,
                                                              DBetaDataType,
                                                              DXDataType,
                                                              ComputeDataType>;

        ReferenceInstance ref;
        auto ref_argument = ref.MakeArgument(
            dy, x, gamma, mean, inv_std, host_dgamma, host_dbeta, host_dx, {N, H, W, G, C});
        auto ref_invoker = ref.MakeInvoker();
        ref_invoker.Run(ref_argument);

        dgamma_dev.FromDevice(dgamma.mData.data());
        dbeta_dev.FromDevice(dbeta.mData.data());
        dx_dev.FromDevice(dx.mData.data());

        pass &= ck::utils::check_err(dgamma, host_dgamma, "Error: Incorrect dgamma", 1e-3, 1e-3);
        pass &= ck::utils::check_err(dbeta, host_dbeta, "Error: Incorrect dbeta", 1e-3, 1e-3);
        pass &= ck::utils::check_err(dx, host_dx, "Error: Incorrect dx", 1e-3, 1e-3);
    }

    return (pass ? 0 : 1);
}
