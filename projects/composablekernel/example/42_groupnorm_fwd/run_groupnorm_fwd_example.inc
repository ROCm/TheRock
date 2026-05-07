// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

int run_groupnorm_fwd_example(int argc, char* argv[])
{
    ck::index_t N = 32;
    ck::index_t H = 16;
    ck::index_t W = 16;
    ck::index_t G = 64;
    ck::index_t C = 128;

    bool do_verification = true;
    bool time_kernel     = false;
    bool log_kernel      = true;

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        time_kernel     = std::stoi(argv[2]);
        log_kernel      = std::stoi(argv[3]);
    }
    else if(argc == 9)
    {
        do_verification = std::stoi(argv[1]);
        time_kernel     = std::stoi(argv[2]);
        log_kernel      = std::stoi(argv[3]);
        N               = std::stoi(argv[4]);
        H               = std::stoi(argv[5]);
        W               = std::stoi(argv[6]);
        G               = std::stoi(argv[7]);
        C               = std::stoi(argv[8]);
    }
    else
    {
        std::cerr << "arg1 = verify(0=no, 1=yes), arg2 = time kernels(0=no, 1=yes), arg3 = log "
                     "kernels(0=no, 1=yes), arg4 to 8: N, H, W, G, C"
                  << std::endl;

        return 1;
    }

    Tensor<XDataType> x({N, H, W, G, C});
    Tensor<YDataType> y({N, H, W, G, C});
    Tensor<GammaDataType> gamma({G, C});
    Tensor<BetaDataType> beta({G, C});
    Tensor<SaveMeanInvStdDataType> save_mean({N, G});
    Tensor<SaveMeanInvStdDataType> save_inv_std({N, G});

    ck::utils::FillUniformDistribution<XDataType>{0.f, 1.f}(x);
    ck::utils::FillUniformDistribution<GammaDataType>{0.f, 1.f}(gamma);
    ck::utils::FillUniformDistribution<BetaDataType>{0.f, 1.f}(beta);

    DeviceMem x_dev(sizeof(XDataType) * x.mDesc.GetElementSpaceSize());
    DeviceMem gamma_dev(sizeof(GammaDataType) * gamma.mDesc.GetElementSpaceSize());
    DeviceMem beta_dev(sizeof(BetaDataType) * beta.mDesc.GetElementSpaceSize());
    DeviceMem y_dev(sizeof(YDataType) * y.mDesc.GetElementSpaceSize());
#ifdef SAVE_MEAN_INV_STD
    DeviceMem save_mean_dev(sizeof(SaveMeanInvStdDataType) * save_mean.mDesc.GetElementSpaceSize());
    DeviceMem save_inv_std_dev(sizeof(SaveMeanInvStdDataType) *
                               save_inv_std.mDesc.GetElementSpaceSize());
#endif

    x_dev.ToDevice(x.mData.data());
    gamma_dev.ToDevice(gamma.mData.data());
    beta_dev.ToDevice(beta.mData.data());

    const auto y_element_op = YElementOp{};

    auto device_instance = DeviceInstance{};
    auto argument_ptr    = device_instance.MakeArgumentPointer(
        {N, H, W, G, C},
        std::vector<ck::index_t>{x.mDesc.GetStrides().begin(), x.mDesc.GetStrides().end()},
        {0, 0, 0, C, 1},
        {0, 0, 0, C, 1},
        std::vector<ck::index_t>{y.mDesc.GetStrides().begin(), y.mDesc.GetStrides().end()},
        std::vector<ck::index_t>{save_mean.mDesc.GetStrides().begin(),
                                    save_mean.mDesc.GetStrides().end()},
        std::vector<ck::index_t>{save_mean.mDesc.GetStrides().begin(),
                                    save_mean.mDesc.GetStrides().end()},
        {1, 2, 4}, // reduction dimension: [H, W, C]
        1e-6,
        x_dev.GetDeviceBuffer(),
        gamma_dev.GetDeviceBuffer(),
        beta_dev.GetDeviceBuffer(),
        y_dev.GetDeviceBuffer(),
#ifdef SAVE_MEAN_INV_STD
        save_mean_dev.GetDeviceBuffer(),
        save_inv_std_dev.GetDeviceBuffer(),
#else
        nullptr,
        nullptr,
#endif
        y_element_op);

    if(!device_instance.IsSupportedArgument(argument_ptr.get()))
    {
        std::cout << "The runtime parameters are not supported" << std::endl;
        return 1;
    };

    size_t workspace_sz = device_instance.GetWorkSpaceSize(argument_ptr.get());
    DeviceMem workspace_dev(workspace_sz);
    device_instance.SetWorkSpacePointer(argument_ptr.get(), workspace_dev.GetDeviceBuffer());

    auto invoker_ptr = device_instance.MakeInvokerPointer();
    float ave_time =
        invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel, log_kernel});

    std::size_t num_btype = sizeof(XDataType) * N * H * W * G * C +
                            sizeof(YDataType) * N * H * W * G * C + sizeof(GammaDataType) * G * C +
                            sizeof(BetaDataType) * G * C;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s, "
              << device_instance.GetTypeString() << std::endl;

    bool pass = true;
    if(do_verification)
    {
        Tensor<YDataType> host_y({N, H, W, G, C});
        Tensor<SaveMeanInvStdDataType> host_save_mean(HostTensorDescriptor{N, G});
        Tensor<SaveMeanInvStdDataType> host_save_inv_std(HostTensorDescriptor{N, G});
        using ReferenceInstance =
            ck::tensor_operation::host::ReferenceGroupnorm<XDataType,
                                                           GammaDataType,
                                                           BetaDataType,
                                                           YDataType,
                                                           SaveMeanInvStdDataType,
                                                           ComputeDataType,
                                                           YElementOp>;

        ReferenceInstance ref;
        auto ref_argument = ref.MakeArgument(x,
                                             gamma,
                                             beta,
                                             host_y,
                                             host_save_mean,
                                             host_save_inv_std,
                                             y_element_op,
                                             {N, H, W, G, C},
                                             1e-6);
        auto ref_invoker  = ref.MakeInvoker();
        ref_invoker.Run(ref_argument);

        y_dev.FromDevice(y.mData.data());
        pass &= ck::utils::check_err(y, host_y, "Error: Incorrect results", 1e-3, 1e-3);
#ifdef SAVE_MEAN_INV_STD
        save_mean_dev.FromDevice(save_mean.mData.data());
        save_inv_std_dev.FromDevice(save_inv_std.mData.data());
        pass &= ck::utils::check_err(
            save_mean, host_save_mean, "Error: Incorrect results (mean)", 1e-3, 1e-3);
        pass &= ck::utils::check_err(
            save_inv_std, host_save_inv_std, "Error: Incorrect results (inv_std)", 1e-3, 1e-3);
#endif
    }

    return (pass ? 0 : 1);
}
