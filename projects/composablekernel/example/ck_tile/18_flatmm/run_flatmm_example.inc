// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
template <typename PrecType,
          typename FlatmmConfig,
          int ScaleGranularityM    = -1,
          int ScaleGranularityN    = -1,
          bool UsePersistentKernel = false,
          typename ALayout,
          typename BLayout,
          typename CLayout>
int run_flatmm_example_with_layouts(int argc,
                                    char* argv[],
                                    const ALayout a_layout                  = ALayout{},
                                    const BLayout b_layout                  = BLayout{},
                                    [[maybe_unused]] const CLayout c_layout = CLayout{})
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    using ADataType   = typename GemmBasicTypeConfig<PrecType>::ADataType;
    using BDataType   = typename GemmBasicTypeConfig<PrecType>::BDataType;
    using CDataType   = typename GemmBasicTypeConfig<PrecType>::CDataType;
    using AccDataType = typename GemmBasicTypeConfig<PrecType>::AccDataType;

    ck_tile::index_t M = arg_parser.get_int("m");
    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t K = arg_parser.get_int("k");

    ck_tile::index_t stride_A = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_B = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_C = arg_parser.get_int("stride_c");

    ck_tile::index_t kbatch      = arg_parser.get_int("split_k");
    int n_warmup                 = arg_parser.get_int("warmup");
    int n_repeat                 = arg_parser.get_int("repeat");
    ck_tile::index_t init_method = arg_parser.get_int("init");
    // persistent not added

    stride_A = ck_tile::get_default_stride(M, K, stride_A, is_row_major(a_layout));
    stride_B = ck_tile::get_default_stride(K, N, stride_B, is_row_major(b_layout));
    stride_C = ck_tile::get_default_stride(M, N, stride_C, is_row_major(CLayout{}));

    ck_tile::HostTensor<ADataType> a_host(
        ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(a_layout)));
    ck_tile::HostTensor<BDataType> b_origin_host(
        ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(b_layout)));
    ck_tile::HostTensor<CDataType> c_rslt_host(
        ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

    ck_tile::HostTensor<AccDataType> per_token_scale(ck_tile::HostTensorDescriptor({M}, {1}));
    ck_tile::HostTensor<AccDataType> per_channel_scale(ck_tile::HostTensorDescriptor({N}, {1}));

    // TODO: add different init types
    if(init_method == 0)
    {
        // ck_tile::FillUniformDistribution<ADataType>{-.5f, .5f}(a_host);
        // ck_tile::FillUniformDistribution<BDataType>{-.5f, .5f}(b_origin_host);
        ck_tile::FillUniformDistribution<ADataType>{0.0f, 1.0f}(a_host);
        ck_tile::FillUniformDistribution<BDataType>{-.5f, .5f}(b_origin_host);
        ck_tile::FillUniformDistribution<AccDataType>{-1.f, 1.f}(per_token_scale);
        ck_tile::FillUniformDistribution<AccDataType>{-1.f, 1.f}(per_channel_scale);
    }
    else if(init_method == 1)
    {
        ck_tile::FillMonotonicSeq<ADataType>{}(a_host);
        ck_tile::FillMonotonicSeq<BDataType>{}(b_origin_host);
        ck_tile::FillUniformDistribution<AccDataType>{1.f, 1.f}(per_token_scale);
        ck_tile::FillUniformDistribution<AccDataType>{1.f, 1.f}(per_channel_scale);
    }
    else if(init_method == 2)
    {
        ck_tile::FillUniformDistribution<ADataType>{1.f, 1.f}(a_host);
        ck_tile::FillUniformDistribution<BDataType>{1.f, 1.f}(b_origin_host);
        ck_tile::FillUniformDistribution<AccDataType>{1.f, 1.f}(per_token_scale);
        ck_tile::FillUniformDistribution<AccDataType>{1.f, 1.f}(per_channel_scale);
    }
    else
    {
        a_host.SetZero();
        b_origin_host.SetZero();
    }

    ck_tile::DeviceMem a_dev_buf(a_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_dev_buf(c_rslt_host.get_element_space_size_in_bytes());

    ck_tile::DeviceMem per_token_scale_dev_buf(per_token_scale.get_element_space_size_in_bytes());
    ck_tile::DeviceMem per_channel_scale_dev_buf(
        per_channel_scale.get_element_space_size_in_bytes());

    a_dev_buf.ToDevice(a_host.data());
    c_rslt_host.SetZero();
    per_token_scale_dev_buf.ToDevice(per_token_scale.data());
    per_channel_scale_dev_buf.ToDevice(per_channel_scale.data());

    // do pre-shuffle
    ck_tile::HostTensor<BDataType> b_shuffle_host = [&]() {
        if constexpr(FlatmmConfig::TiledMMAPermuteN)
        {
            return shuffle_b_v1<FlatmmConfig>(b_origin_host);
        }
        else
        {
            return shuffle_b_v0<FlatmmConfig>(b_origin_host);
        }
    }();
    ck_tile::DeviceMem b_shuffle_dev_buf(b_shuffle_host.get_element_space_size_in_bytes());
    b_shuffle_dev_buf.ToDevice(b_shuffle_host.data());

    auto per_token_scale_dev_ptr = ck_tile::FlatmmScalePointer<ScaleGranularityM>{
        static_cast<float*>(per_token_scale_dev_buf.GetDeviceBuffer())};
    auto per_channel_scale_dev_ptr = ck_tile::FlatmmScalePointer<ScaleGranularityN>{
        static_cast<float*>(per_channel_scale_dev_buf.GetDeviceBuffer())};

    invoke_flatmm<FlatmmConfig,
                  ADataType,
                  BDataType,
                  ck_tile::tuple<>,
                  AccDataType,
                  CDataType,
                  ALayout,
                  BLayout,
                  ck_tile::tuple<>,
                  CLayout,
                  decltype(per_token_scale_dev_ptr),
                  decltype(per_channel_scale_dev_ptr),
                  UsePersistentKernel>(a_dev_buf,
                                       b_shuffle_dev_buf,
                                       c_dev_buf,
                                       M,
                                       N,
                                       K,
                                       stride_A,
                                       stride_B,
                                       stride_C,
                                       kbatch,
                                       per_token_scale_dev_ptr,
                                       per_channel_scale_dev_ptr,
                                       n_warmup,
                                       n_repeat);

    c_dev_buf.FromDevice(c_rslt_host.data());

    bool pass = true;

    if(arg_parser.get_int("v") == 1)
    {
        if(ScaleGranularityM != -1 || ScaleGranularityN != -1)
            throw std::runtime_error("ScaleAB is not supported for CPU verification!\n");
        ck_tile::HostTensor<CDataType> c_ref_host(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        c_ref_host.SetZero();

        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_host, b_origin_host, c_ref_host);
        const float max_accumulated_value =
            *std::max_element(c_ref_host.mData.begin(), c_ref_host.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, kbatch, max_accumulated_value);
        pass = ck_tile::check_err(c_rslt_host,
                                  c_ref_host,
                                  "Error: Incorrect results!",
                                  rtol_atol.at(ck_tile::number<0>{}),
                                  rtol_atol.at(ck_tile::number<1>{}));

        std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                  << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                  << std::endl;
        std::cout << "The CPU veification result is:" << (pass ? "correct" : "fail") << std::endl;
    }
    else if(arg_parser.get_int("v") == 2)
    {
        ck_tile::DeviceMem b_origin_dev_buf(b_origin_host.get_element_space_size_in_bytes());
        b_origin_dev_buf.ToDevice(b_origin_host.data());

        ck_tile::HostTensor<CDataType> c_gpu_ref_host(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        ck_tile::DeviceMem c_gpu_ref_dev_buf(c_gpu_ref_host.get_element_space_size_in_bytes());
        c_gpu_ref_host.SetZero();
        c_gpu_ref_dev_buf.SetZero();

        ADataType* d_A;
        BDataType* d_B;
        CDataType* d_C;

        ck_tile::hip_check_error(hipMalloc(&d_A, M * K * sizeof(ADataType)));
        ck_tile::hip_check_error(hipMalloc(&d_B, N * K * sizeof(BDataType)));
        ck_tile::hip_check_error(hipMalloc(&d_C, M * N * sizeof(CDataType)));

        ck_tile::hip_check_error(hipMemcpy(
            d_A, a_dev_buf.GetDeviceBuffer(), M * K * sizeof(ADataType), hipMemcpyHostToDevice));
        ck_tile::hip_check_error(hipMemcpy(d_B,
                                           b_origin_dev_buf.GetDeviceBuffer(),
                                           N * K * sizeof(BDataType),
                                           hipMemcpyHostToDevice));

        if constexpr(ScaleGranularityM == -1 && ScaleGranularityN == -1)
        {
            ck_tile::reference_gemm_gpu<ADataType,
                                        BDataType,
                                        AccDataType,
                                        CDataType,
                                        ALayout,
                                        BLayout,
                                        CLayout>(
                d_A, d_B, d_C, M, N, K, stride_A, stride_B, stride_C);
        }
        else
        {
            ck_tile::reference_blockwise_gemm_gpu<ADataType,
                                                  BDataType,
                                                  AccDataType,
                                                  CDataType,
                                                  ALayout,
                                                  BLayout,
                                                  CLayout>(
                d_A,
                d_B,
                d_C,
                M,
                N,
                K,
                stride_A,
                stride_B,
                stride_C,
                ScaleGranularityM,
                ScaleGranularityN,
                K,
                static_cast<float*>(per_token_scale_dev_buf.GetDeviceBuffer()),
                static_cast<float*>(per_channel_scale_dev_buf.GetDeviceBuffer()));
        }

        ck_tile::hip_check_error(hipMemcpy(c_gpu_ref_dev_buf.GetDeviceBuffer(),
                                           d_C,
                                           M * N * sizeof(CDataType),
                                           hipMemcpyDeviceToHost));

        ck_tile::hip_check_error(hipFree(d_A));
        ck_tile::hip_check_error(hipFree(d_B));
        ck_tile::hip_check_error(hipFree(d_C));

        c_gpu_ref_dev_buf.FromDevice(c_gpu_ref_host.data());
        const float max_accumulated_value =
            *std::max_element(c_gpu_ref_host.mData.begin(), c_gpu_ref_host.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, kbatch, max_accumulated_value);
        pass = ck_tile::check_err(c_rslt_host,
                                  c_gpu_ref_host,
                                  "Error: Incorrect results!",
                                  rtol_atol.at(ck_tile::number<0>{}),
                                  rtol_atol.at(ck_tile::number<1>{}));

        std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                  << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                  << std::endl;
        std::cout << "The GPU veification result is: " << (pass ? "correct" : "fail") << std::endl;
    }

    return pass;
}
