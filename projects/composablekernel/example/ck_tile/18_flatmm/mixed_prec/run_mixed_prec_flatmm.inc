// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

template <typename PrecActType,
          typename PrecWeightType,
          typename FlatmmConfig,
          bool UsePersistentKernel = false,
          typename ALayout,
          typename BLayout,
          typename CLayout>
int run_mixed_prec_flatmm_with_layouts(int argc,
                                       char* argv[],
                                       const ALayout a_layout                  = ALayout{},
                                       const BLayout b_layout                  = BLayout{},
                                       [[maybe_unused]] const CLayout c_layout = CLayout{})
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    using ADataType   = PrecActType;
    using BDataType   = PrecWeightType;
    using CDataType   = PrecActType;
    using AccDataType = float;

    using ScaleType = ck_tile::e8m0_t;

    constexpr int DequantGranularityN = 1;
    constexpr int DequantGranularityK = 32;

    ck_tile::index_t M = arg_parser.get_int("m");
    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t K = arg_parser.get_int("k");

    ck_tile::index_t stride_A = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_B = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_C = arg_parser.get_int("stride_c");

    ck_tile::index_t kbatch      = arg_parser.get_int("split_k");
    ck_tile::index_t init_method = arg_parser.get_int("init");
    ck_tile::index_t n_warmup    = arg_parser.get_int("warmup");
    ck_tile::index_t n_repeat    = arg_parser.get_int("repeat");

    stride_A = ck_tile::get_default_stride(M, K, stride_A, is_row_major(a_layout));
    stride_B = ck_tile::get_default_stride(K, N, stride_B, is_row_major(b_layout));
    stride_C = ck_tile::get_default_stride(M, N, stride_C, is_row_major(CLayout{}));

    ck_tile::HostTensor<ADataType> a_host(
        ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(a_layout)));
    ck_tile::HostTensor<BDataType> b_origin_host(
        ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(b_layout)));
    ck_tile::HostTensor<CDataType> c_rslt_host(
        ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

    ck_tile::HostTensor<ScaleType> scale_b(ck_tile::HostTensorDescriptor(
        {K / DequantGranularityK, N / DequantGranularityN}, {N / DequantGranularityN, 1}));

    if(init_method == 0)
    {
        ck_tile::FillUniformDistribution<ADataType>{0.0f, 1.0f}(a_host);
        ck_tile::FillUniformDistribution<BDataType>{-.5f, .5f}(b_origin_host);
        ck_tile::FillUniformDistribution<ScaleType>{-2.f, 2.f}(scale_b);
    }
    else if(init_method == 1)
    {
        ck_tile::FillUniformDistribution<ADataType>{1.f, 1.f}(a_host);
        ck_tile::FillUniformDistribution<BDataType>{1.f, 1.f}(b_origin_host);
        ck_tile::FillUniformDistribution<ScaleType>{1.f, 1.f}(scale_b);
    }

    ck_tile::HostTensor<BDataType> b_shuffle_host(
        ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(b_layout)));
    preShuffleWeight<FlatmmConfig>(b_origin_host.begin(), b_shuffle_host.begin(), N, K);

    ck_tile::HostTensor<ScaleType> scale_b_shuffle = preShuffleScale<FlatmmConfig>(scale_b);

    ck_tile::DeviceMem a_dev_buf(a_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_shuffle_dev_buf(b_shuffle_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_dev_buf(c_rslt_host.get_element_space_size_in_bytes());

    ck_tile::DeviceMem scale_b_dev_buf(scale_b_shuffle.get_element_space_size_in_bytes());

    a_dev_buf.ToDevice(a_host.data());
    b_shuffle_dev_buf.ToDevice(b_shuffle_host.data());
    c_rslt_host.SetZero();
    scale_b_dev_buf.ToDevice(scale_b_shuffle.data());

    auto scale_b_dev_ptr =
        ck_tile::FlatmmScalePointer<DequantGranularityN, DequantGranularityK, ScaleType>{
            static_cast<ScaleType*>(scale_b_dev_buf.GetDeviceBuffer()), N / DequantGranularityN};

    invoke_mixed_prec_flatmm<FlatmmConfig,
                             ADataType,
                             BDataType,
                             ck_tile::tuple<>,
                             AccDataType,
                             CDataType,
                             ALayout,
                             BLayout,
                             ck_tile::tuple<>,
                             CLayout,
                             decltype(scale_b_dev_ptr),
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
                                                  scale_b_dev_ptr,
                                                  n_warmup,
                                                  n_repeat);

    c_dev_buf.FromDevice(c_rslt_host.data());

    bool pass = true;
    if(arg_parser.get_int("v") == 1)
    {
        ck_tile::DeviceMem b_origin_dev_buf(b_origin_host.get_element_space_size_in_bytes());
        b_origin_dev_buf.ToDevice(b_origin_host.data());

        ck_tile::HostTensor<CDataType> c_gpu_ref_host(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        ck_tile::DeviceMem c_gpu_ref_dev_buf(c_gpu_ref_host.get_element_space_size_in_bytes());

        ck_tile::HostTensor<AccDataType> scale_A(
            ck_tile::HostTensorDescriptor({1, K / DequantGranularityK}, {1, 1}));

        // scaleA = 1 has no effect on the result
        ck_tile::FillUniformDistribution<AccDataType>{1.f, 1.f}(scale_A);
        ck_tile::DeviceMem scale_A_dev_buf(scale_A.get_element_space_size_in_bytes());
        scale_A_dev_buf.ToDevice(scale_A.data());

        // convert scale_b from e8m0 to float
        ck_tile::HostTensor<AccDataType> scale_b_float(ck_tile::HostTensorDescriptor(
            {K / DequantGranularityK, N / DequantGranularityN}, {N / DequantGranularityN, 1}));
        std::copy(scale_b.begin(), scale_b.end(), scale_b_float.begin());
        ck_tile::DeviceMem scale_b_float_dev_buf(scale_b_float.get_element_space_size_in_bytes());
        scale_b_float_dev_buf.ToDevice(scale_b_float.data());

        c_gpu_ref_dev_buf.SetZero();
        ck_tile::reference_blockwise_gemm_gpu<ADataType,
                                              BDataType,
                                              AccDataType,
                                              CDataType,
                                              ALayout,
                                              BLayout,
                                              CLayout>(
            static_cast<ADataType*>(a_dev_buf.GetDeviceBuffer()),
            static_cast<BDataType*>(b_origin_dev_buf.GetDeviceBuffer()),
            static_cast<CDataType*>(c_gpu_ref_dev_buf.GetDeviceBuffer()),
            M,
            N,
            K,
            stride_A,
            stride_B,
            stride_C,
            M,
            DequantGranularityN,
            DequantGranularityK,
            static_cast<float*>(scale_A_dev_buf.GetDeviceBuffer()),
            static_cast<float*>(scale_b_float_dev_buf.GetDeviceBuffer()));

        c_gpu_ref_dev_buf.FromDevice(c_gpu_ref_host.data());

        const float rtol = std::is_same_v<ADataType, ck_tile::half_t> ? 1e-3 : 1e-2;
        const float atol = std::is_same_v<ADataType, ck_tile::half_t> ? 1e-3 : 1e-2;

        pass = ck_tile::check_err(
            c_rslt_host, c_gpu_ref_host, "Error: Incorrect results!", rtol, atol);

        std::cout << "Relative error threshold: " << rtol << " Absolute error threshold: " << atol
                  << std::endl;
        std::cout << "The GPU veification result is: " << (pass ? "correct" : "fail") << std::endl;
    }

    return pass;
}
