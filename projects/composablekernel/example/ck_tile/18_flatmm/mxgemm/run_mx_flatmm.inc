// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

template <typename PrecActType,
          typename PrecWeightType,
          typename CDataType,
          typename FlatmmConfig,
          bool UsePersistentKernel = false,
          typename ALayout,
          typename BLayout,
          typename CLayout>
int run_mx_flatmm_with_layouts(int argc,
                               char* argv[],
                               const ALayout a_layout = ALayout{},
                               const BLayout b_layout = BLayout{},
                               const CLayout c_layout = CLayout{})
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    using ADataType   = PrecActType;
    using BDataType   = PrecWeightType;
    using AccDataType = float;

    using ScaleType = ck_tile::e8m0_t;

    constexpr int ScaleGranularityM = 1;
    constexpr int ScaleGranularityN = 1;
    constexpr int ScaleGranularityK = 32;

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
    stride_C = ck_tile::get_default_stride(M, N, stride_C, is_row_major(c_layout));

    auto scale_stride_A = ck_tile::get_default_stride(
        M / ScaleGranularityM, K / ScaleGranularityK, 0, is_row_major(a_layout));
    auto scale_stride_B = ck_tile::get_default_stride(
        K / ScaleGranularityK, N / ScaleGranularityN, 0, is_row_major(b_layout));

    if(K % ScaleGranularityK != 0)
        throw std::runtime_error("wrong! K must be multiple of ScaleGranularityK.");
    if(K % ck_tile::numeric_traits<ADataType>::PackedSize != 0 ||
       K % ck_tile::numeric_traits<BDataType>::PackedSize != 0)
        throw std::runtime_error("wrong! K must be multiple of packed size.");

    ck_tile ::HostTensor<ADataType> a_host(
        ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(a_layout)));
    ck_tile::HostTensor<BDataType> b_origin_host(
        ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(b_layout)));
    ck_tile::HostTensor<CDataType> c_rslt_host(
        ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

    ck_tile::HostTensor<ScaleType> scale_a(ck_tile::host_tensor_descriptor(
        M / ScaleGranularityM, K / ScaleGranularityK, scale_stride_A, is_row_major(a_layout)));
    ck_tile::HostTensor<ScaleType> scale_b(ck_tile::host_tensor_descriptor(
        K / ScaleGranularityK, N / ScaleGranularityN, scale_stride_B, is_row_major(b_layout)));

    if(init_method == 0)
    {
        ck_tile::FillUniformDistribution<>{0.0f, 1.0f}(a_host);
        ck_tile::FillUniformDistribution<>{-.5f, .5f}(b_origin_host);
        ck_tile::FillUniformDistribution<>{-2.f, 2.f}(scale_a);
        ck_tile::FillUniformDistribution<>{-2.f, 2.f}(scale_b);
    }
    else if(init_method == 1)
    {
        ck_tile::FillUniformDistribution<>{1.f, 1.f}(a_host);
        ck_tile::FillUniformDistribution<>{1.f, 1.f}(b_origin_host);
        ck_tile::FillUniformDistribution<>{1.f, 1.f}(scale_a);
        ck_tile::FillUniformDistribution<>{1.f, 1.f}(scale_b);
    }
    else
    {
        throw std::runtime_error("wrong! Unexpected init_method");
    }

    const auto b_shuffled_host  = preShuffleWeight<FlatmmConfig::N_Warp_Tile>(b_origin_host);
    const auto scale_a_shuffled = preShuffleScale<FlatmmConfig, true>(scale_a);
    const auto scale_b_shuffled = preShuffleScale<FlatmmConfig, false>(scale_b);

    ck_tile::DeviceMem a_dev_buf(a_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_shuffled_dev_buf(b_shuffled_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_dev_buf(c_rslt_host.get_element_space_size_in_bytes());

    ck_tile::DeviceMem scale_a_dev_buf(scale_a_shuffled.get_element_space_size_in_bytes());
    ck_tile::DeviceMem scale_b_dev_buf(scale_b_shuffled.get_element_space_size_in_bytes());

    a_dev_buf.ToDevice(a_host.data());
    b_shuffled_dev_buf.ToDevice(b_shuffled_host.data());
    c_rslt_host.SetZero();
    scale_a_dev_buf.ToDevice(scale_a_shuffled.data());
    scale_b_dev_buf.ToDevice(scale_b_shuffled.data());

    auto scale_a_dev_ptr =
        ck_tile::FlatmmScalePointer<ScaleGranularityM, ScaleGranularityK, ScaleType>{
            static_cast<ScaleType*>(scale_a_dev_buf.GetDeviceBuffer()), M / ScaleGranularityM};
    auto scale_b_dev_ptr =
        ck_tile::FlatmmScalePointer<ScaleGranularityN, ScaleGranularityK, ScaleType>{
            static_cast<ScaleType*>(scale_b_dev_buf.GetDeviceBuffer()), N / ScaleGranularityN};

    invoke_mx_flatmm<FlatmmConfig,
                     ADataType,
                     BDataType,
                     ck_tile::tuple<>,
                     AccDataType,
                     CDataType,
                     ALayout,
                     BLayout,
                     ck_tile::tuple<>,
                     CLayout,
                     decltype(scale_a_dev_ptr),
                     decltype(scale_b_dev_ptr),
                     UsePersistentKernel>(a_dev_buf,
                                          b_shuffled_dev_buf,
                                          c_dev_buf,
                                          M,
                                          N,
                                          K,
                                          stride_A,
                                          stride_B,
                                          stride_C,
                                          kbatch,
                                          scale_a_dev_ptr,
                                          scale_b_dev_ptr,
                                          n_warmup,
                                          n_repeat);

    c_dev_buf.FromDevice(c_rslt_host.data());

    bool pass = true;
    if(arg_parser.get_int("v") == 1)
    {
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        ck_tile::reference_mx_gemm<ADataType, BDataType, ScaleType, AccDataType, CDataType>(
            a_host, b_origin_host, c_m_n_host_ref, scale_a, scale_b);

        const float rtol = std::is_same_v<ADataType, ck_tile::half_t> ? 1e-3 : 1e-2;
        const float atol = std::is_same_v<ADataType, ck_tile::half_t> ? 1e-3 : 1e-2;

        pass = ck_tile::check_err(
            c_rslt_host, c_m_n_host_ref, "Error: Incorrect results!", rtol, atol);

        std::cout << "Relative error threshold: " << rtol << " Absolute error threshold: " << atol
                  << std::endl;
        std::cout << "The GPU veification result is: " << (pass ? "correct" : "fail") << std::endl;
    }

    return pass ? 0 : -1;
}
