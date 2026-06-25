// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    // Calculate thresholds
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
    // Calculate error due to split_k accumulation
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
        max_accumulated_value, kbatch);
    // Use higher threshold
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

template <typename FlatmmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDatatype,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename ScaleM,
          typename ScaleN,
          typename CDEElementWise = ck_tile::element_wise::PassThrough>
float invoke_gemm(int n_warmup,
                  int n_repeat,
                  const ck_tile::ContiguousGroupedFlatmmHostArgs<ScaleM, ScaleN>& args)
{
    float ave_time = grouped_flatmm<FlatmmConfig,
                                    ADataType,
                                    BDataType,
                                    DsDatatype,
                                    AccDataType,
                                    CDataType,
                                    ALayout,
                                    BLayout,
                                    DsLayout,
                                    CLayout,
                                    false,
                                    CDEElementWise>(
        args, ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat});

    std::string op_name{"Grouped Gemm"};

    std::size_t flop     = std::size_t(2) * args.M * args.N * args.K;
    std::size_t num_byte = sizeof(ADataType) * args.M * args.K +
                           sizeof(BDataType) * args.N * args.K +
                           sizeof(CDataType) * args.M * args.N;

    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
              << gb_per_sec << " GB/s, " << op_name << std::endl;

    return ave_time;
}

template <typename FlatmmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDatatype,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename ScaleM,
          typename ScaleN,
          typename CDEElementWise = ck_tile::element_wise::PassThrough>
float invoke_gemm(int n_warmup,
                  int n_repeat,
                  int val_m,
                  const ck_tile::MaskedGroupedFlatmmHostArgs<ScaleM, ScaleN>& args)
{
    float ave_time = grouped_flatmm<FlatmmConfig,
                                    ADataType,
                                    BDataType,
                                    DsDatatype,
                                    AccDataType,
                                    CDataType,
                                    ALayout,
                                    BLayout,
                                    DsLayout,
                                    CLayout,
                                    false,
                                    CDEElementWise>(
        args, ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat});

    std::string op_name{"Grouped Gemm"};

    std::size_t flop     = std::size_t(2) * val_m * args.N * args.K;
    std::size_t num_byte = sizeof(ADataType) * val_m * args.K +
                           sizeof(BDataType) * args.N * args.K * args.group_count +
                           sizeof(CDataType) * val_m * args.N;

    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
              << gb_per_sec << " GB/s, " << op_name << std::endl;

    return ave_time;
}

template <typename PrecType,
          typename FlatmmConfig,
          int ScaleGranularityM = -1,
          int ScaleGranularityN = -1,
          typename ALayout,
          typename BLayout,
          typename CLayout>
int run_contiguous_grouped_flatmm_example_with_layouts(
    int argc,
    char* argv[],
    const ALayout a_layout                  = ALayout{},
    const BLayout b_layout                  = BLayout{},
    [[maybe_unused]] const CLayout c_layout = CLayout{})
{
    auto [result, arg_parser] = create_args(argc, argv);

    if(!result)
    {
        return -1;
    };

    using ADataType   = typename GemmBasicTypeConfig<PrecType>::ADataType;
    using BDataType   = typename GemmBasicTypeConfig<PrecType>::BDataType;
    using CDataType   = typename GemmBasicTypeConfig<PrecType>::CDataType;
    using AccDataType = typename GemmBasicTypeConfig<PrecType>::AccDataType;

    constexpr int BlockM = FlatmmConfig::M_Tile;

    const int group_count = arg_parser.get_int("group_count");
    const int repeat      = arg_parser.get_int("repeat");
    const int warmup      = arg_parser.get_int("warmup");

    std::vector<ck_tile::index_t> Ms = arg_parser.get_int_vec("Ms");
    std::vector<ck_tile::index_t> Ns = arg_parser.get_int_vec("Ns");
    std::vector<ck_tile::index_t> Ks = arg_parser.get_int_vec("Ks");

    if(!(int(Ms.size()) == group_count))
    {
        std::cout << "Please check the input data." << std::endl;
        // padding additional Ms if needed
        for(int i = 0; i < group_count; i++)
        {
            Ms.push_back(256 + 64 * i);
        }
    }

    ck_tile::index_t M =
        std::reduce(Ms.begin(), Ms.begin() + group_count, 0, [](auto acc, auto group_m) {
            // round up to the multiple of BlockM
            return acc + (group_m + BlockM - 1) / BlockM * BlockM;
        });
    std::cout << "Total M: " << M << std::endl;
    ck_tile::index_t N = Ns[0];
    ck_tile::index_t K = Ks[0];

    ck_tile::index_t kbatch = arg_parser.get_int("split_k");

    ck_tile::index_t stride_A = 0;
    ck_tile::index_t stride_B = 0;
    ck_tile::index_t stride_C = 0;

    stride_A = ck_tile::get_default_stride(M, K, stride_A, is_row_major(a_layout));
    stride_B = ck_tile::get_default_stride(K, N * group_count, stride_B, is_row_major(b_layout));
    stride_C = ck_tile::get_default_stride(M, N, stride_C, is_row_major(c_layout));

    ck_tile::HostTensor<ADataType> a_m_k_tensor(
        ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(a_layout)));
    ck_tile::HostTensor<BDataType> b_k_n_tensor(ck_tile::HostTensor<BDataType>(
        ck_tile::host_tensor_descriptor(K, N * group_count, stride_B, is_row_major(b_layout))));
    ck_tile::HostTensor<CDataType> c_m_n_tensor(ck_tile::HostTensor<CDataType>(
        ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(c_layout))));

    ck_tile::HostTensor<AccDataType> per_token_scale(ck_tile::HostTensorDescriptor({M}, {1}));
    ck_tile::HostTensor<AccDataType> per_channel_scale(ck_tile::HostTensorDescriptor({N}, {1}));

    std::vector<ck_tile::index_t> m_indices(M);
    int indices_fill_start = 0;
    for(int i = 0; i < group_count; ++i)
    {
        int group_m        = Ms[i];
        int padded_group_m = (group_m + BlockM - 1) / BlockM * BlockM;
        for(int j = 0; j < padded_group_m; j++)
        {
            m_indices[indices_fill_start + j] = j < group_m ? i : -1; // -1 for padding
        }
        indices_fill_start += padded_group_m;
    }

    ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k_tensor);
    ck_tile::FillUniformDistribution<BDataType>{-.5f, .5f}(b_k_n_tensor);
    ck_tile::FillUniformDistribution<AccDataType>{-1.f, 1.f}(per_token_scale);
    ck_tile::FillUniformDistribution<AccDataType>{-1.f, 1.f}(per_channel_scale);

    assert(N % N_Warp_Tile == 0 &&
           "N must be divisible by N_Warp_Tile for contiguous grouped gemm");
    ck_tile::HostTensor<BDataType> b_shuffle_host =
        ck_tile::shuffle_b<FlatmmConfig, BDataType>(b_k_n_tensor);

    std::unique_ptr<ck_tile::DeviceMem> a_m_k_dev_buf(
        std::make_unique<ck_tile::DeviceMem>(a_m_k_tensor.get_element_space_size_in_bytes()));
    std::unique_ptr<ck_tile::DeviceMem> b_shfl_dev_buf(
        std::make_unique<ck_tile::DeviceMem>(b_shuffle_host.get_element_space_size_in_bytes()));
    std::unique_ptr<ck_tile::DeviceMem> c_m_n_dev_buf(
        std::make_unique<ck_tile::DeviceMem>(c_m_n_tensor.get_element_space_size_in_bytes()));

    ck_tile::DeviceMem per_token_scale_dev_buf(per_token_scale.get_element_space_size_in_bytes());
    ck_tile::DeviceMem per_channel_scale_dev_buf(
        per_channel_scale.get_element_space_size_in_bytes());

    c_m_n_dev_buf->SetZero();

    ck_tile::DeviceMem m_indices_dev_buf(M * sizeof(ck_tile::index_t));
    m_indices_dev_buf.ToDevice(m_indices.data());

    a_m_k_dev_buf->ToDevice(a_m_k_tensor.data());
    b_shfl_dev_buf->ToDevice(b_shuffle_host.data());

    per_token_scale_dev_buf.ToDevice(per_token_scale.data());
    per_channel_scale_dev_buf.ToDevice(per_channel_scale.data());

    auto per_token_scale_dev_ptr = ck_tile::FlatmmScalePointer<ScaleGranularityM>{
        static_cast<float*>(per_token_scale_dev_buf.GetDeviceBuffer())};
    auto per_channel_scale_dev_ptr = ck_tile::FlatmmScalePointer<ScaleGranularityN>{
        static_cast<float*>(per_channel_scale_dev_buf.GetDeviceBuffer())};

    ck_tile::ContiguousGroupedFlatmmHostArgs<decltype(per_token_scale_dev_ptr),
                                             decltype(per_channel_scale_dev_ptr)>
        kernal_args{static_cast<ck_tile::index_t*>(m_indices_dev_buf.GetDeviceBuffer()),
                    M,
                    N,
                    K,
                    a_m_k_dev_buf->GetDeviceBuffer(),
                    stride_A,
                    b_shfl_dev_buf->GetDeviceBuffer(),
                    stride_B,
                    {},
                    {},
                    c_m_n_dev_buf->GetDeviceBuffer(),
                    stride_C,
                    kbatch,
                    static_cast<float*>(per_token_scale_dev_buf.GetDeviceBuffer()),
                    static_cast<float*>(per_channel_scale_dev_buf.GetDeviceBuffer())};

    invoke_gemm<FlatmmConfig,
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
                decltype(per_channel_scale_dev_ptr)>(warmup, repeat, kernal_args);
    c_m_n_dev_buf->FromDevice(c_m_n_tensor.data());

    bool pass{true};
    if(arg_parser.get_int("v") == 1)
    {
        throw std::runtime_error(
            "Not support v=1 host verification in contiguous grouped gemm, use "
            "v=2 device verification instead");
    }
    else if(arg_parser.get_int("v") == 2)
    {
        BDataType* d_B;
        CDataType* d_C;
        ck_tile::hip_check_error(hipMalloc(&d_B, N * K * sizeof(BDataType)));
        ck_tile::hip_check_error(hipMalloc(&d_C, M * N * sizeof(CDataType)));
        ck_tile::hip_check_error(hipMemset(d_C, 0, M * N * sizeof(CDataType)));

        ck_tile::HostTensor<CDataType> c_gpu_ref_host(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

        ck_tile::index_t acc_m = 0;
        for(int i = 0; i < group_count; ++i)
        {
            ck_tile::index_t padded_M = (Ms[i] + BlockM - 1) / BlockM * BlockM;

            ck_tile::hip_check_error(hipMemcpy(d_B,
                                               b_k_n_tensor.data() + i * N * K,
                                               N * K * sizeof(BDataType),
                                               hipMemcpyHostToDevice));
            ck_tile::reference_gemm_gpu<ADataType,
                                        BDataType,
                                        AccDataType,
                                        CDataType,
                                        ALayout,
                                        BLayout,
                                        CLayout>(
                static_cast<ADataType*>(a_m_k_dev_buf->GetDeviceBuffer()) + acc_m * K,
                d_B,
                d_C + acc_m * N,
                padded_M,
                N,
                K,
                stride_A,
                stride_B,
                stride_C);
            acc_m += padded_M;
        }
        ck_tile::hip_check_error(hipMemcpy(
            c_gpu_ref_host.data(), d_C, M * N * sizeof(CDataType), hipMemcpyDeviceToHost));

        ck_tile::hip_check_error(hipFree(d_B));
        ck_tile::hip_check_error(hipFree(d_C));

        float rtol = 1e-3;
        float atol = 1e-3;

        pass = ck_tile::check_err(
            c_m_n_tensor, c_gpu_ref_host, "Error: Incorrect results!", rtol, atol);

        std::cout << "Relative error threshold: " << rtol << " Absolute error threshold: " << atol
                  << std::endl;
        std::cout << "The GPU veification result is: " << (pass ? "correct" : "fail") << std::endl;
    }

    return pass;
}

template <typename PrecType,
          typename FlatmmConfig,
          int ScaleGranularityM = -1,
          int ScaleGranularityN = -1,
          typename ALayout,
          typename BLayout,
          typename CLayout>
int run_masked_grouped_flatmm_example_with_layouts(
    int argc,
    char* argv[],
    const ALayout a_layout                  = ALayout{},
    const BLayout b_layout                  = BLayout{},
    [[maybe_unused]] const CLayout c_layout = CLayout{})
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
    {
        return -1;
    };

    using ADataType   = typename GemmBasicTypeConfig<PrecType>::ADataType;
    using BDataType   = typename GemmBasicTypeConfig<PrecType>::BDataType;
    using CDataType   = typename GemmBasicTypeConfig<PrecType>::CDataType;
    using AccDataType = typename GemmBasicTypeConfig<PrecType>::AccDataType;

    constexpr int BlockM = FlatmmConfig::M_Tile;

    const int group_count = arg_parser.get_int("group_count");
    const int repeat      = arg_parser.get_int("repeat");
    const int warmup      = arg_parser.get_int("warmup");

    std::vector<ck_tile::index_t> Ms = arg_parser.get_int_vec("Ms");
    std::vector<ck_tile::index_t> Ns = arg_parser.get_int_vec("Ns");
    std::vector<ck_tile::index_t> Ks = arg_parser.get_int_vec("Ks");

    if(!(int(Ms.size()) == group_count))
    {
        std::cout << "Please check the input data." << std::endl;
        // padding additional Ms if needed
        for(int i = 0; i < group_count; i++)
        {
            Ms.push_back(256 + 64 * i);
        }
    }

    ck_tile::index_t M = 4096; // Ms[0];
    ck_tile::index_t N = Ns[0];
    ck_tile::index_t K = Ks[0];

    ck_tile::index_t kbatch = arg_parser.get_int("split_k");

    ck_tile::index_t stride_A = K;
    ck_tile::index_t stride_B = K;
    ck_tile::index_t stride_C = N;

    stride_A = ck_tile::get_default_stride(group_count * M, K, stride_A, is_row_major(a_layout));
    stride_B = ck_tile::get_default_stride(K, N * group_count, stride_B, is_row_major(b_layout));
    stride_C = ck_tile::get_default_stride(group_count * M, N, stride_C, is_row_major(c_layout));

    ck_tile::HostTensor<ADataType> a_m_k_tensor(
        ck_tile::host_tensor_descriptor(group_count * M, K, stride_A, is_row_major(a_layout)));
    ck_tile::HostTensor<BDataType> b_k_n_tensor(ck_tile::HostTensor<BDataType>(
        ck_tile::host_tensor_descriptor(K, N * group_count, stride_B, is_row_major(b_layout))));
    ck_tile::HostTensor<CDataType> c_m_n_tensor(ck_tile::HostTensor<CDataType>(
        ck_tile::host_tensor_descriptor(group_count * M, N, stride_C, is_row_major(c_layout))));

    ck_tile::HostTensor<AccDataType> per_token_scale(
        ck_tile::HostTensorDescriptor({group_count * M}, {1}));
    ck_tile::HostTensor<AccDataType> per_channel_scale(
        ck_tile::HostTensorDescriptor({group_count * N}, {1}));

    std::vector<ck_tile::index_t> m_indices(group_count);
    for(int i = 0; i < group_count; ++i)
    {
        int group_m        = Ms[i];
        int padded_group_m = (group_m + BlockM - 1) / BlockM * BlockM;
        for(int j = 0; j < padded_group_m; j++)
        {
            m_indices[i] = group_m;
        }
    }

    ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k_tensor);
    ck_tile::FillUniformDistribution<BDataType>{-.5f, .5f}(b_k_n_tensor);
    ck_tile::FillUniformDistribution<AccDataType>{-1.f, 1.f}(per_token_scale);
    ck_tile::FillUniformDistribution<AccDataType>{-1.f, 1.f}(per_channel_scale);

    assert(N % N_Warp_Tile == 0 &&
           "N must be divisible by N_Warp_Tile for contiguous grouped gemm");
    ck_tile::HostTensor<BDataType> b_shuffle_host =
        ck_tile::shuffle_b<FlatmmConfig, BDataType>(b_k_n_tensor);

    std::unique_ptr<ck_tile::DeviceMem> a_m_k_dev_buf(
        std::make_unique<ck_tile::DeviceMem>(a_m_k_tensor.get_element_space_size_in_bytes()));
    std::unique_ptr<ck_tile::DeviceMem> b_shfl_dev_buf(
        std::make_unique<ck_tile::DeviceMem>(b_shuffle_host.get_element_space_size_in_bytes()));
    std::unique_ptr<ck_tile::DeviceMem> c_m_n_dev_buf(
        std::make_unique<ck_tile::DeviceMem>(c_m_n_tensor.get_element_space_size_in_bytes()));

    ck_tile::DeviceMem per_token_scale_dev_buf(per_token_scale.get_element_space_size_in_bytes());
    ck_tile::DeviceMem per_channel_scale_dev_buf(
        per_channel_scale.get_element_space_size_in_bytes());
    c_m_n_dev_buf->SetZero();

    ck_tile::DeviceMem m_indices_dev_buf(group_count * sizeof(ck_tile::index_t));
    m_indices_dev_buf.ToDevice(m_indices.data());

    a_m_k_dev_buf->ToDevice(a_m_k_tensor.data());
    b_shfl_dev_buf->ToDevice(b_shuffle_host.data());

    per_token_scale_dev_buf.ToDevice(per_token_scale.data());
    per_channel_scale_dev_buf.ToDevice(per_channel_scale.data());

    auto per_token_scale_dev_ptr = ck_tile::FlatmmScalePointer<ScaleGranularityM>{
        static_cast<float*>(per_token_scale_dev_buf.GetDeviceBuffer())};
    auto per_channel_scale_dev_ptr = ck_tile::FlatmmScalePointer<ScaleGranularityN>{
        static_cast<float*>(per_channel_scale_dev_buf.GetDeviceBuffer())};
    ck_tile::MaskedGroupedFlatmmHostArgs<decltype(per_token_scale_dev_ptr),
                                         decltype(per_channel_scale_dev_ptr)>
        kernal_args{static_cast<ck_tile::index_t*>(m_indices_dev_buf.GetDeviceBuffer()),
                    group_count,
                    M,
                    N,
                    K,
                    a_m_k_dev_buf->GetDeviceBuffer(),
                    stride_A,
                    b_shfl_dev_buf->GetDeviceBuffer(),
                    stride_B,
                    {},
                    {},
                    c_m_n_dev_buf->GetDeviceBuffer(),
                    stride_C,
                    kbatch,
                    static_cast<float*>(per_token_scale_dev_buf.GetDeviceBuffer()),
                    static_cast<float*>(per_channel_scale_dev_buf.GetDeviceBuffer())};
    int sum_val_m = 0;
    for(int gi = 0; gi < group_count; gi++)
    {
        sum_val_m += m_indices[gi];
    }

    invoke_gemm<FlatmmConfig,
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
                decltype(per_channel_scale_dev_ptr)>(warmup, repeat, sum_val_m, kernal_args);
    c_m_n_dev_buf->FromDevice(c_m_n_tensor.data());

    bool pass{true};
    if(arg_parser.get_int("v") == 1)
    {
        throw std::runtime_error(
            "Not support v=1 host verification in contiguous grouped gemm, use "
            "v=2 device verification instead");
    }
    else if(arg_parser.get_int("v") == 2)
    {
        BDataType* d_B;
        CDataType* d_C;
        ck_tile::hip_check_error(hipMalloc(&d_B, N * K * sizeof(BDataType)));
        ck_tile::hip_check_error(hipMalloc(&d_C, group_count * M * N * sizeof(CDataType)));
        ck_tile::hip_check_error(hipMemset(d_C, 0, group_count * M * N * sizeof(CDataType)));

        ck_tile::HostTensor<CDataType> c_gpu_ref_host(
            ck_tile::host_tensor_descriptor(group_count * M, N, stride_C, is_row_major(CLayout{})));
        for(int i = 0; i < group_count; ++i)
        {
            ck_tile::hip_check_error(hipMemcpy(d_B,
                                               b_k_n_tensor.data() + i * N * K,
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
                    static_cast<ADataType*>(a_m_k_dev_buf->GetDeviceBuffer()) + i * M * K,
                    d_B,
                    d_C + i * M * N,
                    m_indices[i],
                    N,
                    K,
                    stride_A,
                    stride_B,
                    stride_C);
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
                    static_cast<ADataType*>(a_m_k_dev_buf->GetDeviceBuffer()) + i * M * K,
                    d_B,
                    d_C + i * M * N,
                    m_indices[i],
                    N,
                    K,
                    stride_A,
                    stride_B,
                    stride_C,
                    ScaleGranularityM,
                    ScaleGranularityN,
                    K,
                    static_cast<float*>(per_token_scale_dev_buf.GetDeviceBuffer()) + i * M,
                    static_cast<float*>(per_channel_scale_dev_buf.GetDeviceBuffer())) +
                    i* N;
            }
            ck_tile::hip_check_error(hipMemcpy(c_gpu_ref_host.data() + i * M * N,
                                               d_C + i * M * N,
                                               M * N * sizeof(CDataType),
                                               hipMemcpyDeviceToHost));
        }

        ck_tile::hip_check_error(hipFree(d_B));
        ck_tile::hip_check_error(hipFree(d_C));

        float rtol = 1e-3;
        float atol = 1e-3;

        pass = ck_tile::check_err(
            c_m_n_tensor, c_gpu_ref_host, "Error: Incorrect results!", rtol, atol);

        std::cout << "Relative error threshold: " << rtol << " Absolute error threshold: " << atol
                  << std::endl;
        std::cout << "The GPU veification result is: " << (pass ? "correct" : "fail") << std::endl;
    }

    return pass;
}
