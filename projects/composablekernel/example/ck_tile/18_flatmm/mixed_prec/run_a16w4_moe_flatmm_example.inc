// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

template <typename FlatmmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDatatype,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ck_tile::MoeFlatmmKind kind,
          typename CDEElementWise = ck_tile::element_wise::PassThrough,
          typename MoeHostArgs>
float invoke_a16w4_moe_gemm(int n_warmup, int n_repeat, const MoeHostArgs& args)
{
    float ave_time = a16w4_moe_gemm<FlatmmConfig,
                                    ADataType,
                                    BDataType,
                                    DsDatatype,
                                    AccDataType,
                                    CDataType,
                                    ALayout,
                                    BLayout,
                                    DsLayout,
                                    ELayout,
                                    kind,
                                    CDEElementWise>(
        args, ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat, true, true, 50});

    std::string op_name{"Moe Gemm"};

    constexpr int PackedSize = ck_tile::numeric_traits<BDataType>::PackedSize;

    std::size_t flop     = std::size_t(2) * args.M * args.N * args.K;
    std::size_t num_byte = sizeof(ADataType) * args.M * args.K +
                           sizeof(BDataType) * args.N * args.K / PackedSize +
                           sizeof(CDataType) * args.M * args.N;
    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
              << gb_per_sec << " GB/s, " << op_name << std::endl;

    return ave_time;
}

template <typename PrecActType,
          typename PrecWeightType,
          typename FlatmmConfig,
          ck_tile::MoeFlatmmKind kind,
          typename ALayout,
          typename BLayout,
          typename CLayout>
int run_a16w4_moe_gemm_example_with_layouts(int argc,
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

    using ADataType = PrecActType;
    using BDataType = PrecWeightType;
    using ADataType = PrecActType;
    using BDataType = PrecWeightType;
    using CDataType =
        std::conditional_t<kind == ck_tile::MoeFlatmmKind::kFFN_gemm1_split_k, float, PrecActType>;
    using AccDataType = float;

    using ScaleType = ck_tile::e8m0_t;

    constexpr int ScaleGranularityN = 1;
    constexpr int ScaleGranularityK = 32;

    const ck_tile::index_t N          = arg_parser.get_int("N");
    const ck_tile::index_t K          = arg_parser.get_int("K");
    ck_tile::index_t stride_A         = arg_parser.get_int("stride_A");
    ck_tile::index_t stride_B         = arg_parser.get_int("stride_B");
    ck_tile::index_t stride_C         = arg_parser.get_int("stride_C");
    ck_tile::index_t init_method      = arg_parser.get_int("init");
    const ck_tile::index_t num_tokens = arg_parser.get_int("NumTokens");
    const ck_tile::index_t topk       = arg_parser.get_int("TopK");
    const ck_tile::index_t warmup     = arg_parser.get_int("warmup");
    const ck_tile::index_t repeat     = arg_parser.get_int("repeat");
    const ck_tile::index_t experts    = arg_parser.get_int("experts");
    const ck_tile::index_t k_batch    = arg_parser.get_int("k_batch");

    // TODO: replace the magic declaration
    const ck_tile::index_t MPerBlock = FlatmmConfig::M_Tile;

    ck_tile::index_t sorted_tile_num = (num_tokens + MPerBlock - 1) / MPerBlock * MPerBlock * topk;
    ck_tile::index_t valid_tile_num  = sorted_tile_num;
    ck_tile::index_t sorted_size     = sorted_tile_num * MPerBlock;

    const ck_tile::index_t M       = sorted_tile_num * MPerBlock;
    const ck_tile::index_t outputN = kind == ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up ? N / 2 : N;

    static_assert(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::ColumnMajor>);
    constexpr bool IsInputGemm = kind != ck_tile::MoeFlatmmKind::kFFN_gemm2;

    stride_A = ck_tile::get_default_stride(
        IsInputGemm ? num_tokens : num_tokens * topk, K, stride_A, is_row_major(a_layout));
    stride_B = ck_tile::get_default_stride(K, N, stride_B, is_row_major(b_layout));
    stride_C = ck_tile::get_default_stride(
        IsInputGemm ? num_tokens * topk : num_tokens, outputN, stride_C, is_row_major(CLayout{}));

    auto a_m_k_tensor = ck_tile::HostTensor<ADataType>(ck_tile::host_tensor_descriptor(
        IsInputGemm ? num_tokens : num_tokens * topk, K, stride_A, is_row_major(a_layout)));
    auto b_k_n_tensor = ck_tile::HostTensor<BDataType>(
        is_row_major(b_layout)
            ? ck_tile::host_tensor_descriptor(experts * N, K, stride_B, is_row_major(b_layout))
            : ck_tile::host_tensor_descriptor(K, experts * N, stride_B, is_row_major(b_layout)));
    auto c_m_n_tensor = ck_tile::HostTensor<CDataType>(ck_tile::host_tensor_descriptor(
        IsInputGemm ? num_tokens * topk : num_tokens, outputN, stride_C, is_row_major(CLayout{})));

    ck_tile::HostTensor<ScaleType> scale_b(ck_tile::HostTensorDescriptor(
        {K * experts / ScaleGranularityK, N / ScaleGranularityN}, {N / ScaleGranularityN, 1}));

    if(init_method == 0)
    {
        ck_tile::FillUniformDistribution<ADataType>{0.0f, 1.0f}(a_m_k_tensor);
        ck_tile::FillUniformDistribution<BDataType>{-.5f, .5f}(b_k_n_tensor);
        ck_tile::FillUniformDistribution<ScaleType>{0.f, 1.f}(scale_b);
    }
    else
    {
        ck_tile::FillUniformDistribution<ADataType>{1.0f, 1.0f}(a_m_k_tensor);
        ck_tile::FillUniformDistribution<BDataType>{1.0f, 1.0f}(b_k_n_tensor);
        ck_tile::FillUniformDistribution<ScaleType>{1.0f, 1.0f}(scale_b);
    }

    ck_tile::HostTensor<BDataType> b_shuffle_host(
        ck_tile::host_tensor_descriptor(K, experts * N, stride_B, is_row_major(b_layout)));
    shuffle_mxfp4_weight<FlatmmConfig, kind>(
        b_k_n_tensor.begin(), b_shuffle_host.begin(), experts, N, K);

    ck_tile::HostTensor<ScaleType> scale_b_shuffle =
        shuffle_mxfp4_scale<FlatmmConfig, kind>(scale_b, experts);
    ck_tile::DeviceMem scale_b_shuffle_dev_buf(scale_b_shuffle.get_element_space_size_in_bytes());

    std::cout << "moe_flatmm:" << "\n  num_experts: " << experts << "\n  num_tokens: " << num_tokens
              << "\n  topk: " << topk << "\n  sorted_tile_num: " << sorted_tile_num
              << "\n  problem_n: " << N << "\n  problem_k: " << K
              << "\n  a_m_k: " << a_m_k_tensor.mDesc << "\n  b_k_n: " << b_k_n_tensor.mDesc
              << "\n  b_shuffle: " << b_shuffle_host.mDesc << "\n  c_m_n: " << c_m_n_tensor.mDesc
              << std::endl;

    ck_tile::HostTensor<ck_tile::index_t> expert_ids(
        ck_tile::HostTensorDescriptor({sorted_tile_num}, {1}));
    ck_tile::HostTensor<ck_tile::index_t> sorted_token_ids(
        ck_tile::HostTensorDescriptor({sorted_size}, {1}));
    ck_tile::HostTensor<AccDataType> expert_weight(
        ck_tile::HostTensorDescriptor({sorted_size}, {1}));
    ck_tile::HostTensor<ck_tile::index_t> max_token_id(
        ck_tile::HostTensorDescriptor({1 + sorted_tile_num}));
    ck_tile::HostTensor<AccDataType> expert_bias(ck_tile::HostTensorDescriptor({experts * N}, {1}));

    if(init_method == 0)
    {
        // for verification only, no need to satify weight normalization
        ck_tile::FillUniformDistribution<AccDataType>{0.0f, 1.0f}(expert_weight);
        ck_tile::FillUniformDistribution<AccDataType>{-1.0f, 1.0f}(expert_bias);
    }
    else
    {
        ck_tile::FillUniformDistribution<AccDataType>{1.0f, 1.0f}(expert_weight);
        ck_tile::FillUniformDistribution<AccDataType>{0.0f, 0.0f}(expert_bias);
    }

    max_token_id.mData = {valid_tile_num * MPerBlock, 0, 1, 2, 3, 4, 6, 7, 8, 8};
    // int eids[]         = {0, 1, 2, 3, 4, 4, 5, 6, 3, 3, 3, 3}; // {2, 1, 1, 2, 2, 2, 1, 2}

    for(int i = 0; i < sorted_tile_num; i++)
    {
        expert_ids.mData[i] = i / ((valid_tile_num + experts - 1) / experts);
    }

    int token_per_tile = (num_tokens * topk + valid_tile_num - 1) / valid_tile_num;
    // int token_per_tile = num_tokens * topk / valid_tile_num;
    int tokenid = 0;
    // sorted_token_ids.mData[0] = 0;
    for(int i = 0; i < sorted_tile_num * MPerBlock; i++)
    {
        int tile_off = i % MPerBlock;
        if(tile_off < token_per_tile && tokenid < num_tokens * topk)
        {
            sorted_token_ids.mData[i] = (tokenid % num_tokens) | ((tokenid / num_tokens) << 24);
            tokenid++;
        }
        else
        {
            sorted_token_ids.mData[i] = num_tokens;
        }
    }

    ck_tile::DeviceMem a_m_k_dev_buf{a_m_k_tensor.get_element_space_size_in_bytes()};
    ck_tile::DeviceMem b_origin_dev_buf{b_k_n_tensor.get_element_space_size_in_bytes()};
    ck_tile::DeviceMem b_shuffle_dev_buf{b_shuffle_host.get_element_space_size_in_bytes()};
    ck_tile::DeviceMem c_m_n_dev_buf{c_m_n_tensor.get_element_space_size_in_bytes()};

    a_m_k_dev_buf.ToDevice(a_m_k_tensor.data());
    b_origin_dev_buf.ToDevice(b_k_n_tensor.data());
    b_shuffle_dev_buf.ToDevice(b_shuffle_host.data());
    c_m_n_dev_buf.SetZero();
    c_m_n_tensor.SetZero();

    ck_tile::DeviceMem sorted_token_ids_dev{sorted_token_ids.get_element_space_size_in_bytes()};
    ck_tile::DeviceMem expert_ids_dev{expert_ids.get_element_space_size_in_bytes()};
    ck_tile::DeviceMem max_token_id_dev{max_token_id.get_element_space_size_in_bytes()};
    ck_tile::DeviceMem expert_weight_dev{expert_weight.get_element_space_size_in_bytes()};
    ck_tile::DeviceMem expert_bias_dev{expert_bias.get_element_space_size_in_bytes()};

    sorted_token_ids_dev.ToDevice(sorted_token_ids.data());
    expert_ids_dev.ToDevice(expert_ids.data());
    max_token_id_dev.ToDevice(max_token_id.data());
    expert_weight_dev.ToDevice(expert_weight.data());
    expert_bias_dev.ToDevice(expert_bias.data());
    scale_b_shuffle_dev_buf.ToDevice(scale_b_shuffle.data());

    const ck_tile::index_t* p_sorted_token_ids_dev =
        static_cast<ck_tile::index_t*>(sorted_token_ids_dev.GetDeviceBuffer());
    const ck_tile::index_t* p_expert_ids_dev =
        static_cast<ck_tile::index_t*>(expert_ids_dev.GetDeviceBuffer());
    const ck_tile::index_t* p_max_token_id_dev =
        static_cast<ck_tile::index_t*>(max_token_id_dev.GetDeviceBuffer());
    const AccDataType* p_sorted_expert_weight_dev =
        static_cast<AccDataType*>(expert_weight_dev.GetDeviceBuffer());

    auto scale_b_shuffle_dev_ptr =
        ck_tile::FlatmmScalePointer<ScaleGranularityN, ScaleGranularityK, ScaleType>{
            static_cast<ScaleType*>(scale_b_shuffle_dev_buf.GetDeviceBuffer()),
            N / ScaleGranularityN};
    auto exp_bias_dev_ptr = ck_tile::FlatmmScalePointer<1>{
        static_cast<float*>(expert_bias_dev.GetDeviceBuffer()), experts * N};

    using MoeFlatmmArgs = ck_tile::MoeFlatmmHostArgs<
        ck_tile::FlatmmScalePointer<-1>,
        ck_tile::FlatmmScalePointer<ScaleGranularityN, ScaleGranularityK, ScaleType>,
        ck_tile::FlatmmScalePointer<1>>;
    MoeFlatmmArgs gemm_desc{p_sorted_token_ids_dev,
                            p_sorted_expert_weight_dev,
                            p_expert_ids_dev,
                            p_max_token_id_dev,
                            a_m_k_dev_buf.GetDeviceBuffer(),
                            b_shuffle_dev_buf.GetDeviceBuffer(),
                            c_m_n_dev_buf.GetDeviceBuffer(),
                            num_tokens,
                            experts,
                            topk,
                            k_batch, // k_batch
                            M,
                            N,
                            K,
                            stride_A,
                            stride_B,
                            stride_C,
                            nullptr,
                            scale_b_shuffle_dev_ptr,
                            exp_bias_dev_ptr};

    invoke_a16w4_moe_gemm<FlatmmConfig,
                          ADataType,
                          BDataType,
                          ck_tile::tuple<>,
                          AccDataType,
                          CDataType,
                          ALayout,
                          BLayout,
                          ck_tile::tuple<>,
                          CLayout,
                          kind>(warmup, repeat, gemm_desc);

    c_m_n_dev_buf.FromDevice(c_m_n_tensor.data());

    bool pass{true};
    if(arg_parser.get_int("validate"))
    {
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(IsInputGemm ? num_tokens * topk : num_tokens,
                                            outputN,
                                            stride_C,
                                            is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        ck_tile::HostTensor<AccDataType> scale_A(
            ck_tile::HostTensorDescriptor({1, K / ScaleGranularityK}, {1, 1}));

        // scaleA = 1 has no effect on the result
        ck_tile::FillUniformDistribution<AccDataType>{1.f, 1.f}(scale_A);
        ck_tile::DeviceMem scale_A_dev_buf(scale_A.get_element_space_size_in_bytes());
        scale_A_dev_buf.ToDevice(scale_A.data());

        // convert scale_b from e8m0 to float
        ck_tile::HostTensor<AccDataType> scale_b_float(ck_tile::HostTensorDescriptor(
            {K * experts / ScaleGranularityK, N / ScaleGranularityN}, {N / ScaleGranularityN, 1}));
        std::copy(scale_b.begin(), scale_b.end(), scale_b_float.begin());
        ck_tile::DeviceMem scale_b_float_dev_buf(scale_b_float.get_element_space_size_in_bytes());
        scale_b_float_dev_buf.ToDevice(scale_b_float.data());

        std::unique_ptr<ck_tile::DeviceMem> c_m_n_ref_buf =
            std::make_unique<ck_tile::DeviceMem>(c_m_n_tensor.get_element_space_size_in_bytes());
        c_m_n_ref_buf->SetZero();

        ck_tile::reference_moe_gemm_gpu<ADataType,
                                        BDataType,
                                        AccDataType,
                                        CDataType,
                                        ALayout,
                                        BLayout,
                                        CLayout,
                                        static_cast<int>(kind),
                                        ck_tile::moe::Swiglu>(
            p_sorted_token_ids_dev,
            p_expert_ids_dev,
            p_max_token_id_dev,
            static_cast<const ADataType*>(a_m_k_dev_buf.GetDeviceBuffer()),
            static_cast<const BDataType*>(b_origin_dev_buf.GetDeviceBuffer()),
            static_cast<CDataType*>(c_m_n_ref_buf->GetDeviceBuffer()),
            p_sorted_expert_weight_dev,
            num_tokens,
            MPerBlock,
            topk,
            M,
            N,
            K,
            stride_A,
            stride_B,
            stride_C,
            M,
            1,
            ScaleGranularityK,
            static_cast<float*>(scale_A_dev_buf.GetDeviceBuffer()),
            static_cast<float*>(scale_b_float_dev_buf.GetDeviceBuffer()),
            static_cast<float*>(expert_bias_dev.GetDeviceBuffer()));

        c_m_n_ref_buf->FromDevice(c_m_n_host_ref.data());

        const float rtol = std::is_same_v<ADataType, ck_tile::half_t> && IsInputGemm ? 1e-3 : 1e-2;
        const float atol = std::is_same_v<ADataType, ck_tile::half_t> && IsInputGemm ? 1e-3 : 1e-2;

        pass = ck_tile::check_err(
            c_m_n_tensor, c_m_n_host_ref, "Error: Incorrect results!", rtol, atol);

        std::cout << "Relative error threshold: " << rtol << " Absolute error threshold: " << atol
                  << std::endl;
        std::cout << "The CPU verification result is:" << (pass ? "correct" : "fail") << std::endl;
    }

    return pass;
}
