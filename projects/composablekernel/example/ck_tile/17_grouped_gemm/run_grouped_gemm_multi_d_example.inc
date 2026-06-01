// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

struct MultiplyMultiply
{
    template <typename E, typename C, typename D0, typename D1>
    CK_TILE_HOST_DEVICE auto operator()(E& e, const C& c, const D0& d0, const D1& d1) const -> void
    {
        const float x0_f = ck_tile::type_convert<float>(c) * ck_tile::type_convert<float>(d0) *
                           ck_tile::type_convert<float>(d1);

        e = ck_tile::type_convert<E>(x0_f);
    }
};

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <typename ADataType,
          typename BDataType,
          typename D0DataType,
          typename EDataType,
          typename AccDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeTypeAB =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;

    using ComputeType =
        std::conditional_t<sizeof(ComputeTypeAB) < sizeof(D0DataType), ComputeTypeAB, D0DataType>;
    // Calculate thresholds
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, EDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));

    const auto atol = ck_tile::get_absolute_threshold<ComputeType, EDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));

    // Calculate error due to split_k accumulation
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<EDataType, EDataType, EDataType>(kbatch);

    const auto atol_split_k = ck_tile::get_absolute_threshold<EDataType, EDataType, EDataType>(
        max_accumulated_value, kbatch);

    // Use higher threshold
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename EDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename CDEElementWise>
float invoke_gemm(int n_warmup,
                  int n_repeat,
                  int group_count,
                  const std::vector<grouped_gemm_multi_d_kargs>& args)
{
    // Workspace memory allocated to hold the gemm descriptions.
    ck_tile::DeviceMem gemm_workspace;
    gemm_workspace.Realloc(get_workspace_size(args));

    float ave_time = 0;
    if constexpr(!GemmConfig::Persistent)
    {
        ave_time = grouped_gemm_multi_d<GemmConfig,
                                        ADataType,
                                        BDataType,
                                        DsDataType,
                                        AccDataType,
                                        EDataType,
                                        ALayout,
                                        BLayout,
                                        DsLayout,
                                        ELayout,
                                        CDEElementWise>(
            args,
            ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat},
            gemm_workspace.GetDeviceBuffer());
    }
    else
    {
        std::vector<ck_tile::GemmTransKernelArg<NumDTensor>> kargs;
        void* kargs_ptr = gemm_workspace.GetDeviceBuffer();
        for(const auto& arg : args)
        {
            kargs.emplace_back(ck_tile::UniversalGemmKernelArgs<1, 1, NumDTensor>{{arg.a_ptr},
                                                                                  {arg.b_ptr},
                                                                                  arg.ds_ptr,
                                                                                  arg.e_ptr,
                                                                                  arg.M,
                                                                                  arg.N,
                                                                                  arg.K,
                                                                                  {arg.stride_A},
                                                                                  {arg.stride_B},
                                                                                  arg.stride_Ds,
                                                                                  arg.stride_E,
                                                                                  arg.k_batch});
        }
        const auto stream = ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat};
        HIP_CHECK_ERROR(
            hipMemcpyWithStream(kargs_ptr,
                                kargs.data(),
                                kargs.size() * sizeof(ck_tile::GemmTransKernelArg<NumDTensor>),
                                hipMemcpyHostToDevice,
                                stream.stream_id_));
        ave_time = grouped_gemm_multi_d_tileloop<GemmConfig,
                                                 ADataType,
                                                 BDataType,
                                                 DsDataType,
                                                 AccDataType,
                                                 EDataType,
                                                 ALayout,
                                                 BLayout,
                                                 DsLayout,
                                                 ELayout,
                                                 CDEElementWise>(stream, group_count, kargs_ptr);
    }
    return ave_time;
}

template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename D0DataType,
          typename D1DataType,
          typename AccDataType,
          typename EDataType,
          typename ALayout,
          typename BLayout,
          typename D0Layout,
          typename D1Layout,
          typename ELayout>
int run_grouped_gemm_multi_d_example_with_layouts(int argc,
                                                  char* argv[],
                                                  const ALayout a_layout   = ALayout{},
                                                  const BLayout b_layout   = BLayout{},
                                                  const D0Layout d0_layout = D0Layout{},
                                                  const D1Layout d1_layout = D1Layout{},
                                                  const ELayout e_layout   = ELayout{})
{
    auto [result, arg_parser] = create_args(argc, argv);

    using CDElementWise = MultiplyMultiply;
    using DsLayout      = ck_tile::tuple<D0Layout, D1Layout>;
    using DsDataType    = ck_tile::tuple<D0DataType, D1DataType>;

    auto valid_input_data = [&](int group_count, const auto&... args) {
        return group_count != 0 && ((args.size() == static_cast<size_t>(group_count)) && ...);
    };

    const int group_count = arg_parser.get_int("group_count");
    const int repeat      = arg_parser.get_int("repeat");
    const int warmup      = arg_parser.get_int("warmup");
    const int kbatch      = arg_parser.get_int("kbatch");
    bool validate         = arg_parser.get_bool("validate");

    if(kbatch > 1 && validate && warmup + repeat > 1)
    {
        std::cout << "WARNING: Data validation enabled with SplitK and more than"
                  << "1 warmup/repeat. Disabling validation." << std::endl;
        validate = false;
    }

    std::vector<ck_tile::index_t> Ms        = arg_parser.get_int_vec("Ms");
    std::vector<ck_tile::index_t> Ns        = arg_parser.get_int_vec("Ns");
    std::vector<ck_tile::index_t> Ks        = arg_parser.get_int_vec("Ks");
    std::vector<ck_tile::index_t> stride_As = arg_parser.get_int_vec("stride_As");
    std::vector<ck_tile::index_t> stride_Bs = arg_parser.get_int_vec("stride_Bs");
    std::vector<ck_tile::index_t> stride_D0 = arg_parser.get_int_vec("stride_Ds");
    std::vector<ck_tile::index_t> stride_D1 = arg_parser.get_int_vec("stride_Ds");
    std::vector<ck_tile::index_t> stride_Es = arg_parser.get_int_vec("stride_Es");

    if(!valid_input_data(
           group_count, Ms, Ns, Ks, stride_As, stride_Bs, stride_D0, stride_D1, stride_Es))
    {
        std::cout << "Please check the input data. Default values will be used." << std::endl;
        std::cout << "Default values: Ms (256, 512, 768, 1024..), Ns (256, 768, 1280..), Ks (512, "
                     "896, 1280..), stride_As (Ks), stride_Bs (Ks), stride_D0 (Ns), stride_D1 "
                     "(Ns), stride_Es (Ns)"
                  << std::endl;
        for(int i = 0; i < group_count; i++)
        {
            Ms.push_back(256 + 256 * i);
            Ns.push_back(256 + 512 * i);
            Ks.push_back(512 + 384 * i);

            stride_As.push_back(Ks[i]);
            stride_Bs.push_back(Ks[i]);
            stride_D0.push_back(Ns[i]);
            stride_D1.push_back(Ns[i]);
            stride_Es.push_back(Ns[i]);
        }
    }

    std::vector<ck_tile::HostTensor<ADataType>> a_m_k_tensors;
    std::vector<ck_tile::HostTensor<BDataType>> b_k_n_tensors;
    std::vector<ck_tile::HostTensor<D0DataType>> d0_m_n_tensors;
    std::vector<ck_tile::HostTensor<D1DataType>> d1_m_n_tensors;
    std::vector<ck_tile::HostTensor<EDataType>> e_m_n_tensors;

    a_m_k_tensors.reserve(group_count);
    b_k_n_tensors.reserve(group_count);
    d0_m_n_tensors.reserve(group_count);
    d1_m_n_tensors.reserve(group_count);
    e_m_n_tensors.reserve(group_count);

    std::vector<std::unique_ptr<ck_tile::DeviceMem>> a_m_k_dev_buf;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> b_k_n_dev_buf;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> d0_m_n_dev_buf;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> d1_m_n_dev_buf;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> e_m_n_dev_buf;

    a_m_k_dev_buf.reserve(group_count);
    b_k_n_dev_buf.reserve(group_count);
    d0_m_n_dev_buf.reserve(group_count);
    d1_m_n_dev_buf.reserve(group_count);
    e_m_n_dev_buf.reserve(group_count);

    std::vector<grouped_gemm_multi_d_kargs> gemm_descs;
    gemm_descs.reserve(group_count);

    for(int i = 0; i < group_count; ++i)
    {

        const ck_tile::index_t M = Ms[i];
        const ck_tile::index_t N = Ns[i];
        const ck_tile::index_t K = Ks[i];

        stride_As[i] = ck_tile::get_default_stride(M, K, stride_As[i], is_row_major(a_layout));
        stride_Bs[i] = ck_tile::get_default_stride(K, N, stride_Bs[i], is_row_major(b_layout));

        stride_D0[i] = ck_tile::get_default_stride(M, N, stride_D0[i], is_row_major(d0_layout));
        stride_D1[i] = ck_tile::get_default_stride(M, N, stride_D1[i], is_row_major(d1_layout));

        stride_Es[i] = ck_tile::get_default_stride(M, N, stride_Es[i], is_row_major(e_layout));

        a_m_k_tensors.push_back(ck_tile::HostTensor<ADataType>(
            ck_tile::host_tensor_descriptor(M, K, stride_As[i], is_row_major(a_layout))));
        b_k_n_tensors.push_back(ck_tile::HostTensor<BDataType>(
            ck_tile::host_tensor_descriptor(K, N, stride_Bs[i], is_row_major(b_layout))));

        d0_m_n_tensors.push_back(ck_tile::HostTensor<D0DataType>(
            ck_tile::host_tensor_descriptor(M, N, stride_D0[i], is_row_major(d0_layout))));
        d1_m_n_tensors.push_back(ck_tile::HostTensor<D1DataType>(
            ck_tile::host_tensor_descriptor(M, N, stride_D1[i], is_row_major(d1_layout))));

        e_m_n_tensors.push_back(ck_tile::HostTensor<EDataType>(
            ck_tile::host_tensor_descriptor(M, N, stride_Es[i], is_row_major(e_layout))));

        std::cout << "gemm[" << i << "]" << " a_m_k: " << a_m_k_tensors[i].mDesc
                  << " b_k_n: " << b_k_n_tensors[i].mDesc << " d0_m_n: " << d0_m_n_tensors[i].mDesc
                  << " d1_m_n: " << d1_m_n_tensors[i].mDesc << " e_m_n: " << e_m_n_tensors[i].mDesc
                  << std::endl;

        ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k_tensors[i]);
        ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n_tensors[i]);
        ck_tile::FillUniformDistribution<D0DataType>{-1.f, 1.f}(d0_m_n_tensors[i]);
        ck_tile::FillUniformDistribution<D1DataType>{-1.f, 1.f}(d1_m_n_tensors[i]);

        a_m_k_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(a_m_k_tensors[i]));

        b_k_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(b_k_n_tensors[i]));

        d0_m_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(d0_m_n_tensors[i]));
        d1_m_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(d1_m_n_tensors[i]));
        e_m_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(e_m_n_tensors[i]));

        e_m_n_dev_buf[i]->SetZero();

        const void* p_a = a_m_k_dev_buf[i]->GetDeviceBuffer();
        const void* p_b = b_k_n_dev_buf[i]->GetDeviceBuffer();
        void* p_e       = e_m_n_dev_buf[i]->GetDeviceBuffer();

        std::array<const void*, DsDataType::size()> ds_ptr_buf = {
            d0_m_n_dev_buf[i]->GetDeviceBuffer(), d1_m_n_dev_buf[i]->GetDeviceBuffer()};
        std::array<ck_tile::index_t, DsDataType::size()> stridesDs = {stride_D0[i], stride_D1[i]};

        gemm_descs.push_back({p_a,
                              p_b,
                              ds_ptr_buf,
                              p_e,
                              kbatch,
                              M,
                              N,
                              K,
                              stride_As[i],
                              stride_Bs[i],
                              stridesDs,
                              stride_Es[i]});
    }

    float ave_time = invoke_gemm<GemmConfig,
                                 ADataType,
                                 BDataType,
                                 DsDataType,
                                 AccDataType,
                                 EDataType,
                                 ALayout,
                                 BLayout,
                                 DsLayout,
                                 ELayout,
                                 CDElementWise>(warmup, repeat, group_count, gemm_descs);

    std::string op_name{"Grouped Gemm Multiple-D"};

    std::size_t flop = 0, num_btype = 0;
    for(int j = 0; j < group_count; ++j)
    {
        flop += std::size_t(2) * gemm_descs[j].M * gemm_descs[j].N * gemm_descs[j].K;
        ck_tile::static_for<0, DsDataType::size(), 1>{}([&](auto i) {
            num_btype += sizeof(ck_tile::remove_cvref_t<std::tuple_element_t<i, DsDataType>>) *
                         gemm_descs[j].M * gemm_descs[j].N;
            flop += sizeof(ck_tile::remove_cvref_t<std::tuple_element_t<i, DsDataType>>) *
                    gemm_descs[j].M * gemm_descs[j].N;
        });

        num_btype += sizeof(ADataType) * gemm_descs[j].M * gemm_descs[j].K +
                     sizeof(BDataType) * gemm_descs[j].K * gemm_descs[j].N +
                     sizeof(EDataType) * gemm_descs[j].M * gemm_descs[j].N;
    }

    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
              << gb_per_sec << " GB/s, " << op_name << std::endl;

    std::vector<ck_tile::HostTensor<EDataType>> e_m_n_host_refs;
    e_m_n_host_refs.reserve(group_count);

    // copy e_m_n_tensors result from device to host and initialize host tensors to zero
    for(int i = 0; i < group_count; i++)
    {
        e_m_n_dev_buf[i]->FromDevice(e_m_n_tensors[i].data());
    }

    bool pass{true};
    if(validate)
    {
        for(int i = 0; i < group_count; ++i)
        {
            e_m_n_host_refs.push_back(ck_tile::HostTensor<EDataType>(
                host_tensor_descriptor(Ms[i], Ns[i], stride_Es[i], is_row_major(e_layout))));

            e_m_n_host_refs[i].SetZero();

            ck_tile::reference_gemm_multiple_d<ADataType,
                                               BDataType,
                                               DsDataType,
                                               AccDataType,
                                               EDataType,
                                               CDElementWise>(
                a_m_k_tensors[i],
                b_k_n_tensors[i],
                {d0_m_n_tensors[i], d1_m_n_tensors[i]},
                e_m_n_host_refs[i]);

            const float max_accumulated_value =
                *std::max_element(e_m_n_host_refs[i].mData.begin(), e_m_n_host_refs[i].mData.end());

            const auto rtol_atol =
                calculate_rtol_atol<ADataType, BDataType, D0DataType, EDataType, AccDataType>(
                    Ks[i], 1, max_accumulated_value);

            pass &=
                ck_tile::check_err(e_m_n_tensors[i],
                                   e_m_n_host_refs[i],
                                   "Error: Incorrect results! in group [" + std::to_string(i) + "]",
                                   rtol_atol.at(ck_tile::number<0>{}),
                                   rtol_atol.at(ck_tile::number<1>{}));

            std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                      << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                      << std::endl;
        }
        std::cout << "The CPU verification result is: " << (pass ? "correct" : "fail") << std::endl;
    }

    if(arg_parser.get_int("json") == 1)
    {
        dump_grouped_gemm_json_results<ALayout, BLayout, ELayout>(arg_parser.get_str("jsonfile"),
                                                                  op_name,
                                                                  group_count,
                                                                  pass,
                                                                  ave_time,
                                                                  tflops,
                                                                  gb_per_sec);
    }

    return pass;
}

template <typename GemmConfig, typename PrecType>
int run_gemm_multi_d_example_prec_type(
    std::string a_layout, std::string b_layout, std::string ds_layout, int argc, char* argv[])
{
    using Row   = ck_tile::tensor_layout::gemm::RowMajor;
    using Col   = ck_tile::tensor_layout::gemm::ColumnMajor;
    using Types = GemmMultiDTypeConfig<PrecType>;

    using ADataType   = typename Types::ADataType;
    using BDataType   = typename Types::BDataType;
    using D0DataType  = typename Types::D0DataType;
    using D1DataType  = typename Types::D1DataType;
    using AccDataType = typename Types::AccDataType;
    using EDataType   = typename Types::EDataType;

    if(a_layout == "R" && b_layout == "C" && ds_layout == "R")
    {
        return run_grouped_gemm_multi_d_example_with_layouts<GemmConfig,
                                                             ADataType,
                                                             BDataType,
                                                             D0DataType,
                                                             D1DataType,
                                                             AccDataType,
                                                             EDataType>(
            argc, argv, Row{}, Col{}, Row{}, Row{}, Row{});
    }
    else
    {
        throw std::runtime_error("Unsupported data layout configuration for provided tensors!");
    }
}

template <typename GemmConfig>
int run_grouped_gemm_multi_d_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
    {
        return -1;
    }
    const std::string a_layout  = arg_parser.get_str("a_layout");
    const std::string b_layout  = arg_parser.get_str("b_layout");
    const std::string ds_layout = arg_parser.get_str("ds_layout");
    const std::string data_type = arg_parser.get_str("prec");

    if(data_type == "fp16")
    {
        return run_gemm_multi_d_example_prec_type<GemmConfig, ck_tile::half_t>(
            a_layout, b_layout, ds_layout, argc, argv);
    }
    else if(data_type == "bf16")
    {
        return run_gemm_multi_d_example_prec_type<GemmConfig, ck_tile::bf16_t>(
            a_layout, b_layout, ds_layout, argc, argv);
    }
    else
    {
        throw std::runtime_error(
            "Unsupported data type configuration. Only fp16 and bf16 are supported.");
    }
}
