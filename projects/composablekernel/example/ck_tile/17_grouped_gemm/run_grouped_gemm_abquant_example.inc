// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

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

template <typename GemmConfig,
          typename ADataType,
          typename AQDataType,
          typename BDataType,
          typename BQDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename AQLayout,
          typename BLayout,
          typename BQLayout,
          typename CLayout,
          typename AQuantGroupSize,
          typename BQuantGroupSize,
          ck_tile::QuantType QuantMode = ck_tile::QuantType::ABQuantGrouped,
          typename CDEElementWise      = ck_tile::element_wise::PassThrough>
float invoke_abquant_gemm(int n_warmup,
                          int n_repeat,
                          int group_count,
                          const std::vector<grouped_gemm_kargs>& args)
{
    // Workspace memory allocated to hold the gemm descriptions.
    ck_tile::DeviceMem gemm_workspace;
    gemm_workspace.Realloc(get_workspace_size(args));

    float ave_time = 0;

    if constexpr(!GemmConfig::Persistent)
    {
        ave_time = grouped_gemm_abquant<GemmConfig,
                                        ALayout,
                                        AQLayout,
                                        BLayout,
                                        BQLayout,
                                        CLayout,
                                        ADataType,
                                        AQDataType,
                                        BDataType,
                                        BQDataType,
                                        AccDataType,
                                        CDataType,
                                        AQuantGroupSize,
                                        BQuantGroupSize,
                                        QuantMode>(
            args,
            ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat},
            gemm_workspace.GetDeviceBuffer());
    }
    else
    {
        // NOTE: With the persistent TileLoop kernel, we do not necessarily need to have
        // the gemm problems known on the host. Instead, we can just pass the pointer
        // to the kernel and let the workgroups figure out which tiles to work on.
        // This is useful when the gemm problems are generated dynamically.
        // In this example however, we generate the `kargs` using the known gemm_descs,
        // and copy the gemm descriptions to the device memory.
        // The contents of the memory pointed to by `kargs_ptr` pointer could be
        // written by e.g. another kernel from earlier stage.
        std::vector<ck_tile::QuantGemmTransKernelArg> kargs;
        void* kargs_ptr = gemm_workspace.GetDeviceBuffer();
        if(args[0].k_batch != 1)
        {
            throw std::runtime_error("Split-K not supported yet for persistent kernel");
        }

        for(const auto& arg : args)
        {
            kargs.emplace_back(ck_tile::QuantGroupedGemmKernelArgs{arg.a_ptr,
                                                                   arg.b_ptr,
                                                                   arg.aq_ptr,
                                                                   arg.bq_ptr,
                                                                   arg.e_ptr,
                                                                   arg.M,
                                                                   arg.N,
                                                                   arg.K,
                                                                   arg.QK_A,
                                                                   arg.QK_B,
                                                                   arg.stride_A,
                                                                   arg.stride_B,
                                                                   arg.stride_E,
                                                                   arg.stride_AQ,
                                                                   arg.stride_BQ,
                                                                   arg.k_batch});
        }
        const auto stream = ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat};
        HIP_CHECK_ERROR(hipMemcpyWithStream(kargs_ptr,
                                            kargs.data(),
                                            kargs.size() * sizeof(ck_tile::QuantGemmTransKernelArg),
                                            hipMemcpyHostToDevice,
                                            stream.stream_id_));
        ave_time = grouped_gemm_tileloop<GemmConfig,
                                         ALayout,
                                         AQLayout,
                                         BLayout,
                                         BQLayout,
                                         CLayout,
                                         ADataType,
                                         AQDataType,
                                         BDataType,
                                         BQDataType,
                                         AccDataType,
                                         CDataType,
                                         AQuantGroupSize,
                                         BQuantGroupSize,
                                         QuantMode>(stream, group_count, kargs_ptr);
    }

    return ave_time;
}

template <typename GemmConfig,
          typename ADataType,
          typename AQDataType,
          typename BDataType,
          typename BQDataType,
          typename CDataType,
          typename AccDataType,
          typename AQuantGroupSize,
          typename BQuantGroupSize,
          ck_tile::QuantType QuantMode,
          typename ALayout,
          typename AQLayout,
          typename BLayout,
          typename BQLayout,
          typename CLayout>
int run_abquant_grouped_gemm_example_with_layouts(
    int argc,
    char* argv[],
    const ALayout a_layout                  = ALayout{},
    const AQLayout aq_layout                = AQLayout{},
    const BLayout b_layout                  = BLayout{},
    const BQLayout bq_layout                = BQLayout{},
    [[maybe_unused]] const CLayout c_layout = CLayout{})
{

    auto [result, arg_parser] = create_args(argc, argv);

    auto valid_input_data = [&](int group_count, const auto&... args) {
        return group_count != 0 && ((args.size() == static_cast<size_t>(group_count)) && ...);
    };

    const int group_count = arg_parser.get_int("group_count");
    const int repeat      = arg_parser.get_int("repeat");
    const int warmup      = arg_parser.get_int("warmup");
    const int kbatch      = arg_parser.get_int("kbatch");
    const int init_method = arg_parser.get_int("init");
    bool validate         = arg_parser.get_bool("validate");

    if(kbatch > 1 && validate && warmup + repeat > 1)
    {
        std::cout << "WARNING: Data validation enabled with SplitK and more than"
                  << "1 warmup/repeat. Disabling validation." << std::endl;
        validate = false;
    }

    std::vector<ck_tile::index_t> Ms = arg_parser.get_int_vec("Ms");
    std::vector<ck_tile::index_t> Ns = arg_parser.get_int_vec("Ns");
    std::vector<ck_tile::index_t> Ks = arg_parser.get_int_vec("Ks");
    std::vector<ck_tile::index_t> AQs; // dimension of AQ tensor is calculated from A tensor
    std::vector<ck_tile::index_t> BQs; // dimension of BQ tensor is calculated from B tensor
    std::vector<ck_tile::index_t> stride_As  = arg_parser.get_int_vec("stride_As");
    std::vector<ck_tile::index_t> stride_Bs  = arg_parser.get_int_vec("stride_Bs");
    std::vector<ck_tile::index_t> stride_Cs  = arg_parser.get_int_vec("stride_Cs");
    std::vector<ck_tile::index_t> stride_AQs = arg_parser.get_int_vec("stride_AQs");
    std::vector<ck_tile::index_t> stride_BQs = arg_parser.get_int_vec("stride_BQs");

    ck_tile::index_t AQK, BQK;

    if(!valid_input_data(
           group_count, Ms, Ns, Ks, stride_As, stride_Bs, stride_Cs, stride_AQs, stride_BQs))
    {
        std::cout << "Please check the input data. Default values will be used." << std::endl;

        // Clear existing (invalid) data before adding defaults
        Ms.clear();
        Ns.clear();
        Ks.clear();
        stride_As.clear();
        stride_Bs.clear();
        stride_Cs.clear();
        stride_AQs.clear();
        stride_BQs.clear();

        for(int i = 0; i < group_count; i++)
        {

            Ms.push_back(256 + 256 * i);
            Ns.push_back(256 + 512 * i);
            Ks.push_back(512 + 128 * i);

            // Let get_default_stride calculate based on layout
            stride_As.push_back(0);
            stride_Bs.push_back(0);
            stride_Cs.push_back(0);
            stride_AQs.push_back(0);
            stride_BQs.push_back(0);
        }
    }

    std::vector<ck_tile::HostTensor<ADataType>> a_m_k_tensors;
    std::vector<ck_tile::HostTensor<BDataType>> b_k_n_tensors;
    std::vector<ck_tile::HostTensor<CDataType>> c_m_n_tensors;
    std::vector<ck_tile::HostTensor<AQDataType>> aq_tensors;
    std::vector<ck_tile::HostTensor<BQDataType>> bq_tensors;

    a_m_k_tensors.reserve(group_count);
    b_k_n_tensors.reserve(group_count);
    c_m_n_tensors.reserve(group_count);
    aq_tensors.reserve(group_count);
    bq_tensors.reserve(group_count);

    std::vector<std::unique_ptr<ck_tile::DeviceMem>> a_m_k_dev_buf;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> b_k_n_dev_buf;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> c_m_n_dev_buf;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> aq_dev_buf;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> bq_dev_buf;

    a_m_k_dev_buf.reserve(group_count);
    b_k_n_dev_buf.reserve(group_count);
    c_m_n_dev_buf.reserve(group_count);
    aq_dev_buf.reserve(group_count);
    bq_dev_buf.reserve(group_count);

    std::vector<grouped_gemm_kargs> gemm_descs;
    gemm_descs.reserve(group_count);

    for(int i = 0; i < group_count; ++i)
    {

        const ck_tile::index_t M = Ms[i];
        const ck_tile::index_t N = Ns[i];
        const ck_tile::index_t K = Ks[i];

        // For ABQuantGrouped, both A and B need quantization
        static_assert(QuantMode == ck_tile::QuantType::ABQuantGrouped,
                      "This file only supports ABQuantGrouped mode");

        AQK = K / AQuantGroupSize::kK; // Group quantization: AQK = K / AQuantGroupSize
        BQK = K / BQuantGroupSize::kK; // Group quantization: BQK = K / BQuantGroupSize
        if(K % AQuantGroupSize::kK != 0)
        {
            throw std::runtime_error(
                "K must be divisible by AQuantGroupSize::kK for ABQuantGrouped mode");
        }
        if(K % BQuantGroupSize::kK != 0)
        {
            throw std::runtime_error(
                "K must be divisible by BQuantGroupSize::kK for ABQuantGrouped mode");
        }

        stride_As[i]  = ck_tile::get_default_stride(M, K, stride_As[i], is_row_major(a_layout));
        stride_Bs[i]  = ck_tile::get_default_stride(K, N, stride_Bs[i], is_row_major(b_layout));
        stride_Cs[i]  = ck_tile::get_default_stride(M, N, stride_Cs[i], is_row_major(CLayout{}));
        stride_AQs[i] = ck_tile::get_default_stride(M, AQK, stride_AQs[i], is_row_major(aq_layout));
        stride_BQs[i] = ck_tile::get_default_stride(BQK, N, stride_BQs[i], is_row_major(bq_layout));

        a_m_k_tensors.push_back(ck_tile::HostTensor<ADataType>(
            ck_tile::host_tensor_descriptor(M, K, stride_As[i], is_row_major(a_layout))));
        b_k_n_tensors.push_back(ck_tile::HostTensor<BDataType>(
            ck_tile::host_tensor_descriptor(K, N, stride_Bs[i], is_row_major(b_layout))));
        c_m_n_tensors.push_back(ck_tile::HostTensor<CDataType>(
            ck_tile::host_tensor_descriptor(M, N, stride_Cs[i], is_row_major(CLayout{}))));
        aq_tensors.push_back(ck_tile::HostTensor<AQDataType>(
            ck_tile::host_tensor_descriptor(M, AQK, stride_AQs[i], is_row_major(aq_layout))));
        bq_tensors.push_back(ck_tile::HostTensor<BQDataType>(
            ck_tile::host_tensor_descriptor(BQK, N, stride_BQs[i], is_row_major(bq_layout))));

        std::cout << "gemm[" << i << "]" << " a_m_k: " << a_m_k_tensors[i].mDesc
                  << " b_k_n: " << b_k_n_tensors[i].mDesc << " c_m_n: " << c_m_n_tensors[i].mDesc
                  << " aq: " << aq_tensors[i].mDesc << " bq: " << bq_tensors[i].mDesc << std::endl;

        if(init_method == 2)
        {
            ck_tile::FillUniformDistribution<ADataType>{1.f, 1.f}(a_m_k_tensors[i]);
            ck_tile::FillUniformDistribution<BDataType>{1.f, 1.f}(b_k_n_tensors[i]);
            ck_tile::FillUniformDistribution<AQDataType>{1.f, 1.f}(aq_tensors[i]);
            ck_tile::FillUniformDistribution<BQDataType>{1.f, 1.f}(bq_tensors[i]);
        }
        else
        {
            ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k_tensors[i]);
            ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n_tensors[i]);
            ck_tile::FillUniformDistribution<AQDataType>{-1.f, 1.f}(aq_tensors[i]);
            ck_tile::FillUniformDistribution<BQDataType>{-1.f, 1.f}(bq_tensors[i]);
        }

        a_m_k_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
            a_m_k_tensors[i].get_element_space_size_in_bytes()));
        b_k_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
            b_k_n_tensors[i].get_element_space_size_in_bytes()));
        c_m_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
            c_m_n_tensors[i].get_element_space_size_in_bytes()));
        aq_dev_buf.push_back(
            std::make_unique<ck_tile::DeviceMem>(aq_tensors[i].get_element_space_size_in_bytes()));
        bq_dev_buf.push_back(
            std::make_unique<ck_tile::DeviceMem>(bq_tensors[i].get_element_space_size_in_bytes()));

        a_m_k_dev_buf[i]->ToDevice(a_m_k_tensors[i].data());
        b_k_n_dev_buf[i]->ToDevice(b_k_n_tensors[i].data());
        aq_dev_buf[i]->ToDevice(aq_tensors[i].data());
        bq_dev_buf[i]->ToDevice(bq_tensors[i].data());
        c_m_n_dev_buf[i]->SetZero();
        c_m_n_tensors[i].SetZero();

        const void* p_a  = a_m_k_dev_buf[i]->GetDeviceBuffer();
        const void* p_b  = b_k_n_dev_buf[i]->GetDeviceBuffer();
        void* p_c        = c_m_n_dev_buf[i]->GetDeviceBuffer();
        const void* p_aq = aq_dev_buf[i]->GetDeviceBuffer();
        const void* p_bq = bq_dev_buf[i]->GetDeviceBuffer();

        gemm_descs.push_back({p_a,
                              p_b,
                              p_c,
                              p_aq,
                              p_bq,
                              kbatch,
                              M,
                              N,
                              K,
                              AQK,
                              BQK,
                              stride_As[i],
                              stride_Bs[i],
                              stride_Cs[i],
                              stride_AQs[i],
                              stride_BQs[i]});
    }

    float ave_time = invoke_abquant_gemm<GemmConfig,
                                         ADataType,
                                         AQDataType,
                                         BDataType,
                                         BQDataType,
                                         AccDataType,
                                         CDataType,
                                         ALayout,
                                         AQLayout,
                                         BLayout,
                                         BQLayout,
                                         CLayout,
                                         AQuantGroupSize,
                                         BQuantGroupSize,
                                         QuantMode>(warmup, repeat, group_count, gemm_descs);

    std::string op_name = "ABQuant Grouped Gemm (" + ck_tile::quant_type_to_string(QuantMode) + ")";

    std::size_t flop = 0, num_btype = 0;
    for(int j = 0; j < group_count; ++j)
    {
        flop += std::size_t(2) * gemm_descs[j].M * gemm_descs[j].N * gemm_descs[j].K;

        num_btype += sizeof(ADataType) * gemm_descs[j].M * gemm_descs[j].K +
                     sizeof(BDataType) * gemm_descs[j].K * gemm_descs[j].N +
                     sizeof(CDataType) * gemm_descs[j].M * gemm_descs[j].N;
    }

    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
              << gb_per_sec << " GB/s, " << op_name << std::endl;

    for(int i = 0; i < group_count; i++)
    {
        c_m_n_dev_buf[i]->FromDevice(c_m_n_tensors[i].data());
    }

    bool pass{true};
    if(validate)
    {
        for(int i = 0; i < group_count; ++i)
        {
            ck_tile::HostTensor<CDataType> c_m_n_host_ref(ck_tile::host_tensor_descriptor(
                Ms[i], Ns[i], stride_Cs[i], is_row_major(CLayout{})));
            c_m_n_host_ref.SetZero();

            // Reference implementation for ABQuantGrouped
            ck_tile::reference_gemm_abquant<ADataType,
                                            AQDataType,
                                            BDataType,
                                            BQDataType,
                                            AccDataType,
                                            CDataType,
                                            AQuantGroupSize,
                                            BQuantGroupSize>(
                a_m_k_tensors[i], aq_tensors[i], b_k_n_tensors[i], bq_tensors[i], c_m_n_host_ref);

            const float max_accumulated_value =
                *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
            const auto rtol_atol =
                calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
                    Ks[i], kbatch, max_accumulated_value);
            pass &=
                ck_tile::check_err(c_m_n_tensors[i],
                                   c_m_n_host_ref,
                                   "Error: Incorrect results! in group [" + std::to_string(i) + "]",
                                   rtol_atol.at(ck_tile::number<0>{}),
                                   rtol_atol.at(ck_tile::number<1>{}));
            std::cout << "gemm[" << i
                      << "] Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                      << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                      << std::endl;
        }
        std::cout << "The CPU verification result is:" << (pass ? "correct" : "fail") << std::endl;
    }

    if(arg_parser.get_int("json") == 1)
    {
        dump_grouped_gemm_json_results<ALayout, BLayout, CLayout>(arg_parser.get_str("jsonfile"),
                                                                  op_name,
                                                                  group_count,
                                                                  pass,
                                                                  ave_time,
                                                                  tflops,
                                                                  gb_per_sec);
    }

    return pass;
}

template <typename PrecType, typename GemmConfig, typename BQuantGroupSize>
int run_abquant_grouped_gemm_example_prec_type_with_bquant(
    std::string a_layout, std::string b_layout, std::string c_layout, int argc, char* argv[])
{
    using Row   = ck_tile::tensor_layout::gemm::RowMajor;
    using Col   = ck_tile::tensor_layout::gemm::ColumnMajor;
    using Types = GemmTypeConfig<PrecType>;
    // Specific type aliases for easy access
    using ADataType       = typename Types::ADataType;
    using BDataType       = typename Types::BDataType;
    using AccDataType     = typename Types::AccDataType;
    using CDataType       = typename Types::CDataType;
    using AQDataType      = typename Types::AccDataType;
    using BQDataType      = typename Types::AccDataType;
    using AQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;

    constexpr auto QuantMode = ck_tile::QuantType::ABQuantGrouped;

    if(a_layout == "R" && b_layout == "C" && c_layout == "R")
    {
        return run_abquant_grouped_gemm_example_with_layouts<GemmConfig,
                                                             ADataType,
                                                             AQDataType,
                                                             BDataType,
                                                             BQDataType,
                                                             CDataType,
                                                             AccDataType,
                                                             AQuantGroupSize,
                                                             BQuantGroupSize,
                                                             QuantMode>(
            argc, argv, Row{}, Row{}, Col{}, Col{}, Row{});
    }
    else if(a_layout == "R" && b_layout == "R" && c_layout == "R")
    {
        return run_abquant_grouped_gemm_example_with_layouts<GemmConfig,
                                                             ADataType,
                                                             AQDataType,
                                                             BDataType,
                                                             BQDataType,
                                                             CDataType,
                                                             AccDataType,
                                                             AQuantGroupSize,
                                                             BQuantGroupSize,
                                                             QuantMode>(
            argc, argv, Row{}, Row{}, Row{}, Col{}, Row{});
    }
    else if(a_layout == "C" && b_layout == "R" && c_layout == "R")
    {
        return run_abquant_grouped_gemm_example_with_layouts<GemmConfig,
                                                             ADataType,
                                                             AQDataType,
                                                             BDataType,
                                                             BQDataType,
                                                             CDataType,
                                                             AccDataType,
                                                             AQuantGroupSize,
                                                             BQuantGroupSize,
                                                             QuantMode>(
            argc, argv, Col{}, Row{}, Row{}, Col{}, Row{});
    }
    else
    {
        throw std::runtime_error("Unsupported data layout configuration for A,B and C tensors!");
    }
}

template <typename PrecType, typename GemmConfig>
int run_abquant_grouped_gemm_example_prec_type(std::string a_layout,
                                               std::string b_layout,
                                               std::string c_layout,
                                               std::string bquant_group_size,
                                               int argc,
                                               char* argv[])
{
    if(bquant_group_size == "1x1x128")
    {
        using BQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
        return run_abquant_grouped_gemm_example_prec_type_with_bquant<PrecType,
                                                                      GemmConfig,
                                                                      BQuantGroupSize>(
            a_layout, b_layout, c_layout, argc, argv);
    }
    else if(bquant_group_size == "1x128x128")
    {
        using BQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;
        return run_abquant_grouped_gemm_example_prec_type_with_bquant<PrecType,
                                                                      GemmConfig,
                                                                      BQuantGroupSize>(
            a_layout, b_layout, c_layout, argc, argv);
    }
    else
    {
        throw std::runtime_error("Unsupported BQuantGroupSize! Use 1x1x128 or 1x128x128.");
    }
}

template <typename PrecType>
int run_abquant_gemm_example_persistency(std::string a_layout,
                                         std::string b_layout,
                                         std::string c_layout,
                                         bool persistent,
                                         std::string bquant_group_size,
                                         int argc,
                                         char* argv[])
{
    if(persistent)
    {
        using GemmConfig = typename GemmQuantConfig<
            ck_tile::QuantType::ABQuantGrouped>::template GemmConfig<PrecType, true>;
        return run_abquant_grouped_gemm_example_prec_type<PrecType, GemmConfig>(
            a_layout, b_layout, c_layout, bquant_group_size, argc, argv);
    }
    else
    {
        using GemmConfig = typename GemmQuantConfig<
            ck_tile::QuantType::ABQuantGrouped>::template GemmConfig<PrecType, false>;
        return run_abquant_grouped_gemm_example_prec_type<PrecType, GemmConfig>(
            a_layout, b_layout, c_layout, bquant_group_size, argc, argv);
    }
}

int run_abquant_grouped_gemm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
    {
        return -1;
    }

    const std::string a_layout          = arg_parser.get_str("a_layout");
    const std::string b_layout          = arg_parser.get_str("b_layout");
    const std::string c_layout          = arg_parser.get_str("c_layout");
    const std::string data_type         = arg_parser.get_str("prec");
    bool persistent                     = arg_parser.get_bool("persistent");
    const std::string bquant_group_size = arg_parser.get_str("bquant_group_size");

    if(data_type == "fp8")
    {
        return run_abquant_gemm_example_persistency<ck_tile::fp8_t>(
            a_layout, b_layout, c_layout, persistent, bquant_group_size, argc, argv);
    }
    else if(data_type == "bf8")
    {
        return run_abquant_gemm_example_persistency<ck_tile::bf8_t>(
            a_layout, b_layout, c_layout, persistent, bquant_group_size, argc, argv);
    }
    else
    {
        throw std::runtime_error("Unsupported data type configuration.");
    }
}
