// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
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
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename CDEElementWise = ck_tile::element_wise::PassThrough>
float invoke_batched_gemm(ck_tile::DeviceMem& a_m_k_dev_buf,
                          ck_tile::DeviceMem& b_k_n_dev_buf,
                          ck_tile::DeviceMem& c_m_n_dev_buf,
                          ck_tile::index_t M,
                          ck_tile::index_t N,
                          ck_tile::index_t K,
                          ck_tile::index_t stride_A,
                          ck_tile::index_t stride_B,
                          ck_tile::index_t stride_C,
                          ck_tile::index_t batch_stride_A,
                          ck_tile::index_t batch_stride_B,
                          ck_tile::index_t batch_stride_C,
                          ck_tile::index_t batch_count,
                          ck_tile::index_t kbatch,
                          int n_warmup,
                          int n_repeat)
{
    ck_tile::BatchedGemmHostArgs args{a_m_k_dev_buf.GetDeviceBuffer(),
                                      b_k_n_dev_buf.GetDeviceBuffer(),
                                      c_m_n_dev_buf.GetDeviceBuffer(),
                                      kbatch,
                                      M,
                                      N,
                                      K,
                                      stride_A,
                                      stride_B,
                                      stride_C,
                                      batch_stride_A,
                                      batch_stride_B,
                                      batch_stride_C,
                                      batch_count};

    float ave_time = batched_gemm<GemmConfig,
                                  ADataType,
                                  BDataType,
                                  DsDataType,
                                  AccDataType,
                                  CDataType,
                                  ALayout,
                                  BLayout,
                                  DsLayout,
                                  CLayout,
                                  CDEElementWise>(
        args, ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat});

    return ave_time;
}

template <typename GemmConfig, typename ALayout, typename BLayout, typename CLayout>
int run_batched_gemm_example_with_layouts(int argc,
                                          char* argv[],
                                          const ALayout a_layout                  = ALayout{},
                                          const BLayout b_layout                  = BLayout{},
                                          [[maybe_unused]] const CLayout c_layout = CLayout{})
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    ck_tile::index_t M = arg_parser.get_int("m");
    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t K = arg_parser.get_int("k");

    ck_tile::index_t stride_A = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_B = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_C = arg_parser.get_int("stride_c");

    ck_tile::index_t batch_stride_A = arg_parser.get_int("batch_stride_a");
    ck_tile::index_t batch_stride_B = arg_parser.get_int("batch_stride_b");
    ck_tile::index_t batch_stride_C = arg_parser.get_int("batch_stride_c");
    ck_tile::index_t batch_count    = arg_parser.get_int("batch_count");
    ck_tile::index_t kbatch         = arg_parser.get_int("split_k");

    int n_warmup = arg_parser.get_int("warmup");
    int n_repeat = arg_parser.get_int("repeat");

    using namespace ck_tile::literals;

    auto f_host_tensor_descriptor = [](std::size_t batch_count_,
                                       std::size_t row,
                                       std::size_t col,
                                       std::size_t stride,
                                       std::size_t batch_stride,
                                       auto layout) {
        if constexpr(std::is_same_v<decltype(layout), ck_tile::tensor_layout::gemm::RowMajor>)
        {
            return ck_tile::HostTensorDescriptor({batch_count_, row, col},
                                                 {batch_stride, stride, 1_uz});
        }
        else
        {
            return ck_tile::HostTensorDescriptor({batch_count_, row, col},
                                                 {batch_stride, 1_uz, stride});
        }
    };

    auto f_get_default_stride = [](std::size_t row,
                                   std::size_t col,
                                   std::size_t stride,
                                   auto layout) {
        if(stride == 0)
        {
            // give a chance if stride is zero, return a default packed stride
            if constexpr(std::is_same_v<decltype(layout), ck_tile::tensor_layout::gemm::RowMajor>)
            {
                return col;
            }
            else
            {
                return row;
            }
        }
        else
            return stride;
    };

    stride_A = f_get_default_stride(M, K, stride_A, a_layout);
    stride_B = f_get_default_stride(K, N, stride_B, b_layout);
    stride_C = f_get_default_stride(M, N, stride_C, c_layout);

    ck_tile::HostTensor<ADataType> a_m_k(
        f_host_tensor_descriptor(batch_count, M, K, stride_A, batch_stride_A, a_layout));
    ck_tile::HostTensor<BDataType> b_k_n(
        f_host_tensor_descriptor(batch_count, K, N, stride_B, batch_stride_B, b_layout));
    ck_tile::HostTensor<CDataType> c_m_n_dev_result(
        f_host_tensor_descriptor(batch_count, M, N, stride_C, batch_stride_C, c_layout));

    ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_m_k);
    ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_k_n);

    ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

    a_m_k_dev_buf.ToDevice(a_m_k.data());
    b_k_n_dev_buf.ToDevice(b_k_n.data());
    c_m_n_dev_buf.SetZero();
    c_m_n_dev_result.SetZero();

    float ave_time = invoke_batched_gemm<GemmConfig,
                                         ADataType,
                                         BDataType,
                                         ck_tile::tuple<>,
                                         AccDataType,
                                         CDataType,
                                         ALayout,
                                         BLayout,
                                         ck_tile::tuple<>,
                                         CLayout>(a_m_k_dev_buf,
                                                  b_k_n_dev_buf,
                                                  c_m_n_dev_buf,
                                                  M,
                                                  N,
                                                  K,
                                                  stride_A,
                                                  stride_B,
                                                  stride_C,
                                                  batch_stride_A,
                                                  batch_stride_B,
                                                  batch_stride_C,
                                                  batch_count,
                                                  kbatch,
                                                  n_warmup,
                                                  n_repeat);
    c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());

    std::string op_name{"Batched Gemm"};
    std::size_t flop     = std::size_t(2) * batch_count * M * N * K;
    std::size_t num_byte = sizeof(ADataType) * batch_count * M * K +
                           sizeof(BDataType) * batch_count * N * K +
                           sizeof(CDataType) * batch_count * M * N;
    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Run " << op_name << "kernel with M =" << M << " N =" << N << " K =" << K
              << " StrideA =" << stride_A << " StrideB =" << stride_B << " StrideC =" << stride_C
              << " batch_stride_A =" << batch_stride_A << " batch_stride_B =" << batch_stride_B
              << " batch_stride_C =" << batch_stride_C << " batch_count =" << batch_count << " : "
              << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << std::endl;

    bool pass = true;

    if(arg_parser.get_int("v") == 1)
    {
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            f_host_tensor_descriptor(batch_count, M, N, stride_C, batch_stride_C, CLayout{}));
        c_m_n_host_ref.SetZero();

        const auto b_n_k = b_k_n.transpose({0, 2, 1});

        ck_tile::reference_batched_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_m_k, b_n_k, c_m_n_host_ref);
        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto rtol_atol = calculate_rtol_atol(K, kbatch, max_accumulated_value);
        pass                 = ck_tile::check_err(c_m_n_dev_result,
                                  c_m_n_host_ref,
                                  "Error: Incorrect results!",
                                  rtol_atol.at(ck_tile::number<0>{}),
                                  rtol_atol.at(ck_tile::number<1>{}));

        std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                  << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                  << std::endl;

        std::cout << "The CPU verification result is:" << (pass ? "correct" : "fail") << std::endl;
    }
    else if(arg_parser.get_int("v") == 2)
    {
        ck_tile::HostTensor<CDataType> c_m_n_gpu_ref(
            f_host_tensor_descriptor(batch_count, M, N, stride_C, batch_stride_C, CLayout{}));
        ck_tile::DeviceMem c_m_n_gpu_buf_ref(c_m_n_gpu_ref.get_element_space_size_in_bytes());
        c_m_n_gpu_ref.SetZero();
        c_m_n_gpu_buf_ref.SetZero();

        ADataType* d_A = static_cast<ADataType*>(a_m_k_dev_buf.GetDeviceBuffer());
        BDataType* d_B = static_cast<BDataType*>(b_k_n_dev_buf.GetDeviceBuffer());
        CDataType* d_C = static_cast<CDataType*>(c_m_n_gpu_buf_ref.GetDeviceBuffer());

        ck_tile::reference_batched_gemm_gpu<ADataType,
                                            BDataType,
                                            AccDataType,
                                            CDataType,
                                            ALayout,
                                            BLayout,
                                            CLayout>(d_A,
                                                     d_B,
                                                     d_C,
                                                     M,
                                                     N,
                                                     K,
                                                     stride_A,
                                                     stride_B,
                                                     stride_C,
                                                     batch_stride_A,
                                                     batch_stride_B,
                                                     batch_stride_C,
                                                     batch_count);

        c_m_n_gpu_buf_ref.FromDevice(c_m_n_gpu_ref.data());
        const float max_accumulated_value =
            *std::max_element(c_m_n_gpu_ref.mData.begin(), c_m_n_gpu_ref.mData.end());
        const auto rtol_atol = calculate_rtol_atol(K, kbatch, max_accumulated_value);
        pass                 = ck_tile::check_err(c_m_n_dev_result,
                                  c_m_n_gpu_ref,
                                  "Error: Incorrect results!",
                                  rtol_atol.at(ck_tile::number<0>{}),
                                  rtol_atol.at(ck_tile::number<1>{}));

        std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                  << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                  << std::endl;

        std::cout << "The GPU verification result is: " << (pass ? "correct" : "fail") << std::endl;
    }

    if(arg_parser.get_int("json") == 1)
    {
        dump_batched_gemm_json_results(arg_parser.get_str("jsonfile"),
                                       op_name,
                                       M,
                                       N,
                                       K,
                                       stride_A,
                                       stride_B,
                                       stride_C,
                                       batch_stride_A,
                                       batch_stride_B,
                                       batch_stride_C,
                                       batch_count,
                                       pass,
                                       ave_time,
                                       tflops,
                                       gb_per_sec,
                                       "batched_gemm");
    }

    return pass;
}

template <typename GemmConfig>
int run_batched_gemm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    std::string a_layout = arg_parser.get_str("a_layout");
    std::string b_layout = arg_parser.get_str("b_layout");

    // if(a_layout == "R" && b_layout == "R")
    // {
    //     return run_batched_gemm_example_with_layouts(argc, argv, Row{}, Row{}, Row{});
    // }
    if(a_layout == "R" && b_layout == "C")
    {
        return run_batched_gemm_example_with_layouts<GemmConfig>(argc, argv, Row{}, Col{}, Row{});
    }
    // TODO: Fixme: with latest changes to GemmPipelineAGmemBGmemCRegV1DefaultPolicy below do not
    // work else if(a_layout == "C" && b_layout == "C")
    // {
    //     return run_batched_gemm_example_with_layouts(argc, argv, Col{}, Col{}, Row{});
    // }
    // else if(a_layout == "C" && b_layout == "R")
    // {
    //     return run_batched_gemm_example_with_layouts(argc, argv, Col{}, Row{}, Row{});
    // }
    else
    {
        throw std::runtime_error("Unsupported data layout configuration for A,B and C tensors!");
    }
}
