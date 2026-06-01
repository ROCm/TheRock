// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <cstddef>

template <typename GemmConfig,
          typename AsDataType,
          typename BsDataType,
          typename DsDataType,
          typename AccDataType,
          typename EDataType,
          typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename ELayout,
          typename AElementWise   = ck_tile::element_wise::PassThrough,
          typename BElementWise   = ck_tile::element_wise::PassThrough,
          typename CDEElementWise = ck_tile::element_wise::PassThrough>
float invoke_gemm_multi_abd(const std::array<const void*, AsDataType::size()>& as_m_k_dev_buf,
                            const std::array<const void*, BsDataType::size()>& bs_k_n_dev_buf,
                            const std::array<const void*, DsDataType::size()>& ds_m_n_dev_buf,
                            void* e_m_n_dev_buf,
                            ck_tile::index_t M,
                            ck_tile::index_t N,
                            ck_tile::index_t K,
                            const std::array<ck_tile::index_t, AsDataType::size()>& StrideAs,
                            const std::array<ck_tile::index_t, BsDataType::size()>& StrideBs,
                            const std::array<ck_tile::index_t, DsDataType::size()>& StrideDs,
                            ck_tile::index_t StrideE,
                            int n_warmup,
                            int n_repeat,
                            int k_batch)
{
    gemm_multi_abd_kargs gemm_descs({as_m_k_dev_buf,
                                     bs_k_n_dev_buf,
                                     ds_m_n_dev_buf,
                                     e_m_n_dev_buf,
                                     k_batch,
                                     M,
                                     N,
                                     K,
                                     StrideAs,
                                     StrideBs,
                                     StrideDs,
                                     StrideE});

    float ave_time = gemm_multi_abd<GemmConfig,
                                    AsDataType,
                                    BsDataType,
                                    DsDataType,
                                    AccDataType,
                                    EDataType,
                                    AsLayout,
                                    BsLayout,
                                    DsLayout,
                                    ELayout,
                                    AElementWise,
                                    BElementWise,
                                    CDEElementWise>(
        gemm_descs, ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat});

    std::string op_name{"Gemm Multiple-ABD"};

    std::size_t flop = 0, num_btype = 0;

    flop += std::size_t(2) * M * N * K;

    num_btype +=
        sizeof(A0DataType) * M * K + sizeof(B0DataType) * K * N + sizeof(EDataType) * M * N;

    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Run Gemm Multiple-ABD kernel with:\n";
    std::cout << "M =" << M << " N =" << N << " K =" << K << "\n";
    std::cout << "StrideA = " << StrideAs[0] << " StrideB = " << StrideBs[0]
              << " StrideE = " << StrideE << "\n";
    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << "\n";

    return ave_time;
}

template <typename GemmConfig,
          typename A0Layout,
          typename A1Layout,
          typename B0Layout,
          typename B1Layout,
          typename D0Layout,
          typename D1Layout,
          typename ELayout>
int run_gemm_multi_abd_example_with_layouts(int argc,
                                            char* argv[],
                                            const A0Layout a0_layout = A0Layout{},
                                            const A1Layout a1_layout = A1Layout{},
                                            const B0Layout b0_layout = B0Layout{},
                                            const B1Layout b1_layout = B1Layout{},
                                            const D0Layout d0_layout = D0Layout{},
                                            const D1Layout d1_layout = D1Layout{},
                                            const ELayout e_layout   = ELayout{})
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
    {
        return -1;
    }
    using AElementWiseFn   = ck_tile::element_wise::AddScale;
    using BElementWiseFn   = ck_tile::element_wise::AddScale;
    using CDEElementWiseFn = ck_tile::element_wise::MultiDMultiply;
    using AsLayout         = ck_tile::tuple<A0Layout, A1Layout>;
    using BsLayout         = ck_tile::tuple<B0Layout, B1Layout>;
    using DsLayout         = ck_tile::tuple<D0Layout, D1Layout>;

    ck_tile::index_t M = arg_parser.get_int("m");
    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t K = arg_parser.get_int("k");

    ck_tile::index_t StrideA = arg_parser.get_int("stride_as");
    ck_tile::index_t StrideB = arg_parser.get_int("stride_bs");
    ck_tile::index_t StrideD = arg_parser.get_int("stride_ds");
    ck_tile::index_t StrideE = arg_parser.get_int("stride_e");

    ck_tile::index_t StrideA0 = StrideA;
    ck_tile::index_t StrideA1 = StrideA;

    ck_tile::index_t StrideB0 = StrideB;
    ck_tile::index_t StrideB1 = StrideB;

    ck_tile::index_t StrideD0 = StrideD;
    ck_tile::index_t StrideD1 = StrideD;

    const int n_warmup = arg_parser.get_int("warmup");
    const int n_repeat = arg_parser.get_int("repeat");
    const int k_batch  = arg_parser.get_int("kbatch");

    StrideA0 = get_default_stride(M, N, StrideA0, is_row_major(a1_layout));
    StrideA1 = get_default_stride(M, N, StrideA1, is_row_major(a1_layout));

    StrideB0 = get_default_stride(K, N, StrideB0, is_row_major(b0_layout));
    StrideB1 = get_default_stride(K, N, StrideB1, is_row_major(b1_layout));

    StrideD0 = get_default_stride(M, N, StrideD0, is_row_major(d0_layout));
    StrideD1 = get_default_stride(M, N, StrideD1, is_row_major(d1_layout));

    StrideE = get_default_stride(M, N, StrideE, is_row_major(e_layout));

    ck_tile::HostTensor<A0DataType> a0_m_k_tesnor(
        host_tensor_descriptor(M, K, StrideA0, is_row_major(a0_layout)));
    ck_tile::HostTensor<A1DataType> a1_m_k_tesnor(
        host_tensor_descriptor(M, K, StrideA1, is_row_major(a1_layout)));

    ck_tile::HostTensor<B0DataType> b0_k_n_tensors(
        host_tensor_descriptor(K, N, StrideB0, is_row_major(b0_layout)));
    ck_tile::HostTensor<B1DataType> b1_k_n_tensors(
        host_tensor_descriptor(K, N, StrideB1, is_row_major(b1_layout)));

    ck_tile::HostTensor<D0DataType> d0_m_n_tensors(
        host_tensor_descriptor(M, N, StrideD0, is_row_major(d0_layout)));
    ck_tile::HostTensor<D1DataType> d1_m_n_tensors(
        host_tensor_descriptor(M, N, StrideD1, is_row_major(d1_layout)));

    ck_tile::HostTensor<EDataType> e_m_n_device_result(
        host_tensor_descriptor(M, N, StrideE, is_row_major(e_layout)));

    ck_tile::FillUniformDistribution<A0DataType>{-1.f, 1.f}(a0_m_k_tesnor);
    ck_tile::FillUniformDistribution<A1DataType>{-1.f, 1.f}(a1_m_k_tesnor);

    ck_tile::FillUniformDistribution<B0DataType>{-1.f, 1.f}(b0_k_n_tensors);
    ck_tile::FillUniformDistribution<B1DataType>{-1.f, 1.f}(b1_k_n_tensors);

    ck_tile::FillUniformDistribution<D0DataType>{-1.f, 1.f}(d0_m_n_tensors);
    ck_tile::FillUniformDistribution<D1DataType>{-1.f, 1.f}(d1_m_n_tensors);

    ck_tile::DeviceMem a0_m_k_dev_buf(a0_m_k_tesnor.get_element_space_size_in_bytes());
    ck_tile::DeviceMem a1_m_k_dev_buf(a1_m_k_tesnor.get_element_space_size_in_bytes());

    ck_tile::DeviceMem b0_k_n_dev_buf(b0_k_n_tensors.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b1_k_n_dev_buf(b1_k_n_tensors.get_element_space_size_in_bytes());

    ck_tile::DeviceMem d0_m_n_dev_buf(d0_m_n_tensors.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d1_m_n_dev_buf(d1_m_n_tensors.get_element_space_size_in_bytes());

    ck_tile::DeviceMem e_m_n_dev_buf(e_m_n_device_result.get_element_space_size_in_bytes());

    a0_m_k_dev_buf.ToDevice(a0_m_k_tesnor.mData.data());
    a1_m_k_dev_buf.ToDevice(a1_m_k_tesnor.mData.data());

    b0_k_n_dev_buf.ToDevice(b0_k_n_tensors.mData.data());
    b1_k_n_dev_buf.ToDevice(b1_k_n_tensors.mData.data());

    d0_m_n_dev_buf.ToDevice(d0_m_n_tensors.mData.data());
    d1_m_n_dev_buf.ToDevice(d1_m_n_tensors.mData.data());

    e_m_n_dev_buf.SetZero();
    e_m_n_device_result.SetZero();

    std::array<const void*, DsDataType::size()> as_ptr_buf = {a0_m_k_dev_buf.GetDeviceBuffer(),
                                                              a1_m_k_dev_buf.GetDeviceBuffer()};

    std::array<const void*, DsDataType::size()> bs_ptr_buf = {b0_k_n_dev_buf.GetDeviceBuffer(),
                                                              b1_k_n_dev_buf.GetDeviceBuffer()};

    std::array<const void*, DsDataType::size()> ds_ptr_buf = {d0_m_n_dev_buf.GetDeviceBuffer(),
                                                              d1_m_n_dev_buf.GetDeviceBuffer()};

    std::array<ck_tile::index_t, AsDataType::size()> strideAs = {StrideA0, StrideA1};
    std::array<ck_tile::index_t, BsDataType::size()> strideBs = {StrideB0, StrideB1};
    std::array<ck_tile::index_t, DsDataType::size()> strideDs = {StrideD0, StrideD1};

    invoke_gemm_multi_abd<GemmConfig,
                          AsDataType,
                          BsDataType,
                          DsDataType,
                          AccDataType,
                          EDataType,
                          AsLayout,
                          BsLayout,
                          DsLayout,
                          ELayout,
                          AElementWiseFn,
                          BElementWiseFn,
                          CDEElementWiseFn>(as_ptr_buf,
                                            bs_ptr_buf,
                                            ds_ptr_buf,
                                            e_m_n_dev_buf.GetDeviceBuffer(),
                                            M,
                                            N,
                                            K,
                                            strideAs,
                                            strideBs,
                                            strideDs,
                                            StrideE,
                                            n_warmup,
                                            n_repeat,
                                            k_batch);

    e_m_n_dev_buf.FromDevice(e_m_n_device_result.data());

    ck_tile::HostTensor<A0DataType> a_m_k_host_ref_element_result(
        host_tensor_descriptor(M, K, StrideA0, is_row_major(a0_layout)));
    ck_tile::HostTensor<B0DataType> b_k_n_host_ref_element_result(
        host_tensor_descriptor(K, N, StrideB0, is_row_major(b0_layout)));
    ck_tile::HostTensor<EDataType> e_m_n_host_ref(
        host_tensor_descriptor(M, N, StrideE, is_row_major(e_layout)));
    a_m_k_host_ref_element_result.SetZero();
    b_k_n_host_ref_element_result.SetZero();
    e_m_n_host_ref.SetZero();

    ck_tile::reference_gemm_multiple_abd<AsDataType,
                                         BsDataType,
                                         DsDataType,
                                         AccDataType,
                                         EDataType,
                                         AElementWiseFn,
                                         BElementWiseFn,
                                         CDEElementWiseFn>({a0_m_k_tesnor, a1_m_k_tesnor},
                                                           {b0_k_n_tensors, b1_k_n_tensors},
                                                           {d0_m_n_tensors, d1_m_n_tensors},
                                                           a_m_k_host_ref_element_result,
                                                           b_k_n_host_ref_element_result,
                                                           e_m_n_host_ref);

    bool pass{true};
    if(arg_parser.get_int("v"))
    {
        const float max_accumulated_value =
            *std::max_element(e_m_n_host_ref.mData.begin(), e_m_n_host_ref.mData.end());

        const auto rtol_atol = calculate_rtol_atol(K, 1, max_accumulated_value);

        pass &= ck_tile::check_err(e_m_n_device_result,
                                   e_m_n_host_ref,
                                   "Error: Incorrect results!",
                                   rtol_atol.at(ck_tile::number<0>{}),
                                   rtol_atol.at(ck_tile::number<1>{}));

        std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                  << std::endl;
        std::cout << "Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                  << std::endl;
        std::cout << "The CPU veification result is: " << (pass ? "correct" : "fail") << std::endl;
    }
    return pass;
}

template <typename GemmConfig>
int run_multiple_abd_gemm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
    {
        return -1;
    }

    const std::string as_layout = arg_parser.get_str("as_layout");
    const std::string bs_layout = arg_parser.get_str("bs_layout");

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    if(as_layout == "R" && bs_layout == "C")
    {
        return run_gemm_multi_abd_example_with_layouts<GemmConfig>(
            argc, argv, Row{}, Row{}, Col{}, Col{}, Row{}, Row{}, Row{});
    }
    else
    {
        throw std::runtime_error("Unsupported data layout configuration for A,B and C tensors!");
    }
}
