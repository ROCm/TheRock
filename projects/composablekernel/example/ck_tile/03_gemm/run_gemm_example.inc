// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck_tile/host/permute_pk_int4.hpp"
#include "ck_tile/host/tensor_shuffle_utils.hpp"
#include "ck_tile/ops/common/utils.hpp"

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
          typename Tensor,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
void permute_tensor_b(Tensor& tensor)
{
    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
        ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
        ck_tile::
            sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>,
        GemmConfig::PermuteA,
        GemmConfig::PermuteB>;

    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
                                                                 GemmConfig::kPadN,
                                                                 GemmConfig::kPadK,
                                                                 GemmConfig::DoubleSmemBuffer,
                                                                 ALayout,
                                                                 BLayout,
                                                                 CLayout,
                                                                 GemmConfig::TransposeC,
                                                                 GemmConfig::UseStructuredSparsity>;

    using UniversalGemmProblem =
        ck_tile::UniversalGemmPipelineProblem<ADataType,
                                              BDataType,
                                              AccDataType,
                                              GemmShape,
                                              GemmUniversalTraits,
                                              GemmConfig::Scheduler,
                                              ck_tile::element_wise::PassThrough,
                                              ck_tile::element_wise::PassThrough,
                                              ADataType,
                                              true>;

    using GemmPipeline = typename PipelineTypeTraits<GemmConfig::Pipeline>::template GemmPipeline<
        UniversalGemmProblem>;

    const ck_tile::index_t K  = tensor.get_length(0);
    const ck_tile::index_t N  = tensor.get_length(1);
    const ck_tile::index_t K1 = GemmPipeline::GetSmemPackB();
    const ck_tile::index_t K0 = K / K1;

    Tensor tensor_copy = tensor;

    // int K0, N, K1
    for(int j = 0; j < K0; j++)
    {
        for(int i = 0; i < N; i++)
        {
            for(int jj = 0; jj < K1; jj++)
            {
                tensor(j * N * K1 + i * K1 + jj) = tensor_copy(i * K + (j * K1 + jj));
            }
        }
    }
}

template <typename GemmConfig,
          typename Invoker,
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
float invoke_gemm(ck_tile::DeviceMem& a_m_k_dev_buf,
                  ck_tile::DeviceMem& b_k_n_dev_buf,
                  ck_tile::DeviceMem& c_m_n_dev_buf,
                  ck_tile::index_t M,
                  ck_tile::index_t N,
                  ck_tile::index_t K,
                  ck_tile::index_t stride_A,
                  ck_tile::index_t stride_B,
                  ck_tile::index_t stride_C,
                  ck_tile::index_t kbatch,
                  int n_warmup,
                  int n_repeat,
                  bool persistent,
                  bool flush_cache,
                  int rotating_count)
{
    ck_tile::GemmHostArgs args = {a_m_k_dev_buf.GetDeviceBuffer(),
                                  b_k_n_dev_buf.GetDeviceBuffer(),
                                  c_m_n_dev_buf.GetDeviceBuffer(),
                                  kbatch,
                                  M,
                                  N,
                                  K,
                                  stride_A,
                                  stride_B,
                                  stride_C};

    float ave_time;
    if(persistent)
    {
        ave_time = Invoker::template gemm<GemmConfig,
                                          ADataType,
                                          BDataType,
                                          DsDataType,
                                          AccDataType,
                                          CDataType,
                                          ALayout,
                                          BLayout,
                                          DsLayout,
                                          CLayout,
                                          true,
                                          CDEElementWise>(
            args,
            ck_tile::stream_config{
                nullptr, true, 1, n_warmup, n_repeat, true, flush_cache, rotating_count});
    }
    else
    {
        ave_time = Invoker::template gemm<GemmConfig,
                                          ADataType,
                                          BDataType,
                                          DsDataType,
                                          AccDataType,
                                          CDataType,
                                          ALayout,
                                          BLayout,
                                          DsLayout,
                                          CLayout,
                                          false,
                                          CDEElementWise>(
            args,
            ck_tile::stream_config{
                nullptr, true, 1, n_warmup, n_repeat, true, flush_cache, rotating_count});
    }

    return ave_time;
}

template <typename CDataType>
bool do_verify(const ck_tile::HostTensor<CDataType>& c_m_n_dev_result,
               const ck_tile::HostTensor<CDataType>& c_m_n_ref,
               const ck_tile::tuple<double, double>& rtol_atol,
               const char* variant)
{
    bool pass = ck_tile::check_err(c_m_n_dev_result,
                                   c_m_n_ref,
                                   "Error: Incorrect results!",
                                   rtol_atol.at(ck_tile::number<0>{}),
                                   rtol_atol.at(ck_tile::number<1>{}));

    std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
              << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{}) << std::endl;
    std::cout << "The " << variant << " verification result is:" << (pass ? "correct" : "fail")
              << std::endl;
    return pass;
}

std::tuple<ck_tile::index_t, ck_tile::index_t, ck_tile::index_t> inline parse_gemm_size(
    ck_tile::ArgParser& arg_parser)
{
    ck_tile::index_t M = arg_parser.get_int("m");
    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t K = arg_parser.get_int("k");
    return std::make_tuple(M, N, K);
}

template <typename GemmConfig,
          typename Invoker,
          typename ADataType,
          typename BDataType = ADataType,
          typename CDataType = ADataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
int run_gemm_example_with_layouts(ck_tile::ArgParser& arg_parser,
                                  const ALayout a_layout                  = ALayout{},
                                  const BLayout b_layout                  = BLayout{},
                                  [[maybe_unused]] const CLayout c_layout = CLayout{})
{
    using AccDataType = typename GemmTypeConfig<ADataType, BDataType, CDataType>::AccDataType;

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
    bool persistent              = arg_parser.get_int("persistent");
    bool flush_cache             = arg_parser.get_bool("flush_cache");
    int rotating_count           = arg_parser.get_int("rotating_count");

    const bool preshuffle = GemmConfig::Preshuffle;

    stride_A = ck_tile::get_default_stride(M, K, stride_A, is_row_major(a_layout));
    stride_B = ck_tile::get_default_stride(K, N, stride_B, is_row_major(b_layout));
    stride_C = ck_tile::get_default_stride(M, N, stride_C, is_row_major(CLayout{}));

    ck_tile::HostTensor<ADataType> a_m_k(
        ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(a_layout)));
    ck_tile::HostTensor<BDataType> b_k_n(
        ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(b_layout)));
    ck_tile::HostTensor<CDataType> c_m_n_dev_result(
        ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

    if(init_method == 0)
    {
        ck_tile::FillUniformDistribution<ADataType>{-2.f, 2.f}(a_m_k);
        ck_tile::FillUniformDistribution<BDataType>{-2.f, 2.f}(b_k_n);
    }
    else if(init_method == 1)
    {
        ck_tile::FillMonotonicSeq<ADataType>{}(a_m_k);
        ck_tile::FillMonotonicSeq<BDataType>{}(b_k_n);
    }
    else if(init_method == 2)
    {
        ck_tile::FillUniformDistribution<ADataType>{1.f, 1.f}(a_m_k);
        ck_tile::FillUniformDistribution<BDataType>{1.f, 1.f}(b_k_n);
    }
    else
    {
        a_m_k.SetZero();
        b_k_n.SetZero();
    }

    if(!preshuffle && GemmConfig::UseStructuredSparsity)
    {
        ck_tile::AdjustToStructuredSparsity<ADataType>{}(a_m_k);
    }

    ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

    static_assert(!GemmConfig::PermuteA, "Not implemented");

    if constexpr(preshuffle)
    {
        ck_tile::HostTensor<BDataType> b_shuffle_host = [&]() {
            if constexpr(GemmConfig::TiledMMAPermuteN)
            {
                std::cout << "Run with PermuteN" << std::endl;
                return ck_tile::shuffle_b_permuteN<GemmConfig>(b_k_n);
            }
            else
            {
                std::cout << "Run without PermuteN" << std::endl;
                return ck_tile::shuffle_b<GemmConfig>(b_k_n);
            }
        }();
        // shuffled buffer B for device implementation
        if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
        {
            ck_tile::permute_vectors_i4x4_b(b_shuffle_host);
        }
        b_k_n_dev_buf.ToDevice(b_shuffle_host.data());
    }
    else
    {
        if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
        {
            // Permute vector pk_i4x4 data for device implementation
            ck_tile::HostTensor<BDataType> b_k_n_dev = b_k_n;
            if constexpr(GemmConfig::PermuteB)
            {
                permute_tensor_b<GemmConfig,
                                 decltype(b_k_n_dev),
                                 ADataType,
                                 BDataType,
                                 AccDataType,
                                 CDataType,
                                 ALayout,
                                 BLayout,
                                 CLayout>(b_k_n_dev);
            }
            ck_tile::permute_vectors_i4x4_b(b_k_n_dev);
            b_k_n_dev_buf.ToDevice(b_k_n_dev.data());
        }
        else
        {
            if constexpr(GemmConfig::PermuteB)
            {
                std::cout << "Permute for this DataType is not implemented." << std::endl;
                return false;
            }
            b_k_n_dev_buf.ToDevice(b_k_n.data());
        }
    }

    a_m_k_dev_buf.ToDevice(a_m_k.data());
    c_m_n_dev_buf.SetZero();
    c_m_n_dev_result.SetZero();

    float ave_time = invoke_gemm<GemmConfig,
                                 Invoker,
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
                                          kbatch,
                                          n_warmup,
                                          n_repeat,
                                          persistent,
                                          flush_cache,
                                          rotating_count);

    c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_byte =
        sizeof(ADataType) * M * K + sizeof(BDataType) * N * K + sizeof(CDataType) * M * N;
    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Run Gemm kernel with M=" << M << " N=" << N << " K=" << K
              << " StrideA=" << stride_A << " StrideB=" << stride_B << " StrideC=" << stride_C
              << " A_Layout=" << ALayout::name << " B_Layout =" << BLayout::name
              << " C_Layout=" << CLayout::name
              << " A_Type=" << ck_tile::DataTypeTraits<ADataType>::name
              << " B_Type=" << ck_tile::DataTypeTraits<BDataType>::name
              << " C_Type=" << ck_tile::DataTypeTraits<CDataType>::name
              << " StructuredSparsity=" << (GemmConfig::UseStructuredSparsity ? "on" : "off")
              << " Persistent=" << (persistent ? "on" : "off") << " : " << ave_time << " ms, "
              << tflops << " TFlops, " << gb_per_sec << " GB/s, " << std::endl;

    bool pass = true;

    // memory on host to store gpu reference result
    ck_tile::HostTensor<CDataType> c_m_n_ref(
        ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
    c_m_n_ref.SetZero();

    if(arg_parser.get_int("v") == 1)
    {
        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_m_k, b_k_n, c_m_n_ref);
        const float max_accumulated_value =
            *std::max_element(c_m_n_ref.mData.begin(), c_m_n_ref.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, kbatch, max_accumulated_value);
        pass = do_verify(c_m_n_dev_result, c_m_n_ref, rtol_atol, "CPU");
    }
    else if(arg_parser.get_int("v") == 2)
    {
        if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
        {
            // Restore input for B for gpu reference
            b_k_n_dev_buf.ToDevice(b_k_n.data());
        }
        if constexpr(GemmConfig::Preshuffle)
        {
            b_k_n_dev_buf.ToDevice(b_k_n.data());
        }

        // memory on device to store gpu reference result
        ck_tile::DeviceMem c_m_n_gpu_buf_ref(c_m_n_ref.get_element_space_size_in_bytes());
        c_m_n_gpu_buf_ref.SetZero();

        ADataType* d_A = static_cast<ADataType*>(a_m_k_dev_buf.GetDeviceBuffer());
        BDataType* d_B = static_cast<BDataType*>(b_k_n_dev_buf.GetDeviceBuffer());
        CDataType* d_C = static_cast<CDataType*>(c_m_n_gpu_buf_ref.GetDeviceBuffer());

        ck_tile::reference_gemm_gpu<ADataType,
                                    BDataType,
                                    AccDataType,
                                    CDataType,
                                    ALayout,
                                    BLayout,
                                    CLayout>(d_A, d_B, d_C, M, N, K, stride_A, stride_B, stride_C);

        c_m_n_gpu_buf_ref.FromDevice(c_m_n_ref.data());

        const float max_accumulated_value =
            *std::max_element(c_m_n_ref.mData.begin(), c_m_n_ref.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, kbatch, max_accumulated_value);
        pass = do_verify(c_m_n_dev_result, c_m_n_ref, rtol_atol, "GPU");
    }

    if(arg_parser.get_int("json") == 1)
    {
        dump_gemm_json_results<ALayout,
                               BLayout,
                               CLayout,
                               ADataType,
                               BDataType,
                               CDataType,
                               GemmConfig,
                               ck_tile::DataTypeTraits>(arg_parser.get_str("jsonfile"),
                                                        M,
                                                        N,
                                                        K,
                                                        stride_A,
                                                        stride_B,
                                                        stride_C,
                                                        persistent,
                                                        pass,
                                                        ave_time,
                                                        tflops,
                                                        gb_per_sec);
    }

    return pass;
}
