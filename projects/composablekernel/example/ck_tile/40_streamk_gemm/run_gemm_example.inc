// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck_tile/ops/common/utils.hpp"

template <typename Layout>
static constexpr inline auto is_row_major(Layout)
{
    return ck_tile::bool_constant<
        std::is_same_v<ck_tile::remove_cvref_t<Layout>, ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculateRtolAtol(const ck_tile::index_t k_dim,
                       const ck_tile::index_t k_batch,
                       const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    // Calculate thresholds
    const auto relative_tolerance =
        ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
            ck_tile::integer_divide_ceil(k_dim, k_batch));
    const auto absolute_tolerance =
        ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
            max_accumulated_value / k_batch, ck_tile::integer_divide_ceil(k_dim, k_batch));
    // Calculate error due to multiple WGs working in the same C macro tile
    const auto relative_tolerance_split_k =
        ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(k_batch);
    const auto absolute_tolerance_split_k =
        ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(max_accumulated_value,
                                                                         k_batch);
    // Use higher threshold
    return ck_tile::make_tuple(std::max(relative_tolerance, relative_tolerance_split_k),
                               std::max(absolute_tolerance, absolute_tolerance_split_k));
}

template <typename GemmConfiguration,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccumulatorDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename CDEElementWise = ck_tile::element_wise::PassThrough,
          ck_tile::StreamKReductionStrategy ReductionStrategy>
std::tuple<float, ck_tile::index_t> gemm(const ck_tile::StreamKHostArgs& args,
                                         const ck_tile::stream_config& stream_config);

template <typename GemmConfiguration,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccumulatorDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename CDEElementWise = ck_tile::element_wise::PassThrough>
std::tuple<float, ck_tile::index_t> invokeGemm(ck_tile::DeviceMem& a_m_k_device_memory,
                                               ck_tile::DeviceMem& b_k_n_device_memory,
                                               ck_tile::DeviceMem& c_m_n_device_memory,
                                               ck_tile::index_t m_dim,
                                               ck_tile::index_t n_dim,
                                               ck_tile::index_t k_dim,
                                               ck_tile::index_t stride_a,
                                               ck_tile::index_t stride_b,
                                               ck_tile::index_t stride_c,
                                               int warmup_iterations,
                                               int repeat_iterations,
                                               bool flush_cache,
                                               ck_tile::StreamKReductionStrategy reduction_strategy)
{
    ck_tile::StreamKHostArgs args{a_m_k_device_memory.GetDeviceBuffer(),
                                  b_k_n_device_memory.GetDeviceBuffer(),
                                  c_m_n_device_memory.GetDeviceBuffer(),
                                  m_dim,
                                  n_dim,
                                  k_dim,
                                  stride_a,
                                  stride_b,
                                  stride_c};

    std::tuple<float, ck_tile::index_t> average_time_and_batch;

    if(reduction_strategy == ck_tile::StreamKReductionStrategy::Atomic)
    {
        average_time_and_batch = gemm<GemmConfiguration,
                                      ADataType,
                                      BDataType,
                                      DsDataType,
                                      AccumulatorDataType,
                                      CDataType,
                                      ALayout,
                                      BLayout,
                                      DsLayout,
                                      CLayout,
                                      CDEElementWise,
                                      ck_tile::StreamKReductionStrategy::Atomic>(
            args,
            ck_tile::stream_config{
                nullptr, true, 1, warmup_iterations, repeat_iterations, true, flush_cache});
    }
    else /*Reduction*/
    {
        average_time_and_batch = gemm<GemmConfiguration,
                                      ADataType,
                                      BDataType,
                                      DsDataType,
                                      AccumulatorDataType,
                                      CDataType,
                                      ALayout,
                                      BLayout,
                                      DsLayout,
                                      CLayout,
                                      CDEElementWise,
                                      ck_tile::StreamKReductionStrategy::Reduction>(
            args,
            ck_tile::stream_config{
                nullptr, true, 1, warmup_iterations, repeat_iterations, true, flush_cache});
    }

    return average_time_and_batch;
}

template <typename CDataType>
bool doVerify(const ck_tile::HostTensor<CDataType>& c_m_n_device_result,
              const ck_tile::HostTensor<CDataType>& c_m_n_reference,
              const ck_tile::tuple<double, double>& relative_absolute_tolerances,
              const char* variant)
{
    bool pass = ck_tile::check_err(c_m_n_device_result,
                                   c_m_n_reference,
                                   "Error: Incorrect results!",
                                   relative_absolute_tolerances.at(ck_tile::number<0>{}),
                                   relative_absolute_tolerances.at(ck_tile::number<1>{}));

    std::cout << "Relative error threshold: "
              << relative_absolute_tolerances.at(ck_tile::number<0>{})
              << " Absolute error threshold: "
              << relative_absolute_tolerances.at(ck_tile::number<1>{}) << std::endl;
    std::cout << "The " << variant << " verification result is:" << (pass ? "correct" : "fail")
              << std::endl;
    return pass;
}

ck_tile::StreamKReductionStrategy getReductionStrategyValue(const std::string& strategy)
{
    if(strategy == "atomic")
    {
        return ck_tile::StreamKReductionStrategy::Atomic;
    }
    else if(strategy == "reduction")
    {
        return ck_tile::StreamKReductionStrategy::Reduction;
    }
    else
    {
        throw std::runtime_error("Unsupported Stream-K reduction strategy !!!");
    }
}

template <typename GemmConfiguration,
          typename TypeConfiguration,
          typename ALayout,
          typename BLayout,
          typename CLayout>
int runGemmExampleWithLayouts(int argc,
                              char* argv[],
                              const ALayout a_layout                  = ALayout{},
                              const BLayout b_layout                  = BLayout{},
                              [[maybe_unused]] const CLayout c_layout = CLayout{})
{
    auto [result, arg_parser] = createArgs(argc, argv);
    if(!result)
        return -1;

    static_assert(!GemmConfiguration::PRESHUFFLE, "Not implemented");
    static_assert(!GemmConfiguration::USE_STRUCTURED_SPARSITY, "Not implemented");
    static_assert(!GemmConfiguration::PERMUTE_A, "Not implemented");
    static_assert(!GemmConfiguration::PERMUTE_B, "Not implemented");

    using ADataType           = typename TypeConfiguration::ADataType;
    using BDataType           = typename TypeConfiguration::BDataType;
    using AccumulatorDataType = typename TypeConfiguration::AccDataType;
    using CDataType           = typename TypeConfiguration::CDataType;

    ck_tile::index_t m_dim = arg_parser.get_int("m");
    ck_tile::index_t n_dim = arg_parser.get_int("n");
    ck_tile::index_t k_dim = arg_parser.get_int("k");

    ck_tile::index_t stride_a    = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_b    = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_c    = arg_parser.get_int("stride_c");
    int warmup_iterations        = arg_parser.get_int("warmup");
    int repeat_iterations        = arg_parser.get_int("repeat");
    ck_tile::index_t init_method = arg_parser.get_int("init");
    bool flush_cache             = arg_parser.get_bool("flush_cache");
    ck_tile::StreamKReductionStrategy reduction_strategy =
        getReductionStrategyValue(arg_parser.get_str("reduction_strategy"));

    stride_a = ck_tile::get_default_stride(m_dim, k_dim, stride_a, is_row_major(a_layout));
    stride_b = ck_tile::get_default_stride(k_dim, n_dim, stride_b, is_row_major(b_layout));
    stride_c = ck_tile::get_default_stride(m_dim, n_dim, stride_c, is_row_major(CLayout{}));

    ck_tile::HostTensor<ADataType> a_m_k_host(
        ck_tile::host_tensor_descriptor(m_dim, k_dim, stride_a, is_row_major(a_layout)));
    ck_tile::HostTensor<BDataType> b_k_n_host(
        ck_tile::host_tensor_descriptor(k_dim, n_dim, stride_b, is_row_major(b_layout)));
    ck_tile::HostTensor<CDataType> c_m_n_device_result(
        ck_tile::host_tensor_descriptor(m_dim, n_dim, stride_c, is_row_major(CLayout{})));

    if(init_method == 0)
    {
        ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_m_k_host);
        ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_k_n_host);
    }
    else if(init_method == 1)
    {
        ck_tile::FillMonotonicSeq<ADataType>{}(a_m_k_host);
        ck_tile::FillMonotonicSeq<BDataType>{}(b_k_n_host);
    }
    else if(init_method == 2)
    {
        ck_tile::FillUniformDistribution<ADataType>{1.f, 1.f}(a_m_k_host);
        ck_tile::FillUniformDistribution<BDataType>{1.f, 1.f}(b_k_n_host);
    }
    else
    {
        a_m_k_host.SetZero();
        b_k_n_host.SetZero();
    }

    ck_tile::DeviceMem a_m_k_device_memory(a_m_k_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_k_n_device_memory(b_k_n_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_m_n_device_memory(c_m_n_device_result.get_element_space_size_in_bytes());

    a_m_k_device_memory.ToDevice(a_m_k_host.data());
    b_k_n_device_memory.ToDevice(b_k_n_host.data());
    c_m_n_device_memory.SetZero();
    c_m_n_device_result.SetZero();
    auto [average_time, num_wgs_per_tile] = invokeGemm<GemmConfiguration,
                                                       ADataType,
                                                       BDataType,
                                                       ck_tile::tuple<>,
                                                       AccumulatorDataType,
                                                       CDataType,
                                                       ALayout,
                                                       BLayout,
                                                       ck_tile::tuple<>,
                                                       CLayout>(a_m_k_device_memory,
                                                                b_k_n_device_memory,
                                                                c_m_n_device_memory,
                                                                m_dim,
                                                                n_dim,
                                                                k_dim,
                                                                stride_a,
                                                                stride_b,
                                                                stride_c,
                                                                warmup_iterations,
                                                                repeat_iterations,
                                                                flush_cache,
                                                                reduction_strategy);

    c_m_n_device_memory.FromDevice(c_m_n_device_result.data());

    std::size_t flop     = std::size_t(2) * m_dim * n_dim * k_dim;
    std::size_t num_byte = sizeof(ADataType) * m_dim * k_dim + sizeof(BDataType) * n_dim * k_dim +
                           sizeof(CDataType) * m_dim * n_dim;
    float tflops     = static_cast<float>(flop) / 1.E9 / average_time;
    float gb_per_sec = num_byte / 1.E6 / average_time;
    std::cout << "Run Gemm kernel with M=" << m_dim << " N=" << n_dim << " K=" << k_dim
              << " StrideA=" << stride_a << " StrideB=" << stride_b << " StrideC=" << stride_c
              << " A_Layout=" << ALayout::name << " B_Layout=" << BLayout::name
              << " C_Layout=" << CLayout::name
              << " A_Type=" << ck_tile::DataTypeTraits<ADataType>::name
              << " B_Type=" << ck_tile::DataTypeTraits<BDataType>::name
              << " C_Type=" << ck_tile::DataTypeTraits<CDataType>::name
              << " reduction_strategy=" << arg_parser.get_str("reduction_strategy") << " "
              << " persistent_dp=" << arg_parser.get_str("persistent_dp") << " " << average_time
              << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, " << std::endl;
    bool pass = false;

    // Memory on host to store gpu reference result
    ck_tile::HostTensor<CDataType> c_m_n_reference(
        ck_tile::host_tensor_descriptor(m_dim, n_dim, stride_c, is_row_major(CLayout{})));
    c_m_n_reference.SetZero();

    if(arg_parser.get_int("v") == 1) // Validate on the CPU
    {
        ck_tile::reference_gemm<ADataType, BDataType, AccumulatorDataType, CDataType>(
            a_m_k_host, b_k_n_host, c_m_n_reference);
        const float max_accumulated_value =
            *std::max_element(c_m_n_reference.mData.begin(), c_m_n_reference.mData.end());
        const auto relative_absolute_tolerances =
            calculateRtolAtol<ADataType, BDataType, AccumulatorDataType, CDataType>(
                k_dim, num_wgs_per_tile, max_accumulated_value);
        pass = doVerify(c_m_n_device_result, c_m_n_reference, relative_absolute_tolerances, "CPU");
    }
    else if(arg_parser.get_int("v") == 2) // Validate on the GPU
    {
        // Memory on device to store gpu reference result
        ck_tile::DeviceMem c_m_n_gpu_buffer_reference(
            c_m_n_reference.get_element_space_size_in_bytes());
        c_m_n_gpu_buffer_reference.SetZero();

        ADataType* d_A = static_cast<ADataType*>(a_m_k_device_memory.GetDeviceBuffer());
        BDataType* d_B = static_cast<BDataType*>(b_k_n_device_memory.GetDeviceBuffer());
        CDataType* d_C = static_cast<CDataType*>(c_m_n_gpu_buffer_reference.GetDeviceBuffer());
        ck_tile::reference_gemm_gpu<ADataType,
                                    BDataType,
                                    AccumulatorDataType,
                                    CDataType,
                                    ALayout,
                                    BLayout,
                                    CLayout>(
            d_A, d_B, d_C, m_dim, n_dim, k_dim, stride_a, stride_b, stride_c);
        c_m_n_gpu_buffer_reference.FromDevice(c_m_n_reference.data());

        const float max_accumulated_value =
            *std::max_element(c_m_n_reference.mData.begin(), c_m_n_reference.mData.end());
        const auto relative_absolute_tolerances =
            calculateRtolAtol<ADataType, BDataType, AccumulatorDataType, CDataType>(
                k_dim, num_wgs_per_tile, max_accumulated_value);
        pass = doVerify(c_m_n_device_result, c_m_n_reference, relative_absolute_tolerances, "GPU");
    }

    return pass;
}
