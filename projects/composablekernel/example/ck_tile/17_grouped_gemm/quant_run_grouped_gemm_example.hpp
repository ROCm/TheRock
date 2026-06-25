// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>
#include <memory>
#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp"
#include "ck_tile/ops/gemm_quant.hpp"
#include "ck_tile/host.hpp"

#include "quant_grouped_gemm_config.hpp"
#include "quant_invoke_grouped_gemm_kernel.hpp"

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
static auto calculate_rtol_atol(const ck_tile::index_t K,
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
          typename QuantGroupSize,
          ck_tile::QuantType QuantMode = ck_tile::QuantType::BQuantGrouped,
          typename CDEElementWise      = ck_tile::element_wise::PassThrough>
float invoke_gemm(int n_warmup,
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
        ave_time =
            grouped_gemm<GemmConfig,
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
                         QuantGroupSize,
                         QuantMode>(args,
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
                                         QuantGroupSize,
                                         QuantMode>(stream, group_count, kargs_ptr);
    }

    std::string op_name = "Quant Grouped Gemm (" + ck_tile::quant_type_to_string(QuantMode) + ")";

    std::size_t flop = 0, num_btype = 0;
    for(int j = 0; j < group_count; ++j)
    {
        flop += std::size_t(2) * args[j].M * args[j].N * args[j].K;

        num_btype += sizeof(ADataType) * args[j].M * args[j].K +
                     sizeof(BDataType) * args[j].K * args[j].N +
                     sizeof(CDataType) * args[j].M * args[j].N;
    }

    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
              << gb_per_sec << " GB/s, " << op_name << std::endl;

    return ave_time;
}

template <typename GemmConfig,
          typename ADataType,
          typename AQDataType,
          typename BDataType,
          typename BQDataType,
          typename CDataType,
          typename AccDataType,
          typename QuantGroupSize,
          ck_tile::QuantType QuantMode,
          typename ALayout,
          typename AQLayout,
          typename BLayout,
          typename BQLayout,
          typename CLayout>
int run_grouped_gemm_example_with_layouts(const ck_tile::ArgParser& arg_parser,
                                          const ALayout a_layout                  = ALayout{},
                                          const AQLayout aq_layout                = AQLayout{},
                                          const BLayout b_layout                  = BLayout{},
                                          const BQLayout bq_layout                = BQLayout{},
                                          [[maybe_unused]] const CLayout c_layout = CLayout{})
{
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
        if constexpr(QuantMode == ck_tile::QuantType::RowColQuant ||
                     QuantMode == ck_tile::QuantType::TensorQuant)
        {
            AQK = 1; // Row quantization: tensor shape [M, 1] or [1]
            BQK = 1; // Column quantization: tensor shape [1, N] or [1]
        }
        else if constexpr(QuantMode == ck_tile::QuantType::AQuantGrouped)
        {
            AQK = K / QuantGroupSize::kK; // Group quantization: AQK = K / GroupSize
            BQK = 0;                      // No B quantization
            if(K % QuantGroupSize::kK != 0)
            {
                throw std::runtime_error(
                    "K must be divisible by QuantGroupSize::kK for AQuantGrouped mode");
            }
        }
        else if constexpr(QuantMode == ck_tile::QuantType::BQuantGrouped)
        {
            AQK = 0;                      // No A quantization
            BQK = K / QuantGroupSize::kK; // Group quantization: BQK = K / GroupSize
            if(K % QuantGroupSize::kK != 0)
            {
                throw std::runtime_error(
                    "K must be divisible by QuantGroupSize::kK for BQuantGrouped mode");
            }
        }

        stride_As[i] = ck_tile::get_default_stride(M, K, stride_As[i], is_row_major(a_layout));
        stride_Bs[i] = ck_tile::get_default_stride(K, N, stride_Bs[i], is_row_major(b_layout));
        stride_Cs[i] = ck_tile::get_default_stride(M, N, stride_Cs[i], is_row_major(CLayout{}));
        if constexpr(QuantMode == ck_tile::QuantType::RowColQuant)
        {
            stride_AQs[i] =
                ck_tile::get_default_stride(M, 1, stride_AQs[i], is_row_major(aq_layout));
            stride_BQs[i] =
                ck_tile::get_default_stride(1, N, stride_BQs[i], is_row_major(bq_layout));
        }
        else if constexpr(QuantMode == ck_tile::QuantType::TensorQuant)
        {
            stride_AQs[i] = 1; // Tensor quantization: tensor shape [1]
            stride_BQs[i] = 1; // Tensor quantization: tensor shape [1]
        }
        else if constexpr(QuantMode == ck_tile::QuantType::AQuantGrouped)
        {
            stride_AQs[i] =
                ck_tile::get_default_stride(M, AQK, stride_AQs[i], is_row_major(aq_layout));
            stride_BQs[i] = 0; // No B quantization
        }
        else if constexpr(QuantMode == ck_tile::QuantType::BQuantGrouped)
        {
            stride_AQs[i] = 0; // No A quantization
            stride_BQs[i] =
                ck_tile::get_default_stride(BQK, N, stride_BQs[i], is_row_major(bq_layout));
        }

        a_m_k_tensors.push_back(ck_tile::HostTensor<ADataType>(
            ck_tile::host_tensor_descriptor(M, K, stride_As[i], is_row_major(a_layout))));
        b_k_n_tensors.push_back(ck_tile::HostTensor<BDataType>(
            ck_tile::host_tensor_descriptor(K, N, stride_Bs[i], is_row_major(b_layout))));
        c_m_n_tensors.push_back(ck_tile::HostTensor<CDataType>(
            ck_tile::host_tensor_descriptor(M, N, stride_Cs[i], is_row_major(CLayout{}))));
        if constexpr(QuantMode == ck_tile::QuantType::RowColQuant)
        {
            aq_tensors.push_back(ck_tile::HostTensor<AQDataType>(
                ck_tile::host_tensor_descriptor(M, AQK, stride_AQs[i], is_row_major(aq_layout))));
            bq_tensors.push_back(ck_tile::HostTensor<BQDataType>(
                ck_tile::host_tensor_descriptor(BQK, N, stride_BQs[i], is_row_major(bq_layout))));
        }
        else if constexpr(QuantMode == ck_tile::QuantType::TensorQuant)
        {
            aq_tensors.push_back(ck_tile::HostTensor<AQDataType>(
                ck_tile::host_tensor_descriptor(1, 1, stride_AQs[i], is_row_major(aq_layout))));
            bq_tensors.push_back(ck_tile::HostTensor<BQDataType>(
                ck_tile::host_tensor_descriptor(1, 1, stride_BQs[i], is_row_major(bq_layout))));
        }
        else if constexpr(QuantMode == ck_tile::QuantType::AQuantGrouped)
        {
            aq_tensors.push_back(ck_tile::HostTensor<AQDataType>(
                ck_tile::host_tensor_descriptor(M, AQK, stride_AQs[i], is_row_major(aq_layout))));
            bq_tensors.push_back(ck_tile::HostTensor<BQDataType>(
                ck_tile::host_tensor_descriptor(0, 0, stride_BQs[i], is_row_major(bq_layout))));
        }
        else if constexpr(QuantMode == ck_tile::QuantType::BQuantGrouped)
        {
            aq_tensors.push_back(ck_tile::HostTensor<AQDataType>(
                ck_tile::host_tensor_descriptor(0, 0, stride_AQs[i], is_row_major(aq_layout))));
            bq_tensors.push_back(ck_tile::HostTensor<BQDataType>(
                ck_tile::host_tensor_descriptor(BQK, N, stride_BQs[i], is_row_major(bq_layout))));
        }

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

        if constexpr(GemmConfig::PreshuffleB && QuantMode == ck_tile::QuantType::BQuantGrouped)
        {
            ck_tile::HostTensor<BDataType> b_shuffle_host =
                ck_tile::shuffle_b<GemmConfig>(b_k_n_tensors[i]);
            b_k_n_dev_buf[i]->ToDevice(b_shuffle_host.data());
        }
        else
        {
            b_k_n_dev_buf[i]->ToDevice(b_k_n_tensors[i].data());
        }

        a_m_k_dev_buf[i]->ToDevice(a_m_k_tensors[i].data());
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

    invoke_gemm<GemmConfig,
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
                QuantGroupSize,
                QuantMode>(warmup, repeat, group_count, gemm_descs);

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
            if constexpr(QuantMode == ck_tile::QuantType::RowColQuant)
            {
                ck_tile::reference_gemm_rowcol_quant<ADataType,
                                                     AQDataType,
                                                     BDataType,
                                                     BQDataType,
                                                     AccDataType,
                                                     CDataType>(a_m_k_tensors[i],
                                                                aq_tensors[i],
                                                                b_k_n_tensors[i],
                                                                bq_tensors[i],
                                                                c_m_n_host_ref);
            }
            else if constexpr(QuantMode == ck_tile::QuantType::TensorQuant)
            {
                ck_tile::reference_gemm_tensor_quant<ADataType,
                                                     AQDataType,
                                                     BDataType,
                                                     BQDataType,
                                                     AccDataType,
                                                     CDataType>(a_m_k_tensors[i],
                                                                aq_tensors[i],
                                                                b_k_n_tensors[i],
                                                                bq_tensors[i],
                                                                c_m_n_host_ref);
            }
            else if constexpr(QuantMode == ck_tile::QuantType::AQuantGrouped)
            {
                ck_tile::reference_gemm_quant<ADataType,
                                              AQDataType,
                                              BDataType,
                                              AccDataType,
                                              CDataType,
                                              QuantGroupSize,
                                              true>(
                    a_m_k_tensors[i], aq_tensors[i], b_k_n_tensors[i], c_m_n_host_ref);
            }
            else if constexpr(QuantMode == ck_tile::QuantType::BQuantGrouped)
            {
                ck_tile::reference_gemm_quant<ADataType,
                                              BQDataType,
                                              BDataType,
                                              AccDataType,
                                              CDataType,
                                              QuantGroupSize,
                                              false>(
                    a_m_k_tensors[i], bq_tensors[i], b_k_n_tensors[i], c_m_n_host_ref);
            }

            const float max_accumulated_value =
                *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
            const auto rtol_atol =
                calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
                    Ks[i], kbatch, max_accumulated_value);
            pass &= ck_tile::check_err(c_m_n_tensors[i],
                                       c_m_n_host_ref,
                                       "Error: Incorrect results!",
                                       rtol_atol.at(ck_tile::number<0>{}),
                                       rtol_atol.at(ck_tile::number<1>{}));
            std::cout << "gemm[" << i
                      << "] Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                      << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                      << std::endl;
        }
        std::cout << "The CPU verification result is:" << (pass ? "correct" : "fail") << std::endl;
    }

    return pass;
}

template <typename PrecType, ck_tile::QuantType QuantMode, typename GemmConfig>
int run_gemm_example_prec_type(const ck_tile::ArgParser& arg_parser,
                               std::string a_layout,
                               std::string b_layout)
{
    using Row   = ck_tile::tensor_layout::gemm::RowMajor;
    using Col   = ck_tile::tensor_layout::gemm::ColumnMajor;
    using Types = GemmTypeConfig<PrecType>;
    // Specific type aliases for easy access
    using ADataType      = typename Types::ADataType;
    using BDataType      = typename Types::BDataType;
    using AccDataType    = typename Types::AccDataType;
    using CDataType      = typename Types::CDataType;
    using AQDataType     = typename Types::AccDataType;
    using BQDataType     = typename Types::AccDataType;
    using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;

    if(a_layout == "R" && b_layout == "C")
    {
        return run_grouped_gemm_example_with_layouts<GemmConfig,
                                                     ADataType,
                                                     AQDataType,
                                                     BDataType,
                                                     BQDataType,
                                                     CDataType,
                                                     AccDataType,
                                                     QuantGroupSize,
                                                     QuantMode>(
            arg_parser, Row{}, Row{}, Col{}, Col{}, Row{});
    }

    if constexpr(QuantMode == ck_tile::QuantType::AQuantGrouped ||
                 (QuantMode == ck_tile::QuantType::BQuantGrouped && !GemmConfig::PreshuffleB))
    {
        if(a_layout == "R" && b_layout == "R")
        {
            return run_grouped_gemm_example_with_layouts<GemmConfig,
                                                         ADataType,
                                                         AQDataType,
                                                         BDataType,
                                                         BQDataType,
                                                         CDataType,
                                                         AccDataType,
                                                         QuantGroupSize,
                                                         QuantMode>(
                arg_parser, Row{}, Row{}, Row{}, Row{}, Row{});
        }
        else if(a_layout == "C" && b_layout == "R")
        {
            return run_grouped_gemm_example_with_layouts<GemmConfig,
                                                         ADataType,
                                                         AQDataType,
                                                         BDataType,
                                                         BQDataType,
                                                         CDataType,
                                                         AccDataType,
                                                         QuantGroupSize,
                                                         QuantMode>(
                arg_parser, Col{}, Col{}, Row{}, Row{}, Row{});
        }
        else if(a_layout == "C" && b_layout == "C")
        {
            return run_grouped_gemm_example_with_layouts<GemmConfig,
                                                         ADataType,
                                                         AQDataType,
                                                         BDataType,
                                                         BQDataType,
                                                         CDataType,
                                                         AccDataType,
                                                         QuantGroupSize,
                                                         QuantMode>(
                arg_parser, Col{}, Col{}, Col{}, Col{}, Row{});
        }
    }

    throw std::runtime_error("Unsupported data layout configuration for A,B and C tensors!");
}

template <typename PrecType, ck_tile::QuantType QuantMode>
int run_gemm_example_persistency(const ck_tile::ArgParser& arg_parser,
                                 std::string a_layout,
                                 std::string b_layout,
                                 bool persistent)
{
    if(persistent)
    {
        using GemmConfig = GemmQuantConfig<QuantMode>::template GemmConfig<PrecType, true>;
        return run_gemm_example_prec_type<PrecType, QuantMode, GemmConfig>(
            arg_parser, a_layout, b_layout);
    }
    else
    {
        using GemmConfig = GemmQuantConfig<QuantMode>::template GemmConfig<PrecType, false>;
        return run_gemm_example_prec_type<PrecType, QuantMode, GemmConfig>(
            arg_parser, a_layout, b_layout);
    }
}
