// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>

#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "gemm_multi_d_benchmark.hpp"

class GemmMultiDProfiler
{
    public:
    static GemmMultiDProfiler& instance(Setting setting)
    {
        static GemmMultiDProfiler instance{setting};
        return instance;
    }

    // Overload for single kernel benchmarking
    void benchmark(GemmMultiDProblem& gemm_multi_d_problem,
                   std::function<float(const ck_tile::GemmMultiDHostArgs<DsDataType::size()>&,
                                       const ck_tile::stream_config&)> kernel_func)
    {
        // Create a vector with a single callable that returns both name and time
        std::vector<std::function<std::tuple<std::string, float>(
            ck_tile::GemmMultiDHostArgs<DsDataType::size()>&, const ck_tile::stream_config&)>>
            callables;

        callables.push_back([kernel_func](ck_tile::GemmMultiDHostArgs<DsDataType::size()>& args,
                                          const ck_tile::stream_config& stream) {
            float time = kernel_func(args, stream);
            return std::make_tuple(std::string(KERNEL_NAME), time);
        });

        benchmark(gemm_multi_d_problem, callables);
    }

    void benchmark(
        GemmMultiDProblem& gemm_multi_d_problem,
        std::vector<std::function<std::tuple<std::string, float>(
            ck_tile::GemmMultiDHostArgs<DsDataType::size()>&, const ck_tile::stream_config&)>>&
            callables)
    {
        const ALayout layout_a   = ALayout{};
        const BLayout layout_b   = BLayout{};
        const D0Layout layout_d0 = D0Layout{};
        const D1Layout layout_d1 = D1Layout{};
        const CLayout layout_c   = CLayout{};

        gemm_multi_d_problem.stride_a_ = ck_tile::get_default_stride(gemm_multi_d_problem.m_,
                                                                     gemm_multi_d_problem.k_,
                                                                     gemm_multi_d_problem.stride_a_,
                                                                     is_row_major(layout_a));
        gemm_multi_d_problem.stride_b_ = ck_tile::get_default_stride(gemm_multi_d_problem.k_,
                                                                     gemm_multi_d_problem.n_,
                                                                     gemm_multi_d_problem.stride_b_,
                                                                     is_row_major(layout_b));
        gemm_multi_d_problem.stride_d0_ =
            ck_tile::get_default_stride(gemm_multi_d_problem.m_,
                                        gemm_multi_d_problem.n_,
                                        gemm_multi_d_problem.stride_d0_,
                                        is_row_major(layout_d0));
        gemm_multi_d_problem.stride_d1_ =
            ck_tile::get_default_stride(gemm_multi_d_problem.m_,
                                        gemm_multi_d_problem.n_,
                                        gemm_multi_d_problem.stride_d1_,
                                        is_row_major(layout_d1));
        gemm_multi_d_problem.stride_c_ = ck_tile::get_default_stride(gemm_multi_d_problem.m_,
                                                                     gemm_multi_d_problem.n_,
                                                                     gemm_multi_d_problem.stride_c_,
                                                                     is_row_major(layout_c));

        ck_tile::HostTensor<ADataType> a_m_k(
            ck_tile::host_tensor_descriptor(gemm_multi_d_problem.m_,
                                            gemm_multi_d_problem.k_,
                                            gemm_multi_d_problem.stride_a_,
                                            is_row_major(layout_a)));
        ck_tile::HostTensor<BDataType> b_k_n(
            ck_tile::host_tensor_descriptor(gemm_multi_d_problem.k_,
                                            gemm_multi_d_problem.n_,
                                            gemm_multi_d_problem.stride_b_,
                                            is_row_major(layout_b)));
        ck_tile::HostTensor<D0DataType> d0_m_n(
            ck_tile::host_tensor_descriptor(gemm_multi_d_problem.m_,
                                            gemm_multi_d_problem.n_,
                                            gemm_multi_d_problem.stride_d0_,
                                            is_row_major(layout_d0)));
        ck_tile::HostTensor<D1DataType> d1_m_n(
            ck_tile::host_tensor_descriptor(gemm_multi_d_problem.m_,
                                            gemm_multi_d_problem.n_,
                                            gemm_multi_d_problem.stride_d1_,
                                            is_row_major(layout_d1)));
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(
            ck_tile::host_tensor_descriptor(gemm_multi_d_problem.m_,
                                            gemm_multi_d_problem.n_,
                                            gemm_multi_d_problem.stride_c_,
                                            is_row_major(layout_c)));

        ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_m_k);
        ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_k_n);
        ck_tile::FillUniformDistribution<D0DataType>{-1.f, 1.f}(d0_m_n);
        ck_tile::FillUniformDistribution<D1DataType>{-1.f, 1.f}(d1_m_n);

        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d0_m_n_dev_buf(d0_m_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d1_m_n_dev_buf(d1_m_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

        a_m_k_dev_buf.ToDevice(a_m_k.mData.data());
        b_k_n_dev_buf.ToDevice(b_k_n.mData.data());
        d0_m_n_dev_buf.ToDevice(d0_m_n.mData.data());
        d1_m_n_dev_buf.ToDevice(d1_m_n.mData.data());

        c_m_n_dev_buf.SetZero();
        c_m_n_dev_result.SetZero();

        std::array<const void*, DsDataType::size()> ds_ptr_buf = {d0_m_n_dev_buf.GetDeviceBuffer(),
                                                                  d1_m_n_dev_buf.GetDeviceBuffer()};

        std::array<ck_tile::index_t, DsDataType::size()> stridesDs = {
            gemm_multi_d_problem.stride_d0_, gemm_multi_d_problem.stride_d1_};

        ck_tile::GemmMultiDHostArgs<DsDataType::size()> gemm_multi_d_args = {
            a_m_k_dev_buf.GetDeviceBuffer(),
            b_k_n_dev_buf.GetDeviceBuffer(),
            ds_ptr_buf,
            c_m_n_dev_buf.GetDeviceBuffer(),
            gemm_multi_d_problem.split_k_,
            gemm_multi_d_problem.m_,
            gemm_multi_d_problem.n_,
            gemm_multi_d_problem.k_,
            gemm_multi_d_problem.stride_a_,
            gemm_multi_d_problem.stride_b_,
            stridesDs,
            gemm_multi_d_problem.stride_c_,
        };

        ck_tile::HostTensor<CDataType> c_m_n_host_result(
            ck_tile::host_tensor_descriptor(gemm_multi_d_problem.m_,
                                            gemm_multi_d_problem.n_,
                                            gemm_multi_d_problem.stride_c_,
                                            is_row_major(layout_c)));

        if(setting_.verify_)
        {
            gemm_multi_d_host_reference(
                setting_.verify_, a_m_k, b_k_n, d0_m_n, d1_m_n, c_m_n_host_result);
        }

        for(auto& callable : callables)
        {
            auto kernel_run_result =
                callable(gemm_multi_d_args,
                         ck_tile::stream_config{
                             nullptr, true, setting_.log_, setting_.n_warmup_, setting_.n_repeat_});
            process_result(gemm_multi_d_problem,
                           c_m_n_dev_buf,
                           c_m_n_host_result,
                           c_m_n_dev_result,
                           kernel_run_result);
        }
    }

    void process_result(const GemmMultiDProblem& gemm_multi_d_problem,
                        ck_tile::DeviceMem& c_m_n_dev_buf,
                        ck_tile::HostTensor<CDataType>& c_m_n_host_result,
                        ck_tile::HostTensor<CDataType>& c_m_n_dev_result,
                        const std::tuple<std::string, float>& kernel_run_result)
    {
        auto [name, avg_time] = kernel_run_result;

        KernelInstance kernel_instance{name, gemm_multi_d_problem, {-1.0f, -1.0f, -1.0f}};

        // compute performance metric
        std::size_t flop = std::size_t(2) * gemm_multi_d_problem.m_ * gemm_multi_d_problem.n_ *
                           gemm_multi_d_problem.k_;
        std::size_t num_byte =
            sizeof(ADataType) * gemm_multi_d_problem.m_ * gemm_multi_d_problem.k_ +
            sizeof(BDataType) * gemm_multi_d_problem.n_ * gemm_multi_d_problem.k_ +
            sizeof(CDataType) * gemm_multi_d_problem.m_ * gemm_multi_d_problem.n_;

        // Dth Dimension Updates
        ck_tile::static_for<0, DsDataType::size(), 1>{}([&](auto i) {
            num_byte += sizeof(ck_tile::remove_cvref_t<std::tuple_element_t<i, DsDataType>>) *
                        gemm_multi_d_problem.m_ * gemm_multi_d_problem.n_;
            flop += sizeof(ck_tile::remove_cvref_t<std::tuple_element_t<i, DsDataType>>) *
                    gemm_multi_d_problem.m_ * gemm_multi_d_problem.n_;
        });

        // update
        kernel_instance.perf_result_.latency_   = avg_time;
        kernel_instance.perf_result_.tflops_    = static_cast<float>(flop) / 1.E9 / avg_time;
        kernel_instance.perf_result_.bandwidth_ = num_byte / 1.E6 / avg_time;

        if(setting_.log_ > 0 && !setting_.json_output_)
        {
            std::cout << kernel_instance << std::endl;
        }

        // verify result
        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());
        bool verified_correct =
            !setting_.verify_ || compare(name,
                                         gemm_multi_d_problem.k_,
                                         1, // Multi d currently supports only k_batch  = 1
                                         c_m_n_dev_result,
                                         c_m_n_host_result);

        if(verified_correct)
        {
            kernel_instances_.emplace_back(kernel_instance);
        }
        else
        {
            std::cout << "Verification failed, skip kernel: " << name << std::endl;
        }

        // clear tensor
        c_m_n_dev_buf.SetZero();
        c_m_n_dev_result.SetZero();
    }

    KernelInstance select_best_instance(Metric metric)
    {
        if(kernel_instances_.empty())
            throw std::runtime_error("Empty instances");

        auto kernel_instance = *std::max_element(kernel_instances_.begin(),
                                                 kernel_instances_.end(),
                                                 [metric](const auto& a, const auto& b) {
                                                     return PerformanceResult::compare(
                                                         b.perf_result_, a.perf_result_, metric);
                                                 });

        if(setting_.json_output_)
        {
            // Output clean JSON only
            std::cout << kernel_instance << std::endl;
        }
        else
        {
            std::cout << "**********************************" << std::endl;
            std::cout << "According to given metrics: " << get_metric_name(metric) << "\n"
                      << "Current kernel performance is: " << kernel_instance << std::endl;
            std::cout << "**********************************" << std::endl;
        }

        if(!setting_.csv_filename_.empty())
        {
            std::ofstream file(setting_.csv_filename_ + ".csv", std::ios::app);

            if(!file.is_open())
            {
                std::cerr << "Warning: Failed to open CSV file for writing." << std::endl;
            }
            else
            {
                if(file.tellp() == 0)
                {
                    file << "rocm_version,device_name,"
                         << "split_k,m,n,k,stride_a,stride_b,stride_c,"
                         << "dtype_a,dtype_b,dtype_acc,dtype_c," << "layout_a,layout_b,layout_c,"
                         << "structured_sparsity," << "name,"
                         << "latency(ms),tflops(TFlops),bandwidth(GB/s),metric\n";
                }

                const auto& problem = kernel_instance.problem_;
                const auto& name    = kernel_instance.name_;
                const auto& perf    = kernel_instance.perf_result_;

                file << get_rocm_version() << "," << ck_tile::get_device_name() << ","
                     << problem.split_k_ << "," << problem.m_ << "," << problem.n_ << ","
                     << problem.k_ << "," << problem.stride_a_ << "," << problem.stride_b_ << ","
                     << problem.stride_c_ << "," << problem.dtype_a_ << "," << problem.dtype_b_
                     << "," << problem.dtype_acc_ << "," << problem.dtype_c_ << ","
                     << problem.layout_a_ << "," << problem.layout_b_ << "," << problem.layout_c_
                     << "," << name << "," << std::fixed << std::setprecision(4) << perf.latency_
                     << "," << std::fixed << std::setprecision(4) << perf.tflops_ << ","
                     << std::fixed << std::setprecision(4) << perf.bandwidth_ << ","
                     << get_metric_name(metric) << "\n";

                if(!file)
                {
                    std::cerr << "Warning: Error occurred while writing to CSV file." << std::endl;
                }
            }
        }

        return kernel_instance;
    }

    GemmMultiDProfiler(const GemmMultiDProfiler&)            = delete;
    GemmMultiDProfiler& operator=(const GemmMultiDProfiler&) = delete;

    private:
    ~GemmMultiDProfiler() { kernel_instances_.clear(); }
    GemmMultiDProfiler(Setting setting) : setting_(setting) {}

    Setting setting_;

    std::vector<KernelInstance> kernel_instances_;
};
