// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_abd.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/gemm_multi_abd.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

namespace ck {
namespace profiler {

// this function is also defined in CK but because of the way we use it in
// profile_gemm_multi_impl, it requires the arguments to not be const
template <typename... X, typename... Y>
auto concat_tuple_of_refs(ck::Tuple<X&...>& tx, ck::Tuple<Y&...>& ty)
{
    return ck::unpack2(
        [&](auto&&... zs) { return ck::Tuple<decltype(zs)...>{ck::forward<decltype(zs)>(zs)...}; },
        tx,
        ty);
}

template <typename AsDataType,
          typename BsDataType,
          typename AccDataType,
          typename DsDataType,
          typename EDataType,
          typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename ELayout,
          typename AElementOp,
          typename BElementOp,
          typename CDEElementOp>
bool profile_gemm_multi_abd_impl(int do_verification,
                                 int init_method,
                                 bool /*do_log*/,
                                 bool time_kernel,
                                 int M,
                                 int N,
                                 int K,
                                 int StrideA,
                                 int StrideB,
                                 int StrideD,
                                 int StrideE)
{
    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            using namespace ck::literals;

            if(is_same<decltype(layout), tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    static constexpr index_t NumATensor = AsDataType::Size();
    auto as_m_k                         = generate_tuple(
        [&](auto i) {
            using ADataType = remove_cvref_t<tuple_element_t<i.value, AsDataType>>;
            using ALayout   = remove_cvref_t<tuple_element_t<i.value, AsLayout>>;

            return Tensor<ADataType>(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
        },
        Number<NumATensor>{});

    static constexpr index_t NumBTensor = BsDataType::Size();
    auto bs_k_n                         = generate_tuple(
        [&](auto i) {
            using BDataType = remove_cvref_t<tuple_element_t<i.value, BsDataType>>;
            using BLayout   = remove_cvref_t<tuple_element_t<i.value, BsLayout>>;

            return Tensor<BDataType>(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
        },
        Number<NumBTensor>{});

    static constexpr index_t NumDTensor = DsDataType::Size();
    auto ds_m_n                         = generate_tuple(
        [&](auto i) {
            using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
            using DLayout   = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

            return Tensor<DDataType>(f_host_tensor_descriptor(M, N, StrideD, DLayout{}));
        },
        Number<NumDTensor>{});

    Tensor<EDataType> e_m_n_device_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));
    Tensor<EDataType> e_m_n_host_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));

    static_for<0, NumATensor, 1>{}(
        [&](auto i) { std::cout << "a" << i.value << "_m_k: " << as_m_k(i).mDesc << std::endl; });
    static_for<0, NumBTensor, 1>{}(
        [&](auto i) { std::cout << "b" << i.value << "_k_n: " << bs_k_n(i).mDesc << std::endl; });
    static_for<0, NumDTensor, 1>{}(
        [&](auto i) { std::cout << "d" << i.value << "_m_n: " << ds_m_n(i).mDesc << std::endl; });
    std::cout << "e_m_n: " << e_m_n_device_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        static_for<0, NumATensor, 1>{}([&](auto i) {
            using ADataType = remove_cvref_t<tuple_element_t<i.value, AsDataType>>;

            as_m_k(i).GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        });

        static_for<0, NumBTensor, 1>{}([&](auto i) {
            using BDataType = remove_cvref_t<tuple_element_t<i.value, BsDataType>>;

            bs_k_n(i).GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        });

        static_for<0, NumDTensor, 1>{}([&](auto i) {
            using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

            ds_m_n(i).GenerateTensorValue(GeneratorTensor_2<DDataType>{-5, 5});
        });

        break;
    default:
        static_for<0, NumATensor, 1>{}([&](auto i) {
            using ADataType = remove_cvref_t<tuple_element_t<i.value, AsDataType>>;

            as_m_k(i).GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        });

        static_for<0, NumBTensor, 1>{}([&](auto i) {
            using BDataType = remove_cvref_t<tuple_element_t<i.value, BsDataType>>;

            bs_k_n(i).GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        });

        static_for<0, NumDTensor, 1>{}([&](auto i) {
            using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

            ds_m_n(i).GenerateTensorValue(GeneratorTensor_3<DDataType>{0.0, 1.0});
        });
    }

    const auto a_element_op   = AElementOp{};
    const auto b_element_op   = BElementOp{};
    const auto cde_element_op = CDEElementOp{};

    using DeviceOp = ck::tensor_operation::device::DeviceGemmMultipleABD<AsLayout,
                                                                         BsLayout,
                                                                         DsLayout,
                                                                         ELayout,
                                                                         AsDataType,
                                                                         BsDataType,
                                                                         DsDataType,
                                                                         EDataType,
                                                                         AElementOp,
                                                                         BElementOp,
                                                                         CDEElementOp>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    // run reference
    if(do_verification)
    {
        using PassThrough = ck::tensor_operation::element_wise::PassThrough;
        Tensor<AccDataType> c_m_n({M, N});

        using AComputeType =
            typename std::conditional<(NumATensor > 1),
                                      EDataType,
                                      remove_cvref_t<tuple_element_t<0, AsDataType>>>::type;

        Tensor<AComputeType> a_m_k({M, K});
        for(int m = 0; m < M; ++m)
        {
            for(int k = 0; k < K; ++k)
            {
                // result
                auto data_refs1 = ck::tie(a_m_k(m, k));
                // inputs
                auto data_refs2 =
                    generate_tie([&](auto i) -> auto& { return as_m_k(Number<i>{})(m, k); },
                                 Number<NumATensor>{});
                auto data_refs = concat_tuple_of_refs(data_refs1, data_refs2);
                unpack(a_element_op, data_refs);
            }
        }

        using BComputeType =
            typename std::conditional<(NumBTensor > 1),
                                      EDataType,
                                      remove_cvref_t<tuple_element_t<0, BsDataType>>>::type;

        Tensor<BComputeType> b_k_n({K, N});
        for(int k = 0; k < K; ++k)
        {
            for(int n = 0; n < N; ++n)
            {
                // result
                auto data_refs1 = ck::tie(b_k_n(k, n));
                // inputs
                auto data_refs2 =
                    generate_tie([&](auto i) -> auto& { return bs_k_n(Number<i>{})(k, n); },
                                 Number<NumBTensor>{});
                auto data_refs = concat_tuple_of_refs(data_refs1, data_refs2);
                unpack(b_element_op, data_refs);
            }
        }

        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<AComputeType,
                                                                                BComputeType,
                                                                                AccDataType,
                                                                                AccDataType,
                                                                                PassThrough,
                                                                                PassThrough,
                                                                                PassThrough>;
        auto ref_gemm               = ReferenceGemmInstance{};
        auto ref_invoker            = ref_gemm.MakeInvoker();

        auto ref_argument =
            ref_gemm.MakeArgument(a_m_k, b_k_n, c_m_n, PassThrough{}, PassThrough{}, PassThrough{});

        ref_invoker.Run(ref_argument);

        for(int m = 0; m < M; ++m)
        {
            for(int n = 0; n < N; ++n)
            {
                // compulsory
                auto data_refs1 = ck::tie(e_m_n_host_result(m, n), c_m_n(m, n));
                // optional (if multiple Ds)
                auto data_refs2 =
                    generate_tie([&](auto i) -> auto& { return ds_m_n(Number<i>{})(m, n); },
                                 Number<NumDTensor>{});
                auto data_refs = concat_tuple_of_refs(data_refs1, data_refs2);
                unpack(cde_element_op, data_refs);
            }
        }
    }

    std::array<DeviceMem*, NumATensor> as_device_buf;
    static_for<0, NumATensor, 1>{}([&](auto i) {
        using ADataType  = remove_cvref_t<tuple_element_t<i.value, AsDataType>>;
        as_device_buf[i] = new DeviceMem(sizeof(ADataType) * as_m_k(i).mDesc.GetElementSpaceSize());
    });

    std::array<DeviceMem*, NumBTensor> bs_device_buf;
    static_for<0, NumBTensor, 1>{}([&](auto i) {
        using BDataType  = remove_cvref_t<tuple_element_t<i.value, BsDataType>>;
        bs_device_buf[i] = new DeviceMem(sizeof(BDataType) * bs_k_n(i).mDesc.GetElementSpaceSize());
    });

    std::array<DeviceMem*, NumDTensor> ds_device_buf;
    static_for<0, NumDTensor, 1>{}([&](auto i) {
        using DDataType  = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
        ds_device_buf[i] = new DeviceMem(sizeof(DDataType) * ds_m_n(i).mDesc.GetElementSpaceSize());
    });

    DeviceMem e_device_buf(sizeof(EDataType) * e_m_n_device_result.mDesc.GetElementSpaceSize());

    static_for<0, NumATensor, 1>{}(
        [&](auto i) { as_device_buf[i]->ToDevice(as_m_k(i).mData.data()); });

    static_for<0, NumBTensor, 1>{}(
        [&](auto i) { bs_device_buf[i]->ToDevice(bs_k_n(i).mData.data()); });

    static_for<0, NumDTensor, 1>{}(
        [&](auto i) { ds_device_buf[i]->ToDevice(ds_m_n(i).mData.data()); });

    std::string best_op_name;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    bool pass = true;

    // profile device operation instances
    for(auto& op_ptr : op_ptrs)
    {
        std::array<const void*, NumATensor> as_pointer;
        std::array<ck::index_t, NumATensor> as_stride;
        static_for<0, NumATensor, 1>{}([&](auto i) {
            as_pointer[i] = as_device_buf[i]->GetDeviceBuffer();
            as_stride[i]  = StrideA;
        });

        std::array<const void*, NumBTensor> bs_pointer;
        std::array<ck::index_t, NumBTensor> bs_stride;
        static_for<0, NumBTensor, 1>{}([&](auto i) {
            bs_pointer[i] = bs_device_buf[i]->GetDeviceBuffer();
            bs_stride[i]  = StrideB;
        });
        std::array<const void*, NumDTensor> ds_pointer;
        std::array<ck::index_t, NumDTensor> ds_stride;
        static_for<0, NumDTensor, 1>{}([&](auto i) {
            ds_pointer[i] = ds_device_buf[i]->GetDeviceBuffer();
            ds_stride[i]  = StrideD;
        });

        auto argument_ptr = op_ptr->MakeArgumentPointer(as_pointer,
                                                        bs_pointer,
                                                        ds_pointer,
                                                        e_device_buf.GetDeviceBuffer(),
                                                        M,
                                                        N,
                                                        K,
                                                        as_stride,
                                                        bs_stride,
                                                        ds_stride,
                                                        StrideE,
                                                        a_element_op,
                                                        b_element_op,
                                                        cde_element_op);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            // re-init E to zero before profiling a kernel
            e_device_buf.SetZero();

            float ave_time =
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

            std::size_t flop = std::size_t(2) * M * N * K;

            std::size_t sizeADataType = 0;
            static_for<0, NumATensor, 1>{}([&](auto i) {
                using ADataType = remove_cvref_t<tuple_element_t<i.value, AsDataType>>;
                sizeADataType   = std::max(sizeADataType, sizeof(ADataType));
            });
            std::size_t sizeBDataType = 0;
            static_for<0, NumBTensor, 1>{}([&](auto i) {
                using BDataType = remove_cvref_t<tuple_element_t<i.value, BsDataType>>;
                sizeBDataType   = std::max(sizeBDataType, sizeof(BDataType));
            });

            std::size_t num_btype =
                sizeADataType * M * K + sizeBDataType * K * N + sizeof(EDataType) * M * N;

            float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

            float gb_per_sec = num_btype / 1.E6 / ave_time;

            std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << op_name << std::endl;
            if(tflops > best_tflops)
            {
                best_op_name    = op_name;
                best_tflops     = tflops;
                best_ave_time   = ave_time;
                best_gb_per_sec = gb_per_sec;
            }

            if(do_verification)
            {
                e_device_buf.FromDevice(e_m_n_device_result.mData.data());
                pass = pass && ck::utils::check_err(e_m_n_device_result, e_m_n_host_result);
            }
        }
        else
        {
            std::cout << op_name << " does not support this problem" << std::endl;
        }
    }

    static_for<0, NumATensor, 1>{}([&](auto i) { delete as_device_buf[i]; });

    static_for<0, NumBTensor, 1>{}([&](auto i) { delete bs_device_buf[i]; });

    static_for<0, NumDTensor, 1>{}([&](auto i) { delete ds_device_buf[i]; });

    std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
              << best_gb_per_sec << " GB/s, " << best_op_name << std::endl;

    return pass;
}

} // namespace profiler
} // namespace ck
