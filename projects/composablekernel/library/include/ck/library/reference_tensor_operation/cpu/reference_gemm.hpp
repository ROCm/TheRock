// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/host_utility/device_prop.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/library/utility/host_tensor.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename ComputeTypeA = CDataType,
          typename ComputeTypeB = ComputeTypeA>
struct ReferenceGemm : public device::BaseOperator
{

    using ElementDataTypeA =
        ck::conditional_t<is_same_v<ComputeTypeA, ck::tf32_t>, float, ComputeTypeA>;
    using ElementDataTypeB =
        ck::conditional_t<is_same_v<ComputeTypeB, ck::tf32_t>, float, ComputeTypeB>;

    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(const Tensor<ADataType>& a_m_k,
                 const Tensor<BDataType>& b_k_n,
                 Tensor<CDataType>& c_m_n,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
            : a_m_k_{a_m_k},
              b_k_n_{b_k_n},
              c_m_n_{c_m_n},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op},
              device_name_{ck::get_device_name()}
        {
        }

        const Tensor<ADataType>& a_m_k_;
        const Tensor<BDataType>& b_k_n_;
        Tensor<CDataType>& c_m_n_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
        ::std::string device_name_; // the device which this gemm is compared with
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        using Argument = ReferenceGemm::Argument;

        float Run(const Argument& arg)
        {
            auto f_mk_kn_mn = [&](auto m, auto n) {
                const int K = arg.a_m_k_.mDesc.GetLengths()[1];

                AccDataType v_acc{0};
                ElementDataTypeA v_a{0};
                ElementDataTypeB v_b{0};

                for(int k = 0; k < K; ++k)
                {
                    if constexpr(is_same_v<ADataType, pk_i4_t>)
                    {
                        uint8_t i4x2 = arg.a_m_k_(m, k).data;
                        int8_t i4    = 0;
                        if(k % 2 == 1)
                            i4 = (i4x2 >> 0) & 0xf;
                        else
                            i4 = (i4x2 >> 4) & 0xf;
                        i4  = i4 - 8;
                        v_a = type_convert<ElementDataTypeA>(i4);
                    }
                    else if constexpr(is_same_v<ADataType, f4x2_pk_t>)
                    {
                        // TODO: add support for ColMajor layout as well
                        if(k % 2 == 1)
                            v_a = type_convert<ElementDataTypeA>(
                                f4_t(arg.a_m_k_(m, k).template unpack<>(Number<1>{})));
                        else
                            v_a = type_convert<ElementDataTypeA>(
                                f4_t(arg.a_m_k_(m, k).template unpack<>(Number<0>{})));
                    }
                    else if constexpr(is_same_v<ADataType, f6x16_pk_t> ||
                                      is_same_v<ADataType, bf6x16_pk_t> ||
                                      is_same_v<ADataType, f6x32_pk_t> ||
                                      is_same_v<ADataType, bf6x32_pk_t>)
                    {
                        v_a = type_convert<ElementDataTypeA>(
                            arg.a_m_k_(m, k).unpack(k % ADataType::packed_size));
                    }
                    else
                    {
                        arg.a_element_op_(v_a, arg.a_m_k_(m, k));
                    }

                    if constexpr(is_same_v<BDataType, pk_i4_t>)
                    {
                        uint8_t i4x2 = arg.b_k_n_(k, n).data;
                        int8_t i4    = 0;
                        if(k % 2 == 1)
                            i4 = (i4x2 >> 0) & 0xf;
                        else
                            i4 = (i4x2 >> 4) & 0xf;
                        i4  = i4 - 8;
                        v_b = type_convert<ElementDataTypeB>(i4);
                    }
                    else if constexpr(is_same_v<BDataType, f4x2_pk_t>)
                    {
                        // TODO: add support for RowMajor layout as well
                        if(k % 2 == 1)
                            v_b = type_convert<ElementDataTypeB>(
                                f4_t(arg.b_k_n_(k, n).template unpack<>(Number<1>{})));
                        else
                            v_b = type_convert<ElementDataTypeB>(
                                f4_t(arg.b_k_n_(k, n).template unpack<>(Number<0>{})));
                    }
                    else if constexpr(is_same_v<BDataType, f6x16_pk_t> ||
                                      is_same_v<BDataType, bf6x16_pk_t> ||
                                      is_same_v<BDataType, f6x32_pk_t> ||
                                      is_same_v<BDataType, bf6x32_pk_t>)
                    {
                        v_b = type_convert<ElementDataTypeB>(
                            arg.b_k_n_(k, n).unpack(k % BDataType::packed_size));
                    }
                    else
                    {
                        arg.b_element_op_(v_b, arg.b_k_n_(k, n));
                    }

                    if constexpr(is_same_v<ADataType, float> && is_same_v<BDataType, float> &&
                                 is_same_v<CDataType, float> && is_same_v<AccDataType, float> &&
                                 is_same_v<ComputeTypeA, ck::tf32_t> &&
                                 is_same_v<ComputeTypeB, ck::tf32_t>)
                    {
                        if(arg.device_name_ == "gfx942")
                        {
                            v_acc +=
                                ck::type_convert<AccDataType>(ck::type_convert<ck::tf32_t>(v_a)) *
                                ck::type_convert<AccDataType>(ck::type_convert<ck::tf32_t>(v_b));
                        }
                        else if(arg.device_name_ == "gfx950")
                        {
                            ck::bhalf_t v_a_bf16_big   = ck::type_convert<ck::bhalf_t>(v_a);
                            ck::bhalf_t v_a_bf16_small = ck::type_convert<ck::bhalf_t>(
                                v_a - type_convert<float>(v_a_bf16_big));
                            ck::bhalf_t v_b_bf16_big   = ck::type_convert<ck::bhalf_t>(v_b);
                            ck::bhalf_t v_b_bf16_small = ck::type_convert<ck::bhalf_t>(
                                v_b - type_convert<float>(v_b_bf16_big));

                            v_acc += ck::type_convert<AccDataType>(v_a_bf16_big) *
                                         ck::type_convert<AccDataType>(v_b_bf16_small) +
                                     ck::type_convert<AccDataType>(v_a_bf16_small) *
                                         ck::type_convert<AccDataType>(v_b_bf16_big) +
                                     ck::type_convert<AccDataType>(v_a_bf16_big) *
                                         ck::type_convert<AccDataType>(v_b_bf16_big);
                        }
                        else
                        {
                            throw std::runtime_error("Unsupported device: " + arg.device_name_);
                        }
                    }
                    else
                    {
                        v_acc +=
                            ck::type_convert<AccDataType>(v_a) * ck::type_convert<AccDataType>(v_b);
                    }
                }

                CDataType v_c{0};

                arg.c_element_op_(v_c, v_acc);

                arg.c_m_n_(m, n) = v_c;
            };

            make_ParallelTensorFunctor(
                f_mk_kn_mn, arg.c_m_n_.mDesc.GetLengths()[0], arg.c_m_n_.mDesc.GetLengths()[1])(
                std::thread::hardware_concurrency());

            return 0;
        }

        float Run(const device::BaseArgument* p_arg,
                  const StreamConfig& /* stream_config */ = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg));
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    bool IsSupportedArgument(const device::BaseArgument*) override { return true; }

    static auto MakeArgument(const Tensor<ADataType>& a_m_k,
                             const Tensor<BDataType>& b_k_n,
                             Tensor<CDataType>& c_m_n,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        return Argument{a_m_k, b_k_n, c_m_n, a_element_op, b_element_op, c_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    virtual std::unique_ptr<device::BaseInvoker> MakeInvokerPointer()
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "ReferenceGemm"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
