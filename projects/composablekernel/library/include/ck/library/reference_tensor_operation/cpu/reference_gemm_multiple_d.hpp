// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/library/utility/host_tensor.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

// assumption: every D matrix has the same layout and the same datatype
template <typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename CDataType,
          typename AccDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename ComputeTypeA = ADataType,
          typename ComputeTypeB = ComputeTypeA>
struct ReferenceGemmMultipleD : public device::BaseOperator
{
    using DDataType = remove_cvref_t<tuple_element_t<0, DsDataType>>;
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(const Tensor<ADataType>& a_m_k,
                 const Tensor<BDataType>& b_k_n,
                 const std::array<Tensor<DDataType>, DsDataType::Size()>& ds_m_n,
                 Tensor<CDataType>& c_m_n,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation cde_element_op)
            : a_m_k_{a_m_k},
              b_k_n_{b_k_n},
              ds_m_n_{ds_m_n},
              c_m_n_{c_m_n},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op}
        {
        }

        const Tensor<ADataType>& a_m_k_;
        const Tensor<BDataType>& b_k_n_;
        const std::array<Tensor<DDataType>, DsDataType::Size()>& ds_m_n_;
        Tensor<CDataType>& c_m_n_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        using Argument = ReferenceGemmMultipleD::Argument;

        float Run(const Argument& arg)
        {
            auto f_mk_kn_mn = [&](auto m, auto n) {
                const int K = arg.a_m_k_.mDesc.GetLengths()[1];

                AccDataType v_acc = 0;
                ComputeTypeA v_a  = 0;
                ComputeTypeB v_b  = 0;

                for(int k = 0; k < K; ++k)
                {
                    arg.a_element_op_(v_a, arg.a_m_k_(m, k));
                    arg.b_element_op_(v_b, arg.b_k_n_(k, n));

                    v_acc +=
                        ck::type_convert<AccDataType>(v_a) * ck::type_convert<AccDataType>(v_b);
                }

                CDataType v_c = 0;

                if constexpr(DsDataType::Size() == 0)
                {
                    arg.cde_element_op_(v_c, v_acc);
                }
                else if constexpr(DsDataType::Size() == 1)
                {
                    arg.cde_element_op_(v_c, v_acc, arg.ds_m_n_[0](m, n));
                }
                else if constexpr(DsDataType::Size() == 2)
                {
                    arg.cde_element_op_(v_c, v_acc, arg.ds_m_n_[0](m, n), arg.ds_m_n_[1](m, n));
                }

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
                             const std::array<Tensor<DDataType>, DsDataType::Size()>& ds_m_n,
                             Tensor<CDataType>& c_m_n,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CDEElementwiseOperation cde_element_op)
    {
        return Argument{a_m_k, b_k_n, ds_m_n, c_m_n, a_element_op, b_element_op, cde_element_op};
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
        str << "ReferenceGemmMultipleD"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
