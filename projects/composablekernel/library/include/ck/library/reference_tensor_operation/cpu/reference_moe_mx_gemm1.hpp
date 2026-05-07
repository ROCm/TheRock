// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>
#include <unordered_map>

#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/library/utility/host_tensor.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

template <typename ADataType,
          typename AScaleDataType,
          typename BDataType,
          typename BScaleDataType,
          typename CDataType,
          typename D0DataType, // expert weight
          typename AccDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          index_t ActivationType_ = 0,
          bool MulRoutedWeight    = true,
          typename ComputeTypeA   = CDataType,
          typename ComputeTypeB   = ComputeTypeA>
struct ReferenceMoeMXGemm1 : public device::BaseOperator
{
    // Argument
    static constexpr auto ActivationType = ActivationType_;
    struct Argument : public device::BaseArgument
    {
        Argument(const Tensor<ck::index_t>& sorted_token_ids,
                 const Tensor<ck::index_t>& expert_ids,
                 const Tensor<ck::index_t>& max_token_id,
                 const index_t sorted_tile_size,
                 const Tensor<ADataType>& a_t_k,
                 const Tensor<AScaleDataType>& a_t_k_scale,
                 const Tensor<BDataType>& b_e_n_k,
                 const Tensor<BScaleDataType>& b_e_n_k_scale,
                 const Tensor<D0DataType>& d2,
                 Tensor<CDataType>& c_t_k_n,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
            : sorted_token_ids_{sorted_token_ids},
              expert_ids_{expert_ids},
              max_token_id_{max_token_id},
              sorted_tile_size_{sorted_tile_size},
              a_t_k_{a_t_k},
              a_t_k_scale_{a_t_k_scale},
              b_e_n_k_{b_e_n_k},
              b_e_n_k_scale_{b_e_n_k_scale},
              d2_{d2},
              c_t_k_n_{c_t_k_n},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op}
        {
        }

        const Tensor<ck::index_t>& sorted_token_ids_;
        const Tensor<ck::index_t>& expert_ids_;
        const Tensor<ck::index_t>& max_token_id_;
        index_t sorted_tile_size_;
        const Tensor<ADataType>& a_t_k_;
        const Tensor<AScaleDataType>& a_t_k_scale_;
        const Tensor<BDataType>& b_e_n_k_;
        const Tensor<BScaleDataType>& b_e_n_k_scale_;
        const Tensor<D0DataType>& d2_;
        Tensor<CDataType>& c_t_k_n_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        using Argument = ReferenceMoeMXGemm1::Argument;

        float Run(const Argument& arg)
        {
            static_assert(ActivationType < 2, "Not supported activation type");
            const int full_n = arg.c_t_k_n_.mDesc.GetLengths()[2];
            arg.c_t_k_n_.SetZero();
            auto f_mk_kn_mn = [&](auto m, auto n) {
                const int K                   = arg.a_t_k_.mDesc.GetLengths()[1];
                const ck::index_t SCALE_BLOCK = K / arg.b_e_n_k_scale_.mDesc.GetLengths()[1];
                AccDataType v_acc{0};
                AccDataType v_acc_up{0};
                ComputeTypeA v_a{0};
                ComputeTypeB v_b{0};
                ComputeTypeB v_b_up{0};
                const int t         = arg.sorted_token_ids_(m) & 0xffffff;
                const int topk_id   = arg.sorted_token_ids_(m) >> 24;
                const int e         = arg.expert_ids_(m / arg.sorted_tile_size_);
                const int token_cnt = arg.c_t_k_n_.mDesc.GetLengths()[0];
                D0DataType v_topk_w = arg.d2_(m, 0); // expert

                if(t < token_cnt)
                {
                    for(int k = 0; k < K; ++k)
                    {
                        auto a_f4x2  = arg.a_t_k_(t, k).data;
                        auto a_scale = arg.a_t_k_scale_(t, k / SCALE_BLOCK);
                        if constexpr(is_same_v<ADataType, f4x2_pk_t>)
                        {

                            f4_t f4 = 0;
                            if(k % 2 == 1)
                                f4 = (a_f4x2 >> 0) & 0xf;
                            else
                                f4 = (a_f4x2 >> 4) & 0xf;
                            v_a = type_convert<ComputeTypeA>(f4) *
                                  type_convert<ComputeTypeA>(a_scale);
                        }
                        else
                        {
                            v_a = type_convert<ComputeTypeA>(a_f4x2) *
                                  type_convert<ComputeTypeA>(a_scale);
                            arg.a_element_op_(v_a, v_a);
                        }
                        auto b_f4x2     = arg.b_e_n_k_(e, k, n).data;
                        auto b_f4x2_up  = arg.b_e_n_k_(e, k, n + full_n).data;
                        auto b_scale    = arg.b_e_n_k_scale_(e, k / SCALE_BLOCK, n);
                        auto b_scale_up = arg.b_e_n_k_scale_(e, k / SCALE_BLOCK, n + full_n);
                        if constexpr(is_same_v<BDataType, f4x2_pk_t>)
                        {

                            f4_t f4    = 0;
                            f4_t f4_up = 0;
                            if(k % 2 == 1)
                            {
                                f4    = (b_f4x2 >> 0) & 0xf;
                                f4_up = (b_f4x2_up >> 0) & 0xf;
                            }
                            else
                            {
                                f4    = (b_f4x2 >> 4) & 0xf;
                                f4_up = (b_f4x2_up >> 4) & 0xf;
                            }
                            v_b = type_convert<ComputeTypeB>(f4) *
                                  type_convert<ComputeTypeB>(b_scale);
                            v_b_up = type_convert<ComputeTypeB>(f4_up) *
                                     type_convert<ComputeTypeB>(b_scale_up);
                        }
                        else
                        {
                            v_b = type_convert<ComputeTypeB>(b_f4x2) *
                                  type_convert<ComputeTypeB>(b_scale);
                            v_b_up = type_convert<ComputeTypeB>(b_f4x2_up) *
                                     type_convert<ComputeTypeB>(b_scale_up);
                            arg.b_element_op_(v_b, v_b);
                            arg.b_element_op_(v_b_up, v_b_up);
                        }

                        v_acc +=
                            ck::type_convert<AccDataType>(v_a) * ck::type_convert<AccDataType>(v_b);
                        v_acc_up += ck::type_convert<AccDataType>(v_a) *
                                    ck::type_convert<AccDataType>(v_b_up);
                    }
                    CDataType v_c{0};
                    CDataType v_c_up{0};
                    if constexpr(MulRoutedWeight)
                    {
                        v_acc *= v_topk_w;
                        v_acc_up *= v_topk_w;
                    }
                    arg.c_element_op_(v_c, v_acc);
                    arg.c_element_op_(v_c_up, v_acc_up);
                    if constexpr(ActivationType == 1)
                    {
                        tensor_operation::element_wise::Silu{}(v_c, v_c);
                        arg.c_t_k_n_(t, topk_id, n) = v_c * v_c_up;
                    }
                    else if constexpr(ActivationType == 0)
                    {
                        tensor_operation::element_wise::Gelu{}(v_c, v_c);
                        arg.c_t_k_n_(t, topk_id, n) = v_c * v_c_up;
                    }
                }
            };

            const ck::index_t max_token_id = arg.max_token_id_(0);
            make_ParallelTensorFunctor(f_mk_kn_mn, max_token_id, full_n)(
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

    static auto MakeArgument(const Tensor<ck::index_t>& sorted_token_ids,
                             const Tensor<ck::index_t>& expert_ids,
                             const Tensor<ck::index_t>& max_token_id,
                             const index_t sorted_tile_size,
                             const Tensor<ADataType>& a_t_k,
                             const Tensor<AScaleDataType>& a_t_k_scale,
                             const Tensor<BDataType>& b_e_n_k,
                             const Tensor<BScaleDataType>& b_e_n_k_scale,
                             const Tensor<D0DataType>& d2,
                             Tensor<CDataType>& c_t_k_n,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        return Argument{sorted_token_ids,
                        expert_ids,
                        max_token_id,
                        sorted_tile_size,
                        a_t_k,
                        a_t_k_scale,
                        b_e_n_k,
                        b_e_n_k_scale,
                        d2,
                        c_t_k_n,
                        a_element_op,
                        b_element_op,
                        c_element_op};
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
        str << "ReferenceMoeMxGemm1"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
