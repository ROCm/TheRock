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
          typename D0DataType, // expert weight
          typename CDataType,
          typename AccDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          bool MulRoutedWeight  = true,
          typename ComputeTypeA = CDataType,
          typename ComputeTypeB = ComputeTypeA>
struct ReferenceMoeMXGemm2 : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(const Tensor<ck::index_t>& sorted_token_ids,
                 const Tensor<ck::index_t>& expert_ids,
                 const Tensor<ck::index_t>& max_token_id,
                 const index_t sorted_tile_size,
                 const Tensor<ADataType>& a_t_k_k,
                 const Tensor<AScaleDataType>& a_t_k_k_scale,
                 const Tensor<BDataType>& b_e_n_k,
                 const Tensor<BScaleDataType>& b_e_n_k_scale,
                 const Tensor<D0DataType>& d2,
                 Tensor<CDataType>& c_t_n,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
            : sorted_token_ids_{sorted_token_ids},
              expert_ids_{expert_ids},
              max_token_id_{max_token_id},
              sorted_tile_size_{sorted_tile_size},
              a_t_k_k_{a_t_k_k},
              a_t_k_k_scale_{a_t_k_k_scale},
              b_e_n_k_{b_e_n_k},
              b_e_n_k_scale_{b_e_n_k_scale},
              d2_{d2},
              c_t_n_{c_t_n},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op}
        {
        }

        const Tensor<ck::index_t>& sorted_token_ids_;
        const Tensor<ck::index_t>& expert_ids_;
        const Tensor<ck::index_t>& max_token_id_;
        index_t sorted_tile_size_;
        const Tensor<ADataType>& a_t_k_k_;
        const Tensor<AScaleDataType>& a_t_k_k_scale_;
        const Tensor<BDataType>& b_e_n_k_;
        const Tensor<BScaleDataType>& b_e_n_k_scale_;
        const Tensor<D0DataType>& d2_;
        Tensor<CDataType>& c_t_n_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        using Argument = ReferenceMoeMXGemm2::Argument;

        float Run(const Argument& arg)
        {
            arg.c_t_n_.SetZero();
            auto f_mk_kn_mn = [&](auto m, auto n) {
                const int K                   = arg.a_t_k_k_.mDesc.GetLengths()[2];
                const ck::index_t SCALE_BLOCK = K / arg.b_e_n_k_scale_.mDesc.GetLengths()[1];
                AccDataType v_acc{0};
                ComputeTypeA v_a{0};
                ComputeTypeB v_b{0};
                const int t         = arg.sorted_token_ids_(m) & 0xffffff;
                const int topk_id   = arg.sorted_token_ids_(m) >> 24;
                const int e         = arg.expert_ids_(m / arg.sorted_tile_size_);
                const int token_cnt = arg.c_t_n_.mDesc.GetLengths()[0];
                D0DataType v_topk_w = arg.d2_(m, 0); // expert

                if(t < token_cnt)
                {
                    for(int k = 0; k < K; ++k)
                    {
                        if constexpr(is_same_v<ADataType, f4x2_pk_t>)
                        {
                            auto f4x2    = arg.a_t_k_k_(t, topk_id, k).data;
                            auto a_scale = arg.a_t_k_k_scale_(t, topk_id, k / SCALE_BLOCK);

                            f4_t f4 = 0;
                            if(k % 2 == 1)
                                f4 = (f4x2 >> 0) & 0xf;
                            else
                                f4 = (f4x2 >> 4) & 0xf;

                            v_a = type_convert<ComputeTypeA>(f4) *
                                  type_convert<ComputeTypeA>(a_scale);
                        }
                        else
                        {
                            arg.a_element_op_(
                                v_a, type_convert<ComputeTypeA>(arg.a_t_k_k_(t, topk_id, k)));
                        }
                        if constexpr(is_same_v<BDataType, f4x2_pk_t>)
                        {
                            auto f4x2    = arg.b_e_n_k_(e, k, n).data;
                            auto b_scale = arg.b_e_n_k_scale_(e, k / SCALE_BLOCK, n);

                            f4_t f4 = 0;
                            if(k % 2 == 1)
                                f4 = (f4x2 >> 0) & 0xf;
                            else
                                f4 = (f4x2 >> 4) & 0xf;

                            v_b = type_convert<ComputeTypeB>(f4) *
                                  type_convert<ComputeTypeB>(b_scale);
                        }
                        else
                        {
                            arg.b_element_op_(v_b,
                                              type_convert<ComputeTypeB>(arg.b_e_n_k_(e, k, n)));
                        }

                        v_acc +=
                            ck::type_convert<AccDataType>(v_a) * ck::type_convert<AccDataType>(v_b);
                    }
                    CDataType v_c{0};
                    if constexpr(MulRoutedWeight)
                    {
                        arg.c_element_op_(v_c, v_acc, 1.f, 1.f, v_topk_w); // hacky, need to fix
                    }
                    else
                    {
                        arg.c_element_op_(v_c, v_acc, 1.f, 1.f, 1.f);
                    }
                    arg.c_t_n_(t, n) += v_c;
                }
            };

            const std::size_t max_token_id = arg.max_token_id_(0);
            // avoid parallelizing over the m dim to prevent data race
            make_ParallelTensorFunctor(
                [&](auto n) {
                    for(std::size_t m = 0; m < max_token_id; ++m)
                        f_mk_kn_mn(m, n);
                },
                arg.c_t_n_.mDesc.GetLengths()[1])(std::thread::hardware_concurrency());

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
                             const Tensor<ADataType>& a_t_k_k,
                             const Tensor<AScaleDataType>& a_t_k_k_scale,
                             const Tensor<BDataType>& b_e_n_k,
                             const Tensor<BScaleDataType>& b_e_n_k_scale,
                             const Tensor<D0DataType>& d2,
                             Tensor<CDataType>& c_t_n,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        return Argument{sorted_token_ids,
                        expert_ids,
                        max_token_id,
                        sorted_tile_size,
                        a_t_k_k,
                        a_t_k_k_scale,
                        b_e_n_k,
                        b_e_n_k_scale,
                        d2,
                        c_t_n,
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
        str << "ReferenceMoeGemm2"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
