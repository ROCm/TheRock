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
          typename BDataType,
          typename CDataType,
          typename D2DataType,
          typename AccDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          index_t ActivationType_ = 0,
          bool MulRoutedWeight    = true,
          typename ComputeTypeA   = AccDataType,
          typename ComputeTypeB   = AccDataType>
struct ReferenceMoeGemm1BlockScale : public device::BaseOperator
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
                 const Tensor<BDataType>& b_e_n_k,
                 const Tensor<D2DataType>& d2,
                 Tensor<CDataType>& c_t_k_n,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
            : sorted_token_ids_{sorted_token_ids},
              expert_ids_{expert_ids},
              max_token_id_{max_token_id},
              sorted_tile_size_{sorted_tile_size},
              a_t_k_{a_t_k},
              b_e_n_k_{b_e_n_k},
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
        const Tensor<BDataType>& b_e_n_k_;
        const Tensor<D2DataType>& d2_;
        Tensor<CDataType>& c_t_k_n_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        using Argument = ReferenceMoeGemm1BlockScale::Argument;

        float Run(const Argument& arg)
        {
            static_assert(ActivationType < 2, "Not supported activation type");
            const int full_n = arg.c_t_k_n_.mDesc.GetLengths()[2];
            auto f_mk_kn_mn  = [&](auto m, auto n) {
                const int K = arg.a_t_k_.mDesc.GetLengths()[1];
                AccDataType v_acc_up{0};
                ComputeTypeB v_b_up{0};
                AccDataType v_acc{0};

                ComputeTypeA v_a{0};
                ComputeTypeB v_b{0};

                const int t         = arg.sorted_token_ids_(m) & 0xffffff;
                const int topk_id   = (arg.sorted_token_ids_(m) & 0xff000000) >> 24;
                const int e         = arg.expert_ids_(m / arg.sorted_tile_size_);
                const int token_cnt = arg.a_t_k_.mDesc.GetLengths()[0];
                D2DataType v_topk_w = arg.d2_(m, 0); // expert
                if(t < token_cnt)
                {
                    for(int k = 0; k < K; ++k)
                    {
                        if constexpr(is_same_v<ADataType, pk_i4_t>)
                        {
                            uint8_t i4x2 = arg.a_t_k_(t, k).data;
                            uint8_t i4   = 0;
                            if(k % 2 == 1)
                                i4 = (i4x2 >> 0) & 0xf;
                            else
                                i4 = (i4x2 >> 4) & 0xf;
#if CK_USE_PK4_LAYOUT_SHUFFLE
                            v_a = i4_to_f32_gfx9(i4);
#else
                            v_a = i4 - 8;
#endif
                        }
                        else
                        {
                            arg.a_element_op_(v_a, arg.a_t_k_(t, k));
                        }
                        // same for B matrix
                        if constexpr(is_same_v<BDataType, pk_i4_t>)
                        {
                            uint8_t i4x2    = arg.b_e_n_k_(e, k, n).data;
                            uint8_t i4x2_up = arg.b_e_n_k_(e, k, n + full_n).data;
                            uint8_t i4      = 0;
                            uint8_t i4_up   = 0;
                            if(k % 2 == 1)
                            {
                                i4    = (i4x2 >> 0) & 0xf;
                                i4_up = (i4x2_up >> 0) & 0xf;
                            }
                            else
                            {
                                i4    = (i4x2 >> 4) & 0xf;
                                i4_up = (i4x2_up >> 4) & 0xf;
                            }
#if CK_USE_PK4_LAYOUT_SHUFFLE
                            v_b    = i4_to_f32_gfx9(i4);
                            v_b_up = i4_to_f32_gfx9(i4_up);
#else
                            v_b    = i4 - 8;
                            v_b_up = i4_up - 8;
#endif
                        }
                        else
                        {
                            arg.b_element_op_(v_b, arg.b_e_n_k_(e, k, n));
                            arg.b_element_op_(v_b_up, arg.b_e_n_k_(e, k, n + full_n));
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
                        if constexpr(is_same_v<BDataType, pk_i4_t>)
                        {
                            v_c_up *= 16;
                            v_c *= 16;
                        }
                        tensor_operation::element_wise::Silu{}(v_c, v_c);
                        arg.c_t_k_n_(t, topk_id, n) = v_c * v_c_up;
                    }
                    else if constexpr(ActivationType == 0)
                    {
                        if constexpr(is_same_v<BDataType, pk_i4_t>)
                        {
                            v_c_up *= 16;
                            v_c *= 16;
                        }
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
                             const Tensor<BDataType>& b_e_n_k,
                             const Tensor<D2DataType>& d2,
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
                        b_e_n_k,
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
        str << "ReferenceMoeGemm1BlaockScale"
            << std::endl;
        // clang-format on

        return str.str();
    }

    static float i4_to_f32_gfx9(uint8_t i4)
    {
        static std::unordered_map<uint8_t, float> u = {{0b1000, -0.5000f},
                                                       {0b1001, -0.4375f},
                                                       {0b1010, -0.3750f},
                                                       {0b1011, -0.3125f},
                                                       {0b1100, -0.2500f},
                                                       {0b1101, -0.1875f},
                                                       {0b1110, -0.1250f},
                                                       {0b1111, -0.0625f},
                                                       {0b0, +0.0000f},
                                                       {0b1, +0.0625f},
                                                       {0b10, +0.1250f},
                                                       {0b11, +0.1875f},
                                                       {0b100, +0.2500f},
                                                       {0b101, +0.3125f},
                                                       {0b110, +0.3750f},
                                                       {0b111, +0.4375f}};

        return u[i4];
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
