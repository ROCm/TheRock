// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>
#include <unordered_map>
#include <mutex>

#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/library/utility/host_tensor.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename D0DataType,
          typename D1DataType,
          typename D2DataType,
          typename AccDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          bool MulRoutedWeight  = true,
          typename ComputeTypeA = CDataType,
          typename ComputeTypeB = ComputeTypeA>
struct ReferenceMoeGemm2 : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(const Tensor<ck::index_t>& sorted_token_ids,
                 const Tensor<ck::index_t>& expert_ids,
                 const Tensor<ck::index_t>& max_token_id,
                 const index_t sorted_tile_size,
                 const Tensor<ADataType>& a_t_k_k,
                 const Tensor<BDataType>& b_e_n_k,
                 const Tensor<D0DataType>& d0,
                 const Tensor<D1DataType>& d1,
                 const Tensor<D2DataType>& d2,
                 Tensor<CDataType>& c_t_n,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
            : sorted_token_ids_{sorted_token_ids},
              expert_ids_{expert_ids},
              max_token_id_{max_token_id},
              sorted_tile_size_{sorted_tile_size},
              a_t_k_k_{a_t_k_k},
              b_e_n_k_{b_e_n_k},
              d0_{d0},
              d1_{d1},
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
        const Tensor<BDataType>& b_e_n_k_;
        const Tensor<D0DataType>& d0_;
        const Tensor<D1DataType>& d1_;
        const Tensor<D2DataType>& d2_;
        Tensor<CDataType>& c_t_n_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        using Argument = ReferenceMoeGemm2::Argument;

        float Run(const Argument& arg)
        {
            std::vector<std::mutex> n_locks(arg.c_t_n_.mDesc.GetLengths()[1]);
            arg.c_t_n_.SetZero();
            auto f_mk_kn_mn = [&](auto m, auto n) {
                const int K = arg.a_t_k_k_.mDesc.GetLengths()[2];
                AccDataType v_acc{0};
                ComputeTypeA v_a{0};
                ComputeTypeB v_b{0};
                const int t         = arg.sorted_token_ids_(m) & 0xffffff;
                const int topk_id   = arg.sorted_token_ids_(m) >> 24;
                const int e         = arg.expert_ids_(m / arg.sorted_tile_size_);
                const int token_cnt = arg.c_t_n_.mDesc.GetLengths()[0];
                D2DataType v_topk_w = arg.d2_(m, 0); // expert

                if(t < token_cnt)
                {
                    for(int k = 0; k < K; ++k)
                    {
                        if constexpr(is_same_v<ADataType, pk_i4_t>)
                        {
                            uint8_t i4x2 = arg.a_t_k_(t, topk_id, k).data;
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
                            arg.a_element_op_(v_a, arg.a_t_k_k_(t, topk_id, k));
                        }
                        if constexpr(is_same_v<BDataType, pk_i4_t>)
                        {
                            uint8_t i4x2 = arg.b_e_n_k_(e, k, n).data;
                            uint8_t i4   = 0;
                            if(k % 2 == 1)
                                i4 = (i4x2 >> 0) & 0xf;
                            else
                                i4 = (i4x2 >> 4) & 0xf;
#if CK_USE_PK4_LAYOUT_SHUFFLE
                            v_b = i4_to_f32_gfx9(i4);
#else
                            v_b = i4 - 8;
#endif
                        }
                        else
                        {
                            arg.b_element_op_(v_b, arg.b_e_n_k_(e, k, n));
                        }

                        v_acc +=
                            ck::type_convert<AccDataType>(v_a) * ck::type_convert<AccDataType>(v_b);
                    }
                    CDataType v_c{0};
                    D0DataType v_d0 = arg.d0_.mDesc.GetNumOfDimension() == 3
                                          ? arg.d0_(t, topk_id, 0)
                                          : arg.d0_(t, topk_id); // a

                    D0DataType v_d1 = arg.d1_(e, n); // b
                    if constexpr(MulRoutedWeight)
                    {
                        arg.c_element_op_(v_c, v_acc, v_d0, v_d1, v_topk_w);
                    }
                    else
                    {
                        arg.c_element_op_(v_c, v_acc, v_d0, v_d1, 1.f);
                    }
                    std::lock_guard<std::mutex> lock(n_locks[n]);
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
                             const Tensor<BDataType>& b_e_n_k,
                             const Tensor<D0DataType>& d0,
                             const Tensor<D1DataType>& d1,
                             const Tensor<D2DataType>& d2,
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
                        b_e_n_k,
                        d0,
                        d1,
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

#if CK_USE_PK4_LAYOUT_SHUFFLE
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
                                                       { 0b111,
                                                         +0.4375f }};

        return u[i4];
    }
#endif
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
