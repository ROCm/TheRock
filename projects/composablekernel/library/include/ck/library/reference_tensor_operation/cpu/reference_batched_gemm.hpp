// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include <stdexcept>

namespace ck {
namespace tensor_operation {
namespace host {

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct ReferenceBatchedGemm : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(const Tensor<ADataType>& a_g_m_k,
                 const Tensor<BDataType>& b_g_k_n,
                 Tensor<CDataType>& c_g_m_n,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op,
                 const int k_batch = 1)
            : a_g_m_k_{a_g_m_k},
              b_g_k_n_{b_g_k_n},
              c_g_m_n_{c_g_m_n},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op},
              k_batch_(k_batch)
        {
            if(k_batch < 1)
                throw std::invalid_argument("Batch size must be at least 1");
        }

        const Tensor<ADataType>& a_g_m_k_;
        const Tensor<BDataType>& b_g_k_n_;
        Tensor<CDataType>& c_g_m_n_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;

        const int k_batch_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        using Argument = ReferenceBatchedGemm::Argument;

        float Run(const Argument& arg)
        {
            auto f_gmk_gkn_gmn = [&](auto g, auto m, auto n) {
                const int K = arg.a_g_m_k_.mDesc.GetLengths()[2];

                // simulate fp accuacy implications of k batching
                std::vector<CDataType> partialSums(arg.k_batch_);

                for(int batchIdx = 0; batchIdx < arg.k_batch_; ++batchIdx)
                {
                    int batchSize  = std::max(K / arg.k_batch_, 1);
                    int batchStart = batchSize * batchIdx;
                    int batchEnd   = batchSize * (batchIdx + 1);
                    // add any extra round-off to last batch
                    if(batchIdx == arg.k_batch_ - 1)
                        batchEnd = K;

                    AccDataType v_acc = 0;
                    for(int k = batchStart; k < batchEnd; ++k)
                    {
                        ADataType v_a;
                        BDataType v_b;

                        arg.a_element_op_(v_a, arg.a_g_m_k_(g, m, k));
                        arg.b_element_op_(v_b, arg.b_g_k_n_(g, k, n));

                        v_acc +=
                            ck::type_convert<AccDataType>(v_a) * ck::type_convert<AccDataType>(v_b);
                    }

                    AccDataType v_c;
                    arg.c_element_op_(v_c, v_acc);
                    partialSums[batchIdx] = ck::type_convert<CDataType>(v_c);
                }

                // finally, sum up partial sums
                // note that we can't simulate the random nature of atomic additions, but at least
                // we can simulate the effect of partial sums
                AccDataType v_c = 0;
                if(arg.k_batch_ > 1)
                {
                    for(int batchIdx = 0; batchIdx < arg.k_batch_; batchIdx++)
                    {
                        // mimic the way fp operations would be done on GPU for k-batching
                        v_c = ck::type_convert<AccDataType>(ck::type_convert<CDataType>(
                            ck::type_convert<AccDataType>(v_c) +
                            ck::type_convert<AccDataType>(partialSums[batchIdx])));
                    }
                }
                else
                {
                    v_c = ck::type_convert<AccDataType>(partialSums[0]);
                }

                arg.c_g_m_n_(g, m, n) = ck::type_convert<CDataType>(v_c);
            };

            make_ParallelTensorFunctor(f_gmk_gkn_gmn,
                                       arg.c_g_m_n_.mDesc.GetLengths()[0],
                                       arg.c_g_m_n_.mDesc.GetLengths()[1],
                                       arg.c_g_m_n_.mDesc.GetLengths()[2])(
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

    static auto MakeArgument(const Tensor<ADataType>& a_g_m_k,
                             const Tensor<BDataType>& b_g_k_n,
                             Tensor<CDataType>& c_g_m_n,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op,
                             const int k_batch = 1)
    {
        return Argument{
            a_g_m_k, b_g_k_n, c_g_m_n, a_element_op, b_element_op, c_element_op, k_batch};
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
        str << "ReferenceBatchedGemm"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct ReferenceBatchedGemm_MQA : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(const Tensor<ADataType>& a_g0_g1_m_k,
                 const Tensor<BDataType>& b_g0_1_k_n,
                 Tensor<CDataType>& c_g0_g1_m_n,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
            : a_g0_g1_m_k_{a_g0_g1_m_k},
              b_g0_1_k_n_{b_g0_1_k_n},
              c_g0_g1_m_n_{c_g0_g1_m_n},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op}
        {
        }

        const Tensor<ADataType>& a_g0_g1_m_k_;
        const Tensor<BDataType>& b_g0_1_k_n_;
        Tensor<CDataType>& c_g0_g1_m_n_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        using Argument = ReferenceBatchedGemm_MQA::Argument;

        float Run(const Argument& arg)
        {
            auto f_g0g1mk_g01kn_g0g1mn = [&](auto g0, auto g1, auto m, auto n) {
                const int K = arg.a_g0_g1_m_k_.mDesc.GetLengths()[3];

                AccDataType v_acc = 0;

                for(int k = 0; k < K; ++k)
                {
                    ADataType v_a;
                    BDataType v_b;

                    arg.a_element_op_(v_a, arg.a_g0_g1_m_k_(g0, g1, m, k));
                    arg.b_element_op_(v_b, arg.b_g0_1_k_n_(g0, 0, k, n));

                    v_acc +=
                        ck::type_convert<AccDataType>(v_a) * ck::type_convert<AccDataType>(v_b);
                }

                AccDataType v_c;

                arg.c_element_op_(v_c, v_acc);

                arg.c_g0_g1_m_n_(g0, g1, m, n) = ck::type_convert<CDataType>(v_c);
            };

            make_ParallelTensorFunctor(f_g0g1mk_g01kn_g0g1mn,
                                       arg.c_g0_g1_m_n_.mDesc.GetLengths()[0],
                                       arg.c_g0_g1_m_n_.mDesc.GetLengths()[1],
                                       arg.c_g0_g1_m_n_.mDesc.GetLengths()[2],
                                       arg.c_g0_g1_m_n_.mDesc.GetLengths()[3])(
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

    static auto MakeArgument(const Tensor<ADataType>& a_g0_g1_m_k,
                             const Tensor<BDataType>& b_g0_1_k_n,
                             Tensor<CDataType>& c_g0_g1_m_n,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        return Argument{
            a_g0_g1_m_k, b_g0_1_k_n, c_g0_g1_m_n, a_element_op, b_element_op, c_element_op};
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
        str << "ReferenceBatchedGemm_MQA"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          ck::index_t QueryGroupNumber>
struct ReferenceBatchedGemm_GQA : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(const Tensor<ADataType>& a_g0_g1_m_k,
                 const Tensor<BDataType>& b_g0_gq_k_n,
                 Tensor<CDataType>& c_g0_g1_m_n,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
            : a_g0_g1_m_k_{a_g0_g1_m_k},
              b_g0_gq_k_n_{b_g0_gq_k_n},
              c_g0_g1_m_n_{c_g0_g1_m_n},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op}
        {
        }

        const Tensor<ADataType>& a_g0_g1_m_k_;
        const Tensor<BDataType>& b_g0_gq_k_n_;
        Tensor<CDataType>& c_g0_g1_m_n_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        using Argument = ReferenceBatchedGemm_GQA::Argument;

        float Run(const Argument& arg)
        {
            auto f_g0g1mk_g0gqkn_g0g1mn = [&](auto g0, auto g1, auto m, auto n) {
                const int G1 = arg.a_g0_g1_m_k_.mDesc.GetLengths()[1];
                const int K  = arg.a_g0_g1_m_k_.mDesc.GetLengths()[3];

                AccDataType v_acc = 0;

                for(int k = 0; k < K; ++k)
                {
                    ADataType v_a;
                    BDataType v_b;

                    arg.a_element_op_(v_a, arg.a_g0_g1_m_k_(g0, g1, m, k));
                    arg.b_element_op_(v_b, arg.b_g0_gq_k_n_(g0, g1 * QueryGroupNumber / G1, k, n));

                    v_acc +=
                        ck::type_convert<AccDataType>(v_a) * ck::type_convert<AccDataType>(v_b);
                }

                AccDataType v_c;

                arg.c_element_op_(v_c, v_acc);

                arg.c_g0_g1_m_n_(g0, g1, m, n) = ck::type_convert<CDataType>(v_c);
            };

            make_ParallelTensorFunctor(f_g0g1mk_g0gqkn_g0g1mn,
                                       arg.c_g0_g1_m_n_.mDesc.GetLengths()[0],
                                       arg.c_g0_g1_m_n_.mDesc.GetLengths()[1],
                                       arg.c_g0_g1_m_n_.mDesc.GetLengths()[2],
                                       arg.c_g0_g1_m_n_.mDesc.GetLengths()[3])(
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

    static auto MakeArgument(const Tensor<ADataType>& a_g0_g1_m_k,
                             const Tensor<BDataType>& b_g0_gq_k_n,
                             Tensor<CDataType>& c_g0_g1_m_n,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        return Argument{
            a_g0_g1_m_k, b_g0_gq_k_n, c_g0_g1_m_n, a_element_op, b_element_op, c_element_op};
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
        str << "ReferenceBatchedGemm_GQA"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
