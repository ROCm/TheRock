// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#if __clang_major__ >= 20
#include "ck/tensor_operation/gpu/grid/gridwise_sparse_embeddings_forward_layernorm_builtins.hpp"
#else
#include "ck/tensor_operation/gpu/grid/gridwise_sparse_embeddings_forward_layernorm.hpp"
#endif

namespace ck {
namespace tensor_operation {
namespace device {

template <typename EmbType,
          typename IndexType,
          typename GammaDataType,
          typename BetaDataType,
          typename AccDataType,
          typename OutType,
          typename EmbElementwiseOperation,
          ck::index_t BlockSize,
          ck::index_t DimClusterSize,
          ck::index_t RowClusterSize,
          ck::index_t DimPerBlock,
          ck::index_t RowPerBlock,
          ck::index_t DimThreadSize,
          ck::index_t RowVectorSize,
          ck::index_t NumEmbeddings>
struct DeviceSparseEmbeddingsForwardLayernorm : public BaseOperator
{
    static auto MakeOutputDescriptor(const index_t index_length, const index_t rows)
    {
        return make_naive_tensor_descriptor_packed(make_tuple(index_length, rows));
    }

    struct Argument : public BaseArgument
    {
        Argument(OutType* p_out,
                 const ck::Array<EmbType*, NumEmbeddings>& p_embs,
                 const ck::Array<IndexType*, NumEmbeddings>& p_indexs,
                 const GammaDataType* p_gamma,
                 const BetaDataType* p_beta,
                 const ck::index_t EmbeddingDim,
                 const ck::index_t IndexLength,
                 const AccDataType epsilon,
                 const EmbElementwiseOperation emb_elementwise_op)
            : p_out_(p_out),
              p_embs_(p_embs),
              p_indexs_(p_indexs),
              p_gamma_(p_gamma),
              p_beta_(p_beta),
              EmbeddingDim_(EmbeddingDim),
              IndexLength_(IndexLength),
              epsilon_(epsilon),
              emb_elementwise_op_(emb_elementwise_op)
        {
            grid_size_ = (IndexLength + DimClusterSize - 1) / DimClusterSize;
        }

        OutType* p_out_;
        ck::Array<EmbType*, NumEmbeddings> p_embs_;
        ck::Array<IndexType*, NumEmbeddings> p_indexs_;
        const GammaDataType* p_gamma_;
        const BetaDataType* p_beta_;
        ck::index_t EmbeddingDim_;
        ck::index_t IndexLength_;
        AccDataType epsilon_;
        EmbElementwiseOperation emb_elementwise_op_;

        size_t grid_size_;
    };

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(void* p_out,
                        const ck::Array<EmbType*, NumEmbeddings>& p_embs,
                        const ck::Array<IndexType*, NumEmbeddings>& p_indexs,
                        const void* p_gamma,
                        const void* p_beta,
                        ck::index_t EmbeddingDim,
                        ck::index_t IndexLength,
                        const AccDataType epsilon,
                        const EmbElementwiseOperation emb_elementwise_op)
    {
        return std::make_unique<Argument>(reinterpret_cast<OutType*>(p_out),
                                          p_embs,
                                          p_indexs,
                                          reinterpret_cast<const GammaDataType*>(p_gamma),
                                          reinterpret_cast<const BetaDataType*>(p_beta),
                                          EmbeddingDim,
                                          IndexLength,
                                          epsilon,
                                          emb_elementwise_op);
    }

    using GridwiseSparseEmbedding =
        GridwiseSparseEmbeddingsForwardLayernorm<EmbType,
                                                 IndexType,
                                                 GammaDataType,
                                                 BetaDataType,
                                                 AccDataType,
                                                 OutType,
                                                 decltype(MakeOutputDescriptor(1, 1)),
                                                 EmbElementwiseOperation,
                                                 BlockSize,
                                                 DimClusterSize,
                                                 RowClusterSize,
                                                 DimPerBlock,
                                                 RowPerBlock,
                                                 DimThreadSize,
                                                 RowVectorSize,
                                                 NumEmbeddings>;

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            auto out_desc = MakeOutputDescriptor(arg.IndexLength_, arg.EmbeddingDim_);
            const auto kernel_main =
                kernel_sparse_embeddings_forward_layernorm<GridwiseSparseEmbedding,
                                                           EmbType,
                                                           IndexType,
                                                           GammaDataType,
                                                           BetaDataType,
                                                           AccDataType,
                                                           OutType,
                                                           decltype(out_desc),
                                                           EmbElementwiseOperation,
                                                           NumEmbeddings>;
            float avg_time = 0;
            avg_time += launch_and_time_kernel(stream_config,
                                               kernel_main,
                                               dim3(arg.grid_size_),
                                               dim3(BlockSize),
                                               0,
                                               arg.p_out_,
                                               arg.p_embs_,
                                               arg.p_indexs_,
                                               arg.p_gamma_,
                                               arg.p_beta_,
                                               out_desc,
                                               arg.epsilon_,
                                               arg.emb_elementwise_op_);

            return (avg_time);
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        };
    };

    static bool IsSupportedArgument(const Argument* p_arg)
    {
        return (RowPerBlock == p_arg->EmbeddingDim_);
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(dynamic_cast<const Argument*>(p_arg));
    }

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer()
    {
        return std::make_unique<Invoker>();
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceSparseEmbeddingsForwardLayernorm_"<< BlockSize << "_" <<
            DimClusterSize << "x" << RowClusterSize << "_" <<
            DimPerBlock << "x" << RowPerBlock << "_" <<
            DimThreadSize << "x" << RowVectorSize;
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
