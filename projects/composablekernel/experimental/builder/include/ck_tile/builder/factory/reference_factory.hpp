// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ref/naive_grouped_conv_fwd_gpu.hpp"
#include "ck_tile/ref/naive_grouped_conv_bwd_data_gpu.hpp"
#include "ck_tile/ref/naive_grouped_conv_bwd_weight_gpu.hpp"
#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/conv_algorithm_concepts.hpp"
#include "ck_tile/builder/types.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_type.hpp"
#include "ck_tile/builder/factory/reference_common.hpp"
#include "ck_tile/core.hpp"
#include <memory>

namespace ck_tile::builder::factory {

// Unified Factory for GPU Reference Convolution (all directions)
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          StringLiteral VERSION>
struct ReferenceFactory
{
    // Validate that only PassThrough elementwise operations are specified
    static constexpr auto kValidation = (internal::ValidateReferenceSignature<SIGNATURE>(), 0);

    static constexpr size_t SPATIAL_DIM = SIGNATURE.spatial_dim;
    using Types                         = internal::FwdConvTensorDataTypes<SIGNATURE>;

    using InDataType  = typename Types::ADataType;
    using WeiDataType = typename Types::BDataType;
    using OutDataType = typename Types::EDataType;

    struct Instance
    {
        // Store template parameters for InstanceTraits reflection
        static constexpr auto kSignature = SIGNATURE;
        static constexpr auto kAlgorithm = ALGORITHM;
        static constexpr auto kVersion   = VERSION;

        // Argument and Invoker types depend on direction
        // Forward: const input, const weight, mutable output
        // Backward Data: mutable input, const weight, const output_grad
        // Backward Weight: const input, mutable weight_grad, const output_grad

        // Use appropriate Argument type based on direction
        using Argument = std::conditional_t<
            ConvDirectionIsForward<SIGNATURE>,
            internal::ReferenceConvArgument<const InDataType*, const WeiDataType*, OutDataType*>,
            std::conditional_t<
                ConvDirectionIsBackwardData<SIGNATURE>,
                internal::
                    ReferenceConvArgument<InDataType*, const WeiDataType*, const OutDataType*>,
                internal::
                    ReferenceConvArgument<const InDataType*, WeiDataType*, const OutDataType*>>>;

        // Invoker calls the appropriate reference implementation based on direction
        struct Invoker
        {
            float Run(const Argument* arg, const StreamConfig& stream_config = StreamConfig{})
            {
                (void)stream_config; // Unused for reference implementation

                if constexpr(ConvDirectionIsForward<SIGNATURE>)
                {
                    ck_tile::
                        naive_grouped_conv_fwd<SPATIAL_DIM, InDataType, WeiDataType, OutDataType>(
                            arg->input_,
                            arg->weight_,
                            arg->output_,
                            arg->G_,
                            arg->N_,
                            arg->K_,
                            arg->C_,
                            arg->input_spatial_,
                            arg->filter_spatial_,
                            arg->output_spatial_,
                            arg->strides_,
                            arg->dilations_,
                            arg->left_pads_);
                }
                else if constexpr(ConvDirectionIsBackwardData<SIGNATURE>)
                {
                    ck_tile::naive_grouped_conv_bwd_data<SPATIAL_DIM,
                                                         InDataType,
                                                         WeiDataType,
                                                         OutDataType>(arg->input_,
                                                                      arg->weight_,
                                                                      arg->output_,
                                                                      arg->G_,
                                                                      arg->N_,
                                                                      arg->K_,
                                                                      arg->C_,
                                                                      arg->input_spatial_,
                                                                      arg->filter_spatial_,
                                                                      arg->output_spatial_,
                                                                      arg->strides_,
                                                                      arg->dilations_,
                                                                      arg->left_pads_);
                }
                else if constexpr(ConvDirectionIsBackwardWeight<SIGNATURE>)
                {
                    ck_tile::naive_grouped_conv_bwd_weight<SPATIAL_DIM,
                                                           InDataType,
                                                           WeiDataType,
                                                           OutDataType>(arg->input_,
                                                                        arg->weight_,
                                                                        arg->output_,
                                                                        arg->G_,
                                                                        arg->N_,
                                                                        arg->K_,
                                                                        arg->C_,
                                                                        arg->input_spatial_,
                                                                        arg->filter_spatial_,
                                                                        arg->output_spatial_,
                                                                        arg->strides_,
                                                                        arg->dilations_,
                                                                        arg->left_pads_);
                }

                return 0.0f; // Reference implementation doesn't track timing
            }
        };

        // Direct Run method (simpler interface, direction-agnostic)
        template <typename InPtrType, typename WeiPtrType, typename OutPtrType>
        static void Run(InPtrType* input,
                        WeiPtrType* weight,
                        OutPtrType* output,
                        int G,
                        int N,
                        int K,
                        int C,
                        const std::vector<ck_tile::long_index_t>& input_spatial,
                        const std::vector<ck_tile::long_index_t>& filter_spatial,
                        const std::vector<ck_tile::long_index_t>& output_spatial,
                        const std::vector<ck_tile::long_index_t>& strides,
                        const std::vector<ck_tile::long_index_t>& dilations,
                        const std::vector<ck_tile::long_index_t>& left_pads)
        {
            if constexpr(ConvDirectionIsForward<SIGNATURE>)
            {
                ck_tile::naive_grouped_conv_fwd<SPATIAL_DIM, InDataType, WeiDataType, OutDataType>(
                    static_cast<const InDataType*>(input),
                    static_cast<const WeiDataType*>(weight),
                    static_cast<OutDataType*>(output),
                    G,
                    N,
                    K,
                    C,
                    input_spatial,
                    filter_spatial,
                    output_spatial,
                    strides,
                    dilations,
                    left_pads);
            }
            else if constexpr(ConvDirectionIsBackwardData<SIGNATURE>)
            {
                ck_tile::
                    naive_grouped_conv_bwd_data<SPATIAL_DIM, InDataType, WeiDataType, OutDataType>(
                        static_cast<InDataType*>(input),
                        static_cast<const WeiDataType*>(weight),
                        static_cast<const OutDataType*>(output),
                        G,
                        N,
                        K,
                        C,
                        input_spatial,
                        filter_spatial,
                        output_spatial,
                        strides,
                        dilations,
                        left_pads);
            }
            else if constexpr(ConvDirectionIsBackwardWeight<SIGNATURE>)
            {
                ck_tile::naive_grouped_conv_bwd_weight<SPATIAL_DIM,
                                                       InDataType,
                                                       WeiDataType,
                                                       OutDataType>(
                    static_cast<const InDataType*>(input),
                    static_cast<WeiDataType*>(weight),
                    static_cast<const OutDataType*>(output),
                    G,
                    N,
                    K,
                    C,
                    input_spatial,
                    filter_spatial,
                    output_spatial,
                    strides,
                    dilations,
                    left_pads);
            }
        }

        std::string GetTypeString() const
        {
            std::string dir_str;
            if constexpr(ConvDirectionIsForward<SIGNATURE>)
                dir_str = "Forward";
            else if constexpr(ConvDirectionIsBackwardData<SIGNATURE>)
                dir_str = "BackwardData";
            else if constexpr(ConvDirectionIsBackwardWeight<SIGNATURE>)
                dir_str = "BackwardWeight";

            return std::string("GPU_Reference_") + dir_str + "_" + std::to_string(SPATIAL_DIM) +
                   "D";
        }

        // Old CK interface: Create argument pointer
        template <typename InPtrType, typename WeiPtrType, typename OutPtrType>
        std::unique_ptr<Argument>
        MakeArgumentPointer(InPtrType input,
                            WeiPtrType weight,
                            OutPtrType output,
                            int G,
                            int N,
                            int K,
                            int C,
                            const std::vector<ck_tile::long_index_t>& input_spatial,
                            const std::vector<ck_tile::long_index_t>& filter_spatial,
                            const std::vector<ck_tile::long_index_t>& output_spatial,
                            const std::vector<ck_tile::long_index_t>& strides,
                            const std::vector<ck_tile::long_index_t>& dilations,
                            const std::vector<ck_tile::long_index_t>& left_pads) const
        {
            return std::make_unique<Argument>(input,
                                              weight,
                                              output,
                                              G,
                                              N,
                                              K,
                                              C,
                                              input_spatial,
                                              filter_spatial,
                                              output_spatial,
                                              strides,
                                              dilations,
                                              left_pads);
        }

        // Old CK interface: Create invoker pointer
        std::unique_ptr<Invoker> MakeInvokerPointer() const { return std::make_unique<Invoker>(); }
    };
};

} // namespace ck_tile::builder::factory
