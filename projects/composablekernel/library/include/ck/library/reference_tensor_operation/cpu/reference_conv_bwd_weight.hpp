// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/tensor_operation/gpu/device/device_base.hpp"

#include "ck/library/utility/host_tensor.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

// input descriptor in [G, N, C, Do, Ho, Wo] order
// weight descriptor in [G, K, C, Z, Y, X] order
// output descriptor in [G, N, K, Di, Hi, Wi] order
// phyiscal layout is irrelavent
template <ck::index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          ck::index_t NumAElementwiseTensor                                         = 0,
          ck::index_t NumBElementwiseTensor                                         = 0,
          ck::index_t NumDElementwiseTensor                                         = 0,
          typename ComputeTypeA                                                     = OutDataType,
          typename ComputeTypeB                                                     = InDataType,
          typename std::enable_if<NDimSpatial >= 1 && NDimSpatial <= 3, bool>::type = false>
struct ReferenceConvBwdWeight : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(
            const Tensor<InDataType>& in_n_c_hi_wi,
            Tensor<WeiDataType>& wei_k_c_y_x,
            const Tensor<OutDataType>& out_n_k_ho_wo,
            std::vector<ck::long_index_t> conv_filter_strides,
            std::vector<ck::long_index_t> conv_filter_dilations,
            std::vector<ck::long_index_t> input_left_pads,
            std::vector<ck::long_index_t> input_right_pads,
            InElementwiseOperation in_element_op,
            WeiElementwiseOperation wei_element_op,
            OutElementwiseOperation out_element_op,
            const std::array<Tensor<OutDataType>, NumAElementwiseTensor>& elementwise_a_tensors,
            const std::array<Tensor<InDataType>, NumBElementwiseTensor>& elementwise_b_tensors,
            const std::array<Tensor<WeiDataType>, NumDElementwiseTensor>& elementwise_d_tensors)
            : input_{in_n_c_hi_wi},
              weight_{wei_k_c_y_x},
              output_{out_n_k_ho_wo},
              elementwise_a_tensors_{elementwise_a_tensors},
              elementwise_b_tensors_{elementwise_b_tensors},
              elementwise_d_tensors_{elementwise_d_tensors},
              conv_strides_{conv_filter_strides},
              conv_dilations_{conv_filter_dilations},
              in_left_pads_{input_left_pads},
              in_right_pads_{input_right_pads},
              in_element_op_{in_element_op},
              wei_element_op_{wei_element_op},
              out_element_op_{out_element_op}
        {
        }

        const Tensor<InDataType>& input_;
        Tensor<WeiDataType>& weight_;
        const Tensor<OutDataType>& output_;

        const std::array<Tensor<OutDataType>, NumAElementwiseTensor>& elementwise_a_tensors_;
        const std::array<Tensor<InDataType>, NumBElementwiseTensor>& elementwise_b_tensors_;
        const std::array<Tensor<WeiDataType>, NumDElementwiseTensor>& elementwise_d_tensors_;

        std::vector<long_index_t> conv_strides_;
        std::vector<long_index_t> conv_dilations_;
        std::vector<long_index_t> in_left_pads_;
        std::vector<long_index_t> in_right_pads_;

        InElementwiseOperation in_element_op_;
        WeiElementwiseOperation wei_element_op_;
        OutElementwiseOperation out_element_op_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        using Argument = ReferenceConvBwdWeight::Argument;

        float Run(const Argument& arg)
        {
            if(!(arg.input_.GetNumOfDimension() == NDimSpatial + 3 &&
                 arg.weight_.GetNumOfDimension() == NDimSpatial + 3 &&
                 arg.output_.GetNumOfDimension() == NDimSpatial + 3))
            {
                throw std::runtime_error("wrong! inconsistent dimension");
            }

            if constexpr(NDimSpatial == 1)
            {
                auto f_kcx = [&](auto g, auto k, auto c, auto x) {
                    float v_acc = 0;

                    for(std::size_t n = 0; n < arg.output_.GetLengths()[1]; ++n)
                    {
                        for(std::size_t wo = 0; wo < arg.output_.GetLengths()[3]; ++wo)
                        {
                            auto wi = static_cast<ck::long_index_t>(wo * arg.conv_strides_[0]) +
                                      static_cast<ck::long_index_t>(x * arg.conv_dilations_[0]) -
                                      static_cast<ck::long_index_t>(arg.in_left_pads_[0]);

                            if(wi >= 0 &&
                               ck::type_convert<std::size_t>(wi) < arg.input_.GetLengths()[3])
                            {
                                ComputeTypeA v_out;
                                ComputeTypeB v_in;

                                ExecuteElementwiseOp(
                                    arg.out_element_op_,
                                    arg.elementwise_a_tensors_,
                                    Number<NumAElementwiseTensor>{},
                                    v_out,
                                    ck::type_convert<float>(arg.output_(g, n, k, wo)),
                                    g,
                                    n,
                                    k,
                                    wo);
                                ExecuteElementwiseOp(
                                    arg.in_element_op_,
                                    arg.elementwise_b_tensors_,
                                    Number<NumBElementwiseTensor>{},
                                    v_in,
                                    ck::type_convert<float>(arg.input_(g, n, c, wi)),
                                    g,
                                    n,
                                    c,
                                    wi);

                                v_acc += type_convert<float>(v_out) * type_convert<float>(v_in);
                            }
                        }
                    }

                    WeiDataType v_acc_converted = ck::type_convert<WeiDataType>(v_acc);
                    WeiDataType& v_wei          = arg.weight_(g, k, c, x);
                    ExecuteElementwiseOp(arg.wei_element_op_,
                                         arg.elementwise_d_tensors_,
                                         Number<NumDElementwiseTensor>{},
                                         v_wei,
                                         v_acc_converted,
                                         g,
                                         k,
                                         c,
                                         x);
                };

                make_ParallelTensorFunctor(f_kcx,
                                           arg.weight_.GetLengths()[0],
                                           arg.weight_.GetLengths()[1],
                                           arg.weight_.GetLengths()[2],
                                           arg.weight_.GetLengths()[3])(
                    std::thread::hardware_concurrency());

                return 0;
            }
            else if constexpr(NDimSpatial == 2)
            {
                auto f_kcyx = [&](auto g, auto k, auto c, auto y, auto x) {
                    std::size_t N = arg.output_.GetLengths()[1];

                    std::size_t Ho = arg.output_.GetLengths()[3];
                    std::size_t Wo = arg.output_.GetLengths()[4];

                    float v_acc = 0;

                    for(std::size_t n = 0; n < N; ++n)
                    {
                        for(std::size_t ho = 0; ho < Ho; ++ho)
                        {
                            auto hi = static_cast<ck::long_index_t>(ho * arg.conv_strides_[0]) +
                                      static_cast<ck::long_index_t>(y * arg.conv_dilations_[0]) -
                                      static_cast<ck::long_index_t>(arg.in_left_pads_[0]);

                            for(std::size_t wo = 0; wo < Wo; ++wo)
                            {
                                auto wi =
                                    static_cast<ck::long_index_t>(wo * arg.conv_strides_[1]) +
                                    static_cast<ck::long_index_t>(x * arg.conv_dilations_[1]) -
                                    static_cast<ck::long_index_t>(arg.in_left_pads_[1]);

                                if(hi >= 0 &&
                                   ck::type_convert<std::size_t>(hi) < arg.input_.GetLengths()[3] &&
                                   wi >= 0 &&
                                   ck::type_convert<std::size_t>(wi) < arg.input_.GetLengths()[4])
                                {
                                    ComputeTypeA v_out;
                                    ComputeTypeB v_in;

                                    ExecuteElementwiseOp(
                                        arg.out_element_op_,
                                        arg.elementwise_a_tensors_,
                                        Number<NumAElementwiseTensor>{},
                                        v_out,
                                        ck::type_convert<float>(arg.output_(g, n, k, ho, wo)),
                                        g,
                                        n,
                                        k,
                                        ho,
                                        wo);
                                    ExecuteElementwiseOp(
                                        arg.in_element_op_,
                                        arg.elementwise_b_tensors_,
                                        Number<NumBElementwiseTensor>{},
                                        v_in,
                                        ck::type_convert<float>(arg.input_(g, n, c, hi, wi)),
                                        g,
                                        n,
                                        c,
                                        hi,
                                        wi);

                                    v_acc += type_convert<float>(v_out) * type_convert<float>(v_in);
                                }
                            }
                        }
                    }

                    WeiDataType v_acc_converted = ck::type_convert<WeiDataType>(v_acc);
                    WeiDataType& v_wei          = arg.weight_(g, k, c, y, x);
                    ExecuteElementwiseOp(arg.wei_element_op_,
                                         arg.elementwise_d_tensors_,
                                         Number<NumDElementwiseTensor>{},
                                         v_wei,
                                         v_acc_converted,
                                         g,
                                         k,
                                         c,
                                         y,
                                         x);
                };

                make_ParallelTensorFunctor(f_kcyx,
                                           arg.weight_.GetLengths()[0],
                                           arg.weight_.GetLengths()[1],
                                           arg.weight_.GetLengths()[2],
                                           arg.weight_.GetLengths()[3],
                                           arg.weight_.GetLengths()[4])(
                    std::thread::hardware_concurrency());

                return 0;
            }
            else if constexpr(NDimSpatial == 3)
            {
                auto f_kczyx = [&](auto g, auto k, auto c, auto z, auto y, auto x) {
                    float v_acc = 0;

                    for(std::size_t n = 0; n < arg.output_.GetLengths()[1]; ++n)
                    {
                        for(std::size_t do_ = 0; do_ < arg.output_.GetLengths()[3]; ++do_)
                        {
                            auto di = static_cast<ck::long_index_t>(do_ * arg.conv_strides_[0]) +
                                      static_cast<ck::long_index_t>(z * arg.conv_dilations_[0]) -
                                      static_cast<ck::long_index_t>(arg.in_left_pads_[0]);
                            for(std::size_t ho = 0; ho < arg.output_.GetLengths()[4]; ++ho)
                            {
                                auto hi =
                                    static_cast<ck::long_index_t>(ho * arg.conv_strides_[1]) +
                                    static_cast<ck::long_index_t>(y * arg.conv_dilations_[1]) -
                                    static_cast<ck::long_index_t>(arg.in_left_pads_[1]);
                                for(std::size_t wo = 0; wo < arg.output_.GetLengths()[5]; ++wo)
                                {
                                    auto wi =
                                        static_cast<ck::long_index_t>(wo * arg.conv_strides_[2]) +
                                        static_cast<ck::long_index_t>(x * arg.conv_dilations_[2]) -
                                        static_cast<ck::long_index_t>(arg.in_left_pads_[2]);

                                    if(di >= 0 &&
                                       ck::type_convert<std::size_t>(di) <
                                           arg.input_.GetLengths()[3] &&
                                       hi >= 0 &&
                                       ck::type_convert<std::size_t>(hi) <
                                           arg.input_.GetLengths()[4] &&
                                       wi >= 0 &&
                                       ck::type_convert<std::size_t>(wi) <
                                           arg.input_.GetLengths()[5])
                                    {
                                        ComputeTypeA v_out;
                                        ComputeTypeB v_in;

                                        ExecuteElementwiseOp(arg.out_element_op_,
                                                             arg.elementwise_a_tensors_,
                                                             Number<NumAElementwiseTensor>{},
                                                             v_out,
                                                             ck::type_convert<float>(
                                                                 arg.output_(g, n, k, do_, ho, wo)),
                                                             g,
                                                             n,
                                                             k,
                                                             do_,
                                                             ho,
                                                             wo);
                                        ExecuteElementwiseOp(arg.in_element_op_,
                                                             arg.elementwise_b_tensors_,
                                                             Number<NumBElementwiseTensor>{},
                                                             v_in,
                                                             ck::type_convert<float>(
                                                                 arg.input_(g, n, c, di, hi, wi)),
                                                             g,
                                                             n,
                                                             c,
                                                             di,
                                                             hi,
                                                             wi);

                                        v_acc +=
                                            type_convert<float>(v_out) * type_convert<float>(v_in);
                                    }
                                }
                            }
                        }
                    }

                    WeiDataType v_acc_converted = ck::type_convert<WeiDataType>(v_acc);
                    WeiDataType& v_wei          = arg.weight_(g, k, c, z, y, x);
                    ExecuteElementwiseOp(arg.wei_element_op_,
                                         arg.elementwise_d_tensors_,
                                         Number<NumDElementwiseTensor>{},
                                         v_wei,
                                         v_acc_converted,
                                         g,
                                         k,
                                         c,
                                         z,
                                         y,
                                         x);
                };

                make_ParallelTensorFunctor(f_kczyx,
                                           arg.weight_.GetLengths()[0],
                                           arg.weight_.GetLengths()[1],
                                           arg.weight_.GetLengths()[2],
                                           arg.weight_.GetLengths()[3],
                                           arg.weight_.GetLengths()[4],
                                           arg.weight_.GetLengths()[5])(
                    std::thread::hardware_concurrency());

                return 0;
            }
            throw std::runtime_error("Conv_bwd: number of dimensions must be between 1 and 3.");
            return 1;
        }

        float Run(const device::BaseArgument* p_arg,
                  const StreamConfig& /*stream_config*/ = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg));
        }
    };

    template <typename... Args,
              typename ElementwiseOp,
              typename ElementwiseTensor,
              typename NumTensor,
              typename Y,
              typename X>
    static void ExecuteElementwiseOp(ElementwiseOp& elementwise_op,
                                     ElementwiseTensor& elementwise_tensors,
                                     NumTensor,
                                     Y& y,
                                     const X& x,
                                     Args... dims)
    {
        if constexpr(NumTensor::value == 0)
        {
            elementwise_op(y, x);
        }
        else if constexpr(NumTensor::value == 1)
        {
            elementwise_op(y, x, elementwise_tensors[0](dims...));
        }
        else if constexpr(NumTensor::value == 2)
        {
            elementwise_op(y, x, elementwise_tensors[0](dims...), elementwise_tensors[1](dims...));
        }
        else
        {
            throw std::runtime_error("ElementOp not supported in reference.");
        }
    }

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    bool IsSupportedArgument(const device::BaseArgument*) override { return true; }

    static auto MakeArgument(
        const Tensor<InDataType>& in_n_c_hi_wi,
        Tensor<WeiDataType>& wei_k_c_y_x,
        const Tensor<OutDataType>& out_n_k_ho_wo,
        std::vector<ck::long_index_t> conv_filter_strides,
        std::vector<ck::long_index_t> conv_filter_dilations,
        std::vector<ck::long_index_t> input_left_pads,
        std::vector<ck::long_index_t> input_right_pads,
        InElementwiseOperation in_element_op,
        WeiElementwiseOperation wei_element_op,
        OutElementwiseOperation out_element_op,
        const std::array<Tensor<OutDataType>, NumAElementwiseTensor>& elementwise_a_tensors = {},
        const std::array<Tensor<InDataType>, NumBElementwiseTensor>& elementwise_b_tensors  = {},
        const std::array<Tensor<WeiDataType>, NumDElementwiseTensor>& elementwise_d_tensors = {})
    {
        return Argument{in_n_c_hi_wi,
                        wei_k_c_y_x,
                        out_n_k_ho_wo,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        in_element_op,
                        wei_element_op,
                        out_element_op,
                        elementwise_a_tensors,
                        elementwise_b_tensors,
                        elementwise_d_tensors};
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
        str << "ReferenceConvBwdWeight"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
