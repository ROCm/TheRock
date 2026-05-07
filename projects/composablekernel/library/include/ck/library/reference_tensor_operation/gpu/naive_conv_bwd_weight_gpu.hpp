// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/type_convert.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/reference_tensor_operation/gpu/naive_conv_utils.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {
namespace ref {

// Optimized backward weight convolution kernel working with packed (contiguous) tensors
// Assumes row-major packing: input[G][N][C][spatial], output_grad[G][N][K][spatial],
// weight_grad[G][K][C][filter]
// Computes gradient with respect to weights
template <index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InElementOp,
          typename WeiElementOp,
          typename OutElementOp>
__global__ void naive_conv_bwd_weight_packed(const InDataType* __restrict__ p_in,
                                             WeiDataType* __restrict__ p_wei_grad,
                                             const OutDataType* __restrict__ p_out_grad,
                                             index_t G,
                                             index_t N,
                                             index_t K,
                                             index_t C,
                                             index_t Di,
                                             index_t Hi,
                                             index_t Wi,
                                             index_t Z,
                                             index_t Y,
                                             index_t X,
                                             index_t Do,
                                             index_t Ho,
                                             index_t Wo,
                                             index_t stride_z,
                                             index_t stride_y,
                                             index_t stride_x,
                                             index_t dilation_z,
                                             index_t dilation_y,
                                             index_t dilation_x,
                                             index_t pad_z,
                                             index_t pad_y,
                                             index_t pad_x,
                                             InElementOp in_op,
                                             WeiElementOp wei_op,
                                             OutElementOp out_op)
{
    const long_index_t tid         = blockIdx.x * blockDim.x + threadIdx.x;
    const long_index_t num_threads = blockDim.x * gridDim.x;

    InDataType in_val   = InDataType{0};
    WeiDataType wei_val = WeiDataType{0};
    OutDataType out_val = OutDataType{0};

    if constexpr(NDimSpatial == 1)
    {
        const long_index_t num_wei      = G * K * C * X;
        const long_index_t in_stride_g  = N * C * Wi;
        const long_index_t in_stride_n  = C * Wi;
        const long_index_t in_stride_c  = Wi;
        const long_index_t out_stride_g = N * K * Wo;
        const long_index_t out_stride_n = K * Wo;
        const long_index_t out_stride_k = Wo;
        const long_index_t wei_stride_g = K * C * X;
        const long_index_t wei_stride_k = C * X;
        const long_index_t wei_stride_c = X;

        for(long_index_t idx = tid; idx < num_wei; idx += num_threads)
        {
            index_t remaining = idx;
            const index_t x   = remaining % X;
            remaining /= X;
            const index_t c = remaining % C;
            remaining /= C;
            const index_t k = remaining % K;
            const index_t g = remaining / K;

            float acc                   = 0.0f;
            const InDataType* in_g      = p_in + g * in_stride_g;
            const OutDataType* out_grad = p_out_grad + g * out_stride_g;

            // Loop over batch and output positions
            for(index_t n = 0; n < N; ++n)
            {
                const InDataType* in_gn     = in_g + n * in_stride_n + c * in_stride_c;
                const OutDataType* out_gn_k = out_grad + n * out_stride_n + k * out_stride_k;

                for(index_t wo = 0; wo < Wo; ++wo)
                {
                    long_index_t wi = wo * stride_x + x * dilation_x - pad_x;
                    if(wi >= 0 && wi < Wi)
                    {
                        in_op(in_val, in_gn[wi]);
                        out_op(out_val, out_gn_k[wo]);
                        acc += type_convert<float>(out_val) * type_convert<float>(in_val);
                    }
                }
            }

            WeiDataType result = type_convert<WeiDataType>(acc);
            wei_op(wei_val, result);
            p_wei_grad[g * wei_stride_g + k * wei_stride_k + c * wei_stride_c + x] = wei_val;
        }
    }
    else if constexpr(NDimSpatial == 2)
    {
        const long_index_t num_wei      = G * K * C * Y * X;
        const long_index_t in_stride_g  = N * C * Hi * Wi;
        const long_index_t in_stride_n  = C * Hi * Wi;
        const long_index_t in_stride_c  = Hi * Wi;
        const long_index_t in_stride_h  = Wi;
        const long_index_t out_stride_g = N * K * Ho * Wo;
        const long_index_t out_stride_n = K * Ho * Wo;
        const long_index_t out_stride_k = Ho * Wo;
        const long_index_t out_stride_h = Wo;
        const long_index_t wei_stride_g = K * C * Y * X;
        const long_index_t wei_stride_k = C * Y * X;
        const long_index_t wei_stride_c = Y * X;
        const long_index_t wei_stride_y = X;

        for(long_index_t idx = tid; idx < num_wei; idx += num_threads)
        {
            index_t remaining = idx;
            const index_t x   = remaining % X;
            remaining /= X;
            const index_t y = remaining % Y;
            remaining /= Y;
            const index_t c = remaining % C;
            remaining /= C;
            const index_t k = remaining % K;
            const index_t g = remaining / K;

            float acc                   = 0.0f;
            const InDataType* in_g      = p_in + g * in_stride_g;
            const OutDataType* out_grad = p_out_grad + g * out_stride_g;

            // Loop over batch and output positions
            for(index_t n = 0; n < N; ++n)
            {
                const InDataType* in_gnc    = in_g + n * in_stride_n + c * in_stride_c;
                const OutDataType* out_gn_k = out_grad + n * out_stride_n + k * out_stride_k;

                for(index_t ho = 0; ho < Ho; ++ho)
                {
                    long_index_t hi = ho * stride_y + y * dilation_y - pad_y;
                    if(hi >= 0 && hi < Hi)
                    {
                        const InDataType* in_gnch    = in_gnc + hi * in_stride_h;
                        const OutDataType* out_gn_kh = out_gn_k + ho * out_stride_h;

                        for(index_t wo = 0; wo < Wo; ++wo)
                        {
                            long_index_t wi = wo * stride_x + x * dilation_x - pad_x;
                            if(wi >= 0 && wi < Wi)
                            {
                                in_op(in_val, in_gnch[wi]);
                                out_op(out_val, out_gn_kh[wo]);
                                acc += type_convert<float>(out_val) * type_convert<float>(in_val);
                            }
                        }
                    }
                }
            }

            WeiDataType result = type_convert<WeiDataType>(acc);
            wei_op(wei_val, result);
            p_wei_grad[g * wei_stride_g + k * wei_stride_k + c * wei_stride_c + y * wei_stride_y +
                       x] = wei_val;
        }
    }
    else if constexpr(NDimSpatial == 3)
    {
        const long_index_t num_wei      = G * K * C * Z * Y * X;
        const long_index_t in_stride_g  = N * C * Di * Hi * Wi;
        const long_index_t in_stride_n  = C * Di * Hi * Wi;
        const long_index_t in_stride_c  = Di * Hi * Wi;
        const long_index_t in_stride_d  = Hi * Wi;
        const long_index_t in_stride_h  = Wi;
        const long_index_t out_stride_g = N * K * Do * Ho * Wo;
        const long_index_t out_stride_n = K * Do * Ho * Wo;
        const long_index_t out_stride_k = Do * Ho * Wo;
        const long_index_t out_stride_d = Ho * Wo;
        const long_index_t out_stride_h = Wo;
        const long_index_t wei_stride_g = K * C * Z * Y * X;
        const long_index_t wei_stride_k = C * Z * Y * X;
        const long_index_t wei_stride_c = Z * Y * X;
        const long_index_t wei_stride_z = Y * X;
        const long_index_t wei_stride_y = X;

        for(long_index_t idx = tid; idx < num_wei; idx += num_threads)
        {
            index_t remaining = idx;
            const index_t x   = remaining % X;
            remaining /= X;
            const index_t y = remaining % Y;
            remaining /= Y;
            const index_t z = remaining % Z;
            remaining /= Z;
            const index_t c = remaining % C;
            remaining /= C;
            const index_t k = remaining % K;
            const index_t g = remaining / K;

            float acc                   = 0.0f;
            const InDataType* in_g      = p_in + g * in_stride_g;
            const OutDataType* out_grad = p_out_grad + g * out_stride_g;

            // Loop over batch and output positions
            for(index_t n = 0; n < N; ++n)
            {
                const InDataType* in_gnc    = in_g + n * in_stride_n + c * in_stride_c;
                const OutDataType* out_gn_k = out_grad + n * out_stride_n + k * out_stride_k;

                for(index_t do_idx = 0; do_idx < Do; ++do_idx)
                {
                    long_index_t di = do_idx * stride_z + z * dilation_z - pad_z;
                    if(di >= 0 && di < Di)
                    {
                        const InDataType* in_gncd    = in_gnc + di * in_stride_d;
                        const OutDataType* out_gn_kd = out_gn_k + do_idx * out_stride_d;

                        for(index_t ho = 0; ho < Ho; ++ho)
                        {
                            long_index_t hi = ho * stride_y + y * dilation_y - pad_y;
                            if(hi >= 0 && hi < Hi)
                            {
                                const InDataType* in_gncdh    = in_gncd + hi * in_stride_h;
                                const OutDataType* out_gn_kdh = out_gn_kd + ho * out_stride_h;

                                for(index_t wo = 0; wo < Wo; ++wo)
                                {
                                    long_index_t wi = wo * stride_x + x * dilation_x - pad_x;
                                    if(wi >= 0 && wi < Wi)
                                    {
                                        in_op(in_val, in_gncdh[wi]);
                                        out_op(out_val, out_gn_kdh[wo]);
                                        acc += type_convert<float>(out_val) *
                                               type_convert<float>(in_val);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            WeiDataType result = type_convert<WeiDataType>(acc);
            wei_op(wei_val, result);
            p_wei_grad[g * wei_stride_g + k * wei_stride_k + c * wei_stride_c + z * wei_stride_z +
                       y * wei_stride_y + x] = wei_val;
        }
    }
}

// GPU reference backward weight convolution - takes ConvParam directly
template <typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename TIn,
          typename TWei,
          typename TOut,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation>
void naive_conv_bwd_weight(const TIn* p_in,
                           TWei* p_wei_grad,
                           const TOut* p_out,
                           const ck::utils::conv::ConvParam& conv_param,
                           InElementwiseOperation in_element_op   = InElementwiseOperation{},
                           WeiElementwiseOperation wei_element_op = WeiElementwiseOperation{},
                           OutElementwiseOperation out_element_op = OutElementwiseOperation{},
                           hipStream_t stream                     = nullptr)
{
    const auto ndim = conv_param.num_dim_spatial_;

    const index_t G = conv_param.G_;
    const index_t N = conv_param.N_;
    const index_t C = conv_param.C_;
    const index_t K = conv_param.K_;

    std::vector<index_t> in_lengths  = {G, N, C};
    std::vector<index_t> wei_lengths = {G, K, C};
    std::vector<index_t> out_lengths = {G, N, K};

    for(index_t i = 0; i < ndim; ++i)
    {
        in_lengths.push_back(static_cast<index_t>(conv_param.input_spatial_lengths_[i]));
        wei_lengths.push_back(static_cast<index_t>(conv_param.filter_spatial_lengths_[i]));
        out_lengths.push_back(static_cast<index_t>(conv_param.output_spatial_lengths_[i]));
    }

    // Calculate total elements for buffer allocation
    long_index_t in_total = 1, wei_total = 1, out_total = 1;
    for(auto l : in_lengths)
        in_total *= l;
    for(auto l : wei_lengths)
        wei_total *= l;
    for(auto l : out_lengths)
        out_total *= l;

    // Allocate packed buffers
    SimpleDeviceMem in_packed_buf(in_total * sizeof(TIn));
    SimpleDeviceMem wei_grad_packed_buf(wei_total * sizeof(TWei));
    SimpleDeviceMem out_grad_packed_buf(out_total * sizeof(TOut));

    TIn* p_in_packed        = static_cast<TIn*>(in_packed_buf.GetDeviceBuffer());
    TWei* p_wei_grad_packed = static_cast<TWei*>(wei_grad_packed_buf.GetDeviceBuffer());
    TOut* p_out_grad_packed = static_cast<TOut*>(out_grad_packed_buf.GetDeviceBuffer());

    // Compute strides and allocate device arrays for pack/unpack
    std::vector<index_t> in_strides  = compute_conv_tensor_strides<InLayout>(in_lengths, ndim);
    std::vector<index_t> wei_strides = compute_conv_tensor_strides<WeiLayout>(wei_lengths, ndim);
    std::vector<index_t> out_strides = compute_conv_tensor_strides<OutLayout>(out_lengths, ndim);

    const size_t dim_count = in_lengths.size();
    SimpleDeviceMem in_lengths_buf(dim_count * sizeof(index_t));
    SimpleDeviceMem in_strides_buf(dim_count * sizeof(index_t));
    SimpleDeviceMem wei_lengths_buf(dim_count * sizeof(index_t));
    SimpleDeviceMem wei_strides_buf(dim_count * sizeof(index_t));
    SimpleDeviceMem out_lengths_buf(dim_count * sizeof(index_t));
    SimpleDeviceMem out_strides_buf(dim_count * sizeof(index_t));

    index_t* d_in_lengths  = static_cast<index_t*>(in_lengths_buf.GetDeviceBuffer());
    index_t* d_in_strides  = static_cast<index_t*>(in_strides_buf.GetDeviceBuffer());
    index_t* d_wei_lengths = static_cast<index_t*>(wei_lengths_buf.GetDeviceBuffer());
    index_t* d_wei_strides = static_cast<index_t*>(wei_strides_buf.GetDeviceBuffer());
    index_t* d_out_lengths = static_cast<index_t*>(out_lengths_buf.GetDeviceBuffer());
    index_t* d_out_strides = static_cast<index_t*>(out_strides_buf.GetDeviceBuffer());

    HIP_CHECK_ERROR(hipMemcpy(
        d_in_lengths, in_lengths.data(), dim_count * sizeof(index_t), hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(
        d_in_strides, in_strides.data(), dim_count * sizeof(index_t), hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(
        d_wei_lengths, wei_lengths.data(), dim_count * sizeof(index_t), hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(
        d_wei_strides, wei_strides.data(), dim_count * sizeof(index_t), hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(
        d_out_lengths, out_lengths.data(), dim_count * sizeof(index_t), hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(
        d_out_strides, out_strides.data(), dim_count * sizeof(index_t), hipMemcpyHostToDevice));

    // Pack input and output_grad tensors to contiguous layout (inputs to bwd weight)
    constexpr int block_size = 256;
    strided_copy_kernel<TIn, false>
        <<<(in_total + block_size - 1) / block_size, block_size, 0, stream>>>(
            p_in, p_in_packed, d_in_lengths, d_in_strides, dim_count, in_total);
    strided_copy_kernel<TOut, false>
        <<<(out_total + block_size - 1) / block_size, block_size, 0, stream>>>(
            p_out, p_out_grad_packed, d_out_lengths, d_out_strides, dim_count, out_total);

    // Build conv parameter vectors for kernel invocation
    std::vector<index_t> conv_strides(ndim);
    std::vector<index_t> conv_dilations(ndim);
    std::vector<index_t> input_pads(ndim);
    for(index_t i = 0; i < ndim; ++i)
    {
        conv_strides[i]   = static_cast<index_t>(conv_param.conv_filter_strides_[i]);
        conv_dilations[i] = static_cast<index_t>(conv_param.conv_filter_dilations_[i]);
        input_pads[i]     = static_cast<index_t>(conv_param.input_left_pads_[i]);
    }

    // Run backward weight convolution kernel on packed data
    const int wei_grid = (wei_total + block_size - 1) / block_size;

    if(ndim == 1)
    {
        naive_conv_bwd_weight_packed<1,
                                     TIn,
                                     TWei,
                                     TOut,
                                     InElementwiseOperation,
                                     WeiElementwiseOperation,
                                     OutElementwiseOperation>
            <<<wei_grid, block_size, 0, stream>>>(p_in_packed,
                                                  p_wei_grad_packed,
                                                  p_out_grad_packed,
                                                  G,
                                                  N,
                                                  K,
                                                  C,
                                                  1,
                                                  1,
                                                  in_lengths[3],
                                                  1,
                                                  1,
                                                  wei_lengths[3],
                                                  1,
                                                  1,
                                                  out_lengths[3],
                                                  1,
                                                  1,
                                                  conv_strides[0],
                                                  1,
                                                  1,
                                                  conv_dilations[0],
                                                  0,
                                                  0,
                                                  input_pads[0],
                                                  in_element_op,
                                                  wei_element_op,
                                                  out_element_op);
    }
    else if(ndim == 2)
    {
        naive_conv_bwd_weight_packed<2,
                                     TIn,
                                     TWei,
                                     TOut,
                                     InElementwiseOperation,
                                     WeiElementwiseOperation,
                                     OutElementwiseOperation>
            <<<wei_grid, block_size, 0, stream>>>(p_in_packed,
                                                  p_wei_grad_packed,
                                                  p_out_grad_packed,
                                                  G,
                                                  N,
                                                  K,
                                                  C,
                                                  1,
                                                  in_lengths[3],
                                                  in_lengths[4],
                                                  1,
                                                  wei_lengths[3],
                                                  wei_lengths[4],
                                                  1,
                                                  out_lengths[3],
                                                  out_lengths[4],
                                                  1,
                                                  conv_strides[0],
                                                  conv_strides[1],
                                                  1,
                                                  conv_dilations[0],
                                                  conv_dilations[1],
                                                  0,
                                                  input_pads[0],
                                                  input_pads[1],
                                                  in_element_op,
                                                  wei_element_op,
                                                  out_element_op);
    }
    else // 3D
    {
        naive_conv_bwd_weight_packed<3,
                                     TIn,
                                     TWei,
                                     TOut,
                                     InElementwiseOperation,
                                     WeiElementwiseOperation,
                                     OutElementwiseOperation>
            <<<wei_grid, block_size, 0, stream>>>(p_in_packed,
                                                  p_wei_grad_packed,
                                                  p_out_grad_packed,
                                                  G,
                                                  N,
                                                  K,
                                                  C,
                                                  in_lengths[3],
                                                  in_lengths[4],
                                                  in_lengths[5],
                                                  wei_lengths[3],
                                                  wei_lengths[4],
                                                  wei_lengths[5],
                                                  out_lengths[3],
                                                  out_lengths[4],
                                                  out_lengths[5],
                                                  conv_strides[0],
                                                  conv_strides[1],
                                                  conv_strides[2],
                                                  conv_dilations[0],
                                                  conv_dilations[1],
                                                  conv_dilations[2],
                                                  input_pads[0],
                                                  input_pads[1],
                                                  input_pads[2],
                                                  in_element_op,
                                                  wei_element_op,
                                                  out_element_op);
    }

    // Unpack weight gradient
    strided_copy_kernel<TWei, true><<<wei_grid, block_size, 0, stream>>>(
        p_wei_grad_packed, p_wei_grad, d_wei_lengths, d_wei_strides, dim_count, wei_total);

    HIP_CHECK_ERROR(hipGetLastError());

    // Memory automatically freed by SimpleDeviceMem destructors
}

} // namespace ref
} // namespace ck
