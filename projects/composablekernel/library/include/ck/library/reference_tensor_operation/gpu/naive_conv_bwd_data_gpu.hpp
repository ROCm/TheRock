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

// Optimized backward data convolution kernel working with packed (contiguous) tensors
// Computes gradients w.r.t. input from output gradients and weights
// Assumes row-major packing: input[G][N][C][spatial], weight[G][K][C][filter],
// output[G][N][K][spatial]
template <index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InElementOp,
          typename WeiElementOp,
          typename OutElementOp>
__global__ void naive_conv_bwd_data_packed(InDataType* __restrict__ p_in,
                                           const WeiDataType* __restrict__ p_wei,
                                           const OutDataType* __restrict__ p_out,
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
        const long_index_t num_in       = G * N * C * Wi;
        const long_index_t out_stride_g = N * K * Wo;
        const long_index_t out_stride_n = K * Wo;
        const long_index_t out_stride_k = Wo;
        const long_index_t wei_stride_g = K * C * X;
        const long_index_t wei_stride_k = C * X;
        const long_index_t wei_stride_c = X;
        const long_index_t in_stride_g  = N * C * Wi;
        const long_index_t in_stride_n  = C * Wi;
        const long_index_t in_stride_c  = Wi;

        for(long_index_t idx = tid; idx < num_in; idx += num_threads)
        {
            index_t remaining = idx;
            const index_t wi  = remaining % Wi;
            remaining /= Wi;
            const index_t c = remaining % C;
            remaining /= C;
            const index_t n = remaining % N;
            const index_t g = remaining / N;

            float acc                 = 0.0f;
            const OutDataType* out_gn = p_out + g * out_stride_g + n * out_stride_n;
            const WeiDataType* wei_g  = p_wei + g * wei_stride_g;

            for(index_t x = 0; x < X; ++x)
            {
                long_index_t w_tmp = wi + pad_x - x * dilation_x;
                if(w_tmp % stride_x == 0)
                {
                    long_index_t wo = w_tmp / stride_x;
                    if(wo >= 0 && wo < Wo)
                    {
                        const OutDataType* out_gnk = out_gn;
                        const WeiDataType* wei_gkc = wei_g + c * wei_stride_c;

                        for(index_t k = 0; k < K; ++k)
                        {
                            out_op(out_val, out_gnk[k * out_stride_k + wo]);
                            wei_op(wei_val, wei_gkc[k * wei_stride_k + x]);
                            acc += type_convert<float>(out_val) * type_convert<float>(wei_val);
                        }
                    }
                }
            }

            InDataType result = type_convert<InDataType>(acc);
            in_op(in_val, result);
            p_in[g * in_stride_g + n * in_stride_n + c * in_stride_c + wi] = in_val;
        }
    }
    else if constexpr(NDimSpatial == 2)
    {
        const long_index_t num_in       = G * N * C * Hi * Wi;
        const long_index_t out_stride_g = N * K * Ho * Wo;
        const long_index_t out_stride_n = K * Ho * Wo;
        const long_index_t out_stride_k = Ho * Wo;
        const long_index_t out_stride_h = Wo;
        const long_index_t wei_stride_g = K * C * Y * X;
        const long_index_t wei_stride_k = C * Y * X;
        const long_index_t wei_stride_c = Y * X;
        const long_index_t wei_stride_y = X;
        const long_index_t in_stride_g  = N * C * Hi * Wi;
        const long_index_t in_stride_n  = C * Hi * Wi;
        const long_index_t in_stride_c  = Hi * Wi;
        const long_index_t in_stride_h  = Wi;

        for(long_index_t idx = tid; idx < num_in; idx += num_threads)
        {
            index_t remaining = idx;
            const index_t wi  = remaining % Wi;
            remaining /= Wi;
            const index_t hi = remaining % Hi;
            remaining /= Hi;
            const index_t c = remaining % C;
            remaining /= C;
            const index_t n = remaining % N;
            const index_t g = remaining / N;

            float acc                 = 0.0f;
            const OutDataType* out_gn = p_out + g * out_stride_g + n * out_stride_n;
            const WeiDataType* wei_g  = p_wei + g * wei_stride_g;

            for(index_t y = 0; y < Y; ++y)
            {
                long_index_t h_tmp = hi + pad_y - y * dilation_y;
                if(h_tmp % stride_y == 0)
                {
                    long_index_t ho = h_tmp / stride_y;
                    if(ho >= 0 && ho < Ho)
                    {
                        const OutDataType* out_gnkh = out_gn + ho * out_stride_h;
                        const WeiDataType* wei_gkcy = wei_g + c * wei_stride_c + y * wei_stride_y;

                        for(index_t x = 0; x < X; ++x)
                        {
                            long_index_t w_tmp = wi + pad_x - x * dilation_x;
                            if(w_tmp % stride_x == 0)
                            {
                                long_index_t wo = w_tmp / stride_x;
                                if(wo >= 0 && wo < Wo)
                                {
                                    for(index_t k = 0; k < K; ++k)
                                    {
                                        out_op(out_val, out_gnkh[k * out_stride_k + wo]);
                                        wei_op(wei_val, wei_gkcy[k * wei_stride_k + x]);
                                        acc += type_convert<float>(out_val) *
                                               type_convert<float>(wei_val);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            InDataType result = type_convert<InDataType>(acc);
            in_op(in_val, result);
            p_in[g * in_stride_g + n * in_stride_n + c * in_stride_c + hi * in_stride_h + wi] =
                in_val;
        }
    }
    else if constexpr(NDimSpatial == 3)
    {
        const long_index_t num_in       = G * N * C * Di * Hi * Wi;
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
        const long_index_t in_stride_g  = N * C * Di * Hi * Wi;
        const long_index_t in_stride_n  = C * Di * Hi * Wi;
        const long_index_t in_stride_c  = Di * Hi * Wi;
        const long_index_t in_stride_d  = Hi * Wi;
        const long_index_t in_stride_h  = Wi;

        for(long_index_t idx = tid; idx < num_in; idx += num_threads)
        {
            index_t remaining = idx;
            const index_t wi  = remaining % Wi;
            remaining /= Wi;
            const index_t hi = remaining % Hi;
            remaining /= Hi;
            const index_t di = remaining % Di;
            remaining /= Di;
            const index_t c = remaining % C;
            remaining /= C;
            const index_t n = remaining % N;
            const index_t g = remaining / N;

            float acc                 = 0.0f;
            const OutDataType* out_gn = p_out + g * out_stride_g + n * out_stride_n;
            const WeiDataType* wei_g  = p_wei + g * wei_stride_g;

            for(index_t z = 0; z < Z; ++z)
            {
                long_index_t d_tmp = di + pad_z - z * dilation_z;
                if(d_tmp % stride_z == 0)
                {
                    long_index_t do_idx = d_tmp / stride_z;
                    if(do_idx >= 0 && do_idx < Do)
                    {
                        const OutDataType* out_gnkd = out_gn + do_idx * out_stride_d;
                        const WeiDataType* wei_gkcz = wei_g + c * wei_stride_c + z * wei_stride_z;

                        for(index_t y = 0; y < Y; ++y)
                        {
                            long_index_t h_tmp = hi + pad_y - y * dilation_y;
                            if(h_tmp % stride_y == 0)
                            {
                                long_index_t ho = h_tmp / stride_y;
                                if(ho >= 0 && ho < Ho)
                                {
                                    const OutDataType* out_gnkdh = out_gnkd + ho * out_stride_h;
                                    const WeiDataType* wei_gkczy = wei_gkcz + y * wei_stride_y;

                                    for(index_t x = 0; x < X; ++x)
                                    {
                                        long_index_t w_tmp = wi + pad_x - x * dilation_x;
                                        if(w_tmp % stride_x == 0)
                                        {
                                            long_index_t wo = w_tmp / stride_x;
                                            if(wo >= 0 && wo < Wo)
                                            {
                                                for(index_t k = 0; k < K; ++k)
                                                {
                                                    out_op(out_val,
                                                           out_gnkdh[k * out_stride_k + wo]);
                                                    wei_op(wei_val,
                                                           wei_gkczy[k * wei_stride_k + x]);
                                                    acc += type_convert<float>(out_val) *
                                                           type_convert<float>(wei_val);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            InDataType result = type_convert<InDataType>(acc);
            in_op(in_val, result);
            p_in[g * in_stride_g + n * in_stride_n + c * in_stride_c + di * in_stride_d +
                 hi * in_stride_h + wi] = in_val;
        }
    }
}

// GPU reference backward data convolution - takes ConvParam directly
template <typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename TIn,
          typename TWei,
          typename TOut,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation>
void naive_conv_bwd_data(TIn* p_in,
                         const TWei* p_wei,
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
    SimpleDeviceMem wei_packed_buf(wei_total * sizeof(TWei));
    SimpleDeviceMem out_packed_buf(out_total * sizeof(TOut));

    TIn* p_in_packed   = static_cast<TIn*>(in_packed_buf.GetDeviceBuffer());
    TWei* p_wei_packed = static_cast<TWei*>(wei_packed_buf.GetDeviceBuffer());
    TOut* p_out_packed = static_cast<TOut*>(out_packed_buf.GetDeviceBuffer());

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

    // Pack output and weight tensors to contiguous layout (inputs to bwd data)
    constexpr int block_size = 256;
    strided_copy_kernel<TOut, false>
        <<<(out_total + block_size - 1) / block_size, block_size, 0, stream>>>(
            p_out, p_out_packed, d_out_lengths, d_out_strides, dim_count, out_total);
    strided_copy_kernel<TWei, false>
        <<<(wei_total + block_size - 1) / block_size, block_size, 0, stream>>>(
            p_wei, p_wei_packed, d_wei_lengths, d_wei_strides, dim_count, wei_total);

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

    // Run backward data convolution kernel on packed data
    const int in_grid = (in_total + block_size - 1) / block_size;

    if(ndim == 1)
    {
        naive_conv_bwd_data_packed<1,
                                   TIn,
                                   TWei,
                                   TOut,
                                   InElementwiseOperation,
                                   WeiElementwiseOperation,
                                   OutElementwiseOperation>
            <<<in_grid, block_size, 0, stream>>>(p_in_packed,
                                                 p_wei_packed,
                                                 p_out_packed,
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
        naive_conv_bwd_data_packed<2,
                                   TIn,
                                   TWei,
                                   TOut,
                                   InElementwiseOperation,
                                   WeiElementwiseOperation,
                                   OutElementwiseOperation>
            <<<in_grid, block_size, 0, stream>>>(p_in_packed,
                                                 p_wei_packed,
                                                 p_out_packed,
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
        naive_conv_bwd_data_packed<3,
                                   TIn,
                                   TWei,
                                   TOut,
                                   InElementwiseOperation,
                                   WeiElementwiseOperation,
                                   OutElementwiseOperation>
            <<<in_grid, block_size, 0, stream>>>(p_in_packed,
                                                 p_wei_packed,
                                                 p_out_packed,
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

    // Unpack result back to strided layout
    strided_copy_kernel<TIn, true><<<in_grid, block_size, 0, stream>>>(
        p_in_packed, p_in, d_in_lengths, d_in_strides, dim_count, in_total);

    HIP_CHECK_ERROR(hipGetLastError());

    // Memory automatically freed by SimpleDeviceMem destructors
}

} // namespace ref
} // namespace ck
