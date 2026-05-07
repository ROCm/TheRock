// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/utility/amd_smfmac.hpp"
#include "ck/library/utility/fill.hpp"

namespace ck {
namespace smfmac_op_util {

template <typename src_vec1, typename src_vec2, typename acc_vec>
__device__ void
builtin_smfmac_naive_selector(const src_vec1&, const src_vec2&, const int32_t&, acc_vec&)
{
}

template <>
__device__ void
builtin_smfmac_naive_selector<half4_t,
                              half8_t,
                              StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 4, true>>(
    const half4_t& reg_a,
    const half8_t& reg_b,
    const int32_t& reg_idx,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 4, true>& reg_c)
{
    intrin_smfmac_f32_16x16x32f16<16, 16>::Run(
        reg_a, reg_b, reg_idx, reg_c.GetVectorTypeReference(Number<0>{}));
}

template <>
__device__ void
builtin_smfmac_naive_selector<bhalf4_t,
                              bhalf8_t,
                              StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 4, true>>(
    const bhalf4_t& reg_a,
    const bhalf8_t& reg_b,
    const int32_t& reg_idx,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 4, true>& reg_c)
{
    intrin_smfmac_f32_16x16x32bf16<16, 16>::Run(
        reg_a, reg_b, reg_idx, reg_c.GetVectorTypeReference(Number<0>{}));
}

template <>
__device__ void builtin_smfmac_naive_selector<
    half4_t,
    half8_t,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 16, true>>(
    const half4_t& reg_a,
    const half8_t& reg_b,
    const int32_t& reg_idx,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 16, true>& reg_c)
{
    intrin_smfmac_f32_32x32x16f16<32, 32>::Run(
        reg_a, reg_b, reg_idx, reg_c.GetVectorTypeReference(Number<0>{}));
}

template <>
__device__ void builtin_smfmac_naive_selector<
    bhalf4_t,
    bhalf8_t,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 16, true>>(
    const bhalf4_t& reg_a,
    const bhalf8_t& reg_b,
    const int32_t& reg_idx,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 16, true>& reg_c)
{
    intrin_smfmac_f32_32x32x16bf16<32, 32>::Run(
        reg_a, reg_b, reg_idx, reg_c.GetVectorTypeReference(Number<0>{}));
}

// Smfmac instructions are using 4:2 structural sparsity, that means that in every contignuous
// subgroup of 4 elements, atleast 2 must be equal to zero and the position of non-zero elements is
// stored in idx register to allow selection of corresponding B matrix elements for multiplication.
// Currently smfmac instructions support only A matrix as sparse
template <typename src1_t,
          index_t src1_vec_size,
          typename src2_t,
          index_t src2_vec_size,
          typename acc_t,
          index_t acc_vec_size,
          typename dst_t,
          int32_t M,
          int32_t N,
          int32_t K>
__global__ void matmul(const src1_t* a, const src2_t* b, dst_t* c)
{
    __shared__ src1_t a_shared[M * K];
    __shared__ src2_t b_shared[K * N];
    const int lane = threadIdx.x;
    // smfmac's A part is storing only non-zero elements in 2VGPRs
    // smfmac's B part is storing all elements in 4VGPRs
    using src1_vec      = typename vector_type<src1_t, src1_vec_size>::type;
    using src1_full_vec = typename vector_type<src1_t, src1_vec_size * 2>::type;
    using src2_vec      = typename vector_type<src2_t, src2_vec_size>::type;
    src1_vec a_frag     = {};
    src2_vec b_frag     = {};

    src1_full_vec a_temp = {};
    src2_vec b_temp      = {};
    // initialize c fragment to 0
    using acc_vec = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, acc_t, 1, acc_vec_size, true>;
    acc_vec c_thread_buf_;

    for(int i = 0; i < 8; ++i)
    {
        a_temp[i] = a[(lane % M) * K + (lane / M) * 8 + i]; // M K
    }

    for(int i = 0; i < 8; ++i)
    {
        b_temp[i] = b[(8 * (lane / N) + i) * N + (lane % N)]; // K N
    }

    __syncthreads();

    for(int i = 0; i < 8; ++i)
    {
        a_shared[(lane % M) * K + (lane / M) * 8 + i] = a_temp[i];
    }
    for(int i = 0; i < 8; ++i)
    {
        b_shared[(8 * (lane / N) + i) * N + (lane % N)] = b_temp[i];
    }

    __syncthreads();

    // Idx must be a 32-bit register and it is storing 4 2-bit indexes of A's non zero elements.
    // It starts with last two elements of every 4 elements subgroup set as non-zero
    int32_t idx = 0b11101110;
    // Bit masks are for zeroing 0-3rd position of idx
    static constexpr int32_t bit_clear_masks[4] = {0b11, 0b1100, 0b110000, 0b11000000};

    src1_t curr_val;
    int32_t a_pos = 0;
    for(int j = 0; j < 2; ++j)
    {
        a_pos = j * 2;
        for(int i = 0; i < 4; ++i)
        {
            curr_val = a_shared[(lane % M) * K + (lane / M) * 8 + 4 * j + i];
            if(curr_val != 0.0f)
            {
                idx &= ~bit_clear_masks[a_pos];
                idx |= (i % 4) << 2 * a_pos;
                a_frag[a_pos] = curr_val;
                a_pos++;
            }
        }
    }

    for(int i = 0; i < 8; ++i)
    {
        b_frag[i] = b_shared[(8 * (lane / N) + i) * N + (lane % N)];
    }

    builtin_smfmac_naive_selector<src1_vec, src2_vec, acc_vec>(a_frag, b_frag, idx, c_thread_buf_);
    __syncthreads();

    // store results from unpacked c_thread_buf_ output
    if constexpr(K == 32)
    {
        static_for<0, acc_vec_size, 1>{}([&](auto i) {
            c[(4 * (lane / 16) + i) * N + lane % 16] =
                ck::type_convert<dst_t>(c_thread_buf_[Number<i>{}]);
        });
    }
    else
    {
        static_for<0, acc_vec_size, 1>{}([&](auto i) {
            c[((8 * (i / 4)) % 32 + 4 * (lane / 32) + i % 4) * N + lane % 32] =
                ck::type_convert<dst_t>(c_thread_buf_[Number<i>{}]);
        });
    }
}

struct GemmParams
{
    GemmParams() : M(16), N(16), K(32), StrideA(32), StrideB(16), StrideC(16), alpha(1), beta(0) {}

    ck::index_t M;
    ck::index_t N;
    ck::index_t K;

    ck::index_t StrideA;
    ck::index_t StrideB;
    ck::index_t StrideC;

    float alpha;
    float beta;
};

template <typename GemmInstance,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
void RunHostGEMM(const Tensor<ADataType>& A,
                 const Tensor<BDataType>& B,
                 Tensor<CDataType>& C,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
{
    auto ref_gemm     = GemmInstance{};
    auto ref_invoker  = ref_gemm.MakeInvoker();
    auto ref_argument = ref_gemm.MakeArgument(A, B, C, a_element_op, b_element_op, c_element_op);

    ref_invoker.Run(ref_argument);
}

template <typename KernelType, typename ADataType, typename BDataType, typename CDataType>
bool RunDeviceGEMM(KernelType kernel,
                   const Tensor<ADataType>& A,
                   const Tensor<BDataType>& B,
                   Tensor<CDataType>& C)
{
    DeviceMem a_m_k_device_buf(sizeof(ADataType) * A.mDesc.GetElementSpaceSize());
    DeviceMem b_n_k_device_buf(sizeof(BDataType) * B.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * C.mDesc.GetElementSpaceSize());

    a_m_k_device_buf.ToDevice(A.mData.data());
    b_n_k_device_buf.ToDevice(B.mData.data());
    kernel<<<1, 64>>>(static_cast<const ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                      static_cast<const BDataType*>(b_n_k_device_buf.GetDeviceBuffer()),
                      static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()));
    c_m_n_device_buf.FromDevice(C.mData.data());

    return true;
}

template <typename DeviceSmfmac,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename GPUAccDataType,
          typename CPUAccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          index_t CAccNum,
          index_t M,
          index_t N,
          index_t K>
struct TestSmfmac
{
    auto PrepareGemmTensor(const ck::smfmac_op_util::GemmParams& params)
    {
        auto f_host_tensor_descriptor =
            [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
                if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
                {
                    return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                                std::vector<std::size_t>({stride, 1}));
                }
                else
                {
                    return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                                std::vector<std::size_t>({1, stride}));
                }
            };

        Tensor<ADataType> a_m_k(
            f_host_tensor_descriptor(params.M, params.K, params.StrideA, ALayout{}));
        Tensor<BDataType> b_n_k(
            f_host_tensor_descriptor(params.K, params.N, params.StrideB, BLayout{}));
        Tensor<CDataType> c_m_n_host_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));
        Tensor<CDataType> c_m_n_device_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));

        auto f_generate_tensor_value = [](auto& tensor, auto type) {
            using dataType = decltype(type);
            tensor.GenerateTensorValue(GeneratorTensor_2<dataType>{-5, 5});
        };

        f_generate_tensor_value(a_m_k, ADataType{});
        f_generate_tensor_value(b_n_k, BDataType{});
        ck::utils::TransformIntoStructuralSparsity<ADataType>{}(a_m_k);

        return std::make_tuple(a_m_k, b_n_k, c_m_n_host_result, c_m_n_device_result);
    }

    auto operator()(const DeviceSmfmac& smfmac_kernel)
    {
        std::cout << "ALayout = " << ALayout{}.name << ", BLayout = " << BLayout{}.name
                  << ", CLayout = " << CLayout{}.name << std::endl;

        // Arrange
        ck::smfmac_op_util::GemmParams params;
        params.M       = M;
        params.N       = N;
        params.K       = K;
        params.StrideA = K; // M K
        params.StrideB = N; // K N
        params.StrideC = N; // M N

        auto host_tensors = PrepareGemmTensor(params);

        const Tensor<ADataType>& a  = std::get<0>(host_tensors);
        const Tensor<BDataType>& b  = std::get<1>(host_tensors);
        Tensor<CDataType>& c_host   = std::get<2>(host_tensors);
        Tensor<CDataType>& c_device = std::get<3>(host_tensors);

        auto a_element_op = AElementwiseOperation{};
        auto b_element_op = BElementwiseOperation{};
        auto c_element_op = CElementwiseOperation{};

        using ReferenceGemmInstance =
            ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                      BDataType,
                                                      CDataType,
                                                      CPUAccDataType,
                                                      AElementwiseOperation,
                                                      BElementwiseOperation,
                                                      CElementwiseOperation>;
        ck::smfmac_op_util::RunHostGEMM<ReferenceGemmInstance>(
            a, b, c_host, a_element_op, b_element_op, c_element_op);

        // Act
        bool is_supported = ck::smfmac_op_util::RunDeviceGEMM(smfmac_kernel, a, b, c_device);

        if(is_supported)
        {
            // Assert
            bool res = false;
            if(std::is_same<CDataType, float>::value)
            {
                res = ck::utils::check_err(c_device.mData, c_host.mData);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else
            {
                std::cout << "UNSUPPORTED CDataType" << std::endl;
            }

            return res;
        }
        else
        {
            return true;
        }
    }
};

} // namespace smfmac_op_util
} // namespace ck
