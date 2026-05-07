// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_moe_gemm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_moe_gemm.hpp"
#include "ck/library/utility/check_err.hpp"

#include "ck/utility/blkgemmpipe_scheduler.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using I4  = ck::pk_i4_t;
using F16 = ck::half_t;
using F8  = ck::f8_t;
using F32 = float;

using Row    = ck::tensor_layout::gemm::RowMajor;
using Col    = ck::tensor_layout::gemm::ColumnMajor;
using Bypass = ck::tensor_layout::BypassLayoutVerification;

using A0DataType       = F8;
using B0DataType       = I4;
using EDataType        = F16;
using AccDataType      = F32;
using CShuffleDataType = F16;
using D0DataType       = F32;
using D1DataType       = F32;
using D2DataType       = F32;
using DsDataType       = ck::Tuple<D0DataType, D1DataType, D2DataType>;

using A0Layout = Row;
using B0Layout = Col;
using ELayout  = Row;
using D0Layout = Row;
using D1Layout = Col;
using D2Layout = ELayout;
using DsLayout = ck::Tuple<D0Layout, D1Layout, D2Layout>;

// for gate, a_scale, b_scale
struct MulABScale
{
    template <typename E, typename C, typename D0, typename D1>
    __host__ __device__ constexpr void
    operator()(E& e, const C& c, const D0& d0, const D1& d1) const;

    template <>
    __host__ __device__ constexpr void operator()<EDataType, EDataType, float, float>(
        EDataType& e, const EDataType& c, const float& d0, const float& d1) const
    {
        (void)d0;
        (void)d1;
#if CK_USE_PK4_LAYOUT_SHUFFLE
        e = ck::type_convert<EDataType>(c);
#else
        e = ck::type_convert<EDataType>(c);
#endif
    }
    template <>
    __host__ __device__ constexpr void operator()<EDataType, float, float, float>(
        EDataType& e, const float& c, const float& d0, const float& d1) const
    {
        (void)d0;
        (void)d1;
#if CK_USE_PK4_LAYOUT_SHUFFLE
        e = ck::type_convert<EDataType>(c);
#else
        e = ck::type_convert<EDataType>(c);
#endif
    }
};

struct MulABScaleExpertWeight
{
    template <typename E, typename C, typename D0, typename D1, typename D2>
    __host__ __device__ constexpr void
    operator()(E& e, const C& c, const D0& d0, const D1& d1, const D2& d2) const;
    // for real kernel use
    template <>
    __host__ __device__ constexpr void operator()<EDataType, float, float, float, float>(
        EDataType& e, const float& c, const float& d0, const float& d1, const float& d2) const
    {
        (void)d0;
        (void)d1;
        (void)d2;
        e = ck::type_convert<EDataType>(c);
    }
    template <>
    __host__ __device__ constexpr void operator()<EDataType, EDataType, float, float, float>(
        EDataType& e, const EDataType& c, const float& d0, const float& d1, const float& d2) const
    {
        (void)d0;
        (void)d1;
        (void)d2;
        e = ck::type_convert<EDataType>(c);
    }
    // for reference cpu
    template <>
    __host__ __device__ constexpr void operator()<float, float, float, float, float>(
        float& e, const float& c, const float& d0, const float& d1, const float& d2) const
    {
        // for reference cpu
        (void)d0;
        (void)d1;
        (void)d2;
        e = ck::type_convert<EDataType>(c);
    }
};

static constexpr bool MulRoutedWeight = true;
static constexpr ck::index_t KPack    = 32;

using CDEElementOp = MulABScaleExpertWeight; // combine MulRoutedWeight = true

// using CDEElementOp = MulABScale; // combine MulRoutedWeight = true

#if 1
void preShuffleBuffer(const I4* src, I4* dst, int N, int K, int NXdl)
{
    int NLane = NXdl;
    int KLane = 64 / NLane;

    int K0 = K / (KLane * KPack);
    // K -> K0 KLane KPack
    // N -> N0 NLane
    // N, K -> N0 K0 KLane NLane KPack
    int tempk;
    for(int n = 0; n < N; ++n)
    {
        for(int k = 0; k < K; ++k)
        {
            int n0 = n / NLane;
            int n1 = n % NLane;

            int k0 = k / (KLane * KPack);
            tempk  = k % (KLane * KPack);
            int k1 = tempk / KPack;
            int k2 = tempk % KPack;

            int outputIndex = n0 * KPack * NLane * KLane * K0 + k0 * KPack * NLane * KLane +
                              k1 * KPack * NLane + n1 * KPack + k2;

            dst[outputIndex / 2] = src[(n * K + k) / 2];
        }
    }
}
#endif

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using AElementOp = PassThrough;
using BElementOp = PassThrough;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;

static constexpr ck::index_t MPerBlock = 128;
static constexpr ck::index_t Nswizzle  = false;
static constexpr ck::index_t Act_OP    = 1; // 0: gelu_and_mul, 1: silu_and_mul

// clang-format off
using DeviceOpInstance = ck::tensor_operation::device::DeviceMoeGemm<
            Row, Col, DsLayout, ELayout, 
            A0DataType, B0DataType, DsDataType, EDataType, AccDataType, CShuffleDataType,
            AElementOp,  BElementOp, CDEElementOp,       GemmSpec,   
            256,   MPerBlock,   64,    128,
            16,   KPack,
            16,   16,
            4,    2,
            S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, 0,
            S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 32, 32, 0,
            2,    1,   S<1, 32, 1, 8>, S<4, 1, 1>,
            ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1, Act_OP, Nswizzle, true, MulRoutedWeight, true, ck::index_t, A0DataType>;
// clang-format on

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // per expert:
    // GEMM shape
    ck::index_t N               = 14336;
    ck::index_t K               = 4096;
    ck::index_t experts         = 8;
    ck::index_t sorted_tile_num = 16;
    ck::index_t valid_tile_num  = 13;
    ck::index_t sorted_size     = sorted_tile_num * MPerBlock;
    ck::index_t valid_size      = valid_tile_num * MPerBlock;
    ck::index_t tokens          = 644;
    ck::index_t topk            = 2;

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 7)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
        N               = std::stoi(argv[4]);
        K               = std::stoi(argv[5]);
        tokens          = std::stoi(argv[6]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 5: N, K, tokens\n");
        exit(1);
    }

    if(tokens * topk > valid_size)
    {
        printf("err config, tokens * topk > valid_size\n");
        exit(-1);
    }
    ck::index_t StrideA              = K;
    ck::index_t StrideB              = K;
    ck::index_t StrideE              = N;
    constexpr ck::index_t NumDTensor = DsDataType::Size();
    constexpr auto StrideDs          = std::array<ck::index_t, NumDTensor>{1, 1, 1};

    ck::index_t KBatch = 1;

    Tensor<ck::index_t> expert_ids(HostTensorDescriptor({sorted_tile_num}, {1}));
    Tensor<ck::index_t> sorted_token_ids(HostTensorDescriptor({sorted_size}, {1}));
    Tensor<ck::index_t> max_token_id(HostTensorDescriptor({1 + sorted_tile_num}));
    max_token_id.mData = {valid_size};
    int eids[]         = {0, 0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 3, 3, 3};
    for(int i = 0; i < sorted_tile_num; i++)
    {
        expert_ids.mData[i] = eids[i];
    }
    int token_per_tile = (tokens * topk + valid_tile_num - 1) / valid_tile_num;
    int tokenid        = 0;
    for(int i = 0; i < sorted_size; i++)
    {
        int tile_off = i % MPerBlock;
        if(tile_off < token_per_tile)
        {
            sorted_token_ids.mData[i] = (tokenid % tokens) | ((tokenid / tokens) << 24);
            tokenid++;
        }
        else
        {
            sorted_token_ids.mData[i] = tokens;
        }
    }

    Tensor<A0DataType> a0_t_k(HostTensorDescriptor({tokens, K}, {K, 1}));
    Tensor<B0DataType> b0_e_n_k(
        HostTensorDescriptor({experts, K, N * 2}, {N * 2 * K, 1, K}, Col{}));
    Tensor<B0DataType> b0_preshuffled(
        HostTensorDescriptor({experts, K, N * 2}, {N * 2 * K, 1, K}, Col{}));
    Tensor<D0DataType> d0_t_n(HostTensorDescriptor({tokens, N}, {StrideDs[0], 0}, Bypass{}));
    Tensor<D1DataType> d1_e_n(
        HostTensorDescriptor({experts, N * 2}, {StrideDs[1] * N * 2, StrideDs[1]}));
    Tensor<D2DataType> d2_e_n(HostTensorDescriptor({sorted_size, N}, {1, 0}, Bypass{}));
    Tensor<EDataType> e_t_n_host_result(
        HostTensorDescriptor({tokens, topk, N}, {topk * N, N, 1}, Row{}));
    Tensor<EDataType> e_t_n_device_result(
        HostTensorDescriptor({tokens, topk, N}, {topk * N, N, 1}, Row{}));

    std::cout << "a0_t_k: " << a0_t_k.mDesc << std::endl;
    std::cout << "b0_e_n_k: " << b0_e_n_k.mDesc << std::endl;
    std::cout << "d2_e_n: " << d2_e_n.mDesc << std::endl;
    std::cout << "d1_e_n: " << d1_e_n.mDesc << std::endl;
    std::cout << "d0_t_n: " << d0_t_n.mDesc << std::endl;
    std::cout << "e_t_n: " << e_t_n_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a0_t_k.GenerateTensorValue(GeneratorTensor_3<A0DataType>{0.0, 1.0});
        d0_t_n.GenerateTensorValue(GeneratorTensor_3<D0DataType>{0.0, 1.0});
        d1_e_n.GenerateTensorValue(GeneratorTensor_3<D1DataType>{0.0, 1.0});
        d2_e_n.GenerateTensorValue(GeneratorTensor_3<D2DataType>{0.0, 1.0});
        break;
    case 2:
        a0_t_k.GenerateTensorValue(GeneratorTensor_1<A0DataType>{});
        b0_e_n_k.GenerateTensorValue(GeneratorTensor_1<B0DataType>{});
        d0_t_n.GenerateTensorValue(GeneratorTensor_1<D0DataType>{});
        d1_e_n.GenerateTensorValue(GeneratorTensor_1<D1DataType>{});
        d2_e_n.GenerateTensorValue(GeneratorTensor_3<D2DataType>{});
        break;
    default:
        a0_t_k.GenerateTensorValue(GeneratorTensor_3<A0DataType>{0.0, 1.0});
        d0_t_n.GenerateTensorValue(GeneratorTensor_3<D0DataType>{0.0, 1.0});
        d1_e_n.GenerateTensorValue(GeneratorTensor_3<D1DataType>{0.0, 1.0});
        d2_e_n.GenerateTensorValue(GeneratorTensor_3<D2DataType>{0.0, 1.0});
    }
    DeviceMem sorted_token_ids_dev(sizeof(ck::index_t) *
                                   sorted_token_ids.mDesc.GetElementSpaceSize());
    DeviceMem expert_ids_dev(sizeof(ck::index_t) * expert_ids.mDesc.GetElementSpaceSize());
    DeviceMem max_token_id_dev(sizeof(ck::index_t) * max_token_id.mDesc.GetElementSpaceSize());
    DeviceMem a0_device_buf(sizeof(A0DataType) * a0_t_k.mDesc.GetElementSpaceSize());
    DeviceMem b0_device_buf(sizeof(B0DataType) * b0_e_n_k.mDesc.GetElementSpaceSize() / 2);
    DeviceMem d0_device_buf(sizeof(D0DataType) * d0_t_n.mDesc.GetElementSpaceSize());
    DeviceMem d1_device_buf(sizeof(D1DataType) * d1_e_n.mDesc.GetElementSpaceSize());
    DeviceMem d2_device_buf(sizeof(D2DataType) * d2_e_n.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) * e_t_n_device_result.mDesc.GetElementSpaceSize());

    sorted_token_ids_dev.ToDevice(sorted_token_ids.mData.data());
    expert_ids_dev.ToDevice(expert_ids.mData.data());
    max_token_id_dev.ToDevice(max_token_id.mData.data());
    a0_device_buf.ToDevice(a0_t_k.mData.data());
    d0_device_buf.ToDevice(d0_t_n.mData.data());
    d1_device_buf.ToDevice(d1_e_n.mData.data());
    d2_device_buf.ToDevice(d2_e_n.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    // do GEMM
    auto device_op = DeviceOpInstance{};

#if 1
    preShuffleBuffer(b0_e_n_k.mData.data(),
                     b0_preshuffled.mData.data(),
                     N * experts,
                     K,
                     device_op.GetPreShuffleParameters());
#else
    // weight pre-shuffle
    int KPack = 32; // int4 -> 32, fp8 -> 16, fp16 -> 8
    int NLane = device_op.GetPreShuffleParameters();
    int KLane = 64 / NLane;

    int K0 = K / (KLane * KPack);
    // K -> K0 KLane KPack
    // N -> N0 NLane
    // N, K -> N0 K0 KLane NLane KPack
    int tempk;
    for(int e = 0; e < experts; ++e)
    {
        for(int n = 0; n < N; ++n)
        {
            for(int k = 0; k < K; ++k)
            {
                int n0 = n / NLane;
                int n1 = n % NLane;

                int k0 = k / (KLane * KPack);
                tempk  = k % (KLane * KPack);
                int k1 = tempk / KPack;
                int k2 = tempk % KPack;

                int outputIndex = n0 * KPack * NLane * KLane * K0 + k0 * KPack * NLane * KLane +
                                  k1 * KPack * NLane + n1 * KPack + k2;

                b0_preshuffled(e, outputIndex % K, outputIndex / K) = b0_e_n_k(e, k, n);
            }
        }
    }
#endif

#if CK_USE_PK4_LAYOUT_SHUFFLE
    // vector pk_i4x4 permute
    for(int e = 0; e < experts; e++)
    {
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < K; j += 8)
            {
                int input[8];

                for(int k = 0; k < 4; k++)
                {
                    int i4x2         = b0_preshuffled(e, j + k * 2, i).data;
                    input[k * 2 + 0] = (i4x2 >> 4) & 0xf;
                    input[k * 2 + 1] = (i4x2 >> 0) & 0xf;
                }

                // permute 01234567->20643175
                {
                    int hi   = input[2];
                    int lo   = input[0];
                    int i4x2 = (hi << 4) | lo;

                    b0_preshuffled(e, j + 0, i) = i4x2;
                }

                {
                    int hi   = input[6];
                    int lo   = input[4];
                    int i4x2 = (hi << 4) | lo;

                    b0_preshuffled(e, j + 2, i) = i4x2;
                }

                {
                    int hi   = input[3];
                    int lo   = input[1];
                    int i4x2 = (hi << 4) | lo;

                    b0_preshuffled(e, j + 4, i) = i4x2;
                }

                {
                    int hi   = input[7];
                    int lo   = input[5];
                    int i4x2 = (hi << 4) | lo;

                    b0_preshuffled(e, j + 6, i) = i4x2;
                }
            }
        }
    }
#endif

    b0_device_buf.ToDevice(b0_preshuffled.mData.data());

    auto invoker = device_op.MakeInvoker();
    auto argument =
        device_op.MakeArgument(sorted_token_ids_dev.GetDeviceBuffer(),
                               expert_ids_dev.GetDeviceBuffer(),
                               max_token_id_dev.GetDeviceBuffer(),
                               a0_device_buf.GetDeviceBuffer(),
                               b0_device_buf.GetDeviceBuffer(),
                               std::array<const void*, NumDTensor>{d0_device_buf.GetDeviceBuffer(),
                                                                   d1_device_buf.GetDeviceBuffer(),
                                                                   d2_device_buf.GetDeviceBuffer()},
                               e_device_buf.GetDeviceBuffer(),
                               tokens,
                               topk,
                               sorted_size,
                               N,
                               K,
                               StrideA,
                               StrideB,
                               StrideDs,
                               StrideE,
                               KBatch,
                               a_element_op,
                               b_element_op,
                               cde_element_op);

    if(!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    if(!(ck::get_device_name() == "gfx942" || ck::get_device_name() == "gfx950" ||
         ck::is_gfx11_supported() || ck::is_gfx12_supported()))
    {
        std::cout << "This kernel support gfx942, gfx950, gfx11 and gfx12 only" << std::endl;
    }

    if(time_kernel)
    {
        float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

        std::size_t flop      = std::size_t(2) * tokens * topk * N * 2 * K;
        std::size_t num_btype = sizeof(A0DataType) * valid_tile_num * K +
                                sizeof(B0DataType) / 2 * K * N * 2 * experts +
                                sizeof(EDataType) * valid_tile_num * N;

        float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

        float gb_per_sec = num_btype / 1.E6 / ave_time;

        std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s" << device_op.GetTypeString() << std::endl;
    }

    if(do_verification)
    {
        invoker.Run(argument, StreamConfig{nullptr, false, 0, 0, 1});

        e_device_buf.FromDevice(e_t_n_device_result.mData.data());

        Tensor<CShuffleDataType> c_t_k_n({tokens, topk, N}, {topk * N, N, 1}, Row{});

        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceMoeGemm<A0DataType,
                                                                                   B0DataType,
                                                                                   CShuffleDataType,
                                                                                   D2DataType,
                                                                                   AccDataType,
                                                                                   PassThrough,
                                                                                   PassThrough,
                                                                                   PassThrough,
                                                                                   Act_OP,
                                                                                   MulRoutedWeight>;
        auto ref_moe_gemm           = ReferenceGemmInstance{};
        auto ref_invoker            = ref_moe_gemm.MakeInvoker();

        auto ref_argument = ref_moe_gemm.MakeArgument(sorted_token_ids,
                                                      expert_ids,
                                                      max_token_id,
                                                      MPerBlock,
                                                      a0_t_k,
                                                      d0_t_n,
                                                      b0_e_n_k,
                                                      d1_e_n,
                                                      c_t_k_n,
                                                      d2_e_n,
                                                      PassThrough{},
                                                      PassThrough{},
                                                      PassThrough{});

        ref_invoker.Run(ref_argument);
        for(int m = 0; m < valid_size; ++m)
        {

            const int fuse_t  = sorted_token_ids.mData[m];
            const int t       = fuse_t & 0xffffff;
            const int topk_id = (fuse_t & 0xff000000) >> 24;

            if(t >= tokens)
            {
                continue;
            }
            const int e = expert_ids(m / MPerBlock);
            for(int n = 0; n < N; ++n)
            {
                cde_element_op(e_t_n_host_result(t, topk_id, n),
                               c_t_k_n(t, topk_id, n),
                               d0_t_n(t, n),
                               d1_e_n(e, n),
                               d2_e_n(e, n));
            }
        }

        e_device_buf.FromDevice(e_t_n_device_result.mData.data());
        return ck::utils::check_err(
                   e_t_n_device_result, e_t_n_host_result, "Error: Incorrect results!", 1e-3, 5e-1)
                   ? 0
                   : 1;
    }

    return 0;
}
