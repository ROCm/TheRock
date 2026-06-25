// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_moe_gemm_blockscale.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_moe_gemm2_blockscale.hpp"
#include "ck/library/utility/check_err.hpp"

#include "ck/utility/blkgemmpipe_scheduler.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using F8   = ck::f8_t;
using F32  = float;
using I64  = int64_t;

using Row    = ck::tensor_layout::gemm::RowMajor;
using Col    = ck::tensor_layout::gemm::ColumnMajor;
using Bypass = ck::tensor_layout::BypassLayoutVerification;

using A0DataType = F8;
using A1DataType = F32;
using B0DataType = F8;
using B1DataType = F32;
using EDataType  = F16;
// using EDataType        = BF16;
using AccDataType      = F32;
using CShuffleDataType = EDataType;
using D2DataType       = F32;
using DsDataType       = ck::Tuple<D2DataType>;

using A0Layout = Row;
using B0Layout = Col;
using ELayout  = Row;
using D0Layout = Row;
using D1Layout = Col;
using D2Layout = ELayout;
// using DsLayoutGate = ck::Tuple<D0Layout, D1Layout>;
using DsLayout = ck::Tuple<D2Layout>;

// d0: ascale, d1: bscale, d2:expert weight
struct MulABScaleExpertWeight
{
    template <typename E, typename C, typename D2>
    __host__ __device__ constexpr void operator()(E& e, const C& c, const D2& d2) const;
    // for real kernel use

    template <>
    __host__ __device__ constexpr void
    operator()<EDataType, EDataType, float>(EDataType& e, const EDataType& c, const float& d2) const
    {
        // for real kernel use
        (void)d2;
        e = ck::type_convert<EDataType>(c);
    }
    template <>
    __host__ __device__ constexpr void
    operator()<EDataType, float, float>(EDataType& e, const float& c, const float& d2) const
    {
        // for real kernel use
        (void)d2;
        e = ck::type_convert<EDataType>(c);
    }
    template <>
    __host__ __device__ constexpr void
    operator()<float, float, float>(float& e, const float& c, const float& d2) const
    {
        // for reference cpu
        e = ck::type_convert<EDataType>(c * d2);
    }
};

void preShuffleBuffer(const B0DataType* src, B0DataType* dst, int N, int K, int NXdl)
{
    int KPack = 16 / sizeof(B0DataType);
    int NLane = NXdl;
    int KLane = 64 / NLane;

    int K0 = K / (KLane * KPack);
    // K -> K0 KLane KPack
    // N -> N0 NLane
    // N, K -> N0 K0 KLane NLane KPack
    int tempk;
    for(I64 n = 0; n < N; ++n)
    {
        for(I64 k = 0; k < K; ++k)
        {
            I64 n0 = n / NLane;
            I64 n1 = n % NLane;

            I64 k0 = k / (KLane * KPack);
            tempk  = k % (KLane * KPack);
            I64 k1 = tempk / KPack;
            I64 k2 = tempk % KPack;

            I64 outputIndex = n0 * KPack * NLane * KLane * K0 + k0 * KPack * NLane * KLane +
                              k1 * KPack * NLane + n1 * KPack + k2;

            dst[outputIndex] = src[n * static_cast<I64>(K) + k];
        }
    }
}
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = MulABScaleExpertWeight;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;

static constexpr ck::index_t Scale_Block_M = 1;
static constexpr ck::index_t Scale_Block_N = 128;
static constexpr ck::index_t Scale_Block_K = 128;
static constexpr bool MulRoutedWeight      = true;

#if 0
static constexpr ck::index_t MPerBlock = 32;
static constexpr ck::index_t BLOCKSIZE = 256;
static constexpr ck::index_t MXDLPerWave = 2;
static constexpr ck::index_t NXDLPerWave = 2;
static constexpr ck::index_t NPerBlock   = 128;
static constexpr ck::index_t MNPerXDL    = 16;
static constexpr ck::index_t KPerBlock   = 256 / sizeof(A0DataType);

static constexpr ck::index_t CShuffleNLane = 16;
static constexpr ck::index_t CShuffleMLane = BLOCKSIZE / CShuffleNLane;
static constexpr ck::index_t AK1           = 16 / sizeof(A0DataType);
static constexpr ck::index_t BK1           = 16 / sizeof(B0DataType);
static constexpr ck::index_t EVec          = 2;
static constexpr ck::index_t D0Vec         = 1;
static constexpr ck::index_t D1Vec         = 1;
static constexpr ck::index_t D2Vec         = 1;

// clang-format off

using DeviceOpInstance = ck::tensor_operation::device::DeviceMoeGemmBlockScale<
               Row, Col, DsLayout, ELayout,
               A0DataType, A1DataType, B0DataType, B1DataType, DsDataType, EDataType, AccDataType, CShuffleDataType,
               AElementOp,  BElementOp, CDEElementOp,       GemmSpec,   
               BLOCKSIZE, Scale_Block_M, Scale_Block_N, Scale_Block_K,
               MPerBlock,   NPerBlock,    KPerBlock,
               AK1,   BK1,
               MNPerXDL,   MNPerXDL,
               MXDLPerWave,  NXDLPerWave,
               S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, AK1, AK1, 0,
               S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, AK1, AK1, 0,
               2,        2,         S<1, CShuffleMLane, 1, CShuffleNLane>, S<EVec, D0Vec, D1Vec, D2Vec>,
               ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1, 0, false, false, false, MulRoutedWeight, int32_t, A0DataType>;

#else
static constexpr ck::index_t MPerBlock = 64; using DeviceOpInstance = ck::tensor_operation::device::DeviceMoeGemmBlockScale<
               Row, Col, DsLayout, ELayout,
               A0DataType, A1DataType, B0DataType, B1DataType, DsDataType, EDataType, AccDataType, CShuffleDataType,
               AElementOp,  BElementOp, CDEElementOp,   GemmSpec,   
               256,  Scale_Block_M, Scale_Block_N, Scale_Block_K,
               MPerBlock,   128,    128,
               16,   16,
               16,   16,
               4,    2,
               S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, 0,
               S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, 0,
               2,    2,   S<1, 32, 1, 8>, S<2, 1, 1, 1>,
               ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, 0, false, false, false, MulRoutedWeight, int32_t, A0DataType>;
#endif
// clang-format on

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // tokens = 1
    // topk = 1
    // experts = 8
    // per expert:

    constexpr ck::index_t valid_tile_num =
        26; // 13 for 128; 52 for 32; 4096 for ds  // > token * topk / MPerBlock
    constexpr ck::index_t sorted_tile_num = valid_tile_num + 3;
    ck::index_t sorted_size               = sorted_tile_num * MPerBlock;
    ck::index_t valid_size                = valid_tile_num * MPerBlock;
#if 1
    // GEMM shape
    ck::index_t N       = 6144;
    ck::index_t K       = 4096;
    ck::index_t experts = 8;
    ck::index_t tokens  = 832;
    ck::index_t topk    = 2;
#else
    // deepseek
    ck::index_t N       = 2048;
    ck::index_t K       = 7160;
    ck::index_t experts = 256;
    ck::index_t tokens  = 1;
    ck::index_t topk    = 8;
#endif

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        // use default case
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
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
        printf("arg4 to 6: N, K, tokens\n");
        exit(0);
    }

    ck::index_t StrideA              = K;
    ck::index_t StrideB              = K;
    ck::index_t StrideE              = N;
    constexpr ck::index_t NumDTensor = DsDataType::Size();
    constexpr auto StrideDs          = std::array<ck::index_t, NumDTensor>{0};
    ck::index_t Scale_Stride_AM      = (K + Scale_Block_K - 1) / Scale_Block_K;
    ck::index_t Scale_Stride_BN      = (K + Scale_Block_K - 1) / Scale_Block_K;
    ck::index_t Scale_Stride_B       = (N + Scale_Block_N - 1) / Scale_Block_N;

    ck::index_t KBatch = 1;

    Tensor<ck::index_t> expert_ids(HostTensorDescriptor({sorted_tile_num}, {1}));
    Tensor<ck::index_t> sorted_token_ids(HostTensorDescriptor({sorted_size}, {1}));
    Tensor<ck::index_t> max_token_id(HostTensorDescriptor({1}));

    max_token_id.mData = {valid_size, 0, 1, 2, 3, 4, 5, 6, 7, 8};
    // int eids[]         = {0, 1, 3, 3, 3};
    //  int eids[]         = {0, 1, 2, 3, 4, 5, 6, 7}; //, 3, 3, 3}; // {2, 1, 1, 2, 2, 2, 1, 2}
    // int eids[] = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 3, 3, 3};
    // int eids[]         = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    //                     2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    //                     3, 3, 3, 3, 3, 3, 3, 3, 4, 4,
    //                     5, 5, 5, 5, 6, 6, 6, 6, 7, 7,
    //                     7, 7,
    //                     3, 3, 3};
    for(int i = 0; i < sorted_tile_num; i++)
    {
        expert_ids.mData[i] = i / ck::math::integer_divide_ceil(valid_tile_num, experts);
    }
    if(tokens * topk > valid_size)
    {
        printf("err config, tokens * topk > valid_size\n");
        exit(-1);
    }
    int token_per_tile = tokens * topk / valid_tile_num;
    int tokenid        = 0;

    for(int i = 0; i < sorted_size; i++)
    {
        int tile_off = i % MPerBlock;
        if(tile_off < token_per_tile && tokenid < tokens * topk)
        {
            sorted_token_ids.mData[i] = (tokenid % tokens) | ((tokenid / tokens) << 24);
            tokenid++;
        }
        else
        {
            sorted_token_ids.mData[i] = tokens;
        }
    }

    Tensor<A0DataType> a0_t_k_k(HostTensorDescriptor({tokens, topk, K}, {topk * K, K, 1}, Row{}));
    Tensor<A1DataType> a1_t_k_k(
        HostTensorDescriptor({tokens, topk, (K + Scale_Block_K - 1) / Scale_Block_K},
                             {(topk * Scale_Stride_AM), Scale_Stride_AM, 1},
                             Row{}));

    Tensor<B0DataType> b0_e_n_k(HostTensorDescriptor({experts, K, N}, {N * K, 1, K}, Col{}));
    Tensor<B1DataType> b1_e_n_k(HostTensorDescriptor(
        {experts, (K + Scale_Block_K - 1) / Scale_Block_K, (N + Scale_Block_N - 1) / Scale_Block_N},
        {(Scale_Stride_B * Scale_Stride_BN), 1, Scale_Stride_BN},
        Col{}));

    Tensor<B0DataType> b0_preshuffled(HostTensorDescriptor({experts, K, N}, {N * K, 1, K}, Col{}));
    Tensor<D2DataType> d2_e_n(HostTensorDescriptor({sorted_size, N}, {1, 0}, Bypass{}));
    Tensor<EDataType> e_t_n_host_result(HostTensorDescriptor({tokens, N}, {N, 1}));
    Tensor<EDataType> e_t_n_device_result(HostTensorDescriptor({tokens, N}, {N, 1}));
    e_t_n_device_result.SetZero();
    std::cout << "a0_t_k_k: " << a0_t_k_k.mDesc << std::endl;
    std::cout << "a1_t_k_k: " << a1_t_k_k.mDesc << std::endl;
    std::cout << "b0_e_n_k: " << b0_e_n_k.mDesc << std::endl;
    std::cout << "b1_e_n_k: " << b1_e_n_k.mDesc << std::endl;
    std::cout << "d2_e_n: " << d2_e_n.mDesc << std::endl;
    std::cout << "e_t_n: " << e_t_n_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a0_t_k_k.GenerateTensorValue(GeneratorTensor_3<A0DataType>{-1.0, 1.0});
        a1_t_k_k.GenerateTensorValue(GeneratorTensor_3<A1DataType>{0, 1.0});
        b0_e_n_k.GenerateTensorValue(GeneratorTensor_3<B0DataType>{-1.0, 1.0});
        b1_e_n_k.GenerateTensorValue(GeneratorTensor_3<B1DataType>{0, 1.0});
        d2_e_n.GenerateTensorValue(GeneratorTensor_3<D2DataType>{0, 1.0});
        break;
    case 2:
        a0_t_k_k.GenerateTensorValue(GeneratorTensor_1<A0DataType>{});
        a1_t_k_k.GenerateTensorValue(GeneratorTensor_1<A1DataType>{});
        b0_e_n_k.GenerateTensorValue(GeneratorTensor_1<B0DataType>{});
        b1_e_n_k.GenerateTensorValue(GeneratorTensor_1<B1DataType>{});
        d2_e_n.GenerateTensorValue(GeneratorTensor_1<D2DataType>{});
        break;
    case 3:
        a0_t_k_k.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-2, 2});
        a1_t_k_k.GenerateTensorValue(GeneratorTensor_1<A1DataType>{});
        b0_e_n_k.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        b1_e_n_k.GenerateTensorValue(GeneratorTensor_1<B1DataType>{});
        d2_e_n.GenerateTensorValue(GeneratorTensor_1<D2DataType>{});
        break;
    case 4:
        a0_t_k_k.GenerateTensorValue(GeneratorTensor_1<A0DataType>{});
        a1_t_k_k.GenerateTensorValue(GeneratorTensor_3<A1DataType>{0, 1.0});
        b0_e_n_k.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        b1_e_n_k.GenerateTensorValue(GeneratorTensor_3<B1DataType>{0, 1.0});
        d2_e_n.GenerateTensorValue(GeneratorTensor_3<D2DataType>{0, 1.0});
        break;
    case 5:
        a0_t_k_k.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-2, 2});
        a1_t_k_k.GenerateTensorValue(GeneratorTensor_3<A1DataType>{0, 1.0});
        b0_e_n_k.GenerateTensorValue(GeneratorTensor_1<B0DataType>{});
        b1_e_n_k.GenerateTensorValue(GeneratorTensor_3<B1DataType>{0, 1.0});
        d2_e_n.GenerateTensorValue(GeneratorTensor_3<D2DataType>{0, 1.0});
        break;
    case 6:
        a0_t_k_k.GenerateTensorValue(GeneratorTensor_3<A0DataType>{1.0, 1.0});
        a1_t_k_k.GenerateTensorValue(GeneratorTensor_3<A1DataType>{1.0, 1.0});
        b0_e_n_k.GenerateTensorValue(GeneratorTensor_3<B0DataType>{1.0, 1.0});
        b1_e_n_k.GenerateTensorValue(GeneratorTensor_3<B1DataType>{1.0, 1.0});
        d2_e_n.GenerateTensorValue(GeneratorTensor_3<D2DataType>{1.0, 1.0});
        for(auto i = 0; i < N * K; i++)
        {
            b0_e_n_k.mData[i]         = ck::type_convert<B0DataType>(static_cast<float>(0.1));
            b0_e_n_k.mData[i + N * K] = ck::type_convert<B0DataType>(static_cast<float>(0.2));
        }
        break;
    default:
        a0_t_k_k.GenerateTensorValue(GeneratorTensor_3<A0DataType>{0.0, 1.0});
        a1_t_k_k.GenerateTensorValue(GeneratorTensor_3<A1DataType>{0, 1.0});
        b0_e_n_k.GenerateTensorValue(GeneratorTensor_3<B0DataType>{-0.5, 0.5});
        b1_e_n_k.GenerateTensorValue(GeneratorTensor_3<B1DataType>{0, 1.0});
        d2_e_n.GenerateTensorValue(GeneratorTensor_3<D2DataType>{0.0, 1.0});
    }

    DeviceMem sorted_token_ids_dev(sizeof(ck::index_t) *
                                   sorted_token_ids.mDesc.GetElementSpaceSize());
    DeviceMem expert_ids_dev(sizeof(ck::index_t) * expert_ids.mDesc.GetElementSpaceSize());
    DeviceMem max_token_id_dev(sizeof(ck::index_t) * max_token_id.mDesc.GetElementSpaceSize());
    DeviceMem a0_device_buf(sizeof(A0DataType) * a0_t_k_k.mDesc.GetElementSpaceSize());
    DeviceMem a1_device_buf(sizeof(A1DataType) * a1_t_k_k.mDesc.GetElementSpaceSize());
    DeviceMem b0_device_buf(sizeof(B0DataType) * b0_e_n_k.mDesc.GetElementSpaceSize());
    DeviceMem b1_device_buf(sizeof(B1DataType) * b1_e_n_k.mDesc.GetElementSpaceSize());
    DeviceMem d2_device_buf(sizeof(D2DataType) * d2_e_n.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) * e_t_n_device_result.mDesc.GetElementSpaceSize());

    sorted_token_ids_dev.ToDevice(sorted_token_ids.mData.data());
    expert_ids_dev.ToDevice(expert_ids.mData.data());
    max_token_id_dev.ToDevice(max_token_id.mData.data());
    a0_device_buf.ToDevice(a0_t_k_k.mData.data());
    a1_device_buf.ToDevice(a1_t_k_k.mData.data());
    b1_device_buf.ToDevice(b1_e_n_k.mData.data());
    d2_device_buf.ToDevice(d2_e_n.mData.data());
    e_device_buf.ToDevice(e_t_n_device_result.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    // do GEMM
    auto device_op = DeviceOpInstance{};

    int NPerXdl = device_op.GetPreShuffleParameters();

    preShuffleBuffer(b0_e_n_k.mData.data(), b0_preshuffled.mData.data(), N * experts, K, NPerXdl);
    b0_device_buf.ToDevice(b0_preshuffled.mData.data());

    auto invoker = device_op.MakeInvoker();
    auto argument =
        device_op.MakeArgument(sorted_token_ids_dev.GetDeviceBuffer(),
                               expert_ids_dev.GetDeviceBuffer(),
                               max_token_id_dev.GetDeviceBuffer(),
                               a0_device_buf.GetDeviceBuffer(),
                               b0_device_buf.GetDeviceBuffer(),
                               std::array<const void*, NumDTensor>{d2_device_buf.GetDeviceBuffer()},
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
                               a1_device_buf.GetDeviceBuffer(),
                               b1_device_buf.GetDeviceBuffer(),
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

    if(time_kernel)
    {
        // not result correct here because output buf not setzero
        float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

        std::size_t flop      = std::size_t(2) * tokens * topk * N * K;
        std::size_t num_btype = sizeof(A0DataType) * tokens * K * topk +
                                sizeof(B0DataType) * K * N * experts +
                                sizeof(EDataType) * tokens * N;

        float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

        float gb_per_sec = num_btype / 1.E6 / ave_time;

        std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s.\n"
                  << device_op.GetTypeString() << std::endl;
    }

    if(do_verification)
    {
        // gemm2 use atomic, so need to reinit outputs
        e_device_buf.ToDevice(e_t_n_device_result.mData.data());
        invoker.Run(argument, StreamConfig{nullptr, false, 0, 0, 1});

        Tensor<float> a_t_k_k({tokens, topk, K});
        Tensor<float> b_e_n_k({experts, K, N});
        Tensor<float> c_t_n({tokens, N});

        for(int t = 0; t < tokens; ++t)
        {
            for(int tk = 0; tk < topk; ++tk)
            {
                for(int k = 0; k < K; ++k)
                {
                    a_t_k_k(t, tk, k) = ck::type_convert<float>(a0_t_k_k(t, tk, k)) *
                                        a1_t_k_k(t, tk, k / Scale_Block_K);
                }
            }
        }

        for(int e = 0; e < experts; ++e)
        {
            for(int k = 0; k < K; ++k)
            {
                for(int n = 0; n < N; ++n)
                {
                    b_e_n_k(e, k, n) = ck::type_convert<float>(b0_e_n_k(e, k, n)) *
                                       b1_e_n_k(e, k / Scale_Block_K, n / Scale_Block_N);
                }
            }
        }

        using ReferenceGemmInstance =
            ck::tensor_operation::host::ReferenceMoeGemm2BlockScale<float,
                                                                    float,
                                                                    float,
                                                                    D2DataType,
                                                                    AccDataType,
                                                                    PassThrough,
                                                                    PassThrough,
                                                                    CDEElementOp,
                                                                    MulRoutedWeight>;
        auto ref_moe_gemm = ReferenceGemmInstance{};
        auto ref_invoker  = ref_moe_gemm.MakeInvoker();
        auto ref_argument = ref_moe_gemm.MakeArgument(sorted_token_ids,
                                                      expert_ids,
                                                      max_token_id,
                                                      MPerBlock,
                                                      a_t_k_k,
                                                      b_e_n_k,
                                                      d2_e_n,
                                                      c_t_n,
                                                      PassThrough{},
                                                      PassThrough{},
                                                      cde_element_op);

        ref_invoker.Run(ref_argument);
        for(int t = 0; t < tokens; ++t)
        {

            for(int n = 0; n < N; ++n)
            {
                e_t_n_host_result(t, n) = ck::type_convert<EDataType>(c_t_n(t, n));
            }
        }

        e_device_buf.FromDevice(e_t_n_device_result.mData.data());

        auto status =
            ck::utils::check_err(
                e_t_n_device_result, e_t_n_host_result, "Error: Incorrect results!", 1e-3, 5e-2)
                ? 0
                : 1;
        if(status == 0)
        {
            printf("Validation Pass.\n");
        }
        return status;
    }

    return 0;
}
