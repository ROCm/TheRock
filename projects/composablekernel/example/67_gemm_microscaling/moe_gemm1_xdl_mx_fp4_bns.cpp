// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_moe_mx_gemm_bns.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_moe_mx_gemm1.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/utility/blkgemmpipe_scheduler.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F4              = ck::f4x2_pk_t;
using F16             = ck::half_t;
using BF16            = ck::bhalf_t;
using F32             = float;
using XDataType       = ck::e8m0_bexp_t;
using XPackedDataType = int32_t; // 4 packed e8m0_bexp_t

using Row    = ck::tensor_layout::gemm::RowMajor;
using Col    = ck::tensor_layout::gemm::ColumnMajor;
using Bypass = ck::tensor_layout::BypassLayoutVerification;

using A0DataType       = F4;
using A1DataType       = XPackedDataType;
using B0DataType       = F4;
using B1DataType       = XPackedDataType;
using EDataType        = F16;
using AccDataType      = F32;
using CShuffleDataType = F32;
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

// d0: ascale, d1: bscale, d2:expert weight
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

using CDEElementOp = MulABScaleExpertWeight;

// A, B Scale preshuffle
template <bool KLast>
void preShuffleScaleBuffer(ck::e8m0_bexp_t* src, ck::e8m0_bexp_t* dst, int MN, int K)
{
    int MNXdlPack = 2;
    int KXdlPack  = 2;

    int XdlMNThread = 16;
    int XdlKThread  = 64 / XdlMNThread;

    int K0 = K / KXdlPack / XdlKThread; // KRepeat

    // The 4 16x128 building blocks will be packed into 1 32x256 for F4
    // The 8 16x16x128 mfma will be packed into 1 32x32x256 for F4

    // unfold the MN32xK(256/32) scale buffer
    //    4            16             2           2
    // To XdlKThread-> XdlMNThread -> KXdlPack -> MNXdlPack
    // Then, MNRepeat->KRepeat

    for(int n = 0; n < MN; ++n)
    {
        for(int k = 0; k < K; ++k)
        {
            int n0    = n / (XdlMNThread * MNXdlPack); // i MNRepeat
            int tempn = n % (XdlMNThread * MNXdlPack);
            int n1    = tempn % XdlMNThread; // i XdlMNThread
            int n2    = tempn / XdlMNThread; // i MNXdlPack

            int k0    = k / (XdlKThread * KXdlPack); // i KRepeat
            int tempk = k % (XdlKThread * KXdlPack);
            int k1    = tempk % XdlKThread; // i XdlKThread
            int k2    = tempk / XdlKThread; // i KXdlPack

            int outputIndex = n0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread * K0 +
                              k0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread +
                              k1 * MNXdlPack * KXdlPack * XdlMNThread + n1 * MNXdlPack * KXdlPack +
                              k2 * MNXdlPack + n2;
            // src[n * K + k] = ck::type_convert<ck::e8m0_bexp_t>(static_cast<float>(powf(2.0f, n2 +
            // k2 * MNXdlPack)));
            if constexpr(KLast)
                dst[outputIndex] = src[n * K + k];
            else
                dst[outputIndex] = src[k * MN + n];
        }
    }
}

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = MulABScaleExpertWeight;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;

constexpr ck::index_t DataPackedSize   = 2;                    // Packed representation of data
constexpr ck::index_t ScaleBlockSize   = 32;                   // scaling block size
constexpr ck::index_t KPerBlock        = 256 / DataPackedSize; // 256 f4 = 128 fp4x2
static constexpr ck::index_t Nswizzle  = false;
static constexpr ck::index_t ActOP     = 0; // 0: gelu_and_mul, 1: silu_and_mul
static constexpr ck::index_t MPerBlock = 128;
static constexpr ck::index_t NPerBlock = 64;
static constexpr ck::index_t BlockSize = 256;
static constexpr bool MulRoutedWeight  = true;

// clang-format off
using DeviceOpInstance                     = ck::tensor_operation::device::DeviceMoeGemmMXBNS<      
    A0Layout,    B0Layout,    DsLayout,    ELayout, 
    A0DataType,  A1DataType,  B0DataType,  B1DataType,  DsDataType, EDataType, AccDataType, CShuffleDataType,
    AElementOp,  BElementOp, CDEElementOp, GemmSpec,   
    ScaleBlockSize, BlockSize,   
    MPerBlock,      NPerBlock,    KPerBlock,
    16,   16, 
    16,   16,
    4,     2,
    S<8, 32, 1>, S<1, 0, 2>,     S<1, 0, 2>,    2, 16, 16, 0,
    S<8, 32, 1>, S<1, 0, 2>,     S<1, 0, 2>,    2, 16, 16, 0,
    2,    2,     S<1, 32, 1, 8>, S<8, 1, 1, 1>,
    ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, 
    ActOP, Nswizzle, true, MulRoutedWeight, ck::index_t, A0DataType>;
// clang-format on

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // per expert:
    // GEMM shape
    constexpr ck::index_t sorted_tile_num = 13;
    constexpr ck::index_t valid_tile_num  = sorted_tile_num;
    ck::index_t sorted_size               = sorted_tile_num * MPerBlock;
    ck::index_t valid_size                = valid_tile_num * MPerBlock;

    ck::index_t N       = 4096;
    ck::index_t K       = 6144;
    ck::index_t experts = 8;
    ck::index_t tokens  = 832;
    ck::index_t topk    = 2;

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

    if(K % ScaleBlockSize != 0)
    {
        throw std::runtime_error("wrong! K must be multiple of ScaleBlockSize.");
    };

    ck::index_t StrideA              = K;
    ck::index_t StrideB              = K;
    ck::index_t StrideE              = N;
    ck::index_t Scale_Stride_AM      = (K + ScaleBlockSize - 1) / ScaleBlockSize;
    ck::index_t Scale_Stride_BN      = (K + ScaleBlockSize - 1) / ScaleBlockSize;
    constexpr ck::index_t NumDTensor = DsDataType::Size();
    constexpr auto StrideDs          = std::array<ck::index_t, NumDTensor>{0, 0, 0};

    ck::index_t KBatch = 1;

    Tensor<ck::index_t> expert_ids(HostTensorDescriptor({sorted_tile_num}, {1}));
    Tensor<ck::index_t> sorted_token_ids(HostTensorDescriptor({sorted_size}, {1}));
    Tensor<ck::index_t> max_token_id(HostTensorDescriptor({sorted_tile_num + 1}));
    max_token_id.mData[0] = valid_size;

    if(tokens * topk > valid_size)
    {
        printf("err config, tokens * topk > valid_size\n");
        exit(-1);
    }

    for(int i = 0; i < sorted_tile_num; i++)
    {
        expert_ids.mData[i] = i / ck::math::integer_divide_ceil(valid_tile_num, experts);
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
    Tensor<XDataType> a1_t_k(HostTensorDescriptor(
        {tokens, (K + ScaleBlockSize - 1) / ScaleBlockSize}, {Scale_Stride_AM, 1}));
    Tensor<B0DataType> b0_e_n_k(
        HostTensorDescriptor({experts, K, N * 2}, {N * 2 * K, 1, K}, Col{}));
    Tensor<XDataType> b1_e_n_k(
        HostTensorDescriptor({experts, (K + ScaleBlockSize - 1) / ScaleBlockSize, N * 2},
                             {(N * 2 * Scale_Stride_BN), 1, Scale_Stride_BN},
                             Col{}));

    // A, B Scale preshuffle
    Tensor<XDataType> a_scale_sorted(HostTensorDescriptor(
        {sorted_size, (K + ScaleBlockSize - 1) / ScaleBlockSize}, {Scale_Stride_AM, 1}));
    Tensor<XDataType> a_scale_preshuffled(HostTensorDescriptor(
        {sorted_size, (K + ScaleBlockSize - 1) / ScaleBlockSize}, {Scale_Stride_AM, 1}));
    Tensor<XDataType> b_scale_preshuffled(
        HostTensorDescriptor({experts, (K + ScaleBlockSize - 1) / ScaleBlockSize, N * 2},
                             {N * 2 * Scale_Stride_BN, 1, Scale_Stride_BN},
                             Col{}));
    Tensor<D2DataType> d2_e_n(HostTensorDescriptor({sorted_size, N}, {1, 0}, Bypass{}));
    Tensor<EDataType> e_t_k_n_host_result(
        HostTensorDescriptor({tokens, topk, N}, {topk * N, N, 1}, Row{}));
    Tensor<EDataType> e_t_k_n_device_result(
        HostTensorDescriptor({tokens, topk, N}, {topk * N, N, 1}, Row{}));

    e_t_k_n_device_result.SetZero();
    std::cout << "a0_t_k:   " << a0_t_k.mDesc << std::endl;
    std::cout << "a1_t_k:   " << a1_t_k.mDesc << std::endl;
    std::cout << "b0_e_n_k: " << b0_e_n_k.mDesc << std::endl;
    std::cout << "b1_e_n_k: " << b1_e_n_k.mDesc << std::endl;
    std::cout << "d2_e_n:   " << d2_e_n.mDesc << std::endl;
    std::cout << "e_t_k_n:  " << e_t_k_n_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a0_t_k.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-1, 1});
        b0_e_n_k.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-1, 1});
        a1_t_k.GenerateTensorValue(GeneratorTensor_3<XDataType>{0, 1.0});
        b1_e_n_k.GenerateTensorValue(GeneratorTensor_3<XDataType>{0, 1.0});
        d2_e_n.GenerateTensorValue(GeneratorTensor_3<D2DataType>{0, 1.0});
        break;
    case 2:
        a0_t_k.GenerateTensorValue(GeneratorTensor_1<A0DataType>{});
        b0_e_n_k.GenerateTensorValue(GeneratorTensor_1<B0DataType>{});
        a1_t_k.GenerateTensorValue(GeneratorTensor_1<XDataType>{});
        b1_e_n_k.GenerateTensorValue(GeneratorTensor_1<XDataType>{});
        d2_e_n.GenerateTensorValue(GeneratorTensor_1<D2DataType>{0.1f});
        break;
    case 3:
        a0_t_k.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-1, 1});
        b0_e_n_k.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-1, 1});
        a1_t_k.GenerateTensorValue(GeneratorTensor_3<XDataType>{0, 1.0});
        b1_e_n_k.GenerateTensorValue(GeneratorTensor_3<XDataType>{0, 1.0});
        d2_e_n.GenerateTensorValue(GeneratorTensor_1<D2DataType>{});
        break;
    case 4:
        a0_t_k.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-2, 2});
        b0_e_n_k.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        a1_t_k.GenerateTensorValue(GeneratorTensor_1<XDataType>{});
        b1_e_n_k.GenerateTensorValue(GeneratorTensor_3<XDataType>{0, 5.0});
        d2_e_n.GenerateTensorValue(GeneratorTensor_1<D2DataType>{});
        break;
    case 5:
        a0_t_k.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-2, 2});
        b0_e_n_k.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        a1_t_k.GenerateTensorValue(GeneratorTensor_3<XDataType>{0, 1.0});
        b1_e_n_k.GenerateTensorValue(GeneratorTensor_1<XDataType>{});
        d2_e_n.GenerateTensorValue(GeneratorTensor_1<D2DataType>{1});
        break;
    case 6:
        a0_t_k.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-2, 2});
        b0_e_n_k.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        a1_t_k.GenerateTensorValue(GeneratorTensor_3<XDataType>{0, 1.0});
        b1_e_n_k.GenerateTensorValue(GeneratorTensor_1<XDataType>{});
        d2_e_n.GenerateTensorValue(GeneratorTensor_1<D2DataType>{});
        break;
    case 7:
        a0_t_k.GenerateTensorValue(GeneratorTensor_1<A0DataType>{0.5f});
        b0_e_n_k.GenerateTensorValue(GeneratorTensor_1<B0DataType>{1.5f});
        a1_t_k.GenerateTensorValue(GeneratorTensor_1<XDataType>{1.0f});
        b1_e_n_k.GenerateTensorValue(GeneratorTensor_1<XDataType>{1.0f});
        d2_e_n.GenerateTensorValue(GeneratorTensor_1<D2DataType>{0.1f});
        break;
    default:
        a0_t_k.GenerateTensorValue(GeneratorTensor_3<A0DataType>{0.0, 1.0});
        b0_e_n_k.GenerateTensorValue(GeneratorTensor_3<B0DataType>{-0.5, 0.5});
        a1_t_k.GenerateTensorValue(GeneratorTensor_3<XDataType>{0.0, 1.0});
        b1_e_n_k.GenerateTensorValue(GeneratorTensor_3<XDataType>{0.0, 1.0});
        d2_e_n.GenerateTensorValue(GeneratorTensor_3<D2DataType>{0.0, 1.0});
    }
    DeviceMem sorted_token_ids_dev(sizeof(ck::index_t) * sorted_token_ids.GetElementSpaceSize());
    DeviceMem expert_ids_dev(sizeof(ck::index_t) * expert_ids.GetElementSpaceSize());
    DeviceMem max_token_id_dev(sizeof(ck::index_t) * max_token_id.GetElementSpaceSize());
    DeviceMem a0_device_buf(sizeof(A0DataType) * a0_t_k.GetElementSpaceSize());
    DeviceMem a1_device_buf(sizeof(XDataType) * a_scale_sorted.GetElementSpaceSize());
    DeviceMem b0_device_buf(sizeof(B0DataType) * b0_e_n_k.GetElementSpaceSize());
    DeviceMem b1_device_buf(sizeof(XDataType) * b1_e_n_k.GetElementSpaceSize());
    DeviceMem d2_device_buf(sizeof(D2DataType) * d2_e_n.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) * e_t_k_n_device_result.GetElementSpaceSize());

    // A scale sorted
    for(int i = 0; i < sorted_size; i++)
    {
        int token_id = sorted_token_ids.mData[i] & 0x00FFFFFF;

        for(int k = 0; k < (K + ScaleBlockSize - 1) / ScaleBlockSize; k++)
        {
            if(token_id == tokens)
            {
                a_scale_sorted(i, k) = ck::type_convert<XDataType>(0);
            }
            else
            {
                a_scale_sorted(i, k) = a1_t_k(token_id, k);
            }
        }
    }

    // A/B scale shuffle
    preShuffleScaleBuffer<ck::is_same_v<A0Layout, Row>>(a_scale_sorted.mData.data(),
                                                        a_scale_preshuffled.mData.data(),
                                                        sorted_size,
                                                        K / ScaleBlockSize);
    preShuffleScaleBuffer<ck::is_same_v<B0Layout, Col>>(b1_e_n_k.mData.data(),
                                                        b_scale_preshuffled.mData.data(),
                                                        N * 2 * experts,
                                                        K / ScaleBlockSize);

    sorted_token_ids_dev.ToDevice(sorted_token_ids.mData.data());
    expert_ids_dev.ToDevice(expert_ids.mData.data());
    max_token_id_dev.ToDevice(max_token_id.mData.data());
    a0_device_buf.ToDevice(a0_t_k.mData.data());
    b0_device_buf.ToDevice(b0_e_n_k.mData.data());
    a1_device_buf.ToDevice(a_scale_preshuffled.mData.data());
    b1_device_buf.ToDevice(b_scale_preshuffled.mData.data());
    d2_device_buf.ToDevice(d2_e_n.mData.data());
    e_device_buf.ToDevice(e_t_k_n_device_result.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    // do GEMM
    auto device_op = DeviceOpInstance{};

    auto invoker  = device_op.MakeInvoker();
    auto argument = device_op.MakeArgument(
        sorted_token_ids_dev.GetDeviceBuffer(),
        expert_ids_dev.GetDeviceBuffer(),
        max_token_id_dev.GetDeviceBuffer(),
        a0_device_buf.GetDeviceBuffer(),
        a1_device_buf.GetDeviceBuffer(),
        b0_device_buf.GetDeviceBuffer(),
        b1_device_buf.GetDeviceBuffer(),
        std::array<const void*, NumDTensor>{nullptr, nullptr, d2_device_buf.GetDeviceBuffer()},
        e_device_buf.GetDeviceBuffer(),
        tokens,
        topk,
        sorted_size,
        N,
        K,
        StrideA,
        Scale_Stride_AM,
        StrideB,
        Scale_Stride_BN,
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

    if(!(ck::get_device_name() == "gfx942" || ck::get_device_name() == "gfx950"))
    {
        std::cout << "This kernel support gfx942 and gfx950 only" << std::endl;
    }

    if(time_kernel)
    {
        // not result correct here because output buf not setzero
        float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

        std::size_t flop =
            // FMA * tokens * N * (Gate+Up) * topk * K +
            // FMA * tokens * N * (Gate+Up) * topk * (K/BlockScale)
            std::size_t(2) * tokens * N * 2 * topk * K +
            std::size_t(2) * tokens * N * 2 * topk * K / ScaleBlockSize;

        std::size_t num_btype = sizeof(A0DataType) / 2 * tokens * topk * K +
                                sizeof(B0DataType) / 2 * K * N * 2 * experts +
                                sizeof(XDataType) * tokens * topk * K / ScaleBlockSize +
                                sizeof(XDataType) * K / ScaleBlockSize * N * 2 * experts +
                                sizeof(EDataType) * tokens * topk * N;

        float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

        float gb_per_sec = num_btype / 1.E6 / ave_time;

        std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s" << device_op.GetTypeString() << std::endl;
    }

    if(do_verification)
    {
        // gemm2 use atomic, so need to reinit outputs
        e_device_buf.ToDevice(e_t_k_n_device_result.mData.data());
        invoker.Run(argument, StreamConfig{nullptr, false, 0, 0, 1});

        Tensor<CShuffleDataType> c_t_k_n({tokens, topk, N}, {topk * N, N, 1}, Row{});

        using ReferenceGemmInstance =
            ck::tensor_operation::host::ReferenceMoeMXGemm1<A0DataType,
                                                            XDataType,
                                                            B0DataType,
                                                            XDataType,
                                                            CShuffleDataType,
                                                            D2DataType,
                                                            AccDataType,
                                                            PassThrough,
                                                            PassThrough,
                                                            PassThrough,
                                                            ActOP,
                                                            MulRoutedWeight>;
        auto ref_moe_gemm = ReferenceGemmInstance{};
        auto ref_invoker  = ref_moe_gemm.MakeInvoker();

        auto ref_argument = ref_moe_gemm.MakeArgument(sorted_token_ids,
                                                      expert_ids,
                                                      max_token_id,
                                                      MPerBlock,
                                                      a0_t_k,
                                                      a1_t_k,
                                                      b0_e_n_k,
                                                      b1_e_n_k,
                                                      d2_e_n,
                                                      c_t_k_n,
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
            for(int n = 0; n < N; ++n)
            {
                e_t_k_n_host_result(t, topk_id, n) =
                    ck::type_convert<EDataType>(c_t_k_n(t, topk_id, n));
            }
        }

        e_device_buf.FromDevice(e_t_k_n_device_result.mData.data());

        auto status =
            ck::utils::check_err(
                e_t_k_n_device_result, e_t_k_n_host_result, "Error: Incorrect results!", 1e-3, 5e-1)
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
