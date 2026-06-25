// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_b_scale.hpp"

using ADataType        = ck::half_t;
using BDataType        = ck::pk_i4_t;
using BScaleDataType   = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
using CDataType        = ck::half_t;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

static constexpr bool PermuteA = false;
static constexpr bool PermuteB = true;

static constexpr ck::index_t Scale_Block_N = 1;
static constexpr ck::index_t Scale_Block_K = 128;

static constexpr ck::index_t KPerBlock = 64;

// clang-format off
using DeviceGemmV2Instance = 
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV3<
        ALayout,   BLayout,  CLayout,   
        ADataType, BDataType, BScaleDataType, CDataType, AccDataType, CShuffleDataType, 
        AElementOp, BElementOp, CElementOp, GemmDefault, 
        256, Scale_Block_N, Scale_Block_K,
        128, 128,
        KPerBlock, 8, 16,
        16,   16,
        8,    2,
        S<8, 32, 1>,  S<1, 0, 2>,  S<1, 0, 2>,
        2, 8, 8, 0,
        S<2, 128, 1>,  S<1, 0, 2>,  S<1, 0, 2>,
        2, 16, 16, 0,
        1, 1, S<1, 16, 1, 16>, 4,
        ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, CDataType, CDataType, PermuteA, PermuteB>;

// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                        AccDataType,
                                                                        CDataType,
                                                                        AccDataType,
                                                                        PassThrough,
                                                                        PassThrough,
                                                                        PassThrough>;
template <typename ProblemType>
bool run_gemm(const ProblemType& problem_size, const ExecutionConfig& config)
{
    using namespace ck::literals;

    auto M       = problem_size.M;
    auto N       = problem_size.N;
    auto K       = problem_size.K;
    auto StrideA = problem_size.StrideA;
    auto StrideB = problem_size.StrideB;
    auto StrideC = problem_size.StrideC;
    auto KBatch  = problem_size.KBatch;

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    auto f_get_default_stride =
        [](std::size_t row, std::size_t col, ck::index_t stride, auto layout) {
            if(stride == -1)
            {
                // give a chance if stride is -1, return a default packed stride
                if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
                {
                    return static_cast<std::size_t>(col);
                }
                else
                {
                    return static_cast<std::size_t>(row);
                }
            }
            else
                return static_cast<std::size_t>(stride);
        };

    ck::index_t Scale_Stride_BN = (K + Scale_Block_K - 1) / Scale_Block_K;

    StrideA = f_get_default_stride(M, K, StrideA, ALayout{});
    StrideB = f_get_default_stride(K, N, StrideB, BLayout{});
    StrideC = f_get_default_stride(M, N, StrideC, CLayout{});

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
    Tensor<BDataType> b_k_n_permute(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
    Tensor<BScaleDataType> b1_k_n(f_host_tensor_descriptor((K + Scale_Block_K - 1) / Scale_Block_K,
                                                           (N + Scale_Block_N - 1) / Scale_Block_N,
                                                           Scale_Stride_BN,
                                                           BLayout{}));

    switch(config.init_method)
    {
    case 0:
        a_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        b_k_n.GenerateTensorValue(GeneratorTensor_1<BDataType>{1});
        b1_k_n.GenerateTensorValue(GeneratorTensor_1<BScaleDataType>{1});
        break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        b1_k_n.GenerateTensorValue(GeneratorTensor_3<BScaleDataType>{0, 1.0});
        break;
    case 2:
        a_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        b1_k_n.GenerateTensorValue(GeneratorTensor_1<BScaleDataType>{1});
        break;
    case 3:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_k_n.GenerateTensorValue(GeneratorTensor_1<BDataType>{1});
        b1_k_n.GenerateTensorValue(GeneratorTensor_1<BScaleDataType>{1});
        break;
    case 4:
        a_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        b_k_n.GenerateTensorValue(GeneratorTensor_1<BDataType>{1});
        b1_k_n.GenerateTensorValue(GeneratorTensor_3<BScaleDataType>{0, 1.0});
        break;
    case 5:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        b1_k_n.GenerateTensorValue(GeneratorTensor_1<BScaleDataType>{1});
        break;
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.5, 0.5});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        b1_k_n.GenerateTensorValue(GeneratorTensor_3<BScaleDataType>{0, 1.0});
    }

    Tensor<CDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "b1_k_n: " << b1_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;

    DeviceMem a_m_k_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_device_buf(sizeof(BDataType) * b_k_n_permute.mDesc.GetElementSpaceSize() / 2);
    DeviceMem b1_scale_device_buf(sizeof(BScaleDataType) * b1_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpaceSize());

    // weight permute
    if constexpr(PermuteB)
    {
        int K1 = KPerBlock;
        int K0 = K / KPerBlock;

        // int K0, N, K1
        for(int j = 0; j < K0; j++)
        {
            for(int i = 0; i < N; i++)
            {
                for(int jj = 0; jj < K1; jj++)
                {
                    b_k_n_permute(j * N * K1 + i * K1 + jj) = b_k_n(i * K + (j * K1 + jj));
                }
            }
        }
    }
    else
    {
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < K; j++)
            {
                b_k_n_permute(i * K + j) = b_k_n(i * K + j);
            }
        }
    }

    // vector pk_i4x4 permute
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < K; j += 8)
        {
            int input[8];

            for(int k = 0; k < 4; k++)
            {
                int i4x2         = b_k_n_permute(j + k * 2, i).data;
                input[k * 2 + 0] = (i4x2 >> 4) & 0xf;
                input[k * 2 + 1] = (i4x2 >> 0) & 0xf;
            }

            // permute 01234567->20643175
            {
                int hi   = input[2];
                int lo   = input[0];
                int i4x2 = (hi << 4) | lo;

                b_k_n_permute(j + 0, i) = i4x2;
            }

            {
                int hi   = input[6];
                int lo   = input[4];
                int i4x2 = (hi << 4) | lo;

                b_k_n_permute(j + 2, i) = i4x2;
            }

            {
                int hi   = input[3];
                int lo   = input[1];
                int i4x2 = (hi << 4) | lo;

                b_k_n_permute(j + 4, i) = i4x2;
            }

            {
                int hi   = input[7];
                int lo   = input[5];
                int i4x2 = (hi << 4) | lo;

                b_k_n_permute(j + 6, i) = i4x2;
            }
        }
    }

    a_m_k_device_buf.ToDevice(a_m_k.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n_permute.mData.data());
    b1_scale_device_buf.ToDevice(b1_k_n.mData.data());
    DeviceMem workspace;

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    // do GEMM
    auto gemm      = DeviceGemmV2Instance{};
    auto invoker   = gemm.MakeInvoker();
    float ave_time = 0;

    auto argument =
        gemm.MakeArgument(static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                          static_cast<BDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
                          static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
                          M,
                          N,
                          K,
                          StrideA,
                          StrideB,
                          StrideC,
                          Scale_Stride_BN,
                          static_cast<BScaleDataType*>(b1_scale_device_buf.GetDeviceBuffer()),
                          KBatch,
                          a_element_op,
                          b_element_op,
                          c_element_op);

    if(!gemm.IsSupportedArgument(argument))
    {
        std::cerr << gemm.GetTypeString() << " does not support this problem" << std::endl;

        return true;
    }

    if(!(ck::get_device_name() == "gfx942" || ck::get_device_name() == "gfx950" ||
         ck::is_gfx11_supported() || ck::is_gfx12_supported()))
    {
        std::cout << "This kernel support gfx942, gfx950, gfx11 and gfx12 only" << std::endl;

        return true;
    }

    bool pass = true;
    if(config.do_verification)
    {
        Tensor<float> b_k_n_dequant({K, N});

        float v_b = 0;
        for(int n = 0; n < N; n++)
        {
            for(int k = 0; k < K; k++)
            {
                ck::pk_i4_t i4x2 = b_k_n(k, n).data;
                int8_t i4        = 0;
                if(k % 2 == 1)
                    i4 = (i4x2.data >> 0) & 0xf;
                else
                    i4 = (i4x2.data >> 4) & 0xf;
                i4  = i4 - 8;
                v_b = ck::type_convert<float>(i4);

                b_k_n_dequant(k, n) =
                    ck::type_convert<float>(v_b) *
                    ck::type_convert<float>(b1_k_n(k / Scale_Block_K, n / Scale_Block_N));
            }
        }

        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n_dequant, c_m_n_host_result, PassThrough{}, PassThrough{}, PassThrough{});

        ref_invoker.Run(ref_argument);

        ave_time = invoker.Run(argument, StreamConfig{nullptr, false, 0});
        c_m_n_device_buf.FromDevice(c_m_n_device_result.mData.data());

        pass &= ck::utils::check_err(c_m_n_device_result,
                                     c_m_n_host_result,
                                     "Error: Incorrect results!",
                                     get_rtol<CDataType>(),
                                     get_atol<CDataType>());
    }

    if(config.time_kernel)
    {
        ave_time =
            invoker.Run(argument, StreamConfig{nullptr, config.time_kernel, 0, 20, 50, true, 50});

        std::size_t flop = 2_uz * M * N * K;
        std::size_t num_btype =
            sizeof(ADataType) * M * K +
            sizeof(BDataType) * K * N /
                (ck::is_same_v<ck::remove_cvref_t<BDataType>, ck::pk_i4_t> ? 2 : 1) +
            sizeof(CDataType) * M * N;

        float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

        float gb_per_sec = num_btype / 1.E6 / ave_time;

        std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s, " << gemm.GetTypeString() << std::endl;
    }
    return pass;
}

bool run_gemm_splitk_example(int argc, char* argv[])
{
    ProblemSizeSplitK problem_size;
    ExecutionConfig config;

    return !parse_cmd_args(argc, argv, problem_size, config) || run_gemm(problem_size, config);
}

int main(int argc, char* argv[]) { return !run_gemm_splitk_example(argc, argv); }
