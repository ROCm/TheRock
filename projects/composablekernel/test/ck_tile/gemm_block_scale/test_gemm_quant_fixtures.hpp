// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "test_gemm_quant_base.hpp"
#include "ck_tile/host/permute_pk_int4.hpp"
#include "ck_tile/host/tensor_shuffle_utils.hpp"

template <bool is_8bit>
constexpr ck_tile::index_t get_k_warp_tile()
{
#if CK_TILE_USE_WMMA
    return 16;
#else
    return is_8bit ? 64 : 32;
#endif
}

struct GemmConfigBase
{
    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool PermuteA = false;
    static constexpr bool PermuteB = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                         = 1;
    static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    static constexpr ck_tile::index_t TileParitionerM01      = 4;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool PreshuffleQuant           = false;
    static constexpr bool PreshuffleB               = false;
    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr bool TiledMMAPermuteN          = false;

    // Default GEMM tile sizes for tests
    static constexpr ck_tile::index_t M_Tile = 16;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 256;

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = get_k_warp_tile<false>();
};

struct GemmConfigPrefill : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128;
};

struct GemmConfigMxFp4 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128;
};

struct GemmConfigPreshuffleQuant : public GemmConfigBase
{
    static constexpr bool PreshuffleQuant = true;
};

struct GemmConfigTransposeC : public GemmConfigBase
{
    static constexpr bool TransposeC = true;
};

struct GemmConfigPreshuffleQuantTransposeC : public GemmConfigBase
{
    static constexpr bool PreshuffleQuant = true;
    static constexpr bool TransposeC      = true;
};

struct GemmConfigPadding : public GemmConfigBase
{
    static constexpr bool kPadN = true;
    static constexpr bool kPadK = true;
};

struct GemmConfigPreshuffleBDecode : public GemmConfigBase
{
    static constexpr bool PreshuffleB      = true;
    static constexpr bool DoubleSmemBuffer = true;

    // Default GEMM tile sizes for tests
    static constexpr ck_tile::index_t M_Tile = 16;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 256;

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = get_k_warp_tile<true>();
};

struct GemmConfigPreshuffleBPrefill : public GemmConfigBase
{
    static constexpr bool PreshuffleB      = true;
    static constexpr bool DoubleSmemBuffer = true;

    // Default GEMM tile sizes for tests
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128;

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = get_k_warp_tile<true>();
};

struct GemmConfigPreshuffleBPrefillTiledPermuteN : public GemmConfigPreshuffleBPrefill
{
    static constexpr int N_Repeat          = N_Tile / N_Warp_Tile / N_Warp;
    static constexpr bool TiledMMAPermuteN = N_Repeat % 2 == 0;
};

struct GemmConfigPreshuffleBPreshuffleQuantDecode : public GemmConfigPreshuffleBDecode
{
    static constexpr bool PreshuffleQuant = true;
};

template <typename Tuple>
class TestCkTileGemmAQuant : public TestCkTileGemmQuantBase<Tuple, TestCkTileGemmAQuant<Tuple>>
{
    using Base = TestCkTileGemmQuantBase<Tuple, TestCkTileGemmAQuant<Tuple>>;
    friend Base;

    public:
    using typename Base::AccDataType;
    using typename Base::ADataType;
    using typename Base::ALayout;
    using typename Base::AQLayout;
    using typename Base::BDataType;
    using typename Base::BLayout;
    using typename Base::CDataType;
    using typename Base::CLayout;
    using typename Base::ComputeDataType;
    using typename Base::QDataType;
    using typename Base::QuantGroupSize;

    static constexpr auto QuantType = Base::QuantType;

    protected:
    void SetUpQuantTypeSpecific() {}
    void TearDownQuantTypeSpecific() {}

    // AQuant-specific data generation
    void run_test_with_validation(ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K)
    {
        const ck_tile::index_t stride_A =
            ck_tile::get_default_stride(M, K, 0, this->is_row_major(ALayout{}));
        const ck_tile::index_t stride_B =
            ck_tile::get_default_stride(K, N, 0, this->is_row_major(BLayout{}));
        const ck_tile::index_t stride_C =
            ck_tile::get_default_stride(M, N, 0, this->is_row_major(CLayout{}));

        // AQuant uses grouped quantization for A matrix
        const ck_tile::index_t AQK = ck_tile::integer_divide_ceil(K, QuantGroupSize::kK);
        // AQLayout is parameterized in the test tuple (can be RowMajor or ColumnMajor for AQuant)
        const ck_tile::index_t stride_AQ =
            ck_tile::get_default_stride(M, AQK, 0, this->is_row_major(AQLayout{}));

        // Generate test data
        ck_tile::HostTensor<ADataType> a_m_k(
            ck_tile::host_tensor_descriptor(M, K, stride_A, this->is_row_major(ALayout{})));
        // AQLayout is independently specified for each test case
        ck_tile::HostTensor<QDataType> aq_m_aqk(
            ck_tile::host_tensor_descriptor(M, AQK, stride_AQ, this->is_row_major(AQLayout{})));
        ck_tile::HostTensor<BDataType> b_k_n(
            ck_tile::host_tensor_descriptor(K, N, stride_B, this->is_row_major(BLayout{})));

        // Initialize data with random values
        if constexpr(std::is_same_v<ADataType, ck_tile::pk_int4_t>)
        {
            ck_tile::FillUniformDistribution<ADataType>{-5.0f, 5.0f}(a_m_k);
        }
        else
        {
            ck_tile::FillUniformDistribution<ADataType>{-2.0f, 3.0f}(a_m_k);
        }
        ck_tile::FillUniformDistribution<BDataType>{-5.0f, 5.0f}(b_k_n);
        ck_tile::FillUniformDistribution<QDataType>{-2.0f, 2.0f}(aq_m_aqk);

        // Allocate device memory
        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size() * sizeof(ADataType));
        ck_tile::DeviceMem aq_m_aqk_dev_buf(aq_m_aqk.get_element_space_size() * sizeof(QDataType));
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size() * sizeof(BDataType));
        ck_tile::DeviceMem c_m_n_dev_buf(M * N * sizeof(CDataType));

        // Copy to device
        if constexpr(std::is_same_v<ADataType, ck_tile::pk_int4_t>)
        {
            // Permute vector pk_i4x4 data for device implementation
            ck_tile::HostTensor<ADataType> temp = a_m_k;
            ck_tile::permute_vectors_i4x4_b(temp);
            a_m_k_dev_buf.ToDevice(temp.data());
        }
        else
        {
            a_m_k_dev_buf.ToDevice(a_m_k.data());
        }
        // aq_m_aqk_dev_buf.ToDevice(aq_m_aqk.data());
        if constexpr(Base::GemmConfig::PreshuffleQuant)
        {
            ck_tile::HostTensor<QDataType> aq_shuffle_host =
                ck_tile::shuffle_aq(&aq_m_aqk, Base::GemmConfig::K_Tile / QuantGroupSize::kK);
            aq_m_aqk_dev_buf.ToDevice(aq_shuffle_host.data());
        }
        else
        {
            aq_m_aqk_dev_buf.ToDevice(aq_m_aqk.data());
        }
        b_k_n_dev_buf.ToDevice(b_k_n.data());

        // Create args for kernel execution
        ck_tile::QuantGemmHostArgs args{
            a_m_k_dev_buf.GetDeviceBuffer(),    // a_ptr
            b_k_n_dev_buf.GetDeviceBuffer(),    // b_ptr
            c_m_n_dev_buf.GetDeviceBuffer(),    // c_ptr
            aq_m_aqk_dev_buf.GetDeviceBuffer(), // aq_ptr (scales)
            nullptr,                            // bq_ptr (not used for AQuant)
            1,                                  // k_batch
            M,
            N,
            K,   // M, N, K
            AQK, // QK_A
            0,   // QK_B (not used for AQuant)
            stride_A,
            stride_B,
            stride_C,
            stride_AQ,
            0 // strides
        };

        // Run the kernel
        ck_tile::stream_config stream_config{};
        this->invoke_quant_gemm(args, stream_config);

        // Validation using reference implementation
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, this->is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        // Run reference AQuant implementation
        ck_tile::reference_gemm_quant<ADataType,
                                      QDataType,
                                      BDataType,
                                      AccDataType,
                                      CDataType,
                                      QuantGroupSize,
                                      true>(a_m_k, aq_m_aqk, b_k_n, c_m_n_host_ref);

        // Get device result
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(
            ck_tile::host_tensor_descriptor(M, N, stride_C, this->is_row_major(CLayout{})));
        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.mData.data());

        // Calculate error tolerances
        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto rtol_atol =
            this->template calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
                K, 1, max_accumulated_value);

        // Validate results
        bool pass = ck_tile::check_err(c_m_n_dev_result,
                                       c_m_n_host_ref,
                                       "Error: Incorrect results!",
                                       rtol_atol.at(ck_tile::number<0>{}),
                                       rtol_atol.at(ck_tile::number<1>{}));

        EXPECT_TRUE(pass) << "AQuantGrouped validation failed with M=" << M << ", N=" << N
                          << ", K=" << K;

        if(!pass)
        {
            std::cout << "AQuantGrouped - Relative error threshold: "
                      << rtol_atol.at(ck_tile::number<0>{})
                      << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                      << std::endl;
        }
    }

    private:
    // AQuant-specific pipeline implementation
    template <typename CodegenGemmShape, typename TilePartitioner, typename CodegenGemmTraits>
    void run_quant_gemm_impl(const ck_tile::QuantGemmHostArgs& args,
                             const ck_tile::stream_config& s)
    {
        using GemmPipelineProblem = ck_tile::GemmPipelineProblemBase<ADataType,
                                                                     BDataType,
                                                                     AccDataType,
                                                                     CodegenGemmShape,
                                                                     CodegenGemmTraits,
                                                                     ComputeDataType>;

        using BaseGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmPipelineProblem>;

        const ck_tile::index_t K_split  = (args.K + Base::K_Tile - 1) / Base::K_Tile * Base::K_Tile;
        const ck_tile::index_t num_loop = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop         = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v  = tail_number_.value;
            constexpr bool transpose_c    = CodegenGemmTraits::TransposeC;

            using PipelineProblem =
                ck_tile::GemmAQuantPipelineProblem<ADataType,
                                                   QDataType,
                                                   BDataType,
                                                   AccDataType,
                                                   CodegenGemmShape,
                                                   CodegenGemmTraits,
                                                   QuantGroupSize,
                                                   transpose_c,
                                                   ComputeDataType,
                                                   ck_tile::GemmPipelineScheduler::Intrawave,
                                                   has_hot_loop_v,
                                                   tail_number_v>;

            using GemmPipeline = ck_tile::AQuantGemmPipelineAgBgCrCompV3<PipelineProblem>;
            using GemmEpilogue = ck_tile::CShuffleEpilogue<
                ck_tile::CShuffleEpilogueProblem<ADataType,
                                                 BDataType,
                                                 ck_tile::tuple<>,
                                                 AccDataType,
                                                 CDataType,
                                                 ck_tile::tuple<>,
                                                 CLayout,
                                                 ck_tile::element_wise::PassThrough,
                                                 TilePartitioner::MPerBlock,
                                                 TilePartitioner::NPerBlock,
                                                 Base::M_Warp,
                                                 Base::N_Warp,
                                                 Base::M_Warp_Tile,
                                                 Base::N_Warp_Tile,
                                                 Base::K_Warp_Tile,
                                                 transpose_c>>;

            using Kernel = ck_tile::QuantGemmKernel<TilePartitioner,
                                                    GemmPipeline,
                                                    GemmEpilogue,
                                                    ck_tile::QuantType::AQuantGrouped>;

            auto kargs        = Kernel::MakeKernelArgs(args);
            const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
            const dim3 blocks = Kernel::BlockSize();

            if(!Kernel::IsSupportedArgument(kargs))
            {
                throw std::runtime_error("Arguments not supported for AQuant kernel");
            }

            ck_tile::launch_kernel(s,
                                   ck_tile::make_kernel<GemmConfigBase::kBlockPerCu>(
                                       Kernel{}, grids, blocks, 0, kargs));
        };

        return BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
    }
};

// BQuant-specific test fixture
template <typename Tuple>
class TestCkTileGemmBQuant : public TestCkTileGemmQuantBase<Tuple, TestCkTileGemmBQuant<Tuple>>
{
    using Base = TestCkTileGemmQuantBase<Tuple, TestCkTileGemmBQuant<Tuple>>;
    friend Base;

    public:
    using typename Base::AccDataType;
    using typename Base::ADataType;
    using typename Base::ALayout;
    using typename Base::BDataType;
    using typename Base::BLayout;
    using typename Base::CDataType;
    using typename Base::CLayout;
    using typename Base::ComputeDataType;
    using typename Base::GemmConfig;
    using typename Base::QDataType;
    using typename Base::QuantGroupSize;

    // Re-use AQLayout from tuple parameters as BQLayout
    using BQLayout = typename Base::AQLayout;

    static constexpr auto QuantType        = Base::QuantType;
    static constexpr auto PreshuffleB      = Base::PreshuffleB;
    static constexpr auto TiledMMAPermuteN = Base::TiledMMAPermuteN;

    protected:
    void SetUpQuantTypeSpecific() {}
    void TearDownQuantTypeSpecific() {}

    void run_test_with_validation(ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K)
    {
        const ck_tile::index_t stride_A = K;
        const ck_tile::index_t stride_B =
            std::is_same_v<BDataType, ck_tile::pk_fp4_raw_t> ? (K / 2) : K;
        const ck_tile::index_t stride_C = N;

        // BQuant uses block/grouped quantization for B matrix
        const ck_tile::index_t BQN       = ck_tile::integer_divide_ceil(N, QuantGroupSize::kN);
        const ck_tile::index_t BQK       = ck_tile::integer_divide_ceil(K, QuantGroupSize::kK);
        const ck_tile::index_t stride_BQ = this->is_row_major(BQLayout{}) ? BQN : BQK;

        // Generate test data
        ck_tile::HostTensor<ADataType> a_m_k(
            ck_tile::host_tensor_descriptor(M, K, stride_A, this->is_row_major(ALayout{})));
        ck_tile::HostTensor<BDataType> b_k_n(ck_tile::host_tensor_descriptor(
            std::is_same_v<BDataType, ck_tile::pk_fp4_raw_t> ? K / 2 : K,
            N,
            stride_B,
            this->is_row_major(BLayout{})));
        ck_tile::HostTensor<QDataType> bq_bqk_bqn(
            ck_tile::host_tensor_descriptor(BQK, BQN, stride_BQ, this->is_row_major(BQLayout{})));

        // Initialize data with random values
        ck_tile::FillUniformDistribution<ADataType>{-0.5f, 0.5f}(a_m_k);
        if constexpr(std::is_same_v<BDataType, ck_tile::pk_fp4_raw_t>)
        {
            ck_tile::FillUniformDistribution<BDataType>{-5.0f, 5.0f}(b_k_n);
            ck_tile::FillUniformDistribution<QDataType>{125.f, 130.f}(bq_bqk_bqn);
        }
        else
        {
            ck_tile::FillUniformDistribution<BDataType>{0.f, 1.f}(b_k_n);
            ck_tile::FillUniformDistribution<QDataType>{-1.0f, 1.0f}(bq_bqk_bqn);
        }

        // Allocate device memory
        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size() * sizeof(ADataType));
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size() * sizeof(BDataType));
        ck_tile::DeviceMem bq_bqk_bqn_dev_buf(bq_bqk_bqn.get_element_space_size() *
                                              sizeof(QDataType));
        ck_tile::DeviceMem c_m_n_dev_buf(M * N * sizeof(CDataType));

        // Copy to device
        a_m_k_dev_buf.ToDevice(a_m_k.data());
        ck_tile::HostTensor<BDataType> b_k_n_dev = b_k_n;
        if constexpr(PreshuffleB)
        {
            if constexpr(TiledMMAPermuteN && QuantGroupSize::kN == 1)
            {
                printf("PreshuffleB with TiledMMAPermuteN\n");
                b_k_n_dev = ck_tile::shuffle_b_permuteN<GemmConfig>(b_k_n);
            }
            else
            {
                printf("PreshuffleB without TiledMMAPermuteN\n");
                b_k_n_dev = ck_tile::shuffle_b<GemmConfig>(b_k_n);
            }
        }
        if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
        {
            ck_tile::permute_vectors_i4x4_b(b_k_n_dev);
        }

        b_k_n_dev_buf.ToDevice(b_k_n_dev.data());

        if constexpr(PreshuffleB && TiledMMAPermuteN && QuantGroupSize::kN == 1)
        {
            printf("Preshuffle BQ with TiledMMAPermuteN \n");
            ck_tile::HostTensor<QDataType> bq_shuffle_host =
                ck_tile::bq_permuteN<GemmConfig>(bq_bqk_bqn, QuantGroupSize::kN);
            bq_bqk_bqn_dev_buf.ToDevice(bq_shuffle_host.data());
        }
        else if constexpr(GemmConfig::PreshuffleQuant)
        {
            ck_tile::HostTensor<QDataType> bq_shuffle_host =
                ck_tile::shuffle_bq(&bq_bqk_bqn, GemmConfig::K_Tile / QuantGroupSize::kK);
            bq_bqk_bqn_dev_buf.ToDevice(bq_shuffle_host.data());
        }
        else
        {
            bq_bqk_bqn_dev_buf.ToDevice(bq_bqk_bqn.data());
        }

        // Create args for kernel execution
        ck_tile::QuantGemmHostArgs args{
            a_m_k_dev_buf.GetDeviceBuffer(),      // a_ptr
            b_k_n_dev_buf.GetDeviceBuffer(),      // b_ptr
            c_m_n_dev_buf.GetDeviceBuffer(),      // c_ptr
            nullptr,                              // aq_ptr (not used for BQuant)
            bq_bqk_bqn_dev_buf.GetDeviceBuffer(), // bq_ptr (scales)
            1,                                    // k_batch
            M,
            N,
            K,   // M, N, K
            0,   // QK_A (not used for BQuant)
            BQK, // QK_B - TODO: we can remove BQK and BQN from args later?
            stride_A,
            stride_B,
            stride_C,
            0,
            stride_BQ // strides
        };

        // Run the kernel
        ck_tile::stream_config stream_config{};
        this->invoke_quant_gemm(args, stream_config);

        // Validation using reference implementation
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, this->is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        // Run reference BQuant implementation
        if constexpr(std::is_same_v<BDataType, ck_tile::pk_fp4_raw_t>)
            ck_tile::reference_mxfp4gemm_quant<ADataType,
                                               QDataType,
                                               BDataType,
                                               AccDataType,
                                               CDataType,
                                               QuantGroupSize,
                                               false>(a_m_k, bq_bqk_bqn, b_k_n, c_m_n_host_ref);
        else
            ck_tile::reference_gemm_quant<ADataType,
                                          QDataType,
                                          BDataType,
                                          AccDataType,
                                          CDataType,
                                          QuantGroupSize,
                                          false>(a_m_k, bq_bqk_bqn, b_k_n, c_m_n_host_ref);

        // Get device result
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(
            ck_tile::host_tensor_descriptor(M, N, stride_C, this->is_row_major(CLayout{})));
        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.mData.data());

        // Calculate error tolerances
        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto rtol_atol =
            this->template calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
                K, 1, max_accumulated_value);

        // Validate results
        bool pass = ck_tile::check_err(c_m_n_dev_result,
                                       c_m_n_host_ref,
                                       "Error: Incorrect results!",
                                       rtol_atol.at(ck_tile::number<0>{}),
                                       rtol_atol.at(ck_tile::number<1>{}));

        EXPECT_TRUE(pass) << "BQuantGrouped validation failed with M=" << M << ", N=" << N
                          << ", K=" << K;

        if(!pass)
        {
            std::cout << "BQuantGrouped - Relative error threshold: "
                      << rtol_atol.at(ck_tile::number<0>{})
                      << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                      << std::endl;
        }
    }

    private:
    // BQuant-specific pipeline implementation
    template <typename CodegenGemmShape, typename TilePartitioner, typename CodegenGemmTraits>
    void run_quant_gemm_impl(const ck_tile::QuantGemmHostArgs& args,
                             const ck_tile::stream_config& s)
    {
        using GemmPipelineProblem = ck_tile::GemmPipelineProblemBase<ADataType,
                                                                     BDataType,
                                                                     AccDataType,
                                                                     CodegenGemmShape,
                                                                     CodegenGemmTraits,
                                                                     ComputeDataType>;

        using BaseGemmPipeline = std::conditional_t<
            PreshuffleB == false,
            ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmPipelineProblem>,
            ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2<GemmPipelineProblem>>;

        const ck_tile::index_t K_split  = (args.K + Base::K_Tile - 1) / Base::K_Tile * Base::K_Tile;
        const ck_tile::index_t num_loop = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop         = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v  = tail_number_.value;

            using PipelineProblem =
                ck_tile::GemmBQuantPipelineProblem<ADataType,
                                                   BDataType,
                                                   QDataType,
                                                   AccDataType,
                                                   CodegenGemmShape,
                                                   CodegenGemmTraits,
                                                   QuantGroupSize,
                                                   ComputeDataType,
                                                   ck_tile::GemmPipelineScheduler::Intrawave,
                                                   has_hot_loop_v,
                                                   tail_number_v>;

            using GemmPipeline = std::conditional_t<
                PreshuffleB == false,
                std::conditional_t<std::is_same_v<BDataType, ck_tile::pk_fp4_raw_t>,
                                   ck_tile::MxFp4GemmPipelineAgBgCrCompV3<PipelineProblem>,
                                   ck_tile::BQuantGemmPipelineAgBgCrCompV3<PipelineProblem>>,
                ck_tile::WPQuantBPipelineAgBgCrV2<PipelineProblem>>;

            using GemmEpilogue = ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
                ADataType,
                std::conditional_t<std::is_same_v<BDataType, ck_tile::pk_fp4_raw_t>,
                                   ADataType,
                                   BDataType>,
                ck_tile::tuple<>,
                AccDataType,
                CDataType,
                ck_tile::tuple<>,
                CLayout,
                ck_tile::element_wise::PassThrough,
                TilePartitioner::MPerBlock,
                TilePartitioner::NPerBlock,
                Base::M_Warp,
                Base::N_Warp,
                Base::M_Warp_Tile,
                Base::N_Warp_Tile,
                Base::K_Warp_Tile,
                false, // transpose_c
                1,
                false,
                1,
                TiledMMAPermuteN>>;

            using Kernel = ck_tile::QuantGemmKernel<TilePartitioner,
                                                    GemmPipeline,
                                                    GemmEpilogue,
                                                    ck_tile::QuantType::BQuantGrouped>;

            auto kargs        = Kernel::MakeKernelArgs(args);
            const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
            const dim3 blocks = Kernel::BlockSize();

            if(!Kernel::IsSupportedArgument(kargs))
            {
                throw std::runtime_error("Arguments not supported for BQuant kernel");
            }

            ck_tile::launch_kernel(s,
                                   ck_tile::make_kernel<GemmConfigBase::kBlockPerCu>(
                                       Kernel{}, grids, blocks, 0, kargs));
        };

        return BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
    }
};

// ABQuant-specific test fixture
template <typename Tuple>
class TestCkTileGemmABQuant : public TestCkTileGemmQuantBase<Tuple, TestCkTileGemmABQuant<Tuple>>
{
    using Base = TestCkTileGemmQuantBase<Tuple, TestCkTileGemmABQuant<Tuple>>;
    friend Base;

    public:
    using typename Base::AccDataType;
    using typename Base::ADataType;
    using typename Base::ALayout;
    using typename Base::AQLayout;
    using typename Base::AQuantGroupSize;
    using typename Base::BDataType;
    using typename Base::BLayout;
    using typename Base::BQuantGroupSize;
    using typename Base::CDataType;
    using typename Base::CLayout;
    using typename Base::ComputeDataType;
    using typename Base::GemmConfig;
    using typename Base::QDataType;
    using BQLayout = ck_tile::tensor_layout::gemm::ColumnMajor;

    static constexpr auto QuantType        = Base::QuantType;
    static constexpr auto PreshuffleB      = Base::PreshuffleB;
    static constexpr auto TiledMMAPermuteN = Base::TiledMMAPermuteN;

    protected:
    void SetUpQuantTypeSpecific() {}
    void TearDownQuantTypeSpecific() {}

    void run_test_with_validation(ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K)
    {
        const ck_tile::index_t stride_A =
            ck_tile::get_default_stride(M, K, 0, this->is_row_major(ALayout{}));
        const ck_tile::index_t stride_B =
            ck_tile::get_default_stride(K, N, 0, this->is_row_major(BLayout{}));
        const ck_tile::index_t stride_C =
            ck_tile::get_default_stride(M, N, 0, this->is_row_major(CLayout{}));

        // AQuant uses grouped quantization for A matrix
        const ck_tile::index_t AQK = ck_tile::integer_divide_ceil(K, AQuantGroupSize::kK);
        // BQuant uses block/grouped quantization for B matrix
        const ck_tile::index_t BQN = ck_tile::integer_divide_ceil(N, BQuantGroupSize::kN);
        const ck_tile::index_t BQK = ck_tile::integer_divide_ceil(K, BQuantGroupSize::kK);
        const ck_tile::index_t stride_AQ =
            ck_tile::get_default_stride(M, AQK, 0, this->is_row_major(AQLayout{}));
        const ck_tile::index_t stride_BQ =
            ck_tile::get_default_stride(BQK, BQN, 0, this->is_row_major(BQLayout{}));
        // Generate test data
        ck_tile::HostTensor<ADataType> a_m_k(
            ck_tile::host_tensor_descriptor(M, K, stride_A, this->is_row_major(ALayout{})));
        // AQLayout is independently specified for each test case
        ck_tile::HostTensor<QDataType> aq_m_aqk( // AQDataType
            ck_tile::host_tensor_descriptor(M, AQK, stride_AQ, this->is_row_major(AQLayout{})));
        ck_tile::HostTensor<BDataType> b_k_n(
            ck_tile::host_tensor_descriptor(K, N, stride_B, this->is_row_major(BLayout{})));
        ck_tile::HostTensor<QDataType> bq_bqk_bqn(
            ck_tile::host_tensor_descriptor(BQK, BQN, stride_BQ, this->is_row_major(BQLayout{})));

        // Initialize data with random values
        if constexpr(std::is_same_v<ADataType, ck_tile::pk_int4_t>)
        {
            ck_tile::FillUniformDistribution<ADataType>{-5.0f, 5.0f}(a_m_k);
        }
        else
        {
            ck_tile::FillUniformDistribution<ADataType>{-2.0f, 3.0f}(a_m_k);
        }
        ck_tile::FillUniformDistribution<BDataType>{-5.0f, 5.0f}(b_k_n);
        ck_tile::FillUniformDistribution<QDataType>{-2.0f, 2.0f}(aq_m_aqk);
        ck_tile::FillUniformDistribution<QDataType>{-2.0f, 2.0f}(bq_bqk_bqn);
        // Allocate device memory
        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size() * sizeof(ADataType));
        ck_tile::DeviceMem aq_m_aqk_dev_buf(aq_m_aqk.get_element_space_size() *
                                            sizeof(QDataType)); // AQDataType
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size() * sizeof(BDataType));
        ck_tile::DeviceMem bq_bqk_bqn_dev_buf(bq_bqk_bqn.get_element_space_size() *
                                              sizeof(QDataType));
        ck_tile::DeviceMem c_m_n_dev_buf(M * N * sizeof(CDataType));

        // Copy to device
        if constexpr(std::is_same_v<ADataType, ck_tile::pk_int4_t>)
        {
            // Permute vector pk_i4x4 data for device implementation
            ck_tile::HostTensor<ADataType> temp = a_m_k;
            ck_tile::permute_vectors_i4x4_b(temp);
            a_m_k_dev_buf.ToDevice(temp.data());
        }
        else
        {
            a_m_k_dev_buf.ToDevice(a_m_k.data());
        }
        ck_tile::HostTensor<BDataType> b_k_n_dev = b_k_n;
        if constexpr(PreshuffleB)
        {
            if constexpr(TiledMMAPermuteN && BQuantGroupSize::kN == 1)
            {
                printf("PreshuffleB with TiledMMAPermuteN\n");
                b_k_n_dev = ck_tile::shuffle_b_permuteN<GemmConfig>(b_k_n);
            }
            else
            {
                printf("PreshuffleB without TiledMMAPermuteN\n");
                b_k_n_dev = ck_tile::shuffle_b<GemmConfig>(b_k_n);
            }
        }
        if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
        {
            ck_tile::permute_vectors_i4x4_b(b_k_n_dev);
        }

        b_k_n_dev_buf.ToDevice(b_k_n_dev.data());

        if constexpr(Base::GemmConfig::PreshuffleQuant)
        {
            ck_tile::HostTensor<QDataType> aq_shuffle_host =
                ck_tile::shuffle_aq(&aq_m_aqk, Base::GemmConfig::K_Tile / AQuantGroupSize::kK);
            aq_m_aqk_dev_buf.ToDevice(aq_shuffle_host.data());
        }
        else
        {
            aq_m_aqk_dev_buf.ToDevice(aq_m_aqk.data());
        }
        if constexpr(PreshuffleB && TiledMMAPermuteN && BQuantGroupSize::kN == 1)
        {
            printf("Preshuffle BQ with TiledMMAPermuteN \n");
            ck_tile::HostTensor<QDataType> bq_shuffle_host =
                ck_tile::bq_permuteN<GemmConfig>(bq_bqk_bqn, BQuantGroupSize::kN);
            bq_bqk_bqn_dev_buf.ToDevice(bq_shuffle_host.data());
        }
        else if constexpr(GemmConfig::PreshuffleQuant)
        {
            ck_tile::HostTensor<QDataType> bq_shuffle_host =
                ck_tile::shuffle_bq(&bq_bqk_bqn, GemmConfig::K_Tile / BQuantGroupSize::kK);
            bq_bqk_bqn_dev_buf.ToDevice(bq_shuffle_host.data());
        }
        else
        {
            bq_bqk_bqn_dev_buf.ToDevice(bq_bqk_bqn.data());
        }

        // Create args for kernel execution
        ck_tile::QuantGemmHostArgs args{
            a_m_k_dev_buf.GetDeviceBuffer(),      // a_ptr
            b_k_n_dev_buf.GetDeviceBuffer(),      // b_ptr
            c_m_n_dev_buf.GetDeviceBuffer(),      // c_ptr
            aq_m_aqk_dev_buf.GetDeviceBuffer(),   // aq_ptr (scales)
            bq_bqk_bqn_dev_buf.GetDeviceBuffer(), // bq_ptr (scales)
            1,                                    // k_batch
            M,
            N,
            K,   // M, N, K
            AQK, // QK_A
            BQK, // QK_B
            stride_A,
            stride_B,
            stride_C,
            stride_AQ,
            stride_BQ // strides
        };

        // Run the kernel
        ck_tile::stream_config stream_config{};
        this->invoke_quant_gemm(args, stream_config);

        // Validation using reference implementation
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, this->is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        // Run reference ABQuant implementation
        ck_tile::reference_gemm_abquant<ADataType,
                                        QDataType, // AQDataType
                                        BDataType,
                                        QDataType,
                                        AccDataType,
                                        CDataType,
                                        AQuantGroupSize,
                                        BQuantGroupSize>(
            a_m_k, aq_m_aqk, b_k_n, bq_bqk_bqn, c_m_n_host_ref);

        // Get device result
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(
            ck_tile::host_tensor_descriptor(M, N, stride_C, this->is_row_major(CLayout{})));
        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.mData.data());

        // Calculate error tolerances
        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto rtol_atol =
            this->template calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
                K, 1, max_accumulated_value);

        // Validate results
        bool pass = ck_tile::check_err(c_m_n_dev_result,
                                       c_m_n_host_ref,
                                       "Error: Incorrect results!",
                                       rtol_atol.at(ck_tile::number<0>{}),
                                       rtol_atol.at(ck_tile::number<1>{}));

        EXPECT_TRUE(pass) << "ABQuantGrouped validation failed with M=" << M << ", N=" << N
                          << ", K=" << K;

        if(!pass)
        {
            std::cout << "ABQuantGrouped - Relative error threshold: "
                      << rtol_atol.at(ck_tile::number<0>{})
                      << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                      << std::endl;
        }
    }

    private:
    // ABQuant-specific pipeline implementation
    template <typename CodegenGemmShape, typename TilePartitioner, typename CodegenGemmTraits>
    void run_quant_gemm_impl(const ck_tile::QuantGemmHostArgs& args,
                             const ck_tile::stream_config& s)
    {
        using GemmPipelineProblem = ck_tile::GemmPipelineProblemBase<ADataType,
                                                                     BDataType,
                                                                     AccDataType,
                                                                     CodegenGemmShape,
                                                                     CodegenGemmTraits,
                                                                     ComputeDataType>;

        using BaseGemmPipeline = std::conditional_t<
            PreshuffleB == true,
            ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2<GemmPipelineProblem>,
            ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmPipelineProblem>>;

        const ck_tile::index_t K_split  = (args.K + Base::K_Tile - 1) / Base::K_Tile * Base::K_Tile;
        const ck_tile::index_t num_loop = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop         = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v  = tail_number_.value;
            constexpr bool transpose_c    = CodegenGemmTraits::TransposeC;

            using PipelineProblem =
                ck_tile::GemmABQuantPipelineProblem<ADataType,
                                                    QDataType, // AQDataType
                                                    BDataType,
                                                    QDataType,
                                                    AccDataType,
                                                    CodegenGemmShape,
                                                    CodegenGemmTraits,
                                                    AQuantGroupSize,
                                                    BQuantGroupSize,
                                                    transpose_c,
                                                    ComputeDataType,
                                                    ck_tile::GemmPipelineScheduler::Intrawave,
                                                    has_hot_loop_v,
                                                    tail_number_v>;

            using GemmPipeline =
                std::conditional_t<PreshuffleB == true,
                                   ck_tile::WPABQuantBPipelineAgBgCrV2<PipelineProblem>,
                                   ck_tile::ABQuantGemmPipelineAgBgCrCompV3<PipelineProblem>>;

            using GemmEpilogue = ck_tile::CShuffleEpilogue<
                ck_tile::CShuffleEpilogueProblem<ADataType,
                                                 BDataType,
                                                 ck_tile::tuple<>,
                                                 AccDataType,
                                                 CDataType,
                                                 ck_tile::tuple<>,
                                                 CLayout,
                                                 ck_tile::element_wise::PassThrough,
                                                 TilePartitioner::MPerBlock,
                                                 TilePartitioner::NPerBlock,
                                                 Base::M_Warp,
                                                 Base::N_Warp,
                                                 Base::M_Warp_Tile,
                                                 Base::N_Warp_Tile,
                                                 Base::K_Warp_Tile,
                                                 transpose_c,
                                                 1,
                                                 false,
                                                 1,
                                                 TiledMMAPermuteN>>;

            using Kernel = ck_tile::QuantGemmKernel<TilePartitioner,
                                                    GemmPipeline,
                                                    GemmEpilogue,
                                                    ck_tile::QuantType::ABQuantGrouped>;

            auto kargs        = Kernel::MakeKernelArgs(args);
            const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
            const dim3 blocks = Kernel::BlockSize();

            if(!Kernel::IsSupportedArgument(kargs))
            {
                throw std::runtime_error("Arguments not supported for ABQuant kernel");
            }

            ck_tile::launch_kernel(s,
                                   ck_tile::make_kernel<GemmConfigBase::kBlockPerCu>(
                                       Kernel{}, grids, blocks, 0, kargs));
        };

        return BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
    }
};

template <typename Tuple>
class TestCkTileGemmPreshuffleBBQuant : public TestCkTileGemmBQuant<Tuple>
{
};

// RowColQuant-specific test fixture
template <typename Tuple>
class TestCkTileGemmRowColQuant
    : public TestCkTileGemmQuantBase<Tuple, TestCkTileGemmRowColQuant<Tuple>>
{
    using Base = TestCkTileGemmQuantBase<Tuple, TestCkTileGemmRowColQuant<Tuple>>;
    friend Base;

    public:
    using typename Base::AccDataType;
    using typename Base::ADataType;
    using typename Base::ALayout;
    using typename Base::BDataType;
    using typename Base::BLayout;
    using typename Base::CDataType;
    using typename Base::CLayout;
    using typename Base::ComputeDataType;
    using typename Base::QDataType;
    using typename Base::QuantGroupSize;

    static constexpr auto QuantType = Base::QuantType;

    protected:
    void SetUpQuantTypeSpecific() {}
    void TearDownQuantTypeSpecific() {}

    void run_test_with_validation(ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K)
    {
        const ck_tile::index_t stride_A = K;
        const ck_tile::index_t stride_B = K;
        const ck_tile::index_t stride_C = N;

        // RowColQuant uses per-row and per-column scales
        const ck_tile::index_t stride_row_scales = 1;
        const ck_tile::index_t stride_col_scales = 1;

        // Generate test data
        ck_tile::HostTensor<ADataType> a_m_k(
            ck_tile::host_tensor_descriptor(M, K, stride_A, this->is_row_major(ALayout{})));
        ck_tile::HostTensor<BDataType> b_k_n(
            ck_tile::host_tensor_descriptor(K, N, stride_B, this->is_row_major(BLayout{})));
        ck_tile::HostTensor<QDataType> row_scales_m(ck_tile::host_tensor_descriptor(
            M, 1, stride_row_scales, ck_tile::bool_constant<true>{}));
        ck_tile::HostTensor<QDataType> col_scales_n(ck_tile::host_tensor_descriptor(
            N, 1, stride_col_scales, ck_tile::bool_constant<true>{}));

        // Initialize data with random values
        ck_tile::FillUniformDistribution<ADataType>{-0.5f, 0.5f}(a_m_k);
        ck_tile::FillUniformDistribution<BDataType>{-0.5f, 0.5f}(b_k_n);
        ck_tile::FillUniformDistribution<QDataType>{0.001f, 0.01f}(row_scales_m);
        ck_tile::FillUniformDistribution<QDataType>{0.001f, 0.01f}(col_scales_n);

        // Allocate device memory
        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size() * sizeof(ADataType));
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size() * sizeof(BDataType));
        ck_tile::DeviceMem row_scales_dev_buf(row_scales_m.get_element_space_size() *
                                              sizeof(QDataType));
        ck_tile::DeviceMem col_scales_dev_buf(col_scales_n.get_element_space_size() *
                                              sizeof(QDataType));
        ck_tile::DeviceMem c_m_n_dev_buf(M * N * sizeof(CDataType));

        // Copy to device
        a_m_k_dev_buf.ToDevice(a_m_k.data());
        b_k_n_dev_buf.ToDevice(b_k_n.data());
        row_scales_dev_buf.ToDevice(row_scales_m.data());
        col_scales_dev_buf.ToDevice(col_scales_n.data());

        // Create args for kernel execution
        ck_tile::QuantGemmHostArgs args{
            a_m_k_dev_buf.GetDeviceBuffer(),      // a_ptr
            b_k_n_dev_buf.GetDeviceBuffer(),      // b_ptr
            c_m_n_dev_buf.GetDeviceBuffer(),      // c_ptr
            row_scales_dev_buf.GetDeviceBuffer(), // aq_ptr (row scales)
            col_scales_dev_buf.GetDeviceBuffer(), // bq_ptr (col scales)
            1,                                    // k_batch
            M,
            N,
            K, // M, N, K
            1, // QK_A (row scales)
            1, // QK_B (col scales)
            stride_A,
            stride_B,
            stride_C,
            stride_row_scales,
            stride_col_scales // strides
        };

        // Run the kernel
        ck_tile::stream_config stream_config{};
        this->invoke_quant_gemm(args, stream_config);

        // Validation using reference implementation
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, this->is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        // Run reference RowColQuant implementation
        ck_tile::reference_gemm_rowcol_quant<ADataType,
                                             QDataType,
                                             BDataType,
                                             QDataType,
                                             AccDataType,
                                             CDataType>(
            a_m_k, row_scales_m, b_k_n, col_scales_n, c_m_n_host_ref);

        // Get device result
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(
            ck_tile::host_tensor_descriptor(M, N, stride_C, this->is_row_major(CLayout{})));
        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.mData.data());

        // Calculate error tolerances
        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto rtol_atol =
            this->template calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
                K, 1, max_accumulated_value);

        // Validate results
        bool pass = ck_tile::check_err(c_m_n_dev_result,
                                       c_m_n_host_ref,
                                       "Error: Incorrect results!",
                                       rtol_atol.at(ck_tile::number<0>{}),
                                       rtol_atol.at(ck_tile::number<1>{}));

        EXPECT_TRUE(pass) << "RowColQuant validation failed with M=" << M << ", N=" << N
                          << ", K=" << K;

        if(!pass)
        {
            std::cout << "RowColQuant - Relative error threshold: "
                      << rtol_atol.at(ck_tile::number<0>{})
                      << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                      << std::endl;
        }
    }

    private:
    // RowColQuant-specific pipeline implementation
    template <typename CodegenGemmShape, typename TilePartitioner, typename CodegenGemmTraits>
    void run_quant_gemm_impl(const ck_tile::QuantGemmHostArgs& args,
                             const ck_tile::stream_config& s)
    {
        using GemmPipelineProblem = ck_tile::GemmPipelineProblemBase<ADataType,
                                                                     BDataType,
                                                                     AccDataType,
                                                                     CodegenGemmShape,
                                                                     CodegenGemmTraits,
                                                                     ComputeDataType>;

        using BaseGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmPipelineProblem>;

        const ck_tile::index_t K_split  = (args.K + Base::K_Tile - 1) / Base::K_Tile * Base::K_Tile;
        const ck_tile::index_t num_loop = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop         = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v  = tail_number_.value;
            constexpr bool transpose_c    = CodegenGemmTraits::TransposeC;

            using PipelineProblem = ck_tile::GemmRowColTensorQuantPipelineProblem<
                ADataType,
                BDataType,
                AccDataType,
                AccDataType,
                CodegenGemmShape,
                CodegenGemmTraits,
                transpose_c,
                ComputeDataType,
                ck_tile::GemmPipelineScheduler::Intrawave,
                has_hot_loop_v,
                tail_number_v>;

            using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem>;
            using GemmEpilogue = ck_tile::CShuffleEpilogue<
                ck_tile::CShuffleEpilogueProblem<ADataType,
                                                 BDataType,
                                                 ck_tile::tuple<>,
                                                 AccDataType,
                                                 CDataType,
                                                 ck_tile::tuple<>,
                                                 CLayout,
                                                 ck_tile::element_wise::PassThrough,
                                                 TilePartitioner::MPerBlock,
                                                 TilePartitioner::NPerBlock,
                                                 Base::M_Warp,
                                                 Base::N_Warp,
                                                 Base::M_Warp_Tile,
                                                 Base::N_Warp_Tile,
                                                 Base::K_Warp_Tile,
                                                 transpose_c>>;

            using Kernel = ck_tile::QuantGemmKernel<TilePartitioner,
                                                    GemmPipeline,
                                                    GemmEpilogue,
                                                    ck_tile::QuantType::RowColQuant>;

            auto kargs        = Kernel::MakeKernelArgs(args);
            const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
            const dim3 blocks = Kernel::BlockSize();

            if(!Kernel::IsSupportedArgument(kargs))
            {
                throw std::runtime_error("Arguments not supported for RowColQuant kernel");
            }

            ck_tile::launch_kernel(s,
                                   ck_tile::make_kernel<GemmConfigBase::kBlockPerCu>(
                                       Kernel{}, grids, blocks, 0, kargs));
        };

        return BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
    }
};

// TensorQuant-specific test fixture
template <typename Tuple>
class TestCkTileGemmTensorQuant
    : public TestCkTileGemmQuantBase<Tuple, TestCkTileGemmTensorQuant<Tuple>>
{
    using Base = TestCkTileGemmQuantBase<Tuple, TestCkTileGemmTensorQuant<Tuple>>;
    friend Base;

    public:
    using typename Base::AccDataType;
    using typename Base::ADataType;
    using typename Base::ALayout;
    using typename Base::BDataType;
    using typename Base::BLayout;
    using typename Base::CDataType;
    using typename Base::CLayout;
    using typename Base::ComputeDataType;
    using typename Base::QDataType;
    using typename Base::QuantGroupSize;

    static constexpr auto QuantType = Base::QuantType;

    protected:
    void SetUpQuantTypeSpecific() {}
    void TearDownQuantTypeSpecific() {}

    void run_test_with_validation(ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K)
    {
        const ck_tile::index_t stride_A = K;
        const ck_tile::index_t stride_B = K;
        const ck_tile::index_t stride_C = N;

        // TensorQuant uses single scalar scale for each tensor
        const ck_tile::index_t stride_scale_a = 1;
        const ck_tile::index_t stride_scale_b = 1;

        // Generate test data
        ck_tile::HostTensor<ADataType> a_m_k(
            ck_tile::host_tensor_descriptor(M, K, stride_A, this->is_row_major(ALayout{})));
        ck_tile::HostTensor<BDataType> b_k_n(
            ck_tile::host_tensor_descriptor(K, N, stride_B, this->is_row_major(BLayout{})));
        ck_tile::HostTensor<QDataType> scale_a(
            ck_tile::host_tensor_descriptor(1, 1, stride_scale_a, ck_tile::bool_constant<true>{}));
        ck_tile::HostTensor<QDataType> scale_b(
            ck_tile::host_tensor_descriptor(1, 1, stride_scale_b, ck_tile::bool_constant<true>{}));

        // Initialize data with random values
        ck_tile::FillUniformDistribution<ADataType>{-0.5f, 0.5f}(a_m_k);
        ck_tile::FillUniformDistribution<BDataType>{-0.5f, 0.5f}(b_k_n);
        ck_tile::FillUniformDistribution<QDataType>{0.001f, 0.01f}(scale_a);
        ck_tile::FillUniformDistribution<QDataType>{0.001f, 0.01f}(scale_b);

        // Allocate device memory
        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size() * sizeof(ADataType));
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size() * sizeof(BDataType));
        ck_tile::DeviceMem scale_a_dev_buf(scale_a.get_element_space_size() * sizeof(QDataType));
        ck_tile::DeviceMem scale_b_dev_buf(scale_b.get_element_space_size() * sizeof(QDataType));
        ck_tile::DeviceMem c_m_n_dev_buf(M * N * sizeof(CDataType));

        // Copy to device
        a_m_k_dev_buf.ToDevice(a_m_k.data());
        b_k_n_dev_buf.ToDevice(b_k_n.data());
        scale_a_dev_buf.ToDevice(scale_a.data());
        scale_b_dev_buf.ToDevice(scale_b.data());

        // Create args for kernel execution
        ck_tile::QuantGemmHostArgs args{
            a_m_k_dev_buf.GetDeviceBuffer(),   // a_ptr
            b_k_n_dev_buf.GetDeviceBuffer(),   // b_ptr
            c_m_n_dev_buf.GetDeviceBuffer(),   // c_ptr
            scale_a_dev_buf.GetDeviceBuffer(), // aq_ptr (scale A)
            scale_b_dev_buf.GetDeviceBuffer(), // bq_ptr (scale B)
            1,                                 // k_batch
            M,
            N,
            K, // M, N, K
            1, // QK_A (tensor scale)
            1, // QK_B (tensor scale)
            stride_A,
            stride_B,
            stride_C,
            stride_scale_a,
            stride_scale_b // strides
        };

        // Run the kernel
        ck_tile::stream_config stream_config{};
        this->invoke_quant_gemm(args, stream_config);

        // Validation using reference implementation
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, this->is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        // Run reference TensorQuant implementation
        ck_tile::reference_gemm_tensor_quant<ADataType,
                                             QDataType,
                                             BDataType,
                                             QDataType,
                                             AccDataType,
                                             CDataType>(
            a_m_k, scale_a, b_k_n, scale_b, c_m_n_host_ref);

        // Get device result
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(
            ck_tile::host_tensor_descriptor(M, N, stride_C, this->is_row_major(CLayout{})));
        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.mData.data());

        // Calculate error tolerances
        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto rtol_atol =
            this->template calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
                K, 1, max_accumulated_value);

        // Validate results
        bool pass = ck_tile::check_err(c_m_n_dev_result,
                                       c_m_n_host_ref,
                                       "Error: Incorrect results!",
                                       rtol_atol.at(ck_tile::number<0>{}),
                                       rtol_atol.at(ck_tile::number<1>{}));

        EXPECT_TRUE(pass) << "TensorQuant validation failed with M=" << M << ", N=" << N
                          << ", K=" << K;

        if(!pass)
        {
            std::cout << "TensorQuant - Relative error threshold: "
                      << rtol_atol.at(ck_tile::number<0>{})
                      << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                      << std::endl;
        }
    }

    private:
    // TensorQuant-specific pipeline implementation
    template <typename CodegenGemmShape, typename TilePartitioner, typename CodegenGemmTraits>
    void run_quant_gemm_impl(const ck_tile::QuantGemmHostArgs& args,
                             const ck_tile::stream_config& s)
    {
        using GemmPipelineProblem = ck_tile::GemmPipelineProblemBase<ADataType,
                                                                     BDataType,
                                                                     AccDataType,
                                                                     CodegenGemmShape,
                                                                     CodegenGemmTraits,
                                                                     ComputeDataType>;

        using BaseGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmPipelineProblem>;

        const ck_tile::index_t K_split  = (args.K + Base::K_Tile - 1) / Base::K_Tile * Base::K_Tile;
        const ck_tile::index_t num_loop = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop         = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v  = tail_number_.value;
            constexpr bool transpose_c    = CodegenGemmTraits::TransposeC;

            using PipelineProblem = ck_tile::GemmRowColTensorQuantPipelineProblem<
                ADataType,
                BDataType,
                AccDataType,
                AccDataType,
                CodegenGemmShape,
                CodegenGemmTraits,
                transpose_c,
                ComputeDataType,
                ck_tile::GemmPipelineScheduler::Intrawave,
                has_hot_loop_v,
                tail_number_v>;

            using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem>;
            using GemmEpilogue = ck_tile::CShuffleEpilogue<
                ck_tile::CShuffleEpilogueProblem<ADataType,
                                                 BDataType,
                                                 ck_tile::tuple<>,
                                                 AccDataType,
                                                 CDataType,
                                                 ck_tile::tuple<>,
                                                 CLayout,
                                                 ck_tile::element_wise::PassThrough,
                                                 TilePartitioner::MPerBlock,
                                                 TilePartitioner::NPerBlock,
                                                 Base::M_Warp,
                                                 Base::N_Warp,
                                                 Base::M_Warp_Tile,
                                                 Base::N_Warp_Tile,
                                                 Base::K_Warp_Tile,
                                                 transpose_c>>;

            using Kernel = ck_tile::QuantGemmKernel<TilePartitioner,
                                                    GemmPipeline,
                                                    GemmEpilogue,
                                                    ck_tile::QuantType::TensorQuant>;

            auto kargs        = Kernel::MakeKernelArgs(args);
            const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
            const dim3 blocks = Kernel::BlockSize();

            if(!Kernel::IsSupportedArgument(kargs))
            {
                throw std::runtime_error("Arguments not supported for TensorQuant kernel");
            }

            ck_tile::launch_kernel(s,
                                   ck_tile::make_kernel<GemmConfigBase::kBlockPerCu>(
                                       Kernel{}, grids, blocks, 0, kargs));
        };

        return BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
    }
};
