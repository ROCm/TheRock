// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/literals.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/ops/flatmm/kernel/flatmm_kernel.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp"
#include "ck_tile/host.hpp"

// #define disable_tile_gs

namespace ck_tile {

template <class ScaleM     = FlatmmScalePointer<-1>,
          class ScaleN     = FlatmmScalePointer<-1>,
          class ExpertBias = FlatmmScalePointer<-1>>
struct MoeFlatmmHostArgs : ScaleFlatmmHostArgs<ScaleM, ScaleN, 0>
{
    ck_tile::index_t NumTokens;
    ck_tile::index_t NumExperts;
    ck_tile::index_t TopK;
    const ck_tile::index_t* p_sorted_token_ids;
    const ck_tile::index_t* p_sorted_expert_ids;
    const ck_tile::index_t* p_max_token_id;
    const void* p_sorted_expert_weights;
    const ck_tile::index_t n_padded_zeros;
    const ck_tile::index_t k_padded_zeros;
    ExpertBias exp_bias;

    CK_TILE_HOST MoeFlatmmHostArgs() noexcept = default;

    CK_TILE_HOST MoeFlatmmHostArgs(const ck_tile::index_t* p_sorted_token_ids_,
                                   const void* p_sorted_expert_weights_,
                                   const ck_tile::index_t* p_sorted_expert_ids_,
                                   const ck_tile::index_t* p_max_token_id_,
                                   const void* a_ptr_,
                                   const void* b_ptr_,
                                   void* c_ptr_,
                                   ck_tile::index_t NumTokens_,
                                   ck_tile::index_t NumExperts_,
                                   ck_tile::index_t TopK_,
                                   ck_tile::index_t k_batch_,
                                   ck_tile::index_t M_,
                                   ck_tile::index_t N_,
                                   ck_tile::index_t K_,
                                   ck_tile::index_t stride_A_,
                                   ck_tile::index_t stride_B_,
                                   ck_tile::index_t stride_C_,
                                   ScaleM scale_m_      = {},
                                   ScaleN scale_n_      = {},
                                   ExpertBias exp_bias_ = {})
        : MoeFlatmmHostArgs(p_sorted_token_ids_,
                            p_sorted_expert_weights_,
                            p_sorted_expert_ids_,
                            p_max_token_id_,
                            a_ptr_,
                            b_ptr_,
                            c_ptr_,
                            NumTokens_,
                            NumExperts_,
                            TopK_,
                            k_batch_,
                            M_,
                            N_,
                            K_,
                            stride_A_,
                            stride_B_,
                            stride_C_,
                            0, // n_padded_zeros_
                            0, // k_padded_zeros_
                            scale_m_,
                            scale_n_,
                            exp_bias_)
    {
    }

    CK_TILE_HOST MoeFlatmmHostArgs(const ck_tile::index_t* p_sorted_token_ids_,
                                   const void* p_sorted_expert_weights_,
                                   const ck_tile::index_t* p_sorted_expert_ids_,
                                   const ck_tile::index_t* p_max_token_id_,
                                   const void* a_ptr_,
                                   const void* b_ptr_,
                                   void* c_ptr_,
                                   ck_tile::index_t NumTokens_,
                                   ck_tile::index_t NumExperts_,
                                   ck_tile::index_t TopK_,
                                   ck_tile::index_t k_batch_,
                                   ck_tile::index_t M_,
                                   ck_tile::index_t N_,
                                   ck_tile::index_t K_,
                                   ck_tile::index_t stride_A_,
                                   ck_tile::index_t stride_B_,
                                   ck_tile::index_t stride_C_,
                                   ck_tile::index_t n_padded_zeros_ = 0,
                                   ck_tile::index_t k_padded_zeros_ = 0,
                                   ScaleM scale_m_                  = {},
                                   ScaleN scale_n_                  = {},
                                   ExpertBias exp_bias_             = {})
        : ScaleFlatmmHostArgs<ScaleM, ScaleN, 0>(a_ptr_,
                                                 b_ptr_,
                                                 {}, // d_ptr_array
                                                 c_ptr_,
                                                 k_batch_,
                                                 M_,
                                                 N_,
                                                 K_,
                                                 stride_A_,
                                                 stride_B_,
                                                 {}, // d_stride_array
                                                 stride_C_,
                                                 scale_m_,
                                                 scale_n_),
          NumTokens(NumTokens_),
          NumExperts(NumExperts_),
          TopK(TopK_),
          p_sorted_token_ids(p_sorted_token_ids_),
          p_sorted_expert_ids(p_sorted_expert_ids_),
          p_max_token_id(p_max_token_id_),
          p_sorted_expert_weights(p_sorted_expert_weights_),
          n_padded_zeros(n_padded_zeros_),
          k_padded_zeros(k_padded_zeros_),
          exp_bias(exp_bias_)
    {
    }
};

enum class MoeFlatmmKind
{
    kFFN_gemm1_gate_only,
    kFFN_gemm1_gate_up,
    kFFN_gemm2,
    kFFN_gemm1_split_k,
};

namespace moe {

struct MoeSilu
{
    template <typename T>
    CK_TILE_HOST_DEVICE T operator()(T gate, T linear = 1) const
    {
        ck_tile::element_wise::Silu{}(gate, gate);
        return gate * linear;
    };
};

struct Swiglu
{
    const float alpha;
    const float limit;

    CK_TILE_HOST_DEVICE
    Swiglu(float alpha_ = 1.702f, float limit_ = 7.0f) // use value in gpt-oss as default
        : alpha(alpha_), limit(limit_)
    {
    }

    template <typename T>
    CK_TILE_HOST_DEVICE T operator()(T gate, T linear) const
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, ck_tile::fp16_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");

        constexpr T one = type_convert<T>(1);

        gate   = gate < limit ? gate : limit;
        linear = linear < limit ? (linear > -limit ? linear : -limit) : limit;

        if constexpr(std::is_same_v<T, float>)
        {
            return gate * __builtin_amdgcn_rcpf(one + ck_tile::exp(alpha * -gate)) * (linear + 1);
        }
        else
        {
            return gate * (one / (one + ck_tile::exp(alpha * -gate))) * (linear + 1);
        }
    }
};

} // namespace moe

template <typename TilePartitioner_,
          typename FlatmmPipeline_,
          typename EpiloguePipeline_,
          MoeFlatmmKind kind,
          typename FusedActivation = moe::MoeSilu>
struct MoeFlatmmKernel
{
    using TilePartitioner = remove_cvref_t<TilePartitioner_>;
    using FlatmmPipeline  = remove_cvref_t<FlatmmPipeline_>;
    using BlockGemmShape =
        remove_cvref_t<typename FlatmmPipeline::BlockGemmShape>; // TileFlatmmShape
    using EpiloguePipeline              = remove_cvref_t<EpiloguePipeline_>;
    using ALayout                       = remove_cvref_t<typename FlatmmPipeline::ALayout>;
    using BLayout                       = remove_cvref_t<typename FlatmmPipeline::BLayout>;
    using ELayout                       = remove_cvref_t<typename FlatmmPipeline::CLayout>;
    using DsLayout                      = remove_cvref_t<typename EpiloguePipeline::DsLayout>;
    using DsDataType                    = remove_cvref_t<typename EpiloguePipeline::DsDataType>;
    static constexpr index_t kBlockSize = FlatmmPipeline::BlockSize;
    static constexpr bool UsePersistentKernel = FlatmmPipeline::UsePersistentKernel;

    using ADataType = remove_cvref_t<typename FlatmmPipeline::ADataType>;
    using BDataType = remove_cvref_t<typename FlatmmPipeline::BDataType>;
    // Below type is actually accumulation data type - the output of block GEMM.
    using EDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    using AccDataType  = float;
    using ActivationOp = FusedActivation;

    static constexpr index_t NumDTensor = DsDataType::size();

    static constexpr auto I0 = number<0>();
    static constexpr auto I1 = number<1>();
    static constexpr auto I2 = number<2>();
    static constexpr auto I3 = number<3>();
    static constexpr auto I4 = number<4>();

    static_assert(DsLayout::size() == DsDataType::size(),
                  "The size of DsLayout and DsDataType should be the same");

    static constexpr bool IsInputGemm   = kind != MoeFlatmmKind::kFFN_gemm2;
    static constexpr bool IsGateUp      = kind == MoeFlatmmKind::kFFN_gemm1_gate_up;
    static constexpr bool IsGemm1SplitK = kind == MoeFlatmmKind::kFFN_gemm1_split_k;
    static constexpr bool IsBShuffled   = true;

    // static constexpr index_t kBlockSize     = EpiloguePipeline::kBlockSize;
    static constexpr index_t kMPerBlock     = EpiloguePipeline::kMPerBlock;
    static constexpr index_t kNPerBlock     = EpiloguePipeline::kNPerBlock;
    static constexpr index_t MWave          = EpiloguePipeline::MWave;
    static constexpr index_t NWave          = EpiloguePipeline::NWave;
    static constexpr index_t MPerXdl        = EpiloguePipeline::MPerXdl;
    static constexpr index_t NPerXdl        = EpiloguePipeline::NPerXdl;
    static constexpr index_t KPerXdl        = EpiloguePipeline::KPerXdl;
    static constexpr index_t isCTransposed  = EpiloguePipeline::isCTransposed;
    static constexpr index_t kMPerIteration = MPerXdl * MWave;
    static constexpr index_t kNPerIteration = NPerXdl * NWave;
    static constexpr index_t kNRepeat       = kNPerBlock / kNPerIteration;

    static constexpr int OutputNPerBlock =
        IsGateUp ? TilePartitioner::NPerBlock / 2 : TilePartitioner::NPerBlock;

    // MXF4_Pipeline only has the of scale B and granularityK is 32
    static constexpr bool AQUANT_Pipeline = std::is_same_v<ADataType, bf8_t> ||
                                            std::is_same_v<ADataType, fp8_t> ||
                                            std::is_same_v<ADataType, pk_fp4_t>;
    static constexpr bool BMXFP4_Pipeline = std::is_same_v<BDataType, pk_fp4_t>;

    static constexpr bool MXF8F6F4MFMA =
#ifdef __gfx950__
        AQUANT_Pipeline && BMXFP4_Pipeline;
#else
        false;
#endif
    static constexpr int MXFP4M_Pack = 2;
    static constexpr int MXFP4N_Pack = 2;
    static constexpr int MXFP4K_Pack = 2;

    static constexpr int M_Pack = AQUANT_Pipeline ? MXFP4M_Pack : 1;
    static constexpr int N_Pack = BMXFP4_Pipeline ? MXFP4N_Pack : 1;
    static constexpr int K_Pack = BMXFP4_Pipeline ? MXFP4K_Pack : 1;

    static constexpr int WeightPackedSize = numeric_traits<BDataType>::PackedSize;

    template <class ScaleM     = FlatmmScalePointer<-1>,
              class ScaleN     = FlatmmScalePointer<-1>,
              class ExpertBias = FlatmmScalePointer<-1>>
    struct MoeFlatmmKernelArgs
    {
        const ck_tile::index_t* p_sorted_token_ids;
        const ck_tile::index_t* p_sorted_expert_ids;
        const ck_tile::index_t* p_max_token_id;
        const void* p_sorted_expert_weights;
        const void* a_ptr;
        const void* b_ptr;
        void* e_ptr;
        ck_tile::index_t NumTokens;
        ck_tile::index_t TopK;
        ck_tile::index_t M;
        ck_tile::index_t N;
        ck_tile::index_t K;
        ck_tile::index_t stride_A;
        ck_tile::index_t stride_B;
        ck_tile::index_t stride_C;
        ck_tile::index_t k_batch;
        ck_tile::index_t n_padded_zeros;
        ck_tile::index_t k_padded_zeros;
        ScaleM scale_m;
        ScaleN scale_n;
        ExpertBias exp_bias;
    };

    template <class ScaleM     = FlatmmScalePointer<-1>,
              class ScaleN     = FlatmmScalePointer<-1>,
              class ExpertBias = FlatmmScalePointer<-1>>
    CK_TILE_HOST static constexpr auto
    MakeKernelArgs(const MoeFlatmmHostArgs<ScaleM, ScaleN, ExpertBias>& hostArgs)
    {
        return MoeFlatmmKernelArgs<ScaleM, ScaleN, ExpertBias>{hostArgs.p_sorted_token_ids,
                                                               hostArgs.p_sorted_expert_ids,
                                                               hostArgs.p_max_token_id,
                                                               hostArgs.p_sorted_expert_weights,
                                                               hostArgs.a_ptr,
                                                               hostArgs.b_ptr,
                                                               hostArgs.e_ptr,
                                                               hostArgs.NumTokens,
                                                               hostArgs.TopK,
                                                               hostArgs.M,
                                                               hostArgs.N,
                                                               hostArgs.K,
                                                               hostArgs.stride_A,
                                                               hostArgs.stride_B,
                                                               hostArgs.stride_C,
                                                               hostArgs.k_batch,
                                                               hostArgs.n_padded_zeros,
                                                               hostArgs.k_padded_zeros,
                                                               hostArgs.scale_m,
                                                               hostArgs.scale_n,
                                                               hostArgs.exp_bias};
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        return concat(
            '_', "moe_flatmm", gemm_prec_str<ADataType, BDataType>, FlatmmPipeline::GetName());
    }

    static constexpr auto BlockSize() -> dim3 { return dim3(kBlockSize); }

    static constexpr auto GridSize(index_t M, index_t N, index_t KBatch)
    {
        return dim3(TilePartitioner::GridSize(M, N), 1, KBatch);
    }
    template <class MoeFlatmmKernelArgs>
    static constexpr auto GridSize(const MoeFlatmmKernelArgs& kargs)
    {
        if constexpr(UsePersistentKernel)
        {
            hipDeviceProp_t prop;
            int deviceId = 0; // default device

            constexpr int block_size = MoeFlatmmKernel::BlockSize().x;
            int dync_smem_size       = 0;
            int maxActiveBlocksPerCU = 0;

            [[maybe_unused]] auto e = hipGetDeviceProperties(&prop, deviceId);

            e = hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &maxActiveBlocksPerCU,
                reinterpret_cast<void*>(kentry<1, MoeFlatmmKernel, MoeFlatmmKernelArgs>),
                block_size,
                dync_smem_size);

            const int persistent_block_size = prop.multiProcessorCount * maxActiveBlocksPerCU;
            const int total_work_tile_cnt   = TilePartitioner::GridSize(kargs.M, kargs.N);

            // std::cout << "maxActiveBlocksPerCU: " << maxActiveBlocksPerCU
            //           << ", persistent_block_size: " << persistent_block_size
            //           << ", total_work_tile_cnt: " << total_work_tile_cnt << std::endl;

            assert(kargs.k_batch == 1);
            return dim3(min(persistent_block_size, total_work_tile_cnt), 1, kargs.k_batch);
        }
        else
        {
            return dim3(TilePartitioner::GridSize(kargs.M, kargs.N), 1, kargs.k_batch);
        }
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemPingSize()
    {
        return max(FlatmmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemPongSize()
    {
        return FlatmmPipeline::GetSmemSize();
    }

    struct SplitKBatchOffset
    {
        template <class KernelArgs>
        __device__ SplitKBatchOffset(const KernelArgs& kargs, const std::size_t k_id = blockIdx.z)
        {
            constexpr auto K1   = BlockGemmShape::WarpTile::at(number<2>{});
            const index_t K_t   = kargs.k_batch * K1;
            const index_t KRead = (kargs.K + K_t - 1) / K_t * K1;

            if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                a_k_split_offset = k_id * KRead;
            }
            else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                a_k_split_offset = k_id * KRead * kargs.stride_A;
            }

            if(k_id < static_cast<uint32_t>(kargs.k_batch - 1))
            {
                splitted_k = KRead;
            }
            else
            {
                splitted_k = kargs.K - KRead * (kargs.k_batch - 1);
            }

            if constexpr(IsBShuffled)
            {
                b_k_split_offset = k_id * splitted_k * NPerXdl;
            }
            else
            {
                if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, BLayout>)
                {
                    b_k_split_offset = k_id * KRead * kargs.stride_B;
                }
                else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, BLayout>)
                {
                    b_k_split_offset = k_id * KRead;
                }
            }
        }

        index_t a_k_split_offset;
        index_t b_k_split_offset;
        index_t splitted_k;
    };

    template <typename KernelArgs>
    CK_TILE_HOST static bool IsSupportedArgument(const KernelArgs& kargs)
    {
        if constexpr(EpiloguePipeline::GetVectorSizeC() % 2 != 0 &&
                     is_any_of<EDataType, fp16_t, bf16_t>::value)
        {
            if(kargs.k_batch != 1)
            {
                std::cerr << "Conditions not met for Kbatch >1 !" << std::endl;
                return false;
            }
        }
        if constexpr(UsePersistentKernel)
        {
            if(kargs.k_batch != 1)
            {
                std::cerr << "Persistent mode doesn't support Kbatch >1 !" << std::endl;
                return false;
            }
        }

        if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.K % TilePartitioner::KPerBlock != 0 && FlatmmPipeline::kPadK == false)
            {
                std::cerr << "Can't support K that is not a multiple of KPerBlock"
                             " without padding!"
                          << std::endl;
                return false;
            }
            if(kargs.K % FlatmmPipeline::GetVectorSizeA() != 0)
            {
                std::cerr << "K is not a multiple of vector load size for A tensor!" << std::endl;
                return false;
            }
        }
        else
        {
            if(kargs.M % TilePartitioner::MPerBlock != 0 && FlatmmPipeline::kPadM == false)
            {
                std::cerr << "Can't support M that is not a multiple of MPerBlock"
                             " without padding!"
                          << std::endl;
                return false;
            }
            if(kargs.M % FlatmmPipeline::GetVectorSizeA() != 0)
            {
                std::cerr << "M is not a multiple of vector load size for A tensor!" << std::endl;
                return false;
            }
        }

        if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
        {
            // if(kargs.N % TilePartitioner::NPerBlock != 0 && FlatmmPipeline::kPadN == false)
            // {
            //     std::cerr << "Can't support N that is not a multiple of NPerBlock"
            //                  " without padding!"
            //               << std::endl;
            //     return false;
            // }
            if(kargs.N % FlatmmPipeline::GetVectorSizeB() != 0)
            {
                std::cerr << "N is not a multiple of vector load size for B tensor!" << std::endl;
                return false;
            }
        }
        else
        {
            if(kargs.K % TilePartitioner::KPerBlock != 0 && FlatmmPipeline::kPadK == false)
            {
                std::cerr << "Can't support K that is not a multiple of KPerBlock"
                             " without padding!"
                          << std::endl;
                return false;
            }
            if(kargs.K % FlatmmPipeline::GetVectorSizeB() != 0)
            {
                std::cerr << "K is not a multiple of vector load size for B tensor!" << std::endl;
                return false;
            }
        }

        bool DTesnorIsValid = {true};
        static_for<0, NumDTensor, 1>{}([&](auto index) {
            using DiLayout = remove_cvref_t<std::tuple_element_t<index.value, DsLayout>>;
            if(std::is_same_v<DiLayout, ELayout> == false)
            {
                DTesnorIsValid = false;
            }
            if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
            {
                if(kargs.N % TilePartitioner::NPerBlock != 0 && FlatmmPipeline::kPadN == false)
                {
                    CK_TILE_ERROR("Can't support N for tensor D that is not a multiple of "
                                  "NPerBlock without padding!");
                    DTesnorIsValid = false;
                }
                if(kargs.N % EpiloguePipeline::GetVectorSizeD(index) != 0)
                {
                    CK_TILE_ERROR("N is not a multiple of vector load size for D tensor!");
                    DTesnorIsValid = false;
                }
            }
            else
            {
                if(kargs.M % TilePartitioner::MPerBlock != 0 && FlatmmPipeline::kPadM == false)
                {
                    CK_TILE_ERROR("Can't support M for tensor D that is not a multiple of "
                                  "MPerBlock without padding!");

                    DTesnorIsValid = false;
                }
                if(kargs.M % EpiloguePipeline::GetVectorSizeD(index) != 0)
                {
                    CK_TILE_ERROR("M is not a multiple of vector load size for D tensor!");
                    DTesnorIsValid = false;
                }
            }
        });

        if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.stride_C % TilePartitioner::NPerBlock != 0 && FlatmmPipeline::kPadN == false)
            {
                std::cerr << "Can't support N that is not a multiple of NPerBlock"
                             " without padding!"
                          << std::endl;
                return false;
            }
            if(kargs.N % EpiloguePipeline::GetVectorSizeC() != 0)
            {
                std::cerr << "N is not a multiple of vector load size for C tensor!" << std::endl;
                return false;
            }
        }
        else
        {
            if(kargs.M % TilePartitioner::MPerBlock != 0 && FlatmmPipeline::kPadM == false)
            {
                std::cerr << "Can't support M that is not a multiple of MPerBlock"
                             " without padding!"
                          << std::endl;
                return false;
            }
            if(kargs.M % EpiloguePipeline::GetVectorSizeC() != 0)
            {
                std::cerr << "M is not a multiple of vector load size for C tensor!" << std::endl;
                return false;
            }
        }
        return DTesnorIsValid;
    }

    template <memory_operation_enum DstInMemOp = (IsInputGemm && !IsGemm1SplitK)
                                                     ? memory_operation_enum::set
                                                     : memory_operation_enum::atomic_add,
              typename KernelArgs>
    CK_TILE_DEVICE static auto
    MakeGemmTensorViews(const ADataType* a_ptr,
                        const BDataType* b_flat_ptr,
                        EDataType* e_ptr,
                        [[maybe_unused]] const AccDataType* exp_weight_ptr,
                        [[maybe_unused]] const int expert_id,
                        const KernelArgs& kargs,
                        const SplitKBatchOffset& splitk_batch_offset)
    {
        const auto& a_tensor_view = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_ptr,
                    make_tuple(IsInputGemm ? kargs.NumTokens : kargs.NumTokens * kargs.TopK,
                               splitk_batch_offset.splitted_k),
                    make_tuple(kargs.stride_A, 1),
                    number<FlatmmPipeline::GetVectorSizeA()>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_ptr,
                    make_tuple(splitk_batch_offset.splitted_k,
                               IsInputGemm ? kargs.NumTokens : kargs.NumTokens * kargs.TopK),
                    make_tuple(kargs.stride_A, 1),
                    number<FlatmmPipeline::GetVectorSizeA()>{},
                    number<1>{});
            }
        }();

        const auto& b_flat_tensor_view = [&]() {
            if constexpr(!FlatmmPipeline::BPreShufflePermute)
            {
                index_t kFlatK =
                    kargs.K * BlockGemmShape::WarpTile::at(I1); // TODO (support splitK)
                index_t kFlatN = kargs.N * kargs.K / kFlatK;

                return make_naive_tensor_view<address_space_enum::global,
                                              memory_operation_enum::set,
                                              FlatmmPipeline::BMemNTType>(
                    b_flat_ptr,
                    make_tuple(kFlatN - kargs.n_padded_zeros / NPerXdl, kFlatK),
                    make_tuple(kFlatK, 1),
                    number<FlatmmPipeline::GetVectorSizeB()>{},
                    number<1>{});
            }
            else
            {
                index_t kFlatK  = FlatmmPipeline::flatKPerWarp;
                index_t kFlatN0 = (kargs.N >> 4);
                index_t kFlatK0 = (kargs.K >> 7);

                auto b_tensor_view_naive = make_naive_tensor_view<address_space_enum::global,
                                                                  memory_operation_enum::set,
                                                                  FlatmmPipeline::BMemNTType>(
                    b_flat_ptr,
                    make_tuple(kFlatK0, kFlatN0 - kargs.n_padded_zeros / NPerXdl, kFlatK),
                    make_tuple(kFlatK * (kFlatN0 - kargs.n_padded_zeros / NPerXdl), kFlatK, 1),
                    number<FlatmmPipeline::GetVectorSizeB()>{},
                    number<1>{});
                return transform_tensor_view(
                    b_tensor_view_naive,
                    make_tuple(
                        make_pass_through_transform(kFlatN0 - kargs.n_padded_zeros / NPerXdl),
                        make_merge_transform_v3_division_mod(make_tuple(kFlatK0, kFlatK))),
                    make_tuple(sequence<1>{}, sequence<0, 2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
        }();

        // TODO: enable vector write for C in ColMajor
        const auto& c_tensor_view = [&]() {
            if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    e_ptr,
                    make_tuple(IsInputGemm ? kargs.NumTokens * kargs.TopK : kargs.NumTokens,
                               IsGateUp ? kargs.N / 2 : kargs.N),
                    make_tuple(kargs.stride_C, 1),
                    number<EpiloguePipeline::GetVectorSizeC()>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    e_ptr,
                    make_tuple(IsInputGemm ? kargs.NumTokens * kargs.TopK : kargs.NumTokens,
                               IsGateUp ? kargs.N / 2 : kargs.N),
                    make_tuple(1, kargs.stride_C),
                    number<1>{},
                    number<1>{});
            }
        }();

        const auto& scale_a_tensor_view = [&]() {
            auto scale_m_desc = kargs.scale_m;
            if constexpr(AQUANT_Pipeline)
            {
                constexpr int AGranularityK = decltype(scale_m_desc)::GranularityK == 0
                                                  ? 1
                                                  : decltype(scale_m_desc)::GranularityK;

                constexpr int MThreadPerXdl = BlockGemmShape::WarpTile::at(I0);
                constexpr int KThreadPerXdl = 64 / BlockGemmShape::WarpTile::at(I0);
                index_t scale_m_packs       = kargs.M / (MXFP4M_Pack * MThreadPerXdl);
                index_t scale_k_packs = kargs.K / (MXFP4K_Pack * AGranularityK * KThreadPerXdl);
                // Pack 2x2 e8m0 over M/K dimension into 1 int32_t to trigger dword width load
                const auto scale_a_naive_desc = make_naive_tensor_descriptor_packed(
                    make_tuple(scale_m_packs, scale_k_packs, KThreadPerXdl, MThreadPerXdl));
                const auto scale_a_desc = transform_tensor_descriptor(
                    scale_a_naive_desc,
                    make_tuple(make_merge_transform(make_tuple(scale_m_packs, MThreadPerXdl)),
                               make_merge_transform(make_tuple(scale_k_packs, KThreadPerXdl))),
                    make_tuple(sequence<0, 3>{}, sequence<1, 2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
                return make_tensor_view<address_space_enum::global>(
                    reinterpret_cast<const int32_t*>(scale_m_desc.ptr), scale_a_desc);
            }
            else
            {
                constexpr int AGranularityK = 32;
                constexpr int MThreadPerXdl = BlockGemmShape::WarpTile::at(I0);
                constexpr int KThreadPerXdl = 64 / BlockGemmShape::WarpTile::at(I0);
                index_t scale_m_packs       = kargs.M / (MXFP4M_Pack * MThreadPerXdl);
                index_t scale_k_packs = kargs.K / (MXFP4K_Pack * AGranularityK * KThreadPerXdl);
                return make_naive_tensor_view<address_space_enum::global>(
                    reinterpret_cast<const int32_t*>(scale_m_desc.ptr),
                    make_tuple(scale_m_packs * MThreadPerXdl, scale_k_packs * KThreadPerXdl),
                    make_tuple(scale_k_packs * KThreadPerXdl, 1),
                    number<8>{},
                    number<1>{});
            }
        }();

        const auto scale_b_flat_view = [&]() {
            auto scale_n = kargs.scale_n;
            constexpr int BGranularityK =
                decltype(scale_n)::GranularityK == 0 ? 1 : decltype(scale_n)::GranularityK;
            if constexpr(AQUANT_Pipeline)
            {
                index_t scale_k =
                    BGranularityK == 0 ? 1 : (kargs.K + BGranularityK - 1) / BGranularityK;
                constexpr int NThreadPerXdl = BlockGemmShape::WarpTile::at(I1);
                constexpr int KThreadPerXdl = 64 / BlockGemmShape::WarpTile::at(I1);
                index_t scale_n_packs       = kargs.N / (MXFP4N_Pack * NThreadPerXdl);
                index_t scale_k_packs = kargs.K / (MXFP4K_Pack * BGranularityK * KThreadPerXdl);
                const auto scale_b_navie_desc = make_naive_tensor_descriptor_packed(
                    make_tuple(scale_n_packs, scale_k_packs, KThreadPerXdl, NThreadPerXdl));
                const auto scale_b_desc = transform_tensor_descriptor(
                    scale_b_navie_desc,
                    make_tuple(make_merge_transform(make_tuple(scale_n_packs, NThreadPerXdl)),
                               make_merge_transform(make_tuple(scale_k_packs, KThreadPerXdl))),
                    make_tuple(sequence<0, 3>{}, sequence<1, 2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));

                return make_tensor_view<address_space_enum::global>(
                    reinterpret_cast<const int32_t*>(scale_n.ptr) +
                        expert_id * kargs.N * scale_k / 4,
                    scale_b_desc);
            }
            else
            {
                index_t scale_k =
                    BGranularityK == 0 ? 1 : (kargs.K + BGranularityK - 1) / BGranularityK;
                const auto scale_k_offset =
                    (splitk_batch_offset.b_k_split_offset / BGranularityK) * K_Pack;
                index_t FlatScaleK = scale_k * N_Pack * BlockGemmShape::WarpTile::at(I1);
                index_t FlatScaleN = kargs.N / N_Pack / BlockGemmShape::WarpTile::at(I1);

                return make_naive_tensor_view<address_space_enum::global>(
                    scale_n.ptr + expert_id * kargs.N * scale_k + scale_k_offset,
                    make_tuple(FlatScaleN - kargs.n_padded_zeros / NPerXdl / N_Pack, FlatScaleK),
                    make_tuple(FlatScaleK, 1),
                    number<8>{},
                    number<1>{});
            }
        }();

        return make_tuple(a_tensor_view,
                          b_flat_tensor_view,
                          c_tensor_view,
                          scale_a_tensor_view,
                          scale_b_flat_view);
    }

    template <typename TensorView>
    CK_TILE_DEVICE static auto MakeGemmPadViews(const TensorView& views)
    {
        const auto& a_pad_view = [&]() {
            const auto& a_tensor_view = views.at(I0);
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(a_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::KPerBlock>{}),
                                       sequence<false, FlatmmPipeline::kPadK>{});
            }
            else
            {
                return pad_tensor_view(a_tensor_view,
                                       make_tuple(number<TilePartitioner::KPerBlock>{},
                                                  number<TilePartitioner::MPerBlock>{}),
                                       sequence<false, FlatmmPipeline::kPadM>{});
            }
        }();

        // TODO vector write in for C in ColMajor
        const auto& c_pad_view = [&]() {
            const auto& c_tensor_view = views.at(I2);
            if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(
                    c_tensor_view,
                    make_tuple(number<TilePartitioner::MPerBlock>{}, number<OutputNPerBlock>{}),
                    sequence<false, FlatmmPipeline::kPadN>{});
            }
            else
            {
                return pad_tensor_view(
                    c_tensor_view,
                    make_tuple(number<TilePartitioner::MPerBlock>{}, number<OutputNPerBlock>{}),
                    sequence<FlatmmPipeline::kPadM, false>{});
            }
        }();

        return make_tuple(a_pad_view, views.at(I1), c_pad_view, views.at(I3), views.at(I4));
    }

    template <typename PadView>
    CK_TILE_DEVICE static auto MakeGemmTileWindows(const PadView& views,
                                                   [[maybe_unused]] const index_t coord_m,
                                                   const index_t coord_n)
    {
        const auto& a_pad_view      = views.at(number<0>{});
        const auto& b_flat_pad_view = views.at(number<1>{});
        const auto& c_pad_view      = views.at(number<2>{});

        const auto& a_block_window = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_tile_window(a_pad_view,
                                        make_tuple(number<TilePartitioner::MPerBlock>{},
                                                   number<TilePartitioner::KPerBlock>{}),
                                        {coord_m, 0}); // NOTE!
            }
            else
            {
                return make_tile_window(a_pad_view,
                                        make_tuple(number<TilePartitioner::KPerBlock>{},
                                                   number<TilePartitioner::MPerBlock>{}),
                                        {0, 0}); // NOTE!
            }
        }();

        constexpr bool isNonInterleaveGateUp = !IsGateUp || BMXFP4_Pipeline;

        const auto& b_flat_block_window =
            make_tile_window(b_flat_pad_view,
                             make_tuple(number<FlatmmPipeline::flatNPerWarp>{},
                                        number<FlatmmPipeline::flatKPerWarp>{}),
                             {static_cast<int>(coord_n / BlockGemmShape::WarpTile::at(I1) /
                                               (isNonInterleaveGateUp ? 1 : 2)),
                              0});

        const int output_N_offset = IsGateUp ? coord_n / 2 : coord_n;

        auto c_block_window = make_tile_window(
            c_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<OutputNPerBlock>{}),
            {0, // offset_m is included when construct C-scatter-window offsets
             output_N_offset});

        constexpr int GranularityK = 32; // fixed config for MXF4_Pipeline
        auto a_scale_block_window  = make_tile_window(
            views.at(I3),
            make_tuple(number<TilePartitioner::MPerBlock / M_Pack>{},
                       number<TilePartitioner::KPerBlock / (GranularityK * K_Pack)>{}),
            {coord_m / M_Pack, 0});

        constexpr int XDLPerLoadScaleB =
            BMXFP4_Pipeline ? 4 : 1; // GranularityK32 / XDL16x16x32_K8 = 4

        auto b_scale_block_window = [&]() {
            if constexpr(MXF8F6F4MFMA)
            {
                return make_tile_window(
                    views.at(I4),
                    make_tuple(number<TilePartitioner::NPerBlock / N_Pack>{},
                               number<TilePartitioner::KPerBlock / (GranularityK * K_Pack)>{}),
                    {coord_n / N_Pack, 0});
            }
            else
            {
                return make_tile_window(
                    views.at(I4),
                    make_tuple(number<FlatmmPipeline::flatNPerWarp>{},
                               number<FlatmmPipeline::flatKPerWarp * N_Pack * K_Pack *
                                      XDLPerLoadScaleB / GranularityK>{}),
                    {coord_n / BlockGemmShape::WarpTile::at(I1) / N_Pack, 0});
            }
        }();

        return make_tuple(a_block_window,
                          b_flat_block_window,
                          c_block_window,
                          a_scale_block_window,
                          b_scale_block_window);
    }

    template <class MoeFlatmmKernelArgs>
    CK_TILE_DEVICE void operator()(MoeFlatmmKernelArgs kargs) const
    {
        int partition_idx       = blockIdx.x;
        int total_work_tile_cnt = TilePartitioner::GridSize(kargs.M, kargs.N);
        do
        {
            const auto [block_offset_m, block_offset_n] =
                TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(partition_idx);

            this->operator()(kargs, block_offset_m, block_offset_n);
            partition_idx += gridDim.x;
        } while(UsePersistentKernel && partition_idx < total_work_tile_cnt);
    }

    template <class MoeFlatmmKernelArgs>
    CK_TILE_DEVICE void operator()(MoeFlatmmKernelArgs kargs, index_t iM, index_t iN) const
    {

        // const auto [iM, iN]   = TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(blockIdx.x);
        const index_t coord_m = __builtin_amdgcn_readfirstlane(iM * TilePartitioner::MPerBlock);
        const index_t coord_n = __builtin_amdgcn_readfirstlane(iN * TilePartitioner::NPerBlock);
        const index_t max_token_id = kargs.p_max_token_id[0];
        // allocate LDS
        __shared__ char smem_ptr_ping[GetSmemPingSize()];
        __shared__ char smem_ptr_pong[GetSmemPongSize()];

        const index_t expert_id = kargs.p_sorted_expert_ids[iM];

        constexpr auto a_dram_dist = FlatmmPipeline::GetADramTileDistribution();
        const auto a_coord = a_dram_dist.calculate_index(); // 2d thread offset, [i_row, i_col]

        constexpr ck_tile::index_t DramMRepeat =
            decltype(a_dram_dist)::DstrEncode::hs_lengthss_[number<0>{}][number<0>{}];
        statically_indexed_array<ck_tile::index_t, DramMRepeat> a_offsets;

        constexpr index_t token_id_offset = 24;
        constexpr index_t token_id_mask   = (1 << token_id_offset) - 1;

        auto row_to_token_idx = [&](auto row_idx) {
            const index_t fused_token =
                kargs.p_sorted_token_ids[row_idx]; // topk-idx[31:24] + token_idx[23:0]
            index_t gather_token_id = fused_token & token_id_mask;
            if constexpr(!IsInputGemm)
            {
                gather_token_id = gather_token_id * kargs.TopK + (fused_token >> token_id_offset);
            }
            return gather_token_id;
        };

        if(coord_m >= max_token_id)
            return;
        static_for<0, DramMRepeat, 1>{}([&](auto m0) {
            const auto row_idx =
                coord_m + m0 * (TilePartitioner::MPerBlock / DramMRepeat) + a_coord[I0];
            index_t gather_token_id = row_to_token_idx(row_idx);
            a_offsets[m0]           = std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>
                                          ? gather_token_id * kargs.stride_A
                                          : gather_token_id;
        });

        const SplitKBatchOffset splitk_batch_offset(kargs);
        const long_index_t expert_stride =
            __builtin_amdgcn_readfirstlane(long_index_t(kargs.N) * kargs.K);

        const ADataType* a_ptr =
            static_cast<const ADataType*>(kargs.a_ptr) + splitk_batch_offset.a_k_split_offset;
        const BDataType* b_flat_ptr =
            static_cast<const BDataType*>(kargs.b_ptr) +
            (splitk_batch_offset.b_k_split_offset + expert_stride * expert_id) / WeightPackedSize;
        EDataType* e_ptr = static_cast<EDataType*>(kargs.e_ptr);

        const AccDataType* exp_weight_ptr =
            static_cast<const AccDataType*>(kargs.p_sorted_expert_weights);

        const auto& gemm_tensor_views_tuple = MakeGemmTensorViews(
            a_ptr, b_flat_ptr, e_ptr, exp_weight_ptr, expert_id, kargs, splitk_batch_offset);
        const auto& gemm_pad_views = MakeGemmPadViews(gemm_tensor_views_tuple);

        auto gemm_tile_windows = MakeGemmTileWindows(gemm_pad_views, coord_m, coord_n);

        const index_t num_loop = TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k);

        // Run GEMM cooperatively by whole workgroup.
        const auto& a_block_window       = gemm_tile_windows.at(I0);
        const auto& b_block_window       = gemm_tile_windows.at(I1);
        const auto& a_scale_block_window = gemm_tile_windows.at(I3);
        const auto& b_scale_block_window = gemm_tile_windows.at(I4);

        auto a_gather_block_tile =
            ck_tile::make_tile_scatter_gather(a_block_window.get_bottom_tensor_view(),
                                              a_block_window.get_window_lengths(),
                                              a_block_window.get_window_origin(),
                                              a_dram_dist,
                                              a_offsets); // K DRAM tile window for

        auto c_block_tile = [&] {
            if constexpr(BMXFP4_Pipeline)
            {
                // BMXFP4_Pipeline uses gate-up interleave 16 layout for weight
                // so don't need extra processing
                if constexpr(AQUANT_Pipeline)
                {
                    return FlatmmPipeline{}(
                        a_gather_block_tile,
                        b_block_window,
                        a_scale_block_window, // weight scale with granularityK = 32
                        b_scale_block_window, // weight scale with granularityK = 32
                        num_loop,
                        smem_ptr_ping,
                        smem_ptr_pong);
                }
                else
                {
                    return FlatmmPipeline{}(
                        a_gather_block_tile,
                        b_block_window,
                        b_scale_block_window, // weight scale with granularityK = 32
                        num_loop,
                        kargs.k_padded_zeros,
                        smem_ptr_ping,
                        smem_ptr_pong);
                }
            }
            else
            {
                return FlatmmPipeline{}(a_gather_block_tile,
                                        b_block_window,
                                        number<IsGateUp>{},
                                        num_loop,
                                        smem_ptr_ping,
                                        smem_ptr_pong);
            }
        }();

        auto& c_block_window = gemm_tile_windows.at(number<2>{});

        // Run EpiloguePipeline
        {
            using EpiProblem = typename EpiloguePipeline::Problem;
            using ODataType  = typename EpiloguePipeline::ODataType;
            using CWarpDstr  = typename EpiloguePipeline::CWarpDstr;

            constexpr index_t NumMXdlPerWavePerShuffle = EpiloguePipeline::NumMXdlPerWavePerShuffle;
            constexpr index_t NumNXdlPerWavePerShuffle = EpiloguePipeline::NumNXdlPerWavePerShuffle;
            constexpr index_t MPerIterationShuffle     = EpiloguePipeline::MPerIterationShuffle;
            constexpr index_t NPerIterationShuffle     = EpiloguePipeline::NPerIterationShuffle;

            constexpr index_t MRepeat       = EpiloguePipeline::MRepeat;
            constexpr index_t NRepeat       = EpiloguePipeline::NRepeat;
            constexpr index_t OutputNRepeat = IsGateUp ? NRepeat / 2 : NRepeat;

            [[maybe_unused]] constexpr index_t EpiVectorSizeC = EpiloguePipeline::GetVectorSizeC();
            [[maybe_unused]] constexpr index_t BlockedXDLN_PerWarp =
                EpiloguePipeline::BlockedXDLN_PerWarp;

            static_assert(!IsGateUp || NumNXdlPerWavePerShuffle % 2 == 0);

            constexpr index_t OutputNumNXdlPerWavePerShuffle =
                IsGateUp ? NumNXdlPerWavePerShuffle / 2 : NumNXdlPerWavePerShuffle;
            constexpr index_t LDS_NPerIterationShuffle =
                IsGateUp ? NPerIterationShuffle / 2 : NPerIterationShuffle;

            constexpr auto lds_block_desc = make_naive_tensor_descriptor(
                make_tuple(number<MPerIterationShuffle>{}, number<LDS_NPerIterationShuffle>{}),
                make_tuple(number<LDS_NPerIterationShuffle>{}, number<1>{}));

            // EpiloguePipeline::template MakeLdsBlockDescriptor<EpiProblem>();
            auto o_lds_block = make_tensor_view<address_space_enum::lds>(
                reinterpret_cast<ODataType*>(smem_ptr_ping), lds_block_desc);

            constexpr int ScaleGranularityM = decltype(kargs.scale_m)::GranularityMN;
            constexpr int ScaleGranularityN = decltype(kargs.scale_n)::GranularityMN;

            constexpr index_t scale_stride_m = ScaleGranularityM == 0 ? 0  // per-tensor scale
                                                                      : 1; // per-token scale
            constexpr index_t scale_stride_n = ScaleGranularityN == 0 ? 0  // per-tensor scale
                                                                      : 1; // per-channel scale

            auto output_acc_tile_distr =
                make_static_tile_distribution(detail::make_embed_tile_distribution_encoding(
                    tile_distribution_encoding<
                        sequence<>,
                        tuple<sequence<MRepeat, MWave>, sequence<OutputNRepeat, NWave>>,
                        tuple<sequence<1, 2>>,
                        tuple<sequence<1, 1>>,
                        sequence<1, 2>,
                        sequence<0, 0>>{},
                    typename CWarpDstr::DstrEncode{}));

            const auto scale_m_coord =
                output_acc_tile_distr.calculate_index(); // 2d thread offset, [i_row, i_col]

            constexpr index_t kM2 = 4;                         // Val-dim
            constexpr index_t kM1 = get_warp_size() / NPerXdl; // Thr-dim
            constexpr index_t kM0 = MPerXdl / kM1 / kM2;       // Var-dim

            constexpr index_t ScaleMRepeat = MRepeat * kM0 * kM2;
            statically_indexed_array<index_t, ScaleMRepeat> scale_m_offsets;

            if constexpr(!BMXFP4_Pipeline)
                static_for<0, MRepeat, 1>{}([&](auto mIter) {
                    static_for<0, kM0, 1>{}([&](auto m0) {
                        static_for<0, kM2, 1>{}([&](auto m2) {
                            const auto row_idx =
                                coord_m + mIter * MPerXdl + m0 * kM1 * kM2 + m2 + scale_m_coord[I0];
                            scale_m_offsets[mIter * number<kM0 * kM2>{} + m0 * number<kM2>{} + m2] =
                                row_to_token_idx(row_idx);
                        });
                    });
                });

            constexpr int DynamicTileOffsetFlag = 0;

            constexpr bool EnableBias = decltype(kargs.exp_bias)::GranularityMN != -1;

            auto permute_tensor_view = [&](auto naive_view, auto is_needed_to_permute_N_PACK) {
                if constexpr(!is_needed_to_permute_N_PACK)
                {
                    return naive_view;
                }
                else
                {
                    auto view1 = transform_tensor_view(
                        naive_view,
                        make_tuple(
                            make_pass_through_transform(number<DynamicTileOffsetFlag>{}),
                            make_unmerge_transform(make_tuple(number<DynamicTileOffsetFlag>{},
                                                              number<NRepeat / N_Pack>{},
                                                              number<NWave>{},
                                                              number<N_Pack>{},
                                                              number<NPerXdl>{}))),
                        make_tuple(sequence<0>{}, sequence<1>{}),
                        make_tuple(sequence<0>{}, sequence<1, 2, 3, 4, 5>{}));
                    return transform_tensor_view(
                        view1,
                        make_tuple(make_pass_through_transform(number<DynamicTileOffsetFlag>{}),
                                   make_merge_transform_v3_division_mod(
                                       make_tuple(number<DynamicTileOffsetFlag>{},
                                                  number<NRepeat / N_Pack>{},
                                                  number<N_Pack>{},
                                                  number<NWave>{},
                                                  number<NPerXdl>{}))),
                        make_tuple(sequence<0>{}, sequence<1, 2, 4, 3, 5>{}),
                        make_tuple(sequence<0>{}, sequence<1>{}));
                }
            };

            auto scale_m_window =
                make_tile_scatter_gather(make_naive_tensor_view<address_space_enum::global>(
                                             kargs.scale_m.ptr,
                                             make_tuple(kargs.M, 1),
                                             make_tuple(scale_stride_m, 0),
                                             number<1>{}, // gather load can't vectorize
                                             number<1>{}),
                                         make_tuple(number<TilePartitioner::MPerBlock>{},
                                                    number<TilePartitioner::NPerBlock>{}),
                                         {0, 0}, // offset m is included in gather offsets
                                         output_acc_tile_distr,
                                         scale_m_offsets);

            auto scale_n_window = make_tile_window(
                make_naive_tensor_view<address_space_enum::global>(
                    kargs.scale_n.ptr + expert_id * kargs.N,
                    make_tuple(1, kargs.N),
                    make_tuple(0, scale_stride_n),
                    number < ScaleGranularityN == 1 ? FlatmmPipeline::GetVectorSizeB() : 1 > {},
                    number<1>{}), // MXF4_Pipeline does't use scale_n, so there is no need to
                                  // permute as n_pack
                make_tuple(number<TilePartitioner::MPerBlock>{},
                           number < IsGateUp ? TilePartitioner::NPerBlock / 2
                                             : TilePartitioner::NPerBlock > {}),
                {0, IsGateUp ? coord_n / 2 : coord_n},
                output_acc_tile_distr);

            auto scale_n_up_window = make_tile_window(
                make_naive_tensor_view<address_space_enum::global>(
                    kargs.scale_n.ptr + expert_id * kargs.N + kargs.N / 2,
                    make_tuple(1, kargs.N),
                    make_tuple(0, scale_stride_n),
                    number < ScaleGranularityN == 1 ? FlatmmPipeline::GetVectorSizeB() : 1 > {},
                    number<1>{}),
                make_tuple(number<TilePartitioner::MPerBlock>{},
                           number<TilePartitioner::NPerBlock / 2>{}),
                {0, coord_n / 2},
                output_acc_tile_distr);

            auto exp_bias_view = make_naive_tensor_view<address_space_enum::global>(
                kargs.exp_bias.ptr + expert_id * kargs.N,
                make_tuple(1, kargs.N),
                make_tuple(0, scale_stride_n),
                number<FlatmmPipeline::GetVectorSizeB()>{},
                number<1>{});

            auto exp_bias_window = make_tile_window(
                permute_tensor_view(exp_bias_view, number<(BMXFP4_Pipeline && !IsInputGemm)>{}),
                make_tuple(number<TilePartitioner::MPerBlock>{},
                           number < IsGateUp ? TilePartitioner::NPerBlock / 2
                                             : TilePartitioner::NPerBlock > {}),
                {0, IsGateUp ? coord_n / 2 : coord_n},
                output_acc_tile_distr);

            auto exp_bias_up_window =
                make_tile_window(make_naive_tensor_view<address_space_enum::global>(
                                     kargs.exp_bias.ptr + expert_id * kargs.N + kargs.N / 2,
                                     make_tuple(1, kargs.N),
                                     make_tuple(0, scale_stride_n),
                                     number<FlatmmPipeline::GetVectorSizeB()>{},
                                     number<1>{}),
                                 make_tuple(number<TilePartitioner::MPerBlock>{},
                                            number<TilePartitioner::NPerBlock / 2>{}),
                                 {0, coord_n / 2},
                                 output_acc_tile_distr);

            auto exp_weight_window =
                make_tile_window(make_naive_tensor_view<address_space_enum::global>(
                                     static_cast<const float*>(kargs.p_sorted_expert_weights),
                                     make_tuple(kargs.M, 1),
                                     make_tuple(1, 0),
                                     number<FlatmmPipeline::GetVectorSizeA()>{},
                                     number<1>{}),
                                 make_tuple(number<TilePartitioner::MPerBlock>{},
                                            number<TilePartitioner::NPerBlock>{}),
                                 {coord_m, 0},
                                 output_acc_tile_distr);

            using ScaleMBuffer    = decltype(load_tile(scale_m_window));
            using ScaleNBuffer    = decltype(load_tile(scale_n_window));
            using ExpBiasBuffer   = decltype(load_tile(exp_bias_window));
            using ExpWeightBuffer = decltype(load_tile(exp_weight_window));

            ScaleMBuffer scale_m_buffer;
            ScaleNBuffer scale_n_buffer, scale_n_up_buffer;

            ExpBiasBuffer exp_bias_buffer, exp_bias_up_buffer;
            ExpWeightBuffer exp_weight_buffer;

            if constexpr(!BMXFP4_Pipeline)
            {
                scale_m_window.load(scale_m_buffer);
                scale_n_buffer = load_tile(scale_n_window);
                if constexpr(IsGateUp)
                    scale_n_up_buffer = load_tile(scale_n_up_window);
            }

            if constexpr(EnableBias)
            {
                exp_bias_buffer = load_tile(exp_bias_window);
                if constexpr(IsGateUp)
                    exp_bias_up_buffer = load_tile(exp_bias_up_window);
            }
            if constexpr(!IsInputGemm)
                exp_weight_buffer = load_tile(exp_weight_window);

            auto in_lds_window = make_tile_window(
                o_lds_block,
                make_tuple(number<MPerIterationShuffle>{}, number<LDS_NPerIterationShuffle>{}),
                {0, 0});

            auto out_lds_window = make_tile_window(
                o_lds_block,
                make_tuple(number<MPerIterationShuffle>{}, number<LDS_NPerIterationShuffle>{}),
                {0, 0});

            using SFC = space_filling_curve<sequence<kMPerBlock, kNPerBlock>,
                                            sequence<0, 1>,
                                            sequence<MPerIterationShuffle, NPerIterationShuffle>>;

            constexpr index_t num_access = SFC::get_num_of_access();

            static_assert(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>,
                          "Currently, the CShuffle EpiloguePipeline only supports the Row Major "
                          "Output layout");

            using TileEncodingPattern = tile_distribution_encoding_pattern_2d<
                kBlockSize,
                MPerIterationShuffle,
                LDS_NPerIterationShuffle,
                kind == MoeFlatmmKind::kFFN_gemm2 ? 2 : EpiloguePipeline::GetVectorSizeC(),
                tile_distribution_pattern::thread_raked,
                EpiProblem::kNumWaveGroups>;

            constexpr auto dram_tile_distribution =
                TileEncodingPattern::make_2d_static_tile_distribution();

            constexpr auto LdsTileDistr = [&] {
                if constexpr(IsGateUp)
                    return make_static_tile_distribution(
                        detail::make_embed_tile_distribution_encoding(
                            tile_distribution_encoding<
                                sequence<>,
                                tuple<sequence<NumMXdlPerWavePerShuffle, MWave>,
                                      // merge two contiguous N
                                      sequence<OutputNumNXdlPerWavePerShuffle, NWave>>,
                                tuple<sequence<1, 2>>,
                                tuple<sequence<1, 1>>,
                                sequence<1, 2>,
                                sequence<0, 0>>{},
                            typename CWarpDstr::DstrEncode{}));
                else
                    return make_static_tile_distribution(
                        EpiloguePipeline::MakeLdsDistributionEncode());
            }();

            using LDSTileTensor =
                decltype(make_static_distributed_tensor<AccDataType>(LdsTileDistr));
            LDSTileTensor lds_tile[2];

            constexpr auto c_warp_y_lengths =
                to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
            constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};
            constexpr int ActVectorSize = c_warp_y_lengths.product() * NumMXdlPerWavePerShuffle *
                                          OutputNumNXdlPerWavePerShuffle;

            auto epi_tile_idx_slice =
                [&](const auto& acc_tile_like_tensor, auto epi_m_idx, auto epi_n_idx) {
                    return acc_tile_like_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<epi_m_idx * NumMXdlPerWavePerShuffle,
                                                 epi_n_idx * OutputNumNXdlPerWavePerShuffle>{},
                                        c_warp_y_index_zeros),
                        merge_sequences(
                            sequence<NumMXdlPerWavePerShuffle, OutputNumNXdlPerWavePerShuffle>{},
                            c_warp_y_lengths));
                };

            auto gate_up_epi_tile_idx_interleave_slice = [&](auto& dest_gate_tensor,
                                                             auto& dest_up_tensor,
                                                             const auto& acc_tile_like_tensor,
                                                             auto epi_m_idx,
                                                             auto epi_n_idx) {
                static_for<0, OutputNumNXdlPerWavePerShuffle, 1>{}([&](auto n_xdl) {
                    dest_gate_tensor.set_y_sliced_thread_data(
                        merge_sequences(sequence<0, n_xdl>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<NumMXdlPerWavePerShuffle, 1>{}, c_warp_y_lengths),
                        acc_tile_like_tensor.get_y_sliced_thread_data(
                            merge_sequences(
                                sequence<epi_m_idx * NumMXdlPerWavePerShuffle,
                                         epi_n_idx * NumNXdlPerWavePerShuffle + 2 * n_xdl>{},
                                c_warp_y_index_zeros),
                            merge_sequences(sequence<NumMXdlPerWavePerShuffle, 1>{},
                                            c_warp_y_lengths)));
                    dest_up_tensor.set_y_sliced_thread_data(
                        merge_sequences(sequence<0, n_xdl>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<NumMXdlPerWavePerShuffle, 1>{}, c_warp_y_lengths),
                        acc_tile_like_tensor.get_y_sliced_thread_data(
                            merge_sequences(
                                sequence<epi_m_idx * NumMXdlPerWavePerShuffle,
                                         epi_n_idx * NumNXdlPerWavePerShuffle + 2 * n_xdl + 1>{},
                                c_warp_y_index_zeros),
                            merge_sequences(sequence<NumMXdlPerWavePerShuffle, 1>{},
                                            c_warp_y_lengths)));
                });
            };

            auto process_epi_tile = [&](auto lds_stage, auto epi_m, auto epi_n) {
                if constexpr(IsGateUp)
                {
                    LDSTileTensor gate_tensor, up_tensor;

                    gate_up_epi_tile_idx_interleave_slice(
                        gate_tensor, up_tensor, c_block_tile, epi_m, epi_n);
                    auto epi_scale_m    = epi_tile_idx_slice(scale_m_buffer, epi_m, epi_n);
                    auto epi_scale_n    = epi_tile_idx_slice(scale_n_buffer, epi_m, epi_n);
                    auto epi_scale_n_up = epi_tile_idx_slice(scale_n_up_buffer, epi_m, epi_n);

                    auto epi_exp_bias    = epi_tile_idx_slice(exp_bias_buffer, epi_m, epi_n);
                    auto epi_exp_bias_up = epi_tile_idx_slice(exp_bias_up_buffer, epi_m, epi_n);

                    static_for<0, ActVectorSize, 1>{}([&](auto idx) {
                        if constexpr(!BMXFP4_Pipeline)
                        {
                            gate_tensor.get_thread_buffer()[idx] *=
                                epi_scale_m[idx] * epi_scale_n[idx];
                            up_tensor.get_thread_buffer()[idx] *=
                                epi_scale_m[idx] * epi_scale_n_up[idx];
                        }
                        if constexpr(EnableBias)
                        {
                            gate_tensor.get_thread_buffer()[idx] += epi_exp_bias[idx];
                            up_tensor.get_thread_buffer()[idx] += epi_exp_bias_up[idx];
                        }
                        lds_tile[lds_stage].get_thread_buffer().at(idx) =
                            ActivationOp{}(gate_tensor.get_thread_buffer().at(idx),
                                           up_tensor.get_thread_buffer().at(idx));
                    });
                }
                else
                {
                    lds_tile[lds_stage].get_thread_buffer() =
                        epi_tile_idx_slice(c_block_tile, epi_m, epi_n);
                    auto epi_scale_m    = epi_tile_idx_slice(scale_m_buffer, epi_m, epi_n);
                    auto epi_scale_n    = epi_tile_idx_slice(scale_n_buffer, epi_m, epi_n);
                    auto epi_exp_weight = epi_tile_idx_slice(exp_weight_buffer, epi_m, epi_n);
                    auto epi_exp_bias   = epi_tile_idx_slice(exp_bias_buffer, epi_m, epi_n);

                    static_for<0, ActVectorSize, 1>{}([&](auto idx) {
                        if constexpr(!BMXFP4_Pipeline)
                            lds_tile[lds_stage].get_thread_buffer()[idx] *=
                                epi_scale_m[idx] * epi_scale_n[idx];
                        if(kind !=
                           MoeFlatmmKind::kFFN_gemm1_split_k) // disable weight and bias for split-k
                        {
                            if constexpr(EnableBias)
                                lds_tile[lds_stage].get_thread_buffer()[idx] += epi_exp_bias[idx];
                            if constexpr(!IsInputGemm)
                                lds_tile[lds_stage].get_thread_buffer()[idx] *= epi_exp_weight[idx];
                        }
                        if constexpr(kind ==
                                     MoeFlatmmKind::kFFN_gemm1_gate_only) // for mlp1 gate-only
                            lds_tile[lds_stage].get_thread_buffer()[idx] =
                                ActivationOp{}(lds_tile[lds_stage].get_thread_buffer()[idx]);
                    });
                }
            };

            constexpr int NumMEpiTile = MRepeat / NumMXdlPerWavePerShuffle;
            constexpr int MPerThread  = TileEncodingPattern::Y2;
            statically_indexed_array<statically_indexed_array<index_t, MPerThread>, NumMEpiTile>
                c_scatter_offsets;
            statically_indexed_array<statically_indexed_array<bool, MPerThread>, NumMEpiTile>
                c_scatter_valids;
            auto c_coord = dram_tile_distribution.calculate_index();
            static_for<0, NumMEpiTile, 1>{}([&](auto mIter) {
                static_for<0, MPerThread, 1>{}([&](auto m0) {
                    auto row_idx = coord_m + mIter * MPerIterationShuffle + c_coord[0] + m0;
                    auto fused_token =
                        kargs.p_sorted_token_ids[row_idx]; // topk-idx[31:24] + token_idx[23:0]

                    index_t scatter_token_id    = fused_token & token_id_mask;
                    c_scatter_valids[mIter][m0] = (scatter_token_id < kargs.NumTokens);
                    if constexpr(IsInputGemm)
                        scatter_token_id =
                            scatter_token_id * kargs.TopK + (fused_token >> token_id_offset);
                    c_scatter_offsets[mIter][m0] = scatter_token_id * kargs.stride_C;
                });
            });

            //===----------------------------------------------------------------------===//
            // Pingpong process start
            //===----------------------------------------------------------------------===//
            process_epi_tile(number<0>{}, number<0>{}, number<0>{});

            static_for<0, num_access, 1>{}([&](auto iAccess) {
                constexpr int read_stage  = iAccess % 2;
                constexpr int write_stage = read_stage ^ 1;

                block_sync_lds();
                constexpr auto idx_y_start = SFC::get_index(number<iAccess.value>{});
                constexpr auto mIter = number<idx_y_start.at(number<0>{}) / MPerIterationShuffle>{};

                const auto c_warptile_in_tensor_casted = cast_tile<ODataType>(lds_tile[read_stage]);

                store_tile(in_lds_window, c_warptile_in_tensor_casted);

                if constexpr(iAccess < num_access - 1)
                {
                    constexpr auto idx_y_start_next = SFC::get_index(number<iAccess.value + 1>{});
                    constexpr auto mIter_next =
                        number<idx_y_start_next.at(number<0>{}) / MPerIterationShuffle>{};
                    constexpr auto nIter_next =
                        number<idx_y_start_next.at(number<1>{}) / NPerIterationShuffle>{};

                    process_epi_tile(number<write_stage>{}, mIter_next, nIter_next);
                }

                block_sync_lds();

                auto c_out_tensor =
                    load_tile(make_tile_window(out_lds_window, dram_tile_distribution));
                auto c_scatter_tile_window =
                    make_tile_scatter_gather(c_block_window.get_bottom_tensor_view(),
                                             c_block_window.get_window_lengths(),
                                             c_block_window.get_window_origin(),
                                             dram_tile_distribution,
                                             c_scatter_offsets[mIter],
                                             c_scatter_valids[mIter]);

                if constexpr(!IsInputGemm ||
                             decltype(c_block_window.get_bottom_tensor_view())::DstInMemOp ==
                                 memory_operation_enum::atomic_add)
                    c_scatter_tile_window.update(c_out_tensor);
                else
                    c_scatter_tile_window.store(c_out_tensor);

                if constexpr(iAccess != num_access - 1)
                {
                    constexpr auto step = SFC::get_forward_step(iAccess);
                    // row_offset of out windows has been included in scatter offset
                    move_tile_window(c_block_window,
                                     {0, step.at(number<1>{}) / number < IsGateUp ? 2 : 1 > {}});
                }
            });
        }
    }
};

} // namespace ck_tile
