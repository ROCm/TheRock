// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

#include "ck_tile/ops/flatmm/kernel/flatmm_kernel.hpp"

namespace ck_tile {

template <typename TilePartitioner_, typename FlatmmPipeline_, typename EpiloguePipeline_>
struct F16xMXF4FlatmmKernel : FlatmmKernel<TilePartitioner_, FlatmmPipeline_, EpiloguePipeline_>
{
    using Underlying = FlatmmKernel<TilePartitioner_, FlatmmPipeline_, EpiloguePipeline_>;

    using TilePartitioner = remove_cvref_t<TilePartitioner_>;
    using FlatmmPipeline  = remove_cvref_t<FlatmmPipeline_>;
    using BlockGemmShape =
        remove_cvref_t<typename FlatmmPipeline::BlockGemmShape>; // TileFlatmmShape
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;
    using ALayout          = remove_cvref_t<typename FlatmmPipeline::ALayout>;
    using BLayout          = remove_cvref_t<typename FlatmmPipeline::BLayout>;
    using ELayout          = remove_cvref_t<typename FlatmmPipeline::CLayout>;
    using DsLayout         = remove_cvref_t<typename EpiloguePipeline::DsLayout>;
    using DsDataType       = remove_cvref_t<typename EpiloguePipeline::DsDataType>;
    static constexpr index_t KernelBlockSize  = FlatmmPipeline::BlockSize;
    static constexpr bool UsePersistentKernel = FlatmmPipeline::UsePersistentKernel;

    using ADataType = remove_cvref_t<typename FlatmmPipeline::ADataType>;
    using BDataType = remove_cvref_t<typename FlatmmPipeline::BDataType>;
    // Below type is actually accumulation data type - the output of block GEMM.
    using EDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    static constexpr int QuantPackedSize = numeric_traits<BDataType>::PackedSize;
    static constexpr int N_Pack          = 2;

    static constexpr index_t NumDTensor = DsDataType::size();

    static constexpr auto I0 = number<0>();
    static constexpr auto I1 = number<1>();
    static constexpr auto I2 = number<2>();
    static constexpr auto I3 = number<3>();
    static constexpr auto I4 = number<4>();

    static_assert(DsLayout::size() == DsDataType::size(),
                  "The size of DsLayout and DsDataType should be the same");
    // using KernelArgs = FlatmmKernelArgs<DsLayout::size()>;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "mixed_prec_gemm", gemm_prec_str<ADataType, BDataType>, FlatmmPipeline::GetName());
        // clang-format on
    }

    template <class ScaleM, class ScaleN>
    CK_TILE_HOST static constexpr auto
    GridSize(const FlatmmKernelArgs<ScaleM, ScaleN, DsDataType::size()>& kargs)
    {
        if constexpr(UsePersistentKernel)
        {
            hipDeviceProp_t prop;
            int deviceId = 0; // default device

            constexpr int block_size = F16xMXF4FlatmmKernel::BlockSize().x;
            int dync_smem_size       = 0;
            int maxActiveBlocksPerCU = 0;

            [[maybe_unused]] auto e = hipGetDeviceProperties(&prop, deviceId);

            e = hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &maxActiveBlocksPerCU,
                reinterpret_cast<void*>(
                    kentry<1,
                           F16xMXF4FlatmmKernel,
                           FlatmmKernelArgs<ScaleM, ScaleN, DsDataType::size()>>),
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

    using SplitKBatchOffset = typename Underlying::SplitKBatchOffset;

    template <typename KernelArgs>
    CK_TILE_DEVICE static auto MakeABlockWindow(const ADataType* a_ptr,
                                                const KernelArgs& kargs,
                                                const index_t k_size,
                                                const index_t block_idx_m)
    {
        // Step 1: Create tensor view
        const auto& a_tensor_view = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_ptr,
                    make_tuple(kargs.M, k_size),
                    make_tuple(kargs.stride_A, 1),
                    number<FlatmmPipeline::GetVectorSizeA()>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_ptr,
                    make_tuple(k_size, kargs.M),
                    make_tuple(kargs.stride_A, 1),
                    number<FlatmmPipeline::GetVectorSizeA()>{},
                    number<1>{});
            }
        }();

        // Step 2: Create padded view
        const auto& a_pad_view = [&]() {
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

        // Step 3: Create tile window
        if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
        {
            return make_tile_window(a_pad_view,
                                    make_tuple(number<TilePartitioner::MPerBlock>{},
                                               number<TilePartitioner::KPerBlock>{}),
                                    {block_idx_m, 0});
        }
        else
        {
            return make_tile_window(a_pad_view,
                                    make_tuple(number<TilePartitioner::KPerBlock>{},
                                               number<TilePartitioner::MPerBlock>{}),
                                    {0, block_idx_m});
        }
    }

    template <typename KernelArgs>
    CK_TILE_DEVICE static auto MakeBFlatBlockWindow(const BDataType* b_flat_ptr,
                                                    const KernelArgs& kargs,
                                                    const index_t block_idx_n)
    {
        // Step 1: Create tensor view
        index_t kFlatK = kargs.K * BlockGemmShape::WarpTile::at(I1);
        index_t kFlatN = kargs.N * kargs.K / kFlatK;

        const auto& b_flat_tensor_view = make_naive_tensor_view<address_space_enum::global>(
            b_flat_ptr,
            make_tuple(kFlatN, kFlatK),
            make_tuple(kFlatK, 1),
            number<FlatmmPipeline::GetVectorSizeB()>{},
            number<1>{});

        // Step 2: No padding needed for b_flat
        // Step 3: Create tile window
        return make_tile_window(
            b_flat_tensor_view,
            make_tuple(number<FlatmmPipeline::flatNPerWarp>{},
                       number<FlatmmPipeline::flatKPerWarp>{}),
            {static_cast<int>(block_idx_n / BlockGemmShape::WarpTile::at(I1)), 0});
    }

    template <typename KernelArgs>
    CK_TILE_DEVICE static auto MakeDBlockWindows(const std::array<const void*, NumDTensor>& ds_ptr,
                                                 const KernelArgs& kargs,
                                                 const index_t block_idx_m,
                                                 const index_t block_idx_n)
    {
        // Step 1: Create tensor views
        const auto& ds_tensor_view = generate_tuple(
            [&](auto i) {
                using DiLayout   = remove_cvref_t<std::tuple_element_t<i.value, DsLayout>>;
                using DDataType_ = remove_cvref_t<std::tuple_element_t<i.value, DsDataType>>;
                if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        static_cast<const DDataType_*>(ds_ptr[i]),
                        make_tuple(kargs.M, kargs.N),
                        make_tuple(kargs.stride_Ds[i], 1),
                        number<EpiloguePipeline::GetVectorSizeD(i)>{},
                        number<1>{});
                }
                else
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        static_cast<const DDataType_*>(ds_ptr[i]),
                        make_tuple(kargs.N, kargs.M),
                        make_tuple(kargs.stride_Ds[i], 1),
                        number<EpiloguePipeline::GetVectorSizeD(i)>{},
                        number<1>{});
                }
            },
            number<NumDTensor>{});

        // Step 2: Create padded views
        const auto& ds_pad_view = generate_tuple(
            [&](auto i) {
                using DiLayout = remove_cvref_t<std::tuple_element_t<i.value, DsLayout>>;
                if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
                {
                    return pad_tensor_view(ds_tensor_view[i],
                                           make_tuple(number<TilePartitioner::MPerBlock>{},
                                                      number<TilePartitioner::NPerBlock>{}),
                                           sequence<false, FlatmmPipeline::kPadN>{});
                }
                else
                {
                    return pad_tensor_view(ds_tensor_view[i],
                                           make_tuple(number<TilePartitioner::NPerBlock>{},
                                                      number<TilePartitioner::MPerBlock>{}),
                                           sequence<false, FlatmmPipeline::kPadM>{});
                }
            },
            number<NumDTensor>{});

        // Step 3: Create tile windows
        return generate_tuple(
            [&](auto i) {
                using DiLayout = remove_cvref_t<std::tuple_element_t<i.value, DsLayout>>;
                if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
                {
                    return make_tile_window(ds_pad_view[i],
                                            make_tuple(number<TilePartitioner::MPerBlock>{},
                                                       number<TilePartitioner::NPerBlock>{}),
                                            {block_idx_m, block_idx_n});
                }
                else
                {
                    return make_tile_window(ds_pad_view[i],
                                            make_tuple(number<TilePartitioner::NPerBlock>{},
                                                       number<TilePartitioner::MPerBlock>{}),
                                            {block_idx_n, block_idx_m});
                }
            },
            number<NumDTensor>{});
    }

    template <memory_operation_enum DstInMemOp = memory_operation_enum::set, typename KernelArgs>
    CK_TILE_DEVICE static auto MakeEBlockWindow(EDataType* e_ptr,
                                                const KernelArgs& kargs,
                                                const index_t block_idx_m,
                                                const index_t block_idx_n)
    {
        // Step 1: Create tensor view
        const auto& e_tensor_view = [&]() {
            if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    e_ptr,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(kargs.stride_E, 1),
                    number<EpiloguePipeline::GetVectorSizeC()>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    e_ptr,
                    make_tuple(kargs.N, kargs.M),
                    make_tuple(kargs.stride_E, 1),
                    number<1>{},
                    number<1>{});
            }
        }();

        // Step 2: Create padded view
        const auto& e_pad_view = [&]() {
            if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(e_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<false, FlatmmPipeline::kPadN>{});
            }
            else
            {
                return pad_tensor_view(e_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<FlatmmPipeline::kPadM, false>{});
            }
        }();

        // Step 3: Create tile window
        return make_tile_window(
            e_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {block_idx_m, block_idx_n});
    }

    template <typename KernelArgs>
    CK_TILE_DEVICE static auto MakeScaleBBlockWindow(const KernelArgs& kargs,
                                                     const index_t block_idx_n)
    {
        auto scale_n = kargs.scale_n_ptr;

        // Step 1: Create tensor view
        index_t FlatScaleK =
            (kargs.K / decltype(scale_n)::GranularityK) * N_Pack * BlockGemmShape::WarpTile::at(I1);
        index_t FlatScaleN = kargs.N / N_Pack / BlockGemmShape::WarpTile::at(I1);

        const auto scale_b_flat_view = make_naive_tensor_view<address_space_enum::global>(
            reinterpret_cast<const e8m0_t*>(scale_n.ptr),
            make_tuple(FlatScaleN, FlatScaleK),
            make_tuple(FlatScaleK, 1),
            number<8>{},
            number<1>{});

        // Step 2: Create tile window
        return make_tile_window(
            scale_b_flat_view,
            make_tuple(number<FlatmmPipeline::flatNPerWarp>{},
                       number<FlatmmPipeline::flatKPerWarp * N_Pack * 4 / 32>{}),
            {block_idx_n / BlockGemmShape::WarpTile::at(I1) / N_Pack, 0});
    }

    template <class ScaleM, class ScaleN, bool UseDefaultScheduler = true>
    CK_TILE_DEVICE static void
    RunFlatmm(const ADataType* a_ptr,
              const BDataType* b_flat_ptr,
              const std::array<const void*, NumDTensor>& ds_ptr,
              EDataType* e_ptr,
              void* smem_ptr_ping,
              void* smem_ptr_pong,
              const FlatmmKernelArgs<ScaleM, ScaleN, DsDataType::size()>& kargs,
              const SplitKBatchOffset& splitk_batch_offset,
              const index_t block_idx_m,
              const index_t block_idx_n)
    {
        // Create block windows using specialized methods
        const auto& a_block_window =
            MakeABlockWindow(a_ptr, kargs, splitk_batch_offset.splitted_k, block_idx_m);
        const auto& b_flat_block_window = MakeBFlatBlockWindow(b_flat_ptr, kargs, block_idx_n);
        const auto& ds_block_window    = MakeDBlockWindows(ds_ptr, kargs, block_idx_m, block_idx_n);
        const auto& scale_block_window = MakeScaleBBlockWindow(kargs, block_idx_n);

        const index_t num_loop = TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k);

        static_assert(ScaleM::GranularityK == ScaleN::GranularityK // have the same granK
                          || ScaleM::GranularityMN == -1           // or ScaleA is disable
                          || ScaleN::GranularityMN == -1,          // or ScaleB is disable
                      "ScaleM and ScaleN should have the same GranularityK");
        constexpr bool DoEpiScale =
            (ScaleM::GranularityMN != -1 && ScaleM::GranularityK == 0) || // per token
            (ScaleN::GranularityMN != -1 && ScaleN::GranularityK == 0);   // per channel

        // Run GEMM cooperatively by whole workgroup.
        auto a_block_window_with_distr =
            ck_tile::make_tile_window(a_block_window.get_bottom_tensor_view(),
                                      a_block_window.get_window_lengths(),
                                      a_block_window.get_window_origin(),
                                      FlatmmPipeline::GetADramTileDistribution());
        const auto& c_block_tile = FlatmmPipeline{}(a_block_window_with_distr,
                                                    b_flat_block_window,
                                                    scale_block_window,
                                                    num_loop,
                                                    smem_ptr_ping,
                                                    smem_ptr_pong);

        // Run Epilogue Pipeline with k_batch dispatching
        if constexpr(DoEpiScale)
        {
            if(kargs.k_batch == 1)
            {
                auto e_block_window = MakeEBlockWindow<memory_operation_enum::set>(
                    e_ptr, kargs, block_idx_m, block_idx_n);
                EpiloguePipeline{}(e_block_window,
                                   c_block_tile,
                                   ds_block_window,
                                   smem_ptr_ping,
                                   kargs.scale_m_ptr + block_idx_m,
                                   kargs.scale_n_ptr + block_idx_n);
            }
            else
            {
                auto e_block_window = MakeEBlockWindow<memory_operation_enum::atomic_add>(
                    e_ptr, kargs, block_idx_m, block_idx_n);
                EpiloguePipeline{}(e_block_window,
                                   c_block_tile,
                                   ds_block_window,
                                   smem_ptr_ping,
                                   kargs.scale_m_ptr + block_idx_m,
                                   kargs.scale_n_ptr + block_idx_n);
            }
        }
        else if(UseDefaultScheduler || (get_warp_id() == 0))
        {
            if(kargs.k_batch == 1)
            {
                auto e_block_window = MakeEBlockWindow<memory_operation_enum::set>(
                    e_ptr, kargs, block_idx_m, block_idx_n);
                EpiloguePipeline{}(e_block_window, c_block_tile, ds_block_window, smem_ptr_ping);
            }
            else
            {
                auto e_block_window = MakeEBlockWindow<memory_operation_enum::atomic_add>(
                    e_ptr, kargs, block_idx_m, block_idx_n);
                EpiloguePipeline{}(e_block_window, c_block_tile, ds_block_window, smem_ptr_ping);
            }
        }
    }

    template <class ScaleM, class ScaleN>
    CK_TILE_DEVICE void operator()(FlatmmKernelArgs<ScaleM, ScaleN, DsDataType::size()> kargs,
                                   int partition_idx = blockIdx.x) const
    {
        int total_work_tile_cnt = TilePartitioner::GridSize(kargs.M, kargs.N);

        do
        {
            const auto [iM, iN] =
                TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(partition_idx);
            const index_t i_m = __builtin_amdgcn_readfirstlane(iM * TilePartitioner::MPerBlock);
            const index_t i_n = __builtin_amdgcn_readfirstlane(iN * TilePartitioner::NPerBlock);

            const SplitKBatchOffset splitk_batch_offset(kargs);
            // options
            const ADataType* a_ptr =
                static_cast<const ADataType*>(kargs.a_ptr) + splitk_batch_offset.a_k_split_offset;
            const BDataType* b_flat_ptr = static_cast<const BDataType*>(kargs.b_ptr) +
                                          splitk_batch_offset.b_k_split_offset / QuantPackedSize;
            EDataType* e_ptr = static_cast<EDataType*>(kargs.e_ptr);

            // allocate LDS
            __shared__ char smem_ptr_ping[Underlying::GetSmemPingSize()];
            __shared__ char smem_ptr_pong[Underlying::GetSmemPongSize()];

            if constexpr(!(EpiloguePipeline::GetVectorSizeC() % 2 != 0 &&
                           is_any_of<EDataType, fp16_t, bf16_t>::value))
            {
                constexpr auto scheduler_type = (FlatmmPipeline::NumWaveGroups == 1);
                RunFlatmm<ScaleM, ScaleN, scheduler_type>(a_ptr,
                                                          b_flat_ptr,
                                                          kargs.ds_ptr,
                                                          e_ptr,
                                                          smem_ptr_ping,
                                                          smem_ptr_pong,
                                                          kargs,
                                                          splitk_batch_offset,
                                                          i_m,
                                                          i_n);
            }
            partition_idx += gridDim.x;
        } while(UsePersistentKernel && partition_idx < total_work_tile_cnt);
    }
};

} // namespace ck_tile
