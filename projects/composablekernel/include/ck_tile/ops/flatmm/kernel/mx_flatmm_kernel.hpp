// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

#include "ck_tile/ops/flatmm/kernel/flatmm_kernel.hpp"

namespace ck_tile {

template <typename TilePartitioner_, typename MXFlatmmPipeline_, typename EpiloguePipeline_>
struct MXFlatmmKernel : FlatmmKernel<TilePartitioner_, MXFlatmmPipeline_, EpiloguePipeline_>
{
    using Underlying = FlatmmKernel<TilePartitioner_, MXFlatmmPipeline_, EpiloguePipeline_>;

    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using MXFlatmmPipeline = remove_cvref_t<MXFlatmmPipeline_>;
    using BlockGemmShape =
        remove_cvref_t<typename MXFlatmmPipeline_::BlockGemmShape>; // TileFlatmmShape
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;
    using ALayout          = remove_cvref_t<typename MXFlatmmPipeline::ALayout>;
    using BLayout          = remove_cvref_t<typename MXFlatmmPipeline::BLayout>;
    using ELayout          = remove_cvref_t<typename MXFlatmmPipeline::CLayout>;
    using DsLayout         = remove_cvref_t<typename EpiloguePipeline::DsLayout>;
    using DsDataType       = remove_cvref_t<typename EpiloguePipeline::DsDataType>;
    static constexpr index_t KernelBlockSize  = MXFlatmmPipeline::BlockSize;
    static constexpr bool UsePersistentKernel = MXFlatmmPipeline::UsePersistentKernel;

    using ADataType = remove_cvref_t<typename MXFlatmmPipeline::ADataType>;
    using BDataType = remove_cvref_t<typename MXFlatmmPipeline::BDataType>;
    // Below type is actually accumulation data type - the output of block GEMM.
    using EDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    static constexpr int MThreadPerXdl = BlockGemmShape::WarpTile::at(number<0>{});
    static constexpr int NThreadPerXdl = BlockGemmShape::WarpTile::at(number<1>{});
    static constexpr int KThreadPerXdl = 64 / MThreadPerXdl;

    static constexpr int APackedSize = numeric_traits<ADataType>::PackedSize;
    static constexpr int BPackedSize = numeric_traits<BDataType>::PackedSize;

    static constexpr int MXdlPack = MXFlatmmPipeline::MXdlPack;
    static constexpr int NXdlPack = MXFlatmmPipeline::NXdlPack;
    static constexpr int KXdlPack = MXFlatmmPipeline::KXdlPack;

    static constexpr index_t NumDTensor = DsDataType::size();

    static constexpr auto I0 = number<0>();
    static constexpr auto I1 = number<1>();
    static constexpr auto I2 = number<2>();
    static constexpr auto I3 = number<3>();
    static constexpr auto I4 = number<4>();
    static constexpr auto I5 = number<5>();

    static_assert(DsLayout::size() == DsDataType::size(),
                  "The size of DsLayout and DsDataType should be the same");
    // using KernelArgs = FlatmmKernelArgs<DsLayout::size()>;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "mx_flatmm_gemm", gemm_prec_str<ADataType, BDataType>, MXFlatmmPipeline::GetName());
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

            constexpr int block_size = MXFlatmmKernel::BlockSize().x;
            int dync_smem_size       = 0;
            int maxActiveBlocksPerCU = 0;

            if(hipGetDeviceProperties(&prop, deviceId) != hipSuccess)
                throw std::runtime_error(std::string("hipGetDeviceProperties failed: ") +
                                         hipGetErrorName(hipGetLastError()));

            if(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                   &maxActiveBlocksPerCU,
                   reinterpret_cast<void*>(
                       kentry<1, MXFlatmmKernel, remove_cvref_t<decltype(kargs)>>),
                   block_size,
                   dync_smem_size) != hipSuccess)
                throw std::runtime_error(
                    std::string("hipOccupancyMaxActiveBlocksPerMultiprocessor failed: ") +
                    hipGetErrorName(hipGetLastError()));

            const int persistent_block_size = prop.multiProcessorCount * maxActiveBlocksPerCU;
            const int total_work_tile_cnt   = TilePartitioner::GridSize(kargs.M, kargs.N);

            // std::cout << "maxActiveBlocksPerCU: " << maxActiveBlocksPerCU
            //           << ", persistent_block_size: " << persistent_block_size
            //           << ", total_work_tile_cnt: " << total_work_tile_cnt << std::endl;

            if(kargs.k_batch != 1)
                throw std::runtime_error("Wrong! k_batch != 1 not supported in persistent kernel");
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
            static_assert(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>,
                          "A tensor for mx must be RowMajor");
            return make_naive_tensor_view<address_space_enum::global>(
                a_ptr,
                make_tuple(kargs.M, k_size),
                make_tuple(kargs.stride_A, 1),
                number<MXFlatmmPipeline::GetVectorSizeA()>{},
                number<1>{});
        }();

        // Step 2: Create padded view
        const auto& a_pad_view = pad_tensor_view(
            a_tensor_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::KPerBlock>{}),
            sequence<false, MXFlatmmPipeline::kPadK>{});

        // Step 3: Create tile window
        return make_tile_window(
            a_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::KPerBlock>{}),
            {block_idx_m, 0});
    }

    template <typename KernelArgs>
    CK_TILE_DEVICE static auto MakeBFlatBlockWindow(const BDataType* b_flat_ptr,
                                                    const KernelArgs& kargs,
                                                    const index_t block_idx_n)
    {
        // Step 1: Create tensor view with special flat layout
        constexpr index_t kKPerBlock    = MXFlatmmPipeline::kKPerBlock;
        constexpr index_t kNWarpTile    = BlockGemmShape::WarpTile::at(I1);
        constexpr index_t flatKPerBlock = kKPerBlock * kNWarpTile;
        const index_t kFlatKBlocks      = kargs.K / kKPerBlock;
        const index_t kFlatN            = kargs.N / kNWarpTile;

        const auto& b_flat_tensor_view = [&]() {
            static_assert(flatKPerBlock % MXFlatmmPipeline::GetVectorSizeB() == 0,
                          "wrong! vector size for B tensor");
            auto&& naive_desc = make_naive_tensor_descriptor_packed(
                make_tuple(kFlatN, kFlatKBlocks, number<flatKPerBlock>{}));
            auto&& desc = transform_tensor_descriptor(
                naive_desc,
                make_tuple(make_pass_through_transform(kFlatN),
                           make_merge_transform_v3_division_mod(
                               make_tuple(kFlatKBlocks, number<flatKPerBlock>{}))),
                make_tuple(sequence<0>{}, sequence<1, 2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
            return make_tensor_view<address_space_enum::global>(b_flat_ptr, desc);
        }();

        // Step 2: No padding for flat B
        // Step 3: Create tile window
        return make_tile_window(
            b_flat_tensor_view,
            make_tuple(number<MXFlatmmPipeline::flatNPerWarp>{},
                       number<MXFlatmmPipeline::flatKPerWarp>{}),
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
                                           sequence<false, MXFlatmmPipeline::kPadN>{});
                }
                else
                {
                    return pad_tensor_view(ds_tensor_view[i],
                                           make_tuple(number<TilePartitioner::NPerBlock>{},
                                                      number<TilePartitioner::MPerBlock>{}),
                                           sequence<false, MXFlatmmPipeline::kPadM>{});
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
                                       sequence<false, MXFlatmmPipeline::kPadN>{});
            }
            else
            {
                return pad_tensor_view(e_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<MXFlatmmPipeline::kPadM, false>{});
            }
        }();

        // Step 3: Create tile window
        return make_tile_window(
            e_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {block_idx_m, block_idx_n});
    }

    template <typename KernelArgs>
    CK_TILE_DEVICE static auto MakeScaleABlockWindow(const KernelArgs& kargs,
                                                     const index_t block_idx_m)
    {
        static constexpr int BlockScaleSize = 32;

        const auto&& scale_packs_m = integer_divide_ceil(kargs.M, (MXdlPack * MThreadPerXdl));
        const auto&& scale_packs_k = kargs.K / BlockScaleSize / (KXdlPack * KThreadPerXdl);

        // Step 1: Create tensor view
        const auto scale_a_naive_desc = make_naive_tensor_descriptor_packed(
            make_tuple(scale_packs_m, scale_packs_k, KThreadPerXdl, MThreadPerXdl));
        const auto scale_a_desc = transform_tensor_descriptor(
            scale_a_naive_desc,
            make_tuple(make_merge_transform(make_tuple(scale_packs_m, MThreadPerXdl)),
                       make_merge_transform(make_tuple(scale_packs_k, KThreadPerXdl))),
            make_tuple(sequence<0, 3>{}, sequence<1, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        const auto& scale_a_tensor_view = make_tensor_view<address_space_enum::global>(
            reinterpret_cast<const int32_t*>(kargs.scale_m_ptr.ptr), scale_a_desc);

        // Step 2: Create tile window
        return make_tile_window(
            scale_a_tensor_view,
            make_tuple(number<TilePartitioner::MPerBlock / MXdlPack>{},
                       number<TilePartitioner::KPerBlock / (BlockScaleSize * KXdlPack)>{}),
            {block_idx_m / MXdlPack, 0});
    }

    template <typename KernelArgs>
    CK_TILE_DEVICE static auto MakeScaleBBlockWindow(const KernelArgs& kargs,
                                                     const index_t block_idx_n)
    {
        static constexpr int BlockScaleSize = 32;

        const auto&& scale_packs_n = integer_divide_ceil(kargs.N, (NXdlPack * NThreadPerXdl));
        const auto&& scale_packs_k = kargs.K / BlockScaleSize / (KXdlPack * KThreadPerXdl);

        // Step 1: Create tensor view
        const auto scale_b_naive_desc = make_naive_tensor_descriptor_packed(
            make_tuple(scale_packs_n, scale_packs_k, KThreadPerXdl, NThreadPerXdl));
        const auto scale_b_desc = transform_tensor_descriptor(
            scale_b_naive_desc,
            make_tuple(make_merge_transform(make_tuple(scale_packs_n, NThreadPerXdl)),
                       make_merge_transform(make_tuple(scale_packs_k, KThreadPerXdl))),
            make_tuple(sequence<0, 3>{}, sequence<1, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        const auto& scale_b_tensor_view = make_tensor_view<address_space_enum::global>(
            reinterpret_cast<const int32_t*>(kargs.scale_n_ptr.ptr), scale_b_desc);

        // Step 2: Create tile window
        return make_tile_window(
            scale_b_tensor_view,
            make_tuple(number<TilePartitioner::NPerBlock / NXdlPack>{},
                       number<TilePartitioner::KPerBlock / (BlockScaleSize * KXdlPack)>{}),
            {block_idx_n / NXdlPack, 0});
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
        const auto& ds_block_window = MakeDBlockWindows(ds_ptr, kargs, block_idx_m, block_idx_n);
        const auto& scale_a_block_window = MakeScaleABlockWindow(kargs, block_idx_m);
        const auto& scale_b_block_window = MakeScaleBBlockWindow(kargs, block_idx_n);

        const index_t num_loop = TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k);

        static_assert(ScaleM::GranularityK == ScaleN::GranularityK // have the same granK
                          || ScaleM::GranularityMN == -1           // or ScaleA is disable
                          || ScaleN::GranularityMN == -1,          // or ScaleB is disable
                      "ScaleM and ScaleN should have the same GranularityK");
        constexpr bool DoEpiScale =
            (ScaleM::GranularityMN != -1 && ScaleM::GranularityK == 0) || // per token
            (ScaleN::GranularityMN != -1 && ScaleN::GranularityK == 0);   // per channel

        const auto& c_block_tile = MXFlatmmPipeline{}(a_block_window,
                                                      b_flat_block_window,
                                                      scale_a_block_window,
                                                      scale_b_block_window,
                                                      num_loop,
                                                      smem_ptr_ping,
                                                      smem_ptr_pong);

        // Run Epilogue Pipeline with split_k dispatch
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
            const index_t i_m = amd_wave_read_first_lane(iM * TilePartitioner::MPerBlock);
            const index_t i_n = amd_wave_read_first_lane(iN * TilePartitioner::NPerBlock);

            const SplitKBatchOffset splitk_batch_offset(kargs);
            // options
            const auto a_ptr = static_cast<const ADataType*>(kargs.a_ptr) +
                               splitk_batch_offset.a_k_split_offset / APackedSize;
            const auto b_flat_ptr = static_cast<const BDataType*>(kargs.b_ptr) +
                                    splitk_batch_offset.b_k_split_offset / BPackedSize;
            EDataType* e_ptr = static_cast<EDataType*>(kargs.e_ptr);

            // allocate LDS
            __shared__ char smem_ptr_ping[Underlying::GetSmemPingSize()];
            __shared__ char smem_ptr_pong[Underlying::GetSmemPongSize()];

            constexpr auto scheduler_type = (MXFlatmmPipeline::NumWaveGroups == 1);
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
            partition_idx += gridDim.x;
        } while(UsePersistentKernel && partition_idx < total_work_tile_cnt);
    }
};

} // namespace ck_tile
