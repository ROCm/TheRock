// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/loop_scheduler.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"

namespace lds_direct_load {

__device__ void sched_barrier()
{
#if CK_USE_AMD_LDS_DIRECT_LOAD_INLINE_ASM
    // When direct loads and `waitcnt` instructions are submitted using inline asm, the usage of
    // `sched_barrier` is necessary to make sure no instructions that use the loaded memory
    // are scheduled by the compiler before the `waitcnt` instruction.
    __builtin_amdgcn_sched_barrier(0);
#endif
}

} // namespace lds_direct_load

namespace ck {

template <index_t NumPrefetch>
struct GridwiseGemmPipeline_v4;

// 1-stage prefetch
template <>
struct GridwiseGemmPipeline_v4<1>
{
    static constexpr auto I0 = Number<0>{};

    __host__ __device__ static constexpr bool IsSupported(index_t /* num_loop */) { return true; }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 1;
    }

    template <bool HasMainLoop,
              typename AGridDesc,
              typename ABlockDesc,
              typename ABlockTransfer,
              typename AGridBuffer,
              typename ABlockBuffers,
              typename ABlockTransferStep,
              typename BGridDesc,
              typename BBlockDesc,
              typename BBlockTransfer,
              typename BGridBuffer,
              typename BBlockBuffers,
              typename BBlockTransferStep,
              typename BlockwiseGemm,
              typename CThreadBuffer>
    __device__ static void Run(const AGridDesc& a_grid_desc,
                               const ABlockDesc& a_block_desc,
                               ABlockTransfer& a_blockwise_copy,
                               const AGridBuffer& a_grid_buf,
                               ABlockBuffers& a_block_bufs,
                               const ABlockTransferStep& a_block_copy_step,
                               const BGridDesc& b_grid_desc,
                               const BBlockDesc& b_block_desc,
                               BBlockTransfer& b_blockwise_copy,
                               const BGridBuffer& b_grid_buf,
                               BBlockBuffers& b_block_bufs,
                               const BBlockTransferStep& b_block_copy_step,
                               const BlockwiseGemm& blockwise_gemm,
                               CThreadBuffer& c_thread_buf,
                               index_t num_loop)
    {
        static_assert(ABlockBuffers::Size() == 1 && BBlockBuffers::Size() == 1);
        auto& a_block_buf = a_block_bufs.At(I0);
        auto& b_block_buf = b_block_bufs.At(I0);

        a_blockwise_copy.Run(a_grid_desc, a_grid_buf, a_block_desc, a_block_buf);
        b_blockwise_copy.Run(b_grid_desc, b_grid_buf, b_block_desc, b_block_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                block_sync_lds_direct_load();
                lds_direct_load::sched_barrier();

                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds_direct_load();
                lds_direct_load::sched_barrier();

                a_blockwise_copy.Run(a_grid_desc, a_grid_buf, a_block_desc, a_block_buf);
                b_blockwise_copy.Run(b_grid_desc, b_grid_buf, b_block_desc, b_block_buf);

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds_direct_load();
            lds_direct_load::sched_barrier();

            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }
    }
};

// 2-stages prefetch
template <>
struct GridwiseGemmPipeline_v4<2>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    __host__ __device__ static constexpr bool IsSupported(index_t num_loop)
    {
        return num_loop % 2 == 0;
    }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return (num_loop / 2) > 1;
    }

    template <bool HasMainLoop,
              typename AGridDesc,
              typename ABlockDesc,
              typename ABlockTransfer,
              typename AGridBuffer,
              typename ABlockBuffers,
              typename ABlockTransferStep,
              typename BGridDesc,
              typename BBlockDesc,
              typename BBlockTransfer,
              typename BGridBuffer,
              typename BBlockBuffers,
              typename BBlockTransferStep,
              typename BlockwiseGemm,
              typename CThreadBuffer>
    __device__ static void Run(const AGridDesc& a_grid_desc,
                               const ABlockDesc& a_block_desc,
                               ABlockTransfer& a_blockwise_copy,
                               const AGridBuffer& a_grid_buf,
                               ABlockBuffers& a_block_bufs,
                               const ABlockTransferStep& a_block_copy_step,
                               const BGridDesc& b_grid_desc,
                               const BBlockDesc& b_block_desc,
                               BBlockTransfer& b_blockwise_copy,
                               const BGridBuffer& b_grid_buf,
                               BBlockBuffers& b_block_bufs,
                               const BBlockTransferStep& b_block_copy_step,
                               const BlockwiseGemm& blockwise_gemm,
                               CThreadBuffer& c_thread_buf,
                               index_t num_loop)
    {
        static_assert(ABlockBuffers::Size() == 2 && BBlockBuffers::Size() == 2);
        auto& a_block_buf1 = a_block_bufs.At(I0);
        auto& a_block_buf2 = a_block_bufs.At(I1);
        auto& b_block_buf1 = b_block_bufs.At(I0);
        auto& b_block_buf2 = b_block_bufs.At(I1);

        a_blockwise_copy.Run(a_grid_desc, a_grid_buf, a_block_desc, a_block_buf1);
        b_blockwise_copy.Run(b_grid_desc, b_grid_buf, b_block_desc, b_block_buf1);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                block_sync_lds_direct_load();
                lds_direct_load::sched_barrier();

                a_blockwise_copy.Run(a_grid_desc, a_grid_buf, a_block_desc, a_block_buf2);
                b_blockwise_copy.Run(b_grid_desc, b_grid_buf, b_block_desc, b_block_buf2);

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                blockwise_gemm.Run(a_block_buf1, b_block_buf1, c_thread_buf);

                block_sync_lds_direct_load();
                lds_direct_load::sched_barrier();

                a_blockwise_copy.Run(a_grid_desc, a_grid_buf, a_block_desc, a_block_buf1);
                b_blockwise_copy.Run(b_grid_desc, b_grid_buf, b_block_desc, b_block_buf1);

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                blockwise_gemm.Run(a_block_buf2, b_block_buf2, c_thread_buf);

                i += 2;
            } while(i < (num_loop - 2));
        }

        // tail
        {
            block_sync_lds_direct_load();
            lds_direct_load::sched_barrier();

            a_blockwise_copy.Run(a_grid_desc, a_grid_buf, a_block_desc, a_block_buf2);
            b_blockwise_copy.Run(b_grid_desc, b_grid_buf, b_block_desc, b_block_buf2);

            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
            b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

            blockwise_gemm.Run(a_block_buf1, b_block_buf1, c_thread_buf);

            block_sync_lds_direct_load();
            lds_direct_load::sched_barrier();

            blockwise_gemm.Run(a_block_buf2, b_block_buf2, c_thread_buf);
        }
    }
};

} // namespace ck
