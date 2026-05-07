// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>

#include "ck_tile/host.hpp"
#include "ck_tile/ops/common/utils.hpp"
#include "ck_tile/ops/reduce.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp"
#include "gemm_utils.hpp"
#include "run_gemm_example.inc"

/**
 * @brief Tile partitioner with output offset support.
 *
 * This partitioner extends the spatially local tile partitioner to support
 * split-K reduction by providing workspace output offset calculation. Each K-split
 * writes to a separate slice of the workspace: workspace[k_id * M * N].
 */
template <typename BlockGemmShapeType, ck_tile::index_t GroupNum, ck_tile::index_t M01>
struct GemmSplitKTilePartitioner
    : public ck_tile::GemmSpatiallyLocalTilePartitioner<BlockGemmShapeType, GroupNum, M01>
{
    using Base = ck_tile::GemmSpatiallyLocalTilePartitioner<BlockGemmShapeType, GroupNum, M01>;

    // Inherit constructors and methods
    using Base::Base;
    using Base::GetLoopNum;

    /**
     * @brief Calculate output pointer offset for split-K reduction.
     *
     * @param kargs  Kernel arguments.
     * @param k_id   Current K-split ID (from blockIdx.z or calculated k_batch).
     * @return ck_tile::index_t  The offset for this K-split.
     */
    template <typename KernelArgs>
    CK_TILE_HOST_DEVICE static ck_tile::index_t GetOutputOffset(const KernelArgs& kargs,
                                                                ck_tile::index_t k_id) noexcept
    {
        // Each K-split gets its own M*N workspace slice
        return (kargs.k_batch > 1) ? (k_id * kargs.M * kargs.N) : 0;
    }
};

/**
 * @brief Extended GEMM host arguments for two-stage split-K implementation
 *
 * This structure supports the two-stage split-K approach where:
 * 1. Stage 1: GEMM writes partial results to workspace memory
 * 2. Stage 2: Reduction kernel sums workspace results to final output
 *
 * The base class e_ptr points to workspace, while final_output_ptr points to the actual output
 */
struct GemmSplitKHostArgs : public ck_tile::GemmHostArgs
{
    using BaseArgs = ck_tile::GemmHostArgs;

    CK_TILE_HOST GemmSplitKHostArgs() = default;
    CK_TILE_HOST GemmSplitKHostArgs(const void* a_ptr_,
                                    const void* b_ptr_,
                                    void* workspace_ptr_, // Workspace for partial results
                                    void* e_ptr_,         // Final output destination
                                    ck_tile::index_t k_batch_,
                                    ck_tile::index_t M_,
                                    ck_tile::index_t N_,
                                    ck_tile::index_t K_,
                                    ck_tile::index_t stride_A_,
                                    ck_tile::index_t stride_B_,
                                    ck_tile::index_t workspace_stride_,
                                    ck_tile::index_t stride_E_)
        : BaseArgs(a_ptr_,
                   b_ptr_,
                   workspace_ptr_, // Base e_ptr = workspace_ptr
                   k_batch_,
                   M_,
                   N_,
                   K_,
                   stride_A_,
                   stride_B_,
                   workspace_stride_),
          final_output_ptr(e_ptr_),
          final_stride_E(stride_E_)
    {
    }

    void* final_output_ptr;          // Pointer to final output tensor
    ck_tile::index_t final_stride_E; // Stride for final output tensor
};

/**
 * @brief Stage 1: GEMM kernel that writes partial split-K results to workspace
 *
 * This function performs the matrix multiplication with split-K, where each
 * K-split writes its partial result to a separate section of the workspace.
 *
 * Workspace layout: [k_batch, M, N] where each [M, N] slice contains
 * partial results for one K-split.
 *
 * @param args Extended arguments containing workspace and final output pointers
 * @param s Stream configuration for kernel execution
 * @return Execution time in milliseconds
 */
template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          bool Persistent,
          typename CDEElementWise>
float gemm_stage1(const GemmSplitKHostArgs& args, const ck_tile::stream_config& s)
{
    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
        ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
        ck_tile::
            sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>,
        GemmConfig::PermuteA,
        GemmConfig::PermuteB>;

    using TilePartitioner = GemmSplitKTilePartitioner<GemmShape,
                                                      GemmConfig::TileParitionerGroupNum,
                                                      GemmConfig::TileParitionerM01>;

    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
                                                                 GemmConfig::kPadN,
                                                                 GemmConfig::kPadK,
                                                                 GemmConfig::DoubleSmemBuffer,
                                                                 ALayout,
                                                                 BLayout,
                                                                 ELayout,
                                                                 GemmConfig::TransposeC,
                                                                 GemmConfig::UseStructuredSparsity,
                                                                 Persistent,
                                                                 GemmConfig::NumWaveGroups,
                                                                 GemmConfig::Preshuffle>;

    // Create base GEMM arguments pointing to workspace instead of final output
    // The workspace will store partial results from each K-split
    ck_tile::GemmHostArgs base_args(args.a_ptr,
                                    args.b_ptr,
                                    args.e_ptr,
                                    args.k_batch,
                                    args.M,
                                    args.N,
                                    args.K,
                                    args.stride_A,
                                    args.stride_B,
                                    args.stride_E);
    constexpr auto scheduler = GemmConfig::Scheduler;

    using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                       BDataType,
                                                                       AccDataType,
                                                                       GemmShape,
                                                                       GemmUniversalTraits,
                                                                       scheduler>;

    using GemmPipeline = typename PipelineTypeTraits<GemmConfig::Pipeline>::template GemmPipeline<
        UniversalGemmProblem>;

    using GemmEpilogue =
        ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<ADataType,
                                                                   BDataType,
                                                                   DsDataType,
                                                                   AccDataType,
                                                                   CDataType,
                                                                   DsLayout,
                                                                   ELayout,
                                                                   CDEElementWise,
                                                                   TilePartitioner::MPerBlock,
                                                                   TilePartitioner::NPerBlock,
                                                                   GemmConfig::M_Warp,
                                                                   GemmConfig::N_Warp,
                                                                   GemmConfig::M_Warp_Tile,
                                                                   GemmConfig::N_Warp_Tile,
                                                                   GemmConfig::K_Warp_Tile,
                                                                   UniversalGemmProblem::TransposeC,
                                                                   GemmConfig::NumWaveGroups>>;

    using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
    auto kargs   = Kernel::MakeKernelArgs(base_args);

    dim3 grids;
    if constexpr(Persistent)
    {
        grids = Kernel::MaxOccupancyGridSize(s);
    }
    else
    {
        grids = Kernel::GridSize(args.M, args.N, args.k_batch);
    }
    const dim3 blocks = Kernel::BlockSize();

    if(!Kernel::IsSupportedArgument(kargs))
    {
        throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
    }

    if(s.log_level_ > 0)
    {
        std::cout << "Stage 1 - Launching GEMM kernel: " << Kernel::GetName() << '\n'
                  << "shape: " << GemmShape::GetName() << '\n'
                  << "problem: " << UniversalGemmProblem::GetName() << '\n'
                  << "pipeline: " << GemmPipeline::GetName() << '\n'
                  << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                  << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                  << std::endl;
    }

    if(s.flush_cache_)
    {
        std::cout << "Flushing cache..." << std::endl;

        ck_tile::HostTensor<ADataType> a_m(ck_tile::host_tensor_descriptor(
            args.M, args.K, args.stride_A, is_row_major(ALayout{})));
        ck_tile::HostTensor<BDataType> b_n(ck_tile::host_tensor_descriptor(
            args.K, args.N, args.stride_B, is_row_major(BLayout{})));

        auto size_a_buffer = a_m.get_element_space_size_in_bytes();
        auto size_b_buffer = b_n.get_element_space_size_in_bytes();

        ck_tile::RotatingMemWrapper<ADataType, BDataType> rotating_mem(
            kargs.as_ptr[0], kargs.bs_ptr[0], s.rotating_count_, size_a_buffer, size_b_buffer);
        rotating_mem.Print();

        auto run_flush_cache = [&]() {
            // flush icache
            ck_tile::flush_icache();
            // rotating mem
            rotating_mem.Next();
            // clear c mem
            if(args.k_batch > 1)
                hipGetErrorString(hipMemsetAsync(
                    args.e_ptr, 0, args.M * args.N * sizeof(CDataType), s.stream_id_));
        };
        return ck_tile::launch_kernel_time_mask(
            s,
            run_flush_cache,
            ck_tile::make_kernel<GemmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
    }
    else
    {
        return ck_tile::launch_kernel(
            s, ck_tile::make_kernel<GemmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
    }
}

/**
 * @brief Stage 2: Reduction kernel that sums partial split-K results to final output
 *
 * This function reduces the partial results stored in workspace memory by stage 1.
 * It sums across the k_batch dimension to produce the final GEMM result.
 *
 * Workspace layout: [k_batch, M, N] -> Final output: [M, N]
 *
 * @tparam CDataType Output data type
 * @tparam ComputeDataType Computation precision for reduction
 * @tparam ELayout Memory layout of output tensor
 * @param args Extended arguments containing workspace and output information
 * @param s Stream configuration for kernel execution
 * @return Execution time in milliseconds
 */
template <typename CDataType,
          typename ComputeDataType = float,
          typename ELayout         = ck_tile::tensor_layout::gemm::RowMajor>
float reduce_stage2(const GemmSplitKHostArgs& args, const ck_tile::stream_config& s)
{
    // Calculate output size based on the final output tensor dimensions
    const ck_tile::index_t output_size = args.M * args.N;

    // Workspace layout: [k_batch, M, N] where each [M, N] slice has the same layout as final output
    // The workspace strides need to account for the layout of the final output tensor
    auto workspace_shape = ck_tile::make_tuple(args.k_batch, args.M, args.N);
    auto workspace_strides =
        ck_tile::make_tuple(args.M * args.N,     // k_batch stride: jump to next K split
                            args.final_stride_E, // stride same as final output stride
                            1);

    // Define kept and reduced dimensions
    constexpr auto kept_dim    = ck_tile::sequence<1, 2>{}; // Keep M, N dimensions
    constexpr auto reduce_dims = ck_tile::sequence<0>{};    // Reduce k_batch dimension

    using ReduceOp   = ck_tile::ReduceOp::Add;
    using BlockWarps = ck_tile::sequence<1, 1>;
    using BlockTile  = ck_tile::sequence<256, 1>;
    using WarpTile   = ck_tile::sequence<256, 1>;
    using ThreadTile = ck_tile::sequence<1, 1>;

    constexpr ck_tile::index_t kBlockPerCu = 1;

    ck_tile::index_t kGridSize = (output_size + BlockTile::at(ck_tile::number<0>{}) - 1) /
                                 BlockTile::at(ck_tile::number<0>{});

    using Shape   = ck_tile::Reduce2dShape<BlockWarps, BlockTile, WarpTile, ThreadTile>;
    using Problem = ck_tile::Reduce2dProblem<CDataType,
                                             ComputeDataType,
                                             CDataType,
                                             Shape,
                                             ReduceOp,
                                             decltype(kept_dim),
                                             decltype(reduce_dims),
                                             3>;
    using Kernel  = ck_tile::ReduceKernel<Problem>;
    const ck_tile::index_t kBlockSize = Kernel::BlockSize();

    if(s.log_level_ > 0)
    {
        std::cout << "Stage 2 - Launching Reduction kernel" << '\n'
                  << "workspace shape: [" << args.k_batch << ", " << args.M << ", " << args.N << "]"
                  << '\n'
                  << "output shape: [" << args.M << ", " << args.N << "]" << '\n'
                  << "grid size: " << kGridSize << std::endl;
    }

    float ave_time =
        ck_tile::launch_kernel(s,
                               ck_tile::make_kernel<kBlockPerCu>(
                                   Kernel{},
                                   kGridSize,
                                   kBlockSize,
                                   0,                                         // LDS size
                                   static_cast<const CDataType*>(args.e_ptr), // workspace input
                                   static_cast<CDataType*>(args.final_output_ptr), // final output
                                   workspace_shape,
                                   workspace_strides));

    return ave_time;
}

/**
 * @brief Orchestrator for two-stage split-K GEMM implementation
 *
 * This function coordinates the two-stage approach:
 * 1. Stage 1: Execute GEMM with each K-split writing to workspace
 * 2. Stage 2: Reduce workspace results to final output (if k_batch > 1)
 *
 * @param args Extended arguments for two-stage execution
 * @param s Stream configuration
 * @return Total execution time (GEMM + Reduction)
 */
template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          bool Persistent,
          typename CDEElementWise>
float gemm_splitk_two_stage(const GemmSplitKHostArgs& args, const ck_tile::stream_config& s)
{
    float gemm_time   = 0.0f;
    float reduce_time = 0.0f;

    if(s.log_level_ > 0)
    {
        std::cout << "Starting Two-Stage GEMM+SplitK with k_batch=" << args.k_batch << std::endl;
        std::cout << "Workspace size: " << args.k_batch << " x " << args.M << " x " << args.N
                  << " = " << args.k_batch * args.M * args.N * sizeof(CDataType) << " bytes"
                  << std::endl;
    }

    // Stage 1: GEMM to workspace
    gemm_time = gemm_stage1<GemmConfig,
                            ADataType,
                            BDataType,
                            DsDataType,
                            AccDataType,
                            CDataType,
                            ALayout,
                            BLayout,
                            DsLayout,
                            ELayout,
                            Persistent,
                            CDEElementWise>(args, s);

    // Synchronize before stage 2
    auto sync_result = hipStreamSynchronize(s.stream_id_);
    if(sync_result != hipSuccess)
    {
        throw std::runtime_error("Stream synchronization failed");
    }

    // Stage 2: Reduction from workspace to final output (if needed)
    if(args.k_batch > 1)
    {
        // Use appropriate precision for reduction computations
        using ComputeDataType = std::conditional_t<
            std::is_same_v<CDataType, ck_tile::half_t>,
            float,
            std::conditional_t<std::is_same_v<CDataType, ck_tile::bf16_t>, float, CDataType>>;
        reduce_time = reduce_stage2<CDataType, ComputeDataType, ELayout>(args, s);
    }
    else
    {
        // Single K-split: simple copy from workspace to final output
        auto copy_result = hipMemcpyAsync(args.final_output_ptr,
                                          args.e_ptr,
                                          args.M * args.N * sizeof(CDataType),
                                          hipMemcpyDeviceToDevice,
                                          s.stream_id_);
        if(copy_result != hipSuccess)
        {
            throw std::runtime_error("Memory copy failed");
        }
    }

    if(s.log_level_ > 0)
    {
        std::cout << "GEMM stage time: " << gemm_time << " ms" << std::endl;
        if(args.k_batch > 1)
        {
            std::cout << "Reduction stage time: " << reduce_time << " ms" << std::endl;
        }
        std::cout << "Total time: " << gemm_time + reduce_time << " ms" << std::endl;
    }

    return gemm_time + reduce_time;
}

/**
 * @brief High-level interface for two-stage split-K GEMM execution
 *
 * @param a_m_k_dev_buf Input matrix A device buffer
 * @param b_k_n_dev_buf Input matrix B device buffer
 * @param c_m_n_dev_buf Output matrix C device buffer
 * @param M Matrix M dimension
 * @param N Matrix N dimension
 * @param K Matrix K dimension
 * @param stride_A Memory stride for matrix A
 * @param stride_B Memory stride for matrix B
 * @param stride_C Memory stride for matrix C
 * @param kbatch Number of K-splits for split-K execution
 * @param n_warmup Number of warmup iterations
 * @param n_repeat Number of repeat iterations for benchmarking
 * @param persistent Whether to use persistent kernel execution
 * @return Average execution time in milliseconds
 */
template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename CDEElementWise = ck_tile::element_wise::PassThrough>
float invoke_gemm_splitk_two_stage(ck_tile::DeviceMem& a_m_k_dev_buf,
                                   ck_tile::DeviceMem& b_k_n_dev_buf,
                                   ck_tile::DeviceMem& c_m_n_dev_buf,
                                   ck_tile::index_t M,
                                   ck_tile::index_t N,
                                   ck_tile::index_t K,
                                   ck_tile::index_t stride_A,
                                   ck_tile::index_t stride_B,
                                   ck_tile::index_t stride_C,
                                   ck_tile::index_t kbatch,
                                   int n_warmup,
                                   int n_repeat,
                                   bool persistent)
{
    // Calculate workspace size: kbatch * M * N elements
    const ck_tile::index_t workspace_size   = kbatch * M * N * sizeof(CDataType);
    const ck_tile::index_t workspace_stride = stride_C; // Stride for k_batch dimension

    // Allocate workspace memory
    ck_tile::DeviceMem workspace_buf(workspace_size);
    workspace_buf.SetZero();

    // Create extended args for two-stage approach
    GemmSplitKHostArgs args{
        a_m_k_dev_buf.GetDeviceBuffer(), // a_ptr
        b_k_n_dev_buf.GetDeviceBuffer(), // b_ptr
        workspace_buf.GetDeviceBuffer(), // workspace_ptr (used as e_ptr for stage 1)
        c_m_n_dev_buf.GetDeviceBuffer(), // final_output_ptr
        kbatch,                          // k_batch
        M,
        N,
        K, // dimensions
        stride_A,
        stride_B,         // input strides
        workspace_stride, // workspace stride
        stride_C          // final output stride
    };

    float ave_time;
    ck_tile::stream_config config{nullptr, true, 1, n_warmup, n_repeat, true, true, 50};

    if(persistent)
    {
        ave_time = gemm_splitk_two_stage<GemmConfig,
                                         ADataType,
                                         BDataType,
                                         DsDataType,
                                         AccDataType,
                                         CDataType,
                                         ALayout,
                                         BLayout,
                                         DsLayout,
                                         CLayout,
                                         true,
                                         CDEElementWise>(args, config);
    }
    else
    {
        ave_time = gemm_splitk_two_stage<GemmConfig,
                                         ADataType,
                                         BDataType,
                                         DsDataType,
                                         AccDataType,
                                         CDataType,
                                         ALayout,
                                         BLayout,
                                         DsLayout,
                                         CLayout,
                                         false,
                                         CDEElementWise>(args, config);
    }

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_byte =
        sizeof(ADataType) * M * K + sizeof(BDataType) * N * K + sizeof(CDataType) * M * N;
    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Run Two-Stage GEMM+SplitK with M=" << M << " N=" << N << " K=" << K
              << " StrideA=" << stride_A << " StrideB=" << stride_B << " StrideC=" << stride_C
              << " kbatch=" << kbatch << " WorkspaceSize=" << workspace_size << " bytes"
              << " A_Layout=" << ALayout::name << " B_Layout =" << BLayout::name
              << " C_Layout=" << CLayout::name
              << " A_Type=" << ck_tile::DataTypeTraits<ADataType>::name
              << " B_Type=" << ck_tile::DataTypeTraits<BDataType>::name
              << " C_Type=" << ck_tile::DataTypeTraits<CDataType>::name
              << " StructuredSparsity=" << (GemmConfig::UseStructuredSparsity ? "on" : "off")
              << " Persistent=" << (persistent ? "on" : "off") << " : " << ave_time << " ms, "
              << tflops << " TFlops, " << gb_per_sec << " GB/s" << std::endl;

    return ave_time;
}

// Two-stage implementation of run_gemm_example_with_layouts
template <typename GemmConfig,
          typename ADataType,
          typename BDataType = ADataType,
          typename CDataType = ADataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
int run_gemm_example_with_layouts_two_stage(ck_tile::ArgParser& arg_parser,
                                            const ALayout a_layout                  = ALayout{},
                                            const BLayout b_layout                  = BLayout{},
                                            [[maybe_unused]] const CLayout c_layout = CLayout{})
{
    using AccDataType = typename GemmTypeConfig<ADataType, BDataType, CDataType>::AccDataType;

    ck_tile::index_t M = arg_parser.get_int("m");
    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t K = arg_parser.get_int("k");

    ck_tile::index_t stride_A = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_B = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_C = arg_parser.get_int("stride_c");

    ck_tile::index_t kbatch      = arg_parser.get_int("split_k");
    int n_warmup                 = arg_parser.get_int("warmup");
    int n_repeat                 = arg_parser.get_int("repeat");
    ck_tile::index_t init_method = arg_parser.get_int("init");
    bool persistent              = arg_parser.get_int("persistent");

    const bool preshuffle = GemmConfig::Preshuffle;

    stride_A = ck_tile::get_default_stride(M, K, stride_A, is_row_major(a_layout));
    stride_B = ck_tile::get_default_stride(K, N, stride_B, is_row_major(b_layout));
    stride_C = ck_tile::get_default_stride(M, N, stride_C, is_row_major(CLayout{}));

    ck_tile::HostTensor<ADataType> a_m_k(
        ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(a_layout)));
    ck_tile::HostTensor<BDataType> b_k_n(
        ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(b_layout)));
    ck_tile::HostTensor<CDataType> c_m_n_dev_result(
        ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

    if(init_method == 0)
    {
        if constexpr(preshuffle)
        {
            ck_tile::FillUniformDistribution<ADataType>{-.5f, .5f}(a_m_k);
            ck_tile::FillUniformDistribution<BDataType>{-.5f, .5f}(b_k_n);
        }
        else
        {
            ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_m_k);
            ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_k_n);
        }
    }
    else if(init_method == 1)
    {
        ck_tile::FillMonotonicSeq<ADataType>{}(a_m_k);
        ck_tile::FillMonotonicSeq<BDataType>{}(b_k_n);
    }
    else if(init_method == 2)
    {
        ck_tile::FillUniformDistribution<ADataType>{1.f, 1.f}(a_m_k);
        ck_tile::FillUniformDistribution<BDataType>{1.f, 1.f}(b_k_n);
    }
    else
    {
        a_m_k.SetZero();
        b_k_n.SetZero();
    }

    if(!preshuffle && GemmConfig::UseStructuredSparsity)
    {
        ck_tile::AdjustToStructuredSparsity<ADataType>{}(a_m_k);
    }

    ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

    static_assert(!GemmConfig::PermuteA, "Not implemented");

    if constexpr(preshuffle)
    {
        ck_tile::HostTensor<BDataType> b_shuffle_host = ck_tile::shuffle_b<GemmConfig>(b_k_n);
        // shuffled buffer B for device implementation
        b_k_n_dev_buf.ToDevice(b_shuffle_host.data());
    }
    else
    {
        if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
        {
            // Permute vector pk_i4x4 data for device implementation
            ck_tile::HostTensor<BDataType> b_k_n_dev = b_k_n;
            if constexpr(GemmConfig::PermuteB)
            {
                permute_tensor_b<GemmConfig,
                                 decltype(b_k_n_dev),
                                 ADataType,
                                 BDataType,
                                 AccDataType,
                                 CDataType,
                                 ALayout,
                                 BLayout,
                                 CLayout>(b_k_n_dev);
            }
            permute_vectors_i4x4_b(b_k_n_dev);
            b_k_n_dev_buf.ToDevice(b_k_n_dev.data());
        }
        else
        {
            if constexpr(GemmConfig::PermuteB)
            {
                std::cout << "Permute for this DataType is not implemented." << std::endl;
                return false;
            }
            b_k_n_dev_buf.ToDevice(b_k_n.data());
        }
    }

    a_m_k_dev_buf.ToDevice(a_m_k.data());
    c_m_n_dev_buf.SetZero();
    c_m_n_dev_result.SetZero();

    std::cout << "Using Workspace Split-K Mode (Two-Stage with Reduction)" << std::endl;
    // Use the new two-stage approach
    invoke_gemm_splitk_two_stage<GemmConfig,
                                 ADataType,
                                 BDataType,
                                 ck_tile::tuple<>,
                                 AccDataType,
                                 CDataType,
                                 ALayout,
                                 BLayout,
                                 ck_tile::tuple<>,
                                 CLayout>(a_m_k_dev_buf,
                                          b_k_n_dev_buf,
                                          c_m_n_dev_buf,
                                          M,
                                          N,
                                          K,
                                          stride_A,
                                          stride_B,
                                          stride_C,
                                          kbatch,
                                          n_warmup,
                                          n_repeat,
                                          persistent);

    c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());
    bool pass = true;

    if(arg_parser.get_int("v") == 1)
    {
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_m_k, b_k_n, c_m_n_host_ref);
        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, kbatch, max_accumulated_value);
        pass = ck_tile::check_err(c_m_n_dev_result,
                                  c_m_n_host_ref,
                                  "Error: Incorrect results!",
                                  rtol_atol.at(ck_tile::number<0>{}),
                                  rtol_atol.at(ck_tile::number<1>{}));

        std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                  << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                  << std::endl;
        std::cout << "The CPU verification result is:" << (pass ? "correct" : "fail") << std::endl;
    }
    else if(arg_parser.get_int("v") == 2)
    {
        if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
        {
            // Restore input for B for gpu reference
            b_k_n_dev_buf.ToDevice(b_k_n.data());
        }
        if constexpr(GemmConfig::Preshuffle)
        {
            b_k_n_dev_buf.ToDevice(b_k_n.data());
        }

        // memory on host to store gpu reference result
        ck_tile::HostTensor<CDataType> c_m_n_gpu_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        // memory on device to store gpu reference result
        ck_tile::DeviceMem c_m_n_gpu_buf_ref(c_m_n_gpu_ref.get_element_space_size_in_bytes());

        c_m_n_gpu_ref.SetZero();
        c_m_n_gpu_buf_ref.SetZero();

        ADataType* d_A = static_cast<ADataType*>(a_m_k_dev_buf.GetDeviceBuffer());
        BDataType* d_B = static_cast<BDataType*>(b_k_n_dev_buf.GetDeviceBuffer());
        CDataType* d_C = static_cast<CDataType*>(c_m_n_gpu_buf_ref.GetDeviceBuffer());

        ck_tile::reference_gemm_gpu<ADataType,
                                    BDataType,
                                    AccDataType,
                                    CDataType,
                                    ALayout,
                                    BLayout,
                                    CLayout>(d_A, d_B, d_C, M, N, K, stride_A, stride_B, stride_C);

        c_m_n_gpu_buf_ref.FromDevice(c_m_n_gpu_ref.data());

        const float max_accumulated_value =
            *std::max_element(c_m_n_gpu_ref.mData.begin(), c_m_n_gpu_ref.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, kbatch, max_accumulated_value);
        pass = ck_tile::check_err(c_m_n_dev_result,
                                  c_m_n_gpu_ref,
                                  "Error: Incorrect results!",
                                  rtol_atol.at(ck_tile::number<0>{}),
                                  rtol_atol.at(ck_tile::number<1>{}));
        std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                  << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                  << std::endl;
        std::cout << "The GPU verification result is: " << (pass ? "correct" : "fail") << std::endl;
    }

    return pass;
}

template <typename GemmConfig,
          typename APrecType,
          typename BPrecType = APrecType,
          typename CPrecType = APrecType>
int run_gemm_example_prec_type(std::string a_layout,
                               std::string b_layout,
                               ck_tile::ArgParser& arg_parser)
{
    using Row       = ck_tile::tensor_layout::gemm::RowMajor;
    using Col       = ck_tile::tensor_layout::gemm::ColumnMajor;
    bool preshuffle = GemmConfig::Preshuffle;

    if(preshuffle && std::is_same_v<BPrecType, ck_tile::pk_int4_t>)
    {
        throw std::runtime_error("Preshuffle is not supported for this int4 datatype!");
    }

    if(preshuffle && a_layout != "R" && b_layout != "C")
    {
        throw std::runtime_error(
            "Preshuffle is supported only for A(Row major), B(column major) input matrices!");
    }

    // Use new two-stage approach for both int4 and other data types
    if constexpr(std::is_same_v<BPrecType, ck_tile::pk_int4_t>)
    {
        if(a_layout == "R" && b_layout == "C")
        {
            return run_gemm_example_with_layouts_two_stage<GemmConfig,
                                                           APrecType,
                                                           BPrecType,
                                                           CPrecType,
                                                           Row,
                                                           Col,
                                                           Row>(arg_parser, Row{}, Col{}, Row{});
        }
        else if(a_layout == "C" && b_layout == "C")
        {
            return run_gemm_example_with_layouts_two_stage<GemmConfig,
                                                           APrecType,
                                                           BPrecType,
                                                           CPrecType,
                                                           Col,
                                                           Col,
                                                           Row>(arg_parser, Col{}, Col{}, Row{});
        }
        else
        {
            throw std::runtime_error("Unsupported memory layout for the input matrices when "
                                     "BPrecType is ck_tile::pk_int4_t!");
        }
    }
    else
    {
        if(a_layout == "R" && b_layout == "R")
        {
            return run_gemm_example_with_layouts_two_stage<GemmConfig,
                                                           APrecType,
                                                           BPrecType,
                                                           CPrecType>(
                arg_parser, Row{}, Row{}, Row{});
        }
        if(a_layout == "R" && b_layout == "C")
        {
            return run_gemm_example_with_layouts_two_stage<GemmConfig,
                                                           APrecType,
                                                           BPrecType,
                                                           CPrecType>(
                arg_parser, Row{}, Col{}, Row{});
        }
        else if(a_layout == "C" && b_layout == "R")
        {
            return run_gemm_example_with_layouts_two_stage<GemmConfig,
                                                           APrecType,
                                                           BPrecType,
                                                           CPrecType>(
                arg_parser, Col{}, Row{}, Row{});
        }
        else if(a_layout == "C" && b_layout == "C")
        {
            return run_gemm_example_with_layouts_two_stage<GemmConfig,
                                                           APrecType,
                                                           BPrecType,
                                                           CPrecType>(
                arg_parser, Col{}, Col{}, Row{});
        }
        else
        {
            throw std::runtime_error("Unsupported memory layout for the input matrices!");
        }
    }
    return 0;
}

template <template <typename PreType> typename GemmConfig>
int run_gemm_example(ck_tile::ArgParser& arg_parser)
{
    std::string data_type = arg_parser.get_str("prec");
    std::string a_layout  = arg_parser.get_str("a_layout");
    std::string b_layout  = arg_parser.get_str("b_layout");

    if(data_type == "fp16")
    {
        return run_gemm_example_prec_type<GemmConfig<ck_tile::half_t>, ck_tile::half_t>(
            a_layout, b_layout, arg_parser);
    }
    else if(data_type == "bf16")
    {
        return run_gemm_example_prec_type<GemmConfig<ck_tile::half_t>, ck_tile::bf16_t>(
            a_layout, b_layout, arg_parser);
    }
    else if(data_type == "fp8")
    {
        return run_gemm_example_prec_type<GemmConfig<ck_tile::fp8_t>,
                                          ck_tile::fp8_t,
                                          ck_tile::fp8_t,
                                          ck_tile::half_t>(a_layout, b_layout, arg_parser);
    }
    else if(data_type == "bf8")
    {
        return run_gemm_example_prec_type<GemmConfig<ck_tile::bf8_t>,
                                          ck_tile::bf8_t,
                                          ck_tile::bf8_t,
                                          ck_tile::half_t>(a_layout, b_layout, arg_parser);
    }
    else if(data_type == "int8")
    {
        return run_gemm_example_prec_type<GemmConfig<ck_tile::int8_t>,
                                          ck_tile::int8_t,
                                          ck_tile::int8_t,
                                          ck_tile::int32_t>(a_layout, b_layout, arg_parser);
    }
    else if(data_type == "pk_int4_t")
    {
        // TODO: Add support for bhalf_t ADataType
        if constexpr(GemmConfig<ck_tile::half_t>::Pipeline == ck_tile::GemmPipeline::COMPUTE_V3)
        {
            return run_gemm_example_prec_type<GemmConfig<ck_tile::half_t>,
                                              ck_tile::half_t,
                                              ck_tile::pk_int4_t,
                                              ck_tile::half_t>(a_layout, b_layout, arg_parser);
        }
        else
        {
            throw std::runtime_error("Unsupported pipeline for this operation !!!");
        }
    }
    else
    {
        throw std::runtime_error("Unsupported data type for this operation !!!");
    }
    return 0;
}

int main(int argc, char* argv[])
{
    auto arg_parser = create_args();
    auto result     = arg_parser.parse(argc, argv);

    if(!result)
        return -1;

    try
    {
#if CK_TILE_USE_WMMA
        return !run_gemm_example<GemmConfigComputeV3_WMMA>(arg_parser);
#else
        return !run_gemm_example<GemmConfigComputeV3>(arg_parser);
#endif
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Caught runtime error: " << e.what() << '\n';
        // Return a non-zero code to indicate failure
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
