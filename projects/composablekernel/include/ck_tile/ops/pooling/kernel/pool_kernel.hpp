// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/pooling/pipeline/pool_default_policy.hpp"
#include "ck_tile/ops/common.hpp"
#include <type_traits>

namespace ck_tile {

/// @brief Host arguments for pooling operations
template <typename TensorShape, typename WindowShape>
struct PoolHostArgs
{

    CK_TILE_HOST PoolHostArgs(const void* input_ptr_,
                              void* output_ptr_,
                              void* output_index_ptr_,
                              TensorShape input_shape_,
                              TensorShape output_shape_,
                              TensorShape input_strides_,
                              TensorShape output_strides_,
                              WindowShape window_lengths_,
                              WindowShape window_strides_,
                              WindowShape window_dilations_,
                              WindowShape input_left_pads_,
                              WindowShape input_right_pads_)
        : input_ptr(input_ptr_),
          output_ptr(output_ptr_),
          output_index_ptr(output_index_ptr_),
          input_shape(input_shape_),
          output_shape(output_shape_),
          input_strides(input_strides_),
          output_strides(output_strides_),
          window_lengths(window_lengths_),
          window_strides(window_strides_),
          window_dilations(window_dilations_),
          input_left_pads(input_left_pads_),
          input_right_pads(input_right_pads_)
    {
    }

    const void* input_ptr;
    void* output_ptr;
    void* output_index_ptr;

    TensorShape input_shape;
    TensorShape output_shape;
    TensorShape input_strides;
    TensorShape output_strides;
    WindowShape window_lengths;
    WindowShape window_strides;
    WindowShape window_dilations;
    WindowShape input_left_pads;
    WindowShape input_right_pads;
};

/// @brief Kernel arguments for pooling operations
template <typename TensorShape, typename WindowShape>
struct PoolKernelArgs
{
    const void* input_ptr;
    void* output_ptr;
    void* output_index_ptr;
    TensorShape input_shape;
    TensorShape output_shape;
    TensorShape input_strides;
    TensorShape output_strides;
    WindowShape window_lengths;
    WindowShape window_strides;
    WindowShape window_dilations;
    WindowShape input_left_pads;
    WindowShape input_right_pads;
};

template <typename Problem_, typename Policy_ = PoolDefaultPolicy>
struct PoolKernel
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using InDataType      = ck_tile::remove_cvref_t<typename Problem::InDataType>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using OutDataType     = ck_tile::remove_cvref_t<typename Problem::OutDataType>;
    using IndexDataType   = ck_tile::remove_cvref_t<typename Problem::IndexDataType>;

    static constexpr index_t kBlockSize = Problem::BlockShape::BlockSize;

    CK_TILE_HOST static constexpr auto BlockSize()
    {
        return is_wave32() ? kBlockSize / 2 : kBlockSize;
    }

    template <typename TensorShape, typename WindowShape>
    static CK_TILE_DEVICE auto MakeTensorView2D(PoolKernelArgs<TensorShape, WindowShape> kargs)
    {
        using S = typename Problem::BlockShape;

        // Compile-time validation for 2D pooling
        static_assert(TensorShape::size() == 4, "2D pooling requires 4D input tensor (N,H,W,C)");
        static_assert(WindowShape::size() == 2, "2D pooling requires 2D window shape (Y,X)");

        // Extract dimension values
        const index_t N = kargs.input_shape.at(number<0>{});
        const index_t H = kargs.input_shape.at(number<1>{});
        const index_t W = kargs.input_shape.at(number<2>{});
        const index_t C = kargs.input_shape.at(number<3>{});

        const index_t No = kargs.output_shape.at(number<0>{});
        const index_t Ho = kargs.output_shape.at(number<1>{});
        const index_t Wo = kargs.output_shape.at(number<2>{});
        const index_t Co = kargs.output_shape.at(number<3>{});

        const index_t Y = kargs.window_lengths.at(number<0>{});
        const index_t X = kargs.window_lengths.at(number<1>{});

        const index_t WindowStrideH = kargs.window_strides.at(number<0>{});
        const index_t WindowStrideW = kargs.window_strides.at(number<1>{});

        const index_t WindowDilationH = kargs.window_dilations.at(number<0>{});
        const index_t WindowDilationW = kargs.window_dilations.at(number<1>{});

        const index_t InLeftPadH = kargs.input_left_pads.at(number<0>{});
        const index_t InLeftPadW = kargs.input_left_pads.at(number<1>{});

        const index_t InRightPadH = kargs.input_right_pads.at(number<0>{});
        const index_t InRightPadW = kargs.input_right_pads.at(number<1>{});

        const index_t MRaw = N * Ho * Wo * C;
        const index_t KRaw = Y * X;
        const index_t MPad = integer_least_multiple(MRaw, S::Block_M) - MRaw;
        const index_t KPad = integer_least_multiple(KRaw, S::Block_N) - KRaw;

        auto reduce_op = typename Problem::ReduceOp{};

        // Create input descriptor with all transformations
        auto in_desc = make_naive_tensor_descriptor(kargs.input_shape, kargs.input_strides);

        // Apply spatial padding to input descriptor
        const auto padded_in_desc = transform_tensor_descriptor(
            in_desc,
            make_tuple(make_pass_through_transform(N),
                       make_pad_transform(H, InLeftPadH, InRightPadH),
                       make_pad_transform(W, InLeftPadW, InRightPadW),
                       make_pass_through_transform(C)),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));

        // Create sliding windows by embedding pooling windows into descriptor
        const auto embed_in_desc = transform_tensor_descriptor(
            padded_in_desc,
            make_tuple(
                make_pass_through_transform(N),
                make_embed_transform(make_tuple(Y, Ho), make_tuple(WindowDilationH, WindowStrideH)),
                make_embed_transform(make_tuple(X, Wo), make_tuple(WindowDilationW, WindowStrideW)),
                make_pass_through_transform(C)),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
            make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3, 4>{}, sequence<5>{}));

        // Reshape into 2D matrix: output positions (M) x pooling window elements (K)
        const auto merged_embed_in_desc =
            transform_tensor_descriptor(embed_in_desc,
                                        make_tuple(make_merge_transform(make_tuple(N, Ho, Wo, C)),
                                                   make_merge_transform(make_tuple(Y, X))),
                                        make_tuple(sequence<0, 2, 4, 5>{}, sequence<1, 3>{}),
                                        make_tuple(sequence<0>{}, sequence<1>{}));

        const auto in_desc_padded = transform_tensor_descriptor(
            merged_embed_in_desc,
            make_tuple(make_right_pad_transform(MRaw, MPad), make_right_pad_transform(KRaw, KPad)),
            make_tuple(sequence<0>{}, sequence<1>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        // Create output descriptor with transformations
        auto out_desc = make_naive_tensor_descriptor(kargs.output_shape, kargs.output_strides);

        const auto merged_out_desc = transform_tensor_descriptor(
            out_desc,
            make_tuple(make_merge_transform(make_tuple(No, Ho, Wo, Co))),
            make_tuple(sequence<0, 1, 2, 3>{}),
            make_tuple(sequence<0>{}));

        const auto out_desc_padded =
            transform_tensor_descriptor(merged_out_desc,
                                        make_tuple(make_right_pad_transform(MRaw, MPad)),
                                        make_tuple(sequence<0>{}),
                                        make_tuple(sequence<0>{}));

        // Now create buffer views and tensor views with the fully transformed descriptors
        const InDataType in_identity =
            type_convert<InDataType>(reduce_op.template GetIdentityValue<ComputeDataType>());
        const OutDataType out_identity =
            type_convert<OutDataType>(reduce_op.template GetIdentityValue<ComputeDataType>());

        auto in_buffer_view = make_buffer_view<address_space_enum::global>(
            static_cast<const InDataType*>(kargs.input_ptr),
            in_desc.get_element_space_size(),
            in_identity);
        const auto in_tensor_padded =
            tensor_view<decltype(in_buffer_view), decltype(in_desc_padded)>{in_buffer_view,
                                                                            in_desc_padded};

        auto out_buffer_view = make_buffer_view<address_space_enum::global>(
            static_cast<OutDataType*>(kargs.output_ptr),
            out_desc.get_element_space_size(),
            out_identity);
        const auto out_tensor_padded =
            tensor_view<decltype(out_buffer_view), decltype(out_desc_padded)>{out_buffer_view,
                                                                              out_desc_padded};

        if constexpr(Problem::kOutputIndex)
        {
            auto out_index_buffer_view = make_buffer_view<address_space_enum::global>(
                static_cast<IndexDataType*>(kargs.output_index_ptr),
                out_desc.get_element_space_size(),
                IndexDataType(-1));
            const auto out_index_tensor_padded =
                tensor_view<decltype(out_index_buffer_view), decltype(out_desc_padded)>{
                    out_index_buffer_view, out_desc_padded};

            return make_tuple(in_tensor_padded, out_tensor_padded, out_index_tensor_padded);
        }
        else
        {
            // Return a dummy tensor for the third element when index output is not needed
            return make_tuple(in_tensor_padded, out_tensor_padded, null_tensor{});
        }
    }

    template <typename TensorShape, typename WindowShape>
    static CK_TILE_DEVICE auto MakeTensorView3D(PoolKernelArgs<TensorShape, WindowShape> kargs)
    {
        using S = typename Problem::BlockShape;

        // Compile-time validation for 3D pooling
        static_assert(TensorShape::size() == 5, "3D pooling requires 5D input tensor (N,D,H,W,C)");
        static_assert(WindowShape::size() == 3, "3D pooling requires 3D window shape (Z,Y,X)");

        // Extract dimension values
        const index_t N = kargs.input_shape.at(number<0>{});
        const index_t D = kargs.input_shape.at(number<1>{});
        const index_t H = kargs.input_shape.at(number<2>{});
        const index_t W = kargs.input_shape.at(number<3>{});
        const index_t C = kargs.input_shape.at(number<4>{});

        const index_t No = kargs.output_shape.at(number<0>{});
        const index_t Do = kargs.output_shape.at(number<1>{});
        const index_t Ho = kargs.output_shape.at(number<2>{});
        const index_t Wo = kargs.output_shape.at(number<3>{});
        const index_t Co = kargs.output_shape.at(number<4>{});

        const index_t Z = kargs.window_lengths.at(number<0>{});
        const index_t Y = kargs.window_lengths.at(number<1>{});
        const index_t X = kargs.window_lengths.at(number<2>{});

        const index_t WindowStrideD = kargs.window_strides.at(number<0>{});
        const index_t WindowStrideH = kargs.window_strides.at(number<1>{});
        const index_t WindowStrideW = kargs.window_strides.at(number<2>{});

        const index_t WindowDilationD = kargs.window_dilations.at(number<0>{});
        const index_t WindowDilationH = kargs.window_dilations.at(number<1>{});
        const index_t WindowDilationW = kargs.window_dilations.at(number<2>{});

        const index_t InLeftPadD = kargs.input_left_pads.at(number<0>{});
        const index_t InLeftPadH = kargs.input_left_pads.at(number<1>{});
        const index_t InLeftPadW = kargs.input_left_pads.at(number<2>{});

        const index_t InRightPadD = kargs.input_right_pads.at(number<0>{});
        const index_t InRightPadH = kargs.input_right_pads.at(number<1>{});
        const index_t InRightPadW = kargs.input_right_pads.at(number<2>{});

        const index_t MRaw = N * Do * Ho * Wo * C;
        const index_t KRaw = Z * Y * X;
        const index_t MPad = integer_least_multiple(MRaw, S::Block_M) - MRaw;
        const index_t KPad = integer_least_multiple(KRaw, S::Block_N) - KRaw;

        auto reduce_op = typename Problem::ReduceOp{};

        // Create input descriptor with all transformations
        auto in_desc = make_naive_tensor_descriptor(kargs.input_shape, kargs.input_strides);

        // Apply spatial padding to input descriptor (all 3D dimensions)
        const auto padded_in_desc = transform_tensor_descriptor(
            in_desc,
            make_tuple(make_pass_through_transform(N),
                       make_pad_transform(D, InLeftPadD, InRightPadD),
                       make_pad_transform(H, InLeftPadH, InRightPadH),
                       make_pad_transform(W, InLeftPadW, InRightPadW),
                       make_pass_through_transform(C)),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}));

        // Create 3D sliding windows by embedding pooling windows into descriptor
        const auto embed_in_desc = transform_tensor_descriptor(
            padded_in_desc,
            make_tuple(
                make_pass_through_transform(N),
                make_embed_transform(make_tuple(Z, Do), make_tuple(WindowDilationD, WindowStrideD)),
                make_embed_transform(make_tuple(Y, Ho), make_tuple(WindowDilationH, WindowStrideH)),
                make_embed_transform(make_tuple(X, Wo), make_tuple(WindowDilationW, WindowStrideW)),
                make_pass_through_transform(C)),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}),
            make_tuple(sequence<0>{},
                       sequence<1, 2>{},
                       sequence<3, 4>{},
                       sequence<5, 6>{},
                       sequence<7>{}));

        // Reshape into 2D matrix: output positions (M) x pooling window elements (K)
        const auto merged_embed_in_desc = transform_tensor_descriptor(
            embed_in_desc,
            make_tuple(make_merge_transform(make_tuple(N, Do, Ho, Wo, C)),
                       make_merge_transform(make_tuple(Z, Y, X))),
            make_tuple(sequence<0, 2, 4, 6, 7>{}, sequence<1, 3, 5>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        const auto in_desc_padded = transform_tensor_descriptor(
            merged_embed_in_desc,
            make_tuple(make_right_pad_transform(MRaw, MPad), make_right_pad_transform(KRaw, KPad)),
            make_tuple(sequence<0>{}, sequence<1>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        // Create output descriptor with transformations
        auto out_desc = make_naive_tensor_descriptor(kargs.output_shape, kargs.output_strides);

        const auto merged_out_desc = transform_tensor_descriptor(
            out_desc,
            make_tuple(make_merge_transform(make_tuple(No, Do, Ho, Wo, Co))),
            make_tuple(sequence<0, 1, 2, 3, 4>{}),
            make_tuple(sequence<0>{}));

        const auto out_desc_padded =
            transform_tensor_descriptor(merged_out_desc,
                                        make_tuple(make_right_pad_transform(MRaw, MPad)),
                                        make_tuple(sequence<0>{}),
                                        make_tuple(sequence<0>{}));

        // Now create buffer views and tensor views with the fully transformed descriptors
        const InDataType in_identity =
            type_convert<InDataType>(reduce_op.template GetIdentityValue<ComputeDataType>());
        const OutDataType out_identity =
            type_convert<OutDataType>(reduce_op.template GetIdentityValue<ComputeDataType>());

        auto in_buffer_view = make_buffer_view<address_space_enum::global>(
            static_cast<const InDataType*>(kargs.input_ptr),
            in_desc.get_element_space_size(),
            in_identity);
        const auto in_tensor_padded =
            tensor_view<decltype(in_buffer_view), decltype(in_desc_padded)>{in_buffer_view,
                                                                            in_desc_padded};

        auto out_buffer_view = make_buffer_view<address_space_enum::global>(
            static_cast<OutDataType*>(kargs.output_ptr),
            out_desc.get_element_space_size(),
            out_identity);
        const auto out_tensor_padded =
            tensor_view<decltype(out_buffer_view), decltype(out_desc_padded)>{out_buffer_view,
                                                                              out_desc_padded};

        if constexpr(Problem::kOutputIndex)
        {
            auto out_index_buffer_view = make_buffer_view<address_space_enum::global>(
                static_cast<IndexDataType*>(kargs.output_index_ptr),
                out_desc.get_element_space_size(),
                IndexDataType(-1));
            const auto out_index_tensor_padded =
                tensor_view<decltype(out_index_buffer_view), decltype(out_desc_padded)>{
                    out_index_buffer_view, out_desc_padded};

            return make_tuple(in_tensor_padded, out_tensor_padded, out_index_tensor_padded);
        }
        else
        {
            // Return a dummy tensor for the third element when index output is not needed
            return make_tuple(in_tensor_padded, out_tensor_padded, null_tensor{});
        }
    }

    public:
    template <typename TensorShape, typename WindowShape>
    CK_TILE_DEVICE void operator()(PoolKernelArgs<TensorShape, WindowShape> kargs) const
    {
        using S = typename Problem::BlockShape;

        // Compile-time validation for supported window dimensions
        static_assert(WindowShape::size() == 2 || WindowShape::size() == 3,
                      "Only 2D and 3D pooling operations are supported");

        const auto iM = get_block_id() * S::Block_M;

        // Get tensors based on dimensionality
        auto [in_tensor_padded, out_tensor_padded, out_index_tensor_padded] = [&]() {
            if constexpr(WindowShape::size() == 2)
                return MakeTensorView2D(kargs);
            else if constexpr(WindowShape::size() == 3)
                return MakeTensorView3D(kargs);
            else
                static_assert(WindowShape::size() == 2 || WindowShape::size() == 3,
                              "Unsupported WindowShape rank: only 2D or 3D pooling is supported");
        }();

        auto reduce_op = typename Problem::ReduceOp{};

        auto x_window = make_tile_window(in_tensor_padded,
                                         make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                         {iM, 0},
                                         Policy::template MakeXBlockTileDistribution<Problem>());
        auto y_window = make_tile_window(out_tensor_padded, make_tuple(number<S::Block_M>{}), {iM});

        __shared__ char smem[Policy::template GetSmemSize<Problem>()];

        const auto reduce_len =
            in_tensor_padded.get_tensor_descriptor().get_lengths().at(number<1>{});
        index_t num_k_tiles =
            __builtin_amdgcn_readfirstlane(integer_divide_ceil(reduce_len, S::Block_N));

        auto block_reduce2d            = Policy::template GetBlockReduce2d<Problem>();
        auto block_reduce2d_sync       = Policy::template GetBlockReduce2dSync<Problem>();
        auto block_reduce2d_cross_warp = Policy::template GetBlockReduce2dCrossWarpSync<Problem>();

        using XTensorTile = decltype(load_tile(x_window));
        auto y_tile       = block_reduce2d.template MakeYBlockTile<XTensorTile>();
        set_tile(y_tile, reduce_op.template GetIdentityValue<ComputeDataType>());

        if constexpr(Problem::kOutputIndex)
        {
            auto y_index_window =
                make_tile_window(out_index_tensor_padded, make_tuple(number<S::Block_M>{}), {iM});

            auto y_index_tile =
                block_reduce2d.template MakeYIndexBlockTile<XTensorTile, IndexDataType>();
            set_tile(y_index_tile, IndexDataType(0));

            // Main reduction loop - with index tracking
            for(int k_tile = amd_wave_read_first_lane(0); k_tile < num_k_tiles; ++k_tile)
            {
                const auto x_tile = load_tile(x_window);
                const auto& in_tensor_padded_ref =
                    in_tensor_padded; // structured bindings cannot be captured prior to cpp20
                auto index_calculator = [&](const auto& x_indices) {
                    // Get global coordinates in the 2D matrix space (M, N)
                    const auto global_M = x_indices.at(number<0>{}) + iM;
                    const auto global_N = (k_tile * S::Block_N) + x_indices.at(number<1>{});
                    return in_tensor_padded_ref.get_tensor_descriptor().calculate_offset(
                        make_tuple(global_M, global_N));
                };

                block_reduce2d(x_tile, y_tile, y_index_tile, reduce_op, index_calculator);
                move_tile_window(x_window, {0, S::Block_N});
            }

            block_reduce2d_sync(y_tile, y_index_tile, reduce_op);
            if constexpr(Problem::kNeedCrossWarpSync)
            {
                __shared__ char smem_indices[Policy::template GetIndicesSmemSize<Problem>()];

                block_reduce2d_cross_warp(y_tile, y_index_tile, smem, smem_indices, reduce_op);
            }

            store_tile(y_window, cast_tile<OutDataType>(y_tile));
            store_tile(y_index_window, cast_tile<IndexDataType>(y_index_tile));
        }
        else
        {
            // Main reduction loop - without index tracking
            for(int k_tile = __builtin_amdgcn_readfirstlane(0); k_tile < num_k_tiles; ++k_tile)
            {
                const auto x_tile = load_tile(x_window);
                block_reduce2d(x_tile, y_tile, reduce_op);
                move_tile_window(x_window, {0, S::Block_N});
            }

            block_reduce2d_sync(y_tile, reduce_op);
            block_reduce2d_cross_warp(y_tile, smem, reduce_op);

            store_tile(y_window, cast_tile<OutDataType>(y_tile));
        }
    }

    /// @brief Validates if the given arguments are supported by the pooling kernel.
    ///
    /// @param kargs The pooling kernel arguments containing all necessary parameters.
    ///
    /// @return true if the arguments are supported, false otherwise.
    ///
    /// @note Requirements:
    ///       - Last dimension (C) must be contiguous (stride = 1) for vectorized access
    ///       - Window dimensions must be supported (2D or 3D)
    ///       - All dimension sizes must be consistent between input and output
    template <typename TensorShape, typename WindowShape>
    CK_TILE_HOST static bool IsSupportedArgument(PoolKernelArgs<TensorShape, WindowShape> kargs)
    {
        constexpr index_t InputRank  = TensorShape::size();
        constexpr index_t OutputRank = TensorShape::size(); // Same as input rank
        constexpr index_t WindowRank = WindowShape::size();

        // Validate window dimensions (only 2D and 3D supported)
        if constexpr(WindowRank != 2 && WindowRank != 3)
        {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR("Only 2D and 3D pooling are supported!");
            }
            return false;
        }

        // Validate that input rank matches expected rank for window dimensions
        if constexpr((WindowRank == 2 && InputRank != 4) || (WindowRank == 3 && InputRank != 5))
        {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR("Input tensor rank doesn't match window dimensions!");
            }
            return false;
        }

        // Check that channel dimension (last dimension) is contiguous for both input and output
        if(kargs.input_strides.at(number<InputRank - 1>{}) != 1)
        {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR("Input tensor's channel dimension must have stride 1!");
            }
            return false;
        }

        if(kargs.output_strides.at(number<OutputRank - 1>{}) != 1)
        {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR("Output tensor's channel dimension must have stride 1!");
            }
            return false;
        }

        return true;
    }

    /// @param kargs The pooling kernel arguments
    /// @return The calculated grid size
    template <typename TensorShape, typename WindowShape>
    CK_TILE_HOST static constexpr index_t
    CalculateGridSize(PoolKernelArgs<TensorShape, WindowShape> kargs)
    {
        using S = typename Problem::BlockShape;

        // Calculate total output elements (M dimension)
        index_t M = 1;
        static_for<0, TensorShape::size(), 1>{}([&](auto i) { M *= kargs.output_shape.at(i); });

        // Calculate grid size: ceil(M / Block_M)
        return (M + S::Block_M - 1) / S::Block_M;
    }

    /// @brief Create kernel arguments from host arguments
    template <typename TensorShape, typename WindowShape>
    CK_TILE_HOST static constexpr auto
    MakeKernelArgs(PoolHostArgs<TensorShape, WindowShape>& host_args)
    {
        return PoolKernelArgs<TensorShape, WindowShape>{host_args.input_ptr,
                                                        host_args.output_ptr,
                                                        host_args.output_index_ptr,
                                                        host_args.input_shape,
                                                        host_args.output_shape,
                                                        host_args.input_strides,
                                                        host_args.output_strides,
                                                        host_args.window_lengths,
                                                        host_args.window_strides,
                                                        host_args.window_dilations,
                                                        host_args.input_left_pads,
                                                        host_args.input_right_pads};
    }
};

} // namespace ck_tile
