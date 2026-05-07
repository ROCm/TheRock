// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
namespace ck_tile {

template <typename WarpGemmAttribute_>
struct WarpGemmImpl
{
    using WarpGemmAttribute = remove_cvref_t<WarpGemmAttribute_>;

    static constexpr index_t kM      = WarpGemmAttribute::kM;
    static constexpr index_t kN      = WarpGemmAttribute::kN;
    static constexpr index_t kK      = WarpGemmAttribute::kK;
    static constexpr index_t kCMLane = WarpGemmAttribute::kCMLane;
    /// @brief The number of elements in K dimension processed by single thread in wavefront.
    ///
    /// @note  Note that WarpGemm may run MFMA instruction multiple times (on different K).
    ///        In such situation this value reflects this fact.
    static constexpr index_t kKPerThread = WarpGemmAttribute::kKPerThread;

    using ADataType = typename WarpGemmAttribute::ADataType;
    using BDataType = typename WarpGemmAttribute::BDataType;
    using CDataType = typename WarpGemmAttribute::CDataType;

    using AWarpDstrEncoding = typename WarpGemmAttribute::AWarpDstrEncoding;
    using BWarpDstrEncoding = typename WarpGemmAttribute::BWarpDstrEncoding;
    using CWarpDstrEncoding = typename WarpGemmAttribute::CWarpDstrEncoding;

    using AWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(AWarpDstrEncoding{}))>;
    using BWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(BWarpDstrEncoding{}))>;
    using CWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(CWarpDstrEncoding{}))>;

    using AWarpTensor = static_distributed_tensor<ADataType, AWarpDstr>;
    using BWarpTensor = static_distributed_tensor<BDataType, BWarpDstr>;
    using CWarpTensor = static_distributed_tensor<CDataType, CWarpDstr>;

    CK_TILE_HOST_DEVICE static constexpr auto get_num_of_access()
    {
        return WarpGemmAttribute_::get_num_of_access();
    }

    template <typename CTensor, typename ATensor, typename BTensor, bool post_nop_ = false>
    CK_TILE_DEVICE void
    operator()(CTensor& c, const ATensor& a, const BTensor& b, bool_constant<post_nop_> = {}) const
    {
        static_assert(detail::is_similiar_distributed_tensor_v<CTensor, CWarpTensor> &&
                      detail::is_similiar_distributed_tensor_v<ATensor, AWarpTensor> &&
                      detail::is_similiar_distributed_tensor_v<BTensor, BWarpTensor>);
        using AVec = ext_vector_t<ADataType, ATensor::get_thread_buffer_size()>;
        using BVec = ext_vector_t<BDataType, BTensor::get_thread_buffer_size()>;
        using CVec = ext_vector_t<CDataType, CTensor::get_thread_buffer_size()>;

        constexpr auto I0 = number<0>{};

        const auto a_vec = a.get_thread_buffer().template get_as<AVec>()[I0];
        const auto b_vec = b.get_thread_buffer().template get_as<BVec>()[I0];
        auto c_vec       = c.get_thread_buffer().template get_as<CVec>()[I0];

        // c_vec += a_vec * b_vec
        WarpGemmAttribute{}(c_vec, a_vec, b_vec, bool_constant<post_nop_>{});

        c.get_thread_buffer().template set_as<CVec>(I0, c_vec);
    }

    template <typename CTensor,
              typename ATensor,
              typename BTensor,
              index_t i_subk,
              bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CTensor& c,
                                   const ATensor& a,
                                   const BTensor& b,
                                   number<i_subk>,
                                   bool_constant<post_nop_> = {}) const
    {
        using AVec = ext_vector_t<ADataType, ATensor::get_thread_buffer_size()>;
        using BVec = ext_vector_t<BDataType, BTensor::get_thread_buffer_size()>;
        using CVec = ext_vector_t<CDataType, CTensor::get_thread_buffer_size()>;

        constexpr auto I0 = number<0>{};

        const auto a_vec = a.get_thread_buffer().template get_as<AVec>()[I0];
        const auto b_vec = b.get_thread_buffer().template get_as<BVec>()[I0];
        auto c_vec       = c.get_thread_buffer().template get_as<CVec>()[I0];

        // c_vec += a_vec * b_vec
        WarpGemmAttribute{}(c_vec, a_vec, b_vec, number<i_subk>{}, bool_constant<post_nop_>{});

        c.get_thread_buffer().template set_as<CVec>(I0, c_vec);
    }

    template <index_t opselA,
              index_t opselB,
              typename CTensor,
              typename ATensor,
              typename BTensor,
              bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CTensor& c,
                                   const ATensor& a,
                                   const BTensor& b,
                                   const int32_t& a_scale,
                                   const int32_t& b_scale,
                                   bool_constant<post_nop_> = {}) const
    {
        static_assert(detail::is_similiar_distributed_tensor_v<CTensor, CWarpTensor> &&
                      detail::is_similiar_distributed_tensor_v<ATensor, AWarpTensor> &&
                      detail::is_similiar_distributed_tensor_v<BTensor, BWarpTensor>);
        using AVec = ext_vector_t<ADataType, ATensor::get_thread_buffer_size()>;
        using BVec = ext_vector_t<BDataType, BTensor::get_thread_buffer_size()>;
        using CVec = ext_vector_t<CDataType, CTensor::get_thread_buffer_size()>;

        constexpr auto I0 = number<0>{};

        const auto a_vec = a.get_thread_buffer().template get_as<AVec>()[I0];
        const auto b_vec = b.get_thread_buffer().template get_as<BVec>()[I0];
        auto c_vec       = c.get_thread_buffer().template get_as<CVec>()[I0];

        // c_vec += a_vec * b_vec
        WarpGemmAttribute{}.template operator()<opselA, opselB>(
            c_vec, a_vec, a_scale, b_vec, b_scale, bool_constant<post_nop_>{});

        c.get_thread_buffer().template set_as<CVec>(I0, c_vec);
    }

    template <typename ATensor, typename BTensor>
    CK_TILE_DEVICE auto operator()(const ATensor& a, const BTensor& b) const
    {
        using CTensor = CWarpTensor;
        static_assert(detail::is_similiar_distributed_tensor_v<ATensor, AWarpTensor> &&
                      detail::is_similiar_distributed_tensor_v<BTensor, BWarpTensor>);
        CTensor c;

        using AVec = ext_vector_t<ADataType, ATensor::get_thread_buffer_size()>;
        using BVec = ext_vector_t<BDataType, BTensor::get_thread_buffer_size()>;
        using CVec = ext_vector_t<CDataType, CTensor::get_thread_buffer_size()>;

        constexpr auto I0 = number<0>{};

        const auto a_vec = a.get_thread_buffer().template get_as<AVec>()[I0];
        const auto b_vec = b.get_thread_buffer().template get_as<BVec>()[I0];

        // c_vec = a_vec * b_vec
        auto c_vec = WarpGemmAttribute{}(a_vec, b_vec);

        c.get_thread_buffer().template set_as<CVec>(I0, c_vec);

        return c;
    }

    template <index_t opselA, index_t opselB, typename ATensor, typename BTensor>
    CK_TILE_DEVICE auto operator()(const ATensor& a,
                                   const BTensor& b,
                                   const int32_t& a_scale,
                                   const int32_t& b_scale) const
    {
        using CTensor = CWarpTensor;
        static_assert(detail::is_similiar_distributed_tensor_v<ATensor, AWarpTensor> &&
                      detail::is_similiar_distributed_tensor_v<BTensor, BWarpTensor>);
        CTensor c;

        using AVec = ext_vector_t<ADataType, ATensor::get_thread_buffer_size()>;
        using BVec = ext_vector_t<BDataType, BTensor::get_thread_buffer_size()>;
        using CVec = ext_vector_t<CDataType, CTensor::get_thread_buffer_size()>;

        constexpr auto I0 = number<0>{};

        const auto a_vec = a.get_thread_buffer().template get_as<AVec>()[I0];
        const auto b_vec = b.get_thread_buffer().template get_as<BVec>()[I0];

        // c_vec = a_vec * b_vec
        auto c_vec =
            WarpGemmAttribute{}.template operator()<opselA, opselB>(a_vec, a_scale, b_vec, b_scale);

        c.get_thread_buffer().template set_as<CVec>(I0, c_vec);

        return c;
    }
};

} // namespace ck_tile
