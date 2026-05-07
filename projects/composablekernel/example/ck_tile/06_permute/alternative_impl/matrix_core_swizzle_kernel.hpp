// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"

// if set to 1, slightly more instructions generated to calculate address
#ifndef MERGE_2D_013425
#define MERGE_2D_013425 0
#endif

enum class matrix_core_inst_enum
{
    MFMA_32x32x8_F16  = 0,
    MFMA_16x16x16_F16 = 1,
};

namespace detail {
template <matrix_core_inst_enum>
struct to_warp_gemm;

template <>
struct to_warp_gemm<matrix_core_inst_enum::MFMA_32x32x8_F16>
{
    using type = ck_tile::WarpGemmMfmaF16F16F32M32N32K8;
};

template <>
struct to_warp_gemm<matrix_core_inst_enum::MFMA_16x16x16_F16>
{
    using type = ck_tile::WarpGemmMfmaF16F16F32M16N16K16;
};
} // namespace detail
template <matrix_core_inst_enum Inst>
using to_warp_gemm_t = typename detail::to_warp_gemm<Inst>::type;

// TODO: in below permute pattern, the last 3 dim is within wave
enum class matrix_core_permute_style
{
    permute_b_n0_k0_n1_k1_n2_k2 = 0, // 0,1,4,2,5,3,6
    permute_b_n0_n1_k0_k1_n2_k2 = 1, // 0,1,2,4,5,3,6
    b_nr_kr_kw_nw_kv            = 2, // 0,1,3,4,2,5
    b_nr_kr_waveflatten         = b_nr_kr_kw_nw_kv,
};

// assume this is B matrix, originally we have batch*n*k
// now batch* n0*n1*n2*k0*k1*k2 -> batch* n0*k0*n1*k1*n2*k2
// assume using 32x32x8-f16, 4 waves and extend the KPerLane to 8xfp16(dwordx4)
//
//                                      4(waves)  32(mfma_m lane)
//                                          |      |
// batch* n0*n1*n2*k0*k1*k2 -> batch* n0*k0*n1*k1*n2*k2 -> 8(thread loading)
//                                    nr  kr    |
//        nr  4  32 kr 2  8                     2(klane)
//
// permute: 0,1,4,2,5,3,6
// or
// batch* n0*n1*n2*k0*k1*k2 -> batch* n0*n1*k0*k1*n2*k2 -> 8(thread loading)
// permute: 0,1,2,4,5,3,6
//
// this kernel only deal with fp16/bf16 data(16bit), and use 2d block size to do the swizzling
// for simplicity, only consider n/k is multiple of block-size

// independend host arg with no template
struct matrix_core_swizzle_host_args
{
    const void* p_src;
    void* p_dst;
    int32_t batch;
    int32_t n;
    int32_t k;
};

// NOTE: this kernel could follow the style of generic permute kernel
// but here we pass in fixed layout as template arg and generate different kernel instance
// purposely
template <int BLOCK_SIZE_ = 256,
          int NPerBlock_  = 256,
          int KPerBlock_  = 128,
          matrix_core_permute_style pstyle_ =
              matrix_core_permute_style::permute_b_n0_k0_n1_k1_n2_k2,
          matrix_core_inst_enum Inst_ = matrix_core_inst_enum::MFMA_32x32x8_F16>
struct matrix_core_swizzle_kernel
{
    using karg = matrix_core_swizzle_host_args;
    using harg = matrix_core_swizzle_host_args;

    static constexpr int BLOCK_SIZE                   = BLOCK_SIZE_;
    static constexpr int WavesPerBlock_N              = BLOCK_SIZE / ck_tile::get_warp_size();
    static constexpr int WavesPerBlock_K              = 1;
    static constexpr int NPerBlock                    = NPerBlock_;
    static constexpr int KPerBlock                    = KPerBlock_;
    static constexpr matrix_core_permute_style pstyle = pstyle_;
    static constexpr matrix_core_inst_enum Inst       = Inst_;

    static constexpr ck_tile::index_t Alignment = 8;
    karg a;
    dim3 grids;

    using WarpGemm = to_warp_gemm_t<Inst>;

    __host__ matrix_core_swizzle_kernel(harg h)
    {
        a                   = h;
        ck_tile::index_t ns = (h.n + NPerBlock - 1) / NPerBlock;
        ck_tile::index_t ks = (h.k + KPerBlock - 1) / KPerBlock;
        grids               = dim3(ks, ns, h.batch);
    }

    __host__ bool is_applicable(harg h) { return h.n % NPerBlock == 0 && h.k % KPerBlock == 0; }

    __host__ void operator()(const ck_tile::stream_config& s) const
    {
        ck_tile::kentry<1, kernel><<<grids, BLOCK_SIZE, 0, s.stream_id_>>>(a);
    }

    struct kernel
    {
        static constexpr int kBlockSize = BLOCK_SIZE;
        __device__ static constexpr auto get_src_dist()
        {
            using namespace ck_tile;
            constexpr index_t K2 = Alignment;
            constexpr index_t N2 = WarpGemm::WarpGemmAttribute::Impl::kAMLane;
            constexpr index_t K1 = WarpGemm::WarpGemmAttribute::Impl::kABKLane;
            constexpr index_t N1 = BLOCK_SIZE / get_warp_size();

            static_assert(NPerBlock % (N1 * N2) == 0);
            static_assert(KPerBlock % (K1 * K2) == 0);

            constexpr index_t K0 = KPerBlock / (K1 * K2);
            constexpr index_t N0 = NPerBlock / (N1 * N2);

            // clang-format off
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<1>,// 0
                    //             1              2            3             4             5             6
                    tuple<sequence<N0>, sequence<N1>, sequence<N2>, sequence<K0>, sequence<K1>, sequence<K2>>,

                    //            N1           K1  N2
                    tuple<sequence<2>, sequence<5, 3>>,
                    tuple<sequence<0>, sequence<0, 0>>,

                    //       N0 K0 K2
                    sequence<1, 4, 6>,
                    sequence<0, 0, 0>>{});
            // clang-format on
        }
        __device__ static constexpr auto get_dst_dist()
        {
            using namespace ck_tile;
            constexpr index_t K2 = Alignment;
            constexpr index_t N2 = WarpGemm::WarpGemmAttribute::Impl::kAMLane;
            constexpr index_t K1 = WarpGemm::WarpGemmAttribute::Impl::kABKLane;
            constexpr index_t N1 = BLOCK_SIZE / get_warp_size();

            static_assert(NPerBlock % (N1 * N2) == 0);
            static_assert(KPerBlock % (K1 * K2) == 0);

            constexpr index_t K0 = KPerBlock / (K1 * K2);
            constexpr index_t N0 = NPerBlock / (N1 * N2);

            if constexpr(pstyle == matrix_core_permute_style::permute_b_n0_k0_n1_k1_n2_k2)
            {
                // clang-format off
                return make_static_tile_distribution(
                    tile_distribution_encoding<
                        sequence<1>,// 0
                        //             1              2            3             4             5             6
                        tuple<sequence<N0>, sequence<K0>, sequence<N1>, sequence<K1>, sequence<N2>, sequence<K2>>,

                        //            N1           K1  N2
                        tuple<sequence<3>, sequence<4, 5>>,
                        tuple<sequence<0>, sequence<0, 0>>,

                        //       N0 K0 K2
                        sequence<1, 2, 6>,
                        sequence<0, 0, 0>>{});
                // clang-format on
            }
            else if constexpr(pstyle == matrix_core_permute_style::permute_b_n0_n1_k0_k1_n2_k2)
            {
                // clang-format off
                return make_static_tile_distribution(
                    tile_distribution_encoding<
                        sequence<1>,// 0
                        //             1              2            3             4             5             6
                        tuple<sequence<N0>, sequence<N1>, sequence<K0>, sequence<K1>, sequence<N2>, sequence<K2>>,

                        //            N1           K1  N2
                        tuple<sequence<2>, sequence<4, 5>>,
                        tuple<sequence<0>, sequence<0, 0>>,

                        //       N0 K0 K2
                        sequence<1, 3, 6>,
                        sequence<0, 0, 0>>{});
                // clang-format on
            }
            else
            {
                // clang-format off
                // b_nr_kr_kw_nw_kv or b_nr_kr_waveflatten
                constexpr index_t Kv = Alignment;
                constexpr index_t Nw = WarpGemm::WarpGemmAttribute::Impl::kAMLane;
                constexpr index_t Kw = WarpGemm::WarpGemmAttribute::Impl::kABKLane;

                static_assert(KPerBlock % (K1 * K2) == 0);
                constexpr index_t Nr = NPerBlock / Nw;
                constexpr index_t Kr = KPerBlock / (Kv * Kw);

                constexpr index_t Nr_p = WavesPerBlock_N;
                constexpr index_t Kr_p = WavesPerBlock_K;
                constexpr index_t Nr_y = Nr / Nr_p;
                constexpr index_t Kr_y = Kr / Kr_p;

                return make_static_tile_distribution(
#if MERGE_2D_013425
                    tile_distribution_encoding<
                        sequence<1>,// 0    R
                        // major       1                         2
                        // minor       0     1     2             0     1     2   3
                        tuple<sequence<Nr_y, Nr_p, Nw>, sequence<Kr_y, Kr_p, Kw, Kv>>,    // H

                        //            Nr_p, Kr_p         Kw Nw
                        tuple<sequence<1  , 2>, sequence<2, 1>>,    // p major
                        tuple<sequence<1  , 1>, sequence<2, 2>>,    // p minor

                        //       Nr_y Kr_y Kv
                        sequence<1,   2,   2>,          // Y major
                        sequence<0,   0,   3>>{});      // y minor
#else
                    tile_distribution_encoding<
                        sequence<1>,// 0    R
                        // major       1                     2                     3
                        // minor       0     1               0     1               0   1   2
                        tuple<sequence<Nr_y, Nr_p>, sequence<Kr_y, Kr_p>, sequence<Kw, Nw, Kv>>,    // H

                        //            Nr_p, Kr_p         Kw Nw
                        tuple<sequence<1  , 2>, sequence<3, 3>>,    // p major
                        tuple<sequence<1  , 1>, sequence<0, 1>>,    // p minor

                        //       Nr_y Kr_y Kv
                        sequence<1,   2,   3>,          // Y major
                        sequence<0,   0,   2>>{});      // y minor
#endif
                // clang-format on
            }
        }

        __device__ void operator()(karg a_)
        {
            using namespace ck_tile;
            index_t i_k = blockIdx.x;
            index_t i_n = blockIdx.y;
            index_t i_b = blockIdx.z;

            constexpr index_t k2 = Alignment;
            constexpr index_t n2 = WarpGemm::WarpGemmAttribute::Impl::kAMLane;
            constexpr index_t k1 = WarpGemm::WarpGemmAttribute::Impl::kABKLane;
            constexpr index_t n1 = BLOCK_SIZE / get_warp_size();
            const index_t k0     = a_.k / (k1 * k2);
            const index_t n0     = a_.n / (n1 * n2);

            constexpr index_t k2_tile = Alignment;
            constexpr index_t n2_tile = WarpGemm::WarpGemmAttribute::Impl::kAMLane;
            constexpr index_t k1_tile = WarpGemm::WarpGemmAttribute::Impl::kABKLane;
            constexpr index_t n1_tile = BLOCK_SIZE / get_warp_size();
            constexpr index_t k0_tile = KPerBlock / (k1_tile * k2_tile);
            constexpr index_t n0_tile = NPerBlock / (n1_tile * n2_tile);

            const fp16_t* p_src = reinterpret_cast<const fp16_t*>(a_.p_src) + i_b * a_.k * a_.n;
            fp16_t* p_dst       = reinterpret_cast<fp16_t*>(a_.p_dst) + i_b * a_.k * a_.n;

            const auto src_view = [&]() {
                const auto tmp = make_naive_tensor_view_packed<address_space_enum::global>(
                    p_src,
                    make_tuple(n0, n1, n2, k0, k1, k2),
                    number<Alignment>{}); // control vector load
                return tmp;
            }();

            const auto src_window = make_tile_window(src_view,
                                                     make_tuple(number<n0_tile>{},
                                                                number<n1_tile>{},
                                                                number<n2_tile>{},
                                                                number<k0_tile>{},
                                                                number<k1_tile>{},
                                                                number<k2_tile>{}),
                                                     {i_n * n0_tile, 0, 0, i_k * k0_tile, 0, 0},
                                                     get_src_dist());

            auto dst_view = [&]() {
                if constexpr(pstyle == matrix_core_permute_style::permute_b_n0_k0_n1_k1_n2_k2)
                {
                    auto tmp = make_naive_tensor_view_packed<address_space_enum::global>(
                        p_dst,
                        make_tuple(n0, k0, n1, k1, n2, k2),
                        number<Alignment>{}); // control vector load
                    return tmp;
                }
                else if constexpr(pstyle == matrix_core_permute_style::permute_b_n0_n1_k0_k1_n2_k2)
                {
                    auto tmp = make_naive_tensor_view_packed<address_space_enum::global>(
                        p_dst,
                        make_tuple(n0, n1, k0, k1, n2, k2),
                        number<Alignment>{}); // control vector load
                    return tmp;
                }
                else
                {
#if MERGE_2D_013425
                    constexpr index_t kv = Alignment;
                    constexpr index_t nw = WarpGemm::WarpGemmAttribute::Impl::kAMLane;
                    constexpr index_t kw = WarpGemm::WarpGemmAttribute::Impl::kABKLane;
                    // constexpr index_t waveflatten = kw*nw*kv;
                    const index_t kr = a_.k / (k1 * k2);
                    const index_t nr = a_.n / nw;
                    auto tmp         = make_naive_tensor_view_packed<address_space_enum::global>(
                        p_dst,
                        make_tuple(nr, kr, number<kw>{}, number<nw>{}, number<kv>{}),
                        number<Alignment>{}); // control vector load
                    auto tmp_1 = transform_tensor_view(
                        tmp,
                        make_tuple(
                            make_merge_transform(make_tuple(nr, number<nw>{})),
                            make_merge_transform(make_tuple(kr, number<kw>{}, number<kv>{}))),
                        make_tuple(sequence<0, 3>{}, sequence<1, 2, 4>{}),
                        make_tuple(sequence<0>{}, sequence<1>{}));
                    return tmp_1;
#else
                    // b_nr_kr_waveflatten = b_nr_kr_kw_nw_kv,
                    constexpr index_t kv          = Alignment;
                    constexpr index_t nw          = WarpGemm::WarpGemmAttribute::Impl::kAMLane;
                    constexpr index_t kw          = WarpGemm::WarpGemmAttribute::Impl::kABKLane;
                    constexpr index_t waveflatten = kw * nw * kv;
                    const index_t kr              = a_.k / (k1 * k2);
                    const index_t nr              = a_.n / nw;
                    auto tmp = make_naive_tensor_view_packed<address_space_enum::global>(
                        p_dst,
                        make_tuple(nr, kr, waveflatten),
                        number<Alignment>{}); // control vector load
                    return tmp;
#endif
                }
            }();

            auto dst_window = [&]() {
                if constexpr(pstyle == matrix_core_permute_style::permute_b_n0_k0_n1_k1_n2_k2)
                {
                    return make_tile_window(dst_view,
                                            make_tuple(number<n0_tile>{},
                                                       number<k0_tile>{},
                                                       number<n1_tile>{},
                                                       number<k1_tile>{},
                                                       number<n2_tile>{},
                                                       number<k2_tile>{}),
                                            {i_n * n0_tile, i_k * k0_tile, 0, 0, 0, 0},
                                            get_dst_dist());
                }
                else if constexpr(pstyle == matrix_core_permute_style::permute_b_n0_n1_k0_k1_n2_k2)
                {
                    return make_tile_window(dst_view,
                                            make_tuple(number<n0_tile>{},
                                                       number<n1_tile>{},
                                                       number<k0_tile>{},
                                                       number<k1_tile>{},
                                                       number<n2_tile>{},
                                                       number<k2_tile>{}),
                                            {i_n * n0_tile, 0, i_k * k0_tile, 0, 0, 0},
                                            get_dst_dist());
                }
                else
                {
#if MERGE_2D_013425
                    // b_nr_kr_waveflatten = b_nr_kr_kw_nw_kv
                    return make_tile_window(dst_view,
                                            make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                                            {i_n * NPerBlock, i_k * KPerBlock},
                                            get_dst_dist());
#else
                    // b_nr_kr_waveflatten = b_nr_kr_kw_nw_kv
                    constexpr index_t kv = Alignment;
                    constexpr index_t nw = WarpGemm::WarpGemmAttribute::Impl::kAMLane;
                    constexpr index_t kw = WarpGemm::WarpGemmAttribute::Impl::kABKLane;
                    constexpr index_t waveflatten_tile = kw * nw * kv;
                    constexpr index_t nr_tile          = NPerBlock / nw;
                    constexpr index_t kr_tile          = KPerBlock / (kw * kv);
                    return make_tile_window(dst_view,
                                            make_tuple(number<nr_tile>{},
                                                       number<kr_tile>{},
                                                       number<waveflatten_tile>{}),
                                            {i_n * nr_tile, i_k * kr_tile, 0},
                                            get_dst_dist());
#endif
                }
            }();

            // actual load store
            auto src_tile = load_tile(src_window);

            // now we only swap the distribution from src to dst, no extra movement occurs
            auto dst_tile                = make_static_distributed_tensor<fp16_t>(get_dst_dist());
            dst_tile.get_thread_buffer() = src_tile.get_thread_buffer();

            // final store
            store_tile(dst_window, dst_tile);
        }
    };
};
