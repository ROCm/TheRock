// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/container/thread_buffer.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/core/utility/functional.hpp"

namespace ck_tile {

// S: scalar type (or it can be non-scalar type)
// NX: # of vector before transpose
// NY: # of vector after transpose
// we got [NX, NY] amount of S data to be transposed into [NY, NX] amount of S data
template <typename S_, index_t NX, index_t NY>
struct transpose_vectors
{
    static constexpr index_t s_per_x = NY;
    static constexpr index_t s_per_y = NX;

    using S = remove_cvref_t<S_>;

    using VX = array<S, s_per_x>;
    using VY = array<S, s_per_y>;

    struct generic_tag
    {
    };
    struct bytesize2_2x2_tag
    {
    };
    struct bytesize1_4x4_tag
    {
    };
    struct bytesize1_2x2_tag
    {
    };

    CK_TILE_DEVICE static constexpr void
    apply_impl(const thread_buffer<VX, NX>& vx_tuple, thread_buffer<VY, NY>& vy_tuple, generic_tag)
    {
        static_for<0, NY, 1>{}([&](auto iy) {
            static_for<0, NX, 1>{}([&](auto ix) { vy_tuple(iy)(ix) = vx_tuple[ix][iy]; });
        });
    }

    CK_TILE_DEVICE static constexpr void apply_impl(const thread_buffer<VX, NX>& vx_tuple,
                                                    thread_buffer<VY, NY>& vy_tuple,
                                                    bytesize2_2x2_tag)
    {
        static_assert(sizeof(S) == 2 && NX % 2 == 0 && NY % 2 == 0, "wrong!");

        constexpr auto I1 = number<1>{};
        constexpr auto I2 = number<2>{};
        using S2          = array<S, 2>;
        // loop over 2x2 tiles and transpose data from vx_tuple into vy_tuple
        static_for<0, NY, 2>{}([&](auto iy) {
            static_for<0, NX, 2>{}([&](auto ix) {
                // 2 16bitx2 data from vx_tuple to be transposed
                const S2 x_s2_0 = vx_tuple[ix].template get_as<S2>(iy / I2);
                const S2 x_s2_1 = vx_tuple[ix + I1].template get_as<S2>(iy / I2);

                // transpose 2x2 16bit
                // ex: v_perm_b32(0x 11 22 33 44, 0x 55 66 77 88, 0x 05 01 04 00) -> 0x33774488
                //                   -- -- -- --     -- -- -- --      -  -  -  -
                //             index  7  6  5  4      3  2  1  0     33 77 44 88
                // index is reversed because of little endianness (least significant bits first)
                const S2 y_s2_0 = bit_cast<S2>(
                    __builtin_amdgcn_perm(bit_cast<uint32_t>(x_s2_0),
                                          bit_cast<uint32_t>(x_s2_1),
                                          // (A0.B0.C0.D0.A1.B1.C1.D1)[1, 0, 5, 4] = (C1.D1.C0.D0)
                                          0x01'00'05'04));
                const S2 y_s2_1 = bit_cast<S2>(
                    __builtin_amdgcn_perm(bit_cast<uint32_t>(x_s2_0),
                                          bit_cast<uint32_t>(x_s2_1),
                                          // (A0.B0.C0.D0.A1.B1.C1.D1)[3, 2, 7, 6] = (A1.B1.A0.B0)
                                          0x03'02'07'06));

                // write transposed 2x2 result:
                // write (C1.D1.C0.D0)
                vy_tuple(iy).set_as(ix / I2, y_s2_0);
                // write (A1.B1.A0.B0)
                vy_tuple(iy + I1).set_as(ix / I2, y_s2_1);
            });
        });
    }

    CK_TILE_DEVICE static constexpr void apply_impl(const thread_buffer<VX, NX>& vx_tuple,
                                                    thread_buffer<VY, NY>& vy_tuple,
                                                    bytesize1_4x4_tag)
    {
        static_assert(sizeof(S) == 1 && NX % 4 == 0 && NY % 4 == 0, "wrong!");

        constexpr auto I1 = number<1>{};
        constexpr auto I2 = number<2>{};
        constexpr auto I3 = number<3>{};
        constexpr auto I4 = number<4>{};
        using S4          = array<S, 4>;
        // loop over 4x4 tiles and transpose data from vx_tuple into vy_tuple
        static_for<0, NY, 4>{}([&](auto iy) {
            static_for<0, NX, 4>{}([&](auto ix) {
                // read A0.B0.C0.D0
                const S4 x_s4_0 = vx_tuple[ix].template get_as<S4>(iy / I4);
                // read A1.B1.C1.D1
                const S4 x_s4_1 = vx_tuple[ix + I1].template get_as<S4>(iy / I4);
                // read A2.B2.C2.D2
                const S4 x_s4_2 = vx_tuple[ix + I2].template get_as<S4>(iy / I4);
                // read A3.B3.C3.D3
                const S4 x_s4_3 = vx_tuple[ix + I3].template get_as<S4>(iy / I4);

                // (A1.B1.C1.D1.A0.B0.C0.D0)[5, 1, 4, 0] = (C1.C0.D1.D0)
                uint32_t t_s4_0 = __builtin_amdgcn_perm(
                    bit_cast<uint32_t>(x_s4_1), bit_cast<uint32_t>(x_s4_0), 0x05'01'04'00);
                // (A3.B3.C3.D3.A2.B2.C2.D2)[5, 1, 4, 0] = (C3.C2.D3.D2)
                uint32_t t_s4_1 = __builtin_amdgcn_perm(
                    bit_cast<uint32_t>(x_s4_3), bit_cast<uint32_t>(x_s4_2), 0x05'01'04'00);
                // (C3.C2.D3.D2.C1.C0.D1.D0)[5, 4, 1, 0] = (D3.D2.D1.D0)
                const S4 y_s4_0 =
                    bit_cast<S4>(__builtin_amdgcn_perm(t_s4_1, t_s4_0, 0x05'04'01'00));
                // (C3.C2.D3.D2.C1.C0.D1.D0)[7, 6, 3, 2] = (C3.C2.C1.C0)
                const S4 y_s4_1 =
                    bit_cast<S4>(__builtin_amdgcn_perm(t_s4_1, t_s4_0, 0x07'06'03'02));
                // (A1.B1.C1.D1.A0.B0.C0.D0)[7, 3, 6, 2] = (A1.A0.B1.B0)
                t_s4_0 = __builtin_amdgcn_perm(
                    bit_cast<uint32_t>(x_s4_1), bit_cast<uint32_t>(x_s4_0), 0x07'03'06'02);
                // (A3.B3.C3.D3.A2.B2.C2.D2)[7, 3, 6, 2] = (A3.A2.B3.B2)
                t_s4_1 = __builtin_amdgcn_perm(
                    bit_cast<uint32_t>(x_s4_3), bit_cast<uint32_t>(x_s4_2), 0x07'03'06'02);
                // (A3.A2.B3.B2.A1.A0.B1.B0)[5, 4, 1, 0] = (B3.B2.B1.B0)
                const S4 y_s4_2 =
                    bit_cast<S4>(__builtin_amdgcn_perm(t_s4_1, t_s4_0, 0x05'04'01'00));
                // (A3.A2.B3.B2.A1.A0.B1.B0)[7, 6, 3, 2] = (A3.A2.A1.A0)
                const S4 y_s4_3 =
                    bit_cast<S4>(__builtin_amdgcn_perm(t_s4_1, t_s4_0, 0x07'06'03'02));

                // write transposed 4x4 result:
                // write (D3.D2.D1.D0)
                vy_tuple(iy).set_as(ix / I4, y_s4_0);
                // write (C3.C2.C1.C0)
                vy_tuple(iy + I1).set_as(ix / I4, y_s4_1);
                // write (B3.B2.B1.B0)
                vy_tuple(iy + I2).set_as(ix / I4, y_s4_2);
                // write (A3.A2.A1.A0)
                vy_tuple(iy + I3).set_as(ix / I4, y_s4_3);
            });
        });
    }

    CK_TILE_DEVICE static constexpr void apply_impl(const thread_buffer<VX, NX>& vx_tuple,
                                                    thread_buffer<VY, NY>& vy_tuple,
                                                    bytesize1_2x2_tag)
    {
        static_assert(sizeof(S) == 1 && NX % 2 == 0 && NY % 2 == 0, "wrong!");

        constexpr auto I1 = number<1>{};
        constexpr auto I2 = number<2>{};
        using S2          = array<S, 2>;
        // loop over 2x2 tiles and transpose data from vx_tuple into vy_tuple
        static_for<0, NY, 2>{}([&](auto iy) {
            static_for<0, NX, 2>{}([&](auto ix) {
                // read A0.B0
                const S2 x_s2_0 = vx_tuple[ix].template get_as<S2>(iy / I2);
                // read A1.B1
                const S2 x_s2_1 = vx_tuple[ix + I1].template get_as<S2>(iy / I2);

                // v_perm_b32: pick 4 bytes from 8 bytes in (input0.input1) using the mask
                const S2 y_s2_0 = bit_cast<S2>(static_cast<uint16_t>(__builtin_amdgcn_perm(
                    static_cast<uint32_t>(bit_cast<uint16_t>(x_s2_0)),
                    static_cast<uint32_t>(bit_cast<uint16_t>(x_s2_1)),
                    // (XX.XX.A0.B0.XX.XX.A1.B1)[clear, clear, 0, 4] = (00.00.B1.B0)
                    0x0C'0C'00'04)));

                const S2 y_s2_1 = bit_cast<S2>(static_cast<uint16_t>(__builtin_amdgcn_perm(
                    static_cast<uint32_t>(bit_cast<uint16_t>(x_s2_0)),
                    static_cast<uint32_t>(bit_cast<uint16_t>(x_s2_1)),
                    // (XX.XX.A0.B0.XX.XX.A1.B1)[clear, clear, 1, 5] = (00.00.A1.A0)
                    0x0C'0C'01'05)));

                // write transposed 2x2 result:
                // write (B1.B0)
                vy_tuple(iy).set_as(ix / I2, y_s2_0);
                // write (A1.A0)
                vy_tuple(iy + I1).set_as(ix / I2, y_s2_1);
            });
        });
    }

    CK_TILE_DEVICE static constexpr auto tag_dispatch()
    {
        if constexpr(sizeof(S) == 2 && NX % 2 == 0 && NY % 2 == 0)
        {
            return bytesize2_2x2_tag{};
        }
        else if constexpr(sizeof(S) == 1 && NX % 4 == 0 && NY % 4 == 0)
        {
            return bytesize1_4x4_tag{};
        }
        else if constexpr(sizeof(S) == 1 && NX % 2 == 0 && NY % 2 == 0)
        {
            return bytesize1_2x2_tag{};
        }
        else
        {
            return generic_tag{};
        }
    }

    CK_TILE_DEVICE void operator()(const thread_buffer<VX, NX>& vx_tuple,
                                   thread_buffer<VY, NY>& vy_tuple) const
    {
        apply_impl(vx_tuple, vy_tuple, tag_dispatch());
    }
};

} // namespace ck_tile
