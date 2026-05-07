// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/core.hpp"
#include <type_traits>

// clang-format off
template<typename SliceStart_ = ck_tile::sequence<0, 0>,
        typename SliceEnd_ = ck_tile::sequence<64, 16>,
        typename Y_Origin_ = ck_tile::sequence<0, 0, 0, 0>>
void test_slice_distribution_from_x_case_0(SliceStart_ = {}, SliceEnd_={}, Y_Origin_ = {})
{
    // slice length [-1, 16]
    using namespace ck_tile;
    constexpr auto r = detail::slice_distribution_from_x(
        make_static_tile_distribution(
                tile_distribution_encoding<sequence<>,
                                        tuple<sequence<1, 4, 16>, sequence<2, 2, 1, 4, 4>>,
                                        //             Y  P  P             Y  P  Y  P  Y
                                        tuple<sequence<1, 2>, sequence<2, 1>>,
                                        tuple<sequence<1, 1>, sequence<3, 2>>,
                                        sequence<1, 2, 2, 2>,
                                        sequence<0, 0, 2, 4>>{}),
        SliceStart_{},
        SliceEnd_{});

    using sliced_dist_enc = remove_cvref_t<decltype(r[number<0>{}].get_static_tile_distribution_encoding())>;
    using target_dist_enc = tile_distribution_encoding<sequence<>,
                                        tuple<sequence<1, 4, 16>, sequence<1, 2, 1, 4, 2>>,
                                        //             Y  P  P             Y  P  Y  P  Y
                                        tuple<sequence<1, 2>, sequence<2, 1>>,
                                        tuple<sequence<1, 1>, sequence<3, 2>>,
                                        sequence<1, 2, 2, 2>,
                                        sequence<0, 0, 2, 4>>;

    static_assert(std::is_same_v<sliced_dist_enc, target_dist_enc>);

    using sliced_y_origins = remove_cvref_t<decltype(r[number<1>{}])>;
    using sliced_y_lengths = remove_cvref_t<decltype(r[number<2>{}])>;
    static_assert(std::is_same_v<sliced_y_origins, Y_Origin_>);
    static_assert(std::is_same_v<sliced_y_lengths, sequence<1, 1, 1, 2>>);
}

template<typename SliceStart_ = ck_tile::sequence<0, 0>,
        typename SliceEnd_ = ck_tile::sequence<16, 16>,
        typename Y_Origin_ = ck_tile::sequence<0, 0, 0, 0, 0>>
void test_slice_distribution_from_x_case_1(SliceStart_ = {}, SliceEnd_={}, Y_Origin_ = {})
{
    // slice length [16, 16]
    using namespace ck_tile;
    constexpr auto r = detail::slice_distribution_from_x(
        make_static_tile_distribution(
                tile_distribution_encoding<sequence<>,
                                        tuple<sequence<4, 8, 2>, sequence<2, 4, 2, 8, 2>>,
                                        //             Y  P  Y            Y  P  Y  Y  P
                                        tuple<sequence<1>, sequence<2, 2>>,
                                        tuple<sequence<1>, sequence<4, 1>>,
                                        sequence<1, 1, 2, 2, 2>,
                                        sequence<0, 2, 0, 2, 3>>{}),
        SliceStart_{},
        SliceEnd_{});

    using sliced_dist_enc = remove_cvref_t<decltype(r[number<0>{}].get_static_tile_distribution_encoding())>;
    using target_dist_enc = tile_distribution_encoding<sequence<>,
                                        tuple<sequence<1, 8, 2>, sequence<1, 4, 1, 2, 2>>,
                                        //             Y  P  Y            Y  P  Y  Y  P
                                        tuple<sequence<1>, sequence<2, 2>>,
                                        tuple<sequence<1>, sequence<4, 1>>,
                                        sequence<1, 1, 2, 2, 2>,
                                        sequence<0, 2, 0, 2, 3>>;

    static_assert(std::is_same_v<sliced_dist_enc, target_dist_enc>);

    using sliced_y_origins = remove_cvref_t<decltype(r[number<1>{}])>;
    using sliced_y_lengths = remove_cvref_t<decltype(r[number<2>{}])>;
    static_assert(std::is_same_v<sliced_y_origins, Y_Origin_>);
    static_assert(std::is_same_v<sliced_y_lengths, sequence<1, 2, 1, 1, 2>>);
}

template<typename SliceStart_ = ck_tile::sequence<0, 0>,
        typename SliceEnd_ = ck_tile::sequence<12, 48>,
        typename Y_Origin_ = ck_tile::sequence<0, 0, 0, 0, 0>>
void test_slice_distribution_from_x_case_2(SliceStart_ = {}, SliceEnd_={}, Y_Origin_ = {})
{
    // slice length [12, 48]
    using namespace ck_tile;
    constexpr auto r = detail::slice_distribution_from_x(
        make_static_tile_distribution(
                tile_distribution_encoding<sequence<4, 5>,
                                        tuple<sequence<4, 3, 2>, sequence<2, 2, 1, 4, 3, 4>>,
                                        //             Y  P  Y            Y  P, Y, P  P, Y
                                        tuple<sequence<0, 1, 0>, sequence<2, 2, 2>>,
                                        tuple<sequence<0, 1, 1>, sequence<4, 1, 3>>,
                                        sequence<1, 2, 1, 2, 2>,
                                        sequence<2, 0, 0, 5, 2>>{}),
        SliceStart_{},
        SliceEnd_{});

    using sliced_dist_enc = remove_cvref_t<decltype(r[number<0>{}].get_static_tile_distribution_encoding())>;
    using target_dist_enc = tile_distribution_encoding<sequence<4, 5>,
                                        tuple<sequence<2, 3, 2>, sequence<1, 2, 1, 4, 3, 2>>,
                                        //             Y  P  Y            Y  P, Y, P  P, Y
                                        tuple<sequence<0, 1, 0>, sequence<2, 2, 2>>,
                                        tuple<sequence<0, 1, 1>, sequence<4, 1, 3>>,
                                        sequence<1, 2, 1, 2, 2>,
                                        sequence<2, 0, 0, 5, 2>>;

    static_assert(std::is_same_v<sliced_dist_enc, target_dist_enc>);

    using sliced_y_origins = remove_cvref_t<decltype(r[number<1>{}])>;
    using sliced_y_lengths = remove_cvref_t<decltype(r[number<2>{}])>;
    static_assert(std::is_same_v<sliced_y_origins, Y_Origin_>);
    static_assert(std::is_same_v<sliced_y_lengths, sequence<2, 1, 2, 2, 1>>);
}

void test_slice_distribution_from_x()
{
    using namespace ck_tile;

    test_slice_distribution_from_x_case_0(sequence< 0,  0>{}, sequence<-1, 16>{}, sequence<0, 0, 0, 0>{});
    test_slice_distribution_from_x_case_0(sequence< 0, 16>{}, sequence<-1, 32>{}, sequence<0, 0, 0, 2>{});
    test_slice_distribution_from_x_case_0(sequence< 0, 32>{}, sequence<-1, 48>{}, sequence<0, 1, 0, 0>{});
    test_slice_distribution_from_x_case_0(sequence< 0, 48>{}, sequence<-1, 64>{}, sequence<0, 1, 0, 2>{});

    test_slice_distribution_from_x_case_1(sequence< 0,  0>{}, sequence<16, 16>{}, sequence<0, 0, 0, 0, 0>{});
    test_slice_distribution_from_x_case_1(sequence<16, 16>{}, sequence<32, 32>{}, sequence<1, 0, 0, 0, 2>{});
    test_slice_distribution_from_x_case_1(sequence<32, 64>{}, sequence<48, 80>{}, sequence<2, 0, 0, 1, 0>{});
    test_slice_distribution_from_x_case_1(sequence<48, 208>{}, sequence<64, 224>{}, sequence<3, 0, 1, 1, 2>{});

    test_slice_distribution_from_x_case_2(sequence< 0,  0>{}, sequence<12, 48>{}, sequence<0, 0, 0, 0, 0>{});
    test_slice_distribution_from_x_case_2(sequence<12, 144>{}, sequence<24, 192>{}, sequence<0, 1, 2, 2, 0>{});
}

// clang-format on
int main() { test_slice_distribution_from_x(); }
