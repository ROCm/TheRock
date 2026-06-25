// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"

using ck::bf8_t;
using ck::bhalf_t;
using ck::f8_t;
using ck::half_t;
using ck::Number;
using ck::type_convert;
using ck::vector_type;

TEST(Custom_bool, TestSize)
{
    struct custom_bool_t
    {
        bool data;
    };
    ASSERT_EQ(sizeof(custom_bool_t), sizeof(bool));
    ASSERT_EQ(sizeof(vector_type<custom_bool_t, 2>), sizeof(vector_type<bool, 2>));
    ASSERT_EQ(sizeof(vector_type<custom_bool_t, 4>), sizeof(vector_type<bool, 4>));
    ASSERT_EQ(sizeof(vector_type<custom_bool_t, 8>), sizeof(vector_type<bool, 8>));
    ASSERT_EQ(sizeof(vector_type<custom_bool_t, 16>), sizeof(vector_type<bool, 16>));
    ASSERT_EQ(sizeof(vector_type<custom_bool_t, 32>), sizeof(vector_type<bool, 32>));
    ASSERT_EQ(sizeof(vector_type<custom_bool_t, 64>), sizeof(vector_type<bool, 64>));
}

TEST(Custom_bool, TestAsType)
{
    struct custom_bool_t
    {
        using type = bool;
        type data;
        custom_bool_t() : data{type{}} {}
        custom_bool_t(type init) : data{init} {}
    };

    // test size
    const int size             = 4;
    std::vector<bool> test_vec = {false, true, false, true};
    // reference vector
    vector_type<custom_bool_t, size> right_vec;
    // check default CTOR
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(right_vec.template AsType<custom_bool_t>()(Number<i>{}).data, false);
    });
    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<custom_bool_t>()(Number<i>{}) = custom_bool_t{test_vec.at(i)};
    });
    vector_type<custom_bool_t, size> left_vec;
    // check copy assignment op
    left_vec = right_vec;
    // overwrite right_vec with 0s
    right_vec = vector_type<custom_bool_t, size>{};
    // check if values were copied correctly
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<custom_bool_t>()(Number<i>{}).data, test_vec.at(i));
    });
}

TEST(Custom_bool, TestAsTypeReshape)
{
    struct custom_bool_t
    {
        using type = bool;
        type data;
        custom_bool_t() : data{type{}} {}
        custom_bool_t(type init) : data{init} {}
    };

    // test size
    const int size             = 4;
    std::vector<bool> test_vec = {false, true, false, true};
    // reference vector
    vector_type<custom_bool_t, size> right_vec;
    // check default CTOR
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(right_vec.template AsType<custom_bool_t>()(Number<i>{}).data, false);
    });
    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<custom_bool_t>()(Number<i>{}) = custom_bool_t{test_vec.at(i)};
    });
    // copy the first half of a vector
    vector_type<custom_bool_t, size / 2> left_vec{
        right_vec.template AsType<vector_type<custom_bool_t, size / 2>::type>()(Number<0>{})};
    // check if values were copied correctly
    ck::static_for<0, size / 2, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<custom_bool_t>()(Number<i>{}).data, test_vec.at(i));
    });
}

TEST(Custom_int8, TestSize)
{
    struct custom_int8_t
    {
        int8_t data;
    };
    ASSERT_EQ(sizeof(custom_int8_t), sizeof(int8_t));
    ASSERT_EQ(sizeof(vector_type<custom_int8_t, 2>), sizeof(vector_type<int8_t, 2>));
    ASSERT_EQ(sizeof(vector_type<custom_int8_t, 4>), sizeof(vector_type<int8_t, 4>));
    ASSERT_EQ(sizeof(vector_type<custom_int8_t, 8>), sizeof(vector_type<int8_t, 8>));
    ASSERT_EQ(sizeof(vector_type<custom_int8_t, 16>), sizeof(vector_type<int8_t, 16>));
    ASSERT_EQ(sizeof(vector_type<custom_int8_t, 32>), sizeof(vector_type<int8_t, 32>));
    ASSERT_EQ(sizeof(vector_type<custom_int8_t, 64>), sizeof(vector_type<int8_t, 64>));
}

TEST(Custom_int8, TestAsType)
{
    struct custom_int8_t
    {
        using type = int8_t;
        type data;
        custom_int8_t() : data{type{}} {}
        custom_int8_t(type init) : data{init} {}
    };

    // test size
    const int size               = 4;
    std::vector<int8_t> test_vec = {3, -6, 8, -2};
    // reference vector
    vector_type<custom_int8_t, size> right_vec;
    // check default CTOR
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(right_vec.template AsType<custom_int8_t>()(Number<i>{}).data, 0);
    });
    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<custom_int8_t>()(Number<i>{}) = custom_int8_t{test_vec.at(i)};
    });
    vector_type<custom_int8_t, size> left_vec;
    // check copy assignment op
    left_vec = right_vec;
    // overwrite right_vec with 0s
    right_vec = vector_type<custom_int8_t, size>{};
    // check if values were copied correctly
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<custom_int8_t>()(Number<i>{}).data, test_vec.at(i));
    });
}

TEST(Custom_int8, TestAsTypeReshape)
{
    struct custom_int8_t
    {
        using type = int8_t;
        type data;
        custom_int8_t() : data{type{}} {}
        custom_int8_t(type init) : data{init} {}
    };

    // test size
    const int size               = 4;
    std::vector<int8_t> test_vec = {3, -6, 8, -2};
    // reference vector
    vector_type<custom_int8_t, size> right_vec;
    // check default CTOR
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(right_vec.template AsType<custom_int8_t>()(Number<i>{}).data, 0);
    });
    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<custom_int8_t>()(Number<i>{}) = custom_int8_t{test_vec.at(i)};
    });
    // copy the first half of a vector
    vector_type<custom_int8_t, size / 2> left_vec{
        right_vec.template AsType<vector_type<custom_int8_t, size / 2>::type>()(Number<0>{})};
    // check if values were copied correctly
    ck::static_for<0, size / 2, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<custom_int8_t>()(Number<i>{}).data, test_vec.at(i));
    });
}

TEST(Custom_uint8, TestSize)
{
    struct custom_uint8_t
    {
        uint8_t data;
    };
    ASSERT_EQ(sizeof(custom_uint8_t), sizeof(uint8_t));
    ASSERT_EQ(sizeof(vector_type<custom_uint8_t, 2>), sizeof(vector_type<uint8_t, 2>));
    ASSERT_EQ(sizeof(vector_type<custom_uint8_t, 4>), sizeof(vector_type<uint8_t, 4>));
    ASSERT_EQ(sizeof(vector_type<custom_uint8_t, 8>), sizeof(vector_type<uint8_t, 8>));
    ASSERT_EQ(sizeof(vector_type<custom_uint8_t, 16>), sizeof(vector_type<uint8_t, 16>));
    ASSERT_EQ(sizeof(vector_type<custom_uint8_t, 32>), sizeof(vector_type<uint8_t, 32>));
    ASSERT_EQ(sizeof(vector_type<custom_uint8_t, 64>), sizeof(vector_type<uint8_t, 64>));
}

TEST(Custom_uint8, TestAsType)
{
    struct custom_uint8_t
    {
        using type = uint8_t;
        type data;
        custom_uint8_t() : data{type{}} {}
        custom_uint8_t(type init) : data{init} {}
    };

    // test size
    const int size                = 4;
    std::vector<uint8_t> test_vec = {3, 6, 8, 2};
    // reference vector
    vector_type<custom_uint8_t, size> right_vec;
    // check default CTOR
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(right_vec.template AsType<custom_uint8_t>()(Number<i>{}).data, 0);
    });
    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<custom_uint8_t>()(Number<i>{}) = custom_uint8_t{test_vec.at(i)};
    });
    vector_type<custom_uint8_t, size> left_vec;
    // check copy assignment op
    left_vec = right_vec;
    // overwrite right_vec with 0s
    right_vec = vector_type<custom_uint8_t, size>{};
    // check if values were copied correctly
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<custom_uint8_t>()(Number<i>{}).data, test_vec.at(i));
    });
}

TEST(Custom_uint8, TestAsTypeReshape)
{
    struct custom_uint8_t
    {
        using type = uint8_t;
        type data;
        custom_uint8_t() : data{type{}} {}
        custom_uint8_t(type init) : data{init} {}
    };

    // test size
    const int size                = 4;
    std::vector<uint8_t> test_vec = {3, 6, 8, 2};
    // reference vector
    vector_type<custom_uint8_t, size> right_vec;
    // check default CTOR
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(right_vec.template AsType<custom_uint8_t>()(Number<i>{}).data, 0);
    });
    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<custom_uint8_t>()(Number<i>{}) = custom_uint8_t{test_vec.at(i)};
    });
    // copy the first half of a vector
    vector_type<custom_uint8_t, size / 2> left_vec{
        right_vec.template AsType<vector_type<custom_uint8_t, size / 2>::type>()(Number<0>{})};
    // check if values were copied correctly
    ck::static_for<0, size / 2, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<custom_uint8_t>()(Number<i>{}).data, test_vec.at(i));
    });
}

TEST(Custom_f8, TestSize)
{
    struct custom_f8_t
    {
        _BitInt(8) data;
    };
    ASSERT_EQ(sizeof(custom_f8_t), sizeof(_BitInt(8)));
    ASSERT_EQ(sizeof(vector_type<custom_f8_t, 2>), sizeof(vector_type<_BitInt(8), 2>));
    ASSERT_EQ(sizeof(vector_type<custom_f8_t, 4>), sizeof(vector_type<_BitInt(8), 4>));
    ASSERT_EQ(sizeof(vector_type<custom_f8_t, 8>), sizeof(vector_type<_BitInt(8), 8>));
    ASSERT_EQ(sizeof(vector_type<custom_f8_t, 16>), sizeof(vector_type<_BitInt(8), 16>));
    ASSERT_EQ(sizeof(vector_type<custom_f8_t, 32>), sizeof(vector_type<_BitInt(8), 32>));
    ASSERT_EQ(sizeof(vector_type<custom_f8_t, 64>), sizeof(vector_type<_BitInt(8), 64>));
}

TEST(Custom_f8, TestAsType)
{
    struct custom_f8_t
    {
        using type = _BitInt(8);
        type data;
        custom_f8_t() : data{type{}} {}
        custom_f8_t(type init) : data{init} {}
    };

    // test size
    const int size                   = 4;
    std::vector<_BitInt(8)> test_vec = {type_convert<_BitInt(8)>(0.3f),
                                        type_convert<_BitInt(8)>(-0.6f),
                                        type_convert<_BitInt(8)>(0.8f),
                                        type_convert<_BitInt(8)>(-0.2f)};
    // reference vector
    vector_type<custom_f8_t, size> right_vec;
    // check default CTOR
    ck::static_for<0, size, 1>{}(
        [&](auto i) { ASSERT_EQ(right_vec.template AsType<custom_f8_t>()(Number<i>{}).data, 0); });
    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<custom_f8_t>()(Number<i>{}) = custom_f8_t{test_vec.at(i)};
    });
    vector_type<custom_f8_t, size> left_vec;
    // check copy assignment op
    left_vec = right_vec;
    // overwrite right_vec with 0s
    right_vec = vector_type<custom_f8_t, size>{};
    // check if values were copied correctly
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<custom_f8_t>()(Number<i>{}).data, test_vec.at(i));
    });
}

TEST(Custom_f8, TestAsTypeReshape)
{
    struct custom_f8_t
    {
        using type = _BitInt(8);
        type data;
        custom_f8_t() : data{type{}} {}
        custom_f8_t(type init) : data{init} {}
    };

    // test size
    const int size                   = 4;
    std::vector<_BitInt(8)> test_vec = {type_convert<_BitInt(8)>(0.3f),
                                        type_convert<_BitInt(8)>(-0.6f),
                                        type_convert<_BitInt(8)>(0.8f),
                                        type_convert<_BitInt(8)>(-0.2f)};
    // reference vector
    vector_type<custom_f8_t, size> right_vec;
    // check default CTOR
    ck::static_for<0, size, 1>{}(
        [&](auto i) { ASSERT_EQ(right_vec.template AsType<custom_f8_t>()(Number<i>{}).data, 0); });
    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<custom_f8_t>()(Number<i>{}) = custom_f8_t{test_vec.at(i)};
    });
    // copy the first half of a vector
    vector_type<custom_f8_t, size / 2> left_vec{
        right_vec.template AsType<vector_type<custom_f8_t, size / 2>::type>()(Number<0>{})};
    // check if values were copied correctly
    ck::static_for<0, size / 2, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<custom_f8_t>()(Number<i>{}).data, test_vec.at(i));
    });
}

TEST(Custom_bf8, TestSize)
{
    struct custom_bf8_t
    {
        unsigned _BitInt(8) data;
    };
    ASSERT_EQ(sizeof(custom_bf8_t), sizeof(unsigned _BitInt(8)));
    ASSERT_EQ(sizeof(vector_type<custom_bf8_t, 2>), sizeof(vector_type<unsigned _BitInt(8), 2>));
    ASSERT_EQ(sizeof(vector_type<custom_bf8_t, 4>), sizeof(vector_type<unsigned _BitInt(8), 4>));
    ASSERT_EQ(sizeof(vector_type<custom_bf8_t, 8>), sizeof(vector_type<unsigned _BitInt(8), 8>));
    ASSERT_EQ(sizeof(vector_type<custom_bf8_t, 16>), sizeof(vector_type<unsigned _BitInt(8), 16>));
    ASSERT_EQ(sizeof(vector_type<custom_bf8_t, 32>), sizeof(vector_type<unsigned _BitInt(8), 32>));
    ASSERT_EQ(sizeof(vector_type<custom_bf8_t, 64>), sizeof(vector_type<unsigned _BitInt(8), 64>));
}

TEST(Custom_bf8, TestAsType)
{
    struct custom_bf8_t
    {
        using type = unsigned _BitInt(8);
        type data;
        custom_bf8_t() : data{type{}} {}
        custom_bf8_t(type init) : data{init} {}
    };

    // test size
    const int size                            = 4;
    std::vector<unsigned _BitInt(8)> test_vec = {type_convert<unsigned _BitInt(8)>(0.3f),
                                                 type_convert<unsigned _BitInt(8)>(-0.6f),
                                                 type_convert<unsigned _BitInt(8)>(0.8f),
                                                 type_convert<unsigned _BitInt(8)>(-0.2f)};
    // reference vector
    vector_type<custom_bf8_t, size> right_vec;
    // check default CTOR
    ck::static_for<0, size, 1>{}(
        [&](auto i) { ASSERT_EQ(right_vec.template AsType<custom_bf8_t>()(Number<i>{}).data, 0); });
    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<custom_bf8_t>()(Number<i>{}) = custom_bf8_t{test_vec.at(i)};
    });
    vector_type<custom_bf8_t, size> left_vec;
    // check copy assignment op
    left_vec = right_vec;
    // overwrite right_vec with 0s
    right_vec = vector_type<custom_bf8_t, size>{};
    // check if values were copied correctly
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<custom_bf8_t>()(Number<i>{}).data, test_vec.at(i));
    });
}

TEST(Custom_bf8, TestAsTypeReshape)
{
    struct custom_bf8_t
    {
        using type = unsigned _BitInt(8);
        type data;
        custom_bf8_t() : data{type{}} {}
        custom_bf8_t(type init) : data{init} {}
    };

    // test size
    const int size                            = 4;
    std::vector<unsigned _BitInt(8)> test_vec = {type_convert<unsigned _BitInt(8)>(0.3f),
                                                 type_convert<unsigned _BitInt(8)>(-0.6f),
                                                 type_convert<unsigned _BitInt(8)>(0.8f),
                                                 type_convert<unsigned _BitInt(8)>(-0.2f)};
    // reference vector
    vector_type<custom_bf8_t, size> right_vec;
    // check default CTOR
    ck::static_for<0, size, 1>{}(
        [&](auto i) { ASSERT_EQ(right_vec.template AsType<custom_bf8_t>()(Number<i>{}).data, 0); });
    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<custom_bf8_t>()(Number<i>{}) = custom_bf8_t{test_vec.at(i)};
    });
    // copy the first half of a vector
    vector_type<custom_bf8_t, size / 2> left_vec{
        right_vec.template AsType<vector_type<custom_bf8_t, size / 2>::type>()(Number<0>{})};
    // check if values were copied correctly
    ck::static_for<0, size / 2, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<custom_bf8_t>()(Number<i>{}).data, test_vec.at(i));
    });
}

TEST(Custom_half, TestSize)
{
    struct custom_half_t
    {
        half_t data;
    };
    ASSERT_EQ(sizeof(custom_half_t), sizeof(half_t));
    ASSERT_EQ(sizeof(vector_type<custom_half_t, 2>), sizeof(vector_type<half_t, 2>));
    ASSERT_EQ(sizeof(vector_type<custom_half_t, 4>), sizeof(vector_type<half_t, 4>));
    ASSERT_EQ(sizeof(vector_type<custom_half_t, 8>), sizeof(vector_type<half_t, 8>));
    ASSERT_EQ(sizeof(vector_type<custom_half_t, 16>), sizeof(vector_type<half_t, 16>));
    ASSERT_EQ(sizeof(vector_type<custom_half_t, 32>), sizeof(vector_type<half_t, 32>));
    ASSERT_EQ(sizeof(vector_type<custom_half_t, 64>), sizeof(vector_type<half_t, 64>));
}

TEST(Custom_half, TestAsType)
{
    struct custom_half_t
    {
        using type = half_t;
        type data;
        custom_half_t() : data{type{}} {}
        custom_half_t(type init) : data{init} {}
    };

    // test size
    const int size               = 4;
    std::vector<half_t> test_vec = {half_t{0.3f}, half_t{-0.6f}, half_t{0.8f}, half_t{-0.2f}};
    // reference vector
    vector_type<custom_half_t, size> right_vec;
    // check default CTOR
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(right_vec.template AsType<custom_half_t>()(Number<i>{}).data,
                  type_convert<half_t>(0.0f));
    });
    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<custom_half_t>()(Number<i>{}) = custom_half_t{test_vec.at(i)};
    });
    vector_type<custom_half_t, size> left_vec;
    // check copy assignment op
    left_vec = right_vec;
    // overwrite right_vec with 0s
    right_vec = vector_type<custom_half_t, size>{};
    // check if values were copied correctly
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<custom_half_t>()(Number<i>{}).data, test_vec.at(i));
    });
}

TEST(Custom_half, TestAsTypeReshape)
{
    struct custom_half_t
    {
        using type = half_t;
        type data;
        custom_half_t() : data{type{}} {}
        custom_half_t(type init) : data{init} {}
    };

    // test size
    const int size               = 4;
    std::vector<half_t> test_vec = {half_t{0.3f}, half_t{-0.6f}, half_t{0.8f}, half_t{-0.2f}};
    // reference vector
    vector_type<custom_half_t, size> right_vec;
    // check default CTOR
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(right_vec.template AsType<custom_half_t>()(Number<i>{}).data,
                  type_convert<half_t>(0.0f));
    });
    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<custom_half_t>()(Number<i>{}) = custom_half_t{test_vec.at(i)};
    });
    // copy the first half of a vector
    vector_type<custom_half_t, size / 2> left_vec{
        right_vec.template AsType<vector_type<custom_half_t, size / 2>::type>()(Number<0>{})};
    // check if values were copied correctly
    ck::static_for<0, size / 2, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<custom_half_t>()(Number<i>{}).data, test_vec.at(i));
    });
}

TEST(Custom_bhalf, TestSize)
{
    struct custom_bhalf_t
    {
        bhalf_t data;
    };
    ASSERT_EQ(sizeof(custom_bhalf_t), sizeof(bhalf_t));
    ASSERT_EQ(sizeof(vector_type<custom_bhalf_t, 2>), sizeof(vector_type<bhalf_t, 2>));
    ASSERT_EQ(sizeof(vector_type<custom_bhalf_t, 4>), sizeof(vector_type<bhalf_t, 4>));
    ASSERT_EQ(sizeof(vector_type<custom_bhalf_t, 8>), sizeof(vector_type<bhalf_t, 8>));
    ASSERT_EQ(sizeof(vector_type<custom_bhalf_t, 16>), sizeof(vector_type<bhalf_t, 16>));
    ASSERT_EQ(sizeof(vector_type<custom_bhalf_t, 32>), sizeof(vector_type<bhalf_t, 32>));
    ASSERT_EQ(sizeof(vector_type<custom_bhalf_t, 64>), sizeof(vector_type<bhalf_t, 64>));
}

TEST(Custom_bhalf, TestAsType)
{
    struct custom_bhalf_t
    {
        using type = bhalf_t;
        type data;
        custom_bhalf_t() : data{type{}} {}
        custom_bhalf_t(type init) : data{init} {}
    };

    // test size
    const int size                = 4;
    std::vector<bhalf_t> test_vec = {type_convert<bhalf_t>(0.3f),
                                     type_convert<bhalf_t>(-0.6f),
                                     type_convert<bhalf_t>(0.8f),
                                     type_convert<bhalf_t>(-0.2f)};
    // reference vector
    vector_type<custom_bhalf_t, size> right_vec;
    // check default CTOR
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(right_vec.template AsType<custom_bhalf_t>()(Number<i>{}).data,
                  type_convert<bhalf_t>(0.0f));
    });
    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<custom_bhalf_t>()(Number<i>{}) = custom_bhalf_t{test_vec.at(i)};
    });
    vector_type<custom_bhalf_t, size> left_vec;
    // check copy assignment op
    left_vec = right_vec;
    // overwrite right_vec with 0s
    right_vec = vector_type<custom_bhalf_t, size>{};
    // check if values were copied correctly
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<custom_bhalf_t>()(Number<i>{}).data, test_vec.at(i));
    });
}

TEST(Custom_bhalf, TestAsTypeReshape)
{
    struct custom_bhalf_t
    {
        using type = bhalf_t;
        type data;
        custom_bhalf_t() : data{type{}} {}
        custom_bhalf_t(type init) : data{init} {}
    };

    // test size
    const int size                = 4;
    std::vector<bhalf_t> test_vec = {type_convert<bhalf_t>(0.3f),
                                     type_convert<bhalf_t>(-0.6f),
                                     type_convert<bhalf_t>(0.8f),
                                     type_convert<bhalf_t>(-0.2f)};
    // reference vector
    vector_type<custom_bhalf_t, size> right_vec;
    // check default CTOR
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(right_vec.template AsType<custom_bhalf_t>()(Number<i>{}).data,
                  type_convert<bhalf_t>(0.0f));
    });
    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<custom_bhalf_t>()(Number<i>{}) = custom_bhalf_t{test_vec.at(i)};
    });
    // copy the first half of a vector
    vector_type<custom_bhalf_t, size / 2> left_vec{
        right_vec.template AsType<vector_type<custom_bhalf_t, size / 2>::type>()(Number<0>{})};
    // check if values were copied correctly
    ck::static_for<0, size / 2, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<custom_bhalf_t>()(Number<i>{}).data, test_vec.at(i));
    });
}

TEST(Custom_float, TestSize)
{
    struct custom_float_t
    {
        float data;
    };
    ASSERT_EQ(sizeof(custom_float_t), sizeof(float));
    ASSERT_EQ(sizeof(vector_type<custom_float_t, 2>), sizeof(vector_type<float, 2>));
    ASSERT_EQ(sizeof(vector_type<custom_float_t, 4>), sizeof(vector_type<float, 4>));
    ASSERT_EQ(sizeof(vector_type<custom_float_t, 8>), sizeof(vector_type<float, 8>));
    ASSERT_EQ(sizeof(vector_type<custom_float_t, 16>), sizeof(vector_type<float, 16>));
    ASSERT_EQ(sizeof(vector_type<custom_float_t, 32>), sizeof(vector_type<float, 32>));
    ASSERT_EQ(sizeof(vector_type<custom_float_t, 64>), sizeof(vector_type<float, 64>));
}

TEST(Custom_float, TestAsType)
{
    struct custom_float_t
    {
        using type = float;
        type data;
        custom_float_t() : data{type{}} {}
        custom_float_t(type init) : data{init} {}
    };

    // test size
    const int size              = 4;
    std::vector<float> test_vec = {0.3f, -0.6f, 0.8f, -0.2f};
    // reference vector
    vector_type<custom_float_t, size> right_vec;
    // check default CTOR
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(right_vec.template AsType<custom_float_t>()(Number<i>{}).data, 0.0f);
    });
    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<custom_float_t>()(Number<i>{}) = custom_float_t{test_vec.at(i)};
    });
    vector_type<custom_float_t, size> left_vec;
    // check copy assignment op
    left_vec = right_vec;
    // overwrite right_vec with 0s
    right_vec = vector_type<custom_float_t, size>{};
    // check if values were copied correctly
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<custom_float_t>()(Number<i>{}).data, test_vec.at(i));
    });
}

TEST(Custom_float, TestAsTypeReshape)
{
    struct custom_float_t
    {
        using type = float;
        type data;
        custom_float_t() : data{type{}} {}
        custom_float_t(type init) : data{init} {}
    };

    // test size
    const int size              = 4;
    std::vector<float> test_vec = {0.3f, -0.6f, 0.8f, -0.2f};
    // reference vector
    vector_type<custom_float_t, size> right_vec;
    // check default CTOR
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(right_vec.template AsType<custom_float_t>()(Number<i>{}).data, 0.0f);
    });
    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<custom_float_t>()(Number<i>{}) = custom_float_t{test_vec.at(i)};
    });
    // copy the first half of a vector
    vector_type<custom_float_t, size / 2> left_vec{
        right_vec.template AsType<vector_type<custom_float_t, size / 2>::type>()(Number<0>{})};
    // check if values were copied correctly
    ck::static_for<0, size / 2, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<custom_float_t>()(Number<i>{}).data, test_vec.at(i));
    });
}

TEST(Custom_double, TestSize)
{
    struct custom_double_t
    {
        double data;
    };
    ASSERT_EQ(sizeof(custom_double_t), sizeof(double));
    ASSERT_EQ(sizeof(vector_type<custom_double_t, 2>), sizeof(vector_type<double, 2>));
    ASSERT_EQ(sizeof(vector_type<custom_double_t, 4>), sizeof(vector_type<double, 4>));
    ASSERT_EQ(sizeof(vector_type<custom_double_t, 8>), sizeof(vector_type<double, 8>));
    ASSERT_EQ(sizeof(vector_type<custom_double_t, 16>), sizeof(vector_type<double, 16>));
    ASSERT_EQ(sizeof(vector_type<custom_double_t, 32>), sizeof(vector_type<double, 32>));
    ASSERT_EQ(sizeof(vector_type<custom_double_t, 64>), sizeof(vector_type<double, 64>));
}

TEST(Custom_double, TestAsType)
{
    struct custom_double_t
    {
        using type = double;
        type data;
        custom_double_t() : data{type{}} {}
        custom_double_t(type init) : data{init} {}
    };

    // test size
    const int size               = 4;
    std::vector<double> test_vec = {0.3, 0.6, 0.8, 0.2};
    // reference vector
    vector_type<custom_double_t, size> right_vec;
    // check default CTOR
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(right_vec.template AsType<custom_double_t>()(Number<i>{}).data, 0.0);
    });
    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<custom_double_t>()(Number<i>{}) = custom_double_t{test_vec.at(i)};
    });
    vector_type<custom_double_t, size> left_vec;
    // check copy assignment op
    left_vec = right_vec;
    // overwrite right_vec with 0s
    right_vec = vector_type<custom_double_t, size>{};
    // check if values were copied correctly
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<custom_double_t>()(Number<i>{}).data, test_vec.at(i));
    });
}

TEST(Custom_double, TestAsTypeReshape)
{
    struct custom_double_t
    {
        using type = double;
        type data;
        custom_double_t() : data{type{}} {}
        custom_double_t(type init) : data{init} {}
    };

    // test size
    const int size               = 4;
    std::vector<double> test_vec = {0.3, 0.6, 0.8, 0.2};
    // reference vector
    vector_type<custom_double_t, size> right_vec;
    // check default CTOR
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(right_vec.template AsType<custom_double_t>()(Number<i>{}).data, 0.0);
    });
    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<custom_double_t>()(Number<i>{}) = custom_double_t{test_vec.at(i)};
    });
    // copy the first half of a vector
    vector_type<custom_double_t, size / 2> left_vec{
        right_vec.template AsType<vector_type<custom_double_t, size / 2>::type>()(Number<0>{})};
    // check if values were copied correctly
    ck::static_for<0, size / 2, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<custom_double_t>()(Number<i>{}).data, test_vec.at(i));
    });
}

TEST(Complex_half, TestSize)
{
    struct complex_half_t
    {
        half_t real;
        half_t img;
    };
    ASSERT_EQ(sizeof(complex_half_t), sizeof(half_t) + sizeof(half_t));
    ASSERT_EQ(sizeof(vector_type<complex_half_t, 2>),
              sizeof(vector_type<half_t, 2>) + sizeof(vector_type<half_t, 2>));
    ASSERT_EQ(sizeof(vector_type<complex_half_t, 4>),
              sizeof(vector_type<half_t, 4>) + sizeof(vector_type<half_t, 4>));
    ASSERT_EQ(sizeof(vector_type<complex_half_t, 8>),
              sizeof(vector_type<half_t, 8>) + sizeof(vector_type<half_t, 8>));
    ASSERT_EQ(sizeof(vector_type<complex_half_t, 16>),
              sizeof(vector_type<half_t, 16>) + sizeof(vector_type<half_t, 16>));
    ASSERT_EQ(sizeof(vector_type<complex_half_t, 32>),
              sizeof(vector_type<half_t, 32>) + sizeof(vector_type<half_t, 32>));
    ASSERT_EQ(sizeof(vector_type<complex_half_t, 64>),
              sizeof(vector_type<half_t, 64>) + sizeof(vector_type<half_t, 64>));
}

TEST(Complex_half, TestAlignment)
{
    struct complex_half_t
    {
        half_t real;
        half_t img;
    };
    ASSERT_EQ(alignof(vector_type<complex_half_t, 2>),
              alignof(vector_type<half_t, 2>) + alignof(vector_type<half_t, 2>));
    ASSERT_EQ(alignof(vector_type<complex_half_t, 4>),
              alignof(vector_type<half_t, 4>) + alignof(vector_type<half_t, 4>));
    ASSERT_EQ(alignof(vector_type<complex_half_t, 8>),
              alignof(vector_type<half_t, 8>) + alignof(vector_type<half_t, 8>));
    ASSERT_EQ(alignof(vector_type<complex_half_t, 16>),
              alignof(vector_type<half_t, 16>) + alignof(vector_type<half_t, 16>));
    ASSERT_EQ(alignof(vector_type<complex_half_t, 32>),
              alignof(vector_type<half_t, 32>) + alignof(vector_type<half_t, 32>));
    ASSERT_EQ(alignof(vector_type<complex_half_t, 64>),
              alignof(vector_type<half_t, 64>) + alignof(vector_type<half_t, 64>));
}

TEST(Complex_half, TestAsType)
{
    struct complex_half_t
    {
        using type = half_t;
        type real;
        type img;
        complex_half_t() : real{type{}}, img{type{}} {}
        complex_half_t(type real_init, type img_init) : real{real_init}, img{img_init} {}
    };

    // test size
    const int size = 4;
    // custom type number of elements
    const int num_elem           = sizeof(complex_half_t) / sizeof(complex_half_t::type);
    std::vector<half_t> test_vec = {half_t{0.3f},
                                    half_t{-0.6f},
                                    half_t{0.8f},
                                    half_t{-0.2f},
                                    half_t{0.5f},
                                    half_t{-0.7f},
                                    half_t{0.9f},
                                    half_t{-0.3f}};
    // reference vector
    vector_type<complex_half_t, size> right_vec;
    // check default CTOR
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(right_vec.template AsType<complex_half_t>()(Number<i>{}).real,
                  type_convert<half_t>(0.0f));
        ASSERT_EQ(right_vec.template AsType<complex_half_t>()(Number<i>{}).img,
                  type_convert<half_t>(0.0f));
    });
    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<complex_half_t>()(Number<i>{}) =
            complex_half_t{test_vec.at(num_elem * i), test_vec.at(num_elem * i + 1)};
    });
    vector_type<complex_half_t, size> left_vec;
    // check copy assignment op
    left_vec = right_vec;
    // overwrite right_vec with 0s
    right_vec = vector_type<complex_half_t, size>{};
    // check if values were copied correctly
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<complex_half_t>()(Number<i>{}).real,
                  test_vec.at(num_elem * i));
        ASSERT_EQ(left_vec.template AsType<complex_half_t>()(Number<i>{}).img,
                  test_vec.at(num_elem * i + 1));
    });
}

TEST(Complex_half, TestAsTypeReshape)
{
    struct complex_half_t
    {
        using type = half_t;
        type real;
        type img;
        complex_half_t() : real{type{}}, img{type{}} {}
        complex_half_t(type real_init, type img_init) : real{real_init}, img{img_init} {}
    };

    // test size
    const int size = 4;
    // custom type number of elements
    const int num_elem           = sizeof(complex_half_t) / sizeof(complex_half_t::type);
    std::vector<half_t> test_vec = {half_t{0.3f},
                                    half_t{-0.6f},
                                    half_t{0.8f},
                                    half_t{-0.2f},
                                    half_t{0.5f},
                                    half_t{-0.7f},
                                    half_t{0.9f},
                                    half_t{-0.3f}};
    // reference vector
    vector_type<complex_half_t, size> right_vec;
    // check default CTOR
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(right_vec.template AsType<complex_half_t>()(Number<i>{}).real,
                  type_convert<half_t>(0.0f));
        ASSERT_EQ(right_vec.template AsType<complex_half_t>()(Number<i>{}).img,
                  type_convert<half_t>(0.0f));
    });
    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<complex_half_t>()(Number<i>{}) =
            complex_half_t{test_vec.at(num_elem * i), test_vec.at(num_elem * i + 1)};
    });
    // copy the first half of a vector
    vector_type<complex_half_t, size / 2> left_vec{
        right_vec.template AsType<vector_type<complex_half_t, size / 2>::type>()(Number<0>{})};
    // check if values were copied correctly
    ck::static_for<0, size / 2, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<complex_half_t>()(Number<i>{}).real,
                  test_vec.at(num_elem * i));
        ASSERT_EQ(left_vec.template AsType<complex_half_t>()(Number<i>{}).img,
                  test_vec.at(num_elem * i + 1));
    });
}

#if CK_USE_OCP_FP8

TEST(FP8OCP, TestSize)
{
    static_assert(std::is_same_v<f8_t, ck::f8_ocp_t>, "OCP FP8 is not enabled");
    ASSERT_EQ(sizeof(f8_t), sizeof(ck::fp8_storage_t));
    ASSERT_EQ(sizeof(vector_type<f8_t, 2>), sizeof(vector_type<ck::fp8_storage_t, 2>));
    ASSERT_EQ(sizeof(vector_type<f8_t, 4>), sizeof(vector_type<ck::fp8_storage_t, 4>));
    ASSERT_EQ(sizeof(vector_type<f8_t, 8>), sizeof(vector_type<ck::fp8_storage_t, 8>));
    ASSERT_EQ(sizeof(vector_type<f8_t, 16>), sizeof(vector_type<ck::fp8_storage_t, 16>));
    ASSERT_EQ(sizeof(vector_type<f8_t, 32>), sizeof(vector_type<ck::fp8_storage_t, 32>));
    ASSERT_EQ(sizeof(vector_type<f8_t, 64>), sizeof(vector_type<ck::fp8_storage_t, 64>));
}

TEST(FP8OCP, TestAsType)
{
    static_assert(std::is_same_v<f8_t, ck::f8_ocp_t>, "OCP FP8 is not enabled");

    // test size
    std::array<float, 8> test_vec = {-4, -2, -0.5, -0.25, 1.0 / 8.0, 1, 1.5, 16};
    constexpr int size            = test_vec.size();

    // reference vector
    vector_type<f8_t, size> right_vec;

    // check default CTOR
    ck::static_for<0, size, 1>{}(
        [&](auto i) { ASSERT_EQ(right_vec.template AsType<f8_t>()(Number<i>{}), f8_t{0}); });

    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<f8_t>()(Number<i>{}) = ck::type_convert<f8_t>(test_vec.at(i));
    });

    vector_type<f8_t, size> left_vec;
    // check copy assignment op
    left_vec = right_vec;
    // overwrite right_vec with 0s
    right_vec = vector_type<f8_t, size>{};

    // check if values were copied correctly
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<f8_t>()(Number<i>{}),
                  ck::type_convert<f8_t>(test_vec.at(i)));
    });

    ck::non_native_vector_base<ck::f8_ocp_t, 2> nnvb_f8x2(ck::type_convert<f8_t>(-10.0f));
    ASSERT_EQ(nnvb_f8x2.template AsType<f8_t>()(Number<0>{}), ck::type_convert<f8_t>(-10.0f));
    ASSERT_EQ(nnvb_f8x2.template AsType<f8_t>()(Number<1>{}), ck::type_convert<f8_t>(-10.0f));
}

TEST(FP8OCP, TestAsTypeReshape)
{
    static_assert(std::is_same_v<f8_t, ck::f8_ocp_t>, "OCP FP8 is not enabled");

    // test size
    std::array<float, 8> test_vec = {-8, -0.5, -0.25, 1.0 / 8.0, 1 / 256, 1, 1.5, 16};
    constexpr int size            = test_vec.size();

    // reference vector
    vector_type<f8_t, size> right_vec;

    // check default CTOR
    ck::static_for<0, size, 1>{}(
        [&](auto i) { ASSERT_EQ(right_vec.template AsType<f8_t>()(Number<i>{}), f8_t{0}); });

    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<f8_t>()(Number<i>{}) = ck::type_convert<f8_t>(test_vec.at(i));
    });

    // copy the first half of a vector
    vector_type<f8_t, size / 2> left_vec{
        right_vec.template AsType<vector_type<f8_t, size / 2>::type>()(Number<0>{})};

    // check if values were copied correctly
    ck::static_for<0, size / 2, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<f8_t>()(Number<i>{}),
                  ck::type_convert<f8_t>(test_vec.at(i)));
    });
}

TEST(BF8OCP, TestSize)
{
    static_assert(std::is_same_v<bf8_t, ck::bf8_ocp_t>, "OCP BF8 is not enabled");
    ASSERT_EQ(sizeof(bf8_t), sizeof(ck::fp8_storage_t));
    ASSERT_EQ(sizeof(vector_type<bf8_t, 2>), sizeof(vector_type<ck::fp8_storage_t, 2>));
    ASSERT_EQ(sizeof(vector_type<bf8_t, 4>), sizeof(vector_type<ck::fp8_storage_t, 4>));
    ASSERT_EQ(sizeof(vector_type<bf8_t, 8>), sizeof(vector_type<ck::fp8_storage_t, 8>));
    ASSERT_EQ(sizeof(vector_type<bf8_t, 16>), sizeof(vector_type<ck::fp8_storage_t, 16>));
    ASSERT_EQ(sizeof(vector_type<bf8_t, 32>), sizeof(vector_type<ck::fp8_storage_t, 32>));
    ASSERT_EQ(sizeof(vector_type<bf8_t, 64>), sizeof(vector_type<ck::fp8_storage_t, 64>));
}

TEST(BF8OCP, TestAsType)
{
    static_assert(std::is_same_v<bf8_t, ck::bf8_ocp_t>, "OCP BF8 is not enabled");

    // test size
    std::array<float, 8> test_vec = {-4, -2, -0.5, -0.25, 1.0 / 8.0, 1, 1.5, 16};
    constexpr int size            = test_vec.size();

    // reference vector
    vector_type<bf8_t, size> right_vec;

    // check default CTOR
    ck::static_for<0, size, 1>{}(
        [&](auto i) { ASSERT_EQ(right_vec.template AsType<bf8_t>()(Number<i>{}), bf8_t{0}); });

    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<bf8_t>()(Number<i>{}) = ck::type_convert<bf8_t>(test_vec.at(i));
    });

    vector_type<bf8_t, size> left_vec{right_vec};
    // check copy assignment op
    left_vec = right_vec;
    // overwrite right_vec with 0s
    right_vec = vector_type<bf8_t, size>{};

    // check if values were copied correctly
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<bf8_t>()(Number<i>{}),
                  ck::type_convert<bf8_t>(test_vec.at(i)));
    });

    ck::non_native_vector_base<bf8_t, 2> nnvb_bf8x2(ck::type_convert<bf8_t>(-10.0f));
    ASSERT_EQ(nnvb_bf8x2.template AsType<bf8_t>()(Number<0>{}), ck::type_convert<bf8_t>(-10.0f));
    ASSERT_EQ(nnvb_bf8x2.template AsType<bf8_t>()(Number<1>{}), ck::type_convert<bf8_t>(-10.0f));
}

TEST(BF8OCP, TestAsTypeReshape)
{
    static_assert(std::is_same_v<bf8_t, ck::bf8_ocp_t>, "OCP BF8 is not enabled");

    // test size
    std::array<float, 8> test_vec = {-8, -0.5, -0.25, 1.0 / 8.0, 1 / 256, 1, 1.5, 16};
    constexpr int size            = test_vec.size();

    // reference vector
    vector_type<bf8_t, size> right_vec;

    // check default CTOR
    ck::static_for<0, size, 1>{}(
        [&](auto i) { ASSERT_EQ(right_vec.template AsType<bf8_t>()(Number<i>{}), bf8_t{0}); });

    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<bf8_t>()(Number<i>{}) = ck::type_convert<bf8_t>(test_vec.at(i));
    });

    // copy the first half of a vector
    vector_type<bf8_t, size / 2> left_vec{
        right_vec.template AsType<vector_type<bf8_t, size / 2>::type>()(Number<0>{})};

    // check if values were copied correctly
    ck::static_for<0, size / 2, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<bf8_t>()(Number<i>{}),
                  ck::type_convert<bf8_t>(test_vec.at(i)));
    });
}

#endif
