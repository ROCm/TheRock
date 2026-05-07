// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck_tile/core.hpp"

using namespace ck_tile;

class TestCkTileTupleApply : public ::testing::Test
{
    public:
    // Test functors for different scenarios
    struct AddFunction
    {
        template <typename... Args>
        CK_TILE_HOST_DEVICE constexpr auto operator()(Args... args) const
        {
            return (args + ...);
        }
    };

    struct MultiplyFunction
    {
        template <typename... Args>
        CK_TILE_HOST_DEVICE constexpr auto operator()(Args... args) const
        {
            return (args * ...);
        }
    };

    struct MaxFunction
    {
        template <typename T>
        CK_TILE_HOST_DEVICE constexpr T operator()(T a) const
        {
            return a;
        }

        template <typename T, typename... Args>
        CK_TILE_HOST_DEVICE constexpr T operator()(T a, Args... args) const
        {
            auto rest_max = operator()(args...);
            return a > rest_max ? a : rest_max;
        }
    };

    struct ReturnTupleFunction
    {
        template <typename... Args>
        CK_TILE_HOST_DEVICE constexpr auto operator()(Args... args) const
        {
            return make_tuple(args..., sizeof...(args));
        }
    };
};

TEST_F(TestCkTileTupleApply, BasicArithmetic)
{
    // Test with simple arithmetic operations
    auto t1      = make_tuple(1, 2, 3);
    auto result1 = apply(AddFunction{}, t1);
    EXPECT_EQ(result1, 6);

    auto t2      = make_tuple(2, 3, 4, 5);
    auto result2 = apply(MultiplyFunction{}, t2);
    EXPECT_EQ(result2, 120);
}

TEST_F(TestCkTileTupleApply, SingleElement)
{
    // Test with single element tuple
    auto t1      = make_tuple(42);
    auto result1 = apply(AddFunction{}, t1);
    EXPECT_EQ(result1, 42);

    auto result2 = apply(MultiplyFunction{}, t1);
    EXPECT_EQ(result2, 42);
}

TEST_F(TestCkTileTupleApply, EmptyTuple)
{
    // Test with empty tuple
    auto t      = tuple<>{};
    auto result = apply([]() { return 100; }, t);
    EXPECT_EQ(result, 100);
}

TEST_F(TestCkTileTupleApply, DifferentTypes)
{
    // Test with different data types
    auto t1      = make_tuple(1, 2.5f, 3.0);
    auto result1 = apply(AddFunction{}, t1);
    EXPECT_FLOAT_EQ(result1, 6.5f);

    // Test with mixed integer and floating point
    auto t2      = make_tuple(10, 0.5f);
    auto result2 = apply(MultiplyFunction{}, t2);
    EXPECT_FLOAT_EQ(result2, 5.0f);
}

TEST_F(TestCkTileTupleApply, ReturnTuple)
{
    // Test function that returns a tuple
    auto t      = make_tuple(1, 2, 3);
    auto result = apply(ReturnTupleFunction{}, t);

    EXPECT_EQ(result.get<0>(), 1);
    EXPECT_EQ(result.get<1>(), 2);
    EXPECT_EQ(result.get<2>(), 3);
    EXPECT_EQ(result.get<3>(), 3); // size
}

TEST_F(TestCkTileTupleApply, LambdaFunction)
{
    // Test with lambda functions
    auto t1      = make_tuple(5, 10, 15);
    auto result1 = apply([](auto a, auto b, auto c) { return a + b + c; }, t1);
    EXPECT_EQ(result1, 30);

    // Test lambda with capture
    int multiplier = 2;
    auto result2 =
        apply([multiplier](auto a, auto b) { return (a + b) * multiplier; }, make_tuple(3, 7));
    EXPECT_EQ(result2, 20);
}

TEST_F(TestCkTileTupleApply, ConstexprContext)
{
    // Test in constexpr context
    constexpr auto t      = make_tuple(2, 3, 4);
    constexpr auto result = apply(MultiplyFunction{}, t);
    static_assert(result == 24, "Constexpr apply should work");
    EXPECT_EQ(result, 24);
}

TEST_F(TestCkTileTupleApply, ReferenceTypes)
{
    // Test with reference types using tie
    int a = 1, b = 2, c = 3;
    auto ref_tuple = tie(a, b, c);

    // Function that modifies references
    apply(
        [](auto& x, auto& y, auto& z) {
            x += 10;
            y += 20;
            z += 30;
        },
        ref_tuple);

    EXPECT_EQ(a, 11);
    EXPECT_EQ(b, 22);
    EXPECT_EQ(c, 33);
}

TEST_F(TestCkTileTupleApply, MoveSemantics)
{
    // Test with move semantics
    auto t      = make_tuple(1, 2, 3);
    auto result = apply(AddFunction{}, std::move(t));
    EXPECT_EQ(result, 6);
}

TEST_F(TestCkTileTupleApply, NumberTypes)
{
    // Test with ck_tile::number types
    auto t      = make_tuple(number<1>{}, number<2>{}, number<3>{});
    auto result = apply([](auto a, auto b, auto c) { return a + b + c; }, t);
    EXPECT_EQ(result, 6);
}

TEST_F(TestCkTileTupleApply, ElementwiseOperation)
{
    // Test simulating elementwise operations
    auto input1 = make_tuple(1.0f, 2.0f, 3.0f);
    auto input2 = make_tuple(4.0f, 5.0f, 6.0f);

    auto add_elementwise = [](const auto& a, const auto& b) {
        return apply(
            [&b](auto... args_a) {
                return apply(
                    [args_a...](auto... args_b) { return make_tuple((args_a + args_b)...); }, b);
            },
            a);
    };

    auto result = add_elementwise(input1, input2);

    EXPECT_FLOAT_EQ(result.get<0>(), 5.0f);
    EXPECT_FLOAT_EQ(result.get<1>(), 7.0f);
    EXPECT_FLOAT_EQ(result.get<2>(), 9.0f);
}

template <typename T>
class TestCkTileTupleApplySize : public TestCkTileTupleApply
{
    protected:
    static constexpr int Size = T::value;
};

using TupleSizes = ::testing::Types<std::integral_constant<int, 1>,
                                    std::integral_constant<int, 2>,
                                    std::integral_constant<int, 3>,
                                    std::integral_constant<int, 4>,
                                    std::integral_constant<int, 8>,
                                    std::integral_constant<int, 16>>;

TYPED_TEST_SUITE(TestCkTileTupleApplySize, TupleSizes);

TYPED_TEST(TestCkTileTupleApplySize, GeneratedTupleSum)
{
    constexpr int N = TypeParam::value;

    // Generate tuple with values 1, 2, 3, ..., N
    constexpr auto t = generate_tuple([](auto i) { return i.value + 1; }, number<N>{});

    // Sum all elements
    constexpr auto result = apply(TestCkTileTupleApply::AddFunction{}, t);

    // Expected sum: 1 + 2 + ... + N = N*(N+1)/2
    constexpr int expected = N * (N + 1) / 2;
    static_assert(result == expected);
}
