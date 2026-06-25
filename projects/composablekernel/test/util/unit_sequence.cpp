// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck/utility/sequence.hpp"
#include "ck/utility/functional.hpp"

using namespace ck;

// Test basic Sequence construction and properties
TEST(Sequence, BasicConstruction)
{
    using Seq = Sequence<1, 2, 3, 4, 5>;
    EXPECT_EQ(Seq::Size(), 5);
    EXPECT_EQ(Seq::mSize, 5);
}

TEST(Sequence, EmptySequence)
{
    using Seq = Sequence<>;
    EXPECT_EQ(Seq::Size(), 0);
    EXPECT_EQ(Seq::mSize, 0);
}

// Test At() method
TEST(Sequence, AtRuntime)
{
    using Seq = Sequence<10, 20, 30, 40>;
    EXPECT_EQ(Seq::At(0), 10);
    EXPECT_EQ(Seq::At(1), 20);
    EXPECT_EQ(Seq::At(2), 30);
    EXPECT_EQ(Seq::At(3), 40);
}

TEST(Sequence, AtCompileTime)
{
    using Seq = Sequence<10, 20, 30, 40>;
    EXPECT_EQ(Seq::At(Number<0>{}), 10);
    EXPECT_EQ(Seq::At(Number<1>{}), 20);
    EXPECT_EQ(Seq::At(Number<2>{}), 30);
    EXPECT_EQ(Seq::At(Number<3>{}), 40);
}

TEST(Sequence, OperatorBracket)
{
    constexpr auto seq = Sequence<5, 10, 15>{};
    EXPECT_EQ(seq[Number<0>{}], 5);
    EXPECT_EQ(seq[Number<1>{}], 10);
    EXPECT_EQ(seq[Number<2>{}], 15);
}

// Test Front() and Back()
TEST(Sequence, FrontBack)
{
    using Seq = Sequence<100, 200, 300>;
    EXPECT_EQ(Seq::Front(), 100);
    EXPECT_EQ(Seq::Back(), 300);
}

TEST(Sequence, FrontBackSingleElement)
{
    using Seq = Sequence<42>;
    EXPECT_EQ(Seq::Front(), 42);
    EXPECT_EQ(Seq::Back(), 42);
}

// Test PushFront and PushBack
TEST(Sequence, PushFront)
{
    using Seq      = Sequence<2, 3, 4>;
    using Result   = decltype(Seq::PushFront(Sequence<1>{}));
    using Expected = Sequence<1, 2, 3, 4>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(Sequence, PushFrontNumbers)
{
    using Seq      = Sequence<3, 4>;
    using Result   = decltype(Seq::PushFront(Number<1>{}, Number<2>{}));
    using Expected = Sequence<1, 2, 3, 4>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(Sequence, PushBack)
{
    using Seq      = Sequence<1, 2, 3>;
    using Result   = decltype(Seq::PushBack(Sequence<4, 5>{}));
    using Expected = Sequence<1, 2, 3, 4, 5>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(Sequence, PushBackNumbers)
{
    using Seq      = Sequence<1, 2>;
    using Result   = decltype(Seq::PushBack(Number<3>{}, Number<4>{}));
    using Expected = Sequence<1, 2, 3, 4>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

// Test PopFront and PopBack
TEST(Sequence, PopFront)
{
    using Seq      = Sequence<1, 2, 3, 4>;
    using Result   = decltype(Seq::PopFront());
    using Expected = Sequence<2, 3, 4>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(Sequence, PopBack)
{
    using Seq      = Sequence<1, 2, 3, 4>;
    using Result   = decltype(Seq::PopBack());
    using Expected = Sequence<1, 2, 3>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

// Test Extract
TEST(Sequence, ExtractByNumbers)
{
    using Seq      = Sequence<10, 20, 30, 40, 50>;
    using Result   = decltype(Seq::Extract(Number<0>{}, Number<2>{}, Number<4>{}));
    using Expected = Sequence<10, 30, 50>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(Sequence, ExtractBySequence)
{
    using Seq      = Sequence<10, 20, 30, 40, 50>;
    using Result   = decltype(Seq::Extract(Sequence<1, 3>{}));
    using Expected = Sequence<20, 40>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

// Test Modify
TEST(Sequence, Modify)
{
    using Seq      = Sequence<1, 2, 3, 4>;
    using Result   = decltype(Seq::Modify(Number<2>{}, Number<99>{}));
    using Expected = Sequence<1, 2, 99, 4>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

// Test Transform
TEST(Sequence, Transform)
{
    using Seq      = Sequence<1, 2, 3, 4>;
    auto double_it = [](auto x) { return 2 * x; };
    using Result   = decltype(Seq::Transform(double_it));
    using Expected = Sequence<2, 4, 6, 8>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

// Test Reverse
TEST(Sequence, Reverse)
{
    using Seq      = Sequence<1, 2, 3, 4, 5>;
    using Result   = decltype(Seq::Reverse());
    using Expected = Sequence<5, 4, 3, 2, 1>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(Sequence, ReverseSingleElement)
{
    using Seq      = Sequence<42>;
    using Result   = decltype(Seq::Reverse());
    using Expected = Sequence<42>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

// Test ReorderGivenNew2Old
TEST(Sequence, ReorderGivenNew2Old)
{
    using Seq      = Sequence<10, 20, 30, 40>;
    using Result   = decltype(Seq::ReorderGivenNew2Old(Sequence<3, 1, 2, 0>{}));
    using Expected = Sequence<40, 20, 30, 10>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

// Test ReorderGivenOld2New
TEST(Sequence, ReorderGivenOld2New)
{
    using Seq      = Sequence<10, 20, 30, 40>;
    using Result   = decltype(Seq::ReorderGivenOld2New(Sequence<3, 1, 2, 0>{}));
    using Expected = Sequence<40, 20, 30, 10>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

// Test arithmetic_sequence_gen
TEST(SequenceGen, ArithmeticSequence)
{
    using Result   = typename arithmetic_sequence_gen<0, 5, 1>::type;
    using Expected = Sequence<0, 1, 2, 3, 4>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceGen, ArithmeticSequenceWithIncrement)
{
    using Result   = typename arithmetic_sequence_gen<0, 10, 2>::type;
    using Expected = Sequence<0, 2, 4, 6, 8>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceGen, ArithmeticSequenceNegativeIncrement)
{
    using Result   = typename arithmetic_sequence_gen<10, 5, -1>::type;
    using Expected = Sequence<10, 9, 8, 7, 6>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceGen, ArithmeticSequenceEmpty)
{
    using Result   = typename arithmetic_sequence_gen<5, 5, 1>::type;
    using Expected = Sequence<>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

// Test uniform_sequence_gen
TEST(SequenceGen, UniformSequence)
{
    using Result   = typename uniform_sequence_gen<5, 42>::type;
    using Expected = Sequence<42, 42, 42, 42, 42>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceGen, UniformSequenceZeroSize)
{
    using Result   = typename uniform_sequence_gen<0, 42>::type;
    using Expected = Sequence<>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

// Test make_index_sequence
TEST(SequenceGen, MakeIndexSequence)
{
    using Result   = make_index_sequence<5>;
    using Expected = Sequence<0, 1, 2, 3, 4>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceGen, MakeIndexSequenceZero)
{
    using Result   = make_index_sequence<0>;
    using Expected = Sequence<>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

// Test sequence_merge
TEST(SequenceMerge, MergeTwoSequences)
{
    using Seq1     = Sequence<1, 2, 3>;
    using Seq2     = Sequence<4, 5, 6>;
    using Result   = typename sequence_merge<Seq1, Seq2>::type;
    using Expected = Sequence<1, 2, 3, 4, 5, 6>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceMerge, MergeMultipleSequences)
{
    using Seq1     = Sequence<1, 2>;
    using Seq2     = Sequence<3, 4>;
    using Seq3     = Sequence<5, 6>;
    using Result   = typename sequence_merge<Seq1, Seq2, Seq3>::type;
    using Expected = Sequence<1, 2, 3, 4, 5, 6>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceMerge, MergeSingleSequence)
{
    using Seq      = Sequence<1, 2, 3>;
    using Result   = typename sequence_merge<Seq>::type;
    using Expected = Sequence<1, 2, 3>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

// Test sequence_split
TEST(SequenceSplit, SplitInMiddle)
{
    using Seq           = Sequence<1, 2, 3, 4, 5, 6>;
    using Split         = sequence_split<Seq, 3>;
    using ExpectedLeft  = Sequence<1, 2, 3>;
    using ExpectedRight = Sequence<4, 5, 6>;
    EXPECT_TRUE((is_same<typename Split::left_type, ExpectedLeft>::value));
    EXPECT_TRUE((is_same<typename Split::right_type, ExpectedRight>::value));
}

TEST(SequenceSplit, SplitAtBeginning)
{
    using Seq           = Sequence<1, 2, 3, 4>;
    using Split         = sequence_split<Seq, 0>;
    using ExpectedLeft  = Sequence<>;
    using ExpectedRight = Sequence<1, 2, 3, 4>;
    EXPECT_TRUE((is_same<typename Split::left_type, ExpectedLeft>::value));
    EXPECT_TRUE((is_same<typename Split::right_type, ExpectedRight>::value));
}

TEST(SequenceSplit, SplitAtEnd)
{
    using Seq           = Sequence<1, 2, 3, 4>;
    using Split         = sequence_split<Seq, 4>;
    using ExpectedLeft  = Sequence<1, 2, 3, 4>;
    using ExpectedRight = Sequence<>;
    EXPECT_TRUE((is_same<typename Split::left_type, ExpectedLeft>::value));
    EXPECT_TRUE((is_same<typename Split::right_type, ExpectedRight>::value));
}

// Test sequence_sort
TEST(SequenceSort, SortAscending)
{
    using Seq      = Sequence<5, 2, 8, 1, 9>;
    using Result   = typename sequence_sort<Seq, math::less<index_t>>::type;
    using Expected = Sequence<1, 2, 5, 8, 9>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceSort, SortDescending)
{
    // Create a greater-than comparator
    struct greater
    {
        __host__ __device__ constexpr bool operator()(index_t x, index_t y) const { return x > y; }
    };
    using Seq      = Sequence<5, 2, 8, 1, 9>;
    using Result   = typename sequence_sort<Seq, greater>::type;
    using Expected = Sequence<9, 8, 5, 2, 1>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceSort, SortAlreadySorted)
{
    using Seq      = Sequence<1, 2, 3, 4, 5>;
    using Result   = typename sequence_sort<Seq, math::less<index_t>>::type;
    using Expected = Sequence<1, 2, 3, 4, 5>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceSort, SortWithDuplicates)
{
    using Seq      = Sequence<3, 1, 4, 1, 5, 9, 2, 6, 5>;
    using Result   = typename sequence_sort<Seq, math::less<index_t>>::type;
    using Expected = Sequence<1, 1, 2, 3, 4, 5, 5, 6, 9>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceSort, SortEmptySequence)
{
    using Seq      = Sequence<>;
    using Result   = typename sequence_sort<Seq, math::less<index_t>>::type;
    using Expected = Sequence<>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceSort, SortSingleElement)
{
    using Seq      = Sequence<42>;
    using Result   = typename sequence_sort<Seq, math::less<index_t>>::type;
    using Expected = Sequence<42>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

// Test sequence_unique_sort
TEST(SequenceUniqueSort, UniqueSort)
{
    using Seq = Sequence<3, 1, 4, 1, 5, 9, 2, 6, 5>;
    using Result =
        typename sequence_unique_sort<Seq, math::less<index_t>, math::equal<index_t>>::type;
    using Expected = Sequence<1, 2, 3, 4, 5, 6, 9>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceUniqueSort, UniqueSortNoDuplicates)
{
    using Seq = Sequence<5, 2, 8, 1, 9>;
    using Result =
        typename sequence_unique_sort<Seq, math::less<index_t>, math::equal<index_t>>::type;
    using Expected = Sequence<1, 2, 5, 8, 9>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceUniqueSort, UniqueSortAllSame)
{
    using Seq = Sequence<5, 5, 5, 5>;
    using Result =
        typename sequence_unique_sort<Seq, math::less<index_t>, math::equal<index_t>>::type;
    using Expected = Sequence<5>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

// Test is_valid_sequence_map
TEST(SequenceMap, ValidMap)
{
    using Map = Sequence<0, 1, 2, 3>;
    EXPECT_TRUE((is_valid_sequence_map<Map>::value));
}

TEST(SequenceMap, ValidMapPermuted)
{
    using Map = Sequence<2, 0, 3, 1>;
    EXPECT_TRUE((is_valid_sequence_map<Map>::value));
}

TEST(SequenceMap, InvalidMapDuplicate)
{
    using Map = Sequence<0, 1, 1, 3>;
    EXPECT_FALSE((is_valid_sequence_map<Map>::value));
}

TEST(SequenceMap, InvalidMapMissing)
{
    using Map = Sequence<0, 1, 3, 4>;
    EXPECT_FALSE((is_valid_sequence_map<Map>::value));
}

// Test sequence_map_inverse
// Note: sequence_map_inverse inverts a mapping where Map[i] = j means old position i maps to new
// position j The inverse gives us new position i came from old position inverse[i]
TEST(SequenceMapInverse, InverseMap)
{
    // Map = <2, 0, 3, 1> means: old[0]->new[2], old[1]->new[0], old[2]->new[3], old[3]->new[1]
    // Inverse should be: new[0]<-old[1], new[1]<-old[3], new[2]<-old[0], new[3]<-old[2]
    using Map    = Sequence<2, 0, 3, 1>;
    using Result = typename sequence_map_inverse<Map>::type;
    // Verify by checking that Map[Result[i]] == i for all i
    EXPECT_EQ((Map::At(Number<Result::At(Number<0>{})>{}) == 0), true);
    EXPECT_EQ((Map::At(Number<Result::At(Number<1>{})>{}) == 1), true);
    EXPECT_EQ((Map::At(Number<Result::At(Number<2>{})>{}) == 2), true);
    EXPECT_EQ((Map::At(Number<Result::At(Number<3>{})>{}) == 3), true);
}

TEST(SequenceMapInverse, InverseIdentityMap)
{
    using Map    = Sequence<0, 1, 2, 3>;
    using Result = typename sequence_map_inverse<Map>::type;
    // Verify by checking that Map[Result[i]] == i for all i (same as the other test)
    EXPECT_EQ((Map::At(Number<Result::At(Number<0>{})>{}) == 0), true);
    EXPECT_EQ((Map::At(Number<Result::At(Number<1>{})>{}) == 1), true);
    EXPECT_EQ((Map::At(Number<Result::At(Number<2>{})>{}) == 2), true);
    EXPECT_EQ((Map::At(Number<Result::At(Number<3>{})>{}) == 3), true);
}

// Test sequence operators
TEST(SequenceOperators, Equality)
{
    constexpr auto seq1 = Sequence<1, 2, 3>{};
    constexpr auto seq2 = Sequence<1, 2, 3>{};
    constexpr auto seq3 = Sequence<1, 2, 4>{};
    EXPECT_TRUE(seq1 == seq2);
    EXPECT_FALSE(seq1 == seq3);
}

TEST(SequenceOperators, Addition)
{
    using Seq1     = Sequence<1, 2, 3>;
    using Seq2     = Sequence<4, 5, 6>;
    using Result   = decltype(Seq1{} + Seq2{});
    using Expected = Sequence<5, 7, 9>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceOperators, Subtraction)
{
    using Seq1     = Sequence<10, 20, 30>;
    using Seq2     = Sequence<1, 2, 3>;
    using Result   = decltype(Seq1{} - Seq2{});
    using Expected = Sequence<9, 18, 27>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceOperators, Multiplication)
{
    using Seq1     = Sequence<2, 3, 4>;
    using Seq2     = Sequence<5, 6, 7>;
    using Result   = decltype(Seq1{} * Seq2{});
    using Expected = Sequence<10, 18, 28>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceOperators, Division)
{
    using Seq1     = Sequence<10, 20, 30>;
    using Seq2     = Sequence<2, 4, 5>;
    using Result   = decltype(Seq1{} / Seq2{});
    using Expected = Sequence<5, 5, 6>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceOperators, Modulo)
{
    using Seq1     = Sequence<10, 20, 30>;
    using Seq2     = Sequence<3, 7, 8>;
    using Result   = decltype(Seq1{} % Seq2{});
    using Expected = Sequence<1, 6, 6>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceOperators, AdditionWithNumber)
{
    using Seq      = Sequence<1, 2, 3>;
    using Result   = decltype(Seq{} + Number<10>{});
    using Expected = Sequence<11, 12, 13>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceOperators, SubtractionWithNumber)
{
    using Seq      = Sequence<10, 20, 30>;
    using Result   = decltype(Seq{} - Number<5>{});
    using Expected = Sequence<5, 15, 25>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceOperators, MultiplicationWithNumber)
{
    using Seq      = Sequence<2, 3, 4>;
    using Result   = decltype(Seq{} * Number<3>{});
    using Expected = Sequence<6, 9, 12>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceOperators, DivisionWithNumber)
{
    using Seq      = Sequence<10, 20, 30>;
    using Result   = decltype(Seq{} / Number<5>{});
    using Expected = Sequence<2, 4, 6>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceOperators, NumberAddition)
{
    using Seq      = Sequence<1, 2, 3>;
    using Result   = decltype(Number<10>{} + Seq{});
    using Expected = Sequence<11, 12, 13>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceOperators, NumberMultiplication)
{
    using Seq      = Sequence<2, 3, 4>;
    using Result   = decltype(Number<3>{} * Seq{});
    using Expected = Sequence<6, 9, 12>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

// Test helper functions
TEST(SequenceHelpers, MergeSequences)
{
    using Seq1     = Sequence<1, 2>;
    using Seq2     = Sequence<3, 4>;
    using Seq3     = Sequence<5, 6>;
    using Result   = decltype(merge_sequences(Seq1{}, Seq2{}, Seq3{}));
    using Expected = Sequence<1, 2, 3, 4, 5, 6>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceHelpers, TransformSequencesSingle)
{
    auto double_it = [](auto x) { return 2 * x; };
    using Seq      = Sequence<1, 2, 3>;
    using Result   = decltype(transform_sequences(double_it, Seq{}));
    using Expected = Sequence<2, 4, 6>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceHelpers, TransformSequencesTwo)
{
    auto add       = [](auto x, auto y) { return x + y; };
    using Seq1     = Sequence<1, 2, 3>;
    using Seq2     = Sequence<4, 5, 6>;
    using Result   = decltype(transform_sequences(add, Seq1{}, Seq2{}));
    using Expected = Sequence<5, 7, 9>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceHelpers, TransformSequencesThree)
{
    auto add3      = [](auto x, auto y, auto z) { return x + y + z; };
    using Seq1     = Sequence<1, 2, 3>;
    using Seq2     = Sequence<4, 5, 6>;
    using Seq3     = Sequence<7, 8, 9>;
    using Result   = decltype(transform_sequences(add3, Seq1{}, Seq2{}, Seq3{}));
    using Expected = Sequence<12, 15, 18>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceHelpers, ReduceOnSequence)
{
    auto add              = [](auto x, auto y) { return x + y; };
    constexpr auto seq    = Sequence<1, 2, 3, 4, 5>{};
    constexpr auto result = reduce_on_sequence(seq, add, Number<0>{});
    EXPECT_EQ(result, 15);
}

TEST(SequenceHelpers, SequenceAnyOf)
{
    auto is_even        = [](auto x) { return x % 2 == 0; };
    constexpr auto seq1 = Sequence<1, 3, 5, 7>{};
    constexpr auto seq2 = Sequence<1, 3, 4, 7>{};
    EXPECT_FALSE(sequence_any_of(seq1, is_even));
    EXPECT_TRUE(sequence_any_of(seq2, is_even));
}

TEST(SequenceHelpers, SequenceAllOf)
{
    auto is_positive    = [](auto x) { return x > 0; };
    constexpr auto seq1 = Sequence<1, 2, 3, 4>{};
    constexpr auto seq2 = Sequence<1, -2, 3, 4>{};
    EXPECT_TRUE(sequence_all_of(seq1, is_positive));
    EXPECT_FALSE(sequence_all_of(seq2, is_positive));
}

// Test scan operations
TEST(SequenceScan, ReverseInclusiveScan)
{
    using Seq = Sequence<1, 2, 3, 4>;
    using Result =
        decltype(reverse_inclusive_scan_sequence(Seq{}, math::plus<index_t>{}, Number<0>{}));
    using Expected = Sequence<10, 9, 7, 4>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceScan, ReverseExclusiveScan)
{
    using Seq = Sequence<1, 2, 3, 4>;
    using Result =
        decltype(reverse_exclusive_scan_sequence(Seq{}, math::plus<index_t>{}, Number<0>{}));
    using Expected = Sequence<9, 7, 4, 0>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceScan, InclusiveScan)
{
    using Seq      = Sequence<1, 2, 3, 4>;
    using Result   = decltype(inclusive_scan_sequence(Seq{}, math::plus<index_t>{}, Number<0>{}));
    using Expected = Sequence<1, 3, 6, 10>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

// Test pick and modify operations
TEST(SequencePickModify, PickElementsByIds)
{
    using Seq      = Sequence<10, 20, 30, 40, 50>;
    using Ids      = Sequence<0, 2, 4>;
    using Result   = decltype(pick_sequence_elements_by_ids(Seq{}, Ids{}));
    using Expected = Sequence<10, 30, 50>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequencePickModify, PickElementsByMask)
{
    using Seq      = Sequence<10, 20, 30, 40, 50>;
    using Mask     = Sequence<1, 0, 1, 0, 1>;
    using Result   = decltype(pick_sequence_elements_by_mask(Seq{}, Mask{}));
    using Expected = Sequence<10, 30, 50>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequencePickModify, ModifyElementsByIds)
{
    using Seq      = Sequence<10, 20, 30, 40, 50>;
    using Values   = Sequence<99, 88>;
    using Ids      = Sequence<1, 3>;
    using Result   = decltype(modify_sequence_elements_by_ids(Seq{}, Values{}, Ids{}));
    using Expected = Sequence<10, 99, 30, 88, 50>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

// Test sequence_reduce
TEST(SequenceReduce, ReduceTwoSequences)
{
    using Seq1     = Sequence<1, 2, 3>;
    using Seq2     = Sequence<4, 5, 6>;
    using Result   = typename sequence_reduce<math::plus<index_t>, Seq1, Seq2>::type;
    using Expected = Sequence<5, 7, 9>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}

TEST(SequenceReduce, ReduceMultipleSequences)
{
    using Seq1     = Sequence<1, 2>;
    using Seq2     = Sequence<3, 4>;
    using Seq3     = Sequence<5, 6>;
    using Result   = typename sequence_reduce<math::plus<index_t>, Seq1, Seq2, Seq3>::type;
    using Expected = Sequence<9, 12>;
    EXPECT_TRUE((is_same<Result, Expected>::value));
}
