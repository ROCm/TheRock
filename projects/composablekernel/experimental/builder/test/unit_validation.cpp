// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/builder/testing/error.hpp"
#include "ck_tile/builder/testing/tensor_buffer.hpp"
#include "ck_tile/builder/testing/tensor_descriptor.hpp"
#include "ck_tile/builder/testing/validation.hpp"
#include "ck_tile/builder/testing/tensor_foreach.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_type.hpp"
#include "ck_tile/builder/testing/testing.hpp"
#include "testing_utils.hpp"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <span>
#include <array>

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;

using testing::ElementsAreArray;
using testing::Eq;
using testing::StrEq;

using ck_tile::test::MatchesReference;
using ck_tile::test::StringEqWithDiff;

// Googletest cannot have both type AND value parameterized tests.
// For now just act lazy and use value template parameters.
template <ckb::DataType DT, ckt::Extent SHAPE, auto STRIDES>
struct Param
{
    constexpr static auto data_type = DT;
    constexpr static auto shape     = SHAPE;
    constexpr static auto strides   = STRIDES;

    constexpr static auto rank = shape.size();

    static ckt::TensorDescriptor<data_type, rank> get_descriptor()
    {
        return ckt::make_descriptor<data_type, rank>(shape, strides);
    }
};

template <typename Param>
struct ValidationReportTests : public ::testing::Test
{
};

using Types = ::testing::Types<
    Param<ckb::DataType::FP32, ckt::Extent{52, 152, 224}, ckt::PackedRightLayout{}>,
    Param<ckb::DataType::FP32, ckt::Extent{72, 1, 49, 2, 4, 5}, ckt::PackedLeftLayout{}>,
    Param<ckb::DataType::FP32, ckt::Extent{}, ckt::Extent{}>,
    Param<ckb::DataType::FP32, ckt::Extent{12, 34, 43, 21}, ckt::Extent{41, 1, 43210, 1831}>>;

TYPED_TEST_SUITE(ValidationReportTests, Types);

TYPED_TEST(ValidationReportTests, SingleCorrect)
{
    const auto desc = TypeParam::get_descriptor();

    auto a = ckt::alloc_tensor_buffer(desc);
    auto b = ckt::alloc_tensor_buffer(desc);

    ckt::clear_tensor_buffer(desc, a.get());
    ckt::clear_tensor_buffer(desc, b.get());

    // Generate a sort-of-random looking sequence
    auto generator = [strides = desc.get_strides()](const auto& index) {
        const auto flat_index = ckt::calculate_offset(index, strides);
        return static_cast<float>((flat_index + 1) * 10'000'019 % 768'351);
    };

    ckt::fill_tensor(desc, a.get(), generator);
    ckt::fill_tensor(desc, b.get(), generator);

    ckt::ValidationReport report;
    report.check("correct", desc, b.get(), a.get());

    EXPECT_THAT(report.get_errors().size(), Eq(0));
}

TYPED_TEST(ValidationReportTests, SingleIncorrect)
{
    const auto desc           = TypeParam::get_descriptor();
    const auto packed_strides = ckt::PackedRightLayout{}(desc.get_lengths());

    auto a = ckt::alloc_tensor_buffer(desc);
    auto b = ckt::alloc_tensor_buffer(desc);

    ckt::clear_tensor_buffer(desc, a.get());
    ckt::clear_tensor_buffer(desc, b.get());

    ckt::fill_tensor(desc, a.get(), []([[maybe_unused]] const auto& i) { return 123; });
    ckt::fill_tensor(desc, b.get(), [packed_strides](const auto& index) {
        const auto flat_index = ckt::calculate_offset(index, packed_strides);
        return flat_index == 0 ? 0 : flat_index == 12345 ? 456 : flat_index == 999999 ? 1 : 123;
    });

    ckt::ValidationReport report;
    report.check("incorrect", desc, b.get(), a.get());

    const auto errors = report.get_errors();

    const auto flat_size       = desc.get_element_size();
    const auto expected_errors = flat_size >= 999999 ? 3 : flat_size >= 12345 ? 2 : 1;

    ASSERT_THAT(errors.size(), Eq(1));
    EXPECT_THAT(errors[0].tensor_name, StrEq("incorrect"));
    EXPECT_THAT(errors[0].wrong_elements, Eq(expected_errors));
    EXPECT_THAT(errors[0].total_elements, Eq(desc.get_element_size()));
}

TYPED_TEST(ValidationReportTests, ZeroIsIncorrect)
{
    const auto desc = TypeParam::get_descriptor();

    auto a = ckt::alloc_tensor_buffer(desc);
    auto b = ckt::alloc_tensor_buffer(desc);

    ckt::clear_tensor_buffer(desc, a.get());
    ckt::clear_tensor_buffer(desc, b.get());

    ckt::ValidationReport report;
    report.check("zero_is_incorrect", desc, b.get(), a.get());

    const auto errors = report.get_errors();
    ASSERT_THAT(errors.size(), Eq(1));
    EXPECT_THAT(errors[0].tensor_name, StrEq("zero_is_incorrect"));
    EXPECT_THAT(errors[0].wrong_elements, Eq(0));
    EXPECT_THAT(errors[0].total_elements, Eq(desc.get_element_size()));
    EXPECT_THAT(errors[0].zero_elements, Eq(desc.get_element_size()));
}

TEST(ValidationReportTests, MultipleSomeIncorrect)
{
    ckt::ValidationReport report;

    {
        auto desc = ckt::make_descriptor<ckb::DataType::BF16, 4>({'R', 'O', 'C', 'm'},
                                                                 ckt::PackedLeftLayout{});

        auto a = ckt::alloc_tensor_buffer(desc);
        auto b = ckt::alloc_tensor_buffer(desc);

        ckt::fill_tensor_buffer(
            desc, a.get(), [](size_t i) { return ck::type_convert<ck::bhalf_t>(i % 100); });
        ckt::fill_tensor_buffer(
            desc, b.get(), [](size_t i) { return ck::type_convert<ck::bhalf_t>(i % 101); });

        report.check("incorrect 1", desc, b.get(), a.get());
    }

    {
        auto desc =
            ckt::make_descriptor<ckb::DataType::U8, 3>({'H', 'I', 'P'}, ckt::PackedRightLayout{});

        auto a = ckt::alloc_tensor_buffer(desc);
        auto b = ckt::alloc_tensor_buffer(desc);

        ckt::fill_tensor_buffer(desc, a.get(), [](size_t i) { return "ROCm"[i % 4]; });
        ckt::fill_tensor_buffer(desc, b.get(), [](size_t i) {
            switch(i % 4)
            {
            case 0: return 'R';
            case 1: return 'O';
            case 2: return 'C';
            case 3: return 'm';
            default: return 'x';
            }
        });

        report.check("correct", desc, b.get(), a.get());
    }

    {
        auto desc =
            ckt::make_descriptor<ckb::DataType::I32, 3>({'G', 'P', 'U'}, ckt::PackedRightLayout{});

        auto a = ckt::alloc_tensor_buffer(desc);
        auto b = ckt::alloc_tensor_buffer(desc);

        ckt::fill_tensor_buffer(desc, a.get(), []([[maybe_unused]] size_t i) { return 1; });
        ckt::fill_tensor_buffer(desc, b.get(), []([[maybe_unused]] size_t i) { return 555; });

        report.check("incorrect 2", desc, b.get(), a.get());
    }

    const auto errors = report.get_errors();

    ASSERT_THAT(errors.size(), Eq(2));
    EXPECT_THAT(errors[0].tensor_name, StrEq("incorrect 1"));
    EXPECT_THAT(errors[0].wrong_elements, Eq(46840334));
    EXPECT_THAT(errors[1].tensor_name, StrEq("incorrect 2"));
    EXPECT_THAT(errors[1].wrong_elements, Eq(482800));
}

// MatchesReference operates on the types defined in testing.hpp, so just
// quickly define a bunch of dummy values for that.

struct DummySignature
{
};

constexpr DummySignature DUMMY_SIGNATURE = {};

namespace ck_tile::builder::test {
template <>
struct Args<DUMMY_SIGNATURE>
{
    auto make_a_descriptor() const
    {
        return make_descriptor<builder::DataType::FP32>(Extent{5, 5, 5, 5}, PackedRightLayout{});
    }

    auto make_b_descriptor() const
    {
        return make_descriptor<builder::DataType::FP16>(Extent{100000}, PackedLeftLayout{});
    }
};

template <>
struct Outputs<DUMMY_SIGNATURE>
{
    void* a;
    void* b;
};

template <>
ValidationReport validate<DUMMY_SIGNATURE>(const Args<DUMMY_SIGNATURE>& args,
                                           Outputs<DUMMY_SIGNATURE> actual,
                                           Outputs<DUMMY_SIGNATURE> expected)
{
    ValidationReport report;
    report.check("a", args.make_a_descriptor(), actual.a, expected.a);
    report.check("b", args.make_b_descriptor(), actual.b, expected.b);
    return report;
}

} // namespace ck_tile::builder::test

TEST(MatchesReference, Correct)
{
    const ckt::Args<DUMMY_SIGNATURE> args;

    const auto a_desc = args.make_a_descriptor();
    const auto b_desc = args.make_b_descriptor();

    auto a_actual = ckt::alloc_tensor_buffer(a_desc);
    auto b_actual = ckt::alloc_tensor_buffer(b_desc);
    ckt::clear_tensor_buffer(a_desc, a_actual.get(), 1);
    ckt::clear_tensor_buffer(b_desc, b_actual.get(), 2);
    const auto actual = ckt::Outputs<DUMMY_SIGNATURE>{
        .a = a_actual.get(),
        .b = b_actual.get(),
    };

    auto a_expected = ckt::alloc_tensor_buffer(a_desc);
    auto b_expected = ckt::alloc_tensor_buffer(b_desc);
    ckt::clear_tensor_buffer(a_desc, a_expected.get(), 1);
    ckt::clear_tensor_buffer(b_desc, b_expected.get(), 2);
    const auto expected = ckt::Outputs<DUMMY_SIGNATURE>{
        .a = a_expected.get(),
        .b = b_expected.get(),
    };

    EXPECT_THAT(actual, MatchesReference(args, expected));
}

TEST(MatchesReference, Incorrect)
{
    const ckt::Args<DUMMY_SIGNATURE> args;

    const auto a_desc = args.make_a_descriptor();
    const auto b_desc = args.make_b_descriptor();

    auto a_actual = ckt::alloc_tensor_buffer(a_desc);
    auto b_actual = ckt::alloc_tensor_buffer(b_desc);
    ckt::clear_tensor_buffer(a_desc, a_actual.get(), 1);
    ckt::clear_tensor_buffer(b_desc, b_actual.get(), 2);
    const auto actual = ckt::Outputs<DUMMY_SIGNATURE>{
        .a = a_actual.get(),
        .b = b_actual.get(),
    };

    auto a_expected = ckt::alloc_tensor_buffer(a_desc);
    auto b_expected = ckt::alloc_tensor_buffer(b_desc);
    ckt::clear_tensor_buffer(a_desc, a_expected.get(), 2);
    ckt::clear_tensor_buffer(b_desc, b_expected.get(), 2);
    const auto expected = ckt::Outputs<DUMMY_SIGNATURE>{
        .a = a_expected.get(),
        .b = b_expected.get(),
    };

    testing::StringMatchResultListener listener;
    EXPECT_TRUE(!ExplainMatchResult(MatchesReference(args, expected), actual, &listener));

    EXPECT_THAT(listener.str(), StringEqWithDiff("1 tensors failed to validate"));
}
