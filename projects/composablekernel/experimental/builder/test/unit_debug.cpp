// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/builder/testing/tensor_descriptor.hpp"
#include "ck_tile/builder/testing/tensor_foreach.hpp"
#include "ck_tile/builder/testing/debug.hpp"
#include "testing_utils.hpp"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <sstream>
#include <vector>

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;

using ck_tile::test::StringEqWithDiff;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::Gt;

TEST(Debug, PrintDescriptor)
{
    auto desc =
        ckt::make_descriptor<ckb::DataType::I32>(ckt::Extent{10, 11, 12}, ckt::PackedRightLayout{});

    std::stringstream ss;
    ckt::print_descriptor("test", desc, ss);

    EXPECT_THAT(ss.str(),
                StringEqWithDiff( //
                    "Descriptor \"test\":\n"
                    "  data type: I32\n"
                    "  size:      1'320 elements\n"
                    "  space:     1'320 elements (5'280 bytes)\n"
                    "  lengths:   [10, 11, 12]\n"
                    "  strides:   [132, 12, 1]\n"
                    "  packed:    yes\n"));

    // Make sure that the stream locale does not leak.
    ss.str("");
    ss << 1000;
    EXPECT_THAT(ss.str(), StringEqWithDiff("1000"));
}

TEST(Debug, LimitedForeach)
{
    {
        std::vector<size_t> values;
        size_t delim_count = 0;
        ckt::detail::limited_foreach(
            10,
            2,
            [&](auto i) { values.push_back(i); },
            [&](auto skip_count) {
                ++delim_count;
                EXPECT_THAT(skip_count, Eq(10 - 2));
            });
        EXPECT_THAT(values, ElementsAreArray({0, 9}));
        EXPECT_THAT(delim_count, Eq(1));
    }

    {
        std::vector<size_t> values;
        size_t delim_count = 0;
        ckt::detail::limited_foreach(
            100,
            9,
            [&](auto i) { values.push_back(i); },
            [&](auto skip_count) {
                ++delim_count;
                EXPECT_THAT(skip_count, Eq(100 - 9));
            });
        EXPECT_THAT(values, ElementsAreArray({0, 1, 2, 3, 4, 96, 97, 98, 99}));
        EXPECT_THAT(delim_count, Eq(1));
    }

    {
        size_t call_count  = 0;
        size_t delim_count = 0;
        ckt::detail::limited_foreach(
            50,
            100,
            [&](auto i) {
                EXPECT_THAT(i, Eq(call_count));
                ++call_count;
            },
            [&]([[maybe_unused]] auto skip_count) { ++delim_count; });
        EXPECT_THAT(call_count, Eq(50));
        EXPECT_THAT(delim_count, Eq(0));
    }
}

TEST(Debug, PrintTensor0D)
{
    auto desc = ckt::make_descriptor<ckb::DataType::I32>(ckt::Extent{}, ckt::PackedRightLayout{});

    auto a = ckt::alloc_tensor_buffer(desc);
    ckt::fill_tensor_buffer(desc, a.get(), []([[maybe_unused]] size_t i) { return 123; });

    std::stringstream ss;
    ckt::print_tensor("0D", desc, a.get(), {}, ss);

    EXPECT_THAT(ss.str(),
                StringEqWithDiff( //
                    "Tensor \"0D\": shape = []\n"
                    "  123\n"));
}

TEST(Debug, PrintTensor1D)
{
    auto desc = ckt::make_descriptor<ckb::DataType::I32>(ckt::Extent{44}, ckt::PackedRightLayout{});

    auto a = ckt::alloc_tensor_buffer(desc);
    ckt::fill_tensor_buffer(desc, a.get(), [](size_t i) { return i % 7; });

    std::stringstream ss;
    ckt::print_tensor("1D", desc, a.get(), {}, ss);

    // Note: output does not involve the size of the matrix separator fields,
    // since these are not printed.
    EXPECT_THAT(ss.str(),
                StringEqWithDiff( //
                    "Tensor \"1D\": shape = [44]\n"
                    "  0 1 2 3 4 ... 4 5 6 0 1\n"));
}

TEST(Debug, PrintTensor4D)
{
    auto desc = ckt::make_descriptor<ckb::DataType::I32>(ckt::Extent{100, 110, 120, 130},
                                                         ckt::PackedRightLayout{});

    auto a = ckt::alloc_tensor_buffer(desc);
    ckt::fill_tensor_buffer(desc, a.get(), [](size_t i) { return i; });

    std::stringstream ss;
    ckt::print_tensor("4D",
                      desc,
                      a.get(),
                      {
                          // Reduce default limits to have smaller output here.
                          // That also tests that we can configure these (to some
                          // extent).
                          .col_limit   = 4,
                          .row_limit   = 4,
                          .slice_limit = 4,
                      },
                      ss);

    EXPECT_THAT(ss.str(),
                StringEqWithDiff( //
                    "Tensor \"4D\": shape = [100, 110, 120, 130]\n"
                    "Tensor \"4D\", slice [0, 0, :, :]\n"
                    "          0         1 ...       128       129\n"
                    "        130       131 ...       258       259\n"
                    "        ...       ... ...       ...       ...\n"
                    "      15340     15341 ...     15468     15469\n"
                    "      15470     15471 ...     15598     15599\n"
                    "\n"
                    "Tensor \"4D\", slice [0, 1, :, :]\n"
                    "      15600     15601 ...     15728     15729\n"
                    "      15730     15731 ...     15858     15859\n"
                    "        ...       ... ...       ...       ...\n"
                    "      30940     30941 ...     31068     31069\n"
                    "      31070     31071 ...     31198     31199\n"
                    "\n"
                    "(skipping 10'996 slices...)\n"
                    "\n"
                    "Tensor \"4D\", slice [99, 108, :, :]\n"
                    "  171568800 171568801 ... 171568928 171568929\n"
                    "  171568930 171568931 ... 171569058 171569059\n"
                    "        ...       ... ...       ...       ...\n"
                    "  171584140 171584141 ... 171584268 171584269\n"
                    "  171584270 171584271 ... 171584398 171584399\n"
                    "\n"
                    "Tensor \"4D\", slice [99, 109, :, :]\n"
                    "  171584400 171584401 ... 171584528 171584529\n"
                    "  171584530 171584531 ... 171584658 171584659\n"
                    "        ...       ... ...       ...       ...\n"
                    "  171599740 171599741 ... 171599868 171599869\n"
                    "  171599870 171599871 ... 171599998 171599999\n"));
}

TEST(Debug, PrintTensorCustomConfig)
{
    auto desc =
        ckt::make_descriptor<ckb::DataType::I32>(ckt::Extent{10, 10, 10}, ckt::PackedRightLayout{});

    auto a = ckt::alloc_tensor_buffer(desc);
    ckt::fill_tensor_buffer(desc, a.get(), [](size_t i) { return i * 101 % 77; });

    std::stringstream ss;
    ckt::print_tensor("CustomConfig",
                      desc,
                      a.get(),
                      {
                          // Reduce default limits to have smaller output here.
                          // That also tests that we can configure these.
                          .col_limit   = 4,
                          .row_limit   = 2,
                          .slice_limit = 6,
                          // Try with different sizes to make sure that the alignment
                          // is still correct after changing these.
                          .row_prefix          = ">>>>",
                          .row_field_sep       = "|||||",
                          .row_skip_val        = "-------",
                          .matrix_row_skip_val = "&&&&&&&&",
                      },
                      ss);

    EXPECT_THAT(ss.str(),
                StringEqWithDiff( //
                    "Tensor \"CustomConfig\": shape = [10, 10, 10]\n"
                    "Tensor \"CustomConfig\", slice [0, :, :]\n"
                    ">>>>|||||       0|||||      24|||||-------|||||      38|||||      62\n"
                    ">>>>|||||&&&&&&&&|||||&&&&&&&&|||||-------|||||&&&&&&&&|||||&&&&&&&&\n"
                    ">>>>|||||       4|||||      28|||||-------|||||      42|||||      66\n"
                    "\n"
                    "Tensor \"CustomConfig\", slice [1, :, :]\n"
                    ">>>>|||||      13|||||      37|||||-------|||||      51|||||      75\n"
                    ">>>>|||||&&&&&&&&|||||&&&&&&&&|||||-------|||||&&&&&&&&|||||&&&&&&&&\n"
                    ">>>>|||||      17|||||      41|||||-------|||||      55|||||       2\n"
                    "\n"
                    "Tensor \"CustomConfig\", slice [2, :, :]\n"
                    ">>>>|||||      26|||||      50|||||-------|||||      64|||||      11\n"
                    ">>>>|||||&&&&&&&&|||||&&&&&&&&|||||-------|||||&&&&&&&&|||||&&&&&&&&\n"
                    ">>>>|||||      30|||||      54|||||-------|||||      68|||||      15\n"
                    "\n"
                    "(skipping 4 slices...)\n"
                    "\n"
                    "Tensor \"CustomConfig\", slice [7, :, :]\n"
                    ">>>>|||||      14|||||      38|||||-------|||||      52|||||      76\n"
                    ">>>>|||||&&&&&&&&|||||&&&&&&&&|||||-------|||||&&&&&&&&|||||&&&&&&&&\n"
                    ">>>>|||||      18|||||      42|||||-------|||||      56|||||       3\n"
                    "\n"
                    "Tensor \"CustomConfig\", slice [8, :, :]\n"
                    ">>>>|||||      27|||||      51|||||-------|||||      65|||||      12\n"
                    ">>>>|||||&&&&&&&&|||||&&&&&&&&|||||-------|||||&&&&&&&&|||||&&&&&&&&\n"
                    ">>>>|||||      31|||||      55|||||-------|||||      69|||||      16\n"
                    "\n"
                    "Tensor \"CustomConfig\", slice [9, :, :]\n"
                    ">>>>|||||      40|||||      64|||||-------|||||       1|||||      25\n"
                    ">>>>|||||&&&&&&&&|||||&&&&&&&&|||||-------|||||&&&&&&&&|||||&&&&&&&&\n"
                    ">>>>|||||      44|||||      68|||||-------|||||       5|||||      29\n"));
}

TEST(Debug, PrintTensorUnlimitedMatrix)
{
    // To limit the output of the test, split the "unlimited" test up into one for the
    // matrices and one for the slices.

    const ckt::Extent shape = ckt::Extent{12, 12};
    const ckt::TensorPrintConfig default_config;

    // The shape should be larger than the default, otherwise this test doesn't make
    // any sense.
    ASSERT_THAT(shape[1], Gt(default_config.col_limit));
    ASSERT_THAT(shape[2], Gt(default_config.row_limit));

    auto desc = ckt::make_descriptor<ckb::DataType::I32>(shape, ckt::PackedRightLayout{});

    auto a = ckt::alloc_tensor_buffer(desc);
    ckt::fill_tensor_buffer(desc, a.get(), [](size_t i) { return i ^ 0xF; });

    std::stringstream ss;
    ckt::print_tensor("UnlimitedConfig", desc, a.get(), ckt::TensorPrintConfig::unlimited(), ss);

    EXPECT_THAT(ss.str(),
                StringEqWithDiff( //
                    "Tensor \"UnlimitedConfig\": shape = [12, 12]\n"
                    "   15  14  13  12  11  10   9   8   7   6   5   4\n"
                    "    3   2   1   0  31  30  29  28  27  26  25  24\n"
                    "   23  22  21  20  19  18  17  16  47  46  45  44\n"
                    "   43  42  41  40  39  38  37  36  35  34  33  32\n"
                    "   63  62  61  60  59  58  57  56  55  54  53  52\n"
                    "   51  50  49  48  79  78  77  76  75  74  73  72\n"
                    "   71  70  69  68  67  66  65  64  95  94  93  92\n"
                    "   91  90  89  88  87  86  85  84  83  82  81  80\n"
                    "  111 110 109 108 107 106 105 104 103 102 101 100\n"
                    "   99  98  97  96 127 126 125 124 123 122 121 120\n"
                    "  119 118 117 116 115 114 113 112 143 142 141 140\n"
                    "  139 138 137 136 135 134 133 132 131 130 129 128\n"));
}

TEST(Debug, PrintTensorUnlimitedSlices)
{
    // To limit the output of the test, split the "unlimited" test up into one for the
    // matrices and one for the slices.

    const ckt::Extent shape = ckt::Extent{13, 1, 1};
    const ckt::TensorPrintConfig default_config;

    // The shape should be larger than the default, otherwise this test doesn't make
    // any sense.
    ASSERT_THAT(shape[0], Gt(default_config.slice_limit));

    auto desc = ckt::make_descriptor<ckb::DataType::I32>(shape, ckt::PackedRightLayout{});

    auto a = ckt::alloc_tensor_buffer(desc);
    ckt::fill_tensor_buffer(desc, a.get(), [](size_t i) { return i * 3; });

    std::stringstream ss;
    ckt::print_tensor("UnlimitedConfig", desc, a.get(), ckt::TensorPrintConfig::unlimited(), ss);

    EXPECT_THAT(ss.str(),
                StringEqWithDiff( //
                    "Tensor \"UnlimitedConfig\": shape = [13, 1, 1]\n"
                    "Tensor \"UnlimitedConfig\", slice [0, :, :]\n"
                    "   0\n"
                    "\n"
                    "Tensor \"UnlimitedConfig\", slice [1, :, :]\n"
                    "   3\n"
                    "\n"
                    "Tensor \"UnlimitedConfig\", slice [2, :, :]\n"
                    "   6\n"
                    "\n"
                    "Tensor \"UnlimitedConfig\", slice [3, :, :]\n"
                    "   9\n"
                    "\n"
                    "Tensor \"UnlimitedConfig\", slice [4, :, :]\n"
                    "  12\n"
                    "\n"
                    "Tensor \"UnlimitedConfig\", slice [5, :, :]\n"
                    "  15\n"
                    "\n"
                    "Tensor \"UnlimitedConfig\", slice [6, :, :]\n"
                    "  18\n"
                    "\n"
                    "Tensor \"UnlimitedConfig\", slice [7, :, :]\n"
                    "  21\n"
                    "\n"
                    "Tensor \"UnlimitedConfig\", slice [8, :, :]\n"
                    "  24\n"
                    "\n"
                    "Tensor \"UnlimitedConfig\", slice [9, :, :]\n"
                    "  27\n"
                    "\n"
                    "Tensor \"UnlimitedConfig\", slice [10, :, :]\n"
                    "  30\n"
                    "\n"
                    "Tensor \"UnlimitedConfig\", slice [11, :, :]\n"
                    "  33\n"
                    "\n"
                    "Tensor \"UnlimitedConfig\", slice [12, :, :]\n"
                    "  36\n"));
}

TEST(Debug, PrintTensorFP32)
{
    auto desc =
        ckt::make_descriptor<ckb::DataType::FP32>(ckt::Extent{5, 5}, ckt::PackedRightLayout{});

    auto a = ckt::alloc_tensor_buffer(desc);
    ckt::fill_tensor_buffer(desc, a.get(), [](size_t i) { return std::pow(1.9999, i); });

    std::stringstream ss;
    ckt::print_tensor("FP32", desc, a.get(), {}, ss);

    EXPECT_THAT(ss.str(),
                StringEqWithDiff( //
                    "Tensor \"FP32\": shape = [5, 5]\n"
                    "         1.000        2.000        4.000        7.999       15.997\n"
                    "        31.992       63.981      127.955      255.898      511.770\n"
                    "      1023.488     2046.874     4093.543     8186.677    16372.535\n"
                    "     32743.432    65483.590   130960.633   261908.172   523790.156\n"
                    "   1047527.938  2094951.125  4189692.750  8378966.500 16757095.000\n"));
}

TEST(Debug, PrintTensorBF16)
{
    auto desc =
        ckt::make_descriptor<ckb::DataType::BF16>(ckt::Extent{5, 5}, ckt::PackedRightLayout{});

    auto a = ckt::alloc_tensor_buffer(desc);
    ckt::fill_tensor_buffer(
        desc, a.get(), [](size_t i) { return ck::type_convert<ck::bhalf_t>(1.2345678f * i); });

    std::stringstream ss;
    ckt::print_tensor("BF16", desc, a.get(), {}, ss);

    EXPECT_THAT(ss.str(),
                StringEqWithDiff( //
                    "Tensor \"BF16\": shape = [5, 5]\n"
                    "   0.000  1.234  2.469  3.703  4.938\n"
                    "   6.188  7.406  8.625  9.875 11.125\n"
                    "  12.375 13.562 14.812 16.000 17.250\n"
                    "  18.500 19.750 21.000 22.250 23.500\n"
                    "  24.750 25.875 27.125 28.375 29.625\n"));
}

TEST(Debug, PrintTensorFP8)
{
    auto desc =
        ckt::make_descriptor<ckb::DataType::FP8>(ckt::Extent{5, 5}, ckt::PackedRightLayout{});

    auto a = ckt::alloc_tensor_buffer(desc);
    ckt::fill_tensor_buffer(
        desc, a.get(), [](size_t i) { return ck::type_convert<ck::f8_t>(i * 0.1f); });

    std::stringstream ss;
    ckt::print_tensor("FP8", desc, a.get(), {}, ss);

    EXPECT_THAT(ss.str(),
                StringEqWithDiff( //
                    "Tensor \"FP8\": shape = [5, 5]\n"
                    "  0.000 0.102 0.203 0.312 0.406\n"
                    "  0.500 0.625 0.688 0.812 0.875\n"
                    "  1.000 1.125 1.250 1.250 1.375\n"
                    "  1.500 1.625 1.750 1.750 1.875\n"
                    "  2.000 2.000 2.250 2.250 2.500\n"));
}

TEST(Debug, PrintTensorSpecialFloats)
{
    auto desc =
        ckt::make_descriptor<ckb::DataType::FP32>(ckt::Extent{5, 5}, ckt::PackedRightLayout{});

    auto a = ckt::alloc_tensor_buffer(desc);
    ckt::fill_tensor_buffer(desc, a.get(), [](size_t i) {
        if(i % 8 == 1)
            return 0.f / 0.f;
        else if(i % 7 == 1)
            return std::sqrt(-1.f);
        else if(i % 6 == 1)
            return 1.f / 0.f;
        else if(i % 5 == 1)
            return -1.f / 0.f;
        else
            return static_cast<float>(i);
    });

    std::stringstream ss;
    ckt::print_tensor("specials", desc, a.get(), {}, ss);

    EXPECT_THAT(ss.str(),
                StringEqWithDiff( //
                    "Tensor \"specials\": shape = [5, 5]\n"
                    "   0.000    nan  2.000  3.000  4.000\n"
                    "   5.000   -inf    inf   -nan    nan\n"
                    "  10.000   -inf 12.000    inf 14.000\n"
                    "    -nan   -inf    nan 18.000    inf\n"
                    "  20.000   -inf   -nan 23.000 24.000\n"));
}

TEST(Debug, PrintTensorFloatPrecision)
{
    auto desc = ckt::make_descriptor<ckb::DataType::FP32>(ckt::Extent{5}, ckt::PackedRightLayout{});

    auto a = ckt::alloc_tensor_buffer(desc);
    ckt::fill_tensor_buffer(desc, a.get(), [](size_t i) { return std::pow(0.9, i); });

    std::stringstream ss;
    ckt::print_tensor("FloatPrecision",
                      desc,
                      a.get(),
                      {
                          .float_precision = 10,
                      },
                      ss);

    EXPECT_THAT(ss.str(),
                StringEqWithDiff( //
                    "Tensor \"FloatPrecision\": shape = [5]\n"
                    "  1.0000000000 0.8999999762 0.8100000024 0.7289999723 0.6560999751\n"));
}
