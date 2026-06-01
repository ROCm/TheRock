// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <ck/library/tensor_operation_instance/device_operation_instance_factory.hpp>
#include "ck_tile/builder/testing/testing.hpp"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <string>
#include <sstream>
#include <iosfwd>
#include <set>
#include <vector>
#include <array>

/// @brief ostream-overload for hipError
///
/// Google Test likes to print errors to ostream, and this provides integration
/// with that. Since we only expect to use this with CK-Builder's own tests,
/// providing this implementation seems not problematic, but if it starts to
/// clash with another implementation then we will need to provide this
/// implementation another way. Unfortunately Google Test does not have a
/// dedicated function to override to provide printing support.
std::ostream& operator<<(std::ostream& os, hipError_t status);

namespace ck_tile::builder::test {

template <auto SIGNATURE>
std::ostream& operator<<(std::ostream& os, [[maybe_unused]] Outputs<SIGNATURE> outputs)
{
    return os << "<tensor outputs>";
}

} // namespace ck_tile::builder::test

namespace ck_tile::test {

static bool isTerminalOutput() { return isatty(fileno(stdout)) || isatty(fileno(stderr)); }

// Returns a string highlighting differences between actual and expected.
// Differences are enclosed in brackets with actual and expected parts separated by '|'.
std::string inlineDiff(const std::string& actual,
                       const std::string& expected,
                       bool use_color = isTerminalOutput());

// A convenience alias for inlineDiff to improve readability in test assertions.
// Note that the function has O(n^2) complexity both in compute and in memory - do not use for very
// long strings
std::string formatInlineDiff(const std::string& actual, const std::string& expected);

// Gmock matcher for string equality with inline diff output on failure
class StringEqWithDiffMatcher : public ::testing::MatcherInterface<std::string>
{
    public:
    explicit StringEqWithDiffMatcher(const std::string& expected);

    bool MatchAndExplain(std::string actual,
                         ::testing::MatchResultListener* listener) const override;

    void DescribeTo(std::ostream* os) const override;
    void DescribeNegationTo(std::ostream* os) const override;

    private:
    std::string expected_;
};

// Factory function for the StringEqWithDiff matcher
::testing::Matcher<std::string> StringEqWithDiff(const std::string& expected);

using ck::tensor_operation::device::instance::DeviceOperationInstanceFactory;

// This utility concept checks whether a type is a valid "Device Operation" -
// that is, there is a valid specialization of `DeviceOperationInstanceFactory`
// for it available.
template <typename DeviceOp>
concept HasCkFactory = requires {
    {
        DeviceOperationInstanceFactory<DeviceOp>::GetInstances()
    } -> std::convertible_to<std::vector<std::unique_ptr<DeviceOp>>>;
};

// This structure represents a (unique) set of instances, either a statically
// defined one (for testing) or one obtained from DeviceOperationInstanceFactory.
// The idea is that we use this structure as a utility to compare a set of
// instances. Instances are stored in a set so that they can be lexicographically
// compared, this helps generating readable error messages which just contain
// the differenses between sets.
struct InstanceSet
{
    explicit InstanceSet() {}

    explicit InstanceSet(std::initializer_list<const char*> items)
        : instances(items.begin(), items.end())
    {
    }

    template <HasCkFactory DeviceOp>
    static InstanceSet from_factory()
    {
        auto set = InstanceSet();

        const auto ops = DeviceOperationInstanceFactory<DeviceOp>::GetInstances();
        for(const auto& op : ops)
        {
            set.instances.insert(op->GetInstanceString());
        }

        return set;
    }

    std::set<std::string> instances;
};

std::ostream& operator<<(std::ostream& os, const InstanceSet& set);

// This is a custom Google Test matcher which can be used to compare two sets
// of instance names, with utility functions that print a helpful error
// message about the difference between the checked sets. Use `InstancesMatch`
// to obtain an instance of this type.
struct InstanceMatcher : public ::testing::MatcherInterface<InstanceSet>
{
    explicit InstanceMatcher(const InstanceSet& expected);

    bool MatchAndExplain(InstanceSet actual,
                         ::testing::MatchResultListener* listener) const override;
    void DescribeTo(std::ostream* os) const override;
    void DescribeNegationTo(std::ostream* os) const override;

    InstanceSet expected_;
};

::testing::Matcher<InstanceSet> InstancesMatch(const InstanceSet& expected);

/// @brief Google Test hipError_t matcher.
///
/// This is a custom Google Test matcher implementation which can be used to
/// compare HIP status codes. Use `HipSuccess()` or `HipError()` to obtain
/// an instance.
///
/// @see HipSuccess
/// @see HipError
/// @see ::testing::MatcherInterface
struct HipStatusMatcher : public ::testing::MatcherInterface<hipError_t>
{
    HipStatusMatcher(hipError_t expected) : expected_(expected) {}

    bool MatchAndExplain(hipError_t actual,
                         ::testing::MatchResultListener* listener) const override;
    void DescribeTo(std::ostream* os) const override;
    void DescribeNegationTo(std::ostream* os) const override;

    hipError_t expected_;
};

/// @brief Construct a Google Test matcher that checks that a HIP operation
/// was successful.
::testing::Matcher<hipError_t> HipSuccess();

/// @brief Construct a Google Test matcher that checks that a HIP operation
/// returned a particular error code.
///
/// @param error The error to expect.
::testing::Matcher<hipError_t> HipError(hipError_t error);

template <auto SIGNATURE>
struct ReferenceOutputMatcher
    : public ::testing::MatcherInterface<builder::test::Outputs<SIGNATURE>>
{
    ReferenceOutputMatcher(const builder::test::Args<SIGNATURE>& args,
                           builder::test::Outputs<SIGNATURE> expected)
        : args_(&args), expected_(expected)
    {
    }

    bool MatchAndExplain(builder::test::Outputs<SIGNATURE> actual,
                         [[maybe_unused]] ::testing::MatchResultListener* listener) const override
    {
        const auto report = ck_tile::builder::test::validate(*args_, actual, expected_);
        const auto errors = report.get_errors();

        if(listener->IsInterested() && !errors.empty())
        {
            *listener << errors.size() << " tensors failed to validate";
        }

        return errors.empty();
    }

    void DescribeTo(std::ostream* os) const override { *os << "<tensor outputs>"; }

    void DescribeNegationTo(std::ostream* os) const override
    {
        *os << "isn't equal to <tensor outputs>";
    }

    const builder::test::Args<SIGNATURE>* args_;
    builder::test::Outputs<SIGNATURE> expected_;
};

template <auto SIGNATURE>
::testing::Matcher<builder::test::Outputs<SIGNATURE>>
MatchesReference(const builder::test::Args<SIGNATURE>& args,
                 builder::test::Outputs<SIGNATURE> expected)
{
    return ::testing::MakeMatcher(new ReferenceOutputMatcher<SIGNATURE>(args, expected));
}

} // namespace ck_tile::test
