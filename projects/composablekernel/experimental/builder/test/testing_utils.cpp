// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "testing_utils.hpp"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <unistd.h>
#include <string>
#include <sstream>
#include <ostream>
#include <vector>
#include <algorithm>

std::ostream& operator<<(std::ostream& os, hipError_t status)
{
    return os << hipGetErrorString(status);
}

namespace ck_tile::test {

// Wagner-Fischer Algorithm for Computing Edit Distance and Inline Diff
//
// OUTPUT FORMAT: [expected|actual] for differences, plain text for matches
// Example: "hello world" vs "hello earth" → "hello [world|earth]"
//
// This function implements the Wagner-Fischer algorithm (1974), which is the classic
// dynamic programming solution for computing the minimum edit distance (Levenshtein distance)
// between two strings. The algorithm has O(n*m) time and space complexity.
//
// ALGORITHM OVERVIEW:
// 1. Build a 2D DP table where dp[i][j] represents the minimum edit distance
//    between the first i characters of 'expected' and first j characters of 'actual'
// 2. Fill the table using the recurrence relation:
//    dp[i][j] = min(
//        dp[i-1][j] + 1,     // deletion (remove char from expected)
//        dp[i][j-1] + 1,     // insertion (add char to expected)
//        dp[i-1][j-1] + cost // substitution (cost=0 if chars match, 1 if different)
//    )
// 3. Backtrack through the table to reconstruct the optimal edit sequence
//
// REFERENCES:
// - Wagner, R. A.; Fischer, M. J. (1974). "The String-to-String Correction Problem"
// - Also known as: Levenshtein distance, edit distance, string alignment
// - Similar to sequence alignment algorithms used in bioinformatics (Needleman-Wunsch)
std::string inlineDiff(const std::string& actual, const std::string& expected, bool use_color)
{

    const char* EXPECTED_COLOR = use_color ? "\033[36m" : ""; // Cyan
    const char* ACTUAL_COLOR   = use_color ? "\033[35m" : ""; // Magenta
    const char* RESET          = use_color ? "\033[0m" : "";

    const size_t n = expected.length(); // Length of expected string
    const size_t m = actual.length();   // Length of actual string

    // PHASE 1: Build the Dynamic Programming Table
    // dp[i][j] = minimum edit distance between expected[0..i-1] and actual[0..j-1]
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1));

    // Base cases: transforming empty string to/from prefixes
    for(size_t i = 0; i <= n; ++i)
    {
        dp[i][0] = i; // Delete i characters from expected to get empty string
    }
    for(size_t j = 0; j <= m; ++j)
    {
        dp[0][j] = j; // Insert j characters to empty string to get actual[0..j-1]
    }

    // Fill the DP table using the Wagner-Fischer recurrence relation
    for(size_t i = 1; i <= n; ++i)
    {
        for(size_t j = 1; j <= m; ++j)
        {
            // Cost is 0 if characters match, 1 if they need substitution
            int cost = (expected[i - 1] == actual[j - 1]) ? 0 : 1;

            // Choose the minimum cost operation:
            dp[i][j] = std::min({
                dp[i - 1][j] + 1,       // Deletion: remove expected[i-1]
                dp[i][j - 1] + 1,       // Insertion: add actual[j-1]
                dp[i - 1][j - 1] + cost // Substitution/Match
            });
        }
    }

    // PHASE 2: Backtrack to Reconstruct the Optimal Edit Sequence
    // We trace back from dp[n][m] to dp[0][0] to find which operations were used
    std::vector<char> operations; // 'M'atch, 'S'ubstitution, 'I'nsertion, 'D'eletion
    std::vector<std::pair<char, char>> diff_chars; // Character pairs for each operation

    size_t i = n, j = m; // Start from bottom-right corner of DP table
    while(i > 0 || j > 0)
    {
        // Determine which operation led to the current cell's value
        int cost = (i > 0 && j > 0 && expected[i - 1] == actual[j - 1]) ? 0 : 1;

        // Check if we came from diagonal (substitution/match)
        if(i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] + cost)
        {
            if(cost == 0)
            {
                operations.push_back('M'); // Characters match
                diff_chars.push_back({expected[i - 1], actual[j - 1]});
            }
            else
            {
                operations.push_back('S'); // Substitution needed
                diff_chars.push_back({expected[i - 1], actual[j - 1]});
            }
            --i;
            --j; // Move diagonally up-left
        }
        // Check if we came from left (insertion)
        else if(j > 0 && dp[i][j] == dp[i][j - 1] + 1)
        {
            operations.push_back('I'); // Insertion: actual has extra character
            diff_chars.push_back({'\0', actual[j - 1]});
            --j; // Move left
        }
        // Must have come from above (deletion)
        else if(i > 0 && dp[i][j] == dp[i - 1][j] + 1)
        {
            operations.push_back('D'); // Deletion: expected has extra character
            diff_chars.push_back({expected[i - 1], '\0'});
            --i; // Move up
        }
    }

    // PHASE 3: Reverse and Build the Human-Readable Diff String
    // Backtracking gives us operations in reverse order, so we reverse to get forward order
    std::reverse(operations.begin(), operations.end());
    std::reverse(diff_chars.begin(), diff_chars.end());

    // Build the final diff string with color highlighting
    std::ostringstream diff;
    std::string expected_diff, actual_diff; // Accumulate consecutive differences
    bool in_diff = false;                   // Track whether we're inside a diff section

    for(size_t k = 0; k < operations.size(); ++k)
    {
        char op       = operations[k];
        char exp_char = diff_chars[k].first;  // Expected character ('\0' for insertions)
        char act_char = diff_chars[k].second; // Actual character ('\0' for deletions)

        if(op == 'M') // Match - characters are identical
        {
            if(in_diff)
            {
                // Close the current diff section and output it
                diff << "[" << EXPECTED_COLOR << expected_diff << RESET << "|" << ACTUAL_COLOR
                     << actual_diff << RESET << "]";
                expected_diff.clear();
                actual_diff.clear();
                in_diff = false;
            }
            diff << exp_char; // Output the matching character as-is
        }
        else // Difference (substitution, insertion, or deletion)
        {
            in_diff = true;
            // Accumulate characters for the diff section
            if(exp_char != '\0')
                expected_diff += exp_char; // Add to expected side
            if(act_char != '\0')
                actual_diff += act_char; // Add to actual side
        }
    }

    // Close any remaining diff section at the end
    if(in_diff)
    {
        diff << "[" << EXPECTED_COLOR << expected_diff << RESET << "|" << ACTUAL_COLOR
             << actual_diff << RESET << "]";
    }

    return diff.str();
}

std::string formatInlineDiff(const std::string& actual, const std::string& expected)
{
    return std::string("Inline diff:  \"") + inlineDiff(actual, expected) + "\"";
}

// StringEqWithDiffMatcher implementation
StringEqWithDiffMatcher::StringEqWithDiffMatcher(const std::string& expected) : expected_(expected)
{
}

bool StringEqWithDiffMatcher::MatchAndExplain(std::string actual,
                                              ::testing::MatchResultListener* listener) const
{
    if(actual == expected_)
    {
        return true;
    }

    // On failure, provide detailed diff information
    if(listener->IsInterested())
    {
        *listener << "\n    Diff: \"" << inlineDiff(actual, expected_) << "\"";
    }
    return false;
}

void StringEqWithDiffMatcher::DescribeTo(std::ostream* os) const
{
    *os << "\"" << expected_ << "\"";
}

void StringEqWithDiffMatcher::DescribeNegationTo(std::ostream* os) const
{
    *os << "is not equal to \"" << expected_ << "\"";
}

// Factory function for the StringEqWithDiff matcher
::testing::Matcher<std::string> StringEqWithDiff(const std::string& expected)
{
    return ::testing::MakeMatcher(new StringEqWithDiffMatcher(expected));
}

std::ostream& operator<<(std::ostream& os, const InstanceSet& set)
{
    // These sets can grow very large, and so its not very nice or useful to print them
    // in the event of a mismatch. Just print a brief description here, and use
    // InstancesMatcher to print a more useful message.
    return (os << "(set of " << set.instances.size() << " instances)");
}

InstanceMatcher::InstanceMatcher(const InstanceSet& expected) : expected_(expected) {}

::testing::Matcher<InstanceSet> InstancesMatch(const InstanceSet& expected)
{
    return ::testing::MakeMatcher(new InstanceMatcher(expected));
}

bool InstanceMatcher::MatchAndExplain(InstanceSet actual,
                                      ::testing::MatchResultListener* listener) const
{
    if(actual.instances == expected_.instances)
    {
        return true;
    }

    if(listener->IsInterested())
    {
        std::vector<std::string> instances;
        std::set_difference(expected_.instances.begin(),
                            expected_.instances.end(),
                            actual.instances.begin(),
                            actual.instances.end(),
                            std::back_inserter(instances));

        *listener << "\n";

        if(instances.size() > 0)
        {
            *listener << " Missing: " << instances.size() << "\n";
            for(const auto& instance : instances)
            {
                if(instance == "")
                {
                    *listener << "- (empty string)\n";
                }
                else
                {
                    *listener << "- " << instance << "\n";
                }
            }
        }

        instances.clear();
        std::set_difference(actual.instances.begin(),
                            actual.instances.end(),
                            expected_.instances.begin(),
                            expected_.instances.end(),
                            std::back_inserter(instances));

        if(instances.size() > 0)
        {
            *listener << "Unexpected: " << instances.size() << "\n";
            for(const auto& instance : instances)
            {
                if(instance == "")
                {
                    *listener << "- (empty string)\n";
                }
                else
                {
                    *listener << "- " << instance << "\n";
                }
            }
        }
    }

    return false;
}

void InstanceMatcher::DescribeTo(std::ostream* os) const { *os << expected_; }

void InstanceMatcher::DescribeNegationTo(std::ostream* os) const
{
    *os << "is not equal to " << expected_;
}

bool HipStatusMatcher::MatchAndExplain(hipError_t actual,
                                       ::testing::MatchResultListener* listener) const
{
    (void)listener;

    if(actual == expected_)
    {
        return true;
    }

    return false;
}

void HipStatusMatcher::DescribeTo(std::ostream* os) const { *os << hipGetErrorString(expected_); }

void HipStatusMatcher::DescribeNegationTo(std::ostream* os) const
{
    if(expected_ == hipSuccess)
    {
        *os << "any error";
    }
    else
    {
        *os << "isn't equal to " << hipGetErrorString(expected_);
    }
}

::testing::Matcher<hipError_t> HipSuccess()
{
    return ::testing::MakeMatcher(new HipStatusMatcher(hipSuccess));
}

::testing::Matcher<hipError_t> HipError(hipError_t error)
{
    return ::testing::MakeMatcher(new HipStatusMatcher(error));
}

} // namespace ck_tile::test
