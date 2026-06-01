// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace ck_tile::reflect {

// Helper class for formatting hierarchical tree structures with proper indentation
// and tree-drawing characters (├─, └─, │, etc.)
//
// Example Usage:
//
//   TreeFormatter f;
//   f.writeLine(0, "Root");
//   f.writeLine(1, "Branch 1");
//   f.writeLine(2, "Item 1a");
//   f.writeLast(2, "Item 1b");
//   f.writeLast(1, "Branch 2");
//   f.writeLast(2, "Item 2a");
//   std::cout << f.getString() << "\n";
//
// Generated Output:
//
//   Root
//   ├─ Branch 1
//   │  ├─ Item 1a
//   │  └─ Item 1b
//   └─ Branch 2
//      └─ Item 2a
class TreeFormatter
{
    public:
    TreeFormatter() = default;

    // Write a line at the specified indentation level (branch continues after this)
    template <typename... Args>
    void writeLine(int indent_level, Args&&... args)
    {
        writeLineImpl(indent_level, false, std::forward<Args>(args)...);
    }

    // Write the last line at the specified indentation level (branch ends)
    template <typename... Args>
    void writeLast(int indent_level, Args&&... args)
    {
        writeLineImpl(indent_level, true, std::forward<Args>(args)...);
    }

    // Get the formatted string (removes trailing newline if present)
    std::string getString() const
    {
        std::string result = oss_.str();
        if(!result.empty() && result.back() == '\n')
        {
            result.pop_back();
        }
        return result;
    }

    private:
    std::ostringstream oss_;
    std::vector<bool> is_last_at_level_; // Tracks which levels have ended

    // Implementation of line writing with tree symbols
    template <typename... Args>
    void writeLineImpl(int indent_level, bool is_last, Args&&... args)
    {
        // Ensure we have enough tracking space
        if(static_cast<size_t>(indent_level) >= is_last_at_level_.size())
        {
            is_last_at_level_.resize(indent_level + 1, false);
            // Level 0 (root) should always be treated as "last" since it has no tree symbols
            if(is_last_at_level_.size() > 0)
            {
                is_last_at_level_[0] = true;
            }
        }

        // Draw the tree structure
        // Start from level 1 (skip level 0 which is the root with no symbols)
        for(int i = 1; i < indent_level; ++i)
        {
            // For all parent levels, draw vertical line or space based on whether they ended
            oss_ << (is_last_at_level_[i] ? "   " : "│  ");
        }

        // Draw the branch symbol for the current level
        if(indent_level > 0)
        {
            oss_ << (is_last ? "└─ " : "├─ ");
        }

        // Write the content using fold expression with direct stream insertion
        ((oss_ << std::forward<Args>(args)), ...);

        oss_ << '\n';

        // Update tracking for this level AFTER writing the line
        // This ensures future lines at deeper levels know if this level ended
        is_last_at_level_[indent_level] = is_last;
    }
};

} // namespace ck_tile::reflect
