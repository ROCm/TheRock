// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <atomic>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <functional>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <vector>

#ifdef __linux__
#include <unistd.h>
#endif

#ifndef MIGRAPHX_GUARD_TEST_TEST_HPP
#define MIGRAPHX_GUARD_TEST_TEST_HPP

namespace test {
// clang-format off
// NOLINTNEXTLINE
#define TEST_FOREACH_BINARY_OPERATORS(m) \
    m(==, equal) \
    m(!=, not_equal) \
    m(<=, less_than_equal) \
    m(>=, greater_than_equal) \
    m(<, less_than) \
    m(>, greater_than) \
    m(and, and_op) \
    m(or, or_op)
// clang-format on

// clang-format off
// NOLINTNEXTLINE
#define TEST_FOREACH_UNARY_OPERATORS(m) \
    m(not, not_op)
// clang-format on

// NOLINTNEXTLINE
#define TEST_EACH_BINARY_OPERATOR_OBJECT(op, name)     \
    struct name                                        \
    {                                                  \
        static std::string as_string() { return #op; } \
        template <class T, class U>                    \
        static decltype(auto) call(T&& x, U&& y)       \
        {                                              \
            return x op y;                             \
        }                                              \
    };

// NOLINTNEXTLINE
#define TEST_EACH_UNARY_OPERATOR_OBJECT(op, name)      \
    struct name                                        \
    {                                                  \
        static std::string as_string() { return #op; } \
        template <class T>                             \
        static decltype(auto) call(T&& x)              \
        {                                              \
            return op x;                               \
        }                                              \
    };

TEST_FOREACH_BINARY_OPERATORS(TEST_EACH_BINARY_OPERATOR_OBJECT)
TEST_FOREACH_UNARY_OPERATORS(TEST_EACH_UNARY_OPERATOR_OBJECT)

struct nop
{
    static std::string as_string() { return ""; }
    template <class T>
    static auto call(T&& x)
    {
        return static_cast<T&&>(x);
    }
};

struct function
{
    static std::string as_string() { return ""; }
    template <class T>
    static decltype(auto) call(T&& x)
    {
        return x();
    }
};

template <class Stream, class Iterator>
Stream& stream_range(Stream& s, Iterator start, Iterator last);

template <class Stream>
inline Stream& operator<<(Stream& s, std::nullptr_t)
{
    s << "nullptr";
    return s;
}

template <class Stream,
          class Range,
          class = typename std::enable_if<not std::is_convertible<Range, std::string>{}>::type>
inline auto operator<<(Stream& s, const Range& v) -> decltype(stream_range(s, v.begin(), v.end()))
{
    s << "{ ";
    stream_range(s, v.begin(), v.end());
    s << "}";
    return s;
}

template <class Stream, class Iterator>
inline Stream& stream_range(Stream& s, Iterator start, Iterator last)
{
    if(start != last)
    {
        s << *start;
        std::for_each(std::next(start), last, [&](auto&& x) { s << ", " << x; });
    }
    return s;
}

template <class T>
const T& get_value(const T& x)
{
    return x;
}

template <class T, class Operator = nop>
struct lhs_expression;

template <class T>
lhs_expression<T> make_lhs_expression(T&& lhs);

template <class T, class Operator>
lhs_expression<T, Operator> make_lhs_expression(T&& lhs, Operator);

// NOLINTNEXTLINE
#define TEST_EXPR_BINARY_OPERATOR(op, name)                       \
    template <class V>                                            \
    auto operator op(const V& rhs2) const                         \
    {                                                             \
        return make_expression(*this, rhs2, name{}); /* NOLINT */ \
    }

// NOLINTNEXTLINE
#define TEST_EXPR_UNARY_OPERATOR(op, name) \
    auto operator op() const { return make_lhs_expression(lhs, name{}); /* NOLINT */ }

template <class T, class U, class Operator>
struct expression
{
    T lhs;
    U rhs;

    friend std::ostream& operator<<(std::ostream& s, const expression& self)
    {
        s << self.lhs << " " << Operator::as_string() << " " << self.rhs;
        return s;
    }

    friend decltype(auto) get_value(const expression& e) { return e.value(); }

    decltype(auto) value() const { return Operator::call(get_value(lhs), get_value(rhs)); };

    TEST_FOREACH_UNARY_OPERATORS(TEST_EXPR_UNARY_OPERATOR)
    TEST_FOREACH_BINARY_OPERATORS(TEST_EXPR_BINARY_OPERATOR)
};

// TODO: Remove rvalue references
template <class T, class U, class Operator>
expression<T, U, Operator> make_expression(T&& rhs, U&& lhs, Operator)
{
    return {std::forward<T>(rhs), std::forward<U>(lhs)};
}

// TODO: Remove rvalue reference
template <class T>
lhs_expression<T> make_lhs_expression(T&& lhs)
{
    return lhs_expression<T>{std::forward<T>(lhs)};
}

template <class T, class Operator>
lhs_expression<T, Operator> make_lhs_expression(T&& lhs, Operator)
{
    return lhs_expression<T, Operator>{std::forward<T>(lhs)};
}

template <class T, class Operator>
struct lhs_expression
{
    T lhs;
    explicit lhs_expression(T e) : lhs(e) {}

    friend std::ostream& operator<<(std::ostream& s, const lhs_expression& self)
    {
        std::string op = Operator::as_string();
        if(not op.empty())
            s << Operator::as_string() << " ";
        s << self.lhs;
        return s;
    }

    friend decltype(auto) get_value(const lhs_expression& e) { return e.value(); }

    decltype(auto) value() const { return Operator::call(get_value(lhs)); }

    TEST_FOREACH_BINARY_OPERATORS(TEST_EXPR_BINARY_OPERATOR)
    TEST_FOREACH_UNARY_OPERATORS(TEST_EXPR_UNARY_OPERATOR)

// NOLINTNEXTLINE
#define TEST_LHS_REOPERATOR(op)                 \
    template <class U>                          \
    auto operator op(const U& rhs) const        \
    {                                           \
        return make_lhs_expression(lhs op rhs); \
    }
    TEST_LHS_REOPERATOR(+)
    TEST_LHS_REOPERATOR(-)
    TEST_LHS_REOPERATOR(*)
    TEST_LHS_REOPERATOR(/)
    TEST_LHS_REOPERATOR(%)
    TEST_LHS_REOPERATOR(&)
    TEST_LHS_REOPERATOR(|)
    TEST_LHS_REOPERATOR(^)
};

template <class F>
struct predicate
{
    std::string msg;
    F f;

    friend std::ostream& operator<<(std::ostream& s, const predicate& self)
    {
        s << self.msg;
        return s;
    }

    decltype(auto) operator()() const { return f(); }

    operator decltype(auto)() const { return f(); }
};

template <class F>
auto make_predicate(const std::string& msg, F f)
{
    return make_lhs_expression(predicate<F>{msg, f}, function{});
}

inline std::string as_string(bool x)
{
    if(x)
        return "true";
    return "false";
}

template <class T>
std::string as_string(const T& x)
{
    std::stringstream ss;
    ss << x;
    return ss.str();
}

template <class Iterator>
std::string as_string(Iterator start, Iterator last)
{
    std::stringstream ss;
    stream_range(ss, start, last);
    return ss.str();
}

template <class F>
auto make_function(const std::string& name, F f)
{
    return [=](auto&&... xs) {
        std::vector<std::string> args = {as_string(xs)...};
        return make_predicate(name + "(" + as_string(args.begin(), args.end()) + ")",
                              [=] { return f(xs...); });
    };
}

struct capture
{
    template <class T>
    auto operator->*(const T& x) const
    {
        return make_lhs_expression(x);
    }

    template <class T, class Operator>
    auto operator->*(const lhs_expression<T, Operator>& x) const
    {
        return x;
    }
};

enum class color
{
    reset      = 0,
    bold       = 1,
    underlined = 4,
    fg_red     = 31,
    fg_green   = 32,
    fg_yellow  = 33,
    fg_blue    = 34,
    fg_default = 39,
    bg_red     = 41,
    bg_green   = 42,
    bg_yellow  = 43,
    bg_blue    = 44,
    bg_default = 49
};
inline std::ostream& operator<<(std::ostream& os, const color& c)
{
#ifndef _WIN32
    static const bool use_color = isatty(STDOUT_FILENO) != 0;
    if(use_color)
        return os << "\033[" << static_cast<std::size_t>(c) << "m";
#else
    (void)c;
#endif
    return os;
}

inline std::atomic<int>& failures()
{
    // NOLINTNEXTLINE
    static std::atomic<int> f = 0;
    return f;
}

template <class T, class F>
void failed(T x, const char* msg, const char* func, const char* file, int line, F f)
{
    if(not bool(x.value()))
    {
        failures()++;
        std::cout << func << std::endl;
        std::cout << file << ":" << line << ":" << std::endl;
        std::cout << color::bold << color::fg_red << "    FAILED: " << color::reset << msg << " "
                  << "[ " << x << " ]" << std::endl;
        f();
    }
}

template <class F>
bool throws(F f)
{
    try
    {
        f();
        return false;
    }
    catch(...)
    {
        return true;
    }
}

template <class Exception, class F>
bool throws(F f, const std::string& msg = "")
{
    try
    {
        f();
        return false;
    }
    catch(const Exception& ex)
    {
        return std::string(ex.what()).find(msg) != std::string::npos;
    }
}

template <class T, class U>
auto within_abs(T px, U py, double ptol = 1e-6f)
{
    return make_function("near", [](auto x, auto y, auto tol) { return std::abs(x - y) < tol; })(
        px, py, ptol);
}

// This implements the basic globbing algorithm where `*` matches any number
// of characters(including none) and `?` matches any single character. It
// doesnt support character classes.
//
// This is a simple recursive implementation that scans the string where the
// string and pattern matches. When a `*` is found in the pattern, the
// `glob_match` function is called recursively to compare the rest of the
// pattern to the rest of the string. If the recursive call returns true,
// then we have a match. However, if it returns false, then we advance one
// character and call the recusrsive call again. This is referred to as a
// star-loop, which will consume zero or more characters.
//
// This simple recursive implementation works well for short string and
// patterns with few stars. First, it is unlikely to use many stars to glob
// test names. Secondly, using many stars is still signficantly faster than
// using the equivalent std::regex, which has a much slower time complexity.
template <class Iterator1, class Iterator2>
bool glob_match(Iterator1 start, Iterator1 last, Iterator2 pattern_start, Iterator2 pattern_last)
{
    std::tie(start, pattern_start) =
        std::mismatch(start, last, pattern_start, pattern_last, [](auto c, auto m) {
            if(m == '?')
                return true;
            // We need a loop for star, so bail and handle the loop below
            if(m == '*')
                return false;
            return c == m;
        });
    // If there is no more pattern then return true if there is no more string to match
    if(pattern_start == pattern_last)
        return start == last;
    // If the pattern is not a star then its a mismatch
    if(*pattern_start != '*')
        return false;
    // Multiple stars are the same as a single star so skip over multiple stars
    pattern_start = std::find_if(pattern_start, pattern_last, [](auto c) { return c != '*'; });
    // If the star is at the end then return true
    if(pattern_start == pattern_last)
        return true;
    // star-loop: match the rest of the pattern and text
    while(not glob_match(start, last, pattern_start, pattern_last) and start != last)
        start++;
    // If the string is empty then it means a match was never found
    return start != last;
}

using string_map = std::unordered_map<std::string, std::vector<std::string>>;

template <class Keyword>
string_map generic_parse(std::vector<std::string> as, Keyword keyword)
{
    string_map result;

    std::string flag;
    for(auto&& x : as)
    {
        auto f = keyword(x);
        if(f.empty())
        {
            result[flag].push_back(x);
        }
        else
        {
            flag = f.front();
            result[flag]; // Ensure the flag exists
            flag = f.back();
        }
    }
    return result;
}

using test_case = std::function<void()>;

inline auto& get_test_cases()
{
    // NOLINTNEXTLINE
    static std::vector<std::pair<std::string, test_case>> cases;
    return cases;
}

inline void add_test_case(std::string name, test_case f)
{
    get_test_cases().emplace_back(std::move(name), std::move(f));
}

struct auto_register_test_case
{
    template <class F>
    auto_register_test_case(const char* name, F f) noexcept
    {
        add_test_case(name, f);
    }
};

struct failure_error
{
};

[[noreturn]] inline void fail() { throw failure_error{}; }

struct driver
{
    driver()
    {
        add_flag({"--help", "-h"}, "Show help");
        add_flag({"--list", "-l"}, "List all test cases");
        add_flag({"--continue", "-c"}, "Continue after failure");
        add_flag({"--quiet", "-q"}, "Don't print out extra output");
    }
    struct argument
    {
        std::vector<std::string> flags = {};
        std::string help               = "";
        int nargs                      = 1;
    };

    void add_arg(const std::vector<std::string>& flags, const std::string& help = "")
    {
        arguments.push_back(argument{flags, help, 1});
    }

    void add_flag(const std::vector<std::string>& flags, const std::string& help = "")
    {
        arguments.push_back(argument{flags, help, 0});
    }

    static void wrap(std::ostream& os,
                     const std::string& text,
                     const std::string& prefix = "",
                     unsigned int line_length  = 80)
    {
        std::istringstream iss(text);
        std::string line = prefix;
        do
        {
            std::string word;
            iss >> word;
            if(line.length() + word.length() > line_length)
            {
                os << line << std::endl;
                line = prefix;
            }
            line += word + " ";
        } while(iss);
        if(not line.empty())
            os << line << std::endl;
    }

    void show_help(const std::string& exe) const
    {
        const std::string prefix = "    ";
        std::cout << std::endl;
        std::cout << color::fg_yellow << "USAGE:" << color::reset << std::endl;
        std::cout << "    ";
        std::cout << exe << " <test-case>... <options>" << std::endl;
        std::cout << std::endl;

        std::cout << color::fg_yellow << "ARGS:" << color::reset << std::endl;
        std::cout << "    ";
        std::cout << color::fg_green << "<test-case>..." << color::reset;
        std::cout << std::endl;

        wrap(std::cout,
             "Test cases to run. A test case can be either the exact test case name or a glob. A "
             "glob expression uses a '*' to select zero or more characters or a '?' to select any "
             "single character.",
             prefix + prefix);

        std::cout << std::endl;
        std::cout << color::fg_yellow << "OPTIONS:" << color::reset << std::endl;
        for(auto&& arg : arguments)
        {
            std::cout << color::fg_green;
            std::string arg_prefix = prefix;
            for(const std::string& a : arg.flags)
            {
                std::cout << arg_prefix;
                std::cout << a;
                arg_prefix = ", ";
            }
            std::cout << color::reset << std::endl;
            wrap(std::cout, arg.help, prefix + prefix);
        }
    }

    std::ostream& out() const
    {
        struct null_buffer : std::streambuf
        {
            virtual int overflow(int c) override { return c; }
        };
        static null_buffer buffer;
        static std::ostream null_stream(&buffer);
        if(quiet)
            return null_stream;
        return std::cout;
    }

    string_map parse(int argc, const char* argv[]) const
    {
        std::vector<std::string> args(argv + 1, argv + argc);
        string_map keys;
        for(auto&& arg : arguments)
        {
            for(auto&& flag : arg.flags)
            {
                keys[flag] = {arg.flags.front()};
                if(arg.nargs == 0)
                    keys[flag].push_back("");
            }
        }
        auto result = generic_parse(args, [&](auto&& s) -> std::vector<std::string> {
            if(keys.count(s) > 0)
                return keys[s];
            else
                return {};
        });
        result["__exe__"].push_back(argv[0]);
        return result;
    }

    static std::string create_command(const string_map& args)
    {
        std::stringstream ss;
        ss << args.at("__exe__").front();
        if(args.count("") > 0)
        {
            for(auto&& arg : args.at(""))
                ss << " \"" << arg << "\"";
        }
        for(auto&& p : args)
        {
            if(p.first == "__exe__")
                continue;
            if(p.first.empty())
                continue;
            ss << " " << p.first;
            for(auto&& arg : p.second)
                ss << " \"" << arg << "\"";
        }
        return ss.str();
    }

    static std::string fork(const std::string& name, string_map args)
    {
        std::string msg;
        args[""] = {name};
        args.erase("--continue");
        args["--quiet"];
        auto cmd = create_command(args);
        auto r   = std::system(cmd.c_str()); // NOLINT
        if(r != 0)
            msg = "Exited with " + std::to_string(r);
        return msg;
    }

    static std::vector<std::pair<std::string, test_case>> glob_tests(const std::string& pattern)
    {
        std::vector<std::pair<std::string, test_case>> result;
        std::copy_if(get_test_cases().begin(),
                     get_test_cases().end(),
                     std::back_inserter(result),
                     [&](auto&& p) {
                         return glob_match(
                             p.first.begin(), p.first.end(), pattern.begin(), pattern.end());
                     });
        return result;
    }

    void run_test_case(const std::string& name, const test_case& f, const string_map& args)
    {
        ran++;
        out() << color::fg_green << "[   RUN    ] " << color::reset << color::bold << name
              << color::reset << std::endl;
        std::string msg;
        auto start = std::chrono::steady_clock::now();
        if(args.count("--continue") > 0)
        {
            msg = fork(name, args);
        }
        else
        {
            try
            {
                failures() = 0;
                f();
            }
            // cppcheck-suppress migraphx-EmptyCatchStatement
            catch(const failure_error&)
            {
            }
        }
        auto finish = std::chrono::steady_clock::now();
        auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(finish - start)
                .count();
        if(msg.empty() and failures() != 0)
        {
            if(failures() == 1)
                msg = "Test failure";
            else
                msg = std::to_string(failures()) + " test failures";
        }
        if(msg.empty())
        {
            out() << color::fg_green << "[ COMPLETE ] " << color::reset;
        }
        else
        {
            failed.push_back(name);
            out() << color::fg_red << "[  FAILED  ] " << color::reset;
        }
        out() << color::bold << name << color::reset;
        out() << color::fg_blue << " (" << elapsed_ms << "ms)" << color::reset;
        if(not msg.empty())
            out() << ": " << color::fg_yellow << msg << color::reset;
        out() << std::endl;
    }

    void run(int argc, const char* argv[])
    {
        auto args = parse(argc, argv);
        if(args.count("--help") > 0)
        {
            show_help(args.at("__exe__").front());
            return;
        }
        if(args.count("--list") > 0)
        {
            for(auto&& tc : get_test_cases())
                out() << tc.first << std::endl;
            return;
        }

        if(args.count("--quiet") > 0)
            quiet = true;

        auto cases = args[""];
        if(cases.empty())
        {
            for(auto&& tc : get_test_cases())
                run_test_case(tc.first, tc.second, args);
        }
        else
        {
            std::unordered_map<std::string, test_case> m(get_test_cases().begin(),
                                                         get_test_cases().end());

            for(auto&& iname : cases)
            {
                std::vector<std::pair<std::string, test_case>> found_cases;
                for(auto&& pattern : get_case_names(iname))
                {
                    auto f = m.find(pattern);
                    if(f == m.end())
                    {
                        found_cases = glob_tests(pattern);
                    }
                    else
                    {
                        found_cases.push_back(*f);
                    }
                }
                if(found_cases.empty())
                {
                    out() << color::fg_red << "[  ERROR   ] Test case '" << iname << "' not found."
                          << color::reset << std::endl;
                    failed.push_back(iname);
                }
                for(auto&& p : found_cases)
                    run_test_case(p.first, p.second, args);
            }
        }
        out() << color::fg_green << "[==========] " << color::fg_yellow << ran << " tests ran"
              << color::reset << std::endl;
        if(not failed.empty())
        {
            out() << color::fg_red << "[  FAILED  ] " << color::fg_yellow << failed.size()
                  << " tests failed" << color::reset << std::endl;
            for(auto&& name : failed)
                out() << color::fg_red << "[  FAILED  ] " << color::fg_yellow << name
                      << color::reset << std::endl;
            std::exit(1);
        }
    }

    std::function<std::vector<std::string>(const std::string&)> get_case_names =
        [](const std::string& name) -> std::vector<std::string> { return {name}; };
    std::vector<argument> arguments = {};
    std::vector<std::string> failed = {};
    std::size_t ran                 = 0;
    bool quiet                      = false;
};

inline void run(int argc, const char* argv[])
{
    driver d{};
    d.run(argc, argv);
}

} // namespace test

// NOLINTNEXTLINE
#define TEST_CAPTURE(...) test::capture{}->*__VA_ARGS__

// NOLINTNEXTLINE
#define CHECK(...) \
    test::failed(  \
        TEST_CAPTURE(__VA_ARGS__), #__VA_ARGS__, __PRETTY_FUNCTION__, __FILE__, __LINE__, [] {})

// NOLINTNEXTLINE
#define EXPECT(...)                         \
    test::failed(TEST_CAPTURE(__VA_ARGS__), \
                 #__VA_ARGS__,              \
                 __PRETTY_FUNCTION__,       \
                 __FILE__,                  \
                 __LINE__,                  \
                 &test::fail)
// NOLINTNEXTLINE
#define STATUS(...) EXPECT((__VA_ARGS__) == 0)

// NOLINTNEXTLINE
#define TEST_CAT(x, ...) TEST_PRIMITIVE_CAT(x, __VA_ARGS__)
// NOLINTNEXTLINE
#define TEST_PRIMITIVE_CAT(x, ...) x##__VA_ARGS__

// NOLINTNEXTLINE
#define TEST_CASE_REGISTER(...)                                                    \
    static test::auto_register_test_case TEST_CAT(register_test_case_, __LINE__) = \
        test::auto_register_test_case(#__VA_ARGS__, &__VA_ARGS__);

// NOLINTNEXTLINE
#define TEST_CASE(...)              \
    void __VA_ARGS__();             \
    TEST_CASE_REGISTER(__VA_ARGS__) \
    void __VA_ARGS__()

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#endif

#endif
