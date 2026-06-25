// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rtc/tmp_dir.hpp>
#include <algorithm>
#include <random>
#include <thread>
#include <unistd.h>

namespace rtc {
std::string random_string(std::string::size_type length)
{
    static const std::string& chars = "0123456789"
                                      "abcdefghijklmnopqrstuvwxyz"
                                      "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    std::mt19937 rg{std::random_device{}()};
    std::uniform_int_distribution<std::string::size_type> pick(0, chars.length() - 1);

    std::string str(length, 0);
    std::generate(str.begin(), str.end(), [&] { return chars[pick(rg)]; });

    return str;
}

std::string unique_string(const std::string& prefix)
{
    auto pid = getpid();
    auto tid = std::this_thread::get_id();
    auto clk = std::chrono::steady_clock::now().time_since_epoch().count();
    std::stringstream ss;
    ss << std::hex << prefix << "-" << pid << "-" << tid << "-" << clk << "-" << random_string(16);
    return ss.str();
}

tmp_dir::tmp_dir(const std::string& prefix)
    : path(fs::temp_directory_path() /
           unique_string(prefix.empty() ? "ck-rtc" : "ck-rtc-" + prefix))
{
    fs::create_directories(this->path);
}

void tmp_dir::execute(const std::string& cmd) const
{
    std::string s = "cd " + path.string() + "; " + cmd;
    std::system(s.c_str());
}

tmp_dir::~tmp_dir() { fs::remove_all(this->path); }

} // namespace rtc
