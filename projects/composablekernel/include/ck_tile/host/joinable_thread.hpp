// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef __linux__
#include <sched.h>
#endif
#include <thread>
#include <utility>

namespace ck_tile {

struct joinable_thread : std::thread
{
    template <typename... Xs>
    joinable_thread(Xs&&... xs) : std::thread(std::forward<Xs>(xs)...)
    {
    }

    joinable_thread(joinable_thread&&)            = default;
    joinable_thread& operator=(joinable_thread&&) = default;

    ~joinable_thread()
    {
        if(this->joinable())
            this->join();
    }
};

inline unsigned int get_available_cpu_cores()
{
#if defined(__linux__)
    cpu_set_t cpu_set;
    if(sched_getaffinity(0, sizeof(cpu_set_t), &cpu_set) == 0)
    {
        unsigned int cpu_count = CPU_COUNT(&cpu_set);
        if(cpu_count > 0)
            return cpu_count;
    }
#endif
    // Fallback if sched_getaffinity unavailable or fails
    return std::thread::hardware_concurrency();
}

class cpu_core_guard
{
#if defined(__linux__)
    cpu_set_t original_cpu_set_;

    public:
    cpu_core_guard(unsigned int num_cores) : original_cpu_set_()
    {
        // save original cpu set
        sched_getaffinity(0, sizeof(cpu_set_t), &original_cpu_set_);

        // set new cpu set
        cpu_set_t new_cpu_set;
        CPU_ZERO(&new_cpu_set);
        for(unsigned int i = 0; i < num_cores; ++i)
        {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
            CPU_SET(i, &new_cpu_set); // NOLINT(old-style-cast)
#pragma clang diagnostic pop
        }
        sched_setaffinity(0, sizeof(cpu_set_t), &new_cpu_set);
    }
    ~cpu_core_guard()
    {
        // restore original cpu set
        sched_setaffinity(0, sizeof(cpu_set_t), &original_cpu_set_);
    }
#endif
};
} // namespace ck_tile
