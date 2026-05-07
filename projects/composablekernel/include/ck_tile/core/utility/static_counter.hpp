// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"

namespace ck_tile {

template <typename Context, index_t Start = 0, index_t Step = 1>
struct static_counter
{
    public:
    template <typename Unique>
    static constexpr index_t next()
    {
        return next<Unique>(0) * Step + Start;
    }

    template <unsigned long long>
    static constexpr index_t next()
    {
        struct Unique
        {
        };
        return next<Unique>(0) * Step + Start;
    }

    template <typename Unique>
    static constexpr index_t current()
    {
        return current<Unique>(0) * Step + Start;
    }

    template <unsigned long long>
    static constexpr index_t current()
    {
        struct Unique
        {
        };
        return current<Unique>(0) * Step + Start;
    }

    private:
    template <index_t I>
    struct slot
    {
        _Pragma("GCC diagnostic push");
        _Pragma("GCC diagnostic ignored \"-Wundefined-internal\"");
        friend constexpr bool slot_allocated(slot<I>);
        _Pragma("GCC diagnostic pop");
    };

    template <index_t I>
    struct allocate_slot
    {
        friend constexpr bool slot_allocated(slot<I>) { return true; }
        enum
        {
            value = I
        };
    };

    // If slot_allocated(slot<I>) has NOT been defined, then SFINAE will keep this function out of
    // the overload set...
    template <typename Unique, index_t I = 0, bool = slot_allocated(slot<I>())>
    static constexpr index_t next(index_t)
    {
        return next<Unique, I + 1>(0);
    }

    // ...And this function will be used, instead, which will define slot_allocated(slot<I>) via
    // allocate_slot<I>.
    template <typename Unique, index_t I = 0>
    static constexpr index_t next(double)
    {
        return allocate_slot<I>::value;
    }

    // If slot_allocated(slot<I>) has NOT been defined, then SFINAE will keep this function out of
    // the overload set...
    template <typename Unique, index_t I = Start, bool = slot_allocated(slot<I>())>
    static constexpr index_t current(index_t)
    {
        return current<Unique, I + 1>(0);
    }

    // ...And this function will be used, instead, which will return the current counter, or assert
    // in case next() hasn't been called yet.
    template <typename Unique, index_t I = Start>
    static constexpr index_t current(double)
    {
        static_assert(I != 0, "You must invoke next() first");

        return I - 1;
    }
};

namespace impl {
template <int I>
struct static_counter_uniq_;
}

// clang-format off
#define MAKE_SC()                                                                       \
    _Pragma("clang diagnostic push")                                                    \
    _Pragma("clang diagnostic ignored \"-Wpre-c2y-compat\"")                            \
    _Pragma("clang diagnostic ignored \"-Wc2y-extensions\"")                            \
    ck_tile::static_counter<ck_tile::impl::static_counter_uniq_<__COUNTER__>>{}         \
    _Pragma("clang diagnostic pop")
#define MAKE_SC_WITH(start_, step_)                                                     \
    _Pragma("clang diagnostic push")                                                    \
    _Pragma("clang diagnostic ignored \"-Wpre-c2y-compat\"")                            \
    _Pragma("clang diagnostic ignored \"-Wc2y-extensions\"") ck_tile::                  \
    static_counter<ck_tile::impl::static_counter_uniq_<__COUNTER__>, start_, step_>{}   \
    _Pragma("clang diagnostic pop")
#define NEXT_SC(c_)                                                                     \
    _Pragma("clang diagnostic push")                                                    \
    _Pragma("clang diagnostic ignored \"-Wpre-c2y-compat\"")                            \
    _Pragma("clang diagnostic ignored \"-Wc2y-extensions\"") c_.next<__COUNTER__>()     \
    _Pragma("clang diagnostic pop")
#define NEXT_SCI(c_, static_i_)                                                         \
    _Pragma("clang diagnostic push")                                                    \
    _Pragma("clang diagnostic ignored \"-Wpre-c2y-compat\"")                            \
    _Pragma("clang diagnostic ignored \"-Wc2y-extensions\"")                            \
    c_.next<__COUNTER__ + static_i_>() _Pragma("clang diagnostic pop")
// clang-format on

// Usage:
// constexpr auto c = MAKE_SC()
// NEXT_SC(c)    // -> constexpr 0
// NEXT_SC(c)    // -> constexpr 1
// NEXT_SC(c)    // -> constexpr 2
} // namespace ck_tile
