// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename F, typename... Fs>
struct composes : private composes<F>
{
    template <typename FirstArg, typename... RestArgs>
    CK_TILE_HOST_DEVICE constexpr explicit composes(FirstArg&& firstArg, RestArgs&&... restArgs)
        : composes<F>(std::forward<FirstArg>(firstArg)), inner_(std::forward<RestArgs>(restArgs)...)
    {
    }

    template <typename Arg>
    CK_TILE_HOST_DEVICE constexpr auto operator()(Arg&& arg) const
    {
        return static_cast<const composes<F>&>(*this)(inner_(std::forward<Arg>(arg)));
    }

    private:
    composes<Fs...> inner_;
};

template <typename F>
struct composes<F>
{
    static_assert(!std::is_reference_v<F>);

    template <typename Arg, typename = std::enable_if_t<std::is_constructible_v<F, Arg>>>
    CK_TILE_HOST_DEVICE constexpr explicit composes(Arg&& arg) : f_(std::forward<Arg>(arg))
    {
    }

    template <typename Arg,
              typename = std::enable_if_t<std::is_invocable_v<std::add_const_t<F>&, Arg>>>
    CK_TILE_HOST_DEVICE constexpr auto operator()(Arg&& arg) const
    {
        return f_(std::forward<Arg>(arg));
    }

    private:
    F f_;
};

template <typename... Ts>
CK_TILE_HOST_DEVICE_EXTERN composes(Ts&&...) -> composes<remove_cvref_t<Ts>...>;

template <typename SaturateType>
struct saturates
{
    // NOTE: this function does not return SaturateType value
    // it is user's responsiblity to do further cast or not
    template <typename AccType>
    CK_TILE_HOST_DEVICE constexpr auto
    operator()(const AccType& a_) const -> std::enable_if_t<std::is_arithmetic_v<AccType>, AccType>
    {
        return clamp(a_,
                     type_convert<AccType>(numeric<SaturateType>::lowest()),
                     type_convert<AccType>(numeric<SaturateType>::max()));
    }
};

} // namespace ck_tile
