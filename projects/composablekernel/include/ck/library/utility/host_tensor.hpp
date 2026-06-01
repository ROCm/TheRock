// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
#include <numeric>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include "ck/utility/data_type.hpp"
#include "ck/utility/span.hpp"
#include "ck/utility/type_convert.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/ranges.hpp"
#include "ck/library/utility/thread.hpp"

#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

namespace ck {

template <typename Range>
std::ostream& LogRange(std::ostream& os, Range&& range, std::string delim)
{
    bool first = true;
    for(auto&& v : range)
    {
        if(first)
            first = false;
        else
            os << delim;
        os << v;
    }
    return os;
}

template <typename T, typename Range>
std::ostream& LogRangeAsType(std::ostream& os, Range&& range, std::string delim)
{
    bool first = true;
    for(auto&& v : range)
    {
        if(first)
            first = false;
        else
            os << delim;

        using RangeType = ck::remove_cvref_t<decltype(v)>;
        if constexpr(std::is_same_v<RangeType, ck::f8_t> || std::is_same_v<RangeType, ck::bf8_t> ||
                     std::is_same_v<RangeType, ck::bhalf_t>)
        {
            os << ck::type_convert<float>(v);
        }
        else if constexpr(std::is_same_v<RangeType, ck::pk_i4_t> ||
                          std::is_same_v<RangeType, ck::f4x2_pk_t>)
        {
            const auto packed_floats = ck::type_convert<ck::float2_t>(v);
            const ck::vector_type<float, 2> vector_of_floats{packed_floats};
            os << vector_of_floats.template AsType<float>()[ck::Number<0>{}] << delim
               << vector_of_floats.template AsType<float>()[ck::Number<1>{}];
        }
        else
        {
            os << static_cast<T>(v);
        }
    }
    return os;
}

template <typename F, typename T, std::size_t... Is>
auto call_f_unpack_args_impl(F f, T args, std::index_sequence<Is...>)
{
    return f(std::get<Is>(args)...);
}

template <typename F, typename T>
auto call_f_unpack_args(F f, T args)
{
    constexpr std::size_t N = std::tuple_size<T>{};

    return call_f_unpack_args_impl(f, args, std::make_index_sequence<N>{});
}

template <typename F, typename T, std::size_t... Is>
auto construct_f_unpack_args_impl(T args, std::index_sequence<Is...>)
{
    return F(std::get<Is>(args)...);
}

template <typename F, typename T>
auto construct_f_unpack_args(F, T args)
{
    constexpr std::size_t N = std::tuple_size<T>{};

    return construct_f_unpack_args_impl<F>(args, std::make_index_sequence<N>{});
}

/**
 * @brief A descriptor class for host tensors that manages tensor dimensions, strides, and layout.
 *
 * The HostTensorDescriptor provides a comprehensive interface for describing multi-dimensional
 * tensors with configurable layouts and automatic stride calculation capabilities.
 *
 * @section stride_handling Stride Handling
 *
 * The descriptor supports multiple stride specification modes:
 *
 * 1. **Explicit Strides**: When strides are provided explicitly, they are validated against
 *    the specified layout to ensure memory access patterns are correct.
 *
 * 2. **Auto-calculated Strides**: When strides are empty or all-zero, they are automatically
 *    calculated based on the tensor layout:
 *    - For RowMajor layout: rightmost dimension has stride 1, others calculated as cumulative
 * products
 *    - For ColumnMajor layout: similar to RowMajor but with swapped stride positions for last two
 * dimensions
 *
 * 3. **Partial Stride Specification**: For GEMM layouts, unknown strides (represented as 0 or
 * negative values) in the last two dimensions can be auto-calculated while preserving higher
 * dimension strides.
 *
 * 4. **Bypass**: When using `BypassLayoutVerification` layout, no stride calculation or validation
 * is performed. That allows to pass in any arbitrary strides including 0.
 *
 * For more details see `CalculateStrides` method.
 *
 * @section layout_support Layout Support
 *
 * - **GEMM Layouts**: Supports RowMajor and ColumnMajor layouts with full validation
 * - **Convolution Layouts**: Recognized but validation is not yet implemented
 * - **Abstract Layouts**: BaseTensorLayout will attempt automatic layout detection for 2D tensors
 *
 * @section limitations Limitations
 *
 * 1. **Layout Detection**: Automatic layout detection only works reliably for 2D tensors.
 *    This is done mostly for legacy GEMM cases to avoid modifying many existing GEMM tests to pass
 *    RowMajor/ColumnMajor explicitly. Higher-dimensional tensors with BaseTensorLayout will throw
 *    validation errors. For more details see `HandleDefaultLayout` method.
 *
 * 2. **Stride Validation**: Only GEMM layouts (RowMajor/ColumnMajor) have full stride validation.
 *    Convolution layouts are accepted but not validated. For more details see `ValidateStrides`.
 *
 * 3. **GEMM Assumptions**: For tensors with more than 2 dimensions, GEMM layout validation
 *    assumes the last two dimensions represent the height-width pattern (e.g., BHW or BWH for
 * batched GEMM).
 *
 * 4. **Negative Stride Handling**: Negative stride values are interpreted as "unknown" and
 *    converted to auto-calculated values only for supported layouts.
 *
 * @section thread_safety Thread Safety
 * This class is not thread-safe. External synchronization is required for concurrent access.
 *
 * @section examples Usage Examples
 *
 * ```cpp
 * // Auto-calculate strides for RowMajor layout
 * HostTensorDescriptor desc1({4, 3}, ck::tensor_layout::gemm::RowMajor{});
 *
 * // Explicit strides with validation
 * HostTensorDescriptor desc2({4, 3}, {3, 1}, ck::tensor_layout::gemm::RowMajor{});
 *
 * // Partial stride specification (auto-calculate unknown dimension)
 * HostTensorDescriptor desc3({4, 3}, {0, 1}, ck::tensor_layout::gemm::RowMajor{});
 * ```
 */
struct HostTensorDescriptor
{
    using BaseTensorLayout = ck::tensor_layout::BaseTensorLayout;
    using DefaultLayout    = BaseTensorLayout;

    // Runtime tag describing which layout is picked when layout is not specified explicitly at
    // construction time.
    enum class ChosenLayout
    {
        Original,
        RowMajor,
        ColumnMajor
    };

    // Master constructor
    template <typename Layout>
    HostTensorDescriptor(std::vector<std::size_t> lens,
                         std::vector<std::size_t> strides,
                         const Layout& layout = DefaultLayout())
        : mLens(std::move(lens)), mStrides(std::move(strides))
    {
        // To support legacy use cases, when layout is not passed in
        const auto new_layout = HandleDefaultLayout(layout);
        if(dbg)
        {
            std::cout << "Original Lens: [";
            LogRange(std::cout, mLens, ", ") << "] and Strides: [";
            LogRange(std::cout, mStrides, ", ") << "]" << std::endl;
            std::cout << "Layout: " << layout << " --> " << new_layout << std::endl;
        }

        // Handling the strides and validation based on the chosen layout
        DispatchChosenLayout(new_layout, layout, [&](auto selected_layout) {
            this->CalculateStrides(selected_layout);
            this->ValidateStrides(selected_layout);
        });
    }

    HostTensorDescriptor() : HostTensorDescriptor({}, {}, DefaultLayout()){};

    // Helper that invokes a callable with a concrete layout object whose type
    // matches the chosen tag (so template code depending on the layout type
    // can still leverage if constexpr branches).
    template <typename F, typename OrigLayout>
    void DispatchChosenLayout(ChosenLayout tag, const OrigLayout& orig, F&& f) const
    {
        switch(tag)
        {
        case ChosenLayout::RowMajor: f(ck::tensor_layout::gemm::RowMajor{}); break;
        case ChosenLayout::ColumnMajor: f(ck::tensor_layout::gemm::ColumnMajor{}); break;
        case ChosenLayout::Original:
        default: f(orig); break;
        }
    }

    template <typename Layout>
    ChosenLayout HandleDefaultLayout(const Layout&)
    {
        if constexpr(!std::is_same_v<Layout, DefaultLayout>)
        {
            return ChosenLayout::Original;
        }
        else
        {
            if(mStrides.empty())
            {
                // No strides provided -> assume RowMajor
                return ChosenLayout::RowMajor;
            }

            const auto rank = mLens.size();

            if(rank > 2)
            {
                // Keep as-is - validation will warn/throw later
                return ChosenLayout::Original;
            }

            if(rank == 0)
            {
                // Keep as-is - validation will warn/throw later
                return ChosenLayout::Original;
            }

            if(rank == 1)
            {
                // Treat 1D tensor as RowMajor
                return ChosenLayout::RowMajor;
            }

            // rank == 2
            if(mStrides.size() == 2)
            {
                // RowMajor pattern (?, 1)
                if(mStrides[1] == 1)
                {
                    return ChosenLayout::RowMajor;
                }

                // ColumnMajor pattern (1, ?)
                if(mStrides[0] == 1)
                {
                    return ChosenLayout::ColumnMajor;
                }
            }

            // Fallback: leave as-is
            return ChosenLayout::Original;
        }
    }

    template <typename Layout>
    void CalculateStrides(const Layout& layout)
    {
        if constexpr(std::is_same_v<Layout, ck::tensor_layout::BypassLayoutVerification>)
            return;
        // This is a workaround if the original stride value is -1 (which means "unknown") has been
        // passed in and casted to size_t (unsigned).
        auto strides_int = AsInt(mStrides);

        // case of empty strides or all-zero: auto-calculate based on layout and tensor dimensions
        if(mStrides.empty() || std::all_of(strides_int.begin(), strides_int.end(), [](int stride) {
               return stride <= 0;
           }))
        {

            if constexpr(!(std::is_same_v<ck::tensor_layout::gemm::RowMajor, Layout> ||
                           std::is_same_v<ck::tensor_layout::gemm::ColumnMajor, Layout>))
            {
                std::cerr << "Only RowMajor and ColumnMajor layouts are supported for empty "
                             "strides, got "
                          << layout << ". Will calculate strides as RowMajor." << std::endl;
            }

            mStrides.clear();
            mStrides.resize(mLens.size(), 0);
            if(mStrides.empty())
                return;

            mStrides.back() = 1;
            std::partial_sum(mLens.rbegin(),
                             mLens.rend() - 1,
                             mStrides.rbegin() + 1,
                             std::multiplies<std::size_t>());

            if constexpr(std::is_same_v<ck::tensor_layout::gemm::ColumnMajor, Layout>)
            {
                // swap the last two strides
                if(mStrides.size() >= 2)
                    std::swap(mStrides[mStrides.size() - 1], mStrides[mStrides.size() - 2]);
            }
        }
        // The other case is if one of the strides is unknown
        // Currently, only GEMM RowMajor and ColumnMajor layouts are supported and only in the lower
        // two dimensions, e.g. {..., 0, N} or {..., M, 0}. The higher dimensions are left
        // untouched.
        else if constexpr(std::is_same_v<ck::tensor_layout::gemm::RowMajor, Layout> ||
                          std::is_same_v<ck::tensor_layout::gemm::ColumnMajor, Layout>)
        {
            auto rank = mStrides.size();
            if(mLens.size() >= 2 && rank >= 2)
            {
                const auto inner_idx =
                    std::is_same_v<ck::tensor_layout::gemm::RowMajor, Layout> ? rank - 1 : rank - 2;
                const auto outer_idx = inner_idx == rank - 1 ? rank - 2 : rank - 1;
                if(mStrides[inner_idx] <= 0)
                {
                    mStrides[inner_idx] = 1;
                }
                if(mStrides[outer_idx] <= 0)
                {
                    mStrides[outer_idx] = mLens[inner_idx] * mStrides[inner_idx];
                }
            }
        }
    }

    template <typename Layout>
    void ValidateStrides(const Layout& layout) const
    {
        if constexpr(std::is_same_v<ck::tensor_layout::BypassLayoutVerification, Layout>)
        {
            return;
        }

        if(mLens.empty())
        {
            throw std::runtime_error(
                "HostTensorDescriptor::ValidateStrides: empty tensor dimensions is not allowed.");
        }

        const int rank = mLens.size();
        if(rank == 1) // skip any 1D tensors
        {
            return;
        }

        if constexpr(std::is_same_v<ck::tensor_layout::BaseTensorLayout, Layout>)
        {
            // Any legacy code that doesn't pass layout to HostTensorDescriptor ctor will
            // hit this case (unless it is a special case - see `HandleDefaultLayout`).
            throw std::runtime_error("HostTensorDescriptor::ValidateStrides: Abstract tensor "
                                     "layout BaseTensorLayout can't be verified. Pls "
                                     "pass specific tensor layout to HostTensorDescriptor (or "
                                     "ck::tensor_layout::BypassLayoutVerification)");
        }

        // GEMM cases
        if constexpr(std::is_base_of_v<ck::tensor_layout::gemm::BaseGemmLayout, Layout>)
        {
            if(mLens.size() != mStrides.size())
            {
                std::ostringstream oss;
                oss << "HostTensorDescriptor::ValidateStrides: mismatch between tensor rank and "
                       "size of strides: "
                    << *this;
                throw std::runtime_error(oss.str());
            }

            // in GEMM, strides must be all positive or all zeros (auto-derived from tensor
            // dimensions)
            auto strides_int = AsInt(mStrides);
            if(std::any_of(
                   strides_int.begin(), strides_int.end(), [](int stride) { return stride <= 0; }))
            {
                std::ostringstream oss;
                oss << "Stride values must be positive or all-zeros (auto-derived from tensor "
                       "dimensions). Instead got ";
                std::copy(
                    strides_int.begin(), strides_int.end(), std::ostream_iterator<int>(oss, " "));
                throw std::runtime_error(oss.str());
            }

            if constexpr(std::is_same_v<ck::tensor_layout::gemm::RowMajor, Layout> ||
                         std::is_same_v<ck::tensor_layout::gemm::ColumnMajor, Layout>)
            {
                // The logic here assumes the GEMM with tensor of more than 2 dims, will always have
                // HW dimesnsions as the inner ones e.g. batched GEMM is either BHW or BWH
                const auto inner_idx =
                    std::is_same_v<ck::tensor_layout::gemm::RowMajor, Layout> ? rank - 1 : rank - 2;
                const auto outer_idx = inner_idx == rank - 1 ? rank - 2 : rank - 1;

                if(mStrides[outer_idx] < mLens[inner_idx] * mStrides[inner_idx])
                {
                    std::ostringstream oss;
                    oss << "Invalid strides for " << layout << ": " << *this;
                    throw std::runtime_error(oss.str());
                }

                // For higher dimensions, validate strides assuming RowMajor
                for(int i = 1; i < rank - 2; ++i)
                {
                    if(mStrides[i - 1] < mStrides[i] * mLens[i])
                    {
                        std::ostringstream oss;
                        oss << "Invalid strides for higher dimensions in " << layout << ": "
                            << *this;
                        throw std::runtime_error(oss.str());
                    }
                }
            }
            else
            {
                std::ostringstream oss;
                oss << "Error: Unsupported GEMM layout: " << layout;
                throw std::runtime_error(oss.str());
            }
        }
        // Convolution cases
        else if constexpr(std::is_base_of_v<ck::tensor_layout::convolution::BaseConvolutionLayout,
                                            Layout>)
        {
            // TBD: implement verification for Conv layouts
            // For now, just print warning and return
            std::cerr << "Warning: Tensor layout verification for ck::tensor_layout::convolution "
                         "layouts is not supported yet. Skipping..."
                      << std::endl;
            return;
        }
        else
        {
            std::ostringstream oss;
            oss << "Error: Tensor layout verification for " << layout << " is not supported yet.";
            throw std::runtime_error(oss.str());
        }
    }

    template <typename X,
              typename Layout = DefaultLayout,
              typename        = std::enable_if_t<std::is_convertible_v<X, std::size_t> &&
                                                 std::is_convertible_v<Layout, BaseTensorLayout>>>
    HostTensorDescriptor(const std::initializer_list<X>& lens, const Layout& layout = Layout{})
        : HostTensorDescriptor(std::vector<std::size_t>(lens.begin(), lens.end()), {}, layout)
    {
        if(dbg)
            std::cout << "HostTensorDescriptor ctor (" << __LINE__ << ")" << std::endl;
    }

    template <typename Layout = DefaultLayout,
              typename        = std::enable_if_t<std::is_convertible_v<Layout, BaseTensorLayout>>>
    HostTensorDescriptor(const std::initializer_list<ck::long_index_t>& lens,
                         const Layout& layout = Layout{})
        : HostTensorDescriptor(std::vector<std::size_t>(lens.begin(), lens.end()), {}, layout)
    {
        if(dbg)
            std::cout << "HostTensorDescriptor ctor (" << __LINE__ << ")" << std::endl;
    }

    template <typename Lengths,
              typename Layout = DefaultLayout,
              typename        = std::enable_if_t<
                         (std::is_convertible_v<ck::ranges::range_value_t<Lengths>, std::size_t> ||
                   std::is_convertible_v<ck::ranges::range_value_t<Lengths>, ck::long_index_t>) &&
                         std::is_convertible_v<Layout, BaseTensorLayout>>>
    HostTensorDescriptor(const Lengths& lens, const Layout& layout = Layout{})
        : HostTensorDescriptor(std::vector<std::size_t>(lens.begin(), lens.end()), {}, layout)
    {
        if(dbg)
            std::cout << "HostTensorDescriptor ctor (" << __LINE__ << ")" << std::endl;
    }

    template <typename X,
              typename Y,
              typename        = std::enable_if_t<std::is_convertible_v<X, std::size_t> &&
                                                 std::is_convertible_v<Y, std::size_t>>,
              typename Layout = DefaultLayout>
    HostTensorDescriptor(const std::initializer_list<X>& lens,
                         const std::initializer_list<Y>& strides,
                         const Layout& layout = Layout{})
        : HostTensorDescriptor(std::vector<std::size_t>(lens.begin(), lens.end()),
                               std::vector<std::size_t>(strides.begin(), strides.end()),
                               layout)
    {
        if(dbg)
            std::cout << "HostTensorDescriptor ctor (" << __LINE__ << ")" << std::endl;
    }

    // HostTensorDescriptor({row, col}, {row_stride, col_stride})
    template <typename Layout = DefaultLayout>
    HostTensorDescriptor(const std::initializer_list<ck::long_index_t>& lens,
                         const std::initializer_list<ck::long_index_t>& strides,
                         const Layout& layout = Layout{})
        : HostTensorDescriptor(std::vector<std::size_t>(lens.begin(), lens.end()),
                               std::vector<std::size_t>(strides.begin(), strides.end()),
                               layout)
    {
        if(dbg)
            std::cout << "HostTensorDescriptor ctor (" << __LINE__ << ")" << std::endl;
    }

    // HostTensorDescriptor({row, col}, strides)
    template <typename Strides, typename Layout = DefaultLayout>
    HostTensorDescriptor(const std::initializer_list<std::size_t>& lens,
                         const Strides& strides,
                         const Layout& layout = Layout{})
        : HostTensorDescriptor(std::vector<std::size_t>(lens.begin(), lens.end()),
                               std::vector<std::size_t>(strides.begin(), strides.end()),
                               layout)
    {
        if(dbg)
            std::cout << "HostTensorDescriptor ctor (" << __LINE__ << ")" << std::endl;
    }

    template <typename Lengths,
              typename Strides,
              typename Layout = DefaultLayout,
              typename        = std::enable_if_t<
                         ((std::is_convertible_v<ck::ranges::range_value_t<Lengths>, std::size_t> &&
                    std::is_convertible_v<ck::ranges::range_value_t<Strides>, std::size_t>) ||
                   (std::is_convertible_v<ck::ranges::range_value_t<Lengths>, ck::long_index_t> &&
                    std::is_convertible_v<ck::ranges::range_value_t<Strides>, ck::long_index_t>)) &&
                         std::is_convertible_v<Layout, BaseTensorLayout>>>
    HostTensorDescriptor(const Lengths& lens,
                         const Strides& strides,
                         const Layout& layout = Layout{})
        : HostTensorDescriptor(std::vector<std::size_t>(lens.begin(), lens.end()),
                               std::vector<std::size_t>(strides.begin(), strides.end()),
                               layout)
    {
        if(dbg)
            std::cout << "HostTensorDescriptor ctor (" << __LINE__ << ")" << std::endl;
    }

    std::size_t GetNumOfDimension() const;
    std::size_t GetElementSize() const;
    std::size_t GetElementSpaceSize() const;

    const std::vector<std::size_t>& GetLengths() const;
    const std::vector<std::size_t>& GetStrides() const;

    template <typename... Is>
    std::size_t GetOffsetFromMultiIndex(Is... is) const
    {
        assert(sizeof...(Is) == this->GetNumOfDimension());
        std::initializer_list<std::size_t> iss{static_cast<std::size_t>(is)...};
        return std::inner_product(iss.begin(), iss.end(), mStrides.begin(), std::size_t{0});
    }

    std::size_t GetOffsetFromMultiIndex(const std::vector<std::size_t>& iss) const
    {
        return std::inner_product(iss.begin(), iss.end(), mStrides.begin(), std::size_t{0});
    }

    friend std::ostream& operator<<(std::ostream& os, const HostTensorDescriptor& desc);
    friend std::ostream& operator<<(std::ostream& os, ChosenLayout tag);

    private:
    std::vector<std::size_t> mLens;
    std::vector<std::size_t> mStrides;
    static constexpr bool dbg = false;

    /**
     * @brief Converts a vector of size_t values to a vector of int values.
     *
     * @param vec The input vector of size_t values to be converted.
     * @return std::vector<int> A vector containing the converted int values.
     */
    std::vector<int> AsInt(const std::vector<size_t>& vec) const
    {
        std::vector<int> strides_int(vec.size());
        std::transform(vec.begin(), vec.end(), strides_int.begin(), [](std::size_t stride) {
            return static_cast<int>(stride);
        });
        return strides_int;
    }
};

template <typename New2Old, typename NewLayout = HostTensorDescriptor::BaseTensorLayout>
HostTensorDescriptor
transpose_host_tensor_descriptor_given_new2old(const HostTensorDescriptor& a,
                                               const New2Old& new2old,
                                               const NewLayout& new_layout = NewLayout())
{
    std::vector<std::size_t> new_lengths(a.GetNumOfDimension());
    std::vector<std::size_t> new_strides(a.GetNumOfDimension());

    for(std::size_t i = 0; i < a.GetNumOfDimension(); i++)
    {
        new_lengths[i] = a.GetLengths()[new2old[i]];
        new_strides[i] = a.GetStrides()[new2old[i]];
    }

    return HostTensorDescriptor(new_lengths, new_strides, new_layout);
}

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

template <typename F, typename... Xs>
struct ParallelTensorFunctor
{
    F mF;
    static constexpr std::size_t NDIM = sizeof...(Xs);
    std::array<std::size_t, NDIM> mLens;
    std::array<std::size_t, NDIM> mStrides;
    std::size_t mN1d;

    ParallelTensorFunctor(F f, Xs... xs) : mF(f), mLens({static_cast<std::size_t>(xs)...})
    {
        mStrides.back() = 1;
        std::partial_sum(mLens.rbegin(),
                         mLens.rend() - 1,
                         mStrides.rbegin() + 1,
                         std::multiplies<std::size_t>());
        mN1d = mStrides[0] * mLens[0];
    }

    std::array<std::size_t, NDIM> GetNdIndices(std::size_t i) const
    {
        std::array<std::size_t, NDIM> indices;

        for(std::size_t idim = 0; idim < NDIM; ++idim)
        {
            indices[idim] = i / mStrides[idim];
            i -= indices[idim] * mStrides[idim];
        }

        return indices;
    }

    void operator()(std::size_t num_thread = 1) const
    {
        std::size_t work_per_thread = (mN1d + num_thread - 1) / num_thread;

        std::vector<joinable_thread> threads(num_thread);

        for(std::size_t it = 0; it < num_thread; ++it)
        {
            std::size_t iw_begin = it * work_per_thread;
            std::size_t iw_end   = std::min((it + 1) * work_per_thread, mN1d);

            auto f = [=, *this] {
                for(std::size_t iw = iw_begin; iw < iw_end; ++iw)
                {
                    call_f_unpack_args(mF, GetNdIndices(iw));
                }
            };
            threads[it] = joinable_thread(f);
        }
    }
};

template <typename F, typename... Xs>
auto make_ParallelTensorFunctor(F f, Xs... xs)
{
    return ParallelTensorFunctor<F, Xs...>(f, xs...);
}

template <typename T>
struct Tensor
{
    using Descriptor = HostTensorDescriptor;
    using Data       = std::vector<T>;

    template <typename X>
    Tensor(std::initializer_list<X> lens) : mDesc(lens), mData(GetElementSpaceSize())
    {
    }

    template <typename X, typename Y>
    Tensor(std::initializer_list<X> lens, std::initializer_list<Y> strides)
        : mDesc(lens, strides), mData(GetElementSpaceSize())
    {
    }

    template <typename Lengths>
    Tensor(const Lengths& lens) : mDesc(lens), mData(GetElementSpaceSize())
    {
    }

    template <typename Lengths, typename Strides>
    Tensor(const Lengths& lens, const Strides& strides)
        : mDesc(lens, strides), mData(GetElementSpaceSize())
    {
    }

    template <typename X, typename... Rest, std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    Tensor(std::initializer_list<X> lens, Rest&&... rest)
        : mDesc(lens, std::forward<Rest>(rest)...), mData(GetElementSpaceSize())
    {
    }

    template <typename X,
              typename Y,
              typename... Rest,
              std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    Tensor(std::initializer_list<X> lens, std::initializer_list<Y> strides, Rest&&... rest)
        : mDesc(lens, strides, std::forward<Rest>(rest)...), mData(GetElementSpaceSize())
    {
    }

    template <typename Lengths, typename... Rest, std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    Tensor(const Lengths& lens, Rest&&... rest)
        : mDesc(lens, std::forward<Rest>(rest)...), mData(GetElementSpaceSize())
    {
    }

    template <typename Lengths,
              typename Strides,
              typename... Rest,
              std::enable_if_t<(sizeof...(Rest) > 0), int> = 0>
    Tensor(const Lengths& lens, const Strides& strides, Rest&&... rest)
        : mDesc(lens, strides, std::forward<Rest>(rest)...), mData(GetElementSpaceSize())
    {
    }

    Tensor(const Descriptor& desc) : mDesc(desc), mData(GetElementSpaceSize()) {}

    template <typename OutT>
    Tensor<OutT> CopyAsType() const
    {
        Tensor<OutT> ret(mDesc);

        ck::ranges::transform(
            mData, ret.mData.begin(), [](auto value) { return ck::type_convert<OutT>(value); });

        return ret;
    }

    Tensor()              = delete;
    Tensor(const Tensor&) = default;
    Tensor(Tensor&&)      = default;

    ~Tensor() = default;

    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&)      = default;

    template <typename FromT>
    explicit Tensor(const Tensor<FromT>& other) : Tensor(other.template CopyAsType<T>())
    {
    }
    void savetxt(std::string file_name, std::string dtype = "float")
    {
        std::ofstream file(file_name);

        if(file.is_open())
        {
            for(auto& itm : mData)
            {
                if(dtype == "float")
                    file << ck::type_convert<float>(itm) << std::endl;
                else if(dtype == "int")
                    file << ck::type_convert<int>(itm) << std::endl;
                else
                    // TODO: we didn't implement operator<< for all custom
                    // data types, here fall back to float in case compile error
                    file << ck::type_convert<float>(itm) << std::endl;
            }
            file.close();
        }
        else
        {
            // Print an error message to the standard error
            // stream if the file cannot be opened.
            throw std::runtime_error(std::string("unable to open file:") + file_name);
        }
    }
    decltype(auto) GetLengths() const { return mDesc.GetLengths(); }

    decltype(auto) GetStrides() const { return mDesc.GetStrides(); }

    std::size_t GetNumOfDimension() const { return mDesc.GetNumOfDimension(); }

    std::size_t GetElementSize() const { return mDesc.GetElementSize(); }

    std::size_t GetElementSpaceSize() const
    {
        if constexpr(ck::is_packed_type_v<ck::remove_cvref_t<T>>)
        {
            return (mDesc.GetElementSpaceSize() + 1) / ck::packed_size_v<ck::remove_cvref_t<T>>;
        }
        else
        {
            return mDesc.GetElementSpaceSize();
        }
    }

    std::size_t GetElementSpaceSizeInBytes() const { return sizeof(T) * GetElementSpaceSize(); }

    void SetZero() { ck::ranges::fill<T>(mData, T{0}); }

    template <typename F>
    void ForEach_impl(F&& f, std::vector<size_t>& idx, size_t rank)
    {
        if(rank == mDesc.GetNumOfDimension())
        {
            f(*this, idx);
            return;
        }
        // else
        for(size_t i = 0; i < mDesc.GetLengths()[rank]; i++)
        {
            idx[rank] = i;
            ForEach_impl(std::forward<F>(f), idx, rank + 1);
        }
    }

    template <typename F>
    void ForEach(F&& f)
    {
        std::vector<size_t> idx(mDesc.GetNumOfDimension(), 0);
        ForEach_impl(std::forward<F>(f), idx, size_t(0));
    }

    template <typename F>
    void ForEach_impl(const F&& f, std::vector<size_t>& idx, size_t rank) const
    {
        if(rank == mDesc.GetNumOfDimension())
        {
            f(*this, idx);
            return;
        }
        // else
        for(size_t i = 0; i < mDesc.GetLengths()[rank]; i++)
        {
            idx[rank] = i;
            ForEach_impl(std::forward<const F>(f), idx, rank + 1);
        }
    }

    template <typename F>
    void ForEach(const F&& f) const
    {
        std::vector<size_t> idx(mDesc.GetNumOfDimension(), 0);
        ForEach_impl(std::forward<const F>(f), idx, size_t(0));
    }

    template <typename G>
    void GenerateTensorValue(G g, std::size_t num_thread = 1)
    {
        switch(mDesc.GetNumOfDimension())
        {
        case 1: {
            auto f = [&](auto i) { (*this)(i) = g(i); };
            make_ParallelTensorFunctor(f, mDesc.GetLengths()[0])(num_thread);
            break;
        }
        case 2: {
            auto f = [&](auto i0, auto i1) { (*this)(i0, i1) = g(i0, i1); };
            make_ParallelTensorFunctor(f, mDesc.GetLengths()[0], mDesc.GetLengths()[1])(num_thread);
            break;
        }
        case 3: {
            auto f = [&](auto i0, auto i1, auto i2) { (*this)(i0, i1, i2) = g(i0, i1, i2); };
            make_ParallelTensorFunctor(
                f, mDesc.GetLengths()[0], mDesc.GetLengths()[1], mDesc.GetLengths()[2])(num_thread);
            break;
        }
        case 4: {
            auto f = [&](auto i0, auto i1, auto i2, auto i3) {
                (*this)(i0, i1, i2, i3) = g(i0, i1, i2, i3);
            };
            make_ParallelTensorFunctor(f,
                                       mDesc.GetLengths()[0],
                                       mDesc.GetLengths()[1],
                                       mDesc.GetLengths()[2],
                                       mDesc.GetLengths()[3])(num_thread);
            break;
        }
        case 5: {
            auto f = [&](auto i0, auto i1, auto i2, auto i3, auto i4) {
                (*this)(i0, i1, i2, i3, i4) = g(i0, i1, i2, i3, i4);
            };
            make_ParallelTensorFunctor(f,
                                       mDesc.GetLengths()[0],
                                       mDesc.GetLengths()[1],
                                       mDesc.GetLengths()[2],
                                       mDesc.GetLengths()[3],
                                       mDesc.GetLengths()[4])(num_thread);
            break;
        }
        case 6: {
            auto f = [&](auto i0, auto i1, auto i2, auto i3, auto i4, auto i5) {
                (*this)(i0, i1, i2, i3, i4, i5) = g(i0, i1, i2, i3, i4, i5);
            };
            make_ParallelTensorFunctor(f,
                                       mDesc.GetLengths()[0],
                                       mDesc.GetLengths()[1],
                                       mDesc.GetLengths()[2],
                                       mDesc.GetLengths()[3],
                                       mDesc.GetLengths()[4],
                                       mDesc.GetLengths()[5])(num_thread);
            break;
        }
        case 12: {
            auto f = [&](auto i0,
                         auto i1,
                         auto i2,
                         auto i3,
                         auto i4,
                         auto i5,
                         auto i6,
                         auto i7,
                         auto i8,
                         auto i9,
                         auto i10,
                         auto i11) {
                (*this)(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11) =
                    g(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11);
            };
            make_ParallelTensorFunctor(f,
                                       mDesc.GetLengths()[0],
                                       mDesc.GetLengths()[1],
                                       mDesc.GetLengths()[2],
                                       mDesc.GetLengths()[3],
                                       mDesc.GetLengths()[4],
                                       mDesc.GetLengths()[5],
                                       mDesc.GetLengths()[6],
                                       mDesc.GetLengths()[7],
                                       mDesc.GetLengths()[8],
                                       mDesc.GetLengths()[9],
                                       mDesc.GetLengths()[10],
                                       mDesc.GetLengths()[11])(num_thread);
            break;
        }
        default: throw std::runtime_error("unspported dimension");
        }
    }

    // Generate random values with multiple threads. Guaranteed to give the same sequence with any
    // number of threads provided.
    template <typename Distribution = std::uniform_real_distribution<float>,
              typename Mapping      = ck::identity,
              typename Generator    = std::minstd_rand>
    void GenerateTensorDistr(Distribution dis       = {0.f, 1.f},
                             Mapping fn             = {},
                             const Generator g      = Generator(0), // default seed 0
                             std::size_t num_thread = -1)
    {
        using ck::math::integer_divide_ceil;
        using ck::math::min;
        if(num_thread == -1ULL)
            num_thread = min(ck::get_available_cpu_cores(), 80U); // max 80 threads
        // At least 2MB per thread
        num_thread = min(num_thread, integer_divide_ceil(this->GetElementSpaceSize(), 0x200000));
        constexpr std::size_t BLOCK_BYTES = 64;
        constexpr std::size_t BLOCK_SIZE  = BLOCK_BYTES / sizeof(T);

        const std::size_t num_blocks = integer_divide_ceil(this->GetElementSpaceSize(), BLOCK_SIZE);
        const std::size_t blocks_per_thread = integer_divide_ceil(num_blocks, num_thread);

        std::vector<std::thread> threads;
        threads.reserve(num_thread - 1);
        const auto dst                = const_cast<T*>(this->mData.data());
        const auto element_space_size = this->GetElementSpaceSize();
        for(int it = num_thread - 1; it >= 0; --it)
        {
            std::size_t ib_begin = it * blocks_per_thread;
            std::size_t ib_end   = min(ib_begin + blocks_per_thread, num_blocks);

            auto job = [=]() {
                auto g_   = g;   // copy
                auto dis_ = dis; // copy
                g_.discard(ib_begin * BLOCK_SIZE * ck::packed_size_v<T>);
                auto t_fn = [&]() {
                    // As user can pass integer distribution in dis, we must ensure that the correct
                    // constructor/converter is called at all times. For f4/f6/f8 types, to ensure
                    // correct results, we convert from float to the target type. In these cases
                    // integer constructors are interpreted as direct initialization of the internal
                    // storage with binary values instead of treating integers as subset of floats.
                    if constexpr(ck::is_same_v<T, ck::f8_t> || ck::is_same_v<T, ck::bf8_t>)
                        return ck::type_convert<T>(static_cast<float>(fn(dis_(g_))));
                    else if constexpr(ck::packed_size_v<T> == 1)
                        return ck::type_convert<T>(fn(dis_(g_)));
                    else if constexpr(ck::is_same_v<T, ck::f4x2_pk_t>)
                        return ck::f4x2_pk_t{ck::type_convert<ck::f4x2_t>(
                            ck::float2_t{ck::type_convert<float>(fn(dis_(g_))),
                                         ck::type_convert<float>(fn(dis_(g_)))})};
                    else if constexpr(ck::is_same_v<T, ck::f6x32_pk_t> ||
                                      ck::is_same_v<T, ck::bf6x32_pk_t>)
                    {
                        return ck::type_convert<T>(
                            ck::float32_t{ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_)))});
                    }
                    else if constexpr(ck::is_same_v<T, ck::f6x16_pk_t> ||
                                      ck::is_same_v<T, ck::bf6x16_pk_t>)
                    {
                        return ck::type_convert<T>(
                            ck::float16_t{ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_))),
                                          ck::type_convert<float>(fn(dis_(g_)))});
                    }
                    else
                        static_assert(false, "Unsupported packed size for T");
                };

                std::size_t ib = ib_begin;
                for(; ib < ib_end - 1; ++ib)
                    ck::static_for<0, BLOCK_SIZE, 1>{}([&](auto iw_) {
                        constexpr size_t iw       = iw_.value;
                        dst[ib * BLOCK_SIZE + iw] = t_fn();
                    });
                for(std::size_t iw = 0; iw < BLOCK_SIZE; ++iw)
                    if(ib * BLOCK_SIZE + iw < element_space_size)
                        dst[ib * BLOCK_SIZE + iw] = t_fn();
            };

            if(it > 0)
                threads.emplace_back(std::move(job));
            else
                job(); // last job run in the main thread
        }
        for(auto& t : threads)
            t.join();
    }

    template <typename... Is>
    std::size_t GetOffsetFromMultiIndex(Is... is) const
    {
        return mDesc.GetOffsetFromMultiIndex(is...) / ck::packed_size_v<ck::remove_cvref_t<T>>;
    }

    template <typename... Is>
    T& operator()(Is... is)
    {
        return mData[mDesc.GetOffsetFromMultiIndex(is...) /
                     ck::packed_size_v<ck::remove_cvref_t<T>>];
    }

    template <typename... Is>
    const T& operator()(Is... is) const
    {
        return mData[mDesc.GetOffsetFromMultiIndex(is...) /
                     ck::packed_size_v<ck::remove_cvref_t<T>>];
    }

    T& operator()(const std::vector<std::size_t>& idx)
    {
        return mData[mDesc.GetOffsetFromMultiIndex(idx) / ck::packed_size_v<ck::remove_cvref_t<T>>];
    }

    const T& operator()(const std::vector<std::size_t>& idx) const
    {
        return mData[mDesc.GetOffsetFromMultiIndex(idx) / ck::packed_size_v<ck::remove_cvref_t<T>>];
    }

    typename Data::iterator begin() { return mData.begin(); }

    typename Data::iterator end() { return mData.end(); }

    typename Data::pointer data() { return mData.data(); }

    typename Data::const_iterator begin() const { return mData.begin(); }

    typename Data::const_iterator end() const { return mData.end(); }

    typename Data::const_pointer data() const { return mData.data(); }

    typename Data::size_type size() const { return mData.size(); }

    template <typename U = T>
    auto AsSpan() const
    {
        constexpr std::size_t FromSize = sizeof(T);
        constexpr std::size_t ToSize   = sizeof(U);

        using Element = std::add_const_t<std::remove_reference_t<U>>;
        return ck::span<Element>{reinterpret_cast<Element*>(data()), size() * FromSize / ToSize};
    }

    template <typename U = T>
    auto AsSpan()
    {
        constexpr std::size_t FromSize = sizeof(T);
        constexpr std::size_t ToSize   = sizeof(U);

        using Element = std::remove_reference_t<U>;
        return ck::span<Element>{reinterpret_cast<Element*>(data()), size() * FromSize / ToSize};
    }

    Descriptor mDesc;
    Data mData;
};

} // namespace ck
