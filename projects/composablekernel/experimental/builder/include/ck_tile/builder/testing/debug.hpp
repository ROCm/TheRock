// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/testing/tensor_descriptor.hpp"
#include "ck_tile/builder/testing/error.hpp"
#include "ck_tile/builder/testing/type_traits.hpp"
#include "ck/utility/type_convert.hpp"
#include <iostream>
#include <locale>
#include <string>
#include <string_view>
#include <syncstream>
#include <concepts>
#include <limits>

/// This file contains a few debugging utilities, mainly focused around
/// tensor data. The idea is that the functionality in this file is not
/// necessarily used in any testing directly, but is available for the
/// programmer to help with debugging problems. These utilities themselves
/// should be tested just the same, though, so that they don't undergo
/// bitrot while they are not actively being used.

namespace ck_tile::builder::test {

namespace detail {

/// @brief Custom number punctuation for CK-Builder debugging.
///
/// During debugging, the locale is usually left to the default C locale.
/// The C locale does not have any thousands separator, which makes
/// large numbers hard to read. This is a specialization of the default
/// C++ number punctuation (`std::numpunct`) which separates thousands
/// using `'`, which helps getting a quick overview of the magnitude of
/// a number. This character is chosen because C++14 allows number literals
/// to have this character.
///
/// @note When using this locale, be sure to restore the old locale in the
/// event that the user actually wants to use a non-standard locale.
///
/// @see std::numpunct
struct numpunct : std::numpunct<char>
{
    char do_thousands_sep() const override { return '\''; }

    std::string do_grouping() const override
    {
        // See std::numpunct, this separates by thousands.
        return "\3";
    }
};

} // namespace detail

/// @brief Print information about a tensor descriptor.
///
/// This function dumps useful information from a tensor descriptor to a
/// stream, `std::cout` by default. This includes the number of elements
/// in the tensor, the size of the backing space, lengths, strides, etc.
///
/// @note All information is printed using a lightly modified locale to
/// get a unified printing experience. The original locale in `stream` is
/// temporarily replaced, but restored before the function returns.
///
/// @tparam DT The tensor element datatype
/// @tparam RANK The rank (number of spatial dimensions) of the tensor.
///
/// @param name A name for the tensor descriptor.
/// @param desc The tensor descriptor to print.
/// @param out The stream to print to, `std::cout` by default.
template <DataType DT, size_t RANK>
void print_descriptor(std::string_view name,
                      const TensorDescriptor<DT, RANK>& desc,
                      std::ostream& out = std::cout)
{
    // Create a custom stream with a completely new config (locale,
    /// precision, fill, etc). Use an osyncstream to buffer the output
    /// while were at it (its not likely to help a lot, but why not).
    std::osyncstream stream(out.rdbuf());
    stream.imbue(std::locale(std::locale(), new detail::numpunct{}));

    // Print name along with some generic info
    const auto size   = desc.get_element_size();
    const auto space  = desc.get_element_space_size();
    const auto bytes  = desc.get_element_space_size_in_bytes();
    const auto packed = desc.is_packed();

    stream << "Descriptor \"" << name << "\":\n"
           << "  data type: " << DT << '\n'
           << "  size:      " << size << " elements\n"
           << "  space:     " << space << " elements (" << bytes << " bytes)\n"
           << "  lengths:   " << desc.get_lengths() << '\n'
           << "  strides:   " << desc.get_strides() << '\n'
           << "  packed:    " << (packed ? "yes" : "no") << std::endl;
}

/// @brief User configuration for printing tensors.
///
/// This structure houses some configuration fields for customizing how tensors
/// are printed. The default is usually good, though `TensorPrintConfig::unlimited()`
/// is useful if you want to print the entire tensor to the output regardless of size.
struct TensorPrintConfig
{
    /// @brief A limit for the number of columns in a tensor row to print.
    ///
    /// Each row of a tensor will be printed as a sequence of values. At most
    /// this number of values are printed, if there are more, `row_skip_val`
    /// will be printed in between.
    size_t col_limit = 10;

    /// @brief A limit for the number of rows in a 2D matrix to print
    ///
    /// Tensors with rank higher than 1 are printed as a single matrix or a series
    /// of matrix slices. At most this number of rows of the matrix will be printed.
    /// If there are more rows, a row of `matrix_row_skip_val` and possibly
    /// `row_skip_val` will be printed in between.
    size_t row_limit = 10;

    /// @brief A limit for the number of 2D tensor slices to print.
    ///
    /// Tensors with rank higher than 2 are flattened into a sequence of slices. At
    /// most this number of slices will be printed.
    size_t slice_limit = 8;

    /// @brief Text to print at the start of a row of values.
    ///
    /// This is used by `TensorPrinter`, and printed at the start of a row of tensor
    /// values.
    std::string_view row_prefix = " ";

    /// @brief Text to print between fields of a row.
    ///
    /// This is used by `TensorPrinter`, and printed between each value of a row of
    /// tensor values.
    std::string_view row_field_sep = " ";

    /// @brief Text to print when skipping some number of row values.
    ///
    /// This is used by `TensorPrinter`, and printed instead of some number of values
    /// when the number of values in a row is too large to all print.
    std::string_view row_skip_val = "...";

    /// @brief Text to print when skipping a row of a matrix.
    ///
    /// This is used by `TensorPrinter`, and printed instead of a value when some
    /// number of rows is skipped when printing a matrix. This is similar to
    /// `row_skip_val`, except in the vertical direction. Note that ALL values
    /// in the skip row is printed this way.
    std::string_view matrix_row_skip_val = "...";

    /// @brief The precision of tensor floating point values.
    ///
    /// Set the number of decimal digits that is printed for a floating point value.
    int float_precision = 3;

    /// @brief Return the default print config, but without any printing limits.
    ///
    /// This is useful if you want to print the *entire* tensor, but be aware that
    /// this may print a lot of data if the tensor is large!
    constexpr static TensorPrintConfig unlimited()
    {
        return {
            .col_limit   = std::numeric_limits<size_t>::max(),
            .row_limit   = std::numeric_limits<size_t>::max(),
            .slice_limit = std::numeric_limits<size_t>::max(),
        };
    }
};

namespace detail {

/// @brief Iterate over a range of values, but limit the amount of iterations.
///
/// Iterate over values `0..n`, but if `limit > n`, only iterate over the
/// first and last few (`limit // 2)` items. This can be used to iterate over
/// large ranges in a way that not too many values are visited. Its primarily
/// used when printing tensors so that not all values of a giant tensor are
/// dumped to the user's terminal.
///
/// @param n The total number of items to iterate over.
/// @param limit The maximum number of items to iterate over. Use even values
/// for best results, as this will lead to the same amount of values in the
/// "begin" and "end" sections.
/// @param f A functor to invoke for each element. The sole parameter is the
/// index.
/// @param delim A functor to invoke between the begin and end sections. This
/// function is only invoked if any items are skipped at all.
void limited_foreach(size_t n, size_t limit, auto f, auto delim)
{
    if(n <= limit)
    {
        for(size_t i = 0; i < n; ++i)
            f(i);
    }
    else
    {
        const auto begin_count = (limit + 1) / 2; // Round up in case `delim` is odd.
        const auto end_count   = limit / 2;
        const auto skip_count  = n - limit;

        for(size_t i = 0; i < begin_count; ++i)
            f(i);

        delim(skip_count);

        for(size_t i = n - end_count; i < n; ++i)
            f(i);
    }
};

/// @brief Output stream requirements for use with `TensorPrinter`.
///
/// The `TensorPrinter` does not write to an ostream directly, but rather writes to
/// a custom stream object. This is mainly so that the user of `TensorPrinter` can
/// get more details than directly with an ostream. Basically, a valid implementation
/// of `TensorPrintStream` exposes 3 things:
/// - A way to print (stringified) tensor elements.
/// - A way to print arbitrary text messages. These are mostly for formatting. This
///   should be implemented using varargs which are directly folded into an ostream,
///   so that <iomanip> functions can be used.
/// - A way to query the max width of any `val` field.
///
/// @see TensorPrinter for more information.
template <typename Stream>
concept TensorPrintStream = requires(Stream& stream, std::string_view val) {
    { stream.max_width } -> std::convertible_to<size_t>;
    { stream.val(val) } -> std::same_as<void>;
    { stream.msg() } -> std::same_as<void>;
    { stream.msg("msg") } -> std::same_as<void>;
    { stream.msg(std::setw(3), std::setfill(4), "msg", val) } -> std::same_as<void>;
};

/// @brief Utility to print tensors.
///
/// This structure implements the main logic for printing tensors to a stream.
/// In order to help with formatting, the `TensorPrinter` abstracts over a custom
/// stream type, see `TensorPrintStream`. This type is actually mostly an internal
/// helper and mainly used by `print_tensor`. Its supposed to be constructed
/// manually, but see the field docs for what is required.
///
/// @tparam DT The data type of the tensor to print.
/// @tparam RANK The rank (number of spatial dimensions) of the tensor to print.
///
/// @see print_tensor
template <DataType DT, size_t RANK>
struct TensorPrinter
{
    /// The name of this tensor. This will be used during printing to add extra
    /// clarity about what the user is seeing.
    std::string_view name;

    /// Configuration details of how to print the tensor. This should be able to
    /// be specified by the user, but the default is good in most cases.
    TensorPrintConfig config;

    /// The lengths of the tensor to print. These values are directly from
    /// `TensorDescriptor::get_lengths()`, stored here to avoid querying them
    /// repeatedly.
    Extent<RANK> lengths;

    /// The strides of the tensor to print. These values are directly from
    /// `TensorDescriptor::get_strides()`, stored here to avoid querying them
    /// repeatedly.
    Extent<RANK> strides;

    /// The tensor's backing buffer. This memory should be host-accessible, for
    /// example by copying it back to the host first.
    const void* h_buffer;

    /// A common stringstream for stringifying tensor values. This is here mostly
    /// so that we can cache the internal allocation.
    std::stringstream ss;

    /// @brief Low-level tensor value stringifying function.
    ///
    /// Print value `value` to the stringstream `ss` (member value). This function
    /// is the actual low-level printing function that prints each element of the
    /// tensor. In order to get a robust printing implementation, the value is written
    /// directly into a stringstream, which is then further processed to be actually
    /// written to the output. This way, the format doesn't depend on the ostream
    /// configuration.
    ///
    /// @param value The value to print to the stream.
    void stringify_value(const void* value)
    {
        if constexpr(DT == DataType::UNDEFINED_DATA_TYPE)
        {
            ss << "??";
            return;
        }

        using CKType        = detail::cpp_type_t<DT>;
        const auto ck_value = *static_cast<const CKType*>(value);

        if constexpr(DT == DataType::I32 || DT == DataType::I8 || DT == DataType::U8)
            ss << ck_value;
        else if constexpr(DT == DataType::FP64 || DT == DataType::FP32)
            ss << std::fixed << std::setprecision(config.float_precision) << ck_value;
        else if constexpr(DT == DataType::FP16 || DT == DataType::BF16 || DT == DataType::FP8 ||
                          DT == DataType::BF8)
            ss << std::fixed
               << std::setprecision(config.float_precision)
               // Note: We are using CK types here (cpp_type_t uses DataTypeToCK), so
               // use CK's type_convert function.
               << ::ck::type_convert<float>(ck_value);
        else
            // TODO: Tuple types? Currently not implemented in DataTypeToCK...
            static_assert(false, "stringify_value unsupported data type, please implement");
    }

    /// @brief Print the value at an index to a stream.
    ///
    /// This function reads the value at `index` and prints it to `stream` (using
    /// `stream.val(...)`).
    ///
    /// @param stream The stream to print to.
    /// @param index The index in the tensor of the value to print.
    void print_value(TensorPrintStream auto& stream, const Extent<RANK>& index)
    {
        const auto offset = calculate_offset(index, strides);
        const auto* value_ptr =
            &static_cast<const std::byte*>(h_buffer)[offset * data_type_sizeof(DT)];

        // Reset the stream without allocating.
        // ss.str("") allocates...
        ss.clear();
        ss.seekg(0);
        ss.seekp(0);
        stringify_value(value_ptr);
        // ss.view() returns a view of the ENTIRE buffer, which may have
        // lingering data since we used seekp() and seekg() to reset the
        // stream. For some reason std::stringstream works this way...
        // Fortunately tellp() returns how many bytes we've actually
        // written.
        const auto view = ss.view().substr(0, ss.tellp());
        stream.val(view);
    }

    /// @brief Print a 1D row to a stream.
    ///
    /// Print a row of tensor values to the stream. This function is used for both
    /// 1D tensors and for rows of 2D tensors, in which the base coordinate is given
    /// by `index`. Note that the print configuration is taken into account to avoid
    /// flooding the user's terminal with values.
    ///
    /// @param stream The stream to print to.
    /// @param index The index of the row to print. The rightmost index element is
    /// ignored, as that is the index of the value _within_ the row.
    void print_row(TensorPrintStream auto& stream, Extent<RANK>& index)
    {
        // See note in `print_matrix`.
        stream.msg(config.row_prefix);
        limited_foreach(
            lengths[RANK - 1],
            config.col_limit,
            [&](auto i) {
                stream.msg(config.row_field_sep);
                index[RANK - 1] = i;
                print_value(stream, index);
            },
            [&]([[maybe_unused]] auto skip_count) {
                stream.msg(config.row_field_sep);
                // Note: Not using stream.val(...) here because we don't want this
                // field to partake in max_width computation, nor do we want to
                // pad it to the max width.
                stream.msg(config.row_skip_val);
            });

        stream.msg('\n');
    }

    /// @brief Print a 2D matrix to a stream.
    ///
    /// Print a matrix of tensor values to the stream. This function is used for both
    /// 2D and slices of higher-dimensional tensors, in which the base coordinate is
    /// given by `index`. Note that the print configuration is taken into account to
    /// avoid flooding the user's terminal with values.
    ///
    /// @param stream The stream to print to.
    /// @param index The index of the row to print. The 2 rightmost index elements are
    /// ignored, as those are the indices of values _within_ the matrix.
    void print_matrix(TensorPrintStream auto& stream, Extent<RANK>& index)
    {
        limited_foreach(
            lengths[RANK - 2],
            config.row_limit,
            [&](auto i) {
                index[RANK - 2] = i;
                print_row(stream, index);
            },
            [&]([[maybe_unused]] auto row_skip_count) {
                // When we encounter a skip row, continue with the same logic
                // as printing 1D tensor rows. Instead of actual values, we will
                // simply print MATRIX_ROW_SKIP_VAL (usually something like "...").
                stream.msg(config.row_prefix);
                limited_foreach(
                    lengths[RANK - 1],
                    config.col_limit,
                    [&]([[maybe_unused]] auto i) {
                        stream.msg(config.row_field_sep);
                        // Note: We're using `stream.val(...)` here because we *do* want this field
                        // to partake in max_width computation, and we *do* want to pad it like
                        // value fields. This is so that these appear the same width as actual
                        // values, so that everything is neatly aligned. This also ensures that if
                        // there are no skip values, then the size of the skip field is not taken
                        // into account.
                        stream.val(config.matrix_row_skip_val);
                    },
                    [&]([[maybe_unused]] auto col_skip_count) {
                        stream.msg(config.row_field_sep);
                        // Note: Not using stream.val(...) here because we don't want this
                        // field to partake in max_width computation, nor do we want to
                        // pad it to the max width.
                        stream.msg(config.row_skip_val);
                    });
                stream.msg('\n');
            });
    }

    /// @brief Print a tensor to a stream.
    ///
    /// This is the main tensor printing function. It calls `print_row` or `print_matrix`
    /// (possibly repeatedly) as required. This function prints the entire tensor in
    /// `h_buffer` regardless.
    ///
    /// @param stream The stream to print to.
    void print_tensor(TensorPrintStream auto& stream)
    {
        Extent<RANK> zero_coord = {};
        if constexpr(RANK == 0)
        {
            // 0D case: just print the one value
            stream.msg(config.row_prefix);
            stream.msg(config.row_field_sep);
            print_value(stream, zero_coord);
            stream.msg('\n');
        }
        else if constexpr(RANK == 1)
        {
            // 1D case: dump everything on one line
            print_row(stream, zero_coord);
        }
        else if constexpr(RANK == 2)
        {
            // 2D case: print a 2D matrix
            print_matrix(stream, zero_coord);
        }
        else
        {
            // For higher dimensions, print each window as a slice
            // We want to limit the *total* number of slices using `slice_limit`,
            // not the number in each axis. So flatten the remaining dimensions.
            // This also avoids recursion in this function in general.

            // First get the shape minus the 2 inner dimensions
            Extent<RANK - 2> outer_shape;
            std::copy_n(lengths.begin(), RANK - 2, outer_shape.begin());

            NdIter iter(outer_shape);
            detail::limited_foreach(
                iter.numel(),
                config.slice_limit,
                [&](auto outer_flat_index) {
                    // Now decode the outer index and turn it back into a complete index
                    const auto outer_index = iter(outer_flat_index);
                    Extent<RANK> index     = {};
                    std::copy_n(outer_index.begin(), RANK - 2, index.begin());

                    // Print an extra separating line between two slices
                    if(outer_flat_index != 0)
                        stream.msg('\n');

                    // Print an information header about the current slice
                    stream.msg("Tensor \"", name, "\", slice [");
                    for(auto x : outer_index)
                        stream.msg(x, ", ");
                    stream.msg(":, :]\n");

                    // And print is as matrix
                    print_matrix(stream, index);
                },
                [&](auto skip_count) { stream.msg("\n(skipping ", skip_count, " slices...)\n"); });
        }
    }
};

/// @brief Implementation of `TensorPrintStream` to figure out the maximum
/// width of a field.
///
/// In order to produce neatly aligned tensors, where all values of each row
/// appear on the same columns, we have to figure out the maximum width of
/// each field. This print stream helps with that: It does not actually print
/// anything, it just figures out the maximum width of any value (not message).
///
/// @details OK, this function does actually print things, but only to an
/// internal `stringstream`. This is so that we can easily figure out the
/// width of the field (in bytes), just by counting the amount of bytes
/// written into the string stream.
///
/// @see TensorPrintStream
struct MaxFieldWidthStream
{
    size_t max_width = 0;

    /// @brief Print a tensor value to the stream
    ///
    /// "Print" a value to the stream. This function figures out the width
    /// of the value when printed, and then composes it with `max_width` to
    /// figure out the total maximum.
    ///
    /// @param value The value to print.
    void val(std::string_view value) { max_width = std::max(max_width, value.size()); }

    /// @brief Print a message to the stream.
    ///
    /// "Print" a non-value message to the stream. In this implementation,
    /// everything is discarded.
    ///
    /// @tparam Args the types of the values to print.
    ///
    /// @param args The values to print.
    template <typename... Args>
    void msg([[maybe_unused]] const Args&... args)
    {
    }
};

/// @brief Implementation of `TensorPrintStream` which actually prints.
///
/// In contrast to `MaxFieldWidthStream`, this function actually prints
/// to an ostream, taking the value produced by that type into account.
struct OutputStream
{
    std::ostream& stream;
    // The maximum width of each tensor value.
    size_t max_width;

    /// @brief Print a tensor value to the stream
    ///
    /// Actually print a value into the stream, (right-)padding it to
    /// `max_width`.
    ///
    /// @param value The value to print.
    void val(std::string_view value)
    {
        stream << std::setfill(' ') << std::setw(max_width) << value;
    }

    /// @brief Print a message to the stream.
    ///
    /// This prints a non-value message directly to the ostream, as if
    /// folded via `operator<<`.
    ///
    /// @tparam Args the types of the values to print.
    ///
    /// @param args The values to print.
    template <typename... Args>
    void msg(const Args&... args)
    {
        (stream << ... << args);
    }
};

} // namespace detail

/// @brief Print device tensor values to an ostream.
///
/// Print the values of a tensor to an ostream. This function neatly formats
/// the tensor according to `config`, tabulating the values so that they are
/// vertically aligned and skipping values to prevent flooding the terminal.
/// With the default config, this function is good to get a quick overview
/// of what a tensor looks like. For a more complete overview, consider
/// supplying `TensorPrintConfig::unlimited()` to get everything (but beware
/// of flooding the terminal). Tensors are printed with the rightmost-dimension
/// as inner dimension, these values appear on the same row in the output.
///
/// @tparam DT The data type of the tensor.
/// @tparam RANK The rank (number of spatial dimensions) of the tensor.
///
/// @param name A name for the tensor. This will be used to add some extra identifying
/// information during printing.
/// @param desc The descriptor for the tensor memory layout.
/// @param d_buffer The tensor's actual data buffer. This is expected to be
/// _device accessible_ memory, as its copied back to the host first.
/// @param config Tensor printing configuration. This allows tweaking some details
/// of the printing process.
/// @param out The ostream to print to, `std::cout` by default.
template <DataType DT, size_t RANK>
void print_tensor(std::string_view name,
                  const TensorDescriptor<DT, RANK>& desc,
                  const void* d_buffer,
                  TensorPrintConfig config = {},
                  std::ostream& out        = std::cout)
{
    // Copy memory to the host (printing from device is sketchy)
    const auto space = desc.get_element_space_size_in_bytes();
    std::vector<std::byte> h_buffer(space);
    check_hip(hipMemcpy(h_buffer.data(), d_buffer, space, hipMemcpyDeviceToHost));

    // Create a custom stream with a completely new config (locale,
    /// precision, fill, etc). Use an osyncstream to buffer the output
    /// while were at it (its not likely to help a lot, but why not).
    std::osyncstream stream(out.rdbuf());
    stream.imbue(std::locale(std::locale(), new detail::numpunct{}));

    // Print a header for the entire tensor (regardless of if there are multiple slices).
    stream << "Tensor \"" << name << "\": shape = " << desc.get_lengths() << "\n";

    detail::TensorPrinter<DT, RANK> printer = {
        .name     = name,
        .config   = config,
        .lengths  = desc.get_lengths(),
        .strides  = desc.get_strides(),
        .h_buffer = h_buffer.data(),
        .ss       = std::stringstream(),
    };

    // We're actually going to print twice: once to figure out the
    // maximum width of the fields, and once to actually print to the stream.

    // Print once to figure out the maximum field width.
    detail::MaxFieldWidthStream max_field_width;
    printer.print_tensor(max_field_width);

    // Actually print to the output stream.
    detail::OutputStream tensor_out = {
        .stream    = stream,
        .max_width = max_field_width.max_width,
    };
    printer.print_tensor(tensor_out);
}

} // namespace ck_tile::builder::test
