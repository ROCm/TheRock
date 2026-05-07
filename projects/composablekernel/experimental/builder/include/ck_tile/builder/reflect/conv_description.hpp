// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// @file
/// @brief Provides utilities to reflect on convolution kernel instances and generate
/// human-readable descriptions of their configuration.
///
/// This file contains the necessary components to transform a convolution kernel's
/// compile-time properties into a structured, descriptive format. This is primarily
/// used for debugging, logging, and generating documentation.
///
/// Key components:
/// - ck_tile::reflect::conv::ConvDescription: A struct that holds the extracted
///   properties and provides methods to format them into strings.
/// - ck_tile::reflect::conv::Describe(): A factory function that creates a
///   ConvDescription from a given kernel instance type.

#pragma once

#include <concepts>
#include <string_view>
#include <sstream>
#include <type_traits>
#include <variant>
#include <functional>

#include <ck_tile/builder/conv_signature_concepts.hpp>
#include <ck_tile/builder/reflect/conv_types.hpp>
#include <ck_tile/builder/reflect/description.hpp>
#include <ck_tile/builder/reflect/instance_traits.hpp>
#include <ck_tile/builder/reflect/tree_formatter.hpp>

namespace ck_tile::reflect {

namespace conv {

/// @brief Signature information for a convolution operation
/// Contains high-level properties that define the convolution's interface,
/// including dimensionality, data layout, data types, and elementwise operations.
struct ConvSignatureInfo
{
    int spatial_dim;
    builder::ConvDirection direction;
    builder::TensorLayout input_layout;
    builder::TensorLayout weight_layout;
    builder::TensorLayout output_layout;
    builder::DataType data_type;
    builder::ElementwiseOperation input_element_op;
    builder::ElementwiseOperation weight_element_op;
    builder::ElementwiseOperation output_element_op;
};

/// @brief Algorithm configuration for a convolution kernel
/// Contains low-level implementation details including thread block configuration,
/// tile dimensions, memory access patterns, and pipeline settings.
struct GemmAlgorithmInfo
{
    int thread_block_size;
    DataTileInfo tile_dims;
    WarpGemmParams warp_gemm;
    InputTileTransferInfo a_tile_transfer;
    InputTileTransferInfo b_tile_transfer;
    OutputTileTransferInfo c_tile_transfer;
    builder::PipelineVersion pipeline_version;
    builder::PipelineScheduler pipeline_scheduler;
    std::variant<builder::ConvFwdSpecialization,
                 builder::ConvBwdDataSpecialization,
                 builder::ConvBwdWeightSpecialization>
        conv_specialization;
    builder::GemmPadding padding;
};

/// @brief Provides human-readable descriptions of convolution kernel instances
/// Generates formatted text descriptions at various levels of detail for
/// understanding and documenting convolution kernel configurations.
class ConvDescription : public Description
{
    public:
    /// @brief Constructor for ConvDescription
    /// @param sig The signature information containing high-level convolution properties
    /// @param algo The algorithm configuration containing low-level implementation details
    /// @param instance_string_getter A callable that returns a string representation of the
    /// instance
    ConvDescription(ConvSignatureInfo sig,
                    GemmAlgorithmInfo algo,
                    std::function<std::string()> instance_string_getter)
        : signature_(std::move(sig)),
          algorithm_(std::move(algo)),
          instance_string_getter_(std::move(instance_string_getter))
    {
    }

    /// @brief Generate a brief one-line summary of the convolution
    /// @return A concise description (e.g., "2D Forward convolution")
    std::string brief() const override
    {
        std::ostringstream oss;
        oss << signature_.spatial_dim << "D " << signature_.direction << " convolution";
        return oss.str();
    }

    /// @brief Generate a detailed hierarchical description of the convolution
    /// @return A multi-line tree-formatted description covering signature and algorithm details
    std::string detailed() const override
    {
        TreeFormatter f;
        f.writeLine(0, signature_.spatial_dim, "D ", signature_.direction, " Convolution Kernel");
        f.writeLine(1, "Signature");
        f.writeLine(2, "Tensor Type: ", signature_.data_type);
        f.writeLine(2, "Input Layout: ", signature_.input_layout);
        f.writeLine(2, "Weight Layout: ", signature_.weight_layout);
        f.writeLine(2, "Output Layout: ", signature_.output_layout);
        f.writeLine(2, "Input elementwise operation: ", signature_.input_element_op);
        f.writeLine(2, "Weights elementwise operation: ", signature_.weight_element_op);
        f.writeLast(2, "Output elementwise operation: ", signature_.output_element_op);

        f.writeLast(1, "Algorithm");
        // Compute Block section
        f.writeLine(2, "Thread block size: ", algorithm_.thread_block_size);
        f.writeLine(2,
                    "Data tile size: ",
                    algorithm_.tile_dims.m,
                    "×",
                    algorithm_.tile_dims.n,
                    "×",
                    algorithm_.tile_dims.k);
        f.writeLine(2, "Gemm padding: ", algorithm_.padding);
        f.writeLine(2, "Convolution specialization: ", algorithm_.conv_specialization);
        // Pipeline section
        f.writeLine(2, "Pipeline version: ", algorithm_.pipeline_version);
        f.writeLine(2, "Pipeline scheduler: ", algorithm_.pipeline_scheduler);
        f.writeLine(2, "Warp Gemm parameters: ");
        f.writeLine(
            3, "subtile size: ", algorithm_.warp_gemm.gemm_m, "×", algorithm_.warp_gemm.gemm_n);
        f.writeLast(3,
                    "Number of warp gemm iterations: ",
                    algorithm_.warp_gemm.m_iter,
                    "×",
                    algorithm_.warp_gemm.n_iter);

        // Memory Access section
        f.writeLast(2, "Memory access:");

        f.writeLine(3, "A Tile transfer: ");
        f.writeLine(4,
                    "Tile dimensions: ",
                    algorithm_.a_tile_transfer.tile_dimensions.k0,
                    "×",
                    algorithm_.a_tile_transfer.tile_dimensions.m_or_n,
                    "×",
                    algorithm_.a_tile_transfer.tile_dimensions.k1,
                    "×");
        f.writeLine(4,
                    "The innermost K subdimension size: ",
                    algorithm_.a_tile_transfer.transfer_params.k1);
        f.writeLine(4,
                    "Spatial thread distribution over the data tile: ",
                    algorithm_.a_tile_transfer.transfer_params.thread_cluster_order[0],
                    "×",
                    algorithm_.a_tile_transfer.transfer_params.thread_cluster_order[1],
                    "×",
                    algorithm_.a_tile_transfer.transfer_params.thread_cluster_order[2]);
        f.writeLine(4,
                    "The order of accessing data tile axes: ",
                    algorithm_.a_tile_transfer.transfer_params.src_access_order[0],
                    "×",
                    algorithm_.a_tile_transfer.transfer_params.src_access_order[1],
                    "×",
                    algorithm_.a_tile_transfer.transfer_params.src_access_order[2]);
        f.writeLine(4,
                    "Vectorized memory access axis index (with contiguous memory): ",
                    algorithm_.a_tile_transfer.transfer_params.src_vector_dim);
        f.writeLine(4,
                    "Vector access (GMEM read) instruction size: ",
                    algorithm_.a_tile_transfer.transfer_params.src_scalar_per_vector);
        f.writeLine(4,
                    "Vector access (LDS write) instruction size: ",
                    algorithm_.a_tile_transfer.transfer_params.dst_scalar_per_vector_k1);
        f.writeLast(4,
                    "LDS data layout padding (to prevent bank conflicts): ",
                    algorithm_.a_tile_transfer.transfer_params.dst_scalar_per_vector_k1);

        f.writeLine(3, "B Tile transfer: ");
        f.writeLine(4,
                    "Tile dimensions: ",
                    algorithm_.b_tile_transfer.tile_dimensions.k0,
                    "×",
                    algorithm_.b_tile_transfer.tile_dimensions.m_or_n,
                    "×",
                    algorithm_.b_tile_transfer.tile_dimensions.k1,
                    "×");
        f.writeLine(4,
                    "The innermost K subdimension size: ",
                    algorithm_.b_tile_transfer.transfer_params.k1);
        f.writeLine(4,
                    "Spatial thread distribution over the data tile: ",
                    algorithm_.b_tile_transfer.transfer_params.thread_cluster_order[0],
                    "×",
                    algorithm_.b_tile_transfer.transfer_params.thread_cluster_order[1],
                    "×",
                    algorithm_.b_tile_transfer.transfer_params.thread_cluster_order[2]);
        f.writeLine(4,
                    "The order of accessing data tile axes: ",
                    algorithm_.b_tile_transfer.transfer_params.src_access_order[0],
                    "×",
                    algorithm_.b_tile_transfer.transfer_params.src_access_order[1],
                    "×",
                    algorithm_.b_tile_transfer.transfer_params.src_access_order[2]);
        f.writeLine(4,
                    "Vectorized memory access axis index (with contiguous memory): ",
                    algorithm_.b_tile_transfer.transfer_params.src_vector_dim);
        f.writeLine(4,
                    "Vector access (GMEM read) instruction size: ",
                    algorithm_.b_tile_transfer.transfer_params.src_scalar_per_vector);
        f.writeLine(4,
                    "Vector access (LDS write) instruction size: ",
                    algorithm_.b_tile_transfer.transfer_params.dst_scalar_per_vector_k1);
        f.writeLast(4,
                    "LDS data layout padding (to prevent bank conflicts): ",
                    algorithm_.b_tile_transfer.transfer_params.dst_scalar_per_vector_k1);

        f.writeLast(3, "C Tile transfer: ");
        f.writeLine(4,
                    "Data shuffle (number of gemm instructions per iteration): ",
                    algorithm_.c_tile_transfer.shuffle_params.m_gemms_per_shuffle,
                    "×",
                    algorithm_.c_tile_transfer.shuffle_params.n_gemms_per_shuffle);
        f.writeLine(4,
                    "Spatial thread distribution used to store data: ",
                    algorithm_.c_tile_transfer.thread_cluster_dims[0],
                    "×",
                    algorithm_.c_tile_transfer.thread_cluster_dims[1],
                    "×",
                    algorithm_.c_tile_transfer.thread_cluster_dims[2],
                    "×",
                    algorithm_.c_tile_transfer.thread_cluster_dims[3]);
        f.writeLast(4,
                    "Vector access (GMEM write) instruction size: ",
                    algorithm_.c_tile_transfer.scalar_per_vector);
        return f.getString();
    }

    /// @brief Generate a string representation of the instance
    /// @return A string that represents the instance
    std::string instance_string() const override { return instance_string_getter_(); }

    private:
    ConvSignatureInfo signature_;
    GemmAlgorithmInfo algorithm_;
    std::function<std::string()> instance_string_getter_;
};

} // namespace conv

} // namespace ck_tile::reflect
